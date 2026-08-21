"""
GPU smoke test for the HF-decoder redesign — the parts unit tests can't cover on CPU
with random weights. Run on a GPU node (see tests/gpu_smoke_test.slurm).

Unlike tests/test_redesign.py and tests/test_kv_cache_invariant.py (which use
load_backbone=False + random weights), this drives the REAL paths a training run hits:

  * load_backbone=True  -> real pretrained ViT (SigLIP2) + real HF decoder weights.
  * one training step under the exact recipe in train.py: torch.autocast(bfloat16)
    forward -> backward -> clip_grad_norm_ -> optimizer.step(), asserting a finite loss,
    a finite positive grad norm, and that weights actually move.
  * generate() under inference_mode (the Blocker-3 / lmms-eval path).
  * save_pretrained -> from_pretrained round-trip (the resume path): shapes/dtypes match
    and the reloaded model reproduces the original loss.

It also re-verifies, on the REAL weights, the properties the redesign relies on and that
the CPU tests could only check on random skeletons:
  * uniform float32 master weights across decoder + ViT + MP (the dtype fix),
  * cfg.lm_vocab_size propagated from the tokenizer (Gap 4),
  * tie-weights state matches cfg.lm_tie_weights (Gap 5),
  * head(base(x)) == model(x).logits, i.e. base returns post-final-norm states (Gap 5).

Usage:
    python tests/gpu_smoke_test.py --model both      # smol + lfm (default)
    python tests/gpu_smoke_test.py --model lfm
    python tests/gpu_smoke_test.py --model smol --skip-generate

Exit code is non-zero if any check fails, so SLURM marks the job FAILED.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import traceback

import torch
import torch.optim as optim

# Allow running as a plain script (python tests/gpu_smoke_test.py): Python puts tests/ on
# sys.path, not the repo root, so put the repo root (parent of tests/) first to import models.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.config import VLMConfig
from models.vision_language_model import VisionLanguageModel


MODEL_IDS = {
    "smol": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "lfm": "LiquidAI/LFM2.5-230M",
}


# --------------------------------------------------------------------------------------
# Minimal pass/fail harness (no pytest on the cluster venv).
# --------------------------------------------------------------------------------------
class Runner:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, name: str, fn) -> None:
        try:
            fn()
            print(f"  [PASS] {name}", flush=True)
        except Exception as e:  # noqa: BLE001 — smoke test wants every failure, not the first
            self.failures.append(name)
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()


def build_cfg(model_key: str) -> VLMConfig:
    """Real config on the hf backend. lm_hidden_dim is auto-corrected by the Decoder."""
    model_id = MODEL_IDS[model_key]
    return VLMConfig(lm_backend="hf", lm_model_type=model_id, lm_tokenizer=model_id)


def make_batch(model: VisionLanguageModel, device: torch.device, batch_size: int = 2, n_text: int = 8):
    """A synthetic multimodal batch: one image per sample + text, with image slots masked."""
    cfg = model.cfg
    with torch.no_grad():
        probe = model.MP(model.vision_encoder(torch.randn(1, 3, cfg.vit_img_size, cfg.vit_img_size, device=device)))
    n_img_tokens = probe.size(1)
    itid = model.tokenizer.image_token_id

    rows = []
    for _ in range(batch_size):
        text = torch.randint(10, 1000, (n_text,)).tolist()
        rows.append([itid] * n_img_tokens + text)
    input_ids = torch.tensor(rows, dtype=torch.long, device=device)          # [B, T]
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[input_ids == itid] = -100                                         # no loss on image slots
    images = [torch.randn(1, 3, cfg.vit_img_size, cfg.vit_img_size, device=device) for _ in range(batch_size)]
    return input_ids, attention_mask, labels, images


def run_for_model(model_key: str, runner: Runner, skip_generate: bool) -> None:
    device = torch.device("cuda")
    torch.manual_seed(0)
    print(f"\n=== {model_key.upper()} ({MODEL_IDS[model_key]}) ===", flush=True)

    # Real pretrained backbones (this is the load_backbone=True path train.py uses).
    model = VisionLanguageModel(build_cfg(model_key), load_backbone=True).to(device)
    input_ids, attention_mask, labels, images = make_batch(model, device)

    # ---- Property checks on REAL weights ------------------------------------------------
    def dtype_uniform_fp32():
        assert {p.dtype for p in model.parameters()} == {torch.float32}, \
            {p.dtype for p in model.parameters()}
    runner.check("dtype: whole VLM is uniform float32 (master weights)", dtype_uniform_fp32)

    def vocab_propagated():
        assert model.cfg.lm_vocab_size == len(model.tokenizer), \
            (model.cfg.lm_vocab_size, len(model.tokenizer))
    runner.check("Gap4: cfg.lm_vocab_size == len(tokenizer)", vocab_propagated)

    def tie_state_matches():
        ie = model.decoder.model.get_input_embeddings().weight
        oe = model.decoder.model.get_output_embeddings().weight
        assert (ie.data_ptr() == oe.data_ptr()) == model.cfg.lm_tie_weights
    runner.check("Gap5: tie-weights state matches cfg.lm_tie_weights", tie_state_matches)

    def head_of_base_matches_logits():
        text_ids = input_ids[:, -8:]  # a short text-only slice
        x = model.decoder.token_embedding(text_ids)
        with torch.no_grad():
            manual = model.decoder.head(model.decoder.base(inputs_embeds=x).last_hidden_state)
            official = model.decoder.model(inputs_embeds=x).logits
        assert torch.allclose(manual, official, atol=1e-3, rtol=1e-3)
    runner.check("Gap5: head(base(x)) == model(x).logits (post-final-norm)", head_of_base_matches_logits)

    # ---- Test A: one real training step (train.py recipe) -------------------------------
    def train_step():
        model.train()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        # snapshot a decoder weight to confirm it moves
        pname, p = next((n, p) for n, p in model.decoder.model.named_parameters() if p.requires_grad)
        before = p.detach().clone()

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)
        assert torch.isfinite(loss), f"loss not finite: {loss}"
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        assert torch.isfinite(grad_norm) and grad_norm > 0, f"grad_norm={grad_norm}"
        optimizer.step()
        assert not torch.equal(before, p.detach()), f"{pname} did not update after optimizer.step()"

        # a second forward still produces a finite loss (no NaN blow-up post-update)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss2 = model(input_ids, images, attention_mask=attention_mask, targets=labels)
        assert torch.isfinite(loss2)
    runner.check("Test A: autocast(bf16) forward+backward+step (finite loss/grad, weights move)", train_step)

    # ---- Test A2: generate() (Blocker 3 / lmms-eval path) ------------------------------
    if not skip_generate:
        def generate_runs():
            model.eval()
            with torch.inference_mode():
                out = model.generate(input_ids, images, max_new_tokens=5, greedy=True)
            assert out.shape == (input_ids.size(0), 5), out.shape
            assert out.dtype == torch.long
            assert (out >= 0).all() and (out < len(model.tokenizer)).all()
        runner.check("Test A2: generate() returns valid in-range token ids", generate_runs)

    # ---- Test B: save_pretrained -> from_pretrained (resume path) ----------------------
    def save_resume_roundtrip():
        model.eval()
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as d:
            model.save_pretrained(d)
            reloaded = VisionLanguageModel.from_pretrained(d).to(device).eval()

        # (1) every param matches in name/shape/dtype -> resume skeleton lines up with save.
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), reloaded.named_parameters()):
            assert n1 == n2, (n1, n2)
            assert p1.shape == p2.shape, (n1, p1.shape, p2.shape)
            assert p1.dtype == p2.dtype, (n1, p1.dtype, p2.dtype)

        # (2) same input -> same loss (weights actually round-tripped, not just shapes).
        with torch.no_grad():
            _, l_orig = model(input_ids, images, attention_mask=attention_mask, targets=labels)
            _, l_reload = reloaded(input_ids, images, attention_mask=attention_mask, targets=labels)
        assert torch.allclose(l_orig, l_reload, atol=1e-4, rtol=1e-4), (l_orig.item(), l_reload.item())
    runner.check("Test B: save_pretrained -> from_pretrained round-trip (shapes + loss)", save_resume_roundtrip)

    del model
    torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["smol", "lfm", "both"], default="both")
    parser.add_argument("--skip-generate", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available — this smoke test must run on a GPU node.", file=sys.stderr)
        return 2

    print(f"torch {torch.__version__} | device {torch.cuda.get_device_name(0)}", flush=True)
    keys = ["smol", "lfm"] if args.model == "both" else [args.model]

    runner = Runner()
    for key in keys:
        try:
            run_for_model(key, runner, args.skip_generate)
        except Exception as e:  # construction itself failed
            runner.failures.append(f"{key}: construction")
            print(f"  [FAIL] {key}: construction: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

    print("\n" + "=" * 60, flush=True)
    if runner.failures:
        print(f"SMOKE TEST FAILED — {len(runner.failures)} check(s) failed:", flush=True)
        for f in runner.failures:
            print(f"  - {f}", flush=True)
        return 1
    print("SMOKE TEST PASSED — all checks green.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
