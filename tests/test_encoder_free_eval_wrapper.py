"""Rigorous tests for the encoder-free lmms-eval wrapper (`eval/lmms_eval_wrapper.py`).

These tests define the CONTRACT the encoder-free wrapper must satisfy. They are written first
(TDD); the wrapper code is hand-written by the user to make them pass. Only the image-handling
path of the wrapper changes — the formatting table, chat-template/tokenize/left-pad, decode, and
result-ordering glue are meant to stay as they are, and Tier 0 / Tier 4 guard that.

Division of labour: this file (tests) is the agent's; `eval/lmms_eval_wrapper.py` is the user's.

WHAT "EVAL WORKS CORRECTLY" MEANS HERE
--------------------------------------
We verify MECHANICAL correctness with a tiny random-weight model, deterministically: the right
image features land on the right `<|image|>` positions in the right order; inputs and outputs stay
aligned to their sample; the count invariant holds; nothing fails silently. We do NOT verify SCORE
quality (that needs a trained checkpoint + a trusted golden number) — that is deferred, not faked.

The scary failures are SILENT (no crash, wrong scores): wrong image on wrong placeholders, sample↔
image misalignment, result-order scramble, a broad except turning bugs into empty predictions.
Tiers 2-4 target exactly those.

-------------------------------------------------------------------------------------------
INTERFACE THESE TESTS PIN (adjustable -- tell me and I'll update the tests before you implement):

  * `NanoVLMWrapper(model=<VisionLanguageModel>, device=..., batch_size=...)` builds
    `self.tokenizer` and an encoder-free `self.image_processor` from `model.cfg`.
  * `_prepare_visual_input(visuals)` -- `visuals` is one entry per sample (None/[] or a list of
    images: PIL/path/ndarray). Returns `(per_sample_image_dicts, per_sample_soft_token_counts)`:
      - per_sample_image_dicts[i]: list[dict], one per image of sample i, each
        {"pixel_values": (N, flat), "image_position_ids": (N, 2)} (N=max_soft_tokens, -1=padding);
        [] if sample i has no image.
      - per_sample_soft_token_counts[i]: list[int], real (non-padding) rows per image; [] if none.
  * `_build_images_dict(per_sample_image_dicts)` -- stacks all images across the batch in
    sample-major then image-major order into {"pixel_values": (num_images, N, flat),
    "image_position_ids": (num_images, N, 2)}, or None when the batch has no images.
  * `_trim_to_placeholder_count(per_sample_image_dicts, per_sample_soft_token_counts, input_ids)`
    -- after tokenize/left-pad/truncation, for each sample turn extra trailing real rows into
    padding (last image first) so the sample's total real rows == surviving `<|image|>` count.
    Returns the same (dicts, counts) structure, trimmed.
  * `generate_until` calls `self.model.generate(input_ids, images_dict, attention_mask=...,
    max_new_tokens=..., greedy=..., temperature=..., top_p=...)` -- one images arg, no positions.
-------------------------------------------------------------------------------------------
"""

from __future__ import annotations

import unittest

import numpy as np
import torch
from PIL import Image

from models.config import VLMConfig
from models.vision_language_model import VisionLanguageModel
from eval.lmms_eval_wrapper import NanoVLMWrapper
from data.processors import get_tokenizer
from lmms_eval.api.instance import Instance
from contracts import assert_image_dict


SMOL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _eval_cfg(**overrides) -> VLMConfig:
    """Encoder-free cfg with tiny, real-processor-compatible vision geometry (matches the
    producer-consumer / end-to-end tests: teacher_patch=2, pooling=2 -> flat = 3*4*4 = 48)."""
    defaults = dict(
        vision_backend="encoder_free",
        lm_backend="hf",
        lm_model_type=SMOL_ID,
        lm_tokenizer=SMOL_ID,
        lm_hidden_dim=576,
        teacher_patch_size=2,
        pooling_kernel_size=2,
        model_patch_size=4,
        model_flat_patch_dim=48,
        max_soft_tokens=70,
        mm_embed_dim=8,
        mm_posemb_size=64,
    )
    defaults.update(overrides)
    return VLMConfig(**defaults)


# ============================================================================================
# Tier 0 -- characterization / regression guard for the UNTOUCHED helpers. CPU, no model.
# These pass on today's code and must stay green after the image-path edit: proof we changed
# only the image handling. If the formatting table drifts intentionally, update these values.
# ============================================================================================
class TestBenchmarkFormattingUnchanged(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Pure methods (no model): build a bare instance without running __init__.
        cls.w = NanoVLMWrapper.__new__(NanoVLMWrapper)

    def test_mmstar_row(self):
        f = self.w.get_benchmark_formatting("mmstar")
        self.assertEqual(f["assistant_prefix"], "Answer:")
        self.assertEqual(f["user_prefix"], "")
        self.assertEqual(f["user_suffix"], "")

    def test_mme_row(self):
        f = self.w.get_benchmark_formatting("mme")
        self.assertEqual(f["user_suffix"], "\nGive a very brief answer.")
        self.assertEqual(f["assistant_prefix"], "")

    def test_unknown_task_is_all_empty(self):
        self.assertEqual(
            self.w.get_benchmark_formatting("no_such_task_xyz"),
            {"text_replacements": {}, "assistant_prefix": "", "user_prefix": "", "user_suffix": ""},
        )

    def test_apply_mme_appends_suffix_to_context_only(self):
        ctx, prompt = self.w.apply_benchmark_formatting("What is shown?", "", "mme")
        self.assertEqual(ctx, "What is shown?\nGive a very brief answer.")
        self.assertEqual(prompt, "")

    def test_apply_mmstar_rewrites_choices_and_appends_answer_prefix(self):
        ctx, prompt = self.w.apply_benchmark_formatting("Q\nA. cat", "PROMPT", "mmstar")
        self.assertEqual(ctx, "Q\nChoices:\nA. cat")
        self.assertEqual(prompt, "PROMPTAnswer:")

    # --- All 9+ benchmark groups: one representative per distinct formatting rule. -------------
    # These lock the exact apply_benchmark_formatting output so a drift in the formatting table
    # (a changed replacement, prefix, suffix, or assistant cue) fails loudly. Task names and rules
    # are the wrapper's own (get_benchmark_formatting), not ported from any other repo.

    def test_group1_multichoice_tasks_share_identical_rules(self):
        # ai2d / mmstar / seedbench / scienceqa are one tuple key -> identical formatting.
        rules = [self.w.get_benchmark_formatting(t)
                 for t in ("ai2d", "mmstar", "seedbench", "scienceqa")]
        for r in rules[1:]:
            self.assertEqual(r, rules[0])

    def test_apply_scienceqa_rewrites_choices_and_post_prompt(self):
        # Group-1 member other than mmstar, exercising the choices header + post-prompt rewrite.
        ctx, prompt = self.w.apply_benchmark_formatting(
            "What is shown?\nA. cat\nB. dog\nAnswer with the option's letter from the given choices directly.",
            "", "scienceqa")
        self.assertEqual(
            ctx,
            "What is shown?\nChoices:\nA. cat\nB. dog\nAnswer with the letter directly.")
        self.assertEqual(prompt, "Answer:")   # assistant cue lands even on an empty prompt

    def test_apply_docvqa_prepends_question_prefix_no_answer_cue(self):
        for task in ("docvqa_val", "docvqa_test"):
            ctx, prompt = self.w.apply_benchmark_formatting("How many?", "P", task)
            self.assertTrue(ctx.startswith("Give a short and terse answer to the following question."))
            self.assertTrue(ctx.endswith("Question: How many?"))
            self.assertEqual(prompt, "P")     # no assistant prefix for docvqa

    def test_apply_chartvqa_prepends_instruction_block(self):
        ctx, prompt = self.w.apply_benchmark_formatting("What is the value?", "P", "chartvqa")
        self.assertTrue(ctx.startswith("For the question below, follow the following instructions:"))
        self.assertTrue(ctx.endswith("Question: What is the value?"))
        self.assertEqual(prompt, "P")

    def test_apply_textvqa_prepends_prefix_no_answer_cue(self):
        for task in ("textvqa_val", "textvqa_test"):
            ctx, prompt = self.w.apply_benchmark_formatting("What sign?", "P", task)
            self.assertTrue(ctx.startswith("Answer the following question about the image"))
            self.assertTrue(ctx.endswith("Question: What sign?"))
            self.assertEqual(prompt, "P")

    def test_apply_mmmu_strips_question_label_rewrites_choices_and_cues(self):
        for task in ("mmmu_val", "mmmu_test"):
            ctx, prompt = self.w.apply_benchmark_formatting(
                "Question: What is X?\nA. a\nB. b\nAnswer with the option's letter from the given choices directly.",
                "", task)
            self.assertEqual(
                ctx,
                " What is X?\nChoices:\nA. a\nB. b\nAnswer with the letter directly.")
            self.assertEqual(prompt, "Answer:")

    def test_apply_infovqa_mme_ocrbench_share_brief_suffix(self):
        for task in ("infovqa_val", "mme", "ocrbench"):
            ctx, prompt = self.w.apply_benchmark_formatting("What is shown?", "", task)
            self.assertEqual(ctx, "What is shown?\nGive a very brief answer.")
            self.assertEqual(prompt, "")      # no assistant prefix for these


# ============================================================================================
# Tier 0b -- the FULL assembled prompt (real code path). Exercises NanoVLMWrapper._assemble_prompt
# -- the exact prompt string generate_until feeds the tokenizer -- so we catch end-to-end shape
# bugs the per-method formatting tests can't: wrong <|image|> placeholder count, and the assistant
# cue not landing at the very end (a stray trailing char shifts tokenization and misgrades). Needs
# only the tokenizer, not a model -> CPU, no GPU.
# ============================================================================================
class TestAssembledPrompt(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = _eval_cfg()
        cls.tok = get_tokenizer(cls.cfg.lm_tokenizer, cls.cfg.vlm_extra_tokens, cls.cfg.lm_chat_template)
        # Bare wrapper carrying only the tokenizer -- _assemble_prompt needs nothing else.
        cls.w = NanoVLMWrapper.__new__(NanoVLMWrapper)
        cls.w.tokenizer = cls.tok
        cls.IMG = cls.tok.image_token

    def test_placeholder_count_equals_sum_of_soft_token_counts(self):
        # One <|image|> per real patch, across all of the sample's images.
        prompt = self.w._assemble_prompt("What is shown?", [3, 2], "mme")
        self.assertEqual(prompt.count(self.IMG), 5)

    def test_text_only_sample_has_no_placeholders(self):
        prompt = self.w._assemble_prompt("What is shown?", [], "mme")
        self.assertEqual(prompt.count(self.IMG), 0)

    def test_answer_cue_lands_at_the_very_end_for_mmstar(self):
        # The grader reads a leading letter; the "Answer:" cue must be the LAST thing in the prompt
        # (after the chat template), with no trailing whitespace, or tokenization/steering is off.
        prompt = self.w._assemble_prompt("Q\nA. cat\nB. dog", [1], "mmstar")
        self.assertTrue(prompt.endswith("Answer:"), repr(prompt[-20:]))
        self.assertEqual(prompt, prompt.rstrip(), "stray trailing whitespace after the cue")

    def test_choices_header_and_placeholders_present_together_for_mmstar(self):
        prompt = self.w._assemble_prompt("Q\nA. cat\nB. dog", [4], "mmstar")
        self.assertIn("\nChoices:\nA. cat", prompt)
        self.assertEqual(prompt.count(self.IMG), 4)

    def test_docvqa_has_prefix_and_no_answer_cue(self):
        prompt = self.w._assemble_prompt("How many?", [2], "docvqa_val")
        self.assertIn("Give a short and terse answer to the following question.", prompt)
        self.assertFalse(prompt.endswith("Answer:"))
        self.assertEqual(prompt.count(self.IMG), 2)


# ============================================================================================
# Shared fixture for the model-loading tiers (GPU under srun; CPU fallback). Builds a tiny
# encoder-free model + the wrapper + two differently-sized images ONCE.
# ============================================================================================
class _EvalWrapperFixture(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_default_device("cpu")

    @classmethod
    def setUpClass(cls) -> None:
        torch.set_default_device(DEVICE)
        torch.manual_seed(0)
        cls.cfg = _eval_cfg()
        model = VisionLanguageModel(cls.cfg, load_backbone=False).to(DEVICE)
        model.eval()
        cls.w = NanoVLMWrapper(model=model, device=str(DEVICE), batch_size=8)
        cls.model = cls.w.model
        cls.tok = cls.w.tokenizer
        cls.IMG = cls.tok.image_token_id
        cls.PAD = cls.tok.pad_token_id if cls.tok.pad_token_id is not None else (cls.tok.eos_token_id or 0)
        cls.TXT = 5  # any fixed non-special text token id

        rng = np.random.RandomState(0)
        # sides are multiples of patch*pooling (=4); different sizes -> different real-row counts.
        cls.img_a = Image.fromarray(rng.randint(0, 256, (16, 16, 3), dtype=np.uint8))
        cls.img_b = Image.fromarray(rng.randint(0, 256, (12, 16, 3), dtype=np.uint8))

    # --- helpers ---------------------------------------------------------------------------
    def _real_rows(self, per_image_dict) -> int:
        pos = per_image_dict["image_position_ids"]
        return int((pos >= 0).all(dim=-1).sum())

    def _left_pad_rows(self, rows):
        """rows: list[list[int]] -> (input_ids, attention_mask) left-padded to the longest row."""
        maxlen = max(len(r) for r in rows)
        ids, mask = [], []
        for r in rows:
            pad = maxlen - len(r)
            ids.append([self.PAD] * pad + r)
            mask.append([0] * pad + [1] * len(r))
        return (torch.tensor(ids, dtype=torch.long, device=DEVICE),
                torch.tensor(mask, dtype=torch.long, device=DEVICE))


# ============================================================================================
# Tier 1 -- `_prepare_visual_input` output + placeholder-string consistency (the count rule).
# ============================================================================================
class TestPrepareVisualInput(_EvalWrapperFixture):
    def test_shapes_counts_and_grouping(self):
        visuals = [[self.img_a, self.img_b], None, [self.img_a]]
        dicts, counts = self.w._prepare_visual_input(visuals)
        self.assertEqual([len(d) for d in dicts], [2, 0, 1])   # per-sample grouping preserved
        self.assertEqual([len(c) for c in counts], [2, 0, 1])
        N, flat = self.cfg.max_soft_tokens, self.cfg.model_flat_patch_dim
        for sample_dicts, sample_counts in zip(dicts, counts):
            for d, c in zip(sample_dicts, sample_counts):
                self.assertEqual(tuple(d["pixel_values"].shape), (N, flat))
                self.assertEqual(tuple(d["image_position_ids"].shape), (N, 2))
                self.assertGreater(c, 0)
                self.assertEqual(self._real_rows(d), c)         # count == real rows in that image

    def test_counts_match_processor(self):
        ref = self.w.image_processor([self.img_a, self.img_b])["num_soft_tokens_per_image"]
        _, counts = self.w._prepare_visual_input([[self.img_a, self.img_b]])
        self.assertEqual(list(counts[0]), list(ref))

    def test_text_only_sample_is_empty(self):
        dicts, counts = self.w._prepare_visual_input([None])
        self.assertEqual(dicts, [[]])
        self.assertEqual(counts, [[]])

    def test_placeholder_string_has_exactly_sum_counts_tokens(self):
        from data.processors import get_image_string_encoder_free
        _, counts = self.w._prepare_visual_input([[self.img_a, self.img_b]])
        s = get_image_string_encoder_free(self.tok, counts[0])
        ids = self.tok(s, add_special_tokens=False)["input_ids"]
        self.assertEqual(sum(1 for t in ids if t == self.IMG), sum(counts[0]))


# ============================================================================================
# Tier 2 -- CROWN JEWEL: the image features written into each sample's <|image|> positions
# equal that sample's own embedder->projector->real-rows, in sample-major then image-major
# order. Catches "wrong image on wrong placeholders" and sample<->image misalignment directly.
# ============================================================================================
class TestImageScatterOrdering(_EvalWrapperFixture):
    def test_scattered_rows_match_expected_per_image_in_order(self):
        visuals = [[self.img_a, self.img_b], [self.img_a]]   # sample0: 2 imgs, sample1: 1 img
        dicts, counts = self.w._prepare_visual_input(visuals)
        images_dict = self.w._build_images_dict(dicts)
        assert_image_dict(images_dict, num_images=3, N=self.cfg.max_soft_tokens,
                          flat_dim=self.cfg.model_flat_patch_dim)

        # input_ids: sample0 placeholders first, then sample1 (row-major matches stack order).
        row0 = [self.TXT] + [self.IMG] * sum(counts[0]) + [self.TXT]
        row1 = [self.TXT] + [self.IMG] * sum(counts[1]) + [self.TXT]
        input_ids, attn = self._left_pad_rows([row0, row1])

        # Expected features derived INDEPENDENTLY from the per-sample dicts, in sample-major then
        # image-major order (NOT from images_dict). This is the key: if _build_images_dict stacks
        # images in the wrong order, `captured` (which follows images_dict) will disagree with this
        # sample-ordered `expected`, catching the "wrong image on wrong sample" bug.
        with torch.no_grad():
            expected_rows = []
            for sample_dicts in dicts:
                for d in sample_dicts:
                    pv = d["pixel_values"].unsqueeze(0)                  # (1, N, flat)
                    pos = d["image_position_ids"].unsqueeze(0)           # (1, N, 2)
                    feat = self.model.vision_projector(self.model.vision_embedder(pv, pos))
                    expected_rows.append(feat[(pos >= 0).all(dim=-1)])   # (real_i, lm_hidden)
            expected = torch.cat(expected_rows, dim=0)                   # (total_real, lm_hidden)

        captured = {}
        def hook(module, args, kwargs):
            captured["tok"] = args[0].detach().clone()
        handle = self.model.decoder.register_forward_pre_hook(hook, with_kwargs=True)
        try:
            self.model(input_ids=input_ids, images=images_dict, attention_mask=attn)
        finally:
            handle.remove()

        got = captured["tok"][input_ids == self.IMG]                    # row-major over the batch
        self.assertEqual(got.shape, expected.shape)
        self.assertTrue(torch.allclose(got, expected, atol=1e-5, rtol=1e-4),
                        "scattered image features are in the wrong order / wrong sample")

    def test_build_images_dict_none_when_no_images(self):
        self.assertIsNone(self.w._build_images_dict([[], []]))


# ============================================================================================
# Tier 3 -- batch-vs-single equivalence (greedy, deterministic). One test that fails if
# left-padding, attention masking, image ordering, or per-sample alignment is wrong.
# NOTE: a failure here can also indicate a decoder-side left-padding / position-id issue
# (not the wrapper) -- either way it means batched eval is not output-invariant, which is a
# real eval-correctness problem worth surfacing.
# ============================================================================================
class TestBatchSingleEquivalence(_EvalWrapperFixture):
    def _assemble(self, visuals):
        dicts, counts = self.w._prepare_visual_input(visuals)
        images_dict = self.w._build_images_dict(dicts)
        rows = [[self.TXT] + [self.IMG] * sum(c) + [self.TXT] for c in counts]
        input_ids, attn = self._left_pad_rows(rows)
        return input_ids, attn, images_dict, dicts

    def test_batched_generation_equals_per_sample(self):
        visuals = [[self.img_a], [self.img_a, self.img_b]]   # different lengths -> real left-pad
        input_ids, attn, images_dict, dicts = self._assemble(visuals)
        with torch.no_grad():
            out_batch = self.model.generate(input_ids, images_dict, attention_mask=attn,
                                            max_new_tokens=6, greedy=True)
        for i, sample_visuals in enumerate(visuals):
            single_ids, single_attn, single_dict, _ = self._assemble([sample_visuals])
            with torch.no_grad():
                out_single = self.model.generate(single_ids, single_dict, attention_mask=single_attn,
                                                 max_new_tokens=6, greedy=True)
            self.assertTrue(torch.equal(out_batch[i], out_single[0]),
                            f"sample {i}: batched output differs from single -> padding/align/order bug")


# ============================================================================================
# Tier 4 -- generate_until glue, OFFLINE with mock Instances (no network, no checkpoint).
# Guards the result-ordering (get_original) that must NOT scramble scores, and that a valid
# batch never silently returns "" (the broad except hazard).
# ============================================================================================
class TestGenerateUntilGlue(_EvalWrapperFixture):
    def _instance(self, idx, context, visual, gen_kwargs=None):
        task, split, doc_id = "faketask", "test", idx
        self.w.task_dict.setdefault(task, {}).setdefault(split, {})[doc_id] = {"visual": visual}
        args = (context, gen_kwargs or {"max_new_tokens": 4},
                (lambda doc: doc["visual"]), doc_id, task, split)
        return Instance(request_type="generate_until", arguments=args, idx=idx,
                        metadata={"task": task, "doc_id": doc_id, "repeats": 1})

    def test_results_returned_in_original_order_text_only(self):
        # Echo stub: model.generate returns the input_ids, so decoded output contains the prompt
        # (hence its unique marker). If get_original mis-orders, the markers land on the wrong slot.
        orig = self.model.generate
        try:
            self.model.generate = lambda input_ids, images, attention_mask=None, **kw: input_ids
            reqs = [
                self._instance(0, "MARKERAAA short", None),
                self._instance(1, "MARKERBBB a considerably longer context so the collator reorders", None),
            ]
            res = self.w.generate_until(reqs)
        finally:
            self.model.generate = orig
        self.assertEqual(len(res), 2)
        self.assertIn("MARKERAAA", res[0])
        self.assertIn("MARKERBBB", res[1])

    def test_image_batch_runs_through_generate_until_without_silent_empty(self):
        # Drives the full image path through generate_until (assembly + trim + build dict), with an
        # echo stub for generate so we test the plumbing, not generation. A crash would be swallowed
        # by the wrapper's except and returned as "" -- so non-empty results prove the path ran.
        orig = self.model.generate
        try:
            self.model.generate = lambda input_ids, images, attention_mask=None, **kw: input_ids
            reqs = [
                self._instance(0, "MARKERONE", [self.img_a]),
                self._instance(1, "MARKERTWO", [self.img_a, self.img_b]),
            ]
            res = self.w.generate_until(reqs)
        finally:
            self.model.generate = orig
        self.assertEqual(len(res), 2)
        self.assertTrue(all(r != "" for r in res), "a sample came back empty -> except swallowed a bug")
        self.assertIn("MARKERONE", res[0])
        self.assertIn("MARKERTWO", res[1])


# ============================================================================================
# Tier 5 -- edge cases + the loud oracle.
# ============================================================================================
class TestTruncationAndCountOracle(_EvalWrapperFixture):
    def test_trim_reconciles_dropped_placeholders_and_generate_runs(self):
        visuals = [[self.img_a]]
        dicts, counts = self.w._prepare_visual_input(visuals)
        real = sum(counts[0])
        self.assertGreater(real, 1)
        # Simulate truncation that dropped one image placeholder for this sample.
        kept = real - 1
        row = [self.IMG] * kept + [self.TXT]
        input_ids, attn = self._left_pad_rows([row])
        trimmed_dicts, trimmed_counts = self.w._trim_to_placeholder_count(dicts, counts, input_ids)
        self.assertEqual(sum(self._real_rows(d) for d in trimmed_dicts[0]), kept)
        images_dict = self.w._build_images_dict(trimmed_dicts)
        with torch.no_grad():   # must NOT raise: real rows now == surviving placeholders
            self.model.generate(input_ids, images_dict, attention_mask=attn,
                                 max_new_tokens=3, greedy=True)

    def test_count_mismatch_raises_runtimeerror(self):
        # The oracle the trim exists to satisfy: more real rows than placeholders -> RuntimeError.
        dicts, counts = self.w._prepare_visual_input([[self.img_a]])
        images_dict = self.w._build_images_dict(dicts)
        real = sum(counts[0])
        bad_row = [self.IMG] * (real - 1) + [self.TXT]      # one placeholder short of the real rows
        input_ids, attn = self._left_pad_rows([bad_row])
        with self.assertRaises(RuntimeError):
            self.model.generate(input_ids, images_dict, attention_mask=attn,
                                max_new_tokens=1, greedy=True)


if __name__ == "__main__":
    unittest.main()
