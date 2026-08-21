"""Seam tests for the encoder-free path wired into the VLM (step 3, sub-step 2).

Branch-by-abstraction: a new config flag ``vision_backend: "vit" | "encoder_free"`` (default
``"vit"``) selects the path. The ViT path stays the default so all existing tests keep passing;
these tests exercise ONLY the new ``"encoder_free"`` branch.

THE ONE RULE under test (the whole reason this file exists):

    number of ``<|image|>`` placeholder tokens in input_ids
      == number of REAL (non-filler) image feature vectors written into the sequence.

New (encoder-free) path per the settled choices (memory ``encoder-free-step3-token-scaffolding``):
run ``VisionEmbedder -> VisionProjector`` over the (padded) patches, then write ONLY the real rows
(skip the filler) into the ``<|image|>`` positions. A row is filler iff its position is
``(-1, -1)``; a real row has both coordinates ``>= 0``.

-------------------------------------------------------------------------------------------
INTERFACE THIS TEST PINS (all adjustable -- tell me and I'll update the test before you write
the module; the module is yours to hand-write):

  * config: ``VLMConfig`` gains ``vision_backend: str = "vit"``.
  * __init__: when ``cfg.vision_backend == "encoder_free"``, the VLM builds
      ``self.vision_embedder`` (VisionEmbedder) and ``self.vision_projector`` (VisionProjector).
  * forward's existing ``images`` argument carries, on the encoder-free branch, a DICT:
      ``images = {"pixel_values":       (num_images, N, model_flat_patch_dim) float, padded to N,
                  "image_position_ids": (num_images, N, 2) int, (-1,-1) marks padding}``
    ``images=None`` means "no images" (text-only) on either branch.
  * the scatter must drop padding: only the non-padding rows (both coords >= 0) are scattered,
    reusing the existing ``_replace_img_tokens_with_embd`` (masked assignment), so a count
    mismatch surfaces as a RuntimeError rather than a silent mis-scatter.
-------------------------------------------------------------------------------------------

Design notes (mirror tests/test_redesign.py):
  * ``load_backbone=False`` -> the LM is randomly initialized, so no large weight files are
    downloaded (only the small cached config.json + tokenizer).
  * The heavy model + the synthetic batch are built once in ``setUpClass`` and shared.
  * Vision dims are shrunk to tiny, hand-checkable sizes via config overrides.
"""

from __future__ import annotations

import unittest

import torch

from models.config import VLMConfig
from models.vision_language_model import VisionLanguageModel
from data.image_processing import ImageProcessor
from data.datasets import VQADataset
from data.collators import VQACollator
from data.processors import get_tokenizer
from contracts import assert_image_dict


# Smallest cached HF decoder (hidden size 576). random-init via load_backbone=False.
SMOL_ID: str = "HuggingFaceTB/SmolLM2-135M-Instruct"

# Tiny, hand-checkable vision geometry.
FLAT_PATCH_DIM = 12   # model_flat_patch_dim: values per patch row
MM_EMBED_DIM = 8      # embedder output width
POSEMB_SIZE = 10      # factorized 2D pos-table size per axis (all positions used are < this)
N_MAX = 6             # padded patches per image (max_soft_tokens for the batch)

# Model-loading tests run on the GPU when one is present (e.g. under srun on a GPU node) and fall
# back to CPU otherwise. Each model-building class sets this as the default device in setUpClass so
# the model and every tensor the tests create land on it, and resets it in tearDownClass.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _encoder_free_cfg(**overrides) -> VLMConfig:
    """A VLMConfig on the encoder-free backend with tiny vision dims and a small HF decoder."""
    defaults = dict(
        vision_backend="encoder_free",
        lm_backend="hf",
        lm_model_type=SMOL_ID,
        lm_tokenizer=SMOL_ID,
        lm_hidden_dim=576,  # SmolLM2-135M hidden size; Decoder overrides cfg from the model anyway
        model_flat_patch_dim=FLAT_PATCH_DIM,
        mm_embed_dim=MM_EMBED_DIM,
        mm_posemb_size=POSEMB_SIZE,
        max_soft_tokens=N_MAX,
    )
    defaults.update(overrides)
    return VLMConfig(**defaults)


class TestEncoderFreeConfigDefault(unittest.TestCase):
    """Cheap, model-free: the flag exists and is additive (default keeps the ViT path)."""

    def test_default_vision_backend_is_vit(self):
        self.assertEqual(VLMConfig().vision_backend, "vit")


class TestEncoderFreeSeam(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_default_device("cpu")  # undo the class-scoped default-device change

    @classmethod
    def setUpClass(cls) -> None:
        torch.set_default_device(DEVICE)  # build model + fixture tensors on the GPU when available
        torch.manual_seed(0)
        cls.cfg = _encoder_free_cfg()
        cls.model = VisionLanguageModel(cls.cfg, load_backbone=False).to(DEVICE)
        cls.model.eval()

        tok = cls.model.tokenizer
        cls.IMG = tok.image_token_id
        cls.PAD = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
        cls.TXT = 5  # any fixed, valid, non-special token id

        # --- Two images of DIFFERENT real length, both padded to N_MAX=6 ------------------
        # image 0: 3 real rows + 3 padding;  image 1: 5 real rows + 1 padding.  Sum real = 8.
        cls.SOFT = [3, 5]
        pv = torch.randn(2, N_MAX, FLAT_PATCH_DIM)
        pv[0, 3:] = 0.0  # padding rows are zero-filled, mirroring the image processor
        pv[1, 5:] = 0.0
        pos = torch.full((2, N_MAX, 2), -1, dtype=torch.long)
        pos[0, :3] = torch.tensor([[0, 0], [0, 1], [1, 0]])
        pos[1, :5] = torch.tensor([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]])
        cls.pixel_values = pv
        cls.image_position_ids = pos
        # The encoder-free path receives image data bundled in the `images` argument as a dict.
        cls.images = {"pixel_values": pv, "image_position_ids": pos}
        # a real row has both coords >= 0; a padding row is (-1, -1)
        cls.real_mask = (pos >= 0).all(dim=-1)  # (2, N_MAX); sums to 8

        # --- Text with EXACTLY sum(real) = 8 image placeholders across the batch ----------
        # row 0: 2 leading pads + text + 3 image tokens (image 0) + text
        # row 1: text + 5 image tokens (image 1) + text  (no leading pad)
        IMG, PAD, TXT = cls.IMG, cls.PAD, cls.TXT
        row0 = [PAD, PAD, TXT, TXT, IMG, IMG, IMG, TXT]
        row1 = [TXT, IMG, IMG, IMG, IMG, IMG, TXT, TXT]
        cls.input_ids = torch.tensor([row0, row1], dtype=torch.long)
        cls.attention_mask = torch.tensor(
            [[0, 0, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.long
        )
        # targets: ignore pad + image positions; real text positions supervise (any valid id).
        targets = cls.input_ids.clone()
        targets[cls.attention_mask == 0] = -100
        targets[cls.input_ids == IMG] = -100
        cls.targets = targets

    # --- helpers -------------------------------------------------------------------------

    def _expected_scattered_features(self) -> torch.Tensor:
        """Recompute, outside forward, the image features the model SHOULD scatter, in order:
        embedder -> projector over the padded patches, then keep only the non-padding rows,
        flattened image-major then position-major (matching the placeholder order)."""
        with torch.no_grad():
            embd = self.model.vision_embedder(self.pixel_values, self.image_position_ids)
            feats = self.model.vision_projector(embd)  # (2, N_MAX, lm_hidden_dim)
        return feats[self.real_mask]  # (8, lm_hidden_dim)

    def _capture_decoder_input(self, **forward_kwargs) -> torch.Tensor:
        """Run forward while capturing the embedding tensor handed to the decoder (the point
        right after the scatter), via a pre-hook on model.decoder."""
        captured = {}

        def hook(module, args, kwargs):
            captured["tok"] = args[0].detach().clone()

        handle = self.model.decoder.register_forward_pre_hook(hook, with_kwargs=True)
        try:
            self.model(**forward_kwargs)
        finally:
            handle.remove()
        return captured["tok"]

    # --- the fixture itself encodes the invariant ----------------------------------------

    def test_fixture_encodes_the_count_invariant(self):
        # Guard the guard: the number of image placeholders must equal the number of non-padding
        # rows, or every downstream assertion would be meaningless.
        n_placeholders = int((self.input_ids == self.IMG).sum())
        n_real_rows = int(self.real_mask.sum())
        self.assertEqual(n_placeholders, n_real_rows)
        self.assertEqual(n_placeholders, sum(self.SOFT))  # == 8

    # --- core: the right rows land in the right places -----------------------------------

    def test_scattered_features_match_and_are_ordered(self):
        # The strongest correctness check: after the scatter, the decoder's input at the image
        # positions equals the projector's non-padding output rows, IN ORDER (image 0's rows
        # fill row 0's placeholders, image 1's fill row 1's). Text/pad positions are the plain
        # token embeddings, untouched.
        tok = self._capture_decoder_input(
            input_ids=self.input_ids,
            images=self.images,
            attention_mask=self.attention_mask,
        )
        expected = self._expected_scattered_features()
        img_mask = (self.input_ids == self.IMG)

        self.assertEqual(int(img_mask.sum()), expected.shape[0])  # count matches
        self.assertTrue(
            torch.allclose(tok[img_mask], expected, atol=1e-5, rtol=1e-4),
            "scattered image features differ from projector output or are out of order",
        )
        # Non-image positions must be exactly the token embeddings (nothing else was touched).
        plain = self.model.decoder.token_embedding(self.input_ids)
        self.assertTrue(torch.allclose(tok[~img_mask], plain[~img_mask], atol=1e-6))

    def test_only_nonpadding_rows_are_scattered(self):
        # Drop-padding: exactly sum(num_soft)=8 rows are scattered, strictly fewer than the
        # padded total (2 images x 6 = 12). Padding rows never reach the sequence.
        expected = self._expected_scattered_features()
        self.assertEqual(expected.shape[0], sum(self.SOFT))
        self.assertLess(expected.shape[0], self.pixel_values.shape[0] * N_MAX)

    # --- the invariant is load-bearing ---------------------------------------------------

    def test_count_mismatch_raises(self):
        # If input_ids carries the wrong number of placeholders (here 7, not 8), the masked
        # assignment in the scatter must fail loudly rather than silently mis-place features.
        IMG, PAD, TXT = self.IMG, self.PAD, self.TXT
        bad_row0 = [PAD, PAD, TXT, TXT, TXT, IMG, IMG, TXT]  # only 2 image tokens (was 3)
        bad_row1 = [TXT, IMG, IMG, IMG, IMG, IMG, TXT, TXT]  # 5 image tokens -> total 7 != 8
        bad_input_ids = torch.tensor([bad_row0, bad_row1], dtype=torch.long)
        with self.assertRaises(RuntimeError):
            self.model(
                input_ids=bad_input_ids,
                images=self.images,
                attention_mask=self.attention_mask,
            )

    # --- shapes, finiteness, and the text-only branch ------------------------------------

    def test_forward_runs_and_logits_are_finite(self):
        logits, loss = self.model(
            input_ids=self.input_ids,
            images=self.images,
            attention_mask=self.attention_mask,
        )
        self.assertEqual(logits.shape[0], self.input_ids.shape[0])
        self.assertEqual(logits.shape[1], self.input_ids.shape[1])
        self.assertTrue(torch.isfinite(logits).all())
        self.assertIsNone(loss)  # no targets passed

    def test_text_only_forward_runs(self):
        # No images: the encoder-free branch must still produce finite logits for pure text.
        text_ids = torch.tensor([[self.TXT, self.TXT, self.TXT, self.TXT]], dtype=torch.long)
        logits, _ = self.model(
            input_ids=text_ids,
            images=None,
            attention_mask=torch.ones_like(text_ids),
        )
        self.assertEqual(logits.shape[:2], (1, 4))
        self.assertTrue(torch.isfinite(logits).all())

    # --- gradients reach the vision modules ----------------------------------------------

    def test_loss_finite_and_grads_reach_vision_modules(self):
        # Full training step on the encoder-free path: loss is finite, and gradient flows back
        # through the scatter into BOTH new vision modules (each must have >=1 nonzero, finite
        # grad). If the gather/scatter detached the graph, these would be zero or None.
        self.model.train()
        try:
            self.model.zero_grad(set_to_none=True)
            _, loss = self.model(
                input_ids=self.input_ids,
                images=self.images,
                attention_mask=self.attention_mask,
                targets=self.targets,
            )
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for module_name in ("vision_embedder", "vision_projector"):
                module = getattr(self.model, module_name)
                got_nonzero = False
                for pname, p in module.named_parameters():
                    self.assertIsNotNone(p.grad, f"{module_name}.{pname} has no grad")
                    self.assertTrue(torch.isfinite(p.grad).all(),
                                    f"{module_name}.{pname} grad not finite")
                    if (p.grad != 0).any():
                        got_nonzero = True
                self.assertTrue(got_nonzero, f"no nonzero grad reached {module_name}")
        finally:
            self.model.eval()
            self.model.zero_grad(set_to_none=True)

    # --- extra rigor -------------------------------------------------------------------

    def test_mm_embed_dim_equals_lm_hidden_dim(self):
        # Gemma 4 architecture: the decoder (models/decoder.py) sets cfg.mm_embed_dim =
        # cfg.lm_hidden_dim = its own hidden_size when built. Guards that line against removal.
        self.assertEqual(self.model.cfg.mm_embed_dim, self.model.cfg.lm_hidden_dim)

    def test_vision_embedder_output_width_matches_mm_embed_dim(self):
        # Rigorous version: the cfg field must match the actual module. The VisionEmbedder must
        # OUTPUT vectors of width cfg.mm_embed_dim -- and since the decoder set mm_embed_dim ==
        # lm_hidden_dim, that width equals lm_hidden_dim too. If the embedder was built from a
        # different (pre-overwrite) mm_embed_dim, this catches the mismatch the cfg equality hides.
        with torch.no_grad():
            out = self.model.vision_embedder(self.pixel_values, self.image_position_ids)
        self.assertEqual(out.shape[-1], self.model.cfg.mm_embed_dim)
        self.assertEqual(out.shape[-1], self.model.cfg.lm_hidden_dim)

    def test_structural_branch_builds_encoder_free_modules(self):
        # The encoder-free model must actually build the new modules and NOT the ViT ones,
        # proving __init__ genuinely swaps paths (not just an unused flag).
        self.assertTrue(hasattr(self.model, "vision_embedder"))
        self.assertTrue(hasattr(self.model, "vision_projector"))
        self.assertFalse(hasattr(self.model, "vision_encoder"))
        self.assertFalse(hasattr(self.model, "MP"))

    def test_forward_is_deterministic(self):
        # Same inputs (eval mode, no dropout) -> identical logits.
        a, _ = self.model(input_ids=self.input_ids, images=self.images,
                          attention_mask=self.attention_mask)
        b, _ = self.model(input_ids=self.input_ids, images=self.images,
                          attention_mask=self.attention_mask)
        self.assertTrue(torch.equal(a, b))

    def test_autocast_bf16_forward_is_finite(self):
        # Training runs under autocast(bf16): the image features come out bf16 while the text
        # embeddings are fp32, so the write-in step must reconcile dtypes and stay finite.
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits, _ = self.model(input_ids=self.input_ids, images=self.images,
                                   attention_mask=self.attention_mask)
        self.assertTrue(torch.isfinite(logits).all())

    def test_count_mismatch_too_many_placeholders_raises(self):
        # Complements test_count_mismatch_raises (too few): too MANY placeholders (9 vs 8) must
        # also fail loudly, not silently.
        IMG, PAD, TXT = self.IMG, self.PAD, self.TXT
        bad_row0 = [PAD, PAD, TXT, IMG, IMG, IMG, IMG, TXT]  # 4 image tokens (was 3)
        bad_row1 = [TXT, IMG, IMG, IMG, IMG, IMG, TXT, TXT]  # 5 -> total 9 != 8
        bad_input_ids = torch.tensor([bad_row0, bad_row1], dtype=torch.long)
        with self.assertRaises(RuntimeError):
            self.model(input_ids=bad_input_ids, images=self.images,
                       attention_mask=self.attention_mask)

    def test_mixed_batch_some_samples_have_no_images(self):
        # Sample 0 has one image (3 real rows); sample 1 is pure text (no placeholders). The
        # dict carries just the 1 image. Its rows must land in sample 0; sample 1 is untouched.
        pv = torch.randn(1, N_MAX, FLAT_PATCH_DIM)
        pv[0, 3:] = 0.0
        pos = torch.full((1, N_MAX, 2), -1, dtype=torch.long)
        pos[0, :3] = torch.tensor([[0, 0], [0, 1], [1, 0]])
        images = {"pixel_values": pv, "image_position_ids": pos}

        IMG, TXT = self.IMG, self.TXT
        input_ids = torch.tensor([[TXT, IMG, IMG, IMG, TXT],
                                  [TXT, TXT, TXT, TXT, TXT]], dtype=torch.long)
        attn = torch.ones_like(input_ids)
        tok = self._capture_decoder_input(input_ids=input_ids, images=images, attention_mask=attn)

        with torch.no_grad():
            feats = self.model.vision_projector(self.model.vision_embedder(pv, pos))
        expected = feats[(pos >= 0).all(dim=-1)]  # (3, D_lm)
        img_mask = (input_ids == IMG)
        self.assertEqual(int(img_mask.sum()), 3)
        self.assertTrue(torch.allclose(tok[img_mask], expected, atol=1e-5, rtol=1e-4))
        # the text-only sample is exactly its plain token embeddings
        plain = self.model.decoder.token_embedding(input_ids)
        self.assertTrue(torch.allclose(tok[1], plain[1], atol=1e-6))

    def test_multiple_images_in_one_sample_are_ordered(self):
        # One sample, TWO images (3 then 2 real rows). Image 0's rows fill the first 3
        # placeholders, image 1's the next 2 -- intra-sample ordering is image-major.
        pv = torch.randn(2, N_MAX, FLAT_PATCH_DIM)
        pv[0, 3:] = 0.0
        pv[1, 2:] = 0.0
        pos = torch.full((2, N_MAX, 2), -1, dtype=torch.long)
        pos[0, :3] = torch.tensor([[0, 0], [0, 1], [1, 0]])
        pos[1, :2] = torch.tensor([[0, 0], [0, 1]])
        images = {"pixel_values": pv, "image_position_ids": pos}

        IMG, TXT = self.IMG, self.TXT
        input_ids = torch.tensor([[TXT, IMG, IMG, IMG, IMG, IMG, TXT]], dtype=torch.long)  # 3+2
        attn = torch.ones_like(input_ids)
        tok = self._capture_decoder_input(input_ids=input_ids, images=images, attention_mask=attn)

        with torch.no_grad():
            feats = self.model.vision_projector(self.model.vision_embedder(pv, pos))
        expected = feats[(pos >= 0).all(dim=-1)]  # (5, D_lm): image 0's 3 then image 1's 2
        img_mask = (input_ids == IMG)
        self.assertEqual(int(img_mask.sum()), 5)
        self.assertTrue(torch.allclose(tok[img_mask], expected, atol=1e-5, rtol=1e-4))

    # --- generate() (sub-step 2b) --------------------------------------------------------
    # generate only changes the PREFILL (embed the prompt + write image features into the
    # <|image|> positions); the KV-cache decode loop, sampling, and EOS handling are
    # backend-agnostic and already covered by tests/test_kv_cache_invariant.py. So the tests
    # below pin: (1) prefill embeds identically to forward, (2) it runs and is well-behaved.

    def _capture_generate_prefill(self, **generate_kwargs) -> torch.Tensor:
        """The embeddings handed to the decoder on its FIRST call inside generate = the prefill."""
        captured = {}

        def hook(module, args, kwargs):
            captured.setdefault("tok", args[0].detach().clone())  # first call only = prefill

        handle = self.model.decoder.register_forward_pre_hook(hook, with_kwargs=True)
        try:
            self.model.generate(**generate_kwargs)
        finally:
            handle.remove()
        return captured["tok"]

    def test_generate_prefill_matches_forward(self):
        # The linchpin: generate's prefill must embed the sequence (text + written-in image
        # features) exactly as forward does. If these agree, generate reuses the same count
        # rule / drop-padding / write-in, and only the (separately tested) decode loop follows.
        fwd = self._capture_decoder_input(
            input_ids=self.input_ids, images=self.images, attention_mask=self.attention_mask)
        gen = self._capture_generate_prefill(
            input_ids=self.input_ids, images=self.images, attention_mask=self.attention_mask,
            max_new_tokens=1, greedy=True)
        self.assertEqual(gen.shape, fwd.shape)
        self.assertTrue(torch.allclose(gen, fwd, atol=1e-5, rtol=1e-4),
                        "generate prefill embeddings differ from forward")

    def test_generate_runs_and_returns_finite_token_ids(self):
        out = self.model.generate(
            input_ids=self.input_ids, images=self.images, attention_mask=self.attention_mask,
            max_new_tokens=3, greedy=True)
        self.assertEqual(out.shape[0], self.input_ids.shape[0])   # one row per sample
        self.assertLessEqual(out.shape[1], 3)                     # at most max_new_tokens
        self.assertEqual(out.dtype, torch.long)                   # token ids

    def test_generate_greedy_is_deterministic(self):
        a = self.model.generate(input_ids=self.input_ids, images=self.images,
                                attention_mask=self.attention_mask, max_new_tokens=3, greedy=True)
        b = self.model.generate(input_ids=self.input_ids, images=self.images,
                                attention_mask=self.attention_mask, max_new_tokens=3, greedy=True)
        self.assertTrue(torch.equal(a, b))

    def test_generate_text_only_runs(self):
        text_ids = torch.tensor([[self.TXT, self.TXT, self.TXT]], dtype=torch.long)
        out = self.model.generate(input_ids=text_ids, images=None,
                                  attention_mask=torch.ones_like(text_ids),
                                  max_new_tokens=2, greedy=True)
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(out.dtype, torch.long)

    def test_generate_count_mismatch_raises(self):
        # The count rule must hold in prefill too. A wrong placeholder count fails loudly.
        # (Guard against a false pass: the not-yet-implemented NotImplementedError is a subclass
        # of RuntimeError, so require the raised error to NOT be NotImplementedError.)
        IMG, PAD, TXT = self.IMG, self.PAD, self.TXT
        bad = torch.tensor([[PAD, PAD, TXT, TXT, TXT, IMG, IMG, TXT],   # 2 image tokens (was 3)
                            [TXT, IMG, IMG, IMG, IMG, IMG, TXT, TXT]],  # 5 -> total 7 != 8
                           dtype=torch.long)
        with self.assertRaises(RuntimeError) as ctx:
            self.model.generate(input_ids=bad, images=self.images,
                                attention_mask=self.attention_mask, max_new_tokens=1, greedy=True)
        self.assertNotIsInstance(ctx.exception, NotImplementedError)


class TestEncoderFreeProducerConsumer(unittest.TestCase):
    """Producer->consumer contract test: the REAL ImageProcessor builds the dict, the model
    consumes it. Proves the two sides agree on the format (the gap a synthetic dict can't cover).
    Only the dataset/collator plumbing (ordering + packing) is left for sub-step 3.
    """

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_default_device("cpu")  # undo the class-scoped default-device change

    @classmethod
    def setUpClass(cls) -> None:
        torch.set_default_device(DEVICE)  # build model + fixture tensors on the GPU when available
        torch.manual_seed(0)
        # Consistent vision geometry so the processor's output width matches the embedder's input:
        # teacher_patch_size=2, pooling=2 -> model_patch_size=4 -> model_flat_patch_dim = 3*4*4 = 48.
        # max_soft_tokens must be one of the supported values (70 is the smallest).
        cls.cfg = VLMConfig(
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
        cls.processor = ImageProcessor(cls.cfg)
        cls.model = VisionLanguageModel(cls.cfg, load_backbone=False).to(DEVICE)
        cls.model.eval()
        cls.IMG = cls.model.tokenizer.image_token_id
        cls.TXT = 5
        # Two tiny images of different sizes (both sides multiples of patch_size*pooling = 4),
        # so they produce different real-row counts and exercise multi-image ordering.
        cls.img_a = torch.randint(0, 256, (3, 16, 16)).float()
        cls.img_b = torch.randint(0, 256, (3, 12, 16)).float()

    def test_processor_dict_matches_contract_and_forward_consumes_it(self):
        d = self.processor([self.img_a, self.img_b])

        # 1) the produced dict matches the shared single-source-of-truth format.
        assert_image_dict(d, num_images=2, N=self.cfg.max_soft_tokens,
                          flat_dim=self.cfg.model_flat_patch_dim)
        soft = list(d["num_soft_tokens_per_image"])
        self.assertEqual(len(soft), 2)
        self.assertTrue(all(s > 0 for s in soft))

        # 2) feed the real dict into forward with exactly sum(soft) placeholders (image 0 then 1).
        IMG, TXT = self.IMG, self.TXT
        ids = [TXT] + [IMG] * soft[0] + [IMG] * soft[1] + [TXT]
        input_ids = torch.tensor([ids], dtype=torch.long)
        attn = torch.ones_like(input_ids)

        captured = {}

        def hook(module, args, kwargs):
            captured["tok"] = args[0].detach().clone()

        handle = self.model.decoder.register_forward_pre_hook(hook, with_kwargs=True)
        try:
            logits, _ = self.model(input_ids=input_ids, images=d, attention_mask=attn)
        finally:
            handle.remove()

        self.assertTrue(torch.isfinite(logits).all())

        # 3) the written-in features equal embedder->projector->(keep real rows), in order.
        pos = d["image_position_ids"]
        with torch.no_grad():
            feats = self.model.vision_projector(self.model.vision_embedder(d["pixel_values"], pos))
        expected = feats[(pos >= 0).all(dim=-1)]
        img_mask = (input_ids == IMG)
        self.assertEqual(int(img_mask.sum()), sum(soft))
        self.assertEqual(expected.shape[0], sum(soft))
        self.assertTrue(torch.allclose(captured["tok"][img_mask], expected, atol=1e-5, rtol=1e-4))


class TestEncoderFreeEndToEndLoss(unittest.TestCase):
    """The full training path, end to end: a REAL VQADataset (encoder-free) builds a sample, the
    REAL collator batches it, and the model's forward computes a loss from the REAL labels the
    dataset produced. The check: that loss is a finite number.

    WHY THIS TEST EXISTS (the gap that let the NaN through):
    every earlier model-side test fed forward SYNTHETIC targets it built by hand, so none of them
    exercised the dataset's own label-masking. The overfit run's loss was NaN from step 0 because
    `_prepare_inputs_and_loss_mask` used `len(apply_chat_template(..., tokenize=True))` -- which
    returns a BatchEncoding, so len == 2 (its key count), every message "segment" was length 2,
    the whole loss mask came out zero, every label became -100, and cross-entropy over an
    all-ignored batch is NaN (memory `transformers5-apply-chat-template-batchencoding`). No test
    caught it because none ran the dataset's labels into the loss. This one does.

    Model-loading test -> runs on the GPU under srun (memory `model-loading-tests-on-gpu`).
    Geometry matches TestEncoderFreeProducerConsumer so the processor's patch width lines up with
    the embedder's input width.
    """

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_default_device("cpu")  # undo the class-scoped default-device change

    @classmethod
    def setUpClass(cls) -> None:
        import numpy as np
        from PIL import Image

        torch.set_default_device(DEVICE)  # build model + fixture tensors on the GPU when available
        torch.manual_seed(0)
        cls.cfg = VLMConfig(
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
        cls.model = VisionLanguageModel(cls.cfg, load_backbone=False).to(DEVICE)
        cls.processor = ImageProcessor(cls.cfg)
        # One tokenizer, shared by the dataset and the model, so image_token_id etc. agree.
        cls.tokenizer = cls.model.tokenizer

        # Two small images of different sizes (each side a multiple of patch*pooling = 4) so the
        # sample has real content in the image slots and multi-image ordering is exercised.
        rng = np.random.RandomState(0)
        cls.img_a = Image.fromarray(rng.randint(0, 256, (16, 16, 3), dtype=np.uint8))
        cls.img_b = Image.fromarray(rng.randint(0, 256, (12, 16, 3), dtype=np.uint8))

    def _raw_example(self, images, user="describe the picture", assistant="a red stop sign"):
        return {"images": images, "texts": [{"user": user, "assistant": assistant}]}

    def _batch(self, images, **text):
        """Build one sample through the real dataset, then batch it through the real collator.
        Returns the collated batch (input_ids, attention_mask, images, labels) on DEVICE."""
        ds = VQADataset([self._raw_example(images, **text)], self.tokenizer, self.processor,
                        mp_image_token_length=0, vision_backend="encoder_free")
        collator = VQACollator(self.tokenizer, max_length=1024, vision_backend="encoder_free")
        batch = collator([ds[0]])
        # tensors already land on DEVICE via the class-scoped default device; images is either the
        # {"pixel_values","image_position_ids"} dict or None (text-only), matching forward's arg.
        return batch

    def test_labels_supervise_something(self):
        # The teeth of the finite-loss check: the dataset's labels must include at least one
        # supervised (non -100) position. If they were all -100 (the old bug), the loss below
        # would be NaN -- so this guards that the finite-loss test is actually meaningful.
        batch = self._batch([self.img_a, self.img_b])
        self.assertGreater(int((batch["labels"] != -100).sum()), 0)

    def test_loss_is_finite_with_real_dataset_labels(self):
        # THE end-to-end check: real dataset -> real collator -> forward with the dataset's own
        # labels -> a finite scalar loss. This is exactly the path that produced NaN before the fix.
        batch = self._batch([self.img_a, self.img_b])
        _, loss = self.model(input_ids=batch["input_ids"], images=batch["images"],
                             attention_mask=batch["attention_mask"], targets=batch["labels"])
        self.assertEqual(loss.ndim, 0)                         # a scalar
        self.assertTrue(torch.isfinite(loss).all(), f"loss is not finite: {loss}")

    def test_text_only_batch_also_gives_finite_loss(self):
        # The images=None branch must also produce a finite loss end to end (no image slots at all).
        batch = self._batch(None)
        self.assertIsNone(batch["images"])
        _, loss = self.model(input_ids=batch["input_ids"], images=batch["images"],
                             attention_mask=batch["attention_mask"], targets=batch["labels"])
        self.assertTrue(torch.isfinite(loss).all(), f"text-only loss is not finite: {loss}")

    def test_backward_grads_are_finite_and_reach_the_vision_modules(self):
        # Gradient sanity (CLAUDE.md): the loss backpropagates to gradients that are finite AND
        # carry signal (non-zero) into the from-scratch modules we train (embedder + projector) --
        # so the image path is genuinely in the loss and actually gets a learning signal.
        # The non-zero part matters: under the all-labels-masked bug the loss is NaN but its
        # backward yields ALL-ZERO grads (every position is ignored), which are finite; requiring
        # a non-zero grad catches that failure mode too, not just outright NaN/Inf.
        batch = self._batch([self.img_a, self.img_b])
        self.model.zero_grad(set_to_none=True)
        _, loss = self.model(input_ids=batch["input_ids"], images=batch["images"],
                             attention_mask=batch["attention_mask"], targets=batch["labels"])
        loss.backward()
        for name, module in (("vision_embedder", self.model.vision_embedder),
                             ("vision_projector", self.model.vision_projector)):
            grads = [p.grad for p in module.parameters() if p.requires_grad]
            self.assertTrue(len(grads) > 0, f"{name} has no trainable params")
            self.assertTrue(any(g is not None for g in grads),
                            f"no gradient reached {name} -- the image path is detached from the loss")
            for g in grads:
                if g is not None:
                    self.assertTrue(torch.isfinite(g).all(), f"non-finite gradient in {name}")
            total = sum(float(g.abs().sum()) for g in grads if g is not None)
            self.assertGreater(total, 0.0,
                               f"gradients into {name} are all zero -- no learning signal (all labels masked?)")


if __name__ == "__main__":
    unittest.main()
