"""Tests for the encoder-free DATA pipeline (sub-step 3).

The model side is already done and tested (tests/test_encoder_free_wiring.py). These tests pin the
DATA side: turning raw images + text into the two things the model consumes on the encoder-free
path — the token string (with the right number of <|image|> placeholders) and the image dict.

The file grows one piece at a time, one planned commit each:
  * PIECE 1 (this block): the text template — get_image_string on the encoder-free path.
  * PIECE 2 (later): the dataset splits the processor's batched dict into a per-image list.
  * PIECE 3 (later): the collator combines the batch's per-image entries into the one dict the
    model wants (or None), checked against tests/contracts.py::assert_image_dict.

Ownership: the user writes the data-pipeline code; these tests define the contract it must meet.
"""

import unittest

import torch

from contracts import assert_image_dict
from data.collators import VQACollator
from data.data_utils import _is_batch_valid
from data.datasets import VQADataset

# `get_image_string_encoder_free` is imported lazily inside its test class (below) so that the
# independent collator tests in this file still collect and run before that function exists.

# Smallest cached HF tokenizer (also used by tests/test_encoder_free_wiring.py).
SMOL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"


class _FakeTokenizer:
    """Minimal stand-in. The template only reads `image_token`.

    It also carries `global_image_token` / a row-col token on purpose: the encoder-free template
    must NOT emit those even when the tokenizer has them (the 2D layout is carried numerically in
    image_position_ids, decision `encoder-free-step3-token-scaffolding`). Their presence here lets
    the tests prove they are absent from the output.
    """

    image_token = "<|image|>"
    global_image_token = "<|global_image|>"
    r1c1 = "<row_1_col_1>"


class TestGetImageStringEncoderFree(unittest.TestCase):
    """The encoder-free text template.

    Contract:
      * for each image i, emit exactly num_soft_tokens_per_image[i] copies of the image placeholder
        token, in image order.
      * placeholders only — no <|global_image|> and no row/col scaffolding tokens.
      * placeholders back to back, nothing between images (no separator). If we later decide a
        per-image separator is wanted, only the "strict concatenation" assertions below change.
      * an empty list (a text-only sample) gives the empty string.
    """

    def setUp(self):
        from data.processors import get_image_string_encoder_free
        self.get_image_string_encoder_free = get_image_string_encoder_free
        self.tok = _FakeTokenizer()

    def test_single_image_emits_exact_count(self):
        out = self.get_image_string_encoder_free(self.tok, [5])
        self.assertEqual(out, self.tok.image_token * 5)
        self.assertEqual(out.count(self.tok.image_token), 5)

    def test_multiple_images_use_their_own_counts(self):
        # Different images have different real-patch counts (aspect-preserving resize), so the
        # template must use each image's own count, not a fixed number.
        counts = [3, 7]
        out = self.get_image_string_encoder_free(self.tok, counts)
        self.assertEqual(out.count(self.tok.image_token), sum(counts))
        # strict concatenation: exactly count[0] then count[1] placeholders, nothing between.
        self.assertEqual(out, self.tok.image_token * 10)

    def test_no_scaffolding_tokens(self):
        out = self.get_image_string_encoder_free(self.tok, [4])
        self.assertNotIn(self.tok.global_image_token, out)
        self.assertNotIn(self.tok.r1c1, out)
        # nothing but the placeholder token remains once placeholders are stripped.
        self.assertEqual(out.replace(self.tok.image_token, ""), "")

    def test_empty_list_is_empty_string(self):
        self.assertEqual(self.get_image_string_encoder_free(self.tok, []), "")


class TestGetImageProcessorEncoderFree(unittest.TestCase):
    """The image-processor factory on the encoder-free path.

    New sibling function (leaves the ViT `get_image_processor` untouched): it builds the
    `ImageProcessor` from the config, since that is what the processor reads its geometry from
    (teacher_patch_size, pooling_kernel_size, max_soft_tokens).

    Contract: `get_image_processor_encoder_free(cfg)` returns an `ImageProcessor` whose geometry
    matches the config it was built from.
    """

    def setUp(self):
        from data.processors import get_image_processor_encoder_free
        from models.config import VLMConfig
        self.get_image_processor_encoder_free = get_image_processor_encoder_free
        self.cfg = VLMConfig()  # defaults: teacher_patch_size=16, pooling=3, max_soft_tokens=280

    def test_returns_image_processor_built_from_cfg(self):
        from data.image_processing import ImageProcessor
        proc = self.get_image_processor_encoder_free(self.cfg)
        self.assertIsInstance(proc, ImageProcessor)
        self.assertEqual(proc.patch_size, self.cfg.teacher_patch_size)
        self.assertEqual(proc.pooling_kernel_size, self.cfg.pooling_kernel_size)
        self.assertEqual(proc.max_soft_tokens, self.cfg.max_soft_tokens)


# ------------------------------------------------------------------------------------------------
# PIECE 3: the collator combines the batch's per-image entries into the one dict the model wants.
# ------------------------------------------------------------------------------------------------

# Tiny, hand-checkable geometry for the fake per-image entries.
_N = 4          # padded patches per image (max_soft_tokens)
_FLAT = 3       # values per patch row (model_flat_patch_dim)


class _FakeTok:
    """The collator only reads `pad_token_id` (to pad input_ids/labels/attention_mask)."""

    pad_token_id = 0


def _one_image(n_real: int, fill: float) -> dict:
    """Build ONE image's per-image entry, padded to _N, the way the dataset would after slicing the
    ImageProcessor's batched dict: `n_real` real rows (all set to `fill` so the image is
    identifiable), the rest zero-filled padding. Positions: real rows get valid coords, padding
    rows get (-1, -1). `fill` is unique per image so the collated order can be read back.
    """
    pv = torch.zeros(_N, _FLAT)
    pv[:n_real] = fill
    pos = torch.full((_N, 2), -1, dtype=torch.long)
    for r in range(n_real):
        pos[r] = torch.tensor([0, r])  # any valid (x, y); one row per real patch
    return {"pixel_values": pv, "image_position_ids": pos}


def _sample(images: list[dict], seq_len: int = 3, id_value: int = 1) -> dict:
    """A minimal dataset sample. The collator does not inspect input_ids for image counting on the
    encoder-free path, so any short ids/labels/mask are fine. `id_value` fills input_ids with a
    recognizable constant (distinct from the pad id 0) so tests can check padding; `seq_len` lets a
    test make a sample long enough to be discarded.
    """
    ids = torch.full((seq_len,), id_value, dtype=torch.long)
    return {
        "input_ids": ids,
        "labels": ids.clone(),
        "attention_mask": torch.ones(seq_len, dtype=torch.long),
        "images": images,
    }


class TestVQACollatorEncoderFree(unittest.TestCase):
    """The collator turns per-sample lists of per-image entries into the single dict the model
    consumes (or None for a fully text-only batch).

    -------------------------------------------------------------------------------------------
    INTERFACE THIS TEST PINS (adjustable -- tell me and I'll update it before you write the code):

      * per-sample ``sample["images"]`` on the encoder-free path is a LIST, one entry per image,
        each entry a dict ``{"pixel_values": (N, flat) float, "image_position_ids": (N, 2) int}``
        (the dataset produces this by slicing the ImageProcessor's batched dict; a text-only sample
        carries ``[]``). Keeping it a per-image list -- not the batched dict -- is what lets the
        packing step's ``len(sample["images"])`` keep counting images.
      * the collator learns the backend from an additive constructor arg
        ``VQACollator(tokenizer, max_length, vision_backend="vit")`` (default keeps the ViT path,
        so every existing test is untouched).
      * on the encoder-free path the collated ``batch["images"]`` is EITHER
          - a dict ``{"pixel_values": (total_images, N, flat), "image_position_ids": (total_images,
            N, 2)}`` with the images stacked in the order their <|image|> placeholders appear
            (sample by sample, image within sample), OR
          - ``None`` when the whole batch is text-only (it must NOT build an empty dict / stack []).
      * the images dict is built from the SURVIVING samples only: a sample dropped for being longer
        than ``max_length`` takes its images with it (text and images must stay aligned), so the
        image combine happens AFTER the over-length discard.
    -------------------------------------------------------------------------------------------
    """

    def _collate(self, batch: list[dict]) -> dict:
        collator = VQACollator(_FakeTok(), max_length=8, vision_backend="encoder_free")
        return collator(batch)

    def test_output_dict_matches_contract(self):
        # Two samples, three images total (2 in sample 0, 1 in sample 1).
        batch = [
            _sample([_one_image(2, 1.0), _one_image(3, 2.0)]),
            _sample([_one_image(1, 3.0)]),
        ]
        out = self._collate(batch)
        self.assertIsInstance(out["images"], dict)
        assert_image_dict(out["images"], num_images=3, N=_N, flat_dim=_FLAT)

    def test_images_are_stacked_in_placeholder_order(self):
        # The silent-failure trap: a wrong ORDER mis-places every image's features and nothing
        # raises (only a wrong COUNT raises, downstream in the model). So pin the order explicitly.
        # Order must be sample-major, image-within-sample: A, B (sample 0), then C (sample 1).
        A, B, C = _one_image(2, 1.0), _one_image(3, 2.0), _one_image(1, 3.0)
        out = self._collate([_sample([A, B]), _sample([C])])

        pv = out["images"]["pixel_values"]          # (3, N, flat)
        pos = out["images"]["image_position_ids"]   # (3, N, 2)
        self.assertEqual(tuple(pv.shape), (3, _N, _FLAT))
        for slot, img in enumerate((A, B, C)):
            self.assertTrue(torch.equal(pv[slot], img["pixel_values"]),
                            f"image in slot {slot} is out of order (pixel_values mismatch)")
            self.assertTrue(torch.equal(pos[slot], img["image_position_ids"]),
                            f"image in slot {slot} is out of order (positions mismatch)")

    def test_text_only_batch_gives_none(self):
        # No images anywhere: the collator must return None, never call the processor or stack [].
        out = self._collate([_sample([]), _sample([])])
        self.assertIsNone(out["images"])

    def test_mixed_batch_keeps_only_real_images_in_order(self):
        # sample 0 text-only, sample 1 has one image, sample 2 has two. Expect 3 images, in order.
        C = _one_image(1, 3.0)
        D, E = _one_image(2, 4.0), _one_image(3, 5.0)
        out = self._collate([_sample([]), _sample([C]), _sample([D, E])])
        assert_image_dict(out["images"], num_images=3, N=_N, flat_dim=_FLAT)
        pv = out["images"]["pixel_values"]
        for slot, img in enumerate((C, D, E)):
            self.assertTrue(torch.equal(pv[slot], img["pixel_values"]),
                            f"image in slot {slot} is out of order")

    def test_single_image_single_sample(self):
        # Smallest non-empty case: one sample, one image -> a (1, N, ...) dict.
        A = _one_image(2, 1.0)
        out = self._collate([_sample([A])])
        assert_image_dict(out["images"], num_images=1, N=_N, flat_dim=_FLAT)
        self.assertTrue(torch.equal(out["images"]["pixel_values"][0], A["pixel_values"]))
        self.assertTrue(torch.equal(out["images"]["image_position_ids"][0], A["image_position_ids"]))

    def test_ordering_across_larger_varied_batch(self):
        # Stronger ordering stress: 4 samples with image counts [2, 0, 3, 1] -> 6 images total.
        # Each image has a unique fill (1..6) and a varied real-row count, so any mis-ordering or
        # mis-count across the whole batch is caught, not just a 3-image swap.
        layout = [2, 0, 3, 1]
        expected: list[dict] = []
        samples: list[dict] = []
        counter = 0
        for n in layout:
            entries = []
            for _ in range(n):
                counter += 1
                entries.append(_one_image(n_real=1 + counter % 3, fill=float(counter)))
            samples.append(_sample(entries))
            expected.extend(entries)
        out = self._collate(samples)
        assert_image_dict(out["images"], num_images=6, N=_N, flat_dim=_FLAT)
        pv = out["images"]["pixel_values"]
        pos = out["images"]["image_position_ids"]
        for slot, img in enumerate(expected):
            self.assertTrue(torch.equal(pv[slot], img["pixel_values"]), f"slot {slot} out of order")
            self.assertTrue(torch.equal(pos[slot], img["image_position_ids"]), f"slot {slot} pos")

    def test_over_length_sample_is_dropped_with_its_images(self):
        # The highest-risk path: a sample longer than max_length is discarded ([collators.py]
        # _discard_samples_that_are_too_long), and its images MUST go with it. If the image combine
        # ran before the discard (or ignored it), image X would leak in and text<->image alignment
        # would break silently. Survivors are samples 0 and 2 -> images A, B, C (X gone).
        A, B = _one_image(2, 1.0), _one_image(3, 2.0)   # sample 0 (kept)
        X = _one_image(2, 9.0)                           # sample 1 (over-length -> dropped)
        C = _one_image(1, 3.0)                           # sample 2 (kept)
        out = self._collate([
            _sample([A, B], seq_len=3),
            _sample([X], seq_len=20),                    # 20 > max_length 8 -> discarded
            _sample([C], seq_len=3),
        ])
        assert_image_dict(out["images"], num_images=3, N=_N, flat_dim=_FLAT)  # 3, not 4
        pv = out["images"]["pixel_values"]
        for slot, img in enumerate((A, B, C)):
            self.assertTrue(torch.equal(pv[slot], img["pixel_values"]), f"slot {slot} out of order")
        self.assertFalse(bool((pv == 9.0).any()), "dropped sample's image leaked into the batch")
        self.assertEqual(out["input_ids"].shape[0], 2, "only the 2 surviving samples' text remains")

    def test_text_tensors_are_stacked_and_padded_correctly(self):
        # The image combine must not disturb the text tensors: input_ids/labels/attention_mask are
        # still left-padded to max_length, with the originals intact -- alongside a valid image dict.
        s0 = _sample([_one_image(2, 1.0)], seq_len=3, id_value=7)
        s1 = _sample([_one_image(1, 2.0)], seq_len=5, id_value=8)
        out = self._collate([s0, s1])  # max_length = 8
        for key in ("input_ids", "labels", "attention_mask"):
            self.assertEqual(tuple(out[key].shape), (2, 8), f"{key} not stacked/padded to (2, 8)")
        # left padding: originals sit at the RIGHT, pad id 0 fills the left.
        self.assertTrue(torch.equal(out["input_ids"][0, -3:], torch.full((3,), 7)))
        self.assertTrue(torch.equal(out["input_ids"][1, -5:], torch.full((5,), 8)))
        self.assertTrue(bool((out["input_ids"][0, :5] == _FakeTok.pad_token_id).all()))
        self.assertTrue(bool((out["attention_mask"][0, :5] == 0).all()))   # pad positions masked
        self.assertTrue(bool((out["attention_mask"][0, 5:] == 1).all()))   # real positions attended
        assert_image_dict(out["images"], num_images=2, N=_N, flat_dim=_FLAT)

    def test_repeated_call_is_deterministic(self):
        # Same inputs -> byte-identical output dict (no hidden ordering nondeterminism).
        def build():
            return [_sample([_one_image(2, 1.0), _one_image(3, 2.0)]), _sample([_one_image(1, 3.0)])]
        a = self._collate(build())["images"]
        b = self._collate(build())["images"]
        self.assertTrue(torch.equal(a["pixel_values"], b["pixel_values"]))
        self.assertTrue(torch.equal(a["image_position_ids"], b["image_position_ids"]))

    def test_does_not_mutate_input_image_entries(self):
        # The collator must not modify the caller's per-image tensors in place.
        A = _one_image(2, 1.0)
        pv_before = A["pixel_values"].clone()
        pos_before = A["image_position_ids"].clone()
        self._collate([_sample([A])])
        self.assertTrue(torch.equal(A["pixel_values"], pv_before))
        self.assertTrue(torch.equal(A["image_position_ids"], pos_before))

    def test_empty_batch_does_not_produce_an_image_dict(self):
        # A fully empty batch (no samples at all) must not fabricate an image dict or crash.
        out = self._collate([])
        self.assertNotIsInstance(out["images"], dict)

    def test_vit_default_leaves_images_as_list(self):
        # Additivity guard: with the default backend the collator behaves exactly as before --
        # images pass through as the per-sample list, so no existing ViT test changes.
        collator = VQACollator(_FakeTok(), max_length=8)  # default vision_backend="vit"
        out = collator([_sample([_one_image(2, 1.0)]), _sample([])])
        self.assertIsInstance(out["images"], list)
        self.assertEqual(len(out["images"]), 2)


# ------------------------------------------------------------------------------------------------
# PIECE 2: the dataset turns ONE raw example (text + images) into ONE finished sample.
# ------------------------------------------------------------------------------------------------


def _raw_example(images, user="describe the picture", assistant="a cat"):
    """A minimal raw dataset item in the shape VQADataset reads: `images` (a list, or None for
    text-only) and `texts` (a list of user/assistant turns). No rating keys -> nothing filtered.
    """
    return {"images": images, "texts": [{"user": user, "assistant": assistant}]}


class _EncoderFreeDatasetFixture(unittest.TestCase):
    """Shared fixture for the encoder-free VQADataset tests (both the internal-method tests and the
    finished-output tests below inherit it).

    Builds a real (cached, small) tokenizer + the real ImageProcessor once; no model is loaded, so
    this runs on the login node. Has no tests of its own.

    -------------------------------------------------------------------------------------------
    INTERFACE THESE TESTS PIN (adjustable -- tell me and I'll update it):

      * ``VQADataset`` gains an additive keyword arg ``vision_backend`` (default keeps the ViT
        path). On ``"encoder_free"`` the passed ``image_processor`` is an ``ImageProcessor``.
      * INTERNAL SEAMS (settled 2026-07-11 -- all backend dispatch lives in ``_process_data``):
          - ``_process_images_encoder_free(images)`` -- a NEW sibling of the ViT ``_process_images``
            (which is left untouched). Returns a tuple ``(per_image_list, num_soft_tokens_per_image)``,
            mirroring the ViT return ``(processed_images, splitted_image_counts)``. Each per-image
            entry is ``{"pixel_values": (N, flat), "image_position_ids": (N, 2)}``.
          - ``_get_messages(item, image_string)`` -- now backend-AGNOSTIC (one shared method): it
            just prepends the already-built ``image_string`` to the first message. ``_process_data``
            builds the right string per backend (``get_image_string`` vs
            ``get_image_string_encoder_free``) and passes it in.
      * per-sample ``images`` in the finished sample is that same per-image list (matches the
        collator, PIECE 3).
    -------------------------------------------------------------------------------------------
    """

    @classmethod
    def setUpClass(cls) -> None:
        import numpy as np
        from PIL import Image

        from data.image_processing import ImageProcessor
        from data.processors import get_tokenizer
        from models.config import VLMConfig

        # Small, consistent vision geometry (teacher_patch=2, pooling=2 -> flat_dim = 3*4*4 = 48).
        cls.cfg = VLMConfig(
            vision_backend="encoder_free",
            lm_tokenizer=SMOL_ID,
            teacher_patch_size=2,
            pooling_kernel_size=2,
            model_patch_size=4,
            model_flat_patch_dim=48,
            max_soft_tokens=70,
            mm_embed_dim=8,
            mm_posemb_size=64,
        )
        cls.N = cls.cfg.max_soft_tokens          # 70
        cls.FLAT = cls.cfg.model_flat_patch_dim  # 48
        cls.processor = ImageProcessor(cls.cfg)
        cls.tokenizer = get_tokenizer(cls.cfg.lm_tokenizer, cls.cfg.vlm_extra_tokens,
                                      cls.cfg.lm_chat_template)
        cls.IMG_ID = cls.tokenizer.image_token_id

        # Two images of different sizes -> different real-patch counts, so ordering matters and the
        # count rule is a non-trivial check. Fixed seed -> deterministic.
        rng = np.random.RandomState(0)
        cls.img_a = Image.fromarray(rng.randint(0, 256, (16, 16, 3), dtype=np.uint8))
        cls.img_b = Image.fromarray(rng.randint(0, 256, (12, 16, 3), dtype=np.uint8))
        # Ground-truth per-image real-patch counts, straight from the processor (currently [64, 63]).
        ref = cls.processor([cls.img_a, cls.img_b])
        cls.expected_real = list(ref["num_soft_tokens_per_image"])
        assert cls.expected_real[0] != cls.expected_real[1], "pick sizes with distinct counts"

    def _build(self, raw_items):
        return VQADataset(raw_items, self.tokenizer, self.processor,
                          mp_image_token_length=0, vision_backend="encoder_free")

    def _real_rows(self, entry) -> int:
        pos = entry["image_position_ids"]
        return int((pos >= 0).all(dim=-1).sum())


class TestProcessImagesEncoderFree(_EncoderFreeDatasetFixture):
    """INTERNAL seam: `_process_images_encoder_free` -- turn raw images into the per-image list +
    the per-image real-patch counts. A new sibling; the ViT `_process_images` is left untouched.
    """

    def test_returns_per_image_list_and_counts(self):
        # Returns a tuple, mirroring the ViT `_process_images`: first the per-image list (one dict
        # per image, right shapes), second the per-image real-patch counts.
        ds = self._build([_raw_example([self.img_a, self.img_b])])
        per_image_list, counts = ds._process_images_encoder_free([self.img_a, self.img_b])
        self.assertIsInstance(per_image_list, list)
        self.assertEqual(len(per_image_list), 2)  # one entry per image, not one stacked bundle
        for entry in per_image_list:
            self.assertIsInstance(entry, dict)
            self.assertEqual(tuple(entry["pixel_values"].shape), (self.N, self.FLAT))
            self.assertEqual(tuple(entry["image_position_ids"].shape), (self.N, 2))
        self.assertEqual(list(counts), self.expected_real)  # matches the processor

    def test_counts_agree_with_the_lists_real_rows(self):
        # Internal consistency: the counts it reports equal the real (non-filler) rows it actually
        # put in each per-image entry -- so the text and the image data can never disagree.
        ds = self._build([_raw_example([self.img_a, self.img_b])])
        per_image_list, counts = ds._process_images_encoder_free([self.img_a, self.img_b])
        self.assertEqual([self._real_rows(e) for e in per_image_list], list(counts))


class TestGetMessagesTakesImageString(_EncoderFreeDatasetFixture):
    """INTERNAL seam: `_get_messages(item, image_string)` -- now backend-agnostic. It builds the
    chat turns and prepends the ALREADY-BUILT `image_string` to the first message. It does not know
    about backends or counts; `_process_data` builds the right string and passes it in.
    """

    def test_prepends_the_given_image_string(self):
        # The first (user) message becomes: image_string + original user text, verbatim.
        item = _raw_example([self.img_a], user="hello")
        image_string = self.tokenizer.image_token * 5
        messages = self._build([item])._get_messages(item, image_string)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], image_string + "hello")
        self.assertEqual(messages[0]["content"].count(self.tokenizer.image_token), 5)

    def test_empty_string_leaves_the_text_unchanged(self):
        # Text-only example: _process_data passes "" and nothing is prepended.
        item = _raw_example(None, user="just text")
        messages = self._build([item])._get_messages(item, "")
        self.assertEqual(messages[0]["content"], "just text")
        self.assertEqual(messages[0]["content"].count(self.tokenizer.image_token), 0)

    def test_strips_stray_image_tokens_from_source_text_before_prepending(self):
        # If the raw user text already contains an image placeholder, the safety check removes it
        # first, so junk in the source can't inflate the placeholder count. Only the prepended run
        # (3 here) survives.
        stray = self.tokenizer.image_token
        item = _raw_example([self.img_a], user=f"a{stray}b")
        image_string = self.tokenizer.image_token * 3
        messages = self._build([item])._get_messages(item, image_string)
        self.assertEqual(messages[0]["content"].count(self.tokenizer.image_token), 3)
        self.assertEqual(messages[0]["content"], image_string + "ab")


class TestVQADatasetEncoderFree(_EncoderFreeDatasetFixture):
    """The dataset's finished-sample (public output) contract on the encoder-free path.

    Given a raw example, `VQADataset` must return a sample whose:
      * ``images`` is a LIST, one entry per image, each the per-image dict
        ``{"pixel_values": (N, flat), "image_position_ids": (N, 2)}`` -- the exact shape the
        collator (PIECE 3) consumes, so the two sides provably agree;
      * ``input_ids`` contains exactly as many <|image|> placeholder tokens as there are real
        (non-filler) patches across those images -- the one rule that must hold.
    A text-only example yields ``images == []`` and no placeholders.
    """

    def test_images_is_a_per_image_list_with_the_right_shapes(self):
        ds = self._build([_raw_example([self.img_a, self.img_b])])
        sample = ds[0]
        self.assertIsInstance(sample["images"], list)
        self.assertEqual(len(sample["images"]), 2)  # one entry per image, not one stacked bundle
        for entry in sample["images"]:
            self.assertIsInstance(entry, dict)
            self.assertEqual(tuple(entry["pixel_values"].shape), (self.N, self.FLAT))
            self.assertEqual(tuple(entry["image_position_ids"].shape), (self.N, 2))

    def test_placeholder_count_equals_total_real_patches(self):
        ds = self._build([_raw_example([self.img_a, self.img_b])])
        sample = ds[0]
        n_placeholders = int((sample["input_ids"] == self.IMG_ID).sum())
        total_real = sum(self._real_rows(e) for e in sample["images"])
        self.assertEqual(n_placeholders, total_real)              # the rule that must hold
        self.assertEqual(n_placeholders, sum(self.expected_real))  # and it matches the processor

    def test_per_image_real_counts_match_the_processor(self):
        ds = self._build([_raw_example([self.img_a, self.img_b])])
        sample = ds[0]
        self.assertEqual([self._real_rows(e) for e in sample["images"]], self.expected_real)

    def test_text_only_example_has_no_images_and_no_placeholders(self):
        ds = self._build([_raw_example(None)])
        sample = ds[0]
        self.assertEqual(sample["images"], [])
        self.assertEqual(int((sample["input_ids"] == self.IMG_ID).sum()), 0)

    def test_dataset_output_feeds_the_collator(self):
        # The two sides fit: the dataset's per-image list, run through the collator, yields the one
        # dict the model consumes -- proving the dataset and collator agree on the format.
        ds = self._build([_raw_example([self.img_a, self.img_b])])
        sample = ds[0]
        collator = VQACollator(self.tokenizer, max_length=1024, vision_backend="encoder_free")
        batch = collator([sample])
        assert_image_dict(batch["images"], num_images=2, N=self.N, flat_dim=self.FLAT)


class TestLossMaskEncoderFree(_EncoderFreeDatasetFixture):
    """Regression + correctness for the training loss mask (the `labels` a sample carries).

    THE bug this guards (memory `transformers5-apply-chat-template-batchencoding`):
    `apply_chat_template([msg], tokenize=True)` returns a BatchEncoding, so `len(...)` == 2 (the
    key count) instead of the token count. That makes every per-message segment length 2, the loss
    mask all-zero, every label -100, and the loss NaN from step 0 -- exactly the overfit run's
    failure. These tests fail loudly if that regresses. The mask lives in BaseDataset (shared, not
    encoder-free-specific); we exercise it through the encoder-free dataset.
    """

    def _labels(self, images, user="describe the picture", assistant="a red stop sign"):
        sample = self._build([_raw_example(images, user=user, assistant=assistant)])[0]
        return sample["input_ids"], sample["labels"]

    def test_not_all_labels_are_masked(self):
        # THE regression: at least one supervised (non -100) label. All -100 -> loss is NaN.
        _, labels = self._labels(None)
        self.assertGreater(int((labels != -100).sum()), 0,
                           "every label is -100 -> loss would be NaN (BatchEncoding seg_len bug?)")

    def test_some_labels_are_masked(self):
        # The prompt side (user turn, image placeholders, padding) must NOT be supervised, so the
        # mask is a proper subset -- some -100 and some not.
        _, labels = self._labels(None)
        self.assertGreater(int((labels == -100).sum()), 0)

    def test_supervised_count_scales_with_answer_length(self):
        # The sharpest check: a longer assistant answer -> strictly more supervised tokens. Under
        # the bug (constant seg_len == 2) the count would NOT grow with the answer, so this catches
        # the exact failure even if a few positions happened to be supervised by accident.
        _, short = self._labels(None, assistant="yes")
        _, long = self._labels(None, assistant="yes it is a long and detailed answer with many words")
        n_short = int((short != -100).sum())
        n_long = int((long != -100).sum())
        self.assertGreater(n_short, 0)
        self.assertGreater(n_long, n_short)

    def test_last_label_is_ignored(self):
        # `_get_labels` sets the final position to -100 (no next token to predict after it).
        _, labels = self._labels(None)
        self.assertEqual(int(labels[-1]), -100)

    def test_image_placeholders_are_never_prediction_targets(self):
        # For a sample WITH an image, no supervised label is the <|image|> id -- the model is never
        # trained to predict an image placeholder (they sit in the masked user turn).
        _, labels = self._labels([self.img_a])
        supervised = set(labels[labels != -100].tolist())
        self.assertNotIn(self.IMG_ID, supervised)


class TestIsBatchValid(unittest.TestCase):
    """`data_utils._is_batch_valid` is the guard the DDP loop uses to SKIP invalid batches. It must
    understand BOTH collator output shapes:
      * encoder-free: batch["images"] is a dict of tensors, or None for a no-image batch.
      * ViT:          batch["images"] is a list of lists of tensors, or [] for a no-image batch.
    Regression for the crash in run 22366827: a text-only encoder-free batch has images=None, and
    _is_batch_valid did `len(None)` -> TypeError, killing the rank INSIDE the safety check instead
    of returning False so the batch is skipped.
    """

    def _collate_encoder_free(self, batch: list[dict]) -> dict:
        return VQACollator(_FakeTok(), max_length=8, vision_backend="encoder_free")(batch)

    # --- encoder-free path (dict / None) ---------------------------------------------------
    def test_text_only_encoder_free_batch_is_invalid_not_crash(self):
        # The exact crash: the real collator emits images=None for an all-text batch; the guard
        # must return False (skip), never raise.
        out = self._collate_encoder_free([_sample([]), _sample([])])
        self.assertIsNone(out["images"])           # precondition: this is the None-images batch
        self.assertIs(_is_batch_valid(out), False)

    def test_image_encoder_free_batch_is_valid(self):
        out = self._collate_encoder_free([_sample([_one_image(2, 1.0)])])
        self.assertIsInstance(out["images"], dict)  # precondition: a real images dict
        self.assertIs(_is_batch_valid(out), True)

    def test_empty_encoder_free_batch_is_invalid(self):
        # Empty input -> collator returns empty lists + images=None; must be invalid, not raise.
        out = self._collate_encoder_free([])
        self.assertIs(_is_batch_valid(out), False)

    # --- ViT path (list-of-lists / []) preserved -------------------------------------------
    def test_vit_batch_with_image_is_valid(self):
        batch = {"input_ids": [[1, 2, 3]], "images": [[torch.zeros(3, 4, 4)]]}
        self.assertIs(_is_batch_valid(batch), True)

    def test_vit_empty_images_batch_is_invalid(self):
        self.assertIs(_is_batch_valid({"input_ids": [[1, 2, 3]], "images": []}), False)

    def test_vit_all_none_images_batch_is_invalid(self):
        # images present as a list, but no actual image in any sample -> invalid (DDP-deadlock guard).
        self.assertIs(_is_batch_valid({"input_ids": [[1, 2, 3]], "images": [[]]}), False)


if __name__ == "__main__":
    unittest.main()
