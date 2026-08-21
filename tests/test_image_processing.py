"""Behavioral / architectural spec for data.image_processing.

This is the data-side counterpart to the VisionEmbedder: it turns raw images into the exact
inputs the embedder consumes -- flattened merged-pixel patches plus their XY grid positions,
padded to a fixed length. We follow Gemma 4's image-processing CONTRACT (not its exact code),
so several tests here are ported from Gemma 4's own image-processor test
(transformers_gemma4_unified/test_image_processing_gemma4_unified.py). We do NOT inherit
transformers' ImageProcessingTestMixin -- that scaffolding (from_dict / save-load / registry)
is bound to the HF base image-processor class we deliberately don't subclass.

Assumed module surface (this is the RED spec):

    Pure helpers (reshape-only, no learned weights):
      get_aspect_ratio_preserving_size(height, width, patch_size, max_patches,
                                       pooling_kernel_size) -> (target_h, target_w)
      convert_image_to_patches(image (C,H,W), patch_size) -> (n_patches, patch_size**2 * C)
      patches_merge(patches (*,L,D), positions_xy (*,L,2), length) -> (merged (*,length,k**2*D),
                                                                       new_positions (*,length,2))
      pad_along_first_dim(image, positions, target_length) -> (padded_image, padded_positions)

    Class (constructed from cfg ONLY -- like VisionEmbedder):
      ImageProcessor(cfg)                        # reads teacher_patch_size, pooling_kernel_size,
                                                 # max_soft_tokens off cfg
      proc(images) -> dict  (a plain dict, not a BatchFeature) with keys:
          "pixel_values"                (B, max_soft_tokens, model_patch_size**2 * 3)  float
          "image_position_ids"          (B, max_soft_tokens, 2)                        int, -1 pad
          "num_soft_tokens_per_image"   list[int]  (real patch count per image)
      Accepts a single PIL image or a list of PIL images.

Output contract that ties this to the embedder: with teacher_patch_size=16, pooling_kernel_size=3
the model patch is 48x48, so the flat patch dim is 48*48*3 = 6912 == VLMConfig.model_flat_patch_dim.
"""

import unittest
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from data.image_processing import (
    ImageProcessor,
    convert_image_to_patches,
    get_aspect_ratio_preserving_size,
    pad_along_first_dim,
    patches_merge,
)


# --- Fixtures ----------------------------------------------------------------

def make_config(**overrides):
    """Minimal cfg exposing exactly the attributes ImageProcessor reads. A SimpleNamespace (not
    the full VLMConfig) keeps each test self-contained and lets a test override one field in
    isolation -- same pattern as tests/test_embedder.py. Defaults mirror Gemma 4's chosen config
    (patch 16, pooling 3, 280 soft tokens)."""
    defaults = dict(
        teacher_patch_size=16,
        pooling_kernel_size=3,
        max_soft_tokens=280,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def default_proc(**cfg_overrides):
    return ImageProcessor(make_config(**cfg_overrides))


def rgb_image(h, w, seed=0, low=0, high=256):
    """A random RGB PIL image of size (h, w). low>=1 avoids accidental all-zero pixels so the
    'padding is zero' tests can't pass by coincidence."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(low, high, size=(h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def real_mask_of(position_ids):
    """(N,2) int positions -> (N,) bool mask of real (non-padding) patches. Padding is (-1,-1),
    so a non-negative x-coordinate marks a real patch."""
    return position_ids[:, 0] >= 0


# --- Pure helper: aspect-ratio-preserving resize -----------------------------

class TestAspectRatioResize(unittest.TestCase):
    # Golden table ported verbatim from Gemma 4's test_aspect_ratio_preserving_resize_dimensions
    # (its own "C++ source of truth"). (patch_size, max_patches, pooling, H, W) -> (target_h, target_w).
    GOLDEN = [
        (16, 256, 1, 256, 256, (256, 256)),
        (16, 256, 1, 512, 512, (256, 256)),
        (10, 200, 1, 50, 10000, (10, 2000)),
        (10, 200, 1, 25, 10000, (10, 2000)),
        (16, 2304, 6, 2785, 34, (6144, 96)),
        (10, 200, 1, 25, 20000, (10, 2000)),
        (4, 64, 2, 50, 1000, (8, 128)),
        (5, 100, 3, 100, 100, (45, 45)),
        (5, 20, 3, 5, 100, (15, 30)),
    ]

    def test_matches_golden_table(self):
        for patch_size, max_patches, pool, h, w, expected in self.GOLDEN:
            got = get_aspect_ratio_preserving_size(
                height=h, width=w, patch_size=patch_size,
                max_patches=max_patches, pooling_kernel_size=pool,
            )
            self.assertEqual(tuple(got), expected, f"case {(patch_size, max_patches, pool, h, w)}")

    def test_dims_divisible_by_side_mult(self):
        # Both sides must be divisible by patch_size * pooling_kernel_size, else merging fails.
        for patch_size, max_patches, pool, h, w, _ in self.GOLDEN:
            th, tw = get_aspect_ratio_preserving_size(
                height=h, width=w, patch_size=patch_size,
                max_patches=max_patches, pooling_kernel_size=pool,
            )
            side_mult = patch_size * pool
            self.assertEqual(th % side_mult, 0)
            self.assertEqual(tw % side_mult, 0)

    def test_stays_within_patch_budget(self):
        # The resized image must never produce more than max_patches teacher patches.
        for patch_size, max_patches, pool, h, w, _ in self.GOLDEN:
            th, tw = get_aspect_ratio_preserving_size(
                height=h, width=w, patch_size=patch_size,
                max_patches=max_patches, pooling_kernel_size=pool,
            )
            self.assertLessEqual((th // patch_size) * (tw // patch_size), max_patches)


# --- Pure helper: patchify ---------------------------------------------------

class TestConvertImageToPatches(unittest.TestCase):
    def test_output_shape(self):
        # (C,H,W) -> (n_h*n_w, patch**2 * C)
        img = torch.randn(3, 12, 8)
        out = convert_image_to_patches(img, patch_size=4)
        self.assertEqual(out.shape, (2 * 3, 4 * 4 * 3))  # 3 rows x 2 cols of patches

    def test_row_major_patch_order_and_content(self):
        # Build an image where every 4x4 patch is a single constant equal to its row-major
        # index. The k-th output row must then be all-k: proves both ordering and that a patch
        # gathers the right pixels.
        p = 4
        n_h, n_w = 3, 2
        img = torch.zeros(3, n_h * p, n_w * p)
        for i in range(n_h):
            for j in range(n_w):
                img[:, i * p:(i + 1) * p, j * p:(j + 1) * p] = i * n_w + j
        out = convert_image_to_patches(img, patch_size=p)
        for idx in range(n_h * n_w):
            self.assertTrue(torch.all(out[idx] == idx), f"patch {idx} not constant/ordered")


# --- Pure helper: k x k merge ------------------------------------------------

class TestPatchesMerge(unittest.TestCase):
    def _teacher_positions(self, n_h, n_w):
        # position (x=col, y=row), row-major -- same convention the processor builds.
        pos = [[j, i] for i in range(n_h) for j in range(n_w)]
        return torch.tensor(pos).unsqueeze(0)  # (1, L, 2)

    def test_k1_is_identity(self):
        # pooling_kernel_size == 1: merging is a no-op on features and positions.
        L, D = 6, 12
        patches = torch.randn(1, L, D)
        positions = self._teacher_positions(2, 3)
        merged, new_pos = patches_merge(patches, positions, length=L)
        self.assertTrue(torch.allclose(merged, patches))
        self.assertTrue(torch.equal(new_pos, positions))

    def test_k2_dim_and_position(self):
        # 2x2 teacher grid merged into 1 model patch: dim scales by k**2, position is the
        # kernel's top-left (min) coordinate divided by k -> (0, 0).
        D = 3 * 2 * 2  # patch_size=2, 3 channels -> teacher dim 12
        patches = torch.randn(1, 4, D)
        positions = self._teacher_positions(2, 2)  # (0,0),(1,0),(0,1),(1,1)
        merged, new_pos = patches_merge(patches, positions, length=1)
        self.assertEqual(merged.shape, (1, 1, 4 * D))
        self.assertEqual(tuple(new_pos[0, 0].tolist()), (0, 0))

    def test_k2_merged_positions_grid(self):
        # 4x4 teacher grid, k=2 -> 2x2 model grid with positions {(0,0),(1,0),(0,1),(1,1)}.
        D = 12
        patches = torch.randn(1, 16, D)
        positions = self._teacher_positions(4, 4)
        _, new_pos = patches_merge(patches, positions, length=4)
        got = {tuple(p) for p in new_pos[0].tolist()}
        self.assertEqual(got, {(0, 0), (1, 0), (0, 1), (1, 1)})

    def test_fully_padded_kernel_stays_padding(self):
        # patches_merge only ever runs on full real teacher grids in the real pipeline (padding is
        # added afterwards), but its padding-preservation branch guarantees that a kernel whose
        # members are all (-1,-1) merges to a (-1,-1) position rather than a spurious real one.
        D = 12
        patches = torch.randn(1, 4, D)
        positions = torch.full((1, 4, 2), -1)
        _, new_pos = patches_merge(patches, positions, length=1)
        self.assertTrue((new_pos == -1).all())


# --- Pure helper: pad to fixed length ----------------------------------------

class TestPadAlongFirstDim(unittest.TestCase):
    def test_pads_pixels_with_zero_and_positions_with_minus_one(self):
        img = torch.randn(3, 8)
        pos = torch.randint(0, 5, (3, 2))
        padded_img, padded_pos = pad_along_first_dim(img, pos, target_length=5)
        self.assertEqual(padded_img.shape, (5, 8))
        self.assertEqual(padded_pos.shape, (5, 2))
        self.assertTrue(torch.all(padded_img[3:] == 0))
        self.assertTrue(torch.all(padded_pos[3:] == -1))
        self.assertTrue(torch.equal(padded_img[:3], img))
        self.assertTrue(torch.equal(padded_pos[:3], pos))

    def test_noop_when_already_target_length(self):
        img = torch.randn(5, 8)
        pos = torch.randint(0, 5, (5, 2))
        padded_img, padded_pos = pad_along_first_dim(img, pos, target_length=5)
        self.assertTrue(torch.equal(padded_img, img))
        self.assertTrue(torch.equal(padded_pos, pos))


# --- ImageProcessor construction ---------------------------------------------

class TestImageProcessorConstruction(unittest.TestCase):
    def test_rejects_unsupported_max_soft_tokens(self):
        # Ported from Gemma: max_soft_tokens must be one of {70,140,280,560,1120}.
        with self.assertRaises(ValueError):
            ImageProcessor(make_config(max_soft_tokens=100))

    def test_accepts_each_supported_max_soft_tokens(self):
        for mst in (70, 140, 280, 560, 1120):
            ImageProcessor(make_config(max_soft_tokens=mst))  # must not raise


# --- ImageProcessor end-to-end -----------------------------------------------

class TestImageProcessorOutput(unittest.TestCase):
    def test_output_keys(self):
        out = default_proc()(rgb_image(100, 100))
        for key in ("pixel_values", "image_position_ids", "num_soft_tokens_per_image"):
            self.assertIn(key, out)

    def test_returns_plain_dict(self):
        # We deliberately return a plain dict, not a transformers BatchFeature.
        out = default_proc()(rgb_image(100, 100))
        self.assertIs(type(out), dict)

    def test_shapes_and_flat_patch_dim(self):
        # Flat patch dim must equal (teacher_patch_size*pooling)**2 * 3 == the embedder's
        # model_flat_patch_dim (6912 for the defaults).
        out = default_proc()(rgb_image(200, 300))
        model_patch = 16 * 3
        flat = model_patch * model_patch * 3
        self.assertEqual(flat, 6912)
        self.assertEqual(out["pixel_values"].shape, (1, 280, flat))
        self.assertEqual(out["image_position_ids"].shape, (1, 280, 2))

    def test_rescale_to_unit_range_no_normalize(self):
        # do_rescale to [0,1], do_normalize=False -> real pixels land in [0, 1].
        out = default_proc()(rgb_image(96, 96, high=256))
        real = out["pixel_values"][0][real_mask_of(out["image_position_ids"][0])]
        self.assertGreaterEqual(float(real.min()), 0.0)
        self.assertLessEqual(float(real.max()), 1.0)

    def test_position_ids_structure(self):
        # Reals are non-negative, come first contiguously, then (-1,-1) padding.
        out = default_proc()(rgb_image(100, 100))
        pos = out["image_position_ids"][0]
        real = real_mask_of(pos)
        n_real = int(real.sum())
        self.assertGreater(n_real, 0)
        self.assertLessEqual(n_real, 280 * 3 ** 2)
        pad = ~real
        if pad.any():
            self.assertTrue((pos[pad] == -1).all())
            last_real = int(torch.where(real)[0][-1])
            first_pad = int(torch.where(pad)[0][0])
            self.assertEqual(last_real + 1, first_pad)  # contiguous split

    def test_padding_patches_are_zero(self):
        out = default_proc()(rgb_image(100, 100, low=1))
        pos = out["image_position_ids"][0]
        pad = ~real_mask_of(pos)
        if pad.any():
            self.assertTrue((out["pixel_values"][0][pad] == 0).all())

    def test_num_soft_tokens_matches_real_count(self):
        out = default_proc()(rgb_image(150, 220))
        n_real = int(real_mask_of(out["image_position_ids"][0]).sum())
        self.assertEqual(out["num_soft_tokens_per_image"][0], n_real)

    def test_real_patch_count_within_budget(self):
        out = default_proc(max_soft_tokens=70)(rgb_image(200, 300))
        self.assertLessEqual(int(real_mask_of(out["image_position_ids"][0]).sum()), 70 * 3 ** 2)

    def test_supported_max_soft_tokens_shapes(self):
        # Ported from Gemma's test_max_soft_tokens_values: each supported budget yields the
        # matching padded shape.
        for mst in (70, 140, 280, 560, 1120):
            out = default_proc(max_soft_tokens=mst)(rgb_image(200, 300))
            self.assertEqual(out["pixel_values"].shape, (1, mst, 6912))
            self.assertEqual(out["image_position_ids"].shape, (1, mst, 2))

    def test_determinism(self):
        img = rgb_image(120, 160)
        a = default_proc()(img)
        b = default_proc()(img)
        self.assertTrue(torch.equal(a["pixel_values"], b["pixel_values"]))
        self.assertTrue(torch.equal(a["image_position_ids"], b["image_position_ids"]))


class TestImageProcessorBatching(unittest.TestCase):
    def test_single_image_equals_list_of_one(self):
        proc = default_proc()
        img = rgb_image(128, 96)
        one = proc(img)
        as_list = proc([img])
        self.assertTrue(torch.equal(one["pixel_values"], as_list["pixel_values"]))
        self.assertTrue(torch.equal(one["image_position_ids"], as_list["image_position_ids"]))

    def test_batch_of_differing_aspect_ratios_stacks(self):
        # Different shapes produce different real counts but all pad to max_soft_tokens, so
        # they stack into one (B, N, .) tensor.
        proc = default_proc()
        out = proc([rgb_image(64, 256, seed=1), rgb_image(256, 64, seed=2), rgb_image(100, 100, seed=3)])
        self.assertEqual(out["pixel_values"].shape[0], 3)
        self.assertEqual(out["pixel_values"].shape[1], 280)
        self.assertEqual(len(out["num_soft_tokens_per_image"]), 3)

    def test_batch_rows_are_independent(self):
        # A row's output must not depend on what else is in the batch.
        proc = default_proc()
        a, b = rgb_image(80, 120, seed=4), rgb_image(200, 90, seed=5)
        joint = proc([a, b])
        solo_a = proc(a)
        self.assertTrue(torch.equal(joint["pixel_values"][0:1], solo_a["pixel_values"]))


# --- Contract link to the embedder ------------------------------------------

class TestFeedsVisionEmbedder(unittest.TestCase):
    def test_processor_output_runs_through_embedder(self):
        # The processor's pixel_values / image_position_ids must be directly consumable by
        # VisionEmbedder.forward and yield a finite (B, N, mm_embed_dim) tensor -- this is the
        # whole point of matching the contract.
        from models.vision_embedder import VisionEmbedder

        out = default_proc()(rgb_image(96, 144))
        pixel_values = out["pixel_values"]
        positions = out["image_position_ids"]

        max_pos = int(positions[positions >= 0].max())
        cfg = SimpleNamespace(
            model_flat_patch_dim=pixel_values.shape[-1],  # 6912
            mm_embed_dim=32,
            mm_posemb_size=max_pos + 1,
            emb_ln_eps=1e-5,
            pos_embd_table_initializer_range=0.02,
        )
        torch.manual_seed(0)
        embedder = VisionEmbedder(cfg)
        embedded = embedder(pixel_values, positions)
        self.assertEqual(embedded.shape, (pixel_values.shape[0], pixel_values.shape[1], 32))
        self.assertTrue(torch.isfinite(embedded).all())


if __name__ == "__main__":
    unittest.main()
