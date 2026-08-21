"""Behavioral / architectural spec for models.vision_embedder.VisionEmbedder.

The VisionEmbedder follows Gemma 4's encoder-free vision embedder *architecture* up through
`pos_norm` (the final RMSNorm->Linear into LM space is a separate MP module, added at the
VLM-wiring step). We test architectural PROPERTIES, not bit-equivalence to Gemma 4 (a
non-gating equivalence cross-check lives in test_embedder_equivalence.py).

Input contract (matches Gemma 4, see Gemma4UnifiedVision2TextModelTester):
  pixel_values      : (batch, patches, model_flat_patch_dim)  float
  image_position_ids: (batch, patches, 2)                     int, -1 marks padding
  output            : (batch, patches, mm_embed_dim)          float
"""

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from models.vision_embedder import VisionEmbedder


# --- Test fixtures -----------------------------------------------------------

def make_config(**overrides):
    """Minimal config exposing exactly the attributes VisionEmbedder reads.

    A SimpleNamespace (not the full VLMConfig) keeps each test self-contained and lets a
    test override one dimension in isolation. Small defaults are hand-checkable:
    flat patch dim 12, embed dim 8, pos table size 10.
    """
    defaults = dict(
        model_flat_patch_dim=12,
        mm_embed_dim=8,
        mm_posemb_size=10,
        emb_ln_eps=1e-5,
        pos_embd_table_initializer_range=0.02,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def make_embedder(seed=0, **overrides):
    """Seeded VisionEmbedder so parameter initialization is reproducible."""
    torch.manual_seed(seed)
    return VisionEmbedder(make_config(**overrides))


def make_linear_embedder(seed=0, **overrides):
    """A VisionEmbedder with its three LayerNorms swapped out for do-nothing layers.

    The LayerNorms bend the numbers in a way that is hard to predict by hand. With them
    removed, the whole forward pass is just `dense(x) + position_embedding`, so a test can
    compute the exact expected output and check the position part on its own.
    """
    embedder = make_embedder(seed, **overrides)
    embedder.patch_ln1 = nn.Identity()
    embedder.patch_ln2 = nn.Identity()
    embedder.pos_norm = nn.Identity()
    return embedder


def random_positions(batch, patches, table_size, seed=0):
    """Valid (batch, patches, 2) int64 XY positions in [0, table_size)."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, table_size, (batch, patches, 2), generator=generator)


def reference_pos(pos_table, positions):
    """A second, deliberately simple version of the position-embedding lookup, written with
    a plain Python loop so the tests can compare it against the module's tensor version. If
    both agree, the module's compact indexing does what the obvious loop does.

    For a patch at grid position (x, y), its position embedding is
    pos_table[x, 0] + pos_table[y, 1] -- the vector stored for that x plus the vector stored
    for that y. A coordinate of -1 means "padding" and adds nothing.
    """
    batch, patches, _ = positions.shape
    embed_dim = pos_table.shape[-1]
    out = torch.zeros(batch, patches, embed_dim, dtype=pos_table.dtype)
    for b in range(batch):
        for n in range(patches):
            x = int(positions[b, n, 0])
            y = int(positions[b, n, 1])
            if x >= 0:
                out[b, n] += pos_table[x, 0]
            if y >= 0:
                out[b, n] += pos_table[y, 1]
    return out


# --- Shape and dtype ---------------------------------------------------------

class TestShapesAndDtype(unittest.TestCase):
    def test_output_shape(self):
        # (B, N, flat_patch_dim) -> (B, N, mm_embed_dim)
        embedder = make_embedder()
        out = embedder(torch.randn(2, 5, 12), random_positions(2, 5, 10))
        self.assertEqual(out.shape, (2, 5, 8))

    def test_single_patch(self):
        # Degenerate (1, 1, .) batch still yields one embedding row.
        embedder = make_embedder()
        out = embedder(torch.randn(1, 1, 12), random_positions(1, 1, 10))
        self.assertEqual(out.shape, (1, 1, 8))

    def test_output_dtype_follows_weights_not_input(self):
        # forward casts the input to the dense weight dtype (Gemma 4 behavior), so a
        # float32-weighted module returns float32 even from a float64 input.
        embedder = make_embedder()
        out = embedder(torch.randn(2, 4, 12, dtype=torch.float64), random_positions(2, 4, 10))
        self.assertEqual(out.dtype, torch.float32)

    def test_runs_in_double_precision(self):
        # A float64 module up-casts a float32 input: dtype tracks the weights either way.
        embedder = make_embedder().double()
        out = embedder(torch.randn(2, 4, 12, dtype=torch.float32), random_positions(2, 4, 10))
        self.assertEqual(out.dtype, torch.float64)


# --- Padding (-1) masking ----------------------------------------------------

class TestPaddingMasking(unittest.TestCase):
    def test_full_padding_row_gets_zero_positional_term(self):
        # A patch at (-1, -1) must receive no positional contribution: with identity
        # norms its output is exactly dense(x).
        embedder = make_linear_embedder()
        x = torch.randn(1, 3, 12)
        positions = random_positions(1, 3, 10)
        positions[0, 1] = torch.tensor([-1, -1])
        out = embedder(x, positions)
        self.assertTrue(torch.allclose(out[0, 1], embedder.patch_dense(x[0, 1]), atol=1e-6))

    def test_partial_padding_masks_only_that_axis(self):
        # (x, -1) contributes table[x, 0] only; the y term is dropped.
        embedder = make_linear_embedder()
        x = torch.zeros(1, 1, 12)
        table = embedder.pos_embedding.detach()
        out = embedder(x, torch.tensor([[[2, -1]]]))
        baseline = embedder.patch_dense(x[0, 0]).detach()  # dense(0) == bias
        self.assertTrue(torch.allclose(out[0, 0] - baseline, table[2, 0], atol=1e-6))

    def test_matches_reference_with_padding(self):
        # Same comparison as test_matches_reference_implementation below, but with some
        # padding (-1) rows mixed in, so padding is covered by the full comparison too.
        embedder = make_linear_embedder()
        x = torch.randn(2, 4, 12)
        positions = random_positions(2, 4, 10)
        positions[0, 2] = torch.tensor([-1, -1])
        positions[1, 0, 1] = -1
        out = embedder(x, positions)
        expected = embedder.patch_dense(x) + reference_pos(embedder.pos_embedding.detach(), positions)
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_valid_row_unaffected_by_another_rows_padding(self):
        # Per-row independence: turning one row into padding must not change other rows.
        embedder = make_embedder()
        x = torch.randn(1, 5, 12)
        positions = random_positions(1, 5, 10)
        out = embedder(x, positions)

        positions2 = positions.clone()
        positions2[0, 2] = torch.tensor([-1, -1])
        out2 = embedder(x, positions2)

        untouched = [i for i in range(5) if i != 2]
        self.assertTrue(torch.allclose(out[0, untouched], out2[0, untouched], atol=1e-6))
        self.assertFalse(torch.allclose(out[0, 2], out2[0, 2], atol=1e-6))

    def test_real_pipeline_padding_rows_are_finite(self):
        # Mirror the image processor's output: padding rows have zero pixels + (-1,-1)
        # positions. Output must be finite everywhere and correctly shaped.
        embedder = make_embedder()
        x = torch.randn(1, 4, 12)
        x[0, 3] = 0.0
        positions = random_positions(1, 4, 10)
        positions[0, 3] = torch.tensor([-1, -1])
        out = embedder(x, positions)
        self.assertEqual(out.shape, (1, 4, 8))
        self.assertTrue(torch.isfinite(out).all())


# --- Factorized positional embedding semantics -------------------------------

class TestPositionalEmbedding(unittest.TestCase):
    def test_matches_reference_implementation(self):
        # With the LayerNorms removed the output is exactly dense(x) + the position part, so
        # compare it against the simple Python-loop version of the position lookup. If they
        # match, the module's compact tensor indexing computes the intended thing.
        embedder = make_linear_embedder()
        x = torch.randn(2, 6, 12)
        positions = random_positions(2, 6, 10)
        out = embedder(x, positions)
        expected = embedder.patch_dense(x) + reference_pos(embedder.pos_embedding.detach(), positions)
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_axis_zero_is_x_axis_one_is_y(self):
        # Pin the axis convention: with a hand-built table and zero input, the positional
        # term is table[x, 0] + table[y, 1]. Swapping coordinates reads different slots.
        embedder = make_linear_embedder()
        embed_dim = embedder.cfg.mm_embed_dim
        with torch.no_grad():
            embedder.pos_embedding.zero_()
            embedder.pos_embedding[2, 0] = torch.arange(embed_dim, dtype=torch.float32)  # x=2
            embedder.pos_embedding[3, 1] = torch.full((embed_dim,), 5.0)                 # y=3

        zero = torch.zeros(1, 1, embedder.cfg.model_flat_patch_dim)
        baseline = embedder.patch_dense(zero[0, 0]).detach()  # dense(0) == bias

        out_hit = embedder(zero, torch.tensor([[[2, 3]]]))
        self.assertTrue(torch.allclose(
            out_hit[0, 0] - baseline, torch.arange(embed_dim, dtype=torch.float32) + 5.0, atol=1e-6))

        out_miss = embedder(zero, torch.tensor([[[3, 2]]]))  # reads table[3,0], table[2,1], both zero
        self.assertTrue(torch.allclose(out_miss[0, 0] - baseline, torch.zeros(embed_dim), atol=1e-6))

    def test_factorized_decomposition_x_and_y_separable(self):
        # pos(x,y) = table[x,0] + table[y,1]: holding x fixed and changing y shifts the
        # output by exactly table[y,1]-table[y',1]; symmetrically for fixed y.
        embedder = make_linear_embedder()
        zero = torch.zeros(1, 1, embedder.cfg.model_flat_patch_dim)
        table = embedder.pos_embedding.detach()

        same_x = embedder(zero, torch.tensor([[[2, 3]]]))[0, 0] - embedder(zero, torch.tensor([[[2, 7]]]))[0, 0]
        self.assertTrue(torch.allclose(same_x, table[3, 1] - table[7, 1], atol=1e-6))

        same_y = embedder(zero, torch.tensor([[[2, 5]]]))[0, 0] - embedder(zero, torch.tensor([[[6, 5]]]))[0, 0]
        self.assertTrue(torch.allclose(same_y, table[2, 0] - table[6, 0], atol=1e-6))

    def test_position_is_not_x_y_symmetric(self):
        embedder = make_linear_embedder()
        zero = torch.zeros(1, 1, embedder.cfg.model_flat_patch_dim)
        forward = embedder(zero, torch.tensor([[[2, 6]]]))
        swapped = embedder(zero, torch.tensor([[[6, 2]]]))
        self.assertFalse(torch.allclose(forward, swapped, atol=1e-4))

    def test_identical_positions_give_identical_positional_term(self):
        embedder = make_linear_embedder()
        row = torch.randn(1, 12)
        x = torch.stack([row, row], dim=1)  # (1, 2, 12), two identical patches
        positions = torch.tensor([[[4, 1], [4, 1]]])
        out = embedder(x, positions)
        self.assertTrue(torch.allclose(out[0, 0], out[0, 1], atol=1e-6))


# --- Per-patch independence (justifies the flat batched layout) --------------

class TestPerPatchIndependence(unittest.TestCase):
    def test_permutation_equivariance(self):
        # Reorder the patches (and their positions the same way): the outputs should come
        # out in that same new order, since no patch depends on any other.
        embedder = make_embedder()
        x = torch.randn(1, 7, 12)
        positions = random_positions(1, 7, 10)
        perm = torch.randperm(7)
        out = embedder(x, positions)
        out_perm = embedder(x[:, perm], positions[:, perm])
        self.assertTrue(torch.allclose(out_perm, out[:, perm], atol=1e-6))

    def test_changing_one_patch_leaves_others_unchanged(self):
        # Each patch is embedded on its own, so changing one patch's pixels must change only
        # that patch's output row and leave every other row exactly as it was.
        embedder = make_embedder()
        x = torch.randn(1, 5, 12)
        positions = random_positions(1, 5, 10)
        out = embedder(x, positions)
        x2 = x.clone()
        x2[0, 2] = torch.randn(12)
        out2 = embedder(x2, positions)
        untouched = [i for i in range(5) if i != 2]
        self.assertTrue(torch.allclose(out[0, untouched], out2[0, untouched], atol=1e-6))
        self.assertFalse(torch.allclose(out[0, 2], out2[0, 2], atol=1e-6))

    def test_batch_concatenation_equivalence(self):
        # Embedding two batches stacked along the batch axis equals stacking their separate
        # embeddings — the property that makes flat cross-image concatenation safe later.
        embedder = make_embedder()
        xa, pa = torch.randn(1, 3, 12), random_positions(1, 3, 10, seed=1)
        xb, pb = torch.randn(1, 3, 12), random_positions(1, 3, 10, seed=2)
        joint = embedder(torch.cat([xa, xb], dim=0), torch.cat([pa, pb], dim=0))
        separate = torch.cat([embedder(xa, pa), embedder(xb, pb)], dim=0)
        self.assertTrue(torch.allclose(joint, separate, atol=1e-6))


# --- Normalization -----------------------------------------------------------

class TestNormalization(unittest.TestCase):
    def test_output_rows_are_normalized(self):
        # In a freshly built module the final LayerNorm starts with scale 1 and shift 0, so
        # every output row should come out with mean ~0 and variance ~1 across its features.
        embedder = make_embedder(mm_embed_dim=64)
        out = embedder(torch.randn(2, 8, 12), random_positions(2, 8, 10))
        self.assertTrue(torch.allclose(out.mean(dim=-1), torch.zeros(2, 8), atol=1e-5))
        self.assertTrue(torch.allclose(out.var(dim=-1, unbiased=False), torch.ones(2, 8), atol=1e-3))

    def test_input_affine_invariance(self):
        # The first LayerNorm normalizes each patch, so scaling and shifting a patch's pixels
        # (a*x + b with a > 0) makes no difference to the output. Positions kept the same.
        embedder = make_embedder()
        x = torch.randn(1, 5, 12)
        positions = random_positions(1, 5, 10)
        out = embedder(x, positions)
        out_affine = embedder(3.0 * x + 7.0, positions)
        self.assertTrue(torch.allclose(out, out_affine, atol=1e-4))


# --- Parameters --------------------------------------------------------------

class TestParameters(unittest.TestCase):
    def test_pos_table_shape(self):
        embedder = make_embedder(mm_posemb_size=43, mm_embed_dim=8)
        self.assertEqual(tuple(embedder.pos_embedding.shape), (43, 2, 8))

    def test_pos_table_is_normally_initialized(self):
        # Deliberate deviation from Gemma 4 (which zero-inits): we small-normal-init so the
        # positional path is exercised from step 0. Check empirically on a large table.
        init_range = 0.05
        embedder = make_embedder(mm_posemb_size=400, mm_embed_dim=64,
                                 pos_embd_table_initializer_range=init_range)
        table = embedder.pos_embedding.detach()
        self.assertFalse(torch.all(table == 0))
        self.assertAlmostEqual(float(table.mean()), 0.0, delta=5e-3)
        self.assertAlmostEqual(float(table.std()), init_range, delta=init_range * 0.1)


# --- Gradient sanity ---------------------------------------------------------

class TestGradients(unittest.TestCase):
    def test_grads_reach_all_params_and_are_finite(self):
        # Loss is a sum of SQUARES, not a plain sum. The module ends in a LayerNorm, which
        # centers each output row, so the plain sum over features is always ~0 -> a constant
        # loss -> all-zero gradients, which would make this test pass for the wrong reason.
        # Squaring gives an objective that genuinely depends on the output values.
        embedder = make_embedder()
        x = torch.randn(2, 4, 12)
        positions = random_positions(2, 4, 10)
        (embedder(x, positions) ** 2).sum().backward()
        for name, p in embedder.named_parameters():
            self.assertIsNotNone(p.grad, f"{name} has no grad")
            self.assertTrue(torch.isfinite(p.grad).all(), f"{name} grad not finite")
        # Guard against a silently-constant loss: the dense layer must actually get gradient.
        self.assertTrue((embedder.patch_dense.weight.grad != 0).any())

    def test_pos_table_grad_only_on_used_slots(self):
        # The position table is indexed as [coordinate, axis]. Only the exact (coordinate,
        # axis) slots that a real patch looked up should receive gradient; every other slot
        # must stay EXACTLY zero. That includes the (0, .) slots the padding patch's clamped
        # -1 lands on -- their being zero proves the padding mask stops gradient reaching them.
        # (Sum-of-squares loss for the same reason as the test above.)
        embedder = make_embedder(mm_posemb_size=10)
        x = torch.randn(1, 2, 12)
        # patch 0 sits at grid position (x=2, y=3); patch 1 is padding (-1, -1).
        positions = torch.tensor([[[2, 3], [-1, -1]]])
        (embedder(x, positions) ** 2).sum().backward()

        grad = embedder.pos_embedding.grad  # (mm_posemb_size, 2, mm_embed_dim)
        used = {(2, 0), (3, 1)}  # x-axis coord 2 and y-axis coord 3 are the only real lookups
        for coord in range(grad.shape[0]):
            for axis in range(2):
                slot = grad[coord, axis]
                if (coord, axis) in used:
                    self.assertTrue((slot != 0).any(), f"used slot ({coord}, {axis}) got no gradient")
                else:
                    self.assertTrue(torch.all(slot == 0), f"unused slot ({coord}, {axis}) got gradient")


# --- Determinism -------------------------------------------------------------

class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_output(self):
        x = torch.randn(2, 4, 12)
        positions = random_positions(2, 4, 10)
        out_a = make_embedder(seed=0)(x, positions)
        out_b = make_embedder(seed=0)(x, positions)
        self.assertTrue(torch.equal(out_a, out_b))

    def test_different_seed_differs(self):
        x = torch.randn(2, 4, 12)
        positions = random_positions(2, 4, 10)
        out_a = make_embedder(seed=0)(x, positions)
        out_b = make_embedder(seed=1)(x, positions)
        self.assertFalse(torch.allclose(out_a, out_b, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
