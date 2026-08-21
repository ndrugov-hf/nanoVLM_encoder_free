"""Behavioral / architectural spec for models.vision_projector.VisionProjector.

The VisionProjector is the encoder-free "modality projector". The VisionEmbedder deliberately
stops at `pos_norm`, outputting width `mm_embed_dim`; the VisionProjector maps that into the
language model's embedding width `lm_hidden_dim`. Per the embedder/MP boundary decision it is
exactly:

    RMSNorm(mm_embed_dim, eps=lm_rms_eps) -> Linear(mm_embed_dim -> lm_hidden_dim, bias=False)

Unlike the ViT-path ModalityProjector it does NOT pixel-shuffle, so it PRESERVES the number of
tokens (that projector shrinks 1024 -> 64; this one leaves N unchanged).

We test architectural PROPERTIES, not bit-equivalence to any reference implementation
(mirrors the style of test_embedder.py).

Input contract:
  x   : (batch, tokens, mm_embed_dim)   float
  out : (batch, tokens, lm_hidden_dim)  float
"""

import unittest
from types import SimpleNamespace

import torch

from models.vision_projector import VisionProjector


# --- Test fixtures -----------------------------------------------------------

def make_config(**overrides):
    """Minimal config exposing exactly the attributes VisionProjector reads.

    Small, hand-checkable defaults: input width mm_embed_dim=8, output width
    lm_hidden_dim=6 (deliberately different so a shape mix-up is caught).
    """
    defaults = dict(
        mm_embed_dim=8,
        lm_hidden_dim=6,
        lm_rms_eps=1e-5,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def make_projector(seed=0, **overrides):
    """Seeded VisionProjector so parameter initialization is reproducible."""
    torch.manual_seed(seed)
    return VisionProjector(make_config(**overrides))


# --- Shape, dtype and token count --------------------------------------------

class TestShapesAndDtype(unittest.TestCase):
    def test_output_shape(self):
        # (B, N, mm_embed_dim) -> (B, N, lm_hidden_dim)
        proj = make_projector()
        out = proj(torch.randn(2, 5, 8))
        self.assertEqual(out.shape, (2, 5, 6))

    def test_single_token(self):
        proj = make_projector()
        out = proj(torch.randn(1, 1, 8))
        self.assertEqual(out.shape, (1, 1, 6))

    def test_token_count_is_preserved(self):
        # The defining contrast with the ViT-path ModalityProjector, which pixel-shuffles a
        # perfect-square token count down by scale_factor**2. This projector must NOT: feed a
        # perfect square (16) and get all 16 rows back, unchanged in count.
        proj = make_projector()
        out = proj(torch.randn(1, 16, 8))
        self.assertEqual(out.shape[1], 16)

    def test_runs_in_double_precision(self):
        # A float64 module returns float64 from a float64 input.
        proj = make_projector().double()
        out = proj(torch.randn(2, 4, 8, dtype=torch.float64))
        self.assertEqual(out.dtype, torch.float64)


# --- Normalization: it is RMSNorm, not LayerNorm, not nothing ----------------

class TestNormalization(unittest.TestCase):
    def test_scale_invariance(self):
        # RMSNorm divides each row by its root-mean-square, so scaling a row's values by a
        # positive constant leaves the output unchanged. (A plain Linear with no norm would
        # scale the output by the same constant, so this also proves a norm is present.)
        proj = make_projector()
        x = torch.randn(2, 4, 8)
        out = proj(x)
        out_scaled = proj(3.0 * x)
        self.assertTrue(torch.allclose(out, out_scaled, atol=1e-4))

    def test_not_shift_invariant(self):
        # RMSNorm does NOT subtract the mean (unlike LayerNorm). So adding a constant to every
        # feature of a row generally changes the output. This is what pins RMSNorm vs LayerNorm.
        proj = make_projector()
        x = torch.randn(2, 4, 8)
        out = proj(x)
        out_shifted = proj(x + 5.0)
        self.assertFalse(torch.allclose(out, out_shifted, atol=1e-4))


# --- No bias -----------------------------------------------------------------

class TestNoBias(unittest.TestCase):
    def test_zero_input_maps_to_zero(self):
        # An all-zero input row has RMS 0, so RMSNorm sends it to 0 (0 / sqrt(0 + eps) == 0),
        # and a bias-free Linear sends 0 -> 0. A nonzero output here would mean the Linear has
        # a bias, which the contract forbids.
        proj = make_projector()
        out = proj(torch.zeros(2, 3, 8))
        self.assertTrue(torch.allclose(out, torch.zeros(2, 3, 6), atol=1e-6))


# --- Per-token independence --------------------------------------------------

class TestPerTokenIndependence(unittest.TestCase):
    def test_changing_one_token_leaves_others_unchanged(self):
        # Each token is projected on its own; changing one token's values must change only that
        # token's output row and leave every other row exactly as it was.
        proj = make_projector()
        x = torch.randn(1, 5, 8)
        out = proj(x)
        x2 = x.clone()
        x2[0, 2] = torch.randn(8)
        out2 = proj(x2)
        untouched = [i for i in range(5) if i != 2]
        self.assertTrue(torch.allclose(out[0, untouched], out2[0, untouched], atol=1e-6))
        self.assertFalse(torch.allclose(out[0, 2], out2[0, 2], atol=1e-6))

    def test_permutation_equivariance(self):
        # Reorder the tokens: the outputs come out in that same new order.
        proj = make_projector()
        x = torch.randn(1, 7, 8)
        perm = torch.randperm(7)
        out = proj(x)
        out_perm = proj(x[:, perm])
        self.assertTrue(torch.allclose(out_perm, out[:, perm], atol=1e-6))

    def test_batch_concatenation_equivalence(self):
        # Projecting two batches stacked along the batch axis equals stacking their separate
        # projections -- the property that makes flat cross-image concatenation safe.
        proj = make_projector()
        xa = torch.randn(1, 3, 8)
        xb = torch.randn(1, 3, 8)
        joint = proj(torch.cat([xa, xb], dim=0))
        separate = torch.cat([proj(xa), proj(xb)], dim=0)
        self.assertTrue(torch.allclose(joint, separate, atol=1e-6))


# --- Gradient sanity ---------------------------------------------------------

class TestGradients(unittest.TestCase):
    def test_grads_reach_every_param_and_are_finite(self):
        # Sum-of-squares objective so the loss genuinely depends on the output values.
        # The module has exactly two learnable tensors -- the RMSNorm scale and the bias-free
        # Linear weight -- and EVERY one must receive a finite, nonzero gradient. Checking every
        # param (not "some param") is what catches a projector that skips the norm, or a norm
        # scale that never influences the output: that param would come back with a zero grad.
        proj = make_projector()
        x = torch.randn(2, 4, 8)
        (proj(x) ** 2).sum().backward()
        params = dict(proj.named_parameters())
        self.assertGreaterEqual(len(params), 2,
                                "expected at least the RMSNorm scale and the Linear weight")
        for name, p in params.items():
            self.assertIsNotNone(p.grad, f"{name} has no grad")
            self.assertTrue(torch.isfinite(p.grad).all(), f"{name} grad not finite")
            self.assertTrue((p.grad != 0).any(), f"{name} received an all-zero gradient")

    def test_gradient_flows_back_to_input(self):
        # In the assembled VLM the projector sits ON TOP of the VisionEmbedder, so gradient has
        # to pass through it into the embedder's parameters. Prove the module is differentiable
        # w.r.t. its input: a leaf input tensor comes back with a finite, nonzero gradient. If it
        # did not, the embedder underneath would never train.
        proj = make_projector()
        x = torch.randn(2, 4, 8, requires_grad=True)
        (proj(x) ** 2).sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue((x.grad != 0).any())


# --- Determinism -------------------------------------------------------------

class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_output(self):
        x = torch.randn(2, 4, 8)
        out_a = make_projector(seed=0)(x)
        out_b = make_projector(seed=0)(x)
        self.assertTrue(torch.equal(out_a, out_b))

    def test_different_seed_differs(self):
        x = torch.randn(2, 4, 8)
        out_a = make_projector(seed=0)(x)
        out_b = make_projector(seed=1)(x)
        self.assertFalse(torch.allclose(out_a, out_b, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
