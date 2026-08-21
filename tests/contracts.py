"""Single source of truth for the encoder-free image-data dict contract.

Both the MODEL-side consumer tests (tests/test_encoder_free_wiring.py) and the future DATA-side
producer tests (sub-step 3) import `assert_image_dict`, so the two halves of the pipeline are
checked against exactly the same spec and cannot silently drift apart.

The dict that reaches `VLM.forward` on the encoder-free path (and that `data/image_processing.py`
`ImageProcessor` produces) must satisfy:

  * keys include "pixel_values" and "image_position_ids" (extra keys such as
    "num_soft_tokens_per_image" are allowed).
  * pixel_values       : (num_images, N, flat_dim)  float
  * image_position_ids : (num_images, N, 2)          integer, (-1,-1) marks a padding row
  * every position row is EITHER exactly (-1,-1) [padding] OR both coords >= 0 [real] — never mixed.
  * real rows come first, contiguously, per image (reals-first, then padding).
  * padding pixel rows are zero-filled.
  * if present, num_soft_tokens_per_image[i] == the number of real rows in image i.
"""

import torch


def assert_image_dict(d, num_images, N, flat_dim):
    """Assert `d` matches the encoder-free image-dict contract. Raises AssertionError otherwise."""
    assert isinstance(d, dict), f"expected a dict, got {type(d)}"
    assert "pixel_values" in d, "missing key 'pixel_values'"
    assert "image_position_ids" in d, "missing key 'image_position_ids'"

    pv = d["pixel_values"]
    pos = d["image_position_ids"]

    assert tuple(pv.shape) == (num_images, N, flat_dim), \
        f"pixel_values shape {tuple(pv.shape)} != {(num_images, N, flat_dim)}"
    assert tuple(pos.shape) == (num_images, N, 2), \
        f"image_position_ids shape {tuple(pos.shape)} != {(num_images, N, 2)}"
    assert pv.dtype.is_floating_point, f"pixel_values dtype {pv.dtype} is not floating point"
    assert not pos.dtype.is_floating_point, f"image_position_ids dtype {pos.dtype} is not integer"

    # A row is padding iff its position is exactly (-1,-1); a real row has both coords >= 0.
    both_neg1 = (pos == -1).all(dim=-1)     # (num_images, N)
    both_nonneg = (pos >= 0).all(dim=-1)    # (num_images, N)
    assert torch.equal(both_neg1 | both_nonneg, torch.ones_like(both_neg1)), \
        "every position row must be either (-1,-1) or have both coords >= 0 (no mixed rows)"

    # reals-first-contiguous per image: once padding starts, the rest of that image is padding.
    for i in range(num_images):
        row = both_nonneg[i]                        # (N,) True where real
        n_real = int(row.sum())
        expected = torch.zeros(N, dtype=torch.bool)
        expected[:n_real] = True
        assert torch.equal(row, expected), \
            f"image {i}: real rows are not first-contiguous (real mask = {row.tolist()})"

    # padding pixel rows must be zero-filled.
    if both_neg1.any():
        assert int(torch.count_nonzero(pv[both_neg1])) == 0, "padding pixel rows must be zero-filled"

    # optional consistency: the advertised soft-token count matches the real-row count per image.
    if "num_soft_tokens_per_image" in d:
        advertised = list(d["num_soft_tokens_per_image"])
        real_counts = both_nonneg.sum(dim=-1).tolist()
        assert advertised == real_counts, \
            f"num_soft_tokens_per_image {advertised} != per-image real-row counts {real_counts}"
