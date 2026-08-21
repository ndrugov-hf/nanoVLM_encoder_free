# Acknowledgements:
# The pure reshape helpers (get_aspect_ratio_preserving_size, convert_image_to_patches,
# patches_merge, pad_along_first_dim) are adapted/copied from the Transformers implementation of
# Gemma 4 (image_processing_gemma4_unified.py). We keep Gemma 4's image-processing CONTRACT and
# math, but drop its HF-framework packaging (base image-processor class, kwargs plumbing,
# BatchFeature). Instead the processor is built from a VLMConfig and returns a plain dict, matching
# how models/vision_embedder.py is built and what it consumes.

import math

import torch
from PIL import Image
from torchvision.transforms.v2 import functional as tvF

# Pixel values arrive as 0..255 integers; the model wants 0..1 floats. Fixed for this processor
# (Gemma 4 runs with do_rescale=True, do_normalize=False), so it is a constant, not a config knob.
RESCALE_FACTOR = 1.0 / 255.0

# The image processor can emit one of these fixed per-image patch budgets (Gemma 4's supported set).
_SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)


def get_aspect_ratio_preserving_size(
    height: int,
    width: int,
    patch_size: int,
    max_patches: int,
    pooling_kernel_size: int,
) -> tuple[int, int]:
    """
    Image is resized to preserve aspect ratio so it fits within the patch budget.
    Target dimensions are the largest that:
    1) Produce at most `max_patches` patches when patchified with `patch_size`
    2) Have height and width divisible by `pooling_kernel_size * patch_size`
    """
    total_px = height * width
    target_px = max_patches * (patch_size**2)
    factor = math.sqrt(target_px / total_px)
    ideal_height = factor * height
    ideal_width = factor * width
    side_mult = pooling_kernel_size * patch_size

    # Round down to nearest multiple of side_mult
    target_height = int(math.floor(ideal_height / side_mult)) * side_mult
    target_width = int(math.floor(ideal_width / side_mult)) * side_mult

    # Handle edge cases where one or both dimensions round to 0
    if target_height == 0 and target_width == 0:
        raise ValueError(
            "Attempting to resize to a 0 x 0 image. Resized height should be divisible by "
            f"`pooling_kernel_size * patch_size`={pooling_kernel_size * patch_size}."
        )

    max_side_length = (max_patches // pooling_kernel_size**2) * side_mult
    if target_height == 0:
        target_height = side_mult
        target_width = min(
            int(math.floor(width / height)) * side_mult,
            max_side_length,
        )
    elif target_width == 0:
        target_width = side_mult
        target_height = min(
            int(math.floor(height / width)) * side_mult,
            max_side_length,
        )

    if target_height * target_width > target_px:
        raise ValueError(
            f"Resizing [{height}x{width}] to [{target_height}x{target_width}] "
            f"but this exceeds {max_patches} patches with patch_size {patch_size}"
        )

    return target_height, target_width


def convert_image_to_patches(image: "torch.Tensor", patch_size: int) -> "torch.Tensor":
    """
    Convert 3D tensor image of shape (num_channels, image_height, image_width) into 2D tensor of patches of shape
    (num_patches_height * num_patches_width, patch_size * patch_size * num_channels).
    """
    num_channels, image_height, image_width = image.shape
    num_patches_height = image_height // patch_size
    num_patches_width = image_width // patch_size
    patched_image = image.reshape(num_channels, num_patches_height, patch_size, num_patches_width, patch_size)
    patched_image = patched_image.permute(1, 3, 2, 4, 0)
    patched_image = patched_image.reshape(num_patches_height * num_patches_width, -1)
    return patched_image


def pad_along_first_dim(
    image: "torch.Tensor", positions: "torch.Tensor", target_length: int
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """
    Pad the tensor along the first dimension. Padding patches are zero-filled and their positions
    are set to -1 (the marker the embedder masks on).
    """
    current_length = image.shape[0]
    padding_length = target_length - current_length
    if padding_length > 0:
        padding = [0, 0] * (image.ndim - 1) + [0, padding_length]
        pos_padding = (0, 0, 0, padding_length)
        image = torch.nn.functional.pad(image, padding, mode="constant", value=0)
        positions = torch.nn.functional.pad(positions, pos_padding, mode="constant", value=-1)
    return image, positions


def patches_merge(
    patches: "torch.Tensor",
    positions_xy: "torch.Tensor",
    length: int,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Merge k×k groups of small patches into larger patches.

    Given `L` input patches of dimension `D = patch_size² × 3`, merge groups of
    `k×k` spatially adjacent patches into `length` output patches of dimension
    `(k × patch_size)² × 3`. The spatial grouping is determined by integer-dividing
    the XY positions by `k`.

    Args:
        patches: (*, L, D) — input patches.
        positions_xy: (*, L, 2) — integer XY positions for each patch (-1 for padding).
        length: target number of output patches. Must satisfy L = length × k².

    Returns:
        merged_patches: (*, length, k²×D) — merged patch features.
        merged_positions: (*, length, 2) — new XY positions for merged patches.
    """
    patch_size = math.isqrt(patches.shape[-1] // 3)
    if patches.shape[-1] != patch_size * patch_size * 3:
        raise ValueError(f"Patch dimension {patches.shape[-1]} is not a valid `patch_size * patch_size * 3`")

    k = math.isqrt(patches.shape[-2] // length)
    if k * k * length != patches.shape[-2]:
        raise ValueError(f"Cannot merge {patches.shape} to {length}")

    # Compute target ordering for reordering patches into kernel-grouped order.
    # This ensures patches within each k×k kernel are contiguous.
    max_x = positions_xy[..., 0].max(dim=-1, keepdim=True)[0] + 1
    kernel_idxs = torch.div(positions_xy, k, rounding_mode="floor")
    num_patches_from_top_left = k * k * kernel_idxs[..., 0] + k * max_x * kernel_idxs[..., 1]

    position_within_kernel = torch.remainder(positions_xy, k)
    num_patches_from_top_left_of_kernel = position_within_kernel[..., 0] + position_within_kernel[..., 1] * k
    target_ordering = num_patches_from_top_left_of_kernel + num_patches_from_top_left

    # Reorder patches by computing the inverse permutation via argsort,
    # then gathering patches into kernel-grouped order.
    perm = target_ordering.long().argsort(dim=-1)  # inverse permutation
    # Expand perm indices to match patch feature dimension for gathering
    perm_expanded = perm.unsqueeze(-1).expand_as(patches)
    kernel_ordered_patches = patches.gather(-2, perm_expanded)

    batch_shape = patches.shape[:-2]

    # Reshape: (*, length*k*k, patch_size*patch_size*3) → (*, length, (k*patch_size)*(k*patch_size)*3)
    kernel_ordered_patches = kernel_ordered_patches.reshape(*batch_shape, length, k * k, patch_size, patch_size, 3)
    # Rearrange (l, a*b, p, q, c) → (l, a*p, b*q, c)
    kernel_ordered_patches = kernel_ordered_patches.reshape(*batch_shape, length, k, k, patch_size, patch_size, 3)
    kernel_ordered_patches = kernel_ordered_patches.permute(
        *range(len(batch_shape)), -6, -5, -3, -4, -2, -1
    )  # (..., l, k, p, k, q, c)
    merged_patches = kernel_ordered_patches.reshape(*batch_shape, length, k * patch_size * k * patch_size * 3)

    # Compute new positions for merged patches
    perm_pos = perm.unsqueeze(-1).expand_as(positions_xy)
    kernel_ordered_positions = positions_xy.float().gather(-2, perm_pos.long())

    # Handle padding: preserve -1 positions
    padding = (positions_xy == -1).all(dim=-1, keepdim=True)  # (..., L, 1)
    kernel_ordered_positions = kernel_ordered_positions * (~padding).float() + positions_xy.float() * padding.float()

    # Reshape positions and take min within each kernel to get the merged position
    kernel_ordered_positions = kernel_ordered_positions.reshape(*batch_shape, length, k * k, 2)
    new_positions = torch.div(kernel_ordered_positions, k, rounding_mode="floor")
    # For each merged patch, take the minimum position across the kernel
    new_positions = new_positions.min(dim=-2)[0].to(torch.long)

    return merged_patches, new_positions


class ImageProcessor:
    """Turn raw images into the inputs models/vision_embedder.py consumes.

    For each image: resize (aspect-preserving, to fit the patch budget) → rescale to [0, 1] →
    cut into teacher patches → merge each k×k block into one model patch → pad to a fixed length.
    A call returns a plain dict:
        pixel_values              (batch, max_soft_tokens, model_flat_patch_dim)  float
        image_position_ids        (batch, max_soft_tokens, 2)                     int, -1 = padding
        num_soft_tokens_per_image list[int]  (real, non-padding patch count per image)

    Built from a VLMConfig, reading:
        teacher_patch_size    — side length in px of a teacher patch (before merging)
        pooling_kernel_size   — k; each model patch merges a k×k block of teacher patches
        max_soft_tokens       — per-image budget of model patches (padded up to this)
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.patch_size = cfg.teacher_patch_size
        self.pooling_kernel_size = cfg.pooling_kernel_size
        self.max_soft_tokens = cfg.max_soft_tokens
        if self.max_soft_tokens not in _SUPPORTED_SOFT_TOKENS:
            raise ValueError(
                f"`max_soft_tokens` must be one of {_SUPPORTED_SOFT_TOKENS}, got {self.max_soft_tokens}."
            )

    def _to_chw_float(self, image) -> "torch.Tensor":
        """Accept a PIL image or a (C, H, W) tensor; return a float32 (3, H, W) tensor in 0..255.

        The HF base class used to do RGB-conversion + PIL→tensor for us; since we dropped it, we do
        the small amount ourselves. Float (not uint8) so the bicubic resize below interpolates
        smoothly.
        """
        if isinstance(image, Image.Image):
            image = tvF.pil_to_tensor(image.convert("RGB"))  # (3, H, W) uint8
        elif not torch.is_tensor(image):
            raise TypeError(f"ImageProcessor expects a PIL image or a (C,H,W) tensor; got {type(image)}")
        return image.float()

    def aspect_ratio_preserving_resize(self, image: "torch.Tensor", max_patches: int) -> "torch.Tensor":
        height, width = image.shape[-2], image.shape[-1]
        target_height, target_width = get_aspect_ratio_preserving_size(
            height=height,
            width=width,
            patch_size=self.patch_size,
            max_patches=max_patches,
            pooling_kernel_size=self.pooling_kernel_size,
        )
        if target_height == height and target_width == width:
            return image
        return tvF.resize(
            image,
            size=[target_height, target_width],
            interpolation=tvF.InterpolationMode.BICUBIC,
            antialias=True,
        )

    def __call__(self, images) -> dict:
        # Accept a single image or a list; process each on its own (different aspect ratios resize to
        # different sizes, so patchify + pad must happen per-image before stacking).
        if isinstance(images, Image.Image) or torch.is_tensor(images):
            images = [images]

        # Each model patch merges pooling_kernel_size² teacher patches, so the teacher-patch budget
        # is that many times the model-patch (soft-token) budget.
        max_patches = self.max_soft_tokens * self.pooling_kernel_size**2

        pixel_values = []
        position_ids = []
        num_soft_tokens_per_image = []

        for image in images:
            # Step 1: to float (3, H, W), then aspect-preserving resize to fit the patch budget.
            image = self._to_chw_float(image)
            image = self.aspect_ratio_preserving_resize(image, max_patches)

            # Step 2: rescale 0..255 -> 0..1. Bicubic resize can overshoot the source range, so clamp
            # back into [0, 1] to keep the embedder's input clean.
            image = (image * RESCALE_FACTOR).clamp(0.0, 1.0)

            # Step 3: cut into teacher patches. (3, H, W) -> (num_teacher_patches, patch_size²*3)
            patch_height = image.shape[-2] // self.patch_size
            patch_width = image.shape[-1] // self.patch_size
            teacher_patches = convert_image_to_patches(image, self.patch_size)

            # Step 4: teacher-patch XY grid positions (x = column, y = row), row-major to match the
            # patch order above.
            device = image.device
            patch_grid = torch.meshgrid(
                torch.arange(patch_width, device=device),
                torch.arange(patch_height, device=device),
                indexing="xy",
            )
            teacher_positions = torch.stack(patch_grid, dim=-1).reshape(teacher_patches.shape[0], 2)

            # Step 5: merge each k×k block of teacher patches into one model patch.
            num_model_patches = teacher_patches.shape[0] // (self.pooling_kernel_size**2)
            merged_patches, merged_positions = patches_merge(
                teacher_patches.unsqueeze(0),  # patches_merge works on a batch dim
                teacher_positions.unsqueeze(0),
                num_model_patches,
            )
            merged_patches = merged_patches.squeeze(0)
            merged_positions = merged_positions.squeeze(0)
            # Every model patch here is real (we resized to exact multiples, so no teacher padding);
            # this is the real, non-padding count.
            num_soft_tokens_per_image.append(merged_patches.shape[0])

            # Step 6: pad up to the fixed per-image budget so images can be stacked into one batch.
            merged_patches, merged_positions = pad_along_first_dim(
                merged_patches, merged_positions, self.max_soft_tokens
            )
            pixel_values.append(merged_patches)
            position_ids.append(merged_positions)

        return {
            "pixel_values": torch.stack(pixel_values, dim=0),
            "image_position_ids": torch.stack(position_ids, dim=0),
            "num_soft_tokens_per_image": num_soft_tokens_per_image,
        }
