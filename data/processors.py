from transformers import AutoTokenizer
import torchvision.transforms as transforms

from data.custom_transforms import DynamicResize, SplitImage, GlobalAndSplitImages
from data.image_processing import ImageProcessor
from models.config import VLMConfig

TOKENIZERS_CACHE = {}

def get_tokenizer(name, extra_special_tokens=None, chat_template=None):
    if name not in TOKENIZERS_CACHE:
        tokenizer_init_kwargs = {"use_fast": True}
        if extra_special_tokens is not None:
            tokenizer_init_kwargs["extra_special_tokens"] = extra_special_tokens
        if chat_template is not None:
            tokenizer_init_kwargs["chat_template"] = chat_template
        tokenizer = AutoTokenizer.from_pretrained(name, **tokenizer_init_kwargs,)
        tokenizer.pad_token = tokenizer.eos_token
        TOKENIZERS_CACHE[name] = tokenizer
    return TOKENIZERS_CACHE[name]

def get_image_processor(max_img_size, splitted_image_size, resize_to_max_side_len=False):
    return transforms.Compose([
        DynamicResize(splitted_image_size, max_img_size, resize_to_max_side_len),
        transforms.ToTensor(),
        GlobalAndSplitImages(splitted_image_size),
    ])

def get_image_processor_encoder_free(cfg: VLMConfig) -> ImageProcessor:
    """
    Purpose:
        Build the image processor for the encoder-free vision path. That path embeds raw image
        patches directly (no ViT), so image preparation is the `ImageProcessor`: resize each image
        to fit the patch budget, cut it into patches, merge each k x k block of patches into one
        model patch, and pad to a fixed length. The processor reads all of its geometry off the
        config, so it is built from `cfg` alone.

        This is the encoder-free sibling of `get_image_processor` (which returns the ViT transform
        pipeline); both are factory helpers so callers pick a processor by vision backend without
        knowing how either one is built.

    Parameters:
     * cfg : the model config. The processor reads `teacher_patch_size`, `pooling_kernel_size`,
             and `max_soft_tokens` off it to fix its patch geometry and per-image budget.

    Returns:
        An `ImageProcessor` callable. Called on one image or a list of images, it returns the dict
        {"pixel_values", "image_position_ids", "num_soft_tokens_per_image"} that the encoder-free
        model consumes.
    """
    return ImageProcessor(cfg)



def get_image_string(tokenizer, splitted_image_counts, mp_image_token_length):
    image_string = ""
    # splitted_image_counts is a list of tuples (n_h, n_w)
    for idx, (n_h, n_w) in enumerate(splitted_image_counts):
        if len(splitted_image_counts) > 1:
            image_string += f"<image: {idx}>"
        if hasattr(tokenizer, "global_image_token"):
            image_string += tokenizer.global_image_token
            image_string += tokenizer.image_token * mp_image_token_length
            if n_h == 1 and n_w == 1:  # If there is only one patch, treat it as the global image
                continue
        for i in range(n_h):
            for j in range(n_w):
                image_string += getattr(tokenizer, f'r{i+1}c{j+1}')
                image_string += tokenizer.image_token * mp_image_token_length
    return image_string


def get_image_string_encoder_free(tokenizer, num_soft_tokens_per_image: list[int]) -> str:
    """
    Purpose:
        Build the run of placeholder tokens that stands in for an example's images in the text, on
        the encoder-free path. For each image, emit exactly as many image placeholder tokens as it
        has real (non-filler) patches; the model later replaces each placeholder with that patch's
        feature vector, so the counts must match. Unlike the ViT `get_image_string`, this emits no
        `<|global_image|>` and no row/col tokens: the 2D layout is carried numerically in
        `image_position_ids`, not spelled out in the text (decision
        `encoder-free-step3-token-scaffolding`).

    Parameters:
     * tokenizer : provides `image_token`, the placeholder string repeated per patch.

     * num_soft_tokens_per_image : one entry per image, in image order, giving that image's real
                                   patch count (as reported by the `ImageProcessor` in its
                                   `num_soft_tokens_per_image`). An empty list means a text-only
                                   example (no images).

    Returns:
        The placeholder tokens for every image, concatenated in image order. The empty string when
        there are no images.
    """
    image_string = ""
    for num_soft_tokens in num_soft_tokens_per_image:
        image_string += tokenizer.image_token * num_soft_tokens

    return image_string
