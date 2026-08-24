# nanoVLM with a Gemma 4 vision path

This fork adapts [nanoVLM](https://github.com/huggingface/nanoVLM) to follow the vision architecture used by Gemma 4 12B. It splits images into raw pixel patches, turns each patch into an embedding with a small module, and places those embeddings in the language model's input sequence. 

## The vision path

Upstream nanoVLM processes images with a SigLIP ViT in `models/vision_transformer.py`. This fork uses the image patches themselves as model inputs.

The processor resizes each image while preserving its aspect ratio. It then divides the image into 32-pixel patches and flattens each patch into 3,072 pixel values.

- `data/image_processing.py` implements the Gemma 4 image-processing format with a plain `VLMConfig`. It assigns each image a patch budget from `{70, 140, 280, 560, 1120}`, using 280 by default. Pixel values are scaled to the `0..1` range. Data-loading workers handle this work on the CPU.

- `models/vision_embedder.py` turns each flattened patch into an embedding. The patch passes through LayerNorm, a Linear layer with an output width of `mm_embed_dim`, and another LayerNorm. A learned factorized 2D positional embedding records its location in the image grid.

- `models/vision_projector.py` maps each embedding to the language model's width. It applies RMSNorm followed by a Linear layer without bias. The number of patches stays the same.

## Image tokens in the prompt

Each image appears in the prompt as a sequence of `<|image|>` tokens. The processor creates one token for every image patch.

`_replace_img_tokens_with_embd` in `models/vision_language_model.py` places the patch embeddings at those token positions. It raises an error when the number of tokens and embeddings differs.

Patch coordinates travel through `image_position_ids`. Padding uses `(-1, -1)` coordinates, which the embedder removes before processing. The encoder-free path uses these coordinates in place of upstream nanoVLM's grid tokens such as `<|global_image|>` and `<row_i_col_j>`.

## The language model

`models/decoder.py` wraps Hugging Face models loaded through `AutoModelForCausalLM`. The wrapper exposes the interface expected by the rest of nanoVLM.

Set `lm_backend = "hf"` to use a Hugging Face model. The `"custom"` option selects the original Llama-style implementation from nanoVLM.

The default language model is `LiquidAI/LFM2.5-1.2B-Instruct`. Ready-made configurations for LFM2.5-230M and SmolLM2-360M-Instruct are available in `models/config_lfm.py` and `models/config_smollm.py`.

## Choosing a vision backend

Set `vision_backend` in `models/config.py` to choose the vision path.

- `"encoder_free"` uses the Gemma 4-style patch embedder and projector. This is the default.

- `"vit"` uses the original ViT path from nanoVLM.

The ViT path remains available for tests and architecture comparisons.

## Evaluation

`eval/lmms_eval_wrapper.py` prepares encoder-free inputs for [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).

`eval.slurm` distributes benchmark tasks across GPUs. After evaluation, `merge_eval_results.py` combines the task outputs into one result file for each checkpoint.

## Tests

The `tests/` directory contains more than 200 CPU tests. They cover image processing, patch embeddings, projection, data loading, sequence packing, evaluation, and model wiring with randomly initialized weights.

## Citation

If you use this repository, please cite nanoVLM:

```java
@misc{wiedmann2025nanovlm,
  author = {Luis Wiedmann and Aritra Roy Gosthipaty and Andrés Marafioti},
  title = {nanoVLM},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/nanoVLM}}
}
```
