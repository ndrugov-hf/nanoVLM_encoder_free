# nanoVLM (encoder-free)

This is a fork of [nanoVLM](https://github.com/huggingface/nanoVLM) with the vision encoder removed. Raw pixel patches go through one small embedding layer and land directly in the language model's input sequence, following the encoder-free design of Gemma 4. For everything the two repos share (the training loop, data packing, the overall philosophy, how to get started) read the upstream README. This file describes what is different here.

## The vision path

Upstream encodes each image with a SigLIP ViT (`models/vision_transformer.py`) and shrinks the resulting features with a pixel-shuffle projector. Here the image itself becomes the input. The processor resizes each image so it fits a fixed patch budget while keeping its aspect ratio, cuts it into 32 px model patches, and flattens every patch into 3072 raw pixel values. Those values are what the model sees.

The pieces, in the order data flows through them:

- `data/image_processing.py` (~300 lines) reimplements Gemma 4's image-processing contract on top of a plain `VLMConfig`, without the HF framework packaging. Every image gets a patch budget from the fixed set {70, 140, 280, 560, 1120}, with 280 as the default. Pixels are rescaled to the 0..1 range and left otherwise untouched (Gemma 4 runs with `do_normalize=False`). All of this happens in the data-loading workers on CPU, so the GPU never waits on preprocessing.
- `models/vision_embedder.py` (~130 lines) replaces the ViT. Each flattened patch runs through LayerNorm, then a Linear into `mm_embed_dim`, then LayerNorm again, and picks up a learned factorized 2D positional embedding for its (x, y) spot in the image grid. That is the entire vision tower.
- `models/vision_projector.py` (~30 lines) maps embedder output into the language model's width: RMSNorm followed by a Linear without bias. It keeps the patch count unchanged. Pixel-shuffle downsampling exists only on the ViT path.

## Image tokens in the text

Each image appears in the prompt as a run of `<|image|>` placeholders, exactly one per model patch the processor produced for that image. The model writes one patch embedding into each placeholder position (`_replace_img_tokens_with_embd` in `models/vision_language_model.py`), and a count mismatch raises immediately. Upstream's grid-layout tokens (`<|global_image|>`, `<row_i_col_j>`) are gone from the vocabulary on this path. The grid position of every patch travels as plain numbers in `image_position_ids`, where padding rows are marked `(-1, -1)` and dropped before the embedder sees them.

## The language side

`models/decoder.py` (~200 lines) wraps any HuggingFace `AutoModelForCausalLM` behind the same surface the VLM expects from the hand-written language model, so the rest of the code cannot tell the two apart. Set `lm_backend = "hf"` in the config to use the wrapper or `"custom"` for upstream's Llama-style stack. The default backbone is `LiquidAI/LFM2.5-1.2B-Instruct`, and `models/config_lfm.py` / `models/config_smollm.py` carry ready-made configs for LFM2.5-230M and SmolLM2-360M-Instruct.

## Switching between the two vision paths

`vision_backend` in `models/config.py` selects `"encoder_free"` (the default) or `"vit"`. The ViT path stayed fully wired during the migration and still works, which made every step of the switch testable against a known-good reference.

## Evaluation

`eval/lmms_eval_wrapper.py` speaks the encoder-free input format, so [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) benchmarks run against these models the same way they do upstream. `eval.slurm` fans the benchmark tasks out across GPUs and `merge_eval_results.py` combines the per-task outputs into one result file per checkpoint.

## Tests

The `tests/` directory holds over 200 unit tests covering the image processor, the embedder, the projector, the data pipeline, packing, the eval wrapper, and the model wiring, on CPU with random weights. `sbatch tests/gpu_smoke_test.slurm` runs the paths that need real weights and a GPU: an autocast training step, generation, and save/resume.

## Citation

If you use this repository, please cite the upstream nanoVLM work:

```
@misc{wiedmann2025nanovlm,
  author = {Luis Wiedmann and Aritra Roy Gosthipaty and Andrés Marafioti},
  title = {nanoVLM},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/nanoVLM}}
}
```
