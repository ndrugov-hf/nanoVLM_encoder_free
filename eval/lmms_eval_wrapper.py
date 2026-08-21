"""
LMMS-Eval wrapper for nanoVLM model.
This allows using lmms-eval for intermediate evaluation during training.
"""

import torch
from typing import List, Tuple, Optional, Union
from PIL import Image
import numpy as np
import torch.distributed as dist

from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.model import lmms
from lmms_eval.api.instance import Instance

from models.vision_language_model import VisionLanguageModel
# ------------------------ CHANGE ------------------------
# Import encoder-free versions of get_image_processor and get_image_string
from data.processors import get_tokenizer, get_image_processor_encoder_free, get_image_string_encoder_free
# --------------------- END OF CHANGE ---------------------   

from data.collators import VQACollator


class NanoVLMWrapper(lmms):
    """Wrapper to make nanoVLM compatible with lmms-eval framework."""
    
    def __init__(
        self,
        model: str | VisionLanguageModel = "lusxvr/nanoVLM-450M",
        device: str = "cuda",
        batch_size: int = 32,
        **kwargs
    ):
        super().__init__()

        if isinstance(model, str):
            self.model = VisionLanguageModel.from_pretrained(model).to(device)
        else:
            self.model = model.to(device)

        # ------------------------ CHANGE ------------------------
        # Check the backend on the loaded model, not the raw `model` argument: when `model` is a
        # checkpoint-path string (how lmms-eval's CLI instantiates us) it has no `.cfg` until it is
        # loaded, so this assert must run after from_pretrained.
        assert self.model.cfg.vision_backend == "encoder_free", "This eval works only when model.cfg.vision_backend == 'encoder_free'"
        # --------------------- END OF CHANGE ---------------------

        self.device = device
        self.batch_size = batch_size
        
        if dist.is_available() and dist.is_initialized():
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
        else:
            # Fallback for non-distributed execution
            self._rank = 0
            self._world_size = 1
        
        # Get tokenizer and image processor from model config if not provided
        self.tokenizer = get_tokenizer(self.model.cfg.lm_tokenizer, self.model.cfg.vlm_extra_tokens, self.model.cfg.lm_chat_template)
        resize_to_max_side_len = False
        if hasattr(self.model.cfg, "resize_to_max_side_len"):
            resize_to_max_side_len = self.model.cfg.resize_to_max_side_len
        print(f"Resize to max side len: {resize_to_max_side_len}")
        # ------------------------ CHANGE ------------------------
        # Using get_image_processor_encoder_free() instead of get_image_processor()
        self.image_processor = get_image_processor_encoder_free(self.model.cfg)
        # --------------------- END OF CHANGE ---------------------    

    def _prepare_visual_input(
        self,
        visual_list: list[list[Image.Image | str | np.ndarray] | None],
    ) -> tuple[list[list[dict[str, torch.Tensor]]], list[list[int]]]:
        """
        Purpose:
            Turn one evaluation batch's raw images into encoder-free image data, kept grouped
            by sample. lmms-eval hands us the batch's images as one entry per sample: a list of
            that sample's images, or None (or an empty list) for a text-only sample. Each image
            is run through the encoder-free ImageProcessor on its own (resize -> cut into patches
            -> merge each k x k block of patches into one model patch -> pad to a fixed length),
            and its result becomes one dict. The per-image dicts for a sample are collected into
            that sample's own list, so the sample-to-image grouping is preserved for the caller.

            This is the batch-level, eval sibling of the training-path
            `_process_images_encoder_free` (which processes a single sample). It runs over the
            whole batch at once and keeps the per-sample grouping the eval loop needs: to write
            each sample's <|image|> placeholder run, and to stack the batch's images in sample
            order later. It also accepts the input forms lmms-eval can pass (PIL image, file
            path, or numpy array), which the training path never sees.

        Parameters:
         * visual_list : the batch's images, one entry per sample in batch order. Each entry is
                         either a list of that sample's images (each a PIL image, a path string,
                         or an (H, W, C) numpy array) or None / [] for a text-only sample.

        Returns:
            A tuple (processed_images_lists, num_soft_tokens_per_image_lists), both with one
            entry per sample, in batch order:
             * processed_images_lists[i] : list with one dict per image of sample i, in image
                   order. Each dict is
                   {"pixel_values":       (max_soft_tokens, model_flat_patch_dim) float,
                    "image_position_ids": (max_soft_tokens, 2) int, where (-1, -1) marks a
                                          padding patch}.
                   The empty list when sample i is text-only.
             * num_soft_tokens_per_image_lists[i] : list[int], one entry per image of sample i,
                   the real (non-padding) patch count for that image. Equals the number of real
                   rows in that image's pixel_values, so the text can write exactly that many
                   <|image|> placeholders. The empty list when sample i is text-only.

        Raises:
            ValueError: if an image is not a PIL image, a path string, or a numpy array.
        """
        # ------------------------ CHANGE ------------------------
        processed_images_lists = []
        num_soft_tokens_per_image_lists = []

        for sample in visual_list:
            sample_dicts, sample_counts = [], []

            # A text-only sample (None or []) contributes an empty group, keeping the returned
            # lists aligned one-to-one with the batch's samples.
            if sample is not None:
                for visual in sample:
                    if isinstance(visual, Image.Image):
                        image = visual
                    elif isinstance(visual, str): # Keep path loading for convenience
                        image = Image.open(visual).convert("RGB")
                    elif isinstance(visual, np.ndarray): # Keep numpy array loading for convenience
                        image = Image.fromarray(visual)
                    else:
                        # If it's not an Image, a path string, or a numpy array, it's an error
                        raise ValueError(f"Unsupported visual type: {type(visual)}. Expected PIL Image, path string, or numpy array.")

                    processed_image_data = self.image_processor(image)

                    sample_dicts.append({"pixel_values": processed_image_data["pixel_values"].squeeze(0),
                                         "image_position_ids": processed_image_data["image_position_ids"].squeeze(0)})
                    sample_counts.append(processed_image_data["num_soft_tokens_per_image"][0])

            processed_images_lists.append(sample_dicts)
            num_soft_tokens_per_image_lists.append(sample_counts)

        return processed_images_lists, num_soft_tokens_per_image_lists
    # --------------------- END OF CHANGE ---------------------

    # ------------------------ CHANGE ------------------------
    def _build_images_dict(
        self,
        per_sample_image_dicts: list[list[dict[str, torch.Tensor]]],
    ) -> dict[str, torch.Tensor] | None:
        """
        Purpose:
            Collapse the batch's per-sample image dicts into the single stacked image dict the
            encoder-free model consumes. `_prepare_visual_input` keeps images grouped by sample
            (one list per sample); the model instead wants every image in the batch stacked along
            one leading axis. This flattens the grouping and stacks each field, preserving image
            order, so the model's forward can embed all images at once and write their features
            into the <|image|> placeholder positions.

            The stack order is sample-major then image-major: sample 0's images in order, then
            sample 1's, and so on -- the same order the placeholder tokens appear in across the
            (row-major) batch. Keeping these two orders identical is what puts each image's
            features on its own sample's placeholders; any other order would cross the wires.

            Returns None when the batch has no images at all. That None is the model's text-only
            signal (`_embed_inputs` skips the image path when `images` is None), so a batch of
            purely text samples runs as a plain text forward.

        Parameters:
         * per_sample_image_dicts : the grouped image dicts from `_prepare_visual_input`, one
                                    list per sample (empty for a text-only sample). Each dict is
                                    {"pixel_values": (max_soft_tokens, model_flat_patch_dim) float,
                                     "image_position_ids": (max_soft_tokens, 2) int}.

        Returns:
            A single dict
            {"pixel_values":       (num_images, max_soft_tokens, model_flat_patch_dim) float,
             "image_position_ids": (num_images, max_soft_tokens, 2) int},
            where num_images is the total image count across the whole batch and the leading axis
            is in sample-major, image-major order. None when the batch contains no images.
        """
        flat = [d for sample_dicts in per_sample_image_dicts for d in sample_dicts]
        if not flat:
            return None
        return {
            "pixel_values":       torch.stack([d["pixel_values"] for d in flat]),
            "image_position_ids": torch.stack([d["image_position_ids"] for d in flat]),
        }
    # --------------------- END OF CHANGE ---------------------

    # ------------------------ CHANGE ------------------------
    def _trim_to_placeholder_count(
        self,
        per_sample_image_dicts: list[list[dict[str, torch.Tensor]]],
        per_sample_soft_token_counts: list[list[int]],
        input_ids: torch.Tensor,
    ) -> tuple[list[list[dict[str, torch.Tensor]]], list[list[int]]]:
        """
        Purpose:
            Reconcile each sample's image data with the <|image|> placeholders that actually
            survived tokenization. The model writes one image patch feature into each <|image|>
            placeholder, and it requires the two counts to match exactly (see `_embed_inputs`,
            which raises if they differ). But tokenizing with truncation to `max_length` can drop
            some of a sample's placeholders, leaving fewer placeholders than the sample has real
            (non-padding) patches. This turns that sample's extra real patches back into padding
            so its real-patch count matches its surviving placeholder count, letting generation run
            instead of failing on the count mismatch.

            A real patch is marked as padding the same way the processor marks one: its
            image_position_ids row is set to (-1, -1), which drops it from the model's image path.
            Extra patches are removed last-image-first, and within an image from the last real
            patch backward, so trimming eats from the tail of the sample's image patches -- the
            end that a right-truncation would have cut. Samples whose real-patch count already
            fits (nothing was dropped) are returned unchanged.

        Parameters:
         * per_sample_image_dicts : the grouped image dicts from `_prepare_visual_input`, one list
                                    per sample, each dict
                                    {"pixel_values": (max_soft_tokens, model_flat_patch_dim) float,
                                     "image_position_ids": (max_soft_tokens, 2) int}.

         * per_sample_soft_token_counts : the grouped real-patch counts from `_prepare_visual_input`,
                                          one list per sample, one int per image.

         * input_ids : the tokenized, left-padded, possibly-truncated batch, shape (batch, seq).
                       Row i holds sample i's tokens; its <|image|> placeholders are counted to
                       learn how many real patches sample i is allowed to keep.

        Returns:
            A tuple (trimmed_dicts, trimmed_counts) with the same per-sample structure as the
            inputs, where any sample with more real patches than surviving placeholders has had its
            trailing real patches turned into padding (last image first) so that, per sample,
            sum(real patches) == number of surviving <|image|> placeholders. The input structures
            are not modified in place.
        """
        image_token_id = self.tokenizer.image_token_id
        trimmed_dicts: list[list[dict[str, torch.Tensor]]] = []
        trimmed_counts: list[list[int]] = []

        for sample_idx, (sample_dicts, sample_counts) in enumerate(
            zip(per_sample_image_dicts, per_sample_soft_token_counts)
        ):
            surviving = int((input_ids[sample_idx] == image_token_id).sum())
            excess = sum(sample_counts) - surviving

            if excess <= 0:
                # Nothing was dropped for this sample; keep its data as-is.
                trimmed_dicts.append(sample_dicts)
                trimmed_counts.append(list(sample_counts))
                continue

            new_sample_dicts = list(sample_dicts)
            new_sample_counts = list(sample_counts)
            # Remove extra real patches last-image-first, trailing patches first.
            for img_idx in range(len(new_sample_dicts) - 1, -1, -1):
                if excess <= 0:
                    break
                pos = new_sample_dicts[img_idx]["image_position_ids"]
                real_idx = (pos >= 0).all(dim=-1).nonzero(as_tuple=True)[0]  # this image's real rows
                remove = min(excess, real_idx.numel())
                if remove == 0:
                    continue
                new_pos = pos.clone()
                new_pos[real_idx[real_idx.numel() - remove:]] = -1          # mark trailing reals as padding
                new_sample_dicts[img_idx] = {
                    "pixel_values": new_sample_dicts[img_idx]["pixel_values"],
                    "image_position_ids": new_pos,
                }
                new_sample_counts[img_idx] -= remove
                excess -= remove

            trimmed_dicts.append(new_sample_dicts)
            trimmed_counts.append(new_sample_counts)

        return trimmed_dicts, trimmed_counts
    # --------------------- END OF CHANGE ---------------------

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for nanoVLM")

    def get_benchmark_formatting(self, task_name: str) -> dict:
        """Get benchmark-specific formatting rules."""
        benchmark_formats = {
            ("ai2d", "mmstar", "seedbench", "scienceqa"): { #   
                "text_replacements": {
                    "\nOptions:": "\nChoices:",
                    "\nA. ": "\nChoices:\nA. ",
                    "Please select the correct answer from the options above.": "Answer with the letter.",
                    "Answer with the option's letter from the given choices directly": "Answer with the letter directly",
                },
                "assistant_prefix": "Answer:",
                "user_prefix": "",
                "user_suffix": ""
            },
            ("docvqa_val", "docvqa_test"): {
                "text_replacements": {},
                "assistant_prefix": "",
                "user_prefix": "Give a short and terse answer to the following question. "
                                + "Do not paraphrase or reformat the text you see in the image. Do not include any full stops. "
                                + "Just give the answer without additional explanation. Question: ",
                "user_suffix": ""
            },
            "chartvqa": {
                "text_replacements": {},
                "assistant_prefix": "",
                "user_prefix": "For the question below, follow the following instructions:\n"
                                + "-The answer should contain as few words as possible.\n"
                                + "-Don't paraphrase or reformat the text you see in the image.\n"
                                + "-Answer a binary question with Yes or No.\n"
                                + "-When asked to give a numerical value, provide a number like 2 instead of Two.\n"
                                + "-If the final answer has two or more items, provide it in the list format like [1, 2].\n"
                                + "-When asked to give a ratio, give out the decimal value like 0.25 instead of 1:4.\n"
                                + "-When asked to give a percentage, give out the whole value like 17 instead of decimal like 0.17%.\n"
                                + "-Don't include any units in the answer.\n"
                                + "-Do not include any full stops at the end of the answer.\n"
                                + "-Try to include the full label from the graph when asked about an entity.\n"
                                + "Question: ",
                "user_suffix": ""
            },
            ("textvqa_val", "textvqa_test"): {
                "text_replacements": {},
                "assistant_prefix": "",
                "user_prefix": "Answer the following question about the image using as few words as possible. "
                                + "Follow these additional instructions:\n"
                                + "-Always answer a binary question with Yes or No.\n"
                                + "-When asked what time it is, reply with the time seen in the image.\n"
                                + "-Do not put any full stops at the end of the answer.\n"
                                + "-Do not put quotation marks around the answer.\n"
                                + "-An answer with one or two words is favorable.\n"
                                + "-Do not apply common sense knowledge. The answer can be found in the image.\n"
                                + "Question: ",
                "user_suffix": ""
            },
            ("mmmu_val", "mmmu_test"): {
                "text_replacements": {
                    "Question:": "",
                    "Answer with the option's letter from the given choices directly.": "Answer with the letter directly.",
                    "\nA. ": "\nChoices:\nA. "
                },
                "assistant_prefix": "Answer:",
                "user_prefix": "",
                "user_suffix": ""
            },
            ("infovqa_val", "mme", "ocrbench"): {
                "text_replacements": {},
                "assistant_prefix": "",
                "user_prefix": "",
                "user_suffix": "\nGive a very brief answer."
            }
        }
        
        # Check individual task names first
        if task_name in benchmark_formats:
            return benchmark_formats[task_name]
        
        # Check if task is in any list/tuple keys
        for key, formatting in benchmark_formats.items():
            if isinstance(key, (list, tuple)) and task_name in key:
                return formatting
        
        # Default formatting
        return {"text_replacements": {}, "assistant_prefix": "", "user_prefix": "", "user_suffix": ""}
    
    def apply_benchmark_formatting(self, context_str: str, prompt: str, task_name: str) -> tuple[str, str]:
        """Apply benchmark-specific formatting to context and prompt."""
        formatting = self.get_benchmark_formatting(task_name)
        
        # Add user prefix to context
        if formatting["user_prefix"]:
            context_str = formatting["user_prefix"] + context_str
        
        # Apply text replacements to context
        for old_text, new_text in formatting["text_replacements"].items():
            context_str = context_str.replace(old_text, new_text)
        
        # Add user suffix to context
        if formatting["user_suffix"]:
            context_str = context_str + formatting["user_suffix"]
        
        # Add assistant prefix to prompt
        if formatting["assistant_prefix"]:
            prompt = prompt + formatting["assistant_prefix"]

        return context_str, prompt

    # ------------------------ CHANGE ------------------------
    def _assemble_prompt(self, context: str, soft_token_counts: list[int], task_name: str) -> str:
        """
        Purpose:
            Build the exact prompt string one sample is fed to the tokenizer, on the encoder-free
            path. This is the single source of truth for prompt assembly: generate_until calls it
            once per sample. It (1) applies the benchmark's context rewrites/prefixes/suffixes,
            (2) prepends one <|image|> placeholder per real image patch (so the count matches the
            image features the model will splice in), (3) wraps the result as a user turn and
            renders the chat template with the generation prompt, and (4) appends the benchmark's
            assistant cue (e.g. "Answer:") at the very end, where it steers the first generated
            token.

        Parameters:
         * context : the task context for this sample, i.e. task.doc_to_text(doc).

         * soft_token_counts : this sample's real-patch counts, one int per image (the sample's
                               entry from `_prepare_visual_input`'s counts). Empty for a text-only
                               sample, which yields no placeholders.

         * task_name : the lmms-eval task name, used to pick the benchmark formatting.

        Returns:
            The assembled prompt string, ending in the benchmark's assistant cue when it defines
            one, ready to tokenize.
        """
        # 1. benchmark text replacements / prefixes / suffixes on the context
        context, _ = self.apply_benchmark_formatting(context, "", task_name)
        # 2. one <|image|> placeholder per real patch, prepended before the text
        image_string = get_image_string_encoder_free(self.tokenizer, soft_token_counts)
        prompt_content = image_string + context
        # 3. wrap as a user turn and render the chat template (adds the generation prompt)
        messages = [{"role": "user", "content": prompt_content}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # 4. append the benchmark's assistant cue at the very end
        _, prompt = self.apply_benchmark_formatting("", prompt, task_name)
        return prompt
    # --------------------- END OF CHANGE ---------------------

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            # ------------------------ CHANGE ------------------------
            # Encoder-free rewire: keep the batch's images grouped per sample (no flatten), build
            # each sample's placeholder run from its own real-patch counts, and let any prep error
            # propagate -- the old broad try/except turned a real bug into "" predictions, i.e.
            # silently wrong scores.
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            # visuals: one entry per sample -- a list of that sample's images, or None if text-only.
            visuals = [dtv(self.task_dict[t][s][i]) for dtv, i, t, s in zip(doc_to_visual, doc_id, task, split)]
            dicts, counts = self._prepare_visual_input(visuals)

            # One assembled prompt per sample (single source of truth: _assemble_prompt).
            prompts = [self._assemble_prompt(contexts[i], counts[i], task[i]) for i in range(len(contexts))]
            # --------------------- END OF CHANGE ---------------------

            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding="longest",
                padding_side="left",
                truncation=True,
                max_length=self.max_length
            )

            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            # ------------------------ CHANGE ------------------------
            # Truncation to max_length may have dropped some <|image|> placeholders; turn the
            # matching real patches back into padding so, per sample, real patches == surviving
            # placeholders, then stack the batch's images into the single dict the model consumes
            # (None when the batch has no images).
            dicts, counts = self._trim_to_placeholder_count(dicts, counts, input_ids)
            images_dict = self._build_images_dict(dicts)
            # --------------------- END OF CHANGE ---------------------

            # Extract generation parameters for the batch
            # We use the gen_kwargs from the first item in the chunk, assuming they are uniform for the batch.
            # lmms-eval groups requests by gen_kwargs, so this assumption should hold.
            current_gen_kwargs = all_gen_kwargs[0] if all_gen_kwargs else {}
            max_new_tokens = current_gen_kwargs.get("max_new_tokens", 50)
            temperature = current_gen_kwargs.get("temperature", 0.0) # Default to greedy
            top_p = current_gen_kwargs.get("top_p", 1.0)
            # Check if greedy generation is explicitly requested or implied by temperature 0
            greedy = current_gen_kwargs.get("do_sample", False) is False or temperature == 0.0
            # Pass None for temperature/top_p if greedy, as some HF models expect this
            gen_temperature = temperature if not greedy else None
            gen_top_p = top_p if not greedy else None
            
            # Generate
            generated_ids_batch = self.model.generate(
                input_ids,
                images_dict,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                greedy=greedy,
                temperature=gen_temperature,
                top_p=gen_top_p,
            )

            # Decode generated sequences
            # generated_ids_batch from model.generate usually contains only the generated tokens (excluding prompt)
            generated_texts = self.tokenizer.batch_decode(
                generated_ids_batch,
                skip_special_tokens=True
            )
            res.extend(generated_texts)
            pbar.update(len(contexts))

        pbar.close()

        # print(res)
        # re_ords.get_original() will sort the results back to the original order of requests
        return re_ords.get_original(res)

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("Multi Round Generation is not implemented for nanoVLM")
    
    @property
    def max_length(self):
        """Return the maximum sequence length."""
        return self.model.cfg.lm_max_position_embeddings 
    
    @property
    def batch_size_per_gpu(self):
        """Return the batch size."""
        return self.batch_size