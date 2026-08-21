import json
import os
import tempfile
from dataclasses import asdict
from typing import Optional
from jaxtyping import Int, Float
from torch import Tensor


from models.utils import top_k_top_p_filtering
from models.vision_transformer import ViT

# -------------- CHANGE ----------
# Old:
#   from models.language_model import LanguageModel
#   The VLM referenced LanguageModel directly, hardcoding the custom backend.
# New: route decoder construction through build_decoder, which selects the
# custom LanguageModel or the HF Decoder wrapper based on cfg.lm_backend.
from models.build_decoder import build_decoder
from models.decoder import Decoder
# -------------- END OF CHANGE ----------

from models.modality_projector import ModalityProjector
from models.config import VLMConfig
# -------------- CHANGE ----------
# Added new imports
from models.vision_embedder import VisionEmbedder
from models.vision_projector import VisionProjector
# -------------- END OF CHANGE ----------

from data.processors import get_tokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model, save_model

class VisionLanguageModel(nn.Module):
    def __init__(self, cfg: VLMConfig, load_backbone=True):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = get_tokenizer(cfg.lm_tokenizer, cfg.vlm_extra_tokens, cfg.lm_chat_template)
        
        # ------------------------ CHANGE ------------------------
        # What vision pathway gets built now depends on cfg.vision_backend
        # Old:
        #
        # if load_backbone:
        #   print("Loading from backbone weights for ViT")
        #   self.vision_encoder = ViT.from_pretrained(cfg)
        # else:
        #   self.vision_encoder = ViT(cfg)
        #
        # New:
        if cfg.vision_backend == "encoder_free":
            self.vision_embedder = VisionEmbedder(cfg)
            self.vision_projector = VisionProjector(cfg)
        else:
            if load_backbone:
                print("Loading from backbone weights for ViT")
                self.vision_encoder = ViT.from_pretrained(cfg)
            else:
                self.vision_encoder = ViT(cfg)

            self.MP = ModalityProjector(cfg)

        self.decoder = build_decoder(cfg, load_backbone)

        # ------------------------ CHANGE ------------------------
        # Old:
        #   (nothing) — there was no resize call here. With the custom LanguageModel
        #   backend the embedding was built pre-enlarged inside LanguageModel.__init__
        #   (nn.Embedding(cfg.lm_vocab_size, ...) = base_vocab + 66) and from_pretrained
        #   initialized the extra rows, so the resize was implicit.
        #
        # New:
        #   The HF AutoModelForCausalLM arrives at its native vocab and knows nothing about
        #   our 66 VLM special tokens, so we resize its embedding to len(tokenizer) and
        #   init the new rows explicitly (needed for LFM, harmless for SmolLM).
        if cfg.lm_backend == "hf":
            assert isinstance(self.decoder, Decoder)  # hf backend => Decoder wrapper (narrows .model for the type checker)
            self.decoder.model.resize_token_embeddings(len(self.tokenizer))
            self._fit_embeddings_to_tokenizer()
            # Gap 4 (vocab): keep cfg's vocab in sync with the ACTUAL resized embedding.
            # After the resize the table has one row per tokenizer id (base tokenizer vocab
            # + the 66 VLM special tokens), which is authoritative — the model's native
            # config.vocab_size may be padded (e.g. LFM2's 65536) and does not reflect the
            # real row count. Setting cfg here (not in Decoder) is why it lives after resize:
            # only the VLM knows len(tokenizer).
            resized_vocab_size = self.decoder.model.get_input_embeddings().weight.shape[0]
            cfg.lm_vocab_size = resized_vocab_size
            cfg.lm_base_vocab_size = resized_vocab_size - cfg.extra_token_amount

            # ------------------------ CHANGE ------------------------
            # Old:
            #   (nothing) — cfg.lm_tie_weights was never applied on the "hf" path. Whether
            #   the input embedding and the output head shared one matrix was decided purely
            #   by the loaded HF model's own config default, and resize_token_embeddings
            #   silently re-ties on top of that. So the flag was a no-op here (it is honored
            #   only on the custom LanguageModel path, language_model.py:404-405).
            #
            # New:
            #   Enforce cfg.lm_tie_weights explicitly, and do it LAST (after resize + _fit)
            #   so it wins over resize's re-tying. Tie -> share one matrix; untie -> give the
            #   head its own independent weight (a clone of the embedding) so the two train
            #   separately. Also keep model.config in sync so a later save/reload remembers
            #   the choice.
            model = self.decoder.model
            model.config.tie_word_embeddings = cfg.lm_tie_weights
            if cfg.lm_tie_weights:
                # Share one matrix for the input embedding and the output head.
                model.tie_weights()
            else:
                # Untie: setting the config flag alone does NOT split the shared tensor;
                # the head must be given a new Parameter to actually separate them.
                model.get_output_embeddings().weight = nn.Parameter(
                    model.get_input_embeddings().weight.clone()
                )
            # --------------------- END OF CHANGE ---------------------
        # --------------------- END OF CHANGE ---------------------

        self.load_backbone = load_backbone

        # ------------------------ CHANGE ------------------------
        # Old:
        # self.tokenizer = get_tokenizer(cfg.lm_tokenizer, cfg.vlm_extra_tokens, cfg.lm_chat_template)
        # New: 'self.tokenizer = ...' moved up^
        # --------------------- END OF CHANGE ---------------------

    def _fit_embeddings_to_tokenizer(self) -> None:
        """
        Purpose:
            Reconcile the HF decoder's token-embedding table with the tokenizer once the
            vision special tokens have been registered. Re-initializes the rows belonging
            to the added vision special tokens so each starts as a distinct, small-random
            input embedding rather than stale padding values.

        Parameters:
            None — operates on ``self``.

        Returns:
            None. Mutates the embedding rows for ``cfg.vlm_extra_tokens`` with N(0, 0.02)
            samples. Only called when ``cfg.lm_backend == "hf"``.

        Raises:
            AssertionError: if ``len(self.tokenizer)`` exceeds the embedding row count
            after ``resize_token_embeddings``.
        """
        assert isinstance(self.decoder, Decoder)  # only called on the hf path => Decoder wrapper
        model = self.decoder.model
        n_rows_in_embd_matrix = model.get_input_embeddings().weight.shape[0]

        assert len(self.tokenizer) <= n_rows_in_embd_matrix, (
            f"tokenizer has {len(self.tokenizer)} tokens but the embedding only has "
            f"{n_rows_in_embd_matrix} rows; call resize_token_embeddings first"
        )

        extra_ids = [
            self.tokenizer.convert_tokens_to_ids(t)
            for t in self.cfg.vlm_extra_tokens.values()
        ]
        with torch.no_grad():
            embd_matrix: Float[Tensor, "vocab dim"] = model.get_input_embeddings().weight
            embd_matrix[extra_ids] = torch.empty(
                (len(extra_ids), embd_matrix.shape[1]),
                dtype=embd_matrix.dtype,
                device=embd_matrix.device,
            ).normal_(mean=0.0, std=0.02)

    def _replace_img_tokens_with_embd(self, input_ids, token_embd, image_embd):
        """
        Replace every image-token placeholder in `input_ids` with the corresponding slice
        from `image_embd`. Supports an arbitrary number of image-token placeholders per sample.
        The first example in the batch might have 2 images and the second none.
        """
        # Clone the original embeddings to avoid in-place issues
        updated_token_embd = token_embd.clone()

        # Build a mask of all image-token positions: shape [B, T_seq]
        mask = (input_ids == self.tokenizer.image_token_id)
        updated_token_embd[mask] = image_embd.view(-1, image_embd.size(-1)).to(updated_token_embd.dtype) # torch flattens before assigning

        return updated_token_embd

    def _process_images(self, images, device):
        # ------------------------ CHANGE ------------------------
        # Added the if-else branching. Before, there was no branching, and the vision_backend="vit"
        # branch was always executed
        if self.cfg.vision_backend == "encoder_free":
            # If the encoder-free path is followed, 'images' is already either a None or a dictionary
            # In case 'images' is a dictionary, we want to move its values to device
            if isinstance(images, dict):
                images_on_device = {
                                    "pixel_values": images["pixel_values"].to(device),
                                    "image_position_ids": images["image_position_ids"].to(device),
                }
                images = images_on_device

            return images
        
        else:
            if isinstance(images, list):
                if images and isinstance(images[0], list):
                    images = [img for sublist in images for img in sublist]

                if not images:  # Handle cases with no images
                    return None
                else:
                    return torch.cat(images, dim=0).to(device)
            return images # Already a tensor
        # --------------------- END OF CHANGE ---------------------

    # ------------------------ CHANGE ------------------------
    # New function that is used in both forward() and generate()
    def _embed_inputs(
        self,
        input_ids: Int[Tensor, "batch seq"],
        images,
    ) -> Float[Tensor, "batch seq lm_hidden_dim"]:
        """
        Purpose:
            Turn a batch of token ids into the input embeddings the decoder consumes. Shared by
            forward() and generate() so both embed the sequence identically. Look up the text
            embedding for every token id; then, if the batch has images, replace each <|image|>
            placeholder embedding with the matching image feature vector from the active vision
            backend. Encoder-free: run the patches through VisionEmbedder -> VisionProjector and
            keep only the real (non-filler) rows before writing them in. ViT: run the image
            tensor through the ViT encoder -> ModalityProjector.

        Parameters:
         * input_ids : token ids for the batch, left-padded to a common length, containing one
                       <|image|> placeholder id per image feature vector to be written in.
                       Shape (batch, seq).

         * images : the batch's images in the form the active backend expects, or None for a
                    text-only batch. Encoder-free: a dict
                    {"pixel_values": (num_images, patches, model_flat_patch_dim),
                     "image_position_ids": (num_images, patches, 2)}, where a filler patch has
                    position (-1, -1). ViT: a list of per-image (3, H, W) tensors (or a batch of
                    such lists).

        Returns:
            Input embeddings of shape (batch, seq, lm_hidden_dim): the text embeddings
            with the <|image|> positions overwritten by image feature vectors. Returned unchanged
            (text embeddings only) when images is None.

        Raises:
            RuntimeError: if the number of <|image|> placeholder tokens in input_ids does not
            equal the number of image feature vectors to write in. The write-in is a masked
            assignment, which requires the two counts to match.
        """
        token_embd = self.decoder.token_embedding(input_ids) # [B, T_sequence, D_lm]
        if self.cfg.vision_backend == "encoder_free":
            processed_image_data: dict | Tensor | None = self._process_images(images, token_embd.device)

            # Produce image token embeddings ONLY IF 'images' is not None
            if processed_image_data is not None:
                # If we are in the encoder-free path, 'images' must be either None or a dictionary
                assert isinstance(processed_image_data, dict), "'processed_image_data' must be a dictionary if this branch is executed"

                image_embd: Float[Tensor, "num_images patch mm_embed_dim"] = self.vision_embedder(processed_image_data["pixel_values"],
                                                                                             processed_image_data["image_position_ids"])
                image_embd: Float[Tensor, "num_images patch lm_hidden_dim"] = self.vision_projector(image_embd)

                # Leave image embeddings only for the true image patches, not the padding patches
                model_patch_positions = processed_image_data["image_position_ids"] # (num_images, patch, 2)
                true_patches_mask = (model_patch_positions != -1).all(dim=-1) # (num_images, patch)
                image_embd = image_embd[true_patches_mask] # (num_true_patches_from_batch, lm_hidden_dim)

                # Replace embeddings of <|image|> tokens with embeddings of real image patches
                # _replace_img_tokens_with_embd always flattens token_embd into a 2D tensor, so 
                # it's ok that token_embd is a 3D tensor and image_embd is a 2D tensor
                token_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, image_embd)
            
        else:
            images_tensor = self._process_images(images, input_ids.device)
            
            if images_tensor is not None:
                image_embd = self.vision_encoder(images_tensor)
                image_embd = self.MP(image_embd)  # [num_images, mp_image_token_length, D_lm]
                token_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, image_embd)

        return token_embd
    # --------------------- END OF CHANGE ---------------------

    def forward(self, input_ids, images, attention_mask=None, targets=None):
        # ------------------------ CHANGE ------------------------
        # Use the new _embed_inputs function to get embeddings for text tokens,
        # get embedings for image tokens, and combine them into a unified tensor of embeddings
        token_embd: Float[Tensor, "batch seq lm_hidden_dim"] = self._embed_inputs(input_ids, images)
        # --------------------- END OF CHANGE ---------------------

        logits, _ = self.decoder(token_embd, attention_mask=attention_mask)
    
        loss = None
        if targets is not None:
            logits = self.decoder.head(logits) # Apply LM head
            # Loss is calculated over all tokens, but `targets` (labels) will have -100 for non-answer tokens.
            # No need to slice logits based on image embedding size here, as the target mask handles it.
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)

        return logits, loss

    @torch.inference_mode()
    def generate(self, input_ids, images, attention_mask=None, max_new_tokens=5, top_k=50, top_p=0.9, temperature=0.5, greedy=False):
        # ------------------------ CHANGE ------------------------
        # Use the new _embed_inputs function to get embeddings for text tokens,
        # get embedings for image tokens, and combine them into a unified tensor of embeddings
        token_embd: Float[Tensor, "batch seq lm_hidden_dim"] = self._embed_inputs(input_ids, images)
        # --------------------- END OF CHANGE ---------------------

        current_total_seq_len = token_embd.size(1)
        batch_size = input_ids.size(0) # Or token_embd.size(0)
        
        # --- Multimodal Prefill Phase ---
        prefill_output, kv_cache_list = self.decoder(
            token_embd,
            attention_mask=attention_mask, # Use the provided attention mask
            kv_cache=None,
            start_pos=0
        )
        
        last_token_output_from_prefill = prefill_output[:, -1, :] # (B, D_lm) if self.decoder.lm_use_tokens=False, else (B, V_size)
        
        if not self.decoder.lm_use_tokens:
            current_logits = self.decoder.head(last_token_output_from_prefill)  # (B, V_size)
        else:
            current_logits = last_token_output_from_prefill # (B, V_size)

        # Store newly generated token IDs
        newly_generated_ids_list = []

        # --- Decode Phase by sampling tokens autoregressively using the kv-cache ---
        for _ in range(max_new_tokens):
            if greedy:
                next_token_id = torch.argmax(current_logits, dim=-1, keepdim=True) # (B, 1)
            else:
                filtered_logits = top_k_top_p_filtering(current_logits, top_k=top_k, top_p=top_p)
                probs = torch.softmax(filtered_logits / temperature, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            newly_generated_ids_list.append(next_token_id) # list[ Int[Tensor, "B 1"] ]
            
            # Embed the newly generated token
            next_token_embed = self.decoder.token_embedding(next_token_id) # [B, 1, D_lm]
            
            # The start_pos for the new token is the current total sequence length *before* adding this new token
            current_token_start_pos = current_total_seq_len
            current_total_seq_len += 1

            # update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat((attention_mask, torch.ones((batch_size, 1), device=attention_mask.device, dtype=attention_mask.dtype)), dim=1)

            # With KV cache: only process the new token
            decode_step_output, kv_cache_list = self.decoder(
                next_token_embed,
                attention_mask=attention_mask,
                kv_cache=kv_cache_list,
                start_pos=current_token_start_pos
            )
      
            last_token_output = decode_step_output[:, -1, :] 
            
            # Apply head to get logits (if model is in embedding mode)
            if not self.decoder.lm_use_tokens:
                current_logits = self.decoder.head(last_token_output)
            else:
                current_logits = last_token_output
        
        if not newly_generated_ids_list: # Handle case where max_new_tokens might be 0
            return torch.empty((batch_size,0), dtype=torch.long, device=input_ids.device)

        generated_ids = torch.cat(newly_generated_ids_list, dim=1)

        # Post-process to handle EOS token.
        if self.tokenizer.eos_token_id is not None and generated_ids.numel() > 0: # Ensure generated_ids is not empty
            seq_len = generated_ids.size(1)
            device = generated_ids.device

            eos_mask = (generated_ids == self.tokenizer.eos_token_id) # Create a boolean mask for EOS tokens

            col_indices_for_min = torch.arange(seq_len, device=device) # Create column indices [0, 1, ..., seq_len-1]
            
            # In eos_mask, mark positions with actual col_idx, others with a large number
            masked_col_indices = torch.where(eos_mask, col_indices_for_min.unsqueeze(0).expand_as(generated_ids), seq_len + 1) 

            first_eos_indices_values = torch.min(masked_col_indices, dim=1).values
            
            # Clamp values to seq_len (if no EOS found, min will be seq_len + 1, clamp brings it to seq_len0. This means if no EOS, or EOS is the last token, no replacement will happen for that sample.
            actual_first_eos_indices = torch.clamp(first_eos_indices_values, max=seq_len)

            # Create column indices for comparison, shape [batch_size, seq_len]
            col_indices_for_comparison = torch.arange(seq_len, device=device).unsqueeze(0).expand_as(generated_ids)
            
            # Tokens are replaced if their column index is greater than the index of the first EOS token
            replace_mask = col_indices_for_comparison > actual_first_eos_indices.unsqueeze(1)
            
            generated_ids[replace_mask] = self.tokenizer.eos_token_id
        
        return generated_ids

    @classmethod
    def from_pretrained(
        cls, repo_id_or_path: str, *, revision: str | None = None
    ) -> "VisionLanguageModel":
        """
        Load a VisionLanguageModel from a local directory or a repo on the Hugging Face Hub.

        Args:
            repo_id_or_path (str): The path to the local directory or the Hugging Face Hub repo ID.

        Returns:
            VisionLanguageModel: The loaded model.
        """
        # If local folder exists => load from there
        if os.path.exists(repo_id_or_path):
            config_path = os.path.join(repo_id_or_path, "config.json")
            weights_path = os.path.join(repo_id_or_path, "model.safetensors")

            if not os.path.exists(config_path):
                raise ValueError(
                    f"Config file not found at {config_path}. Please provide a valid path."
                )
            if not os.path.exists(weights_path):
                raise ValueError(
                    f"Weights file not found at {weights_path}. Please provide a valid path."
                )
        # Otherwise, assume it's a Hugging Face Hub repo
        else:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="config.json", revision=revision
            )
            weights_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="model.safetensors", revision=revision
            )

        # Load config
        with open(config_path, "r") as f:
            cfg = VLMConfig(**json.load(f))

        # Initialize model without loading the backbone
        model = cls(cfg, load_backbone=False)

        # Load safetensors weights
        load_model(model, weights_path)

        # Done!
        return model

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save the model and configuration to a directory.

        Args:
            save_directory (str): The directory to save the model and config.
        """
        # Create directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)

        # Save config
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            f.write(json.dumps(asdict(self.cfg), indent=4))

        # Save weights as safetensors
        save_model(self, os.path.join(save_directory, "model.safetensors"))

    def push_to_hub(self, repo_id: str, private: bool = False):
        """
        Push the model and configuration to the Hugging Face Hub.

        Args:
            repo_id (str): The repo ID on the Hugging Face Hub.
        """
        from huggingface_hub import create_repo, upload_folder

        # Create repo
        repo_url = create_repo(repo_id=repo_id, private=private, exist_ok=True)
        repo_id = repo_url.repo_id
        print("Created repo: ", repo_url)

        with tempfile.TemporaryDirectory() as save_path:
            # Save to tmp directory
            self.save_pretrained(save_path)

            # Save model card
            with open(os.path.join(save_path, "README.md"), "w") as f:
                f.write(MODEL_CARD_TEMPLATE.format(repo_id=repo_id))

            # Upload
            return upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=save_path,
                commit_message="Upload nanoVLM using push_to_hub",
            )


MODEL_CARD_TEMPLATE = """
---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
library_name: nanovlm
license: mit
pipeline_tag: image-text-to-text
tags:
  - vision-language
  - multimodal
  - research
---

**nanoVLM** is a minimal and lightweight Vision-Language Model (VLM) designed for efficient training and experimentation. Built using pure PyTorch, the entire model architecture and training logic fits within ~750 lines of code. It combines a ViT-based image encoder (SigLIP-B/16-224-85M) with a lightweight causal language model (SmolLM2-135M), resulting in a compact 222M parameter model.

For more information, check out the base model on https://huggingface.co/lusxvr/nanoVLM-222M.

**Usage:**

Clone the nanoVLM repository: https://github.com/huggingface/nanoVLM.
Follow the install instructions and run the following code:

```python
from models.vision_language_model import VisionLanguageModel

model = VisionLanguageModel.from_pretrained("{repo_id}")
```
"""
