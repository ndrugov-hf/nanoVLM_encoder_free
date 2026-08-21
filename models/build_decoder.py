from __future__ import annotations

from models.config import VLMConfig
from models.decoder import Decoder
from models.language_model import LanguageModel


def build_decoder(cfg: VLMConfig, load_backbone: bool) -> Decoder | LanguageModel:
    """
    Purpose:
        Construct the language decoder for VisionLanguageModel, selecting between the
        custom SmolLM-style LanguageModel and the HuggingFace Decoder wrapper
        based on cfg.lm_backend.

    Parameters:
     * cfg (VLMConfig) : model configuration. cfg.lm_backend must be "custom"
       (hand-written Llama/GQA stack in language_model.py) or "hf" (any
       AutoModelForCausalLM, including LFM2.5-230M).

     * load_backbone (bool) : if True, load pretrained backbone weights from
       cfg.lm_model_type; if False, build a randomly-initialized skeleton (used
       when a full VLM checkpoint will supply weights via safetensors).

    Returns:
        An nn.Module exposing the VLM decoder contract:
         * token_embedding — input embedding table
         * head — output projection (vocab logits)
         * forward(embeddings, attention_mask) → (hidden_states, cache)
         * lm_use_tokens — bool; False in VLM embedding mode

    Raises:
        ValueError: if cfg.lm_backend is not "custom" or "hf".
    """
    if cfg.lm_backend == "hf":
        return Decoder(cfg, load_backbone)

    if cfg.lm_backend == "custom":
        if load_backbone:
            return LanguageModel.from_pretrained(cfg)
        return LanguageModel(cfg)

    raise ValueError(
        f"Unknown lm_backend={cfg.lm_backend!r}; expected 'custom' or 'hf'."
    )
