import warnings

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from models.config import VLMConfig
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.cache_utils import Cache


class Decoder(nn.Module):
    """
    Purpose:
        Thin wrapper around HuggingFace ``AutoModelForCausalLM``. Loads any HF causal LM
        (e.g. LiquidAI/LFM2.5-230M) and exposes the same surface API the VLM expects
        from the custom ``LanguageModel``: ``token_embedding``, ``head``, embedding-mode
        ``forward``, and ``lm_use_tokens``.

    Parameters (constructor):
     * cfg (VLMConfig) : must provide ``lm_model_type``, ``lm_hidden_dim``, and
       ``lm_use_tokens``. ``lm_hidden_dim`` must equal the loaded model's
       ``config.hidden_size``.

     * load_backbone (bool) : if True, download and load pretrained weights; if False,
       build a randomly-initialized model of the right architecture (for VLM checkpoint
       resume, where weights arrive via ``safetensors`` afterwards).
    """

    def __init__(self, cfg: VLMConfig, load_backbone: bool) -> None:
        super().__init__()

        # ------------------------ CHANGE ------------------------
        # Old:
        #   self.model = AutoModelForCausalLM.from_pretrained(cfg.lm_model_type)
        #   self.model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(cfg.lm_model_type))
        #   These load in the checkpoint's native dtype, which for LFM2/SmolLM is bfloat16.
        #   That gives the LM bf16 *master weights*, which (a) is inconsistent with the custom
        #   LanguageModel path and the fp32 ViT/ModalityProjector, and (b) fights the training
        #   recipe: train.py uses torch.autocast(bf16), which assumes fp32 master weights and
        #   casts to bf16 only for compute. bf16 master weights => degraded optimizer updates.
        #
        # New: force fp32 at load so master weights are fp32 everywhere; autocast handles the
        # bf16 compute. dtype=torch.float32 makes every parameter fp32 for both branches
        # (loading directly in fp32 avoids allocating a transient bf16 copy).
        if load_backbone:
            # Download pretrained weights
            self.model = AutoModelForCausalLM.from_pretrained(cfg.lm_model_type, dtype=torch.float32)
        else:
            # Initialize with random weights
            self.model = AutoModelForCausalLM.from_config(
                AutoConfig.from_pretrained(cfg.lm_model_type), dtype=torch.float32
            )
        # --------------------- END OF CHANGE ---------------------

        # Get dimension of vectors the model accepts as input
        self.hidden_size = self.model.config.hidden_size

        # ------------------------ CHANGE ------------------------
        # Old:
        #   assert self.hidden_size == cfg.lm_hidden_dim, (
        #       f"{cfg.lm_hidden_dim=} but decoder's {self.hidden_size=}"
        #   )
        #   The user had to hand-set cfg.lm_hidden_dim to each HF model's hidden size or
        #   construction failed. But lm_hidden_dim is the value ModalityProjector builds
        #   from, so the HF model should be the source of truth.
        #
        # New (Gap 4): propagate the HF model's real architecture into cfg so downstream
        # components and the saved config.json stay consistent instead of carrying the stale
        # SmolLM defaults. The HF model is the source of truth for its own dimensions. Each
        # field falls back to the existing cfg value when a given model doesn't expose that
        # attribute (attribute inventory/names differ across architectures).
        #
        # On the "hf" path only lm_hidden_dim is consumed at runtime (the ModalityProjector
        # builds from it), so its mismatch stays a soft warning; the rest are propagated for
        # config honesty + resume, since the HF backbone builds itself from its own config,
        # not from these cfg fields. Vocab is intentionally NOT set here — it is set later in
        # the VLM, after the embedding is resized to len(tokenizer), because the model's
        # native config.vocab_size can be padded (e.g. LFM2's 65536) and would not reflect
        # the real embedding row count.
        hf = self.model.config
        if cfg.lm_hidden_dim != self.hidden_size:
            warnings.warn(
                f"Overriding cfg.lm_hidden_dim ({cfg.lm_hidden_dim}) with the HF model's "
                f"hidden_size ({self.hidden_size}) from {cfg.lm_model_type!r}."
            )
        cfg.lm_hidden_dim = self.hidden_size
        cfg.lm_inter_dim = getattr(hf, "intermediate_size", cfg.lm_inter_dim)
        cfg.lm_n_heads = getattr(hf, "num_attention_heads", cfg.lm_n_heads)
        cfg.lm_n_kv_heads = getattr(hf, "num_key_value_heads", cfg.lm_n_kv_heads)
        cfg.lm_n_blocks = getattr(hf, "num_hidden_layers", cfg.lm_n_blocks)
        cfg.lm_max_position_embeddings = getattr(
            hf, "max_position_embeddings", cfg.lm_max_position_embeddings
        )
        cfg.lm_re_base = getattr(hf, "rope_theta", cfg.lm_re_base)
        cfg.lm_rms_eps = getattr(hf, "rms_norm_eps", cfg.lm_rms_eps)
        # --------------------- END OF CHANGE ---------------------

        # lm_use_tokens = True → "I am a normal standalone LM. Input is token ids and I return logits"
        # lm_use_tokens = False → "I am a backbone inside the VLM. Input is pre-computed embeddings and I return hidden states."
        # In this repo, lm_use_tokens is always False.
        self.lm_use_tokens = cfg.lm_use_tokens

        assert not self.lm_use_tokens, (
            "Decoder only supports backbone mode (lm_use_tokens=False): it takes embeddings "
            "and returns hidden states. Token-in/logits-out is the caller's job via "
            "token_embedding and head."
        )

    @property
    def token_embedding(self) -> nn.Module:
        # The input embedding matrix: maps token ids -> embedding vectors.
        out = self.model.get_input_embeddings()
        assert out is not None

        return out

    @property
    def head(self) -> nn.Module:
        # The output projection (LM head): maps hidden state vectors -> vocab logits.
        out = self.model.get_output_embeddings()
        assert out is not None

        return out

    @property
    def base(self) -> nn.Module:
        # The decoder stack without the LM head: calling it produces hidden states, not logits.
        out = self.model.get_decoder() if hasattr(self.model, "get_decoder") else self.model.model
        assert out is not None

        return out

    def forward(
        self,
        token_embd: Float[Tensor, "batch seq dim"],
        attention_mask: Int[Tensor, "batch seq"] | None = None,
        kv_cache: Cache | None = None,
        start_pos: int = 0
    ) -> tuple[Float[Tensor, "batch seq dim"], Cache | None]:
        """
        Purpose:
            Run input embeddings through the language backbone's decoder stack and
            return the final hidden states (the LM head is applied later, by the VLM).

        Parameters:
         * token_embd : input embeddings of shape (batch, seq, lm_hidden_dim)

         * attention_mask : padding mask of shape (batch, seq) — 1 for real tokens, 0 for
            padding — matching token_embd's batch and sequence dims. Defaults to None,
            meaning nothing is masked.

         * kv_cache : a single HF ``Cache`` object holding the backbone's per-layer
                      state (keys/values for attention layers, and for hybrid models
                      like LFM2 the rolling window for conv layers), used for efficient
                      autoregressive decoding. If None, the model creates a fresh cache
                      (when use_cache is on) or runs without one (during training).

         * start_pos : the absolute position of the first token in ``token_embd``.
                       Accepted for call-site compatibility with the custom LM protocol,
                       but not passed to the backbone — the HF model derives positions
                       from the cache length. Used here only as a sanity check that it
                       agrees with the cache.

        Returns:
            A tuple (last_hidden_state, past_key_values):
             * last_hidden_state : decoder hidden states of shape (batch, seq, dim).
             * past_key_values : the updated Cache holding the per-layer keys/values
               (and conv state, for hybrid models), to be fed back on the next decode
               step; None during training, where the cache is not built.
        """
        # start_pos is not threaded into the backbone (positions come from the cache
        # length), so assert the invariant that makes that safe instead of trusting it.
        if kv_cache is not None and start_pos != kv_cache.get_seq_length():
            raise ValueError(
                f"{start_pos=} disagrees with cache length {kv_cache.get_seq_length()}"
            )
        # `out` is a BaseModelOutputWithPast: a small container with four fields.
        # You read them by name (out.last_hidden_state) or by key (out["..."]).
        #
        #   out.last_hidden_state : (batch, seq, lm_hidden_dim) tensor of final
        #                           hidden states, already normalized. Always present.
        #   out.past_key_values   : the Cache holding each layer's state so the next
        #                           decode step can reuse it instead of recomputing.
        #                           IMPORTANT: this is None unless we pass use_cache=True.
        #                           For LFM2 it is a config-aware DynamicCache that stores
        #                           keys/values for attention layers AND the rolling window
        #                           for the short-conv layers.
        #   out.hidden_states     : per-layer hidden states; None unless output_hidden_states=True.
        #   out.attentions        : per-layer attention weights; None unless output_attentions=True.
        out = self.base(
                        inputs_embeds=token_embd, 
                        attention_mask=attention_mask,
                        past_key_values=kv_cache,
                        use_cache=not self.training,
                        )
        
        return out.last_hidden_state, out.past_key_values
