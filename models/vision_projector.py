# New implementation
import torch.nn as nn
from models.config import VLMConfig

from jaxtyping import Int, Float
from torch import Tensor

class VisionProjector(nn.Module):
    def __init__(self, cfg: VLMConfig):
        super().__init__()

        self.cfg = cfg
        self.input_dim = cfg.mm_embed_dim
        self.output_dim = cfg.lm_hidden_dim
        self.lm_rms_eps = cfg.lm_rms_eps 

        # Multimodal projection: RMSNorm → Linear
        self.norm = nn.RMSNorm(self.input_dim, eps=self.lm_rms_eps)
        self.proj = nn.Linear(self.input_dim, self.output_dim, bias=False)

    def forward(
        self, 
        x: Float[Tensor, "batch patches mm_embed_dim"]
    ) -> Float[Tensor, "batch patches lm_hidden_dim"]:
        # Additional dtype casting
        target_dtype = self.proj.weight.dtype
        if target_dtype.is_floating_point:
            x = x.to(target_dtype)
        
        hidden_states = self.norm(x) # (batch, patches, mm_embed_dim)
        hidden_states = self.proj(hidden_states) # (batch, patches, lm_hidden_dim)

        return hidden_states