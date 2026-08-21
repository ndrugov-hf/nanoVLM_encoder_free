import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.config import VLMConfig

from jaxtyping import Int, Float
from torch import Tensor

class VisionEmbedder(nn.Module):
    def __init__(self, cfg: VLMConfig):
        super().__init__()
        
        self.cfg = cfg 
        self.model_flat_patch_dim = cfg.model_flat_patch_dim
        self.mm_embed_dim = cfg.mm_embed_dim

        self.mm_posemb_size = cfg.mm_posemb_size
        self.emb_ln_eps = cfg.emb_ln_eps
        self.pos_embd_table_initializer_range = cfg.pos_embd_table_initializer_range

        # Patch embedding: LN₁ → Dense → LN₂
        self.patch_ln1 = nn.LayerNorm(self.model_flat_patch_dim, eps=self.emb_ln_eps)
        self.patch_dense = nn.Linear(self.model_flat_patch_dim, self.mm_embed_dim)
        self.patch_ln2 = nn.LayerNorm(self.mm_embed_dim, eps=self.emb_ln_eps)

        # Factorized 2D positional embedding
        self.pos_embedding = nn.Parameter(torch.randn((self.mm_posemb_size, 2, self.mm_embed_dim)) * self.pos_embd_table_initializer_range)
        self.pos_norm = nn.LayerNorm(self.mm_embed_dim, eps=self.emb_ln_eps)

    def forward(        
        self,
        flattened_patches: Float[Tensor, "batch patches model_flat_patch_dim"],
        model_patch_positions: Int[Tensor, "batch patches 2"]
    ) -> Float[Tensor, "batch patches mm_embed_dim"]:
        """
        Purpose:
            Given a batch of flattened patches, return a batch of patch embeddings. 
            For each patch:
                - normalize it
                - project it to the LM's embedding dim
                - add a learned 2D positional embedding

        Parameters:
            * flattened_patches : batch of sequences of flattened model patches. Each sequence contains
                                patches concatenated across all samples and all images.

            * model_patch_positions : tensor containing each model patch's (x, y) position within its image. 
                                    The order of (x, y) positions matches the order of patches in x.
                                    Must contain no -1 padding positions (padding patches are dropped 
                                    before reaching the embedder).

        Returns:
            Tensor of patch embeddings of shape (batch, patches, mm_embed_dim)
        """
        # Step 1: Patch embedding (LN → Dense → LN)

        # patch_dense below multiplies the patches by its weight, and a matmul needs both sides
        # to have the same dtype. The patches come in as float32 from the image processor, but the
        # module's weights may be a different float type (e.g. bfloat16 ), so cast the patches 
        # to the weight's dtype to avoid a dtype-mismatch error.
        # The is_floating_point check skips the cast when the weights are a non-float dtype (e.g.
        # a quantized integer type), which float patches must not be cast into.
        target_dtype = self.patch_dense.weight.dtype
        if target_dtype.is_floating_point:
            flattened_patches = flattened_patches.to(target_dtype)
        
        hidden_states = self.patch_ln1(flattened_patches) # (batch, patches, model_flat_patch_dim)
        hidden_states = self.patch_dense(hidden_states) # (batch, patches, mm_embed_dim)
        hidden_states = self.patch_ln2(hidden_states) # (batch, patches, mm_embed_dim)

        # Step 2: Add factorized positional embeddings + LN
        clamped = model_patch_positions.clamp(min=0).long() # Padded patches have position (-1, -1). Indexing pos_embedding with -1 
                                                        # does not crash in PyTorch — -1 is a valid negative index that wraps
                                                        # around to the last row. If you replace each -1 with 0, 
                                                        # you know which row it lands on, instead of relying on wraparound behavior.
        valid = (model_patch_positions != -1).to(self.pos_embedding.dtype).unsqueeze(-1) # (batch, patches, 2, 1)
        axes = torch.arange(2, device=model_patch_positions.device) # (2,)
        pos_embs = (self.pos_embedding[clamped, axes] * valid).sum(-2) # (batch, patches, mm_embed_dim)
        hidden_states = hidden_states + pos_embs
        hidden_states = self.pos_norm(hidden_states)

        return hidden_states



"""
Explanation of pos_embs = = (self.pos_embedding[clamped, axes]):

    Assume mm_posemb_size = 5, batch = 3, mm_embed_dim = 5, patches = 4
    
    pos_embedding = [
                      [ [a0 a1 a2 a3 a4], [b0 b1 b2 b3 b4] ],
                      [ [c0 c1 c2 c3 c4], [d0 d1 d2 d3 d4] ],
                      [ [e0 e1 e2 e3 e4], [f0 f1 f2 f3 f4] ],
                      [ [g0 g1 g2 g3 g4], [h0 h1 h2 h3 h4] ],
                      [ [i0 i1 i2 i3 i4], [j0 j1 j2 j3 j4] ],
                    ] 
                    (shape = (mm_posemb_size, 2, mm_embed_dim))

    model_patch_positions = [ 
                              [ [2, 3],   [9, 10],   [4, 5],   [1, 1] ], 
                              [ [4, 4],   [1, 8],   [-1, -1], [-1, -1] ],
                              [ [-1, -1], [-1, -1], [-1, -1], [-1, -1] ],
                            ]
                            (shape = (batch, patches, 2))
    
    clamped = [ 
                [ [2, 3],   [9, 10],   [4, 5],   [1, 1] ], 
                [ [4, 4],   [1, 8], [0, 0], [0, 0] ],
                [ [0, 0], [0, 0], [0, 0], [0, 0] ],
              ]
              (shape = (batch, patches, 2))

    axes = [0, 1]
           shape = (2, )

    pos_embedding[clamped, axes]
    1) axes gets broadcasted to (batch, patches, 2)
              [ 
                [ [0, 1], [0, 1], [0, 1], [0, 1] ], 
                [ [0, 1], [0, 1], [0, 1], [0, 1] ],
                [ [0, 1], [0, 1], [0, 1], [0, 1] ],
              ]
              (shape = (batch, patches, 2))

    2) PyTorch creates a tensor of shape (batch, patches, 2, mm_embed_dim),
       where out[b, p, a] = pos_embedding[ clamped[b, p, a], axes[b, p, a] ] = pos_embedding[ clamped[b, p, a], a ]
"""