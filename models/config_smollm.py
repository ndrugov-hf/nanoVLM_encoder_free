from dataclasses import dataclass, field


@dataclass
class VLMConfig:
    vit_hidden_dim: int = 768
    vit_inter_dim: int = 4 * vit_hidden_dim
    vit_patch_size: int = 16
    vit_img_size: int = 512
    vit_n_heads: int = 12
    vit_dropout: float = 0.0
    vit_n_blocks: int = 12
    vit_ln_eps: float = 1e-6
    vit_cls_flag: bool = False
    vit_model_type: str = 'google/siglip2-base-patch16-512'

    lm_hidden_dim: int = 960
    lm_inter_dim: int = 2560
    lm_rms_eps: float = 1e-5
    lm_re_base: int = 100000
    lm_max_position_embeddings: int = 8192
    lm_base_vocab_size: int = 49152
    extra_token_amount: int = 66  # Number of extra tokens for the VLM (image start, image end, image token)
    lm_vocab_size: int = lm_base_vocab_size + extra_token_amount # Not a great way to do this, but it works for now (vlm_extra_tokens cannot be a dict, since this is mutable, and a Field has no len() function)
    lm_n_heads: int = 15
    lm_n_kv_heads: int = 5
    lm_dropout: float = 0.0
    lm_n_blocks: int = 32
    lm_attn_scaling: float = 1.0
    lm_max_length: int = 4096
    lm_use_tokens: bool = False # Decide if the LM expects tokens or embeddings as input (if using as a backbone for the VLM, set to False)
    lm_tie_weights: bool = True # Decide if you want to tie the LM Head weight to the token embedding weights
    lm_model_type: str = 'HuggingFaceTB/SmolLM2-360M-Instruct' #'HuggingFaceTB/SmolLM2-135M' #
    lm_tokenizer: str = 'HuggingFaceTB/SmolLM2-360M-Instruct'
    lm_chat_template: str = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    mp_pixel_shuffle_factor: int = 4
    mp_image_token_length: int = 64

    max_img_size: int = 2048
    resize_to_max_side_len: bool = True

    # ------------------------ CHANGE ------------------------
    # ------------------------------------------------------------------
    # Vision Embedder (encoder-free, Gemma 4 architecture)
    # First encoder-free step: these configure models/vision_embedder.py, which will replace
    # the ViT patch/feature extractor with direct raw-merged-pixel-patch embedding. The vit_*
    # / mp_* fields above are kept for now (the ViT path is still wired); they get removed
    # when the VLM is switched over to the embedder.
    # ------------------------------------------------------------------
    # Image patchification geometry (mirrors Gemma 4's image processor; the data-side
    # patchifier that actually produces these patches is a later step). A "teacher patch" is
    # teacher_patch_size x teacher_patch_size px; a "model patch" merges pooling_kernel_size^2
    # teacher patches.
    vision_backend: str = "encoder_free" # "encoder_free" or "vit"

    teacher_patch_size: int = 16
    pooling_kernel_size: int = 3
    model_patch_size: int = teacher_patch_size * pooling_kernel_size  # 48 px per model patch
    model_flat_patch_dim: int = 3 * (model_patch_size ** 2)           # 3*48*48 = 6912 raw values per patch
    max_soft_tokens: int = 280        # per-image budget of model patches the image processor emits (one of {70,140,280,560,1120}; Gemma 4 default 280). Distinct from mm_posemb_size below. Read by data/image_processing.py.

    # Embedder widths / init.
    mm_embed_dim: int = 1152          # intermediate embedder width (HYPERPARAMETER; Gemma 4 uses 3840). MP maps mm_embed_dim -> lm_hidden_dim at the wiring step; revisit then.
    mm_posemb_size: int = 1120        # factorized 2D pos-embedding table size per axis (>= max grid side in model patches; matches Gemma 4). Coupled to image processing; settle in step 2.
    emb_ln_eps: float = 1e-5          # LayerNorm eps in the embedder (Gemma 4 uses nn.LayerNorm's default, 1e-5)
    pos_embd_table_initializer_range: float = 0.02  # std for normal-init of the pos-embedding table (Gemma 4 zero-inits; small-normal here is a deliberate, documented deviation)

    # ------------------------ END OF CHANGE ------------------------

    vlm_extra_tokens: dict[str, str] = field(default_factory=lambda: {"image_token": "<|image|>", "global_image_token": "<|global_image|>",
      "r1c1": "<row_1_col_1>", "r1c2": "<row_1_col_2>", "r1c3": "<row_1_col_3>", "r1c4": "<row_1_col_4>", "r1c5": "<row_1_col_5>", "r1c6": "<row_1_col_6>", "r1c7": "<row_1_col_7>", "r1c8": "<row_1_col_8>",
      "r2c1": "<row_2_col_1>", "r2c2": "<row_2_col_2>", "r2c3": "<row_2_col_3>", "r2c4": "<row_2_col_4>", "r2c5": "<row_2_col_5>", "r2c6": "<row_2_col_6>", "r2c7": "<row_2_col_7>", "r2c8": "<row_2_col_8>",
      "r3c1": "<row_3_col_1>", "r3c2": "<row_3_col_2>", "r3c3": "<row_3_col_3>", "r3c4": "<row_3_col_4>", "r3c5": "<row_3_col_5>", "r3c6": "<row_3_col_6>", "r3c7": "<row_3_col_7>", "r3c8": "<row_3_col_8>",
      "r4c1": "<row_4_col_1>", "r4c2": "<row_4_col_2>", "r4c3": "<row_4_col_3>", "r4c4": "<row_4_col_4>", "r4c5": "<row_4_col_5>", "r4c6": "<row_4_col_6>", "r4c7": "<row_4_col_7>", "r4c8": "<row_4_col_8>",
      "r5c1": "<row_5_col_1>", "r5c2": "<row_5_col_2>", "r5c3": "<row_5_col_3>", "r5c4": "<row_5_col_4>", "r5c5": "<row_5_col_5>", "r5c6": "<row_5_col_6>", "r5c7": "<row_5_col_7>", "r5c8": "<row_5_col_8>",
      "r6c1": "<row_6_col_1>", "r6c2": "<row_6_col_2>", "r6c3": "<row_6_col_3>", "r6c4": "<row_6_col_4>", "r6c5": "<row_6_col_5>", "r6c6": "<row_6_col_6>", "r6c7": "<row_6_col_7>", "r6c8": "<row_6_col_8>",
      "r7c1": "<row_7_col_1>", "r7c2": "<row_7_col_2>", "r7c3": "<row_7_col_3>", "r7c4": "<row_7_col_4>", "r7c5": "<row_7_col_5>", "r7c6": "<row_7_col_6>", "r7c7": "<row_7_col_7>", "r7c8": "<row_7_col_8>",
      "r8c1": "<row_8_col_1>", "r8c2": "<row_8_col_2>", "r8c3": "<row_8_col_3>", "r8c4": "<row_8_col_4>", "r8c5": "<row_8_col_5>", "r8c6": "<row_8_col_6>", "r8c7": "<row_8_col_7>", "r8c8": "<row_8_col_8>"})
    vlm_load_backbone_weights: bool = True
    vlm_checkpoint_path: str = 'checkpoints'
    hf_repo_name: str = 'nanoVLM'

    # ------------------------ CHANGE ------------------------
    # New function
    def __post_init__(self):
    # Gemma 4 architecture: the vision embedder outputs the LM's hidden width, so mm_embed_dim
    # must equal lm_hidden_dim = the decoder's hidden_size. The VisionEmbedder is built before the
    # decoder, so resolve the decoder's hidden_size now (from its HF config) and pin both dims,
    # so the embedder/projector are built at the right width.
      if self.vision_backend == "encoder_free":
        from transformers import AutoConfig
        hidden_size = AutoConfig.from_pretrained(self.lm_model_type).hidden_size
        self.lm_hidden_dim = hidden_size
        self.mm_embed_dim = hidden_size
    # ------------------------ END OF CHANGE ------------------------



@dataclass
class TrainConfig:
    lr_mp: float = 0.00512
    lr_vision_backbone: float = 5e-5 #0.0005 #
    lr_language_backbone: float = 5e-5

    # ------------------------ CHANGE ------------------------
    # Added learning rates for components of the encoder-free VLM,
    # which are different from the components of the encoder-based VLM
    lr_vision_projector: float = 1e-4 # 0.00512
    lr_vision_embedder: float = 1e-4 # 0.00512

    val_size: int = 50000 # 50000
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    eval_in_epochs: bool = True
    eval_interval: int = 500
    stats_log_interval: int = 100
    max_training_steps: int = 40000
    max_images_per_example: int = 4
    max_images_per_knapsack: int = 18
    max_sample_length: int = 4096
    compile: bool = False
    resume_from_vlm_checkpoint: bool = False # Indicate if the training should be resumed from a checkpoint of the whole VLM or you want to start from scratch


    train_dataset_path: str = 'HuggingFaceM4/FineVision_concat_shuffled_2' # 'HuggingFaceM4/FineVision_concat_shuffled_2' or 'HuggingFaceM4/FineVision'
    train_dataset_name: tuple[str, ...] = ("default", ) #('allava_laion', 'allava_vflan', 'cambrian(filtered)_processed', 'LLaVA_Instruct_150K', 'mmevol', 'sharegpt4o', 'sharegpt4v(coco)', 'sharegpt4v(knowledge)', 'sharegpt4v(llava)', 'sharegpt4v(sam)') # 'vision_flan(filtered)', 'lvis_instruct4v',

    # train_dataset_path: str = 'HuggingFaceM4/FineVision'
    # train_dataset_name: tuple[str, ...] = ('funsd', )


    stream_dataset: bool = True
    relevance_min_rating: int = 1
    image_correspondence_min_rating: int = 1
    visual_dependency_min_rating: int = 1
    formatting_min_rating: int = 1
    wandb_entity: str = "your-wandb-entity" # Indicate the entity to log to in wandb
    log_wandb: bool = True
    use_lmms_eval: bool = True # Use lmms-eval for evaluation
    lmms_eval_tasks: str = 'mmstar,mme' # Pass additional task as one string, seperated by commas without spaces (e.g. 'mmstar,mmmu,ocrbench')
    lmms_eval_limit: float = None
    lmms_eval_batch_size: int = 64
