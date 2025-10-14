from transformers.configuration_utils import PretrainedConfig
from typing import Optional, Any
from transformers.utils import logging

logger = logging.get_logger(__name__)


class SuryaEncoderConfig(PretrainedConfig):
    model_type = "surya-encoder"
    base_config_key = "vision_config"

    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "depth",
    }

    def __init__(
        self,
        # Vision encoder dims
        embed_dim: int = 1536,  # vision encoder embed size
        hidden_size: int = 1536,  # after merger hidden size
        intermediate_size: int = 4224,
        num_hidden_layers: int = 42,
        num_attention_heads: int = 12,
        # Patch/grid parameters
        num_channels: int = 3,
        patch_size: int = 14,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 1,
        # Norm/attn/initialization
        rms_norm_eps: float = 1e-5,
        use_bias: bool = False,
        attn_implementation: str = "flash_attention_2",  # "eager","sdpa","flash_attention_2"
        initializer_range: float = 0.02,
        init_merger_std: float = 0.02,
        is_causal: bool = False,  # vision encoder causal forward
        post_norm: bool = True,
        gradient_checkpointing: bool = False,
        # Optional fixed image size for convenience (not required by the model itself)
        image_size: Optional[int | tuple[int, int]] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        # Some downstream utilities expect `spatial_patch_size` like Qwen
        self.spatial_patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.rms_norm_eps = rms_norm_eps
        self.use_bias = use_bias
        self.attn_implementation = attn_implementation
        self.initializer_range = initializer_range
        self.init_merger_std = init_merger_std
        self.is_causal = is_causal
        self.post_norm = post_norm
        self.gradient_checkpointing = gradient_checkpointing
        self.image_size = image_size

