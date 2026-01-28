from typing import Optional


class SiglipVisionConfig:
    def __init__(
        self,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        num_hidden_layers: int = 27,
        num_attention_heads: int = 16,
        num_channels: int = 3,
        image_size: int = 224,
        patch_size: int = 14,
        layer_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        **kwargs
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = (image_size // patch_size) ** 2


class GemmaConfig:
    def __init__(
        self,
        vocab_size: int = 257216,
        hidden_size: int = 2048,
        intermediate_size: int = 16384,
        num_hidden_layers: int = 18,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 1,
        head_dim: int = 256,
        max_position_embeddings: int = 8192,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        pad_token_id: Optional[int] = None,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id


class PaliGemmaConfig:
    def __init__(
        self,
        vision_config: Optional[dict] = None,
        text_config: Optional[dict] = None,
        projection_dim: int = 2048,
        hidden_size: int = 2048,
        pad_token_id: int = 0,
        image_token_index: int = 257152,
        vocab_size: int = 257216,
        **kwargs
    ):
        self.vision_config = SiglipVisionConfig(**(vision_config or {}))
        self.text_config = GemmaConfig(**(text_config or {}))
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.num_image_tokens = self.vision_config.num_image_tokens
