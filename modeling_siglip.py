import torch
import torch.nn as nn

class SiglipVisionModelConfig:

    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_channels: int = 3,
        image_size: int = 224,
        patch_size: int = 16,
        layer_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        num_image_tokens: int = None,
        **kwargs
    ):
        
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens


class SiglipVisionTransformer(nn.Module):

    def __init__(self, config: SiglipVisionModelConfig):
        super(SiglipVisionTransformer, self).__init__()
        self.config = config
        embed_dim = self.config.hidden_size

        self.embedding_layer = SiglipVisionEmbedding(config)
        self.encoder = SiglipVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, images: torch.Tensor):
        """
        Input  : (Batch_size, C, H, W)
        Output : (Batch_size, num_patches, embedding_dim)
        """
        hidden_states = self.embedding_layer(images)
        last_hidden_state = self.encoder(hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state


class SiglipVisionModel(nn.Module):

    def __init__(self, config: SiglipVisionModelConfig):
        super(SiglipVisionModel, self).__init__()
        self.config = config
        self.vision_encoder = SiglipVisionTransformer(config)

    def forward(self, pixel_values: torch.Tensor):
        """
        Input: (Batch_size, C, H, W)
        Output: (Batch_size, num_patches, embedding_dim)
        """
        return self.vision_encoder(images = pixel_values)

