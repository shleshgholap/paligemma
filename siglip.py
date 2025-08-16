import torch
import torch.nn as nn
from einops import rearrange
from typing import Tuple, Optional

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



class SiglipVisionEmbedding(nn.Module):

    def __init__(self, config: SiglipVisionModelConfig):
        super(SiglipVisionEmbedding, self).__init__()

        self.config = config
        
        self.image_size = config.image_size
        self.num_channels = config.num_channels
        self.embed_dim = config.hidden_size

        self.patch_size = config.patch_size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_position_embeddings = self.num_patches
        self.postional_embedding = nn.Embedding(self.num_positions, self.embed_dim)

        self.path_embedding = nn.Conv2d(
            in_channels = self.num_channels,
            out_channels = self.embed_dim,
            kernel_size = self.patch_size,
            stride = self.patch_size,
            padding = "valid",
        )

        self.register_buffer(
            "positional_ids", 
            torch.arange(self.num_position_embeddings).expand(1, -1),
            persistent=False
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Input  : (Batch_size, C, H, W)
        Output : (Batch_size, num_patches, embedding_dim)
        """
        B, C, H, W = images.shape

        # Extract patches
        # Output size : (Batch_size, embed_dim, num_patches ** 1/2, num_patches ** 1/2)
        patch_embeds = self.path_embedding(images) 

        # Traditional Way to rearrange would be:
        # embeddings = patch_embeds.flatten(2).transpose(1, 2)
        embeddings = rearrange(patch_embeds, 'b embed_dim patches_h patches_w -> b (patches_h patches_w) embed_dim')

        # Add positonal embeddings
        # Output size : (Batch_size, num_patches, embed_dim)
        embeddings = embeddings + self.postional_embedding(self.positional_ids)

        return embeddings
    
class SiglipMLP(nn.Module):

    def __init__(self, config: SiglipVisionModelConfig):
        super(SiglipMLP, self).__init__()

        self.hidden_size = config.hidden_size # same as embedding_dim
        self.intermediate_size = config.intermediate_size

        self.fc1 = nn.Linear(self.hidden_size, self.intermediate_size)
        self.fc2 = nn.Linear(self.intermediate_size, self.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        # [Batch_size, num_patches, embedding_dim] -> [Batch_size, num_patches, intermediate_size]
        hidden_states = self.fc1(hidden_states)
        # [Batch_size, num_patches, embedding_dim] -> [Batch_size, num_patches, embedding_dim]
        hidden_states = nn.functional.gelu(hidden_states, approximate='tanh')
        # [Batch_size, num_patches, intermediate_size] -> [Batch_size, num_patches, embedding_dim]
        hidden_states = self.fc2(hidden_states)

        return hidden_states # [Batch_size, num_patches, embedding_dim]


class SiglipAttention(nn.Module):

    def __init__(self, config:SiglipVisionModelConfig):
        super(SiglipAttention, self).__init__()

        self.config = config

        self.embedding_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.patch_size = config.patch_size
        self.dropout = config.attention_dropout

        self.head_dim = self.embedding_dim // self.num_attention_heads

        self.k_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.q_proj = nn.Linear(self.embedding_dim, self.embedding_dim) 
        self.v_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.out_proj = nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        """
        Input: [Batch_size, num_patches, embedding_dim]
        Output: [Batch_size, num_patches, embedding_dim]
        """
    
        batch_size, seq_len, d_model = hidden_states.shape # d_model is same as embedding_dim

        K = self.k_proj(hidden_states)  # [Batch_size, seq_len, embedding_dim]
        Q = self.q_proj(hidden_states)  # [Batch_size, seq_len, embedding_dim]  
        V = self.v_proj(hidden_states)  # [Batch_size, seq_len, embedding_dim]

        # Traditional Way to rearrange would be:
        # K = K.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        K = rearrange(K, 'b s (h d) -> (b h s d)', h=self.num_attention_heads, d=self.head_dim)  # [Batch_size, num_heads, seq_len, head_dim]
        Q = rearrange(Q, 'b s (h d) -> (b h s d)', h=self.num_attention_heads, d=self.head_dim)  # [Batch_size, num_heads, seq_len, head_dim]
        V = rearrange(V, 'b s (h d) -> (b h s d)', h=self.num_attention_heads, d=self.head_dim)  # [Batch_size, num_heads, seq_len, head_dim]

        attention_output = self.MultiHeadAttention(K, Q, V) # [Batch_size, num_heads, seq_len, head_dim]

        # [Batch_size, num_heads, seq_len, head_dim] -> [Batch_size, seq_len, embedding_dim]
        # Traditional Way to rearrange would be:
        # attention_output = attention_output.transpose(1, 2).reshape(batch_size, seq_len, self.embedding_dim)
        attention_output = rearrange(attention_output, 'b h s d -> b s (h d)', h=self.num_attention_heads, d=self.head_dim)  
        attention_output = self.out_proj(attention_output)

        return attention_output
    
    def MultiHeadAttention(self, K, Q, V):
        """
        Input: K, Q, V of shape [Batch_size, num_heads, seq_len, head_dim]
        Output: Attention output of shape [Batch_size, num_heads, seq_len, head_dim]
        """

        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)  # [Batch_size, num_heads, seq_len, seq_len]
        attention_weights = nn.functional.softmax(attention_scores, dim=-1).to(Q.dtype)  # [Batch_size, num_heads, seq_len, seq_len]
        attention_weights = nn.functional.dropout(attention_weights, p=self.dropout, training=self.training)
        attention_output = torch.matmul(attention_weights, V) # [Batch_size, num_heads, seq_len, head_dim]

        return attention_output, attention_weights 
 

class SiglipVisionEncoderLayer(nn.Module):

    def __init__(self, config: SiglipVisionModelConfig):
        super(SiglipVisionEncoderLayer, self).__init__()

        self.embed_dim = config.hidden_size
        self.attention = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Input : (Batch_size, num_patches, embedding_dim)
        Output : (Batch_size, num_patches, embedding_dim)
        """

        # residual: [Batch_size, num_patches, embedding_dim]
        residual = hidden_states
        # [Batch_size, num_patches, embedding_dim] -> [Batch_size, num_patches, embedding_dim]
        hidden_states = self.layer_norm1(hidden_states)
        # [Batch_size, num_patches, embedding_dim] -> [Batch_size, num_patches, embedding_dim]
        hidden_states, _ = self.attention(hidden_states)

        # hidden_states: [Batch_size, num_patches, embedding_dim]
        hidden_states = residual + hidden_states

        # residual: [Batch_size, num_patches, embedding_dim]
        residual = hidden_states
        # [Batch_size, num_patches, embedding_dim] -> [Batch_size, num_patches, embedding_dim]
        hidden_states = self.layer_norm2(hidden_states)
        # [Batch_size, num_patches, embedding_dim] -> [Batch_size, num_patches, embedding_dim]
        hidden_states = self.mlp(hidden_states)

        # hidden_states: [Batch_size, num_patches, embedding_dim]
        hidden_states = residual + hidden_states

        return hidden_states 


class SiglipVisionEncoder(nn.Module):

    def __init__(self, config: SiglipVisionModelConfig):
        super(SiglipVisionEncoder, self).__init__()

        self.num_hidden_layers = config.num_hidden_layers
        self.encoder_layers = nn.ModuleList(
            [SiglipVisionEncoderLayer(config) for _ in range(self.num_hidden_layers)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Input : (Batch_size, num_patches, embedding_dim)
        Output : (Batch_size, num_patches, embedding_dim)
        """

        for layer in self.encoder_layers:
            # hidden_states: [Batch_size, num_patches, embedding_dim]
            hidden_states = layer(hidden_states)

        return hidden_states  # [Batch_size, num_patches, embedding_dim] 



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

