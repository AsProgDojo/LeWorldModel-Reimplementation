import math
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_channels: int, hidden_size: int):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        # pixels: [B, C, H, W]
        x = self.projection(pixels)             # [B, D, H/P, W/P]
        return x.flatten(2).transpose(1, 2)     # [B, N, D] patchesin row-major order


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        # [B, N, D] -> [B, H, N, dh]
        q = self.query(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)  # [B, H, N, N]
        weights = self.attn_dropout(scores.softmax(dim=-1))
        out = weights @ v                                            # [B, H, N, dh]

        out = out.transpose(1, 2).reshape(B, N, D)                   # merge heads
        return self.out_dropout(self.out(out))

    
class Transformer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attn = MultiHeadSelfAttention(hidden_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)