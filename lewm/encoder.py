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

    
class TransformerBlock(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.dropout(self.fc2(self.act(self.fc1(self.norm2(x)))))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        in_channels: int = 3,
        hidden_size: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_dim: int = 768,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, hidden_size)
        num_patches = self.patch_embedding.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.position_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_size))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, num_heads, mlp_dim, dropout, layer_norm_eps)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        # pixels: [B, C, H, W] -> [B, 1 + N, D]; token 0 is CLS
        x = self.patch_embedding(pixels)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.position_embedding
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)