import torch
import torch.nn.functional as F
from torch import nn

def modulate(
        x: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
) -> torch.Tensor:
    return x * (1 + scale) + shift


class FeedForward(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            dropout: float = 0.0,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            heads: int = 8,
            dim_head: int = 64,
            dropout: float = 0.0,
    ):
        super().__init__()

        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout

        inner_dim = heads * dim_head

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(
            dim,
            inner_dim * 3,
            bias = False
        )

        if heads == 1 and dim_head == dim:
            self.to_out = nn.Identity()
        else:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout),
            )
