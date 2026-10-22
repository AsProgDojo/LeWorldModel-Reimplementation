import torch
import torch.nn as nn

from lewm.models.layers import ConditionalTransformerBlock

class PredictorTransformer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            depth: int,
            heads: int,
            dim_head: int,
            mlp_dim: int,
            dropout: float  = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList(
            [
                ConditionalTransformerBlock(hidden_dim, heads, dim_head, mlp_dim, dropout)
                for _ in range(depth)
            ]
        )
        self.input_proj = (
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity
        )
        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity
        )
        self.output_proj = (
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        c = self.cond_proj(c)
        for block in self.layers:
            x = block(x, c)
        x = self.norm(x)
        return self.output_proj(x)

class ARPredictor(nn.Module):
    def __init__(
            self,
            *,
            num_frames: int,
            depth: int,
            heads: int,
            mlp_dim: int,
            input_dim: int,
            hidden_dim: int,
            output_dim: int | None = None,
            dim_head: int = 64,
            dropout: float = 0.0,
            emb_dropout: float = 0.0,
    ):
        super().__init__()
        self.pos_embedding - nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = PredictorTransformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x: [B, T, input_dim] frame embeddings, c: [B, T, input_dim] action embeddings
        T = x.size(1)
        x = x + self.pos_embeddings[:, :T]
        x = self.dropout(x)
        return self.transformer(x, c)