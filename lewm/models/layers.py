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

    def forward(
            self,
            x: torch.Tensor,
            causal: bool = True,
    ) -> torch.Tensor:
        
        batch_size, sequence_length, _ = x.shape
        x = self.norm(x)
        query, key, value = self.to_qkv(x).chunk(3, dim=-1)

        def split_heads(
                tensor: torch.Tensor
        ) -> torch.Tensor:
            
            tensor = tensor.reshape(
                batch_size,
                sequence_length,
                self.heads,
                self.dim_head
            )

            return tensor.transpose(1, 2)

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)

        dropout_probability = (
            self.dropout
            if self.training
            else 0.0
        )

        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=dropout_probability,
            is_causal=causal
        )

        output = (
            output
            .transpose(1,2)
            .contiguous()
            .reshape(
                batch_size,
                sequence_length,
                self.heads * self.dim_head,
            )
        )

        return self.to_output(output)

class ConditionalTransformerBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            heads: int,
            dim_head: int,
            mlp_dim: int,
            dropout: float = 0.0,
    ):
        super().__init__()

        self.attention = SelfAttention(
            dim=dim,
            heads=heads,
            dim_head = dim_head,
            dropout=dropout,
        )

        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=mlp_dim,
            dropout=dropout,
        )

        self.attention_norm = nn.LayerNorm(
            dim,
            elementwise_affine=False,
            eps=1e-6,
        )

        self.feed_forward_norm = nn.LayerNorm(
            dim,
            elementwise_affine=False,
            eps=1e-6,
        )

        self.condition_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                dim,
                6 * dim,
                bias=True
            ),
        )

        nn.init.zeros_(
            self.condition_modulation[-1].weight
        )
        nn.init.zeros_(
            self.condition_modulation[-1].bias
        )

    def forward(
            self,
            x: torch.Tensor,
            condition: torch.Tensor,
    ) -> torch.Tensor:
        (
            attention_shift,
            attention_scale,
            attention_gate,
            mlp_shift,
            mlp_scale,
            mlp_gate,
        ) = self.condition_modulation(condition).chunk(6, dim=-1)

        attention_input = modulate(
            self.attention_norm(x),
            attention_shift,
            attention_scale,
        )

        x = x + attention_gate * self.attention(attention_input)

        mlp_input = modulate(
            self.feed_forward_norm(x),
            mlp_shift,
            mlp_scale,
        )

        x = x + mlp_gate * self.feed_forward(mlp_input)

        return x