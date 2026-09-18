import torch
from lewm.models.layers import (
    ConditionalTransformerBlock,
    FeedForward,
    SelfAttention,
    modulate,
)

def test_modulate():
    x = torch.tensor([1.0, 2.0, 3.0])
    shift = torch.tensor([0.5, 0.5, 0.5])
    scale = torch.tensor([1.0, 0.0, -0.5])

    output = modulate(x, shift, scale)
    expected = x * (1 + scale) + shift

    assert torch.allclose(output, expected)

def test_feed_forward_preserves_shape():
    feed_forward = FeedForward(
        dim=16, 
        hidden_dim=64,
    )
    x = torch.randn(2, 3, 16)

    output = feed_forward(x)

    assert output.shape == x.shape

def test_attention_preserves_shape():
    attention = SelfAttention(
        dim=16,
        head=2,
        dim_head=8,
    )
    x = torch.randn(2, 4, 16)

    output = attention(x)

    assert output.shape == x.shape

