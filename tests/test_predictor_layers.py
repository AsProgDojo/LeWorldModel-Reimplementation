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

