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

def test_attention_is_causal():
    torch.manual_seed(42)

    attention = SelfAttention(
        dim=16,
        heads=2,
        dim_head=8,
        dropout=0.0,
    )
    attention.eval()

    original = torch.randn(1, 4, 16)
    modified = original.clone()

    modified[:, 3] = torch.randn(1, 16) * 100

    original_output = attention(original)
    modified_output = attention(modified)

    assert torch.allclose(
        original_output[:, :3],
        modified_output[:, :3],
        atol=1e-6,
    )

def test_adaln_modulation_starts_at_zero():
    block = ConditionalTransformerBlock(
        dim=16,
        heads=2,
        dim_head=8,
        mlp_dim=64
    )

    linear = block.condition_modulation[-1]

    assert torch.count_nonzero(linear.weight) == 0
    assert torch.count_nonzero(linear.bias) == 0

def test_conditional_block_is_identity_at_initialization():
    torch.manual_seed(42)

    block = ConditionalTransformerBlock(
        dim=16,
        heads=2,
        dim_head=8,
        mlp_dim=64,
        dropout=0.0
    )

    x = torch.randn(2, 3, 16)
    condition = torch.randn(2, 3, 16)

    output = block(x, condition)

    assert torch.allclose(output, x, atol=1e-7)