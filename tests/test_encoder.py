import torch
from transformers import ViTConfig, ViTModel
from lewm.models.encoder import VisionTransformer, build_vit_tiny

def small_vit():
    return VisionTransformer(
        image_size=56, patch_size=14, hidden_size=32, depth=2, num_heads=4, mlp_dim=128
    )

def test_output_shape():
    out = small_vit()(torch.randn(2, 3, 56, 56))
    assert out.shape == (2, 1 + 16, 32)

def test_full_size_token_count():
    vit = build_vit_tiny().eval()
    with torch.no_grad():
        out = vit(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 257, 192)

def test_patch_embedding_is_shared_linear_on_flattened_patches():
    vit = small_vit()
    pe = vit.patch_embedding
    x = torch.randn(2, 3, 56, 56)
    P = 14
    patches = x.unfold(2, P, P).unfold(3, P, P)           # [B, C, H/P, W/P, P, P]
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(2, 16, 3 * P * P)
    weight = pe.projection.weight.reshape(32, -1)         # [D, C*P*P]
    expected = patches @ weight.T + pe.projection.bias
    assert torch.allclose(pe(x), expected, atol=1e-5)

def test_attention_is_not_causal():
    vit = small_vit().eval()
    x = torch.randn(1, 3, 56, 56)
    x2 = x.clone()
    x2[:, :, 42:, 42:] = torch.randn(3, 14, 14)  # change only the last patch
    out, out2 = vit(x), vit(x2)
    assert not torch.allclose(out[:, 1], out2[:, 1])  # earlier patch tokens do change
