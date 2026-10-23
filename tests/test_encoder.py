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
