import torch
import torch.nn.functional as F

from lewm.models.action_encoder import ActionEncoder

def make_encoder():
    return ActionEncoder(input_dim=10, smoothed_dim=6, emb_dim=16, mlp_scale=4)

def test_output_shape():
    out = make_encoder()(torch.randn(3, 5, 10))
    assert out.shape == (3, 5, 16)

