import pytest
import torch

from lewm.models.predictor import ARPredictor

def make_predictor(**overrides):
    kwargs = dict(
        num_frames=4, depth=2, heads=2, mlp_dim=32,
        input_dim=16, hidden_dim=16, dim_head=8, 
    )
    kwargs.update(overrides)
    return ARPredictor(**kwargs)

def randomize_gates(predictor):
    # AdaLN-Zero starts as identity; randomize so the block actually do something.
    for block in predictor.transformer.layer:
        torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.5)

def test_output_shape():
    out = make_predictor()(torch.randn(2, 3, 16), torch.randn(2, 3, 16))
    assert out.shape == (2, 3, 16)

def test_shorter_than_num_frames_works():
    out = make_predictor()(torch.randn(2, 1, 16), torch.randn(2, 1, 16))
    assert out.shape == (2, 1, 16)



