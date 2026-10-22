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

def test_longer_than_num_frames_works():
    with pytest.raises(RuntimeError):
        make_predictor()(torch.randn(2, 5, 16), torch.randn(2, 5, 16))

def test_fresh_predictor_is_positional_embedding_plus_final_norm():
    # Zero gates make very block an identity, so only pos-embed + final LayerNorm remain.
    predictor = make_predictor().eval()
    x, c = torch.randn(2, 3, 16), torch.randn(2, 3, 16)
    expected = predictor.transformer.norm(x + predictor.pos_embedding[:, :3])
    assert torch.allclose(predictor(x, c), expected, atol=1e-6)

def test_actions_influence_output_once_gates_are_nonzero():
    torch.manual_seed(0)
    predictor = make_predictor()
    randomize_gates(predictor)
    predictor.eval()
    x = torch.randn(2, 3, 16)
    out_a = predictor(x, torch.randn(2, 3, 16))
    out_b = predictor(x, torch.randn(2, 3, 16))
    assert not torch.allclose(out_a, out_b)

