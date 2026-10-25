import torch
import torch.nn as nn

from lewm.losses.prediction import prediction_loss
from lewm.models.action_encoder import ActionEncoder
from lewm.models.encoder import VisionTransformer
from lewm.models.jepa import JEPA
from lewm.models.predictor import ARPredictor
from lewm.models.projector import MLP

D = 16


def make_parts():
    return dict(
        encoder=VisionTransformer(
            image_size=28, patch_size=14, hidden_size=D, depth=1, num_heads=2, mlp_dim=32
        ),
        predictor=ARPredictor(
            num_frames=3, depth=1, heads=2, mlp_dim=32, input_dim=D, hidden_dim=D, dim_head=8
        ),
        action_encoder=ActionEncoder(input_dim=4, smoothed_dim=4, emb_dim=D),
    )


def make_model():
    return JEPA(
        **make_parts(),
        projector=MLP(input_dim=D, hidden_dim=32, output_dim=D, norm_fn=nn.BatchNorm1d),
        pred_proj=MLP(input_dim=D, hidden_dim=32, output_dim=D, norm_fn=nn.BatchNorm1d),
    )


def randomize_gates(model):
    # AdaLN-Zero starts as identity (and the action path gets zero gradient);
    # randomize so the whole network is active.
    for block in model.predictor.transformer.layers:
        nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.5)


def make_batch(B=2, T=4):
    return {"pixels": torch.randn(B, T, 3, 28, 28), "action": torch.randn(B, T, 4)}

def test_encode_shapes():
    out = make_model().encode(make_batch())
    assert out["emb"].shape == (2, 4, D)
    assert out["act_emb"].shape == (2, 4, D)

def test_encode_without_action_has_no_act_emb():
    out = make_model().encode({"pixels": torch.randn(2, 4, 3, 28, 28)})
    assert "act_emb" not in out