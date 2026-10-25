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
        nn.init.normal_(block.condition_modulation[-1].weight, std=0.5)


def make_batch(B=2, T=4):
    return {"pixels": torch.randn(B, T, 3, 28, 28), "action": torch.randn(B, T, 4)}

def test_encode_shapes():
    out = make_model().encode(make_batch())
    assert out["emb"].shape == (2, 4, D)
    assert out["act_emb"].shape == (2, 4, D)

def test_encode_without_action_has_no_act_emb():
    out = make_model().encode({"pixels": torch.randn(2, 4, 3, 28, 28)})
    assert "act_emb" not in out

def test_encode_adds_keys_to_the_same_dict_and_returns_it():
    # The training step relies on this: encode(batch) fills the batch dict.
    batch = make_batch()
    out = make_model().encode(batch)
    assert out is batch
    assert "emb" in batch and "act_emb" in batch

def test_encode_takes_cls_then_projector():
    model = make_model().eval()
    batch = make_batch()
    pixels = batch["pixels"].clone()
    with torch.no_grad():
        emb = model.encode(batch)["emb"]
        expected = model.projector(model.encoder(pixels.flatten(0, 1))[:, 0]).reshape(2, 4, -1)
    assert torch.allclose(emb, expected, atol=1e-6)

def test_frames_are_encoded_independently_in_eval():
    model = make_model().eval()
    batch = make_batch()
    single_pixels = batch["pixels"][:, 2:3].clone()
    with torch.no_grad():
        full = model.encode(batch)["emb"]
        single = model.encode({"pixels": single_pixels})["emb"]
    assert torch.allclose(full[:, 2:3], single, atol=1e-5)

def test_projector_batchnorm_sees_flattened_frames():
    model = make_model().train()
    seen = []
    model.projector.net[1].register_forward_hook(lambda m, i, o: seen.append(i[0].shape))
    model.encode(make_batch(B=2, T=4))
    assert seen == [(8, 32)]  # B*T rows, hidden features

def test_projector_and_pred_proj_default_to_identity():
    model = JEPA(**make_parts())
    assert isinstance(model.projector, nn.Identity)
    assert isinstance(model.pred_proj, nn.Identity)

def test_predictions_are_causal_end_to_end():
    torch.manual_seed(0)
    model = make_model()
    randomize_gates(model)
    model.eval()
    emb, act = torch.randn(1, 3, D), torch.randn(1, 3, D)
    emb2, act2 = emb.clone(), act.clone()
    emb2[:, 2] = torch.randn(D) * 10
    act2[:, 2] = torch.randn(D) * 10
    with torch.no_grad():
        p, p2 = model.predict(emb, act), model.predict(emb2, act2)
    assert torch.allclose(p[:, :2], p2[:, :2], atol=1e-5)
    assert not torch.allclose(p[:, 2], p2[:, 2])

def test_training_step_alignment_and_gradient_flow():
    torch.manual_seed(0)
    model = make_model().train()
    randomize_gates(model)
    history_size, num_preds = 3, 1

    out = model.encode(make_batch(B=2, T=history_size + num_preds))
    emb, act_emb = out["emb"], out["act_emb"]
    emb.retain_grad()

    ctx_emb = emb[:, :history_size]
    ctx_act = act_emb[:, :history_size]
    tgt_emb = emb[:, num_preds:]
    pred_emb = model.predict(ctx_emb, ctx_act)
    assert pred_emb.shape == tgt_emb.shape == (2, 3, D)

    prediction_loss(pred_emb, tgt_emb).backward()

    # The last frame is a target only (never context): its gradient proves
    # the target path is NOT detached.
    assert emb.grad[:, history_size].abs().sum() > 0

    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, missing