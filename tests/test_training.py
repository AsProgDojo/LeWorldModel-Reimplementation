import torch
from lewm.losses.sigreg import SIGReg
from lewm.training import compute_losses
from test_jepa import D, make_batch, make_model, randomize_gates

def run(
        model,
        batch=None, 
        history_size=3, 
        num_preds=1,
        sigreg_weight = 0.09,
):
    sigreg = SIGReg(knots=17, num_projections=32)
    batch = batch or make_batch(B=2, T=history_size + num_preds)
    return compute_losses(model, sigreg, batch, history_size, num_preds, sigreg_weight)



def test_returns_three_finite_scalar_losses():
    losses = run(make_model().train())

    assert set(losses) == {'pred_loss', 'sigreg_loss', 'loss'}
    for value in losses.values():
        assert value.ndim == 0
        assert torch.isfinite(value)

def test_total_is_weighted_sum():
    losses = run(make_model().train(), sigreg_weight=0.5)
    expected = losses['pred_loss'] + 0.5 * losses['sigreg_loss']
    assert torch.allclose(losses['loss'], expected)

def test_nan_actions_are_zeroed():
    batch = make_batch(B=2, T=4)
    batch['action'][0, 2] = float('nan')

    losses = run(make_model().train(), batch=batch)

    assert torch.isfinite(losses['loss'])
    assert torch.isfinite(batch['action']).all()

def test_all_parameters_get_gradients():
    model = make_model().train()
    randomize_gates(model)

    run(model)['loss'].backward()

    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, missing

def test_last_frame_gets_grFadient_through_target_path():
    # With history 3 and 1 pred, frame 3 is only ever a target.
    # Without SIGReg the only way it can get gradient is the non-detached target.
    model = make_model().train()
    randomize_gates(model)
    batch = make_batch(B=2, T=4)

    losses = run(model, batch=batch, sigreg_weight=0.0)
    batch['emb'].retain_grad()
    losses['pred_loss'].backward()

    assert batch['emb'].grad[:, 3].abs().sum() > 0


def test_longer_horizon():
    losses = run(make_model().train(), history_size=2, num_preds=1)
    assert torch.isfinite(losses['loss'])