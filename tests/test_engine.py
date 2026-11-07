import pytest
import torch
from lewm.engine import train_epoch, validate
from lewm.losses.sigreg import SIGReg
from lewm.training import compute_losses
from test_jepa import make_batch, make_model, randomize_gates
from test_optimization import make_setup

HISTORY, PREDS = 3, 1
KEYS = {'pred_loss', 'sigreg_loss', 'loss'}

def batches(sizes=(2, 2, 2)):
    return [make_batch(B=b, T=HISTORY + PREDS) for b in sizes]

def run_train_epoch(setup, loader):
    model, sigreg, optimizer, scheduler = setup
    return train_epoch(
        model, sigreg, optimizer, scheduler, loader,
        history_size=HISTORY, num_preds=PREDS, sigreg_weight=0.09,
        gradient_clip_val=1.0, precision='32',
    )

def run_validate(model, sigreg, loader, **kwargs):
    return validate(
        model, sigreg, loader,
        history_size=HISTORY, num_preds=PREDS,
        sigreg_weight=0.09, precision='32', **kwargs,
    )

def make_eval_parts():
    torch.manual_seed(0)
    model = make_model()
    randomize_gates(model)
    return model, SIGReg(knots=17, num_projections=32)

def test_train_epoch_returns_one_loss_dict_per_batch():
    setup = make_setup()
    step_losses = run_train_epoch(setup, batches((2, 2, 2)))

    assert len(step_losses) == 3
    for losses in step_losses:
        assert set(losses) == KEYS
        assert all(isinstance(v, float) for v in losses.values())
        assert all(torch.isfinite(torch.tensor(v)) for v in losses.values())

def test_train_epoch_steps_optimizer_and_scheduler_once_per_batch():
    setup = make_setup()
    run_train_epoch(setup, batches((2, 2, 2, 2)))

    assert setup[3].last_epoch == 4

def test_train_epoch_puts_model_in_train_mode():
    setup = make_setup()
    setup[0].eval()

    run_train_epoch(setup, batches((2,)))

    assert setup[0].training

def test_validate_returns_mean_losses():
    model, sigreg = make_eval_parts()
    result = run_validate(model, sigreg, batches((2, 4)))

    assert set(result) == KEYS
    assert all(torch.isfinite(torch.tensor(v)) for v in result.values())

def test_validate_does_not_change_weights_buffers_or_gradients():
    model, sigreg = make_eval_parts()
    model.train()
    params_before = [p.detach().clone() for p in model.parameters()]
    buffers_before = [b.detach().clone() for b in model.buffers()]

    run_validate(model, sigreg, batches((2, 2)))

    assert all(torch.equal(a, p) for a, p in zip(params_before, model.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(buffers_before, model.buffers()))
    assert all(p.grad is None for p in model.parameters())

@pytest.mark.parametrize("was_training", [True, False])
def test_validate_runs_in_eval_mode_and_restores_mode(was_training):
    model, sigreg = make_eval_parts()
    model.train(was_training)
    seen = []
    model.projector.register_forward_pre_hook(
        lambda module, args: seen.append(module.training)
    )

    run_validate(model, sigreg, batches((2,)))

    assert seen == [False]
    assert model.training == was_training

def test_validate_mean_is_weighted_by_batch_size():
    model, sigreg = make_eval_parts()
    loader = batches((2, 4))

    expected_total, expected_count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            copy = {k: v.clone() for k, v in batch.items()}
            size = copy["pixels"].size(0)
            losses = compute_losses(model, sigreg, copy, HISTORY, PREDS, 0.09)
            expected_total += losses["pred_loss"].item() * size
            expected_count += size

    result = run_validate(model, sigreg, loader)

    assert result["pred_loss"] == pytest.approx(expected_total / expected_count, rel=1e-5)

def test_validate_max_batches_limits_the_number_of_batches():
    model, sigreg = make_eval_parts()
    calls = []
    model.projector.register_forward_pre_hook(lambda module, args: calls.append(1))

    run_validate(model, sigreg, batches((2, 2, 2)), max_batches=1)

    assert len(calls) == 1

def test_validate_rejects_empty_loader():
    model, sigreg = make_eval_parts()
    with pytest.raises(ValueError):
        run_validate(model, sigreg, [])
