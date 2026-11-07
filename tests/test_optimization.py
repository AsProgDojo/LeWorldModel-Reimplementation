import pytest
import torch

from lewm.losses.sigreg import SIGReg
from lewm.optimization import autocast_context, build_optimizer, train_step
from lewm.scheduler import LinearWarmupCosineAnnealingLR
from test_jepa import make_batch, make_model, randomize_gates

HISTORY, PREDS = 3, 1


def make_setup(optimizer_name="adamw", lr=1e-3, precision="32"):
    torch.manual_seed(0)
    model = make_model().train()
    randomize_gates(model)
    sigreg = SIGReg(knots=17, num_projections=32)
    if optimizer_name == "adamw":
        optimizer = build_optimizer(model, learning_rate=lr, weight_decay=1e-3)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # constant-ish start (warmup_start_lr == lr) so the very first step is not lr=0
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer, warmup_steps=1, max_steps=1000, warmup_start_lr=lr
    )
    return model, sigreg, optimizer, scheduler

HISTORY, PREDS = 3, 1


def step(setup, batch, clip=1.0, precision="32"):
    model, sigreg, optimizer, scheduler = setup
    return train_step(
        model, sigreg, optimizer, scheduler, batch,
        history_size=HISTORY, num_preds=PREDS, sigreg_weight=0.09,
        gradient_clip_val=clip, precision=precision,
    )

def test_build_optimizer_is_adamw_with_given_settings():
    model = make_model()
    optimizer = build_optimizer(model, learning_rate=5e-5, weight_decay=1e-3)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == 5e-5
    assert optimizer.param_groups[0]["weight_decay"] == 1e-3
    assert len(optimizer.param_groups[0]["params"]) == len(list(model.parameters()))

def test_train_step_returns_detached_losses():
    setup = make_setup()
    losses = step(setup, make_batch(B=2, T=4))

    assert set(losses) == {"pred_loss", "sigreg_loss", "loss"}
    for value in losses.values():
        assert not value.requires_grad
        assert torch.isfinite(value)

def test_train_step_updates_parameters_and_clears_gradients():
    setup = make_setup()
    model = setup[0]
    before = [p.detach().clone() for p in model.parameters()]

    step(setup, make_batch(B=2, T=4))

    changed = [not torch.equal(b, p) for b, p in zip(before, model.parameters())]
    assert any(changed)
    assert all(p.grad is None for p in model.parameters())

def test_loss_decreases_on_a_fixed_batch():
    setup = make_setup(lr=1e-2)
    batch = make_batch(B=4, T=4)
    first = step(setup, {k: v.clone() for k, v in batch.items()})["loss"]
    for _ in range(40):
        last = step(setup, {k: v.clone() for k, v in batch.items()})["loss"]

    assert last < first

def test_gradient_norm_is_clipped():
    setup = make_setup(optimizer_name="sgd", lr=1.0)
    model = setup[0]
    before = [p.detach().clone() for p in model.parameters()]

    step(setup, make_batch(B=2, T=4), clip=1e-3)

    delta = torch.sqrt(
        sum(((p.detach() - b) ** 2).sum() for b, p in zip(before, model.parameters()))
    )
    # SGD with lr=1: the update equals the clipped gradient, whose norm is the clip value
    assert delta.item() == pytest.approx(1e-3, rel=1e-2)

def test_scheduler_is_stepped_once_per_train_step():
    setup = make_setup()
    scheduler = setup[3]

    for _ in range(3):
        step(setup, make_batch(B=2, T=4))

    assert scheduler.last_epoch == 3

def test_autocast_context_selection():
    with autocast_context("32", "cpu"):
        assert not torch.is_autocast_enabled("cpu")
    with autocast_context("bf16", "cpu"):
        assert torch.is_autocast_enabled("cpu")
    with pytest.raises(NotImplementedError):
        autocast_context("16", "cpu")

def test_bf16_train_step_keeps_fp32_weights_and_finite_loss():
    setup = make_setup(precision="bf16")
    losses = step(setup, make_batch(B=2, T=4), precision="bf16")

    assert torch.isfinite(losses["loss"])
    assert all(p.dtype == torch.float32 for p in setup[0].parameters())
