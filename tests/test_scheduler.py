import pytest
import torch

from lewm.scheduler import LinearWarmupCosineAnnealingLR, build_scheduler

BASE_LR = 1.0

def make(warmup_steps=10, max_steps=110, **kwargs):
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=BASE_LR)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer, warmup_steps=warmup_steps, max_steps=max_steps, **kwargs
    )
    return optimizer, scheduler

def advance(optimizer, scheduler, steps):
    for _ in range(steps):
        optimizer.step()
        scheduler.step()

def lr(optimizer):
    return optimizer.param_groups[0]['lr']

def test_starts_at_warmup_start_lr():
    optimizer, _ = make(warmup_start_lr=0.0)
    assert lr(optimizer) == 0.0

def test_warmup_is_linear():
    optimizer, scheduler = make()
    advance(optimizer, scheduler, 5)
    assert lr(optimizer) == pytest.approx(0.5 * BASE_LR)

def test_reaches_base_lr_at_end_of_warmup():
    optimizer, scheduler = make()
    advance(optimizer, scheduler, 10)
    assert lr(optimizer) == pytest.approx(BASE_LR)

def test_cosine_midpoint_is_half_of_base_lr():
    optimizer, scheduler = make(warmup_steps=10, max_steps=110)
    advance(optimizer, scheduler, 60)
    assert lr(optimizer) == pytest.approx(0.5 * BASE_LR)

def test_ends_at_eta_min():
    optimizer, scheduler = make(eta_min=0.1)
    advance(optimizer, scheduler, 110)
    assert lr(optimizer) == pytest.approx(0.1)

def test_rises_during_warmup_then_never_increases():
    optimizer, scheduler = make()
    values = [lr(optimizer)]
    for _ in range(110):
        optimizer.step()
        scheduler.step()
        values.append(lr(optimizer))

    assert values[:11] == sorted(values[:11])
    assert values[10:] == sorted(values[10:], reverse=True)

def test_build_scheduler_uses_one_percent_warmup():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=BASE_LR)
    scheduler = build_scheduler(optimizer, total_steps=1000)

    assert scheduler.warmup_steps == 10
    assert scheduler.max_steps == 1000

def test_build_scheduler_warmup_is_at_least_one_step():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=BASE_LR)
    scheduler = build_scheduler(optimizer, total_steps=50)

    assert scheduler.warmup_steps == 1

def test_rejects_max_steps_not_greater_than_warmup():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=BASE_LR)

    with pytest.raises(ValueError):
        LinearWarmupCosineAnnealingLR(optimizer, warmup_steps=10, max_steps=10)