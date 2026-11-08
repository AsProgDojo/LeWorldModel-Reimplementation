from collections.abc import Callable, Iterable

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lewm.losses.sigreg import SIGReg
from lewm.models.jepa import JEPA
from lewm.optimization import autocast_context, train_step
from lewm.training import compute_losses

def move_to_device(
        batch: dict[str, torch.Tensor],
        device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for name, value in batch.items()
    }

def train_epoch(
        model: JEPA,
        sigreg: SIGReg,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        loader: Iterable[dict[str, torch.Tensor]],
        history_size: int,
        num_preds: int,
        sigreg_weight: float,
        gradient_clip_val: float,
        precision: str = "bf16",
        device: torch.device | str = "cpu",
        on_step: Callable[[dict[str, float], float], None] | None = None,
) -> list[dict[str, float]]:
    """
    One pass over the loader. Returns the losses of every step (the original
    logs per step, so nothing is averaged here).

    on_step(losses, learning_rate) is called after every step. The learning
    rate is read after the scheduler step, so it is the one the next step
    uses (the original logs it at the same point).
    """
    device = torch.device(device)
    model.train()

    step_losses = []
    for batch in loader:
        batch = move_to_device(batch, device)
        losses = train_step(
            model, sigreg, optimizer, scheduler, batch,
            history_size, num_preds, sigreg_weight,
            gradient_clip_val, precision,
        )
        values = {name: value.item() for name, value in losses.items()}
        step_losses.append(values)
        if on_step is not None:
            on_step(values, optimizer.param_groups[0]["lr"])

    return step_losses


def validate(
        model: JEPA,
        sigreg: SIGReg,
        loader: Iterable[dict[str, torch.Tensor]],
        history_size: int,
        num_preds: int,
        sigreg_weight: float,
        precision: str = "bf16",
        device: torch.device | str = "cpu",
        max_batches: int | None = None,
) -> dict[str, float]:
    """
    Same losses as training, but in eval mode (BatchNorm uses running statistics, dropout is off) and without gradients.
    
    Returns the mean of each loss, weighted by batch size.
    max_batches=1 gives the original's one-batch sanity check."""
    device = torch.device(device)
    was_training = model.training
    model.eval()

    totals: dict[str, float] = {}
    count = 0

    with torch.no_grad():
        for index, batch in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break

            batch = move_to_device(batch, device)
            batch_size = batch['pixels'].size(0)

            with autocast_context(precision, device.type):
                losses = compute_losses(
                    model, sigreg, batch, history_size, num_preds, sigreg_weight
                )

            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + value.item() * batch_size
            count += batch_size

        model.train(was_training)

        if count == 0:
            raise ValueError("validation loader produced no batches.")

        return {name: total / count for name, total in totals.items()}