import contextlib
import torch

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lewm.losses.sigreg import SIGReg
from lewm.models.jepa import JEPA
from lewm.training import compute_losses

def build_optimizer(
        model: JEPA,
        learning_rate: float,
        weight_decay: float,
) -> torch.optim.AdamW:
    """Plain AdamW over all model parameters (no weight-decay exclusions)."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

def autocast_context(precision: str, device_type: str):
    """
    "bf16": bfloat16 autocast (Lightning's "bf16-mixed"): weights stay fp32,
            only the forward pass and loss run in bf16, no GradScaler.
    "32":   no autocast.
    """
    if precision == "bf16":
        return torch.autocast(device_type, dtype=torch.bfloat16)
    if precision == "32":
        return contextlib.nullcontext()
    raise NotImplementedError(f"precision {precision!r} is not supported, use 'bf16' or '32'.")

def train_step(
        model: JEPA,
        sigreg: SIGReg,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        batch: dict[str, torch.Tensor],
        history_size: int,
        num_preds: int,
        sigreg_weight: float,
        gradient_clip_val: float,
        precision: str = "bf16"
) -> dict[str, torch.Tensor]:
    """One optimization step. Returns the detached losses."""

    device_type = next(model.parameters()).device.type

    with autocast_context(precision, device_type):
        losses = compute_losses(
            model, sigreg, batch, history_size, num_preds, sigreg_weight
        )

    losses['loss'].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)

    return {name: value.detach() for name, value in losses.items()}