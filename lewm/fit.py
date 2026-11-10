from collections.abc import Iterable
from pathlib import Path

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lewm.checkpoint import (
    load_training_state,
    resume_path,
    save_config,
    save_training_state,
    save_weights,
)
from lewm.engine import train_epoch, validate
from lewm.losses.sigreg import SIGReg
from lewm.models.jepa import JEPA

def fit(
        model: JEPA,
        sigreg: SIGReg,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        train_loader: Iterable[dict[str, torch.Tensor]],
        val_loader: Iterable[dict[str, torch.Tensor]],
        *,
        num_epochs: int,
        history_size: int,
        num_preds: int,
        sigreg_weight: float,
        gradient_clip_val: float,
        precision: str,
        device: torch.device | str,
        run_dir: str | Path,
        run_name: str,
        config: dict,
        checkpoint_interval: int = 1,
        logger=None,
        sanity_val_batches: int = 1,
) -> list[dict[str, float]]:
    """
    Full training run. If <run_dir>/<run_name>_weights.ckpt exists, training
    resumes from it. Returns the validation means of the epochs trained in
    this call.
    """
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive.")

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    model.to(device)
    sigreg.to(device)
    save_config(config, run_dir)

    checkpoint = resume_path(run_dir, run_name)
    if checkpoint.exists():
        start_epoch, global_step = load_training_state(
            checkpoint, model, optimizer, scheduler
        )
    else:
        start_epoch, global_step = 0, 0
        if sanity_val_batches > 0:
            # like num_sanity_val_steps=1: catch a broken validation early
            validate(
                model, sigreg, val_loader, history_size, num_preds,
                sigreg_weight, precision, device, max_batches=sanity_val_batches,
            )


    def log_step(losses: dict[str, float], learning_rate: float) -> None:
        nonlocal global_step
        global_step += 1
        if logger is not None:
            metrics = {f"train/{name}": value for name, value in losses.items()}
            metrics["hparams/lr_model_opt"] = learning_rate
            logger.log(metrics, global_step)

    validation_history = []
    for epoch in range(start_epoch, num_epochs):
        train_epoch(
            model, sigreg, optimizer, scheduler, train_loader,
            history_size, num_preds, sigreg_weight, gradient_clip_val,
            precision, device, on_step=log_step,
        )

        validation = validate(
            model, sigreg, val_loader, history_size, num_preds,
            sigreg_weight, precision, device,
        )
        validation_history.append(validation)
        if logger is not None:
            logger.log(
                {f"val/{name}": value for name, value in validation.items()},
                global_step,
            )

        completed = epoch + 1
        if completed % checkpoint_interval == 0 or completed == num_epochs:
            save_weights(model, run_dir, completed)
        save_training_state(
            checkpoint, model, optimizer, scheduler, completed, global_step
        )

    return validation_history


