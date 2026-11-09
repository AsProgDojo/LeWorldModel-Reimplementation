from pathlib import Path

import torch
import yaml

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lewm.models.jepa import JEPA

def save_weights(model: JEPA, run_dir: str | Path, epoch: int) -> Path:
    """Model weights only, like the original's weights_epoch_{N}.pt."""
    path = Path(run_dir) / f'weights_epoch_{epoch}.pt'
    torch.save(model.state_dict(), path)
    return path

def save_config(config: dict, run_dir: str | Path) -> Path:
    path = Path(run_dir) / 'config.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    return path

def resume_path(run_dir: str | Path, run_name: str) -> Path:
    return Path(run_dir) / f'{run_name}_weights.ckpt'

def save_training_state(
        path: str | Path,
        model: JEPA,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        epoch: int,
        global_step: int,
) -> None:
    """
    Everything needed to continue training. `epoch` is the number of
    completed epochs. The RNG state is an addition beyond the original
    (Lightning does not restore it); it makes resumed runs reproducible.
    """
    path = Path(path)
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }
    # write to a temporary file first, so an interrupted save cannot corrupt the checkpoint
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, temporary)
    temporary.replace(path)

def load_training_state(
        path: str | Path,
        model: JEPA,
        optimizer: Optimizer,
        scheduler: LRScheduler,
) -> tuple[int, int]:
    """Restores everything in place. Returns (completed_epochs, global_step)."""
    state = torch.load(path, map_location='cpu', weights_only=True)

    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])

    torch.set_rng_state(state['rng_cpu'])
    if state['rng_cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['rng_cuda'])

    return state['epoch'], state['global_step']



