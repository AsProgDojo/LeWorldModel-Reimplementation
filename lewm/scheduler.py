import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

class LinearWarmupCosineAnnealingLR(LRScheduler):
    """
    Port of stable_pretraining.optim.lr_scheduler.LinearWarmupCosineAnnealingLR.
    
    Step it once per optimizer step (not per epoch).
    """

    def __init__(
            self,
            optimizer: Optimizer,
            warmup_steps: int,
            max_steps: int,
            warmup_start_lr: float = 0.0,
            eta_min: float = 0.0,
            last_epoch: int = -1,
    ):
        if max_steps <= warmup_steps:
            raise ValueError("max_steps must be greater than warmup_steps.")

        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch < self.warmup_steps:
            return [
                self.warmup_start_lr 
                + (base_lr - self.warmup_start_lr)
                * self.last_epoch
                / self.warmup_steps
                for base_lr in self.base_lrs
            ]

        progress = (
            (self.last_epoch - self.warmup_steps)
            / (self.max_steps - self.warmup_steps)
        )
        return [
            self.eta_min
            + (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) / 2
            for base_lr in self.base_lrs
        ]

def build_scheduler(
        optimizer: Optimizer,
        total_steps: int,
        warmup_fraction: float = 0.01,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
) -> LinearWarmupCosineAnnealingLR:
    """Default settings used by the original: warmup is 1% of all steps."""

    return LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_steps=max(1, int(warmup_fraction * total_steps)),
        max_steps = total_steps,
        warmup_start_lr=warmup_start_lr,
        eta_min = eta_min,
    )