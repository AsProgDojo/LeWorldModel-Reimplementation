import torch
import torch.nn.functional as F
from torch import nn

def modulate(
        x: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
) -> torch.Tensor:
    return x * (1 + scale) + shift


