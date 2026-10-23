import math
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_channels: int, hidden_size: int):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        # pixels: [B, C, H, W]
        x = self.projection(pixels)             # [B, D, H/P, W/P]
        return x.flatten(2).transpose(1, 2)     # [B, N, D] patchesin row-major order

