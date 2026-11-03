import numpy as np
import torch
import torch.nn.functional as F

# Source: stable_pretraining.data.dataset_stats.ImageNet
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def preprocess_pixels(
        pixels: torch.Tensor,
        image_size: int = 224,
) -> torch.Tensor:
    """
    Args:
        pixels: uint8 tensor of shape [..., H, W, 3]

    Returns:
        float32 tensor of shape [..., 3, image_size, image_size]
    """
    pixels = torch.as_tensor(pixels)
    if pixels.dtype != torch.uint8:
        raise ValueError(f"pixels must be uint8, got {pixels.dtype}")

    leading_shape = pixels.shape[:-3]
    frames = pixels.reshape(-1, *pixels.shape[-3:])          # [N, H, W, 3]
    frames = frames.permute(0, 3, 1, 2).float() / 255.0       # [N, 3, H, W]

    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    frames = (frames - mean) / std

    frames = F.interpolate(
        frames,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )

    return frames.reshape(*leading_shape, 3, image_size, image_size)


class ZScoreNormalizer:
    """Pickable z-score normalizer (a class, not a closure, so DataLoader workers can pickle it)."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std

    def __call__(self, x:torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / self.std).float()


def build_zscore_normalizer(data) -> ZScoreNormalizer:
    """
    Args:
        data: array-like of shape [N, D] (a whole dataset column)
    """
    data = torch.from_numpy(np.array(data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return ZScoreNormalizer(mean, std)