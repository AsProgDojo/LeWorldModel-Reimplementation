import torch
import torch.nn as nn

class ActionEncoder(nn.Module):
    def __init__(self, input_dim: int, smoothed_dim: int, emb_dim: int, mlp_scale: int = 4):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim)
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, input_dim]
        x = x.float()
        x = x.permute(0, 2, 1)      # [B, input_dim, T]
        x = self.patch_embed(x)     # [B, smoothed_dim, T]
        x = x.permute(0, 2, 1)      # [B, T, smoothed_dim]
        return self.embed(x)        # [B, T, emb_dim]
