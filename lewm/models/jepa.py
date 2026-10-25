import torch.nn as nn

class JEPA(nn.Module):
    def __init__(
            self,
            encoder,
            predictor,
            action_encoder,
            projector=None,
            pred_proj=None,
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info):
        """Encode observations and actions into embeddings.\ninfo: dict with pixels (B, T, C, H, W) and optionally action (B, T, A)"""
        pixels = info["pixels"].float()
        b = pixels.size(0)
        pixels = pixels.flatten(0, 1)                           # (B*T, C, H, W)
        pixels_emb = self.encoder(pixels)[:, 0]                 # CLS token
        emb = self.projector(pixels_emb)                        # (B*T, D)
        info["emb"] = emb.reshape(b, -1, emb.size(-1))          # (B, T, D)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info