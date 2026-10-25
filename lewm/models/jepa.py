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

    