import torch

from lewm.losses.prediction import prediction_loss
from lewm.losses.sigreg import SIGReg

def lewm_objective(
        predicted_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        embeddings: torch.Tensor,
        sigreg: SIGReg,
        sigreg_weight: float,
) -> dict[str, torch.Tensor]:
    pred_loss = prediction_loss(
        predicted_embeddings,
        target_embeddings,
    )

    sigreg_loss = sigreg(embeddings.transpose(0, 1))

    total_loss = pred_loss + sigreg_weight * sigreg_loss

    return {
        "pred_loss": pred_loss,
        "sigreg_loss": sigreg_loss,
        "loss": total_loss,
    }