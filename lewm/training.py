import torch
from lewm.losses.objective import lewm_objective
from lewm.losses.sigreg import SIGReg
from lewm.models.jepa import JEPA

def compute_losses(
        model: JEPA,
        sigreg: SIGReg,
        batch: dict[str, torch.Tensor],
        history_size: int,
        num_preds: int,
        sigreg_weight: float,
) -> dict[str, torch.Tensor]:
    """
    Port of lejepa_forward from the original train.py.

    Args:
        batch: "pixels" [B, T, C, H, W] and "action" [B, T, A],
               with T = history_size + num_preds. The dict is modified in place
               (action NaNs are zeroed, encode adds "emb" and "act_emb").

    Returns:
        {"pred_loss", "sigreg_loss", "loss"}

    """
    # NaN actions occur at sequence boundaries
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)

    output = model.encode(batch)
    embeddings = output['emb']                  # [B, T, D]
    action_embeddings = output['act_emb']       # [B, T, D]

    context_embeddings = embeddings[:, :history_size]
    context_actions = action_embeddings[:, :history_size]
    target_embeddings = embeddings[:, num_preds:]

    predicted_embeddings = model.predict(context_embeddings, context_actions)

    return lewm_objective(
        predicted_embeddings=predicted_embeddings,
        target_embeddings=target_embeddings,
        embeddings=embeddings,
        sigreg=sigreg,
        sigreg_weight=sigreg_weight,
    )

