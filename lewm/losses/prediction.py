import torch

def prediction_loss(predicted_embeddings: torch.Tensor, target_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Computes the prediction loss between predicted embeddings and target embeddings.

    Args:
        predicted_embeddings (torch.Tensor): The predicted embeddings from the model.
        target_embeddings (torch.Tensor): The ground truth embeddings to compare against.

    Returns:
        torch.Tensor: The computed prediction loss.
    """

    # Compute the mean squared error loss
    loss = torch.square(predicted_embeddings - target_embeddings).mean()
    
    return loss