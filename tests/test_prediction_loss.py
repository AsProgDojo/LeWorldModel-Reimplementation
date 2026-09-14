import torch
from lewm.losses.prediction import prediction_loss

def test_prediction_loss_is_zero_for_identical_embeddings():
    # Create identical predicted and target embeddings
    embeddings = torch.randn(2, 3, 192)

    # Compute the prediction loss
    loss = prediction_loss(embeddings, embeddings)

    # Assert that the loss is zero for identical embeddings
    assert torch.isclose(loss, torch.tensor(0.0)), f"Expected loss to be 0.0, but got {loss.item()}"

def test_prediction_loss_computes_mean_squared_error():
    predicted = torch.zeros(2, 3, 4)
    target = torch.ones(2, 3, 4)

    loss = prediction_loss(predicted, target)

    assert torch.isclose(loss, torch.tensor(1.0))

def test_prediction_loss_backpropagate_through_both_sides():
    predicted = torch.randn(2, 3, 4, requires_grad=True)
    target = torch.randn(2, 3, 4, requires_grad=True)

    loss = prediction_loss(predicted, target)
    loss.backward()

    assert predicted.grad is not None
    assert target.grad is not None

    assert torch.isfinite(predicted.grad).all()
    assert torch.isfinite(target.grad).all()