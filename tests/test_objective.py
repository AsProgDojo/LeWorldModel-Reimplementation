import torch
from lewm.losses.objective import lewm_objective
from lewm.losses.sigreg import SIGReg

def test_objective_returns_expected_losses():
    torch.manual_seed(42)

    sigreg = SIGReg(knots=17, num_projections=32)

    predicted = torch.randn(2, 3, 8)
    target = torch.randn(2, 3, 8)
    embeddings = torch.randn(2, 4, 8)

    output = lewm_objective(
        predicted_embeddings=predicted,
        target_embeddings=target,
        embeddings=embeddings,
        sigreg=sigreg,
        sigreg_weight=0.09
    )

    assert "pred_loss" in output
    assert "sigreg_loss" in output
    assert "loss" in output

def test_total_loss_is_weighted_sum():
    torch.manual_seed(42)

    sigreg = SIGReg(
        knots=17,
        num_projections=32,
    )

    predicted = torch.randn(2, 3, 8)
    target = torch.randn(2, 3, 8)
    embeddings = torch.randn(2, 4, 8)

    output = lewm_objective(
        predicted_embeddings=predicted,
        target_embeddings=target,
        embeddings=embeddings,
        sigreg=sigreg,
        sigreg_weight=0.09,
    )

    expected = output["pred_loss"] + 0.09 * output['sigreg_loss']

    assert torch.allclose(
        output["loss"],
        expected,
    )

def test_objective_backpropagates_through_prediction_and_targets():
    torch.manual_seed(42)

    sigreg = SIGReg(
        knots=17,
        num_projections=32
    )

    predicted = torch.randn(2, 3, 8, requires_grad=True)
    target = torch.randn(2, 3, 8, requires_grad=True)
    embeddings = torch.randn(2, 4, 8, requires_grad=True)

    output = lewm_objective(
        predicted_embeddings=predicted,
        target_embeddings=target,
        embeddings=embeddings,
        sigreg=sigreg,
        sigreg_weight=0.09
    )

    output['loss'].backward()

    assert predicted.grad is not None
    assert target.grad is not None
    assert embeddings.grad is not None

    assert torch.isfinite(predicted.grad).all()
    assert torch.isfinite(target.grad).all()
    assert torch.isfinite(embeddings.grad).all()