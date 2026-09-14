import torch
from lewm.losses.sigreg import SIGReg

def test_sigreg_returns_scalar():
    sigreg = SIGReg(
        knots=17,
        num_projections=32,
    )

    embeddings = torch.randn(3, 16, 8)

    loss = sigreg(embeddings)
    assert loss.ndim == 0


def test_sigreg_is_finite():
    sigreg = SIGReg(
        knots=17,
        num_projections=32,
    )

    embeddings = torch.randn(3, 16, 8)

    loss = sigreg(embeddings)
    assert torch.isfinite(loss)

def test_sigreg_backpropagates_to_embeddings():
    sigreg = SIGReg(
        knots=17, num_projections=32,
    )

    embeddings = torch.randn(3, 16, 8, requires_grad=True)

    loss = sigreg(embeddings)
    loss.backward()

    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()

def test_sigreg_matches_reference_computation():
    knots=17
    num_projections=32

    sigreg = SIGReg(
        knots,
        num_projections,
    )

    embeddings = torch.randn(3, 16, 8,)

    torch.manual_seed(42)
    actual = sigreg(embeddings)
    torch.manual_seed(42)

    evaluation_points = torch.linspace(
        0.0, 3.0, knots, dtype=torch.float32,
    )

    step_size = 3.0 / (knots - 1)

    integration_weights = torch.full(
        (knots,),
        2.0 * step_size,
        dtype=torch.float32
    )
    integration_weights[[0, -1]] = step_size

    gaussian = torch.exp(-evaluation_points.square() / 2.0)
    integration_weights = (integration_weights * gaussian)

    directions = torch.randn(embeddings.shape[-1], num_projections)
    directions = directions / directions.norm(p=2, dim = 0)

    values = (
        (embeddings @ directions).unsqueeze(-1) * evaluation_points
    )
    error = (
        values.cos().mean(dim=-3) - gaussian
    ).square() + values.sin().mean(dim=-3).square()

    expected = (
        (error @ integration_weights)
        * embeddings.shape[-2]
    ).mean()

    assert torch.allclose(
        actual,
        expected,
        atol=1e-6,
    )