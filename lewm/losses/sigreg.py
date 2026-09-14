import torch
from torch import nn

class SIGReg(nn.Module):
    def __init__(
            self,
            knots: int = 17,
            num_projections: int = 1024,
    ):
        super().__init__()

        self.num_projections = num_projections

        evaluation_points = torch.linspace(
            0.0,
            3.0,
            knots,
            dtype=torch.float32,
        )
        step_size = 3.0 / (knots - 1)

        integration_weights = torch.full(
            (knots,),
            2.0 * step_size,
            dtype=torch.float32,
        )
        integration_weights[0] = step_size
        integration_weights[-1] = step_size

        gaussian_characteristic_function = torch.exp(-evaluation_points.square() / 2.0)

        self.register_buffer(
            "evaluation_points",
            evaluation_points,
        )
        self.register_buffer(
            "gaussian_characteristic_function",
            gaussian_characteristic_function,
        )
        self.register_buffer(
            "integration_weights",
            integration_weights * gaussian_characteristic_function,
        )

    def forward(
            self,
            embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: Tensor of shape [T, B, D]

        Returns:
            Scalar SIGReg loss.
        """

        embedding_dim = embeddings.shape[-1]
        batch_size = embeddings.shape[-2]

        directions = torch.randn(
            embedding_dim,
            self.num_projections,
            device=embeddings.device,
        )

        directions = directions / directions.norm(p=2, dim=0)
        projected_embeddings = embeddings @ directions

        scaled_projections = (
            projected_embeddings.unsqueeze(-1) * self.evaluation_points
        )

        empirical_real = scaled_projections.cos().mean(dim=-3)
        empirical_imaginary = scaled_projections.sin().mean(dim=-3)

        error = (
            empirical_real - self.gaussian_characteristic_function
        ).square() + empirical_imaginary.square()

        statistic = (
            error @ self.integration_weights
        ) * batch_size

        return statistic.mean()