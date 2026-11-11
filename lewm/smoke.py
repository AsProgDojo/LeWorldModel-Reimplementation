import argparse
import torch
import torch.nn as nn

from lewm.engine import move_to_device, validate
from lewm.fit import fit
from lewm.losses.sigreg import SIGReg
from lewm.models.action_encoder import ActionEncoder
from lewm.models.encoder import VisionTransformer
from lewm.models.jepa import JEPA
from lewm.models.predictor import ARPredictor
from lewm.models.projector import MLP
from lewm.optimization import build_optimizer
from lewm.scheduler import build_scheduler

# Our own debugging setup (not for the reference): a tiny model and a toy world
IMAGE_SIZE = 28
EMBED_DIM = 48
ACTION_DIM = 2
HISTORY_SIZE = 3
NUM_PREDS = 1
SQUARE_RADIUS = 3
STEP_SIZE = 3.0
LOW, HIGH = 4.0, IMAGE_SIZE - 5.0

def make_fake_batches(
        num_batches: int, batch_size: int, seed: int,
) -> list[dict[str, torch.Tensor]]:
    """
    A toy world: a bright square on a black image. The action at frame t is a
    2D velocity that moves the square between frame t and frame t+1, so the
    next frame is predictable from the current frame plus the action.
    pixels: (B, T, 3, S, S), action: (B, T, 2).
    """

    generator = torch.Generator().manual_seed(seed)
    num_frames = HISTORY_SIZE + NUM_PREDS
    coords = torch.arange(IMAGE_SIZE, dtype=torch.float32)
    batches = []
    for _ in range(num_batches):
        actions = torch.rand(batch_size, num_frames, ACTION_DIM, generator=generator) * 2 - 1
        position = LOW + (HIGH - LOW) * torch.rand(batch_size, 2, generator=generator)
        positions = []
        for t in range(num_frames):
            positions.append(position)
            position = (position + STEP_SIZE * actions[:, t]).clamp(LOW, HIGH)
        positions = torch.stack(positions, dim=1)   # (B, T, 2)

        near_x = (coords - positions[..., 0:1]).abs() <= SQUARE_RADIUS  # (B, T, S)
        near_y = (coords - positions[..., 1:2]).abs() <= SQUARE_RADIUS  # (B, T, S)
        square = (near_y.unsqueeze(-1) & near_x.unsqueeze(-2)).float()  # (B, T, S, S)
        pixels = square.unsqueeze(2).expand(-1, -1, 3, -1, -1).contiguous()
        batches.append({"pixels": pixels, "action": actions})
    return batches


def make_smoke_model() -> JEPA:
    return JEPA(
        encoder=VisionTransformer(
            image_size=IMAGE_SIZE, patch_size=14, hidden_size=EMBED_DIM,
            depth=2, num_heads=3, mlp_dim=2 * EMBED_DIM,
        ),
        predictor=ARPredictor(
            num_frames=HISTORY_SIZE, depth=2, heads=2, mlp_dim=2 * EMBED_DIM,
            input_dim=EMBED_DIM, hidden_dim=EMBED_DIM, dim_head=16,
        ),
        action_encoder=ActionEncoder(
            input_dim=ACTION_DIM, smoothed_dim=ACTION_DIM, emb_dim=EMBED_DIM
        ),
        projector=MLP(EMBED_DIM, 2 * EMBED_DIM, EMBED_DIM, norm_fn=nn.BatchNorm1d),
        pred_proj=MLP(EMBED_DIM, 2 * EMBED_DIM, EMBED_DIM, norm_fn=nn.BatchNorm1d),
    )


class ListLogger:
    """Keeps every logged metric in memory."""

    def __init__(self):
        self.records = []

    def log(self, metrics: dict[str, float], step: int) -> None:
        self.records.append({'step': step, **metrics})

    def close(self) -> None:
        pass


def shuffle_actions(batches: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    """Gives every sample the actions of its neighbour in the batch."""
    return [{**batch, "action": batch["action"].roll(1, dims=0)} for batch in batches]


@torch.no_grad()
def embedding_std(
    model: JEPA, batches: list[dict[str, torch.Tensor]], device: torch.device
) -> torch.Tensor:
    """Per-dimension standard deviation of all encoded frames, shape (D,)."""
    model.eval()
    embeddings = []
    for batch in batches:
        batch = move_to_device(batch, device)
        embeddings.append(model.encode(batch)['emb'].flatten(0, 1).float())
    return torch.cat(embeddings).std(dim=0)


def mean_over(records: list[dict], key: str, steps: slice) -> float:
    values = [record[key] for record in records if key in record][steps]
    return sum(values) / len(values)

def run_smoke(
        run_dir,
        device: str = "cpu",
        precision: str = "32",
        num_epochs: int = 5,
        num_train_batches: int = 40,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        seed: int = 0,
) -> dict[str, float]:
    torch.manual_seed(seed)
    train_batches = make_fake_batches(num_train_batches, batch_size, seed=seed)
    val_batches = make_fake_batches(4, batch_size, seed=seed + 1)

    model = make_smoke_model()
    sigreg = SIGReg(knots=17, num_projections=64)
    optimizer = build_optimizer(model, learning_rate=learning_rate, weight_decay=1e-3)
    scheduler = build_scheduler(optimizer, total_steps=num_epochs * num_train_batches)
    logger = ListLogger()

    fit(
        model, sigreg, optimizer, scheduler, train_batches, val_batches,
        num_epochs=num_epochs, history_size=HISTORY_SIZE, num_preds=NUM_PREDS,
        sigreg_weight=0.09, gradient_clip_val=1.0, precision=precision,
        device=device, run_dir=run_dir, run_name="smoke", config={"seed": seed},
        logger=logger,
    )

    steps = [record for record in logger.records if 'train/loss' in record]
    device = torch.device(device)
    real = validate(
        model, sigreg, val_batches, HISTORY_SIZE, NUM_PREDS, 0.09, precision, device
    )
    shuffled = validate(
        model, sigreg, shuffle_actions(val_batches), HISTORY_SIZE, NUM_PREDS, 0.09,
        precision, device,
    )
    std = embedding_std(model, val_batches, device)

    return {
        "first_loss": mean_over(steps, "train/loss", slice(0, 10)),
        "last_loss": mean_over(steps, "train/loss", slice(-10, None)),
        "first_pred_loss": mean_over(steps, "train/pred_loss", slice(0, 10)),
        "last_pred_loss": mean_over(steps, "train/pred_loss", slice(-10, None)),
        "val_pred_loss": real["pred_loss"],
        "val_pred_loss_shuffled_actions": shuffled["pred_loss"],
        "embedding_std_mean": std.mean().item(),
        "embedding_std_min": std.min().item(),
        "embedding_std_max": std.max().item(),
    }

def main() -> None:
    import tempfile

    parser = argparse.ArgumentParser(description="Smoke test of fit() on a toy world.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", default="32", choices=["32", "bf16"])
    parser.add_argument("--epochs", type=int, default=5)
    arguments = parser.parse_args()

    with tempfile.TemporaryDirectory() as run_dir:
        results = run_smoke(
            run_dir, arguments.device, arguments.precision, arguments.epochs
        )
    for name, value in results.items():
        print(f'{name:34s} {value:.4f}')

if __name__ == "__main__":
    main()