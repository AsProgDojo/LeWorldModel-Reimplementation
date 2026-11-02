from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    image_size: int = 224
    patch_size: int = 14
    embed_dim: int = 192
    encoder_depth: int = 12
    encoder_heads: int = 3
    encoder_mlp_dim: int = 768
    action_smoothed_dim: int = 10
    action_mlp_scale: int = 4
    predictor_depth: int = 6
    predictor_heads: int = 16
    predictor_mlp_dim: int = 2048
    predictor_dim_head: int = 64
    predictor_dropout: float = 0.1
    predictor_emb_dropout: float = 0.0
    projector_hidden_dim: int = 2048


@dataclass
class DataConfig:
    frameskip: int = 5
    history_size: int = 3
    num_preds: int = 1
    train_split: float = 0.9

    @property
    def num_steps(self) -> int:
        # Frames per training window (history + predicted steps).
        return self.history_size + self.num_preds


@dataclass
class TrainingConfig:
    batch_size: int = 128
    num_epochs: int = 100
    learning_rate: float = 5e-5
    weight_decay: float = 1e-3
    gradient_clip_val: float = 1.0
    precision: str = "bf16"


@dataclass
class LossConfig:
    sigreg_weight: float = 0.09
    sigreg_knots: int = 17
    sigreg_num_projections: int = 1024


@dataclass
class Config:
    seed: int
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    loss: LossConfig


_PRECISIONS = {"bf16", "16", "32"}


def validate_config(config: Config) -> None:
    m, d, t, loss = config.model, config.data, config.training, config.loss

    positive_ints = {
        "model.image_size": m.image_size,
        "model.patch_size": m.patch_size,
        "model.embed_dim": m.embed_dim,
        "model.encoder_depth": m.encoder_depth,
        "model.encoder_heads": m.encoder_heads,
        "model.encoder_mlp_dim": m.encoder_mlp_dim,
        "model.action_smoothed_dim": m.action_smoothed_dim,
        "model.action_mlp_scale": m.action_mlp_scale,
        "model.predictor_depth": m.predictor_depth,
        "model.predictor_heads": m.predictor_heads,
        "model.predictor_mlp_dim": m.predictor_mlp_dim,
        "model.predictor_dim_head": m.predictor_dim_head,
        "model.projector_hidden_dim": m.projector_hidden_dim,
        "data.frameskip": d.frameskip,
        "data.history_size": d.history_size,
        "data.num_preds": d.num_preds,
        "training.batch_size": t.batch_size,
        "training.num_epochs": t.num_epochs,
        "loss.sigreg_knots": loss.sigreg_knots,
        "loss.sigreg_num_projections": loss.sigreg_num_projections,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    if m.image_size % m.patch_size != 0:
        raise ValueError("model.image_size must be divisible by model.patch_size")
    if m.embed_dim % m.encoder_heads != 0:
        raise ValueError("model.embed_dim must be divisible by model.encoder_heads")
    for name, p in {
        "model.predictor_dropout": m.predictor_dropout,
        "model.predictor_emb_dropout": m.predictor_emb_dropout,
    }.items():
        if not 0.0 <= p < 1.0:
            raise ValueError(f"{name} must be in [0, 1), got {p}")

    if not 0.0 < d.train_split < 1.0:
        raise ValueError("data.train_split must be in (0, 1)")

    if t.learning_rate <= 0:
        raise ValueError("training.learning_rate must be positive")
    if t.weight_decay < 0:
        raise ValueError("training.weight_decay must be non-negative")
    if t.gradient_clip_val <= 0:
        raise ValueError("training.gradient_clip_val must be positive")
    if t.precision not in _PRECISIONS:
        raise ValueError(f"training.precision must be one of {sorted(_PRECISIONS)}")

    if loss.sigreg_weight < 0:
        raise ValueError("loss.sigreg_weight must be non-negative")


def load_config(path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    config = Config(
        seed=raw["seed"],
        model=ModelConfig(**raw["model"]),
        data=DataConfig(**raw["data"]),
        training=TrainingConfig(**raw["training"]),
        loss=LossConfig(**raw["loss"]),
    )
    validate_config(config)
    return config