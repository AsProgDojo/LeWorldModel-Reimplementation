from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    image_size: int
    patch_size: int
    embed_dim: int


@dataclass
class DataConfig:
    frame_skip: int
    sequence_length: int


@dataclass
class TrainingConfig:
    batch_size: int
    num_epochs: int
    learning_rate: float
    weight_decay: float


@dataclass
class LossConfig:
    sigreg_weight: float


@dataclass
class Config:
    seed: int
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    loss: LossConfig


def load_config(path: str | Path) -> Config:
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    config = Config(
        seed=raw["seed"],
        model=ModelConfig(**raw["model"]),
        data=DataConfig(**raw["data"]),
        training=TrainingConfig(**raw["training"]),
        loss=LossConfig(**raw["loss"]),
    )

    validate_config(config)

    return config


def validate_config(config: Config) -> None:
    if config.model.image_size <= 0:
        raise ValueError("image_size must be positive.")

    if config.model.patch_size <= 0:
        raise ValueError("patch_size must be positive.")

    if config.model.image_size % config.model.patch_size != 0:
        raise ValueError(
            "image_size must be divisible by patch_size."
        )

    if config.model.embed_dim <= 0:
        raise ValueError("embed_dim must be positive.")

    if config.data.frame_skip <= 0:
        raise ValueError("frame_skip must be positive.")

    if config.data.sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")

    if config.training.batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if config.training.num_epochs <= 0:
        raise ValueError("num_epochs must be positive.")

    if config.training.learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    if config.training.weight_decay < 0:
        raise ValueError("weight_decay cannot be negative.")

    if config.loss.sigreg_weight < 0:
        raise ValueError("sigreg_weight cannot be negative.")