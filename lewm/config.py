from dataclasses import dataclass, field
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
    # Released config/train/data/tworoom.yaml.
    dataset_name: str = "tworoom.h5"
    keys_to_load: list[str] = field(
        default_factory=lambda: ["pixels", "action", "proprio"]
    )
    keys_to_cache: list[str] = field(default_factory=lambda: ["action", "proprio"])

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
    # Released trainer settings: accelerator "gpu", num_sanity_val_steps=1,
    # SaveCkptCallback(epoch_interval=1).
    device: str = "cuda"
    sanity_val_batches: int = 1
    checkpoint_interval: int = 1
    # Scheduler settings are the stable_pretraining defaults of
    # LinearWarmupCosineAnnealingLR (the released config only names the class).
    warmup_fraction: float = 0.01
    warmup_start_lr: float = 0.0
    eta_min: float = 0.0


@dataclass
class LossConfig:
    sigreg_weight: float = 0.09
    sigreg_knots: int = 17
    sigreg_num_projections: int = 1024


@dataclass
class LoaderConfig:
    # Released config/train/lewm.yaml (loader:). They are passed to DataLoader,
    # which rejects persistent_workers / prefetch_factor when num_workers == 0.
    num_workers: int = 6
    persistent_workers: bool = True
    prefetch_factor: int | None = 3
    pin_memory: bool = True


@dataclass
class WandbConfig:
    # Released launcher/local.yaml has wandb disabled and entity "lewm", which is
    # not ours: set your own entity before enabling it.
    enabled: bool = False
    entity: str = ""
    project: str = "lewm"


@dataclass
class Config:
    seed: int
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    loss: LossConfig
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    # Released lewm.yaml: output_model_name. Our runs go to <output_dir>/<subdir>
    # (the original uses the Hydra job id as subdir, and an empty one is allowed).
    output_model_name: str = "lewm"
    subdir: str = ""
    output_dir: str = "runs"


_PRECISIONS = {"bf16", "16", "32"}
_DEVICE_TYPES = {"cpu", "cuda"}


def validate_config(config: Config) -> None:
    m, d, t, loss = config.model, config.data, config.training, config.loss
    loader, wandb = config.loader, config.wandb

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
        "training.checkpoint_interval": t.checkpoint_interval,
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
    if not d.dataset_name:
        raise ValueError("data.dataset_name must not be empty")
    for required in ("pixels", "action"):
        if required not in d.keys_to_load:
            raise ValueError(f"data.keys_to_load must contain '{required}'")
    for key in d.keys_to_cache:
        if key not in d.keys_to_load:
            raise ValueError(f"data.keys_to_cache has '{key}', which is not in keys_to_load")

    if t.learning_rate <= 0:
        raise ValueError("training.learning_rate must be positive")
    if t.weight_decay < 0:
        raise ValueError("training.weight_decay must be non-negative")
    if t.gradient_clip_val <= 0:
        raise ValueError("training.gradient_clip_val must be positive")
    if t.precision not in _PRECISIONS:
        raise ValueError(f"training.precision must be one of {sorted(_PRECISIONS)}")
    if t.device.split(":")[0] not in _DEVICE_TYPES:
        raise ValueError(f"training.device must start with one of {sorted(_DEVICE_TYPES)}")
    if t.sanity_val_batches < 0:
        raise ValueError("training.sanity_val_batches must be non-negative")
    if not 0.0 <= t.warmup_fraction < 1.0:
        raise ValueError("training.warmup_fraction must be in [0, 1)")
    if t.warmup_start_lr < 0 or t.eta_min < 0:
        raise ValueError("training.warmup_start_lr and training.eta_min must be non-negative")

    if loss.sigreg_weight < 0:
        raise ValueError("loss.sigreg_weight must be non-negative")

    if loader.num_workers < 0:
        raise ValueError("loader.num_workers must be non-negative")
    if loader.prefetch_factor is not None and loader.prefetch_factor <= 0:
        raise ValueError("loader.prefetch_factor must be positive or null")
    if loader.num_workers == 0:
        if loader.persistent_workers:
            raise ValueError("loader.persistent_workers needs loader.num_workers > 0")
        if loader.prefetch_factor is not None:
            raise ValueError("loader.prefetch_factor must be null when loader.num_workers is 0")

    if wandb.enabled and not wandb.entity:
        raise ValueError("wandb.entity must be set when wandb.enabled is true")

    if not config.output_model_name:
        raise ValueError("output_model_name must not be empty")
    if not config.output_dir:
        raise ValueError("output_dir must not be empty")


_VALUE_KEYS = ("seed", "output_model_name", "subdir", "output_dir")
_REQUIRED_SECTIONS = {
    "model": ModelConfig,
    "data": DataConfig,
    "training": TrainingConfig,
    "loss": LossConfig,
}
_OPTIONAL_SECTIONS = {"loader": LoaderConfig, "wandb": WandbConfig}


def load_config(path) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    sections = {**_REQUIRED_SECTIONS, **_OPTIONAL_SECTIONS}

    unknown = set(raw) - set(_VALUE_KEYS) - set(sections)
    if unknown:
        raise TypeError(f"Unknown top-level config keys: {sorted(unknown)}")
    missing = {"seed", *_REQUIRED_SECTIONS} - set(raw)
    if missing:
        raise ValueError(f"Missing required config keys: {sorted(missing)}")

    values = {key: raw[key] for key in _VALUE_KEYS if key in raw}
    for name, section_class in sections.items():
        if name in raw:
            # an unknown key inside a section raises TypeError here
            values[name] = section_class(**(raw[name] or {}))

    config = Config(**values)
    validate_config(config)
    return config
