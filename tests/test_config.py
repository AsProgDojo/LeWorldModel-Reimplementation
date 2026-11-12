from dataclasses import replace

import pytest
import yaml

from lewm.config import Config, load_config, validate_config


def base_config() -> Config:
    return load_config("configs/base.yaml")


def write_yaml(tmp_path, content: dict):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(content), encoding="utf-8")
    return path


MINIMAL = {
    "seed": 1,
    "model": {},
    "data": {},
    "training": {},
    "loss": {},
}


def test_load_base_config():
    config = base_config()

    assert config.seed == 3072
    assert config.output_model_name == "lewm"

    assert config.model.image_size == 224
    assert config.model.patch_size == 14
    assert config.model.embed_dim == 192

    assert config.data.frameskip == 5
    assert config.data.history_size == 3
    assert config.data.num_steps == 4
    assert config.data.dataset_name == "tworoom.h5"
    assert config.data.keys_to_load == ["pixels", "action", "proprio"]
    assert config.data.keys_to_cache == ["action", "proprio"]

    assert config.training.batch_size == 128
    assert config.training.num_epochs == 100
    assert config.training.device == "cuda"
    assert config.training.checkpoint_interval == 1
    assert config.training.sanity_val_batches == 1
    assert config.training.warmup_fraction == 0.01
    assert config.training.warmup_start_lr == 0.0
    assert config.training.eta_min == 0.0

    assert config.loss.sigreg_weight == 0.09

    assert config.loader.num_workers == 6
    assert config.loader.persistent_workers is True
    assert config.loader.prefetch_factor == 3
    assert config.loader.pin_memory is True

    assert config.wandb.enabled is False
    assert config.wandb.project == "lewm"


def test_base_yaml_matches_the_dataclass_defaults():
    # base.yaml spells out the released values; the defaults must agree with it
    from lewm.config import (
        DataConfig, LoaderConfig, LossConfig, ModelConfig, TrainingConfig, WandbConfig,
    )

    config = base_config()
    assert config.model == ModelConfig()
    assert config.data == DataConfig()
    assert config.training == TrainingConfig()
    assert config.loss == LossConfig()
    assert config.loader == LoaderConfig()
    assert config.wandb == WandbConfig()


def test_load_smoke_config():
    config = load_config("configs/smoke.yaml")

    assert config.training.device == "cpu"
    assert config.loader.num_workers == 0
    assert config.loader.prefetch_factor is None


def test_optional_sections_fall_back_to_defaults(tmp_path):
    config = load_config(write_yaml(tmp_path, MINIMAL))

    assert config.loader.num_workers == 6
    assert config.wandb.enabled is False
    assert config.output_dir == "runs"
    assert config.data.dataset_name == "tworoom.h5"


def test_unknown_keys_are_rejected(tmp_path):
    with pytest.raises(TypeError):
        load_config(write_yaml(tmp_path, {**MINIMAL, "wanbd": {}}))
    with pytest.raises(TypeError):
        load_config(write_yaml(tmp_path, {**MINIMAL, "loader": {"num_worker": 2}}))


def test_missing_required_sections_are_rejected(tmp_path):
    incomplete = {key: value for key, value in MINIMAL.items() if key != "loss"}

    with pytest.raises(ValueError, match="loss"):
        load_config(write_yaml(tmp_path, incomplete))


def test_image_size_must_be_divisible_by_patch_size():
    config = base_config()
    config = replace(config, model=replace(config.model, patch_size=15))

    with pytest.raises(ValueError):
        validate_config(config)


def test_history_size_must_be_positive():
    config = base_config()
    config = replace(config, data=replace(config.data, history_size=0))

    with pytest.raises(ValueError):
        validate_config(config)


def test_checkpoint_interval_must_be_positive():
    config = base_config()
    config = replace(config, training=replace(config.training, checkpoint_interval=0))

    with pytest.raises(ValueError, match="checkpoint_interval"):
        validate_config(config)


@pytest.mark.parametrize("warmup_fraction", [-0.1, 1.0])
def test_warmup_fraction_must_be_in_zero_one(warmup_fraction):
    config = base_config()
    config = replace(config, training=replace(config.training, warmup_fraction=warmup_fraction))

    with pytest.raises(ValueError, match="warmup_fraction"):
        validate_config(config)


def test_device_must_be_cpu_or_cuda():
    config = base_config()
    validate_config(replace(config, training=replace(config.training, device="cuda:1")))

    with pytest.raises(ValueError, match="device"):
        validate_config(replace(config, training=replace(config.training, device="tpu")))


def test_keys_to_load_must_contain_pixels_and_action():
    config = base_config()

    for keys in (["action", "proprio"], ["pixels", "proprio"]):
        broken = replace(config, data=replace(config.data, keys_to_load=keys, keys_to_cache=[]))
        with pytest.raises(ValueError, match="keys_to_load"):
            validate_config(broken)


def test_keys_to_cache_must_be_loaded():
    config = base_config()
    broken = replace(config, data=replace(config.data, keys_to_cache=["state"]))

    with pytest.raises(ValueError, match="keys_to_cache"):
        validate_config(broken)


def test_loader_without_workers_cannot_be_persistent_or_prefetch():
    config = base_config()
    no_workers = replace(config.loader, num_workers=0)

    with pytest.raises(ValueError, match="persistent_workers"):
        validate_config(replace(config, loader=replace(no_workers, prefetch_factor=None)))
    with pytest.raises(ValueError, match="prefetch_factor"):
        validate_config(replace(config, loader=replace(no_workers, persistent_workers=False)))

    valid = replace(no_workers, persistent_workers=False, prefetch_factor=None)
    validate_config(replace(config, loader=valid))


def test_negative_workers_are_rejected():
    config = base_config()

    with pytest.raises(ValueError, match="num_workers"):
        validate_config(replace(config, loader=replace(config.loader, num_workers=-1)))


def test_wandb_needs_an_entity_when_enabled():
    config = base_config()

    with pytest.raises(ValueError, match="entity"):
        validate_config(replace(config, wandb=replace(config.wandb, enabled=True)))

    enabled = replace(config.wandb, enabled=True, entity="my-team")
    validate_config(replace(config, wandb=enabled))


def test_loader_settings_are_accepted_by_a_real_dataloader():
    # the config validation must agree with what torch actually allows
    from torch.utils.data import DataLoader

    for loader in (
        base_config().loader,
        load_config("configs/smoke.yaml").loader,
    ):
        DataLoader(
            list(range(4)),
            num_workers=loader.num_workers,
            persistent_workers=loader.persistent_workers,
            prefetch_factor=loader.prefetch_factor,
            pin_memory=loader.pin_memory,
        )
