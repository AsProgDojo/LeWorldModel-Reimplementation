import pytest

from lewm.config import (
    Config,
    DataConfig,
    LossConfig,
    ModelConfig,
    TrainingConfig,
    load_config,
    validate_config,
)


def test_load_base_config():
    config = load_config("configs/base.yaml")

    assert config.seed == 42

    assert config.model.image_size == 224
    assert config.model.patch_size == 14
    assert config.model.embed_dim == 192

    assert config.data.frame_skip == 5
    assert config.data.sequence_length == 3

    assert config.training.batch_size == 32
    assert config.training.num_epochs == 10

    assert config.loss.sigreg_weight == 0.09


def test_image_size_must_be_divisible_by_patch_size():
    config = Config(
        seed=42,
        model=ModelConfig(
            image_size=224,
            patch_size=15,
            embed_dim=192,
        ),
        data=DataConfig(
            frame_skip=5,
            sequence_length=3,
        ),
        training=TrainingConfig(
            batch_size=32,
            num_epochs=10,
            learning_rate=1e-3,
            weight_decay=0.05,
        ),
        loss=LossConfig(
            sigreg_weight=0.09,
        ),
    )

    with pytest.raises(ValueError):
        validate_config(config)


def test_sequence_length_must_be_positive():
    config = Config(
        seed=42,
        model=ModelConfig(
            image_size=224,
            patch_size=14,
            embed_dim=192,
        ),
        data=DataConfig(
            frame_skip=5,
            sequence_length=0,
        ),
        training=TrainingConfig(
            batch_size=32,
            num_epochs=10,
            learning_rate=1e-3,
            weight_decay=0.05,
        ),
        loss=LossConfig(
            sigreg_weight=0.09,
        ),
    )

    with pytest.raises(ValueError):
        validate_config(config)