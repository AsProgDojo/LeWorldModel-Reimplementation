import json
import pytest
import torch
from lewm.fit import fit
from lewm.loggers import build_logger
from test_jepa import make_batch
from test_optimization import make_setup

HISTORY, PREDS = 3, 1

torch.manual_seed(123)
TRAIN = [make_batch(B=2, T=HISTORY + PREDS) for _ in range(3)]
VAL = [make_batch(B=2, T=HISTORY + PREDS) for _ in range(2)]

def run_fit(run_dir, num_epochs, setup=None, **kwargs):
    setup = setup or make_setup()
    model, sigreg, optimizer, scheduler = setup
    history = fit(
        model, sigreg, optimizer, scheduler, TRAIN, VAL,
        num_epochs=num_epochs, history_size=HISTORY, num_preds=PREDS,
        sigreg_weight=0.09, gradient_clip_val=1.0, precision="32",
        device="cpu", run_dir=run_dir, run_name="lewm", config={"seed": 0},
        **kwargs,
    )
    return setup, history


def test_fit_trains_validates_and_saves_everything(tmp_path):
    _, history = run_fit(tmp_path, num_epochs=2)

    assert len(history) == 2
    assert set(history[0]) == {'pred_loss', 'sigreg_loss', 'loss'}
    for name in ("config.yaml", "weights_epoch_1.pt", "weights_epoch_2.pt", "lewm_weights.ckpt"):
        assert (tmp_path / name).exists(), name


def test_checkpoint_interval_still_saves_the_final_epoch(tmp_path):
    run_fit(tmp_path, num_epochs=3, checkpoint_interval=2)

    assert not (tmp_path / "weights_epoch_1.pt").exists()
    assert (tmp_path / "weights_epoch_2.pt").exists()
    assert (tmp_path / "weights_epoch_3.pt").exists()

def test_rejects_non_positive_checkpoint_interval(tmp_path):
    with pytest.raises(ValueError):
        run_fit(tmp_path, num_epochs=1, checkpoint_interval=0)


def test_fit_logs_every_step_and_every_validation(tmp_path):
    with build_logger(tmp_path, {'seed': 0}) as logger:
        run_fit(tmp_path, num_epochs=2, logger=logger)

    lines = [
        json.loads(line)
        for line in (tmp_path / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()
    ]
    train_lines = [line for line in lines if 'train/loss' in line]
    val_lines = [line for line in lines if 'val/loss' in line]

    assert [line["step"] for line in train_lines] == [1, 2, 3, 4, 5, 6]
    assert all("hparams/lr_model_opt" in line for line in train_lines)
    assert [line["step"] for line in val_lines] == [3, 6]
    assert {"train/pred_loss", "train/sigreg_loss"} <= set(train_lines[0])


def test_resumed_run_matches_an_uninterrupted_run(tmp_path):
    straight_setup, straight_history = run_fit(tmp_path / 'straight', num_epochs=2)

    # "crash" after one epoch, then start again from scratch objects
    run_fit(tmp_path / 'resumed', num_epochs=1)
    resumed_setup, resumed_history = run_fit(tmp_path / 'resumed', num_epochs=2)

    assert len(resumed_history) == 1        # only the second epoch was trained
    assert resumed_setup[3].last_epoch == straight_setup[3].last_epoch == 6

    for a, b in zip(
        straight_setup[0].state_dict().values(), resumed_setup[0].state_dict().values()
    ):
        assert torch.allclose(a, b, atol=1e-6)
    assert resumed_history[0]['pred_loss'] == pytest.approx(
        straight_history[1]['pred_loss'], rel=1e-5
    )

def test_resumed_run_continues_the_global_step(tmp_path):
    run_fit(tmp_path, num_epochs=1)

    with build_logger(tmp_path / 'second', {}) as logger:
        run_fit(tmp_path, num_epochs=2, logger=logger)

    lines = [
        json.loads(line)
        for line in (tmp_path / 'second' / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()
    ]
    train_steps = [line['step'] for line in lines if 'train/loss' in line]
    assert train_steps == [4, 5, 6]

def test_nothing_to_do_when_all_epochs_are_done(tmp_path):
    run_fit(tmp_path, num_epochs=2)
    _, history = run_fit(tmp_path, num_epochs=2)

    assert history == []

def test_sanity_validation_only_runs_on_a_fresh_start(tmp_path):
    def eval_calls_during(run_dir, num_epochs):
        setup = make_setup()
        calls = []
        setup[0].projector.register_forward_pre_hook(
            lambda module, args: calls.append(1) if not module.training else None
        )
        run_fit(run_dir, num_epochs, setup=setup)
        return len(calls)

    # fresh: 1 sanity batch + 2 validation batches
    assert eval_calls_during(tmp_path, num_epochs=1) == 1 + len(VAL)
    # resumed for one more epoch: just the 2 validation batches
    assert eval_calls_during(tmp_path, num_epochs=2) == len(VAL)




