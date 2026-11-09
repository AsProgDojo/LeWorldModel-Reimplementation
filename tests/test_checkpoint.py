import torch
import yaml

from lewm.checkpoint import (
    load_training_state,
    resume_path,
    save_config,
    save_training_state,
    save_weights,
)
from test_jepa import make_batch, make_model
from test_optimization import make_setup, step


def test_save_weights_round_trip(tmp_path):
    model = make_model()
    path = save_weights(model, tmp_path, epoch=3)

    assert path.name == 'weights_epoch_3.pt'

    other = make_model()
    other.load_state_dict(torch.load(path, weights_only=True))
    for a, b in zip(model.state_dict().values(), other.state_dict().values()):
        assert torch.equal(a, b)


def test_save_config_writes_readable_yaml(tmp_path):
    path = save_config({"seed": 3072, "model": {"embed_dim": 192}}, tmp_path)

    assert path.name == 'config.yaml'
    assert yaml.safe_load(path.read_text(encoding='utf-8')) == {
        "seed": 3072,
        "model": {"embed_dim": 192},
    }

def test_resume_path_uses_run_name(tmp_path):
    assert resume_path(tmp_path, 'lewm') == tmp_path / 'lewm_weights.ckpt'

def test_training_state_round_trip(tmp_path):
    setup = make_setup()
    model, _, optimizer, scheduler = setup
    for _ in range(3):
        step(setup, make_batch(B=2, T=4))

    path = tmp_path / 'state.ckpt'
    save_training_state(path, model, optimizer, scheduler, epoch=2, global_step=3)
    assert not path.with_suffix(".ckpt.tmp").exists()

    # numbers drawn right after saving must be reproduced after loading
    expected_random = torch.rand(3)

    fresh_model, _, fresh_optimizer, fresh_scheduler = make_setup()
    torch.manual_seed(999)
    epoch, global_step = load_training_state(
        path, fresh_model, fresh_optimizer, fresh_scheduler
    )

    assert (epoch, global_step) == (2, 3)
    for a, b in zip(model.state_dict().values(), fresh_model.state_dict().values()):
        assert torch.equal(a, b)
    assert fresh_scheduler.last_epoch == scheduler.last_epoch
    assert fresh_optimizer.param_groups[0]['lr'] == optimizer.param_groups[0]['lr']

    first_parameter = next(iter(model.parameters()))
    fresh_first_parameter = next(iter(fresh_model.parameters()))
    assert torch.equal(
        optimizer.state[first_parameter]['exp_avg'],
        fresh_optimizer.state[fresh_first_parameter]['exp_avg'],
    )
    assert torch.equal(torch.rand(3), expected_random)




    