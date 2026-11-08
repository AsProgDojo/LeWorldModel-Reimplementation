import json
import sys
import types

from lewm.loggers import JsonlLogger, MultiLogger, WandbLogger, build_logger

def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]

class FakeRun:
    def __init__(self):
        self.logged = []
        self.finished = False

    def log(self, metrics, step):
        self.logged.append((metrics, step))

    def finish(self):
        self.finished = True

def install_fake_wandb(monkeypatch):
    calls = []
    run = FakeRun()

    def init(**kwargs):
        calls.append(kwargs)
        return run

    monkeypatch.setitem(sys.modules, 'wandb', types.SimpleNamespace(init=init))
    return calls, run

def test_jsonl_logger_writes_one_line_per_call(tmp_path):
    path = tmp_path / 'logs' / 'metrics.jsonl'
    logger = JsonlLogger(path)
    logger.log({'train/loss': 1.5}, step=1)
    logger.log({'train/loss': 1.0}, step=2)
    logger.close()

    assert read_lines(path) == [
        {'step': 1, 'train/loss': 1.5},
        {'step': 2, 'train/loss': 1.0},
    ]

def test_jsonl_logger_appends_when_reopened(tmp_path):
    path = tmp_path / 'metrics.jsonl'
    for step in (1, 2):
        logger = JsonlLogger(path)
        logger.log({'x': 0.0}, step=step)
        logger.close()

    assert [line['step'] for line in read_lines(path)] == [1, 2]

def test_multi_logger_fans_out_and_closes_all(tmp_path):
    first, second = FakeRun(), FakeRun()
    first.close = lambda: setattr(first, 'finished', True)
    second.close = lambda: setattr(second, 'finished', True)

    with MultiLogger([first, second]) as logger:
        logger.log({'a': 1.0}, 3)

    assert first.logged == [({'a': 1.0}, 3)]
    assert second.logged == [({'a': 1.0}, 3)]
    assert first.finished and second.finished

def test_build_logger_without_wandb_only_writes_jsonl(tmp_path, monkeypatch):
    # make sure wandb is not touched at all
    monkeypatch.setitem(sys.modules, 'wandb', None)

    with build_logger(tmp_path, {'seed': 1}) as logger:
        assert len(logger.loggers) == 1
        logger.log({'a': 1.0}, 1)

    assert read_lines(tmp_path / 'metrics.jsonl') == [{'step': 1, 'a' : 1.0}]

def test_wandb_logger_initializes_like_the_original_launcher(monkeypatch):
    calls, run = install_fake_wandb(monkeypatch)

    logger = WandbLogger(
        entity="lewm", project="lewm", name="lewm", run_id="123", config={"seed": 1},
    )
    logger.log({'a': 1.0}, step=7)
    logger.close()

    assert calls == [
        {
            "entity": "lewm",
            "project": "lewm",
            "name": "lewm",
            "id": "123",
            "resume": "allow",
            "config": {"seed": 1},
        }
    ]
    assert run.logged == [({"a": 1.0}, 7)]
    assert run.finished

def test_build_logger_with_wandb_logs_to_both(tmp_path, monkeypatch):
    calls, run = install_fake_wandb(monkeypatch)
    settings = {"entity": "e", "project": "p", "name": "n", "run_id": "r"}

    with build_logger(tmp_path, {'seed': 1}, wandb=settings) as logger:
        assert len(logger.loggers) == 2
        logger.log({'a': 2.0}, 4)

    assert calls[0]['config'] == {'seed': 1}
    assert run.logged == [({'a': 2.0}, 4)]
    assert run.finished
    assert read_lines(tmp_path / "metrics.jsonl") == [{"step": 4, "a": 2.0}]




