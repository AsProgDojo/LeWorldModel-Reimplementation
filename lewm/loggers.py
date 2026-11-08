import json
from collections.abc import Iterable
from pathlib import Path

class JsonlLogger:
    """Appends one JSON object per call to a .jsonl file."""

    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, 'a', encoding='utf-8')

    def log(self, metrics: dict[str, float], step: int) -> None:
        self._file.write(json.dumps({'step':step, **metrics}) + '\n')
        self._file.flush()

    def close(self) -> None:
        self._file.close()



class WandbLogger:
    """
    Settings follow the original's launcher config: resume="allow" with a
    fixed run id, so a resumed run continues the same wandb run. Models are
    not uploaded (log_model: False in the original).
    """

    def __init__(
            self,
            entity: str,
            project: str,
            name: str,
            run_id: str,
            config: dict,
    ):
        import wandb    # imported here so wandb is only needed when enabled

        self._run = wandb.init(
            entity=entity,
            project=project,
            name=name,
            id=run_id,
            resume="allow",
            config=config,
        )

    def log(self, metrics: dict[str, float], step: int) -> None:
        self._run.log(metrics, step=step)

    def close(self) -> None:
        self._run.finish()



class MultiLogger:
    def __init__(self, loggers: Iterable):
        self.loggers = list(loggers)

    def log(self, metrics: dict[str, float], step: int) -> None:
        for logger in self.loggers:
            logger.log(metrics, step)

    def close(self) -> None:
        for logger in self.loggers:
            logger.close()

    def __enter__(self) -> 'MultiLogger':
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

def build_logger(
        log_dir: str | Path,
        hyperparameters: dict,
        wandb: dict | None = None,
) -> MultiLogger:
    """
    Always writes <log_dir>/metrics.jsonl. If `wandb` is given, also logs to
    wandb; it needs the keys entity, project, name and run_id.
    """
    loggers = [JsonlLogger(Path(log_dir) / 'metrics.jsonl')]
    if wandb is not None:
        loggers.append(WandbLogger(**wandb, config=hyperparameters))
    return MultiLogger(loggers)