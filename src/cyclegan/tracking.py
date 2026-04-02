from __future__ import annotations

from pathlib import Path
from typing import Any


class ExperimentTracker:
    def __init__(self, enabled: bool, tracking_uri: str, experiment_name: str, run_name: str) -> None:
        self.enabled = enabled
        self.run = None
        self.mlflow = None
        if not enabled:
            return
        try:
            import mlflow
        except ModuleNotFoundError:
            self.enabled = False
            return

        self.mlflow = mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict[str, Any]) -> None:
        if not self.enabled or self.mlflow is None:
            return
        sanitized = {key: str(value) for key, value in params.items()}
        self.mlflow.log_params(sanitized)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        if not self.enabled or self.mlflow is None:
            return
        self.mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        if not self.enabled or self.mlflow is None or not path.exists():
            return
        self.mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def end(self) -> None:
        if not self.enabled or self.mlflow is None or self.run is None:
            return
        self.mlflow.end_run()
