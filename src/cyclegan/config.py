from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CycleGANConfig:
    dataset_name: str = "apple2orange"
    data_root: Path = Path("datasets")
    output_root: Path = Path("outputs")
    checkpoint_root: Path = Path("checkpoints") / "pytorch"
    logs_root: Path = Path("logs") / "pytorch"

    image_size: int = 256
    batch_size: int = 1
    epochs: int = 100
    lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    lambda_cycle: float = 10.0
    lambda_identity: float = 5.0
    save_every_n_epochs: int = 5
    num_workers: int = 0
    subset_size: int | None = None
    seed: int = 42
    replay_buffer_size: int = 50
    sample_count: int = 4

    n_residual_blocks: int = 9
    generator_filters: int = 64
    discriminator_filters: int = 64
    channels: int = 3

    soft_fake_label_range: tuple[float, float] = (0.0, 0.3)
    soft_real_label_range: tuple[float, float] = (0.7, 1.2)

    resume: bool = True
    device: str = "cpu"
    experiment_name: str | None = None
    train_split_x: str = "trainA"
    train_split_y: str = "trainB"
    test_split_x: str = "testA"
    test_split_y: str = "testB"
    metrics_filename: str = "metrics.csv"
    checkpoint_name: str = "training-checkpoint.pt"
    sample_dirname: str = "samples"
    inference_dirname: str = "inference"
    evaluation_dirname: str = "evaluation"
    tracking_uri: str = "mlruns"
    tracking_experiment: str = "cyclegan"
    tracking_enabled: bool = True
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def dataset_dir(self) -> Path:
        return self.data_root / self.dataset_name

    @property
    def run_name(self) -> str:
        return self.experiment_name or self.dataset_name

    @property
    def checkpoint_dir(self) -> Path:
        return self.checkpoint_root / self.run_name

    @property
    def output_dir(self) -> Path:
        return self.output_root / self.run_name

    @property
    def logs_dir(self) -> Path:
        return self.logs_root / self.run_name

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoint_dir / self.checkpoint_name

    @property
    def metrics_path(self) -> Path:
        return self.logs_dir / self.metrics_filename

    @property
    def train_sample_dir(self) -> Path:
        return self.output_dir / self.sample_dirname

    @property
    def inference_dir(self) -> Path:
        return self.output_dir / self.inference_dirname

    @property
    def evaluation_dir(self) -> Path:
        return self.output_dir / self.evaluation_dirname

    def ensure_dirs(self) -> None:
        for path in (
            self.checkpoint_dir,
            self.output_dir,
            self.logs_dir,
            self.train_sample_dir,
            self.inference_dir,
            self.evaluation_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def config_to_dict(config: CycleGANConfig) -> dict[str, Any]:
    raw = asdict(config)
    for key, value in raw.items():
        if isinstance(value, Path):
            raw[key] = str(value)
    return raw


def load_config_from_yaml(path: Path) -> CycleGANConfig:
    content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    path_fields = {"data_root", "output_root", "checkpoint_root", "logs_root"}
    normalized = {key: Path(value) if key in path_fields else value for key, value in content.items()}
    return CycleGANConfig(**normalized)
