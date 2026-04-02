from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a config-driven CycleGAN experiment.")
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("torch is required to train the model. Install dependencies with `pip install -r requirements.txt`.") from exc

    from cyclegan.config import config_to_dict, load_config_from_yaml
    from cyclegan.trainer import CycleGANTrainer

    config = load_config_from_yaml(Path(args.config))
    if config.device == "cpu" and torch.cuda.is_available():
        config.device = "cuda"

    print(f"Python: {sys.executable}")
    print(f"Torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(config_to_dict(config))

    trainer = CycleGANTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
