from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CycleGAN for unpaired image translation.")
    parser.add_argument("--dataset-name", default="apple2orange")
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--checkpoint-root", default="checkpoints/pytorch")
    parser.add_argument("--logs-root", default="logs/pytorch")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--subset-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("torch is required to train the model. Install dependencies with `pip install -r requirements.txt`.") from exc

    from cyclegan.config import CycleGANConfig
    from cyclegan.trainer import CycleGANTrainer, config_to_dict

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Python: {sys.executable}")
    print(f"Torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    config = CycleGANConfig(
        dataset_name=args.dataset_name,
        data_root=Path(args.data_root),
        output_root=Path(args.output_root),
        checkpoint_root=Path(args.checkpoint_root),
        logs_root=Path(args.logs_root),
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        subset_size=None if args.subset_size <= 0 else args.subset_size,
        num_workers=args.num_workers,
        save_every_n_epochs=args.save_every,
        seed=args.seed,
        resume=not args.no_resume,
        device=args.device or default_device,
        experiment_name=args.experiment_name,
    )
    print(json.dumps(config_to_dict(config), indent=2))
    trainer = CycleGANTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
