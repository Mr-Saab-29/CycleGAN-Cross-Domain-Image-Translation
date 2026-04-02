from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained CycleGAN checkpoint.")
    parser.add_argument("--dataset-name", default="apple2orange")
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--checkpoint-root", default="checkpoints/pytorch")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("torch is required to evaluate the model. Install dependencies with `pip install -r requirements.txt`.") from exc

    from cyclegan.config import CycleGANConfig
    from cyclegan.evaluation import evaluate_generators
    from cyclegan.inference import load_generators_for_inference

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    config = CycleGANConfig(
        dataset_name=args.dataset_name,
        data_root=Path(args.data_root),
        checkpoint_root=Path(args.checkpoint_root),
        output_root=Path(args.output_root),
        image_size=args.image_size,
        subset_size=None,
        experiment_name=args.experiment_name,
        device=args.device or default_device,
    )
    config.ensure_dirs()
    generator_x_to_y, generator_y_to_x = load_generators_for_inference(config)
    metrics, preview_path = evaluate_generators(generator_x_to_y, generator_y_to_x, config)
    print(metrics)
    print(f"Saved evaluation preview to {preview_path}")


if __name__ == "__main__":
    main()
