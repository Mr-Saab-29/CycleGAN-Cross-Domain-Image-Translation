from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument("--dataset-name", default="apple2orange")
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--checkpoint-root", default="checkpoints/pytorch")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument("--direction", choices=("x2y", "y2x"), required=True)
    parser.add_argument("--input", required=True, help="Path to the input image.")
    parser.add_argument("--output", required=True, help="Path to save the translated image.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("torch is required for inference. Install dependencies with `pip install -r requirements.txt`.") from exc

    from cyclegan.config import CycleGANConfig
    from cyclegan.inference import build_inference_transform, load_generators_for_inference, tensor_to_pil, translate
    from PIL import Image

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
    generator_x_to_y, generator_y_to_x = load_generators_for_inference(config)
    model = generator_x_to_y if args.direction == "x2y" else generator_y_to_x

    image = Image.open(args.input).convert("RGB")
    tensor = build_inference_transform(config.image_size)(image).unsqueeze(0)
    translated = translate(model, tensor, config.device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(translated).save(output_path)
    print(f"Saved translated image to {output_path}")


if __name__ == "__main__":
    main()
