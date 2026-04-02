from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sample translations from a trained CycleGAN checkpoint.")
    parser.add_argument("--dataset-name", default="apple2orange")
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--checkpoint-root", default="checkpoints/pytorch")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("torch is required to generate samples. Install dependencies with `pip install -r requirements.txt`.") from exc

    from cyclegan.checkpoints import load_checkpoint
    from cyclegan.config import CycleGANConfig
    from cyclegan.data import load_datasets
    from cyclegan.inference import save_bidirectional_preview
    from cyclegan.models import Discriminator, GeneratorResNet

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

    generator_x_to_y = GeneratorResNet(config.channels, config.channels, config.generator_filters, config.n_residual_blocks).to(config.device)
    generator_y_to_x = GeneratorResNet(config.channels, config.channels, config.generator_filters, config.n_residual_blocks).to(config.device)
    discriminator_x = Discriminator(config.channels, config.discriminator_filters).to(config.device)
    discriminator_y = Discriminator(config.channels, config.discriminator_filters).to(config.device)

    load_checkpoint(
        config.checkpoint_path,
        generators={"x_to_y": generator_x_to_y, "y_to_x": generator_y_to_x},
        discriminators={"x": discriminator_x, "y": discriminator_y},
        optimizers=None,
        device=config.device,
    )

    datasets = load_datasets(config)
    sample_x = datasets.test_x[args.sample_index]
    sample_y = datasets.test_y[args.sample_index]
    destination = config.inference_dir / f"sample_{args.sample_index:03d}.png"
    save_bidirectional_preview(
        generator_x_to_y,
        generator_y_to_x,
        sample_x,
        sample_y,
        config.device,
        destination,
        title=f"{config.run_name} - sample {args.sample_index}",
    )
    print(f"Saved preview to {destination}")


if __name__ == "__main__":
    main()
