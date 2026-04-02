from __future__ import annotations

import io
from pathlib import Path

from PIL import Image
import torch
from torchvision import transforms

from .checkpoints import load_checkpoint
from .config import CycleGANConfig
from .models import Discriminator, GeneratorResNet
from .utils import save_translation_preview


@torch.no_grad()
def translate(model: torch.nn.Module, tensor: torch.Tensor, device: str) -> torch.Tensor:
    model.eval()
    return model(tensor.to(device))


def build_inference_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    image = tensor.detach().cpu().squeeze(0)
    image = ((image * 0.5) + 0.5).clamp(0.0, 1.0)
    image = transforms.ToPILImage()(image)
    return image


def image_bytes_to_tensor(image_bytes: bytes, image_size: int) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = build_inference_transform(image_size)
    return transform(image).unsqueeze(0)


def load_generators_for_inference(config: CycleGANConfig) -> tuple[torch.nn.Module, torch.nn.Module]:
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
    generator_x_to_y.eval()
    generator_y_to_x.eval()
    return generator_x_to_y, generator_y_to_x


@torch.no_grad()
def save_bidirectional_preview(
    generator_x_to_y: torch.nn.Module,
    generator_y_to_x: torch.nn.Module,
    sample_x: torch.Tensor,
    sample_y: torch.Tensor,
    device: str,
    destination: Path,
    title: str,
) -> None:
    translated_y = translate(generator_x_to_y, sample_x.unsqueeze(0), device).squeeze(0).cpu()
    translated_x = translate(generator_y_to_x, sample_y.unsqueeze(0), device).squeeze(0).cpu()
    save_translation_preview(sample_x.cpu(), translated_y, sample_y.cpu(), translated_x, destination, title)
