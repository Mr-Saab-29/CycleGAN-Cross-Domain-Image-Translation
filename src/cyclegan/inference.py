from __future__ import annotations

from pathlib import Path

import torch

from .utils import save_translation_preview


@torch.no_grad()
def translate(model: torch.nn.Module, tensor: torch.Tensor, device: str) -> torch.Tensor:
    model.eval()
    return model(tensor.to(device))


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
