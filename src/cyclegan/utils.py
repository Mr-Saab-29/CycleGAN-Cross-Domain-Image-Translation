from __future__ import annotations

import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def denormalize(images: torch.Tensor, mean: float = 0.5, std: float = 0.5) -> torch.Tensor:
    return (images * std) + mean


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    image = denormalize(tensor.detach().cpu()).clamp(0.0, 1.0)
    if image.ndim == 4:
        image = image[0]
    return np.transpose(image.numpy(), (1, 2, 0))


def save_translation_preview(
    input_x: torch.Tensor,
    translated_y: torch.Tensor,
    input_y: torch.Tensor,
    translated_x: torch.Tensor,
    destination: Path,
    title: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    images = (
        (input_x, "Input X"),
        (translated_y, "Translated Y"),
        (input_y, "Input Y"),
        (translated_x, "Translated X"),
    )
    for axis, (image, image_title) in zip(axes.flat, images):
        axis.imshow(tensor_to_image(image))
        axis.set_title(image_title)
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def save_epoch_preview_grid(
    rows: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    destination: Path,
    title: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    row_count = len(rows)
    fig: Figure
    fig, axes = plt.subplots(row_count, 4, figsize=(16, 4 * row_count))
    if row_count == 1:
        axes = np.expand_dims(axes, axis=0)

    headers = ("Input X", "Translated Y", "Input Y", "Translated X")
    for col, header in enumerate(headers):
        axes[0, col].set_title(header)

    for row_idx, images in enumerate(rows):
        for col_idx, image in enumerate(images):
            axes[row_idx, col_idx].imshow(tensor_to_image(image))
            axes[row_idx, col_idx].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def append_metrics_row(path: Path, row: dict[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
