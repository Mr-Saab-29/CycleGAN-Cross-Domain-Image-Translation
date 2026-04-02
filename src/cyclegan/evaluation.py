from __future__ import annotations

import json
from pathlib import Path

import torch

from .config import CycleGANConfig
from .data import load_datasets
from .utils import save_epoch_preview_grid


@torch.no_grad()
def evaluate_generators(
    generator_x_to_y: torch.nn.Module,
    generator_y_to_x: torch.nn.Module,
    config: CycleGANConfig,
    max_batches: int = 20,
) -> tuple[dict[str, float], Path]:
    device = torch.device(config.device)
    datasets = load_datasets(config)
    generator_x_to_y.eval()
    generator_y_to_x.eval()

    cycle_x_total = 0.0
    cycle_y_total = 0.0
    identity_x_total = 0.0
    identity_y_total = 0.0
    batches = 0
    preview_rows: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

    for real_x, real_y in zip(datasets.test_loader_x, datasets.test_loader_y):
        real_x = real_x.to(device)
        real_y = real_y.to(device)
        fake_y = generator_x_to_y(real_x)
        fake_x = generator_y_to_x(real_y)
        cycled_x = generator_y_to_x(fake_y)
        cycled_y = generator_x_to_y(fake_x)
        identity_x = generator_y_to_x(real_x)
        identity_y = generator_x_to_y(real_y)

        cycle_x_total += torch.mean(torch.abs(real_x - cycled_x)).item()
        cycle_y_total += torch.mean(torch.abs(real_y - cycled_y)).item()
        identity_x_total += torch.mean(torch.abs(real_x - identity_x)).item()
        identity_y_total += torch.mean(torch.abs(real_y - identity_y)).item()

        if len(preview_rows) < config.sample_count:
            preview_rows.append((real_x[0].cpu(), fake_y[0].cpu(), real_y[0].cpu(), fake_x[0].cpu()))

        batches += 1
        if batches >= max_batches:
            break

    if batches == 0:
        raise RuntimeError("No evaluation batches available. Check the dataset paths.")

    metrics = {
        "eval_cycle_x_l1": cycle_x_total / batches,
        "eval_cycle_y_l1": cycle_y_total / batches,
        "eval_identity_x_l1": identity_x_total / batches,
        "eval_identity_y_l1": identity_y_total / batches,
    }
    preview_path = config.evaluation_dir / "evaluation_preview.png"
    save_epoch_preview_grid(preview_rows, preview_path, title=f"{config.run_name} - evaluation")
    report_path = config.evaluation_dir / "report.json"
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics, preview_path
