from __future__ import annotations

import torch


def discriminator_loss(
    real_pred: torch.Tensor,
    fake_pred: torch.Tensor,
    soft_real_label_range: tuple[float, float],
    soft_fake_label_range: tuple[float, float],
) -> torch.Tensor:
    real_labels = torch.empty_like(real_pred).uniform_(*soft_real_label_range)
    fake_labels = torch.empty_like(fake_pred).uniform_(*soft_fake_label_range)
    real_loss = 0.5 * torch.mean((real_pred - real_labels) ** 2)
    fake_loss = 0.5 * torch.mean((fake_pred - fake_labels) ** 2)
    return real_loss + fake_loss


def generator_loss(fake_pred: torch.Tensor, soft_real_label_range: tuple[float, float]) -> torch.Tensor:
    real_labels = torch.empty_like(fake_pred).uniform_(*soft_real_label_range)
    return torch.mean((fake_pred - real_labels) ** 2)


def cycle_consistency_loss(real_image: torch.Tensor, cycled_image: torch.Tensor, lambda_cycle: float) -> torch.Tensor:
    return lambda_cycle * torch.mean(torch.abs(real_image - cycled_image))


def identity_loss(real_image: torch.Tensor, generated_image: torch.Tensor, lambda_identity: float) -> torch.Tensor:
    return lambda_identity * torch.mean(torch.abs(real_image - generated_image))
