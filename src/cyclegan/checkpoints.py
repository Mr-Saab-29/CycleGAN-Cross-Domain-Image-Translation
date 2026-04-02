from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(
    path: Path,
    epoch: int,
    generators: dict[str, torch.nn.Module],
    discriminators: dict[str, torch.nn.Module],
    optimizers: dict[str, torch.optim.Optimizer],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "generators": {name: model.state_dict() for name, model in generators.items()},
        "discriminators": {name: model.state_dict() for name, model in discriminators.items()},
        "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
    }
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    generators: dict[str, torch.nn.Module],
    discriminators: dict[str, torch.nn.Module],
    optimizers: dict[str, torch.optim.Optimizer] | None = None,
    device: str = "cpu",
) -> int:
    checkpoint = torch.load(path, map_location=device)
    for name, model in generators.items():
        model.load_state_dict(checkpoint["generators"][name])
    for name, model in discriminators.items():
        model.load_state_dict(checkpoint["discriminators"][name])
    if optimizers is not None:
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(checkpoint["optimizers"][name])
    return int(checkpoint["epoch"])
