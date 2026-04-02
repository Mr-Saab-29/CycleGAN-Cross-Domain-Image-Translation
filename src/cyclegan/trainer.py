from __future__ import annotations

import random
import time

import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from .checkpoints import load_checkpoint, save_checkpoint
from .config import CycleGANConfig, config_to_dict
from .data import load_datasets
from .evaluation import evaluate_generators
from .losses import cycle_consistency_loss, discriminator_loss, generator_loss, identity_loss
from .models import Discriminator, GeneratorResNet, weights_init_normal
from .tracking import ExperimentTracker
from .utils import append_metrics_row, save_epoch_preview_grid, set_seed


class ReplayBuffer:
    def __init__(self, max_size: int) -> None:
        if max_size <= 0:
            raise ValueError("Replay buffer size must be positive.")
        self.max_size = max_size
        self.buffer: list[torch.Tensor] = []

    def push_and_pop(self, batch: torch.Tensor) -> torch.Tensor:
        returned: list[torch.Tensor] = []
        for image in batch.detach():
            image = image.unsqueeze(0)
            if len(self.buffer) < self.max_size:
                self.buffer.append(image)
                returned.append(image)
                continue

            if random.random() > 0.5:
                index = random.randrange(len(self.buffer))
                returned.append(self.buffer[index].clone())
                self.buffer[index] = image
            else:
                returned.append(image)
        return torch.cat(returned, dim=0)


class CycleGANTrainer:
    def __init__(self, config: CycleGANConfig) -> None:
        self.config = config
        self.config.ensure_dirs()
        set_seed(config.seed)
        self.device = torch.device(config.device)
        self.datasets = load_datasets(config)
        self.fake_x_buffer = ReplayBuffer(config.replay_buffer_size)
        self.fake_y_buffer = ReplayBuffer(config.replay_buffer_size)
        self.tracker = ExperimentTracker(
            enabled=config.tracking_enabled,
            tracking_uri=config.tracking_uri,
            experiment_name=config.tracking_experiment,
            run_name=config.run_name,
        )

        self.generator_x_to_y = GeneratorResNet(
            input_channels=config.channels,
            output_channels=config.channels,
            filters=config.generator_filters,
            n_blocks=config.n_residual_blocks,
        ).to(self.device)
        self.generator_y_to_x = GeneratorResNet(
            input_channels=config.channels,
            output_channels=config.channels,
            filters=config.generator_filters,
            n_blocks=config.n_residual_blocks,
        ).to(self.device)
        self.discriminator_x = Discriminator(config.channels, config.discriminator_filters).to(self.device)
        self.discriminator_y = Discriminator(config.channels, config.discriminator_filters).to(self.device)

        self.generators = {"x_to_y": self.generator_x_to_y, "y_to_x": self.generator_y_to_x}
        self.discriminators = {"x": self.discriminator_x, "y": self.discriminator_y}

        self.generator_x_to_y.apply(weights_init_normal)
        self.generator_y_to_x.apply(weights_init_normal)
        self.discriminator_x.apply(weights_init_normal)
        self.discriminator_y.apply(weights_init_normal)

        betas = (config.beta1, config.beta2)
        self.optimizers = {
            "g_x_to_y": torch.optim.Adam(self.generator_x_to_y.parameters(), lr=config.lr, betas=betas),
            "g_y_to_x": torch.optim.Adam(self.generator_y_to_x.parameters(), lr=config.lr, betas=betas),
            "d_x": torch.optim.Adam(self.discriminator_x.parameters(), lr=config.lr, betas=betas),
            "d_y": torch.optim.Adam(self.discriminator_y.parameters(), lr=config.lr, betas=betas),
        }
        self.schedulers = {name: LambdaLR(optimizer, lr_lambda=self._lambda_lr) for name, optimizer in self.optimizers.items()}
        self.start_epoch = 1
        if config.resume and config.checkpoint_path.exists():
            self.start_epoch = load_checkpoint(
                config.checkpoint_path,
                generators=self.generators,
                discriminators=self.discriminators,
                optimizers=self.optimizers,
                device=config.device,
            ) + 1
        self.tracker.log_params(config_to_dict(config))

    def _lambda_lr(self, epoch: int) -> float:
        decay_start_epoch = self.config.epochs // 2
        if epoch < decay_start_epoch:
            return 1.0
        decay_length = max(self.config.epochs - decay_start_epoch, 1)
        return max(0.0, 1.0 - (epoch - decay_start_epoch) / decay_length)

    def _preview_pairs(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        count = min(self.config.sample_count, len(self.datasets.test_x), len(self.datasets.test_y))
        return [(self.datasets.test_x[index], self.datasets.test_y[index]) for index in range(count)]

    @torch.no_grad()
    def _save_epoch_preview(self, epoch: int, preview_pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        self.generator_x_to_y.eval()
        self.generator_y_to_x.eval()
        rows: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for sample_x, sample_y in preview_pairs:
            translated_y = self.generator_x_to_y(sample_x.unsqueeze(0).to(self.device)).squeeze(0).cpu()
            translated_x = self.generator_y_to_x(sample_y.unsqueeze(0).to(self.device)).squeeze(0).cpu()
            rows.append((sample_x.cpu(), translated_y, sample_y.cpu(), translated_x))
        save_epoch_preview_grid(
            rows,
            self.config.train_sample_dir / f"epoch_{epoch:03d}.png",
            title=f"{self.config.run_name} - epoch {epoch}",
        )

    def train(self) -> None:
        preview_pairs = self._preview_pairs()

        try:
            for epoch in range(self.start_epoch, self.config.epochs + 1):
                epoch_start = time.time()
                metrics = self._train_one_epoch(epoch)
                append_metrics_row(self.config.metrics_path, metrics)
                self.tracker.log_metrics({k: float(v) for k, v in metrics.items() if k != "epoch"}, step=epoch)
                self._save_epoch_preview(epoch, preview_pairs)

                if epoch % self.config.save_every_n_epochs == 0 or epoch == self.config.epochs:
                    save_checkpoint(
                        self.config.checkpoint_path,
                        epoch=epoch,
                        generators=self.generators,
                        discriminators=self.discriminators,
                        optimizers=self.optimizers,
                    )
                    self.tracker.log_artifact(self.config.checkpoint_path, artifact_path="checkpoints")

                for scheduler in self.schedulers.values():
                    scheduler.step()

                duration = time.time() - epoch_start
                print(
                    f"Epoch {epoch}/{self.config.epochs} | "
                    f"G_XtoY={metrics['g_x_to_y']:.4f} | G_YtoX={metrics['g_y_to_x']:.4f} | "
                    f"Dx={metrics['d_x']:.4f} | Dy={metrics['d_y']:.4f} | "
                    f"cycle={metrics['cycle']:.4f} | identity={metrics['identity']:.4f} | "
                    f"{duration:.1f}s"
                )

            evaluation_metrics, preview_path = evaluate_generators(
                self.generator_x_to_y,
                self.generator_y_to_x,
                self.config,
            )
            self.tracker.log_metrics(evaluation_metrics, step=self.config.epochs)
            self.tracker.log_artifact(preview_path, artifact_path="evaluation")
            self.tracker.log_artifact(self.config.evaluation_dir / "report.json", artifact_path="evaluation")
        finally:
            self.tracker.end()

    def _train_one_epoch(self, epoch: int) -> dict[str, float | int]:
        self.generator_x_to_y.train()
        self.generator_y_to_x.train()
        self.discriminator_x.train()
        self.discriminator_y.train()

        totals = {"g_x_to_y": 0.0, "g_y_to_x": 0.0, "d_x": 0.0, "d_y": 0.0, "cycle": 0.0, "identity": 0.0}
        steps = min(len(self.datasets.train_loader_x), len(self.datasets.train_loader_y))
        progress = tqdm(zip(self.datasets.train_loader_x, self.datasets.train_loader_y), total=steps, desc=f"Epoch {epoch}", leave=False)

        for real_x, real_y in progress:
            real_x = real_x.to(self.device)
            real_y = real_y.to(self.device)

            fake_y = self.generator_x_to_y(real_x)
            fake_x = self.generator_y_to_x(real_y)
            buffered_fake_x = self.fake_x_buffer.push_and_pop(fake_x)
            buffered_fake_y = self.fake_y_buffer.push_and_pop(fake_y)

            dx_loss = discriminator_loss(
                self.discriminator_x(real_x),
                self.discriminator_x(buffered_fake_x),
                self.config.soft_real_label_range,
                self.config.soft_fake_label_range,
            )
            dy_loss = discriminator_loss(
                self.discriminator_y(real_y),
                self.discriminator_y(buffered_fake_y),
                self.config.soft_real_label_range,
                self.config.soft_fake_label_range,
            )

            self.optimizers["d_x"].zero_grad()
            dx_loss.backward()
            self.optimizers["d_x"].step()

            self.optimizers["d_y"].zero_grad()
            dy_loss.backward()
            self.optimizers["d_y"].step()

            fake_y = self.generator_x_to_y(real_x)
            fake_x = self.generator_y_to_x(real_y)
            g_x_to_y_loss = generator_loss(self.discriminator_y(fake_y), self.config.soft_real_label_range)
            g_y_to_x_loss = generator_loss(self.discriminator_x(fake_x), self.config.soft_real_label_range)

            cycle_loss = cycle_consistency_loss(real_x, self.generator_y_to_x(fake_y), self.config.lambda_cycle) + cycle_consistency_loss(
                real_y, self.generator_x_to_y(fake_x), self.config.lambda_cycle
            )
            identity_component = identity_loss(real_x, self.generator_y_to_x(real_x), self.config.lambda_identity) + identity_loss(
                real_y, self.generator_x_to_y(real_y), self.config.lambda_identity
            )

            total_generator_loss = g_x_to_y_loss + g_y_to_x_loss + cycle_loss + identity_component
            self.optimizers["g_x_to_y"].zero_grad()
            self.optimizers["g_y_to_x"].zero_grad()
            total_generator_loss.backward()
            self.optimizers["g_x_to_y"].step()
            self.optimizers["g_y_to_x"].step()

            totals["g_x_to_y"] += g_x_to_y_loss.item()
            totals["g_y_to_x"] += g_y_to_x_loss.item()
            totals["d_x"] += dx_loss.item()
            totals["d_y"] += dy_loss.item()
            totals["cycle"] += cycle_loss.item()
            totals["identity"] += identity_component.item()
            progress.set_postfix(
                g=(g_x_to_y_loss.item() + g_y_to_x_loss.item()) / 2.0,
                d=(dx_loss.item() + dy_loss.item()) / 2.0,
            )

        return {"epoch": epoch, **{name: value / steps for name, value in totals.items()}}
