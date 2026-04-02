from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from .config import CycleGANConfig


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir: Path, transform: transforms.Compose | None = None) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = sorted(path for path in self.root_dir.iterdir() if path.is_file())
        self.filenames = np.array([path.name for path in self.image_paths])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.image_paths[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image


@dataclass
class DatasetBundle:
    train_x: Dataset
    train_y: Dataset
    test_x: Dataset
    test_y: Dataset
    train_loader_x: DataLoader
    train_loader_y: DataLoader
    test_loader_x: DataLoader
    test_loader_y: DataLoader


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    train_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, test_transform


def maybe_subset(dataset: Dataset, subset_size: int | None, seed: int) -> Dataset:
    if subset_size is None or subset_size >= len(dataset):
        return dataset
    generator = random.Random(seed)
    indices = generator.sample(range(len(dataset)), subset_size)
    return Subset(dataset, indices)


def load_datasets(config: CycleGANConfig) -> DatasetBundle:
    train_transform, test_transform = build_transforms(config.image_size)
    dataset_dir = config.dataset_dir

    train_x = ImageFolderDataset(dataset_dir / config.train_split_x, train_transform)
    train_y = ImageFolderDataset(dataset_dir / config.train_split_y, train_transform)
    test_x = ImageFolderDataset(dataset_dir / config.test_split_x, test_transform)
    test_y = ImageFolderDataset(dataset_dir / config.test_split_y, test_transform)

    train_x = maybe_subset(train_x, config.subset_size, config.seed)
    train_y = maybe_subset(train_y, config.subset_size, config.seed)

    return DatasetBundle(
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        train_loader_x=DataLoader(train_x, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, drop_last=True),
        train_loader_y=DataLoader(train_y, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, drop_last=True),
        test_loader_x=DataLoader(test_x, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers),
        test_loader_y=DataLoader(test_y, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers),
    )
