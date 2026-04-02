from __future__ import annotations

import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(channels),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, filters: int = 64, n_blocks: int = 9) -> None:
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, filters, kernel_size=7),
            nn.InstanceNorm2d(filters),
            nn.ReLU(inplace=True),
        ]

        in_features = filters
        for _ in range(2):
            out_features = in_features * 2
            layers.extend(
                [
                    nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True),
                ]
            )
            in_features = out_features

        for _ in range(n_blocks):
            layers.append(ResidualBlock(in_features))

        for _ in range(2):
            out_features = in_features // 2
            layers.extend(
                [
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=1),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True),
                ]
            )
            in_features = out_features

        layers.extend(
            [
                nn.ReflectionPad2d(3),
                nn.Conv2d(in_features, output_channels, kernel_size=7),
                nn.Tanh(),
            ]
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_channels: int, filters: int = 64) -> None:
        super().__init__()

        def block(in_filters: int, out_filters: int, normalize: bool = True) -> list[nn.Module]:
            layers: list[nn.Module] = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(input_channels, filters, normalize=False),
            *block(filters, filters * 2),
            *block(filters * 2, filters * 4),
            *block(filters * 4, filters * 8),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(filters * 8, 1, kernel_size=4, padding=1),
        )

    def forward(self, x):
        return self.model(x)


def weights_init_normal(module: nn.Module) -> None:
    classname = module.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(module.weight.data, mean=0.0, std=0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif "InstanceNorm" in classname:
        if getattr(module, "weight", None) is not None:
            nn.init.normal_(module.weight.data, mean=1.0, std=0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)
