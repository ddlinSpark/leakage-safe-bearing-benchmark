"""Compact models used by the public CWRU benchmark."""

from __future__ import annotations

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.fc(self.pool(x).flatten(1)).view(x.shape[0], -1, 1, 1)
        return x * weights


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            SEBlock(out_channels),
        )
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(x) + self.skip(x))


class ImageEncoder(nn.Module):
    def __init__(self, in_channels: int, feature_dim: int = 192) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(32, 64),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, feature_dim, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.blocks(self.stem(x.float()))).flatten(1)


class RawCNN(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 2) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=25, stride=2, padding=12),
            nn.BatchNorm1d(64),
            nn.SiLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.SiLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, raw: torch.Tensor, cwt: torch.Tensor | None = None) -> torch.Tensor:
        return self.classifier(self.encoder(raw.float()))


class CWT2DCNN(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 2) -> None:
        super().__init__()
        self.encoder = ImageEncoder(in_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(192, num_classes),
        )

    def forward(self, raw: torch.Tensor, cwt: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(cwt))


class CWTGRU(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        image_channels: int = 2,
        raw_channels: int = 2,
        gru_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(image_channels)
        self.gru = nn.GRU(
            input_size=raw_channels,
            hidden_size=gru_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.15,
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(192 + 2 * gru_hidden),
            nn.Linear(192 + 2 * gru_hidden, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(128, num_classes),
        )

    def forward(self, raw: torch.Tensor, cwt: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(cwt)
        sequence = raw.float().transpose(1, 2)
        _, hidden = self.gru(sequence)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fusion(torch.cat([image_features, hidden], dim=1))


def build_model(
    name: str,
    num_classes: int = 10,
    raw_channels: int = 2,
    image_channels: int = 2,
) -> nn.Module:
    if name == "raw_cnn":
        return RawCNN(num_classes=num_classes, in_channels=raw_channels)
    if name == "cwt_cnn":
        return CWT2DCNN(num_classes=num_classes, in_channels=image_channels)
    if name == "cwt_gru":
        return CWTGRU(
            num_classes=num_classes,
            raw_channels=raw_channels,
            image_channels=image_channels,
        )
    raise ValueError(f"Unknown model: {name}")
