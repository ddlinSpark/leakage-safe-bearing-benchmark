"""Raw waveform one-dimensional CNN baseline."""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseModel


class RawCNN(BaseModel):
    model_name = "raw_cnn"
    input_modalities = ("raw",)

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
