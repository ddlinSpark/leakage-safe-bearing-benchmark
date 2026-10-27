"""Bidirectional GRU baseline operating on raw waveforms."""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseModel


class RawGRU(BaseModel):
    """Bidirectional recurrent baseline without time-frequency images."""

    model_name = "gru_only"
    input_modalities = ("raw",)

    def __init__(
        self,
        num_classes: int = 10,
        raw_channels: int = 2,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=raw_channels,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.15,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * hidden_size),
            nn.Linear(2 * hidden_size, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, raw: torch.Tensor, cwt: torch.Tensor | None = None) -> torch.Tensor:
        _, hidden = self.gru(raw.float().transpose(1, 2))
        return self.classifier(torch.cat([hidden[-2], hidden[-1]], dim=1))
