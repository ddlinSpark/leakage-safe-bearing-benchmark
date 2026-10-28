"""Time-frequency and waveform fusion model."""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseModel
from .layers import ImageEncoder


class CWTGRU(BaseModel):
    model_name = "cwt_gru"
    input_modalities = ("raw", "cwt")

    def __init__(
        self,
        num_classes: int = 10,
        image_channels: int = 2,
        raw_channels: int = 2,
        gru_hidden: int = 64,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(image_channels, use_se=use_se)
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
        return self.fusion(self.fused_features(raw, cwt))

    def embedding(self, raw: torch.Tensor, cwt: torch.Tensor) -> torch.Tensor:
        return self.fusion[:-1](self.fused_features(raw, cwt))

    def fused_features(self, raw: torch.Tensor, cwt: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(cwt)
        sequence = raw.float().transpose(1, 2)
        _, hidden = self.gru(sequence)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return torch.cat([image_features, hidden], dim=1)
