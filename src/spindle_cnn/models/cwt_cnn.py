"""CWT image CNN baseline."""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseModel
from .layers import ImageEncoder


class CWT2DCNN(BaseModel):
    model_name = "cwt_cnn"
    input_modalities = ("cwt",)

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 2,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = ImageEncoder(in_channels, use_se=use_se)
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(192, num_classes),
        )

    def forward(self, raw: torch.Tensor, cwt: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(cwt))
