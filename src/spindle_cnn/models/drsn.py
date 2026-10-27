"""One-dimensional deep residual shrinkage network adaptation.

The implementation follows the residual shrinkage concept of M. Zhao et al.,
"Deep residual shrinkage networks for fault diagnosis," IEEE Transactions on
Industrial Informatics, 2020. It is adapted to the three-channel Paderborn
waveform and the shared benchmark training protocol; it is not claimed to be a
line-for-line reproduction of the authors' original training code.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseModel


def _soft_threshold(features: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    thresholds = thresholds.clamp_min(0)
    return (
        torch.sign(features)
        * torch.relu(torch.abs(features) - thresholds)
    )


class ResidualShrinkageBlock1D(nn.Module):
    """Adaptive soft-threshold residual block for a waveform feature map."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        reduction: int = 8,
    ) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
        )
        hidden = max(out_channels // reduction, 4)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(out_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_channels),
            nn.Sigmoid(),
        )
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.skip(inputs)
        features = self.body(inputs)
        scale = self.attention(features).unsqueeze(-1)
        average_magnitude = features.abs().mean(dim=-1, keepdim=True)
        features = _soft_threshold(features, scale * average_magnitude)
        return self.activation(features + residual)


class DRSN1D(BaseModel):
    """Three-channel, four-stage DRSN adaptation for benchmark comparison."""

    model_name = "drsn"
    input_modalities = ("raw",)

    def __init__(self, num_classes: int = 4, raw_channels: int = 3) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            ResidualShrinkageBlock1D(raw_channels, 32),
            ResidualShrinkageBlock1D(32, 64, stride=2),
            ResidualShrinkageBlock1D(64, 128, stride=2),
            ResidualShrinkageBlock1D(128, 128, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        raw: torch.Tensor,
        cwt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.classifier(self.pool(self.blocks(raw.float())))
