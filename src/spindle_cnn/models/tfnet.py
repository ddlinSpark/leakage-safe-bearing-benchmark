"""TFNet comparison architecture adapted from the authors' public code.

Reference:
Q. Chen, X. Dong, G. Tu, et al., "TFN: An interpretable neural network with
time-frequency transform embedded for intelligent fault diagnosis," Mechanical
Systems and Signal Processing, vol. 207, Art. no. 110952, 2024.

The published comparison-model code was used as the structural source. This
version changes the input channel count to three and output count to four for
the common Paderborn protocol and removes import-time global seed side effects.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel


class TFConv1d(nn.Module):
    """Sinusoidal time-frequency convolution with learnable frequencies."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        minimum_frequency: float = 0.03,
        maximum_frequency: float = 0.45,
        sigma: float = 0.52,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.minimum_frequency = minimum_frequency
        self.maximum_frequency = maximum_frequency
        self.sigma = sigma
        initial = torch.linspace(
            minimum_frequency,
            maximum_frequency,
            out_channels * in_channels,
        ).reshape(out_channels, in_channels)
        self.frequencies = nn.Parameter(initial)

    def _kernels(self) -> torch.Tensor:
        frequencies = self.frequencies.clamp(
            self.minimum_frequency,
            self.maximum_frequency,
        )
        positions = torch.arange(
            -(self.kernel_size // 2),
            self.kernel_size - self.kernel_size // 2,
            device=frequencies.device,
            dtype=frequencies.dtype,
        )
        envelope = torch.exp(
            -(positions / (self.kernel_size // 2)).pow(2)
            / self.sigma**2
            / 2
        )
        angle = 2 * math.pi * frequencies.unsqueeze(-1) * positions
        real = envelope * torch.cos(angle)
        imaginary = envelope * torch.sin(angle)
        return torch.stack((real, imaginary), dim=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        kernels = self._kernels()
        real = F.conv1d(
            inputs,
            kernels[0],
            stride=self.stride,
            padding=self.padding,
        )
        imaginary = F.conv1d(
            inputs,
            kernels[1],
            stride=self.stride,
            padding=self.padding,
        )
        return torch.sqrt(real.square() + imaginary.square() + 1e-8)


class TFNet(BaseModel):
    """Authors' TFN comparison backbone adapted to this benchmark."""

    model_name = "tfnet"
    input_modalities = ("raw",)

    def __init__(
        self,
        num_classes: int = 4,
        in_channels: int = 3,
        mid_channels: int = 16,
        kernel_size: int = 15,
    ) -> None:
        super().__init__()
        function_kernel_size = 2 * mid_channels - 1
        self.function_conv = TFConv1d(
            in_channels,
            mid_channels,
            function_kernel_size,
            padding=function_kernel_size // 2,
        )
        self.layer1 = nn.Sequential(
            nn.Conv1d(mid_channels, 16, kernel_size=kernel_size),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(64, num_classes)

    @staticmethod
    def _adaptive_max_4(features: torch.Tensor) -> torch.Tensor:
        """Equivalent to AdaptiveMaxPool1d(4) with deterministic reductions."""

        segments = torch.tensor_split(features, 4, dim=-1)
        return torch.stack([segment.amax(dim=-1) for segment in segments], dim=-1)

    def forward(
        self,
        raw: torch.Tensor,
        cwt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self.function_conv(raw.float())
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        features = self._adaptive_max_4(self.layer4(features))
        return self.classifier(self.projection(features))
