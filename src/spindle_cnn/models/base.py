"""Common model interface used by all benchmark architectures."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """Minimal contract shared by trainable benchmark models."""

    model_name: str = "base"
    input_modalities: tuple[str, ...] = ()

    @abstractmethod
    def forward(
        self,
        raw: torch.Tensor,
        cwt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return class logits for a batch of waveforms and optional CWT maps."""

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
