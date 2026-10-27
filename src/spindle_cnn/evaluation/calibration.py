"""Temperature-scaling utilities."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def probabilities_from_logits(
    logits: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    tensor = torch.from_numpy(np.asarray(logits, dtype=np.float64)) / temperature
    return F.softmax(tensor, dim=1).numpy()


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    minimum: float = 0.05,
    maximum: float = 20.0,
) -> float:
    """Fit one positive temperature by minimizing validation NLL."""

    logits_tensor = torch.from_numpy(np.asarray(logits, dtype=np.float32))
    labels_tensor = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    log_minimum = math.log(minimum)
    log_maximum = math.log(maximum)
    raw_parameter = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
    optimizer = torch.optim.LBFGS(
        [raw_parameter],
        lr=0.2,
        max_iter=100,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        temperature = torch.exp(
            torch.clamp(raw_parameter, min=log_minimum, max=log_maximum)
        )
        loss = F.cross_entropy(logits_tensor / temperature, labels_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(
        torch.exp(torch.clamp(raw_parameter.detach(), min=log_minimum, max=log_maximum))
    )
    return min(max(temperature, minimum), maximum)
