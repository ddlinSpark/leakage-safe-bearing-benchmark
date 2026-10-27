"""Parameter, FLOP, and latency utilities for benchmark models."""

from __future__ import annotations

import time

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def estimate_bidirectional_gru_flops(
    batch_size: int,
    sequence_length: int,
    input_size: int,
    hidden_size: int,
    layers: int = 2,
) -> int:
    """Estimate GRU matrix-multiply FLOPs omitted by PyTorch's profiler."""

    total_macs = 0
    layer_input = input_size
    for _ in range(layers):
        per_direction = sequence_length * 3 * (
            layer_input * hidden_size + hidden_size * hidden_size
        )
        total_macs += 2 * per_direction
        layer_input = 2 * hidden_size
    return 2 * batch_size * total_macs


@torch.no_grad()
def estimate_model_gflops(
    model: nn.Module,
    raw: torch.Tensor,
    cwt: torch.Tensor,
    *,
    model_key: str,
) -> float:
    model.eval()
    with FlopCounterMode(display=False) as counter:
        model(raw, cwt)
    flops = int(counter.get_total_flops())
    if model_key in {"gru_only", "cwt_gru", "cwt_gru_no_se"}:
        flops += estimate_bidirectional_gru_flops(
            batch_size=raw.shape[0],
            sequence_length=raw.shape[-1],
            input_size=raw.shape[1],
            hidden_size=64,
            layers=2,
        )
    return float(flops / 1e9 / raw.shape[0])


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_batch_inference_ms(
    model: nn.Module,
    raw: torch.Tensor,
    cwt: torch.Tensor,
    *,
    warmup: int = 2,
    repeats: int = 5,
) -> float:
    model.eval()
    device = raw.device
    with torch.inference_mode():
        for _ in range(warmup):
            model(raw, cwt)
        synchronize(device)
        start = time.perf_counter()
        for _ in range(repeats):
            model(raw, cwt)
        synchronize(device)
    return float((time.perf_counter() - start) * 1000 / repeats)
