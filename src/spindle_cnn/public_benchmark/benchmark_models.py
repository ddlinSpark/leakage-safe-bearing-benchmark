"""Measure parameter count, FLOPs, and batch latency on the same device."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from ..paths import OUTPUTS_DIR
from .models import build_model


MODEL_LABELS = {
    "raw_cnn": "Raw 1D-CNN",
    "gru_only": "GRU only",
    "cwt_cnn": "CWT 2D-CNN",
    "cwt_gru_no_se": "CWT-GRU without SE",
    "cwt_gru": "CWT-GRU",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比较模型参数量、FLOPs 与批次延迟")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=8192)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--raw-channels", type=int, default=3)
    parser.add_argument("--image-channels", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUTS_DIR / "public_paderborn_final" / "model_complexity.csv",
    )
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def estimate_bidirectional_gru_flops(
    batch_size: int,
    sequence_length: int,
    input_size: int,
    hidden_size: int,
    layers: int = 2,
) -> int:
    """Estimate GRU matrix-multiply FLOPs when the profiler omits recurrent ops."""

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
def measure_inference(
    model: nn.Module,
    raw: torch.Tensor,
    cwt: torch.Tensor,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> float:
    model.eval()
    for _ in range(warmup):
        model(raw, cwt)
    synchronize(device)
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            model(raw, cwt)
        end.record()
        synchronize(device)
        return float(start.elapsed_time(end) / repeats)
    start_time = time.perf_counter()
    for _ in range(repeats):
        model(raw, cwt)
    return float((time.perf_counter() - start_time) * 1000 / repeats)


def measure_training_step(
    model: nn.Module,
    raw: torch.Tensor,
    cwt: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> float:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(raw, cwt), labels)
        loss.backward()
        optimizer.step()
    synchronize(device)
    timings = []
    for _ in range(repeats):
        optimizer.zero_grad(set_to_none=True)
        synchronize(device)
        start_time = time.perf_counter()
        loss = criterion(model(raw, cwt), labels)
        loss.backward()
        optimizer.step()
        synchronize(device)
        timings.append((time.perf_counter() - start_time) * 1000)
    return float(sum(timings) / len(timings))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = torch.randn(args.batch_size, args.raw_channels, args.window_size, device=device)
    cwt = torch.randn(
        args.batch_size,
        args.image_channels,
        args.image_size,
        args.image_size,
        device=device,
    )
    labels = torch.randint(0, 4, (args.batch_size,), device=device)
    rows = []

    for name, label in MODEL_LABELS.items():
        model = build_model(
            name,
            num_classes=4,
            raw_channels=args.raw_channels,
            image_channels=args.image_channels,
        ).to(device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        with FlopCounterMode(display=False) as counter:
            model(raw, cwt)
        flops = int(counter.get_total_flops())
        if name in {"gru_only", "cwt_gru_no_se", "cwt_gru"}:
            flops += estimate_bidirectional_gru_flops(
                batch_size=args.batch_size,
                sequence_length=args.window_size,
                input_size=args.raw_channels,
                hidden_size=64,
                layers=2,
            )
        inference_ms = measure_inference(
            model,
            raw,
            cwt,
            device,
            args.warmup,
            args.repeats,
        )
        training_ms = measure_training_step(
            model,
            raw,
            cwt,
            labels,
            device,
            max(2, args.warmup // 2),
            max(5, args.repeats // 3),
        )
        rows.append(
            {
                "model": name,
                "label": label,
                "parameters": parameter_count,
                "gflops_batch": flops / 1e9,
                "gflops_sample": flops / 1e9 / args.batch_size,
                "inference_ms_batch": inference_ms,
                "inference_samples_per_s": 1000 * args.batch_size / inference_ms,
                "training_ms_batch": training_ms,
                "training_samples_per_s": 1000 * args.batch_size / training_ms,
            }
        )

    pd.DataFrame(rows).to_csv(args.output, index=False)
    metadata = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "torch": torch.__version__,
        "batch_size": args.batch_size,
        "window_size": args.window_size,
        "image_size": args.image_size,
        "raw_channels": args.raw_channels,
        "image_channels": args.image_channels,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "flop_note": (
            "FlopCounterMode totals plus analytical bidirectional GRU matrix "
            "multiplication estimates; elementwise nonlinearities are excluded."
        ),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
