"""Generate controlled CWT-representation variants from canonical raw windows."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pywt
import torch
import torch.nn.functional as F

from ..paths import PROJECT_ROOT


VARIANTS: dict[str, dict[str, Any]] = {
    "no_max_norm": {
        "scales": 48,
        "image_size": 96,
        "transform": "magnitude",
        "normalize_max": False,
        "description": "48 linear magnitude scales without per-channel max normalization",
    },
    "log_magnitude": {
        "scales": 48,
        "image_size": 96,
        "transform": "log_magnitude",
        "normalize_max": True,
        "description": "48 log1p(magnitude) scales with per-channel max normalization",
    },
    "resolution_128": {
        "scales": 48,
        "image_size": 128,
        "transform": "magnitude",
        "normalize_max": True,
        "description": "48 magnitude scales resized to 128 x 128",
    },
    "scales_32": {
        "scales": 32,
        "image_size": 96,
        "transform": "magnitude",
        "normalize_max": True,
        "description": "32 geometric scales from 1 to 64",
    },
    "scales_64": {
        "scales": 64,
        "image_size": 96,
        "transform": "magnitude",
        "normalize_max": True,
        "description": "64 geometric scales from 1 to 64",
    },
    "real_imag": {
        "scales": 48,
        "image_size": 96,
        "transform": "real_imag",
        "normalize_max": True,
        "description": "48 complex CWT scales represented by real and imaginary parts",
    },
}


def _resize(values: np.ndarray, image_size: int) -> np.ndarray:
    tensor = torch.from_numpy(values.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return (
        F.interpolate(
            tensor,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze()
        .numpy()
    )


def transform_window(
    window: np.ndarray,
    *,
    scales: int,
    image_size: int,
    transform: str,
    normalize_max: bool,
) -> np.ndarray:
    scale_values = np.geomspace(1, 64, num=scales)
    channels = []
    for channel in window:
        coefficients, _ = pywt.cwt(channel, scale_values, "morl")
        if transform == "real_imag":
            pair = np.stack([coefficients.real, coefficients.imag]).astype(np.float32)
            tensor = torch.from_numpy(pair).unsqueeze(1)
            tensor = F.interpolate(
                tensor,
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            if normalize_max:
                tensor = tensor / torch.clamp(tensor.abs().amax(), min=1e-6)
            channels.extend(tensor.numpy())
            continue

        magnitude = np.abs(coefficients).astype(np.float32)
        if transform == "log_magnitude":
            magnitude = np.log1p(magnitude)
        elif transform != "magnitude":
            raise ValueError(f"Unknown CWT transform: {transform}")
        image = _resize(magnitude, image_size)
        if normalize_max:
            image = image / max(float(np.max(image)), 1e-6)
        channels.append(image)
    return np.stack(channels).astype(np.float16)


def _process_chunk(
    raw_path: str,
    start: int,
    stop: int,
    scales: int,
    image_size: int,
    transform: str,
    normalize_max: bool,
) -> np.ndarray:
    with np.load(raw_path, mmap_mode="r") as data:
        raw = data["raw"][start:stop]
    return np.stack(
        [
            transform_window(
                window,
                scales=scales,
                image_size=image_size,
                transform=transform,
                normalize_max=normalize_max,
            )
            for window in raw
        ]
    )


def generate_variant(
    raw_cache: Path,
    output_path: Path,
    *,
    variant: str,
    workers: int,
) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown CWT variant: {variant}")
    settings = VARIANTS[variant]
    with np.load(raw_cache, mmap_mode="r") as data:
        count = len(data["raw"])
        labels = data["labels"]
        loads = data["loads"]
        record_ids = data["record_ids"]
        starts = data["starts"]
    channels = 6 if settings["transform"] == "real_imag" else 3
    shape = (
        count,
        channels,
        int(settings["image_size"]),
        int(settings["image_size"]),
    )
    chunks = [
        (start, min(start + 64, count))
        for start in range(0, count, 64)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = np.empty(shape, dtype=np.float16)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _process_chunk,
                str(raw_cache),
                start,
                stop,
                int(settings["scales"]),
                int(settings["image_size"]),
                str(settings["transform"]),
                bool(settings["normalize_max"]),
            )
            for start, stop in chunks
        ]
        for index, ((start, stop), future) in enumerate(
            zip(chunks, futures, strict=True),
            start=1,
        ):
            result[start:stop] = future.result()
            if index % 10 == 0 or stop == count:
                print(f"{variant}: {stop}/{count} windows", flush=True)
    np.savez_compressed(
        output_path,
        cwt=result,
        labels=labels,
        loads=loads,
        record_ids=record_ids,
        starts=starts,
    )
    metadata = {
        "variant": variant,
        **settings,
        "raw_source": str(raw_cache),
        "shape": list(shape),
        "wavelet": "morl",
        "scale_range": [1, 64],
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-cache",
        type=Path,
        default=PROJECT_ROOT / "data" / "public" / "paderborn" / "pu_4class_w8192.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "public" / "paderborn" / "cwt_sensitivity",
    )
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help="Comma-separated variant names.",
    )
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    for variant in variants:
        output_path = args.output_dir / f"{variant}.npz"
        if output_path.is_file():
            print(f"Reusing {output_path}", flush=True)
            continue
        generate_variant(
            args.raw_cache,
            output_path,
            variant=variant,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()
