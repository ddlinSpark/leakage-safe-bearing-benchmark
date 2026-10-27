"""CWRU 12 kHz drive-end benchmark data preparation.

The official CWRU files use numeric names. This module defines one explicit
10-class label map and uses non-overlapping windows so labels cannot leak
between adjacent training and evaluation samples.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pywt
import torch
import torch.nn.functional as F
from scipy.io import loadmat

from ..paths import DATA_DIR


CLASS_NAMES = [
    "Normal",
    "Ball_007",
    "Ball_014",
    "Ball_021",
    "InnerRace_007",
    "InnerRace_014",
    "InnerRace_021",
    "OuterRace_007",
    "OuterRace_014",
    "OuterRace_021",
]

# Rows are classes and columns correspond to motor loads 0, 1, 2, and 3 hp.
# IDs follow the official CWRU 12 kHz drive-end table.
CWRU_RECORD_IDS = np.array(
    [
        [97, 98, 99, 100],
        [118, 119, 120, 121],
        [185, 186, 187, 188],
        [222, 223, 224, 225],
        [105, 106, 107, 108],
        [169, 170, 171, 172],
        [209, 210, 211, 212],
        [130, 131, 132, 133],
        [197, 198, 199, 200],
        [234, 235, 236, 237],
    ],
    dtype=np.int64,
)

LOAD_TO_RPM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}
SAMPLING_RATE_HZ = 12_000
DEFAULT_WAVELETS = ("morl",)


@dataclass(frozen=True)
class CWRUDataset:
    raw: np.ndarray
    cwt: np.ndarray
    labels: np.ndarray
    loads: np.ndarray
    record_ids: np.ndarray
    starts: np.ndarray


def _signal_from_mat(data: dict, record_id: int, channel: str) -> np.ndarray:
    candidates = [
        key
        for key in data
        if not key.startswith("__")
        and key.endswith(f"_{channel}_time")
        and str(record_id) in key
    ]
    if not candidates:
        raise KeyError(f"Record {record_id} has no {channel} channel")
    return np.asarray(data[candidates[0]], dtype=np.float32).reshape(-1)


def _standardize_window(window: np.ndarray) -> np.ndarray:
    mean = window.mean(axis=1, keepdims=True)
    std = window.std(axis=1, keepdims=True)
    return ((window - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def _cwt_image(
    raw_window: np.ndarray,
    wavelets: tuple[str, ...],
    scales: int,
    image_size: int,
) -> np.ndarray:
    scale_values = np.geomspace(1, 64, num=scales)
    channels: list[np.ndarray] = []
    for channel in raw_window:
        for wavelet in wavelets:
            coefficients, _ = pywt.cwt(channel, scale_values, wavelet)
            magnitude = np.abs(coefficients).astype(np.float32)
            tensor = torch.from_numpy(magnitude).unsqueeze(0).unsqueeze(0)
            tensor = F.interpolate(
                tensor,
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            tensor = tensor / torch.clamp(tensor.amax(), min=1e-6)
            channels.append(tensor.numpy())
    return np.stack(channels, axis=0).astype(np.float16)


def verify_cwru_files(data_root: Path) -> dict[int, list[str]]:
    manifest: dict[int, list[str]] = {}
    expected = CWRU_RECORD_IDS.reshape(-1).tolist()
    missing = [record_id for record_id in expected if not (data_root / f"{record_id}.mat").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing CWRU files in {data_root}: {missing}")

    for record_id in expected:
        data = loadmat(data_root / f"{record_id}.mat")
        _signal_from_mat(data, record_id, "DE")
        _signal_from_mat(data, record_id, "FE")
        manifest[int(record_id)] = [
            key for key in data if not key.startswith("__")
        ]
    return manifest


def build_cwru_dataset(
    data_root: Path,
    *,
    window_size: int = 2048,
    stride: int = 2048,
    wavelets: tuple[str, ...] = DEFAULT_WAVELETS,
    scales: int = 48,
    image_size: int = 96,
) -> CWRUDataset:
    verify_cwru_files(data_root)
    raw_windows: list[np.ndarray] = []
    cwt_windows: list[np.ndarray] = []
    labels: list[int] = []
    loads: list[int] = []
    record_ids: list[int] = []
    starts: list[int] = []

    for class_id, class_records in enumerate(CWRU_RECORD_IDS):
        for load_id, record_id in enumerate(class_records):
            data = loadmat(data_root / f"{record_id}.mat")
            de = _signal_from_mat(data, int(record_id), "DE")
            fe = _signal_from_mat(data, int(record_id), "FE")
            channel_length = min(len(de), len(fe))
            signal = np.stack([de[:channel_length], fe[:channel_length]], axis=0)
            record_window_count = 0
            for start in range(0, channel_length - window_size + 1, stride):
                raw = _standardize_window(signal[:, start : start + window_size])
                raw_windows.append(raw)
                cwt_windows.append(_cwt_image(raw, wavelets, scales, image_size))
                labels.append(class_id)
                loads.append(load_id)
                record_ids.append(int(record_id))
                starts.append(start)
                record_window_count += 1
            print(
                f"class={CLASS_NAMES[class_id]:16s} load={load_id} "
                f"record={record_id} windows={record_window_count}"
            )

    return CWRUDataset(
        raw=np.stack(raw_windows).astype(np.float32),
        cwt=np.stack(cwt_windows).astype(np.float16),
        labels=np.asarray(labels, dtype=np.int64),
        loads=np.asarray(loads, dtype=np.int8),
        record_ids=np.asarray(record_ids, dtype=np.int32),
        starts=np.asarray(starts, dtype=np.int32),
    )


def save_cwru_cache(dataset: CWRUDataset, output_path: Path, metadata: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        raw=dataset.raw,
        cwt=dataset.cwt,
        labels=dataset.labels,
        loads=dataset.loads,
        record_ids=dataset.record_ids,
        starts=dataset.starts,
    )
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_cwru_cache(cache_path: Path) -> CWRUDataset:
    with np.load(cache_path) as data:
        return CWRUDataset(
            raw=data["raw"],
            cwt=data["cwt"],
            labels=data["labels"],
            loads=data["loads"],
            record_ids=data["record_ids"],
            starts=data["starts"],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 CWRU 10 类跨负载实验缓存")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_DIR / "public" / "cwru" / "raw",
        help="包含 40 个官方 .mat 文件的目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "public" / "cwru" / "cwru_10class_w2048_morl.npz",
        help="输出 NPZ 缓存路径",
    )
    parser.add_argument("--window-size", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=2048)
    parser.add_argument("--scales", type=int, default=48)
    parser.add_argument("--image-size", type=int, default=96)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = build_cwru_dataset(
        args.data_root,
        window_size=args.window_size,
        stride=args.stride,
        scales=args.scales,
        image_size=args.image_size,
    )
    metadata = {
        "dataset": "CWRU Bearing Data Center, 12 kHz drive-end data",
        "classes": CLASS_NAMES,
        "record_ids": CWRU_RECORD_IDS.tolist(),
        "load_to_rpm": LOAD_TO_RPM,
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "window_size": args.window_size,
        "stride": args.stride,
        "wavelets": list(DEFAULT_WAVELETS),
        "scales": args.scales,
        "image_size": args.image_size,
        "shape": {
            "raw": list(dataset.raw.shape),
            "cwt": list(dataset.cwt.shape),
        },
    }
    save_cwru_cache(dataset, args.output, metadata)
    print(f"Saved {args.output} with {len(dataset.labels)} windows")


if __name__ == "__main__":
    main()
