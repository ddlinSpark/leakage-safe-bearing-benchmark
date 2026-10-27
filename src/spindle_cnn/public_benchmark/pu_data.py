"""Paderborn University bearing data preparation for grouped experiments."""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pywt
import torch
import torch.nn.functional as F
from scipy.io import loadmat

from ..paths import DATA_DIR
from .cwru_data import CWRUDataset, _standardize_window


CLASS_NAMES = ["Healthy", "OuterRing", "InnerRing", "MultipleDamage"]

BEARING_CODES = {
    "K001": 0,
    "K002": 0,
    "K003": 0,
    "K004": 0,
    "K005": 0,
    "KA01": 1,
    "KA04": 1,
    "KA08": 1,
    "KA15": 1,
    "KI01": 2,
    "KI04": 2,
    "KI08": 2,
    "KI14": 2,
    "KB23": 3,
    "KB24": 3,
    "KB27": 3,
}

BEARING_RECORD_IDS = {code: idx for idx, code in enumerate(BEARING_CODES)}
RECORD_ID_TO_BEARING = {idx: code for code, idx in BEARING_RECORD_IDS.items()}
HELDOUT_RECORD_IDS = {
    BEARING_RECORD_IDS[code] for code in ("K005", "KA15", "KI14", "KB27")
}

CONDITION_TO_LOAD = {
    "N09_M07_F10": 0,
    "N15_M01_F10": 1,
    "N15_M07_F04": 2,
    "N15_M07_F10": 3,
}
LOAD_TO_CONDITION = {value: key for key, value in CONDITION_TO_LOAD.items()}

_FILENAME_PATTERN = re.compile(
    r"^(?P<condition>N\d{2}_M\d{2}_F\d{2})_"
    r"(?P<bearing>K(?:[ABI]\d{2,3}|\d{3}))_(?P<measurement>\d+)$"
)


def _host_service_channels(path: Path) -> np.ndarray:
    data = loadmat(path)
    root = data[path.stem][0, 0]
    signals = {}
    for item in np.asarray(root["Y"]).reshape(-1):
        name = item["Name"].flat[0].item()
        raster = item["Raster"].flat[0].item()
        if raster != "HostService":
            continue
        if name in {"phase_current_1", "phase_current_2", "vibration_1"}:
            signals[name] = np.asarray(item["Data"]).reshape(-1).astype(np.float32)
    required = ["phase_current_1", "phase_current_2", "vibration_1"]
    if any(name not in signals for name in required):
        raise KeyError(f"Missing HostService channels in {path}: {sorted(signals)}")
    length = min(len(signals[name]) for name in required)
    return np.stack([signals[name][:length] for name in required], axis=0)


def _cwt_image(
    raw_window: np.ndarray,
    scales: int,
    image_size: int,
) -> np.ndarray:
    scale_values = np.geomspace(1, 64, num=scales)
    channels = []
    for channel in raw_window:
        coefficients, _ = pywt.cwt(channel, scale_values, "morl")
        tensor = torch.from_numpy(np.abs(coefficients).astype(np.float32))
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        tensor = F.interpolate(
            tensor,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        tensor = tensor / torch.clamp(tensor.amax(), min=1e-6)
        channels.append(tensor.numpy())
    return np.stack(channels).astype(np.float16)


def _process_file(
    path: Path,
    window_size: int,
    stride: int,
    scales: int,
    image_size: int,
    max_measurements: int,
) -> dict[str, Any] | None:
    match = _FILENAME_PATTERN.match(path.stem)
    if match is None:
        return None
    condition = match.group("condition")
    bearing_code = match.group("bearing")
    measurement = int(match.group("measurement"))
    if condition not in CONDITION_TO_LOAD or bearing_code not in BEARING_CODES:
        return None
    if measurement > max_measurements:
        return None
    try:
        signal = _host_service_channels(path)
    except Exception as exc:  # A single damaged official file should not abort the subset.
        return {"error": str(exc), "path": str(path)}

    length = signal.shape[1]
    raw_windows = []
    cwt_windows = []
    starts = []
    for start in range(0, length - window_size + 1, stride):
        raw = _standardize_window(signal[:, start : start + window_size])
        raw_windows.append(raw)
        cwt_windows.append(_cwt_image(raw, scales, image_size))
        starts.append(start)
    return {
        "path": str(path),
        "condition": condition,
        "bearing_code": bearing_code,
        "measurement": measurement,
        "raw": raw_windows,
        "cwt": cwt_windows,
        "starts": starts,
    }


def build_pu_dataset(
    data_root: Path,
    *,
    window_size: int = 8192,
    stride: int = 8192,
    scales: int = 48,
    image_size: int = 96,
    max_measurements: int = 2,
    workers: int = min(8, os.cpu_count() or 1),
) -> CWRUDataset:
    mat_files = sorted(data_root.rglob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No PU .mat files under {data_root}")

    raw_windows: list[np.ndarray] = []
    cwt_windows: list[np.ndarray] = []
    labels: list[int] = []
    loads: list[int] = []
    record_ids: list[int] = []
    starts: list[int] = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = executor.map(
            _process_file,
            mat_files,
            [window_size] * len(mat_files),
            [stride] * len(mat_files),
            [scales] * len(mat_files),
            [image_size] * len(mat_files),
            [max_measurements] * len(mat_files),
        )
        for result in results:
            if result is None:
                continue
            if "error" in result:
                print(f"WARNING skipped {result['path']}: {result['error']}")
                continue
            bearing_code = result["bearing_code"]
            condition = result["condition"]
            raw_windows.extend(result["raw"])
            cwt_windows.extend(result["cwt"])
            starts.extend(result["starts"])
            count = len(result["starts"])
            labels.extend([BEARING_CODES[bearing_code]] * count)
            loads.extend([CONDITION_TO_LOAD[condition]] * count)
            record_ids.extend([BEARING_RECORD_IDS[bearing_code]] * count)
            print(
                f"{condition} {bearing_code} measurement={result['measurement']} "
                f"windows={count}"
            )

    observed_records = set(record_ids)
    expected_records = set(BEARING_RECORD_IDS.values())
    if observed_records != expected_records:
        missing = sorted(expected_records - observed_records)
        raise ValueError(f"Missing records after preprocessing: {missing}")
    observed_classes = set(labels)
    if observed_classes != set(range(len(CLASS_NAMES))):
        raise ValueError(f"Missing classes after preprocessing: {sorted(observed_classes)}")

    return CWRUDataset(
        raw=np.stack(raw_windows).astype(np.float32),
        cwt=np.stack(cwt_windows).astype(np.float16),
        labels=np.asarray(labels, dtype=np.int64),
        loads=np.asarray(loads, dtype=np.int8),
        record_ids=np.asarray(record_ids, dtype=np.int32),
        starts=np.asarray(starts, dtype=np.int32),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 Paderborn 四类跨工况缓存")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_DIR / "public" / "paderborn" / "extracted",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "public" / "paderborn" / "pu_4class_w8192.npz",
    )
    parser.add_argument("--window-size", type=int, default=8192)
    parser.add_argument("--stride", type=int, default=8192)
    parser.add_argument("--scales", type=int, default=48)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--max-measurements", type=int, default=2)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = build_pu_dataset(
        args.data_root,
        window_size=args.window_size,
        stride=args.stride,
        scales=args.scales,
        image_size=args.image_size,
        max_measurements=args.max_measurements,
        workers=args.workers,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        raw=dataset.raw,
        cwt=dataset.cwt,
        labels=dataset.labels,
        loads=dataset.loads,
        record_ids=dataset.record_ids,
        starts=dataset.starts,
    )
    metadata = {
        "dataset": "Paderborn University Bearing Dataset",
        "classes": CLASS_NAMES,
        "bearing_codes": BEARING_CODES,
        "record_ids": BEARING_RECORD_IDS,
        "heldout_record_ids": sorted(HELDOUT_RECORD_IDS),
        "condition_to_load": CONDITION_TO_LOAD,
        "window_size": args.window_size,
        "stride": args.stride,
        "max_measurements": args.max_measurements,
        "shape": {"raw": list(dataset.raw.shape), "cwt": list(dataset.cwt.shape)},
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {args.output} with {len(dataset.labels)} windows")


if __name__ == "__main__":
    main()
