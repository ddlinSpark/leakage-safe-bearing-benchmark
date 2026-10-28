"""Build an audit cache that preserves original Paderborn signal amplitudes."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from ..paths import DATA_DIR, PROJECT_ROOT
from .cwru_data import _standardize_window
from .pu_data import (
    BEARING_CODES,
    BEARING_RECORD_IDS,
    CONDITION_TO_LOAD,
    _FILENAME_PATTERN,
    _host_service_channels,
    CLASS_NAMES,
)


def _read_source(
    path_text: str,
    window_size: int,
    stride: int,
    max_measurements: int,
) -> dict[str, Any] | None:
    path = Path(path_text)
    match = _FILENAME_PATTERN.match(path.stem)
    if match is None:
        return None
    condition = match.group("condition")
    bearing_code = match.group("bearing")
    measurement = int(match.group("measurement"))
    if (
        condition not in CONDITION_TO_LOAD
        or bearing_code not in BEARING_CODES
        or measurement > max_measurements
    ):
        return None
    try:
        signal = _host_service_channels(path)
    except Exception as exc:
        return {
            "source_file": str(path),
            "error": f"{type(exc).__name__}: {exc}",
        }

    windows = []
    starts = []
    for start in range(0, signal.shape[1] - window_size + 1, stride):
        windows.append(signal[:, start : start + window_size].copy())
        starts.append(start)
    return {
        "source_file": str(path),
        "condition": condition,
        "bearing_code": bearing_code,
        "measurement": measurement,
        "raw": np.stack(windows).astype(np.float32) if windows else np.empty(
            (0, signal.shape[0], window_size),
            dtype=np.float32,
        ),
        "starts": np.asarray(starts, dtype=np.int32),
    }


def build_raw_window_cache(
    data_root: Path,
    output_path: Path,
    *,
    window_size: int = 8192,
    stride: int = 8192,
    max_measurements: int = 2,
    workers: int = 4,
) -> dict[str, Any]:
    mat_files = sorted(data_root.rglob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No Paderborn .mat files under {data_root}")

    windows = []
    labels = []
    loads = []
    record_ids = []
    measurements = []
    starts = []
    source_files = []
    errors = []
    source_file_count = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = executor.map(
            _read_source,
            [str(path) for path in mat_files],
            [window_size] * len(mat_files),
            [stride] * len(mat_files),
            [max_measurements] * len(mat_files),
        )
        for result in futures:
            if result is None:
                continue
            if "error" in result:
                errors.append(
                    {
                        "source_file": result["source_file"],
                        "error": result["error"],
                    }
                )
                continue
            source_file_count += 1
            count = len(result["raw"])
            windows.append(result["raw"])
            starts.append(result["starts"])
            labels.extend(
                [BEARING_CODES[result["bearing_code"]]] * count
            )
            loads.extend([CONDITION_TO_LOAD[result["condition"]]] * count)
            record_ids.extend([BEARING_RECORD_IDS[result["bearing_code"]]] * count)
            measurements.extend([result["measurement"]] * count)
            source_files.extend([result["source_file"]] * count)

    if not windows:
        raise ValueError("No readable Paderborn measurement files were found")

    raw = np.concatenate(windows, axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        raw=raw,
        labels=np.asarray(labels, dtype=np.int64),
        loads=np.asarray(loads, dtype=np.int8),
        record_ids=np.asarray(record_ids, dtype=np.int32),
        measurements=np.asarray(measurements, dtype=np.int8),
        starts=np.concatenate(starts),
        source_files=np.asarray(source_files, dtype=np.str_),
    )
    raw_cache_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
    metadata = {
        "dataset": "Paderborn University Bearing Dataset",
        "classes": CLASS_NAMES,
        "bearing_codes": BEARING_CODES,
        "condition_to_load": CONDITION_TO_LOAD,
        "window_size": window_size,
        "stride": stride,
        "max_measurements": max_measurements,
        "shape": {"raw": list(raw.shape)},
        "source_files_read": source_file_count,
        "windows_read": int(len(labels)),
        "raw_cache_sha256": raw_cache_sha256,
        "errors": errors,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def validate_against_canonical(
    raw_cache_path: Path,
    canonical_cache_path: Path,
    *,
    tolerance: float = 2e-6,
) -> dict[str, Any]:
    with np.load(raw_cache_path) as raw_data, np.load(canonical_cache_path) as canonical:
        fields = ("labels", "loads", "record_ids", "starts")
        field_results = {
            field: bool(np.array_equal(raw_data[field], canonical[field]))
            for field in fields
        }
        if len(raw_data["raw"]) != len(canonical["raw"]):
            return {
                "status": "MISMATCH",
                "raw_cache_rows": int(len(raw_data["raw"])),
                "canonical_rows": int(len(canonical["raw"])),
                "field_equal": field_results,
            }
        max_difference = 0.0
        chunk_size = 128
        for start in range(0, len(raw_data["raw"]), chunk_size):
            stop = min(start + chunk_size, len(raw_data["raw"]))
            reconstructed = np.stack(
                [_standardize_window(window) for window in raw_data["raw"][start:stop]]
            )
            expected = canonical["raw"][start:stop]
            difference = float(
                np.max(np.abs(reconstructed.astype(np.float32) - expected))
            )
            max_difference = max(max_difference, difference)
            if difference > tolerance:
                return {
                    "status": "MISMATCH",
                    "first_mismatch_chunk": [start, stop],
                    "maximum_absolute_difference": difference,
                    "field_equal": field_results,
                }
    return {
        "status": "VERIFIED" if all(field_results.values()) else "MISMATCH",
        "maximum_standardization_difference": max_difference,
        "tolerance": tolerance,
        "field_equal": field_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_DIR / "public" / "paderborn" / "extracted",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "public" / "paderborn" / "pu_4class_w8192_raw.npz",
    )
    parser.add_argument(
        "--canonical",
        type=Path,
        default=DATA_DIR / "public" / "paderborn" / "pu_4class_w8192.npz",
    )
    parser.add_argument("--window-size", type=int, default=8192)
    parser.add_argument("--stride", type=int, default=8192)
    parser.add_argument("--max-measurements", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = build_raw_window_cache(
        args.data_root,
        args.output,
        window_size=args.window_size,
        stride=args.stride,
        max_measurements=args.max_measurements,
        workers=args.workers,
    )
    print(
        f"Saved {args.output} with {metadata['shape']['raw'][0]} windows; "
        f"errors={len(metadata['errors'])}"
    )
    if args.validate:
        validation = validate_against_canonical(args.output, args.canonical)
        validation_path = args.output.with_name(
            args.output.stem + "_validation.json"
        )
        validation_path.write_text(
            json.dumps(validation, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(validation, indent=2))
        if validation["status"] != "VERIFIED":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
