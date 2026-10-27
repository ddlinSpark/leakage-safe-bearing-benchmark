"""Shared configuration, provenance, and result persistence utilities."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..paths import PROJECT_ROOT


RESULT_COLUMNS = [
    "run_id",
    "experiment_type",
    "dataset",
    "protocol",
    "split_version",
    "method",
    "model",
    "model_key",
    "evaluation_mode",
    "fold",
    "seed",
    "label_ratio",
    "threshold",
    "macro_f1",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "ece",
    "nll",
    "brier",
    "raw_macro_f1",
    "raw_accuracy",
    "raw_ece",
    "raw_nll",
    "raw_brier",
    "train_time_s",
    "train_time",
    "inference_time_ms",
    "inference_time",
    "inference_ms_per_batch",
    "inference_samples_per_s",
    "parameters",
    "gflops_sample",
    "temperature",
    "train_loss",
    "best_val_loss",
    "best_val_macro_f1",
    "epochs_ran",
    "best_epoch",
    "labeled_windows",
    "unlabeled_candidate_windows",
    "train_windows",
    "validation_windows",
    "test_windows",
    "pseudo_count",
    "pseudo_coverage",
    "pseudo_accuracy",
    "pseudo_macro_f1",
    "self_training_rounds_completed",
    "config_hash",
    "split_hash",
    "dataset_hash",
    "git_commit",
    "git_dirty",
    "config_path",
    "status",
    "failure_reason",
    "timestamp_utc",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_provenance(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    def run_git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return completed.stdout.strip() if completed.returncode == 0 else ""

    commit = run_git("rev-parse", "HEAD")
    dirty = bool(run_git("status", "--porcelain"))
    return {
        "git_commit": commit or "unknown",
        "git_dirty": dirty,
    }


def environment_snapshot(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor() or "CPU"
        ),
    }


def load_json_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Experiment config must be a JSON object")
    return payload


def sanitize_identifier(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def read_results(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    frame = pd.read_csv(path)
    for column in RESULT_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    if frame["model"].isna().all() and "model_key" in frame:
        frame["model"] = frame["model_key"]
    if frame["train_time"].isna().all() and "train_time_s" in frame:
        frame["train_time"] = frame["train_time_s"]
    if frame["inference_time"].isna().all() and "inference_time_ms" in frame:
        frame["inference_time"] = frame["inference_time_ms"]
    ordered = [column for column in RESULT_COLUMNS if column in frame]
    extra = [column for column in frame.columns if column not in ordered]
    return frame[ordered + extra]


def append_result(path: Path, row: dict[str, Any]) -> pd.DataFrame:
    frame = read_results(path)
    row_frame = pd.DataFrame([row])
    frame = pd.concat([frame, row_frame], ignore_index=True)
    frame = frame.loc[:, ~frame.columns.duplicated()]
    atomic_write_csv(path, frame)
    return frame


def has_exact_run(
    results: pd.DataFrame,
    *,
    run_id: str,
    config_hash: str,
) -> bool:
    if results.empty:
        return False
    mask = (
        results["run_id"].astype(str).eq(run_id)
        & results["config_hash"].astype(str).eq(config_hash)
        & results["status"].astype(str).eq("success")
    )
    return bool(mask.any())


def configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(console)
    root.addHandler(file_handler)


def ensure_clean_worktree(allow_dirty: bool) -> dict[str, Any]:
    provenance = git_provenance()
    if provenance["git_dirty"] and not allow_dirty:
        raise RuntimeError(
            "Git worktree is dirty. Commit the code or pass --allow-dirty for an "
            "explicitly marked exploratory run."
        )
    return provenance


def relative_or_absolute(path: Path, root: Path = PROJECT_ROOT) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())
