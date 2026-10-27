"""Group-safe split construction and label-budget sampling."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..public_benchmark.cwru_data import CWRUDataset


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    train_conditions: tuple[int, ...]
    validation_condition: int
    test_condition: int


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch and optionally request determinism."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def load_fold_indices(
    data: CWRUDataset,
    test_load: int,
    heldout_record_ids: np.ndarray | None = None,
) -> SplitIndices:
    """Construct one held-out-condition fold with optional bearing holdout."""

    remaining = [load for load in range(4) if load != test_load]
    if not remaining:
        raise ValueError("At least one training/validation condition is required")
    validation_load = remaining[test_load % len(remaining)]
    train_loads = tuple(load for load in remaining if load != validation_load)

    if heldout_record_ids is not None and len(heldout_record_ids) > 0:
        train_validation_mask = ~np.isin(data.record_ids, heldout_record_ids)
        test_mask = np.isin(data.record_ids, heldout_record_ids)
        train_idx = np.flatnonzero(
            train_validation_mask & np.isin(data.loads, train_loads)
        )
        val_idx = np.flatnonzero(train_validation_mask & (data.loads == validation_load))
        test_idx = np.flatnonzero(test_mask & (data.loads == test_load))
    else:
        train_idx = np.flatnonzero(np.isin(data.loads, train_loads))
        val_idx = np.flatnonzero(data.loads == validation_load)
        test_idx = np.flatnonzero(data.loads == test_load)

    split = SplitIndices(
        train=np.asarray(train_idx, dtype=np.int64),
        validation=np.asarray(val_idx, dtype=np.int64),
        test=np.asarray(test_idx, dtype=np.int64),
        train_conditions=train_loads,
        validation_condition=validation_load,
        test_condition=test_load,
    )
    validate_split_isolation(data, split, heldout_record_ids)
    return split


def validate_split_isolation(
    data: CWRUDataset,
    split: SplitIndices,
    heldout_record_ids: np.ndarray | None = None,
) -> None:
    """Raise when windows or groups overlap across partitions."""

    train = set(split.train.tolist())
    validation = set(split.validation.tolist())
    test = set(split.test.tolist())
    if train & validation or train & test or validation & test:
        raise ValueError("Window indices overlap across train/validation/test")
    if set(split.train_conditions) & {split.test_condition, split.validation_condition}:
        raise ValueError("Training conditions overlap with validation/test")
    if split.validation_condition == split.test_condition:
        raise ValueError("Validation and test conditions must differ")

    for name, indices in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        if len(indices) == 0:
            raise ValueError(f"{name} split is empty")
        observed_conditions = set(data.loads[indices].tolist())
        expected_condition = {
            "train": set(split.train_conditions),
            "validation": {split.validation_condition},
            "test": {split.test_condition},
        }[name]
        if observed_conditions != expected_condition:
            raise ValueError(
                f"{name} contains unexpected conditions: {sorted(observed_conditions)}"
            )

    if heldout_record_ids is not None and len(heldout_record_ids) > 0:
        heldout = set(np.asarray(heldout_record_ids, dtype=np.int64).tolist())
        if set(data.record_ids[split.train].tolist()) & heldout:
            raise ValueError("Held-out bearing IDs occur in the training split")
        if set(data.record_ids[split.validation].tolist()) & heldout:
            raise ValueError("Held-out bearing IDs occur in the validation split")
        if not set(data.record_ids[split.test].tolist()).issubset(heldout):
            raise ValueError("Test split contains non-held-out bearing IDs")


def balanced_indices(
    indices: np.ndarray,
    labels: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample an equal number of windows from every class."""

    if not 0 < fraction <= 1:
        raise ValueError("Label fraction must be in (0, 1]")
    indices = np.asarray(indices, dtype=np.int64)
    num_classes = int(labels.max()) + 1
    by_class = [indices[labels[indices] == class_id] for class_id in range(num_classes)]
    missing = [class_id for class_id, items in enumerate(by_class) if len(items) == 0]
    if missing:
        raise ValueError(f"Training split is missing classes: {missing}")

    per_class = min(len(items) for items in by_class)
    take = max(1, math.ceil(per_class * fraction))
    selected = [rng.choice(items, size=take, replace=False) for items in by_class]
    return np.concatenate(selected).astype(np.int64)


def _sha256_indices(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def build_split_manifest(
    data: CWRUDataset,
    split: SplitIndices,
    *,
    dataset_name: str,
    split_version: str,
    fold: int,
    heldout_record_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return an auditable JSON-compatible description of one fold."""

    def split_summary(indices: np.ndarray) -> dict[str, Any]:
        unique_labels, counts = np.unique(data.labels[indices], return_counts=True)
        unique_records, record_counts = np.unique(data.record_ids[indices], return_counts=True)
        unique_conditions, condition_counts = np.unique(data.loads[indices], return_counts=True)
        return {
            "count": int(len(indices)),
            "indices": indices.astype(int).tolist(),
            "indices_sha256": _sha256_indices(indices),
            "class_counts": {
                str(int(label)): int(count)
                for label, count in zip(unique_labels, counts, strict=True)
            },
            "record_counts": {
                str(int(record)): int(count)
                for record, count in zip(unique_records, record_counts, strict=True)
            },
            "condition_counts": {
                str(int(condition)): int(count)
                for condition, count in zip(
                    unique_conditions,
                    condition_counts,
                    strict=True,
                )
            },
        }

    return {
        "dataset": dataset_name,
        "split_version": split_version,
        "fold": fold,
        "test_condition": split.test_condition,
        "validation_condition": split.validation_condition,
        "train_conditions": list(split.train_conditions),
        "heldout_record_ids": (
            sorted(int(value) for value in heldout_record_ids)
            if heldout_record_ids is not None
            else []
        ),
        "train": split_summary(split.train),
        "validation": split_summary(split.validation),
        "test": split_summary(split.test),
    }


def write_json(path: Path, payload: dict[str, Any]) -> str:
    """Write deterministic, atomically replaced JSON and return a content hash."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded + "\n", encoding="utf-8")
    temporary.replace(path)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
