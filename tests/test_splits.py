from __future__ import annotations

import numpy as np

from spindle_cnn.datasets.splits import (
    balanced_indices,
    load_fold_indices,
    validate_split_isolation,
)
from spindle_cnn.public_benchmark.cwru_data import CWRUDataset


def synthetic_dataset() -> CWRUDataset:
    loads = np.repeat(np.arange(4, dtype=np.int8), 40)
    labels = np.tile(np.repeat(np.arange(4, dtype=np.int64), 10), 4)
    record_ids = (labels * 3 + (np.arange(len(labels)) % 3)).astype(np.int32)
    count = len(labels)
    return CWRUDataset(
        raw=np.zeros((count, 2, 8), dtype=np.float32),
        cwt=np.zeros((count, 2, 4, 4), dtype=np.float16),
        labels=labels,
        loads=loads,
        record_ids=record_ids,
        starts=np.tile(np.arange(5, dtype=np.int32), 32),
    )


def test_heldout_records_and_conditions_are_disjoint() -> None:
    data = synthetic_dataset()
    split = load_fold_indices(data, 0, np.asarray([0, 3], dtype=np.int64))
    validate_split_isolation(data, split, np.asarray([0, 3], dtype=np.int64))
    assert split.test_condition not in split.train_conditions
    assert split.test_condition != split.validation_condition
    assert not set(data.record_ids[split.train]) & {0, 3}
    assert not set(data.record_ids[split.validation]) & {0, 3}


def test_balanced_fraction_samples_each_class_equally() -> None:
    data = synthetic_dataset()
    split = load_fold_indices(data, 0, np.asarray([0, 3], dtype=np.int64))
    selected = balanced_indices(
        split.train,
        data.labels,
        0.5,
        np.random.default_rng(42),
    )
    counts = np.bincount(data.labels[selected], minlength=4)
    assert counts.min() == counts.max()
    assert set(selected).issubset(set(split.train))
