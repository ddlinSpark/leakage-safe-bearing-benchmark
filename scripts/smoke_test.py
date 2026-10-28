"""Dependency-light smoke test for the formal reproducibility repository."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from spindle_cnn.datasets.splits import balanced_indices, load_fold_indices
from spindle_cnn.evaluation.metrics import metrics_from_logits
from spindle_cnn.models.registry import build_model
from spindle_cnn.public_benchmark.cwru_data import CWRUDataset
from spindle_cnn.visualization.style import PAPER_COLORS, apply_paper_style


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = json.loads(
        (ROOT / "configs" / "paderborn_main.json").read_text(encoding="utf-8")
    )
    assert config["folds"] == [0, 1, 2, 3]
    assert config["seeds"] == [42, 123, 2024, 2025, 3407]
    assert len(PAPER_COLORS) == 11
    apply_paper_style()

    heldout = set(config["dataset"]["heldout_record_ids"])
    for fold in config["folds"]:
        manifest = json.loads(
            (ROOT / "results" / "splits" / f"fold{fold}.json").read_text(
                encoding="utf-8"
            )
        )
        assert set(manifest["test"]["record_counts"]) == {
            str(value) for value in heldout
        }
        fitting = set(manifest["train"]["record_counts"]) | set(
            manifest["validation"]["record_counts"]
        )
        assert fitting.isdisjoint({str(value) for value in heldout})

    loads = np.repeat(np.arange(4, dtype=np.int8), 40)
    labels = np.tile(np.repeat(np.arange(4, dtype=np.int64), 10), 4)
    records = (labels * 3 + (np.arange(len(labels)) % 3)).astype(np.int32)
    data = CWRUDataset(
        raw=np.zeros((len(labels), 3, 8), dtype=np.float32),
        cwt=np.zeros((len(labels), 3, 4, 4), dtype=np.float16),
        labels=labels,
        loads=loads,
        record_ids=records,
        starts=np.tile(np.arange(5, dtype=np.int32), 32),
    )
    split = load_fold_indices(data, 0, np.asarray([0, 3]))
    selected = balanced_indices(split.train, labels, 0.5, np.random.default_rng(42))
    counts = np.bincount(labels[selected], minlength=4)
    assert counts.min() == counts.max()

    raw = torch.randn(2, 3, 64)
    cwt = torch.randn(2, 3, 16, 16)
    for model_key in (
        "raw_cnn",
        "gru_only",
        "cwt_cnn",
        "cwt_gru_no_se",
        "cwt_gru",
        "drsn",
        "tfnet",
    ):
        model = build_model(
            model_key,
            num_classes=4,
            raw_channels=3,
            image_channels=3,
        )
        model.eval()
        with torch.inference_mode():
            logits = model(raw, cwt)
        assert logits.shape == (2, 4)

    logits = np.asarray(
        [[2.0, 0.0, -1.0, -2.0], [0.0, 2.0, -1.0, -2.0]],
        dtype=np.float64,
    )
    metric_values = metrics_from_logits(
        logits,
        np.asarray([0, 1], dtype=np.int64),
        temperature=1.0,
    )
    assert 0.0 <= metric_values["accuracy"] <= 1.0
    assert 0.0 <= metric_values["ece"] <= 1.0
    print("SMOKE TEST PASS: config, splits, indexing, metrics, style, and 7 models")


if __name__ == "__main__":
    main()
