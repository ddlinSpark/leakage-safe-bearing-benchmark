from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from spindle_cnn.analysis.recheck_run import recheck_one


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


def test_canonical_result_matrix_has_required_repetitions() -> None:
    raw = pd.read_csv(RESULTS_DIR / "raw_runs.csv")
    required = {
        "model",
        "fold",
        "seed",
        "label_ratio",
        "threshold",
        "macro_f1",
        "accuracy",
        "train_time",
        "inference_time",
        "parameters",
        "config_hash",
        "split_hash",
        "dataset_hash",
        "git_commit",
        "status",
    }
    assert required.issubset(raw.columns)
    assert raw["run_id"].is_unique
    assert raw["status"].eq("success").all()

    main = raw.loc[raw["experiment_type"].eq("main")]
    assert main.groupby("method").size().eq(20).all()
    assert set(main["seed"]) == {42, 123, 2024, 2025, 3407}
    assert set(main["fold"]) == {0, 1, 2, 3}

    label = raw.loc[
        raw["experiment_type"].eq("label_efficiency")
    ].groupby(["method", "label_ratio"]).size()
    assert label.eq(20).all()

    threshold = raw.loc[
        raw["experiment_type"].eq("threshold")
    ].groupby("threshold").size()
    assert threshold.eq(20).all()


def test_summary_values_are_reproduced_from_raw_rows() -> None:
    raw = pd.read_csv(RESULTS_DIR / "raw_runs.csv")
    summary = pd.read_csv(RESULTS_DIR / "summary.csv")
    main = raw.loc[raw["experiment_type"].eq("main")]
    expected = main.groupby("method").agg(
        macro_f1=("macro_f1", "mean"),
        accuracy=("accuracy", "mean"),
    )
    observed = summary.set_index("method")
    np.testing.assert_allclose(
        observed.loc[expected.index, "Macro-F1 Mean"],
        expected["macro_f1"],
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        observed.loc[expected.index, "Accuracy Mean"],
        expected["accuracy"],
        rtol=0,
        atol=1e-12,
    )


def test_split_manifests_exclude_heldout_bearings_from_fitting() -> None:
    for fold in range(4):
        manifest = json.loads(
            (RESULTS_DIR / "splits" / f"fold{fold}.json").read_text(encoding="utf-8")
        )
        heldout = set(manifest["heldout_record_ids"])
        train_records = set(map(int, manifest["train"]["record_counts"]))
        validation_records = set(map(int, manifest["validation"]["record_counts"]))
        test_records = set(map(int, manifest["test"]["record_counts"]))
        assert not train_records & heldout
        assert not validation_records & heldout
        assert test_records == heldout
        assert manifest["test_condition"] not in manifest["train_conditions"]
        assert manifest["test_condition"] != manifest["validation_condition"]


def test_saved_logits_reproduce_recorded_metrics() -> None:
    raw = pd.read_csv(RESULTS_DIR / "raw_runs.csv")
    report = recheck_one(RESULTS_DIR, raw.iloc[0], tolerance=1e-8)
    assert report["status"] == "passed"
