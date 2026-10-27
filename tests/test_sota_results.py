from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOTA_DIR = PROJECT_ROOT / "results" / "sota"


def test_external_retraining_matrix_is_complete() -> None:
    raw = pd.read_csv(SOTA_DIR / "raw_runs.csv")
    assert len(raw) == 40
    assert raw["run_id"].is_unique
    assert raw["status"].eq("success").all()
    assert set(raw["method"]) == {"drsn", "tfnet"}
    assert raw.groupby("method").size().eq(20).all()
    assert set(raw["fold"]) == {0, 1, 2, 3}
    assert set(raw["seed"]) == {42, 123, 2024, 2025, 3407}
    assert not raw.duplicated(["method", "fold", "seed"]).any()
    assert not raw["git_dirty"].astype(str).str.casefold().eq("true").any()
    assert raw["dataset_hash"].nunique() == 1
    assert raw.groupby("fold")["split_hash"].nunique().eq(1).all()


def test_external_summaries_are_computed_from_raw_rows() -> None:
    raw = pd.read_csv(SOTA_DIR / "raw_runs.csv")
    summary = pd.read_csv(SOTA_DIR / "summary_with_ci.csv").set_index("Method")
    for method, group in raw.groupby("method"):
        np.testing.assert_allclose(
            summary.loc[method, "Macro-F1 Mean"],
            group["macro_f1"].mean(),
            rtol=0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            summary.loc[method, "Macro-F1 Std"],
            group["macro_f1"].std(ddof=1),
            rtol=0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            summary.loc[method, "Accuracy Mean"],
            group["accuracy"].mean(),
            rtol=0,
            atol=1e-12,
        )


def test_external_logits_reproduce_all_recorded_metrics() -> None:
    report = json.loads(
        (SOTA_DIR / "rechecks" / "all_runs.json").read_text(encoding="utf-8")
    )
    assert report["passed"] == 40
    assert len(report["runs"]) == 40
    assert all(run["status"] == "passed" for run in report["runs"])
    assert all(
        all(metric["passed"] for metric in run["metrics"].values())
        for run in report["runs"]
    )
