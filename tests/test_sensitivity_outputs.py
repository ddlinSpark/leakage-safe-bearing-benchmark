from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_label_granularity_rows_reuse_the_same_trained_predictions() -> None:
    frame = pd.read_csv(ROOT / "results" / "label_granularity_per_run.csv")
    assert not frame.empty
    counts = frame.groupby("run_id")["granularity"].nunique()
    assert counts.eq(3).all()
    assert frame["runs"].eq(1).all()


def test_domain_shift_has_raw_amplitude_statistics() -> None:
    frame = pd.read_csv(ROOT / "results" / "domain_shift_statistics.csv")
    required = {
        "rms",
        "peak",
        "kurtosis",
        "crest_factor",
        "spectral_centroid_hz",
        "spectral_entropy",
    }
    assert required.issubset(set(frame["metric"]))
    assert frame["mean"].notna().all()


def test_damage_origin_manifest_covers_each_fold_and_split() -> None:
    frame = pd.read_csv(ROOT / "results" / "damage_origin_manifest.csv")
    assert set(frame["fold"]) == {0, 1, 2, 3}
    assert set(frame["split"]) == {"train", "validation", "test"}
    assert frame.groupby(["fold", "split"]).size().ge(1).all()
    test_rows = frame.loc[frame["split"].eq("test")]
    assert set(test_rows["damage_origin"]) <= {"none", "real"}


def test_protocol_window_audit_is_stride_aligned() -> None:
    frame = pd.read_csv(ROOT / "reports" / "window_audit.csv")
    assert len(frame) == 3939
    assert frame["window_start"].mod(8192).eq(0).all()
    assert (frame["window_end_exclusive"] - frame["window_start"]).eq(8192).all()
    assert frame["measurement"].isin([1, 2]).all()


def test_calibration_audit_has_one_row_per_canonical_run() -> None:
    raw = pd.read_csv(ROOT / "results" / "raw_runs.csv")
    calibration = pd.read_csv(ROOT / "results" / "calibration_per_run.csv")
    assert len(calibration) == len(raw) == 320
    assert calibration["run_id"].is_unique
    assert np.isfinite(calibration["test_ece_after"]).all()


def test_cwt_control_matrix_is_complete() -> None:
    variants = [
        "no_max_norm",
        "log_magnitude",
        "resolution_128",
        "scales_32",
        "scales_64",
        "real_imag",
    ]
    for variant in variants:
        frame = pd.read_csv(
            ROOT / "results" / "cwt_sensitivity" / variant / "raw_runs.csv"
        )
        assert len(frame) == 20
        assert frame["status"].eq("success").all()
        assert not frame.duplicated(["method", "fold", "seed"]).any()


def test_coverage_self_training_matrix_is_complete() -> None:
    frame = pd.read_csv(
        ROOT / "results" / "self_training_sensitivity" / "raw_runs.csv"
    )
    assert len(frame) == 40
    assert frame["status"].eq("success").all()
    assert frame.groupby(["experiment_type", "label_ratio"]).size().eq(20).all()
