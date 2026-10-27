from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATS = PROJECT_ROOT / "results" / "hierarchical_statistics.csv"
RAW = PROJECT_ROOT / "results" / "raw_runs.csv"
SOTA = PROJECT_ROOT / "results" / "sota" / "raw_runs.csv"


def test_hierarchical_table_covers_requested_comparisons() -> None:
    table = pd.read_csv(STATS)
    required = {
        "main:raw_cnn:gru_only",
        "main:raw_cnn:cwt_cnn",
        "main:raw_cnn:cwt_gru_no_se",
        "main:raw_cnn:cwt_gru",
        "external:raw_cnn:drsn",
        "external:raw_cnn:tfnet",
        "main:cwt_cnn:cwt_gru",
        "main:cwt_gru_no_se:cwt_gru",
        "label_50:cwt_gru:cwt_gru_selftrain",
        "label_full:cwt_gru:cwt_gru_selftrain",
    }
    assert required.issubset(set(table["comparison"]))
    assert table["n_folds"].eq(4).all()
    assert table["n_seeds"].eq(5).all()
    assert table["n_pairs"].eq(20).all()
    assert (
        table["hierarchical_ci_low"]
        <= table["hierarchical_ci_high"]
    ).all()


def test_reported_main_deltas_match_existing_raw_rows() -> None:
    raw = pd.read_csv(RAW)
    main = raw.loc[raw["experiment_type"].eq("main")]
    pivot = main.pivot_table(
        index=["fold", "seed"],
        columns="method",
        values="macro_f1",
    )
    table = pd.read_csv(STATS).set_index("comparison")
    for reference, method in (
        ("raw_cnn", "gru_only"),
        ("raw_cnn", "cwt_cnn"),
        ("raw_cnn", "cwt_gru_no_se"),
        ("raw_cnn", "cwt_gru"),
        ("cwt_cnn", "cwt_gru"),
        ("cwt_gru_no_se", "cwt_gru"),
    ):
        expected = float(
            (pivot[method] - pivot[reference]).mean()
        )
        observed = float(
            table.loc[f"main:{reference}:{method}", "mean_delta"]
        )
        np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-12)


def test_external_deltas_match_existing_raw_rows() -> None:
    raw = pd.read_csv(RAW)
    external = pd.read_csv(SOTA)
    main = raw.loc[
        raw["experiment_type"].eq("main")
        & raw["method"].eq("raw_cnn")
    ]
    pivot = pd.concat([main, external], ignore_index=True).pivot_table(
        index=["fold", "seed"],
        columns="method",
        values="macro_f1",
    )
    table = pd.read_csv(STATS).set_index("comparison")
    for method in ("drsn", "tfnet"):
        expected = float((pivot[method] - pivot["raw_cnn"]).mean())
        observed = float(
            table.loc[f"external:raw_cnn:{method}", "mean_delta"]
        )
        np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-12)
