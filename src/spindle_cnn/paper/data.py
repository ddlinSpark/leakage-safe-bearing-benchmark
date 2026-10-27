"""Load and validate every numerical source used by the manuscript."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


METHOD_LABELS = {
    "raw_cnn": "Raw 1D-CNN",
    "gru_only": "GRU only",
    "cwt_cnn": "CWT 2D-CNN",
    "cwt_gru_no_se": "CWT-GRU without SE",
    "cwt_gru": "CWT-GRU",
    "cwt_gru_temp": "CWT-GRU + temperature scaling",
    "cwt_gru_selftrain": "CWT-GRU + temperature scaling + self-training",
}

CLASS_LABELS = ["Healthy", "Outer ring", "Inner ring", "Multiple damage"]


class BenchmarkResults:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.raw = pd.read_csv(self.root / "raw_runs.csv")
        self.summary = pd.read_csv(self.root / "summary.csv")
        self.summary_ci = pd.read_csv(self.root / "summary_with_ci.csv")
        self.tests = pd.read_csv(self.root / "statistical_tests.csv")
        self.ablation = pd.read_csv(self.root / "ablation_summary.csv")
        self.domain_gap = pd.read_csv(self.root / "domain_gap_summary.csv")
        self.label = pd.read_csv(self.root / "label_efficiency_summary.csv")
        self.threshold = pd.read_csv(self.root / "threshold_summary.csv")
        self.complexity = pd.read_csv(self.root / "model_complexity.csv")
        self._validate()

    def _validate(self) -> None:
        if len(self.raw) != 320:
            raise ValueError(f"Expected 320 canonical runs, found {len(self.raw)}")
        if not self.raw["status"].astype(str).eq("success").all():
            raise ValueError("Canonical table contains a failed run")
        if not self.raw["run_id"].is_unique:
            raise ValueError("Canonical table contains duplicate run IDs")
        main_counts = self.main.groupby("method").size()
        if not main_counts.eq(20).all():
            raise ValueError(f"Main experiment does not contain 20 runs per method: {main_counts}")
        threshold_counts = self.raw.loc[
            self.raw["experiment_type"].eq("threshold")
        ].groupby("threshold").size()
        if not threshold_counts.eq(20).all():
            raise ValueError(
                f"Threshold experiment does not contain 20 runs per threshold: {threshold_counts}"
            )

    @property
    def main(self) -> pd.DataFrame:
        return self.raw.loc[self.raw["experiment_type"].eq("main")].copy()

    def summary_row(self, method: str, experiment_type: str = "main") -> pd.Series:
        match = self.summary_ci.loc[
            self.summary_ci["experiment_type"].eq(experiment_type)
            & self.summary_ci["method"].eq(method)
        ]
        if len(match) != 1:
            raise ValueError(f"Expected one summary row for {experiment_type}/{method}")
        return match.iloc[0]

    def paired(self, reference: str, method: str, experiment_type: str = "main") -> pd.Series:
        if experiment_type == "main":
            match = self.tests.loc[
                self.tests["experiment_type"].eq("main")
                & self.tests["reference"].eq(reference)
                & self.tests["method"].eq(method)
            ]
        elif experiment_type == "label_efficiency":
            match = self.tests.loc[
                self.tests["experiment_type"].eq("label_efficiency")
                & self.tests["reference"].eq(reference)
                & self.tests["method"].eq(method)
            ]
        else:
            raise ValueError(f"Unsupported paired experiment: {experiment_type}")
        if len(match) != 1:
            raise ValueError(
                f"Expected one paired comparison for {experiment_type}/{reference}/{method}"
            )
        return match.iloc[0]

    def label_row(self, method: str, fraction: float) -> pd.Series:
        match = self.label.loc[
            self.label["method"].eq(method)
            & self.label["label_ratio"].round(4).eq(round(fraction, 4))
        ]
        if len(match) != 1:
            raise ValueError(f"Expected one label row for {method}/{fraction}")
        return match.iloc[0]

    def threshold_row(self, threshold: float) -> pd.Series:
        match = self.threshold.loc[
            self.threshold["threshold"].round(4).eq(round(threshold, 4))
        ]
        if len(match) != 1:
            raise ValueError(f"Expected one threshold row for {threshold}")
        return match.iloc[0]

    def ablation_row(self, method: str) -> pd.Series:
        match = self.ablation.loc[self.ablation["Model"].eq(method)]
        if len(match) != 1:
            raise ValueError(f"Expected one ablation row for {method}")
        return match.iloc[0]

    def main_calibration(self, method: str) -> dict[str, float]:
        group = self.main.loc[self.main["method"].eq(method)]
        if group.empty:
            raise ValueError(f"No calibration rows for {method}")
        return {
            "ece_raw": float(group["raw_ece"].mean()),
            "ece_calibrated": float(group["ece"].mean()),
            "nll_raw": float(group["raw_nll"].mean()),
            "nll_calibrated": float(group["nll"].mean()),
            "brier_raw": float(group["raw_brier"].mean()),
            "brier_calibrated": float(group["brier"].mean()),
            "accuracy": float(group["accuracy"].mean()),
        }

    def main_cost(self) -> pd.DataFrame:
        grouped = (
            self.main.groupby(["method", "model_key"], as_index=False)
            .agg(
                parameters=("parameters", "median"),
                gflops_sample=("gflops_sample", "median"),
                training_seconds=("train_time", "median"),
                inference_ms_per_batch=("inference_ms_per_batch", "median"),
                runs=("seed", "size"),
            )
            .sort_values("training_seconds")
        )
        return grouped

    def split_rows(self) -> list[dict[str, Any]]:
        rows = []
        for path in sorted((self.root / "splits").glob("fold*.json")):
            manifest = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "fold": int(manifest["fold"]),
                    "train_conditions": manifest["train_conditions"],
                    "validation_condition": int(manifest["validation_condition"]),
                    "test_condition": int(manifest["test_condition"]),
                    "train_count": int(manifest["train"]["count"]),
                    "validation_count": int(manifest["validation"]["count"]),
                    "test_count": int(manifest["test"]["count"]),
                }
            )
        if len(rows) != 4:
            raise ValueError(f"Expected four split manifests, found {len(rows)}")
        return rows

    def confusion_matrix(self, method: str) -> pd.DataFrame:
        selected = self.main.loc[
            self.main["method"].eq(method)
        ][["run_id"]]
        frames = []
        for run_id in selected["run_id"]:
            path = self.root / "runs" / str(run_id) / "confusion_matrix.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
            frames.append(pd.read_csv(path))
        combined = pd.concat(frames, ignore_index=True)
        counts = (
            combined.groupby(["true_class", "predicted_class"], as_index=False)["count"]
            .sum()
            .pivot(index="true_class", columns="predicted_class", values="count")
            .reindex(index=range(4), columns=range(4), fill_value=0)
            .fillna(0)
        )
        normalized = counts.div(counts.sum(axis=1).replace(0, 1), axis=0)
        normalized.index = CLASS_LABELS
        normalized.columns = CLASS_LABELS
        return normalized


def mean_sd(row: pd.Series, mean_column: str, std_column: str, digits: int = 3) -> str:
    mean = float(row[mean_column])
    std = float(row[std_column])
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"
