"""Build the complete SCI manuscript from canonical experiment outputs."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt

from ..paths import PROJECT_ROOT
from .data import (
    BenchmarkResults,
    CLASS_LABELS,
    EXTERNAL_METHOD_LABELS,
    METHOD_LABELS,
    mean_sd,
)
from .docx_style import (
    add_body,
    add_bullet_items,
    add_caption,
    add_figure,
    add_numbered_items,
    add_page_break,
    add_reference,
    add_small_text,
    add_three_line_table,
    add_title,
    configure_document,
)
from .equations import add_equation


TITLE = (
    "Leakage-Safe Cross-Domain Benchmarking for Bearing Fault Diagnosis: "
    "Time-Frequency Fusion, Calibration, and Self-Training"
)

FIXED_ZIP_TIMESTAMP = (2013, 12, 23, 23, 15, 0)


def normalize_docx_container(path: Path) -> None:
    """Make OOXML ZIP timestamps deterministic without changing part content."""
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.stem}.normalized.",
        suffix=path.suffix,
        delete=False,
    ) as temporary_file:
        normalized_path = Path(temporary_file.name)
    try:
        with ZipFile(path, "r") as source, ZipFile(
            normalized_path,
            "w",
            compression=ZIP_DEFLATED,
            compresslevel=9,
        ) as target:
            for entry in source.infolist():
                payload = source.read(entry)
                normalized = ZipInfo(entry.filename, FIXED_ZIP_TIMESTAMP)
                normalized.compress_type = entry.compress_type
                normalized.external_attr = entry.external_attr
                normalized.create_system = entry.create_system
                normalized.comment = entry.comment
                target.writestr(normalized, payload)
        os.replace(normalized_path, path)
    finally:
        normalized_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the complete SCI manuscript")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=PROJECT_ROOT / "figures" / "publication",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "docs"
        / "paper"
        / "Leakage_Safe_Bearing_Benchmark_SCI_final.docx",
    )
    return parser.parse_args()


def f3(value: float) -> str:
    return f"{float(value):.3f}"


def f2(value: float) -> str:
    return f"{float(value):.2f}"


def p_text(value: float) -> str:
    return "< 0.001" if float(value) < 0.001 else f"= {float(value):.3f}"


def percent(value: float) -> str:
    return f"{100 * float(value):.1f}%"


def standard_result(results: BenchmarkResults, method: str) -> str:
    row = results.summary_row(method)
    return mean_sd(row, "Macro-F1 Mean", "Macro-F1 Std")


def standard_result_from_external(results: BenchmarkResults, method: str) -> str:
    row = results.external_summary.loc[
        results.external_summary["Method"].eq(method)
    ].iloc[0]
    return mean_sd(row, "Macro-F1 Mean", "Macro-F1 Std")


def main_table_rows(results: BenchmarkResults) -> list[list[str]]:
    rows = []
    for method in METHOD_LABELS:
        row = results.summary_row(method)
        rows.append(
            [
                METHOD_LABELS[method],
                mean_sd(row, "Macro-F1 Mean", "Macro-F1 Std"),
                mean_sd(row, "Accuracy Mean", "Accuracy Std"),
                int(row["Runs"]),
            ]
        )
    return rows


def paired_rows(results: BenchmarkResults) -> list[list[str]]:
    rows = []
    for method in (
        "gru_only",
        "cwt_cnn",
        "cwt_gru_no_se",
        "cwt_gru",
        "cwt_gru_temp",
        "cwt_gru_selftrain",
    ):
        row = results.hierarchical_row(f"main:raw_cnn:{method}")
        supported = bool(row["hierarchical_ci_excludes_zero"])
        rows.append(
            [
                METHOD_LABELS[method],
                f"{float(row['mean_delta']):+.3f}",
                f"{float(row['hierarchical_ci_low']):+.3f} to "
                f"{float(row['hierarchical_ci_high']):+.3f}",
                f"p {p_text(row['fold_aware_p'])}",
                "Yes" if supported else "No",
            ]
        )
    return rows


def label_rows(results: BenchmarkResults) -> list[list[str]]:
    rows = []
    budget_labels = {
        0.1: "10% balanced budget",
        0.5: "50% balanced budget",
        1.0: "Full balanced budget",
    }
    for fraction in (0.1, 0.5, 1.0):
        for method in ("cwt_gru", "cwt_gru_selftrain"):
            row = results.label_row(method, fraction)
            rows.append(
                [
                    budget_labels[fraction],
                    "Supervised" if method == "cwt_gru" else "Calibrated self-training",
                    mean_sd(row, "Macro-F1 Mean", "Macro-F1 Std"),
                    mean_sd(row, "Accuracy Mean", "Accuracy Std"),
                    int(row["Runs"]),
                ]
            )
    return rows


def threshold_rows(results: BenchmarkResults) -> list[list[str]]:
    rows = []
    for threshold in (0.3, 0.4, 0.5, 0.6, 0.7):
        row = results.threshold_row(threshold)
        rows.append(
            [
                f"{threshold:.1f}",
                f"{float(row['pseudo_count_mean']):.1f}",
                percent(row["pseudo_coverage_mean"]),
                f"{float(row['pseudo_accuracy_mean']):.3f}",
                f"{float(row['Macro-F1 Mean']):.3f} +/- "
                f"{float(row['Macro-F1 Std']):.3f}",
                int(row["Runs"]),
            ]
        )
    return rows


def calibration_rows(results: BenchmarkResults) -> list[list[str]]:
    rows = []
    for method in ("raw_cnn", "cwt_cnn", "cwt_gru", "cwt_gru_selftrain"):
        values = results.main_calibration(method)
        rows.append(
            [
                METHOD_LABELS[method],
                f"{values['ece_raw']:.3f}",
                f"{values['ece_calibrated']:.3f}",
                f"{values['nll_raw']:.3f}",
                f"{values['nll_calibrated']:.3f}",
                f"{values['brier_raw']:.3f}",
                f"{values['brier_calibrated']:.3f}",
            ]
        )
    return rows


def sota_rows(results: BenchmarkResults) -> list[list[str]]:
    architecture = {
        "raw_cnn": ("Raw waveform", "Three-layer 1D CNN + linear head"),
        "gru_only": ("Raw waveform", "Two-layer bidirectional GRU + linear head"),
        "cwt_cnn": ("CWT magnitude maps", "Residual CNN with SE + linear head"),
        "cwt_gru_no_se": ("Raw + CWT", "Residual image CNN + two-layer bidirectional GRU"),
        "cwt_gru": ("Raw + CWT", "Residual CNN with SE + two-layer bidirectional GRU"),
        "cwt_gru_temp": ("Raw + CWT", "CWT-GRU weights; validation temperature scaling"),
        "cwt_gru_selftrain": ("Raw + CWT", "CWT-GRU; thresholded pseudo-label retraining"),
        "drsn": (
            "Raw waveform",
            "Four-stage residual shrinkage CNN; adaptive soft thresholding",
        ),
        "tfnet": (
            "Raw waveform",
            "Learnable sinusoidal TF convolution + CNN backbone",
        ),
    }
    rows = []
    main = results.main
    for method in METHOD_LABELS:
        summary = results.summary_row(method)
        parameters = int(
            main.loc[main["method"].eq(method), "parameters"].median()
        )
        representation, model = architecture[method]
        rows.append(
            [
                METHOD_LABELS[method],
                representation,
                model,
                f"{parameters:,}",
                f"{float(summary['Macro-F1 Mean']):.3f} +/- "
                f"{float(summary['Macro-F1 Std']):.3f}",
                f"{float(summary['Accuracy Mean']):.3f} +/- "
                f"{float(summary['Accuracy Std']):.3f}",
                int(summary["Runs"]),
            ]
        )
    for method in EXTERNAL_METHOD_LABELS:
        summary = results.external_summary.loc[
            results.external_summary["Method"].eq(method)
        ].iloc[0]
        parameters = int(
            results.external_raw.loc[
                results.external_raw["method"].eq(method), "parameters"
            ].median()
        )
        representation, model = architecture[method]
        rows.append(
            [
                EXTERNAL_METHOD_LABELS[method],
                representation,
                model,
                f"{parameters:,}",
                mean_sd(summary, "Macro-F1 Mean", "Macro-F1 Std"),
                mean_sd(summary, "Accuracy Mean", "Accuracy Std"),
                int(summary["Runs"]),
            ]
        )
    return rows


def ablation_rows(results: BenchmarkResults) -> list[list[str]]:
    rows = []
    for method in (
        "raw_cnn",
        "gru_only",
        "cwt_cnn",
        "cwt_gru_no_se",
        "cwt_gru",
        "cwt_gru_temp",
        "cwt_gru_selftrain",
    ):
        if method == "raw_cnn":
            cwt, gru, se, calibration, self_training = 0, 0, 0, 0, 0
        else:
            row = results.ablation_row(method)
            cwt = int(row["CWT"])
            gru = int(row["GRU"])
            se = int(row["SE"])
            calibration = int(row["Calibration"])
            self_training = int(row["Self-training"])
        summary = results.summary_row(method)
        rows.append(
            [
                METHOD_LABELS[method],
                cwt,
                gru,
                se,
                calibration,
                self_training,
                f"{float(summary['Macro-F1 Mean']):.3f} +/- "
                f"{float(summary['Macro-F1 Std']):.3f}",
                f"{float(summary['Accuracy Mean']):.3f} +/- "
                f"{float(summary['Accuracy Std']):.3f}",
                int(summary["Runs"]),
            ]
        )
    return rows


def complexity_rows(results: BenchmarkResults) -> list[list[str]]:
    cost = results.main_cost().set_index("method")
    rows = []
    for method in (
        "raw_cnn",
        "gru_only",
        "cwt_cnn",
        "cwt_gru_no_se",
        "cwt_gru",
        "cwt_gru_selftrain",
    ):
        row = cost.loc[method]
        rows.append(
            [
                METHOD_LABELS[method],
                f"{int(row['parameters']):,}",
                f"{float(row['gflops_sample']):.3f}",
                f"{float(row['training_seconds']):.1f}",
                f"{float(row['inference_ms_per_batch']):.1f}",
            ]
        )
    return rows


def fold_rows(results: BenchmarkResults) -> list[list[str]]:
    labels = ["C0", "C1", "C2", "C3"]
    rows = []
    for row in results.split_rows():
        rows.append(
            [
                row["fold"],
                ", ".join(labels[index] for index in row["train_conditions"]),
                labels[row["validation_condition"]],
                labels[row["test_condition"]],
                row["train_count"],
                row["validation_count"],
                row["test_count"],
            ]
        )
    return rows


def domain_gap_rows(results: BenchmarkResults) -> list[list[str]]:
    rows = []
    for row in results.domain_gap.itertuples(index=False):
        method = str(row.Model)
        rows.append(
            [
                METHOD_LABELS.get(method, method),
                f"{float(row._1):.3f} +/- {float(row._2):.3f}",
                f"{float(row._3):.3f} +/- {float(row._4):.3f}",
                f"{float(row._5):.3f}",
                f"{float(row._7):.1f}%",
                int(row.Runs),
            ]
        )
    return rows


def label_granularity_rows(results: BenchmarkResults) -> list[list[str]]:
    if results.label_granularity.empty:
        return []
    selected_methods = [
        "raw_cnn",
        "cwt_cnn",
        "cwt_gru_no_se",
        "cwt_gru",
    ]
    labels = {
        "four_class": "Four class",
        "three_class": "Healthy / single / multiple",
        "binary": "Healthy / fault",
    }
    pivot = results.label_granularity.set_index(
        ["method", "granularity"]
    )
    rows = []
    for method in selected_methods:
        row = [METHOD_LABELS[method]]
        for granularity in ("four_class", "three_class", "binary"):
            values = pivot.loc[(method, granularity)]
            row.append(
                f"{float(values['macro_f1_mean']):.3f} +/- "
                f"{float(values['macro_f1_sd']):.3f}"
            )
        rows.append(row)
    return rows


def domain_shift_rows(results: BenchmarkResults) -> list[list[str]]:
    if results.domain_shift.empty:
        return []
    metrics = [
        ("rms", "RMS"),
        ("kurtosis", "Kurtosis"),
        ("crest_factor", "Crest factor"),
        ("spectral_centroid_hz", "Spectral centroid"),
        ("spectral_entropy", "Spectral entropy"),
    ]
    selected = results.domain_shift.loc[
        results.domain_shift["comparison"].eq("train_vs_test")
        & results.domain_shift["channel"].eq("vibration_1")
    ]
    rows = []
    for metric, label in metrics:
        values = selected.loc[selected["metric"].eq(metric)].set_index("fold")[
            "standardized_mean_difference"
        ]
        rows.append(
            [
                label,
                *[f"{float(values.loc[fold]):+.2f}" for fold in range(4)],
            ]
        )
    return rows


def cwt_sensitivity_rows(results: BenchmarkResults) -> list[list[str]]:
    if results.cwt_sensitivity.empty:
        return []
    labels = {
        "baseline": "Baseline 48 scales, 96 x 96, max norm",
        "no_max_norm": "No max normalization",
        "log_magnitude": "Log magnitude",
        "resolution_128": "128 x 128",
        "scales_32": "32 scales",
        "scales_64": "64 scales",
        "real_imag": "Real + imaginary",
    }
    order = list(labels)
    summary = results.cwt_sensitivity.set_index("variant")
    rows = []
    for variant in order:
        if variant not in summary.index:
            continue
        row = summary.loc[variant]
        rows.append(
            [
                labels[variant],
                f"{float(row['macro_f1_mean']):.3f} +/- "
                f"{float(row['macro_f1_sd']):.3f}",
                f"{float(row['mean_validation_macro_f1']):.3f}",
                f"{float(row['mean_best_epoch']):.1f}",
                int(row["runs"]),
            ]
        )
    return rows


def self_training_sensitivity_rows(
    results: BenchmarkResults,
) -> list[list[str]]:
    if results.self_training_sensitivity.empty:
        return []
    labels = {
        "supervised": "Supervised",
        "fixed_threshold_0.60": "Fixed tau = 0.60",
        "coverage_controlled_0.50": "Per-class top 50%",
    }
    budgets = {
        "50_percent": "50% balanced budget",
        "full_balanced": "Full balanced budget",
    }
    rows = []
    for budget in ("50_percent", "full_balanced"):
        for condition in (
            "supervised",
            "fixed_threshold_0.60",
            "coverage_controlled_0.50",
        ):
            selected = results.self_training_sensitivity.loc[
                results.self_training_sensitivity["budget"].eq(budget)
                & results.self_training_sensitivity["condition"].eq(condition)
            ]
            if selected.empty:
                continue
            row = selected.iloc[0]
            rows.append(
                [
                    budgets[budget],
                    labels[condition],
                    f"{float(row['macro_f1_mean']):.3f} +/- "
                    f"{float(row['macro_f1_sd']):.3f}",
                    int(row["runs"]),
                ]
            )
    return rows


def validation_audit_rows(results: BenchmarkResults) -> list[list[str]]:
    rows = []
    if not results.training_budget.empty:
        for row in results.training_budget.itertuples(index=False):
            rows.append(
                [
                    "Training opportunity",
                    METHOD_LABELS.get(str(row.method), str(row.method)),
                    f"{float(row.validation_delta_mean):+.3f} "
                    f"[{float(row.validation_ci_low):+.3f}, "
                    f"{float(row.validation_ci_high):+.3f}]",
                    f"{int(row.validation_improved_runs)}/20",
                    f"{float(row.budget_30_best_epoch_mean):.1f}",
                ]
            )
    if not results.baseline_tuning_comparisons.empty:
        for row in results.baseline_tuning_comparisons.itertuples(index=False):
            rows.append(
                [
                    "Baseline LR tuning",
                    str(row.model_label),
                    f"{float(row.mean_tuned_minus_default):+.3f} "
                    f"[{float(row.hierarchical_ci_low):+.3f}, "
                    f"{float(row.hierarchical_ci_high):+.3f}]",
                    f"{int(row.fold_wins)}/{int(row.fold_wins) + int(row.fold_ties) + int(row.fold_losses)} folds",
                    "--",
                ]
            )
    return rows


def build_references(document: Document) -> None:
    references = [
        'C. Lessmeier, J. K. Kimotho, D. Zimmer, and W. Sextro, "Condition monitoring of bearing damage in electromechanical drive systems by using motor current signals of electric motors: A benchmark data set for data-driven classification," in Proc. European Conf. PHM Soc., vol. 3, no. 1, 2016, doi: 10.36001/phme.2016.v3i1.1577.',
        'J. Hendriks, P. Dumond, and D. A. Knox, "Towards better benchmarking using the CWRU bearing fault dataset," Mech. Syst. Signal Process., vol. 169, Art. no. 108732, Apr. 2022, doi: 10.1016/j.ymssp.2021.108732.',
        'W. A. Smith and R. B. Randall, "Rolling element bearing diagnostics using the Case Western Reserve University data: A benchmark study," Mech. Syst. Signal Process., vol. 64-65, pp. 100-131, 2015.',
        'Z. Wang and T. Oates, "Imaging time-series to improve classification and imputation," in Proc. IJCAI, 2015, pp. 2627-2633.',
        'C. Torrence and G. P. Compo, "A practical guide to wavelet analysis," Bull. Amer. Meteor. Soc., vol. 79, no. 1, pp. 61-78, 1998.',
        'J. Hu, L. Shen, and G. Sun, "Squeeze-and-excitation networks," in Proc. IEEE CVPR, 2018, pp. 7132-7141.',
        'K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in Proc. IEEE CVPR, 2016, pp. 770-778.',
        'K. Cho, B. van Merrienboer, C. Gulcehre, et al., "Learning phrase representations using RNN encoder-decoder for statistical machine translation," in Proc. EMNLP, 2014, pp. 1724-1734.',
        'D.-H. Lee, "Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks," in Proc. ICML Workshop Challenges Representation Learning, 2013.',
        'C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in Proc. ICML, 2017, pp. 1321-1330.',
        'L. Wheat, M. V. Mohrenschildt, S. Habibi, and D. Al-Ani, "Impact of data leakage in vibration signals used for bearing fault diagnosis," IEEE Access, vol. 12, pp. 169879-169895, 2024, doi: 10.1109/ACCESS.2024.3497716.',
        'C. Zhao, E. Zio, and W. Shen, "Domain generalization for cross-domain fault diagnosis: An application-oriented perspective and a benchmark study," Reliab. Eng. Syst. Saf., vol. 245, Art. no. 109964, 2024, doi: 10.1016/j.ress.2024.109964.',
        'S. Alsafari and A. Yafoz, "Few-shot bearing fault diagnosis under joint fault-severity and load shift: A leak-free cross-domain benchmark," Comput. Model. Eng. Sci., vol. 148, no. 1, Art. no. 17, 2026, doi: 10.32604/cmes.2026.084403.',
        'B. Fang, Y. Hu, G. Zheng, X. Zhang, and L. Xie, "Multiscale time-frequency CNN-transformer model for bearing fault diagnosis: A comprehensive feature extraction approach," Eng. Fail. Anal., vol. 184, Art. no. 110312, 2026, doi: 10.1016/j.engfailanal.2025.110312.',
        'B. Wu and G. Yu, "Time-frequency decomposition and convolution strategy for fault diagnosis of rotating machinery," Eng. Fail. Anal., vol. 191, Art. no. 110793, 2026, doi: 10.1016/j.engfailanal.2026.110793.',
        'M. Zhao, S. Zhong, X. Fu, B. Tang, and M. Pecht, "Deep residual shrinkage networks for fault diagnosis," IEEE Trans. Ind. Informat., vol. 16, no. 7, pp. 4681-4690, 2020.',
        'X. Li, J. Wang, J. Wang, J. Wang, J. Chen, and X. Yu, "Research on CNC machine tool spindle fault diagnosis method based on DRSN-GCE model," Algorithms, vol. 18, no. 6, Art. no. 304, 2025, doi: 10.3390/a18060304.',
        'X. Li, J. Chen, J. Wang, et al., "Research on fault diagnosis method of bearings in the spindle system for CNC machine tools based on DRSN-Transformer," IEEE Access, vol. 12, pp. 74586-74595, 2024, doi: 10.1109/ACCESS.2024.3404968.',
        'D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in Proc. ICLR, 2015.',
        'I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in Proc. ICLR, 2019.',
        'Paderborn University, "Bearing Data Center," https://groups.uni-paderborn.de/kat/BearingDataCenter/ (accessed Oct. 27, 2026).',
        'Q. Chen, X. Dong, G. Tu, et al., "TFN: An interpretable neural network with time-frequency transform embedded for intelligent fault diagnosis," Mech. Syst. Signal Process., vol. 207, Art. no. 110952, 2024, doi: 10.1016/j.ymssp.2023.110952.',
    ]
    for index, text in enumerate(references, 1):
        add_reference(document, index, text)


def build_manuscript(
    results: BenchmarkResults,
    figures_dir: Path,
    output: Path,
) -> None:
    document = Document()
    configure_document(document)
    document.core_properties.title = TITLE
    document.core_properties.subject = "Leakage-safe bearing fault diagnosis benchmark"
    document.core_properties.author = "Anonymous"
    document.core_properties.comments = "Generated from canonical CSV results."

    metadata = json.loads(
        (PROJECT_ROOT / "data" / "public" / "paderborn" / "pu_4class_w8192.json").read_text(
            encoding="utf-8"
        )
    )
    config = json.loads(
        (PROJECT_ROOT / "configs" / "paderborn_main.json").read_text(encoding="utf-8")
    )
    total_windows = int(metadata["shape"]["raw"][0])
    epochs = int(config["training"]["epochs"])
    batch_size = int(config["training"]["batch_size"])
    learning_rate = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"]["weight_decay"])
    patience = int(config["training"]["patience"])
    pseudo_rounds = int(config["training"]["pseudo_rounds"])
    pseudo_threshold = float(config["training"]["pseudo_threshold"])
    labeled_counts = results.main["labeled_windows"]
    runs_per_method = int(results.summary_row("raw_cnn")["Runs"])
    seed_count = int(results.main["seed"].nunique())
    fold_count = int(results.main["fold"].nunique())
    heldout_count = len(config["dataset"]["heldout_record_ids"])
    low_budget_self_training = results.raw.loc[
        results.raw["experiment_type"].eq("label_efficiency")
        & results.raw["method"].eq("cwt_gru_selftrain")
        & results.raw["label_ratio"].round(4).eq(0.1)
    ]
    low_budget_zero_updates = int(
        low_budget_self_training["pseudo_count"].fillna(0).eq(0).sum()
    )
    threshold_counts = (
        results.raw.loc[results.raw["experiment_type"].eq("threshold")]
        .groupby(["fold", "seed"], as_index=False)["pseudo_count"]
        .max()
    )
    threshold_zero_update_cases = int(
        threshold_counts["pseudo_count"].fillna(0).eq(0).sum()
    )
    cwt_gru_confusion = results.confusion_matrix("cwt_gru")
    main_run_count = int(results.raw["experiment_type"].eq("main").sum())
    label_run_count = int(
        results.raw["experiment_type"].eq("label_efficiency").sum()
    )
    threshold_run_count = int(
        results.raw["experiment_type"].eq("threshold").sum()
    )
    threshold_level_count = int(
        results.raw.loc[
            results.raw["experiment_type"].eq("threshold"), "threshold"
        ].nunique()
    )
    first_run_config = json.loads(
        (
            results.root
            / "runs"
            / str(results.main.iloc[0]["run_id"])
            / "config.json"
        ).read_text(encoding="utf-8")
    )
    device_name = str(first_run_config["environment"]["device_name"])
    parameter_count = int(
        results.main.loc[results.main["method"].eq("cwt_gru"), "parameters"].median()
    )
    raw_parameter_count = int(
        results.main.loc[results.main["method"].eq("raw_cnn"), "parameters"].median()
    )

    raw = results.hierarchical_row("main:raw_cnn:cwt_cnn")
    gru = results.hierarchical_row("main:raw_cnn:cwt_gru")
    no_se_vs_full = results.hierarchical_row(
        "main:cwt_gru_no_se:cwt_gru"
    )
    mid_update = results.hierarchical_row(
        "label_50:cwt_gru:cwt_gru_selftrain"
    )
    full_update = results.hierarchical_row(
        "label_full:cwt_gru:cwt_gru_selftrain"
    )
    drsn_delta = results.hierarchical_row("external:raw_cnn:drsn")
    tfn_delta = results.hierarchical_row("external:raw_cnn:tfnet")
    coverage_comparison_50 = (
        results.self_training_comparisons.loc[
            results.self_training_comparisons["budget"].eq("50_percent")
            & results.self_training_comparisons["right"].eq(
                "coverage_controlled_0.50"
            )
            & results.self_training_comparisons["left"].eq("supervised")
        ]
        if not results.self_training_comparisons.empty
        else pd.DataFrame()
    )
    coverage_diagnostics_50 = (
        results.pseudo_label_diagnostics.loc[
            results.pseudo_label_diagnostics["budget"].eq("50_percent")
        ]
        if not results.pseudo_label_diagnostics.empty
        else pd.DataFrame()
    )

    add_title(document, TITLE)
    add_small_text(document, "Anonymous manuscript prepared for double-blind review")

    heading = document.add_paragraph()
    heading.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = heading.add_run("Abstract")
    run.bold = True
    run.font.size = Pt(11)
    run.font.name = "Times New Roman"
    add_body(
        document,
        "High in-domain accuracy does not establish that a bearing diagnostic model "
        "transfers to a new physical unit and operating condition. This study asks "
        "whether time-frequency fusion, recurrent modeling, calibration, and "
        "self-training deliver repeatable gains when familiar bearings and "
        "conditions are simultaneously removed. The Paderborn benchmark contains "
        f"{total_windows:,} non-overlapping windows from vibration and two "
        f"phase-current channels. {fold_count} operating-condition folds and "
        f"{seed_count} seeds provide {runs_per_method} matched evaluations per "
        "method; inference resamples folds before seeds. Validation Macro-F1 was "
        f"approximately {f3(results.domain_gap['Validation Macro-F1 Mean'].min())}-"
        f"{f3(results.domain_gap['Validation Macro-F1 Mean'].max())}, whereas test "
        f"Macro-F1 was {f3(results.domain_gap['Test Macro-F1 Mean'].min())}-"
        f"{f3(results.domain_gap['Test Macro-F1 Mean'].max())}. Fold-aware intervals "
        "did not establish a stable advantage for the evaluated magnitude-CWT, "
        "recurrent fusion, or SE variants. Validation-fitted temperature scaling "
        "improved test ECE and NLL for the CWT models but worsened them for Raw "
        "1D-CNN. Fixed-threshold self-training provided no stable improvement. "
        + (
            "A per-class top-50% rule selected candidates with mean offline "
            f"pseudo-label accuracy {float(coverage_diagnostics_50['pseudo_accuracy'].mean()):.3f}, "
            "but its 50% budget difference from supervised training was "
            f"{float(coverage_comparison_50.iloc[0]['mean_delta']):+.3f} "
            f"(fold-aware interval {float(coverage_comparison_50.iloc[0]['hierarchical_ci_low']):+.3f} "
            f"to {float(coverage_comparison_50.iloc[0]['hierarchical_ci_high']):+.3f}). "
            if not coverage_comparison_50.empty
            and not coverage_diagnostics_50.empty
            else "A coverage-controlled alternative was evaluated as a sensitivity analysis. "
        )
        + "When "
        "the same fitted outputs were coarsened from four classes to healthy versus "
        "fault, Macro-F1 increased for the four principal models, indicating that "
        "part of the failure concerned fault-location boundaries rather than only "
        "fault detection. The evidence supports a configuration-specific boundary "
        "for cross-domain claims, not universal failure of these method families.",
        first=True,
    )
    keywords = document.add_paragraph()
    keywords.paragraph_format.space_before = Pt(4)
    keyword_label = keywords.add_run("Keywords: ")
    keyword_label.bold = True
    keyword_label.font.name = "Times New Roman"
    keyword_label.font.size = Pt(9.5)
    keyword_body = keywords.add_run(
        "bearing fault diagnosis; leakage-safe benchmark; domain generalization; "
        "continuous wavelet transform; self-training; probability calibration; "
        "negative results"
    )
    keyword_body.font.name = "Times New Roman"
    keyword_body.font.size = Pt(9.5)
    add_page_break(document)

    document.add_heading("1 Introduction", level=1)
    add_body(
        document,
        "Bearing condition is central to the reliability of rotating machinery. "
        "Deep learning has made end-to-end diagnosis practical by learning "
        "discriminative representations from vibration and current measurements "
        "[1-3]. High diagnostic accuracies are consequently reported frequently. The "
        "interpretation of these values requires care because a fault class can be "
        "predicted from a signature tied to one recording, one sensor installation, "
        "or one operating point. If those source signatures are shared between "
        "training and test data, high accuracy may reflect identification of the "
        "source rather than transferable fault evidence.",
        first=True,
    )
    add_body(
        document,
        "Time-domain networks learn local impulse patterns directly from ordered "
        "waveforms. Their filters are efficient and preserve sample-level timing, but "
        "non-stationary modulation and narrow spectral components can remain difficult "
        "to separate with local time-domain kernels alone. Time-frequency imaging "
        "addresses this issue by exposing scale and frequency structure [4,5]. "
        "Continuous wavelet representations are especially relevant to bearing "
        "impulses because they offer variable resolution across scales. Multiscale "
        "time-frequency convolution and Transformer attention extend this idea by "
        "preserving real and imaginary time-frequency components and by modeling "
        "long-range dependencies [14]. Decomposition-specific convolution strategies "
        "make the same point from a kernel-design perspective [15].",
    )
    add_body(
        document,
        "Temporal and multimodal models add another route to improved diagnosis. "
        "Residual networks stabilize deep image encoders [7], gated recurrent models "
        "represent ordered variation [8], and channel attention can reweight image "
        "features [6]. Combining vibration with motor current provides mechanical and "
        "electrical views of the same drive [1]. Residual shrinkage and Transformer "
        "variants have reported strong performance in noise-oriented and "
        "spindle-specific studies [16-18]. These results motivate fusion architectures. "
        "They do not establish that a recurrent branch, an attention block, or a "
        "second sensor will improve transfer when the test bearing and operating "
        "condition are both new.",
    )
    add_body(
        document,
        "Semi-supervised learning can reduce annotation cost, while probability "
        "calibration can make confidence more useful for maintenance decisions. "
        "Pseudo-labeling trains a model on its own confident predictions [9], and "
        "temperature scaling adjusts probability sharpness without changing the "
        "predicted class [10]. Under distribution shift, however, a confident "
        "prediction may reflect consistency with a familiar source domain rather than "
        "correctness on the target. A fixed confidence threshold can therefore admit "
        "the wrong samples, reject every useful sample, or redistribute labels toward "
        "a subset of classes.",
    )
    add_body(
        document,
        "The central gap is not a shortage of network components. It is a shortage of "
        "evaluations that isolate the claim from the protocol. Many studies use "
        "random-window splits, mix recordings from the same physical bearing across "
        "partitions, or compare numbers produced under different preprocessing and "
        "test settings. Such evidence cannot establish generalization when unseen "
        "bearing identity and unseen operating condition occur together. Benchmark "
        "audits have shown that split construction can materially change reported "
        "performance [2,3], and leakage-safe cross-domain studies have identified "
        "similar risks on Paderborn [11-13]. The central question is therefore not "
        "whether a model can fit the development domains, but whether its apparent "
        "advantage survives the simultaneous removal of familiar bearing identity "
        "and operating condition.",
    )
    add_body(
        document,
        "This study asks whether a time-frequency representation, a recurrent fusion "
        "path, calibrated self-training, and temperature scaling deliver repeatable "
        "diagnostic benefits under a single strict protocol. It evaluates Raw 1D-CNN, "
        "GRU-only, CWT 2D-CNN, CWT-GRU without SE, full CWT-GRU, temperature-scaled "
        "CWT-GRU, and calibrated self-training with the same folds, windows, metrics, "
        "and optimization settings. The test bearing codes are never used for fitting "
        "or validation, and the test operating condition is absent from every fitting "
        "partition.",
        first=True,
    )
    document.add_heading("1.1 Contributions", level=2)
    add_numbered_items(
        document,
        [
            "A leakage-safe protocol that applies simultaneous physical-bearing "
            "and operating-condition holdout to non-overlapping signal windows and "
            "audits source-window identity against restored raw files.",
            "A protocol-controlled comparison of raw waveform, recurrent, "
            "magnitude-CWT, fusion, SE, calibration, self-training, and external "
            "architecture adaptations on identical fold-seed keys.",
            "Fold-first statistical inference that resamples operating-condition "
            "domains before stochastic seeds and reports exact four-fold sign-flip "
            "resolution explicitly.",
            "A cross-domain evidence analysis that combines raw-signal diagnostics, "
            "damage-origin auditing, label-granularity controls, and feature-space "
            "visualization without assigning causality to an unisolated factor.",
        ],
    )

    document.add_heading("2 Related Work", level=1)
    document.add_heading("2.1 Time-frequency fault diagnosis", level=2)
    add_body(
        document,
        "Time-series imaging creates a matrix representation that convolutional "
        "networks can process as an image [4]. Continuous wavelet analysis provides "
        "scale-dependent localization for non-stationary signals [5]. In bearing "
        "diagnosis, this representation can expose impulse repetition and resonance "
        "bands that are less visible in a single time-domain view. Recent work has "
        "moved beyond generic images: Wu and Yu [15] design decomposition-specific "
        "frequency and time kernels, while Fang et al. [14] combine multiscale "
        "time-frequency convolution with Transformer attention.",
    )
    add_body(
        document,
        "These methods solve two real problems: local fault morphology is made "
        "accessible to convolution, and global spectral or temporal context can be "
        "learned through attention. The unresolved question is whether the gain "
        "survives a test unit and operating condition that were not represented during "
        "development. In addition, a magnitude-only image discards phase, and "
        "resampling a large time-frequency field can suppress transient timing. The "
        "present study therefore treats CWT as one representation under test, not as "
        "an assumed improvement. It preserves the exact magnitude-CWT pipeline used "
        "in code and compares it with a raw-waveform model under the same grouped split.",
    )

    document.add_heading("2.2 Temporal and multimodal representation learning", level=2)
    add_body(
        document,
        "Residual connections make deep encoders easier to optimize [7], and gated "
        "recurrent networks model ordered dependencies [8]. These components are "
        "attractive for bearing signals because fault-related impacts occur in time "
        "and interact with speed, load, and machine dynamics. Channel attention such "
        "as squeeze-and-excitation [6] can emphasize informative feature channels. "
        "Paderborn provides synchronized vibration and motor-current channels [1], "
        "supporting a multimodal input without synthesizing a second view.",
    )
    add_body(
        document,
        "Temporal and multimodal learning has improved matched-condition diagnosis "
        "and noise robustness in prior studies [14,16-18]. The remaining limitation is "
        "that added capacity can improve interpolation on familiar recordings while "
        "remaining sensitive to changes in bearing identity. A recurrent layer also "
        "introduces sequential computation over long windows, which may increase "
        "latency even when FLOPs are not the dominant cost. This study asks whether "
        "the implemented two-layer bidirectional GRU adds diagnostic value beyond the "
        "same CWT image encoder and measures the corresponding cost.",
    )

    document.add_heading("2.3 Semi-supervised learning and confidence calibration", level=2)
    add_body(
        document,
        "Pseudo-label self-training uses model predictions as additional supervision "
        "[9]. It can improve data efficiency when unlabeled samples are close to the "
        "labeled distribution. Temperature scaling fits one scalar to logits and "
        "preserves their ordering [10]. It can improve probability calibration without "
        "changing the predicted class. Both procedures are useful, but they answer "
        "different questions: self-training changes training supervision, whereas "
        "temperature scaling changes confidence and probability scores.",
    )
    add_body(
        document,
        "Under domain shift, a fixed confidence threshold may not transfer. Selected "
        "pseudo-labels can concentrate on classes that the initial model already "
        "separates, so retraining can reinforce existing errors. Calibration fitted on "
        "a validation condition can also fail when the test bearing and condition "
        "differ. The unresolved issue is the joint behavior of selection coverage, "
        "pseudo-label accuracy, end-task performance, and calibration under the same "
        "held-out protocol. This work reports all four rather than treating calibration "
        "or pseudo-label accuracy as a proxy for correct diagnosis.",
    )

    document.add_heading("2.4 Leakage-safe evaluation and domain generalization", level=2)
    add_body(
        document,
        "Benchmark audits have shown that segment-level random splitting can share "
        "nearby signal samples or source signatures between training and test sets "
        "[2,3,11]. Hendriks et al. [2] demonstrated that CWRU performance depends on "
        "benchmark construction. Wheat et al. [11] quantified the gap between "
        "run-to-run and part-to-part evaluation on Paderborn. Domain-generalization "
        "benchmarks have also emphasized application-oriented shifts and unseen-domain "
        "testing [12], while recent leak-free bearing studies explicitly separate "
        "physical units and operating regimes [13].",
    )
    add_body(
        document,
        "Prior leakage-aware studies have already shown that split construction "
        "materially changes bearing-diagnosis estimates. Some recent cross-domain "
        "studies also allow a small labeled target-support set for few-shot "
        "adaptation. In contrast, the present benchmark does not expose labels from "
        "the held-out bearing-condition test domain during model fitting, "
        "calibration, pseudo-label generation, threshold analysis, or model "
        "selection.",
    )
    add_body(
        document,
        "The contribution is therefore not the discovery of leakage itself. This "
        "study jointly evaluates representation choice, recurrent fusion, "
        "calibration, self-training, and external architecture adaptations under "
        f"one bearing-disjoint and condition-disjoint protocol with the same "
        f"{fold_count}-fold, {seed_count}-seed keys. The resulting values are "
        "benchmark evidence, not a claim that accuracies from other splits are "
        "directly comparable.",
    )

    document.add_heading("3 Methodology", level=1)
    document.add_heading("3.1 Overall framework", level=2)
    add_body(
        document,
        "Figure 1 summarizes the complete evaluation pipeline. Public multimodal data "
        "are segmented into non-overlapping windows before any split is constructed. "
        "Each fold assigns bearing identities to fitting or testing, then separates a "
        "test operating condition, a validation condition, and two training "
        "conditions. Model selection and temperature fitting use validation data only. "
        "Self-training uses the unlabeled remainder of the training partition. "
        "Diagnostic metrics, calibration, confusion structure, and computational cost "
        "are computed after all fitting is complete.",
        first=True,
    )
    add_figure(
        document,
        figures_dir / "figure_method_01_overall_framework.png",
        "Figure 1. Overall methodology framework. Bearing-code and operating-condition "
        "isolation are enforced before model fitting, calibration, or pseudo-label "
        "selection.",
        width_inches=6.55,
        alt_text="Methodology flow from public multimodal data and leakage-safe folds "
        "through model comparison, validation-only calibration, held-out evaluation, "
        "and statistical analysis.",
    )

    document.add_heading("3.2 Data and leakage-safe protocol", level=2)
    add_body(
        document,
        "The primary dataset is the Paderborn University bearing dataset [1,21]. The "
        f"benchmark uses 16 bearing codes and {fold_count} operating conditions. The input "
        "channels are vibration, phase-current 1, and phase-current 2 at 64 kHz. For "
        "each bearing and condition, measurements 1 and 2 are included. Signals are "
        "segmented into windows of 8192 samples with stride 8192, so no sample is "
        "shared by two windows.",
        first=True,
    )
    add_body(
        document,
        "The four classes are healthy, outer-ring damage, inner-ring damage, and "
        "multiple damage. Training bearings are K001-K004, KA01, KA04, KA08, KI01, "
        "KI04, KI08, KB23, and KB24. K005, KA15, KI14, and KB27 are reserved for test "
        "data and never enter fitting or validation. Operating conditions are denoted "
        "C0 = N09_M07_F10, C1 = N15_M01_F10, C2 = N15_M07_F04, and "
        "C3 = N15_M07_F10.",
    )
    add_three_line_table(
        document,
        [
            "Fold",
            "Training conditions",
            "Validation",
            "Test",
            "Train windows",
            "Validation windows",
            "Test windows",
        ],
        fold_rows(results),
        [1.0, 3.0, 2.0, 1.4, 2.0, 2.5, 1.8],
        "Table 1. Bearing-code-disjoint and condition-disjoint folds. Every fold uses "
        "the same four held-out codes K005, KA15, KI14, and KB27.",
    )
    add_body(
        document,
        f"The cached benchmark contains {total_windows:,} windows. The training counts "
        "vary across folds because one operating condition is used for validation and "
        "another for testing. The held-out bearing codes are excluded from both "
        "training and validation in every fold. Windows are grouped by bearing code, "
        "so a model cannot obtain the same physical unit through a different segment.",
    )
    add_body(
        document,
        "A separate source audit restored the 16 official archives, verified their "
        "recorded SHA-256 values, retained exact measurement filenames and start "
        "indices, and re-standardized the original-amplitude windows. The resulting "
        "raw tensor matched the canonical cache elementwise. The only unreadable "
        "source file is the known truncated `N15_M01_F10_KA08_2.mat`, which the "
        "canonical preprocessing also excludes.",
    )

    document.add_heading("3.3 Signal normalization and CWT representation", level=2)
    add_body(
        document,
        "Each channel is standardized independently inside each window. For channel "
        "x_c(t), the continuous wavelet coefficient at scale a and position b is",
        first=True,
    )
    add_equation(
        document,
        r"W_{x_c}(a,b)=\frac{1}{\sqrt{a}}\int_{-\infty}^{\infty}x_c(t)\psi^{*}\left(\frac{t-b}{a}\right)dt",
        1,
    )
    add_body(
        document,
        "The implementation uses the Morlet wavelet and 48 scales spaced geometrically "
        "from 1 to 64. It stores the coefficient magnitude, resizes the magnitude map "
        "bilinearly to 96 x 96 pixels, and normalizes each channel by its maximum:",
    )
    add_equation(
        document,
        r"I_c(a,b)=\frac{|W_{x_c}(a,b)|}{\max_{a,b}|W_{x_c}(a,b)|+\epsilon}",
        2,
    )
    add_body(
        document,
        "Normalization is performed window by window after standardization. No global "
        "test statistic is used during preprocessing. The resulting tensor has shape "
        "3 x 96 x 96 and is stored as float16; the model casts it to float32 before "
        "convolution.",
    )

    document.add_heading("3.4 CWT-GRU architecture", level=2)
    add_body(
        document,
        "Figure 2 shows the actual CWT-GRU implementation. The CWT branch starts with "
        "a 5 x 5 convolution with 32 output channels, stride 2, batch normalization, "
        "SiLU activation, and 2 x 2 max pooling. Three residual stages then map 32 to "
        "64, 64 to 128 with stride 2, and 128 to 192 with stride 2. Each residual "
        "block uses two 3 x 3 convolutions and an SE gate with reduction ratio 8. "
        "Global average pooling produces a 192-dimensional image feature.",
        first=True,
    )
    add_body(
        document,
        "The waveform branch transposes the 3 x 8192 input into an 8192 x 3 sequence. "
        "A two-layer bidirectional GRU uses 64 hidden units per direction and dropout "
        "0.15. The last forward and backward hidden states are concatenated into a "
        "128-dimensional recurrent feature. Fusion concatenates the 192- and "
        "128-dimensional vectors, applies layer normalization, and maps 320 to 256, "
        "256 to 128, and 128 to four logits. The projection uses SiLU and dropout "
        "rates of 0.25 and 0.15.",
    )
    add_body(
        document,
        f"The complete model contains {parameter_count:,} trainable parameters for "
        "the four-class problem. This count comes from the executed model and is also "
        "recorded in `results/raw_runs.csv`.",
    )
    add_three_line_table(
        document,
        ["Module", "Operation in code", "Output shape"],
        [
            ["CWT input", "Three channel magnitude maps", "3 x 96 x 96"],
            ["Stem", "Conv5x5(32, stride 2) + BN + SiLU + MaxPool2", "32 x 24 x 24"],
            ["Residual stage 1", "Two Conv3x3 + BN + optional SE", "64 x 24 x 24"],
            ["Residual stage 2", "Two Conv3x3, stride 2 + BN + optional SE", "128 x 12 x 12"],
            ["Residual stage 3", "Two Conv3x3, stride 2 + BN + optional SE", "192 x 6 x 6"],
            ["Image feature", "Global average pooling", "192"],
            ["Raw sequence", "Transpose 3 x 8192 to 8192 x 3", "8192 x 3"],
            ["Recurrent feature", "Two-layer bidirectional GRU, hidden 64", "128"],
            ["Fusion classifier", "320 -> 256 -> 128 -> 4", "4 logits"],
        ],
        [2.5, 8.0, 3.3],
        "Table 2. CWT-GRU tensor flow and layer parameters taken from the implementation.",
    )
    add_figure(
        document,
        figures_dir / "figure_method_02_cwt_gru_architecture.png",
        "Figure 2. CWT-GRU architecture with actual channel sizes, hidden dimensions, "
        "attention placement, and fusion projection.",
        width_inches=6.6,
        alt_text="Two-branch CWT-GRU architecture. A residual SE image encoder produces "
        "192 features while a two-layer bidirectional GRU produces 128 features before "
        "a 320-dimensional fusion classifier.",
    )

    document.add_heading("3.5 Supervised training and baselines", level=2)
    add_body(
        document,
        "All models use the same configuration. The Full balanced budget uses the "
        "smallest class count in that fold and therefore contains about 992-996 "
        f"windows ({int(labeled_counts.min())}-{int(labeled_counts.max())} across "
        "folds). This leaves additional training windows available for self-training.",
        first=True,
    )
    add_three_line_table(
        document,
        ["Parameter", "Value"],
        [
            ["Optimizer", "AdamW [19,20]"],
            ["Initial learning rate", f"{learning_rate:g}"],
            ["Weight decay", f"{weight_decay:g}"],
            ["Batch size", str(batch_size)],
            ["Epochs", f"maximum {epochs}"],
            ["Early stopping", f"validation macro-F1, patience {patience}"],
            ["Scheduler", f"cosine annealing over {epochs} epochs"],
            ["Precision", "mixed precision on CUDA"],
            ["Gradient clipping", "maximum norm 5.0"],
            [
                "Input augmentation",
                "raw circular shift up to +/-32 samples and corresponding CWT shift; "
                "50% probability of additive noise",
            ],
            ["GPU", device_name],
        ],
        [5.5, 9.4],
        "Table 3. Training configuration recorded by the canonical experiment runner.",
    )
    add_body(
        document,
        "The recorded execution environment used Python 3.13.5, PyTorch 2.11.0 "
        "with CUDA 13.0, NumPy 2.1.3, pandas 2.2.3, SciPy 1.15.3, scikit-learn "
        "1.6.1, and PyWavelets 1.8.0 on an NVIDIA GeForce RTX 5080. These versions "
        "are recorded in the reproducibility guide and per-run environment "
        "snapshots where available.",
    )
    add_body(
        document,
        "Raw 1D-CNN uses three 1D convolution layers with kernel sizes 25, 9, and 5, "
        "channels 64, 128, and 256, and a four-class linear head. GRU-only uses the "
        "same two-layer bidirectional recurrent encoder and a 128-to-four linear "
        "classifier. CWT 2D-CNN uses the image encoder and a linear classifier. "
        "CWT-GRU without SE uses the same fusion architecture with all SE gates "
        "disabled. Temperature-scaled CWT-GRU reuses the same trained weights and "
        "changes only the probability transformation.",
    )

    document.add_heading("3.6 Temperature scaling", level=2)
    add_body(
        document,
        "For logits z and validation-fitted temperature T, the calibrated probability "
        "of class k is",
        first=True,
    )
    add_equation(
        document,
        r"p_T(y=k\mid x)=\frac{\exp(z_k/T)}{\sum_{j=1}^{K}\exp(z_j/T)}",
        3,
    )
    add_body(
        document,
        "The temperature is optimized on validation logits by LBFGS with cross-entropy "
        "as the objective and a positive range from 0.05 to 20. Test labels are not "
        "used. Because T is positive, the argmax and all classification metrics remain "
        "unchanged. ECE, NLL, and Brier score can change.",
    )

    document.add_heading("3.7 Calibrated self-training", level=2)
    add_body(
        document,
        "Let L be the verified training-label set and U the unlabeled remainder of the "
        "training partition. Validation and test indices are excluded from U. For "
        "predicted class k, the class-conditional candidate set is",
        first=True,
    )
    add_equation(
        document,
        r"\mathcal{S}_k=\operatorname{TopM}\{i\in U:\hat{y}_i=k,\ p_{i,k}\geq\tau\}",
        4,
    )
    add_body(
        document,
        f"where tau = {pseudo_threshold:.2f} in the main experiment, p_i,k is the temperature-scaled "
        "probability, and M is initially twice the number of labeled windows per class. "
        "M doubles in the next round. Figure 3 shows the complete loop. After selection, "
        "pseudo-labels are combined with verified labels and the model is reinitialized. "
        "The weighted training objective is",
    )
    add_equation(
        document,
        r"\mathcal{L}=-\frac{1}{\sum_i w_i}\sum_i w_i\log p_{\theta}(y_i\mid x_i),\quad w_i=1\text{ for verified labels},\quad w_i=p_{i,\hat{y}_i}\text{ for pseudo-labels}",
        5,
    )
    add_body(
        document,
        f"The main self-training condition permits at most {pseudo_rounds} rounds. The threshold "
        "experiment uses one controlled selection-and-retraining round at a 50% "
        "balanced budget so that the threshold comparison is not confounded by a "
        "different number of updates. At a 10% balanced budget, no candidate passed the fixed "
        f"threshold in any of the {runs_per_method} main evaluations, so the procedure returned the "
        "initial model.",
    )
    add_figure(
        document,
        figures_dir / "figure_method_03_self_training_pipeline.png",
        "Figure 3. Calibrated self-training pipeline. The candidate pool is the "
        "unlabeled portion of the training partition; validation and test data are "
        "excluded from label generation.",
        width_inches=6.55,
        alt_text="Self-training workflow from a balanced labeled set and validation-only "
        "temperature fitting through confidence selection, class-balanced caps, "
        "weighted retraining, and independent held-out evaluation.",
    )

    document.add_heading("3.8 Evaluation metrics and statistical analysis", level=2)
    add_body(
        document,
        "Macro-F1 is the primary metric because it weights each class equally. Accuracy "
        "is secondary. The benchmark also records macro precision, macro recall, ECE "
        "with 15 equal-width bins, multiclass NLL, and Brier score. ECE is",
        first=True,
    )
    add_equation(
        document,
        r"\mathrm{ECE}=\sum_{m=1}^{M}\frac{|B_m|}{n}\left|\mathrm{acc}(B_m)-\overline{\mathrm{conf}}(B_m)\right|",
        6,
    )
    add_body(
        document,
        f"Each mean and sample standard deviation describes {runs_per_method} "
        f"matched evaluations from {fold_count} operating-condition folds crossed "
        f"with {seed_count} stochastic seeds. Seeds within a fold share the same "
        "training, validation, and test partitions. The evaluations are therefore "
        "not treated as 20 independent domains. For paired model comparisons, the "
        "primary inferential analysis uses a two-stage cluster bootstrap: "
        "operating-condition folds are resampled first, and seeds are then "
        "resampled within each selected fold. Model differences remain paired at "
        "the fold-seed level, and percentile 95% intervals are computed from "
        "20,000 bootstrap samples.",
    )
    add_body(
        document,
        "As a coarse sensitivity check, a two-sided exact sign-flip test is applied "
        "to the four fold-level mean differences. With only four domain-level "
        "units, the smallest attainable nonzero two-sided p-value is 0.125. This "
        "coarseness is reported explicitly. Unclustered t intervals and two-sided "
        "Wilcoxon signed-rank tests are retained as sensitivity analyses, not as "
        "the primary evidence. A result is not described as an improvement when "
        "the hierarchical interval includes zero.",
    )
    add_body(
        document,
        "The primary benchmark is the predeclared four-class configuration with "
        "window-wise standardization, a 48-scale 96 x 96 magnitude CWT, 12 maximum "
        "epochs, and the fixed pseudo-label threshold tau = 0.60. Normalization, "
        "CWT scale and resolution, magnitude transformation, phase preservation, "
        "label granularity, baseline learning rate, training opportunity, and "
        "coverage-controlled pseudo-labeling are sensitivity analyses. They are "
        "reported in full and do not replace the primary configuration after "
        "observing held-out test performance.",
    )
    add_three_line_table(
        document,
        ["Experiment", "Purpose", "Runs"],
        [
            [
                "Main",
                f"Seven method conditions; {fold_count} folds x {seed_count} seeds; "
                "Full balanced budget",
                str(main_run_count),
            ],
            [
                "Label efficiency",
                "Supervised and self-training at 10% balanced budget and "
                "50% balanced budget; "
                "Full balanced budget reuses main rows",
                str(label_run_count),
            ],
            [
                "Threshold sensitivity",
                f"{threshold_level_count} thresholds at 50% balanced budget; "
                f"{fold_count} folds x {seed_count} seeds",
                str(threshold_run_count),
            ],
            ["Total", "Canonical successful evaluations", str(len(results.raw))],
        ],
        [3.1, 9.0, 2.0],
        "Table 4. Canonical experiment matrix. The Full balanced budget view reuses "
        "the exact main rows and is not counted as a second training experiment.",
    )

    document.add_heading("4 Results by Research Question", level=1)
    document.add_heading(
        "4.1 RQ1: Do time-frequency representations outperform raw waveform models "
        "under leakage-safe cross-domain evaluation?",
        level=2,
    )
    add_body(
        document,
        f"Raw 1D-CNN reached a macro-F1 of {standard_result(results, 'raw_cnn')}. "
        f"CWT 2D-CNN reached {standard_result(results, 'cwt_cnn')}. The mean "
        f"paired difference was {float(raw['mean_delta']):+.3f}, with a "
        "fold-aware hierarchical 95% interval from "
        f"{float(raw['hierarchical_ci_low']):+.3f} to "
        f"{float(raw['hierarchical_ci_high']):+.3f}. The interval includes zero, "
        "so the evaluation does not support a stable CWT advantage. This result "
        "answers RQ1 "
        "negatively for the implemented magnitude-CWT pipeline.",
        first=True,
    )
    add_three_line_table(
        document,
        ["Method", "Macro-F1", "Accuracy", "Runs"],
        main_table_rows(results),
        [7.0, 3.1, 3.1, 1.1],
        "Table 5. Main comparison under the Full balanced budget and "
        "simultaneous unseen-bearing and "
        "unseen-condition testing. Values are mean +/- standard deviation over "
        f"{runs_per_method} "
        "matched evaluations.",
    )
    add_figure(
        document,
        figures_dir / "figure_01_model_comparison.png",
        "Figure 4. Macro-F1 under the Full balanced budget with one "
        "standard-deviation error bars across "
        f"{runs_per_method} matched evaluations per method condition.",
        width_inches=6.25,
        alt_text="Bar chart comparing macro-F1 of seven method conditions under "
        "unseen-bearing and unseen-condition testing.",
    )

    document.add_heading(
        "4.2 RQ2: Does temporal fusion improve CWT-based fault diagnosis?",
        level=2,
    )
    add_body(
        document,
        "The paired CWT-GRU minus CWT 2D-CNN difference was "
        f"{float(results.hierarchical_row('main:cwt_cnn:cwt_gru')['mean_delta']):+.3f}, "
        "with a fold-aware hierarchical 95% interval from "
        f"{float(results.hierarchical_row('main:cwt_cnn:cwt_gru')['hierarchical_ci_low']):+.3f} "
        "to "
        f"{float(results.hierarchical_row('main:cwt_cnn:cwt_gru')['hierarchical_ci_high']):+.3f}. "
        f"The paired CWT-GRU minus Raw 1D-CNN difference was "
        f"{float(gru['mean_delta']):+.3f}, with interval "
        f"{float(gru['hierarchical_ci_low']):+.3f} to "
        f"{float(gru['hierarchical_ci_high']):+.3f}. "
        "GRU-only was lower than Raw 1D-CNN by "
        f"{abs(float(results.hierarchical_row('main:raw_cnn:gru_only')['mean_delta'])):.3f}; "
        "however, its hierarchical interval "
        f"({float(results.hierarchical_row('main:raw_cnn:gru_only')['hierarchical_ci_low']):+.3f} "
        "to "
        f"{float(results.hierarchical_row('main:raw_cnn:gru_only')['hierarchical_ci_high']):+.3f}) "
        "includes zero. The unclustered Wilcoxon sensitivity result "
        f"(p {p_text(results.paired('raw_cnn', 'gru_only')['p_value'])}) and the "
        "fold-aware sign-flip result "
        f"(p {p_text(results.hierarchical_row('main:raw_cnn:gru_only')['fold_aware_p'])}) "
        "therefore disagree in strength. The conservative conclusion is that no "
        "stable recurrent gain was established.",
        first=True,
    )
    add_three_line_table(
        document,
        [
            "Method",
            "Paired delta Macro-F1",
            "Hierarchical 95% interval",
            "Fold-aware p",
            "Stable direction?",
        ],
        paired_rows(results),
        [5.3, 2.3, 3.5, 1.9, 2.2],
        "Table 6. Fold-aware paired comparisons against Raw 1D-CNN on identical "
        "fold-seed keys. Intervals resample folds before seeds. The exact "
        "sign-flip p-value has a minimum attainable nonzero value of 0.125. "
        "Unclustered t intervals and Wilcoxon p-values are retained in "
        "results/statistical_tests.csv as sensitivity analyses.",
    )

    document.add_heading(
        "4.3 RQ3: What is the contribution of each model component?",
        level=2,
    )
    add_body(
        document,
        f"CWT-GRU without SE reached {standard_result(results, 'cwt_gru_no_se')}, "
        f"numerically above full CWT-GRU at {standard_result(results, 'cwt_gru')}. "
        "The paired full-minus-no-SE difference was "
        f"{float(no_se_vs_full['mean_delta']):+.3f}, with a hierarchical interval "
        f"from {float(no_se_vs_full['hierarchical_ci_low']):+.3f} to "
        f"{float(no_se_vs_full['hierarchical_ci_high']):+.3f}. Equivalently, the "
        "no-SE-minus-full difference was "
        f"{float(-no_se_vs_full['mean_delta']):+.3f}, with interval "
        f"{float(-no_se_vs_full['hierarchical_ci_high']):+.3f} to "
        f"{float(-no_se_vs_full['hierarchical_ci_low']):+.3f}. This does not "
        "establish that SE "
        "improves the model. Adding the recurrent branch to CWT-only did not improve "
        "the mean. The complete component matrix is reported in Section 6 and Table "
        "11.",
        first=True,
    )

    document.add_heading(
        "4.4 RQ4: Can calibrated self-training improve diagnostic performance under "
        "limited labels?",
        level=2,
    )
    add_body(
        document,
        f"At a 10% balanced budget, supervised and self-training results are identical in "
        f"{low_budget_zero_updates} evaluations because no pseudo-label passed the "
        "threshold. At a 50% balanced budget, self-training increased the mean "
        f"from {f3(results.label_row('cwt_gru', 0.5)['Macro-F1 Mean'])} to "
        f"{f3(results.label_row('cwt_gru_selftrain', 0.5)['Macro-F1 Mean'])}; the "
        f"paired difference was {float(mid_update['mean_delta']):+.3f}, with a "
        "fold-aware interval from "
        f"{float(mid_update['hierarchical_ci_low']):+.3f} to "
        f"{float(mid_update['hierarchical_ci_high']):+.3f}. At the Full balanced "
        "budget, the paired difference was "
        f"{float(full_update['mean_delta']):+.3f}, with interval "
        f"{float(full_update['hierarchical_ci_low']):+.3f} to "
        f"{float(full_update['hierarchical_ci_high']):+.3f}. Both intervals "
        "include zero. The answer to RQ4 is that the "
        "procedure did not demonstrate a reliable diagnostic improvement.",
        first=True,
    )
    add_three_line_table(
        document,
        ["Label budget", "Method", "Macro-F1", "Accuracy", "Runs"],
        label_rows(results),
        [1.9, 4.8, 3.4, 3.4, 1.2],
        "Table 7. Label-efficiency results. Full balanced budget uses the minimum per-class "
        "count in each fold and is not all training windows.",
    )
    add_figure(
        document,
        figures_dir / "figure_03_label_efficiency.png",
        "Figure 5. Macro-F1 under 10% balanced budget, 50% balanced budget, and "
        "Full balanced budget conditions. "
        "Self-training makes no update at the 10% balanced budget because no "
        "candidate passes tau = 0.60.",
        width_inches=6.0,
        alt_text="Line plot of supervised and calibrated self-training macro-F1 at "
        "three balanced-budget conditions.",
    )

    document.add_heading(
        "4.5 RQ5: How sensitive is self-training to the pseudo-label confidence "
        "threshold?",
        level=2,
    )
    add_body(
        document,
        f"At a 50% balanced budget, mean macro-F1 ranged from "
        f"{f3(results.threshold['Macro-F1 Mean'].min())} to "
        f"{f3(results.threshold['Macro-F1 Mean'].max())}. Pseudo-label accuracy "
        "increased with the threshold, but the end-task response was not monotonic. "
        f"The highest mean occurred at tau = "
        f"{results.threshold.loc[results.threshold['Macro-F1 Mean'].idxmax(), 'threshold']:.1f}, "
        f"with {percent(results.threshold.loc[results.threshold['Macro-F1 Mean'].idxmax(), 'pseudo_coverage_mean'])} "
        f"candidate coverage. {threshold_zero_update_cases} fold-seed cases produced "
        "no pseudo-label at any "
        "tested threshold. Higher selected-label precision therefore did not guarantee "
        "better classification.",
        first=True,
    )
    add_three_line_table(
        document,
        [
            "Threshold",
            "Mean pseudo-labels",
            "Coverage",
            "Pseudo-label accuracy",
            "Test Macro-F1",
            "Runs",
        ],
        threshold_rows(results),
        [1.6, 2.8, 1.9, 2.8, 3.4, 1.1],
        "Table 8. Threshold sensitivity at a 50% balanced budget. Pseudo-label accuracy is an "
        "offline audit and is never used to train or select the threshold.",
    )
    add_figure(
        document,
        figures_dir / "figure_04_threshold_sensitivity.png",
        "Figure 6. Threshold sensitivity. The lower panel shows pseudo-label accuracy, "
        "which rises while end-task macro-F1 remains non-monotonic.",
        width_inches=6.1,
        alt_text="Two-panel threshold sensitivity plot showing test macro-F1 and "
        "pseudo-label accuracy at five confidence thresholds.",
    )

    document.add_heading(
        "4.6 RQ6: Does probability calibration improve classification performance or "
        "only confidence reliability?",
        level=2,
    )
    add_body(
        document,
        "Temperature scaling cannot change the argmax, and the accuracy and macro-F1 "
        "of each CWT-GRU evaluation are therefore identical before and after "
        "temperature fitting. The probability scores change. For CWT-GRU, mean test "
        f"ECE decreased from {results.main_calibration('cwt_gru')['ece_raw']:.3f} to "
        f"{results.main_calibration('cwt_gru')['ece_calibrated']:.3f}, and NLL "
        f"decreased from {results.main_calibration('cwt_gru')['nll_raw']:.3f} to "
        f"{results.main_calibration('cwt_gru')['nll_calibrated']:.3f}. For Raw "
        "1D-CNN, the validation-selected temperature worsened test ECE and NLL, which "
        "shows that an in-domain calibration fit need not transfer to a new bearing "
        "and condition. The answer to RQ6 is that calibration can improve confidence "
        "reliability for some models, but it does not correct classification errors.",
        first=True,
    )
    add_three_line_table(
        document,
        ["Method", "ECE raw", "ECE scaled", "NLL raw", "NLL scaled", "Brier raw", "Brier scaled"],
        calibration_rows(results),
        [4.2, 1.6, 1.8, 1.6, 1.8, 1.7, 1.9],
        "Table 9. Test-set probability metrics before and after validation-fitted "
        "temperature scaling.",
    )
    add_figure(
        document,
        figures_dir / "figure_05_calibration.png",
        "Figure 7. Mean expected calibration error before and after temperature "
        "scaling. Improvements are model- and domain-dependent.",
        width_inches=6.0,
        alt_text="Bar chart of expected calibration error before and after temperature "
        "scaling for four principal models.",
    )
    add_figure(
        document,
        figures_dir / "figure_08_reliability_diagrams.png",
        "Figure 8. Reliability diagrams aggregated from all "
        f"{runs_per_method} Full balanced budget runs. The diagonal represents "
        "perfect calibration.",
        width_inches=6.3,
        alt_text="Four reliability diagrams comparing uncalibrated and temperature-scaled "
        "confidence with empirical accuracy.",
    )

    document.add_heading(
        "5 Re-trained Model and External Baseline Comparison",
        level=1,
    )
    add_body(
        document,
        "A numerical comparison is included here only for models trained by the present "
        "project on the same dataset split, windowing, preprocessing, and metric code. "
        "The reference MTFC-T study [14] uses multiscale time-frequency convolution and "
        "Transformer attention, but its CWRU experiment uses a 70/10/20 split within "
        "one operating load and different datasets for its main comparison. Those "
        "published values are therefore not inserted into this table and are not used "
        "to rank methods across protocols.",
        first=True,
    )
    add_body(
        document,
        "Table 10 therefore reports the re-trained model boundary. Raw 1D-CNN and "
        "GRU-only represent waveform-only baselines. CWT 2D-CNN isolates the "
        "time-frequency image encoder. CWT-GRU without SE and full CWT-GRU isolate "
        "temporal fusion and channel attention. The temperature and self-training rows "
        "describe evaluation or training procedures applied to the same CWT-GRU "
        "architecture, not additional external architectures.",
    )
    add_body(
        document,
        f"Two external architectures were also adapted and retrained for all "
        f"{runs_per_method} fold-seed combinations. DRSN [16] reached "
        f"{standard_result_from_external(results, 'drsn')} macro-F1 and "
        f"{f3(results.external_summary.loc[results.external_summary['Method'].eq('drsn'), 'Accuracy Mean'].iloc[0])} "
        f"accuracy. TFN [22] reached {standard_result_from_external(results, 'tfnet')} "
        f"macro-F1 and "
        f"{f3(results.external_summary.loc[results.external_summary['Method'].eq('tfnet'), 'Accuracy Mean'].iloc[0])} "
        "accuracy. Their paired macro-F1 differences from Raw 1D-CNN were "
        f"{float(drsn_delta['mean_delta']):+.3f} "
        f"(fold-aware 95% interval {float(drsn_delta['hierarchical_ci_low']):+.3f} "
        f"to {float(drsn_delta['hierarchical_ci_high']):+.3f}; "
        f"sign-flip p {p_text(drsn_delta['fold_aware_p'])}) and "
        f"{float(tfn_delta['mean_delta']):+.3f} "
        f"(interval {float(tfn_delta['hierarchical_ci_low']):+.3f} to "
        f"{float(tfn_delta['hierarchical_ci_high']):+.3f}; "
        f"sign-flip p {p_text(tfn_delta['fold_aware_p'])}), respectively. "
        "These are results for the same-protocol adaptations, not "
        "estimates of the performance reported by the original studies.",
    )
    if not results.baseline_tuning_comparisons.empty:
        tuning = results.baseline_tuning_comparisons.set_index("method")
        drsn_tuning = tuning.loc["drsn"]
        tfn_tuning = tuning.loc["tfnet"]
        add_body(
            document,
            "A finite validation-only learning-rate audit compared the "
            "predeclared 0.002 setting with 0.001 for Raw, CWT 2D-CNN, DRSN, and "
            "TFN. Per-fold selection used validation Macro-F1 only. DRSN changed "
            f"by {float(drsn_tuning['mean_tuned_minus_default']):+.3f} "
            f"({float(drsn_tuning['hierarchical_ci_low']):+.3f} to "
            f"{float(drsn_tuning['hierarchical_ci_high']):+.3f}) and TFN changed "
            f"by {float(tfn_tuning['mean_tuned_minus_default']):+.3f} "
            f"({float(tfn_tuning['hierarchical_ci_low']):+.3f} to "
            f"{float(tfn_tuning['hierarchical_ci_high']):+.3f}) on held-out test "
            "Macro-F1. Both intervals included zero. The 160 candidate rows are "
            "retained in full in the result registry.",
        )
    add_body(
        document,
        "MTFC-T was not retrained because the authors' linked repository does not "
        "yet contain its proposed-model implementation. DRSN-Transformer was also "
        "excluded because neither an official implementation nor sufficient "
        "architecture and training detail was available in the project materials "
        "for a defensible replication. No original-study accuracy "
        "was inserted to fill either gap.",
    )
    add_three_line_table(
        document,
        [
            "Method",
            "Input representation",
            "Core architecture",
            "Parameters",
            "Macro-F1",
            "Accuracy",
            "Runs",
        ],
        sota_rows(results),
        [3.6, 2.6, 4.0, 1.6, 2.2, 2.2, 0.8],
        "Table 10. Models eligible for the same-protocol comparison. DRSN and TFN "
        "are three-channel, four-class adaptations under the common training "
        "protocol. No unreproduced external result is included.",
    )

    document.add_heading("6 Ablation Study", level=1)
    add_body(
        document,
        "Table 11 reports the component matrix. Raw 1D-CNN has no CWT, GRU, SE, "
        "calibration, or self-training component. CWT 2D-CNN includes the image "
        "encoder and its SE gates. GRU-only includes the recurrent branch without "
        "CWT. The remaining rows add recurrent fusion, remove or restore SE, and then "
        "apply temperature scaling and self-training.",
        first=True,
    )
    add_three_line_table(
        document,
        [
            "Model",
            "CWT",
            "GRU",
            "SE",
            "Calibration",
            "Self-training",
            "Macro-F1",
            "Accuracy",
            "Runs",
        ],
        ablation_rows(results),
        [3.6, 0.8, 0.8, 0.8, 1.4, 1.5, 2.5, 2.5, 0.8],
        "Table 11. Component-wise ablation under the same leakage-safe split.",
    )
    add_figure(
        document,
        figures_dir / "figure_02_ablation.png",
        "Figure 9. Component ablation. Error bars show one standard deviation over "
        f"{runs_per_method} matched evaluations.",
        width_inches=6.2,
        alt_text="Bar chart of component-wise ablation macro-F1 across six "
        "time-frequency and recurrent architectures.",
    )
    add_body(
        document,
        "On the stored full-minus-no-SE scale, removing SE changed macro-F1 by "
        f"{float(no_se_vs_full['mean_delta']):+.3f}, with a fold-aware interval "
        f"from {float(no_se_vs_full['hierarchical_ci_low']):+.3f} to "
        f"{float(no_se_vs_full['hierarchical_ci_high']):+.3f}. Equivalently, "
        "the no-SE-minus-full contrast is "
        f"{float(-no_se_vs_full['mean_delta']):+.3f}, with interval "
        f"{float(-no_se_vs_full['hierarchical_ci_high']):+.3f} to "
        f"{float(-no_se_vs_full['hierarchical_ci_low']):+.3f}. Both intervals "
        "include zero. The unclustered two-sided Wilcoxon sensitivity was "
        f"p {p_text(results.paired('cwt_gru_no_se', 'cwt_gru')['p_value'])}. This "
        "is a numerical direction, not stable evidence of an SE penalty. Adding "
        "CWT to GRU-only "
        f"changed macro-F1 by "
        f"{float(results.hierarchical_row('main:gru_only:cwt_gru')['mean_delta']):+.3f} "
        "(fold-aware interval "
        f"{float(results.hierarchical_row('main:gru_only:cwt_gru')['hierarchical_ci_low']):+.3f} "
        "to "
        f"{float(results.hierarchical_row('main:gru_only:cwt_gru')['hierarchical_ci_high']):+.3f}). "
        "Adding GRU "
        "to CWT-only changed it by "
        f"{float(results.hierarchical_row('main:cwt_cnn:cwt_gru')['mean_delta']):+.3f} "
        "(fold-aware interval "
        f"{float(results.hierarchical_row('main:cwt_cnn:cwt_gru')['hierarchical_ci_low']):+.3f} "
        "to "
        f"{float(results.hierarchical_row('main:cwt_cnn:cwt_gru')['hierarchical_ci_high']):+.3f}). The"
        " ablation therefore does not support the assumption that every added "
        "component improves cross-domain robustness.",
    )

    document.add_heading("7 Experimental Analysis", level=1)
    document.add_heading("7.1 Class confusion and partial class collapse", level=2)
    add_body(
        document,
        f"Figure 10 aggregates all {runs_per_method} Full balanced budget runs of each principal "
        "model. Healthy windows are recognized relatively well, but damage classes are "
        "strongly confused. CWT-GRU predicts the correct inner-ring class for "
        f"{percent(cwt_gru_confusion.loc['Inner ring', 'Inner ring'])} of inner-ring "
        "windows, while only "
        f"{percent(cwt_gru_confusion.loc['Outer ring', 'Outer ring'])} of outer-ring "
        "windows remain outer-ring and "
        f"{percent(cwt_gru_confusion.loc['Outer ring', 'Inner ring'])} are assigned to "
        "inner-ring. The multiple-damage class is split among healthy, "
        "outer-ring, and inner-ring predictions. This asymmetric error structure is "
        "consistent with partial class collapse and explains why accuracy can exceed "
        "macro-F1.",
        first=True,
    )
    add_figure(
        document,
        figures_dir / "figure_06_confusion_matrices.png",
        "Figure 10. Row-normalized confusion matrices aggregated across "
        f"{fold_count} folds and {seed_count} seeds ({runs_per_method} runs per method).",
        width_inches=6.35,
        alt_text="Four row-normalized confusion matrices for raw CNN, CWT CNN, "
        "CWT-GRU, and calibrated self-training.",
    )
    document.add_heading("7.2 Pseudo-label coverage and threshold behavior", level=2)
    add_body(
        document,
        "The threshold experiment separates selection quality from end-task behavior. "
        "At tau = 0.30, pseudo-label accuracy is "
        f"{float(results.threshold_row(0.3)['pseudo_accuracy_mean']):.3f} and coverage "
        f"is {percent(results.threshold_row(0.3)['pseudo_coverage_mean'])}. At "
        f"tau = 0.70, pseudo-label accuracy rises to "
        f"{float(results.threshold_row(0.7)['pseudo_accuracy_mean']):.3f}, while "
        f"coverage falls to {percent(results.threshold_row(0.7)['pseudo_coverage_mean'])}. "
        "The corresponding mean test macro-F1 is not ordered by threshold precision. "
        "This pattern shows that a fixed threshold operates as a joint selection rule "
        "over class coverage and confidence, not as a simple estimate of label quality.",
        first=True,
    )
    if not results.threshold_per_class.empty:
        threshold_high = results.threshold_per_class.loc[
            np.isclose(
                results.threshold_per_class["threshold"].to_numpy(dtype=float),
                0.7,
            )
        ]
        coverage_columns = [
            column
            for column in results.threshold_per_class.columns
            if column.startswith("coverage_class_")
        ]
        if not threshold_high.empty and coverage_columns:
            values = threshold_high.iloc[0][coverage_columns].to_numpy(dtype=float)
            add_body(
                document,
                "Reconstructing the stored selections reproduced the logged "
                "pseudo-label count for all 100 threshold cells. At tau = 0.70, "
                f"predicted-class coverage ranged from {100 * values.min():.1f}% "
                f"to {100 * values.max():.1f}% across classes. The class-conditional "
                "coverage, rather than only total pseudo-label accuracy, determines "
                "whether a difficult class receives any additional supervision.",
            )
    document.add_heading("7.3 Calibration and proper scoring rules", level=2)
    add_body(
        document,
        "ECE is sensitive to binning, so NLL and multiclass Brier score are reported "
        "alongside it. For CWT 2D-CNN, CWT-GRU, and self-training, the mean test ECE "
        "and NLL decreased after temperature scaling. The Brier score also decreased "
        "for these three rows. Raw 1D-CNN shows the opposite test behavior despite "
        "using the same validation fitting rule. This difference is consistent with a "
        "validation-to-test domain gap rather than a universal failure of temperature "
        "scaling. Predicted classes did not change in any row.",
        first=True,
    )
    document.add_heading("7.4 Computational cost and latency", level=2)
    add_body(
        document,
        f"Table 12 reports medians from the {runs_per_method} main evaluations. "
        "CWT-GRU uses "
        f"{parameter_count:,} "
        "parameters and a median training duration of "
        f"{float(results.main_cost().set_index('method').loc['cwt_gru', 'training_seconds']):.1f} s, "
        "compared with "
        f"{int(results.main_cost().set_index('method').loc['raw_cnn', 'parameters']):,} "
        "parameters and a median training duration of "
        f"{float(results.main_cost().set_index('method').loc['raw_cnn', 'training_seconds']):.1f} s "
        "for Raw 1D-CNN. The recurrent path also has higher inference time per test "
        "batch. These values are implementation- and hardware-dependent, but the "
        "direction is consistent with sequential processing over 8192 time steps. "
        "Self-training increases median training duration because it repeats model "
        "initialization and training.",
        first=True,
    )
    if not results.cost_audit.empty:
        cwt_sample = results.cost_audit.loc[
            results.cost_audit["item"].eq("Serial 3,939-window CWT estimate"),
            "value",
        ]
        if not cwt_sample.empty:
            add_body(
                document,
                "The recorded training times exclude offline CWT generation and "
                "cache loading. A single-process timing sample projected the "
                "canonical 3,939-window CWT generation to approximately "
                f"{float(cwt_sample.iloc[0]):.0f} s before training. The apparent "
                "training-time similarity between cached CWT and raw models must "
                "not be interpreted as lower end-to-end preprocessing cost.",
            )
            add_body(
                document,
                "For recurrent models, profiler counts were supplemented by an "
                "explicit two-layer bidirectional GRU multiply-accumulate formula. "
                "This FLOP estimate excludes data movement, recurrent control "
                "overhead, activation functions, and hardware-specific kernel costs.",
            )
    add_three_line_table(
        document,
        [
            "Method",
            "Parameters",
            "GFLOPs/sample",
            "Training time (s)",
            "Inference (ms/batch)",
        ],
        complexity_rows(results),
        [5.2, 2.1, 2.2, 2.7, 2.8],
        "Table 12. Median computational cost over the "
        f"{runs_per_method} main evaluations. Training "
        "time is complete-run wall time; inference is per test batch of up to 128 "
        "windows on the recorded RTX 5080 environment.",
    )
    add_figure(
        document,
        figures_dir / "figure_07_computational_cost.png",
        "Figure 11. Parameters and median inference time per batch for the principal "
        "architectures.",
        width_inches=6.25,
        alt_text="Two-panel bar chart of parameter count and inference latency for five "
        "model architectures.",
    )

    document.add_heading(
        "7.5 Validation-to-test generalization gap",
        level=2,
    )
    gap_min = results.domain_gap["Absolute Gap"].min()
    gap_max = results.domain_gap["Absolute Gap"].max()
    drop_min = results.domain_gap["Relative Drop (%)"].min()
    drop_max = results.domain_gap["Relative Drop (%)"].max()
    add_body(
        document,
        f"Across Raw 1D-CNN, CWT 2D-CNN, CWT-GRU, and CWT-GRU without SE, "
        f"validation macro-F1 ranged from "
        f"{f3(results.domain_gap['Validation Macro-F1 Mean'].min())} to "
        f"{f3(results.domain_gap['Validation Macro-F1 Mean'].max())}, whereas "
        f"test macro-F1 ranged from "
        f"{f3(results.domain_gap['Test Macro-F1 Mean'].min())} to "
        f"{f3(results.domain_gap['Test Macro-F1 Mean'].max())}. "
        f"The absolute gap was {f3(gap_min)}-{f3(gap_max)}, corresponding to a "
        f"{drop_min:.1f}%-{drop_max:.1f}% relative drop. Table 13 and Figure 12 "
        "summarize these values.",
        first=True,
    )
    add_three_line_table(
        document,
        [
            "Model",
            "Validation Macro-F1",
            "Test Macro-F1",
            "Absolute gap",
            "Relative drop",
            "Runs",
        ],
        domain_gap_rows(results),
        [3.4, 2.8, 2.8, 2.2, 2.2, 1.0],
        "Table 13. Validation-to-test generalization gap under simultaneous "
        "unseen-bearing and unseen-condition shift.",
    )
    add_figure(
        document,
        figures_dir / "figure_09_validation_test_gap.png",
        "Figure 12. Validation and test macro-F1 for the four principal models. "
        "Labels report the mean relative drop from validation to test.",
        width_inches=6.25,
        alt_text="Grouped bar chart of validation and test macro-F1 with standard "
        "deviations and relative drops for four models.",
    )

    document.add_heading(
        "7.6 Domain-shift and label-granularity diagnostics",
        level=2,
    )
    granularity = results.label_granularity
    if not granularity.empty:
        raw_four = granularity.loc[
            granularity["method"].eq("raw_cnn")
            & granularity["granularity"].eq("four_class")
        ].iloc[0]
        raw_binary = granularity.loc[
            granularity["method"].eq("raw_cnn")
            & granularity["granularity"].eq("binary")
        ].iloc[0]
        raw_three = granularity.loc[
            granularity["method"].eq("raw_cnn")
            & granularity["granularity"].eq("three_class")
        ].iloc[0]
        cwt_gru_four = granularity.loc[
            granularity["method"].eq("cwt_gru")
            & granularity["granularity"].eq("four_class")
        ].iloc[0]
        cwt_gru_binary = granularity.loc[
            granularity["method"].eq("cwt_gru")
            & granularity["granularity"].eq("binary")
        ].iloc[0]
    else:
        raw_four = raw_binary = raw_three = cwt_gru_four = cwt_gru_binary = None
    add_body(
        document,
        "The raw-signal audit shows that the held-out domains differ from the "
        "development partitions in amplitude and spectral structure, and that "
        "the direction and magnitude vary by fold. Figure 13 and Table 14 report "
        "standardized descriptive contrasts; they are not causal decompositions. "
        "The held-out fault bearings all carry real lifetime damage, while "
        "artificial and real damage are mixed in the single-location fitting "
        "bearings. Bearing identity, operating condition, and damage origin "
        "therefore change together.",
        first=True,
    )
    if (figures_dir / "figure_13_domain_shift_diagnostics.png").is_file():
        add_figure(
            document,
            figures_dir / "figure_13_domain_shift_diagnostics.png",
            "Figure 13. Raw-signal domain-shift diagnostics for the four held-out "
            "condition folds. Panel (d) reports the standardized train-to-test "
            "difference; it does not assign causality.",
            width_inches=6.3,
            alt_text="Four-panel raw-signal diagnostic with test-bearing RMS, "
            "condition spectral distributions, entropy, and fold-wise effect sizes.",
        )
    if not results.domain_shift.empty:
        add_three_line_table(
            document,
            ["Metric", "Fold 0", "Fold 1", "Fold 2", "Fold 3"],
            domain_shift_rows(results),
            [4.2, 2.1, 2.1, 2.1, 2.1],
            "Table 14. Standardized train-to-test differences for vibration "
            "statistics. Positive values indicate a larger held-out test mean.",
        )
    if raw_four is not None:
        add_body(
            document,
            f"Table 15 uses the fitted four-class outputs without retraining. "
            f"Raw 1D-CNN improves from four-class Macro-F1 "
            f"{float(raw_four['macro_f1_mean']):.3f} to binary healthy-versus-fault "
            f"Macro-F1 {float(raw_binary['macro_f1_mean']):.3f}; the three-class "
            f"view is {float(raw_three['macro_f1_mean']):.3f}. CWT-GRU shows the "
            f"same qualitative pattern, from {float(cwt_gru_four['macro_f1_mean']):.3f} "
            f"to {float(cwt_gru_binary['macro_f1_mean']):.3f}. This sensitivity "
            "distinguishes partial anomaly detection from stable fault-location "
            "identification. It does not change the primary four-class task.",
        )
        add_three_line_table(
            document,
            ["Model", "Four class", "Healthy / single / multiple", "Healthy / fault"],
            label_granularity_rows(results),
            [3.7, 3.1, 4.4, 3.2],
            "Table 15. Post-hoc label-granularity sensitivity. Values are "
            "mean +/- standard deviation over 20 matched evaluations.",
        )
        if (figures_dir / "figure_16_label_granularity.png").is_file():
            add_figure(
                document,
                figures_dir / "figure_16_label_granularity.png",
                "Figure 14. Macro-F1 after post-hoc grouping of the same trained "
                "four-class predictions.",
                width_inches=6.2,
                alt_text="Grouped bar chart comparing four-class, three-class, and "
                "binary healthy-versus-fault Macro-F1 for four principal models.",
            )

    document.add_heading(
        "7.7 Sensitivity analyses and audit controls",
        level=2,
    )
    if not results.cwt_sensitivity.empty:
        add_body(
            document,
            "The CWT controls vary one preprocessing choice at a time while "
            "retaining the same folds, seeds, metric code, model family, and "
            "training budget. All tested settings are reported; no setting is "
            "promoted to the primary method after its test score is observed.",
            first=True,
        )
        add_three_line_table(
            document,
            [
                "CWT representation",
                "Test Macro-F1",
                "Validation Macro-F1",
                "Mean selected epoch",
                "Runs",
            ],
            cwt_sensitivity_rows(results),
            [4.5, 2.8, 2.8, 2.4, 1.2],
            "Table 16. CWT preprocessing sensitivity with CWT 2D-CNN on all "
            "matched fold-seed keys.",
        )
        if (figures_dir / "figure_19_cwt_sensitivity.png").is_file():
            add_figure(
                document,
                figures_dir / "figure_19_cwt_sensitivity.png",
                "Figure 15. Held-out test Macro-F1 for the canonical magnitude CWT "
                "and six controlled representation variants.",
                width_inches=6.2,
                alt_text="Bar chart comparing test Macro-F1 across CWT scale, "
                "resolution, normalization, magnitude, and phase variants.",
            )
    if not results.normalization.empty:
        norm = results.normalization.loc[
            results.normalization["normalization"].eq("B_training_statistics")
        ]
        norm_range = (
            f"{norm['validation_test_gap_mean'].min():.3f}-"
            f"{norm['validation_test_gap_mean'].max():.3f}"
            if not norm.empty
            else "not available"
        )
        add_body(
            document,
            "Training-statistic normalization preserved cross-partition amplitude "
            "information but changed the validation-test gap to "
            f"{norm_range}. It remains a sensitivity condition: the primary "
            "window-wise normalization is not replaced based on test outcome.",
        )
        if not results.normalization_comparisons.empty:
            norm_compare = results.normalization_comparisons.set_index("method")
            raw_norm = norm_compare.loc["raw_cnn"]
            cwt_gru_norm = norm_compare.loc["cwt_gru"]
            add_body(
                document,
                "For Raw 1D-CNN, training-statistic normalization changed test "
                f"macro-F1 by {float(raw_norm['mean_delta_normalization_b_minus_a']):+.3f}, "
                "with a fold-aware interval from "
                f"{float(raw_norm['hierarchical_ci_low']):+.3f} to "
                f"{float(raw_norm['hierarchical_ci_high']):+.3f} and "
                f"{int(raw_norm['fold_wins_b'])} of 4 fold means in its favor. "
                "For CWT-GRU, the corresponding difference was "
                f"{float(cwt_gru_norm['mean_delta_normalization_b_minus_a']):+.3f} "
                f"(interval {float(cwt_gru_norm['hierarchical_ci_low']):+.3f} to "
                f"{float(cwt_gru_norm['hierarchical_ci_high']):+.3f}). "
                "The primary window-wise standardization is retained because "
                "changing it after test observation would change the benchmark.",
            )
    if not results.self_training_sensitivity.empty:
        add_body(
            document,
            "A coverage-controlled rule selects the highest-confidence 50% within "
            "each predicted class, avoiding an absolute confidence threshold. Its "
            "selection-count and class-conditional accuracy diagnostics are saved "
            "for every run. Table 17 compares it with supervised training and the "
            "fixed tau = 0.60 procedure on identical keys.",
        )
        add_three_line_table(
            document,
            ["Label budget", "Training procedure", "Test Macro-F1", "Runs"],
            self_training_sensitivity_rows(results),
            [3.3, 3.8, 3.5, 1.1],
            "Table 17. Coverage-controlled pseudo-labeling compared with the "
            "predeclared fixed-threshold procedure and supervised training.",
        )
        if (figures_dir / "figure_17_self_training_sensitivity.png").is_file():
            add_figure(
                document,
                figures_dir / "figure_17_self_training_sensitivity.png",
                "Figure 16. Self-training sensitivity at the 50% and Full balanced "
                "budgets. Error bars show one standard deviation over 20 matched runs.",
                width_inches=6.2,
                alt_text="Two-panel bar chart comparing supervised, fixed-threshold, "
                "and coverage-controlled self-training Macro-F1.",
            )
    if not results.training_budget.empty or not results.baseline_tuning.empty:
        add_three_line_table(
            document,
            [
                "Audit",
                "Model",
                "Validation-only difference",
                "Improvement",
                "Mean selected epoch",
            ],
            validation_audit_rows(results),
            [3.0, 3.1, 4.5, 2.0, 2.0],
            "Table 18. Training-opportunity and baseline learning-rate audits. "
            "Intervals resample folds before seeds where reported.",
        )
    if not results.training_budget.empty:
        best_epoch_max = int(results.training_budget["budget_30_best_epoch_max"].max())
        cap_runs = int(results.training_budget["budget_30_cap_selected_runs"].sum())
        validation_wins = int(
            results.training_budget["validation_improved_runs"].sum()
        )
        add_body(
            document,
            "In the 30-epoch opportunity, the latest selected epoch across all "
            f"four architectures was {best_epoch_max}, and no run selected the "
            f"30th epoch (cap-selected runs: {cap_runs}). Validation Macro-F1 "
            f"improved in {validation_wins} of 80 matched checks, but every "
            "held-out test interval for the 30-versus-12 difference included zero. "
            "The 12-epoch primary protocol was therefore retained rather than "
            "changing the benchmark after observing test outcomes.",
        )
    if not results.feature_space.empty:
        add_body(
            document,
            "Post-hoc penultimate embeddings were inspected with PCA and t-SNE "
            "after model fitting. The saved diagnostics compare neighbour agreement "
            "for fault class, physical bearing, and condition. These labels are "
            "used for visualization and offline interpretation only and do not "
            "enter feature optimization or model selection.",
        )
        add_body(
            document,
            "Across folds, domain silhouette ranged from "
            f"{float(results.feature_space['domain_silhouette'].min()):+.3f} to "
            f"{float(results.feature_space['domain_silhouette'].max()):+.3f}, "
            "and held-out test class silhouette ranged from "
            f"{float(results.feature_space['test_class_silhouette'].min()):.3f} to "
            f"{float(results.feature_space['test_class_silhouette'].max()):.3f}. "
            "Nearest-neighbour agreement was higher for fault class "
            f"({float(results.feature_space['nearest_same_class_fraction'].mean()):.3f} "
            "mean) than for bearing identity "
            f"({float(results.feature_space['nearest_same_bearing_fraction'].mean()):.3f}) "
            "or condition "
            f"({float(results.feature_space['nearest_same_condition_fraction'].mean()):.3f}). "
            "The embeddings are therefore not organized as clean train/test domain "
            "clusters, but the low held-out class silhouette shows that class "
            "organization remains weak within the unseen domain.",
        )
        if (figures_dir / "figure_22_feature_space_pca.png").is_file():
            add_figure(
                document,
                figures_dir / "figure_22_feature_space_pca.png",
                "Figure 17. PCA of CWT-GRU penultimate embeddings for train, "
                "validation, and held-out test windows under each fold.",
                width_inches=6.2,
                alt_text="Four PCA panels showing train, validation, and test "
                "penultimate embeddings for the four held-out folds.",
            )
        if (figures_dir / "figure_23_feature_space_tsne.png").is_file():
            add_figure(
                document,
                figures_dir / "figure_23_feature_space_tsne.png",
                "Figure 18. t-SNE of one representative fold, coloured by fault "
                "class and held-out bearing identity after fitting.",
                width_inches=6.2,
                alt_text="Two t-SNE panels of CWT-GRU embeddings coloured by fault "
                "class and by held-out bearing identity.",
            )

    document.add_heading("8 Discussion", level=1)
    document.add_heading("8.1 Complexity is not the same as robustness", level=2)
    add_body(
        document,
        "The main comparison does not show that the complicated model is better. The "
        "CWT 2D-CNN differs from Raw 1D-CNN by "
        f"{abs(float(raw['mean_delta'])):.3f} macro-F1, and the CWT-GRU "
        "mean is lower than both the raw model and the CWT-only model. The no-SE "
        "variant has the highest observed mean, but its fold-aware interval is "
        "centered near zero. This pattern is consistent with a distinction between "
        "representational capacity on the training distribution and invariance to a "
        "new bearing and operating condition. More capacity can fit domain-specific "
        "cues without acquiring the fault mechanism that transfers.",
        first=True,
    )
    add_body(
        document,
        "The two external adaptations did not exceed the simpler baselines "
        "under the joint holdout. This observation does not invalidate their "
        "published results, which were obtained with different datasets, splits, "
        "input channels, and training recipes. It indicates that architecture-level "
        "complexity alone did not transfer to this stricter evaluation.",
    )
    add_body(
        document,
        "Table 13 and Figure 12 show substantially higher validation macro-F1 than "
        "test macro-F1 for every principal model. The held-out test scores therefore "
        "cannot be explained merely by an inability to fit the development domains. "
        "The performance loss coincides with the joint change in bearing identity "
        "and operating condition. This pattern is consistent with joint domain "
        "shift, but the design cannot isolate the causal contributions of bearing "
        "identity, operating condition, and damage-origin differences, nor can it "
        "identify a single failure mechanism.",
    )

    document.add_heading(
        "8.2 Potential information loss in the implemented CWT representation",
        level=2,
    )
    add_body(
        document,
        "The implemented representation takes the absolute CWT coefficient and "
        "bilinearly resizes the field to 96 x 96. Phase is discarded, and the resize "
        "can attenuate short transients, impact timing, and narrow sidebands. The raw "
        "waveform retains the ordered samples, although local convolutional filters "
        "still need to discover the relevant scales. The result does not show that "
        "time-frequency analysis is intrinsically inferior. It shows that a "
        "magnitude-only image and a generic image classifier are not automatically "
        "better than a waveform model under this distribution shift.",
        first=True,
    )
    if not results.cwt_sensitivity_comparisons.empty:
        vs_baseline = results.cwt_sensitivity_comparisons.loc[
            results.cwt_sensitivity_comparisons["variant"].eq("scales_32")
            & results.cwt_sensitivity_comparisons["reference"].eq(
                "baseline_cwt_48"
            )
        ]
        vs_raw = results.cwt_sensitivity_comparisons.loc[
            results.cwt_sensitivity_comparisons["variant"].eq("scales_32")
            & results.cwt_sensitivity_comparisons["reference"].eq("raw_1d_cnn")
        ]
        if not vs_baseline.empty and not vs_raw.empty:
            add_body(
                document,
                "The CWT controls make this boundary more specific. A 32-scale "
                "representation changed test Macro-F1 relative to the primary "
                "48-scale representation by "
                f"{float(vs_baseline.iloc[0]['mean_delta']):+.3f}, with a "
                f"fold-aware interval from "
                f"{float(vs_baseline.iloc[0]['hierarchical_ci_low']):+.3f} to "
                f"{float(vs_baseline.iloc[0]['hierarchical_ci_high']):+.3f} and "
                f"{int(vs_baseline.iloc[0]['fold_wins'])} of 4 fold means in its "
                "favor. Relative to Raw 1D-CNN, however, its difference was "
                f"{float(vs_raw.iloc[0]['mean_delta']):+.3f}, with interval "
                f"{float(vs_raw.iloc[0]['hierarchical_ci_low']):+.3f} to "
                f"{float(vs_raw.iloc[0]['hierarchical_ci_high']):+.3f} and "
                f"{int(vs_raw.iloc[0]['fold_wins'])} of 4 fold means in its favor. "
                "The scale choice can change CWT performance, but it did not "
                "establish a stable advantage over the raw-waveform baseline, and "
                "the 32-scale variant was not promoted after inspecting test data.",
            )
    add_body(
        document,
        "The MTFC-T reference provides a useful design contrast [14]. Its TFCLayer "
        "retains both real and imaginary components before multiscale convolution, "
        "and its Transformer attention explicitly models long-range dependencies. "
        "Those choices address part of the phase and resolution limitation identified "
        "here. The present study did not implement MTFC-T, so its reported results are "
        "not ranked against the present benchmark.",
    )

    document.add_heading("8.3 Recurrent fusion adds cost without a stable gain", level=2)
    add_body(
        document,
        f"CWT-GRU uses {parameter_count / raw_parameter_count:.1f} times as many "
        "parameters as Raw 1D-CNN and has a higher "
        "median training and inference cost. Its paired macro-F1 interval against "
        "Raw 1D-CNN is centered near zero. The recurrent branch traverses 8192 time "
        "steps and therefore exposes limited parallelism compared with local "
        "convolution. The observation is an empirical result for this "
        "implementation, not a claim that every temporal model must fail.",
        first=True,
    )

    document.add_heading("8.4 Self-training risks confirmation bias", level=2)
    add_body(
        document,
        "At a 10% balanced budget, no pseudo-label passed tau = 0.60 in all "
        f"{low_budget_zero_updates} evaluations. At "
        "the 50% balanced and Full balanced budgets, selected pseudo-label accuracy is high "
        "in many runs, yet the fold-aware intervals for diagnostic change include "
        "zero. The selection rule uses the model's own calibrated confidence, so it "
        "is most likely to admit samples "
        "that agree with the current decision geometry. The asymmetric confusion "
        "matrix shows that this geometry is already biased toward particular classes. "
        "A class-balanced cap limits count imbalance but does not create information "
        "about systematically confused classes. The observed pattern is consistent "
        "with confirmation bias rather than a demonstrated improvement in fault "
        "representation.",
        first=True,
    )
    if not results.self_training_sensitivity.empty:
        coverage = results.self_training_sensitivity.loc[
            results.self_training_sensitivity["condition"].eq(
                "coverage_controlled_0.50"
            )
            & results.self_training_sensitivity["budget"].eq("50_percent")
        ]
        comparison = results.self_training_comparisons.loc[
            results.self_training_comparisons["budget"].eq("50_percent")
            & results.self_training_comparisons["right"].eq(
                "coverage_controlled_0.50"
            )
            & results.self_training_comparisons["left"].eq("supervised")
        ]
        diagnostics = results.pseudo_label_diagnostics.loc[
            results.pseudo_label_diagnostics["budget"].eq("50_percent")
        ]
        if (
            not coverage.empty
            and not comparison.empty
            and not diagnostics.empty
        ):
            add_body(
                document,
                "The per-class top-50% rule selected approximately "
                f"{percent(diagnostics['pseudo_coverage'].mean())} of candidates "
                "with offline pseudo-label accuracy "
                f"{float(diagnostics['pseudo_accuracy'].mean()):.3f}. Despite this "
                "high precision, the end-task difference from supervised training "
                f"was {float(comparison.iloc[0]['mean_delta']):+.3f}, with a "
                f"fold-aware interval from "
                f"{float(comparison.iloc[0]['hierarchical_ci_low']):+.3f} to "
                f"{float(comparison.iloc[0]['hierarchical_ci_high']):+.3f}. "
                "The result reinforces the distinction between selection accuracy "
                "and transferable end-task benefit.",
            )
            full_diagnostics = results.pseudo_label_diagnostics.loc[
                results.pseudo_label_diagnostics["budget"].eq("full_balanced")
            ]
            if not full_diagnostics.empty:
                zero_multiple = int(
                    full_diagnostics["selected_class_3"].fillna(0).eq(0).sum()
                )
                add_body(
                    document,
                    "At the Full balanced budget, the coverage rule selected no "
                    f"multiple-damage pseudo-label in {zero_multiple} of "
                    f"{len(full_diagnostics)} runs. Coverage is therefore "
                    "class-conditional in practice even when a global fraction is "
                    "requested.",
                )

    document.add_heading("8.5 Calibration changes confidence, not decisions", level=2)
    add_body(
        document,
        "Temperature scaling is monotonic and therefore preserves the argmax. For the "
        "CWT models, lower test ECE and NLL indicate more reliable probabilities. For "
        "Raw 1D-CNN, the validation-selected temperature worsened both test measures. "
        "A lower ECE for one model and domain should not be described as a repaired "
        "classifier. Calibration and self-training must be reported separately because "
        "one changes confidence and the other changes the training labels.",
        first=True,
    )
    if not results.calibration_summary.empty:
        calibration = results.calibration_summary.set_index("model")
        raw_lower = (
            int(calibration.loc["raw_cnn", "lower_bound_runs"])
            if "raw_cnn" in calibration.index
            else 0
        )
        cwt_rows = calibration.loc[
            calibration.index.intersection(
                ["cwt_cnn", "cwt_gru", "cwt_gru_no_se", "cwt_gru_selftrain"]
            )
        ]
        add_body(
            document,
            f"The Raw 1D-CNN fitted temperature reached the lower bound 0.05 in "
            f"{raw_lower} of 20 runs, whereas the four CWT model conditions had "
            f"mean fitted temperatures between {float(cwt_rows['temperature_mean'].min()):.2f} "
            f"and {float(cwt_rows['temperature_mean'].max()):.2f}. Per-fold "
            "reliability diagrams show that aggregate improvement does not imply "
            "uniform calibration across held-out domains.",
        )

    document.add_heading("8.6 Toward a reproducible reporting standard", level=2)
    add_body(
        document,
        "The benchmark supports eight concrete reporting requirements: non-overlapping "
        "windows; bearing-code-disjoint fitting and testing; explicit operating-domain "
        "holdout; source-file and preprocessing provenance; at least five stochastic "
        "seeds per operating-condition fold with fold-aware paired statistics; "
        "class-conditional pseudo-label coverage as well as aggregate accuracy; "
        "separate reports of classification and calibration; and an explicit "
        "damage-origin manifest. It also supports an auditable distinction between "
        "numerical improvement and statistical evidence.",
    )

    document.add_heading("9 Conclusion", level=1)
    add_body(
        document,
        "This study asked whether time-frequency imaging, temporal fusion, "
        "temperature scaling, and self-training improve bearing diagnosis when a new "
        "bearing and a new operating condition are held out together. Under the "
        "Paderborn protocol, the study did not establish stable improvements: fold-aware intervals for "
        "CWT and CWT-GRU against Raw 1D-CNN include zero in macro-F1, and intervals "
        "for self-training include zero at the tested balanced label budgets. "
        "Same-protocol DRSN and TFN adaptations had lower mean macro-F1, although "
        "the four held-out condition folds limit inferential precision. These results "
        "do not establish a universal failure of time-frequency or recurrent methods. "
        "They establish a stricter boundary for claims based on random-window "
        "benchmarks. Sensitivity analyses show that representation scale and label "
        "granularity can change the observed score, while neither a stronger "
        "pseudo-label selector nor a longer training budget produced a stable "
        "end-task gain in the evaluated matrix.",
        first=True,
    )
    add_body(
        document,
        "The study has several limitations. The Paderborn test rig is not a complete "
        "machine-tool spindle assembly, and the held-out physical units are limited to "
        f"{heldout_count} codes. Only four independent operating-condition folds are "
        "available, which limits inferential resolution. The benchmark uses one "
        "window length, one primary CWT scale range, one principal model family, and "
        "a small validation set. The external DRSN "
        "and TFN implementations are adaptations under the common protocol, so "
        "their results do not estimate performance under the original papers' "
        "datasets or training recipes. The held-out fault bearings are "
        "lifetime-damaged while several training bearings are artificially damaged, "
        "so the test coincides with changes in damage origin, bearing identity, and "
        "condition. These factors cannot be separated by the primary design. No "
        "reviewed second dataset met the simultaneous unit-disjoint and "
        "condition-disjoint criteria, so the conclusions apply to the evaluated "
        "configurations and this Paderborn protocol. The same four fixed test folds "
        "were reused across the broader model-comparison project. This supports a "
        "run-level leakage-free statement, but not a claim that the test set remained "
        "cognitively untouched by every human design decision.",
    )
    add_body(
        document,
        "Future work should test phase-preserving or complex-valued time-frequency "
        "representations, multi-scale kernels that retain real and imaginary "
        "components, target-aware domain adaptation with strictly guarded test data, "
        "physically informed bearing models, and validation on complete spindle "
        "assemblies and naturally degraded bearings. Self-training should be paired "
        "with coverage control and an independent audit of class-conditional label "
        "quality.",
    )

    document.add_heading("Appendix A Additional Diagnostic Figures", level=1)
    supplementary_figures = [
        (
            "figure_14_per_fold_reliability.png",
            "Figure S1. Per-fold reliability diagrams for CWT-GRU before and "
            "after validation-fitted temperature scaling.",
            "Four reliability diagrams that separate the four held-out condition folds.",
        ),
        (
            "figure_15_temperature_distribution.png",
            "Figure S2. Per-run validation-fitted temperature distributions for "
            "the principal calibrated models.",
            "Box plots of fitted temperature for four calibrated model conditions.",
        ),
        (
            "figure_20_foldwise_paired_differences.png",
            "Figure S3. Fold-level paired differences for the principal comparisons.",
            "Dot plot of fold-level paired Macro-F1 differences with zero reference.",
        ),
        (
            "figure_21_training_budget.png",
            "Figure S4. Validation and held-out test changes from 12 to 30 maximum "
            "training epochs.",
            "Paired bar plots for validation and test changes in four architectures.",
        ),
        (
            "figure_24_normalization_sensitivity.png",
            "Figure S5. Window-wise and training-statistic normalization "
            "sensitivity.",
            "Three-panel comparison of test Macro-F1, validation-test gap, and "
            "relative drop for three model families.",
        ),
    ]
    for filename, caption, alt_text in supplementary_figures:
        path = figures_dir / filename
        if path.is_file():
            add_figure(
                document,
                path,
                caption,
                width_inches=6.2,
                alt_text=alt_text,
            )

    document.add_heading("Data and Code Availability", level=1)
    add_body(
        document,
        "The Paderborn data are available from the official Bearing Data Center [21]. "
        "The accompanying project archive contains the preprocessing commands, "
        "canonical configuration, split manifests, per-run configurations, all 320 "
        "result rows and 40 external-baseline retraining rows, automated statistics, "
        "separate preprocessing and training sensitivity rows, method figures, and "
        "manuscript-generation script. Raw signals, large caches, and model weights "
        "are not redistributed.",
        first=True,
    )

    document.add_heading("References", level=1)
    build_references(document)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent,
        prefix=f".{output.stem}.",
        suffix=output.suffix,
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        document.save(temporary_path)
        normalize_docx_container(temporary_path)
        try:
            os.replace(temporary_path, output)
        except PermissionError:
            # Some Windows viewers retain a read handle that blocks rename
            # replacement while still permitting the native CopyFile API.
            if os.name != "nt":
                raise
            if not ctypes.windll.kernel32.CopyFileW(
                str(temporary_path),
                str(output),
                False,
            ):
                raise ctypes.WinError()
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    results = BenchmarkResults(args.results_dir)
    build_manuscript(results, args.figures_dir.resolve(), args.output.resolve())
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
