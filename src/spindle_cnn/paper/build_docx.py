"""Build the complete SCI manuscript from canonical experiment outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt

from ..paths import PROJECT_ROOT
from .data import BenchmarkResults, CLASS_LABELS, METHOD_LABELS, mean_sd
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
    "Leakage-Safe Benchmarking of Time-Frequency Fusion and Calibrated "
    "Self-Training for Bearing Fault Diagnosis"
)


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
        / "Leakage_Safe_Bearing_Benchmark_SCI.docx",
    )
    return parser.parse_args()


def f3(value: float) -> str:
    return f"{float(value):.3f}"


def f2(value: float) -> str:
    return f"{float(value):.2f}"


def p_text(value: float) -> str:
    return "<0.001" if float(value) < 0.001 else f"{float(value):.3f}"


def percent(value: float) -> str:
    return f"{100 * float(value):.1f}%"


def standard_result(results: BenchmarkResults, method: str) -> str:
    row = results.summary_row(method)
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
        row = results.paired("raw_cnn", method)
        rows.append(
            [
                METHOD_LABELS[method],
                f"{float(row['mean_paired_delta']):+.3f}",
                f"{float(row['paired_delta_ci95_low']):+.3f} to "
                f"{float(row['paired_delta_ci95_high']):+.3f}",
                p_text(row["p_value"]),
            ]
        )
    return rows


def label_rows(results: BenchmarkResults) -> list[list[str]]:
    rows = []
    for fraction in (0.1, 0.5, 1.0):
        for method in ("cwt_gru", "cwt_gru_selftrain"):
            row = results.label_row(method, fraction)
            rows.append(
                [
                    f"{fraction * 100:.0f}%",
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

    raw = results.paired("raw_cnn", "cwt_cnn")
    gru = results.paired("raw_cnn", "cwt_gru")
    no_se_vs_full = results.paired("cwt_gru_no_se", "cwt_gru")
    mid_update = results.tests.loc[
        results.tests["experiment_type"].eq("label_efficiency")
        & results.tests["label_ratio"].round(4).eq(0.5)
        & results.tests["reference"].eq("cwt_gru")
    ].iloc[0]
    full_update = results.tests.loc[
        results.tests["experiment_type"].eq("label_efficiency")
        & results.tests["label_ratio"].round(4).eq(1.0)
        & results.tests["reference"].eq("cwt_gru")
    ].iloc[0]

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
        "Deep-learning bearing diagnostics often report high accuracy, but the "
        "estimate can depend on overlapping windows, repeated physical units, and "
        "test information entering model development. This study evaluates "
        "time-frequency fusion and calibrated self-training as a diagnostic "
        "benchmark rather than a leaderboard contest. The Paderborn dataset supplies "
        f"{total_windows:,} non-overlapping windows from vibration and two phase-current "
        f"channels. {fold_count} grouped folds combine unseen bearing codes with unseen "
        f"operating conditions, and {seed_count} seeds produce {runs_per_method} "
        "matched evaluations per method. "
        f"Raw 1D-CNN reached a mean macro-F1 of {f3(results.summary_row('raw_cnn')['Macro-F1 Mean'])}, "
        f"whereas CWT 2D-CNN reached {f3(results.summary_row('cwt_cnn')['Macro-F1 Mean'])} "
        f"and CWT-GRU reached {f3(results.summary_row('cwt_gru')['Macro-F1 Mean'])}. "
        "No CWT-containing model produced a statistically significant gain over the "
        "raw waveform baseline. The highest mean was obtained without SE, but removing "
        f"SE did not produce a significant difference from full CWT-GRU (p={p_text(no_se_vs_full['p_value'])}). "
        "At a 10% class-balanced label budget, no pseudo-label passed the threshold in "
        f"any of the {runs_per_method} evaluations. Self-training did not improve macro-F1 significantly "
        f"at 50% (paired delta {float(mid_update['mean_paired_delta']):+.3f}, "
        f"p={p_text(mid_update['p_value'])}) or 100% "
        f"({float(full_update['mean_paired_delta']):+.3f}, p={p_text(full_update['p_value'])}). "
        "Temperature scaling left predicted classes unchanged. It improved ECE and NLL "
        "for the CWT models on the held-out test folds, whereas the validation-selected "
        "temperature worsened both measures for Raw 1D-CNN. The results show that "
        "architectural complexity and confidence calibration should be evaluated "
        "separately from classification accuracy under domain shift.",
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
        "similar risks on Paderborn [11-13].",
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
            "A leakage-safe bearing benchmark with non-overlapping windows, "
            "bearing-code-disjoint fitting and testing, operating-condition holdout, "
            f"and {runs_per_method} matched fold-seed evaluations per condition.",
            "A controlled comparison of waveform, image, recurrent, attention, and "
            "temperature-scaling conditions that reports unfavorable and "
            "non-significant findings rather than selecting a favorable run.",
            "A separate analysis of thresholded self-training and probability "
            "calibration, including pseudo-label coverage, pseudo-label accuracy, "
            "class confusion, proper scoring rules, parameters, FLOPs, and latency.",
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
        "These studies solve the first part of the problem by showing that split "
        "construction changes the meaning of a score. They leave two questions open: "
        "how several representation choices behave under one identical grouped split, "
        "and how self-training plus calibration behave when test labels cannot inform "
        "selection. This study extends the bearing-code-disjoint principle with "
        "simultaneous operating-condition holdout and evaluates all methods with the "
        f"same {fold_count}-fold, {seed_count}-seed keys. The resulting values are "
        "benchmark evidence, not a "
        "claim that any random-window accuracy is directly comparable.",
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
        "All models use the same configuration. The 100% main condition means 100% of "
        "the class-balanced label budget, not all training windows. The budget is the "
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
        "different number of updates. At 10% labels, no candidate passed the fixed "
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
        "Each reported mean is accompanied by a sample standard deviation across "
        f"{runs_per_method} "
        "matched fold-seed evaluations. A 95% confidence interval uses the t "
        "distribution. Models compared on the same fold and seed are paired. "
        "Two-sided Wilcoxon signed-rank tests are used when at least five matched "
        "pairs are available. A non-significant result is not described as an "
        "improvement.",
    )
    add_three_line_table(
        document,
        ["Experiment", "Purpose", "Runs"],
        [
            [
                "Main",
                f"Seven method conditions; {fold_count} folds x {seed_count} seeds; "
                "100% balanced labels",
                str(main_run_count),
            ],
            [
                "Label efficiency",
                "Supervised and self-training at 10% and 50%; 100% reuses main rows",
                str(label_run_count),
            ],
            [
                "Threshold sensitivity",
                f"{threshold_level_count} thresholds at 50% labels; "
                f"{fold_count} folds x {seed_count} seeds",
                str(threshold_run_count),
            ],
            ["Total", "Canonical successful evaluations", str(len(results.raw))],
        ],
        [3.1, 9.0, 2.0],
        "Table 4. Canonical experiment matrix. The 100% label-efficiency view reuses "
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
        f"CWT 2D-CNN reached {standard_result(results, 'cwt_cnn')}. The paired "
        f"difference was {float(raw['mean_paired_delta']):+.3f} with p="
        f"{p_text(raw['p_value'])} and a 95% interval from "
        f"{float(raw['paired_delta_ci95_low']):+.3f} to "
        f"{float(raw['paired_delta_ci95_high']):+.3f}. The mean CWT estimate is "
        "almost identical to the raw baseline, but the interval includes zero and the "
        "test does not support a stable CWT advantage. This result answers RQ1 "
        "negatively for the implemented magnitude-CWT pipeline.",
        first=True,
    )
    add_three_line_table(
        document,
        ["Method", "Macro-F1", "Accuracy", "Runs"],
        main_table_rows(results),
        [7.0, 3.1, 3.1, 1.1],
        "Table 5. Full-label comparison under simultaneous unseen-bearing and "
        "unseen-condition testing. Values are mean +/- standard deviation over "
        f"{runs_per_method} "
        "matched evaluations.",
    )
    add_figure(
        document,
        figures_dir / "figure_01_model_comparison.png",
        "Figure 4. Full-label macro-F1 with one standard-deviation error bars across "
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
        f"The paired CWT-GRU minus CWT 2D-CNN difference was "
        f"{float(results.paired('cwt_cnn', 'cwt_gru')['mean_paired_delta']):+.3f} "
        f"(p={p_text(results.paired('cwt_cnn', 'cwt_gru')['p_value'])}). A separate "
        f"comparison of CWT-GRU with Raw 1D-CNN gave "
        f"{float(gru['mean_paired_delta']):+.3f} (p={p_text(gru['p_value'])}). "
        "GRU-only was lower than Raw 1D-CNN by "
        f"{abs(float(results.paired('raw_cnn', 'gru_only')['mean_paired_delta'])):.3f} "
        f"(p={p_text(results.paired('raw_cnn', 'gru_only')['p_value'])}). The "
        "recurrent-only model therefore did not replace either local convolutional "
        "representation, and adding the recurrent path to the CWT image encoder did "
        "not produce a repeatable diagnostic gain.",
        first=True,
    )
    add_three_line_table(
        document,
        ["Method", "Paired delta Macro-F1", "95% interval", "Wilcoxon p"],
        paired_rows(results),
        [6.1, 3.2, 3.4, 1.7],
        "Table 6. Paired comparisons against Raw 1D-CNN on identical fold and seed "
        "keys. Positive values indicate a higher method mean than the raw baseline.",
    )

    document.add_heading(
        "4.3 RQ3: What is the contribution of each model component?",
        level=2,
    )
    add_body(
        document,
        f"CWT-GRU without SE reached {standard_result(results, 'cwt_gru_no_se')}, "
        f"numerically above full CWT-GRU at {standard_result(results, 'cwt_gru')}. "
        f"The paired no-SE minus full difference was "
        f"{abs(float(no_se_vs_full['mean_paired_delta'])):.3f} "
        f"(p={p_text(no_se_vs_full['p_value'])}). This does not establish that SE "
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
        f"At 10% labels, supervised and self-training results are identical in "
        f"{low_budget_zero_updates} evaluations because no pseudo-label passed the "
        "threshold. At 50%, self-training increased the mean "
        f"from {f3(results.label_row('cwt_gru', 0.5)['Macro-F1 Mean'])} to "
        f"{f3(results.label_row('cwt_gru_selftrain', 0.5)['Macro-F1 Mean'])}; the "
        f"paired difference was {float(mid_update['mean_paired_delta']):+.3f} "
        f"(p={p_text(mid_update['p_value'])}). At 100%, the paired difference was "
        f"{float(full_update['mean_paired_delta']):+.3f} "
        f"(p={p_text(full_update['p_value'])}). None of the non-zero mean changes "
        "is statistically distinguishable from zero. The answer to RQ4 is that the "
        "procedure did not demonstrate a reliable diagnostic improvement.",
        first=True,
    )
    add_three_line_table(
        document,
        ["Label budget", "Method", "Macro-F1", "Accuracy", "Runs"],
        label_rows(results),
        [1.9, 4.8, 3.4, 3.4, 1.2],
        "Table 7. Label-efficiency results. The 100% setting is the class-balanced "
        "budget, not all training windows.",
    )
    add_figure(
        document,
        figures_dir / "figure_03_label_efficiency.png",
        "Figure 5. Macro-F1 at 10%, 50%, and 100% class-balanced label budgets. "
        "Self-training makes no update at 10% because no candidate passes tau = 0.60.",
        width_inches=6.0,
        alt_text="Line plot of supervised and calibrated self-training macro-F1 at "
        "three class-balanced label budgets.",
    )

    document.add_heading(
        "4.5 RQ5: How sensitive is self-training to the pseudo-label confidence "
        "threshold?",
        level=2,
    )
    add_body(
        document,
        f"At a 50% label budget, mean macro-F1 ranged from "
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
        "Table 8. Threshold sensitivity at 50% labels. Pseudo-label accuracy is an "
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
        f"{runs_per_method} full-label runs. The diagonal represents perfect calibration.",
        width_inches=6.3,
        alt_text="Four reliability diagrams comparing uncalibrated and temperature-scaled "
        "confidence with empirical accuracy.",
    )

    document.add_heading("5 Re-trained Model Comparison and SOTA Interface", level=1)
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
        "architecture, not additional external SOTA models.",
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
        "Table 10. Models eligible for the same-protocol comparison. No unreproduced "
        "external result is included.",
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
        f"Removing SE changed full CWT-GRU macro-F1 by "
        f"{abs(float(no_se_vs_full['mean_paired_delta'])):.3f} in favor of the no-SE "
        f"variant, with p={p_text(no_se_vs_full['p_value'])}. This is a numerical "
        "direction, not evidence of a significant SE penalty. Adding CWT to GRU-only "
        f"changed macro-F1 by "
        f"{float(results.paired('gru_only', 'cwt_gru')['mean_paired_delta']):+.3f} "
        f"(p={p_text(results.paired('gru_only', 'cwt_gru')['p_value'])}). Adding GRU "
        "to CWT-only changed it by "
        f"{float(results.paired('cwt_cnn', 'cwt_gru')['mean_paired_delta']):+.3f} "
        f"(p={p_text(results.paired('cwt_cnn', 'cwt_gru')['p_value'])}). The"
        " ablation therefore does not support the assumption that every added "
        "component improves cross-domain robustness.",
    )

    document.add_heading("7 Experimental Analysis", level=1)
    document.add_heading("7.1 Class confusion and partial class collapse", level=2)
    add_body(
        document,
        f"Figure 10 aggregates all {runs_per_method} full-label runs of each principal "
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

    document.add_heading("8 Discussion", level=1)
    document.add_heading("8.1 Complexity is not the same as robustness", level=2)
    add_body(
        document,
        "The main comparison does not show that the complicated model is better. The "
        "CWT 2D-CNN differs from Raw 1D-CNN by "
        f"{abs(float(raw['mean_paired_delta'])):.3f} macro-F1, and the CWT-GRU "
        "mean is lower than both the raw model and the CWT-only model. The no-SE "
        "variant has the highest observed mean, but its paired difference from the raw "
        "model is not significant. This pattern supports a distinction between "
        "representational capacity on the training distribution and invariance to a "
        "new bearing and operating condition. More capacity can fit domain-specific "
        "cues without acquiring the fault mechanism that transfers.",
        first=True,
    )
    add_body(
        document,
        "The validation-to-test gap is large. Raw 1D-CNN, CWT 2D-CNN, and CWT-GRU "
        f"have mean validation macro-F1 values of "
        f"{results.main.loc[results.main['method'].eq('raw_cnn'), 'best_val_macro_f1'].mean():.3f}, "
        f"{results.main.loc[results.main['method'].eq('cwt_cnn'), 'best_val_macro_f1'].mean():.3f}, and "
        f"{results.main.loc[results.main['method'].eq('cwt_gru'), 'best_val_macro_f1'].mean():.3f}, "
        "while their test means are "
        f"{f3(results.summary_row('raw_cnn')['Macro-F1 Mean'])}, "
        f"{f3(results.summary_row('cwt_cnn')['Macro-F1 Mean'])}, and "
        f"{f3(results.summary_row('cwt_gru')['Macro-F1 Mean'])}. A model can therefore "
        "learn the seen bearings and still fail to generalize to the held-out code and "
        "condition.",
    )

    document.add_heading("8.2 The implemented CWT representation loses information", level=2)
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
        "median training and inference cost. Its mean macro-F1 is not higher by a "
        "statistically distinguishable margin. The recurrent branch traverses 8192 "
        "time steps and therefore exposes limited parallelism compared with local "
        "convolution. The observation is an empirical result for this implementation, "
        "not a claim that every temporal model must fail.",
        first=True,
    )

    document.add_heading("8.4 Self-training risks confirmation bias", level=2)
    add_body(
        document,
        "At 10% labels, no pseudo-label passed tau = 0.60 in all "
        f"{low_budget_zero_updates} evaluations. At "
        "50% and 100%, selected pseudo-label accuracy is high in many runs, yet the "
        "mean diagnostic change is small and non-significant. The selection rule uses "
        "the model's own calibrated confidence, so it is most likely to admit samples "
        "that agree with the current decision geometry. The asymmetric confusion "
        "matrix shows that this geometry is already biased toward particular classes. "
        "A class-balanced cap limits count imbalance but does not create information "
        "about systematically confused classes. The observed pattern is consistent "
        "with confirmation bias rather than a demonstrated improvement in fault "
        "representation.",
        first=True,
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

    document.add_heading("8.6 Toward a reproducible reporting standard", level=2)
    add_body(
        document,
        "The benchmark supports six concrete reporting requirements: non-overlapping "
        "windows; bearing-code-disjoint fitting and testing; explicit operating-domain "
        "holdout; at least five independent seeds with paired statistics; separate "
        "reports of pseudo-label coverage and accuracy; and separate reports of "
        "classification and calibration. It also supports an auditable distinction "
        "between numerical improvement and statistical evidence.",
    )

    document.add_heading("9 Conclusion", level=1)
    add_body(
        document,
        "This study asked whether time-frequency imaging, temporal fusion, "
        "temperature scaling, and self-training improve bearing diagnosis when a new "
        "bearing and a new operating condition are held out together. The answer from "
        "the five-seed Paderborn benchmark is no stable improvement: CWT and CWT-GRU "
        "are statistically indistinguishable from Raw 1D-CNN in macro-F1, and "
        "self-training does not improve the mean significantly at the tested label "
        "budgets. These results do not establish a universal failure of "
        "time-frequency or recurrent methods. They establish a stricter boundary for "
        "claims based on random-window benchmarks.",
        first=True,
    )
    add_body(
        document,
        "The study has several limitations. The Paderborn test rig is not a complete "
        "machine-tool spindle assembly, and the held-out physical units are limited to "
        f"{heldout_count} codes. The benchmark uses one window length, one CWT scale "
        "range, one "
        "model family, and a small validation set. External SOTA models were not "
        "inserted because they were not re-trained under the identical protocol. The "
        "held-out fault bearings are lifetime-damaged while several training bearings "
        "are artificially damaged, so the test combines damage-origin changes with "
        "bearing and condition shifts.",
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

    document.add_heading("Data and Code Availability", level=1)
    add_body(
        document,
        "The Paderborn data are available from the official Bearing Data Center [21]. "
        "The accompanying project archive contains the preprocessing commands, "
        "canonical configuration, split manifests, per-run configurations, all 320 "
        "result rows, automated statistics, method figures, and manuscript-generation "
        "script. Raw signals, large caches, and model weights are not redistributed.",
        first=True,
    )

    document.add_heading("References", level=1)
    build_references(document)

    output.parent.mkdir(parents=True, exist_ok=True)
    document.save(output)


def main() -> None:
    args = parse_args()
    results = BenchmarkResults(args.results_dir)
    build_manuscript(results, args.figures_dir.resolve(), args.output.resolve())
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
