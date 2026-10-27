import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..paths import OUTPUTS_DIR


DEFAULT_VERSIONS = [
    ("gafd_gru_swin_cbam", "v1_baseline"),
    ("gafd_gru_swin_cbam_v2", "v2_balanced_labels"),
    ("gafd_gru_swin_cbam_v3", "v3_exclusive_margin"),
    ("gafd_gru_swin_cbam_v2_focal_sampler", "v4_focal_sampler"),
    ("gafd_gru_swin_cbam_v4_confidence_proto_hier", "v5_conf_proto_hier"),
    ("gafd_gru_swin_cbam_v6_confidence_only", "v6_confidence_only"),
    ("gadf_swin_gam_fusion", "v7_gadf_swin_gam"),
    ("cwt_residual_attention", "v8_cwt_residual_attention"),
    ("cwt_resnet18", "v9_cwt_resnet18"),
    ("cwt_vgg11_bn", "v10_cwt_vgg11_bn"),
    ("cwt_residual_attention_plus", "v11_cwt_residual_attention_plus"),
    ("cwt_residual_attention_plus_morl_only", "v12_v11_morl_only"),
    ("cwt_residual_attention_plus_morl_mexh", "v13_v11_morl_mexh"),
    ("cwt_residual_attention_plus_stage2", "v14_v11_stage2"),
    ("cwt_residual_attention_plus_cgau4", "v15_v11_cgau4"),
    ("cwt_residual_attention_plus_gaus8", "v16_v11_gaus8"),
    ("cwt_residual_attention_plus_gaus8_ce_no_sampler", "v17_v16_ce_no_sampler"),
    ("cwt_residual_attention_plus_gaus8_focal_no_sampler", "v18_v16_focal_no_sampler"),
    ("cwt_residual_attention_plus_gaus8_ce_plain", "v19_v16_ce_plain"),
    ("cwt_residual_attention_plus_gaus8_ce_sampler", "v20_v16_ce_sampler"),
    ("cwt_residual_attention_plus_gaus8_tta_only", "v21_v16_tta_only"),
    ("cwt_residual_attention_plus_gaus8_tta_calibrated", "v22_v16_tta_vector_scaling"),
    ("cwt_residual_attention_plus_gaus8_temperature_scaling", "v23_v16_temperature_scaling"),
    ("cwt_residual_attention_plus_gaus8_isotonic_ovr", "v24_v16_isotonic_ovr"),
    ("cwt_residual_attention_plus_gaus8_op_vector_scaling", "v25_v16_op_vector_scaling"),
    ("cwt_residual_attention_plus_gaus8_uniform_cover_robust_conf", "v26_v16_uniform_cover_robust_conf"),
    ("cwt_residual_attention_plus_gaus8_uniform_cover_robust_conf_tta_vector_scaling", "v27_v26_tta_vector_scaling"),
    ("cwt_residual_attention_plus_gaus8_uniform_cover_robust_conf_op_vector_scaling", "v28_v26_op_vector_scaling"),
    ("cwt_residual_attention_plus_gaus8_op_vector_scaling_local_bias", "v29_v25_local_bias"),
    ("cwt_residual_attention_plus_gaus8_op_vector_scaling_local_bias_v2", "v30_v29_local_bias_v2"),
    ("cwt_residual_attention_plus_gaus8_op_vector_scaling_local_bias_v3a", "v31_v30_local_bias_v3"),
    ("cwt_residual_attention_plus_gaus8_v32a_affine_calibrator", "v32a_v31_frozen_learnable_calibrator"),
    ("cwt_residual_attention_plus_gaus8_v32b_condition_calibrator", "v32b_v32a_condition_aware_calibrator"),
    ("cwt_residual_attention_plus_gaus8_v32c_condition_confusion_calibrator", "v32c_v32b_confusion_aware"),
    ("cwt_residual_attention_plus_gaus8_v32c_lambda05", "v33_v32c_lambda05"),
    ("cwt_residual_attention_plus_gaus8_v32c_key469", "v34_v32c_key469"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总多个实验版本的指标与图表对比")
    parser.add_argument(
        "--outputs-root",
        default=str(OUTPUTS_DIR),
        help="实验输出根目录",
    )
    parser.add_argument(
        "--save-dir",
        default=str(OUTPUTS_DIR / "version_comparison"),
        help="版本对比图输出目录",
    )
    return parser.parse_args()


def normalize_path(path_value: str) -> Path:
    path_str = str(path_value)
    if len(path_str) >= 3 and path_str[1] == ":" and path_str[2] in ("\\", "/") and Path("/").exists():
        drive = path_str[0].lower()
        suffix = path_str[3:].replace("\\", "/")
        return Path(f"/mnt/{drive}/{suffix}")
    return Path(path_str)


def read_best_val_f1(output_dir: Path) -> float | None:
    history_path = output_dir / "training_history.csv"
    if not history_path.exists():
        return None
    history_df = pd.read_csv(history_path)
    if "val_f1_macro" not in history_df.columns or history_df.empty:
        return None
    return float(history_df["val_f1_macro"].max())


def read_active_test_classes(output_dir: Path) -> int | None:
    label_path = output_dir / "label_distribution.csv"
    if not label_path.exists():
        return None
    label_df = pd.read_csv(label_path)
    test_df = label_df[(label_df["dataset_split"] == "test") & (label_df["count"] > 0)]
    return int(test_df["label_name"].nunique())


def read_summary_row(output_dir: Path, display_name: str) -> dict:
    summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
    metrics = summary["metrics"]
    return {
        "version": display_name,
        "output_dir": str(output_dir),
        "accuracy": float(metrics["accuracy"]),
        "precision_macro": float(metrics["precision_macro"]),
        "recall_macro": float(metrics["recall_macro"]),
        "f1_macro": float(metrics["f1_macro"]),
        "f1_weighted": float(metrics["f1_weighted"]),
        "best_val_f1_macro": read_best_val_f1(output_dir),
        "test_active_classes": read_active_test_classes(output_dir),
        "pseudo_label_mode": summary.get("pseudo_label_mode", "legacy"),
        "loss_name": summary.get("loss_name", "cross_entropy"),
        "weighted_sampler": bool(summary.get("use_weighted_sampler", False)),
        "confidence_power": summary.get("confidence_power", ""),
        "coarse_loss_weight": summary.get("coarse_loss_weight", ""),
        "prototype_loss_weight": summary.get("prototype_loss_weight", ""),
        "postprocess_calibration_method": summary.get("postprocess_calibration_method", ""),
        "postprocess_group_column": summary.get("postprocess_group_column", ""),
    }


def build_note(row: pd.Series) -> str:
    notes = []
    if row["test_active_classes"] != 10:
        notes.append("test classes < 10")
    if row["pseudo_label_mode"] != "legacy":
        notes.append(f"labels={row['pseudo_label_mode']}")
    if row["loss_name"] != "cross_entropy":
        notes.append(f"loss={row['loss_name']}")
    if row["weighted_sampler"]:
        notes.append("sampler=on")
    if row["prototype_loss_weight"] not in ("", 0, 0.0):
        notes.append(f"proto={row['prototype_loss_weight']}")
    if row["coarse_loss_weight"] not in ("", 0, 0.0):
        notes.append(f"hier={row['coarse_loss_weight']}")
    if row["postprocess_calibration_method"]:
        calib_note = f"calib={row['postprocess_calibration_method']}"
        if row["postprocess_group_column"]:
            calib_note += f"@{row['postprocess_group_column']}"
        notes.append(calib_note)
    return ", ".join(notes) if notes else "baseline"


def main() -> None:
    args = parse_args()
    outputs_root = normalize_path(args.outputs_root)
    save_dir = normalize_path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for folder_name, display_name in DEFAULT_VERSIONS:
        output_dir = outputs_root / folder_name
        summary_path = output_dir / "run_summary.json"
        if summary_path.exists():
            rows.append(read_summary_row(output_dir, display_name))

    if not rows:
        raise FileNotFoundError("未找到任何可用的 run_summary.json，无法生成版本对比图。")

    compare_df = pd.DataFrame(rows)
    compare_df["note"] = compare_df.apply(build_note, axis=1)
    compare_df.to_csv(save_dir / "version_metrics_comparison.csv", index=False)

    plot_df = compare_df.melt(
        id_vars=["version"],
        value_vars=["accuracy", "recall_macro", "f1_macro", "f1_weighted"],
        var_name="metric",
        value_name="value",
    )

    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(18, 12))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.2])

    metric_order = ["accuracy", "recall_macro", "f1_macro", "f1_weighted"]
    titles = {
        "accuracy": "Accuracy",
        "recall_macro": "Macro Recall",
        "f1_macro": "Macro F1",
        "f1_weighted": "Weighted F1",
    }
    for idx, metric_name in enumerate(metric_order):
        ax = fig.add_subplot(grid[idx // 2, idx % 2])
        metric_df = plot_df[plot_df["metric"] == metric_name]
        sns.barplot(
            data=metric_df,
            x="version",
            y="value",
            hue="version",
            legend=False,
            ax=ax,
            palette="viridis",
        )
        ax.set_title(titles[metric_name])
        ax.set_xlabel("")
        ax.set_ylabel(metric_name)
        ax.tick_params(axis="x", rotation=20)
        for patch, value in zip(ax.patches, metric_df["value"]):
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                patch.get_height() + 0.005,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    text_ax = fig.add_subplot(grid[2, :])
    text_ax.axis("off")
    lines = ["Version Summary", ""]
    for _, row in compare_df.sort_values("f1_macro", ascending=False).iterrows():
        best_val_text = "N/A" if pd.isna(row["best_val_f1_macro"]) else f"{row['best_val_f1_macro']:.3f}"
        lines.append(
            f"{row['version']}: test_f1_macro={row['f1_macro']:.3f}, "
            f"best_val_f1={best_val_text}, active_test_classes={row['test_active_classes']}, {row['note']}"
        )
    text_ax.text(
        0.01,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
        bbox={"facecolor": "#f7f7f7", "edgecolor": "#d0d0d0", "boxstyle": "round,pad=0.6"},
    )

    fig.suptitle("Spindle CNN Version Comparison Dashboard", fontsize=18, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_dir / "version_metrics_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
