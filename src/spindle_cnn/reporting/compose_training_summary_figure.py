import argparse
import json
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from ..paths import OUTPUTS_DIR


MODEL_DISPLAY_NAMES = {
    "gru_swin_cbam": "GRU + SwinTransformer-CBAM",
    "residual_attention": "Residual Attention",
    "residual_attention_plus": "Residual Attention Plus",
    "resnet18": "ResNet18",
    "vgg11_bn": "VGG11_BN",
}

TRANSFORM_DISPLAY_NAMES = {
    "gadf": "GADF",
    "gaf": "GAF",
    "cwt": "CWT",
    "raw": "Raw Signal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将训练结果图表汇总为一张总图")
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUTS_DIR / "gafd_gru_swin_cbam_v2"),
        help="单次训练输出目录",
    )
    parser.add_argument(
        "--save-name",
        default="summary_dashboard.png",
        help="总图文件名",
    )
    return parser.parse_args()


def normalize_path(path_value: str) -> Path:
    path_str = str(path_value)
    if len(path_str) >= 3 and path_str[1] == ":" and path_str[2] in ("\\", "/") and Path("/").exists():
        drive = path_str[0].lower()
        suffix = path_str[3:].replace("\\", "/")
        return Path(f"/mnt/{drive}/{suffix}")
    return Path(path_str)


def format_experiment_name(summary: dict) -> str:
    transform_name = TRANSFORM_DISPLAY_NAMES.get(
        str(summary.get("transform_type", "")).lower(),
        str(summary.get("transform_type", "Unknown")).upper(),
    )
    model_name = MODEL_DISPLAY_NAMES.get(
        str(summary.get("model_name", "")).lower(),
        str(summary.get("model_name", "Model")),
    )
    return f"{transform_name} + {model_name}"


def load_metrics_text(output_dir: Path) -> tuple[str, str]:
    summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
    metrics = summary["metrics"]
    experiment_name = format_experiment_name(summary)
    lines = [
        experiment_name,
        "",
        f"Device: {summary['device']}",
        f"Epochs: {summary['epochs']}",
        f"Batch Size: {summary['batch_size']}",
        f"Window Size: {summary['window_size']}",
        f"GADF Points: {summary.get('gadf_points', 'N/A')}",
        f"Image Size: {summary['image_size']}",
        f"Normal Ratio: {summary.get('normal_ratio', 'N/A')}",
        "",
        "Test Metrics",
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"Precision Macro: {metrics['precision_macro']:.4f}",
        f"Recall Macro: {metrics['recall_macro']:.4f}",
        f"F1 Macro: {metrics['f1_macro']:.4f}",
        f"Precision Weighted: {metrics['precision_weighted']:.4f}",
        f"Recall Weighted: {metrics['recall_weighted']:.4f}",
        f"F1 Weighted: {metrics['f1_weighted']:.4f}",
        "",
        f"Train Samples: {summary['train_samples']}",
        f"Val Samples: {summary['val_samples']}",
        f"Test Samples: {summary['test_samples']}",
    ]
    if summary.get("cwt_wavelets"):
        lines.extend(
            [
                "",
                f"CWT Wavelets: {', '.join(summary['cwt_wavelets'])}",
                f"Gradient Mix: {summary.get('cwt_gradient_mix', 'N/A')}",
            ]
        )

    calibration_method = summary.get("postprocess_calibration_method")
    if calibration_method:
        lines.extend(
            [
                "",
                "Postprocess",
                f"Calibration: {calibration_method}",
                f"Group Column: {summary.get('postprocess_group_column', 'global') or 'global'}",
                f"TTA Views: {len(summary.get('postprocess_tta_shift_pairs', [])) or 1}",
            ]
        )
        val_metrics = summary.get("postprocess_val_metrics", {})
        if val_metrics:
            if "accuracy" in val_metrics:
                lines.append(f"Val Accuracy After Post: {val_metrics['accuracy']:.4f}")
            if "f1_macro" in val_metrics:
                lines.append(f"Val Macro F1 After Post: {val_metrics['f1_macro']:.4f}")

    return experiment_name, "\n".join(lines)


def add_image_panel(ax: plt.Axes, image_path: Path, title: str) -> None:
    image = mpimg.imread(image_path)
    ax.imshow(image)
    ax.set_title(title, fontsize=13)
    ax.axis("off")


def main() -> None:
    args = parse_args()
    output_dir = normalize_path(args.output_dir)

    figure_map = [
        ("training_curves.png", "Training Curves"),
        ("label_distribution.png", "Label Distribution"),
        ("confusion_matrix.png", "Confusion Matrix"),
        ("tsne_raw_data.png", "t-SNE of Raw Data"),
        ("tsne_model_features.png", "t-SNE of Model Features"),
    ]

    experiment_name, metrics_text = load_metrics_text(output_dir)

    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.suptitle(f"Spindle CNN Summary Dashboard: {experiment_name}", fontsize=18, y=0.98)

    text_ax = axes[0, 0]
    text_ax.axis("off")
    text_ax.text(
        0.02,
        0.98,
        metrics_text,
        va="top",
        ha="left",
        fontsize=12,
        family="monospace",
        bbox={"facecolor": "#f5f5f5", "edgecolor": "#d0d0d0", "boxstyle": "round,pad=0.6"},
    )

    panel_axes = [axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]
    for ax, (image_name, title) in zip(panel_axes, figure_map):
        add_image_panel(ax, output_dir / image_name, title)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / args.save_name, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
