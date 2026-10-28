"""Generate method diagrams directly aligned with the benchmark code."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from ..paths import PROJECT_ROOT
from ..visualization.style import (
    AUXILIARY_COLORS,
    INK_COLOR,
    MUTED_COLOR,
    PAPER_COLORS,
    apply_paper_style,
    method_color,
)


BLUE = method_color("raw_cnn")
PALE_BLUE = AUXILIARY_COLORS["reference"]
ORANGE = method_color("cwt_gru_selftrain")
PALE_ORANGE = PAPER_COLORS[7]
GRAY = MUTED_COLOR
PALE_GRAY = PAPER_COLORS[6]
GREEN = method_color("cwt_cnn")
PALE_GREEN = AUXILIARY_COLORS["secondary"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate methodology figures")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "figures" / "publication",
    )
    return parser.parse_args()


def box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str = "",
    *,
    edge: str = GRAY,
    face: str = "white",
    title_size: float = 9,
    body_size: float = 8,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.0,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    if body:
        ax.text(
            x + width / 2,
            y + height * 0.66,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color=INK_COLOR,
        )
        ax.text(
            x + width / 2,
            y + height * 0.30,
            body,
            ha="center",
            va="center",
            fontsize=body_size,
            color=INK_COLOR,
            linespacing=1.25,
        )
    else:
        ax.text(
            x + width / 2,
            y + height / 2,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            color=INK_COLOR,
            linespacing=1.25,
        )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = GRAY,
    width: float = 1.2,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=width,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png")
    plt.close(fig)


def overall_framework(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.6, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(
        ax,
        0.03,
        0.69,
        0.19,
        0.22,
        "Public multimodal data",
        "Paderborn PU\nvibration + 2 phase currents\n3 × 8192 windows",
        edge=BLUE,
        face=PALE_BLUE,
    )
    box(
        ax,
        0.27,
        0.69,
        0.22,
        0.22,
        "Leakage-safe folds",
        "Test: unseen bearing code\n+ unseen condition\nValidation: seen bearing,\nremaining condition",
        edge=ORANGE,
        face=PALE_ORANGE,
    )
    box(
        ax,
        0.54,
        0.69,
        0.19,
        0.22,
        "Model comparison",
        "Raw 1D-CNN\nGRU-only\nCWT 2D-CNN\nCWT-GRU + SE",
        edge=BLUE,
        face=PALE_BLUE,
    )
    box(
        ax,
        0.78,
        0.69,
        0.19,
        0.22,
        "Validation-only fit",
        "Early stopping\nTemperature T by NLL\nNo test-label access",
        edge=GREEN,
        face=PALE_GREEN,
    )

    arrow(ax, (0.22, 0.80), (0.27, 0.80))
    arrow(ax, (0.49, 0.80), (0.54, 0.80))
    arrow(ax, (0.73, 0.80), (0.78, 0.80))

    box(
        ax,
        0.08,
        0.36,
        0.22,
        0.19,
        "Label budget",
        "10% balanced budget\n50% balanced budget\nFull balanced budget",
        edge=BLUE,
        face=PALE_BLUE,
    )
    box(
        ax,
        0.39,
        0.36,
        0.22,
        0.19,
        "Calibrated self-training",
        "τ = 0.60\nclass-balanced pseudo-labels\nup to 2 rounds",
        edge=ORANGE,
        face=PALE_ORANGE,
    )
    box(
        ax,
        0.70,
        0.36,
        0.22,
        0.19,
        "Held-out evaluation",
        "unseen bearing + condition\n20 matched runs per method\nno test-informed update",
        edge=GREEN,
        face=PALE_GREEN,
    )
    arrow(ax, (0.30, 0.455), (0.39, 0.455))
    arrow(ax, (0.61, 0.455), (0.70, 0.455))
    arrow(ax, (0.875, 0.69), (0.81, 0.55), color=GREEN)

    box(
        ax,
        0.15,
        0.08,
        0.70,
        0.16,
        "Analysis layer",
        "Macro-F1 and accuracy | fold-aware paired intervals | ECE, NLL, Brier score\n"
        "class confusion | pseudo-label coverage and accuracy | parameters, FLOPs, latency",
        edge=GRAY,
        face=PALE_GRAY,
        title_size=10,
        body_size=8.3,
    )
    arrow(ax, (0.81, 0.36), (0.81, 0.24), color=GREEN)
    save(fig, output_dir, "figure_method_01_overall_framework")


def cwt_gru_architecture(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 6.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.02,
        0.95,
        "CWT branch",
        fontsize=11,
        fontweight="bold",
        color=BLUE,
        ha="left",
    )
    box(
        ax,
        0.02,
        0.68,
        0.16,
        0.20,
        "Three CWT maps",
        "raw 3 × 8192\nMorlet, 48 scales\nmagnitude 3 × 96 × 96",
        edge=BLUE,
        face=PALE_BLUE,
    )
    box(
        ax,
        0.21,
        0.68,
        0.16,
        0.20,
        "Stem",
        "Conv 5x5, 32, stride 2\nBN + SiLU + MaxPool\n3 × 96 × 96 -> 32 × 24 × 24",
        edge=BLUE,
        face=PALE_BLUE,
    )
    box(
        ax,
        0.40,
        0.68,
        0.17,
        0.20,
        "Residual stages",
        "32->64, stride 1\n64->128, stride 2\n128->192, stride 2",
        edge=BLUE,
        face=PALE_BLUE,
    )
    box(
        ax,
        0.60,
        0.68,
        0.16,
        0.20,
        "SE after each block",
        "GAP -> Linear C->C/8\nReLU -> Linear C/8->C\nSigmoid channel gate",
        edge=BLUE,
        face=PALE_BLUE,
    )
    box(
        ax,
        0.79,
        0.68,
        0.17,
        0.20,
        "Image feature",
        "global average pool\nfeature vector: 192",
        edge=BLUE,
        face=PALE_BLUE,
    )
    for start, end in (
        ((0.18, 0.78), (0.21, 0.78)),
        ((0.37, 0.78), (0.40, 0.78)),
        ((0.57, 0.78), (0.60, 0.78)),
        ((0.76, 0.78), (0.79, 0.78)),
    ):
        arrow(ax, start, end, color=BLUE)

    ax.text(
        0.02,
        0.57,
        "Waveform branch",
        fontsize=11,
        fontweight="bold",
        color=ORANGE,
        ha="left",
    )
    box(
        ax,
        0.02,
        0.32,
        0.20,
        0.19,
        "Raw sequence",
        "vibration + 2 currents\n3 x 8192 -> 8192 x 3",
        edge=ORANGE,
        face=PALE_ORANGE,
    )
    box(
        ax,
        0.26,
        0.32,
        0.27,
        0.19,
        "Two-layer bidirectional GRU",
        "hidden = 64 per direction\nsequence length = 8192\ndropout = 0.15",
        edge=ORANGE,
        face=PALE_ORANGE,
    )
    box(
        ax,
        0.57,
        0.32,
        0.19,
        0.19,
        "Last hidden states",
        "concatenate forward + backward\n64 + 64 = 128",
        edge=ORANGE,
        face=PALE_ORANGE,
    )
    arrow(ax, (0.22, 0.415), (0.26, 0.415), color=ORANGE)
    arrow(ax, (0.53, 0.415), (0.57, 0.415), color=ORANGE)

    box(
        ax,
        0.24,
        0.08,
        0.54,
        0.15,
        "Fusion classifier",
        "[192 image | 128 GRU] -> 320 -> LayerNorm -> Linear 256 -> SiLU -> Dropout 0.25\n"
        "-> Linear 128 -> SiLU -> Dropout 0.15 -> Linear 4 logits",
        edge=GREEN,
        face=PALE_GREEN,
        title_size=10,
        body_size=8.0,
    )
    arrow(ax, (0.875, 0.68), (0.63, 0.23), color=BLUE)
    arrow(ax, (0.665, 0.32), (0.53, 0.23), color=ORANGE)
    save(fig, output_dir, "figure_method_02_cwt_gru_architecture")


def self_training_pipeline(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 5.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(
        ax,
        0.03,
        0.64,
        0.17,
        0.20,
        "Balanced labeled set",
        "L: verified training labels\n"
        "10% balanced budget\n"
        "50% balanced budget\n"
        "Full balanced budget",
        edge=BLUE,
        face=PALE_BLUE,
    )
    box(
        ax,
        0.24,
        0.64,
        0.18,
        0.20,
        "Initialize model",
        "CWT-GRU + SE\nearly stop on validation F1\nAdamW + cosine schedule",
        edge=BLUE,
        face=PALE_BLUE,
    )
    box(
        ax,
        0.46,
        0.64,
        0.19,
        0.20,
        "Fit temperature",
        "validation logits only\nLBFGS minimizing NLL\nmonotonic probabilities",
        edge=GREEN,
        face=PALE_GREEN,
    )
    box(
        ax,
        0.69,
        0.64,
        0.28,
        0.20,
        "Unlabeled candidate pool",
        "U = training indices - L\nvalidation indices excluded\ntest indices excluded",
        edge=ORANGE,
        face=PALE_ORANGE,
    )
    arrow(ax, (0.20, 0.74), (0.24, 0.74))
    arrow(ax, (0.42, 0.74), (0.46, 0.74))
    arrow(ax, (0.65, 0.74), (0.69, 0.74))

    box(
        ax,
        0.06,
        0.34,
        0.23,
        0.18,
        "Confidence selection",
        "p_i,k from temperature scaling\np_i,k ≥ τ, τ = 0.60",
        edge=ORANGE,
        face=PALE_ORANGE,
    )
    box(
        ax,
        0.34,
        0.34,
        0.24,
        0.18,
        "Class-balanced cap",
        "Top-M per predicted class\nM = 2 × labeled count / K\nM doubles each round",
        edge=ORANGE,
        face=PALE_ORANGE,
    )
    box(
        ax,
        0.63,
        0.34,
        0.25,
        0.18,
        "Weighted retraining",
        "verified labels: weight 1\npseudo-labels: confidence weight\nreinitialize; at most 2 rounds",
        edge=BLUE,
        face=PALE_BLUE,
    )
    arrow(ax, (0.29, 0.43), (0.34, 0.43), color=ORANGE)
    arrow(ax, (0.58, 0.43), (0.63, 0.43), color=ORANGE)
    arrow(ax, (0.48, 0.64), (0.17, 0.52), color=GREEN)
    arrow(ax, (0.79, 0.64), (0.23, 0.52), color=ORANGE)

    box(
        ax,
        0.18,
        0.08,
        0.64,
        0.15,
        "Independent evaluation",
        "unseen bearing + unseen condition | no test pseudo-labels\n"
        "Macro-F1, accuracy, ECE/NLL/Brier, confusion, coverage and cost",
        edge=GREEN,
        face=PALE_GREEN,
        title_size=10,
        body_size=8.2,
    )
    arrow(ax, (0.755, 0.34), (0.70, 0.23), color=BLUE)
    save(fig, output_dir, "figure_method_03_self_training_pipeline")


def main() -> None:
    args = parse_args()
    apply_paper_style(font_size=9)
    overall_framework(args.output_dir)
    cwt_gru_architecture(args.output_dir)
    self_training_pipeline(args.output_dir)


if __name__ == "__main__":
    main()
