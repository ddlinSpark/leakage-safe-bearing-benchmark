"""Create the CWT-GRU architecture diagram used in the manuscript."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from ..paths import OUTPUTS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 CWT-GRU 网络结构图")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUTS_DIR
        / "public_paderborn_final"
        / "figures"
        / "figure_architecture.png",
    )
    return parser.parse_args()


def box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    detail: str,
    color: str,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.03,rounding_size=0.06",
            facecolor=color,
            edgecolor="#4A4A4A",
            linewidth=0.9,
        )
    )
    ax.text(x + width / 2, y + height * 0.64, title, ha="center", va="center", weight="bold")
    ax.text(x + width / 2, y + height * 0.28, detail, ha="center", va="center", fontsize=8)


def arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.0,
            color="#333333",
        )
    )


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    box(ax, 0.2, 2.3, 1.8, 1.4, "Raw signals", "3 channels x 8192", "#E8EEF6")
    box(ax, 2.55, 3.55, 2.0, 1.3, "CWT branch", "Morlet, 48 scales", "#E8EEF6")
    box(ax, 5.0, 3.55, 2.0, 1.3, "Image encoder", "3 x 96 x 96", "#E8EEF6")
    box(ax, 7.45, 3.55, 2.0, 1.3, "Residual + SE", "192-d feature", "#E8EEF6")
    box(ax, 10.0, 3.55, 1.7, 1.3, "Global pool", "192-d vector", "#E8EEF6")

    box(ax, 2.55, 1.35, 2.0, 1.3, "Temporal branch", "3 x 8192 sequence", "#F5EEE7")
    box(ax, 5.0, 1.35, 2.0, 1.3, "2-layer BiGRU", "hidden = 64", "#F5EEE7")
    box(ax, 7.45, 1.35, 2.0, 1.3, "Temporal feature", "128-d vector", "#F5EEE7")

    box(ax, 10.0, 2.05, 1.4, 1.3, "Fusion", "320-d", "#EFEFF1")
    box(ax, 12.0, 2.05, 1.7, 1.3, "Classifier", "4 classes", "#EFEFF1")

    arrow(ax, (2.0, 3.15), (2.55, 4.2))
    arrow(ax, (2.0, 2.85), (2.55, 2.0))
    arrow(ax, (4.55, 4.2), (5.0, 4.2))
    arrow(ax, (7.0, 4.2), (7.45, 4.2))
    arrow(ax, (9.45, 4.2), (10.0, 4.2))
    arrow(ax, (4.55, 2.0), (5.0, 2.0))
    arrow(ax, (7.0, 2.0), (7.45, 2.0))
    arrow(ax, (10.85, 3.55), (10.85, 3.35))
    arrow(ax, (9.45, 2.0), (10.25, 2.05))
    arrow(ax, (11.4, 2.7), (12.0, 2.7))

    ax.text(
        5.9,
        5.45,
        "Time-frequency branch: CWT magnitude -> residual CNN with squeeze-and-excitation",
        ha="center",
        fontsize=8.5,
    )
    ax.text(
        5.9,
        0.65,
        "Temporal branch: bidirectional recurrent modeling preserves waveform order and transients",
        ha="center",
        fontsize=8.5,
    )
    ax.text(
        11.8,
        1.4,
        "concat -> 256 -> 128 -> logits",
        ha="center",
        fontsize=8,
    )
    fig.savefig(args.output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
