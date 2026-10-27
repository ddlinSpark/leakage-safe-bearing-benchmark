"""Create the evaluation-flow figure with the project plotting environment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from ..paths import OUTPUTS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成论文评价流程示意图")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUTS_DIR / "public_paderborn" / "figures" / "figure_protocol.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, ax = plt.subplots(figsize=(9.2, 2.7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    boxes = [
        (0.2, 1.35, 1.8, 0.8, "Verified public\nlabels"),
        (2.35, 1.35, 1.8, 0.8, "Non-overlapping\nwindows"),
        (4.5, 1.35, 1.8, 0.8, "Disjoint bearing\nand condition"),
        (6.65, 1.35, 1.8, 0.8, "Labeled + unlabeled\ntraining data"),
        (8.8, 1.35, 1.0, 0.8, "Held-out\ntest"),
    ]
    for x, y, width, height, text in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.03,rounding_size=0.06",
                linewidth=0.9,
                edgecolor="#5B5B5B",
                facecolor="#EDF2F8",
            )
        )
        ax.text(x + width / 2, y + height / 2, text, ha="center", va="center")
    for left in (2.0, 4.15, 6.3, 8.45):
        ax.add_patch(
            FancyArrowPatch(
                (left, 1.75),
                (left + 0.3, 1.75),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=0.9,
                color="#333333",
            )
        )
    ax.text(
        4.15,
        0.55,
        "Calibrated pseudo-labels are generated only from the unlabeled training portion",
        ha="center",
        va="center",
        fontsize=8,
    )
    fig.savefig(args.output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
