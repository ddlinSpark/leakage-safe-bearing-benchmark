"""Render manuscript equations as high-resolution raster fallback images."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..paths import OUTPUTS_DIR


EQUATIONS = {
    1: r"$W_x(a,b)=\frac{1}{\sqrt{a}}\int_{-\infty}^{\infty}x(t)\psi^{*}\left(\frac{t-b}{a}\right)dt$",
    2: r"$I_c(a,b)=\frac{|W_{x_c}(a,b)|}{\max_{a,b}|W_{x_c}(a,b)|+\epsilon}$",
    3: r"$p(y=k\mid x)=\frac{\exp(z_k/T)}{\sum_{j=1}^{K}\exp(z_j/T)}$",
    4: r"$\mathcal{S}_k=\mathrm{TopM}\left\{i\notin\mathcal{L}:\hat{y}_i=k,\;p_{i,k}\geq\tau\right\},\quad \tau=0.60$",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="渲染论文公式图片")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "paper_assets",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
        }
    )
    for number, expression in EQUATIONS.items():
        fig = plt.figure(figsize=(8.0, 1.05), facecolor="white")
        fig.text(0.5, 0.5, expression, ha="center", va="center", fontsize=17, color="black")
        output = args.output_dir / f"equation_{number}.png"
        fig.savefig(output, dpi=320, bbox_inches="tight", pad_inches=0.03, facecolor="white")
        plt.close(fig)
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
