"""Single palette and visual style for all publication figures."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba


PAPER_COLORS = (
    "#7B95C6",
    "#49C2D9",
    "#A1D8E8",
    "#67A583",
    "#A2C986",
    "#D0E2C0",
    "#FDED95",
    "#FFC1A6",
    "#F59C7C",
    "#F47254",
    "#C85E62",
)

MODEL_COLORS = {
    "raw_cnn": "#7B95C6",
    "gru_only": "#49C2D9",
    "cwt_cnn": "#67A583",
    "cwt_gru_no_se": "#A2C986",
    "cwt_gru": "#FDED95",
    "cwt_gru_temp": "#FFC1A6",
    "cwt_gru_selftrain": "#F47254",
    "cwt_gru_selftrain_coverage": "#F47254",
    "drsn": "#C85E62",
    "tfnet": "#F59C7C",
    "tfn": "#F59C7C",
}

AUXILIARY_COLORS = {
    "reference": "#A1D8E8",
    "secondary": "#D0E2C0",
}

INK_COLOR = "#1F1F1F"
GRID_COLOR = "#D9D9D9"
MUTED_COLOR = "#7A7A7A"
WHITE = "#FFFFFF"


def apply_paper_style(*, font_size: float = 9.0) -> None:
    """Apply the common neutral journal style without changing data."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": font_size,
            "axes.labelsize": font_size + 1,
            "axes.titlesize": font_size + 1,
            "legend.fontsize": max(font_size - 1, 6.5),
            "xtick.labelsize": max(font_size - 0.5, 6.5),
            "ytick.labelsize": max(font_size - 0.5, 6.5),
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": INK_COLOR,
            "axes.labelcolor": INK_COLOR,
            "text.color": INK_COLOR,
            "xtick.color": INK_COLOR,
            "ytick.color": INK_COLOR,
            "lines.linewidth": 1.4,
            "lines.markersize": 4,
            "errorbar.capsize": 3,
            "legend.frameon": False,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.6,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def method_color(method: str) -> str:
    """Return the fixed color for a method, with deterministic fallback cycling."""
    if method in MODEL_COLORS:
        return MODEL_COLORS[method]
    fallback = [
        value
        for value in PAPER_COLORS
        if value not in set(MODEL_COLORS.values())
    ]
    code = sum((index + 1) * ord(char) for index, char in enumerate(str(method)))
    return fallback[code % len(fallback)]


def color_with_alpha(color: str, alpha: float) -> tuple[float, float, float, float]:
    """Return an RGBA color while preserving the source palette hue."""
    red, green, blue, _ = to_rgba(color)
    return red, green, blue, alpha


def palette_cmap(
    color: str = "#7B95C6",
    *,
    name: str | None = None,
) -> LinearSegmentedColormap:
    """Create a light-to-palette sequential map for heatmaps and surfaces."""
    return LinearSegmentedColormap.from_list(
        name or "bearing_paper_palette",
        [WHITE, color],
    )


def ordered_colors(values: Sequence[str]) -> list[str]:
    """Map a sequence of method keys to their fixed publication colors."""
    return [method_color(str(value)) for value in values]
