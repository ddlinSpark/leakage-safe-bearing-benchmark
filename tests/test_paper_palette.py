"""Tests for the shared publication palette and figure-style adoption."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spindle_cnn.plotting.style import method_color as legacy_method_color
from spindle_cnn.visualization.style import (
    MODEL_COLORS,
    PAPER_COLORS,
    apply_paper_style,
    color_with_alpha,
    method_color,
    palette_cmap,
)


ROOT = Path(__file__).resolve().parents[1]
PLOT_MODULES = [
    "src/spindle_cnn/analysis/baseline_tuning.py",
    "src/spindle_cnn/analysis/calibration_audit.py",
    "src/spindle_cnn/analysis/cwt_sensitivity.py",
    "src/spindle_cnn/analysis/domain_shift.py",
    "src/spindle_cnn/analysis/feature_space.py",
    "src/spindle_cnn/analysis/foldwise.py",
    "src/spindle_cnn/analysis/label_granularity.py",
    "src/spindle_cnn/analysis/normalization_sensitivity.py",
    "src/spindle_cnn/analysis/self_training.py",
    "src/spindle_cnn/analysis/training_budget.py",
    "src/spindle_cnn/experiments/make_figures.py",
    "src/spindle_cnn/experiments/make_method_figures.py",
]


def test_palette_and_model_mapping_are_fixed() -> None:
    assert PAPER_COLORS == (
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
    assert MODEL_COLORS["raw_cnn"] == "#7B95C6"
    assert MODEL_COLORS["gru_only"] == "#49C2D9"
    assert MODEL_COLORS["cwt_cnn"] == "#67A583"
    assert MODEL_COLORS["cwt_gru_no_se"] == "#A2C986"
    assert MODEL_COLORS["cwt_gru"] == "#FDED95"
    assert MODEL_COLORS["cwt_gru_temp"] == "#FFC1A6"
    assert MODEL_COLORS["cwt_gru_selftrain"] == "#F47254"
    assert MODEL_COLORS["drsn"] == "#C85E62"
    assert MODEL_COLORS["tfnet"] == "#F59C7C"


def test_style_and_color_helpers() -> None:
    apply_paper_style()
    assert plt.rcParams["axes.spines.top"] is False
    assert plt.rcParams["axes.spines.right"] is False
    assert plt.rcParams["figure.facecolor"] == "#FFFFFF"
    assert plt.rcParams["savefig.dpi"] == 600
    assert method_color("raw_cnn") == "#7B95C6"
    assert legacy_method_color("raw_cnn") == "#7B95C6"
    assert color_with_alpha("#7B95C6", 0.4)[3] == 0.4
    assert palette_cmap("#67A583").name == "bearing_paper_palette"


def test_all_plot_modules_use_shared_style_and_no_local_hex_palette() -> None:
    palette = {value.lower() for value in PAPER_COLORS}
    for relative in PLOT_MODULES:
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "apply_paper_style" in text, relative
        local_hex = {
            value.lower()
            for value in re.findall(r"#[0-9A-Fa-f]{6}", text)
        }
        assert local_hex <= palette, f"{relative}: {sorted(local_hex - palette)}"
