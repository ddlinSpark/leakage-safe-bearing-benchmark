"""Project path defaults shared by data, training, and inference modules."""

from __future__ import annotations

import os
from pathlib import Path


def _resolve_project_root() -> Path:
    override = os.environ.get("SPINDLE_CNN_ROOT")
    if override:
        return Path(override).expanduser().resolve()

    source_candidate = Path(__file__).resolve().parents[2]
    if (source_candidate / "pyproject.toml").is_file():
        return source_candidate

    return Path.cwd().resolve()


PROJECT_ROOT = _resolve_project_root()
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = PROJECT_ROOT / "dataset_final"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"

DEFAULT_DATASET_CSV = DATASET_DIR / "final_dataset_table.csv"
DEFAULT_WINDOW_INDEX_CSV = DATASET_DIR / "window_index" / "window_sample_index.csv"
DEFAULT_LIMIT_CSV = DATA_DIR / "ms_cvtlimit.csv"
