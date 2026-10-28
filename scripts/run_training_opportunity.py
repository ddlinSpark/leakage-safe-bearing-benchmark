"""Run the 30-epoch training-opportunity sensitivity."""

from __future__ import annotations

import sys
from pathlib import Path

from spindle_cnn.experiments.run_benchmark import main


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--config",
        str(ROOT / "configs" / "paderborn_training_budget_30.json"),
        "--experiment",
        "main",
        "--output-dir",
        str(ROOT / "results" / "training_budget_30"),
        *sys.argv[1:],
    ]
    main()
