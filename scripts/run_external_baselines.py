"""Run DRSN and TFN same-protocol adaptation experiments."""

from __future__ import annotations

import sys
from pathlib import Path

from spindle_cnn.experiments.run_benchmark import main


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--config",
        str(ROOT / "configs" / "paderborn_sota.json"),
        "--experiment",
        "sota",
        *sys.argv[1:],
    ]
    main()
