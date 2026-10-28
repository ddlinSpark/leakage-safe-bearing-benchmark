"""Run the canonical self-training threshold sensitivity."""

from __future__ import annotations

import sys

from spindle_cnn.experiments.run_benchmark import main


if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--experiment",
        "threshold",
        *sys.argv[1:],
    ]
    main()
