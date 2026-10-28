"""Audit CWT preprocessing cost and clarify reported training timings."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..datasets.splits import load_fold_indices
from ..evaluation.efficiency import estimate_bidirectional_gru_flops
from ..paths import PROJECT_ROOT
from ..public_benchmark.cwru_data import load_cwru_cache
from ..public_benchmark.pu_data import _cwt_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=PROJECT_ROOT / "data" / "public" / "paderborn" / "pu_4class_w8192.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
    )
    parser.add_argument("--samples", type=int, default=48)
    args = parser.parse_args()
    data = load_cwru_cache(args.cache)
    split = load_fold_indices(data, 0, np.asarray([4, 8, 12, 15]))
    selected = split.train[: args.samples]
    start = time.perf_counter()
    generated = [_cwt_image(data.raw[index], 48, 96) for index in selected]
    elapsed = time.perf_counter() - start
    per_window = elapsed / len(selected)
    full_estimate = per_window * len(data.labels)
    cache_size = args.cache.stat().st_size
    raw_bytes = int(data.raw.nbytes)
    cwt_bytes = int(data.cwt.nbytes)
    estimated_gru_flops = estimate_bidirectional_gru_flops(
        batch_size=1,
        sequence_length=8192,
        input_size=3,
        hidden_size=64,
        layers=2,
    )
    rows = pd.DataFrame(
        [
            {
                "item": "Measured CWT generation sample",
                "value": elapsed,
                "unit": "s",
                "note": f"{len(selected)} windows, single process, canonical settings",
            },
            {
                "item": "CWT generation per window",
                "value": per_window,
                "unit": "s/window",
                "note": "Measured sample divided by window count",
            },
            {
                "item": "Serial 3,939-window CWT estimate",
                "value": full_estimate,
                "unit": "s",
                "note": "Extrapolated from the measured sample; not a recorded full-run time",
            },
            {
                "item": "Canonical NPZ bytes",
                "value": cache_size,
                "unit": "bytes",
                "note": "Compressed cache containing raw and CWT arrays",
            },
            {
                "item": "In-memory raw array bytes",
                "value": raw_bytes,
                "unit": "bytes",
                "note": "float32 raw tensor",
            },
            {
                "item": "In-memory CWT array bytes",
                "value": cwt_bytes,
                "unit": "bytes",
                "note": "float16 CWT tensor",
            },
            {
                "item": "Computed bidirectional GRU FLOPs",
                "value": estimated_gru_flops,
                "unit": "FLOPs/sample",
                "note": "Explicit 2-layer formula added to profiler counts",
            },
            {
                "item": "Training-time inclusion",
                "value": np.nan,
                "unit": "status",
                "note": "Recorded training timings exclude offline CWT generation and cache loading",
            },
        ]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(
        args.output_dir / "computational_cost_audit.csv",
        index=False,
        lineterminator="\n",
    )
    print(rows.to_string(index=False))


if __name__ == "__main__":
    main()
