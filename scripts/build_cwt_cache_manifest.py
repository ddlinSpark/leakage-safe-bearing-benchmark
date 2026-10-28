"""Record sizes, hashes, and shapes for generated CWT control caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data" / "public" / "paderborn" / "cwt_sensitivity",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "manifests" / "cwt_sensitivity_caches.csv",
    )
    args = parser.parse_args()
    rows = []
    for path in sorted(args.cache_dir.glob("*.npz")):
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        with np.load(path, mmap_mode="r") as payload:
            shape = list(payload["cwt"].shape)
            dtype = str(payload["cwt"].dtype)
        rows.append(
            {
                "variant": path.stem,
                "shape": "x".join(str(value) for value in shape),
                "dtype": dtype,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "scales": metadata["scales"],
                "image_size": metadata["image_size"],
                "transform": metadata["transform"],
                "normalize_max": metadata["normalize_max"],
            }
        )
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, lineterminator="\n")
    print(f"Wrote {len(frame)} cache hashes to {args.output}")


if __name__ == "__main__":
    main()
