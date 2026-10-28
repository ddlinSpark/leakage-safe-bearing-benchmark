"""Record SHA-256 values for every publication figure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=ROOT / "figures" / "publication",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "figures" / "publication" / "figure_hashes.json",
    )
    args = parser.parse_args()
    files = sorted(
        path
        for path in args.figures_dir.iterdir()
        if path.suffix.lower() in {".png", ".pdf"}
    )
    payload = {
        "files": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        }
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Recorded {len(files)} publication figure files.")


if __name__ == "__main__":
    main()
