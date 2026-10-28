"""Verify local Paderborn archives against the checked-in checksum manifest."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

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
        "--archive-dir",
        type=Path,
        default=ROOT / "data" / "public" / "paderborn" / "rars",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "manifests" / "paderborn_archives_sha256.csv",
    )
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    failures = []
    for row in manifest.itertuples(index=False):
        path = args.archive_dir / f"{row.bearing_code}.rar"
        if not path.is_file():
            failures.append(f"{row.bearing_code}: missing")
            continue
        if path.stat().st_size != int(row.bytes):
            failures.append(
                f"{row.bearing_code}: size {path.stat().st_size} != {int(row.bytes)}"
            )
            continue
        actual = sha256(path)
        if actual != row.sha256:
            failures.append(
                f"{row.bearing_code}: SHA-256 {actual} != {row.sha256}"
            )
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Verified {len(manifest)} Paderborn archives by size and SHA-256.")


if __name__ == "__main__":
    main()
