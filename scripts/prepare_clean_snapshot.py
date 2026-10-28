"""Create a clean source snapshot without caches or local development artifacts."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def source_paths(root: Path) -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        capture_output=True,
        check=True,
    )
    paths = [
        Path(value.decode("utf-8"))
        for value in completed.stdout.split(b"\0")
        if value
    ]
    return sorted(set(paths))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-canonical-cache",
        action="store_true",
        help="Copy the 605 MB canonical NPZ for an isolated training smoke test.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Clean snapshot target is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    paths = source_paths(root)
    for relative in paths:
        source = root / relative
        if not source.is_file():
            continue
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    if args.include_canonical_cache:
        relative = Path(
            "data/public/paderborn/pu_4class_w8192.npz"
        )
        source = root / relative
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        metadata = source.with_suffix(".json")
        shutil.copy2(metadata, target.with_suffix(".json"))
    print(f"Copied {len(paths)} non-ignored source files to {output}")


if __name__ == "__main__":
    main()
