"""Download and extract the focused Paderborn benchmark subset."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlretrieve

from ..paths import DATA_DIR
from .pu_data import BEARING_CODES


BASE_URL = "https://groups.uni-paderborn.de/kat/BearingDataCenter"


def _default_seven_zip() -> Path:
    configured = os.environ.get("SEVEN_ZIP")
    if configured:
        return Path(configured).expanduser()
    resolved = (
        shutil.which("7z")
        or shutil.which("7za")
        or shutil.which("7z.exe")
        or shutil.which("bsdtar")
    )
    return Path(resolved or "bsdtar")


DEFAULT_SEVEN_ZIP = _default_seven_zip()


def download_one(code: str, archive_dir: Path) -> tuple[str, Path]:
    path = archive_dir / f"{code}.rar"
    if not path.exists():
        urlretrieve(f"{BASE_URL}/{code}.rar", path)
    return code, path


def extract_one(code: str, archive: Path, output_dir: Path, seven_zip: Path) -> None:
    destination = output_dir / code
    destination.mkdir(parents=True, exist_ok=True)
    if seven_zip.name.lower().startswith("bsdtar"):
        completed = subprocess.run(
            [
                str(seven_zip),
                "-xf",
                str(archive),
                "-C",
                str(destination),
                "--include=*_1.mat",
                "--include=*_2.mat",
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Failed to extract {code}: {completed.stderr}")
        print(f"extracted {code} with bsdtar")
        return

    command = [
        str(seven_zip),
        "x",
        "-y",
        "-r",
        str(archive),
        "*_1.mat",
        "*_2.mat",
        f"-o{destination}",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Failed to extract {code}: {completed.stderr}")
    print(f"extracted {code} measurements 1-2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备聚焦的 Paderborn 公共数据子集")
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=DATA_DIR / "public" / "paderborn" / "rars",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=DATA_DIR / "public" / "paderborn" / "extracted",
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seven-zip", type=Path, default=DEFAULT_SEVEN_ZIP)
    parser.add_argument("--skip-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.archive_dir.mkdir(parents=True, exist_ok=True)
    args.extract_dir.mkdir(parents=True, exist_ok=True)
    codes = list(BEARING_CODES)

    archives: dict[str, Path] = {}
    if args.skip_download:
        for code in codes:
            path = args.archive_dir / f"{code}.rar"
            if not path.is_file():
                raise FileNotFoundError(path)
            archives[code] = path
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(download_one, code, args.archive_dir): code
                for code in codes
            }
            for future in as_completed(futures):
                code, path = future.result()
                archives[code] = path
                print(f"downloaded {code}")

    for code in codes:
        extract_one(code, archives[code], args.extract_dir, args.seven_zip)


if __name__ == "__main__":
    main()
