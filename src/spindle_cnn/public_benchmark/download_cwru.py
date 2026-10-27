"""Download the 40 official CWRU files used by the public benchmark."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlretrieve

from ..paths import DATA_DIR
from .cwru_data import CWRU_RECORD_IDS, verify_cwru_files


BASE_URL = "https://engineering.case.edu/sites/default/files"


def download_one(record_id: int, output_dir: Path, overwrite: bool) -> int:
    destination = output_dir / f"{record_id}.mat"
    if destination.exists() and not overwrite:
        return record_id
    urlretrieve(f"{BASE_URL}/{record_id}.mat", destination)
    return record_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载 CWRU 10 类基准的 40 个官方文件")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR / "public" / "cwru" / "raw",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record_ids = CWRU_RECORD_IDS.reshape(-1).tolist()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_one, record_id, args.output_dir, args.overwrite): record_id
            for record_id in record_ids
        }
        for future in as_completed(futures):
            record_id = future.result()
            print(f"downloaded {record_id}.mat")
    verify_cwru_files(args.output_dir)
    print(f"verified {len(record_ids)} files in {args.output_dir}")


if __name__ == "__main__":
    main()
