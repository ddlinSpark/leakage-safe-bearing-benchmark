"""Evaluate training-statistic normalization as a preprocessing sensitivity."""

from __future__ import annotations

import argparse
import copy
import hashlib
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..datasets.splits import (
    build_split_manifest,
    load_fold_indices,
    write_json,
)
from ..experiments.common import (
    atomic_write_json,
    ensure_clean_worktree,
    file_sha256,
    load_json_config,
)
from ..experiments.run_benchmark import (
    resolve_device,
    train_and_evaluate,
)
from ..models.registry import METHODS
from ..paths import PROJECT_ROOT
from ..public_benchmark.cwru_data import CWRUDataset
from ..public_benchmark.cwt_sensitivity import _process_chunk


METHODS_TO_RUN = ["raw_cnn", "cwt_cnn", "cwt_gru"]


def make_cwt(raw_path: Path, count: int, *, workers: int) -> np.ndarray:
    result = np.empty((count, 3, 96, 96), dtype=np.float16)
    chunks = [(start, min(start + 64, count)) for start in range(0, count, 64)]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _process_chunk,
                str(raw_path),
                start,
                stop,
                48,
                96,
                "magnitude",
                True,
            )
            for start, stop in chunks
        ]
        for index, ((start, stop), future) in enumerate(
            zip(chunks, futures, strict=True),
            start=1,
        ):
            result[start:stop] = future.result()
            if index % 10 == 0 or stop == count:
                print(f"CWT windows: {stop}/{count}", flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "paderborn_main.json",
    )
    parser.add_argument(
        "--raw-cache",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "public"
        / "paderborn"
        / "pu_4class_w8192_raw.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "normalization_sensitivity",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "public" / "paderborn" / "normalization_b",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.raw_cache.is_file():
        raise FileNotFoundError(args.raw_cache)
    config = copy.deepcopy(load_json_config(args.config.resolve()))
    config["dataset"]["name"] += "-training-stat-normalized"
    config["dataset"]["normalization"] = "training-partition channel mean/std"
    config["dataset"]["sensitivity_analysis"] = True
    provenance = ensure_clean_worktree(args.allow_dirty)
    device = resolve_device(args.device or config["training"]["device"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "config_used.json", config)
    raw_hash = file_sha256(args.raw_cache)

    with np.load(args.raw_cache, allow_pickle=False) as cache:
        raw_original = cache["raw"]
        labels = cache["labels"]
        loads = cache["loads"]
        record_ids = cache["record_ids"]
        starts = cache["starts"]
    heldout = np.asarray(config["dataset"]["heldout_record_ids"], dtype=np.int64)
    base = CWRUDataset(
        raw=raw_original,
        cwt=np.empty((len(raw_original), 3, 1, 1), dtype=np.float16),
        labels=labels,
        loads=loads,
        record_ids=record_ids,
        starts=starts,
    )

    for fold in config["folds"]:
        split = load_fold_indices(base, int(fold), heldout)
        fit_windows = raw_original[split.train].astype(np.float64)
        channel_mean = fit_windows.mean(axis=(0, 2), dtype=np.float64)
        channel_std = fit_windows.std(axis=(0, 2), dtype=np.float64)
        channel_std = np.maximum(channel_std, 1e-6)
        normalized = (
            (raw_original.astype(np.float32) - channel_mean[None, :, None])
            / channel_std[None, :, None]
        ).astype(np.float32)
        prepared_path = args.prepared_dir / f"fold{fold}.npz"
        if prepared_path.is_file():
            with np.load(prepared_path) as prepared:
                cwt = prepared["cwt"]
                saved_mean = prepared["channel_mean"]
                saved_std = prepared["channel_std"]
            if not (
                np.allclose(saved_mean, channel_mean)
                and np.allclose(saved_std, channel_std)
            ):
                raise ValueError(f"Prepared normalization statistics mismatch: fold {fold}")
        else:
            with tempfile.TemporaryDirectory(prefix="bearing-norm-") as temporary:
                raw_path = Path(temporary) / "raw.npz"
                np.savez(raw_path, raw=normalized)
                cwt = make_cwt(raw_path, len(normalized), workers=args.workers)
            if args.prepare_only:
                prepared_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    prepared_path,
                    cwt=cwt,
                    channel_mean=channel_mean,
                    channel_std=channel_std,
                )
                print(f"Prepared normalization cache: {prepared_path}", flush=True)
                continue
        if args.prepare_only:
            print(f"Reusing normalization cache: {prepared_path}", flush=True)
            continue
        data = CWRUDataset(
            raw=normalized,
            cwt=cwt,
            labels=labels,
            loads=loads,
            record_ids=record_ids,
            starts=starts,
        )
        manifest = build_split_manifest(
            data,
            split,
            dataset_name=config["dataset"]["name"],
            split_version=config["dataset"]["split_version"] + "-norm-b",
            fold=int(fold),
            heldout_record_ids=heldout,
        )
        split_hash = write_json(
            args.output_dir / "splits" / f"fold{fold}.json",
            manifest,
        )
        stats_hash = hashlib.sha256(
            np.concatenate([channel_mean, channel_std]).astype("<f8").tobytes()
        ).hexdigest()
        dataset_hash = f"{raw_hash}:fold{fold}:{stats_hash}"
        print(
            f"fold={fold} normalization mean={channel_mean.tolist()} "
            f"std={channel_std.tolist()}",
            flush=True,
        )
        for seed in config["seeds"]:
            for method in METHODS_TO_RUN:
                train_and_evaluate(
                    data=data,
                    split=split,
                    experiment_type="normalization_sensitivity",
                    method=method,
                    method_spec=METHODS[method],
                    fold=int(fold),
                    seed=int(seed),
                    label_ratio=1.0,
                    confidence_threshold=float(
                        config["training"]["pseudo_threshold"]
                    ),
                    config=config,
                    device=device,
                    results_path=args.output_dir / "raw_runs.csv",
                    runs_dir=args.output_dir / "runs",
                    split_hash=split_hash,
                    dataset_hash=dataset_hash,
                    provenance=provenance,
                )


if __name__ == "__main__":
    main()
