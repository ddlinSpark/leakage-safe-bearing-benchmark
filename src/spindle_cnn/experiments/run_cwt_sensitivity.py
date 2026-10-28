"""Run CWT preprocessing controls under the canonical fold-seed protocol."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..experiments.common import (
    atomic_write_json,
    configure_logging,
    ensure_clean_worktree,
    file_sha256,
    load_json_config,
)
from ..experiments.run_benchmark import resolve_device, run_main_experiment, summarize_results
from ..paths import PROJECT_ROOT
from ..public_benchmark.cwru_data import CWRUDataset, load_cwru_cache
from ..public_benchmark.cwt_sensitivity import VARIANTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "paderborn_main.json",
    )
    parser.add_argument(
        "--variants-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "public" / "paderborn" / "cwt_sensitivity",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "cwt_sensitivity",
    )
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--device")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json_config(args.config.resolve())
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown CWT variants: {unknown}")
    base_cache = (PROJECT_ROOT / config["dataset"]["cache"]).resolve()
    base = load_cwru_cache(base_cache)
    base_hash = file_sha256(base_cache)
    provenance = ensure_clean_worktree(args.allow_dirty)
    device = resolve_device(args.device or config["training"]["device"])

    for variant in variants:
        variant_path = args.variants_dir / f"{variant}.npz"
        metadata_path = variant_path.with_suffix(".json")
        if not variant_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                f"Missing CWT variant cache or metadata: {variant_path}"
            )
        with np.load(variant_path) as payload:
            variant_cwt = payload["cwt"]
            labels = payload["labels"]
            loads = payload["loads"]
            record_ids = payload["record_ids"]
            starts = payload["starts"]
        if not (
            np.array_equal(labels, base.labels)
            and np.array_equal(loads, base.loads)
            and np.array_equal(record_ids, base.record_ids)
            and np.array_equal(starts, base.starts)
        ):
            raise ValueError(f"CWT variant metadata mismatch: {variant}")
        data = CWRUDataset(
            raw=base.raw,
            cwt=variant_cwt,
            labels=base.labels,
            loads=base.loads,
            record_ids=base.record_ids,
            starts=base.starts,
        )
        variant_config: dict[str, Any] = copy.deepcopy(config)
        variant_config["dataset"]["name"] = f"Paderborn-4class-CWT-{variant}"
        variant_config["dataset"]["sensitivity_variant"] = variant
        variant_config["experiments"]["main"]["methods"] = ["cwt_cnn"]
        variant_config["experiments"]["main"]["fractions"] = [1.0]
        output_dir = args.output_dir / variant
        output_dir.mkdir(parents=True, exist_ok=True)
        configure_logging(output_dir / "run_cwt_sensitivity.log")
        atomic_write_json(output_dir / "config_used.json", variant_config)
        dataset_hash = file_sha256(variant_path) + ":" + base_hash
        run_main_experiment(
            variant_config,
            methods=["cwt_cnn"],
            fractions=[1.0],
            experiment_type="main",
            output_dir=output_dir,
            data=data,
            device=device,
            provenance=provenance,
            dataset_hash=dataset_hash,
            resume=not args.no_resume,
        )
        summarize_results(output_dir)


if __name__ == "__main__":
    main()
