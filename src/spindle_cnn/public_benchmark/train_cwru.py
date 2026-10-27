"""Cross-load CWRU experiments for supervised and calibrated self-training."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset

from ..paths import DATA_DIR, OUTPUTS_DIR
from .cwru_data import CWRUDataset, load_cwru_cache
from .models import build_model


@dataclass
class TrainState:
    model: nn.Module
    temperature: float
    best_val_f1: float
    epochs_ran: int


class WindowSet(Dataset):
    def __init__(
        self,
        data: CWRUDataset,
        indices: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray | None = None,
        augment: bool = False,
    ) -> None:
        self.raw = torch.from_numpy(data.raw)
        self.cwt = torch.from_numpy(data.cwt)
        self.indices = torch.as_tensor(indices, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.weights = torch.as_tensor(
            np.ones(len(indices), dtype=np.float32) if weights is None else weights,
            dtype=torch.float32,
        )
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, local_index: int):
        global_index = int(self.indices[local_index])
        raw = self.raw[global_index].float()
        cwt = self.cwt[global_index].float()
        if self.augment:
            shift = int(torch.randint(-32, 33, (1,)).item())
            raw = torch.roll(raw, shifts=shift, dims=-1)
            cwt = torch.roll(cwt, shifts=shift // 16, dims=-1)
            if torch.rand(1).item() < 0.5:
                scale = 0.005 + 0.01 * torch.rand(1).item()
                raw = raw + torch.randn_like(raw) * scale
                cwt = torch.clamp(cwt + torch.randn_like(cwt) * 0.01, min=0.0)
        return raw, cwt, self.labels[local_index], self.weights[local_index], local_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CWRU 10 类跨负载基准实验")
    parser.add_argument(
        "--cache",
        type=Path,
        default=DATA_DIR / "public" / "cwru" / "cwru_10class_w2048_morl.npz",
        help="由 cwru_data.py 生成的缓存",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "public_cwru",
        help="结果输出目录",
    )
    parser.add_argument("--dataset-name", default="CWRU-12k-10class")
    parser.add_argument(
        "--heldout-record-ids",
        default="",
        help="只用于测试的轴承记录编号，逗号分隔；留空表示所有记录参与各负载折",
    )
    parser.add_argument(
        "--methods",
        default="raw_cnn,cwt_cnn,cwt_gru,cwt_gru_selftrain",
        help="逗号分隔的方法列表",
    )
    parser.add_argument(
        "--fractions",
        default="0.1,0.25,0.5,1.0",
        help="训练集中有真实标签的类别平衡比例",
    )
    parser.add_argument("--folds", default="0,1,2,3", help="测试负载编号，逗号分隔")
    parser.add_argument("--seeds", default="42,43,44", help="随机种子，逗号分隔")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--pseudo-rounds", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true", help="仅运行一个快速配置")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def forward_model(model: nn.Module, raw: torch.Tensor, cwt: torch.Tensor) -> torch.Tensor:
    if isinstance(model, nn.Module):
        return model(raw, cwt)
    raise TypeError("model must be a torch module")


def balanced_indices(
    indices: np.ndarray,
    labels: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    num_classes = int(labels.max()) + 1
    by_class = [indices[labels[indices] == class_id] for class_id in range(num_classes)]
    per_class = min(len(items) for items in by_class)
    take = max(1, math.ceil(per_class * fraction))
    selected = [rng.choice(items, size=take, replace=False) for items in by_class]
    return np.concatenate(selected)


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for raw, cwt, labels, _, _ in loader:
            logits = forward_model(model, raw.to(device), cwt.to(device))
            logits_parts.append(logits.cpu().numpy())
            label_parts.append(labels.numpy())
    return np.concatenate(logits_parts), np.concatenate(label_parts)


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if not np.any(mask):
            continue
        accuracy = (predicted[mask] == labels[mask]).mean()
        error += mask.mean() * abs(accuracy - confidence[mask].mean())
    return float(error)


def metrics_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    temperature: float = 1.0,
) -> dict[str, float]:
    scaled = logits / temperature
    probabilities = F.softmax(torch.from_numpy(scaled), dim=1).numpy()
    predicted = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro")),
        "macro_precision": float(precision_score(labels, predicted, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, predicted, average="macro", zero_division=0)),
        "ece": expected_calibration_error(probabilities, labels),
    }


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    logits_tensor = torch.from_numpy(logits)
    labels_tensor = torch.from_numpy(labels)
    candidates = np.concatenate(
        [
            np.linspace(0.5, 2.0, 31),
            np.linspace(2.2, 5.0, 15),
        ]
    )
    losses = [
        float(
            F.cross_entropy(logits_tensor / float(temperature), labels_tensor).item()
        )
        for temperature in candidates
    ]
    return float(candidates[int(np.argmin(losses))])


def train_model(
    data: CWRUDataset,
    train_indices: np.ndarray,
    train_labels: np.ndarray,
    train_weights: np.ndarray,
    val_indices: np.ndarray,
    method: str,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    num_workers: int,
    device: torch.device,
    seed: int,
) -> TrainState:
    model_name = "cwt_gru" if method == "cwt_gru_selftrain" else method
    model = build_model(
        model_name,
        num_classes=int(data.labels.max()) + 1,
        raw_channels=data.raw.shape[1],
        image_channels=data.cwt.shape[1],
    ).to(device)
    train_set = WindowSet(
        data,
        train_indices,
        train_labels,
        train_weights,
        augment=True,
    )
    val_set = WindowSet(data, val_indices, data.labels[val_indices])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_state = copy.deepcopy(model.state_dict())
    best_f1 = -1.0
    best_epoch = 0
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for raw, cwt, labels, weights, _ in train_loader:
            raw = raw.to(device, non_blocking=True)
            cwt = cwt.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = forward_model(model, raw, cwt)
                losses = F.cross_entropy(logits, labels, reduction="none")
                loss = (losses * weights).sum() / weights.sum().clamp_min(1e-6)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        val_logits, val_labels = predict(model, val_loader, device)
        val_f1 = metrics_from_logits(val_logits, val_labels)["macro_f1"]
        if val_f1 > best_f1 + 1e-5:
            best_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"    epoch={epoch:02d} val_macro_f1={val_f1:.4f} "
                f"best={best_f1:.4f}"
            )
        if stale >= patience:
            break

    model.load_state_dict(best_state)
    val_logits, val_labels = predict(model, val_loader, device)
    temperature = fit_temperature(val_logits, val_labels)
    return TrainState(
        model=model,
        temperature=temperature,
        best_val_f1=best_f1,
        epochs_ran=best_epoch,
    )


def select_pseudo_labels(
    model: nn.Module,
    data: CWRUDataset,
    candidate_indices: np.ndarray,
    temperature: float,
    max_per_class: int,
    confidence_threshold: float,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = WindowSet(data, candidate_indices, data.labels[candidate_indices])
    loader = DataLoader(
        dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    logits, _ = predict(model, loader, device)
    probabilities = F.softmax(torch.from_numpy(logits / temperature), dim=1).numpy()
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)

    selected: list[int] = []
    num_classes = int(data.labels.max()) + 1
    for class_id in range(num_classes):
        class_positions = np.flatnonzero(predicted == class_id)
        class_positions = class_positions[confidence[class_positions] >= confidence_threshold]
        if len(class_positions) == 0:
            continue
        order = class_positions[np.argsort(confidence[class_positions])[::-1]]
        selected.extend(order[:max_per_class].tolist())
    selected_array = np.asarray(selected, dtype=np.int64)
    return (
        candidate_indices[selected_array],
        predicted[selected_array].astype(np.int64),
        confidence[selected_array].astype(np.float32),
    )


def run_self_training(
    data: CWRUDataset,
    labeled_indices: np.ndarray,
    candidate_indices: np.ndarray,
    val_indices: np.ndarray,
    *,
    rounds: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    num_workers: int,
    device: torch.device,
    seed: int,
) -> TrainState:
    labels = data.labels[labeled_indices]
    weights = np.ones(len(labeled_indices), dtype=np.float32)
    state = train_model(
        data,
        labeled_indices,
        labels,
        weights,
        val_indices,
        "cwt_gru",
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        patience=patience,
        num_workers=num_workers,
        device=device,
        seed=seed,
    )
    num_classes = int(data.labels.max()) + 1
    max_per_class = max(1, len(labeled_indices) // num_classes * 2)
    for round_id in range(1, rounds + 1):
        pseudo_idx, pseudo_labels, pseudo_weights = select_pseudo_labels(
            state.model,
            data,
            candidate_indices,
            state.temperature,
            max_per_class=max_per_class,
            confidence_threshold=0.6,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )
        if len(pseudo_idx) == 0:
            print("    no pseudo-labels passed the confidence threshold")
            break
        combined_idx = np.concatenate([labeled_indices, pseudo_idx])
        combined_labels = np.concatenate([labels, pseudo_labels])
        combined_weights = np.concatenate([weights, pseudo_weights])
        print(
            f"    self-training round={round_id} pseudo={len(pseudo_idx)} "
            f"balanced_cap={max_per_class}"
        )
        state = train_model(
            data,
            combined_idx,
            combined_labels,
            combined_weights,
            val_indices,
            "cwt_gru_selftrain",
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=patience,
            num_workers=num_workers,
            device=device,
            seed=seed + round_id,
        )
        max_per_class *= 2
    return state


def load_fold_indices(
    data: CWRUDataset,
    test_load: int,
    heldout_record_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    remaining = [load for load in range(4) if load != test_load]
    val_load = remaining[test_load % len(remaining)]
    train_loads = [load for load in remaining if load != val_load]
    if heldout_record_ids is not None and len(heldout_record_ids) > 0:
        train_val_mask = ~np.isin(data.record_ids, heldout_record_ids)
        test_mask = np.isin(data.record_ids, heldout_record_ids)
        train_idx = np.flatnonzero(train_val_mask & np.isin(data.loads, train_loads))
        val_idx = np.flatnonzero(train_val_mask & (data.loads == val_load))
        test_idx = np.flatnonzero(test_mask & (data.loads == test_load))
    else:
        train_idx = np.flatnonzero(np.isin(data.loads, train_loads))
        val_idx = np.flatnonzero(data.loads == val_load)
        test_idx = np.flatnonzero(data.loads == test_load)
    return train_idx, val_idx, test_idx


def evaluate_model(
    state: TrainState,
    data: CWRUDataset,
    test_indices: np.ndarray,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float], np.ndarray, np.ndarray]:
    test_set = WindowSet(data, test_indices, data.labels[test_indices])
    loader = DataLoader(
        test_set,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    logits, labels = predict(state.model, loader, device)
    uncalibrated = metrics_from_logits(logits, labels, temperature=1.0)
    calibrated = metrics_from_logits(logits, labels, temperature=state.temperature)
    return uncalibrated, calibrated, logits, labels


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.folds = "3"
        args.seeds = "42"
        args.fractions = "0.1"
        args.epochs = min(args.epochs, 2)
        args.patience = 2
        args.methods = "raw_cnn,cwt_cnn,cwt_gru,cwt_gru_selftrain"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data = load_cwru_cache(args.cache)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    fractions = [float(item) for item in args.fractions.split(",") if item.strip()]
    folds = [int(item) for item in args.folds.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    heldout_record_ids = np.asarray(
        [
            int(item)
            for item in args.heldout_record_ids.split(",")
            if item.strip()
        ],
        dtype=np.int64,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.csv"
    result_rows: list[dict] = []
    if results_path.is_file():
        result_rows = pd.read_csv(results_path).to_dict(orient="records")
    existing_keys = {
        (
            row.get("dataset"),
            int(row.get("test_condition")),
            row.get("method"),
            round(float(row.get("label_fraction")), 4),
            int(row.get("seed")),
        )
        for row in result_rows
    }

    for fold in folds:
        train_idx_all, val_idx, test_idx = load_fold_indices(
            data,
            fold,
            heldout_record_ids=heldout_record_ids,
        )
        print(
            f"fold test_condition={fold} "
            f"train={len(train_idx_all)} val={len(val_idx)} test={len(test_idx)}"
        )
        for method in methods:
            applicable_fractions = [1.0] if method in {"raw_cnn", "cwt_cnn"} else fractions
            for fraction in applicable_fractions:
                for seed in seeds:
                    candidate_key = (
                        args.dataset_name,
                        fold,
                        method,
                        round(fraction, 4),
                        seed,
                    )
                    if candidate_key in existing_keys:
                        print(
                            f"  skip existing method={method} fraction={fraction:.2f} "
                            f"condition={fold} seed={seed}"
                        )
                        continue
                    set_seed(seed)
                    rng = np.random.default_rng(seed + 1000 * fold)
                    labeled_idx = balanced_indices(train_idx_all, data.labels, fraction, rng)
                    candidate_mask = np.ones(len(train_idx_all), dtype=bool)
                    candidate_mask[np.isin(train_idx_all, labeled_idx)] = False
                    candidate_idx = train_idx_all[candidate_mask]
                    print(
                        f"  method={method} fraction={fraction:.2f} seed={seed} "
                        f"labeled={len(labeled_idx)} unlabeled={len(candidate_idx)}"
                    )
                    if method == "cwt_gru_selftrain":
                        state = run_self_training(
                            data,
                            labeled_idx,
                            candidate_idx,
                            val_idx,
                            rounds=args.pseudo_rounds,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            learning_rate=args.learning_rate,
                            patience=args.patience,
                            num_workers=args.num_workers,
                            device=device,
                            seed=seed,
                        )
                    else:
                        state = train_model(
                            data,
                            labeled_idx,
                            data.labels[labeled_idx],
                            np.ones(len(labeled_idx), dtype=np.float32),
                            val_idx,
                            method,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            learning_rate=args.learning_rate,
                            patience=args.patience,
                            num_workers=args.num_workers,
                            device=device,
                            seed=seed,
                        )
                    raw_metrics, calibrated_metrics, test_logits, test_labels = evaluate_model(
                        state,
                        data,
                        test_idx,
                        args.batch_size,
                        args.num_workers,
                        device,
                    )
                    prediction_dir = args.output_dir / "predictions"
                    prediction_dir.mkdir(exist_ok=True)
                    prediction_name = (
                        f"fold{fold}_{method}_fraction{fraction:.2f}_seed{seed}.npz"
                    )
                    np.savez(
                        prediction_dir / prediction_name,
                        logits=test_logits,
                        labels=test_labels,
                        temperature=state.temperature,
                    )
                    row = {
                        "dataset": args.dataset_name,
                        "protocol": "grouped-cross-condition",
                        "test_condition": fold,
                        "method": method,
                        "label_fraction": fraction,
                        "seed": seed,
                        "labeled_windows": len(labeled_idx),
                        "temperature": state.temperature,
                        "best_val_macro_f1": state.best_val_f1,
                        "epochs_ran": state.epochs_ran,
                        **{f"raw_{key}": value for key, value in raw_metrics.items()},
                        **{f"cal_{key}": value for key, value in calibrated_metrics.items()},
                    }
                    result_rows.append(row)
                    pd.DataFrame(result_rows).to_csv(
                        results_path,
                        index=False,
                    )
                    existing_keys.add(candidate_key)
                    print(
                        f"    test_acc={raw_metrics['accuracy']:.4f} "
                        f"macro_f1={raw_metrics['macro_f1']:.4f} "
                        f"ece={calibrated_metrics['ece']:.4f}"
                    )

    results = pd.DataFrame(result_rows)
    results.to_csv(args.output_dir / "results.csv", index=False)
    summary = (
        results.groupby(["method", "label_fraction"])
        .agg(
            accuracy_mean=("raw_accuracy", "mean"),
            accuracy_std=("raw_accuracy", "std"),
            macro_f1_mean=("raw_macro_f1", "mean"),
            macro_f1_std=("raw_macro_f1", "std"),
            calibrated_ece_mean=("cal_ece", "mean"),
            calibrated_ece_std=("cal_ece", "std"),
            runs=("seed", "count"),
        )
        .reset_index()
    )
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
