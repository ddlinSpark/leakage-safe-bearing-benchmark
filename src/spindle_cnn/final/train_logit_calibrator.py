import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from ..paths import DATA_DIR, DATASET_DIR, OUTPUTS_DIR
from . import postprocess_cwt_calibration as post
from . import train_cwt_residual_attention_compare as base
from .models.logit_calibrator import AffineCalibrator, ConditionAwareCalibrator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练独立的 logits calibrator，不重训主干模型")
    parser.add_argument(
        "--source-output-dir",
        default=str(OUTPUTS_DIR / "cwt_residual_attention_plus_gaus8"),
        help="主干模型输出目录，需包含 best_model.pt 与 run_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUTS_DIR / "cwt_residual_attention_plus_gaus8_v32a_affine_calibrator"),
        help="校准器输出目录",
    )
    parser.add_argument(
        "--calibrator-type",
        choices=["affine", "condition_aware", "confusion_aware"],
        default="affine",
        help="校准器类型",
    )
    parser.add_argument("--epochs", type=int, default=50, help="校准器训练轮数")
    parser.add_argument("--batch-size", type=int, default=256, help="校准器训练批大小")
    parser.add_argument("--learning-rate", type=float, default=1e-2, help="校准器学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--patience", type=int, default=10, help="早停 patience")
    parser.add_argument("--num-workers", type=int, default=4, help="主干模型 DataLoader worker 数")
    parser.add_argument(
        "--tta-time-shifts",
        type=str,
        default="0,-2,2",
        help="TTA 时间轴平移，逗号分隔",
    )
    parser.add_argument(
        "--tta-scale-shifts",
        type=str,
        default="0",
        help="TTA 尺度轴平移，逗号分隔",
    )
    parser.add_argument(
        "--key-class-indices",
        type=str,
        default="2,4,6,9",
        help="关键类别索引列表，用于输出关键类 F1 与退化报警",
    )
    parser.add_argument(
        "--condition-column",
        type=str,
        default="op",
        help="condition-aware calibrator 使用的 metadata 列名",
    )
    parser.add_argument("--embedding-dim", type=int, default=8, help="condition embedding 维度")
    parser.add_argument("--hidden-dim", type=int, default=32, help="condition-aware MLP 隐层维度")
    parser.add_argument("--dropout", type=float, default=0.1, help="condition-aware MLP dropout")
    parser.add_argument(
        "--key-loss-weight",
        type=float,
        default=0.0,
        help="关键类保护损失权重；仅在 confusion_aware 下生效",
    )
    parser.add_argument(
        "--baseline-v25-dir",
        default=str(OUTPUTS_DIR / "cwt_residual_attention_plus_gaus8_op_vector_scaling"),
        help="退化报警的 v25 参考目录",
    )
    parser.add_argument(
        "--baseline-v31-dir",
        default=str(
            OUTPUTS_DIR
            / "cwt_residual_attention_plus_gaus8_op_vector_scaling_local_bias_v3a"
        ),
        help="退化报警的 v31 参考目录",
    )
    parser.add_argument(
        "--max-degrade-v25",
        type=float,
        default=0.03,
        help="关键类 F1 相比 v25 允许的最大下降幅度",
    )
    parser.add_argument(
        "--max-degrade-v31",
        type=float,
        default=0.05,
        help="关键类 F1 相比 v31 允许的最大下降幅度",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def save_training_curve(history: List[Dict[str, float]], output_dir: Path) -> None:
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    axes[0].set_title("Calibrator Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["val_acc"], label="val_acc")
    axes[1].plot(history_df["epoch"], history_df["val_f1_macro"], label="val_f1_macro")
    axes[1].set_title("Calibrator Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=180)
    plt.close(fig)


def save_prediction_csv(
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    raw_pred: np.ndarray,
    calibrated_pred: np.ndarray,
    output_path: Path,
) -> None:
    prediction_df = metadata.copy().reset_index(drop=True)
    prediction_df["true_label"] = y_true
    prediction_df["true_label_name"] = [base.CLASS_NAMES[int(v)] for v in y_true]
    prediction_df["raw_pred_label"] = raw_pred
    prediction_df["raw_pred_label_name"] = [base.CLASS_NAMES[int(v)] for v in raw_pred]
    prediction_df["pred_label"] = calibrated_pred
    prediction_df["pred_label_name"] = [base.CLASS_NAMES[int(v)] for v in calibrated_pred]
    prediction_df.to_csv(output_path, index=False)


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def copy_static_artifacts(source_output_dir: Path, output_dir: Path) -> None:
    for file_name in [
        "label_distribution.png",
        "label_distribution.csv",
        "tsne_raw_data.png",
        "tsne_raw_data.csv",
        "tsne_model_features.png",
        "tsne_model_features.csv",
    ]:
        src = source_output_dir / file_name
        if src.exists():
            shutil.copy2(src, output_dir / file_name)


def save_split_cache(
    cache_dir: Path,
    split_name: str,
    logits: np.ndarray,
    labels: np.ndarray,
    embeddings: np.ndarray,
    indices: np.ndarray,
    metadata: pd.DataFrame,
) -> None:
    split_dir = cache_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / f"{split_name}_logits.npy", logits.astype(np.float32))
    np.save(split_dir / f"{split_name}_labels.npy", labels.astype(np.int64))
    np.save(split_dir / f"{split_name}_embeddings.npy", embeddings.astype(np.float32))
    np.save(split_dir / f"{split_name}_indices.npy", indices.astype(np.int64))
    metadata.iloc[indices].copy().reset_index(drop=True).to_csv(
        split_dir / f"{split_name}_metadata.csv",
        index=False,
    )


def load_split_cache(cache_dir: Path, split_name: str) -> Dict[str, object]:
    split_dir = cache_dir / split_name
    return {
        "logits": np.load(split_dir / f"{split_name}_logits.npy"),
        "labels": np.load(split_dir / f"{split_name}_labels.npy"),
        "embeddings": np.load(split_dir / f"{split_name}_embeddings.npy"),
        "indices": np.load(split_dir / f"{split_name}_indices.npy"),
        "metadata": pd.read_csv(split_dir / f"{split_name}_metadata.csv"),
    }


def get_tta_cache_dir(source_output_dir: Path, time_shifts: List[int], scale_shifts: List[int]) -> Path:
    time_token = "_".join(str(item) for item in time_shifts)
    scale_token = "_".join(str(item) for item in scale_shifts)
    return source_output_dir / "logits_cache_tta" / f"time_{time_token}__scale_{scale_token}"


def build_condition_ids(
    val_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame,
    condition_column: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    if condition_column not in val_metadata.columns or condition_column not in test_metadata.columns:
        raise KeyError(f"condition_column '{condition_column}' 不存在于缓存 metadata 中")
    vocab = sorted(
        {
            str(item)
            for item in pd.concat([val_metadata[condition_column], test_metadata[condition_column]], axis=0).tolist()
        }
    )
    mapping = {name: idx for idx, name in enumerate(vocab)}
    val_ids = val_metadata[condition_column].astype(str).map(mapping).to_numpy(dtype=np.int64)
    test_ids = test_metadata[condition_column].astype(str).map(mapping).to_numpy(dtype=np.int64)
    return val_ids, test_ids, mapping


def forward_calibrator(
    calibrator: torch.nn.Module,
    logits: torch.Tensor,
    condition_ids: torch.Tensor | None,
) -> torch.Tensor:
    if condition_ids is None:
        return calibrator(logits)
    return calibrator(logits, condition_ids)


def compute_calibration_loss(
    calibrated_logits: torch.Tensor,
    labels: torch.Tensor,
    criterion: torch.nn.Module,
    key_class_tensor: torch.Tensor | None,
    key_loss_weight: float,
) -> Tuple[torch.Tensor, float, float]:
    base_loss = criterion(calibrated_logits, labels)
    key_loss_value = 0.0
    total_loss = base_loss
    if key_class_tensor is not None and key_loss_weight > 0:
        key_mask = (labels.unsqueeze(1) == key_class_tensor.unsqueeze(0)).any(dim=1)
        if torch.any(key_mask):
            key_loss = criterion(calibrated_logits[key_mask], labels[key_mask])
            total_loss = base_loss + key_loss_weight * key_loss
            key_loss_value = float(key_loss.detach().cpu().item())
    return total_loss, float(base_loss.detach().cpu().item()), key_loss_value


def load_or_build_tta_cache(
    source_output_dir: Path,
    summary: Dict[str, object],
    time_shifts: List[int],
    scale_shifts: List[int],
    num_workers: int,
    seed: int,
) -> Dict[str, Dict[str, object]]:
    cache_dir = get_tta_cache_dir(source_output_dir, time_shifts, scale_shifts)
    manifest_path = cache_dir / "cache_manifest.json"
    required = [
        cache_dir / "val" / "val_logits.npy",
        cache_dir / "val" / "val_labels.npy",
        cache_dir / "val" / "val_metadata.csv",
        cache_dir / "test" / "test_logits.npy",
        cache_dir / "test" / "test_labels.npy",
        cache_dir / "test" / "test_metadata.csv",
    ]
    if manifest_path.exists() and all(path.exists() for path in required):
        return {
            "val": load_split_cache(cache_dir, "val"),
            "test": load_split_cache(cache_dir, "test"),
            "cache_dir": {"path": str(cache_dir)},
        }

    cwt_wavelets = [str(item) for item in summary["cwt_wavelets"]]
    cached = base.build_window_cache(
        dataset_csv=base.normalize_path(str(DATASET_DIR / "final_dataset_table.csv")),
        window_index_csv=base.normalize_path(
            str(DATASET_DIR / "window_index" / "window_sample_index.csv")
        ),
        limit_csv=base.normalize_path(str(DATA_DIR / "ms_cvtlimit.csv")),
        output_dir=source_output_dir,
        window_size=int(summary["window_size"]),
        gadf_points=int(summary["gadf_points"]),
        cwt_wavelets=cwt_wavelets,
        cwt_gradient_mix=float(summary.get("cwt_gradient_mix", 0.15)),
        cwt_add_diff=bool(summary.get("cwt_add_diff", False)),
        derived_quantile=0.90,
        normal_ratio=float(summary.get("normal_ratio", 0.30)),
        pseudo_label_mode=str(summary.get("pseudo_label_mode", "balanced_score")),
        margin_weight=float(summary.get("margin_weight", 0.75)),
        exclusive_ratio=float(summary.get("exclusive_ratio", 0.5)),
        confidence_power=float(summary.get("confidence_power", 1.5)),
        sample_weight_floor=float(summary.get("sample_weight_floor", 0.0)),
        max_samples_per_split=0,
        seed=seed,
    )

    _, val_loader, test_loader, _ = base.build_dataloaders(
        cached=cached,
        batch_size=int(summary["batch_size"]),
        num_workers=num_workers,
        use_weighted_sampler=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = post.build_model(summary, device=device)
    model.load_state_dict(torch.load(source_output_dir / "best_model.pt", map_location=device))
    shift_pairs = [(sy, sx) for sy in scale_shifts for sx in time_shifts]

    val_logits, val_true, val_embeddings, val_indices = post.collect_logits_with_tta(model, val_loader, device, shift_pairs)
    test_logits, test_true, test_embeddings, test_indices = post.collect_logits_with_tta(model, test_loader, device, shift_pairs)

    save_split_cache(cache_dir, "val", val_logits, val_true, val_embeddings, val_indices, cached.metadata)
    save_split_cache(cache_dir, "test", test_logits, test_true, test_embeddings, test_indices, cached.metadata)
    manifest = {
        "source_output_dir": str(source_output_dir),
        "time_shifts": time_shifts,
        "scale_shifts": scale_shifts,
        "seed": seed,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "val": load_split_cache(cache_dir, "val"),
        "test": load_split_cache(cache_dir, "test"),
        "cache_dir": {"path": str(cache_dir)},
    }


def load_class_f1_map(report_path: Path) -> Dict[str, float]:
    if not report_path.exists():
        return {}
    report_df = pd.read_csv(report_path, index_col=0)
    if "f1-score" not in report_df.columns:
        return {}
    return {
        str(class_name): float(report_df.loc[class_name, "f1-score"])
        for class_name in base.CLASS_NAMES
        if class_name in report_df.index
    }


def save_key_class_report(
    output_dir: Path,
    key_class_indices: List[int],
    current_f1: Dict[str, float],
    v25_f1: Dict[str, float],
    v31_f1: Dict[str, float],
    max_degrade_v25: float,
    max_degrade_v31: float,
) -> List[Dict[str, object]]:
    rows = []
    alerts: List[Dict[str, object]] = []
    for class_idx in key_class_indices:
        class_name = base.CLASS_NAMES[class_idx]
        current_value = float(current_f1.get(class_name, 0.0))
        v25_value = float(v25_f1.get(class_name, np.nan))
        v31_value = float(v31_f1.get(class_name, np.nan))
        row = {
            "class_index": class_idx,
            "class_name": class_name,
            "current_f1": current_value,
            "v25_f1": v25_value,
            "v31_f1": v31_value,
            "delta_vs_v25": current_value - v25_value if not np.isnan(v25_value) else np.nan,
            "delta_vs_v31": current_value - v31_value if not np.isnan(v31_value) else np.nan,
        }
        if not np.isnan(v25_value) and current_value < v25_value - max_degrade_v25:
            alerts.append(
                {
                    "type": "degrade_vs_v25",
                    "class_name": class_name,
                    "current_f1": current_value,
                    "baseline_f1": v25_value,
                    "max_drop": max_degrade_v25,
                }
            )
        if not np.isnan(v31_value) and current_value < v31_value - max_degrade_v31:
            alerts.append(
                {
                    "type": "degrade_vs_v31",
                    "class_name": class_name,
                    "current_f1": current_value,
                    "baseline_f1": v31_value,
                    "max_drop": max_degrade_v31,
                }
            )
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "key_class_f1.csv", index=False)
    return alerts


def main() -> None:
    args = parse_args()
    base.set_seed(args.seed)

    source_output_dir = base.normalize_path(args.source_output_dir)
    output_dir = base.normalize_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((source_output_dir / "run_summary.json").read_text(encoding="utf-8"))
    time_shifts = post.parse_shift_list(args.tta_time_shifts)
    scale_shifts = post.parse_shift_list(args.tta_scale_shifts)
    key_class_indices = parse_int_list(args.key_class_indices)
    copy_static_artifacts(source_output_dir, output_dir)

    tta_cache = load_or_build_tta_cache(
        source_output_dir=source_output_dir,
        summary=summary,
        time_shifts=time_shifts,
        scale_shifts=scale_shifts,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    val_cache = tta_cache["val"]
    test_cache = tta_cache["test"]

    val_condition_ids = None
    test_condition_ids = None
    condition_mapping: Dict[str, int] = {}
    if args.calibrator_type in {"condition_aware", "confusion_aware"}:
        val_condition_ids, test_condition_ids, condition_mapping = build_condition_ids(
            val_metadata=val_cache["metadata"],
            test_metadata=test_cache["metadata"],
            condition_column=args.condition_column,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.calibrator_type == "affine":
        calibrator = AffineCalibrator(num_classes=len(base.CLASS_NAMES)).to(device)
    else:
        calibrator = ConditionAwareCalibrator(
            num_classes=len(base.CLASS_NAMES),
            num_conditions=len(condition_mapping),
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        ).to(device)
    optimizer = torch.optim.Adam(
        calibrator.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()
    key_class_tensor = None
    effective_key_loss_weight = args.key_loss_weight if args.calibrator_type == "confusion_aware" else 0.0
    if effective_key_loss_weight > 0:
        key_class_tensor = torch.tensor(key_class_indices, dtype=torch.long, device=device)

    val_logits_tensor = torch.from_numpy(np.asarray(val_cache["logits"], dtype=np.float32))
    val_labels_tensor = torch.from_numpy(np.asarray(val_cache["labels"], dtype=np.int64))
    if val_condition_ids is None:
        train_dataset = TensorDataset(val_logits_tensor, val_labels_tensor)
    else:
        train_dataset = TensorDataset(
            val_logits_tensor,
            val_labels_tensor,
            torch.from_numpy(val_condition_ids.astype(np.int64)),
        )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    best_epoch = 0
    best_val_f1 = -1.0
    best_val_acc = -1.0
    patience_left = args.patience
    history: List[Dict[str, float]] = []
    best_model_path = output_dir / "best_calibrator.pt"

    for epoch in range(1, args.epochs + 1):
        calibrator.train()
        train_losses = []
        train_ce_losses = []
        train_key_losses = []
        for batch in train_loader:
            batch_logits = batch[0].to(device, non_blocking=True)
            batch_labels = batch[1].to(device, non_blocking=True)
            batch_condition_ids = None if len(batch) < 3 else batch[2].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            calibrated_logits = forward_calibrator(calibrator, batch_logits, batch_condition_ids)
            loss, ce_loss_value, key_loss_value = compute_calibration_loss(
                calibrated_logits=calibrated_logits,
                labels=batch_labels,
                criterion=criterion,
                key_class_tensor=key_class_tensor,
                key_loss_weight=effective_key_loss_weight,
            )
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))
            train_ce_losses.append(ce_loss_value)
            train_key_losses.append(key_loss_value)

        calibrator.eval()
        with torch.no_grad():
            val_logits_device = val_logits_tensor.to(device)
            val_labels_device = val_labels_tensor.to(device)
            val_condition_device = (
                None
                if val_condition_ids is None
                else torch.from_numpy(val_condition_ids.astype(np.int64)).to(device)
            )
            calibrated_val_logits = forward_calibrator(calibrator, val_logits_device, val_condition_device)
            val_loss = float(criterion(calibrated_val_logits, val_labels_device).detach().cpu().item())
            val_pred = torch.argmax(calibrated_val_logits, dim=1).cpu().numpy()
            val_true = val_labels_tensor.cpu().numpy()
            val_acc = float(np.mean(val_pred == val_true))
            val_f1 = float(
                f1_score(
                    val_true,
                    val_pred,
                    labels=np.arange(len(base.CLASS_NAMES)),
                    average="macro",
                    zero_division=0,
                )
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
                "train_ce_loss": float(np.mean(train_ce_losses)) if train_ce_losses else 0.0,
                "train_key_loss": float(np.mean(train_key_losses)) if train_key_losses else 0.0,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1_macro": val_f1,
            }
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={history[-1]['train_loss']:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )

        improved = (val_f1 > best_val_f1 + 1e-12) or (
            abs(val_f1 - best_val_f1) <= 1e-12 and val_acc > best_val_acc + 1e-12
        )
        if improved:
            best_epoch = epoch
            best_val_f1 = val_f1
            best_val_acc = val_acc
            patience_left = args.patience
            torch.save(calibrator.state_dict(), best_model_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    save_training_curve(history, output_dir)
    calibrator.load_state_dict(torch.load(best_model_path, map_location=device))
    calibrator.eval()

    with torch.no_grad():
        val_logits_device = torch.from_numpy(np.asarray(val_cache["logits"], dtype=np.float32)).to(device)
        test_logits_device = torch.from_numpy(np.asarray(test_cache["logits"], dtype=np.float32)).to(device)
        val_condition_device = (
            None
            if val_condition_ids is None
            else torch.from_numpy(val_condition_ids.astype(np.int64)).to(device)
        )
        test_condition_device = (
            None
            if test_condition_ids is None
            else torch.from_numpy(test_condition_ids.astype(np.int64)).to(device)
        )
        calibrated_val_logits = forward_calibrator(calibrator, val_logits_device, val_condition_device).cpu().numpy()
        calibrated_test_logits = forward_calibrator(calibrator, test_logits_device, test_condition_device).cpu().numpy()

    val_true = np.asarray(val_cache["labels"], dtype=np.int64)
    test_true = np.asarray(test_cache["labels"], dtype=np.int64)
    val_pred = np.argmax(calibrated_val_logits, axis=1).astype(np.int64)
    test_pred = np.argmax(calibrated_test_logits, axis=1).astype(np.int64)
    raw_test_pred = np.argmax(np.asarray(test_cache["logits"]), axis=1).astype(np.int64)

    metrics = base.save_metrics(test_true, test_pred, output_dir)
    base.save_confusion_matrix(test_true, test_pred, output_dir)
    save_prediction_csv(
        metadata=test_cache["metadata"],
        y_true=test_true,
        raw_pred=raw_test_pred,
        calibrated_pred=test_pred,
        output_path=output_dir / "test_predictions.csv",
    )
    np.save(output_dir / "calibrated_val_logits.npy", calibrated_val_logits.astype(np.float32))
    np.save(output_dir / "calibrated_test_logits.npy", calibrated_test_logits.astype(np.float32))

    v25_f1 = load_class_f1_map(base.normalize_path(args.baseline_v25_dir) / "classification_report.csv")
    v31_f1 = load_class_f1_map(base.normalize_path(args.baseline_v31_dir) / "classification_report.csv")
    current_f1 = load_class_f1_map(output_dir / "classification_report.csv")
    alerts = save_key_class_report(
        output_dir=output_dir,
        key_class_indices=key_class_indices,
        current_f1=current_f1,
        v25_f1=v25_f1,
        v31_f1=v31_f1,
        max_degrade_v25=args.max_degrade_v25,
        max_degrade_v31=args.max_degrade_v31,
    )

    val_metrics = {
        "accuracy": float(np.mean(val_pred == val_true)),
        "f1_macro": float(
            f1_score(
                val_true,
                val_pred,
                labels=np.arange(len(base.CLASS_NAMES)),
                average="macro",
                zero_division=0,
            )
        ),
    }

    calibrator_details = {
        "calibrator_type": args.calibrator_type,
        "source_output_dir": str(source_output_dir),
        "tta_cache_dir": tta_cache["cache_dir"]["path"],
        "tta_time_shifts": time_shifts,
        "tta_scale_shifts": scale_shifts,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "best_val_f1_macro": best_val_f1,
        "key_class_indices": key_class_indices,
        "key_loss_weight": effective_key_loss_weight,
        "alerts": alerts,
        "history": history,
    }
    if args.calibrator_type == "affine":
        calibrator_details["learned_scale"] = calibrator.scale.detach().cpu().numpy().tolist()
        calibrator_details["learned_bias"] = calibrator.bias.detach().cpu().numpy().tolist()
    else:
        calibrator_details["condition_column"] = args.condition_column
        calibrator_details["condition_mapping"] = condition_mapping
        calibrator_details["embedding_dim"] = args.embedding_dim
        calibrator_details["hidden_dim"] = args.hidden_dim
        calibrator_details["dropout"] = args.dropout
    (output_dir / "calibrator_details.json").write_text(
        json.dumps(calibrator_details, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    result_payload = {
        "model_name": summary["model_name"],
        "transform_type": summary.get("transform_type", "cwt"),
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "window_size": summary["window_size"],
        "gadf_points": summary["gadf_points"],
        "image_size": summary["image_size"],
        "cwt_wavelets": summary.get("cwt_wavelets", []),
        "cwt_gradient_mix": summary.get("cwt_gradient_mix", 0.15),
        "normal_ratio": summary.get("normal_ratio", 0.30),
        "train_samples": summary["train_samples"],
        "val_samples": summary["val_samples"],
        "test_samples": summary["test_samples"],
        "metrics": metrics,
        "class_names": base.CLASS_NAMES,
        "postprocess_calibration_method": (
            "learnable_affine"
            if args.calibrator_type == "affine"
            else ("condition_aware_calibrator" if args.calibrator_type == "condition_aware" else "confusion_aware_calibrator")
        ),
        "postprocess_group_column": "global" if args.calibrator_type == "affine" else args.condition_column,
        "postprocess_tta_shift_pairs": [(sy, sx) for sy in scale_shifts for sx in time_shifts],
        "postprocess_val_metrics": val_metrics,
        "postprocess_source_dir": str(source_output_dir),
        "logits_cache_dir": tta_cache["cache_dir"]["path"],
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
