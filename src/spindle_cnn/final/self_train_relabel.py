"""
Self-training: 用 v33 校准模型对全量数据生成新伪标签，并重训主干。

流程:
1. 加载 v16 的 CWT 缓存（raw_sequences + gadf_images）
2. 加载 v16 主干 + v33 校准器
3. 对 train/val/test 全部样本执行 TTA 推理 + 校准
4. 生成平衡的新伪标签
5. 保存到新输出目录并重训主干
"""

import argparse
import json
import shutil
from typing import Dict, List, Tuple
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 非交互式后端，避免 Qt 依赖

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from ..paths import OUTPUTS_DIR
from . import postprocess_cwt_calibration as post
from . import train_cwt_residual_attention_compare as base
from .models.logit_calibrator import ConditionAwareCalibrator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-training: relabel data with calibrator and retrain backbone")
    parser.add_argument(
        "--source-output-dir",
        default=str(OUTPUTS_DIR / "cwt_residual_attention_plus_gaus8"),
        help="v16 主干模型输出目录",
    )
    parser.add_argument(
        "--calibrator-dir",
        default=str(OUTPUTS_DIR / "cwt_residual_attention_plus_gaus8_v32c_lambda05"),
        help="v33 校准器输出目录",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUTS_DIR / "self_train_v1"),
        help="self-training 输出目录",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=7e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--model-name",
        choices=["residual_attention", "residual_attention_gru"],
        default="residual_attention_gru",
        help="重训时使用的模型架构（TTA 推理始终用源主干架构）",
    )
    parser.add_argument("--coarse-loss-weight", type=float, default=0.0,
                        help="层次粗分类损失权重")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def collect_calibrated_logits_for_all_splits(
    model: torch.nn.Module,
    calibrator: torch.nn.Module,
    cached: base.CachedArrays,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    shift_pairs: List[Tuple[int, int]],
    condition_mapping: Dict[str, int],
) -> Dict[str, Dict[str, np.ndarray]]:
    """对 train/val/test 全部样本执行 TTA 推理 + 校准，返回 logits/labels/indices。"""
    results: Dict[str, Dict[str, np.ndarray]] = {}
    condition_column = "op"

    split_meta = cached.metadata
    for split_name in ["train", "val", "test"]:
        split_indices = split_meta.index[
            split_meta["dataset_split"] == split_name
        ].to_numpy()

        dataset = base.WindowDataset(
            raw_sequences=cached.raw_sequences,
            gadf_images=cached.gadf_images,
            labels=cached.labels,
            metadata=cached.metadata,
            indices=split_indices,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

        model.eval()
        calibrator.eval()

        logits_list = []
        targets = []
        indices_list = []

        for batch in loader:
            raw_seq = batch["raw_seq"].to(device, non_blocking=True)
            gadf = batch["gadf"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            batch_indices = batch["index"]

            # TTA
            logits_sum = None
            for shift_y, shift_x in shift_pairs:
                gadf_tta = post.shift_with_replicate(gadf, shift_y=shift_y, shift_x=shift_x)
                logits, _, _ = model(raw_seq, gadf_tta)
                logits_sum = logits if logits_sum is None else logits_sum + logits

            tta_logits = logits_sum / max(len(shift_pairs), 1)

            # 校准器推理: batch_indices 是全局索引
            global_idx_list = batch_indices.cpu().numpy().tolist()
            batch_ops = [
                str(cached.metadata.iloc[int(idx)][condition_column])
                for idx in global_idx_list
            ]
            condition_ids = torch.tensor(
                [condition_mapping.get(op, 0) for op in batch_ops],
                dtype=torch.long,
                device=device,
            )
            calibrated = calibrator(tta_logits, condition_ids)

            logits_list.append(calibrated.cpu().numpy())
            targets.append(label.cpu().numpy())
            indices_list.append(batch_indices.cpu().numpy())

        results[split_name] = {
            "logits": np.concatenate(logits_list, axis=0),
            "labels": np.concatenate(targets, axis=0),
            "indices": np.concatenate(indices_list, axis=0),
        }
        print(
            f"  [{split_name}] 推理完成: {len(results[split_name]['labels'])} 个样本"
        )

    return results


def assign_balanced_labels_from_calibrated_logits(
    sample_df: pd.DataFrame,
    calibrated_logits: np.ndarray,
    normal_ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """基于校准器 logits 生成平衡伪标签。"""
    num_samples = len(sample_df)
    labels = np.zeros(num_samples, dtype=np.int64)
    probs = F.softmax(torch.from_numpy(calibrated_logits.astype(np.float32)), dim=1).numpy()
    confidence = np.max(probs, axis=1).astype(np.float32)

    for split_name in ["train", "val", "test"]:
        split_indices = sample_df.index[
            sample_df["dataset_split"] == split_name
        ].to_numpy()
        if len(split_indices) == 0:
            continue

        split_probs = probs[split_indices]
        split_conf = confidence[split_indices]
        normal_count = int(round(len(split_indices) * normal_ratio))
        normal_count = min(
            max(normal_count, 1),
            max(len(split_indices) - len(base.ABNORMAL_CLASS_NAMES), 1),
        )
        abnormal_count = len(split_indices) - normal_count

        target_counts = np.full(
            len(base.ABNORMAL_CLASS_NAMES),
            abnormal_count // len(base.ABNORMAL_CLASS_NAMES),
            dtype=np.int64,
        )
        target_counts[: abnormal_count % len(base.ABNORMAL_CLASS_NAMES)] += 1

        assigned_local = np.full(len(split_indices), -1, dtype=np.int64)

        # Normal: 选 class 0 概率最高的样本
        normal_scores = split_probs[:, 0]
        normal_local = np.argsort(-normal_scores)[:normal_count]
        assigned_local[normal_local] = 0

        # 异常类: 按校准器预测的最大异常概率分配
        abnormal_probs = split_probs[:, 1:]  # shape: [N, 9]
        abnormal_pred = np.argmax(abnormal_probs, axis=1)  # 0-indexed 异常类
        abnormal_confidence = np.max(abnormal_probs, axis=1)

        for class_offset in range(len(base.ABNORMAL_CLASS_NAMES)):
            need = int(target_counts[class_offset])
            if need <= 0:
                continue
            # 筛选校准器预测为该类的未分配样本
            candidates = np.where(
                (abnormal_pred == class_offset) & (assigned_local == -1)
            )[0]
            if len(candidates) > need:
                # 按置信度选 top-k
                cand_conf = abnormal_confidence[candidates]
                selected = candidates[np.argsort(-cand_conf)[:need]]
            elif len(candidates) > 0:
                selected = candidates
            else:
                # 该类无校准器预测，从未分配样本中补充
                unassigned = np.where(assigned_local == -1)[0]
                selected = unassigned[: min(need, len(unassigned))]
            assigned_local[selected] = class_offset + 1

        # 兜底: 未分配的样本用校准器 argmax
        leftovers = np.where(assigned_local == -1)[0]
        for local_idx in leftovers:
            global_idx = split_indices[local_idx]
            assigned_local[local_idx] = int(np.argmax(probs[global_idx]))

        labels[split_indices] = assigned_local

    return labels, confidence


def main() -> None:
    args = parse_args()
    base.set_seed(args.seed)

    source_output_dir = Path(args.source_output_dir)
    calibrator_dir = Path(args.calibrator_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. 加载 v16 缓存
    v16_summary = json.loads(
        (source_output_dir / "run_summary.json").read_text(encoding="utf-8")
    )
    cache_dir = source_output_dir / "cache"
    print(f"加载 v16 缓存: {cache_dir}")

    cached = base.CachedArrays(
        raw_sequences=np.load(cache_dir / "raw_sequences.npy", mmap_mode="r"),
        gadf_images=np.load(cache_dir / "gadf_images.npy", mmap_mode="r"),
        labels=np.load(cache_dir / "labels.npy", mmap_mode="r"),
        metadata=pd.read_csv(cache_dir / "sample_metadata.csv"),
    )
    print(
        f"  样本数: {len(cached.metadata)} "
        f"(train={len(cached.metadata[cached.metadata['dataset_split']=='train'])}, "
        f"val={len(cached.metadata[cached.metadata['dataset_split']=='val'])}, "
        f"test={len(cached.metadata[cached.metadata['dataset_split']=='test'])})"
    )

    # 2. 加载 v16 主干 + v33 校准器
    print("加载 v16 主干模型...")
    model = post.build_model(v16_summary, device=device)
    model.load_state_dict(
        torch.load(source_output_dir / "best_model.pt", map_location=device)
    )
    model.eval()

    calib_details = json.loads(
        (calibrator_dir / "calibrator_details.json").read_text(encoding="utf-8")
    )
    condition_mapping = calib_details.get("condition_mapping", {})
    print(f"加载 v33 校准器，condition_mapping: {len(condition_mapping)} 个 op")

    calibrator = ConditionAwareCalibrator(
        num_classes=len(base.CLASS_NAMES),
        num_conditions=len(condition_mapping),
        embedding_dim=calib_details.get("embedding_dim", 8),
        hidden_dim=calib_details.get("hidden_dim", 32),
        dropout=calib_details.get("dropout", 0.1),
    ).to(device)
    calibrator.load_state_dict(
        torch.load(calibrator_dir / "best_calibrator.pt", map_location=device)
    )
    calibrator.eval()

    # 3. TTA 推理全部 split
    tta_time_shifts = [0, -2, 2]
    tta_scale_shifts = [0]
    shift_pairs = [(sy, sx) for sy in tta_scale_shifts for sx in tta_time_shifts]
    print(f"TTA shift pairs: {shift_pairs}")

    print("开始 TTA 推理 + 校准...")
    inference_results = collect_calibrated_logits_for_all_splits(
        model=model,
        calibrator=calibrator,
        cached=cached,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        shift_pairs=shift_pairs,
        condition_mapping=condition_mapping,
    )

    # 4. 生成新伪标签
    all_logits = np.concatenate(
        [inference_results[s]["logits"] for s in ["train", "val", "test"]], axis=0
    )
    all_indices = np.concatenate(
        [inference_results[s]["indices"] for s in ["train", "val", "test"]], axis=0
    )
    sort_order = np.argsort(all_indices)
    logits_ordered = all_logits[sort_order]

    new_labels, new_confidence = assign_balanced_labels_from_calibrated_logits(
        sample_df=cached.metadata,
        calibrated_logits=logits_ordered,
        normal_ratio=float(v16_summary.get("normal_ratio", 0.30)),
    )

    # 5. 构建新 metadata 和 labels
    new_metadata = cached.metadata.copy()
    new_metadata["label"] = new_labels
    new_metadata["label_name"] = [base.CLASS_NAMES[label] for label in new_labels]
    new_metadata["sample_confidence"] = new_confidence
    confidence_power = float(v16_summary.get("confidence_power", 1.5))
    scaled_confidence = np.power(np.clip(new_confidence, 1e-4, 1.0), confidence_power)
    new_metadata["sample_weight"] = scaled_confidence

    # 计算标签分布
    print("\n新标签分布:")
    for split_name in ["train", "val", "test"]:
        mask = (new_metadata["dataset_split"] == split_name).to_numpy()
        split_labels = new_labels[mask]
        unique, counts = np.unique(split_labels, return_counts=True)
        print(f"  {split_name}:")
        for cls_idx, cnt in zip(unique, counts):
            print(f"    {base.CLASS_NAMES[cls_idx]}: {cnt}")

    # 6. 保存新缓存
    new_cache_dir = output_dir / "cache"
    new_cache_dir.mkdir(parents=True, exist_ok=True)
    new_metadata.to_csv(new_cache_dir / "sample_metadata.csv", index=False)
    np.save(new_cache_dir / "labels.npy", new_labels)

    # 复制 CWT 缓存文件
    for fname in ["raw_sequences.npy", "gadf_images.npy"]:
        src = cache_dir / fname
        dst = new_cache_dir / fname
        if not dst.exists():
            shutil.copy2(src, dst)

    # 重新加载带新标签的缓存
    new_cached = base.CachedArrays(
        raw_sequences=np.load(new_cache_dir / "raw_sequences.npy", mmap_mode="r"),
        gadf_images=np.load(new_cache_dir / "gadf_images.npy", mmap_mode="r"),
        labels=np.load(new_cache_dir / "labels.npy", mmap_mode="r"),
        metadata=pd.read_csv(new_cache_dir / "sample_metadata.csv"),
    )

    # 7. 保存标签分布图
    base.save_label_distribution(new_cached.metadata, output_dir)

    # 8. 构建 DataLoaders 并重训
    train_loader, val_loader, test_loader, split_indices = base.build_dataloaders(
        cached=new_cached,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=True,
    )

    # 重建模型（从头开始）
    image_channels = int(v16_summary.get("image_channels", len(base.FEATURE_COLUMNS)))
    print(f"\n从头初始化主干模型开始重训（架构: {args.model_name}, channels: {image_channels}）...")
    if args.model_name == "residual_attention_gru":
        new_model = base.ResidualAttentionGRUClassifier(
            image_channels=image_channels,
            image_size=int(v16_summary["image_size"]),
            num_classes=len(base.CLASS_NAMES),
            attention_stages=int(v16_summary.get("attention_stages", 3)),
            channel_attention_reduction=int(v16_summary.get("channel_attention_reduction", 8)),
        ).to(device)
    else:
        new_model = base.ResidualAttentionClassifier(
            image_channels=image_channels,
            image_size=int(v16_summary["image_size"]),
            num_classes=len(base.CLASS_NAMES),
            attention_stages=int(v16_summary.get("attention_stages", 3)),
            channel_attention_reduction=int(v16_summary.get("channel_attention_reduction", 8)),
        ).to(device)

    train_labels = new_cached.labels[split_indices["train"]]
    class_weights = base.compute_class_weights(
        train_labels, num_classes=len(base.CLASS_NAMES)
    ).to(device)

    criterion = base.FocalLoss(gamma=2.0, weight=class_weights)
    coarse_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        new_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = GradScaler(device.type, enabled=device.type == "cuda")

    best_val_f1 = -1.0
    history: List[Dict[str, float]] = []
    best_model_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = base.run_epoch(
            model=new_model,
            loader=train_loader,
            criterion=criterion,
            coarse_criterion=coarse_criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            train=True,
            coarse_loss_weight=args.coarse_loss_weight,
            prototype_loss_weight=0.0,
        )
        val_loss, val_acc = base.run_epoch(
            model=new_model,
            loader=val_loader,
            criterion=criterion,
            coarse_criterion=coarse_criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            train=False,
            coarse_loss_weight=args.coarse_loss_weight,
            prototype_loss_weight=0.0,
        )

        val_pred, val_true, _, _ = base.collect_predictions(
            new_model, val_loader, device
        )
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
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1_macro": val_f1,
            }
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(new_model.state_dict(), best_model_path)

    # 9. 评估
    base.save_training_curve(history, output_dir)
    new_model.load_state_dict(torch.load(best_model_path, map_location=device))

    _, _, val_embeddings, val_indices, val_logits = (
        base.collect_predictions_with_logits(new_model, val_loader, device)
    )
    y_pred, y_true, test_embeddings, test_indices, test_logits = (
        base.collect_predictions_with_logits(new_model, test_loader, device)
    )

    metrics = base.save_metrics(y_true, y_pred, output_dir)
    base.save_confusion_matrix(y_true, y_pred, output_dir)

    print("\n===== Self-training 结果 =====")
    print(f"Test Accuracy : {metrics['accuracy']:.4f}")
    print(f"Test Macro F1  : {metrics['f1_macro']:.4f}")
    print(f"Test Weighted F1: {metrics['f1_weighted']:.4f}")
    print()
    print(f"v16 基线:")
    print(f"  Accuracy: 0.4943 | Macro F1: 0.3204 | Weighted F1: 0.4133")
    print(f"\n提升:")
    print(f"  Accuracy Δ: {metrics['accuracy'] - 0.4943:+.4f}")
    print(f"  Macro F1 Δ: {metrics['f1_macro'] - 0.3204:+.4f}")

    # 保存 run_summary
    result_payload = {
        "model_name": args.model_name,
        "transform_type": "cwt",
        "self_train_source": str(source_output_dir),
        "self_train_calibrator": str(calibrator_dir),
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "window_size": v16_summary["window_size"],
        "gadf_points": v16_summary["gadf_points"],
        "cwt_wavelets": v16_summary["cwt_wavelets"],
        "cwt_gradient_mix": v16_summary.get("cwt_gradient_mix", 0.15),
        "image_size": v16_summary["image_size"],
        "attention_stages": v16_summary.get("attention_stages", 3),
        "channel_attention_reduction": v16_summary.get("channel_attention_reduction", 8),
        "normal_ratio": v16_summary.get("normal_ratio", 0.30),
        "metrics": metrics,
        "train_samples": int((new_cached.metadata["dataset_split"] == "train").sum()),
        "val_samples": int((new_cached.metadata["dataset_split"] == "val").sum()),
        "test_samples": int((new_cached.metadata["dataset_split"] == "test").sum()),
        "num_samples": len(new_cached.metadata),
        "class_names": base.CLASS_NAMES,
        "pseudo_label_source": "v33_calibrator_self_training",
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 保存预测 CSV
    prediction_df = new_cached.metadata.iloc[test_indices].copy().reset_index(drop=True)
    prediction_df["true_label"] = y_true
    prediction_df["true_label_name"] = [
        base.CLASS_NAMES[int(v)] for v in y_true
    ]
    prediction_df["pred_label"] = y_pred
    prediction_df["pred_label_name"] = [
        base.CLASS_NAMES[int(v)] for v in y_pred
    ]
    prediction_df.to_csv(output_dir / "test_predictions.csv", index=False)

    print("\nSelf-training 完成。")


if __name__ == "__main__":
    main()
