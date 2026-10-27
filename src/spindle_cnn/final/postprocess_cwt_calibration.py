import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from ..paths import DATA_DIR, DATASET_DIR, OUTPUTS_DIR
from . import train_cwt_residual_attention_compare as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对已有 CWT 分类模型做 TTA 与后处理校准")
    parser.add_argument(
        "--source-output-dir",
        default=str(OUTPUTS_DIR / "cwt_residual_attention_plus_gaus8"),
        help="已有训练结果目录，需包含 best_model.pt、run_summary.json 和 cache",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUTS_DIR / "cwt_residual_attention_plus_gaus8_tta_calibrated"),
        help="后处理结果输出目录",
    )
    parser.add_argument(
        "--tta-time-shifts",
        type=str,
        default="0,-2,2",
        help="测试时增强的时间轴平移像素，逗号分隔",
    )
    parser.add_argument(
        "--tta-scale-shifts",
        type=str,
        default="0",
        help="测试时增强的尺度轴平移像素，逗号分隔",
    )
    parser.add_argument(
        "--calibration-method",
        choices=["none", "temperature_scaling", "vector_scaling", "isotonic_ovr"],
        default="vector_scaling",
        help="后处理校准方式",
    )
    parser.add_argument(
        "--group-column",
        type=str,
        default="",
        help="按 metadata 中的列分组拟合校准器，例如 op；留空表示全局校准",
    )
    parser.add_argument(
        "--min-group-samples",
        type=int,
        default=40,
        help="分组校准时单组验证样本最小数量，小于该值回退到全局校准",
    )
    parser.add_argument(
        "--min-group-classes",
        type=int,
        default=3,
        help="分组校准时单组验证集中至少需要覆盖的类别数",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--local-bias-classes",
        type=str,
        default="",
        help="对若干弱类做局部偏置微调，传入类别索引列表，如 1,4,6；留空表示关闭",
    )
    parser.add_argument(
        "--local-bias-candidates",
        type=str,
        default="-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6,0.8",
        help="局部偏置搜索候选值，逗号分隔",
    )
    parser.add_argument(
        "--local-bias-max-val-acc-drop",
        type=float,
        default=0.01,
        help="局部偏置搜索时允许验证集准确率相对基线下降的最大幅度",
    )
    return parser.parse_args()


def parse_shift_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def shift_with_replicate(x: torch.Tensor, shift_y: int = 0, shift_x: int = 0) -> torch.Tensor:
    if shift_y == 0 and shift_x == 0:
        return x

    pad_left = max(shift_x, 0)
    pad_right = max(-shift_x, 0)
    pad_top = max(shift_y, 0)
    pad_bottom = max(-shift_y, 0)
    padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")

    start_y = pad_bottom
    end_y = start_y + x.shape[-2]
    start_x = pad_right
    end_x = start_x + x.shape[-1]
    return padded[:, :, start_y:end_y, start_x:end_x]


def build_model(summary: Dict[str, object], device: torch.device) -> torch.nn.Module:
    model_name = str(summary["model_name"])
    image_channels = int(summary.get("image_channels", len(base.FEATURE_COLUMNS)))
    if model_name == "residual_attention":
        model = base.ResidualAttentionClassifier(
            image_channels=image_channels,
            image_size=int(summary["image_size"]),
            num_classes=len(base.CLASS_NAMES),
            attention_stages=int(summary.get("attention_stages", 3)),
            channel_attention_reduction=int(summary.get("channel_attention_reduction", 8)),
        )
    elif model_name == "residual_attention_gru":
        model = base.ResidualAttentionGRUClassifier(
            image_channels=image_channels,
            image_size=int(summary["image_size"]),
            num_classes=len(base.CLASS_NAMES),
            attention_stages=int(summary.get("attention_stages", 3)),
            channel_attention_reduction=int(summary.get("channel_attention_reduction", 8)),
        )
    elif model_name == "vgg11_bn":
        model = base.VGG11BNClassifier(
            image_channels=image_channels,
            num_classes=len(base.CLASS_NAMES),
        )
    else:
        model = base.ResNet18Classifier(
            image_channels=image_channels,
            num_classes=len(base.CLASS_NAMES),
        )
    return model.to(device)


@torch.no_grad()
def collect_logits_with_tta(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    shift_pairs: Sequence[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits_list = []
    targets = []
    embeddings = []
    indices = []

    for batch in loader:
        raw_seq = batch["raw_seq"].to(device, non_blocking=True)
        gadf = batch["gadf"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        logits_sum = None
        feature_sum = None
        for shift_y, shift_x in shift_pairs:
            gadf_tta = shift_with_replicate(gadf, shift_y=shift_y, shift_x=shift_x)
            logits, _, fused_feature = model(raw_seq, gadf_tta)
            logits_sum = logits if logits_sum is None else logits_sum + logits
            feature_sum = fused_feature if feature_sum is None else feature_sum + fused_feature

        logits_mean = logits_sum / max(len(shift_pairs), 1)
        feature_mean = feature_sum / max(len(shift_pairs), 1)

        logits_list.append(logits_mean.cpu().numpy())
        targets.append(label.cpu().numpy())
        embeddings.append(feature_mean.cpu().numpy())
        indices.append(batch["index"].cpu().numpy())

    return (
        np.concatenate(logits_list, axis=0),
        np.concatenate(targets, axis=0),
        np.concatenate(embeddings, axis=0),
        np.concatenate(indices, axis=0),
    )


def fit_vector_scaler(val_logits: np.ndarray, val_true: np.ndarray, seed: int) -> LogisticRegression:
    calibrator = LogisticRegression(
        max_iter=4000,
        solver="lbfgs",
        random_state=seed,
    )
    calibrator.fit(val_logits, val_true)
    return calibrator


def logits_to_probs(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.clip(np.sum(exp_logits, axis=1, keepdims=True), 1e-12, None)


def fit_temperature(val_logits: np.ndarray, val_true: np.ndarray) -> float:
    candidate_temperatures = np.unique(
        np.concatenate(
            [
                np.linspace(0.6, 3.0, 25, dtype=np.float64),
                np.geomspace(0.5, 5.0, 30, dtype=np.float64),
            ]
        )
    )
    best_temperature = 1.0
    best_loss = float("inf")
    for temperature in candidate_temperatures:
        probs = logits_to_probs(val_logits / float(temperature))
        loss = log_loss(val_true, probs, labels=np.arange(len(base.CLASS_NAMES)))
        if loss < best_loss:
            best_loss = loss
            best_temperature = float(temperature)
    return best_temperature


def fit_isotonic_ovr(
    val_logits: np.ndarray,
    val_true: np.ndarray,
) -> Tuple[List[IsotonicRegression], np.ndarray]:
    val_probs = logits_to_probs(val_logits)
    models: List[IsotonicRegression] = []
    for class_idx in range(len(base.CLASS_NAMES)):
        targets = (val_true == class_idx).astype(np.float64)
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(val_probs[:, class_idx], targets)
        models.append(iso)
    return models, val_probs


def apply_isotonic_ovr(models: List[IsotonicRegression], logits: np.ndarray) -> np.ndarray:
    probs = logits_to_probs(logits)
    calibrated = np.zeros_like(probs)
    for class_idx, iso in enumerate(models):
        calibrated[:, class_idx] = iso.transform(probs[:, class_idx])
    calibrated = np.clip(calibrated, 1e-9, None)
    calibrated = calibrated / np.clip(np.sum(calibrated, axis=1, keepdims=True), 1e-12, None)
    return calibrated


def apply_calibration(
    method: str,
    val_logits: np.ndarray,
    val_true: np.ndarray,
    test_logits: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    payload: Dict[str, object] = {"method": method}
    if method == "none":
        return np.argmax(val_logits, axis=1), np.argmax(test_logits, axis=1), payload

    if method == "temperature_scaling":
        temperature = fit_temperature(val_logits, val_true)
        val_pred = np.argmax(val_logits / temperature, axis=1)
        test_pred = np.argmax(test_logits / temperature, axis=1)
        payload["temperature"] = temperature
        return val_pred.astype(np.int64), test_pred.astype(np.int64), payload

    if method == "isotonic_ovr":
        models, _ = fit_isotonic_ovr(val_logits, val_true)
        val_probs = apply_isotonic_ovr(models, val_logits)
        test_probs = apply_isotonic_ovr(models, test_logits)
        val_pred = np.argmax(val_probs, axis=1)
        test_pred = np.argmax(test_probs, axis=1)
        payload["num_models"] = len(models)
        return val_pred.astype(np.int64), test_pred.astype(np.int64), payload

    calibrator = fit_vector_scaler(val_logits, val_true, seed=seed)
    val_pred = calibrator.predict(val_logits)
    test_pred = calibrator.predict(test_logits)
    payload["classes"] = calibrator.classes_.tolist()
    payload["coef"] = calibrator.coef_.tolist()
    payload["intercept"] = calibrator.intercept_.tolist()
    return val_pred.astype(np.int64), test_pred.astype(np.int64), payload


def apply_grouped_calibration(
    method: str,
    val_logits: np.ndarray,
    val_true: np.ndarray,
    val_groups: np.ndarray,
    test_logits: np.ndarray,
    test_groups: np.ndarray,
    seed: int,
    min_group_samples: int,
    min_group_classes: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    global_val_pred, global_test_pred, global_payload = apply_calibration(
        method=method,
        val_logits=val_logits,
        val_true=val_true,
        test_logits=test_logits,
        seed=seed,
    )

    payload: Dict[str, object] = {
        "method": method,
        "group_column_mode": "grouped",
        "global_fallback": global_payload,
        "groups": {},
    }
    val_pred = global_val_pred.copy()
    test_pred = global_test_pred.copy()

    unique_groups = sorted({str(item) for item in np.concatenate([val_groups, test_groups], axis=0)})
    for group_name in unique_groups:
        val_mask = val_groups == group_name
        test_mask = test_groups == group_name
        group_val_count = int(np.sum(val_mask))
        group_test_count = int(np.sum(test_mask))
        if group_test_count == 0:
            continue

        group_info: Dict[str, object] = {
            "val_samples": group_val_count,
            "test_samples": group_test_count,
        }

        if group_val_count < min_group_samples:
            group_info["status"] = "fallback_insufficient_samples"
            payload["groups"][group_name] = group_info
            continue

        group_classes = np.unique(val_true[val_mask])
        group_info["val_classes"] = group_classes.astype(int).tolist()
        if len(group_classes) < min_group_classes:
            group_info["status"] = "fallback_insufficient_classes"
            payload["groups"][group_name] = group_info
            continue

        try:
            group_val_pred, group_test_pred, group_payload = apply_calibration(
                method=method,
                val_logits=val_logits[val_mask],
                val_true=val_true[val_mask],
                test_logits=test_logits[test_mask],
                seed=seed,
            )
            val_pred[val_mask] = group_val_pred
            test_pred[test_mask] = group_test_pred
            group_info["status"] = "group_calibrated"
            group_info["calibration"] = group_payload
        except Exception as exc:
            group_info["status"] = "fallback_error"
            group_info["error"] = str(exc)

        payload["groups"][group_name] = group_info

    return val_pred.astype(np.int64), test_pred.astype(np.int64), payload


def build_vector_scaling_scores(
    logits: np.ndarray,
    payload: Dict[str, object],
    num_classes: int,
) -> np.ndarray:
    classes = np.asarray(payload["classes"], dtype=np.int64)
    coef = np.asarray(payload["coef"], dtype=np.float64)
    intercept = np.asarray(payload["intercept"], dtype=np.float64)
    raw_scores = logits @ coef.T + intercept
    full_scores = np.full((len(logits), num_classes), -1e9, dtype=np.float64)
    for score_idx, class_idx in enumerate(classes):
        full_scores[:, int(class_idx)] = raw_scores[:, score_idx]
    return full_scores


def build_grouped_vector_scaling_scores(
    logits: np.ndarray,
    groups: np.ndarray,
    payload: Dict[str, object],
    num_classes: int,
) -> np.ndarray:
    scores = build_vector_scaling_scores(
        logits=logits,
        payload=payload["global_fallback"],
        num_classes=num_classes,
    )
    for group_name, group_info in payload["groups"].items():
        if group_info.get("status") != "group_calibrated":
            continue
        mask = groups == group_name
        if not np.any(mask):
            continue
        scores[mask] = build_vector_scaling_scores(
            logits=logits[mask],
            payload=group_info["calibration"],
            num_classes=num_classes,
        )
    return scores


def greedy_search_local_bias(
    val_scores: np.ndarray,
    val_true: np.ndarray,
    test_scores: np.ndarray,
    target_classes: List[int],
    bias_candidates: List[float],
    max_val_acc_drop: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    bias_vector = np.zeros(val_scores.shape[1], dtype=np.float64)

    def evaluate(scores: np.ndarray, bias: np.ndarray) -> Tuple[np.ndarray, float, float]:
        pred = np.argmax(scores + bias[None, :], axis=1).astype(np.int64)
        acc = float(accuracy_score(val_true, pred))
        macro_f1 = float(
            f1_score(
                val_true,
                pred,
                labels=np.arange(val_scores.shape[1]),
                average="macro",
                zero_division=0,
            )
        )
        return pred, acc, macro_f1

    _, base_acc, base_f1 = evaluate(val_scores, bias_vector)
    search_payload: Dict[str, object] = {
        "base_val_accuracy": base_acc,
        "base_val_f1_macro": base_f1,
        "selected_biases": {},
    }

    for class_idx in target_classes:
        best_bias = float(bias_vector[class_idx])
        _, best_acc, best_f1 = evaluate(val_scores, bias_vector)
        for candidate in bias_candidates:
            trial_bias = bias_vector.copy()
            trial_bias[class_idx] = float(candidate)
            _, trial_acc, trial_f1 = evaluate(val_scores, trial_bias)
            if trial_acc < base_acc - max_val_acc_drop:
                continue
            if (trial_f1 > best_f1 + 1e-12) or (
                abs(trial_f1 - best_f1) <= 1e-12 and trial_acc > best_acc + 1e-12
            ):
                best_bias = float(candidate)
                best_acc = trial_acc
                best_f1 = trial_f1
        bias_vector[class_idx] = best_bias
        search_payload["selected_biases"][str(class_idx)] = {
            "class_name": base.CLASS_NAMES[class_idx],
            "bias": best_bias,
            "val_accuracy": best_acc,
            "val_f1_macro": best_f1,
        }

    val_pred = np.argmax(val_scores + bias_vector[None, :], axis=1).astype(np.int64)
    test_pred = np.argmax(test_scores + bias_vector[None, :], axis=1).astype(np.int64)
    search_payload["final_bias_vector"] = bias_vector.tolist()
    search_payload["final_val_accuracy"] = float(accuracy_score(val_true, val_pred))
    search_payload["final_val_f1_macro"] = float(
        f1_score(
            val_true,
            val_pred,
            labels=np.arange(val_scores.shape[1]),
            average="macro",
            zero_division=0,
        )
    )
    return val_pred, test_pred, search_payload


def save_prediction_csv(
    metadata: pd.DataFrame,
    indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    prediction_df = metadata.iloc[indices].copy().reset_index(drop=True)
    prediction_df["true_label"] = y_true
    prediction_df["true_label_name"] = [base.CLASS_NAMES[int(v)] for v in y_true]
    prediction_df["pred_label"] = y_pred
    prediction_df["pred_label_name"] = [base.CLASS_NAMES[int(v)] for v in y_pred]
    prediction_df.to_csv(output_path, index=False)


def copy_static_artifacts(source_output_dir: Path, output_dir: Path) -> None:
    for file_name in [
        "training_curves.png",
        "training_history.csv",
        "label_distribution.png",
        "label_distribution.csv",
        "best_model.pt",
    ]:
        src = source_output_dir / file_name
        if src.exists():
            shutil.copy2(src, output_dir / file_name)


def main() -> None:
    args = parse_args()
    base.set_seed(args.seed)

    source_output_dir = base.normalize_path(args.source_output_dir)
    output_dir = base.normalize_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((source_output_dir / "run_summary.json").read_text(encoding="utf-8"))
    copy_static_artifacts(source_output_dir, output_dir)

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
        derived_quantile=0.90,
        normal_ratio=float(summary.get("normal_ratio", 0.30)),
        pseudo_label_mode=str(summary.get("pseudo_label_mode", "balanced_score")),
        margin_weight=float(summary.get("margin_weight", 0.75)),
        exclusive_ratio=float(summary.get("exclusive_ratio", 0.5)),
        confidence_power=float(summary.get("confidence_power", 1.5)),
        sample_weight_floor=float(summary.get("sample_weight_floor", 0.0)),
        max_samples_per_split=0,
        seed=args.seed,
    )

    _, val_loader, test_loader, split_indices = base.build_dataloaders(
        cached=cached,
        batch_size=int(summary["batch_size"]),
        num_workers=4,
        use_weighted_sampler=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(summary, device=device)
    model.load_state_dict(torch.load(source_output_dir / "best_model.pt", map_location=device))

    time_shifts = parse_shift_list(args.tta_time_shifts)
    scale_shifts = parse_shift_list(args.tta_scale_shifts)
    shift_pairs = [(sy, sx) for sy in scale_shifts for sx in time_shifts]

    val_logits, val_true, _, val_indices = collect_logits_with_tta(model, val_loader, device, shift_pairs)
    test_logits, test_true, test_embeddings, test_indices = collect_logits_with_tta(model, test_loader, device, shift_pairs)

    group_column = args.group_column.strip()
    if group_column:
        if group_column not in cached.metadata.columns:
            raise KeyError(f"group_column '{group_column}' 不存在于 metadata 中")
        val_groups = cached.metadata.iloc[val_indices][group_column].astype(str).to_numpy()
        test_groups = cached.metadata.iloc[test_indices][group_column].astype(str).to_numpy()
        val_pred, test_pred, calibration_payload = apply_grouped_calibration(
            method=args.calibration_method,
            val_logits=val_logits,
            val_true=val_true,
            val_groups=val_groups,
            test_logits=test_logits,
            test_groups=test_groups,
            seed=args.seed,
            min_group_samples=args.min_group_samples,
            min_group_classes=args.min_group_classes,
        )
    else:
        val_groups = np.array([], dtype=str)
        test_groups = np.array([], dtype=str)
        val_pred, test_pred, calibration_payload = apply_calibration(
            method=args.calibration_method,
            val_logits=val_logits,
            val_true=val_true,
            test_logits=test_logits,
            seed=args.seed,
        )

    local_bias_classes = parse_int_list(args.local_bias_classes)
    if local_bias_classes:
        if args.calibration_method != "vector_scaling":
            raise ValueError("local_bias_classes 仅支持在 vector_scaling 后处理下使用")
        if group_column:
            val_scores = build_grouped_vector_scaling_scores(
                logits=val_logits,
                groups=val_groups,
                payload=calibration_payload,
                num_classes=len(base.CLASS_NAMES),
            )
            test_scores = build_grouped_vector_scaling_scores(
                logits=test_logits,
                groups=test_groups,
                payload=calibration_payload,
                num_classes=len(base.CLASS_NAMES),
            )
        else:
            val_scores = build_vector_scaling_scores(
                logits=val_logits,
                payload=calibration_payload,
                num_classes=len(base.CLASS_NAMES),
            )
            test_scores = build_vector_scaling_scores(
                logits=test_logits,
                payload=calibration_payload,
                num_classes=len(base.CLASS_NAMES),
            )

        val_pred, test_pred, local_bias_payload = greedy_search_local_bias(
            val_scores=val_scores,
            val_true=val_true,
            test_scores=test_scores,
            target_classes=local_bias_classes,
            bias_candidates=parse_float_list(args.local_bias_candidates),
            max_val_acc_drop=float(args.local_bias_max_val_acc_drop),
        )
        calibration_payload["local_bias_search"] = local_bias_payload

    metrics = base.save_metrics(test_true, test_pred, output_dir)
    base.save_confusion_matrix(test_true, test_pred, output_dir)
    save_prediction_csv(
        metadata=cached.metadata,
        indices=test_indices,
        y_true=test_true,
        y_pred=test_pred,
        output_path=output_dir / "test_predictions.csv",
    )

    test_raw_flatten = cached.raw_sequences[test_indices].reshape(len(test_indices), -1).astype(np.float32)
    base.save_tsne_plot(
        features=test_raw_flatten,
        labels=test_true,
        output_path=output_dir / "tsne_raw_data.png",
        title="t-SNE of Original Raw Window Data",
        seed=args.seed,
    )
    base.save_tsne_plot(
        features=test_embeddings.astype(np.float32),
        labels=test_true,
        output_path=output_dir / "tsne_model_features.png",
        title="t-SNE of Model Features After TTA/Calibration",
        seed=args.seed,
    )

    val_metrics = {
        "accuracy": float(accuracy_score(val_true, val_pred)),
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
    (output_dir / "postprocess_details.json").write_text(
        json.dumps(
            {
                "source_output_dir": str(source_output_dir),
                "tta_shift_pairs": shift_pairs,
                "group_column": group_column,
                "calibration": calibration_payload,
                "val_metrics_after_postprocess": val_metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    result_payload = dict(summary)
    result_payload["postprocess_source_dir"] = str(source_output_dir)
    result_payload["postprocess_tta_shift_pairs"] = shift_pairs
    result_payload["postprocess_calibration_method"] = args.calibration_method
    result_payload["postprocess_group_column"] = group_column
    result_payload["postprocess_val_metrics"] = val_metrics
    result_payload["metrics"] = metrics
    (output_dir / "run_summary.json").write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
