import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models as tv_models

from ..paths import DATA_DIR, DATASET_DIR, OUTPUTS_DIR


FEATURE_COLUMNS = [
    "OilMistTemperature",
    "Motor1_Current",
    "Motor1_Temperature",
    "Motor1_Torque",
    "Motor1_Speed",
    "Motor2_Current",
    "Motor2_Temperature",
    "Motor2_Torque",
    "Motor2_Speed",
]

THRESHOLD_FEATURES = [
    "Motor1_Current",
    "Motor1_Temperature",
    "Motor1_Torque",
    "Motor2_Current",
    "Motor2_Temperature",
    "Motor2_Torque",
]

CLASS_NAMES = [
    "Normal",
    "Motor1_Current_Over",
    "Motor1_Temperature_Over",
    "Motor1_Torque_Over",
    "Motor2_Current_Over",
    "Motor2_Temperature_Over",
    "Motor2_Torque_Over",
    "OilMistTemperature_High",
    "Motor1_Speed_Abnormal",
    "Motor2_Speed_Abnormal",
]

ABNORMAL_CLASS_NAMES = CLASS_NAMES[1:]
COARSE_CLASS_NAMES = [
    "Normal",
    "Motor1_Subsystem",
    "Motor2_Subsystem",
    "OilMist_Subsystem",
    "Speed_Subsystem",
]
COARSE_LABEL_MAP = np.array([0, 1, 1, 1, 2, 2, 2, 3, 4, 4], dtype=np.int64)
SCORE_FEATURE_NAMES = [
    "Motor1_Current",
    "Motor1_Temperature",
    "Motor1_Torque",
    "Motor2_Current",
    "Motor2_Temperature",
    "Motor2_Torque",
    "OilMistTemperature",
    "Motor1_Speed",
    "Motor2_Speed",
]


@dataclass
class CachedArrays:
    raw_sequences: np.ndarray
    gadf_images: np.ndarray
    labels: np.ndarray
    metadata: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 CWT 时频图 + Residual Attention / VGG / ResNet 进行十分类训练与评估"
    )
    parser.add_argument(
        "--dataset-csv",
        default=str(DATASET_DIR / "final_dataset_table.csv"),
        help="最终数据集表 CSV 路径",
    )
    parser.add_argument(
        "--window-index-csv",
        default=str(DATASET_DIR / "window_index" / "window_sample_index.csv"),
        help="滑窗样本索引 CSV 路径",
    )
    parser.add_argument(
        "--limit-csv",
        default=str(DATA_DIR / "ms_cvtlimit.csv"),
        help="阈值规则 CSV 路径",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUTS_DIR / "cwt_residual_attention"),
        help="训练结果输出目录",
    )
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=16, help="批大小")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 数量")
    parser.add_argument(
        "--window-size",
        type=int,
        default=1000,
        help="每个样本的时间序列长度，应与样本索引中的窗口长度一致",
    )
    parser.add_argument(
        "--cwt-scales",
        dest="gadf_points",
        type=int,
        default=64,
        help="CWT 使用的尺度数量，越大时频分辨率越高但计算更慢",
    )
    parser.add_argument(
        "--cwt-wavelets",
        type=str,
        default="morl,mexh,gaus4",
        help="用于多尺度 CWT 融合的小波列表，逗号分隔",
    )
    parser.add_argument(
        "--cwt-gradient-mix",
        type=float,
        default=0.15,
        help="将时间梯度能量混入时频图的比例，用于增强瞬态边缘",
    )
    parser.add_argument(
        "--cwt-add-diff",
        action="store_true",
        help="为每个传感器特征增加一阶差分 CWT 通道（通道数翻倍）",
    )
    parser.add_argument(
        "--model-name",
        choices=["residual_attention", "residual_attention_gru", "vgg11_bn", "resnet18"],
        default="residual_attention",
        help="时频图分类模型类型",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=112,
        help="送入 SwinTransformer 的图像尺寸",
    )
    parser.add_argument(
        "--attention-stages",
        type=int,
        default=3,
        help="Residual Attention 的注意力层级数，推荐 2 或 3",
    )
    parser.add_argument(
        "--channel-attention-reduction",
        type=int,
        default=8,
        help="通道注意力压缩倍率",
    )
    parser.add_argument(
        "--derived-quantile",
        type=float,
        default=0.90,
        help="OilMistTemperature 和 Speed 异常类使用的训练集分位数阈值",
    )
    parser.add_argument(
        "--normal-ratio",
        type=float,
        default=0.30,
        help="每个 split 中划为 Normal 的样本比例，其余样本在 9 个异常类之间平衡分配",
    )
    parser.add_argument(
        "--pseudo-label-mode",
        choices=["balanced_score", "balanced_margin"],
        default="balanced_score",
        help="伪标签分配策略：balanced_score 为纯分数平衡，balanced_margin 为优先选择类间间隔更大的样本",
    )
    parser.add_argument(
        "--margin-weight",
        type=float,
        default=0.75,
        help="balanced_margin 模式下加入类间独占性奖励的权重",
    )
    parser.add_argument(
        "--exclusive-ratio",
        type=float,
        default=0.50,
        help="balanced_margin 模式下先从高独占性候选中分配的比例",
    )
    parser.add_argument(
        "--loss-name",
        choices=["cross_entropy", "focal"],
        default="focal",
        help="分类损失函数，推荐 focal 以提升难分类异常类的权重",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal Loss 的 gamma 参数，越大越关注难样本",
    )
    parser.add_argument(
        "--disable-class-weights",
        action="store_true",
        help="关闭类别权重，默认启用以缓解 Normal 类占优问题",
    )
    parser.add_argument(
        "--disable-weighted-sampler",
        action="store_true",
        help="关闭训练集 WeightedRandomSampler，默认启用以增加少数类采样概率",
    )
    parser.add_argument(
        "--confidence-power",
        type=float,
        default=1.5,
        help="伪标签置信度权重指数，越大越强调高置信样本",
    )
    parser.add_argument(
        "--sample-weight-floor",
        type=float,
        default=0.0,
        help="样本权重下限，最终权重为 floor + (1-floor) * confidence^power",
    )
    parser.add_argument(
        "--coarse-loss-weight",
        type=float,
        default=0.0,
        help="层次化粗分类损失权重",
    )
    parser.add_argument(
        "--prototype-loss-weight",
        type=float,
        default=0.0,
        help="原型一致性损失权重",
    )
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=0,
        help="每个 split 最多保留多少样本，0 表示不限制",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_path(path_value: str) -> Path:
    path_value = str(path_value)
    windows_drive_match = re.match(r"^([A-Za-z]):[\\/](.*)$", path_value)
    # 仅在 WSL/Linux 环境下做 /mnt/ 转换，原生 Windows 保持原路径
    if windows_drive_match and Path("/proc/version").exists():
        drive_letter = windows_drive_match.group(1).lower()
        suffix = windows_drive_match.group(2).replace("\\", "/")
        return Path(f"/mnt/{drive_letter}/{suffix}")
    return Path(path_value)


def resample_series(series: np.ndarray, target_points: int) -> np.ndarray:
    if len(series) == target_points:
        return series.astype(np.float32)
    old_x = np.linspace(0.0, 1.0, len(series))
    new_x = np.linspace(0.0, 1.0, target_points)
    return np.interp(new_x, old_x, series.astype(np.float32)).astype(np.float32)


def scale_to_unit_interval(series: np.ndarray) -> np.ndarray:
    series = series.astype(np.float32)
    min_value = float(np.min(series))
    max_value = float(np.max(series))
    if math.isclose(min_value, max_value):
        return np.zeros_like(series, dtype=np.float32)
    return (series - min_value) / (max_value - min_value)


def scale_to_minus_one_plus_one(series: np.ndarray) -> np.ndarray:
    normalized = scale_to_unit_interval(series)
    return np.clip(normalized * 2.0 - 1.0, -1.0, 1.0)


def compute_feature_cwt(
    series_1d: np.ndarray,
    cwt_scales: int,
    cwt_wavelets: List[str],
    gradient_mix: float,
) -> np.ndarray:
    """对单特征时序计算 CWT 时频图。"""
    sampled = resample_series(series_1d, cwt_scales)
    sampled = sampled - np.mean(sampled)
    scale = np.std(sampled) + 1e-6
    normalized = sampled / scale
    scales = np.arange(1, cwt_scales + 1, dtype=np.float32)
    cwt_maps = []
    for wavelet_name in cwt_wavelets:
        coeffs, _ = pywt.cwt(normalized, scales, wavelet_name)
        magnitude = np.abs(coeffs).astype(np.float32)
        magnitude = scale_to_unit_interval(magnitude)
        cwt_maps.append(magnitude)
    fused_map = np.mean(np.stack(cwt_maps, axis=0), axis=0)
    gradient = np.gradient(normalized.astype(np.float32))
    gradient_energy = np.tile(np.abs(gradient)[None, :], (cwt_scales, 1))
    gradient_energy = scale_to_unit_interval(gradient_energy)
    fused_map = (1.0 - gradient_mix) * fused_map + gradient_mix * gradient_energy
    return fused_map.astype(np.float32)


def build_cwt_stack(
    sequence: np.ndarray,
    cwt_scales: int,
    cwt_wavelets: List[str],
    gradient_mix: float,
    add_diff: bool = False,
) -> np.ndarray:
    channels = []
    for feature_idx in range(sequence.shape[1]):
        series = sequence[:, feature_idx]
        channels.append(compute_feature_cwt(series, cwt_scales, cwt_wavelets, gradient_mix))
        if add_diff:
            diff = np.diff(series, prepend=series[0]).astype(np.float32)
            channels.append(compute_feature_cwt(diff, cwt_scales, cwt_wavelets, gradient_mix))
    return np.stack(channels, axis=0).astype(np.float32)


def load_threshold_table(limit_csv: Path) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    limit_df = pd.read_csv(limit_csv)
    limit_df["上限"] = pd.to_numeric(limit_df["上限"], errors="coerce")
    limit_df = limit_df.dropna(subset=["上限"])
    limit_df = limit_df[limit_df["bj"].isin(THRESHOLD_FEATURES)]

    per_op: Dict[str, Dict[str, float]] = {}
    for _, row in limit_df.iterrows():
        per_op.setdefault(str(row["gwh"]), {})[str(row["bj"])] = float(row["上限"])

    global_map = (
        limit_df.groupby("bj")["上限"]
        .median()
        .to_dict()
    )
    return per_op, {str(k): float(v) for k, v in global_map.items()}


def select_sample_index(sample_df: pd.DataFrame, max_samples_per_split: int, seed: int) -> pd.DataFrame:
    if max_samples_per_split <= 0:
        return sample_df.reset_index(drop=True)

    sampled_parts = []
    for split_name, split_df in sample_df.groupby("dataset_split", sort=False):
        if len(split_df) <= max_samples_per_split:
            sampled_parts.append(split_df)
        else:
            sampled_parts.append(split_df.sample(max_samples_per_split, random_state=seed))
    return pd.concat(sampled_parts, axis=0).sort_values(["dataset_split", "op", "start_row_in_group"]).reset_index(drop=True)


def robust_normalize_by_train_quantile(
    values: np.ndarray,
    sample_df: pd.DataFrame,
    quantile: float = 0.95,
) -> np.ndarray:
    value_array = np.clip(values.astype(np.float32), 0.0, None)
    train_mask = sample_df["dataset_split"].to_numpy() == "train"
    reference = value_array[train_mask] if np.any(train_mask) else value_array
    if reference.size == 0:
        return np.zeros_like(value_array, dtype=np.float32)

    scale = float(np.quantile(reference, quantile))
    scale = max(scale, 1e-6)
    return np.clip(value_array / scale, 0.0, 1.0).astype(np.float32)


def load_group_sequences(
    con: duckdb.DuckDBPyConnection,
    dataset_csv: Path,
    sample_df: pd.DataFrame,
) -> Dict[Tuple[str, str], np.ndarray]:
    group_cache: Dict[Tuple[str, str], np.ndarray] = {}
    selected_columns = ["op", "dataset_split", "event_time", *FEATURE_COLUMNS]
    select_sql = ", ".join(f'"{col}"' for col in selected_columns)

    for op, split_name in sample_df[["op", "dataset_split"]].drop_duplicates().itertuples(index=False):
        sql = f"""
            SELECT {select_sql}
            FROM read_csv_auto('{dataset_csv.as_posix()}', header = true, sample_size = 20000)
            WHERE op = ? AND dataset_split = ?
            ORDER BY event_time
        """
        df = con.execute(sql, [op, split_name]).df()
        array = df[FEATURE_COLUMNS].astype(np.float32).to_numpy()
        group_cache[(str(op), str(split_name))] = array
    return group_cache


def assign_balanced_pseudo_labels(
    sample_df: pd.DataFrame,
    score_matrix: np.ndarray,
    normal_ratio: float,
    pseudo_label_mode: str,
    margin_weight: float,
    exclusive_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_samples = len(sample_df)
    labels = np.zeros(num_samples, dtype=np.int64)
    dominant_idx = np.argmax(score_matrix, axis=1).astype(np.int64)
    dominant_score = np.max(score_matrix, axis=1).astype(np.float32)

    for split_name in ["train", "val", "test"]:
        split_indices = sample_df.index[sample_df["dataset_split"] == split_name].to_numpy()
        if len(split_indices) == 0:
            continue

        split_scores = score_matrix[split_indices]
        split_max_scores = dominant_score[split_indices]
        normal_count = int(round(len(split_indices) * normal_ratio))
        normal_count = min(max(normal_count, 1), max(len(split_indices) - len(ABNORMAL_CLASS_NAMES), 1))
        abnormal_count = len(split_indices) - normal_count

        target_counts = np.full(len(ABNORMAL_CLASS_NAMES), abnormal_count // len(ABNORMAL_CLASS_NAMES), dtype=np.int64)
        target_counts[: abnormal_count % len(ABNORMAL_CLASS_NAMES)] += 1

        assigned_local = np.full(len(split_indices), -1, dtype=np.int64)
        normal_local = np.argsort(split_max_scores)[:normal_count]
        assigned_local[normal_local] = 0

        if pseudo_label_mode == "balanced_score":
            utility_matrix = split_scores.copy()
        else:
            utility_matrix = np.zeros_like(split_scores, dtype=np.float32)
            for class_offset in range(len(ABNORMAL_CLASS_NAMES)):
                other_scores = np.delete(split_scores, class_offset, axis=1)
                max_other = other_scores.max(axis=1)
                margin_bonus = np.maximum(split_scores[:, class_offset] - max_other, 0.0)
                utility_matrix[:, class_offset] = split_scores[:, class_offset] + margin_weight * margin_bonus

            exclusive_targets = np.floor(target_counts * exclusive_ratio).astype(np.int64)
            remaining_slots = target_counts - exclusive_targets
            for class_offset in range(len(ABNORMAL_CLASS_NAMES)):
                need = int(exclusive_targets[class_offset])
                if need <= 0:
                    continue
                ranked_local = np.argsort(-utility_matrix[:, class_offset])
                picked = 0
                for local_idx in ranked_local:
                    if assigned_local[local_idx] != -1:
                        continue
                    if dominant_idx[split_indices[local_idx]] != class_offset:
                        continue
                    assigned_local[local_idx] = class_offset + 1
                    picked += 1
                    if picked >= need:
                        break
                remaining_slots[class_offset] += need - picked
            target_counts = remaining_slots

        for class_offset in range(len(ABNORMAL_CLASS_NAMES)):
            need = int(target_counts[class_offset])
            if need <= 0:
                continue
            ranked_local = np.argsort(-utility_matrix[:, class_offset])
            picked = 0
            for local_idx in ranked_local:
                if assigned_local[local_idx] != -1:
                    continue
                assigned_local[local_idx] = class_offset + 1
                picked += 1
                if picked >= need:
                    break

        leftovers = np.where(assigned_local == -1)[0]
        for local_idx in leftovers:
            assigned_local[local_idx] = int(np.argmax(utility_matrix[local_idx])) + 1

        labels[split_indices] = assigned_local

    return labels, dominant_idx, dominant_score


def build_window_cache(
    dataset_csv: Path,
    window_index_csv: Path,
    limit_csv: Path,
    output_dir: Path,
    window_size: int,
    gadf_points: int,
    cwt_wavelets: List[str],
    cwt_gradient_mix: float,
    cwt_add_diff: bool,
    derived_quantile: float,
    normal_ratio: float,
    pseudo_label_mode: str,
    margin_weight: float,
    exclusive_ratio: float,
    confidence_power: float,
    sample_weight_floor: float,
    max_samples_per_split: int,
    seed: int,
) -> CachedArrays:
    cache_dir = output_dir / "cache"
    raw_path = cache_dir / "raw_sequences.npy"
    gadf_path = cache_dir / "gadf_images.npy"
    label_path = cache_dir / "labels.npy"
    meta_path = cache_dir / "sample_metadata.csv"

    if raw_path.exists() and gadf_path.exists() and label_path.exists() and meta_path.exists():
        metadata = pd.read_csv(meta_path)
        return CachedArrays(
            raw_sequences=np.load(raw_path, mmap_mode="r"),
            gadf_images=np.load(gadf_path, mmap_mode="r"),
            labels=np.load(label_path, mmap_mode="r"),
            metadata=metadata,
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    sample_df = pd.read_csv(window_index_csv)
    sample_df = select_sample_index(sample_df, max_samples_per_split=max_samples_per_split, seed=seed)

    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    group_cache = load_group_sequences(con, dataset_csv, sample_df)
    con.close()

    num_samples = len(sample_df)
    num_features = len(FEATURE_COLUMNS)
    num_cwt_channels = num_features * (2 if cwt_add_diff else 1)
    raw_sequences = np.zeros((num_samples, window_size, num_features), dtype=np.float32)
    gadf_images = np.zeros((num_samples, num_cwt_channels, gadf_points, gadf_points), dtype=np.float16)

    stat_rows: List[Dict[str, float]] = []
    for idx, row in sample_df.iterrows():
        key = (str(row["op"]), str(row["dataset_split"]))
        start_idx = int(row["start_row_in_group"])
        end_idx = int(row["end_row_in_group"]) + 1
        sequence = group_cache[key][start_idx:end_idx]
        if len(sequence) != window_size:
            raise ValueError(f"样本 {row['sample_id']} 的窗口长度为 {len(sequence)}，与预期 {window_size} 不一致")

        raw_sequences[idx] = sequence
        gadf_images[idx] = build_cwt_stack(
            sequence,
            gadf_points,
            cwt_wavelets=cwt_wavelets,
            gradient_mix=cwt_gradient_mix,
            add_diff=cwt_add_diff,
        ).astype(np.float16)

        feature_max = np.max(sequence, axis=0)
        feature_abs_max = np.max(np.abs(sequence), axis=0)
        stat_rows.append(
            {
                "sample_id": row["sample_id"],
                "op": row["op"],
                "dataset_split": row["dataset_split"],
                "OilMistTemperature_max": float(feature_max[0]),
                "Motor1_Current_max": float(feature_max[1]),
                "Motor1_Temperature_max": float(feature_max[2]),
                "Motor1_Torque_max": float(feature_max[3]),
                "Motor1_Speed_abs_max": float(feature_abs_max[4]),
                "Motor2_Current_max": float(feature_max[5]),
                "Motor2_Temperature_max": float(feature_max[6]),
                "Motor2_Torque_max": float(feature_max[7]),
                "Motor2_Speed_abs_max": float(feature_abs_max[8]),
            }
        )

    stats_df = pd.DataFrame(stat_rows)
    per_op_thresholds, global_thresholds = load_threshold_table(limit_csv)

    derived_thresholds: Dict[str, Dict[str, float]] = {}
    for op, op_df in stats_df[stats_df["dataset_split"] == "train"].groupby("op"):
        derived_thresholds[str(op)] = {
            "OilMistTemperature": float(op_df["OilMistTemperature_max"].quantile(derived_quantile)),
            "Motor1_Speed": float(op_df["Motor1_Speed_abs_max"].quantile(derived_quantile)),
            "Motor2_Speed": float(op_df["Motor2_Speed_abs_max"].quantile(derived_quantile)),
        }

    global_derived = {
        "OilMistTemperature": float(stats_df.loc[stats_df["dataset_split"] == "train", "OilMistTemperature_max"].quantile(derived_quantile)),
        "Motor1_Speed": float(stats_df.loc[stats_df["dataset_split"] == "train", "Motor1_Speed_abs_max"].quantile(derived_quantile)),
        "Motor2_Speed": float(stats_df.loc[stats_df["dataset_split"] == "train", "Motor2_Speed_abs_max"].quantile(derived_quantile)),
    }

    score_matrix = np.zeros((num_samples, len(ABNORMAL_CLASS_NAMES)), dtype=np.float32)
    for idx, row in sample_df.iterrows():
        op = str(row["op"])
        op_threshold = per_op_thresholds.get(op, {})
        op_derived = derived_thresholds.get(op, global_derived)
        stat_row = stats_df.iloc[idx]

        score_matrix[idx] = np.array(
            [
                stat_row["Motor1_Current_max"] / max(op_threshold.get("Motor1_Current", global_thresholds["Motor1_Current"]), 1e-6),
                stat_row["Motor1_Temperature_max"] / max(op_threshold.get("Motor1_Temperature", global_thresholds["Motor1_Temperature"]), 1e-6),
                stat_row["Motor1_Torque_max"] / max(op_threshold.get("Motor1_Torque", global_thresholds["Motor1_Torque"]), 1e-6),
                stat_row["Motor2_Current_max"] / max(op_threshold.get("Motor2_Current", global_thresholds["Motor2_Current"]), 1e-6),
                stat_row["Motor2_Temperature_max"] / max(op_threshold.get("Motor2_Temperature", global_thresholds["Motor2_Temperature"]), 1e-6),
                stat_row["Motor2_Torque_max"] / max(op_threshold.get("Motor2_Torque", global_thresholds["Motor2_Torque"]), 1e-6),
                stat_row["OilMistTemperature_max"] / max(op_derived["OilMistTemperature"], 1e-6),
                stat_row["Motor1_Speed_abs_max"] / max(op_derived["Motor1_Speed"], 1e-6),
                stat_row["Motor2_Speed_abs_max"] / max(op_derived["Motor2_Speed"], 1e-6),
            ],
            dtype=np.float32,
        )

    labels, dominant_idx, dominant_score = assign_balanced_pseudo_labels(
        sample_df=sample_df,
        score_matrix=score_matrix,
        normal_ratio=normal_ratio,
        pseudo_label_mode=pseudo_label_mode,
        margin_weight=margin_weight,
        exclusive_ratio=exclusive_ratio,
    )

    metadata = sample_df.copy()
    metadata["label"] = labels
    metadata["label_name"] = [CLASS_NAMES[label] for label in labels]
    metadata["dominant_feature"] = [SCORE_FEATURE_NAMES[int(idx)] for idx in dominant_idx]
    metadata["dominant_label_name"] = [CLASS_NAMES[int(idx) + 1] for idx in dominant_idx]
    metadata["label_score"] = dominant_score
    for feature_idx, feature_name in enumerate(SCORE_FEATURE_NAMES):
        metadata[f"score_{feature_name}"] = score_matrix[:, feature_idx]
    # 使用训练集分位数做稳健归一化，避免个别极端窗口把绝大多数样本权重压成接近 0。
    confidence = robust_normalize_by_train_quantile(dominant_score.astype(np.float32), sample_df=sample_df, quantile=0.95)
    if pseudo_label_mode == "balanced_margin":
        sorted_scores = np.sort(score_matrix, axis=1)
        margin = np.maximum(sorted_scores[:, -1] - sorted_scores[:, -2], 0.0).astype(np.float32)
        margin = robust_normalize_by_train_quantile(margin, sample_df=sample_df, quantile=0.95)
        confidence = 0.7 * confidence + 0.3 * margin
    metadata["sample_confidence"] = confidence
    scaled_confidence = np.power(np.clip(confidence, 1e-4, 1.0), confidence_power)
    floor = float(np.clip(sample_weight_floor, 0.0, 1.0))
    metadata["sample_weight"] = floor + (1.0 - floor) * scaled_confidence
    metadata.to_csv(meta_path, index=False)

    np.save(raw_path, raw_sequences)
    np.save(gadf_path, gadf_images)
    np.save(label_path, labels)
    np.save(cache_dir / "anomaly_scores.npy", score_matrix)

    threshold_payload = {
        "per_op_limit_thresholds": per_op_thresholds,
        "global_limit_thresholds": global_thresholds,
        "per_op_derived_thresholds": derived_thresholds,
        "global_derived_thresholds": global_derived,
        "pseudo_label_strategy": pseudo_label_mode,
        "normal_ratio": normal_ratio,
        "margin_weight": margin_weight,
        "exclusive_ratio": exclusive_ratio,
        "score_feature_names": SCORE_FEATURE_NAMES,
        "class_names": CLASS_NAMES,
        "derived_quantile": derived_quantile,
    }
    (cache_dir / "pseudo_label_thresholds.json").write_text(
        json.dumps(threshold_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return CachedArrays(
        raw_sequences=raw_sequences,
        gadf_images=gadf_images,
        labels=labels,
        metadata=metadata,
    )


class WindowDataset(Dataset):
    def __init__(
        self,
        raw_sequences: np.ndarray,
        gadf_images: np.ndarray,
        labels: np.ndarray,
        metadata: pd.DataFrame,
        indices: np.ndarray,
    ) -> None:
        self.raw_sequences = raw_sequences
        self.gadf_images = gadf_images
        self.labels = labels
        self.metadata = metadata
        self.indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        real_idx = int(self.indices[idx])
        raw_seq = torch.from_numpy(np.array(self.raw_sequences[real_idx], dtype=np.float32, copy=True))
        gadf = torch.from_numpy(np.array(self.gadf_images[real_idx], dtype=np.float32, copy=True))
        label_value = int(self.labels[real_idx])
        label = torch.tensor(label_value, dtype=torch.long)
        coarse_label = torch.tensor(int(COARSE_LABEL_MAP[label_value]), dtype=torch.long)
        confidence = torch.tensor(float(self.metadata.iloc[real_idx]["sample_weight"]), dtype=torch.float32)
        return {
            "raw_seq": raw_seq,
            "gadf": gadf,
            "label": label,
            "coarse_label": coarse_label,
            "confidence": confidence,
            "index": torch.tensor(real_idx, dtype=torch.long),
        }


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(F.adaptive_avg_pool2d(x, 1))
        max_out = self.mlp(F.adaptive_max_pool2d(x, 1))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(concat))


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction=reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class GRUBranch(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, hidden = self.gru(x)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        forward_last = hidden[-2]
        backward_last = hidden[-1]
        fused = torch.cat([forward_last, backward_last], dim=1)
        return self.proj(fused)


class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + identity, inplace=True)


class AttentionModule2D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pre = ResidualBlock2D(channels, channels)
        self.trunk = nn.Sequential(
            ResidualBlock2D(channels, channels),
            ResidualBlock2D(channels, channels),
        )
        self.mask_branch = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock2D(channels, channels),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.post = ResidualBlock2D(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        trunk = self.trunk(x)
        mask = self.mask_branch(x)
        mask = F.interpolate(mask, size=trunk.shape[-2:], mode="bilinear", align_corners=False)
        mask = torch.sigmoid(mask)
        return self.post((1.0 + mask) * trunk)


class ChannelAttention2D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        reduced = max(channels // reduction, 16)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.attn(x)


class ResidualAttentionClassifier(nn.Module):
    def __init__(
        self,
        image_channels: int,
        image_size: int,
        num_classes: int,
        attention_stages: int = 3,
        channel_attention_reduction: int = 8,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.stem = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(ResidualBlock2D(64, 128, stride=2), AttentionModule2D(128))
        self.layer2 = nn.Sequential(ResidualBlock2D(128, 256, stride=2), AttentionModule2D(256))
        if attention_stages >= 3:
            self.layer3 = nn.Sequential(ResidualBlock2D(256, 256), AttentionModule2D(256), ResidualBlock2D(256, 256))
        else:
            self.layer3 = nn.Sequential(ResidualBlock2D(256, 256), ResidualBlock2D(256, 256))
        self.channel_attention = ChannelAttention2D(256, reduction=channel_attention_reduction)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(256, num_classes)
        self.coarse_classifier = nn.Linear(256, len(COARSE_CLASS_NAMES))

    def forward(self, raw_seq: torch.Tensor, gadf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del raw_seq
        if gadf.shape[-1] != self.image_size or gadf.shape[-2] != self.image_size:
            gadf = F.interpolate(gadf, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        x = self.stem(gadf)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.channel_attention(x)
        pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        coarse_logits = self.coarse_classifier(pooled)
        return logits, coarse_logits, pooled


class ResidualAttentionGRUClassifier(nn.Module):
    """CWT 时频图 CNN + 原始时序 GRU 多模态融合分类器。"""

    def __init__(
        self,
        image_channels: int,
        image_size: int,
        num_classes: int,
        attention_stages: int = 3,
        channel_attention_reduction: int = 8,
        gru_input_size: int = 9,
        gru_hidden_size: int = 128,
        gru_num_layers: int = 2,
        gru_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.image_size = image_size

        # CNN 分支（同 ResidualAttentionClassifier，不含 classifier）
        self.stem = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(ResidualBlock2D(64, 128, stride=2), AttentionModule2D(128))
        self.layer2 = nn.Sequential(ResidualBlock2D(128, 256, stride=2), AttentionModule2D(256))
        if attention_stages >= 3:
            self.layer3 = nn.Sequential(
                ResidualBlock2D(256, 256), AttentionModule2D(256), ResidualBlock2D(256, 256)
            )
        else:
            self.layer3 = nn.Sequential(ResidualBlock2D(256, 256), ResidualBlock2D(256, 256))
        self.channel_attention = ChannelAttention2D(256, reduction=channel_attention_reduction)
        self.cnn_dropout = nn.Dropout(0.3)

        # GRU 分支
        self.gru_branch = GRUBranch(
            input_size=gru_input_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            dropout=gru_dropout,
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(256 + 256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(256, num_classes)
        self.coarse_classifier = nn.Linear(256, len(COARSE_CLASS_NAMES))

    def forward(
        self, raw_seq: torch.Tensor, gadf: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # CNN 路径
        if gadf.shape[-1] != self.image_size or gadf.shape[-2] != self.image_size:
            gadf = F.interpolate(gadf, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        cnn_x = self.stem(gadf)
        cnn_x = self.layer1(cnn_x)
        cnn_x = self.layer2(cnn_x)
        cnn_x = self.layer3(cnn_x)
        cnn_x = self.channel_attention(cnn_x)
        cnn_feat = F.adaptive_avg_pool2d(cnn_x, 1).flatten(1)
        cnn_feat = self.cnn_dropout(cnn_feat)

        # GRU 路径
        gru_feat = self.gru_branch(raw_seq)

        # 融合
        fused = torch.cat([cnn_feat, gru_feat], dim=1)
        pooled = self.fusion(fused)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        coarse_logits = self.coarse_classifier(pooled)
        return logits, coarse_logits, pooled


class VGG11BNClassifier(nn.Module):
    def __init__(self, image_channels: int, num_classes: int) -> None:
        super().__init__()
        self.backbone = tv_models.vgg11_bn(weights=None)
        self.backbone.features[0] = nn.Conv2d(image_channels, 64, kernel_size=3, padding=1)
        self.backbone.classifier[6] = nn.Linear(4096, num_classes)
        self.coarse_classifier = nn.Linear(4096, len(COARSE_CLASS_NAMES))

    def forward(self, raw_seq: torch.Tensor, gadf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del raw_seq
        x = self.backbone.features(gadf)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        for layer in self.backbone.classifier[:-1]:
            x = layer(x)
        logits = self.backbone.classifier[-1](x)
        coarse_logits = self.coarse_classifier(x)
        return logits, coarse_logits, x


class ResNet18Classifier(nn.Module):
    def __init__(self, image_channels: int, num_classes: int) -> None:
        super().__init__()
        self.backbone = tv_models.resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Linear(512, num_classes)
        self.coarse_classifier = nn.Linear(512, len(COARSE_CLASS_NAMES))

    def forward(self, raw_seq: torch.Tensor, gadf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del raw_seq
        x = self.backbone.conv1(gadf)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.backbone.fc(x)
        coarse_logits = self.coarse_classifier(x)
        return logits, coarse_logits, x


def build_dataloaders(
    cached: CachedArrays,
    batch_size: int,
    num_workers: int,
    use_weighted_sampler: bool,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, np.ndarray]]:
    metadata = cached.metadata
    split_indices = {
        "train": metadata.index[metadata["dataset_split"] == "train"].to_numpy(),
        "val": metadata.index[metadata["dataset_split"] == "val"].to_numpy(),
        "test": metadata.index[metadata["dataset_split"] == "test"].to_numpy(),
    }

    loaders = {}
    train_labels = cached.labels[split_indices["train"]]
    train_sampler = None
    if use_weighted_sampler:
        train_class_weights = compute_class_weights(train_labels, num_classes=len(CLASS_NAMES))
        sample_weights = train_class_weights[train_labels].double()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    for split_name in ["train", "val", "test"]:
        dataset = WindowDataset(
            raw_sequences=cached.raw_sequences,
            gadf_images=cached.gadf_images,
            labels=cached.labels,
            metadata=cached.metadata,
            indices=split_indices[split_name],
        )
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train" and train_sampler is None),
            sampler=train_sampler if split_name == "train" else None,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
    return loaders["train"], loaders["val"], loaders["test"], split_indices


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        if weight is None:
            self.register_buffer("weight", None)
        else:
            self.register_buffer("weight", weight.clone().detach())

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        target_log_probs = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        focal_factor = torch.pow(1.0 - target_probs, self.gamma)
        per_sample_loss = -target_log_probs * focal_factor
        if self.weight is not None:
            per_sample_loss = per_sample_loss * self.weight[target]
        return per_sample_loss.mean()


def compute_classification_loss(
    criterion: nn.Module,
    logits: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if sample_weight is None:
        return criterion(logits, target)

    if isinstance(criterion, FocalLoss):
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        target_log_probs = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        focal_factor = torch.pow(1.0 - target_probs, criterion.gamma)
        per_sample_loss = -target_log_probs * focal_factor
        if criterion.weight is not None:
            per_sample_loss = per_sample_loss * criterion.weight[target]
        return (per_sample_loss * sample_weight).mean()

    per_sample_loss = F.cross_entropy(
        logits,
        target,
        weight=criterion.weight if hasattr(criterion, "weight") else None,
        reduction="none",
    )
    return (per_sample_loss * sample_weight).mean()


def prototype_loss(embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    unique_labels = torch.unique(labels)
    if len(unique_labels) <= 1:
        return embeddings.new_tensor(0.0)

    loss_terms = []
    for label in unique_labels:
        class_mask = labels == label
        class_embeddings = embeddings[class_mask]
        if len(class_embeddings) <= 1:
            continue
        class_center = class_embeddings.mean(dim=0, keepdim=True)
        loss_terms.append(torch.mean(torch.sum((class_embeddings - class_center) ** 2, dim=1)))
    if not loss_terms:
        return embeddings.new_tensor(0.0)
    return torch.stack(loss_terms).mean()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    coarse_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    train: bool,
    coarse_loss_weight: float,
    prototype_loss_weight: float,
) -> Tuple[float, float]:
    model.train(train)
    total_loss = 0.0
    predictions = []
    targets = []

    for batch in loader:
        raw_seq = batch["raw_seq"].to(device, non_blocking=True)
        gadf = batch["gadf"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        coarse_label = batch["coarse_label"].to(device, non_blocking=True)
        confidence = batch["confidence"].to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits, coarse_logits, fused_feature = model(raw_seq, gadf)
            fine_loss = compute_classification_loss(criterion, logits, label, sample_weight=confidence)
            coarse_loss = compute_classification_loss(
                coarse_criterion,
                coarse_logits,
                coarse_label,
                sample_weight=confidence,
            )
            proto_loss = prototype_loss(fused_feature, label)
            loss = fine_loss + coarse_loss_weight * coarse_loss + prototype_loss_weight * proto_loss

        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += float(loss.item()) * len(label)
        predictions.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())
        targets.extend(label.detach().cpu().numpy().tolist())

    mean_loss = total_loss / max(len(loader.dataset), 1)
    mean_acc = accuracy_score(targets, predictions)
    return mean_loss, mean_acc


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions = []
    targets = []
    embeddings = []
    indices = []

    for batch in loader:
        raw_seq = batch["raw_seq"].to(device, non_blocking=True)
        gadf = batch["gadf"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        logits, _, fused_feature = model(raw_seq, gadf)
        predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
        targets.append(label.cpu().numpy())
        embeddings.append(fused_feature.cpu().numpy())
        indices.append(batch["index"].cpu().numpy())

    return (
        np.concatenate(predictions, axis=0),
        np.concatenate(targets, axis=0),
        np.concatenate(embeddings, axis=0),
        np.concatenate(indices, axis=0),
    )


@torch.no_grad()
def collect_predictions_with_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions = []
    targets = []
    embeddings = []
    indices = []
    logits_list = []

    for batch in loader:
        raw_seq = batch["raw_seq"].to(device, non_blocking=True)
        gadf = batch["gadf"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        logits, _, fused_feature = model(raw_seq, gadf)
        predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
        targets.append(label.cpu().numpy())
        embeddings.append(fused_feature.cpu().numpy())
        indices.append(batch["index"].cpu().numpy())
        logits_list.append(logits.cpu().numpy())

    return (
        np.concatenate(predictions, axis=0),
        np.concatenate(targets, axis=0),
        np.concatenate(embeddings, axis=0),
        np.concatenate(indices, axis=0),
        np.concatenate(logits_list, axis=0),
    )


def save_training_curve(history: List[Dict[str, float]], output_dir: Path) -> None:
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["train_acc"], label="train_acc")
    axes[1].plot(history_df["epoch"], history_df["val_acc"], label="val_acc")
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=180)
    plt.close(fig)


def save_label_distribution(metadata: pd.DataFrame, output_dir: Path) -> None:
    count_df = (
        metadata.groupby(["dataset_split", "label_name"])
        .size()
        .reset_index(name="count")
    )
    count_df.to_csv(output_dir / "label_distribution.csv", index=False)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=count_df, x="label_name", y="count", hue="dataset_split", ax=ax)
    ax.set_title("Class Label Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_dir / "label_distribution.png", dpi=180)
    plt.close(fig)


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASS_NAMES)))
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_df.to_csv(output_dir / "confusion_matrix.csv", index=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("10-Class Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def stratified_subset_indices(labels: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    unique_labels = np.unique(labels)
    per_class = max(max_points // max(len(unique_labels), 1), 1)
    selected = []
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        take = min(per_class, len(label_indices))
        selected.extend(rng.choice(label_indices, size=take, replace=False).tolist())
    return np.array(sorted(selected), dtype=np.int64)


def save_tsne_plot(
    features: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str,
    seed: int,
) -> None:
    if len(features) > 2000:
        keep = stratified_subset_indices(labels, max_points=2000, seed=seed)
        features = features[keep]
        labels = labels[keep]

    scaled = StandardScaler().fit_transform(features)
    reduced = PCA(n_components=min(50, scaled.shape[1], len(scaled))).fit_transform(scaled)
    embedding = TSNE(
        n_components=2,
        random_state=seed,
        learning_rate="auto",
        init="pca",
        perplexity=min(30, max(5, len(reduced) // 20)),
    ).fit_transform(reduced)

    tsne_df = pd.DataFrame(
        {
            "x": embedding[:, 0],
            "y": embedding[:, 1],
            "label": labels,
            "label_name": [CLASS_NAMES[int(label)] for label in labels],
        }
    )
    tsne_df.to_csv(output_path.with_suffix(".csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(data=tsne_df, x="x", y="y", hue="label_name", palette="tab10", s=40, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
) -> Dict[str, float]:
    all_labels = np.arange(len(CLASS_NAMES))
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=all_labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=all_labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=all_labels, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, labels=all_labels, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, labels=all_labels, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=all_labels, average="weighted", zero_division=0)),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )
    pd.DataFrame(report).transpose().to_csv(output_dir / "classification_report.csv", index=True)
    return metrics


def save_split_logits_cache(
    output_dir: Path,
    split_name: str,
    logits: np.ndarray,
    labels: np.ndarray,
    embeddings: np.ndarray,
    indices: np.ndarray,
    metadata: pd.DataFrame,
) -> None:
    cache_dir = output_dir / "logits_cache" / split_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / f"{split_name}_logits.npy", logits.astype(np.float32))
    np.save(cache_dir / f"{split_name}_labels.npy", labels.astype(np.int64))
    np.save(cache_dir / f"{split_name}_embeddings.npy", embeddings.astype(np.float32))
    np.save(cache_dir / f"{split_name}_indices.npy", indices.astype(np.int64))
    metadata.iloc[indices].copy().reset_index(drop=True).to_csv(
        cache_dir / f"{split_name}_metadata.csv",
        index=False,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    cwt_wavelets = [item.strip() for item in args.cwt_wavelets.split(",") if item.strip()]
    if not cwt_wavelets:
        raise ValueError("至少需要提供一个 CWT wavelet。")

    dataset_csv = normalize_path(args.dataset_csv)
    window_index_csv = normalize_path(args.window_index_csv)
    limit_csv = normalize_path(args.limit_csv)
    output_dir = normalize_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cached = build_window_cache(
        dataset_csv=dataset_csv,
        window_index_csv=window_index_csv,
        limit_csv=limit_csv,
        output_dir=output_dir,
        window_size=args.window_size,
        gadf_points=args.gadf_points,
        cwt_wavelets=cwt_wavelets,
        cwt_gradient_mix=args.cwt_gradient_mix,
        cwt_add_diff=args.cwt_add_diff,
        derived_quantile=args.derived_quantile,
        normal_ratio=args.normal_ratio,
        pseudo_label_mode=args.pseudo_label_mode,
        margin_weight=args.margin_weight,
        exclusive_ratio=args.exclusive_ratio,
        confidence_power=args.confidence_power,
        sample_weight_floor=args.sample_weight_floor,
        max_samples_per_split=args.max_samples_per_split,
        seed=args.seed,
    )

    save_label_distribution(cached.metadata, output_dir)
    train_loader, val_loader, test_loader, split_indices = build_dataloaders(
        cached=cached,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=not args.disable_weighted_sampler,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_channels = len(FEATURE_COLUMNS) * (2 if args.cwt_add_diff else 1)
    if args.model_name == "residual_attention":
        model = ResidualAttentionClassifier(
            image_channels=image_channels,
            image_size=args.image_size,
            num_classes=len(CLASS_NAMES),
            attention_stages=args.attention_stages,
            channel_attention_reduction=args.channel_attention_reduction,
        ).to(device)
    elif args.model_name == "residual_attention_gru":
        model = ResidualAttentionGRUClassifier(
            image_channels=image_channels,
            image_size=args.image_size,
            num_classes=len(CLASS_NAMES),
            attention_stages=args.attention_stages,
            channel_attention_reduction=args.channel_attention_reduction,
        ).to(device)
    elif args.model_name == "vgg11_bn":
        model = VGG11BNClassifier(
            image_channels=image_channels,
            num_classes=len(CLASS_NAMES),
        ).to(device)
    else:
        model = ResNet18Classifier(
            image_channels=image_channels,
            num_classes=len(CLASS_NAMES),
        ).to(device)

    train_labels = cached.labels[split_indices["train"]]
    class_weights = None
    if not args.disable_class_weights:
        class_weights = compute_class_weights(train_labels, num_classes=len(CLASS_NAMES)).to(device)

    if args.loss_name == "focal":
        criterion = FocalLoss(gamma=args.focal_gamma, weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    coarse_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = GradScaler(device.type, enabled=device.type == "cuda")

    best_val_f1 = -1.0
    history: List[Dict[str, float]] = []
    best_model_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            coarse_criterion=coarse_criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            train=True,
            coarse_loss_weight=args.coarse_loss_weight,
            prototype_loss_weight=args.prototype_loss_weight,
        )
        val_loss, val_acc = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            coarse_criterion=coarse_criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            train=False,
            coarse_loss_weight=args.coarse_loss_weight,
            prototype_loss_weight=args.prototype_loss_weight,
        )

        val_pred, val_true, _, _ = collect_predictions(model, val_loader, device)
        val_f1 = float(
            f1_score(
                val_true,
                val_pred,
                labels=np.arange(len(CLASS_NAMES)),
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
            torch.save(model.state_dict(), best_model_path)

    save_training_curve(history, output_dir)

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    (
        val_pred,
        val_true,
        val_embeddings,
        val_indices,
        val_logits,
    ) = collect_predictions_with_logits(model, val_loader, device)
    (
        y_pred,
        y_true,
        test_embeddings,
        test_indices,
        test_logits,
    ) = collect_predictions_with_logits(model, test_loader, device)
    metrics = save_metrics(y_true, y_pred, output_dir)
    save_confusion_matrix(y_true, y_pred, output_dir)
    save_split_logits_cache(
        output_dir=output_dir,
        split_name="val",
        logits=val_logits,
        labels=val_true,
        embeddings=val_embeddings,
        indices=val_indices,
        metadata=cached.metadata,
    )
    save_split_logits_cache(
        output_dir=output_dir,
        split_name="test",
        logits=test_logits,
        labels=y_true,
        embeddings=test_embeddings,
        indices=test_indices,
        metadata=cached.metadata,
    )

    prediction_df = cached.metadata.iloc[test_indices].copy().reset_index(drop=True)
    prediction_df["true_label"] = y_true
    prediction_df["true_label_name"] = [CLASS_NAMES[int(v)] for v in y_true]
    prediction_df["pred_label"] = y_pred
    prediction_df["pred_label_name"] = [CLASS_NAMES[int(v)] for v in y_pred]
    prediction_df.to_csv(output_dir / "test_predictions.csv", index=False)

    test_raw_flatten = cached.raw_sequences[test_indices].reshape(len(test_indices), -1).astype(np.float32)
    save_tsne_plot(
        features=test_raw_flatten,
        labels=y_true,
        output_path=output_dir / "tsne_raw_data.png",
        title="t-SNE of Original Raw Window Data",
        seed=args.seed,
    )
    save_tsne_plot(
        features=test_embeddings.astype(np.float32),
        labels=y_true,
        output_path=output_dir / "tsne_model_features.png",
        title="t-SNE of Model Features After Training",
        seed=args.seed,
    )

    result_payload = {
        "model_name": args.model_name,
        "transform_type": "cwt",
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "window_size": args.window_size,
        "gadf_points": args.gadf_points,
        "cwt_wavelets": cwt_wavelets,
        "cwt_gradient_mix": args.cwt_gradient_mix,
        "cwt_add_diff": args.cwt_add_diff,
        "image_channels": image_channels,
        "image_size": args.image_size,
        "attention_stages": args.attention_stages,
        "channel_attention_reduction": args.channel_attention_reduction,
        "normal_ratio": args.normal_ratio,
        "pseudo_label_mode": args.pseudo_label_mode,
        "margin_weight": args.margin_weight,
        "exclusive_ratio": args.exclusive_ratio,
        "loss_name": args.loss_name,
        "focal_gamma": args.focal_gamma,
        "use_class_weights": not args.disable_class_weights,
        "use_weighted_sampler": not args.disable_weighted_sampler,
        "confidence_power": args.confidence_power,
        "sample_weight_floor": args.sample_weight_floor,
        "coarse_loss_weight": args.coarse_loss_weight,
        "prototype_loss_weight": args.prototype_loss_weight,
        "num_samples": int(len(cached.metadata)),
        "train_samples": int(len(split_indices["train"])),
        "val_samples": int(len(split_indices["val"])),
        "test_samples": int(len(split_indices["test"])),
        "metrics": metrics,
        "class_names": CLASS_NAMES,
        "logits_cache_dir": str(output_dir / "logits_cache"),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
