import argparse
import json
import math
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from scipy.signal import stft

from ..paths import DATASET_DIR


# 总图默认展示更多主轴相关特征，让同一窗口下的多变量变化能一起对比。
DEFAULT_FEATURES = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制原始曲线图和 5 种时频变换总图")
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
        "--output-dir",
        default=str(DATASET_DIR / "transform_overviews"),
        help="6 张总图的输出目录",
    )
    parser.add_argument(
        "--sample-id",
        default="",
        help="指定要绘图的 sample_id；为空时默认使用第一条 train 样本",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=1000,
        help="绘图前保留的最大序列长度；默认提升到 1000 点",
    )
    return parser.parse_args()


def load_sample_row(window_index_csv: Path, sample_id: str) -> pd.Series:
    sample_index = pd.read_csv(window_index_csv)
    if sample_id:
        matched = sample_index[sample_index["sample_id"] == sample_id]
        if matched.empty:
            raise ValueError(f"未找到 sample_id={sample_id}")
        return matched.iloc[0]

    train_rows = sample_index[sample_index["dataset_split"] == "train"]
    if not train_rows.empty:
        return train_rows.iloc[0]
    return sample_index.iloc[0]


def load_sample_window(
    con: duckdb.DuckDBPyConnection,
    dataset_csv: Path,
    sample_row: pd.Series,
    feature_columns: list[str],
) -> pd.DataFrame:
    selected_columns = ["id", "machine_model", "op", "event_time", "dataset_split", *feature_columns]
    select_sql = ", ".join(f'"{col}"' for col in selected_columns)
    sql = f"""
        SELECT {select_sql}
        FROM read_csv_auto('{dataset_csv.as_posix()}', header = true, sample_size = 20000)
        WHERE op = ? AND dataset_split = ?
        ORDER BY event_time
    """
    group_df = con.execute(sql, [sample_row["op"], sample_row["dataset_split"]]).df()
    start_idx = int(sample_row["start_row_in_group"])
    end_idx = int(sample_row["end_row_in_group"]) + 1
    window_df = group_df.iloc[start_idx:end_idx].copy()
    window_df["event_time"] = pd.to_datetime(window_df["event_time"])
    return window_df


def resample_series(series: np.ndarray, max_points: int) -> np.ndarray:
    if len(series) <= max_points:
        return series.astype(float)

    old_x = np.linspace(0, 1, len(series))
    new_x = np.linspace(0, 1, max_points)
    return np.interp(new_x, old_x, series.astype(float))


def normalize_series(series: np.ndarray) -> np.ndarray:
    series = series.astype(float)
    min_value = np.min(series)
    max_value = np.max(series)
    if np.isclose(max_value, min_value):
        return np.zeros_like(series, dtype=float)
    return (series - min_value) / (max_value - min_value)


def to_minus_one_plus_one(series: np.ndarray) -> np.ndarray:
    normalized = normalize_series(series)
    return np.clip(normalized * 2.0 - 1.0, -1.0, 1.0)


def transform_mtf(series: np.ndarray, n_bins: int = 8) -> np.ndarray:
    scaled = normalize_series(series)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    states = np.digitize(scaled, bin_edges[1:-1], right=False)

    transition = np.zeros((n_bins, n_bins), dtype=float)
    for i in range(len(states) - 1):
        transition[states[i], states[i + 1]] += 1.0

    row_sums = transition.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    transition = transition / row_sums

    mtf = transition[states][:, states]
    return mtf


def transform_rp(series: np.ndarray) -> np.ndarray:
    scaled = normalize_series(series)
    distance = np.abs(scaled[:, None] - scaled[None, :])
    threshold = np.quantile(distance, 0.2)
    return (distance <= threshold).astype(float)


def transform_gaf(series: np.ndarray) -> np.ndarray:
    scaled = to_minus_one_plus_one(series)
    phi = np.arccos(scaled)
    return np.cos(phi[:, None] + phi[None, :])


def transform_cwt(series: np.ndarray) -> np.ndarray:
    widths = np.arange(1, min(33, len(series) // 2 + 1))
    if len(widths) == 0:
        widths = np.arange(1, 2)
    coefficients, _ = pywt.cwt(series, widths, "morl")
    return np.abs(coefficients)


def transform_stft(series: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nperseg = min(32, len(series))
    if nperseg < 4:
        nperseg = len(series)
    noverlap = max(0, nperseg // 2)
    freqs, times, matrix = stft(series, nperseg=nperseg, noverlap=noverlap)
    return freqs, times, np.abs(matrix)


def make_axes(feature_count: int) -> tuple[plt.Figure, np.ndarray, int, int]:
    cols = 3
    rows = math.ceil(feature_count / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 3.6 * rows), squeeze=False)
    return fig, axes.flatten(), rows, cols


def hide_unused_axes(axes: np.ndarray, used_count: int) -> None:
    for idx in range(used_count, len(axes)):
        axes[idx].axis("off")


def plot_raw_overview(window_df: pd.DataFrame, features: list[str], output_path: Path, sample_row: pd.Series) -> None:
    fig, axes, _, _ = make_axes(len(features))
    x = np.arange(len(window_df))

    for idx, feature in enumerate(features):
        ax = axes[idx]
        ax.plot(x, window_df[feature].to_numpy(dtype=float), linewidth=1.2, color="tab:blue")
        ax.set_title(f"{sample_row['op']} | {feature}")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.25)

    hide_unused_axes(axes, len(features))
    fig.suptitle(
        f"Raw Signal Overview | {sample_row['sample_id']} | points={len(window_df)} | {sample_row['start_time']} -> {sample_row['end_time']}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_matrix_overview(
    matrices: dict[str, np.ndarray],
    output_path: Path,
    title: str,
    sample_row: pd.Series,
) -> None:
    fig, axes, _, _ = make_axes(len(matrices))

    for idx, (feature, matrix) in enumerate(matrices.items()):
        ax = axes[idx]
        image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis")
        ax.set_title(f"{sample_row['op']} | {feature}")
        ax.set_xlabel("Time Index")
        ax.set_ylabel("Time/Frequency Index")
        fig.colorbar(image, ax=ax, shrink=0.82)

    hide_unused_axes(axes, len(matrices))
    fig.suptitle(
        f"{title} | {sample_row['sample_id']} | points={int(sample_row['window_size'])} | {sample_row['start_time']} -> {sample_row['end_time']}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_stft_overview(
    matrices: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_path: Path,
    sample_row: pd.Series,
) -> None:
    fig, axes, _, _ = make_axes(len(matrices))

    for idx, (feature, (freqs, times, magnitude)) in enumerate(matrices.items()):
        ax = axes[idx]
        image = ax.pcolormesh(times, freqs, magnitude, shading="gouraud", cmap="viridis")
        ax.set_title(f"{sample_row['op']} | {feature}")
        ax.set_xlabel("Time Bin")
        ax.set_ylabel("Frequency")
        fig.colorbar(image, ax=ax, shrink=0.82)

    hide_unused_axes(axes, len(matrices))
    fig.suptitle(
        f"STFT Overview | {sample_row['sample_id']} | points={int(sample_row['window_size'])} | {sample_row['start_time']} -> {sample_row['end_time']}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_metadata(output_dir: Path, sample_row: pd.Series, features: list[str]) -> None:
    metadata = {
        "sample_id": sample_row["sample_id"],
        "machine_model": sample_row["machine_model"],
        "op": sample_row["op"],
        "dataset_split": sample_row["dataset_split"],
        "start_time": sample_row["start_time"],
        "end_time": sample_row["end_time"],
        "features": features,
        "outputs": {
            "raw": str(output_dir / "01_raw_signal_overview.png"),
            "mtf": str(output_dir / "02_mtf_overview.png"),
            "rp": str(output_dir / "03_rp_overview.png"),
            "gaf": str(output_dir / "04_gaf_overview.png"),
            "cwt": str(output_dir / "05_cwt_overview.png"),
            "stft": str(output_dir / "06_stft_overview.png"),
        },
    }
    (output_dir / "overview_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    dataset_csv = Path(args.dataset_csv)
    window_index_csv = Path(args.window_index_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_row = load_sample_row(window_index_csv, args.sample_id)
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    window_df = load_sample_window(con, dataset_csv, sample_row, DEFAULT_FEATURES)
    con.close()

    sampled_series = {}
    for feature in DEFAULT_FEATURES:
        sampled_series[feature] = resample_series(window_df[feature].to_numpy(dtype=float), args.max_points)

    plot_raw_overview(window_df, DEFAULT_FEATURES, output_dir / "01_raw_signal_overview.png", sample_row)

    mtf_maps = {feature: transform_mtf(series) for feature, series in sampled_series.items()}
    plot_matrix_overview(mtf_maps, output_dir / "02_mtf_overview.png", "MTF Overview", sample_row)

    rp_maps = {feature: transform_rp(series) for feature, series in sampled_series.items()}
    plot_matrix_overview(rp_maps, output_dir / "03_rp_overview.png", "RP Overview", sample_row)

    gaf_maps = {feature: transform_gaf(series) for feature, series in sampled_series.items()}
    plot_matrix_overview(gaf_maps, output_dir / "04_gaf_overview.png", "GAF Overview", sample_row)

    cwt_maps = {feature: transform_cwt(series) for feature, series in sampled_series.items()}
    plot_matrix_overview(cwt_maps, output_dir / "05_cwt_overview.png", "CWT Overview", sample_row)

    stft_maps = {feature: transform_stft(series) for feature, series in sampled_series.items()}
    plot_stft_overview(stft_maps, output_dir / "06_stft_overview.png", sample_row)

    write_metadata(output_dir, sample_row, DEFAULT_FEATURES)


if __name__ == "__main__":
    main()
