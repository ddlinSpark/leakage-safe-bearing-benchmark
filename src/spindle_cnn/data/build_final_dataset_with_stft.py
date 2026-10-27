import argparse
import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import stft

from ..paths import DATA_DIR, DATASET_DIR


# 第一版数据集先聚焦主轴相关字段，避免特征过多导致预处理和试验成本过高。
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

# 先用 4 个核心特征做 STFT 联合展示，一张图中放多个子图，方便肉眼核查。
PREVIEW_FEATURES = [
    "Motor1_Current",
    "Motor1_Temperature",
    "Motor1_Torque",
    "Motor1_Speed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于 data_1_clean.csv 重新制作最终数据集，并生成 STFT 总览图")
    parser.add_argument(
        "--input-csv",
        default=str(DATA_DIR / "data_1_clean.csv"),
        help="清洗后的宽表 CSV 路径",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DATASET_DIR),
        help="新的最终数据集输出目录",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=256,
        help="STFT 预览窗口长度",
    )
    parser.add_argument(
        "--preview-windows",
        type=int,
        default=2,
        help="总览图中抽取多少个时间窗口，每个窗口会展示多个特征子图",
    )
    return parser.parse_args()


def load_station_list(con: duckdb.DuckDBPyConnection, input_csv: Path) -> list[str]:
    sql = f"""
        SELECT DISTINCT op
        FROM read_csv_auto('{input_csv.as_posix()}', header = true, sample_size = 20000)
        ORDER BY op
    """
    return [row[0] for row in con.execute(sql).fetchall()]


def load_station_frame(
    con: duckdb.DuckDBPyConnection,
    input_csv: Path,
    station: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    selected_columns = ["id", "machine_model", "op", "event_time", *feature_columns]
    select_sql = ", ".join(f'"{col}"' for col in selected_columns)
    sql = f"""
        SELECT {select_sql}
        FROM read_csv_auto('{input_csv.as_posix()}', header = true, sample_size = 20000)
        WHERE op = ?
        ORDER BY event_time
    """
    return con.execute(sql, [station]).df()


def clean_station_frame(df: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, dict]:
    cleaned = df.copy()
    cleaned["event_time"] = pd.to_datetime(cleaned["event_time"])

    for col in feature_columns:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

        # 温度列中的 -200 是明显的传感器缺失占位值，先转成 NaN 再插值。
        if "Temperature" in col:
            cleaned.loc[cleaned[col] <= -100, col] = np.nan

    missing_before = cleaned[feature_columns].isna().sum().to_dict()

    # 先做线性插值，尽量保留时间趋势；首尾缺失再用前后值补齐。
    cleaned[feature_columns] = cleaned[feature_columns].interpolate(
        method="linear",
        limit_direction="both",
    )
    cleaned[feature_columns] = cleaned[feature_columns].bfill().ffill()

    # 若整列都缺失，插值后仍会为空，这里回退到列中位数，再兜底填 0。
    medians = cleaned[feature_columns].median(numeric_only=True)
    cleaned[feature_columns] = cleaned[feature_columns].fillna(medians)
    cleaned[feature_columns] = cleaned[feature_columns].fillna(0.0)

    missing_after = cleaned[feature_columns].isna().sum().to_dict()
    return cleaned, {"missing_before": missing_before, "missing_after": missing_after}


def add_split_column(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    total_rows = len(result)
    train_end = int(total_rows * 0.7)
    val_end = int(total_rows * 0.85)

    # 按时间顺序切分，避免未来信息泄漏到训练集。
    result["dataset_split"] = "test"
    result.iloc[:train_end, result.columns.get_loc("dataset_split")] = "train"
    result.iloc[train_end:val_end, result.columns.get_loc("dataset_split")] = "val"
    return result


def write_dataset_table(
    con: duckdb.DuckDBPyConnection,
    input_csv: Path,
    output_dir: Path,
    feature_columns: list[str],
    preview_window_limit: int,
    window_size: int,
) -> tuple[Path, Path, list[dict], list[pd.DataFrame]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_csv = output_dir / "final_dataset_table.csv"
    summary_csv = output_dir / "station_missing_summary.csv"

    station_list = load_station_list(con, input_csv)
    summary_rows: list[dict] = []
    preview_windows: list[pd.DataFrame] = []

    first_write = True
    for station in station_list:
        station_df = load_station_frame(con, input_csv, station, feature_columns)
        station_df, missing_stats = clean_station_frame(station_df, feature_columns)
        station_df = add_split_column(station_df)

        # 收集少量窗口用于 STFT 总览图，不单独输出一张一张的小图。
        train_df = station_df[station_df["dataset_split"] == "train"].reset_index(drop=True)
        if len(preview_windows) < preview_window_limit and len(train_df) >= window_size:
            preview_windows.append(train_df.iloc[:window_size].copy())

        summary_rows.append(
            {
                "op": station,
                "machine_model": station_df["machine_model"].iloc[0],
                "row_count": len(station_df),
                **{f"{col}_missing_before": int(missing_stats["missing_before"][col]) for col in feature_columns},
                **{f"{col}_missing_after": int(missing_stats["missing_after"][col]) for col in feature_columns},
            }
        )

        station_df.to_csv(final_csv, mode="w" if first_write else "a", index=False, header=first_write)
        first_write = False

    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    return final_csv, summary_csv, summary_rows, preview_windows


def load_dataset_table(dataset_csv: Path, nrows: int = 5) -> pd.DataFrame:
    return pd.read_csv(dataset_csv, parse_dates=["event_time"], nrows=nrows)


def plot_stft_overview(
    windows: list[pd.DataFrame],
    output_path: Path,
    window_size: int,
    preview_features: list[str],
) -> None:
    if not windows:
        return

    rows = len(windows)
    cols = len(preview_features)
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.8 * rows), squeeze=False)

    for row_idx, window_df in enumerate(windows):
        station_name = window_df["op"].iloc[0]
        start_time = pd.to_datetime(window_df["event_time"].iloc[0]).strftime("%Y-%m-%d %H:%M:%S")
        end_time = pd.to_datetime(window_df["event_time"].iloc[-1]).strftime("%Y-%m-%d %H:%M:%S")

        for col_idx, feature in enumerate(preview_features):
            ax = axes[row_idx][col_idx]
            series = window_df[feature].to_numpy(dtype=float)

            nperseg = min(64, len(series))
            noverlap = max(0, nperseg // 2)
            _, freqs, stft_matrix = None, None, None
            freqs, times, stft_matrix = stft(series, nperseg=nperseg, noverlap=noverlap)
            magnitude = np.abs(stft_matrix)

            image = ax.pcolormesh(times, freqs, magnitude, shading="gouraud", cmap="viridis")
            ax.set_title(f"{station_name} | {feature}\n{start_time} -> {end_time}", fontsize=10)
            ax.set_xlabel("Time Bin")
            ax.set_ylabel("Frequency")
            fig.colorbar(image, ax=ax, shrink=0.78)

    fig.suptitle(
        f"STFT Total Overview ({rows} windows x {cols} features, window_size={window_size})",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_metadata(
    output_dir: Path,
    input_csv: Path,
    dataset_csv: Path,
    summary_csv: Path,
    preview_png: Path,
    feature_columns: list[str],
    preview_features: list[str],
    preview_window_count: int,
) -> None:
    preview_sample = pd.read_csv(dataset_csv, nrows=1)
    metadata = {
        "input_csv": str(input_csv),
        "dataset_csv": str(dataset_csv),
        "summary_csv": str(summary_csv),
        "preview_png": str(preview_png),
        "dataset_description": "基于 data_1_clean.csv 重建的最终数据集表，已完成空值处理与时间顺序切分。",
        "feature_columns": feature_columns,
        "preview_features": preview_features,
        "preview_window_count": preview_window_count,
        "missing_value_strategy": [
            "温度列中 <= -100 的值视为无效缺失",
            "按工位、按时间顺序进行线性插值",
            "首尾缺失使用前向/后向填充",
            "仍为空的列使用中位数，再兜底填 0",
        ],
        "split_strategy": "每个工位按时间顺序切分为 train 70% / val 15% / test 15%",
        "output_columns": preview_sample.columns.tolist(),
    }
    (output_dir / "dataset_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("PRAGMA threads=4")

    final_csv, summary_csv, _, preview_windows = write_dataset_table(
        con=con,
        input_csv=input_csv,
        output_dir=output_dir,
        feature_columns=DEFAULT_FEATURES,
        preview_window_limit=args.preview_windows,
        window_size=args.window_size,
    )

    # 这里显式加载一次结果表，确认新数据集可以被下游训练脚本直接读取。
    _ = load_dataset_table(final_csv, nrows=5)

    preview_png = output_dir / "stft_overview.png"
    plot_stft_overview(
        windows=preview_windows[: args.preview_windows],
        output_path=preview_png,
        window_size=args.window_size,
        preview_features=PREVIEW_FEATURES,
    )

    write_metadata(
        output_dir=output_dir,
        input_csv=input_csv,
        dataset_csv=final_csv,
        summary_csv=summary_csv,
        preview_png=preview_png,
        feature_columns=DEFAULT_FEATURES,
        preview_features=PREVIEW_FEATURES,
        preview_window_count=min(len(preview_windows), args.preview_windows),
    )
    con.close()


if __name__ == "__main__":
    main()
