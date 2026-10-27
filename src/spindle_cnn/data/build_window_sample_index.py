import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from ..paths import DATASET_DIR


# 与最终数据集表保持一致，后续训练和时频变换都默认使用这些字段。
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
    parser = argparse.ArgumentParser(description="基于 final_dataset_table.csv 生成滑窗样本索引")
    parser.add_argument(
        "--input-csv",
        default=str(DATASET_DIR / "final_dataset_table.csv"),
        help="最终数据集表 CSV 路径",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DATASET_DIR / "window_index"),
        help="滑窗样本索引输出目录",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=1000,
        help="每个样本窗口包含的时间点数量",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=250,
        help="滑窗步长",
    )
    parser.add_argument(
        "--max-windows-per-group",
        type=int,
        default=200,
        help="每个工位和切分组合最多保留多少个样本，0 表示不限制",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=["uniform_cover", "head"],
        default="uniform_cover",
        help="当窗口数超过上限时的抽样方式：uniform_cover 为全时段均匀抽样，head 为保留最前面的窗口",
    )
    return parser.parse_args()


def load_station_split_groups(con: duckdb.DuckDBPyConnection, input_csv: Path) -> list[tuple[str, str]]:
    sql = f"""
        SELECT DISTINCT op, dataset_split
        FROM read_csv_auto('{input_csv.as_posix()}', header = true, sample_size = 20000)
        ORDER BY op, dataset_split
    """
    return [(row[0], row[1]) for row in con.execute(sql).fetchall()]


def load_group_frame(
    con: duckdb.DuckDBPyConnection,
    input_csv: Path,
    station: str,
    split_name: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    selected_columns = ["id", "machine_model", "op", "event_time", "dataset_split", *feature_columns]
    select_sql = ", ".join(f'"{col}"' for col in selected_columns)
    sql = f"""
        SELECT {select_sql}
        FROM read_csv_auto('{input_csv.as_posix()}', header = true, sample_size = 20000)
        WHERE op = ? AND dataset_split = ?
        ORDER BY event_time
    """
    df = con.execute(sql, [station, split_name]).df()
    df["event_time"] = pd.to_datetime(df["event_time"])
    return df


def build_sample_rows(
    group_df: pd.DataFrame,
    feature_columns: list[str],
    window_size: int,
    stride: int,
    max_windows_per_group: int,
    sampling_mode: str,
) -> list[dict]:
    sample_rows: list[dict] = []
    total_rows = len(group_df)
    if total_rows < window_size:
        return sample_rows

    candidate_start_indices = list(range(0, total_rows - window_size + 1, stride))
    if max_windows_per_group > 0 and len(candidate_start_indices) > max_windows_per_group:
        if sampling_mode == "head":
            selected_start_indices = candidate_start_indices[:max_windows_per_group]
        else:
            # 在整段时间范围内均匀抽样，避免窗口只集中在每个工位切分的前半段。
            picked_positions = np.round(
                np.linspace(0, len(candidate_start_indices) - 1, num=max_windows_per_group)
            ).astype(int)
            selected_start_indices = [candidate_start_indices[pos] for pos in picked_positions]
    else:
        selected_start_indices = candidate_start_indices

    for start_idx in selected_start_indices:
        end_idx = start_idx + window_size
        window_df = group_df.iloc[start_idx:end_idx]

        sample_rows.append(
            {
                "sample_id": f"{group_df['dataset_split'].iloc[0]}_{group_df['op'].iloc[0]}_{start_idx:08d}",
                "machine_model": group_df["machine_model"].iloc[0],
                "op": group_df["op"].iloc[0],
                "dataset_split": group_df["dataset_split"].iloc[0],
                "start_row_in_group": start_idx,
                "end_row_in_group": end_idx - 1,
                "window_size": window_size,
                "stride": stride,
                "start_time": window_df["event_time"].iloc[0].strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": window_df["event_time"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S"),
                "source_first_id": str(window_df["id"].iloc[0]),
                "source_last_id": str(window_df["id"].iloc[-1]),
                # 保留特征列清单，后续训练或变换脚本可以直接读取。
                "feature_columns": ",".join(feature_columns),
            }
        )

    return sample_rows


def write_outputs(
    sample_rows: list[dict],
    output_dir: Path,
    input_csv: Path,
    feature_columns: list[str],
    window_size: int,
    stride: int,
    max_windows_per_group: int,
    sampling_mode: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    index_csv = output_dir / "window_sample_index.csv"
    summary_csv = output_dir / "window_group_summary.csv"

    sample_df = pd.DataFrame(sample_rows)
    sample_df.to_csv(index_csv, index=False)

    if sample_df.empty:
        summary_df = pd.DataFrame(
            columns=["machine_model", "op", "dataset_split", "sample_count", "first_start_time", "last_end_time"]
        )
    else:
        summary_df = (
            sample_df.groupby(["machine_model", "op", "dataset_split"], as_index=False)
            .agg(
                sample_count=("sample_id", "count"),
                first_start_time=("start_time", "min"),
                last_end_time=("end_time", "max"),
            )
            .sort_values(["machine_model", "op", "dataset_split"])
        )
    summary_df.to_csv(summary_csv, index=False)

    metadata = {
        "input_csv": str(input_csv),
        "index_csv": str(index_csv),
        "summary_csv": str(summary_csv),
        "feature_columns": feature_columns,
        "window_size": window_size,
        "stride": stride,
        "max_windows_per_group": max_windows_per_group,
        "sampling_mode": sampling_mode,
        "sample_count": int(len(sample_df)),
        "group_count": int(len(summary_df)),
        "description": "基于 final_dataset_table.csv 生成的滑窗样本索引，不复制原始数据，只记录窗口定位信息。",
    }
    (output_dir / "window_index_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return index_csv, summary_csv


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)

    con = duckdb.connect()
    con.execute("PRAGMA threads=4")

    sample_rows: list[dict] = []
    for station, split_name in load_station_split_groups(con, input_csv):
        group_df = load_group_frame(con, input_csv, station, split_name, DEFAULT_FEATURES)
        sample_rows.extend(
            build_sample_rows(
                group_df=group_df,
                feature_columns=DEFAULT_FEATURES,
                window_size=args.window_size,
                stride=args.stride,
                max_windows_per_group=args.max_windows_per_group,
                sampling_mode=args.sampling_mode,
            )
        )

    write_outputs(
        sample_rows=sample_rows,
        output_dir=output_dir,
        input_csv=input_csv,
        feature_columns=DEFAULT_FEATURES,
        window_size=args.window_size,
        stride=args.stride,
        max_windows_per_group=args.max_windows_per_group,
        sampling_mode=args.sampling_mode,
    )
    con.close()


if __name__ == "__main__":
    main()
