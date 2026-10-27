"""从原始 sbxx.csv + dyxx.csv 构建清洗后的最终数据集，替代 data_1_clean.csv。

流程:
1. Join sbxx (传感器读数) + dyxx (事件元数据) via dyid = id
2. 过滤 9 个目标特征
3. Long → Wide pivot (每 dyid 一行，9 列传感器值)
4. 按 op + event_time 排序
5. 数据清洗 (温度 <= -100 → NaN, 线性插值, 前向/后向填充)
6. 按 op 时间顺序切分 train/val/test (70/15/15)
7. 构建滑窗索引
8. 输出 dataset_final/

用法: python -m spindle_cnn.data.build_dataset_from_raw
"""

import argparse
import json
import math
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from ..paths import DATA_DIR, DATASET_DIR, PROJECT_ROOT

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

PROJECT = PROJECT_ROOT
DATA = DATA_DIR
OUT = DATASET_DIR


def step1_extract_pivoted() -> Path:
    """从 sbxx + dyxx 提取并 pivot 为宽表。"""
    print("=" * 60)
    print("Step 1: 从 sbxx + dyxx 提取 9 特征 pivoted 表")
    print("=" * 60)

    con = duckdb.connect()
    con.execute("SET threads = 8")
    # 大内存模式加速
    con.execute("SET memory_limit = '16GB'")

    # 特征过滤列表
    feat_list = ", ".join(f"'{f}'" for f in FEATURE_COLUMNS)

    sql = f"""
    WITH
    -- dyxx: 事件元数据, id 可能有空格, jlsj 可能有无效时间
    dy_raw AS (
        SELECT
            id AS id_raw,
            gwh AS op_raw,
            jlsj AS jlsj_raw
        FROM read_csv_auto('{DATA.as_posix()}/dyxx.csv',
            all_varchar=true
        )
    ),
    dy AS (
        SELECT
            REPLACE(id_raw, ' ', '') AS id_clean,
            op_raw AS op,
            TRY_STRPTIME(jlsj_raw, '%d/%m/%Y %H:%M:%S') AS event_time
        FROM dy_raw
        WHERE op_raw IS NOT NULL
          AND TRY_STRPTIME(jlsj_raw, '%d/%m/%Y %H:%M:%S') IS NOT NULL
    ),
    -- sbxx: 传感器读数, dyid 可能有脏数据
    sb_raw AS (
        SELECT
            id,
            dyid AS dyid_raw,
            bj,
            sz
        FROM read_csv_auto('{DATA.as_posix()}/sbxx.csv',
            all_varchar=true
        )
    ),
    sb AS (
        SELECT
            id,
            REPLACE(dyid_raw, ' ', '') AS dyid_clean,
            bj,
            TRY_CAST(sz AS DOUBLE) AS sz
        FROM sb_raw
        WHERE bj IN ({feat_list})
          AND TRY_CAST(sz AS DOUBLE) IS NOT NULL
    ),
    -- Join
    joined AS (
        SELECT
            dy.op,
            dy.event_time,
            sb.bj,
            sb.sz
        FROM sb
        INNER JOIN dy ON sb.dyid_clean = dy.id_clean
    )
    -- Pivot: 每个 (op, event_time) 一行
    SELECT
        op,
        event_time,
        MAX(CASE WHEN bj = 'OilMistTemperature' THEN sz END) AS OilMistTemperature,
        MAX(CASE WHEN bj = 'Motor1_Current' THEN sz END) AS Motor1_Current,
        MAX(CASE WHEN bj = 'Motor1_Temperature' THEN sz END) AS Motor1_Temperature,
        MAX(CASE WHEN bj = 'Motor1_Torque' THEN sz END) AS Motor1_Torque,
        MAX(CASE WHEN bj = 'Motor1_Speed' THEN sz END) AS Motor1_Speed,
        MAX(CASE WHEN bj = 'Motor2_Current' THEN sz END) AS Motor2_Current,
        MAX(CASE WHEN bj = 'Motor2_Temperature' THEN sz END) AS Motor2_Temperature,
        MAX(CASE WHEN bj = 'Motor2_Torque' THEN sz END) AS Motor2_Torque,
        MAX(CASE WHEN bj = 'Motor2_Speed' THEN sz END) AS Motor2_Speed
    FROM joined
    GROUP BY op, event_time
    ORDER BY op, event_time
    """
    print("  查询中 (join + pivot 约需 2-5 分钟)...")
    df = con.execute(sql).fetchdf()
    con.close()

    out_path = OUT / "raw_pivoted.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  输出: {out_path} | {len(df):,} 行")
    print(f"  Ops: {sorted(df['op'].unique().tolist())}")
    print(f"  时间范围: {df['event_time'].min()} ~ {df['event_time'].max()}")
    return out_path


def step2_clean(pivoted_path: Path) -> Path:
    """清洗数据。"""
    print()
    print("=" * 60)
    print("Step 2: 数据清洗")
    print("=" * 60)

    df = pd.read_parquet(pivoted_path)

    # 温度列 <= -100 视为无效
    temp_cols = ["OilMistTemperature", "Motor1_Temperature", "Motor2_Temperature"]
    for col in temp_cols:
        mask = df[col] <= -100
        df.loc[mask, col] = np.nan
        print(f"  {col}: {mask.sum():,} 个无效值 (<= -100) 置为 NaN")

    # 按 op 分组，按时间排序，线性插值
    print("  线性插值 + 前后向填充...")
    df = df.sort_values(["op", "event_time"])
    for col in FEATURE_COLUMNS:
        df[col] = df.groupby("op")[col].transform(
            lambda s: s.interpolate(method="linear", limit_direction="both")
        )
    # 兜底: 中位数 + 0
    for col in FEATURE_COLUMNS:
        mask = df[col].isna()
        if mask.any():
            df.loc[mask, col] = df[col].median()
        df[col] = df[col].fillna(0.0)

    df = df.sort_values(["op", "event_time"]).reset_index(drop=True)

    out_path = OUT / "clean_pivoted.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  输出: {out_path} | {len(df):,} 行")
    return out_path


def step3_split(clean_path: Path) -> Path:
    """按 op 时间顺序切分 train/val/test。"""
    print()
    print("=" * 60)
    print("Step 3: 70/15/15 时间顺序切分")
    print("=" * 60)

    df = pd.read_parquet(clean_path)
    df["dataset_split"] = ""

    for op, group in df.groupby("op", sort=False):
        n = len(group)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        indices = group.index
        df.loc[indices[:train_end], "dataset_split"] = "train"
        df.loc[indices[train_end:val_end], "dataset_split"] = "val"
        df.loc[indices[val_end:], "dataset_split"] = "test"

    for split_name in ["train", "val", "test"]:
        cnt = (df["dataset_split"] == split_name).sum()
        print(f"  {split_name}: {cnt:,} 行 ({cnt / len(df) * 100:.1f}%)")

    out_path = OUT / "clean_split.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  输出: {out_path}")
    return out_path


def step4_window_index(split_path: Path, stride: int = 500) -> Path:
    """构建滑窗索引。stride=500 即 50% 重叠，stride=1000 即不重叠。"""
    print()
    print("=" * 60)
    print(f"Step 4: 构建滑窗样本索引 (window=1000, stride={stride}, overlap={1000-stride})")
    print("=" * 60)

    df = pd.read_parquet(split_path)
    window_size = 1000

    rows = []
    sample_id = 0
    for (op, split_name), group in df.groupby(["op", "dataset_split"], sort=False):
        group = group.reset_index(drop=True)
        n = len(group)
        num_windows = max(0, (n - window_size) // stride + 1)
        if num_windows == 0:
            continue
        for w in range(num_windows):
            start = w * stride
            end = start + window_size - 1
            rows.append(
                {
                    "sample_id": sample_id,
                    "op": op,
                    "dataset_split": split_name,
                    "start_row_in_group": start,
                    "end_row_in_group": end,
                }
            )
            sample_id += 1

    window_df = pd.DataFrame(rows)
    print(f"  总样本: {len(window_df):,}")

    # 每个 split 的 op 分布
    for split_name in ["train", "val", "test"]:
        split_samples = window_df[window_df["dataset_split"] == split_name]
        print(f"  {split_name}: {len(split_samples):,} 样本, {split_samples['op'].nunique()} ops")

    wdir = OUT / "window_index"
    wdir.mkdir(parents=True, exist_ok=True)
    out_path = wdir / "window_sample_index.csv"
    window_df.to_csv(out_path, index=False)
    print(f"  输出: {out_path}")
    return out_path


def step5_export_csv(split_path: Path) -> Path:
    """导出兼容现有训练脚本的 CSV 格式。"""
    print()
    print("=" * 60)
    print("Step 5: 导出 final_dataset_table.csv")
    print("=" * 60)

    df = pd.read_parquet(split_path)
    df["id"] = range(len(df))
    df["machine_model"] = ""
    cols = ["id", "machine_model", "op", "event_time"] + FEATURE_COLUMNS + ["dataset_split"]
    df = df[cols]

    out_path = OUT / "final_dataset_table.csv"
    df.to_csv(out_path, index=False)
    size_gb = os.path.getsize(out_path) / (1024 ** 3)
    print(f"  输出: {out_path} ({size_gb:.1f} GB)")
    return out_path


def step6_save_metadata(split_path: Path, window_index_path: Path) -> None:
    """保存数据集元信息。"""
    df = pd.read_parquet(split_path)
    wi = pd.read_csv(window_index_path)

    meta = {
        "source": "sbxx.csv + dyxx.csv",
        "total_events": len(df),
        "window_count": len(wi),
        "window_size": 1000,
        "feature_columns": FEATURE_COLUMNS,
        "ops": sorted(df["op"].unique().tolist()),
        "time_range": [str(df["event_time"].min()), str(df["event_time"].max())],
        "split_strategy": "每个工位按时间顺序切分为 train 70% / val 15% / test 15%",
        "missing_value_strategy": [
            "温度列中 <= -100 的值视为无效缺失",
            "按工位、按时间顺序进行线性插值",
            "首尾缺失使用前向/后向填充",
            "仍为空的列使用中位数，再兜底填 0",
        ],
        "output_columns": ["id", "machine_model", "op", "event_time"] + FEATURE_COLUMNS + ["dataset_split"],
    }
    (OUT / "dataset_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 工位缺失统计
    summary_rows = []
    for op, group in df.groupby("op", sort=False):
        row = {"op": op}
        for col in FEATURE_COLUMNS:
            row[f"{col}_missing_pct"] = group[col].isna().mean()
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(OUT / "station_missing_summary.csv", index=False)

    print()
    print(f"  元数据: {OUT / 'dataset_metadata.json'}")
    print(f"  缺失统计: {OUT / 'station_missing_summary.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 sbxx.csv 和 dyxx.csv 构建清洗、切分后的最终数据集"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=500,
        help="滑窗步长；窗口长度为 1000，默认使用 50%% 重叠",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print("从原始数据 (sbxx.csv + dyxx.csv) 构建最终数据集")
    print(f"输出目录: {OUT}")
    print()

    # Step 1: 提取 + pivot
    pivoted = step1_extract_pivoted()

    # Step 2: 清洗
    clean = step2_clean(pivoted)

    # Step 3: 切分
    split = step3_split(clean)

    # Step 4: 滑窗索引 (50% 重叠)
    wi = step4_window_index(split, stride=args.stride)

    # Step 5: 导出 CSV
    step5_export_csv(split)

    # Step 6: 元数据
    step6_save_metadata(split, wi)

    print()
    print("=" * 60)
    print("完成! 数据集已生成到 dataset_final/")
    print("=" * 60)


if __name__ == "__main__":
    main()
