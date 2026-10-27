import argparse
import json
from pathlib import Path

import duckdb

from ..paths import DATA_DIR


PRIMARY_COLUMNS = {"id", "op", "date"}
def export_clean_csv(input_csv: Path, output_csv: Path, metadata_json: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute(f"PRAGMA temp_directory='{output_csv.parent.as_posix()}'")

    input_path = input_csv.as_posix()
    output_path = output_csv.as_posix()
    schema_rows = con.execute(
        f"""
        DESCRIBE
        SELECT *
        FROM read_csv_auto(
            '{input_path}',
            header = true,
            all_varchar = true,
            sample_size = 20000
        )
        """
    ).fetchall()
    headers = [row[0] for row in schema_rows]
    feature_columns = [col for col in headers if col not in PRIMARY_COLUMNS]
    numeric_selects = [
        f'TRY_CAST(NULLIF(TRIM("{col}"), \'\') AS DOUBLE) AS "{col}"'
        for col in feature_columns
    ]
    select_sql = ",\n                ".join(numeric_selects)

    con.execute(
        f"""
        COPY (
            SELECT
                id,
                regexp_replace(op, '_[0-9]+$', '') AS machine_model,
                op,
                strptime(date, '%d/%m/%Y %H:%M:%S') AS event_time,
                {select_sql}
            FROM read_csv_auto(
                '{input_path}',
                header = true,
                all_varchar = true,
                sample_size = 20000
            )
            WHERE try_strptime(date, '%d/%m/%Y %H:%M:%S') IS NOT NULL
        ) TO '{output_path}' (FORMAT CSV, HEADER)
        """
    )

    stats = con.execute(
        f"""
        SELECT
            COUNT(*) AS row_count,
            COUNT(DISTINCT machine_model) AS machine_model_count,
            COUNT(DISTINCT op) AS gwh_count,
            MIN(event_time) AS min_time,
            MAX(event_time) AS max_time
        FROM read_csv_auto(
            '{output_path}',
            header = true,
            sample_size = 20000
        )
        """
    ).fetchone()

    metadata = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "clean_rules": [
            "保留 id、op，并新增 machine_model",
            "将 date 标准化为 event_time 时间列",
            "所有设备参数列统一使用 DOUBLE 类型导出",
            "过滤无法解析时间的记录",
        ],
        "row_count": int(stats[0]),
        "machine_model_count": int(stats[1]),
        "gwh_count": int(stats[2]),
        "time_range": {
            "min": str(stats[3]) if stats[3] is not None else None,
            "max": str(stats[4]) if stats[4] is not None else None,
        },
        "feature_column_count": len(feature_columns),
        "feature_columns": feature_columns,
    }
    metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    con.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="清洗 data_1.csv，输出适合建模使用的宽表")
    parser.add_argument(
        "--input-csv",
        default=str(DATA_DIR / "data_1.csv"),
        help="原始宽表 CSV 路径",
    )
    parser.add_argument(
        "--output-csv",
        default=str(DATA_DIR / "data_1_clean.csv"),
        help="清洗后的 CSV 路径",
    )
    parser.add_argument(
        "--metadata-json",
        default=str(DATA_DIR / "data_1_clean_metadata.json"),
        help="清洗结果说明文件",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_clean_csv(
        input_csv=Path(args.input_csv),
        output_csv=Path(args.output_csv),
        metadata_json=Path(args.metadata_json),
    )


if __name__ == "__main__":
    main()
