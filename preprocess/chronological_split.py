"""
chronological_split.py
----------------------
按时间顺序切分清洗后的光伏核心特征数据。

功能：
    读取 `data/processed/core_features_clean.csv`，
    按原始时间顺序切分为：
        - Train: 70%
        - Validation: 10%
        - Calibration: 10%
        - Test: 10%

输出：
    默认写入 `data/processed/splits/` 目录：
        - train.csv
        - validation.csv
        - calibration.csv
        - test.csv
        - split_summary.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "core_features_clean.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"

DEFAULT_SPLIT_RATIOS = {
    "train": 0.70,
    "validation": 0.10,
    "calibration": 0.10,
    "test": 0.10,
}


def load_clean_core_features(csv_path: str | Path = INPUT_FILE) -> pd.DataFrame:
    """
    读取清洗后的核心特征数据。
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到清洗后的数据文件: {csv_path}")

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("输入数据必须使用 DatetimeIndex。")

    if df.empty:
        raise ValueError("输入数据为空，无法进行切分。")

    if not df.index.is_monotonic_increasing:
        raise ValueError("输入数据的时间索引不是递增顺序。请先按原始时间顺序排序后再切分。")

    return df


def split_dataframe_chronologically(
    df: pd.DataFrame,
    split_ratios: dict[str, float] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    按时间顺序对 DataFrame 做连续区间切分，不打乱顺序。
    """
    ratios = split_ratios or DEFAULT_SPLIT_RATIOS
    required_keys = ["train", "validation", "calibration", "test"]

    if set(ratios.keys()) != set(required_keys):
        raise ValueError(f"split_ratios 的键必须为: {required_keys}")

    ratio_values = np.array([ratios[key] for key in required_keys], dtype="float64")
    if np.any(ratio_values <= 0):
        raise ValueError("所有切分比例都必须大于 0。")

    if not np.isclose(ratio_values.sum(), 1.0):
        raise ValueError(f"切分比例之和必须为 1.0，当前为 {ratio_values.sum():.6f}")

    total_rows = len(df)
    cumulative = np.cumsum(ratio_values)

    train_end = int(np.floor(total_rows * cumulative[0]))
    validation_end = int(np.floor(total_rows * cumulative[1]))
    calibration_end = int(np.floor(total_rows * cumulative[2]))

    splits = {
        "train": df.iloc[:train_end].copy(),
        "validation": df.iloc[train_end:validation_end].copy(),
        "calibration": df.iloc[validation_end:calibration_end].copy(),
        "test": df.iloc[calibration_end:].copy(),
    }

    return splits


def build_split_summary(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    生成切分摘要表，便于快速检查每一段的行数与时间范围。
    """
    total_rows = sum(len(split_df) for split_df in splits.values())
    summary_rows: list[dict[str, object]] = []

    for split_name, split_df in splits.items():
        if split_df.empty:
            start_time = pd.NaT
            end_time = pd.NaT
        else:
            start_time = split_df.index[0]
            end_time = split_df.index[-1]

        summary_rows.append(
            {
                "split": split_name,
                "row_count": len(split_df),
                "actual_ratio_pct": round(len(split_df) / total_rows * 100, 2),
                "start_time": start_time,
                "end_time": end_time,
            }
        )

    return pd.DataFrame(summary_rows).set_index("split")


def save_splits(
    splits: dict[str, pd.DataFrame],
    output_dir: str | Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """
    保存 4 个切分结果及其摘要表。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_map = {
        "train": output_dir / "train.csv",
        "validation": output_dir / "validation.csv",
        "calibration": output_dir / "calibration.csv",
        "test": output_dir / "test.csv",
    }

    for split_name, split_df in splits.items():
        split_df.to_csv(file_map[split_name], encoding="utf-8-sig")

    summary_df = build_split_summary(splits)
    summary_df.to_csv(output_dir / "split_summary.csv", encoding="utf-8-sig")
    return summary_df


def main() -> None:
    df = load_clean_core_features()
    splits = split_dataframe_chronologically(df)
    summary_df = save_splits(splits)
    print(summary_df)
    print(f"\n切分结果已保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
