from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_FILE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "91-Site_DKA-M9_B-Phase.csv"
)
OUTPUT_FILE = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "91_site_missing_pattern_summary.csv"
)

TARGET_COLUMNS = [
    "Active_Power",
    "Radiation_Global_Tilted",
    "Radiation_Diffuse_Tilted",
    "Weather_Temperature_Celsius",
    "Wind_Speed",
    "Weather_Relative_Humidity",
]


def analyze_specific_missing_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    仅针对 6 个核心列，统计缺失模式摘要。

    返回字段：
    - missing_count: 单列缺失值总数
    - missing_percentage: 缺失值占总行数比例（%）
    - max_consecutive_missing: 最大连续缺失长度
    - missing_run_distribution: 连续缺失段分布，格式 {长度: 次数}
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("输入对象必须是 pandas.DataFrame。")

    df = df.copy()
    df.columns = df.columns.str.strip()
    total_rows = len(df)
    results: dict[str, dict[str, object]] = {}

    for col in TARGET_COLUMNS:
        try:
            if col not in df.columns:
                raise KeyError(
                    f"目标列 '{col}' 不存在。已先执行 df.columns.str.strip()，"
                    "请检查原始 CSV 的列名是否与预期一致。"
                )

            null_mask = df[col].isna()
            missing_count = int(null_mask.sum())
            missing_percentage = round(
                (missing_count / total_rows) * 100, 2
            ) if total_rows else 0.00

            if missing_count == 0:
                max_consecutive_missing = 0
                missing_run_distribution: dict[int, int] = {}
            else:
                # 缺失/非缺失状态切换时生成新分组，仅对缺失区块做长度统计。
                group_id = null_mask.ne(null_mask.shift(fill_value=False)).cumsum()
                run_lengths = null_mask[null_mask].groupby(group_id[null_mask]).size()

                max_consecutive_missing = int(run_lengths.max()) if not run_lengths.empty else 0
                missing_run_distribution = (
                    run_lengths.value_counts().sort_index().astype(int).to_dict()
                )

            results[col] = {
                "missing_count": missing_count,
                "missing_percentage": missing_percentage,
                "max_consecutive_missing": max_consecutive_missing,
                "missing_run_distribution": missing_run_distribution,
            }

        except KeyError as exc:
            results[col] = {
                "missing_count": None,
                "missing_percentage": None,
                "max_consecutive_missing": None,
                "missing_run_distribution": f"错误: {exc}",
            }
        except Exception as exc:
            results[col] = {
                "missing_count": None,
                "missing_percentage": None,
                "max_consecutive_missing": None,
                "missing_run_distribution": f"未知错误: {exc}",
            }

    return pd.DataFrame.from_dict(results, orient="index")


def load_91_site_data(csv_path: str | Path = DATA_FILE) -> pd.DataFrame:
    """
    读取 91-site 原始 CSV，并返回 DataFrame。
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"未找到 91-site 数据文件: {csv_path}\n"
            "当前脚本默认读取 data/raw/91-Site_DKA-M9_B-Phase.csv。"
        )

    return pd.read_csv(csv_path)


def export_summary_to_csv(
    summary_df: pd.DataFrame,
    output_path: str | Path = OUTPUT_FILE,
) -> Path:
    """
    导出缺失模式摘要到 CSV。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, encoding="utf-8-sig", index=True)
    return output_path


def main() -> None:
    df = load_91_site_data()
    summary_df = analyze_specific_missing_patterns(df)
    output_path = export_summary_to_csv(summary_df)
    print(summary_df)
    print(f"\n摘要 CSV 已导出至: {output_path}")


if __name__ == "__main__":
    main()
