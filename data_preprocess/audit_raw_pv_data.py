from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.project_config import load_project_config, resolve_project_path

PROJECT_CONFIG = load_project_config()
DEFAULT_INPUT_FILE = PROJECT_CONFIG.get_path(
    "paths.raw_input_file",
    PROJECT_ROOT / "data" / "raw" / "91-Site_DKA-M9_B-Phase.csv",
)
DEFAULT_OUTPUT_DIR = PROJECT_CONFIG.get_path(
    "paths.results.data_audit_raw_pv",
    PROJECT_ROOT / "results" / "data_audit" / "raw_pv",
)

TIME_COLUMN_CANDIDATES = (
    "timestamp",
    "datetime",
    "date_time",
    "time",
    "date",
    "ds",
)

DEFAULT_REQUIRED_COLUMNS = [
    "Active_Power",
    "Radiation_Global_Tilted",
    "Radiation_Diffuse_Tilted",
    "Weather_Temperature_Celsius",
    "Weather_Relative_Humidity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit raw PV time-series data before selecting a modeling subset. "
            "The script detects timestamps, checks sampling stability, missing "
            "patterns, wind-speed availability, and candidate high-quality intervals."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_FILE),
        help="Raw CSV path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for audit CSV/JSON/Markdown reports.",
    )
    parser.add_argument(
        "--time_col",
        type=str,
        default=None,
        help="Optional explicit timestamp column. If omitted, the script auto-detects it.",
    )
    parser.add_argument(
        "--expected_freq_minutes",
        type=float,
        default=None,
        help="Expected sampling interval in minutes. If omitted, inferred from timestamp diffs.",
    )
    parser.add_argument(
        "--long_gap_steps",
        type=int,
        default=12,
        help="Missing runs with at least this many rows are reported as long gaps.",
    )
    parser.add_argument(
        "--candidate_window_days",
        type=int,
        nargs="+",
        default=[90, 180, 365, 545],
        help="Window lengths used to rank candidate modeling intervals.",
    )
    parser.add_argument(
        "--candidate_step_days",
        type=int,
        default=30,
        help="Step size for scanning candidate intervals.",
    )
    parser.add_argument(
        "--ignore_wind_for_candidates",
        action="store_true",
        help=(
            "Keep reporting wind-variable missingness, but do not use wind columns "
            "when scoring candidate modeling intervals. Use this for newer no-wind experiments."
        ),
    )
    parser.add_argument(
        "--required_columns",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Columns treated as mandatory for PV modeling quality. Defaults to "
            "power, tilted radiation, temperature, and humidity columns when present."
        ),
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of top candidate intervals included in the Markdown report.",
    )
    return parser.parse_args()


def normalize_column_name(name: Any) -> str:
    text = str(name).strip().lstrip("\ufeff")
    return text


def compact_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def is_parseable_datetime(series: pd.Series, min_success_ratio: float = 0.95) -> bool:
    sample = series.dropna()
    if sample.empty:
        return False
    if len(sample) > 10_000:
        sample = sample.iloc[:10_000]

    parsed = pd.to_datetime(sample, errors="coerce")
    success_ratio = float(parsed.notna().mean())
    if success_ratio < min_success_ratio:
        return False

    parsed = parsed.dropna()
    return parsed.nunique() >= 2 and parsed.min() < parsed.max()


def detect_time_column(df: pd.DataFrame, preferred: str | None = None) -> str:
    if preferred is not None:
        if preferred not in df.columns:
            raise KeyError(f"Explicit --time_col '{preferred}' is not in the CSV columns.")
        if not is_parseable_datetime(df[preferred]):
            raise ValueError(f"Explicit --time_col '{preferred}' is not parseable as datetime.")
        return preferred

    checked: set[str] = set()
    for candidate in TIME_COLUMN_CANDIDATES:
        if candidate in df.columns:
            checked.add(candidate)
            if is_parseable_datetime(df[candidate]):
                return candidate

    object_columns = [
        column
        for column in df.columns
        if column not in checked and (pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]))
    ]
    for column in object_columns:
        checked.add(column)
        if is_parseable_datetime(df[column]):
            return column

    time_like_columns = [
        column
        for column in df.columns
        if column not in checked and any(token in compact_name(column) for token in TIME_COLUMN_CANDIDATES)
    ]
    for column in time_like_columns:
        if is_parseable_datetime(df[column]):
            return column

    raise ValueError(
        "Failed to auto-detect a timestamp column. Pass --time_col explicitly."
    )


def load_raw_csv(path: Path, time_col: str | None) -> tuple[pd.DataFrame, str, int]:
    if not path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    df.columns = [normalize_column_name(column) for column in df.columns]
    detected_time_col = detect_time_column(df, preferred=time_col)
    parsed_time = pd.to_datetime(df[detected_time_col], errors="coerce")
    invalid_timestamp_rows = int(parsed_time.isna().sum())

    df = df.loc[parsed_time.notna()].copy()
    df[detected_time_col] = parsed_time.loc[parsed_time.notna()].to_numpy()
    df = df.sort_values(detected_time_col).reset_index(drop=True)
    return df, detected_time_col, invalid_timestamp_rows


def infer_expected_delta(
    timestamps: pd.Series,
    expected_freq_minutes: float | None,
) -> pd.Timedelta:
    if expected_freq_minutes is not None:
        if expected_freq_minutes <= 0:
            raise ValueError("--expected_freq_minutes must be positive.")
        return pd.Timedelta(minutes=float(expected_freq_minutes))

    diffs = timestamps.diff().dropna()
    positive_diffs = diffs[diffs > pd.Timedelta(0)]
    if positive_diffs.empty:
        raise ValueError("Cannot infer sampling interval because no positive timestamp diff exists.")

    value_counts = positive_diffs.value_counts()
    return value_counts.index[0]


def build_time_diff_distribution(timestamps: pd.Series) -> pd.DataFrame:
    diffs = timestamps.diff().dropna()
    if diffs.empty:
        return pd.DataFrame(columns=["delta", "delta_minutes", "count", "ratio_pct"])

    counts = diffs.value_counts().rename_axis("delta").reset_index(name="count")
    counts = counts.sort_values("delta").reset_index(drop=True)
    counts["delta_minutes"] = counts["delta"].dt.total_seconds() / 60.0
    counts["ratio_pct"] = counts["count"] / len(diffs) * 100.0
    counts["delta"] = counts["delta"].astype(str)
    return counts[["delta", "delta_minutes", "count", "ratio_pct"]]


def build_timestamp_gap_table(
    timestamps: pd.Series,
    expected_delta: pd.Timedelta,
) -> pd.DataFrame:
    diffs = timestamps.diff()
    irregular_mask = diffs.notna() & (diffs != expected_delta)
    if not irregular_mask.any():
        return pd.DataFrame(
            columns=[
                "previous_timestamp",
                "next_timestamp",
                "delta",
                "delta_minutes",
                "gap_type",
                "estimated_missing_steps",
            ]
        )

    gap_df = pd.DataFrame(
        {
            "previous_timestamp": timestamps.shift(1).loc[irregular_mask].astype(str).to_numpy(),
            "next_timestamp": timestamps.loc[irregular_mask].astype(str).to_numpy(),
            "delta": diffs.loc[irregular_mask].astype(str).to_numpy(),
            "delta_minutes": diffs.loc[irregular_mask].dt.total_seconds().to_numpy() / 60.0,
        }
    )

    expected_seconds = expected_delta.total_seconds()
    estimated_missing_steps: list[int] = []
    gap_types: list[str] = []
    for delta_minutes in gap_df["delta_minutes"].to_numpy(dtype=float):
        delta_seconds = delta_minutes * 60.0
        if delta_seconds <= 0:
            gap_types.append("duplicate_or_non_positive")
            estimated_missing_steps.append(0)
        elif math.isclose(delta_seconds, expected_seconds):
            gap_types.append("expected")
            estimated_missing_steps.append(0)
        elif delta_seconds > expected_seconds:
            gap_types.append("larger_than_expected")
            estimated_missing_steps.append(max(int(round(delta_seconds / expected_seconds)) - 1, 0))
        else:
            gap_types.append("shorter_than_expected")
            estimated_missing_steps.append(0)

    gap_df["gap_type"] = gap_types
    gap_df["estimated_missing_steps"] = estimated_missing_steps
    return gap_df


def summarize_sampling(
    timestamps: pd.Series,
    expected_delta: pd.Timedelta,
    invalid_timestamp_rows: int,
) -> dict[str, Any]:
    diffs = timestamps.diff().dropna()
    positive_diffs = diffs[diffs > pd.Timedelta(0)]
    expected_count = int((diffs == expected_delta).sum())
    irregular_count = int((diffs != expected_delta).sum())
    larger_than_expected = diffs[diffs > expected_delta]
    duplicated_rows = int(timestamps.duplicated().sum())

    return {
        "rows_after_valid_timestamp_filter": int(len(timestamps)),
        "invalid_timestamp_rows_dropped": int(invalid_timestamp_rows),
        "start_time": str(timestamps.iloc[0]) if len(timestamps) else None,
        "end_time": str(timestamps.iloc[-1]) if len(timestamps) else None,
        "duplicate_timestamp_rows": duplicated_rows,
        "expected_sampling_interval": str(expected_delta),
        "expected_sampling_minutes": expected_delta.total_seconds() / 60.0,
        "timestamp_diff_count": int(len(diffs)),
        "expected_delta_count": expected_count,
        "expected_delta_ratio_pct": expected_count / len(diffs) * 100.0 if len(diffs) else None,
        "irregular_delta_count": irregular_count,
        "larger_than_expected_gap_count": int(len(larger_than_expected)),
        "estimated_missing_timestamp_steps": int(
            sum(max(int(round(delta / expected_delta)) - 1, 0) for delta in larger_than_expected)
        ),
        "min_positive_delta_minutes": (
            positive_diffs.min().total_seconds() / 60.0 if not positive_diffs.empty else None
        ),
        "max_positive_delta_minutes": (
            positive_diffs.max().total_seconds() / 60.0 if not positive_diffs.empty else None
        ),
    }


def build_missing_rate_table(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    data_cols = [column for column in df.columns if column != time_col]
    total_rows = len(df)
    rows: list[dict[str, Any]] = []
    for column in data_cols:
        missing_count = int(df[column].isna().sum())
        rows.append(
            {
                "column": column,
                "non_missing_count": int(total_rows - missing_count),
                "missing_count": missing_count,
                "missing_rate_pct": missing_count / total_rows * 100.0 if total_rows else np.nan,
                "dtype": str(df[column].dtype),
            }
        )
    return pd.DataFrame(rows).sort_values("missing_rate_pct", ascending=False).reset_index(drop=True)


def build_boolean_segments(
    timestamps: pd.Series,
    mask: pd.Series | np.ndarray,
    expected_delta: pd.Timedelta,
) -> pd.DataFrame:
    arr = np.asarray(mask, dtype=bool)
    if len(arr) == 0 or not arr.any():
        return pd.DataFrame(
            columns=[
                "start_time",
                "end_time",
                "start_row",
                "end_row",
                "row_count",
                "calendar_span_minutes",
                "expected_span_minutes",
            ]
        )

    starts = np.flatnonzero(arr & np.r_[True, ~arr[:-1]])
    ends = np.flatnonzero(arr & np.r_[~arr[1:], True])
    timestamp_values = timestamps.reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    expected_minutes = expected_delta.total_seconds() / 60.0

    for start, end in zip(starts, ends):
        start_time = timestamp_values.iloc[int(start)]
        end_time = timestamp_values.iloc[int(end)]
        row_count = int(end - start + 1)
        rows.append(
            {
                "start_time": str(start_time),
                "end_time": str(end_time),
                "start_row": int(start),
                "end_row": int(end),
                "row_count": row_count,
                "calendar_span_minutes": (end_time - start_time).total_seconds() / 60.0,
                "expected_span_minutes": (row_count - 1) * expected_minutes,
            }
        )

    return pd.DataFrame(rows)


def build_missing_segment_tables(
    df: pd.DataFrame,
    time_col: str,
    expected_delta: pd.Timedelta,
    long_gap_steps: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    timestamps = df[time_col]
    data_cols = [column for column in df.columns if column != time_col]
    segment_frames: list[pd.DataFrame] = []

    for column in data_cols:
        segments = build_boolean_segments(timestamps, df[column].isna(), expected_delta)
        if segments.empty:
            continue
        segments.insert(0, "column", column)
        segments["is_long_gap"] = segments["row_count"] >= long_gap_steps
        segment_frames.append(segments)

    if not segment_frames:
        all_segments = pd.DataFrame(
            columns=[
                "column",
                "start_time",
                "end_time",
                "start_row",
                "end_row",
                "row_count",
                "calendar_span_minutes",
                "expected_span_minutes",
                "is_long_gap",
            ]
        )
    else:
        all_segments = pd.concat(segment_frames, ignore_index=True)
        all_segments = all_segments.sort_values(["column", "start_row"]).reset_index(drop=True)

    long_segments = all_segments.loc[all_segments["is_long_gap"]].copy()
    long_segments = long_segments.sort_values(["row_count", "calendar_span_minutes"], ascending=False).reset_index(drop=True)
    return all_segments, long_segments


def build_missing_segment_summary(missing_segments: pd.DataFrame) -> pd.DataFrame:
    if missing_segments.empty:
        return pd.DataFrame(
            columns=[
                "column",
                "missing_segment_count",
                "long_missing_segment_count",
                "max_consecutive_missing_rows",
                "median_consecutive_missing_rows",
                "total_missing_rows_in_segments",
            ]
        )

    summary = (
        missing_segments.groupby("column")
        .agg(
            missing_segment_count=("row_count", "size"),
            long_missing_segment_count=("is_long_gap", "sum"),
            max_consecutive_missing_rows=("row_count", "max"),
            median_consecutive_missing_rows=("row_count", "median"),
            total_missing_rows_in_segments=("row_count", "sum"),
        )
        .reset_index()
    )
    return summary.sort_values("max_consecutive_missing_rows", ascending=False).reset_index(drop=True)


def detect_wind_columns(columns: list[str]) -> tuple[list[str], list[str]]:
    wind_speed_cols: list[str] = []
    wind_related_cols: list[str] = []

    for column in columns:
        lower = column.lower()
        compact = compact_name(column)
        if "wind" in lower or "风" in column:
            wind_related_cols.append(column)
        if ("wind" in lower and "speed" in lower) or "windspeed" in compact or "风速" in column:
            wind_speed_cols.append(column)

    return wind_speed_cols, wind_related_cols


def build_period_quality_table(
    df: pd.DataFrame,
    time_col: str,
    period: str,
    required_cols: list[str],
    wind_cols: list[str],
) -> pd.DataFrame:
    working = df.copy()
    working["_period"] = pd.to_datetime(working[time_col]).dt.to_period(period).astype(str)
    rows: list[dict[str, Any]] = []

    for period_value, group in working.groupby("_period", sort=True):
        row: dict[str, Any] = {
            "period": period_value,
            "start_time": str(group[time_col].iloc[0]),
            "end_time": str(group[time_col].iloc[-1]),
            "rows": int(len(group)),
        }
        if required_cols:
            required_missing = group[required_cols].isna()
            row["required_cell_missing_rate_pct"] = float(required_missing.to_numpy().mean() * 100.0)
            row["required_complete_row_rate_pct"] = float((~required_missing.any(axis=1)).mean() * 100.0)
        if wind_cols:
            wind_missing = group[wind_cols].isna()
            row["wind_cell_missing_rate_pct"] = float(wind_missing.to_numpy().mean() * 100.0)
            row["wind_complete_row_rate_pct"] = float((~wind_missing.any(axis=1)).mean() * 100.0)
            for column in wind_cols:
                row[f"{column}_missing_rate_pct"] = float(group[column].isna().mean() * 100.0)
        rows.append(row)

    return pd.DataFrame(rows)


def make_group_cache(df: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    if not columns:
        return {"columns": [], "missing_cumsum": None, "any_missing_cumsum": None}

    missing = df[columns].isna().to_numpy(dtype=np.int64)
    missing_cumsum = np.vstack(
        [np.zeros((1, missing.shape[1]), dtype=np.int64), np.cumsum(missing, axis=0)]
    )
    any_missing = missing.any(axis=1).astype(np.int64)
    any_missing_cumsum = np.concatenate([[0], np.cumsum(any_missing)])
    return {
        "columns": columns,
        "missing_cumsum": missing_cumsum,
        "any_missing_cumsum": any_missing_cumsum,
    }


def interval_group_stats(cache: dict[str, Any], start: int, end: int) -> dict[str, float | None]:
    columns = cache["columns"]
    row_count = end - start
    if row_count <= 0 or not columns:
        return {
            "cell_missing_rate_pct": None,
            "complete_row_rate_pct": None,
            "max_column_missing_rate_pct": None,
        }

    missing_counts = cache["missing_cumsum"][end] - cache["missing_cumsum"][start]
    any_missing_count = cache["any_missing_cumsum"][end] - cache["any_missing_cumsum"][start]
    column_missing_rates = missing_counts / row_count * 100.0

    return {
        "cell_missing_rate_pct": float(missing_counts.sum() / (row_count * len(columns)) * 100.0),
        "complete_row_rate_pct": float((1.0 - any_missing_count / row_count) * 100.0),
        "max_column_missing_rate_pct": float(column_missing_rates.max()),
    }


def scan_candidate_intervals(
    df: pd.DataFrame,
    time_col: str,
    expected_delta: pd.Timedelta,
    required_cols: list[str],
    wind_cols: list[str],
    window_days: list[int],
    step_days: int,
) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()
    if step_days <= 0:
        raise ValueError("--candidate_step_days must be positive.")

    timestamps = pd.DatetimeIndex(pd.to_datetime(df[time_col]))
    start_min = timestamps.min().floor("D")
    end_max = timestamps.max()
    expected_seconds = expected_delta.total_seconds()

    required_cache = make_group_cache(df, required_cols)
    wind_cache = make_group_cache(df, wind_cols)
    all_data_cols = [column for column in df.columns if column != time_col]
    all_cache = make_group_cache(df, all_data_cols)

    rows: list[dict[str, Any]] = []
    for days in sorted(set(window_days)):
        if days <= 0:
            continue
        window_delta = pd.Timedelta(days=int(days))
        step_delta = pd.Timedelta(days=int(step_days))
        current_start = start_min
        while current_start + window_delta <= end_max:
            current_end = current_start + window_delta
            start_pos = int(timestamps.searchsorted(current_start, side="left"))
            end_pos = int(timestamps.searchsorted(current_end, side="left"))
            row_count = end_pos - start_pos

            if row_count > 0:
                window_diffs = pd.Series(timestamps[start_pos:end_pos]).diff().dropna()
                if len(window_diffs):
                    expected_diff_ratio_pct = float((window_diffs == expected_delta).mean() * 100.0)
                    timestamp_gap_count = int((window_diffs > expected_delta).sum())
                    irregular_delta_count = int((window_diffs != expected_delta).sum())
                else:
                    expected_diff_ratio_pct = None
                    timestamp_gap_count = 0
                    irregular_delta_count = 0
            else:
                expected_diff_ratio_pct = None
                timestamp_gap_count = 0
                irregular_delta_count = 0

            expected_rows = int(round(window_delta.total_seconds() / expected_seconds))
            time_coverage_ratio_pct = (
                min(row_count / expected_rows * 100.0, 100.0) if expected_rows > 0 else None
            )

            required_stats = interval_group_stats(required_cache, start_pos, end_pos)
            wind_stats = interval_group_stats(wind_cache, start_pos, end_pos)
            all_stats = interval_group_stats(all_cache, start_pos, end_pos)

            required_complete = (required_stats["complete_row_rate_pct"] or 0.0) / 100.0
            wind_complete = (wind_stats["complete_row_rate_pct"] or 0.0) / 100.0
            continuity = (expected_diff_ratio_pct or 0.0) / 100.0
            coverage = (time_coverage_ratio_pct or 0.0) / 100.0
            season_score = min(days / 365.0, 1.0)

            if wind_cols:
                quality_score = 100.0 * (
                    0.35 * required_complete
                    + 0.25 * wind_complete
                    + 0.20 * continuity
                    + 0.10 * coverage
                    + 0.10 * season_score
                )
            else:
                quality_score = 100.0 * (
                    0.60 * required_complete
                    + 0.20 * continuity
                    + 0.10 * coverage
                    + 0.10 * season_score
                )

            rows.append(
                {
                    "window_days": int(days),
                    "start_time": str(current_start),
                    "end_time_exclusive": str(current_end),
                    "rows": int(row_count),
                    "expected_rows": int(expected_rows),
                    "time_coverage_ratio_pct": time_coverage_ratio_pct,
                    "expected_delta_ratio_pct": expected_diff_ratio_pct,
                    "timestamp_gap_count": timestamp_gap_count,
                    "irregular_delta_count": irregular_delta_count,
                    "required_cell_missing_rate_pct": required_stats["cell_missing_rate_pct"],
                    "required_complete_row_rate_pct": required_stats["complete_row_rate_pct"],
                    "required_max_column_missing_rate_pct": required_stats["max_column_missing_rate_pct"],
                    "wind_cell_missing_rate_pct": wind_stats["cell_missing_rate_pct"],
                    "wind_complete_row_rate_pct": wind_stats["complete_row_rate_pct"],
                    "wind_max_column_missing_rate_pct": wind_stats["max_column_missing_rate_pct"],
                    "all_columns_cell_missing_rate_pct": all_stats["cell_missing_rate_pct"],
                    "quality_score": float(quality_score),
                }
            )
            current_start += step_delta

    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return candidates

    candidates = candidates.sort_values(
        ["window_days", "quality_score", "required_complete_row_rate_pct", "wind_complete_row_rate_pct"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    candidates["rank_within_window_days"] = (
        candidates.groupby("window_days")["quality_score"].rank(method="first", ascending=False).astype(int)
    )
    return candidates.sort_values("quality_score", ascending=False).reset_index(drop=True)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if pd.isna(value):
        return None
    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(data), ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown_report(
    path: Path,
    input_path: Path,
    time_col: str,
    sampling_summary: dict[str, Any],
    missing_rates: pd.DataFrame,
    missing_segment_summary: pd.DataFrame,
    wind_speed_cols: list[str],
    wind_related_cols: list[str],
    yearly_quality: pd.DataFrame,
    candidate_intervals: pd.DataFrame,
    top_k: int,
) -> None:
    lines: list[str] = []

    def format_optional_float(value: Any) -> str:
        if value is None or pd.isna(value):
            return "n/a"
        return f"{float(value):.3f}"

    lines.append("# Raw PV Data Audit Report")
    lines.append("")
    lines.append(f"- Input file: `{input_path}`")
    lines.append(f"- Detected time column: `{time_col}`")
    lines.append(f"- Time span: {sampling_summary['start_time']} to {sampling_summary['end_time']}")
    lines.append(f"- Rows after timestamp parsing: {sampling_summary['rows_after_valid_timestamp_filter']}")
    lines.append(f"- Expected sampling interval: {sampling_summary['expected_sampling_interval']}")
    lines.append(f"- Expected-delta ratio: {sampling_summary['expected_delta_ratio_pct']:.4f}%")
    lines.append(f"- Larger-than-expected timestamp gaps: {sampling_summary['larger_than_expected_gap_count']}")
    lines.append(f"- Estimated missing timestamp steps: {sampling_summary['estimated_missing_timestamp_steps']}")
    lines.append("")

    lines.append("## Missing Rate by Column")
    for _, row in missing_rates.head(20).iterrows():
        lines.append(
            f"- `{row['column']}`: {row['missing_rate_pct']:.3f}% "
            f"({int(row['missing_count'])} / {int(row['missing_count'] + row['non_missing_count'])})"
        )
    lines.append("")

    lines.append("## Long Missing Runs")
    if missing_segment_summary.empty:
        lines.append("- No missing segments detected.")
    else:
        for _, row in missing_segment_summary.head(20).iterrows():
            lines.append(
                f"- `{row['column']}`: max run {int(row['max_consecutive_missing_rows'])} rows, "
                f"long runs {int(row['long_missing_segment_count'])}, "
                f"segments {int(row['missing_segment_count'])}"
            )
    lines.append("")

    lines.append("## Wind Variables")
    lines.append(f"- Wind-speed columns: {wind_speed_cols if wind_speed_cols else 'not detected'}")
    lines.append(f"- Wind-related columns: {wind_related_cols if wind_related_cols else 'not detected'}")
    if wind_speed_cols and not yearly_quality.empty:
        display_cols = ["period", "rows", "wind_cell_missing_rate_pct", "wind_complete_row_rate_pct"]
        existing_cols = [column for column in display_cols if column in yearly_quality.columns]
        lines.append("")
        lines.append("| year | rows | wind missing % | wind complete rows % |")
        lines.append("| --- | ---: | ---: | ---: |")
        for _, row in yearly_quality[existing_cols].iterrows():
            lines.append(
                f"| {row['period']} | {int(row['rows'])} | "
                f"{row.get('wind_cell_missing_rate_pct', np.nan):.3f} | "
                f"{row.get('wind_complete_row_rate_pct', np.nan):.3f} |"
            )
    lines.append("")

    lines.append("## Candidate Modeling Intervals")
    if candidate_intervals.empty:
        lines.append("- No candidate intervals were generated.")
    else:
        report_cols = [
            "window_days",
            "start_time",
            "end_time_exclusive",
            "rows",
            "quality_score",
            "required_complete_row_rate_pct",
            "wind_complete_row_rate_pct",
            "time_coverage_ratio_pct",
            "expected_delta_ratio_pct",
        ]
        existing_cols = [column for column in report_cols if column in candidate_intervals.columns]
        lines.append(
            "| days | start | end exclusive | rows | score | required complete % | "
            "wind complete % | time coverage % | expected delta % |"
        )
        lines.append("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for _, row in candidate_intervals[existing_cols].head(top_k).iterrows():
            lines.append(
                f"| {int(row['window_days'])} | {row['start_time']} | {row['end_time_exclusive']} | "
                f"{int(row['rows'])} | {row['quality_score']:.3f} | "
                f"{format_optional_float(row.get('required_complete_row_rate_pct', np.nan))} | "
                f"{format_optional_float(row.get('wind_complete_row_rate_pct', np.nan))} | "
                f"{format_optional_float(row.get('time_coverage_ratio_pct', np.nan))} | "
                f"{format_optional_float(row.get('expected_delta_ratio_pct', np.nan))} |"
            )
    lines.append("")
    lines.append(
        "Interpretation note: candidate scores prioritize required PV/weather completeness, "
        "wind-speed availability, timestamp continuity, and seasonal coverage. They are intended "
        "to shortlist intervals for modeling, not to replace research-design judgment."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = resolve_project_path(args.input, PROJECT_ROOT).resolve()
    output_dir = resolve_project_path(args.output_dir, PROJECT_ROOT).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df, time_col, invalid_timestamp_rows = load_raw_csv(input_path, args.time_col)
    if df.empty:
        raise ValueError("No valid timestamp rows remain after loading the raw CSV.")

    expected_delta = infer_expected_delta(df[time_col], args.expected_freq_minutes)
    data_cols = [column for column in df.columns if column != time_col]
    required_cols = args.required_columns if args.required_columns is not None else DEFAULT_REQUIRED_COLUMNS
    required_cols = [column for column in required_cols if column in df.columns]
    wind_speed_cols, wind_related_cols = detect_wind_columns(data_cols)
    wind_quality_cols = [] if args.ignore_wind_for_candidates else (wind_speed_cols if wind_speed_cols else wind_related_cols)

    sampling_summary = summarize_sampling(df[time_col], expected_delta, invalid_timestamp_rows)
    time_diff_distribution = build_time_diff_distribution(df[time_col])
    timestamp_gaps = build_timestamp_gap_table(df[time_col], expected_delta)
    missing_rates = build_missing_rate_table(df, time_col)
    missing_segments, long_missing_segments = build_missing_segment_tables(
        df=df,
        time_col=time_col,
        expected_delta=expected_delta,
        long_gap_steps=args.long_gap_steps,
    )
    missing_segment_summary = build_missing_segment_summary(missing_segments)
    monthly_quality = build_period_quality_table(
        df=df,
        time_col=time_col,
        period="M",
        required_cols=required_cols,
        wind_cols=wind_quality_cols,
    )
    yearly_quality = build_period_quality_table(
        df=df,
        time_col=time_col,
        period="Y",
        required_cols=required_cols,
        wind_cols=wind_quality_cols,
    )
    candidate_intervals = scan_candidate_intervals(
        df=df,
        time_col=time_col,
        expected_delta=expected_delta,
        required_cols=required_cols,
        wind_cols=wind_quality_cols,
        window_days=args.candidate_window_days,
        step_days=args.candidate_step_days,
    )

    wind_missing_segments = missing_segments.loc[
        missing_segments["column"].isin(wind_quality_cols)
    ].copy()
    wind_observed_segments_frames: list[pd.DataFrame] = []
    for column in wind_quality_cols:
        observed_segments = build_boolean_segments(df[time_col], df[column].notna(), expected_delta)
        if observed_segments.empty:
            continue
        observed_segments.insert(0, "column", column)
        wind_observed_segments_frames.append(observed_segments)
    wind_observed_segments = (
        pd.concat(wind_observed_segments_frames, ignore_index=True)
        if wind_observed_segments_frames
        else pd.DataFrame()
    )
    if not wind_observed_segments.empty:
        wind_observed_segments = wind_observed_segments.sort_values(
            ["row_count", "calendar_span_minutes"], ascending=False
        ).reset_index(drop=True)

    missing_rates.to_csv(output_dir / "missing_rates.csv", index=False, encoding="utf-8-sig")
    time_diff_distribution.to_csv(output_dir / "time_diff_distribution.csv", index=False, encoding="utf-8-sig")
    timestamp_gaps.to_csv(output_dir / "timestamp_gaps.csv", index=False, encoding="utf-8-sig")
    missing_segments.to_csv(output_dir / "missing_segments.csv", index=False, encoding="utf-8-sig")
    long_missing_segments.to_csv(output_dir / "long_missing_segments.csv", index=False, encoding="utf-8-sig")
    missing_segment_summary.to_csv(output_dir / "missing_segment_summary.csv", index=False, encoding="utf-8-sig")
    monthly_quality.to_csv(output_dir / "monthly_quality.csv", index=False, encoding="utf-8-sig")
    yearly_quality.to_csv(output_dir / "yearly_quality.csv", index=False, encoding="utf-8-sig")
    candidate_intervals.to_csv(output_dir / "candidate_intervals.csv", index=False, encoding="utf-8-sig")
    wind_missing_segments.to_csv(output_dir / "wind_missing_segments.csv", index=False, encoding="utf-8-sig")
    wind_observed_segments.to_csv(output_dir / "wind_observed_segments.csv", index=False, encoding="utf-8-sig")

    summary = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "time_column": time_col,
        "required_columns_used": required_cols,
        "wind_speed_columns": wind_speed_cols,
        "wind_related_columns": wind_related_cols,
        "wind_quality_columns_used": wind_quality_cols,
        "sampling_summary": sampling_summary,
        "top_missing_rates": missing_rates.head(20).to_dict(orient="records"),
        "top_candidate_intervals": candidate_intervals.head(args.top_k).to_dict(orient="records")
        if not candidate_intervals.empty
        else [],
    }
    write_json(output_dir / "audit_summary.json", summary)
    write_markdown_report(
        path=output_dir / "audit_report.md",
        input_path=input_path,
        time_col=time_col,
        sampling_summary=sampling_summary,
        missing_rates=missing_rates,
        missing_segment_summary=missing_segment_summary,
        wind_speed_cols=wind_speed_cols,
        wind_related_cols=wind_related_cols,
        yearly_quality=yearly_quality,
        candidate_intervals=candidate_intervals,
        top_k=args.top_k,
    )

    print(f"Audit completed. Reports written to: {output_dir}")
    print(f"Detected time column: {time_col}")
    print(f"Expected sampling interval: {sampling_summary['expected_sampling_interval']}")
    print(f"Wind-speed columns: {wind_speed_cols if wind_speed_cols else 'not detected'}")
    if not candidate_intervals.empty:
        best = candidate_intervals.iloc[0]
        print(
            "Best candidate interval: "
            f"{best['start_time']} -> {best['end_time_exclusive']} "
            f"({int(best['window_days'])} days, score={best['quality_score']:.3f})"
        )


if __name__ == "__main__":
    main()
