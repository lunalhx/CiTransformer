from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import pvlib  # type: ignore
except ModuleNotFoundError:
    pvlib = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "91-Site_DKA-M9_B-Phase.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed_selected_2020_2022"
DEFAULT_START = "2020-01-01 00:00:00"
DEFAULT_END = "2022-12-31 23:55:00"
EXPECTED_DELTA = pd.Timedelta(minutes=5)

SITE_LAT = -23.762
SITE_LON = 133.875
SITE_ALT = 546.0
SITE_TZ = "Australia/Darwin"
SITE_UTC_OFFSET_HOURS = 9.5
ELEVATION_THR = 5.0

CORE_FEATURE_COLS = [
    "Active_Pow",
    "Radiation_Global_Tilted",
    "Radiation_Diffuse_Tilted",
    "Weather_T",
    "Weather_R",
    "solar_elevation",
    "sin_time_of_day",
    "cos_time_of_day",
    "sin_day_of_year",
    "cos_day_of_year",
    "day_night_label",
]

NIGHTTIME_ZERO_COLS = [
    "Active_Pow",
    "Radiation_Global_Tilted",
    "Radiation_Diffuse_Tilted",
]

IRRADIANCE_COLS = [
    "Radiation_Global_Tilted",
    "Radiation_Diffuse_Tilted",
]

TARGET_COL = "Active_Pow"

RAW_COL_MAP = {
    "Active_Power": "Active_Pow",
    "Weather_Temperature_Celsius": "Weather_T",
    "Weather_Relative_Humidity": "Weather_R",
}

PV_GAP_RULE_COLS = [
    "Active_Pow",
    "Radiation_Global_Tilted",
    "Radiation_Diffuse_Tilted",
]
WEATHER_GAP_RULE_COLS = [
    "Weather_T",
    "Weather_R",
]
PV_GAP_MAX_STEPS = 12
WEATHER_GAP_MAX_STEPS = 12

CALENDAR_SPLIT_BOUNDS = {
    "train": ("2020-01-01 00:00:00", "2020-12-31 23:55:00"),
    "validation": ("2021-01-01 00:00:00", "2021-06-30 23:55:00"),
    "calibration": ("2021-07-01 00:00:00", "2021-12-31 23:55:00"),
    "test": ("2022-01-01 00:00:00", "2022-12-31 23:55:00"),
}

FEATURE_DECISIONS = [
    {
        "column": "Active_Pow",
        "source_column": "Active_Power",
        "decision": "keep",
        "reason": "Prediction target and autoregressive input feature.",
    },
    {
        "column": "Radiation_Global_Tilted",
        "source_column": "Radiation_Global_Tilted",
        "decision": "keep",
        "reason": "Direct irradiance-related driver for PV generation.",
    },
    {
        "column": "Radiation_Diffuse_Tilted",
        "source_column": "Radiation_Diffuse_Tilted",
        "decision": "keep",
        "reason": "Diffuse irradiance driver for cloudy and low-sun conditions.",
    },
    {
        "column": "Weather_T",
        "source_column": "Weather_Temperature_Celsius",
        "decision": "keep",
        "reason": "Temperature affects PV conversion efficiency.",
    },
    {
        "column": "Weather_R",
        "source_column": "Weather_Relative_Humidity",
        "decision": "keep",
        "reason": "Humidity is a compact weather-state proxy.",
    },
    {
        "column": "solar_elevation",
        "source_column": "derived_from_timestamp",
        "decision": "keep",
        "reason": "Physical solar-geometry feature generated from the site location.",
    },
    {
        "column": "sin_time_of_day",
        "source_column": "derived_from_timestamp",
        "decision": "keep",
        "reason": "Cyclical daily time feature.",
    },
    {
        "column": "cos_time_of_day",
        "source_column": "derived_from_timestamp",
        "decision": "keep",
        "reason": "Cyclical daily time feature.",
    },
    {
        "column": "sin_day_of_year",
        "source_column": "derived_from_timestamp",
        "decision": "keep",
        "reason": "Cyclical annual seasonality feature.",
    },
    {
        "column": "cos_day_of_year",
        "source_column": "derived_from_timestamp",
        "decision": "keep",
        "reason": "Cyclical annual seasonality feature.",
    },
    {
        "column": "day_night_label",
        "source_column": "derived_from_solar_elevation",
        "decision": "keep",
        "reason": f"Physical regime label using solar_elevation >= {ELEVATION_THR} degrees.",
    },
    {
        "column": "Wind_Speed",
        "source_column": "Wind_Speed",
        "decision": "drop",
        "reason": "Wind_Speed is fully missing in the selected 2020-2022 period.",
    },
    {
        "column": "Wind_Direction",
        "source_column": "Wind_Direction",
        "decision": "drop",
        "reason": "Wind direction is not useful without wind speed and would require circular encoding.",
    },
    {
        "column": "Global_Horizontal_Radiation",
        "source_column": "Global_Horizontal_Radiation",
        "decision": "drop",
        "reason": "Tilted-plane irradiance columns are closer to the PV array geometry.",
    },
    {
        "column": "Diffuse_Horizontal_Radiation",
        "source_column": "Diffuse_Horizontal_Radiation",
        "decision": "drop",
        "reason": "Tilted-plane diffuse irradiance is retained instead.",
    },
    {
        "column": "Weather_Daily_Rainfall",
        "source_column": "Weather_Daily_Rainfall",
        "decision": "drop",
        "reason": "Daily rainfall is not part of the original baseline feature protocol.",
    },
    {
        "column": "Current_Phase_Average",
        "source_column": "Current_Phase_Average",
        "decision": "drop",
        "reason": "Highly coupled with the power target and can introduce target-proxy leakage.",
    },
    {
        "column": "Active_Energy_Delivered_Received",
        "source_column": "Active_Energy_Delivered_Received",
        "decision": "drop",
        "reason": "Cumulative energy is not aligned with the instantaneous power target protocol.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess the selected 2020-2022 PV period only. The script stops at "
            "clean feature CSVs and chronological train/validation/calibration/test splits."
        )
    )
    parser.add_argument("--input_path", type=str, default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--time_col", type=str, default="timestamp")
    parser.add_argument("--start", type=str, default=DEFAULT_START)
    parser.add_argument("--end", type=str, default=DEFAULT_END)
    parser.add_argument("--max_power_capacity", type=float, default=11.55)
    parser.add_argument("--max_irradiance", type=float, default=1300.0)
    parser.add_argument(
        "--split_strategy",
        choices=["calendar", "ratio"],
        default="calendar",
        help=(
            "calendar: train=2020, validation=2021-H1, calibration=2021-H2, test=2022. "
            "ratio: original 70/10/10/10 chronological split."
        ),
    )
    return parser.parse_args()


def ensure_output_dirs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "splits").mkdir(parents=True, exist_ok=True)
    (output_dir / "audit").mkdir(parents=True, exist_ok=True)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(column).strip().lstrip("\ufeff") for column in df.columns]
    return df


def ensure_site_timezone(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"df.index must be a DatetimeIndex, got {type(df.index).__name__}.")

    df = df.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize(SITE_TZ)
    elif str(df.index.tz) != SITE_TZ:
        df.index = df.index.tz_convert(SITE_TZ)
    return df


def compute_solar_elevation_noaa(index: pd.DatetimeIndex) -> np.ndarray:
    local_index = index.tz_convert(SITE_TZ) if index.tz is not None else index.tz_localize(SITE_TZ)
    day_of_year = local_index.dayofyear.to_numpy(dtype="float64")
    hour = (
        local_index.hour.to_numpy(dtype="float64")
        + local_index.minute.to_numpy(dtype="float64") / 60.0
        + local_index.second.to_numpy(dtype="float64") / 3600.0
    )

    gamma = 2.0 * np.pi / 365.0 * (day_of_year - 1.0 + (hour - 12.0) / 24.0)
    equation_of_time = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2.0 * gamma)
        - 0.040849 * np.sin(2.0 * gamma)
    )
    declination = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2.0 * gamma)
        + 0.000907 * np.sin(2.0 * gamma)
        - 0.002697 * np.cos(3.0 * gamma)
        + 0.00148 * np.sin(3.0 * gamma)
    )

    time_offset = equation_of_time + 4.0 * SITE_LON - 60.0 * SITE_UTC_OFFSET_HOURS
    true_solar_minutes = (hour * 60.0 + time_offset) % 1440.0
    hour_angle = np.deg2rad(true_solar_minutes / 4.0 - 180.0)
    latitude_rad = np.deg2rad(SITE_LAT)
    sin_elevation = (
        np.sin(latitude_rad) * np.sin(declination)
        + np.cos(latitude_rad) * np.cos(declination) * np.cos(hour_angle)
    )
    return np.rad2deg(np.arcsin(np.clip(sin_elevation, -1.0, 1.0)))


def add_day_night_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    df = ensure_site_timezone(df)
    if pvlib is not None:
        solar_position = pvlib.solarposition.get_solarposition(
            time=df.index,
            latitude=SITE_LAT,
            longitude=SITE_LON,
            altitude=SITE_ALT,
        )
        solar_elevation = solar_position["elevation"].to_numpy(dtype="float64")
        method = "pvlib"
    else:
        solar_elevation = compute_solar_elevation_noaa(df.index)
        method = "noaa_fallback"

    df = df.assign(solar_elevation=solar_elevation)
    df["day_night_label"] = np.where(df["solar_elevation"] < ELEVATION_THR, 0, 1)
    return df, method


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_site_timezone(df)
    local_index = df.index
    minutes_of_day = (
        local_index.hour.to_numpy(dtype="float64") * 60.0
        + local_index.minute.to_numpy(dtype="float64")
    )
    angle_day = 2.0 * np.pi * minutes_of_day / 1440.0

    day_of_year = local_index.dayofyear.to_numpy(dtype="float64")
    days_in_year = np.where(local_index.is_leap_year, 366.0, 365.0)
    year_progress = (day_of_year - 1.0 + minutes_of_day / 1440.0) / days_in_year
    angle_year = 2.0 * np.pi * year_progress

    df["sin_time_of_day"] = np.sin(angle_day).astype("float64")
    df["cos_time_of_day"] = np.cos(angle_day).astype("float64")
    df["sin_day_of_year"] = np.sin(angle_year).astype("float64")
    df["cos_day_of_year"] = np.cos(angle_year).astype("float64")
    return df


def select_core_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in CORE_FEATURE_COLS if column not in df.columns]
    if missing:
        raise KeyError(f"Core feature columns are missing after feature generation: {missing}")
    return df.loc[:, CORE_FEATURE_COLS].copy()


def load_and_filter_raw(
    input_path: Path,
    time_col: str,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input raw CSV not found: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)
    df = normalize_columns(df)
    if time_col not in df.columns:
        raise KeyError(f"Time column `{time_col}` is missing from raw CSV.")

    parsed_time = pd.to_datetime(df[time_col], errors="coerce")
    invalid_timestamp_rows = int(parsed_time.isna().sum())
    df = df.loc[parsed_time.notna()].copy()
    df[time_col] = parsed_time.loc[parsed_time.notna()].to_numpy()

    selected_start = pd.Timestamp(start)
    selected_end = pd.Timestamp(end)
    if selected_start > selected_end:
        raise ValueError("--start must be <= --end.")

    df = df.loc[(df[time_col] >= selected_start) & (df[time_col] <= selected_end)].copy()
    df = df.sort_values(time_col).reset_index(drop=True)

    duplicate_rows = int(df[time_col].duplicated().sum())
    if duplicate_rows:
        raise ValueError(
            f"Selected period contains {duplicate_rows} duplicated timestamps. "
            "Resolve duplicates before preprocessing."
        )

    if df.empty:
        raise ValueError("No rows remain after applying the selected time range.")

    audit = {
        "input_rows_total": int(len(parsed_time)),
        "invalid_timestamp_rows_dropped": invalid_timestamp_rows,
        "selected_start": str(selected_start),
        "selected_end": str(selected_end),
        "selected_rows": int(len(df)),
        "selected_first_timestamp": str(df[time_col].iloc[0]),
        "selected_last_timestamp": str(df[time_col].iloc[-1]),
        "duplicate_timestamp_rows": duplicate_rows,
    }
    return df, audit


def audit_sampling(df: pd.DataFrame, time_col: str) -> tuple[dict[str, Any], pd.DataFrame]:
    timestamps = pd.to_datetime(df[time_col])
    diffs = timestamps.diff().dropna()
    expected_count = int((diffs == EXPECTED_DELTA).sum())
    irregular_mask = diffs != EXPECTED_DELTA
    larger_gap_mask = diffs > EXPECTED_DELTA

    gap_rows: list[dict[str, Any]] = []
    for idx, delta in diffs.loc[irregular_mask].items():
        previous_time = timestamps.iloc[int(idx) - 1]
        next_time = timestamps.iloc[int(idx)]
        missing_steps = 0
        if delta > EXPECTED_DELTA:
            missing_steps = max(int(round(delta / EXPECTED_DELTA)) - 1, 0)
        gap_rows.append(
            {
                "previous_timestamp": str(previous_time),
                "next_timestamp": str(next_time),
                "delta": str(delta),
                "delta_minutes": delta.total_seconds() / 60.0,
                "estimated_missing_steps": int(missing_steps),
            }
        )

    summary = {
        "expected_delta": str(EXPECTED_DELTA),
        "expected_delta_minutes": EXPECTED_DELTA.total_seconds() / 60.0,
        "timestamp_diff_count": int(len(diffs)),
        "expected_delta_count": expected_count,
        "expected_delta_ratio_pct": expected_count / len(diffs) * 100.0 if len(diffs) else None,
        "irregular_delta_count": int(irregular_mask.sum()),
        "larger_than_expected_gap_count": int(larger_gap_mask.sum()),
        "estimated_missing_timestamp_steps": int(
            sum(max(int(round(delta / EXPECTED_DELTA)) - 1, 0) for delta in diffs.loc[larger_gap_mask])
        ),
    }
    return summary, pd.DataFrame(gap_rows)


def build_nan_run_lengths(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    null_mask = series.isna()
    row_run_lengths = pd.Series(0, index=series.index, dtype="int64")
    if not null_mask.any():
        return row_run_lengths, pd.Series(dtype="int64")

    group_id = null_mask.ne(null_mask.shift(fill_value=False)).cumsum()
    segment_lengths = null_mask[null_mask].groupby(group_id[null_mask]).size().astype("int64")
    row_run_lengths.loc[null_mask] = segment_lengths.reindex(group_id[null_mask]).to_numpy()
    return row_run_lengths, segment_lengths


def summarize_missing(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    total_rows = len(df)
    rows: list[dict[str, Any]] = []
    for column in columns:
        null_mask = df[column].isna()
        _, segment_lengths = build_nan_run_lengths(df[column])
        long_threshold = PV_GAP_MAX_STEPS if column in PV_GAP_RULE_COLS else WEATHER_GAP_MAX_STEPS
        rows.append(
            {
                "column": column,
                "missing_count": int(null_mask.sum()),
                "missing_rate_pct": float(null_mask.mean() * 100.0) if total_rows else 0.0,
                "missing_segment_count": int(len(segment_lengths)),
                "long_missing_segment_count": int((segment_lengths > long_threshold).sum())
                if len(segment_lengths)
                else 0,
                "max_consecutive_missing_rows": int(segment_lengths.max()) if len(segment_lengths) else 0,
                "median_consecutive_missing_rows": float(segment_lengths.median()) if len(segment_lengths) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("missing_rate_pct", ascending=False).reset_index(drop=True)


def apply_gap_fill_rule(
    series: pd.Series,
    max_fill_steps: int,
) -> tuple[pd.Series, pd.Series, dict[str, int]]:
    series = series.copy()
    null_mask = series.isna()
    invalid_mask = pd.Series(False, index=series.index, dtype="bool")

    if not null_mask.any():
        return series, invalid_mask, {
            "missing_rows": 0,
            "filled_rows": 0,
            "invalid_rows": 0,
            "fillable_segments": 0,
            "long_segments": 0,
        }

    row_run_lengths, segment_lengths = build_nan_run_lengths(series)
    fill_mask = null_mask & row_run_lengths.le(max_fill_steps)
    long_mask = null_mask & row_run_lengths.gt(max_fill_steps)
    causal_filled = series.ffill()
    series.loc[fill_mask] = causal_filled.loc[fill_mask]

    unresolved_fill_mask = fill_mask & series.isna()
    invalid_mask = invalid_mask | unresolved_fill_mask | long_mask

    stats = {
        "missing_rows": int(null_mask.sum()),
        "filled_rows": int((fill_mask & series.notna()).sum()),
        "invalid_rows": int(invalid_mask.sum()),
        "fillable_segments": int(segment_lengths.le(max_fill_steps).sum()),
        "long_segments": int(segment_lengths.gt(max_fill_steps).sum()),
    }
    return series, invalid_mask, stats


def add_physical_time_features(raw_df: pd.DataFrame, time_col: str) -> tuple[pd.DataFrame, str]:
    df = raw_df.copy()
    df = df.rename(columns=RAW_COL_MAP)
    df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df[time_col]), name="timestamp"))
    df = df.drop(columns=[time_col])
    df, solar_position_method = add_day_night_labels(df)
    df = add_cyclical_time_features(df)
    return df, solar_position_method


def clean_core_features(
    core_df: pd.DataFrame,
    max_power_capacity: float,
    max_irradiance: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    df = core_df.copy()
    df.attrs["max_power_capacity"] = max_power_capacity
    invalid_row_mask = pd.Series(False, index=df.index, dtype="bool")
    cleaning_rows: list[dict[str, Any]] = []

    night_mask = df["day_night_label"] == 0
    for column in NIGHTTIME_ZERO_COLS:
        original_nonzero = int((df.loc[night_mask, column].fillna(0.0) != 0.0).sum())
        original_missing = int(df.loc[night_mask, column].isna().sum())
        df.loc[night_mask, column] = 0.0
        cleaning_rows.append(
            {
                "column": column,
                "rule": "nighttime_zero",
                "affected_rows": int(night_mask.sum()),
                "night_nonzero_values_overwritten": original_nonzero,
                "night_missing_values_overwritten": original_missing,
            }
        )

    for column in PV_GAP_RULE_COLS:
        df[column], column_invalid_mask, stats = apply_gap_fill_rule(df[column], PV_GAP_MAX_STEPS)
        invalid_row_mask = invalid_row_mask | column_invalid_mask
        cleaning_rows.append(
            {
                "column": column,
                "rule": f"causal_ffill_missing_runs_le_{PV_GAP_MAX_STEPS}_steps_else_drop",
                **stats,
            }
        )

    for column in WEATHER_GAP_RULE_COLS:
        df[column], column_invalid_mask, stats = apply_gap_fill_rule(df[column], WEATHER_GAP_MAX_STEPS)
        invalid_row_mask = invalid_row_mask | column_invalid_mask
        cleaning_rows.append(
            {
                "column": column,
                "rule": f"causal_ffill_missing_runs_le_{WEATHER_GAP_MAX_STEPS}_steps_else_drop",
                **stats,
            }
        )

    remaining_nan_mask = df[CORE_FEATURE_COLS].isna().any(axis=1)
    invalid_row_mask = invalid_row_mask | remaining_nan_mask
    rows_before_drop = len(df)
    rows_dropped = int(invalid_row_mask.sum())
    df = df.loc[~invalid_row_mask].copy()

    df["Active_Pow"] = df["Active_Pow"].clip(lower=0.0, upper=max_power_capacity)
    for column in IRRADIANCE_COLS:
        df[column] = df[column].clip(lower=0.0, upper=max_irradiance)

    cleaning_summary = {
        "rows_before_clean_drop": int(rows_before_drop),
        "rows_dropped_after_gap_rules": rows_dropped,
        "rows_after_cleaning": int(len(df)),
        "remaining_missing_values": int(df[CORE_FEATURE_COLS].isna().sum().sum()),
        "nighttime_rows": int(night_mask.sum()),
        "power_clip_range": [0.0, max_power_capacity],
        "irradiance_clip_range": [0.0, max_irradiance],
    }
    return df, pd.DataFrame(cleaning_rows), cleaning_summary


def local_timestamp(timestamp: str) -> pd.Timestamp:
    return pd.Timestamp(timestamp, tz=SITE_TZ)


def split_calendar(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    splits: dict[str, pd.DataFrame] = {}
    for split_name, (start, end) in CALENDAR_SPLIT_BOUNDS.items():
        start_ts = local_timestamp(start)
        end_ts = local_timestamp(end)
        splits[split_name] = df.loc[(df.index >= start_ts) & (df.index <= end_ts)].copy()
    return splits


def split_ratio(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    total_rows = len(df)
    train_end = int(np.floor(total_rows * 0.70))
    validation_end = int(np.floor(total_rows * 0.80))
    calibration_end = int(np.floor(total_rows * 0.90))
    return {
        "train": df.iloc[:train_end].copy(),
        "validation": df.iloc[train_end:validation_end].copy(),
        "calibration": df.iloc[validation_end:calibration_end].copy(),
        "test": df.iloc[calibration_end:].copy(),
    }


def build_split_summary(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    total_rows = sum(len(split_df) for split_df in splits.values())
    rows: list[dict[str, Any]] = []
    for split_name, split_df in splits.items():
        rows.append(
            {
                "split": split_name,
                "row_count": int(len(split_df)),
                "actual_ratio_pct": len(split_df) / total_rows * 100.0 if total_rows else 0.0,
                "start_time": str(split_df.index[0]) if len(split_df) else None,
                "end_time": str(split_df.index[-1]) if len(split_df) else None,
                "missing_values": int(split_df.isna().sum().sum()) if len(split_df) else 0,
            }
        )
    return pd.DataFrame(rows)


def save_splits(splits: dict[str, pd.DataFrame], output_dir: Path) -> pd.DataFrame:
    split_dir = output_dir / "splits"
    for split_name, split_df in splits.items():
        split_df.to_csv(split_dir / f"{split_name}.csv", encoding="utf-8-sig")
    split_summary = build_split_summary(splits)
    split_summary.to_csv(split_dir / "split_summary.csv", index=False, encoding="utf-8-sig")
    return split_summary


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
        return None if np.isnan(value) else float(value)
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if pd.isna(value):
        return None
    return value


def write_markdown_report(
    output_path: Path,
    summary: dict[str, Any],
    raw_missing_summary: pd.DataFrame,
    core_missing_before_clean: pd.DataFrame,
    cleaning_stats: pd.DataFrame,
    split_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Preprocessing Report: Selected PV Period 2020-2022",
        "",
        "## Scope",
        f"- Input: `{summary['input_path']}`",
        f"- Output directory: `{summary['output_dir']}`",
        f"- Selected raw time range: {summary['selection']['selected_start']} to {summary['selection']['selected_end']}",
        f"- Selected rows before cleaning: {summary['selection']['selected_rows']}",
        f"- Final clean rows: {summary['cleaning']['rows_after_cleaning']}",
        f"- Split strategy: {summary['split_strategy']}",
        "",
        "## Time Audit",
        f"- Expected sampling interval: {summary['sampling']['expected_delta']}",
        f"- Expected-delta ratio: {summary['sampling']['expected_delta_ratio_pct']:.4f}%",
        f"- Larger-than-expected gaps: {summary['sampling']['larger_than_expected_gap_count']}",
        f"- Estimated missing timestamp steps: {summary['sampling']['estimated_missing_timestamp_steps']}",
        "",
        "## Raw Missing Summary",
        "| column | missing % | missing rows | max run | long runs |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in raw_missing_summary.iterrows():
        lines.append(
            f"| {row['column']} | {row['missing_rate_pct']:.3f} | {int(row['missing_count'])} | "
            f"{int(row['max_consecutive_missing_rows'])} | {int(row['long_missing_segment_count'])} |"
        )

    lines.extend(
        [
            "",
            "## Core Missing Summary Before Cleaning",
            "| column | missing % | missing rows | max run | long runs |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in core_missing_before_clean.iterrows():
        lines.append(
            f"| {row['column']} | {row['missing_rate_pct']:.3f} | {int(row['missing_count'])} | "
            f"{int(row['max_consecutive_missing_rows'])} | {int(row['long_missing_segment_count'])} |"
        )

    lines.extend(
        [
            "",
            "## Cleaning Rules",
            "- Nighttime Active_Pow and tilted irradiance columns are set to zero when day_night_label = 0.",
            "- Missing runs up to 12 consecutive 5-minute steps are filled by causal forward fill.",
            "- Longer missing runs are not imputed; affected rows are dropped and become temporal gaps for downstream segment-aware windowing.",
            "- Active_Pow is clipped to the physical range and tilted irradiance columns are clipped to a plausible upper bound.",
            "",
            "## Cleaning Stats",
            "| column | rule | filled rows | invalid rows | long segments |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for _, row in cleaning_stats.iterrows():
        lines.append(
            f"| {row.get('column', '')} | {row.get('rule', '')} | "
            f"{int(row.get('filled_rows', 0) if not pd.isna(row.get('filled_rows', 0)) else 0)} | "
            f"{int(row.get('invalid_rows', 0) if not pd.isna(row.get('invalid_rows', 0)) else 0)} | "
            f"{int(row.get('long_segments', 0) if not pd.isna(row.get('long_segments', 0)) else 0)} |"
        )

    lines.extend(
        [
            "",
            "## Split Summary",
            "| split | rows | ratio % | start | end |",
            "| --- | ---: | ---: | --- | --- |",
        ]
    )
    for _, row in split_summary.iterrows():
        lines.append(
            f"| {row['split']} | {int(row['row_count'])} | {row['actual_ratio_pct']:.2f} | "
            f"{row['start_time']} | {row['end_time']} |"
        )

    lines.extend(
        [
            "",
            "## Feature Decision",
            "- Final feature columns follow the existing no-wind baseline protocol.",
            "- Wind_Speed is excluded because it is fully missing in the selected 2020-2022 period.",
            "- Scaling is intentionally not performed here; downstream training code should fit scalers on train only.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_output_dirs(output_dir)

    raw_selected, selection_audit = load_and_filter_raw(
        input_path=input_path,
        time_col=args.time_col,
        start=args.start,
        end=args.end,
    )
    sampling_summary, timestamp_gaps = audit_sampling(raw_selected, args.time_col)

    raw_missing_columns = [
        "Active_Power",
        "Radiation_Global_Tilted",
        "Radiation_Diffuse_Tilted",
        "Weather_Temperature_Celsius",
        "Weather_Relative_Humidity",
        "Wind_Speed",
    ]
    raw_missing_columns = [column for column in raw_missing_columns if column in raw_selected.columns]
    raw_missing_summary = summarize_missing(raw_selected, raw_missing_columns)

    labeled_df, solar_position_method = add_physical_time_features(raw_selected, args.time_col)
    core_before_clean = select_core_features(labeled_df)
    core_missing_before_clean = summarize_missing(core_before_clean, CORE_FEATURE_COLS)
    clean_df, cleaning_stats, cleaning_summary = clean_core_features(
        core_df=core_before_clean,
        max_power_capacity=args.max_power_capacity,
        max_irradiance=args.max_irradiance,
    )

    if args.split_strategy == "calendar":
        splits = split_calendar(clean_df)
    else:
        splits = split_ratio(clean_df)
    split_summary = save_splits(splits, output_dir)

    raw_selected.to_csv(output_dir / "selected_raw_2020_2022.csv", index=False, encoding="utf-8-sig")
    labeled_df.to_csv(output_dir / "nighttime_labeled_2020_2022.csv", encoding="utf-8-sig")
    core_before_clean.to_csv(output_dir / "core_features_before_clean.csv", encoding="utf-8-sig")
    clean_df.to_csv(output_dir / "core_features_clean.csv", encoding="utf-8-sig")

    raw_missing_summary.to_csv(output_dir / "audit" / "raw_missing_summary.csv", index=False, encoding="utf-8-sig")
    core_missing_before_clean.to_csv(
        output_dir / "audit" / "core_missing_before_clean.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cleaning_stats.to_csv(output_dir / "audit" / "cleaning_stats.csv", index=False, encoding="utf-8-sig")
    timestamp_gaps.to_csv(output_dir / "audit" / "timestamp_gaps.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(FEATURE_DECISIONS).to_csv(
        output_dir / "audit" / "feature_decisions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "site": {
            "latitude": SITE_LAT,
            "longitude": SITE_LON,
            "altitude_m": SITE_ALT,
            "timezone": SITE_TZ,
            "day_night_elevation_threshold_deg": ELEVATION_THR,
            "solar_position_method": solar_position_method,
        },
        "selection": selection_audit,
        "sampling": sampling_summary,
        "feature_columns": list(CORE_FEATURE_COLS),
        "target_column": TARGET_COL,
        "split_strategy": args.split_strategy,
        "split_bounds": CALENDAR_SPLIT_BOUNDS if args.split_strategy == "calendar" else "70/10/10/10",
        "cleaning": cleaning_summary,
        "split_summary": split_summary.to_dict(orient="records"),
        "outputs": {
            "selected_raw": str(output_dir / "selected_raw_2020_2022.csv"),
            "nighttime_labeled": str(output_dir / "nighttime_labeled_2020_2022.csv"),
            "core_features_before_clean": str(output_dir / "core_features_before_clean.csv"),
            "core_features_clean": str(output_dir / "core_features_clean.csv"),
            "splits": str(output_dir / "splits"),
            "audit": str(output_dir / "audit"),
        },
    }

    (output_dir / "preprocessing_summary.json").write_text(
        json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown_report(
        output_path=output_dir / "preprocessing_report.md",
        summary=summary,
        raw_missing_summary=raw_missing_summary,
        core_missing_before_clean=core_missing_before_clean,
        cleaning_stats=cleaning_stats,
        split_summary=split_summary,
    )

    print(f"Preprocessing completed: {output_dir}")
    print(f"Selected rows: {selection_audit['selected_rows']}")
    print(f"Clean rows: {cleaning_summary['rows_after_cleaning']}")
    print(split_summary.to_string(index=False))


if __name__ == "__main__":
    main()
