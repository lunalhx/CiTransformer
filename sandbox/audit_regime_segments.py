from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_INPUT_CSV = (
    "results/d1_long_no_wind_2015_2022/regimes/"
    "gmm_hmm_daytime_k7/train_with_regime.csv"
)
DEFAULT_OUTPUT_DIR = (
    "results/d1_long_no_wind_2015_2022/regimes/"
    "gmm_hmm_daytime_k7/segment_audit"
)
TIME_COLUMN_CANDIDATES = ("timestamp", "datetime", "date", "time", "ds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit regime-continuous segment lengths and PCMCI sample feasibility "
            "for regime-labeled PV split CSV files."
        )
    )
    parser.add_argument("--input_csv", default=DEFAULT_INPUT_CSV, help="Regime-labeled CSV to audit.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Directory for audit CSV/Markdown outputs.")
    parser.add_argument("--time_col", default=None, help="Optional timestamp column name.")
    parser.add_argument("--regime_col", default="regime", help="Regime label column.")
    parser.add_argument("--day_col", default="day_night_label", help="Day/night label column.")
    parser.add_argument(
        "--include_night",
        action="store_true",
        help="Include regime 0 in the main summary. By default only daytime regimes are summarized.",
    )
    parser.add_argument(
        "--tau_max",
        type=int,
        default=12,
        help="Maximum PCMCI lag in rows. Default 12 for 60 minutes at 5-minute resolution.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=int,
        default=[13, 24, 48, 96],
        help="Segment length thresholds, in rows, to count per regime.",
    )
    parser.add_argument(
        "--expected_freq_minutes",
        type=int,
        default=None,
        help="Expected sampling interval. If omitted, inferred from timestamp mode.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def is_parseable_datetime(series: pd.Series, min_success_ratio: float = 0.95) -> bool:
    if series.empty:
        return False
    parsed = pd.to_datetime(series, errors="coerce")
    return float(parsed.notna().mean()) >= min_success_ratio


def detect_time_column(df: pd.DataFrame, preferred: str | None = None) -> str:
    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    candidates.extend([candidate for candidate in TIME_COLUMN_CANDIDATES if candidate not in candidates])

    for column in candidates:
        if column in df.columns and is_parseable_datetime(df[column]):
            return column
    for column in df.columns:
        if column not in candidates and is_parseable_datetime(df[column]):
            return column
    raise ValueError("Failed to detect a timestamp column. Pass --time_col explicitly.")


def load_dataframe(path: str | Path, time_col: str | None) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(path)
    detected_time_col = detect_time_column(df, preferred=time_col)
    df[detected_time_col] = pd.to_datetime(df[detected_time_col], errors="raise")
    df = df.sort_values(detected_time_col).reset_index(drop=True)
    if df[detected_time_col].duplicated().any():
        duplicate_count = int(df[detected_time_col].duplicated().sum())
        raise ValueError(f"{path} contains {duplicate_count} duplicated timestamps.")
    return df, detected_time_col


def infer_expected_delta(timestamp: pd.Series, expected_freq_minutes: int | None) -> pd.Timedelta:
    if expected_freq_minutes is not None:
        return pd.Timedelta(minutes=expected_freq_minutes)

    diffs = timestamp.diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        raise ValueError("Cannot infer sampling frequency from timestamps.")
    mode = diffs.mode()
    return mode.iloc[0] if not mode.empty else diffs.iloc[0]


def mark_true_segments(df: pd.DataFrame, time_col: str, expected_delta: pd.Timedelta) -> pd.Series:
    timestamp = df[time_col]
    is_new = timestamp.diff().isna() | (timestamp.diff() != expected_delta)
    return is_new.cumsum().astype(int)


def build_regime_segments(
    df: pd.DataFrame,
    time_col: str,
    regime_col: str,
    true_segment_col: str,
) -> pd.DataFrame:
    regime_change = df[regime_col].ne(df[regime_col].shift())
    true_segment_change = df[true_segment_col].ne(df[true_segment_col].shift())
    regime_segment_id = (regime_change | true_segment_change).cumsum()

    working = df[[time_col, regime_col, true_segment_col]].copy()
    working["regime_segment_id"] = regime_segment_id
    grouped = working.groupby("regime_segment_id", sort=True)
    segments = grouped.agg(
        regime=(regime_col, "first"),
        true_segment_id=(true_segment_col, "first"),
        start_time=(time_col, "first"),
        end_time=(time_col, "last"),
        length_rows=(time_col, "size"),
    ).reset_index(drop=True)
    segments["duration_minutes"] = (
        (segments["end_time"] - segments["start_time"]).dt.total_seconds() / 60.0
    )
    return segments


def summarize_segments(
    segments: pd.DataFrame,
    regimes: list[int],
    tau_max: int,
    thresholds: list[int],
    target_conditioned_counts: pd.Series,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for regime in regimes:
        regime_segments = segments[segments["regime"] == regime].copy()
        lengths = regime_segments["length_rows"].to_numpy(dtype=float)
        total_rows = int(lengths.sum()) if len(lengths) else 0
        same_regime_usable_targets = int(np.maximum(lengths - tau_max, 0).sum()) if len(lengths) else 0
        target_conditioned_usable_targets = int(target_conditioned_counts.get(regime, 0))

        record: dict[str, Any] = {
            "regime": int(regime),
            "n_segments": int(len(regime_segments)),
            "total_rows": total_rows,
            "same_regime_usable_targets_tau_max": same_regime_usable_targets,
            "target_conditioned_usable_targets_tau_max": target_conditioned_usable_targets,
            "same_regime_retention_ratio": (
                float(same_regime_usable_targets / total_rows) if total_rows else 0.0
            ),
            "target_conditioned_to_same_regime_ratio": (
                float(target_conditioned_usable_targets / same_regime_usable_targets)
                if same_regime_usable_targets
                else np.nan
            ),
            "min_length": int(lengths.min()) if len(lengths) else 0,
            "p25_length": float(np.percentile(lengths, 25)) if len(lengths) else 0.0,
            "median_length": float(np.median(lengths)) if len(lengths) else 0.0,
            "mean_length": float(lengths.mean()) if len(lengths) else 0.0,
            "p75_length": float(np.percentile(lengths, 75)) if len(lengths) else 0.0,
            "p90_length": float(np.percentile(lengths, 90)) if len(lengths) else 0.0,
            "max_length": int(lengths.max()) if len(lengths) else 0,
        }
        for threshold in thresholds:
            record[f"segments_ge_{threshold}"] = int((lengths >= threshold).sum())
            record[f"rows_in_segments_ge_{threshold}"] = int(lengths[lengths >= threshold].sum())
        records.append(record)
    return pd.DataFrame(records)


def compute_target_conditioned_usable_counts(
    df: pd.DataFrame,
    regime_col: str,
    true_segment_col: str,
    tau_max: int,
) -> pd.Series:
    position_in_true_segment = df.groupby(true_segment_col).cumcount()
    usable = position_in_true_segment >= tau_max
    return df.loc[usable, regime_col].value_counts().sort_index()


def write_markdown_report(
    output_path: Path,
    input_csv: Path,
    expected_delta: pd.Timedelta,
    tau_max: int,
    summary: pd.DataFrame,
) -> None:
    lines = [
        "# Regime Segment Feasibility Audit",
        "",
        f"- Input CSV: `{input_csv}`",
        f"- Expected sampling interval: `{expected_delta}`",
        f"- tau_max rows: `{tau_max}`",
        "",
        "## Summary",
        "",
        "| Regime | Segments | Rows | Median len | Mean len | Max len | Same-regime usable | Target-conditioned usable | Retention |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {int(row['regime'])} | {int(row['n_segments'])} | {int(row['total_rows'])} | "
            f"{row['median_length']:.1f} | {row['mean_length']:.1f} | {int(row['max_length'])} | "
            f"{int(row['same_regime_usable_targets_tau_max'])} | "
            f"{int(row['target_conditioned_usable_targets_tau_max'])} | "
            f"{row['same_regime_retention_ratio']:.2%} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- `same-regime usable` counts target rows that have `tau_max` previous rows inside the same uninterrupted regime segment.",
            "- `target-conditioned usable` counts target rows whose previous `tau_max` rows are continuous in the original time axis, regardless of whether previous rows have the same regime.",
            "- If same-regime usable counts are much smaller than target-conditioned counts, direct per-regime segment PCMCI will discard many samples.",
            "- In that case, prefer target-regime-conditioned PCMCI: condition on `regime(t)=r` while taking lagged variables from the original continuous timeline.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, time_col = load_dataframe(input_csv, time_col=args.time_col)
    for column in [args.regime_col, args.day_col]:
        if column not in df.columns:
            raise KeyError(f"Missing required column: {column}")

    expected_delta = infer_expected_delta(df[time_col], args.expected_freq_minutes)
    df["true_segment_id"] = mark_true_segments(df, time_col, expected_delta)

    segments = build_regime_segments(
        df,
        time_col=time_col,
        regime_col=args.regime_col,
        true_segment_col="true_segment_id",
    )

    regimes = sorted(df[args.regime_col].dropna().astype(int).unique().tolist())
    if not args.include_night:
        regimes = [regime for regime in regimes if regime != 0]

    target_conditioned_counts = compute_target_conditioned_usable_counts(
        df,
        regime_col=args.regime_col,
        true_segment_col="true_segment_id",
        tau_max=args.tau_max,
    )
    summary = summarize_segments(
        segments=segments,
        regimes=regimes,
        tau_max=args.tau_max,
        thresholds=sorted(set(args.thresholds)),
        target_conditioned_counts=target_conditioned_counts,
    )

    segments.to_csv(output_dir / "regime_segment_details.csv", index=False)
    summary.to_csv(output_dir / "regime_segment_summary.csv", index=False)
    write_markdown_report(
        output_path=output_dir / "regime_segment_audit.md",
        input_csv=input_csv,
        expected_delta=expected_delta,
        tau_max=args.tau_max,
        summary=summary,
    )

    log(f"Wrote segment audit to {output_dir}")
    log(summary.to_string(index=False))


if __name__ == "__main__":
    main()
