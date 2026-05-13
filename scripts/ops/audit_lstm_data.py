from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.datasets import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_TARGET_COLUMN,
    build_segment_boundaries,
    fit_split_scalers,
    infer_expected_timedelta,
    load_split_dataframe,
    resolve_split_dir,
)
from utils.project_config import load_project_config

PROJECT_CONFIG = load_project_config()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit LSTM split, segment, and window statistics.")
    parser.add_argument("--data_dir", type=str, default=str(PROJECT_CONFIG.get_path("paths.data_dir", DEFAULT_DATA_DIR)))
    parser.add_argument("--time_col", type=str, default=None)
    parser.add_argument("--target_col", type=str, default=DEFAULT_TARGET_COLUMN)
    parser.add_argument("--feature_cols", nargs="+", default=DEFAULT_FEATURE_COLUMNS)
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_lens", nargs="+", type=int, default=[1, 12, 24, 48])
    parser.add_argument("--sampling_freq_minutes", type=int, default=None)
    parser.add_argument("--zero_eps", type=float, default=1e-6)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a readable text report.")
    return parser.parse_args()


def scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def describe_numeric(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values)
    if values.size == 0:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "max": None,
            "mean": None,
            "std": None,
        }

    return {
        "count": int(values.size),
        "min": scalar(np.min(values)),
        "p25": scalar(np.percentile(values, 25)),
        "median": scalar(np.percentile(values, 50)),
        "p75": scalar(np.percentile(values, 75)),
        "p90": scalar(np.percentile(values, 90)),
        "p95": scalar(np.percentile(values, 95)),
        "max": scalar(np.max(values)),
        "mean": scalar(np.mean(values)),
        "std": scalar(np.std(values, ddof=1)) if values.size > 1 else 0.0,
    }


def count_target_day_points(
    day_night: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    seq_len: int,
    pred_len: int,
) -> tuple[int, int, int, int]:
    total_points = 0
    daytime_points = 0
    total_windows = 0
    windows_with_daytime = 0

    for start, end in zip(starts, ends, strict=True):
        segment_length = int(end - start)
        window_count = segment_length - seq_len - pred_len + 1
        if window_count <= 0:
            continue

        total_windows += window_count
        for offset in range(window_count):
            target_start = int(start + offset + seq_len)
            target_end = target_start + pred_len
            target_day = day_night[target_start:target_end]
            daytime_count = int(np.sum(target_day == 1))
            daytime_points += daytime_count
            total_points += pred_len
            if daytime_count > 0:
                windows_with_daytime += 1

    return total_windows, total_points, daytime_points, windows_with_daytime


def summarize_split(
    name: str,
    df: pd.DataFrame,
    expected_delta: pd.Timedelta,
    seq_len: int,
    pred_lens: list[int],
    target_col: str,
    zero_eps: float,
    train_target_mean: float,
    train_target_std: float,
) -> dict[str, Any]:
    starts, ends = build_segment_boundaries(df.index, expected_delta)
    segment_lengths = ends - starts
    active_pow = df[target_col].to_numpy(dtype=np.float64)
    day_night = (
        df["day_night_label"].to_numpy(dtype=np.int64)
        if "day_night_label" in df.columns
        else np.ones(len(df), dtype=np.int64)
    )
    scaled_target = (active_pow - train_target_mean) / train_target_std

    pred_len_stats: dict[str, Any] = {}
    for pred_len in pred_lens:
        min_segment_length = seq_len + pred_len
        usable_mask = segment_lengths >= min_segment_length
        valid_windows = np.maximum(segment_lengths - min_segment_length + 1, 0)
        total_windows, total_points, daytime_points, windows_with_daytime = count_target_day_points(
            day_night=day_night,
            starts=starts,
            ends=ends,
            seq_len=seq_len,
            pred_len=pred_len,
        )
        if int(valid_windows.sum()) != total_windows:
            raise RuntimeError(f"Window count mismatch for split={name}, pred_len={pred_len}.")

        pred_len_stats[str(pred_len)] = {
            "min_segment_length": int(min_segment_length),
            "usable_segments": int(np.sum(usable_mask)),
            "dropped_segments": int(len(segment_lengths) - np.sum(usable_mask)),
            "samples": int(total_windows),
            "target_points": int(total_points),
            "target_daytime_points": int(daytime_points),
            "target_nighttime_points": int(total_points - daytime_points),
            "target_daytime_ratio": scalar(daytime_points / total_points) if total_points else None,
            "target_nighttime_ratio": scalar(1.0 - daytime_points / total_points) if total_points else None,
            "windows_with_any_daytime_ratio": scalar(windows_with_daytime / total_windows) if total_windows else None,
        }

    return {
        "split": name,
        "time_start": str(df.index.min()) if len(df) else None,
        "time_end": str(df.index.max()) if len(df) else None,
        "rows": int(len(df)),
        "missing_values": int(df.isna().sum().sum()),
        "expected_delta_minutes": scalar(expected_delta / pd.Timedelta(minutes=1)),
        "segment_count": int(len(segment_lengths)),
        "segment_lengths": describe_numeric(segment_lengths),
        "shortest_segments": [int(value) for value in np.sort(segment_lengths)[:8]],
        "longest_segments": [int(value) for value in np.sort(segment_lengths)[-8:][::-1]],
        "row_daytime_ratio": scalar(np.mean(day_night == 1)) if len(day_night) else None,
        "row_nighttime_ratio": scalar(np.mean(day_night == 0)) if len(day_night) else None,
        "active_pow": {
            **describe_numeric(active_pow),
            "zero_ratio": scalar(np.mean(np.abs(active_pow) <= zero_eps)) if active_pow.size else None,
            "scaled_mean_using_train": scalar(np.mean(scaled_target)) if scaled_target.size else None,
            "scaled_std_using_train": scalar(np.std(scaled_target, ddof=1)) if scaled_target.size > 1 else None,
        },
        "pred_lens": pred_len_stats,
    }


def print_report(report: dict[str, Any]) -> None:
    print("LSTM data audit")
    print("=" * 80)
    print(f"data_dir: {report['data_dir']}")
    print(f"split_dir: {report['split_dir']}")
    print(f"seq_len: {report['seq_len']}")
    print(f"pred_lens: {', '.join(map(str, report['pred_lens']))}")
    print(f"expected_delta_minutes: {report['expected_delta_minutes']}")
    print(f"feature_cols: {', '.join(report['feature_cols'])}")
    print(f"target_col: {report['target_col']}")
    print()

    for split_name, split in report["splits"].items():
        print(f"[{split_name}] {split['time_start']} -> {split['time_end']}")
        print(
            "  rows={rows} missing={missing_values} segments={segment_count} "
            "row_day={row_daytime_ratio:.4f} row_night={row_nighttime_ratio:.4f}".format(**split)
        )
        seg = split["segment_lengths"]
        print(
            "  segment_len min={min} p25={p25:.1f} median={median:.1f} "
            "p75={p75:.1f} max={max}".format(**seg)
        )
        power = split["active_pow"]
        print(
            "  Active_Pow min={min:.6f} max={max:.6f} mean={mean:.6f} std={std:.6f} "
            "zero_ratio={zero_ratio:.4f} scaled_mean={scaled_mean_using_train:.4f} "
            "scaled_std={scaled_std_using_train:.4f}".format(**power)
        )
        for pred_len, pred in split["pred_lens"].items():
            print(
                "  pred_len={pred_len:>2s} samples={samples} usable_segments={usable_segments} "
                "dropped_segments={dropped_segments} target_day={target_daytime_ratio:.4f} "
                "target_night={target_nighttime_ratio:.4f}".format(pred_len=pred_len, **pred)
            )
        print()


def main() -> None:
    args = parse_args()
    split_dir = resolve_split_dir(args.data_dir)
    frames = {
        split: load_split_dataframe(split_dir / f"{split}.csv", time_col=args.time_col)
        for split in ("train", "validation", "calibration", "test")
    }
    expected_delta = (
        pd.Timedelta(minutes=args.sampling_freq_minutes)
        if args.sampling_freq_minutes is not None
        else infer_expected_timedelta(frames["train"].index)
    )
    scalers = fit_split_scalers(frames["train"], list(args.feature_cols), args.target_col)
    train_target_mean = float(scalers.target_scaler.mean_[0])
    train_target_std = float(scalers.target_scaler.scale_[0])

    report = {
        "data_dir": args.data_dir,
        "split_dir": str(split_dir),
        "seq_len": args.seq_len,
        "pred_lens": args.pred_lens,
        "expected_delta_minutes": scalar(expected_delta / pd.Timedelta(minutes=1)),
        "feature_cols": list(args.feature_cols),
        "target_col": args.target_col,
        "train_target_scaler": {
            "mean": train_target_mean,
            "scale": train_target_std,
        },
        "splits": {
            split_name: summarize_split(
                name=split_name,
                df=df,
                expected_delta=expected_delta,
                seq_len=args.seq_len,
                pred_lens=args.pred_lens,
                target_col=args.target_col,
                zero_eps=args.zero_eps,
                train_target_mean=train_target_mean,
                train_target_std=train_target_std,
            )
            for split_name, df in frames.items()
        },
    }

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
