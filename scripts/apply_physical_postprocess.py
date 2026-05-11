from __future__ import annotations

import argparse
import json
import math
import os
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
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_CONFIG.get_path("paths.matplotlib_cache")))

DEFAULT_MODELS = [
    "persistence",
    "lstm",
    "itransformer",
    "itransformer_tuned",
    "itransformer_global_pcmci_11vars",
]
DEFAULT_PRED_LENS = [1, 12, 24, 48]
PRED_COLUMN_PATTERN = re.compile(r"^y_pred_t\+(\d+)$")


def to_float(value: float | np.floating | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def to_percentage(numerator: float | np.floating, denominator: float | np.floating, eps: float) -> float | None:
    denominator = float(denominator)
    if abs(denominator) <= eps:
        return None
    return to_float(float(numerator) / denominator * 100.0)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> dict[str, Any]:
    if len(y_true) == 0:
        return {
            "count": 0,
            "mae": None,
            "mse": None,
            "rmse": None,
            "mbe": None,
            "median_ae": None,
            "p95_ae": None,
            "max_ae": None,
            "smape": None,
            "mape_nonzero": None,
            "wape": None,
            "nmae_by_mean": None,
            "nrmse_by_mean": None,
            "nmbe_by_mean": None,
            "nmae_by_max": None,
            "nrmse_by_max": None,
            "nmbe_by_max": None,
            "r2": None,
            "pearson_r": None,
            "mean_true": None,
            "mean_pred": None,
            "mean_abs_true": None,
            "max_abs_true": None,
            "sum_abs_true": None,
            "nonzero_target_count": 0,
        }

    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    error = y_pred - y_true
    abs_error = np.abs(error)
    squared_error = np.square(error)
    mae = np.mean(abs_error)
    mse = np.mean(squared_error)
    rmse = np.sqrt(mse)
    smape = np.mean(2.0 * np.abs(error) / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100.0
    mbe = np.mean(error)
    median_ae = np.median(abs_error)
    p95_ae = np.percentile(abs_error, 95)
    max_ae = np.max(abs_error)
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    mean_abs_true = np.mean(np.abs(y_true))
    max_abs_true = np.max(np.abs(y_true))
    sum_abs_true = np.sum(np.abs(y_true))

    nonzero_mask = np.abs(y_true) > eps
    if nonzero_mask.any():
        mape = np.mean(np.abs(error[nonzero_mask] / y_true[nonzero_mask])) * 100.0
    else:
        mape = None

    ss_res = np.sum(squared_error)
    ss_tot = np.sum(np.square(y_true - mean_true))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > eps else None

    if len(y_true) > 1 and np.std(y_true) > eps and np.std(y_pred) > eps:
        pearson_r = np.corrcoef(y_true, y_pred)[0, 1]
    else:
        pearson_r = None

    return {
        "count": int(len(y_true)),
        "mae": to_float(mae),
        "mse": to_float(mse),
        "rmse": to_float(rmse),
        "mbe": to_float(mbe),
        "median_ae": to_float(median_ae),
        "p95_ae": to_float(p95_ae),
        "max_ae": to_float(max_ae),
        "smape": to_float(smape),
        "mape_nonzero": to_float(mape),
        "wape": to_percentage(np.sum(abs_error), sum_abs_true, eps),
        "nmae_by_mean": to_percentage(mae, mean_abs_true, eps),
        "nrmse_by_mean": to_percentage(rmse, mean_abs_true, eps),
        "nmbe_by_mean": to_percentage(mbe, mean_abs_true, eps),
        "nmae_by_max": to_percentage(mae, max_abs_true, eps),
        "nrmse_by_max": to_percentage(rmse, max_abs_true, eps),
        "nmbe_by_max": to_percentage(mbe, max_abs_true, eps),
        "r2": to_float(r2),
        "pearson_r": to_float(pearson_r),
        "mean_true": to_float(mean_true),
        "mean_pred": to_float(mean_pred),
        "mean_abs_true": to_float(mean_abs_true),
        "max_abs_true": to_float(max_abs_true),
        "sum_abs_true": to_float(sum_abs_true),
        "nonzero_target_count": int(nonzero_mask.sum()),
    }


def evaluate_prediction_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    day_night_label: np.ndarray,
) -> dict[str, Any]:
    flat_true = y_true.reshape(-1)
    flat_pred = y_pred.reshape(-1)
    flat_day = day_night_label.reshape(-1)

    all_metrics = compute_regression_metrics(flat_true, flat_pred)
    daytime_metrics = compute_regression_metrics(flat_true[flat_day == 1], flat_pred[flat_day == 1])

    per_horizon_all = {
        f"t+{horizon + 1}": compute_regression_metrics(y_true[:, horizon], y_pred[:, horizon])
        for horizon in range(y_true.shape[1])
    }
    per_horizon_daytime = {
        f"t+{horizon + 1}": compute_regression_metrics(
            y_true[:, horizon][day_night_label[:, horizon] == 1],
            y_pred[:, horizon][day_night_label[:, horizon] == 1],
        )
        for horizon in range(y_true.shape[1])
    }

    return {
        "all_timestamps": all_metrics,
        "daytime_only": daytime_metrics,
        "per_horizon_all": per_horizon_all,
        "per_horizon_daytime": per_horizon_daytime,
        "mape_note": "MAPE excludes targets with abs(y_true) <= 1e-6 to avoid night-time zero division.",
        "normalization_note": (
            "nmae_by_mean/nrmse_by_mean/nmbe_by_mean use mean(abs(y_true)) as denominator. "
            "nmae_by_max/nrmse_by_max/nmbe_by_max use max(abs(y_true)) in the evaluated split/scope "
            "as a PV capacity proxy."
        ),
    }


def save_prediction_plot(
    predictions_df: pd.DataFrame,
    output_path: Path,
    split_name: str = "test",
    model_label: str = "Model",
    max_points: int = 4000,
) -> None:
    import matplotlib.pyplot as plt

    plot_df = predictions_df.loc[:, ["target_start_time", "y_true_t+1", "y_pred_t+1"]].copy()
    plot_df["target_start_time"] = pd.to_datetime(plot_df["target_start_time"])
    plot_df = plot_df.sort_values("target_start_time")

    if len(plot_df) > max_points:
        step = max(1, len(plot_df) // max_points)
        plot_df = plot_df.iloc[::step].copy()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(plot_df["target_start_time"], plot_df["y_true_t+1"], label="Ground Truth", linewidth=1.2)
    ax.plot(plot_df["target_start_time"], plot_df["y_pred_t+1"], label="Prediction", linewidth=1.2)
    ax.set_title(f"{model_label} Prediction on {split_name.title()} Set (t+1)")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Active_Pow")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the unified PV physical post-processing protocol to existing experiment outputs: "
            "night predictions are forced to zero and remaining negative predictions are clipped to zero."
        )
    )
    parser.add_argument(
        "--input_root",
        type=str,
        default=str(PROJECT_CONFIG.get_path("paths.results_root", "results/d1_long_no_wind_2015_2022")),
        help="Original experiment result root. This directory is never modified.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="results/d1_long_no_wind_2015_2022_physical_mask",
        help="Destination root for post-processed results.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model result directories to process. Each case must already contain predictions.csv.",
    )
    parser.add_argument("--pred_lens", nargs="+", type=int, default=DEFAULT_PRED_LENS, help="Prediction lengths.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild post-processed outputs even when the destination predictions.csv already exists.",
    )
    parser.add_argument(
        "--strict_missing",
        action="store_true",
        help="Fail immediately when a case is missing predictions.csv. By default, missing cases are skipped.",
    )
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    return resolve_project_path(path_value, PROJECT_ROOT)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def discover_horizons(df: pd.DataFrame) -> list[int]:
    horizons = sorted(
        int(match.group(1))
        for column in df.columns
        for match in [PRED_COLUMN_PATTERN.match(column)]
        if match is not None
    )
    if not horizons:
        raise ValueError("Cannot find any y_pred_t+k columns in predictions.csv.")
    return horizons


def dataframe_to_arrays(df: pd.DataFrame, horizons: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.stack([df[f"y_true_t+{horizon}"].to_numpy(dtype=np.float32) for horizon in horizons], axis=1)
    y_pred = np.stack(
        [
            df[
                f"y_pred_raw_t+{horizon}"
                if f"y_pred_raw_t+{horizon}" in df.columns
                else f"y_pred_t+{horizon}"
            ].to_numpy(dtype=np.float32)
            for horizon in horizons
        ],
        axis=1,
    )
    day_night = np.stack([df[f"day_night_t+{horizon}"].to_numpy(dtype=np.int64) for horizon in horizons], axis=1)
    return y_true, y_pred, day_night


def apply_night_nonnegative_mask(
    y_pred_raw: np.ndarray,
    day_night_label: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, dict[str, int | str | bool | float]]:
    y_pred = y_pred_raw.astype(np.float32, copy=True)
    night_mask = day_night_label != 1

    night_points = int(night_mask.sum())
    night_changed = int(np.sum(np.abs(y_pred[night_mask]) > eps))
    y_pred[night_mask] = 0.0

    negative_mask = y_pred < 0.0
    negative_clip_points = int(negative_mask.sum())
    y_pred[negative_mask] = 0.0

    changed_mask = np.abs(y_pred - y_pred_raw) > eps
    stats: dict[str, int | str | bool | float] = {
        "mode": "night_nonnegative",
        "night_mask_source": "day_night_t+k",
        "night_value": 0.0,
        "nonnegative_clip": True,
        "total_points": int(y_pred_raw.size),
        "night_points": night_points,
        "night_forced_zero_changed_points": night_changed,
        "negative_clip_points": negative_clip_points,
        "total_changed_points": int(changed_mask.sum()),
    }
    return y_pred, stats


def add_raw_and_processed_predictions(
    df: pd.DataFrame,
    horizons: list[int],
    y_pred_processed: np.ndarray,
) -> pd.DataFrame:
    output_df = df.copy()
    for index, horizon in enumerate(horizons):
        pred_col = f"y_pred_t+{horizon}"
        raw_col = f"y_pred_raw_t+{horizon}"
        if raw_col not in output_df.columns:
            output_df[raw_col] = output_df[pred_col].to_numpy(dtype=np.float32)
        output_df[pred_col] = y_pred_processed[:, index]
    return reorder_prediction_columns(output_df, horizons)


def reorder_prediction_columns(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    ordered: list[str] = [
        column
        for column in ["sample_id", "input_start_time", "input_end_time", "target_start_time", "target_end_time"]
        if column in df.columns
    ]
    for horizon in horizons:
        ordered.extend(
            column
            for column in [
                f"y_true_t+{horizon}",
                f"y_pred_raw_t+{horizon}",
                f"y_pred_t+{horizon}",
                f"day_night_t+{horizon}",
            ]
            if column in df.columns
        )
    ordered.extend(column for column in df.columns if column not in ordered)
    return df.loc[:, ordered]


def build_metrics_payload(
    *,
    model_name: str,
    pred_len: int,
    source_dir: Path,
    source_predictions_path: Path,
    source_metrics_path: Path,
    output_dir: Path,
    raw_metrics: dict[str, Any],
    processed_metrics: dict[str, Any],
    postprocess_stats: dict[str, Any],
) -> dict[str, Any]:
    report_split = str(raw_metrics.get("report_split", "test"))
    config = raw_metrics.get("config", {})
    if isinstance(config, dict):
        config = dict(config)
        config["physical_postprocess"] = "night_nonnegative"

    payload: dict[str, Any] = {
        "config": config,
        "experiment_name": raw_metrics.get("experiment_name"),
        "tuning_stage": raw_metrics.get("tuning_stage"),
        "baseline_type": raw_metrics.get("baseline_type", model_name),
        "baseline_definition": raw_metrics.get("baseline_definition"),
        "report_split": report_split,
        "physical_postprocess": postprocess_stats,
        "source_results_dir": str(source_dir),
        "source_predictions_path": str(source_predictions_path),
        "source_metrics_path": str(source_metrics_path) if source_metrics_path.exists() else None,
        "output_results_dir": str(output_dir),
        "raw_reported_metrics": raw_metrics.get("reported_metrics"),
        "reported_metrics": processed_metrics,
        "test_metrics": processed_metrics if report_split == "test" else None,
        "validation_metrics": processed_metrics if report_split == "validation" else None,
        "checkpoint_path": raw_metrics.get("checkpoint_path"),
        "expected_delta_minutes": raw_metrics.get("expected_delta_minutes"),
        "dataset_summary": raw_metrics.get("dataset_summary"),
        "raw_split_rows": raw_metrics.get("raw_split_rows"),
        "target_feature_index": raw_metrics.get("target_feature_index"),
        "trainable_parameter_count": raw_metrics.get("trainable_parameter_count"),
        "postprocess_note": (
            "Predictions were post-processed uniformly for all models: target horizons with day_night_t+k != 1 "
            "were forced to zero, then remaining negative predictions were clipped to zero. The mask is derived "
            "from timestamp/solar-elevation preprocessing labels, not from y_true == 0."
        ),
    }
    if "causal_mask" in raw_metrics:
        payload["causal_mask"] = raw_metrics["causal_mask"]
    if "calibration_usage" in raw_metrics:
        payload["calibration_usage"] = raw_metrics["calibration_usage"]
    if "best_epoch" in raw_metrics:
        payload["best_epoch"] = raw_metrics["best_epoch"]
    if "best_validation_loss" in raw_metrics:
        payload["best_validation_loss"] = raw_metrics["best_validation_loss"]
    return payload


def process_one_case(
    *,
    model_name: str,
    pred_len: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    input_root = resolve_path(args.input_root)
    output_root = resolve_path(args.output_root)
    source_dir = input_root / model_name / f"pred_len_{pred_len}"
    output_dir = output_root / model_name / f"pred_len_{pred_len}"
    output_predictions_path = output_dir / "predictions.csv"

    if output_predictions_path.exists() and not args.force:
        metrics = read_json(output_dir / "metrics.json")
        return build_summary_row(model_name, pred_len, output_dir, metrics, skipped=True)

    source_predictions_path = source_dir / "predictions.csv"
    if not source_predictions_path.exists():
        if args.strict_missing:
            raise FileNotFoundError(f"Missing predictions.csv: {source_predictions_path}")
        print(f"Skipping {model_name}/pred_len_{pred_len}: missing {source_predictions_path}", flush=True)
        return build_missing_summary_row(model_name, pred_len, source_dir, output_dir, source_predictions_path)

    source_metrics_path = source_dir / "metrics.json"
    raw_metrics = read_json(source_metrics_path)

    predictions_df = pd.read_csv(source_predictions_path)
    horizons = discover_horizons(predictions_df)
    if horizons != list(range(1, pred_len + 1)):
        raise ValueError(
            f"Unexpected horizons for {source_predictions_path}: {horizons}. "
            f"Expected consecutive horizons 1..{pred_len}."
        )

    y_true, y_pred_raw, day_night_label = dataframe_to_arrays(predictions_df, horizons)
    y_pred_processed, postprocess_stats = apply_night_nonnegative_mask(y_pred_raw, day_night_label)
    processed_metrics = evaluate_prediction_arrays(
        y_true=y_true,
        y_pred=y_pred_processed,
        day_night_label=day_night_label,
    )
    output_df = add_raw_and_processed_predictions(predictions_df, horizons, y_pred_processed)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_predictions_path, index=False)
    save_prediction_plot(
        output_df,
        output_dir / "pred_plot.png",
        split_name=str(raw_metrics.get("report_split", "test")),
        model_label=f"{model_name} + Physical Mask",
    )

    metrics_payload = build_metrics_payload(
        model_name=model_name,
        pred_len=pred_len,
        source_dir=source_dir,
        source_predictions_path=source_predictions_path,
        source_metrics_path=source_metrics_path,
        output_dir=output_dir,
        raw_metrics=raw_metrics,
        processed_metrics=processed_metrics,
        postprocess_stats=postprocess_stats,
    )
    write_json(output_dir / "metrics.json", metrics_payload)

    return build_summary_row(model_name, pred_len, output_dir, metrics_payload, skipped=False)


def build_summary_row(
    model_name: str,
    pred_len: int,
    output_dir: Path,
    metrics_payload: dict[str, Any],
    skipped: bool,
) -> dict[str, Any]:
    reported_metrics = metrics_payload.get("reported_metrics") or {}
    all_metrics = reported_metrics.get("all_timestamps") or {}
    daytime_metrics = reported_metrics.get("daytime_only") or {}
    postprocess_stats = metrics_payload.get("physical_postprocess") or {}
    return {
        "model": model_name,
        "pred_len": pred_len,
        "status": "skipped_existing" if skipped else "processed",
        "skipped_existing": skipped,
        "output_dir": str(output_dir),
        "all_mae": all_metrics.get("mae"),
        "all_rmse": all_metrics.get("rmse"),
        "all_mbe": all_metrics.get("mbe"),
        "daytime_mae": daytime_metrics.get("mae"),
        "daytime_rmse": daytime_metrics.get("rmse"),
        "daytime_mbe": daytime_metrics.get("mbe"),
        "daytime_r2": daytime_metrics.get("r2"),
        "night_points": postprocess_stats.get("night_points"),
        "night_forced_zero_changed_points": postprocess_stats.get("night_forced_zero_changed_points"),
        "negative_clip_points": postprocess_stats.get("negative_clip_points"),
        "total_changed_points": postprocess_stats.get("total_changed_points"),
    }


def build_missing_summary_row(
    model_name: str,
    pred_len: int,
    source_dir: Path,
    output_dir: Path,
    source_predictions_path: Path,
) -> dict[str, Any]:
    return {
        "model": model_name,
        "pred_len": pred_len,
        "status": "missing_predictions_csv",
        "skipped_existing": False,
        "source_dir": str(source_dir),
        "missing_predictions_path": str(source_predictions_path),
        "output_dir": str(output_dir),
        "all_mae": None,
        "all_rmse": None,
        "all_mbe": None,
        "daytime_mae": None,
        "daytime_rmse": None,
        "daytime_mbe": None,
        "daytime_r2": None,
        "night_points": None,
        "night_forced_zero_changed_points": None,
        "negative_clip_points": None,
        "total_changed_points": None,
    }


def main() -> None:
    args = parse_args()
    output_root = resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for model_name in args.models:
        for pred_len in args.pred_lens:
            print(f"Processing {model_name}/pred_len_{pred_len}", flush=True)
            row = process_one_case(model_name=model_name, pred_len=pred_len, args=args)
            rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_path = output_root / "physical_mask_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    write_json(
        output_root / "physical_mask_manifest.json",
        {
            "input_root": str(resolve_path(args.input_root)),
            "output_root": str(output_root),
            "models": list(args.models),
            "pred_lens": [int(pred_len) for pred_len in args.pred_lens],
            "mode": "night_nonnegative",
            "missing_predictions_policy": "skip",
            "summary_csv": str(summary_path),
        },
    )
    print(f"Saved summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
