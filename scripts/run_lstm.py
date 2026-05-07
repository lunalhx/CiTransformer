from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.project_config import load_project_config, resolve_project_path

PROJECT_CONFIG = load_project_config()
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_CONFIG.get_path("paths.matplotlib_cache")))

import matplotlib.pyplot as plt

from models.baseline import LSTMBaseline
from utils.datasets import (
    DEFAULT_DATA_DIR,
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_TARGET_COLUMN,
    ContinuousSegmentTimeSeriesDataset,
    fit_split_scalers,
    infer_expected_timedelta,
    load_split_dataframe,
    resolve_split_dir,
)


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        self.best_score = math.inf
        self.counter = 0

    def step(self, current_score: float) -> bool:
        if current_score < self.best_score - self.min_delta:
            self.best_score = current_score
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the LSTM baseline for PV power forecasting.")

    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(PROJECT_CONFIG.get_path("paths.data_dir", DEFAULT_DATA_DIR)),
        help="Directory containing split CSV files.",
    )
    parser.add_argument("--time_col", type=str, default=None, help="Optional explicit timestamp column name.")
    parser.add_argument("--target_col", type=str, default=DEFAULT_TARGET_COLUMN, help="Prediction target column.")
    parser.add_argument(
        "--feature_cols",
        nargs="+",
        default=DEFAULT_FEATURE_COLUMNS,
        help="Input feature columns. Defaults to the 11 processed PV features.",
    )
    parser.add_argument("--seq_len", type=int, default=96, help="Look-back sequence length.")
    parser.add_argument("--pred_len", type=int, default=1, help="Forecast horizon length.")
    parser.add_argument(
        "--sampling_freq_minutes",
        type=int,
        default=None,
        help="Optional expected sampling interval in minutes. If omitted, inferred from train timestamps.",
    )

    parser.add_argument("--hidden_size", type=int, default=128, help="LSTM hidden size.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout used in LSTM/decoder head.")

    parser.add_argument("--epochs", type=int, default=30, help="Maximum training epochs.")
    parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Adam weight decay.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm. Set <=0 to disable.")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience.")
    parser.add_argument("--min_delta", type=float, default=1e-5, help="Minimum validation loss improvement.")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=int(PROJECT_CONFIG.get("runtime.num_workers", 0)),
        help="DataLoader workers.",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=0,
        help="Optional extra training heartbeat every N batches. Set <=0 to rely on the progress bar only.",
    )
    parser.add_argument(
        "--progress_mininterval",
        type=float,
        default=15.0,
        help="Minimum seconds between progress bar refreshes and extra heartbeat logs.",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--device",
        type=str,
        default=str(PROJECT_CONFIG.get("runtime.device", "auto")),
        choices=["auto", "cpu", "cuda", "mps"],
        help="Training device. 'auto' picks cuda -> mps -> cpu.",
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(
            PROJECT_CONFIG.get_path(
                "paths.results.lstm",
                "results/d1_long_no_wind_2015_2022/lstm",
            )
        ),
        help="Directory for metrics/predictions/plots.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=str(
            PROJECT_CONFIG.get_path(
                "paths.checkpoints.lstm",
                "checkpoints/d1_long_no_wind_2015_2022/lstm",
            )
            / "best_model.pth"
        ),
        help="Path to save the best checkpoint.",
    )
    parser.add_argument(
        "--eval_checkpoint_path",
        type=str,
        default=None,
        help="Optional existing best_model.pth path. If provided, skip training and only export validation/test results.",
    )
    parser.add_argument(
        "--report_split",
        type=str,
        default="test",
        choices=["validation", "test"],
        help="Which split to export as predictions.csv / pred_plot.png and print as the main summary.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Optional human-readable experiment name stored in metrics.json for tuning bookkeeping.",
    )
    parser.add_argument(
        "--tuning_stage",
        type=str,
        default=None,
        help="Optional tuning stage label stored in metrics.json (for example: s1, s2, final).",
    )
    parser.add_argument(
        "--tuning_only",
        action="store_true",
        help="Validation-only tuning mode. Skips loading/evaluating the test split and requires --report_split validation.",
    )

    parser.add_argument(
        "--max_train_batches",
        type=int,
        default=None,
        help="Optional debug cap for train batches per epoch.",
    )
    parser.add_argument(
        "--max_eval_batches",
        type=int,
        default=None,
        help="Optional debug cap for validation/test batches.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_name: str) -> torch.device:
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")

    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device("mps")

    if device_name == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> dict[str, float | int | None]:
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


def format_metric_for_console(value: float | None) -> str:
    if value is None:
        return "nan"
    return f"{value:.6f}"


def resolve_progress_mininterval(progress_mininterval: float) -> float:
    return max(float(progress_mininterval), 1.0)


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


def ns_to_datetime(ns_array: np.ndarray, timezone: Any) -> pd.DatetimeIndex:
    flat_array = np.asarray(ns_array).reshape(-1)
    if timezone is None:
        return pd.to_datetime(flat_array)
    return pd.to_datetime(flat_array, utc=True).tz_convert(timezone)


def save_prediction_plot(
    predictions_df: pd.DataFrame,
    output_path: Path,
    split_name: str = "test",
    model_label: str = "LSTM Baseline",
    max_points: int = 4000,
) -> None:
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


def create_data_loader(dataset: ContinuousSegmentTimeSeriesDataset, batch_size: int, shuffle: bool, num_workers: int, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def compute_target_daytime_ratio(dataset: ContinuousSegmentTimeSeriesDataset) -> float | None:
    total_points = 0
    daytime_points = 0

    for segment_start, segment_length in zip(dataset.valid_segment_starts, dataset.valid_segment_lengths, strict=True):
        window_count = int(segment_length - dataset.seq_len - dataset.pred_len + 1)
        if window_count <= 0:
            continue

        for offset in range(window_count):
            target_start = int(segment_start + offset + dataset.seq_len)
            target_end = target_start + dataset.pred_len
            target_day = dataset.day_night_label[target_start:target_end]
            daytime_points += int(np.sum(target_day == 1))
            total_points += int(dataset.pred_len)

    if total_points == 0:
        return None
    return float(daytime_points / total_points)


def build_training_run_stats(datasets: dict[str, ContinuousSegmentTimeSeriesDataset]) -> dict[str, int | float | None]:
    return {
        "train_sample_count": int(len(datasets["train"])),
        "val_sample_count": int(len(datasets["validation"])),
        "train_daytime_ratio": compute_target_daytime_ratio(datasets["train"]),
        "val_daytime_ratio": compute_target_daytime_ratio(datasets["validation"]),
    }


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = batch["x"].to(device=device, dtype=torch.float32)
    y = batch["y"].to(device=device, dtype=torch.float32)
    return x, y


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    epoch: int,
    log_interval: int,
    progress_mininterval: float,
    max_batches: int | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    refresh_interval = resolve_progress_mininterval(progress_mininterval)
    last_heartbeat_time = time.monotonic()

    progress = tqdm(
        loader,
        desc=f"Train Epoch {epoch:03d}",
        leave=True,
        dynamic_ncols=True,
        mininterval=refresh_interval,
    )
    for batch_idx, batch in enumerate(progress):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x, y = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        predictions = model(x)
        loss = criterion(predictions, y)
        loss.backward()

        if grad_clip and grad_clip > 0:
            clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        batch_size = x.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        progress.set_postfix(loss=f"{loss.item():.6f}", refresh=False)

        if log_interval > 0 and (batch_idx + 1) % log_interval == 0:
            now = time.monotonic()
            if now - last_heartbeat_time >= refresh_interval:
                avg_loss = total_loss / max(total_samples, 1)
                log(
                    f"[Epoch {epoch:03d}] train batch {batch_idx + 1}/{len(loader)} "
                    f"| batch_loss={loss.item():.6f} avg_loss={avg_loss:.6f}"
                )
                last_heartbeat_time = now

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split_name: str,
    epoch: int | None = None,
    progress_mininterval: float = 15.0,
    max_batches: int | None = None,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    description = f"{split_name.title()} Eval"
    if epoch is not None:
        description = f"{split_name.title()} Epoch {epoch:03d}"

    progress = tqdm(
        loader,
        desc=description,
        leave=True,
        dynamic_ncols=True,
        mininterval=resolve_progress_mininterval(progress_mininterval),
    )
    for batch_idx, batch in enumerate(progress):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x, y = move_batch_to_device(batch, device)
        predictions = model(x)
        loss = criterion(predictions, y)

        batch_size = x.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        progress.set_postfix(loss=f"{loss.item():.6f}", refresh=False)

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    target_scaler: Any,
    device: torch.device,
    split_name: str = "test",
    progress_mininterval: float = 15.0,
    max_batches: int | None = None,
) -> dict[str, np.ndarray]:
    model.eval()

    pred_batches: list[np.ndarray] = []
    true_batches: list[np.ndarray] = []
    input_start_batches: list[np.ndarray] = []
    input_end_batches: list[np.ndarray] = []
    target_time_batches: list[np.ndarray] = []
    target_day_batches: list[np.ndarray] = []

    progress = tqdm(
        loader,
        desc=f"{split_name.title()} Predict",
        leave=True,
        dynamic_ncols=True,
        mininterval=resolve_progress_mininterval(progress_mininterval),
    )
    for batch_idx, batch in enumerate(progress):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = batch["x"].to(device=device, dtype=torch.float32)
        predictions_scaled = model(x).detach().cpu().numpy()
        predictions = target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).reshape(predictions_scaled.shape)

        pred_batches.append(predictions.astype(np.float32))
        true_batches.append(batch["y_raw"].cpu().numpy().astype(np.float32))
        input_start_batches.append(batch["input_start_ns"].cpu().numpy().astype(np.int64))
        input_end_batches.append(batch["input_end_ns"].cpu().numpy().astype(np.int64))
        target_time_batches.append(batch["target_time_ns"].cpu().numpy().astype(np.int64))
        target_day_batches.append(batch["target_day_night"].cpu().numpy().astype(np.int64))

    return {
        "y_pred": np.concatenate(pred_batches, axis=0),
        "y_true": np.concatenate(true_batches, axis=0),
        "input_start_ns": np.concatenate(input_start_batches, axis=0),
        "input_end_ns": np.concatenate(input_end_batches, axis=0),
        "target_time_ns": np.concatenate(target_time_batches, axis=0),
        "target_day_night": np.concatenate(target_day_batches, axis=0),
    }


def build_prediction_dataframe(
    prediction_dict: dict[str, np.ndarray],
    target_timezone: Any,
) -> pd.DataFrame:
    y_pred = prediction_dict["y_pred"]
    y_true = prediction_dict["y_true"]
    input_start_ns = prediction_dict["input_start_ns"]
    input_end_ns = prediction_dict["input_end_ns"]
    target_time_ns = prediction_dict["target_time_ns"]
    target_day_night = prediction_dict["target_day_night"]
    pred_len = y_pred.shape[1]

    target_start_ns = target_time_ns[:, 0]
    target_end_ns = target_time_ns[:, -1]

    frame_dict: dict[str, Any] = {
        "sample_id": np.arange(len(y_pred), dtype=np.int64),
        "input_start_time": ns_to_datetime(input_start_ns, target_timezone).astype(str),
        "input_end_time": ns_to_datetime(input_end_ns, target_timezone).astype(str),
        "target_start_time": ns_to_datetime(target_start_ns, target_timezone).astype(str),
        "target_end_time": ns_to_datetime(target_end_ns, target_timezone).astype(str),
    }

    for horizon in range(pred_len):
        horizon_name = horizon + 1
        frame_dict[f"y_true_t+{horizon_name}"] = y_true[:, horizon]
        frame_dict[f"y_pred_t+{horizon_name}"] = y_pred[:, horizon]
        frame_dict[f"day_night_t+{horizon_name}"] = target_day_night[:, horizon]

    return pd.DataFrame(frame_dict)


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    args: argparse.Namespace,
    expected_delta: pd.Timedelta,
    feature_scaler: Any,
    target_scaler: Any,
    feature_cols: list[str],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "feature_cols": feature_cols,
            "expected_delta_minutes": float(expected_delta / pd.Timedelta(minutes=1)),
            "feature_scaler_mean": feature_scaler.mean_.tolist(),
            "feature_scaler_scale": feature_scaler.scale_.tolist(),
            "target_scaler_mean": target_scaler.mean_.tolist(),
            "target_scaler_scale": target_scaler.scale_.tolist(),
        },
        checkpoint_path,
    )


def prepare_datasets(
    args: argparse.Namespace,
    include_test: bool = True,
) -> tuple[dict[str, ContinuousSegmentTimeSeriesDataset], dict[str, pd.DataFrame], Any, pd.Timedelta]:
    split_dir = resolve_split_dir(args.data_dir)

    train_df = load_split_dataframe(split_dir / "train.csv", time_col=args.time_col)
    validation_df = load_split_dataframe(split_dir / "validation.csv", time_col=args.time_col)
    calibration_df = load_split_dataframe(split_dir / "calibration.csv", time_col=args.time_col)
    test_df = load_split_dataframe(split_dir / "test.csv", time_col=args.time_col) if include_test else None

    feature_cols = list(args.feature_cols)
    scalers = fit_split_scalers(train_df, feature_cols, args.target_col)

    expected_delta = (
        pd.Timedelta(minutes=args.sampling_freq_minutes)
        if args.sampling_freq_minutes is not None
        else infer_expected_timedelta(train_df.index)
    )

    datasets = {
        "train": ContinuousSegmentTimeSeriesDataset(
            df=train_df,
            feature_cols=feature_cols,
            target_col=args.target_col,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            feature_scaler=scalers.feature_scaler,
            target_scaler=scalers.target_scaler,
            expected_delta=expected_delta,
        ),
        "validation": ContinuousSegmentTimeSeriesDataset(
            df=validation_df,
            feature_cols=feature_cols,
            target_col=args.target_col,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            feature_scaler=scalers.feature_scaler,
            target_scaler=scalers.target_scaler,
            expected_delta=expected_delta,
        ),
        "calibration": ContinuousSegmentTimeSeriesDataset(
            df=calibration_df,
            feature_cols=feature_cols,
            target_col=args.target_col,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            feature_scaler=scalers.feature_scaler,
            target_scaler=scalers.target_scaler,
            expected_delta=expected_delta,
        ),
    }
    if test_df is not None:
        datasets["test"] = ContinuousSegmentTimeSeriesDataset(
            df=test_df,
            feature_cols=feature_cols,
            target_col=args.target_col,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            feature_scaler=scalers.feature_scaler,
            target_scaler=scalers.target_scaler,
            expected_delta=expected_delta,
        )

    raw_frames = {
        "train": train_df,
        "validation": validation_df,
        "calibration": calibration_df,
    }
    if test_df is not None:
        raw_frames["test"] = test_df
    return datasets, raw_frames, scalers, expected_delta


def print_dataset_summaries(datasets: dict[str, ContinuousSegmentTimeSeriesDataset]) -> None:
    log("\nDataset summary")
    log("-" * 80)
    for split_name, dataset in datasets.items():
        summary = dataset.summary()
        log(f"{split_name:>11s}: {json.dumps(summary, ensure_ascii=False)}")
    log("-" * 80)


def resolve_path(path_value: str) -> Path:
    return resolve_project_path(path_value, PROJECT_ROOT)


def build_lstm_model(args: argparse.Namespace) -> LSTMBaseline:
    return LSTMBaseline(
        input_size=len(args.feature_cols),
        hidden_size=args.hidden_size,
        pred_len=args.pred_len,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )


def build_eval_args_from_checkpoint(cli_args: argparse.Namespace, checkpoint_args: dict[str, Any]) -> argparse.Namespace:
    eval_args = argparse.Namespace(**checkpoint_args)

    eval_args.data_dir = cli_args.data_dir
    if cli_args.time_col is not None:
        eval_args.time_col = cli_args.time_col
    if cli_args.sampling_freq_minutes is not None:
        eval_args.sampling_freq_minutes = cli_args.sampling_freq_minutes
    eval_args.device = cli_args.device
    eval_args.results_dir = cli_args.results_dir
    eval_args.report_split = cli_args.report_split
    eval_args.experiment_name = cli_args.experiment_name or getattr(eval_args, "experiment_name", None)
    eval_args.tuning_stage = cli_args.tuning_stage or getattr(eval_args, "tuning_stage", None)
    if cli_args.max_eval_batches is not None:
        eval_args.max_eval_batches = cli_args.max_eval_batches
    eval_args.progress_mininterval = cli_args.progress_mininterval
    eval_args.num_workers = cli_args.num_workers
    eval_args.tuning_only = False
    eval_args.eval_checkpoint_path = cli_args.eval_checkpoint_path
    return eval_args


def evaluate_and_export(
    args: argparse.Namespace,
    model: nn.Module,
    datasets: dict[str, ContinuousSegmentTimeSeriesDataset],
    raw_frames: dict[str, pd.DataFrame],
    scalers: Any,
    expected_delta: pd.Timedelta,
    results_dir: Path,
    device: torch.device,
    checkpoint_path: Path,
    best_epoch: int,
    best_val_loss: float | None,
    history: list[dict[str, float | int | None]],
    training_run_stats: dict[str, int | float | None],
) -> None:
    target_feature_index = int(list(args.feature_cols).index(args.target_col)) if args.target_col in args.feature_cols else None
    trainable_parameter_count = int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    validation_loader = create_data_loader(datasets["validation"], args.batch_size, False, args.num_workers, device)

    validation_prediction_dict = collect_predictions(
        model=model,
        loader=validation_loader,
        target_scaler=scalers.target_scaler,
        device=device,
        split_name="validation",
        progress_mininterval=args.progress_mininterval,
        max_batches=args.max_eval_batches,
    )
    validation_metrics = evaluate_prediction_arrays(
        y_true=validation_prediction_dict["y_true"],
        y_pred=validation_prediction_dict["y_pred"],
        day_night_label=validation_prediction_dict["target_day_night"],
    )

    test_metrics: dict[str, Any] | None = None
    reported_split_name = args.report_split
    if reported_split_name == "validation":
        reported_prediction_dict = validation_prediction_dict
    else:
        if "test" not in datasets:
            raise RuntimeError("Test dataset was not loaded, so report_split=test is unavailable in this run mode.")
        test_loader = create_data_loader(datasets["test"], args.batch_size, False, args.num_workers, device)
        reported_prediction_dict = collect_predictions(
            model=model,
            loader=test_loader,
            target_scaler=scalers.target_scaler,
            device=device,
            split_name="test",
            progress_mininterval=args.progress_mininterval,
            max_batches=args.max_eval_batches,
        )
        test_metrics = evaluate_prediction_arrays(
            y_true=reported_prediction_dict["y_true"],
            y_pred=reported_prediction_dict["y_pred"],
            day_night_label=reported_prediction_dict["target_day_night"],
        )

    reported_metrics = validation_metrics if reported_split_name == "validation" else test_metrics
    if reported_metrics is None:
        raise RuntimeError("reported_metrics should not be None after evaluation.")

    predictions_df = build_prediction_dataframe(
        prediction_dict=reported_prediction_dict,
        target_timezone=datasets[reported_split_name].timezone,
    )

    metrics_payload = {
        "config": vars(args),
        "experiment_name": args.experiment_name,
        "tuning_stage": args.tuning_stage,
        "device": str(device),
        "baseline_type": "lstm",
        "baseline_definition": "Sequence-to-vector LSTM that maps the last encoder hidden state to future Active_Pow horizons.",
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "expected_delta_minutes": float(expected_delta / pd.Timedelta(minutes=1)),
        "dataset_summary": {split_name: dataset.summary() for split_name, dataset in datasets.items()},
        "training_run_stats": training_run_stats,
        "raw_split_rows": {split_name: int(len(df)) for split_name, df in raw_frames.items()},
        "target_feature_index": target_feature_index,
        "report_split": args.report_split,
        "calibration_usage": "loaded_but_unused",
        "trainable_parameter_count": trainable_parameter_count,
        "validation_metrics": validation_metrics,
        "reported_metrics": reported_metrics,
        "test_metrics": test_metrics,
        "history": history,
        "checkpoint_path": str(checkpoint_path),
    }

    metrics_path = results_dir / "metrics.json"
    predictions_path = results_dir / "predictions.csv"
    plot_path = results_dir / "pred_plot.png"

    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics_payload, fp, ensure_ascii=False, indent=2)

    predictions_df.to_csv(predictions_path, index=False)
    save_prediction_plot(predictions_df, plot_path, split_name=reported_split_name, model_label="LSTM Baseline")

    log("\nSaved outputs")
    log(f"- metrics: {metrics_path}")
    log(f"- predictions: {predictions_path}")
    log(f"- plot: {plot_path}")
    log(f"- checkpoint: {checkpoint_path}")

    all_metrics = reported_metrics["all_timestamps"]
    daytime_metrics = reported_metrics["daytime_only"]
    log(f"\nReported metrics ({reported_split_name})")
    log(
        "all timestamps | "
        f"MAE={format_metric_for_console(all_metrics['mae'])} "
        f"MSE={format_metric_for_console(all_metrics['mse'])} "
        f"RMSE={format_metric_for_console(all_metrics['rmse'])} "
        f"MBE={format_metric_for_console(all_metrics['mbe'])} "
        f"sMAPE={format_metric_for_console(all_metrics['smape'])} "
        f"MAPE(nonzero)={format_metric_for_console(all_metrics['mape_nonzero'])} "
        f"WAPE={format_metric_for_console(all_metrics['wape'])} "
        f"nRMSE(max)={format_metric_for_console(all_metrics['nrmse_by_max'])}"
    )
    log(
        "daytime only  | "
        f"MAE={format_metric_for_console(daytime_metrics['mae'])} "
        f"MSE={format_metric_for_console(daytime_metrics['mse'])} "
        f"RMSE={format_metric_for_console(daytime_metrics['rmse'])} "
        f"MBE={format_metric_for_console(daytime_metrics['mbe'])} "
        f"sMAPE={format_metric_for_console(daytime_metrics['smape'])} "
        f"MAPE(nonzero)={format_metric_for_console(daytime_metrics['mape_nonzero'])} "
        f"WAPE={format_metric_for_console(daytime_metrics['wape'])} "
        f"nRMSE(max)={format_metric_for_console(daytime_metrics['nrmse_by_max'])}"
    )


def main() -> None:
    args = parse_args()

    if args.tuning_only and args.report_split != "validation":
        raise ValueError("--tuning_only requires --report_split validation.")
    if args.tuning_only and args.eval_checkpoint_path is not None:
        raise ValueError("--tuning_only cannot be combined with --eval_checkpoint_path.")

    eval_checkpoint: dict[str, Any] | None = None
    eval_checkpoint_path: Path | None = None
    if args.eval_checkpoint_path is not None:
        eval_checkpoint_path = resolve_path(args.eval_checkpoint_path)
        eval_checkpoint = torch.load(eval_checkpoint_path, map_location="cpu")
        checkpoint_args = eval_checkpoint.get("args")
        if checkpoint_args is None:
            raise KeyError(f"Checkpoint {eval_checkpoint_path} does not contain saved args.")
        args = build_eval_args_from_checkpoint(args, checkpoint_args)

    set_random_seed(args.seed)
    device = get_device(args.device)

    if eval_checkpoint is not None:
        if eval_checkpoint_path is None:
            raise RuntimeError("eval_checkpoint_path should be resolved before checkpoint evaluation.")

        results_dir = resolve_path(args.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        include_test = args.report_split == "test"
        datasets, raw_frames, scalers, expected_delta = prepare_datasets(args, include_test=include_test)
        print_dataset_summaries(datasets)

        if len(datasets["validation"]) == 0:
            raise RuntimeError("Validation dataset contains no valid windows. Reduce seq_len/pred_len or inspect segmentation.")
        if args.report_split == "test" and len(datasets["test"]) == 0:
            raise RuntimeError("Test dataset contains no valid windows. Reduce seq_len/pred_len or inspect segmentation.")

        model = build_lstm_model(args).to(device)
        model.load_state_dict(eval_checkpoint["model_state_dict"])
        model.eval()

        log(f"\nUsing device: {device}")
        log(f"Results directory: {results_dir}")
        log(f"Checkpoint path: {eval_checkpoint_path}")
        evaluate_and_export(
            args=args,
            model=model,
            datasets=datasets,
            raw_frames=raw_frames,
            scalers=scalers,
            expected_delta=expected_delta,
            results_dir=results_dir,
            device=device,
            checkpoint_path=eval_checkpoint_path,
            best_epoch=int(eval_checkpoint.get("epoch", 0)),
            best_val_loss=(
                float(eval_checkpoint.get("best_val_loss"))
                if eval_checkpoint.get("best_val_loss") is not None
                else None
            ),
            history=[],
            training_run_stats=build_training_run_stats(datasets),
        )
        return

    results_dir = resolve_path(args.results_dir)
    checkpoint_path = resolve_path(args.checkpoint_path)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    datasets, raw_frames, scalers, expected_delta = prepare_datasets(args, include_test=not args.tuning_only)
    print_dataset_summaries(datasets)

    if len(datasets["train"]) == 0:
        raise RuntimeError("Train dataset contains no valid windows. Reduce seq_len/pred_len or inspect segmentation.")
    if len(datasets["validation"]) == 0:
        raise RuntimeError("Validation dataset contains no valid windows. Reduce seq_len/pred_len or inspect segmentation.")
    if not args.tuning_only and len(datasets["test"]) == 0:
        raise RuntimeError("Test dataset contains no valid windows. Reduce seq_len/pred_len or inspect segmentation.")

    train_loader = create_data_loader(datasets["train"], args.batch_size, True, args.num_workers, device)
    validation_loader = create_data_loader(datasets["validation"], args.batch_size, False, args.num_workers, device)
    training_run_stats = build_training_run_stats(datasets)

    model = build_lstm_model(args).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

    history: list[dict[str, float | int | None]] = []
    best_val_loss = math.inf
    best_epoch = 0

    log(f"\nUsing device: {device}")
    log(f"Results directory: {results_dir}")
    log(f"Checkpoint path: {checkpoint_path}")
    for epoch in range(1, args.epochs + 1):
        log(f"\nStarting epoch {epoch:03d}/{args.epochs}")
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
            epoch=epoch,
            log_interval=args.log_interval,
            progress_mininterval=args.progress_mininterval,
            max_batches=args.max_train_batches,
        )
        val_loss = evaluate_loss(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
            split_name="validation",
            epoch=epoch,
            progress_mininterval=args.progress_mininterval,
            max_batches=args.max_eval_batches,
        )

        if val_loss < best_val_loss - args.min_delta:
            best_val_loss = float(val_loss)
            best_epoch = epoch
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
                args=args,
                expected_delta=expected_delta,
                feature_scaler=scalers.feature_scaler,
                target_scaler=scalers.target_scaler,
                feature_cols=list(args.feature_cols),
            )

        should_stop = early_stopping.step(val_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "validation_loss": float(val_loss),
                "best_epoch": int(best_epoch),
                "best_validation_loss": float(best_val_loss),
                "patience_counter": int(early_stopping.counter),
                "learning_rate": get_learning_rate(optimizer),
                **training_run_stats,
            }
        )
        log(
            f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} "
            f"| best_epoch={best_epoch} | best_val_loss={best_val_loss:.6f} "
            f"| patience_counter={early_stopping.counter} | lr={get_learning_rate(optimizer):.6g}"
        )

        if should_stop:
            log(f"Early stopping triggered at epoch {epoch}.")
            break

    log(f"Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    evaluate_and_export(
        args=args,
        model=model,
        datasets=datasets,
        raw_frames=raw_frames,
        scalers=scalers,
        expected_delta=expected_delta,
        results_dir=results_dir,
        device=device,
        checkpoint_path=checkpoint_path,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        history=history,
        training_run_stats=training_run_stats,
    )


if __name__ == "__main__":
    main()
