from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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

from models.baseline import LSTMBaseline
from utils.datasets import (
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

    parser.add_argument("--data_dir", type=str, default="data/processed", help="Directory containing split CSV files.")
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
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument(
        "--log_interval",
        type=int,
        default=200,
        help="Print one training progress log every N batches. Set <=0 to disable batch logs.",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Training device. 'auto' picks cuda -> mps -> cpu.",
    )

    parser.add_argument("--results_dir", type=str, default="results/lstm", help="Directory for metrics/predictions/plots.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/lstm/best_model.pth",
        help="Path to save the best checkpoint.",
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


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> dict[str, float | int | None]:
    if len(y_true) == 0:
        return {
            "count": 0,
            "mae": None,
            "mse": None,
            "rmse": None,
            "smape": None,
            "mape_nonzero": None,
            "nonzero_target_count": 0,
        }

    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    error = y_true - y_pred
    mae = np.mean(np.abs(error))
    mse = np.mean(np.square(error))
    rmse = np.sqrt(mse)
    smape = np.mean(2.0 * np.abs(error) / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100.0

    nonzero_mask = np.abs(y_true) > eps
    if nonzero_mask.any():
        mape = np.mean(np.abs(error[nonzero_mask] / y_true[nonzero_mask])) * 100.0
    else:
        mape = None

    return {
        "count": int(len(y_true)),
        "mae": to_float(mae),
        "mse": to_float(mse),
        "rmse": to_float(rmse),
        "smape": to_float(smape),
        "mape_nonzero": to_float(mape),
        "nonzero_target_count": int(nonzero_mask.sum()),
    }


def format_metric_for_console(value: float | None) -> str:
    if value is None:
        return "nan"
    return f"{value:.6f}"


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
    }


def ns_to_datetime(ns_array: np.ndarray, timezone: Any) -> pd.DatetimeIndex:
    flat_array = np.asarray(ns_array).reshape(-1)
    if timezone is None:
        return pd.to_datetime(flat_array)
    return pd.to_datetime(flat_array, utc=True).tz_convert(timezone)


def save_prediction_plot(predictions_df: pd.DataFrame, output_path: Path, max_points: int = 4000) -> None:
    plot_df = predictions_df.loc[:, ["target_start_time", "y_true_t+1", "y_pred_t+1"]].copy()
    plot_df["target_start_time"] = pd.to_datetime(plot_df["target_start_time"])
    plot_df = plot_df.sort_values("target_start_time")

    if len(plot_df) > max_points:
        step = max(1, len(plot_df) // max_points)
        plot_df = plot_df.iloc[::step].copy()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(plot_df["target_start_time"], plot_df["y_true_t+1"], label="Ground Truth", linewidth=1.2)
    ax.plot(plot_df["target_start_time"], plot_df["y_pred_t+1"], label="Prediction", linewidth=1.2)
    ax.set_title("LSTM Baseline Prediction on Test Set (t+1)")
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
    max_batches: int | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    progress = tqdm(loader, desc=f"Train Epoch {epoch:03d}", leave=True, dynamic_ncols=True)
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
        progress.set_postfix(loss=f"{loss.item():.6f}")

        if log_interval > 0 and (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / max(total_samples, 1)
            log(
                f"[Epoch {epoch:03d}] train batch {batch_idx + 1}/{len(loader)} "
                f"| batch_loss={loss.item():.6f} avg_loss={avg_loss:.6f}"
            )

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split_name: str,
    epoch: int | None = None,
    max_batches: int | None = None,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    description = f"{split_name.title()} Eval"
    if epoch is not None:
        description = f"{split_name.title()} Epoch {epoch:03d}"

    progress = tqdm(loader, desc=description, leave=True, dynamic_ncols=True)
    for batch_idx, batch in enumerate(progress):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x, y = move_batch_to_device(batch, device)
        predictions = model(x)
        loss = criterion(predictions, y)

        batch_size = x.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        progress.set_postfix(loss=f"{loss.item():.6f}")

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    target_scaler: Any,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, np.ndarray]:
    model.eval()

    pred_batches: list[np.ndarray] = []
    true_batches: list[np.ndarray] = []
    input_start_batches: list[np.ndarray] = []
    input_end_batches: list[np.ndarray] = []
    target_time_batches: list[np.ndarray] = []
    target_day_batches: list[np.ndarray] = []

    progress = tqdm(loader, desc="Test Predict", leave=True, dynamic_ncols=True)
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


def prepare_datasets(args: argparse.Namespace) -> tuple[dict[str, ContinuousSegmentTimeSeriesDataset], dict[str, pd.DataFrame], Any, pd.Timedelta]:
    split_dir = resolve_split_dir(args.data_dir)

    train_df = load_split_dataframe(split_dir / "train.csv", time_col=args.time_col)
    validation_df = load_split_dataframe(split_dir / "validation.csv", time_col=args.time_col)
    calibration_df = load_split_dataframe(split_dir / "calibration.csv", time_col=args.time_col)
    test_df = load_split_dataframe(split_dir / "test.csv", time_col=args.time_col)

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
        "test": ContinuousSegmentTimeSeriesDataset(
            df=test_df,
            feature_cols=feature_cols,
            target_col=args.target_col,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            feature_scaler=scalers.feature_scaler,
            target_scaler=scalers.target_scaler,
            expected_delta=expected_delta,
        ),
    }

    raw_frames = {
        "train": train_df,
        "validation": validation_df,
        "calibration": calibration_df,
        "test": test_df,
    }
    return datasets, raw_frames, scalers, expected_delta


def print_dataset_summaries(datasets: dict[str, ContinuousSegmentTimeSeriesDataset]) -> None:
    log("\nDataset summary")
    log("-" * 80)
    for split_name, dataset in datasets.items():
        summary = dataset.summary()
        log(f"{split_name:>11s}: {json.dumps(summary, ensure_ascii=False)}")
    log("-" * 80)


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)
    device = get_device(args.device)

    results_dir = PROJECT_ROOT / args.results_dir
    checkpoint_path = PROJECT_ROOT / args.checkpoint_path
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    datasets, raw_frames, scalers, expected_delta = prepare_datasets(args)
    print_dataset_summaries(datasets)

    if len(datasets["train"]) == 0:
        raise RuntimeError("Train dataset contains no valid windows. Reduce seq_len/pred_len or inspect segmentation.")
    if len(datasets["validation"]) == 0:
        raise RuntimeError("Validation dataset contains no valid windows. Reduce seq_len/pred_len or inspect segmentation.")
    if len(datasets["test"]) == 0:
        raise RuntimeError("Test dataset contains no valid windows. Reduce seq_len/pred_len or inspect segmentation.")

    train_loader = create_data_loader(datasets["train"], args.batch_size, True, args.num_workers, device)
    validation_loader = create_data_loader(datasets["validation"], args.batch_size, False, args.num_workers, device)
    test_loader = create_data_loader(datasets["test"], args.batch_size, False, args.num_workers, device)

    model = LSTMBaseline(
        input_size=len(args.feature_cols),
        hidden_size=args.hidden_size,
        pred_len=args.pred_len,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

    history: list[dict[str, float | int]] = []
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
            max_batches=args.max_train_batches,
        )
        val_loss = evaluate_loss(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
            split_name="validation",
            epoch=epoch,
            max_batches=args.max_eval_batches,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "validation_loss": float(val_loss),
            }
        )
        log(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

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

        if early_stopping.step(val_loss):
            log(f"Early stopping triggered at epoch {epoch}.")
            break

    log(f"Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_prediction_dict = collect_predictions(
        model=model,
        loader=test_loader,
        target_scaler=scalers.target_scaler,
        device=device,
        max_batches=args.max_eval_batches,
    )
    test_metrics = evaluate_prediction_arrays(
        y_true=test_prediction_dict["y_true"],
        y_pred=test_prediction_dict["y_pred"],
        day_night_label=test_prediction_dict["target_day_night"],
    )
    predictions_df = build_prediction_dataframe(
        prediction_dict=test_prediction_dict,
        target_timezone=datasets["test"].timezone,
    )

    metrics_payload = {
        "config": vars(args),
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "expected_delta_minutes": float(expected_delta / pd.Timedelta(minutes=1)),
        "dataset_summary": {split_name: dataset.summary() for split_name, dataset in datasets.items()},
        "raw_split_rows": {split_name: int(len(df)) for split_name, df in raw_frames.items()},
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
    save_prediction_plot(predictions_df, plot_path)

    log("\nSaved outputs")
    log(f"- metrics: {metrics_path}")
    log(f"- predictions: {predictions_path}")
    log(f"- plot: {plot_path}")
    log(f"- checkpoint: {checkpoint_path}")

    all_metrics = test_metrics["all_timestamps"]
    daytime_metrics = test_metrics["daytime_only"]
    log("\nTest metrics")
    log(
        "all timestamps | "
        f"MAE={format_metric_for_console(all_metrics['mae'])} "
        f"MSE={format_metric_for_console(all_metrics['mse'])} "
        f"RMSE={format_metric_for_console(all_metrics['rmse'])} "
        f"sMAPE={format_metric_for_console(all_metrics['smape'])} "
        f"MAPE(nonzero)={format_metric_for_console(all_metrics['mape_nonzero'])}"
    )
    log(
        "daytime only  | "
        f"MAE={format_metric_for_console(daytime_metrics['mae'])} "
        f"MSE={format_metric_for_console(daytime_metrics['mse'])} "
        f"RMSE={format_metric_for_console(daytime_metrics['rmse'])} "
        f"sMAPE={format_metric_for_console(daytime_metrics['smape'])} "
        f"MAPE(nonzero)={format_metric_for_console(daytime_metrics['mape_nonzero'])}"
    )


if __name__ == "__main__":
    main()
