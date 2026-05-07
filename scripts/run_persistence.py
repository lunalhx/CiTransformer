from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.baseline import PersistenceBaseline
from scripts.run_lstm import (
    build_prediction_dataframe,
    collect_predictions,
    create_data_loader,
    evaluate_prediction_arrays,
    format_metric_for_console,
    get_device,
    log,
    prepare_datasets,
    print_dataset_summaries,
    save_prediction_plot,
    set_random_seed,
)
from utils.datasets import DEFAULT_DATA_DIR, DEFAULT_FEATURE_COLUMNS, DEFAULT_TARGET_COLUMN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the persistence / naive baseline for PV power forecasting."
    )

    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Directory containing split CSV files.")
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

    parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size used for evaluation.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Kept for consistency with other baselines.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Evaluation device. 'auto' picks cuda -> mps -> cpu.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/d1_long_no_wind_2015_2022/persistence",
        help="Directory for metrics/predictions/plots.",
    )
    parser.add_argument(
        "--max_eval_batches",
        type=int,
        default=None,
        help="Optional debug cap for validation/test batches.",
    )
    return parser.parse_args()


def resolve_results_dir(results_dir: str) -> Path:
    path = Path(results_dir)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def get_target_feature_index(feature_cols: list[str], target_col: str) -> int:
    if target_col not in feature_cols:
        raise ValueError(
            f"Persistence baseline requires `{target_col}` to appear in feature_cols so the last observed "
            "target value can be copied forward."
        )
    return int(feature_cols.index(target_col))


def build_persistence_model(
    feature_cols: list[str],
    target_col: str,
    pred_len: int,
    feature_scaler: Any,
    target_scaler: Any,
) -> tuple[PersistenceBaseline, int]:
    target_feature_index = get_target_feature_index(feature_cols, target_col)

    model = PersistenceBaseline(
        target_feature_index=target_feature_index,
        pred_len=pred_len,
        feature_mean=float(feature_scaler.mean_[target_feature_index]),
        feature_scale=float(feature_scaler.scale_[target_feature_index]),
        target_mean=float(target_scaler.mean_[0]),
        target_scale=float(target_scaler.scale_[0]),
    )
    return model, target_feature_index


@torch.no_grad()
def evaluate_split(
    split_name: str,
    dataset: Any,
    loader: Any,
    model: PersistenceBaseline,
    target_scaler: Any,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[dict[str, Any], Any]:
    prediction_dict = collect_predictions(
        model=model,
        loader=loader,
        target_scaler=target_scaler,
        device=device,
        split_name=split_name,
        max_batches=max_batches,
    )
    metrics = evaluate_prediction_arrays(
        y_true=prediction_dict["y_true"],
        y_pred=prediction_dict["y_pred"],
        day_night_label=prediction_dict["target_day_night"],
    )
    prediction_df = build_prediction_dataframe(
        prediction_dict=prediction_dict,
        target_timezone=dataset.timezone,
    )
    return metrics, prediction_df


def log_metrics(split_name: str, metrics: dict[str, Any]) -> None:
    all_metrics = metrics["all_timestamps"]
    daytime_metrics = metrics["daytime_only"]

    log(f"\n{split_name.title()} metrics")
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
    set_random_seed(args.seed)
    device = get_device(args.device)

    results_dir = resolve_results_dir(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    datasets, raw_frames, scalers, expected_delta = prepare_datasets(args)
    print_dataset_summaries(datasets)

    if len(datasets["validation"]) == 0:
        raise RuntimeError("Validation dataset contains no valid windows. Reduce seq_len/pred_len or inspect segmentation.")
    if len(datasets["test"]) == 0:
        raise RuntimeError("Test dataset contains no valid windows. Reduce seq_len/pred_len or inspect segmentation.")

    feature_cols = list(args.feature_cols)
    model, target_feature_index = build_persistence_model(
        feature_cols=feature_cols,
        target_col=args.target_col,
        pred_len=args.pred_len,
        feature_scaler=scalers.feature_scaler,
        target_scaler=scalers.target_scaler,
    )
    model = model.to(device)
    model.eval()

    validation_loader = create_data_loader(datasets["validation"], args.batch_size, False, args.num_workers, device)
    test_loader = create_data_loader(datasets["test"], args.batch_size, False, args.num_workers, device)

    log(f"\nUsing device: {device}")
    log(f"Results directory: {results_dir}")
    log("Calibration split was loaded for protocol consistency but is not used for persistence evaluation.")

    validation_metrics, _ = evaluate_split(
        split_name="validation",
        dataset=datasets["validation"],
        loader=validation_loader,
        model=model,
        target_scaler=scalers.target_scaler,
        device=device,
        max_batches=args.max_eval_batches,
    )
    test_metrics, test_predictions_df = evaluate_split(
        split_name="test",
        dataset=datasets["test"],
        loader=test_loader,
        model=model,
        target_scaler=scalers.target_scaler,
        device=device,
        max_batches=args.max_eval_batches,
    )

    metrics_payload = {
        "config": vars(args),
        "experiment_name": None,
        "tuning_stage": None,
        "device": str(device),
        "baseline_type": "persistence",
        "baseline_definition": "Repeat the last observed Active_Pow in the input window for every future step.",
        "best_epoch": None,
        "best_validation_loss": None,
        "expected_delta_minutes": float(expected_delta / pd.Timedelta(minutes=1)),
        "dataset_summary": {split_name: dataset.summary() for split_name, dataset in datasets.items()},
        "raw_split_rows": {split_name: int(len(df)) for split_name, df in raw_frames.items()},
        "target_feature_index": target_feature_index,
        "trainable_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "report_split": "test",
        "calibration_usage": "loaded_but_unused",
        "validation_metrics": validation_metrics,
        "reported_metrics": test_metrics,
        "test_metrics": test_metrics,
        "history": [],
        "checkpoint_path": None,
    }

    metrics_path = results_dir / "metrics.json"
    predictions_path = results_dir / "predictions.csv"
    plot_path = results_dir / "pred_plot.png"

    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics_payload, fp, ensure_ascii=False, indent=2)

    test_predictions_df.to_csv(predictions_path, index=False)
    save_prediction_plot(test_predictions_df, plot_path, split_name="test", model_label="Persistence Baseline")

    log("\nSaved outputs")
    log(f"- metrics: {metrics_path}")
    log(f"- predictions: {predictions_path}")
    log(f"- plot: {plot_path}")

    log_metrics("validation", validation_metrics)
    log_metrics("test", test_metrics)


if __name__ == "__main__":
    main()
