from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.baseline import ITransformerBaseline
from scripts.run_lstm import (
    EarlyStopping,
    build_prediction_dataframe,
    collect_predictions,
    create_data_loader,
    evaluate_loss,
    evaluate_prediction_arrays,
    format_metric_for_console,
    get_device,
    log,
    prepare_datasets,
    print_dataset_summaries,
    save_checkpoint,
    set_random_seed,
    train_one_epoch,
)
from utils.datasets import DEFAULT_DATA_DIR, DEFAULT_FEATURE_COLUMNS, DEFAULT_TARGET_COLUMN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the vanilla iTransformer baseline for PV power forecasting."
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

    parser.add_argument("--d_model", type=int, default=128, help="iTransformer hidden dimension.")
    parser.add_argument("--n_heads", type=int, default=4, help="iTransformer attention head count.")
    parser.add_argument("--e_layers", type=int, default=2, help="iTransformer encoder layer count.")
    parser.add_argument("--d_ff", type=int, default=256, help="iTransformer feed-forward dimension.")
    parser.add_argument("--factor", type=int, default=5, help="iTransformer attention factor.")
    parser.add_argument(
        "--activation",
        type=str,
        default="gelu",
        choices=["relu", "gelu"],
        help="iTransformer feed-forward activation.",
    )
    parser.add_argument(
        "--disable_norm",
        action="store_true",
        help="Disable the built-in normalization / de-normalization used by vanilla iTransformer.",
    )
    parser.add_argument(
        "--output_attention",
        action="store_true",
        help="Compute attention maps inside iTransformer. Training still uses only the forecast tensor.",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout used in attention and FFN blocks.")
    parser.add_argument(
        "--causal_graph_dir",
        type=str,
        default=None,
        help=(
            "Optional directory containing global_causal_adjacency.csv. When provided, the 2D variable-level "
            "causal mask is injected into iTransformer attention."
        ),
    )

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
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Training device. 'auto' picks cuda -> mps -> cpu.",
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/itransformer",
        help="Directory for metrics/predictions/plots.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/itransformer/best_model.pth",
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


def get_target_feature_index(feature_cols: list[str], target_col: str) -> int:
    if target_col not in feature_cols:
        raise ValueError(
            f"Target column `{target_col}` must appear in feature_cols for iTransformer because the vanilla backbone "
            "predicts the same variates that appear in the encoder input."
        )
    return int(feature_cols.index(target_col))


def load_causal_attention_mask(
    causal_graph_dir: str | None,
    feature_cols: list[str],
) -> tuple[torch.Tensor | None, dict[str, Any] | None]:
    if causal_graph_dir is None:
        return None, None

    graph_dir = resolve_path(causal_graph_dir)
    adjacency_path = graph_dir / "global_causal_adjacency.csv"
    if not adjacency_path.exists():
        raise FileNotFoundError(f"Cannot find causal adjacency file: {adjacency_path}")

    adjacency = pd.read_csv(adjacency_path, index_col=0)
    adjacency.index = adjacency.index.astype(str)
    adjacency.columns = adjacency.columns.astype(str)

    expected_order = list(feature_cols)
    index_order = list(adjacency.index)
    column_order = list(adjacency.columns)
    if index_order != expected_order or column_order != expected_order:
        raise ValueError(
            "Causal adjacency variables must exactly match --feature_cols order.\n"
            f"Expected: {expected_order}\n"
            f"Rows:     {index_order}\n"
            f"Columns:  {column_order}"
        )

    matrix = adjacency.to_numpy(dtype=np.float32)
    if matrix.shape != (len(expected_order), len(expected_order)):
        raise ValueError(
            f"Causal adjacency must have shape {(len(expected_order), len(expected_order))}, got {matrix.shape}."
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"Causal adjacency contains non-finite values: {adjacency_path}")
    if np.any(np.diag(matrix) <= 0.0):
        raise ValueError("Causal adjacency diagonal must be positive so every variable can attend to itself.")

    additive_mask = np.where(matrix > 0.0, 0.0, -1e9).astype(np.float32)
    allowed_positions = int(np.sum(matrix > 0.0))
    total_positions = int(matrix.size)
    metadata = {
        "causal_graph_dir": str(graph_dir),
        "causal_adjacency_path": str(adjacency_path),
        "feature_order": expected_order,
        "mask_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "allowed_attention_positions": allowed_positions,
        "total_attention_positions": total_positions,
        "mask_density": float(allowed_positions / total_positions),
        "mask_format": "additive attention mask, 0.0=allowed, -1e9=blocked",
    }
    return torch.from_numpy(additive_mask), metadata


def build_itransformer_model(args: argparse.Namespace, scalers: Any) -> ITransformerBaseline:
    feature_cols = list(args.feature_cols)
    target_feature_index = get_target_feature_index(feature_cols, args.target_col)
    causal_attention_mask, _ = load_causal_attention_mask(getattr(args, "causal_graph_dir", None), feature_cols)

    return ITransformerBaseline(
        target_feature_index=target_feature_index,
        feature_mean=float(scalers.feature_scaler.mean_[target_feature_index]),
        feature_scale=float(scalers.feature_scaler.scale_[target_feature_index]),
        target_mean=float(scalers.target_scaler.mean_[0]),
        target_scale=float(scalers.target_scaler.scale_[0]),
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_ff=args.d_ff,
        factor=args.factor,
        dropout=args.dropout,
        activation=args.activation,
        output_attention=args.output_attention,
        use_norm=not args.disable_norm,
        causal_attention_mask=causal_attention_mask,
    )


def save_prediction_plot(
    predictions_df: pd.DataFrame,
    output_path: Path,
    split_name: str,
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
    ax.set_title(f"iTransformer Baseline Prediction on {split_name.title()} Set (t+1)")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Active_Pow")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


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
    if cli_args.causal_graph_dir is not None:
        eval_args.causal_graph_dir = cli_args.causal_graph_dir
    elif not hasattr(eval_args, "causal_graph_dir"):
        eval_args.causal_graph_dir = None
    if cli_args.max_eval_batches is not None:
        eval_args.max_eval_batches = cli_args.max_eval_batches
    eval_args.progress_mininterval = cli_args.progress_mininterval
    eval_args.num_workers = cli_args.num_workers
    eval_args.tuning_only = False
    eval_args.eval_checkpoint_path = cli_args.eval_checkpoint_path
    return eval_args


def evaluate_and_export(
    args: argparse.Namespace,
    model: ITransformerBaseline,
    datasets: dict[str, Any],
    raw_frames: dict[str, pd.DataFrame],
    scalers: Any,
    expected_delta: pd.Timedelta,
    results_dir: Path,
    device: torch.device,
    checkpoint_path: Path,
    best_epoch: int,
    best_val_loss: float | None,
    history: list[dict[str, float | int]],
) -> None:
    target_feature_index = get_target_feature_index(list(args.feature_cols), args.target_col)
    _, causal_mask_metadata = load_causal_attention_mask(
        getattr(args, "causal_graph_dir", None),
        list(args.feature_cols),
    )
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
        "baseline_type": "itransformer",
        "baseline_definition": (
            "iTransformer backbone with Active_Pow extracted from the multivariate forecast output"
            + (" and a global PCMCI variable-level attention mask." if causal_mask_metadata else ".")
        ),
        "causal_mask": causal_mask_metadata,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "expected_delta_minutes": float(expected_delta / pd.Timedelta(minutes=1)),
        "dataset_summary": {split_name: dataset.summary() for split_name, dataset in datasets.items()},
        "raw_split_rows": {split_name: int(len(df)) for split_name, df in raw_frames.items()},
        "target_feature_index": target_feature_index,
        "trainable_parameter_count": trainable_parameter_count,
        "report_split": args.report_split,
        "calibration_usage": "loaded_but_unused",
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
    save_prediction_plot(predictions_df, plot_path, split_name=reported_split_name)

    log("\nSaved outputs")
    log(f"- metrics: {metrics_path}")
    log(f"- predictions: {predictions_path}")
    log(f"- plot: {plot_path}")
    log(f"- checkpoint: {checkpoint_path}")
    if causal_mask_metadata is not None:
        log(f"- causal mask: {causal_mask_metadata['causal_adjacency_path']}")

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

    if args.eval_checkpoint_path is not None:
        checkpoint_path = resolve_path(args.eval_checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_args = checkpoint.get("args")
        if checkpoint_args is None:
            raise KeyError(f"Checkpoint {checkpoint_path} does not contain saved args.")

        args = build_eval_args_from_checkpoint(args, checkpoint_args)
        set_random_seed(int(args.seed))
        device = get_device(args.device)
        results_dir = resolve_path(args.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        include_test = args.report_split == "test"
        datasets, raw_frames, scalers, expected_delta = prepare_datasets(args, include_test=include_test)
        print_dataset_summaries(datasets)

        if len(datasets["validation"]) == 0:
            raise RuntimeError("Validation dataset contains no valid windows. Reduce seq_len/pred_len or inspect segmentation.")
        if args.report_split == "test" and len(datasets["test"]) == 0:
            raise RuntimeError("Test dataset contains no valid windows. Reduce seq_len/pred_len or inspect segmentation.")

        model = build_itransformer_model(args, scalers).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        log(f"\nUsing device: {device}")
        log(f"Results directory: {results_dir}")
        log(f"Checkpoint path: {checkpoint_path}")

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
            best_epoch=int(checkpoint.get("epoch", 0)),
            best_val_loss=float(checkpoint.get("best_val_loss")) if checkpoint.get("best_val_loss") is not None else None,
            history=[],
        )
        return

    set_random_seed(args.seed)
    device = get_device(args.device)

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

    model = build_itransformer_model(args, scalers).to(device)
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
    )


if __name__ == "__main__":
    main()
