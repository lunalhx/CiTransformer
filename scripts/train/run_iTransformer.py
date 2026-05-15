from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.project_config import load_project_config, resolve_project_path

PROJECT_CONFIG = load_project_config()
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_CONFIG.get_path("paths.matplotlib_cache")))

import matplotlib.pyplot as plt

from models.baseline import ITransformerBaseline
from scripts.train.run_lstm import (
    EarlyStopping,
    build_prediction_dataframe,
    collect_predictions,
    create_data_loader,
    compute_regression_metrics,
    evaluate_loss,
    evaluate_prediction_arrays,
    format_metric_for_console,
    get_device,
    get_learning_rate,
    log,
    prepare_datasets,
    print_dataset_summaries,
    save_checkpoint,
    set_random_seed,
    train_one_epoch,
)
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
class RegimeModelArtifact:
    path: Path
    model: Any
    scaler: Any
    feature_columns: list[str]
    day_column: str
    start_probability: np.ndarray
    transition_matrix: np.ndarray
    daytime_regime_offset: int

    @property
    def daytime_regime_count(self) -> int:
        return int(self.start_probability.shape[0])

    @property
    def total_regime_count(self) -> int:
        return self.daytime_regime_count + self.daytime_regime_offset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the vanilla iTransformer baseline for PV power forecasting."
    )

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
    parser.add_argument(
        "--causal_mask_mode",
        type=str,
        default="hard",
        choices=["none", "hard", "soft_bias", "causal_reward"],
        help=(
            "How to inject the causal graph into attention. 'hard' blocks non-edges, 'soft_bias' adds a finite "
            "negative bias to non-edges, 'causal_reward' adds a finite positive bias to retained causal edges, "
            "and 'none' disables causal attention masking."
        ),
    )
    parser.add_argument(
        "--causal_mask_beta",
        type=float,
        default=1.0,
        help="Finite negative bias magnitude used only when --causal_mask_mode soft_bias.",
    )
    parser.add_argument(
        "--causal_gamma",
        type=float,
        default=1.0,
        help="Positive causal edge reward magnitude used only when --causal_mask_mode causal_reward.",
    )
    parser.add_argument(
        "--causal_reward_strength",
        type=str,
        default="max_abs_mci",
        choices=["max_abs_mci"],
        help="Edge strength column used by --causal_mask_mode causal_reward.",
    )
    parser.add_argument(
        "--causal_strength_normalization",
        type=str,
        default="per_target_max",
        choices=["per_target_max"],
        help="Strength normalization used by --causal_mask_mode causal_reward.",
    )
    parser.add_argument(
        "--regime_graph_root",
        type=str,
        default=None,
        help=(
            "Optional root directory containing regime_*/global_causal_adjacency.csv. "
            "When provided, per-sample regime masks are probability-weighted from online HMM state forecasts."
        ),
    )
    parser.add_argument(
        "--regime_model_path",
        type=str,
        default=str(
            PROJECT_CONFIG.get_path("paths.results_root")
            / "regimes"
            / "gmm_hmm_daytime_k7"
            / "gmm_hmm_regime_model.pkl"
        ),
        help=(
            "Train-only GMM-HMM artifact used for online regime posterior filtering when --regime_graph_root is set."
        ),
    )
    parser.add_argument(
        "--regime_mask_strategy",
        type=str,
        default="transition_weighted",
        choices=["transition_weighted"],
        help="Regime dynamic mask strategy. Uses online HMM alpha_T and transition-weighted future regime probabilities.",
    )

    parser.add_argument("--epochs", type=int, default=30, help="Maximum training epochs.")
    parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Adam weight decay.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm. Set <=0 to disable.")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience.")
    parser.add_argument("--min_delta", type=float, default=1e-5, help="Minimum validation loss improvement.")
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="plateau",
        choices=["plateau", "none"],
        help="Learning-rate scheduler. Use 'none' to keep a fixed learning rate.",
    )
    parser.add_argument(
        "--lr_plateau_factor",
        type=float,
        default=0.5,
        help="Factor for ReduceLROnPlateau when validation loss plateaus.",
    )
    parser.add_argument(
        "--lr_plateau_patience",
        type=int,
        default=2,
        help="Number of stagnant validation epochs before reducing learning rate.",
    )
    parser.add_argument(
        "--lr_plateau_min_lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate for ReduceLROnPlateau.",
    )
    parser.add_argument(
        "--lr_plateau_threshold",
        type=float,
        default=None,
        help="Absolute validation-loss improvement threshold for ReduceLROnPlateau. Defaults to --min_delta.",
    )
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
                "paths.results.itransformer",
                "results/d1_long_no_wind_2015_2022/itransformer",
            )
        ),
        help="Directory for metrics/predictions/plots.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=str(
            PROJECT_CONFIG.get_path(
                "paths.checkpoints.itransformer",
                "checkpoints/d1_long_no_wind_2015_2022/itransformer",
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
    mode: str = "hard",
    beta: float = 1.0,
    gamma: float = 1.0,
    reward_strength: str = "max_abs_mci",
    strength_normalization: str = "per_target_max",
    d_model: int | None = None,
    n_heads: int | None = None,
) -> tuple[torch.Tensor | None, dict[str, Any] | None]:
    if mode not in {"none", "hard", "soft_bias", "causal_reward"}:
        raise ValueError(f"Unsupported causal_mask_mode: {mode}")
    if mode == "soft_bias" and beta <= 0.0:
        raise ValueError("--causal_mask_beta must be positive when --causal_mask_mode soft_bias.")
    if mode == "causal_reward" and gamma < 0.0:
        raise ValueError("--causal_gamma must be non-negative when --causal_mask_mode causal_reward.")
    if mode == "causal_reward" and reward_strength != "max_abs_mci":
        raise ValueError("Only --causal_reward_strength max_abs_mci is supported for causal_reward.")
    if mode == "causal_reward" and strength_normalization != "per_target_max":
        raise ValueError("Only --causal_strength_normalization per_target_max is supported for causal_reward.")
    if causal_graph_dir is None:
        return None, None

    graph_dir = resolve_path(causal_graph_dir)
    if mode == "causal_reward":
        return load_causal_reward_attention_mask(
            graph_dir=graph_dir,
            feature_cols=feature_cols,
            gamma=gamma,
            reward_strength=reward_strength,
            strength_normalization=strength_normalization,
            d_model=d_model,
            n_heads=n_heads,
        )

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

    if mode == "none":
        additive_mask = None
        mask_format = "none; causal graph loaded for metadata only"
    elif mode == "hard":
        additive_mask = np.where(matrix > 0.0, 0.0, -1e9).astype(np.float32)
        mask_format = "hard additive attention mask, 0.0=allowed, -1e9=blocked"
    else:
        if d_model is None or n_heads is None:
            raise ValueError("d_model and n_heads are required when --causal_mask_mode soft_bias.")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}.")
        head_dim = d_model // n_heads
        # FullAttention scales scores after applying additive masks, so compensate here
        # to make beta approximately equal to the final softmax-logit penalty.
        pre_scale_penalty = float(beta) * math.sqrt(float(head_dim))
        additive_mask = np.where(matrix > 0.0, 0.0, -pre_scale_penalty).astype(np.float32)
        mask_format = (
            f"soft additive attention bias, 0.0=allowed, approximately -{float(beta):g} final-logit "
            f"non-edge bias before softmax"
        )

    allowed_positions = int(np.sum(matrix > 0.0))
    total_positions = int(matrix.size)
    metadata = {
        "causal_graph_dir": str(graph_dir),
        "causal_adjacency_path": str(adjacency_path),
        "mask_mode": mode,
        "mask_beta": float(beta) if mode == "soft_bias" else None,
        "pre_scale_non_edge_bias": (
            float(additive_mask[matrix <= 0.0][0])
            if additive_mask is not None and np.any(matrix <= 0.0)
            else None
        ),
        "mask_applied": mode != "none",
        "feature_order": expected_order,
        "mask_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "allowed_attention_positions": allowed_positions,
        "total_attention_positions": total_positions,
        "mask_density": float(allowed_positions / total_positions),
        "mask_format": mask_format,
    }
    return torch.from_numpy(additive_mask) if additive_mask is not None else None, metadata


def load_causal_reward_attention_mask(
    graph_dir: Path,
    feature_cols: list[str],
    gamma: float,
    reward_strength: str,
    strength_normalization: str,
    d_model: int | None,
    n_heads: int | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if d_model is None or n_heads is None:
        raise ValueError("d_model and n_heads are required when --causal_mask_mode causal_reward.")
    if d_model % n_heads != 0:
        raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}.")

    edges_path = graph_dir / "topk_final_edges.csv"
    if not edges_path.exists():
        raise FileNotFoundError(f"Cannot find causal reward edge file: {edges_path}")

    edges = pd.read_csv(edges_path)
    required_columns = {"source", "target", reward_strength}
    missing_columns = sorted(required_columns - set(edges.columns))
    if missing_columns:
        raise ValueError(f"{edges_path} is missing required columns: {missing_columns}")

    feature_to_index = {name: index for index, name in enumerate(feature_cols)}
    feature_count = len(feature_cols)
    raw_strength = np.zeros((feature_count, feature_count), dtype=np.float32)
    invalid_edges: list[dict[str, Any]] = []

    for row in edges.loc[:, ["source", "target", reward_strength]].itertuples(index=False):
        source = str(row.source)
        target = str(row.target)
        if source not in feature_to_index or target not in feature_to_index:
            invalid_edges.append({"source": source, "target": target})
            continue
        strength = float(getattr(row, reward_strength))
        if not math.isfinite(strength):
            raise ValueError(f"{edges_path} contains non-finite {reward_strength} for {source}->{target}.")
        if strength < 0.0:
            raise ValueError(f"{edges_path} contains negative {reward_strength} for {source}->{target}.")
        target_index = feature_to_index[target]
        source_index = feature_to_index[source]
        raw_strength[target_index, source_index] = max(raw_strength[target_index, source_index], strength)

    if invalid_edges:
        raise ValueError(
            f"{edges_path} contains edges outside --feature_cols. First invalid edges: {invalid_edges[:5]}"
        )

    normalized_strength = np.zeros_like(raw_strength, dtype=np.float32)
    row_maxima = raw_strength.max(axis=1)
    non_empty_rows = row_maxima > 0.0
    normalized_strength[non_empty_rows] = (
        raw_strength[non_empty_rows] / row_maxima[non_empty_rows, None]
    )

    head_dim = int(d_model) // int(n_heads)
    pre_scale_multiplier = float(gamma) * math.sqrt(float(head_dim))
    additive_mask = (pre_scale_multiplier * normalized_strength).astype(np.float32)
    final_reward = (float(gamma) * normalized_strength).astype(np.float32)
    positive_reward = final_reward[final_reward > 0.0]

    reward_edge_count = int(np.count_nonzero(normalized_strength > 0.0))
    total_positions = int(normalized_strength.size)
    metadata = {
        "causal_graph_dir": str(graph_dir),
        "causal_edges_path": str(edges_path),
        "mask_mode": "causal_reward",
        "mask_beta": None,
        "causal_gamma": float(gamma),
        "causal_reward_strength": reward_strength,
        "causal_strength_normalization": strength_normalization,
        "reward_edge_count": reward_edge_count,
        "reward_min": float(positive_reward.min()) if positive_reward.size else 0.0,
        "reward_max": float(final_reward.max()) if final_reward.size else 0.0,
        "pre_scale_reward_max": float(additive_mask.max()) if additive_mask.size else 0.0,
        "mask_applied": True,
        "feature_order": list(feature_cols),
        "mask_shape": [int(feature_count), int(feature_count)],
        "allowed_attention_positions": reward_edge_count,
        "total_attention_positions": total_positions,
        "mask_density": float(reward_edge_count / total_positions) if total_positions else 0.0,
        "reward_matrix_density": float(reward_edge_count / total_positions) if total_positions else 0.0,
        "normalization_non_empty_target_count": int(np.count_nonzero(non_empty_rows)),
        "mask_format": (
            "causal edge reward additive attention bias; non-edges are 0.0 and retained edges receive "
            f"approximately +{float(gamma):g} max final-logit reward per target before softmax"
        ),
    }
    return torch.from_numpy(additive_mask), metadata


def build_itransformer_model(args: argparse.Namespace, scalers: Any) -> ITransformerBaseline:
    feature_cols = list(args.feature_cols)
    target_feature_index = get_target_feature_index(feature_cols, args.target_col)
    causal_attention_mask, _ = load_causal_attention_mask(
        getattr(args, "causal_graph_dir", None),
        feature_cols,
        mode=getattr(args, "causal_mask_mode", "hard"),
        beta=float(getattr(args, "causal_mask_beta", 1.0)),
        gamma=float(getattr(args, "causal_gamma", 1.0)),
        reward_strength=str(getattr(args, "causal_reward_strength", "max_abs_mci")),
        strength_normalization=str(getattr(args, "causal_strength_normalization", "per_target_max")),
        d_model=int(getattr(args, "d_model", 0)),
        n_heads=int(getattr(args, "n_heads", 0)),
    )

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
    return resolve_project_path(path_value, PROJECT_ROOT)


def normalize_probability_vector(values: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float64).copy()
    probabilities[~np.isfinite(probabilities)] = 0.0
    probabilities = np.maximum(probabilities, 0.0)
    total = float(probabilities.sum())
    if total > 0.0:
        return (probabilities / total).astype(np.float32)

    if fallback is not None:
        fallback_probabilities = np.asarray(fallback, dtype=np.float64).copy()
        fallback_probabilities[~np.isfinite(fallback_probabilities)] = 0.0
        fallback_probabilities = np.maximum(fallback_probabilities, 0.0)
        fallback_total = float(fallback_probabilities.sum())
        if fallback_total > 0.0:
            return (fallback_probabilities / fallback_total).astype(np.float32)

    return np.full_like(probabilities, 1.0 / max(len(probabilities), 1), dtype=np.float32)


def probabilities_from_log_likelihood(log_likelihood: np.ndarray) -> np.ndarray:
    log_values = np.asarray(log_likelihood, dtype=np.float64)
    finite_mask = np.isfinite(log_values)
    if not finite_mask.any():
        return np.full(log_values.shape, 1.0 / max(log_values.size, 1), dtype=np.float32)
    shifted = log_values - float(np.max(log_values[finite_mask]))
    probabilities = np.exp(shifted)
    probabilities[~finite_mask] = 0.0
    return normalize_probability_vector(probabilities)


def load_regime_model_artifact(regime_model_path: str | None) -> RegimeModelArtifact | None:
    if regime_model_path is None:
        return None

    artifact_path = resolve_path(regime_model_path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Cannot find regime model artifact: {artifact_path}")

    with artifact_path.open("rb") as fp:
        artifact = pickle.load(fp)

    model = artifact.get("model")
    scaler = artifact.get("scaler")
    feature_columns = list(artifact.get("feature_columns") or [])
    if model is None or scaler is None or not feature_columns:
        raise ValueError(
            f"{artifact_path} must contain model, scaler, and feature_columns from train-only GMM-HMM discovery."
        )
    if not hasattr(model, "_compute_log_likelihood"):
        raise TypeError("The loaded regime model must expose hmmlearn-style _compute_log_likelihood for online filtering.")

    start_probability = normalize_probability_vector(np.asarray(model.startprob_, dtype=np.float64))
    transition_matrix = np.asarray(model.transmat_, dtype=np.float64)
    if transition_matrix.ndim != 2 or transition_matrix.shape[0] != transition_matrix.shape[1]:
        raise ValueError(f"Regime transition matrix must be square, got {transition_matrix.shape}.")
    if transition_matrix.shape[0] != start_probability.shape[0]:
        raise ValueError(
            "Regime start probability and transition matrix disagree: "
            f"start={start_probability.shape}, transition={transition_matrix.shape}."
        )
    transition_matrix = np.vstack(
        [
            normalize_probability_vector(row, fallback=start_probability)
            for row in transition_matrix
        ]
    ).astype(np.float32)

    daytime_regime_offset = int(artifact.get("daytime_regime_offset", 1))
    if daytime_regime_offset != 1:
        raise ValueError(
            f"Only daytime_regime_offset=1 is supported for regime mask weighting, got {daytime_regime_offset}."
        )

    return RegimeModelArtifact(
        path=artifact_path,
        model=model,
        scaler=scaler,
        feature_columns=feature_columns,
        day_column=str(artifact.get("day_column", "day_night_label")),
        start_probability=start_probability,
        transition_matrix=transition_matrix,
        daytime_regime_offset=daytime_regime_offset,
    )


def compute_online_regime_posteriors(
    df: pd.DataFrame,
    regime_artifact: RegimeModelArtifact,
    expected_delta: pd.Timedelta,
) -> np.ndarray:
    missing_columns = [column for column in regime_artifact.feature_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Split is missing GMM-HMM regime feature columns: {missing_columns}")

    if regime_artifact.day_column in df.columns:
        day_mask = pd.to_numeric(df[regime_artifact.day_column], errors="coerce").fillna(0).to_numpy(dtype=np.int64) == 1
    else:
        day_mask = np.ones(len(df), dtype=bool)

    feature_values = df[regime_artifact.feature_columns].to_numpy(dtype=float)
    scaled_values = regime_artifact.scaler.transform(feature_values)
    # Emission likelihoods are row-local under the fitted HMM; only the alpha recursion below carries temporal state.
    log_likelihood = regime_artifact.model._compute_log_likelihood(scaled_values)
    if log_likelihood.shape != (len(df), regime_artifact.daytime_regime_count):
        raise ValueError(
            "GMM-HMM emission log-likelihood shape mismatch: "
            f"got {log_likelihood.shape}, expected {(len(df), regime_artifact.daytime_regime_count)}."
        )

    posterior = np.zeros((len(df), regime_artifact.total_regime_count), dtype=np.float32)
    day_alpha: np.ndarray | None = None
    index = df.index

    for row_index in range(len(df)):
        has_gap = row_index == 0 or (index[row_index] - index[row_index - 1]) != expected_delta
        if not day_mask[row_index]:
            posterior[row_index, 0] = 1.0
            day_alpha = None
            continue

        if day_alpha is None or has_gap or not day_mask[row_index - 1]:
            prior = regime_artifact.start_probability
        else:
            prior = normalize_probability_vector(day_alpha @ regime_artifact.transition_matrix, fallback=regime_artifact.start_probability)

        emission_probability = probabilities_from_log_likelihood(log_likelihood[row_index])
        day_alpha = normalize_probability_vector(prior * emission_probability, fallback=prior)
        posterior[row_index, 1:] = day_alpha

    return posterior


def prepare_itransformer_datasets(
    args: argparse.Namespace,
    include_test: bool = True,
    regime_artifact: RegimeModelArtifact | None = None,
) -> tuple[dict[str, ContinuousSegmentTimeSeriesDataset], dict[str, pd.DataFrame], Any, pd.Timedelta]:
    if regime_artifact is None:
        return prepare_datasets(args, include_test=include_test)

    split_dir = resolve_split_dir(args.data_dir)
    split_names = ["train", "validation", "calibration"]
    if include_test:
        split_names.append("test")

    raw_frames = {
        split_name: load_split_dataframe(
            split_dir / f"{split_name}.csv",
            time_col=args.time_col,
        )
        for split_name in split_names
    }

    feature_cols = list(args.feature_cols)
    scalers = fit_split_scalers(raw_frames["train"], feature_cols, args.target_col)
    expected_delta = (
        pd.Timedelta(minutes=args.sampling_freq_minutes)
        if args.sampling_freq_minutes is not None
        else infer_expected_timedelta(raw_frames["train"].index)
    )
    regime_posteriors = {
        split_name: compute_online_regime_posteriors(frame, regime_artifact, expected_delta)
        for split_name, frame in raw_frames.items()
    }

    datasets: dict[str, ContinuousSegmentTimeSeriesDataset] = {}
    for split_name, frame in raw_frames.items():
        datasets[split_name] = ContinuousSegmentTimeSeriesDataset(
            df=frame,
            feature_cols=feature_cols,
            target_col=args.target_col,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            feature_scaler=scalers.feature_scaler,
            target_scaler=scalers.target_scaler,
            expected_delta=expected_delta,
            include_regime_labels=False,
            regime_posterior=regime_posteriors[split_name],
            regime_transition_matrix=regime_artifact.transition_matrix,
            regime_start_probability=regime_artifact.start_probability,
        )
    return datasets, raw_frames, scalers, expected_delta


def load_regime_attention_mask_bank(
    regime_graph_root: str | None,
    feature_cols: list[str],
    mode: str,
    beta: float,
    gamma: float,
    reward_strength: str,
    strength_normalization: str,
    d_model: int,
    n_heads: int,
    regime_mask_strategy: str,
    expected_regime_count: int | None = None,
    regime_model_path: str | None = None,
) -> tuple[torch.Tensor | None, dict[str, Any] | None]:
    if regime_graph_root is None:
        return None, None
    if regime_mask_strategy != "transition_weighted":
        raise ValueError("Only --regime_mask_strategy transition_weighted is supported for dynamic regime masking.")
    if mode not in {"soft_bias", "causal_reward"}:
        raise ValueError("Regime-aware dynamic masking requires --causal_mask_mode soft_bias or causal_reward.")
    if mode == "soft_bias" and not math.isclose(float(beta), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Regime-aware dynamic masking is fixed to --causal_mask_beta 1.0.")

    graph_root = resolve_path(regime_graph_root)
    if not graph_root.exists():
        raise FileNotFoundError(f"Cannot find regime graph root: {graph_root}")

    regime_pattern = re.compile(r"^regime_(\d+)$")
    regime_dirs: dict[int, Path] = {}
    for child in graph_root.iterdir():
        if not child.is_dir():
            continue
        match = regime_pattern.match(child.name)
        if match is None:
            continue
        required_path = child / ("topk_final_edges.csv" if mode == "causal_reward" else "global_causal_adjacency.csv")
        if required_path.exists():
            regime_dirs[int(match.group(1))] = child

    if not regime_dirs:
        expected_name = "topk_final_edges.csv" if mode == "causal_reward" else "global_causal_adjacency.csv"
        raise FileNotFoundError(
            f"Cannot find any regime_*/{expected_name} files under {graph_root}."
        )

    feature_count = len(feature_cols)
    max_regime = max(regime_dirs)
    if expected_regime_count is not None:
        max_regime = max(max_regime, int(expected_regime_count) - 1)
    mask_bank = np.zeros((max_regime + 1, feature_count, feature_count), dtype=np.float32)
    regime_metadata: dict[str, Any] = {}
    for regime_id in sorted(regime_dirs):
        if mode == "causal_reward" and regime_id == 0:
            continue
        mask, metadata = load_causal_attention_mask(
            str(regime_dirs[regime_id]),
            feature_cols,
            mode=mode,
            beta=beta,
            gamma=gamma,
            reward_strength=reward_strength,
            strength_normalization=strength_normalization,
            d_model=d_model,
            n_heads=n_heads,
        )
        if mask is None:
            raise RuntimeError(f"Regime {regime_id} did not produce an additive attention mask.")
        mask_bank[regime_id] = mask.numpy().astype(np.float32)
        regime_metadata[str(regime_id)] = metadata

    loaded_regimes = [int(regime_id) for regime_id in sorted(regime_dirs) if not (mode == "causal_reward" and regime_id == 0)]
    if not loaded_regimes:
        raise FileNotFoundError(f"Cannot find any nonzero regime reward graphs under {graph_root}.")

    metadata = {
        "mask_type": (
            "regime_transition_weighted_causal_reward"
            if mode == "causal_reward"
            else "regime_transition_weighted_soft_bias"
        ),
        "mask_applied": True,
        "mask_mode": mode,
        "mask_beta": float(beta) if mode == "soft_bias" else None,
        "causal_gamma": float(gamma) if mode == "causal_reward" else None,
        "causal_reward_strength": reward_strength if mode == "causal_reward" else None,
        "causal_strength_normalization": strength_normalization if mode == "causal_reward" else None,
        "regime_graph_root": str(graph_root),
        "regime_model_path": str(resolve_path(regime_model_path)) if regime_model_path is not None else None,
        "regime_mask_strategy": regime_mask_strategy,
        "offline_full_split_regime_labels_used": False,
        "regime_probability_source": "online_hmm_forward_filter",
        "future_regime_distribution": "hmm_transition_matrix",
        "selection_detail": (
            "For each sample, alpha_T is computed from observations up to encoder_end - 1; "
            "future pi_{T+h|T} is propagated with the train-fitted HMM transition matrix; "
            "a horizon-average regime probability vector weights the mask bank."
        ),
        "fallback_regime": 0,
        "fallback_mask": "all-zero additive attention bias, equivalent to no structural causal constraint",
        "loaded_regimes": loaded_regimes,
        "mask_bank_shape": [int(value) for value in mask_bank.shape],
        "feature_order": list(feature_cols),
        "regime_metadata": regime_metadata,
        "regime_reward_density": (
            {
                regime_id: float(regime_meta.get("reward_matrix_density", 0.0))
                for regime_id, regime_meta in regime_metadata.items()
            }
            if mode == "causal_reward"
            else None
        ),
        "graph_semantics": (
            "Target-regime-conditioned PCMCI graphs are used as leakage-safe dynamic causal reward bias "
            "weighted by online HMM transition probabilities."
            if mode == "causal_reward"
            else (
                "Target-regime-conditioned PCMCI graphs are used as leakage-safe dynamic soft attention bias "
                "weighted by online HMM transition probabilities."
            )
        ),
    }
    return torch.from_numpy(mask_bank), metadata


def build_regime_model_kwargs_fn(
    regime_mask_bank: torch.Tensor | None,
) -> Any:
    if regime_mask_bank is None:
        return None

    def model_kwargs_fn(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
        if "regime_mask_weights" not in batch:
            raise KeyError(
                "Dynamic regime masking requires `regime_mask_weights` in each batch. "
                "Load a train-only regime model via --regime_model_path."
            )
        weights = batch["regime_mask_weights"].to(device=device, dtype=torch.float32)
        if weights.ndim != 2:
            raise ValueError(f"regime_mask_weights must have shape [batch, regimes], got {tuple(weights.shape)}.")
        bank = regime_mask_bank.to(device=device)
        if weights.size(1) > bank.size(0):
            pad_count = weights.size(1) - bank.size(0)
            padding = torch.zeros(
                pad_count,
                bank.size(1),
                bank.size(2),
                device=device,
                dtype=bank.dtype,
            )
            bank = torch.cat([bank, padding], dim=0)
        elif weights.size(1) < bank.size(0):
            bank = bank[: weights.size(1)]
        weights = torch.clamp(weights, min=0.0)
        weights = weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-12)
        weighted_mask = torch.einsum("br,rnm->bnm", weights, bank)
        # Shape [B, 1, N, N] broadcasts over attention heads [B, H, N, N].
        return {"mask": weighted_mask.unsqueeze(1)}

    return model_kwargs_fn


def value_counts_dict(values: np.ndarray) -> dict[str, int]:
    if values.size == 0:
        return {}
    unique, counts = np.unique(values.astype(np.int64), return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(unique, counts, strict=True)}


def compute_target_regime_metrics(prediction_dict: dict[str, np.ndarray]) -> dict[str, Any] | None:
    if "target_regime" not in prediction_dict:
        return None
    flat_true = prediction_dict["y_true"].reshape(-1)
    flat_pred = prediction_dict["y_pred"].reshape(-1)
    flat_regime = prediction_dict["target_regime"].reshape(-1).astype(np.int64)

    metrics: dict[str, Any] = {}
    for regime_id in sorted(np.unique(flat_regime).tolist()):
        regime_mask = flat_regime == int(regime_id)
        metrics[str(int(regime_id))] = compute_regression_metrics(
            flat_true[regime_mask],
            flat_pred[regime_mask],
        )
    return metrics


def summarize_regime_mask_usage(
    prediction_dict: dict[str, np.ndarray],
    regime_mask_metadata: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if regime_mask_metadata is None or "regime_mask_weights" not in prediction_dict:
        return None

    weights = np.asarray(prediction_dict["regime_mask_weights"], dtype=np.float64)
    if weights.ndim != 2:
        raise ValueError(f"regime_mask_weights must have shape [samples, regimes], got {weights.shape}.")
    loaded_regimes = {int(value) for value in regime_mask_metadata.get("loaded_regimes", [])}
    loaded_mask = np.array([regime_id in loaded_regimes for regime_id in range(weights.shape[1])], dtype=bool)
    fallback_weight = weights[:, ~loaded_mask].sum(axis=1) if weights.size else np.array([], dtype=np.float64)
    dominant_regime = weights.argmax(axis=1).astype(np.int64) if weights.size else np.array([], dtype=np.int64)
    usage = {
        "sample_count": int(weights.shape[0]),
        "regime_mask_strategy": regime_mask_metadata.get("regime_mask_strategy"),
        "offline_full_split_regime_labels_used": False,
        "mean_regime_mask_weights": {
            str(regime_id): float(value)
            for regime_id, value in enumerate(weights.mean(axis=0).tolist() if weights.size else [])
        },
        "dominant_regime_counts": value_counts_dict(dominant_regime),
        "fallback_weight_mean": float(fallback_weight.mean()) if fallback_weight.size else 0.0,
        "fallback_definition": "probability mass on regimes without loaded graphs, including regime 0 night, maps to no-mask bias",
    }
    return usage


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
    if cli_args.regime_graph_root is not None:
        eval_args.regime_graph_root = cli_args.regime_graph_root
    elif not hasattr(eval_args, "regime_graph_root"):
        eval_args.regime_graph_root = None
    if cli_args.regime_model_path is not None:
        eval_args.regime_model_path = cli_args.regime_model_path
    elif not hasattr(eval_args, "regime_model_path"):
        eval_args.regime_model_path = None
    if not hasattr(eval_args, "regime_mask_strategy"):
        eval_args.regime_mask_strategy = cli_args.regime_mask_strategy
    if not hasattr(eval_args, "causal_mask_mode"):
        eval_args.causal_mask_mode = cli_args.causal_mask_mode
    if not hasattr(eval_args, "causal_mask_beta"):
        eval_args.causal_mask_beta = cli_args.causal_mask_beta
    if not hasattr(eval_args, "causal_gamma"):
        eval_args.causal_gamma = cli_args.causal_gamma
    if not hasattr(eval_args, "causal_reward_strength"):
        eval_args.causal_reward_strength = cli_args.causal_reward_strength
    if not hasattr(eval_args, "causal_strength_normalization"):
        eval_args.causal_strength_normalization = cli_args.causal_strength_normalization
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
    history: list[dict[str, Any]],
    model_kwargs_fn: Any = None,
    regime_mask_metadata: dict[str, Any] | None = None,
) -> None:
    target_feature_index = get_target_feature_index(list(args.feature_cols), args.target_col)
    lr_plateau_threshold = getattr(args, "lr_plateau_threshold", None)
    if lr_plateau_threshold is None:
        lr_plateau_threshold = getattr(args, "min_delta", 0.0)
    if regime_mask_metadata is None:
        _, causal_mask_metadata = load_causal_attention_mask(
            getattr(args, "causal_graph_dir", None),
            list(args.feature_cols),
            mode=getattr(args, "causal_mask_mode", "hard"),
            beta=float(getattr(args, "causal_mask_beta", 1.0)),
            gamma=float(getattr(args, "causal_gamma", 1.0)),
            reward_strength=str(getattr(args, "causal_reward_strength", "max_abs_mci")),
            strength_normalization=str(getattr(args, "causal_strength_normalization", "per_target_max")),
            d_model=int(getattr(args, "d_model", 0)),
            n_heads=int(getattr(args, "n_heads", 0)),
        )
    else:
        causal_mask_metadata = None
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
        model_kwargs_fn=model_kwargs_fn,
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
            model_kwargs_fn=model_kwargs_fn,
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
    validation_target_regime_metrics = compute_target_regime_metrics(validation_prediction_dict)
    reported_target_regime_metrics = compute_target_regime_metrics(reported_prediction_dict)
    regime_mask_usage = {
        "validation": summarize_regime_mask_usage(validation_prediction_dict, regime_mask_metadata),
        reported_split_name: summarize_regime_mask_usage(reported_prediction_dict, regime_mask_metadata),
    } if regime_mask_metadata is not None else None
    reward_stats: dict[str, Any] | None = None
    if causal_mask_metadata and causal_mask_metadata.get("mask_mode") == "causal_reward":
        reward_stats = {
            "reward_edge_count": int(causal_mask_metadata.get("reward_edge_count", 0)),
            "reward_min": float(causal_mask_metadata.get("reward_min", 0.0)),
            "reward_max": float(causal_mask_metadata.get("reward_max", 0.0)),
            "pre_scale_reward_max": float(causal_mask_metadata.get("pre_scale_reward_max", 0.0)),
        }
    elif regime_mask_metadata and regime_mask_metadata.get("mask_mode") == "causal_reward":
        regime_reward_metadata = [
            metadata
            for metadata in (regime_mask_metadata.get("regime_metadata") or {}).values()
            if metadata.get("mask_mode") == "causal_reward"
        ]
        reward_mins = [
            float(metadata.get("reward_min", 0.0))
            for metadata in regime_reward_metadata
            if int(metadata.get("reward_edge_count", 0)) > 0
        ]
        reward_stats = {
            "reward_edge_count": int(
                sum(int(metadata.get("reward_edge_count", 0)) for metadata in regime_reward_metadata)
            ),
            "reward_min": min(reward_mins) if reward_mins else 0.0,
            "reward_max": max(
                [float(metadata.get("reward_max", 0.0)) for metadata in regime_reward_metadata],
                default=0.0,
            ),
            "pre_scale_reward_max": max(
                [float(metadata.get("pre_scale_reward_max", 0.0)) for metadata in regime_reward_metadata],
                default=0.0,
            ),
        }

    metrics_payload = {
        "config": vars(args),
        "experiment_name": args.experiment_name,
        "tuning_stage": args.tuning_stage,
        "device": str(device),
        "baseline_type": "itransformer",
        "baseline_definition": (
            "iTransformer backbone with Active_Pow extracted from the multivariate forecast output"
            + (
                (
                    " and transition-weighted target-regime-conditioned PCMCI causal edge reward from online HMM regime probabilities."
                    if regime_mask_metadata.get("mask_mode") == "causal_reward"
                    else " and transition-weighted target-regime-conditioned PCMCI soft attention bias from online HMM regime probabilities."
                )
                if regime_mask_metadata is not None
                else (
                    f" and a global PCMCI variable-level {causal_mask_metadata['mask_mode']} attention mask."
                    if causal_mask_metadata and causal_mask_metadata.get("mask_applied")
                    else "."
                )
            )
        ),
        "causal_mask_mode": getattr(args, "causal_mask_mode", "hard"),
        "causal_gamma": (
            float(getattr(args, "causal_gamma", 1.0))
            if getattr(args, "causal_mask_mode", "hard") == "causal_reward"
            else None
        ),
        "causal_reward_strength": (
            str(getattr(args, "causal_reward_strength", "max_abs_mci"))
            if getattr(args, "causal_mask_mode", "hard") == "causal_reward"
            else None
        ),
        "causal_strength_normalization": (
            str(getattr(args, "causal_strength_normalization", "per_target_max"))
            if getattr(args, "causal_mask_mode", "hard") == "causal_reward"
            else None
        ),
        "reward_edge_count": (
            int(reward_stats["reward_edge_count"]) if reward_stats is not None else None
        ),
        "reward_min": (
            float(reward_stats["reward_min"]) if reward_stats is not None else None
        ),
        "reward_max": (
            float(reward_stats["reward_max"]) if reward_stats is not None else None
        ),
        "pre_scale_reward_max": (
            float(reward_stats["pre_scale_reward_max"]) if reward_stats is not None else None
        ),
        "loaded_regimes": (
            regime_mask_metadata.get("loaded_regimes")
            if regime_mask_metadata is not None
            else None
        ),
        "regime_reward_density": (
            regime_mask_metadata.get("regime_reward_density")
            if regime_mask_metadata is not None and regime_mask_metadata.get("mask_mode") == "causal_reward"
            else None
        ),
        "causal_mask": causal_mask_metadata,
        "regime_causal_mask": regime_mask_metadata,
        "regime_mask_strategy": (
            getattr(args, "regime_mask_strategy", None)
            if regime_mask_metadata is not None
            else None
        ),
        "offline_full_split_regime_labels_used": False if regime_mask_metadata is not None else None,
        "regime_mask_usage": regime_mask_usage,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "lr_scheduler": getattr(args, "lr_scheduler", "none"),
        "lr_plateau_factor": float(getattr(args, "lr_plateau_factor", 0.5)),
        "lr_plateau_patience": int(getattr(args, "lr_plateau_patience", 2)),
        "lr_plateau_min_lr": float(getattr(args, "lr_plateau_min_lr", 1e-6)),
        "lr_plateau_threshold": float(lr_plateau_threshold),
        "expected_delta_minutes": float(expected_delta / pd.Timedelta(minutes=1)),
        "dataset_summary": {split_name: dataset.summary() for split_name, dataset in datasets.items()},
        "raw_split_rows": {split_name: int(len(df)) for split_name, df in raw_frames.items()},
        "target_feature_index": target_feature_index,
        "trainable_parameter_count": trainable_parameter_count,
        "report_split": args.report_split,
        "calibration_usage": "loaded_but_unused",
        "validation_metrics": validation_metrics,
        "validation_target_regime_metrics": validation_target_regime_metrics,
        "reported_metrics": reported_metrics,
        "reported_target_regime_metrics": reported_target_regime_metrics,
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
    if regime_mask_metadata is not None:
        log(f"- regime mask root: {regime_mask_metadata['regime_graph_root']}")
        log(f"- regime mask strategy: {regime_mask_metadata['regime_mask_strategy']}")

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
    if args.regime_graph_root is not None and args.causal_graph_dir is not None:
        raise ValueError("--regime_graph_root and --causal_graph_dir are mutually exclusive.")
    if args.regime_graph_root is not None and args.regime_model_path is None:
        raise ValueError("--regime_model_path is required when --regime_graph_root is set.")

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
        if args.regime_graph_root is not None and args.causal_graph_dir is not None:
            raise ValueError("--regime_graph_root and --causal_graph_dir are mutually exclusive.")
        regime_artifact = (
            load_regime_model_artifact(getattr(args, "regime_model_path", None))
            if getattr(args, "regime_graph_root", None) is not None
            else None
        )
        regime_mask_bank, regime_mask_metadata = load_regime_attention_mask_bank(
            getattr(args, "regime_graph_root", None),
            list(args.feature_cols),
            mode=getattr(args, "causal_mask_mode", "hard"),
            beta=float(getattr(args, "causal_mask_beta", 1.0)),
            gamma=float(getattr(args, "causal_gamma", 1.0)),
            reward_strength=str(getattr(args, "causal_reward_strength", "max_abs_mci")),
            strength_normalization=str(getattr(args, "causal_strength_normalization", "per_target_max")),
            d_model=int(getattr(args, "d_model", 0)),
            n_heads=int(getattr(args, "n_heads", 0)),
            regime_mask_strategy=getattr(args, "regime_mask_strategy", "transition_weighted"),
            expected_regime_count=regime_artifact.total_regime_count if regime_artifact is not None else None,
            regime_model_path=getattr(args, "regime_model_path", None),
        )
        if regime_mask_bank is not None:
            regime_mask_bank = regime_mask_bank.to(device=device)
        model_kwargs_fn = build_regime_model_kwargs_fn(regime_mask_bank)

        include_test = args.report_split == "test"
        datasets, raw_frames, scalers, expected_delta = prepare_itransformer_datasets(
            args,
            include_test=include_test,
            regime_artifact=regime_artifact,
        )
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
            model_kwargs_fn=model_kwargs_fn,
            regime_mask_metadata=regime_mask_metadata,
        )
        return

    set_random_seed(args.seed)
    device = get_device(args.device)

    results_dir = resolve_path(args.results_dir)
    checkpoint_path = resolve_path(args.checkpoint_path)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    regime_artifact = (
        load_regime_model_artifact(getattr(args, "regime_model_path", None))
        if getattr(args, "regime_graph_root", None) is not None
        else None
    )
    regime_mask_bank, regime_mask_metadata = load_regime_attention_mask_bank(
        getattr(args, "regime_graph_root", None),
        list(args.feature_cols),
        mode=getattr(args, "causal_mask_mode", "hard"),
        beta=float(getattr(args, "causal_mask_beta", 1.0)),
        gamma=float(getattr(args, "causal_gamma", 1.0)),
        reward_strength=str(getattr(args, "causal_reward_strength", "max_abs_mci")),
        strength_normalization=str(getattr(args, "causal_strength_normalization", "per_target_max")),
        d_model=int(getattr(args, "d_model", 0)),
        n_heads=int(getattr(args, "n_heads", 0)),
        regime_mask_strategy=getattr(args, "regime_mask_strategy", "transition_weighted"),
        expected_regime_count=regime_artifact.total_regime_count if regime_artifact is not None else None,
        regime_model_path=getattr(args, "regime_model_path", None),
    )
    if regime_mask_bank is not None:
        regime_mask_bank = regime_mask_bank.to(device=device)
    model_kwargs_fn = build_regime_model_kwargs_fn(regime_mask_bank)

    datasets, raw_frames, scalers, expected_delta = prepare_itransformer_datasets(
        args,
        include_test=not args.tuning_only,
        regime_artifact=regime_artifact,
    )
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
    lr_plateau_threshold = args.lr_plateau_threshold if args.lr_plateau_threshold is not None else args.min_delta
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_plateau_factor,
            patience=args.lr_plateau_patience,
            threshold=lr_plateau_threshold,
            threshold_mode="abs",
            min_lr=args.lr_plateau_min_lr,
        )
        if args.lr_scheduler == "plateau"
        else None
    )
    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

    history: list[dict[str, Any]] = []
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
            model_kwargs_fn=model_kwargs_fn,
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
            model_kwargs_fn=model_kwargs_fn,
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

        previous_lr = get_learning_rate(optimizer)
        if scheduler is not None:
            scheduler.step(val_loss)
        current_lr = get_learning_rate(optimizer)
        lr_reduced = bool(current_lr < previous_lr)

        should_stop = early_stopping.step(val_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "validation_loss": float(val_loss),
                "best_epoch": int(best_epoch),
                "best_validation_loss": float(best_val_loss),
                "patience_counter": int(early_stopping.counter),
                "learning_rate": float(current_lr),
                "lr_reduced": lr_reduced,
            }
        )
        log(
            f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} "
            f"| best_epoch={best_epoch} | best_val_loss={best_val_loss:.6f} "
            f"| patience_counter={early_stopping.counter} | lr={current_lr:.6g} "
            f"| lr_reduced={lr_reduced}"
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
        model_kwargs_fn=model_kwargs_fn,
        regime_mask_metadata=regime_mask_metadata,
    )


if __name__ == "__main__":
    main()
