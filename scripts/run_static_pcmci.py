from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

from utils.datasets import (
    DEFAULT_FEATURE_COLUMNS,
    build_segment_boundaries,
    infer_expected_timedelta,
    load_split_dataframe,
)


CI_TEST_NAME = "ParCorr(significance='analytic')"
DEFAULT_TRAIN_PATH = "data/processed/splits/train.csv"
DEFAULT_OUTPUT_DIR = "results/causal_graphs/static"
FULL_FEATURE_COLUMNS = list(DEFAULT_FEATURE_COLUMNS)
ACTIVE_POW_COLUMN = "Active_Pow"
DETERMINISTIC_ROOT_COLUMNS = [
    "solar_elevation",
    "sin_time_of_day",
    "cos_time_of_day",
    "sin_day_of_year",
    "cos_day_of_year",
    "day_night_label",
]


@dataclass
class PreparedPCMCIData:
    time_column: str
    mode: str
    feature_columns: list[str]
    active_feature_columns: list[str]
    excluded_constant_columns: list[str]
    rows_loaded: int
    rows_after_mode_filter: int
    rows_after_missing_drop: int
    rows_used_in_pcmci: int
    expected_delta: pd.Timedelta
    total_segments: int
    kept_segments: int
    dropped_segments_short: int
    all_segment_lengths: list[int]
    kept_segment_lengths: list[int]
    segment_arrays: dict[int, np.ndarray] | np.ndarray
    tigramite_analysis_mode: str


@dataclass(frozen=True)
class ConstraintSpec:
    profile: str
    deterministic_root_columns: list[str]
    forbidden_pair_reasons: dict[tuple[str, str], str]
    description_lines: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a train-only global static causal graph with Tigramite PCMCI "
            "for the PV forecasting project. The script never reads validation / "
            "calibration / test."
        )
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default=DEFAULT_TRAIN_PATH,
        help="Path to train.csv. Only this file is loaded.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Base directory for exported causal graph artifacts. The mode suffix is appended automatically.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="daytime",
        choices=["daytime", "all"],
        help="Use daytime-only samples or the full train split.",
    )
    parser.add_argument("--tau_max", type=int, default=12, help="Maximum lag for PCMCI.")
    parser.add_argument("--pc_alpha", type=float, default=0.05, help="PC stage alpha for PCMCI.")
    parser.add_argument(
        "--alpha_level",
        type=float,
        default=0.05,
        help="Significance threshold used by PCMCI and the exported edge list.",
    )
    parser.add_argument(
        "--min_segment_len",
        type=int,
        default=128,
        help="Drop continuous segments shorter than this length before PCMCI.",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=1,
        help="Verbosity level: 0=silent, 1=key steps + heartbeat, 2=PCMCI stage logs, 3=more Tigramite detail.",
    )
    parser.add_argument(
        "--heartbeat_seconds",
        type=int,
        default=30,
        help="Print a heartbeat during the long PCMCI step every N seconds. Set 0 to disable.",
    )
    parser.add_argument(
        "--constraint_profile",
        type=str,
        default="pv_physical",
        choices=["none", "pv_physical"],
        help=(
            "Post-PCMCI constraint profile used when aggregating adjacency and plotting the final graph. "
            "`pv_physical` treats deterministic time features as exogenous roots and treats Active_Pow as a sink."
        ),
    )
    return parser.parse_args()


def log(message: str, verbosity: int, level: int = 1) -> None:
    if verbosity >= level:
        print(message, flush=True)


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_output_dir(path_str: str, mode: str) -> Path:
    base_dir = resolve_repo_path(path_str)
    if base_dir.name == mode:
        return base_dir
    return base_dir / mode


def validate_arguments(args: argparse.Namespace) -> None:
    if args.tau_max < 0:
        raise ValueError("--tau_max must be >= 0.")
    if not 0 < args.pc_alpha <= 1:
        raise ValueError("--pc_alpha must be in (0, 1].")
    if not 0 < args.alpha_level <= 1:
        raise ValueError("--alpha_level must be in (0, 1].")
    if args.min_segment_len <= 0:
        raise ValueError("--min_segment_len must be positive.")


def normalize_column_name(column: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", column.lower())


def build_mapping_suggestions(
    missing_columns: list[str],
    existing_columns: list[str],
) -> dict[str, list[str]]:
    suggestions: dict[str, list[str]] = {}
    normalized_lookup = {normalize_column_name(column): column for column in existing_columns}

    for missing_column in missing_columns:
        normalized_missing = normalize_column_name(missing_column)
        direct_match = normalized_lookup.get(normalized_missing)
        matches: list[str] = []
        if direct_match is not None:
            matches.append(direct_match)

        fuzzy_matches = get_close_matches(missing_column, existing_columns, n=3, cutoff=0.4)
        for match in fuzzy_matches:
            if match not in matches:
                matches.append(match)
        suggestions[missing_column] = matches

    return suggestions


def validate_required_columns(df: pd.DataFrame) -> None:
    missing_columns = [column for column in FULL_FEATURE_COLUMNS if column not in df.columns]
    if not missing_columns:
        return

    existing_columns = list(df.columns)
    suggestions = build_mapping_suggestions(missing_columns, existing_columns)
    suggestion_lines = [
        f"  - {missing_column}: {candidate_list if candidate_list else 'no close match found'}"
        for missing_column, candidate_list in suggestions.items()
    ]
    raise KeyError(
        "train.csv is missing required feature columns for static PCMCI.\n"
        f"Required columns: {FULL_FEATURE_COLUMNS}\n"
        f"Missing columns: {missing_columns}\n"
        f"Existing columns: {existing_columns}\n"
        "Mapping suggestions:\n"
        + "\n".join(suggestion_lines)
    )


def describe_segment_lengths(lengths: list[int]) -> dict[str, float | int | None]:
    if not lengths:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "median": None,
            "p25": None,
            "p75": None,
        }

    lengths_array = np.asarray(lengths, dtype=np.int64)
    return {
        "count": int(lengths_array.size),
        "min": int(lengths_array.min()),
        "max": int(lengths_array.max()),
        "median": float(np.median(lengths_array)),
        "p25": float(np.quantile(lengths_array, 0.25)),
        "p75": float(np.quantile(lengths_array, 0.75)),
    }


def format_timedelta(delta: pd.Timedelta) -> str:
    return str(delta)


def format_elapsed_seconds(elapsed_seconds: float) -> str:
    total_seconds = max(0, int(round(elapsed_seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def get_tigramite_verbosity(cli_verbosity: int) -> int:
    if cli_verbosity >= 3:
        return 2
    if cli_verbosity >= 2:
        return 1
    return 0


def build_constraint_spec(
    feature_columns: list[str],
    profile: str,
) -> ConstraintSpec:
    if profile == "none":
        return ConstraintSpec(
            profile=profile,
            deterministic_root_columns=[],
            forbidden_pair_reasons={},
            description_lines=["No post-PCMCI physical edge filtering is applied."],
        )

    deterministic_root_columns = [column for column in DETERMINISTIC_ROOT_COLUMNS if column in feature_columns]
    forbidden_pair_reasons: dict[tuple[str, str], str] = {}

    for target in deterministic_root_columns:
        for source in feature_columns:
            if source != target:
                forbidden_pair_reasons[(source, target)] = "deterministic_time_feature_is_exogenous_root"

    if ACTIVE_POW_COLUMN in feature_columns:
        for target in feature_columns:
            if target != ACTIVE_POW_COLUMN:
                forbidden_pair_reasons[(ACTIVE_POW_COLUMN, target)] = "active_pow_is_treated_as_sink_only"

    description_lines = [
        "Deterministic time features are treated as exogenous roots, so cross-variable edges into them are removed.",
        "Active_Pow is treated as an outcome variable, so outgoing edges from Active_Pow to other variables are removed.",
        "Raw PCMCI significance is preserved in edges_lag_level.csv; only the final directed adjacency and plotted graph are filtered.",
    ]
    return ConstraintSpec(
        profile=profile,
        deterministic_root_columns=deterministic_root_columns,
        forbidden_pair_reasons=forbidden_pair_reasons,
        description_lines=description_lines,
    )


def prepare_pcmci_data(
    train_path: Path,
    mode: str,
    min_segment_len: int,
    verbosity: int,
) -> PreparedPCMCIData:
    log(f"[1/6] Loading train split only from {train_path}", verbosity)
    train_df = load_split_dataframe(train_path)
    validate_required_columns(train_df)

    time_column = train_df.index.name or "datetime_index"
    rows_loaded = len(train_df)
    log(f"Detected time column: {time_column}", verbosity, level=2)
    log(f"Using feature order aligned with current repo scripts: {FULL_FEATURE_COLUMNS}", verbosity, level=2)

    working_df = train_df.loc[:, FULL_FEATURE_COLUMNS].copy()

    if mode == "daytime":
        if "day_night_label" not in working_df.columns:
            raise KeyError("`day_night_label` is required for --mode daytime but is missing from train.csv.")
        working_df = working_df.loc[working_df["day_night_label"] == 1].copy()
        if working_df.empty:
            raise ValueError("No rows remain after applying `--mode daytime`.")

    rows_after_mode_filter = len(working_df)

    missing_row_mask = working_df.isna().any(axis=1)
    missing_row_count = int(missing_row_mask.sum())
    if missing_row_count > 0:
        log(
            f"Dropping {missing_row_count} rows with missing values before segment detection. "
            "This is conservative and lets dropped rows create segment boundaries.",
            verbosity,
        )
    working_df = working_df.loc[~missing_row_mask].copy()
    rows_after_missing_drop = len(working_df)
    if working_df.empty:
        raise ValueError("No samples remain after dropping rows with missing values in the required feature columns.")

    expected_delta = infer_expected_timedelta(working_df.index)
    segment_starts, segment_ends = build_segment_boundaries(working_df.index, expected_delta)
    all_segment_lengths = [int(end - start) for start, end in zip(segment_starts, segment_ends)]

    kept_frames: list[pd.DataFrame] = []
    kept_segment_lengths: list[int] = []
    for start, end in zip(segment_starts, segment_ends):
        segment_df = working_df.iloc[int(start) : int(end)].copy()
        if len(segment_df) >= min_segment_len:
            kept_frames.append(segment_df)
            kept_segment_lengths.append(int(len(segment_df)))

    total_segments = len(all_segment_lengths)
    kept_segments = len(kept_frames)
    dropped_segments_short = total_segments - kept_segments
    if not kept_frames:
        raise ValueError(
            "No continuous segments satisfy --min_segment_len after mode filtering and missing-value dropping. "
            f"Detected sampling frequency: {format_timedelta(expected_delta)}. "
            f"Total segments: {total_segments}. "
            f"Longest segment length: {max(all_segment_lengths) if all_segment_lengths else 0}."
        )

    rows_used_in_pcmci = int(sum(kept_segment_lengths))

    scaler_source = pd.concat(kept_frames, axis=0)
    std_series = scaler_source.std(ddof=0)
    excluded_constant_columns = [
        column for column in FULL_FEATURE_COLUMNS if pd.isna(std_series[column]) or np.isclose(std_series[column], 0.0)
    ]
    active_feature_columns = [column for column in FULL_FEATURE_COLUMNS if column not in excluded_constant_columns]
    if not active_feature_columns:
        raise ValueError("All required feature columns became constant after filtering; PCMCI cannot run.")

    if excluded_constant_columns:
        log(
            "Excluding constant columns from PCMCI while keeping them in the final exported 11x11 matrices: "
            f"{excluded_constant_columns}",
            verbosity,
        )

    scaler = StandardScaler()
    scaler.fit(scaler_source.loc[:, active_feature_columns].to_numpy(dtype=np.float64))

    segment_dict: dict[int, np.ndarray] = {}
    for segment_id, segment_df in enumerate(kept_frames):
        standardized_values = scaler.transform(
            segment_df.loc[:, active_feature_columns].to_numpy(dtype=np.float64)
        ).astype(np.float64)
        segment_dict[segment_id] = standardized_values

    if len(segment_dict) == 1:
        tigramite_analysis_mode = "single"
        segment_arrays: dict[int, np.ndarray] | np.ndarray = next(iter(segment_dict.values()))
    else:
        tigramite_analysis_mode = "multiple"
        segment_arrays = segment_dict

    return PreparedPCMCIData(
        time_column=time_column,
        mode=mode,
        feature_columns=list(FULL_FEATURE_COLUMNS),
        active_feature_columns=active_feature_columns,
        excluded_constant_columns=excluded_constant_columns,
        rows_loaded=rows_loaded,
        rows_after_mode_filter=rows_after_mode_filter,
        rows_after_missing_drop=rows_after_missing_drop,
        rows_used_in_pcmci=rows_used_in_pcmci,
        expected_delta=expected_delta,
        total_segments=total_segments,
        kept_segments=kept_segments,
        dropped_segments_short=dropped_segments_short,
        all_segment_lengths=all_segment_lengths,
        kept_segment_lengths=kept_segment_lengths,
        segment_arrays=segment_arrays,
        tigramite_analysis_mode=tigramite_analysis_mode,
    )


def run_pcmci(
    prepared: PreparedPCMCIData,
    tau_max: int,
    pc_alpha: float,
    alpha_level: float,
    verbosity: int,
    heartbeat_seconds: int,
) -> dict[str, Any]:
    log("[2/6] Running Tigramite PCMCI + ParCorr on train-only continuous segments", verbosity)
    tigramite_verbosity = get_tigramite_verbosity(verbosity)
    dataframe = pp.DataFrame(
        prepared.segment_arrays,
        var_names=prepared.active_feature_columns,
        analysis_mode=prepared.tigramite_analysis_mode,
    )
    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=ParCorr(significance="analytic"),
        verbosity=tigramite_verbosity,
    )
    stop_event = threading.Event()
    start_time = time.time()
    heartbeat_thread: threading.Thread | None = None

    if verbosity >= 1 and heartbeat_seconds > 0:
        def heartbeat_worker() -> None:
            while not stop_event.wait(timeout=heartbeat_seconds):
                elapsed_text = format_elapsed_seconds(time.time() - start_time)
                print(
                    "[PCMCI] still running | "
                    f"elapsed={elapsed_text} | "
                    f"active_features={len(prepared.active_feature_columns)} | "
                    f"segments={prepared.kept_segments} | "
                    f"tau_max={tau_max}",
                    flush=True,
                )

        heartbeat_thread = threading.Thread(target=heartbeat_worker, daemon=True)
        heartbeat_thread.start()

    try:
        results = pcmci.run_pcmci(
            tau_max=tau_max,
            pc_alpha=pc_alpha,
            alpha_level=alpha_level,
            fdr_method="none",
        )
    finally:
        stop_event.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=1.0)

    elapsed_text = format_elapsed_seconds(time.time() - start_time)
    log(f"PCMCI core step finished in {elapsed_text}.", verbosity)
    return results


def expand_results_to_full_feature_space(
    results: dict[str, Any],
    full_feature_columns: list[str],
    active_feature_columns: list[str],
    tau_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    full_feature_count = len(full_feature_columns)
    full_shape = (full_feature_count, full_feature_count, tau_max + 1)
    p_matrix_full = np.full(full_shape, np.nan, dtype=np.float64)
    val_matrix_full = np.full(full_shape, np.nan, dtype=np.float64)
    graph_matrix_full = np.full(full_shape, "", dtype=results["graph"].dtype)

    full_index_lookup = {column: idx for idx, column in enumerate(full_feature_columns)}
    active_indices = [full_index_lookup[column] for column in active_feature_columns]

    for active_source_idx, full_source_idx in enumerate(active_indices):
        for active_target_idx, full_target_idx in enumerate(active_indices):
            p_matrix_full[full_source_idx, full_target_idx, :] = results["p_matrix"][active_source_idx, active_target_idx, :]
            val_matrix_full[full_source_idx, full_target_idx, :] = results["val_matrix"][
                active_source_idx,
                active_target_idx,
                :,
            ]
            graph_matrix_full[full_source_idx, full_target_idx, :] = results["graph"][
                active_source_idx,
                active_target_idx,
                :,
            ]

    return p_matrix_full, val_matrix_full, graph_matrix_full


def build_edge_table(
    full_feature_columns: list[str],
    p_matrix_full: np.ndarray,
    val_matrix_full: np.ndarray,
    graph_matrix_full: np.ndarray,
    alpha_level: float,
    excluded_constant_columns: list[str],
    constraint_spec: ConstraintSpec,
) -> pd.DataFrame:
    excluded_set = set(excluded_constant_columns)
    rows: list[dict[str, Any]] = []

    for source_idx, source in enumerate(full_feature_columns):
        for target_idx, target in enumerate(full_feature_columns):
            for tau in range(p_matrix_full.shape[2]):
                lag = -tau
                p_value = p_matrix_full[source_idx, target_idx, tau]
                test_stat = val_matrix_full[source_idx, target_idx, tau]
                graph_symbol = str(graph_matrix_full[source_idx, target_idx, tau])
                significant = bool(np.isfinite(p_value) and p_value <= alpha_level)
                raw_directed_lag_edge = bool(significant and graph_symbol == "-->")
                constraint_reason = constraint_spec.forbidden_pair_reasons.get((source, target), "")
                constraint_allowed = constraint_reason == ""
                used_for_variable_adjacency = bool(raw_directed_lag_edge and constraint_allowed)
                rows.append(
                    {
                        "source": source,
                        "target": target,
                        "source_idx": source_idx,
                        "target_idx": target_idx,
                        "lag": int(lag),
                        "test_stat": float(test_stat) if np.isfinite(test_stat) else np.nan,
                        "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                        "significant": int(significant),
                        "graph_symbol": graph_symbol,
                        "raw_directed_lag_edge": int(raw_directed_lag_edge),
                        "constraint_allowed": int(constraint_allowed),
                        "constraint_reason": constraint_reason,
                        "used_for_variable_adjacency": int(used_for_variable_adjacency),
                        "source_active_in_pcmci": int(source not in excluded_set),
                        "target_active_in_pcmci": int(target not in excluded_set),
                    }
                )

    return pd.DataFrame(rows)


def build_variable_level_adjacency(
    edge_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    adjacency_raw = pd.DataFrame(0, index=feature_columns, columns=feature_columns, dtype=int)
    directed_edges = edge_df.loc[edge_df["used_for_variable_adjacency"] == 1, ["source", "target"]].drop_duplicates()

    for edge in directed_edges.itertuples(index=False):
        adjacency_raw.loc[edge.source, edge.target] = 1

    adjacency_for_mask = adjacency_raw.copy()
    np.fill_diagonal(adjacency_for_mask.values, 1)
    return adjacency_raw, adjacency_for_mask


def build_plot_matrices_from_edge_table(
    feature_columns: list[str],
    edge_df: pd.DataFrame,
    graph_matrix_full: np.ndarray,
    val_matrix_full: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    graph_matrix_for_plot = np.full_like(graph_matrix_full, "", dtype=graph_matrix_full.dtype)
    val_matrix_for_plot = np.full_like(val_matrix_full, np.nan, dtype=np.float64)
    feature_index_lookup = {column: idx for idx, column in enumerate(feature_columns)}

    used_edge_rows = edge_df.loc[
        edge_df["used_for_variable_adjacency"] == 1,
        ["source", "target", "lag", "test_stat"],
    ]

    for edge in used_edge_rows.itertuples(index=False):
        source_idx = feature_index_lookup[edge.source]
        target_idx = feature_index_lookup[edge.target]
        tau = -int(edge.lag)
        graph_matrix_for_plot[source_idx, target_idx, tau] = "-->"
        if np.isfinite(edge.test_stat):
            val_matrix_for_plot[source_idx, target_idx, tau] = float(edge.test_stat)

    return graph_matrix_for_plot, val_matrix_for_plot


def save_adjacency_heatmap(
    adjacency_df: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(11, 9))
    sns.heatmap(
        adjacency_df,
        annot=True,
        fmt="d",
        cmap="Blues",
        square=True,
        cbar=False,
        linewidths=0.5,
        linecolor="white",
    )
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_causal_graph(
    full_feature_columns: list[str],
    graph_matrix_full: np.ndarray,
    val_matrix_full: np.ndarray,
    output_path: Path,
) -> bool:
    try:
        tp.plot_graph(
            graph=graph_matrix_full,
            val_matrix=np.nan_to_num(val_matrix_full, nan=0.0),
            var_names=full_feature_columns,
            figsize=(14, 12),
            save_name=str(output_path),
            show_colorbar=True,
        )
        plt.close("all")
        return True
    except Exception:
        plt.close("all")
        return False


def build_summary(
    args: argparse.Namespace,
    train_path: Path,
    output_dir: Path,
    prepared: PreparedPCMCIData,
    edge_df: pd.DataFrame,
    constraint_spec: ConstraintSpec,
    graph_png_created: bool,
) -> dict[str, Any]:
    significant_count = int(edge_df["significant"].sum())
    raw_directed_edge_count = int(edge_df["raw_directed_lag_edge"].sum())
    adjacency_edge_count = int(edge_df["used_for_variable_adjacency"].sum())
    constrained_out_count = raw_directed_edge_count - adjacency_edge_count
    summary = {
        "train_path": str(train_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "mode": args.mode,
        "time_column": prepared.time_column,
        "feature_columns_output_order": prepared.feature_columns,
        "active_feature_columns_for_pcmci": prepared.active_feature_columns,
        "excluded_constant_columns": prepared.excluded_constant_columns,
        "rows_loaded_from_train": prepared.rows_loaded,
        "rows_after_mode_filter": prepared.rows_after_mode_filter,
        "rows_after_missing_drop": prepared.rows_after_missing_drop,
        "rows_used_in_pcmci": prepared.rows_used_in_pcmci,
        "expected_sampling_frequency": format_timedelta(prepared.expected_delta),
        "total_segments_detected": prepared.total_segments,
        "segments_kept_for_pcmci": prepared.kept_segments,
        "segments_dropped_for_being_short": prepared.dropped_segments_short,
        "all_segment_length_stats": describe_segment_lengths(prepared.all_segment_lengths),
        "kept_segment_length_stats": describe_segment_lengths(prepared.kept_segment_lengths),
        "tigramite_analysis_mode": prepared.tigramite_analysis_mode,
        "ci_test": CI_TEST_NAME,
        "tau_max": int(args.tau_max),
        "pc_alpha": float(args.pc_alpha),
        "alpha_level": float(args.alpha_level),
        "fdr_method": "none",
        "min_segment_len": int(args.min_segment_len),
        "significance_rule": "significant = (p_value <= alpha_level)",
        "variable_level_aggregation_rule": (
            "A[i, j] = 1 if there exists at least one lag-level row with "
            "used_for_variable_adjacency = 1, where "
            "used_for_variable_adjacency = raw_directed_lag_edge and constraint_allowed, "
            "and raw_directed_lag_edge = significant and graph_symbol == '-->'"
        ),
        "constraint_profile": constraint_spec.profile,
        "constraint_deterministic_root_columns": constraint_spec.deterministic_root_columns,
        "constraint_forbidden_pair_count": int(len(constraint_spec.forbidden_pair_reasons)),
        "constraint_description_lines": constraint_spec.description_lines,
        "method_caveat": (
            "ParCorr is a continuous-variable conditional independence test. "
            "`day_night_label` is therefore treated numerically in mode=all, "
            "and becomes an excluded constant column in mode=daytime."
        ),
        "note_on_contemporaneous_edges": (
            "Significant but unoriented contemporaneous links such as 'o-o' are kept in "
            "edges_lag_level.csv but are not aggregated into the directed variable-level adjacency."
        ),
        "note_on_causal_graph_png": (
            "causal_graph.png shows only the constrained directed lagged edges that are used for "
            "the variable-level adjacency. Raw PCMCI graph symbols remain available in "
            "edges_lag_level.csv and pcmci_raw_results.npz."
        ),
        "edge_rows_exported": int(len(edge_df)),
        "significant_lag_level_rows": significant_count,
        "raw_directed_lag_level_rows": raw_directed_edge_count,
        "constrained_directed_rows_used_for_variable_adjacency": adjacency_edge_count,
        "directed_rows_removed_by_constraints": constrained_out_count,
        "causal_graph_png_created": graph_png_created,
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
    }
    return summary


def build_readme_markdown(summary: dict[str, Any]) -> str:
    excluded_constant_columns = summary["excluded_constant_columns"]
    excluded_text = (
        f"- Excluded constant columns from PCMCI estimation: {excluded_constant_columns}\n"
        if excluded_constant_columns
        else ""
    )
    constraint_description_lines = summary["constraint_description_lines"]
    constraint_text = "\n".join(f"- {line}" for line in constraint_description_lines)
    deterministic_roots = summary["constraint_deterministic_root_columns"]
    deterministic_root_text = (
        f"- Deterministic root columns under the active constraint profile: {deterministic_roots}\n"
        if deterministic_roots
        else ""
    )

    return f"""# Static PCMCI Summary

## What This Run Does

- Builds a single global static causal graph from `train.csv` only.
- Does **not** read validation / calibration / test.
- Uses `mode = {summary["mode"]}`.
- Uses Tigramite `PCMCI` with `{summary["ci_test"]}`.
- Uses `constraint_profile = {summary["constraint_profile"]}` for the final adjacency and plotted graph.

## Time-Continuity Protocol

- Auto-detect the timestamp column (`{summary["time_column"]}`) and sort by time.
- Infer the main sampling frequency from the mode of adjacent timestamp differences: `{summary["expected_sampling_frequency"]}`.
- Start a new continuous segment whenever the adjacent difference is not equal to the inferred main frequency.
- Drop segments shorter than `min_segment_len = {summary["min_segment_len"]}`.
- Feed the kept segments into Tigramite through its `{summary["tigramite_analysis_mode"]}` dataset mode so different segments are not stitched together as one fake continuous series.

## Missing-Value And Scaling Policy

- Missing-value handling is conservative: rows with any missing value in the required feature columns are dropped before segment detection, so deleted rows naturally create new segment boundaries.
- Standardization is fit only on the retained train rows that actually enter PCMCI. This improves numerical stability for ParCorr without touching any non-train split.
{excluded_text}- Raw output feature order follows the current repo input order used by `utils.datasets.DEFAULT_FEATURE_COLUMNS`, so the exported adjacency matrix can be aligned with downstream variable tokens.

## Significance And Aggregation Rules

- Lag-level significance rule: `{summary["significance_rule"]}`.
- Variable-level aggregation rule: `{summary["variable_level_aggregation_rule"]}`.
- Method caveat: {summary["method_caveat"]}.
- Contemporaneous links with unresolved direction such as `o-o` remain in `edges_lag_level.csv`, but they are excluded from the directed adjacency mask.
- `causal_graph.png` is a filtered view that shows only the constrained directed lagged edges used in the final adjacency.

## Constraint Profile

{constraint_text}
{deterministic_root_text}- Forbidden source-target pairs generated by the active profile: {summary["constraint_forbidden_pair_count"]}.

## Main Artifacts

- `adjacency_variable_level_raw.csv`: directed binary adjacency without forcing diagonal values.
- `adjacency_variable_level_for_mask.csv`: same adjacency, but the diagonal is forced to 1 for downstream masks.
- `edges_lag_level.csv`: lag-level PCMCI export with p-values, test statistics, graph symbols, raw directed-edge flags, constraint flags, and the final aggregation flag.
- `pcmci_summary.json`: parameters, counts, segment statistics, and rules used in this run.
- `adjacency_heatmap.png`: heatmap of the raw variable-level adjacency.
- `causal_graph.png`: Tigramite graph visualization for the constrained directed lagged edges used by the adjacency.
- `pcmci_raw_results.npz`: raw expanded `p_matrix`, `val_matrix`, and `graph` arrays in full feature space, plus filtered matrices used for plotting.
"""


def export_results(
    output_dir: Path,
    prepared: PreparedPCMCIData,
    edge_df: pd.DataFrame,
    adjacency_raw: pd.DataFrame,
    adjacency_for_mask: pd.DataFrame,
    p_matrix_full: np.ndarray,
    val_matrix_full: np.ndarray,
    graph_matrix_full: np.ndarray,
    graph_matrix_for_plot: np.ndarray,
    val_matrix_for_plot: np.ndarray,
    summary: dict[str, Any],
    verbosity: int,
) -> None:
    log("[5/6] Exporting adjacency matrices, edge list, summary, and figures", verbosity)
    output_dir.mkdir(parents=True, exist_ok=True)

    adjacency_raw.to_csv(output_dir / "adjacency_variable_level_raw.csv", encoding="utf-8-sig")
    adjacency_for_mask.to_csv(output_dir / "adjacency_variable_level_for_mask.csv", encoding="utf-8-sig")
    edge_df.to_csv(output_dir / "edges_lag_level.csv", index=False, encoding="utf-8-sig")

    with (output_dir / "pcmci_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    np.savez_compressed(
        output_dir / "pcmci_raw_results.npz",
        p_matrix=p_matrix_full,
        val_matrix=val_matrix_full,
        graph=graph_matrix_full,
        graph_for_plot=graph_matrix_for_plot,
        val_matrix_for_plot=val_matrix_for_plot,
        feature_names=np.asarray(prepared.feature_columns, dtype="<U64"),
    )

    save_adjacency_heatmap(
        adjacency_raw,
        output_dir / "adjacency_heatmap.png",
        title=f"Static PCMCI Variable-Level Adjacency ({prepared.mode})",
    )

    readme_text = build_readme_markdown(summary)
    (output_dir / "README_static_pcmci.md").write_text(readme_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_arguments(args)

    train_path = resolve_repo_path(args.train_path)
    if not train_path.exists():
        raise FileNotFoundError(f"Cannot find train.csv at: {train_path}")

    output_dir = resolve_output_dir(args.output_dir, args.mode)
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Final output directory: {output_dir}", args.verbosity)
    log(f"Constraint profile: {args.constraint_profile}", args.verbosity)
    if args.min_segment_len <= args.tau_max:
        log(
            "Warning: --min_segment_len is not larger than --tau_max. "
            "The run is still allowed, but very short segments can make lagged estimates unstable.",
            args.verbosity,
        )
    if args.alpha_level >= 0.05:
        log(
            "Warning: --alpha_level >= 0.05 is permissive for large PCMCI samples and can yield very dense graphs. "
            "Consider 0.01 or smaller if you want a sparser graph.",
            args.verbosity,
        )

    prepared = prepare_pcmci_data(
        train_path=train_path,
        mode=args.mode,
        min_segment_len=args.min_segment_len,
        verbosity=args.verbosity,
    )
    constraint_spec = build_constraint_spec(
        feature_columns=prepared.feature_columns,
        profile=args.constraint_profile,
    )
    log(
        "Prepared PCMCI input with "
        f"{prepared.rows_used_in_pcmci} rows across {prepared.kept_segments} kept segments "
        f"(out of {prepared.total_segments} detected, sampling frequency {format_timedelta(prepared.expected_delta)}).",
        args.verbosity,
    )
    if prepared.mode == "daytime" and prepared.rows_after_mode_filter > 0:
        dropped_after_segmenting = 1.0 - (prepared.rows_used_in_pcmci / prepared.rows_after_mode_filter)
        if dropped_after_segmenting > 0.25:
            log(
                "Warning: daytime mode dropped "
                f"{dropped_after_segmenting:.1%} of rows after segment filtering. "
                "A high --min_segment_len can bias the graph toward longer daytime segments.",
                args.verbosity,
            )

    results = run_pcmci(
        prepared=prepared,
        tau_max=args.tau_max,
        pc_alpha=args.pc_alpha,
        alpha_level=args.alpha_level,
        verbosity=args.verbosity,
        heartbeat_seconds=args.heartbeat_seconds,
    )

    log("[3/6] Expanding lag-level PCMCI results back to the full feature space", args.verbosity)
    p_matrix_full, val_matrix_full, graph_matrix_full = expand_results_to_full_feature_space(
        results=results,
        full_feature_columns=prepared.feature_columns,
        active_feature_columns=prepared.active_feature_columns,
        tau_max=args.tau_max,
    )

    log("[4/6] Applying constraint-aware edge aggregation into variable-level adjacency", args.verbosity)
    edge_df = build_edge_table(
        full_feature_columns=prepared.feature_columns,
        p_matrix_full=p_matrix_full,
        val_matrix_full=val_matrix_full,
        graph_matrix_full=graph_matrix_full,
        alpha_level=args.alpha_level,
        excluded_constant_columns=prepared.excluded_constant_columns,
        constraint_spec=constraint_spec,
    )
    adjacency_raw, adjacency_for_mask = build_variable_level_adjacency(
        edge_df=edge_df,
        feature_columns=prepared.feature_columns,
    )
    graph_matrix_for_plot, val_matrix_for_plot = build_plot_matrices_from_edge_table(
        feature_columns=prepared.feature_columns,
        edge_df=edge_df,
        graph_matrix_full=graph_matrix_full,
        val_matrix_full=val_matrix_full,
    )

    graph_png_created = save_causal_graph(
        full_feature_columns=prepared.feature_columns,
        graph_matrix_full=graph_matrix_for_plot,
        val_matrix_full=val_matrix_for_plot,
        output_path=output_dir / "causal_graph.png",
    )

    summary = build_summary(
        args=args,
        train_path=train_path,
        output_dir=output_dir,
        prepared=prepared,
        edge_df=edge_df,
        constraint_spec=constraint_spec,
        graph_png_created=graph_png_created,
    )
    export_results(
        output_dir=output_dir,
        prepared=prepared,
        edge_df=edge_df,
        adjacency_raw=adjacency_raw,
        adjacency_for_mask=adjacency_for_mask,
        p_matrix_full=p_matrix_full,
        val_matrix_full=val_matrix_full,
        graph_matrix_full=graph_matrix_full,
        graph_matrix_for_plot=graph_matrix_for_plot,
        val_matrix_for_plot=val_matrix_for_plot,
        summary=summary,
        verbosity=args.verbosity,
    )

    log("[6/6] Static PCMCI export completed successfully.", args.verbosity)


if __name__ == "__main__":
    main()
