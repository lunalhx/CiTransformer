from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT_FALLBACK = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT_FALLBACK / ".cache" / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI
except ImportError as exc:  # pragma: no cover - exercised only when dependency is missing
    raise ImportError(
        "Missing dependency `tigramite`. Install the causal-discovery dependencies with:\n"
        "  pip install tigramite networkx scikit-learn pandas numpy matplotlib"
    ) from exc


VARIABLES = [
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

TARGET_COLUMN = "Active_Pow"
RADIATION_COLUMNS = {"Radiation_Global_Tilted", "Radiation_Diffuse_Tilted"}
WEATHER_COLUMNS = {"Weather_T", "Weather_R"}
EXOGENOUS_TIME_COLUMNS = {
    "solar_elevation",
    "sin_time_of_day",
    "cos_time_of_day",
    "sin_day_of_year",
    "cos_day_of_year",
    "day_night_label",
}
NON_EXOGENOUS_COLUMNS = {TARGET_COLUMN, *RADIATION_COLUMNS, *WEATHER_COLUMNS}
TIME_COLUMN_CANDIDATES = ["timestamp", "datetime", "date", "time", "Time", "DateTime"]
DEFAULT_OUTPUT_DIR = "results/causal_graphs/global_pcmci_11vars_train"
DEFAULT_FREQ_MINUTES = 5
BLOCKED_ATTENTION_VALUE = -1e9


@dataclass
class PreparedPCMCIData:
    dataframe: pp.DataFrame
    segment_arrays: dict[int, np.ndarray] | np.ndarray
    analysis_mode: str
    variables: list[str]
    rows_used: int
    segment_count: int
    segment_lengths: list[int]
    dropped_short_segments: int
    dropped_missing_rows: int
    scaler_mean: list[float]
    scaler_scale: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a train-only global PCMCI causal graph for PV power forecasting "
            "and export a variable-level causal mask for iTransformer attention."
        )
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default=None,
        help="Optional path to train.csv. If omitted, only project train split candidates are searched.",
    )
    parser.add_argument("--tau_min", type=int, default=1, help="Minimum lag for PCMCI. Use 1 to avoid contemporaneous edges.")
    parser.add_argument("--tau_max", type=int, default=12, help="Maximum lag for PCMCI. 12 means 60 minutes for 5-min data.")
    parser.add_argument("--pc_alpha", type=float, default=0.05, help="PC stage alpha for PCMCI.")
    parser.add_argument("--alpha_level", type=float, default=0.05, help="Significance threshold for exported edges.")
    parser.add_argument(
        "--fdr_method",
        type=str,
        default="fdr_bh",
        help="Tigramite FDR correction method, e.g. fdr_bh or none.",
    )
    parser.add_argument(
        "--topk_active",
        type=int,
        default=5,
        help="Maximum number of parent variables kept for Active_Pow after aggregation.",
    )
    parser.add_argument(
        "--topk_other",
        type=int,
        default=3,
        help="Maximum number of parent variables kept for each non-Active_Pow target.",
    )
    parser.add_argument(
        "--min_segment_length",
        type=int,
        default=24,
        help="Drop continuous selected-data segments shorter than this many rows before PCMCI.",
    )
    parser.add_argument(
        "--freq_minutes",
        type=int,
        default=DEFAULT_FREQ_MINUTES,
        help="Expected sampling interval in minutes. Defaults to 5.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where graph, CSV, NPY and JSON outputs are saved.",
    )
    parser.add_argument(
        "--sample_scope",
        type=str,
        default="full_train",
        choices=["full_train", "daytime"],
        help="Rows used for PCMCI. full_train uses every train row; daytime preserves the old day_night/solar filter.",
    )
    return parser.parse_args()


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / "requirements.txt").exists() or (parent / "data").exists():
            return parent
    return PROJECT_ROOT_FALLBACK


def resolve_repo_path(path: str | Path, project_root: Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else project_root / candidate


def resolve_train_path(project_root: Path, train_path: str | None = None) -> Path:
    if train_path is not None:
        explicit_path = resolve_repo_path(train_path, project_root)
        if explicit_path.exists():
            return explicit_path
        raise FileNotFoundError(f"Specified --train_path does not exist: {explicit_path}")

    candidates = [
        project_root / "data/processed_selected_2020_2022/splits/train.csv",
        # Practical fallback for projects that store split files directly under the processed directory.
        project_root / "data/processed_selected_2020_2022/train.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Cannot find train.csv. Checked:\n" + "\n".join(f"  - {path}" for path in candidates)
    )


def detect_time_column(df: pd.DataFrame) -> str:
    for column in TIME_COLUMN_CANDIDATES:
        if column in df.columns:
            try:
                parsed = pd.to_datetime(df[column], errors="coerce")
            except Exception:
                continue
            if float(parsed.notna().mean()) >= 0.95:
                return column

    raise ValueError(
        "Failed to detect a timestamp column. Expected one of "
        f"{TIME_COLUMN_CANDIDATES}. Available columns: {list(df.columns)}"
    )


def load_train_data(path: Path) -> tuple[pd.DataFrame, str, int]:
    df = pd.read_csv(path)
    rows_loaded = len(df)
    time_col = detect_time_column(df)
    timestamp = pd.to_datetime(df[time_col], errors="raise")
    df = df.copy()
    df[time_col] = timestamp
    df = df.sort_values(time_col).reset_index(drop=True)

    duplicated = int(df[time_col].duplicated().sum())
    if duplicated:
        raise ValueError(f"{path} contains {duplicated} duplicated timestamps in column `{time_col}`.")

    return df, time_col, rows_loaded


def validate_required_columns(df: pd.DataFrame, variables: list[str]) -> None:
    missing = [column for column in variables if column not in df.columns]
    if not missing:
        return
    raise KeyError(
        "train.csv is missing required PCMCI columns.\n"
        f"Missing columns: {missing}\n"
        f"Required columns: {variables}\n"
        f"Available columns: {list(df.columns)}"
    )


def filter_daytime(df: pd.DataFrame) -> pd.DataFrame:
    if "day_night_label" in df.columns:
        daytime_df = df.loc[df["day_night_label"] == 1].copy()
    elif "solar_elevation" in df.columns:
        daytime_df = df.loc[df["solar_elevation"] > 0].copy()
    else:
        raise KeyError(
            "Cannot apply daytime-only filtering: neither `day_night_label` nor `solar_elevation` exists. "
            f"Available columns: {list(df.columns)}"
        )

    if daytime_df.empty:
        raise ValueError("No rows remain after daytime-only filtering.")
    return daytime_df


def select_pcmci_rows(df: pd.DataFrame, sample_scope: str) -> tuple[pd.DataFrame, str]:
    if sample_scope == "full_train":
        return df.copy(), "full train split; no daytime filtering"
    if sample_scope == "daytime":
        return filter_daytime(df), "day_night_label == 1 if available, otherwise solar_elevation > 0"
    raise ValueError(f"Unsupported sample_scope: {sample_scope}")


def split_continuous_segments(
    df: pd.DataFrame,
    time_col: str,
    freq_minutes: int,
    min_segment_length: int,
) -> tuple[list[pd.DataFrame], list[int], int]:
    if freq_minutes <= 0:
        raise ValueError("--freq_minutes must be positive.")
    if min_segment_length <= 0:
        raise ValueError("--min_segment_length must be positive.")

    working_df = df.sort_values(time_col).reset_index(drop=True).copy()
    expected_delta = pd.Timedelta(minutes=freq_minutes)
    diffs = working_df[time_col].diff()
    is_new_segment = diffs.isna() | (diffs != expected_delta)
    segment_ids = is_new_segment.cumsum() - 1

    kept_segments: list[pd.DataFrame] = []
    all_lengths: list[int] = []
    dropped_short = 0

    for _, segment_df in working_df.groupby(segment_ids, sort=True):
        segment = segment_df.copy()
        length = int(len(segment))
        all_lengths.append(length)
        if length >= min_segment_length:
            kept_segments.append(segment)
        else:
            dropped_short += 1

    if not kept_segments:
        longest = max(all_lengths) if all_lengths else 0
        raise ValueError(
            "No continuous selected-data segments satisfy --min_segment_length. "
            f"Expected interval: {expected_delta}. Total segments: {len(all_lengths)}. "
            f"Longest segment length: {longest}."
        )

    return kept_segments, all_lengths, dropped_short


def prepare_pcmci_dataframe(
    segments: list[pd.DataFrame],
    variables: list[str],
    dropped_missing_rows: int = 0,
) -> PreparedPCMCIData:
    if any(segment[variables].isna().any(axis=None) for segment in segments):
        raise ValueError(
            "Missing values must be dropped before segment detection, otherwise removed rows can create false lags."
        )

    scaler_source = pd.concat([segment.loc[:, variables] for segment in segments], axis=0)
    std = scaler_source.std(ddof=0)
    constant_columns = [column for column in variables if pd.isna(std[column]) or np.isclose(std[column], 0.0)]
    if constant_columns:
        raise ValueError(
            "PCMCI variables contain constants after row filtering, which makes ParCorr ill-conditioned.\n"
            f"Constant columns: {constant_columns}"
        )

    scaler = StandardScaler()
    scaler.fit(scaler_source.loc[:, variables].to_numpy(dtype=np.float64))

    segment_dict: dict[int, np.ndarray] = {}
    segment_lengths: list[int] = []
    for segment_id, segment in enumerate(segments):
        values = scaler.transform(segment.loc[:, variables].to_numpy(dtype=np.float64)).astype(np.float64)
        if values.shape[0] > 0:
            segment_dict[segment_id] = values
            segment_lengths.append(int(values.shape[0]))

    if len(segment_dict) == 1:
        analysis_mode = "single"
        segment_arrays: dict[int, np.ndarray] | np.ndarray = next(iter(segment_dict.values()))
    else:
        analysis_mode = "multiple"
        segment_arrays = segment_dict

    dataframe = pp.DataFrame(segment_arrays, var_names=variables, analysis_mode=analysis_mode)
    return PreparedPCMCIData(
        dataframe=dataframe,
        segment_arrays=segment_arrays,
        analysis_mode=analysis_mode,
        variables=variables,
        rows_used=int(sum(segment_lengths)),
        segment_count=len(segment_lengths),
        segment_lengths=segment_lengths,
        dropped_short_segments=0,
        dropped_missing_rows=dropped_missing_rows,
        scaler_mean=[float(value) for value in scaler.mean_],
        scaler_scale=[float(value) for value in scaler.scale_],
    )


def build_physical_prior(variables: list[str]) -> dict[str, Any]:
    return {
        "target_output": [TARGET_COLUMN],
        "radiation": sorted(RADIATION_COLUMNS & set(variables)),
        "weather": sorted(WEATHER_COLUMNS & set(variables)),
        "exogenous_time": sorted(EXOGENOUS_TIME_COLUMNS & set(variables)),
        "rules": [
            "Exogenous time/label variables may point into target, radiation, and weather variables.",
            "No variable may point into exogenous time/label variables, except each variable attends to itself in the final mask.",
            "Cross edges among exogenous time/label variables are removed.",
            "Active_Pow cannot point to radiation, weather, or exogenous time variables.",
            "Non-exogenous variables may keep lagged autoregressive edges.",
            "Radiation, weather, and exogenous time variables may point into Active_Pow.",
            "solar_elevation may point into tilted radiation variables.",
            "Weather_T and Weather_R may point into tilted radiation variables.",
            "Cross-weather and cross-radiation edges are not kept in this first global mask.",
        ],
    }


def is_edge_allowed(source: str, target: str, lag: int) -> bool:
    if lag < 1:
        return False

    if source == target:
        return source in NON_EXOGENOUS_COLUMNS

    if target in EXOGENOUS_TIME_COLUMNS:
        return False

    if source in EXOGENOUS_TIME_COLUMNS:
        return target in NON_EXOGENOUS_COLUMNS

    if source == TARGET_COLUMN and target != TARGET_COLUMN:
        return False

    if target == TARGET_COLUMN:
        return source in (RADIATION_COLUMNS | WEATHER_COLUMNS)

    if target in RADIATION_COLUMNS:
        return source in ({"solar_elevation"} | WEATHER_COLUMNS)

    if target in WEATHER_COLUMNS:
        return False

    return False


def run_pcmci(
    dataframe: pp.DataFrame,
    tau_min: int,
    tau_max: int,
    pc_alpha: float,
    alpha_level: float,
    fdr_method: str,
) -> dict[str, Any]:
    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=ParCorr(significance="analytic"),
        verbosity=0,
    )
    try:
        return pcmci.run_pcmci(
            tau_min=tau_min,
            tau_max=tau_max,
            pc_alpha=pc_alpha,
            alpha_level=alpha_level,
            fdr_method=fdr_method,
        )
    except TypeError:
        # Older Tigramite versions may not expose tau_min in run_pcmci; post-processing still enforces tau_min.
        return pcmci.run_pcmci(
            tau_max=tau_max,
            pc_alpha=pc_alpha,
            alpha_level=alpha_level,
            fdr_method=fdr_method,
        )


def get_effective_p_matrix(results: dict[str, Any], fdr_method: str) -> np.ndarray:
    p_matrix = np.asarray(results["p_matrix"], dtype=np.float64)
    # Tigramite versions differ slightly in naming; prefer explicitly corrected values if present.
    if fdr_method != "none":
        for key in ("q_matrix", "p_matrix_corrected", "corrected_p_matrix"):
            if key in results:
                return np.asarray(results[key], dtype=np.float64)
    return p_matrix


def extract_significant_edges(
    results: dict[str, Any],
    variables: list[str],
    alpha_level: float,
    tau_min: int,
    tau_max: int,
    fdr_method: str,
) -> tuple[list[dict[str, Any]], int, int]:
    p_matrix = get_effective_p_matrix(results, fdr_method=fdr_method)
    val_matrix = np.asarray(results["val_matrix"], dtype=np.float64)

    raw_significant_count = 0
    filtered_by_prior_count = 0
    edges: list[dict[str, Any]] = []
    max_available_tau = min(tau_max, p_matrix.shape[2] - 1)

    for source_idx, source in enumerate(variables):
        for target_idx, target in enumerate(variables):
            for lag in range(max(1, tau_min), max_available_tau + 1):
                p_value = p_matrix[source_idx, target_idx, lag]
                val = val_matrix[source_idx, target_idx, lag]
                if not np.isfinite(p_value) or p_value >= alpha_level:
                    continue

                raw_significant_count += 1
                allowed = is_edge_allowed(source=source, target=target, lag=lag)
                if not allowed:
                    filtered_by_prior_count += 1
                    continue

                edges.append(
                    {
                        "source": source,
                        "target": target,
                        "lag": int(lag),
                        "mci": float(val) if np.isfinite(val) else np.nan,
                        "abs_mci": float(abs(val)) if np.isfinite(val) else np.nan,
                        "p_value": float(p_value),
                    }
                )

    return edges, raw_significant_count, filtered_by_prior_count


def aggregate_lag_edges(edges: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "source",
        "target",
        "significant_lags",
        "strongest_lag",
        "max_abs_mci",
        "signed_mci_at_strongest_lag",
        "min_p_value",
        "sign",
        "lag_count",
    ]
    if not edges:
        return pd.DataFrame(columns=columns)

    edge_df = pd.DataFrame(edges)
    rows: list[dict[str, Any]] = []
    for (source, target), group in edge_df.groupby(["source", "target"], sort=False):
        group = group.sort_values(["abs_mci", "p_value"], ascending=[False, True])
        strongest = group.iloc[0]
        significant_lags = sorted(int(lag) for lag in group["lag"].tolist())
        signed_mci = float(strongest["mci"])
        rows.append(
            {
                "source": source,
                "target": target,
                "significant_lags": ",".join(str(lag) for lag in significant_lags),
                "strongest_lag": int(strongest["lag"]),
                "max_abs_mci": float(abs(signed_mci)),
                "signed_mci_at_strongest_lag": signed_mci,
                "min_p_value": float(group["p_value"].min()),
                "sign": "positive" if signed_mci >= 0 else "negative",
                "lag_count": int(len(significant_lags)),
            }
        )

    return pd.DataFrame(rows, columns=columns).sort_values(
        ["target", "max_abs_mci", "min_p_value"], ascending=[True, False, True]
    )


def apply_topk_filter(edges: pd.DataFrame, topk_active: int, topk_other: int) -> pd.DataFrame:
    if topk_active <= 0 or topk_other <= 0:
        raise ValueError("--topk_active and --topk_other must be positive.")
    if edges.empty:
        return edges.copy()

    kept_frames: list[pd.DataFrame] = []
    for target, group in edges.groupby("target", sort=False):
        topk = topk_active if target == TARGET_COLUMN else topk_other
        group = group.sort_values(["max_abs_mci", "min_p_value"], ascending=[False, True])
        kept_frames.append(group.head(topk))

    return pd.concat(kept_frames, axis=0).sort_values(
        ["target", "max_abs_mci", "min_p_value"], ascending=[True, False, True]
    )


def build_adjacency_matrix(edges: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    # Direction is intentionally target x source for iTransformer variable attention:
    # adjacency[target, source] = 1 means the target query may attend to source key/value.
    adjacency = pd.DataFrame(0, index=variables, columns=variables, dtype=np.int64)
    for variable in variables:
        adjacency.loc[variable, variable] = 1

    for row in edges.itertuples(index=False):
        adjacency.loc[row.target, row.source] = 1

    return adjacency


def build_attention_masks(adjacency: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    binary_adjacency = adjacency.to_numpy(dtype=np.float32)
    additive_mask = np.where(binary_adjacency > 0.0, 0.0, BLOCKED_ATTENTION_VALUE).astype(np.float32)
    return binary_adjacency, additive_mask


def plot_causal_graph(edges: pd.DataFrame, variables: list[str], output_path: Path) -> None:
    graph = nx.DiGraph()
    graph.add_nodes_from(variables)

    for row in edges.itertuples(index=False):
        if row.source == row.target:
            continue
        graph.add_edge(
            row.source,
            row.target,
            label=f"L{row.strongest_lag}, |MCI|={row.max_abs_mci:.3f}",
            weight=float(row.max_abs_mci),
        )

    plt.figure(figsize=(14, 9))
    if graph.number_of_edges() == 0:
        pos = nx.shell_layout(graph)
    else:
        pos = nx.spring_layout(graph, seed=42, k=1.2)

    node_colors = []
    for node in graph.nodes:
        if node == TARGET_COLUMN:
            node_colors.append("#f4a261")
        elif node in RADIATION_COLUMNS:
            node_colors.append("#e9c46a")
        elif node in WEATHER_COLUMNS:
            node_colors.append("#8ab17d")
        else:
            node_colors.append("#7db7d8")

    nx.draw_networkx_nodes(graph, pos, node_size=2600, node_color=node_colors, edgecolors="#333333", linewidths=1.0)
    nx.draw_networkx_labels(graph, pos, font_size=9, font_weight="bold")
    nx.draw_networkx_edges(
        graph,
        pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=18,
        width=1.8,
        connectionstyle="arc3,rad=0.08",
        edge_color="#444444",
    )
    edge_labels = nx.get_edge_attributes(graph, "label")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8, label_pos=0.55)

    plt.title("Global PCMCI Causal Graph (train only)", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def dataframe_to_json_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    return json.loads(df.to_json(orient="records", force_ascii=False))


def save_outputs(
    output_dir: Path,
    results: dict[str, Any],
    aggregated_edges: pd.DataFrame,
    topk_edges: pd.DataFrame,
    adjacency: pd.DataFrame,
    binary_mask: np.ndarray,
    additive_mask: np.ndarray,
    config: dict[str, Any],
    variables: list[str],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "raw_p_matrix": output_dir / "raw_p_matrix.npy",
        "raw_val_matrix": output_dir / "raw_val_matrix.npy",
        "edges": output_dir / "global_causal_edges.csv",
        "adjacency": output_dir / "global_causal_adjacency.csv",
        "mask": output_dir / "global_causal_mask.npy",
        "additive_mask": output_dir / "global_additive_attention_mask.npy",
        "graph": output_dir / "global_causal_graph.png",
        "config": output_dir / "global_pcmci_config.json",
        "active_parents": output_dir / "active_pow_parents.csv",
    }

    np.save(paths["raw_p_matrix"], np.asarray(results["p_matrix"], dtype=np.float64))
    np.save(paths["raw_val_matrix"], np.asarray(results["val_matrix"], dtype=np.float64))

    topk_edges.to_csv(paths["edges"], index=False)
    adjacency.to_csv(paths["adjacency"], index=True, index_label="target\\source")
    np.save(paths["mask"], binary_mask)
    np.save(paths["additive_mask"], additive_mask)

    active_parents = topk_edges.loc[topk_edges["target"] == TARGET_COLUMN].copy()
    active_parents = active_parents.sort_values(["max_abs_mci", "min_p_value"], ascending=[False, True])
    active_parents.to_csv(paths["active_parents"], index=False)

    config = dict(config)
    config["outputs"] = {name: str(path) for name, path in paths.items()}
    config["variables"] = variables
    config["active_pow_parents"] = dataframe_to_json_records(active_parents)
    paths["config"].write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_causal_graph(topk_edges, variables=variables, output_path=paths["graph"])
    return paths


def validate_arguments(args: argparse.Namespace) -> None:
    if args.tau_min < 0:
        raise ValueError("--tau_min must be >= 0.")
    if args.tau_max < args.tau_min:
        raise ValueError("--tau_max must be >= --tau_min.")
    if args.tau_max <= 0:
        raise ValueError("--tau_max must be positive.")
    if not 0 < args.pc_alpha <= 1:
        raise ValueError("--pc_alpha must be in (0, 1].")
    if not 0 < args.alpha_level <= 1:
        raise ValueError("--alpha_level must be in (0, 1].")
    if args.min_segment_length <= args.tau_max:
        raise ValueError("--min_segment_length should be greater than --tau_max.")


def print_final_report(
    paths: dict[str, Path],
    train_path: Path,
    rows_loaded: int,
    selected_rows: int,
    sample_scope: str,
    segment_count: int,
    variables: list[str],
    args: argparse.Namespace,
    raw_significant_count: int,
    filtered_by_prior_count: int,
    topk_edges: pd.DataFrame,
) -> None:
    active_parents = topk_edges.loc[topk_edges["target"] == TARGET_COLUMN].sort_values(
        ["max_abs_mci", "min_p_value"], ascending=[False, True]
    )
    if active_parents.empty:
        parent_text = "None"
    else:
        parent_text = ", ".join(
            f"{row.source}(lags={row.significant_lags}, strongest={row.strongest_lag}, |MCI|={row.max_abs_mci:.4f})"
            for row in active_parents.itertuples(index=False)
        )

    print("\nGlobal PCMCI finished.")
    print(f"Data path: {train_path}")
    print(f"Raw train samples: {rows_loaded}")
    print(f"Sample scope: {sample_scope}")
    print(f"Selected train samples: {selected_rows}")
    print(f"Continuous segments kept: {segment_count}")
    print(f"PCMCI variables: {variables}")
    print(f"tau_min / tau_max: {args.tau_min} / {args.tau_max}")
    print(f"Significant lag-level edges before physical prior: {raw_significant_count}")
    print(f"Edges filtered by physical prior: {filtered_by_prior_count}")
    print(f"Top-K aggregated edges kept: {len(topk_edges)}")
    print(f"Active_Pow parents: {parent_text}")
    print("Output files:")
    for name, path in paths.items():
        print(f"  - {name}: {path}")


def main() -> None:
    args = parse_args()
    validate_arguments(args)

    start_time = time.time()
    project_root = find_project_root()
    train_path = resolve_train_path(project_root, args.train_path)
    output_dir = resolve_repo_path(args.output_dir, project_root)

    print(f"Loading train split only: {train_path}")
    train_df, time_col, rows_loaded = load_train_data(train_path)
    validate_required_columns(train_df, VARIABLES)

    pcmci_candidate_df, row_filter_description = select_pcmci_rows(train_df, sample_scope=args.sample_scope)
    selected_rows = len(pcmci_candidate_df)
    if selected_rows <= args.min_segment_length:
        raise ValueError(
            "Selected data is too small for global PCMCI. "
            f"Rows: {selected_rows}, min_segment_length: {args.min_segment_length}."
        )

    missing_mask = pcmci_candidate_df[VARIABLES].isna().any(axis=1)
    dropped_missing_rows = int(missing_mask.sum())
    if dropped_missing_rows:
        print(
            f"Dropping {dropped_missing_rows} selected train rows with missing PCMCI variables before segment detection; "
            "dropped rows become natural time gaps."
        )
    pcmci_source_df = pcmci_candidate_df.loc[~missing_mask].copy()
    if pcmci_source_df.empty:
        raise ValueError("No usable train rows remain after dropping missing PCMCI variables.")

    segments, all_segment_lengths, dropped_short_segments = split_continuous_segments(
        pcmci_source_df,
        time_col=time_col,
        freq_minutes=args.freq_minutes,
        min_segment_length=args.min_segment_length,
    )
    prepared = prepare_pcmci_dataframe(segments, VARIABLES, dropped_missing_rows=dropped_missing_rows)
    prepared.dropped_short_segments = dropped_short_segments

    print(
        "Running PCMCI + ParCorr "
        f"(segments={prepared.segment_count}, rows={prepared.rows_used}, tau={args.tau_min}..{args.tau_max})"
    )
    results = run_pcmci(
        prepared.dataframe,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        pc_alpha=args.pc_alpha,
        alpha_level=args.alpha_level,
        fdr_method=args.fdr_method,
    )

    lag_edges, raw_significant_count, filtered_by_prior_count = extract_significant_edges(
        results=results,
        variables=VARIABLES,
        alpha_level=args.alpha_level,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        fdr_method=args.fdr_method,
    )
    aggregated_edges = aggregate_lag_edges(lag_edges)
    topk_edges = apply_topk_filter(aggregated_edges, topk_active=args.topk_active, topk_other=args.topk_other)
    adjacency = build_adjacency_matrix(topk_edges, VARIABLES)
    binary_mask, additive_mask = build_attention_masks(adjacency)

    if topk_edges.empty:
        print("WARNING: No significant causal edges remained after physical-prior and Top-K filtering.")

    elapsed_seconds = time.time() - start_time
    config = {
        "script": "causal_algo/run_global_pcmci.py",
        "train_path": str(train_path),
        "output_dir": str(output_dir),
        "time_column": time_col,
        "sample_policy": "train.csv only; validation/calibration/test are never read",
        "sample_scope": args.sample_scope,
        "row_filter": row_filter_description,
        "freq_minutes": args.freq_minutes,
        "raw_train_samples": rows_loaded,
        "selected_train_samples": selected_rows,
        "rows_used_in_pcmci": prepared.rows_used,
        "total_segments_after_filter": len(all_segment_lengths),
        "continuous_segments_kept": prepared.segment_count,
        "segments_dropped_for_being_short": dropped_short_segments,
        "segment_lengths_all": all_segment_lengths,
        "segment_lengths_kept": prepared.segment_lengths,
        "dropped_missing_rows_in_variables": prepared.dropped_missing_rows,
        "analysis_mode": prepared.analysis_mode,
        "tau_min": args.tau_min,
        "tau_max": args.tau_max,
        "pc_alpha": args.pc_alpha,
        "alpha_level": args.alpha_level,
        "fdr_method": args.fdr_method,
        "topk_active": args.topk_active,
        "topk_other": args.topk_other,
        "raw_significant_lag_edges": raw_significant_count,
        "physical_prior_filtered_lag_edges": filtered_by_prior_count,
        "aggregated_edges_before_topk": int(len(aggregated_edges)),
        "aggregated_edges_after_topk": int(len(topk_edges)),
        "adjacency_direction": "adjacency[target, source] = 1; target query may attend to source key/value",
        "mask_format": {
            "global_causal_mask.npy": "binary adjacency, 1=allowed, 0=blocked",
            "global_additive_attention_mask.npy": f"additive mask, 0.0=allowed, {BLOCKED_ATTENTION_VALUE}=blocked",
        },
        "physical_prior": build_physical_prior(VARIABLES),
        "standard_scaler": {
            "fit_scope": f"train {args.sample_scope} continuous segments only",
            "mean": prepared.scaler_mean,
            "scale": prepared.scaler_scale,
        },
        "runtime_seconds": float(elapsed_seconds),
        "generated_at": pd.Timestamp.now().isoformat(),
    }

    paths = save_outputs(
        output_dir=output_dir,
        results=results,
        aggregated_edges=aggregated_edges,
        topk_edges=topk_edges,
        adjacency=adjacency,
        binary_mask=binary_mask,
        additive_mask=additive_mask,
        config=config,
        variables=VARIABLES,
    )

    print_final_report(
        paths=paths,
        train_path=train_path,
        rows_loaded=rows_loaded,
        selected_rows=selected_rows,
        sample_scope=args.sample_scope,
        segment_count=prepared.segment_count,
        variables=VARIABLES,
        args=args,
        raw_significant_count=raw_significant_count,
        filtered_by_prior_count=filtered_by_prior_count,
        topk_edges=topk_edges,
    )


if __name__ == "__main__":
    main()
