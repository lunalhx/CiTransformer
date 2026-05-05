from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_static_pcmci import (
    DEFAULT_SPARSE_MIN_LAG_SUPPORT,
    DEFAULT_SPARSE_TOP_K_PARENTS,
    PreparedPCMCIData,
    build_constraint_spec,
    build_edge_table,
    build_lag_support_count_matrix,
    build_pair_level_support_table,
    build_plot_matrices_from_edge_table,
    build_support_threshold_adjacency,
    build_top_k_parent_adjacency,
    build_variable_level_adjacency,
    enrich_summary_with_sparse_exports,
    export_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill derived CSV/heatmap exports for an existing static PCMCI results directory "
            "without rerunning the expensive PCMCI core step."
        )
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Existing static PCMCI results directory that already contains pcmci_summary.json and pcmci_raw_results.npz.",
    )
    parser.add_argument(
        "--sparse_min_lag_support",
        type=int,
        default=DEFAULT_SPARSE_MIN_LAG_SUPPORT,
        help="Support-threshold sparse adjacency: keep a pair only when at least this many lag supports survive.",
    )
    parser.add_argument(
        "--sparse_top_k_parents",
        type=int,
        default=DEFAULT_SPARSE_TOP_K_PARENTS,
        help="Top-k sparse adjacency: keep at most this many cross-variable parents per target.",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=1,
        help="Verbosity level for export logging.",
    )
    return parser.parse_args()


def build_prepared_from_summary(summary: dict[str, object], feature_columns: list[str]) -> PreparedPCMCIData:
    return PreparedPCMCIData(
        time_column=str(summary["time_column"]),
        mode=str(summary["mode"]),
        feature_columns=feature_columns,
        active_feature_columns=list(summary["active_feature_columns_for_pcmci"]),
        excluded_constant_columns=list(summary.get("excluded_constant_columns", [])),
        rows_loaded=int(summary["rows_loaded_from_train"]),
        rows_after_mode_filter=int(summary["rows_after_mode_filter"]),
        rows_after_missing_drop=int(summary["rows_after_missing_drop"]),
        rows_used_in_pcmci=int(summary["rows_used_in_pcmci"]),
        expected_delta=pd.Timedelta(str(summary["expected_sampling_frequency"])),
        total_segments=int(summary["total_segments_detected"]),
        kept_segments=int(summary["segments_kept_for_pcmci"]),
        dropped_segments_short=int(summary["segments_dropped_for_being_short"]),
        all_segment_lengths=[],
        kept_segment_lengths=[],
        segment_arrays=np.empty(0),
        tigramite_analysis_mode=str(summary["tigramite_analysis_mode"]),
    )


def main() -> None:
    args = parse_args()
    if args.sparse_min_lag_support <= 0:
        raise ValueError("--sparse_min_lag_support must be positive.")
    if args.sparse_top_k_parents <= 0:
        raise ValueError("--sparse_top_k_parents must be positive.")

    results_dir = Path(args.results_dir).expanduser().resolve()
    summary_path = results_dir / "pcmci_summary.json"
    npz_path = results_dir / "pcmci_raw_results.npz"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing raw results file: {npz_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    raw_data = np.load(npz_path, allow_pickle=True)
    feature_columns = [str(value) for value in raw_data["feature_names"]]

    constraint_spec = build_constraint_spec(
        feature_columns=feature_columns,
        profile=str(summary.get("constraint_profile", "none")),
    )
    edge_df = build_edge_table(
        full_feature_columns=feature_columns,
        p_matrix_full=raw_data["p_matrix"],
        val_matrix_full=raw_data["val_matrix"],
        graph_matrix_full=raw_data["graph"],
        alpha_level=float(summary["alpha_level"]),
        excluded_constant_columns=list(summary.get("excluded_constant_columns", [])),
        constraint_spec=constraint_spec,
    )
    adjacency_raw, adjacency_for_mask = build_variable_level_adjacency(
        edge_df=edge_df,
        feature_columns=feature_columns,
    )
    pair_support_df = build_pair_level_support_table(edge_df=edge_df)
    lag_support_df = build_lag_support_count_matrix(
        pair_support_df=pair_support_df,
        feature_columns=feature_columns,
    )
    sparse_support_raw, sparse_support_for_mask = build_support_threshold_adjacency(
        lag_support_df=lag_support_df,
        min_lag_support=args.sparse_min_lag_support,
    )
    top_k_raw, top_k_for_mask = build_top_k_parent_adjacency(
        pair_support_df=pair_support_df,
        feature_columns=feature_columns,
        top_k_parents=args.sparse_top_k_parents,
    )
    graph_matrix_for_plot, val_matrix_for_plot = build_plot_matrices_from_edge_table(
        feature_columns=feature_columns,
        edge_df=edge_df,
        graph_matrix_full=raw_data["graph"],
        val_matrix_full=raw_data["val_matrix"],
    )

    summary["output_dir"] = str(results_dir)
    summary["generated_at_utc"] = pd.Timestamp.utcnow().isoformat()
    summary = enrich_summary_with_sparse_exports(
        summary=summary,
        sparse_min_lag_support=args.sparse_min_lag_support,
        sparse_top_k_parents=args.sparse_top_k_parents,
        pair_support_df=pair_support_df,
        lag_support_df=lag_support_df,
        sparse_support_raw=sparse_support_raw,
        top_k_raw=top_k_raw,
    )

    prepared = build_prepared_from_summary(summary=summary, feature_columns=feature_columns)
    export_results(
        output_dir=results_dir,
        prepared=prepared,
        edge_df=edge_df,
        pair_support_df=pair_support_df,
        adjacency_raw=adjacency_raw,
        adjacency_for_mask=adjacency_for_mask,
        lag_support_df=lag_support_df,
        sparse_support_raw=sparse_support_raw,
        sparse_support_for_mask=sparse_support_for_mask,
        top_k_raw=top_k_raw,
        top_k_for_mask=top_k_for_mask,
        p_matrix_full=raw_data["p_matrix"],
        val_matrix_full=raw_data["val_matrix"],
        graph_matrix_full=raw_data["graph"],
        graph_matrix_for_plot=graph_matrix_for_plot,
        val_matrix_for_plot=val_matrix_for_plot,
        summary=summary,
        verbosity=args.verbosity,
    )

    print(f"Backfilled exports into: {results_dir}")
    print(f"Nonzero pair-level edges: {int((lag_support_df > 0).sum().sum())}")
    print(f"Support-threshold pair edges: {int(sparse_support_raw.to_numpy().sum())}")
    print(f"Top-k pair edges: {int(top_k_raw.to_numpy().sum())}")


if __name__ == "__main__":
    main()
