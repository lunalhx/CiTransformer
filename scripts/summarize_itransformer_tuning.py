#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd


CONFIG_FIELDS = [
    "seq_len",
    "learning_rate",
    "batch_size",
    "d_model",
    "d_ff",
    "e_layers",
    "dropout",
    "weight_decay",
    "activation",
    "n_heads",
    "factor",
    "disable_norm",
    "output_attention",
    "seed",
]

BEST_TSV_FIELDS = [
    "seq_len",
    "learning_rate",
    "batch_size",
    "d_model",
    "d_ff",
    "e_layers",
    "dropout",
    "weight_decay",
    "activation",
    "n_heads",
    "factor",
    "disable_norm",
    "output_attention",
    "seed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize validation-first iTransformer tuning runs into sortable ranking tables."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="results/tuning/itransformer",
        help="Root directory that contains tuning experiment subdirectories.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for summary CSV/JSON files. Defaults to <root_dir>/summary.",
    )
    parser.add_argument(
        "--pred_lens",
        type=int,
        nargs="+",
        default=[12, 24, 48],
        help="Prediction lengths to include in the shared-config ranking.",
    )
    parser.add_argument(
        "--stage_filter",
        type=str,
        default=None,
        help="Optional tuning stage filter, for example: s1 or s2.",
    )
    parser.add_argument(
        "--print_best_tsv",
        action="store_true",
        help="Print the best shared configuration as a single TSV line for shell automation.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def get_nested_metric(payload: dict[str, Any], *keys: str) -> float | None:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if current is None:
        return None
    value = float(current)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def infer_stage(payload: dict[str, Any], metrics_path: Path) -> str | None:
    stage = payload.get("tuning_stage")
    if stage is not None:
        return str(stage)

    experiment_name = payload.get("experiment_name")
    if isinstance(experiment_name, str) and experiment_name:
        return experiment_name.split("_", 1)[0]

    run_dir_name = metrics_path.parent.name
    if "_" in run_dir_name:
        return run_dir_name.split("_", 1)[0]
    return None


def build_signature(config: dict[str, Any]) -> str:
    parts = []
    for field in CONFIG_FIELDS:
        parts.append(f"{field}={config.get(field)}")
    return "|".join(parts)


def collect_rows(root_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(root_dir.rglob("metrics.json")):
        payload = load_json(metrics_path)
        config = payload.get("config", {})
        validation_metrics = payload.get("validation_metrics") or {}
        row: dict[str, Any] = {
            "metrics_path": str(metrics_path),
            "run_dir": str(metrics_path.parent),
            "experiment_name": payload.get("experiment_name") or metrics_path.parent.name,
            "tuning_stage": infer_stage(payload, metrics_path),
            "report_split": payload.get("report_split"),
            "best_epoch": payload.get("best_epoch"),
            "best_validation_loss": payload.get("best_validation_loss"),
            "shared_signature": build_signature(config),
        }

        for field in CONFIG_FIELDS:
            row[field] = config.get(field)
        row["pred_len"] = config.get("pred_len")

        row["val_all_mae"] = get_nested_metric(validation_metrics, "all_timestamps", "mae")
        row["val_all_rmse"] = get_nested_metric(validation_metrics, "all_timestamps", "rmse")
        row["val_daytime_mae"] = get_nested_metric(validation_metrics, "daytime_only", "mae")
        row["val_daytime_rmse"] = get_nested_metric(validation_metrics, "daytime_only", "rmse")
        row["test_all_mae"] = get_nested_metric(payload, "test_metrics", "all_timestamps", "mae")
        row["test_all_rmse"] = get_nested_metric(payload, "test_metrics", "all_timestamps", "rmse")
        row["test_daytime_mae"] = get_nested_metric(payload, "test_metrics", "daytime_only", "mae")
        row["test_daytime_rmse"] = get_nested_metric(payload, "test_metrics", "daytime_only", "rmse")

        rows.append(row)
    return rows


def sort_run_rows(df: pd.DataFrame) -> pd.DataFrame:
    sorted_df = df.sort_values(
        by=[
            "pred_len",
            "val_daytime_rmse",
            "val_daytime_mae",
            "val_all_rmse",
            "val_all_mae",
            "best_validation_loss",
            "experiment_name",
        ],
        ascending=[True, True, True, True, True, True, True],
        na_position="last",
    ).copy()
    sorted_df["rank_within_pred_len"] = sorted_df.groupby("pred_len").cumcount() + 1
    return sorted_df


def build_shared_ranking(df: pd.DataFrame, pred_lens: list[int]) -> pd.DataFrame:
    required_pred_lens = sorted(set(int(pred_len) for pred_len in pred_lens))
    filtered = df[df["pred_len"].isin(required_pred_lens)].copy()
    if filtered.empty:
        return filtered

    grouped = filtered.groupby("shared_signature").agg(
        pred_lens_covered=("pred_len", lambda values: ",".join(str(int(value)) for value in sorted(set(values)))),
        pred_len_count=("pred_len", "nunique"),
        tuning_stages=("tuning_stage", lambda values: ",".join(sorted({str(value) for value in values if pd.notna(value)}))),
        experiment_names=("experiment_name", lambda values: ",".join(sorted({str(value) for value in values if pd.notna(value)}))),
        run_dirs=("run_dir", lambda values: ",".join(sorted({str(value) for value in values if pd.notna(value)}))),
        avg_val_daytime_rmse=("val_daytime_rmse", "mean"),
        max_val_daytime_rmse=("val_daytime_rmse", "max"),
        avg_val_daytime_mae=("val_daytime_mae", "mean"),
        max_val_daytime_mae=("val_daytime_mae", "max"),
        avg_val_all_rmse=("val_all_rmse", "mean"),
        avg_val_all_mae=("val_all_mae", "mean"),
        avg_best_validation_loss=("best_validation_loss", "mean"),
        **{field: (field, "first") for field in CONFIG_FIELDS},
    ).reset_index()

    grouped = grouped[grouped["pred_len_count"] == len(required_pred_lens)].copy()
    grouped = grouped[grouped["pred_lens_covered"] == ",".join(str(pred_len) for pred_len in required_pred_lens)].copy()
    grouped = grouped.sort_values(
        by=[
            "avg_val_daytime_rmse",
            "max_val_daytime_rmse",
            "avg_val_daytime_mae",
            "max_val_daytime_mae",
            "avg_val_all_rmse",
            "avg_val_all_mae",
            "avg_best_validation_loss",
            "shared_signature",
        ],
        ascending=[True, True, True, True, True, True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    grouped["shared_rank"] = grouped.index + 1
    return grouped


def write_outputs(all_runs: pd.DataFrame, shared_ranking: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_runs.to_csv(output_dir / "all_runs.csv", index=False)
    if not all_runs.empty:
        per_pred_len = sort_run_rows(all_runs)
        per_pred_len.to_csv(output_dir / "ranking_by_pred_len.csv", index=False)
    else:
        pd.DataFrame().to_csv(output_dir / "ranking_by_pred_len.csv", index=False)

    shared_ranking.to_csv(output_dir / "ranking_shared_configs.csv", index=False)

    best_shared_payload: dict[str, Any] | None = None
    if not shared_ranking.empty:
        best_shared_payload = shared_ranking.iloc[0].to_dict()

    with (output_dir / "best_shared_config.json").open("w", encoding="utf-8") as fp:
        json.dump(best_shared_payload, fp, ensure_ascii=False, indent=2)


def print_best_tsv(shared_ranking: pd.DataFrame) -> int:
    if shared_ranking.empty:
        return 1

    best_row = shared_ranking.iloc[0]
    values = []
    for field in BEST_TSV_FIELDS:
        value = best_row.get(field)
        values.append("" if value is None else str(value))
    print("\t".join(values))
    return 0


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else root_dir / "summary"

    rows = collect_rows(root_dir)
    all_runs = pd.DataFrame(rows)
    if all_runs.empty:
        write_outputs(all_runs, pd.DataFrame(), output_dir)
        if args.print_best_tsv:
            sys.exit(1)
        print(f"No metrics.json files found under {root_dir}")
        return

    if args.stage_filter:
        all_runs = all_runs[all_runs["tuning_stage"] == args.stage_filter].copy()

    all_runs = sort_run_rows(all_runs)
    shared_ranking = build_shared_ranking(all_runs, args.pred_lens)
    write_outputs(all_runs, shared_ranking, output_dir)

    if args.print_best_tsv:
        sys.exit(print_best_tsv(shared_ranking))

    print(f"Saved tuning summary to {output_dir}")
    print(f"- all runs: {output_dir / 'all_runs.csv'}")
    print(f"- per pred_len ranking: {output_dir / 'ranking_by_pred_len.csv'}")
    print(f"- shared config ranking: {output_dir / 'ranking_shared_configs.csv'}")
    print(f"- best shared config: {output_dir / 'best_shared_config.json'}")


if __name__ == "__main__":
    main()
