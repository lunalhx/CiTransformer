from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.project_config import load_project_config, resolve_project_path


PROJECT_CONFIG = load_project_config()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize iTransformer causal reward validation/test runs.")
    parser.add_argument(
        "--results_root",
        type=str,
        default=str(PROJECT_CONFIG.get_path("paths.results_root")),
        help="Experiment results root.",
    )
    parser.add_argument(
        "--causal_reward_root",
        type=str,
        default=None,
        help="Root containing gamma_<value>/pred_len_<N>/metrics.json runs.",
    )
    parser.add_argument("--pred_lens", nargs="+", type=int, default=[12, 24, 48], help="Prediction lengths.")
    parser.add_argument(
        "--gammas",
        nargs="+",
        type=float,
        default=[0.1, 0.25, 0.5, 1.0, 2.0],
        help="Gamma values to summarize.",
    )
    parser.add_argument("--baseline_none_root", type=str, default=None, help="Matched no-mask baseline root.")
    parser.add_argument("--baseline_soft_beta1_root", type=str, default=None, help="soft_bias beta=1 baseline root.")
    parser.add_argument(
        "--baseline_dynamic_soft_root",
        type=str,
        default=None,
        help="Dynamic regime soft_bias baseline root.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for summary CSV/Markdown. Defaults to <causal_reward_root>/summary.",
    )
    return parser.parse_args()


def resolve_path(path_value: str | Path) -> Path:
    return resolve_project_path(path_value, PROJECT_ROOT)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def gamma_dir_name(gamma: float) -> str:
    return f"gamma_{gamma:.1f}".replace(".", "p") if float(gamma).is_integer() else f"gamma_{gamma:g}".replace(".", "p")


def daytime_rmse(payload: dict[str, Any] | None, split: str) -> float | None:
    if payload is None:
        return None
    metrics = payload.get(f"{split}_metrics")
    if metrics is None and split == "test":
        metrics = payload.get("test_metrics")
        if metrics is None and payload.get("report_split") == "test":
            metrics = payload.get("reported_metrics")
    if metrics is None and split == "validation":
        metrics = payload.get("validation_metrics") or payload.get("reported_metrics")
    if not metrics:
        return None
    daytime = metrics.get("daytime_only") or {}
    value = daytime.get("rmse")
    return float(value) if value is not None else None


def delta(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    return float(candidate - baseline)


def max_reward_from_payload(payload: dict[str, Any] | None) -> float | None:
    if payload is None:
        return None
    if payload.get("reward_max") is not None:
        return float(payload["reward_max"])
    causal_mask = payload.get("causal_mask") or {}
    if causal_mask.get("reward_max") is not None:
        return float(causal_mask["reward_max"])
    regime_mask = payload.get("regime_causal_mask") or {}
    regime_metadata = regime_mask.get("regime_metadata") or {}
    reward_values = [
        float(metadata["reward_max"])
        for metadata in regime_metadata.values()
        if metadata.get("reward_max") is not None
    ]
    return max(reward_values) if reward_values else None


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    results_root = resolve_path(args.results_root)
    causal_reward_root = (
        resolve_path(args.causal_reward_root)
        if args.causal_reward_root
        else results_root / "itransformer_causal_reward"
    )
    baseline_none_root = (
        resolve_path(args.baseline_none_root)
        if args.baseline_none_root
        else results_root / "itransformer_mask_calibration" / "none_matched"
    )
    baseline_soft_beta1_root = (
        resolve_path(args.baseline_soft_beta1_root)
        if args.baseline_soft_beta1_root
        else results_root / "itransformer_mask_calibration" / "soft_bias_beta_1"
    )
    baseline_dynamic_soft_root = (
        resolve_path(args.baseline_dynamic_soft_root)
        if args.baseline_dynamic_soft_root
        else results_root / "itransformer_regime_dynamic_pcmci_k7"
    )

    rows: list[dict[str, Any]] = []
    for pred_len in args.pred_lens:
        none_payload = load_json(baseline_none_root / f"pred_len_{pred_len}" / "metrics.json")
        soft_payload = load_json(baseline_soft_beta1_root / f"pred_len_{pred_len}" / "metrics.json")
        dynamic_payload = load_json(baseline_dynamic_soft_root / f"pred_len_{pred_len}" / "metrics.json")
        none_rmse = daytime_rmse(none_payload, "validation")
        soft_rmse = daytime_rmse(soft_payload, "validation")
        dynamic_rmse = daytime_rmse(dynamic_payload, "validation")

        for gamma in args.gammas:
            metrics_path = causal_reward_root / gamma_dir_name(gamma) / f"pred_len_{pred_len}" / "metrics.json"
            payload = load_json(metrics_path)
            validation_rmse = daytime_rmse(payload, "validation")
            test_rmse = daytime_rmse(payload, "test")
            notes = []
            if payload is None:
                notes.append("missing_metrics")
            elif validation_rmse is None:
                notes.append("missing_validation_rmse")
            if none_rmse is None:
                notes.append("missing_none_baseline")
            if soft_rmse is None:
                notes.append("missing_soft_beta1_baseline")
            if dynamic_rmse is None:
                notes.append("missing_dynamic_soft_baseline")

            rows.append(
                {
                    "pred_len": int(pred_len),
                    "gamma": float(gamma),
                    "validation_daytime_rmse": validation_rmse,
                    "test_daytime_rmse": test_rmse,
                    "delta_vs_none": delta(validation_rmse, none_rmse),
                    "delta_vs_soft_beta1": delta(validation_rmse, soft_rmse),
                    "delta_vs_dynamic_soft": delta(validation_rmse, dynamic_rmse),
                    "best_epoch": payload.get("best_epoch") if payload else None,
                    "reward_max": max_reward_from_payload(payload),
                    "notes": ";".join(notes),
                    "metrics_path": str(metrics_path),
                }
            )
    return rows


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "pred_len",
        "gamma",
        "validation_daytime_rmse",
        "test_daytime_rmse",
        "delta_vs_none",
        "delta_vs_soft_beta1",
        "delta_vs_dynamic_soft",
        "best_epoch",
        "reward_max",
        "notes",
        "metrics_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fieldnames})


def format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def write_markdown(rows: list[dict[str, Any]], output_path: Path) -> None:
    lines = [
        "# iTransformer Causal Reward Summary",
        "",
        "Deltas are validation daytime RMSE differences; negative values favor causal reward.",
        "",
        "| pred_len | gamma | val daytime RMSE | test daytime RMSE | delta vs none | delta vs soft beta=1 | delta vs dynamic soft | best epoch | reward max | notes |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {pred_len} | {gamma:g} | {val} | {test} | {none} | {soft} | {dynamic} | {epoch} | {reward} | {notes} |".format(
                pred_len=row["pred_len"],
                gamma=row["gamma"],
                val=format_float(row["validation_daytime_rmse"]),
                test=format_float(row["test_daytime_rmse"]),
                none=format_float(row["delta_vs_none"]),
                soft=format_float(row["delta_vs_soft_beta1"]),
                dynamic=format_float(row["delta_vs_dynamic_soft"]),
                epoch="" if row["best_epoch"] is None else row["best_epoch"],
                reward=format_float(row["reward_max"]),
                notes=row["notes"],
            )
        )

    complete_rows = [row for row in rows if row["validation_daytime_rmse"] is not None]
    if complete_rows:
        lines.extend(["", "## Best Validation Runs", ""])
        for pred_len in sorted({int(row["pred_len"]) for row in complete_rows}):
            candidates = [row for row in complete_rows if int(row["pred_len"]) == pred_len]
            best = min(candidates, key=lambda row: float(row["validation_daytime_rmse"]))
            lines.append(
                f"- pred_len={pred_len}: gamma={best['gamma']:g}, validation daytime RMSE={best['validation_daytime_rmse']:.6f}"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    causal_reward_root = (
        resolve_path(args.causal_reward_root)
        if args.causal_reward_root
        else resolve_path(args.results_root) / "itransformer_causal_reward"
    )
    output_dir = resolve_path(args.output_dir) if args.output_dir else causal_reward_root / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows(args)
    csv_path = output_dir / "causal_reward_summary.csv"
    markdown_path = output_dir / "causal_reward_summary.md"
    write_csv(rows, csv_path)
    write_markdown(rows, markdown_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
