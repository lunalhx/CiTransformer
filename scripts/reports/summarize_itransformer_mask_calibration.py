from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.project_config import load_project_config, resolve_project_path


PROJECT_CONFIG = load_project_config()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize iTransformer causal mask calibration results.")
    parser.add_argument(
        "--results_root",
        type=str,
        default=str(PROJECT_CONFIG.get_path("paths.results_root")),
        help="Experiment results root.",
    )
    parser.add_argument("--pred_lens", nargs="+", type=int, default=[12, 24, 48], help="Prediction lengths.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for summary CSV/Markdown. Defaults to <results_root>/itransformer_mask_calibration/summary.",
    )
    return parser.parse_args()


def resolve_path(path_value: str | Path) -> Path:
    return resolve_project_path(path_value, PROJECT_ROOT)


def load_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def pct_delta(value: float, baseline: float) -> float:
    return (value - baseline) / baseline * 100.0


def metric_row(case: str, pred_len: int, metrics_path: Path) -> dict[str, Any]:
    payload = load_metrics(metrics_path)
    test_metrics = payload.get("test_metrics") or payload["reported_metrics"]
    daytime = test_metrics["daytime_only"]
    all_timestamps = test_metrics["all_timestamps"]
    return {
        "case": case,
        "pred_len": pred_len,
        "daytime_mae": float(daytime["mae"]),
        "daytime_rmse": float(daytime["rmse"]),
        "daytime_mbe": float(daytime["mbe"]),
        "daytime_r2": float(daytime["r2"]),
        "daytime_pearson_r": float(daytime["pearson_r"]),
        "all_mae": float(all_timestamps["mae"]),
        "all_rmse": float(all_timestamps["rmse"]),
        "metrics_path": str(metrics_path),
    }


def build_rows(results_root: Path, pred_lens: list[int]) -> list[dict[str, Any]]:
    cases = {
        "none_matched": results_root / "itransformer_mask_calibration" / "none_matched",
        "hard_existing": results_root / "itransformer_global_pcmci_11vars",
        "soft_beta_1": results_root / "itransformer_mask_calibration" / "soft_bias_beta_1",
        "soft_beta_2": results_root / "itransformer_mask_calibration" / "soft_bias_beta_2",
    }
    rows: list[dict[str, Any]] = []
    for pred_len in pred_lens:
        for case, case_dir in cases.items():
            rows.append(metric_row(case, pred_len, case_dir / f"pred_len_{pred_len}" / "metrics.json"))
    return rows


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "case",
        "pred_len",
        "daytime_mae",
        "daytime_rmse",
        "daytime_mbe",
        "daytime_r2",
        "daytime_pearson_r",
        "all_mae",
        "all_rmse",
        "metrics_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], output_path: Path) -> None:
    by_pred_case = {(int(row["pred_len"]), str(row["case"])): row for row in rows}
    pred_lens = sorted({int(row["pred_len"]) for row in rows})

    lines = [
        "# iTransformer Mask Calibration Summary",
        "",
        "Negative deltas mean the candidate has lower daytime RMSE than the baseline.",
        "",
        "| pred_len | none RMSE | hard RMSE | soft beta=1 RMSE | soft beta=2 RMSE | soft1 vs hard | soft1 vs none | soft2 vs hard | soft2 vs none |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    soft1_better_than_hard_long = True
    soft1_within_one_pct_none_long = True
    soft1_better_than_none_count = 0
    soft2_better_than_none_count = 0

    for pred_len in pred_lens:
        none = by_pred_case[(pred_len, "none_matched")]["daytime_rmse"]
        hard = by_pred_case[(pred_len, "hard_existing")]["daytime_rmse"]
        soft1 = by_pred_case[(pred_len, "soft_beta_1")]["daytime_rmse"]
        soft2 = by_pred_case[(pred_len, "soft_beta_2")]["daytime_rmse"]
        soft1_vs_hard = pct_delta(soft1, hard)
        soft1_vs_none = pct_delta(soft1, none)
        soft2_vs_hard = pct_delta(soft2, hard)
        soft2_vs_none = pct_delta(soft2, none)

        if pred_len in {24, 48}:
            soft1_better_than_hard_long = soft1_better_than_hard_long and soft1 < hard
            soft1_within_one_pct_none_long = soft1_within_one_pct_none_long and soft1_vs_none <= 1.0
        soft1_better_than_none_count += int(soft1 < none)
        soft2_better_than_none_count += int(soft2 < none)

        lines.append(
            "| {pred_len} | {none:.6f} | {hard:.6f} | {soft1:.6f} | {soft2:.6f} | "
            "{soft1_vs_hard:+.2f}% | {soft1_vs_none:+.2f}% | {soft2_vs_hard:+.2f}% | {soft2_vs_none:+.2f}% |".format(
                pred_len=pred_len,
                none=none,
                hard=hard,
                soft1=soft1,
                soft2=soft2,
                soft1_vs_hard=soft1_vs_hard,
                soft1_vs_none=soft1_vs_none,
                soft2_vs_hard=soft2_vs_hard,
                soft2_vs_none=soft2_vs_none,
            )
        )

    lines.extend(["", "## Decision Hints", ""])
    if soft1_better_than_hard_long and soft1_within_one_pct_none_long:
        lines.append("- `soft_bias beta=1` satisfies the primary criterion for using soft masks in later situation-aware experiments.")
    else:
        lines.append("- `soft_bias beta=1` does not satisfy the primary criterion; inspect beta=2 and no-mask before using masks as the main predictive claim.")

    if max(soft1_better_than_none_count, soft2_better_than_none_count) >= 2:
        lines.append("- At least one soft-mask setting beats matched no-mask on two or more horizons, so predictive benefit remains plausible.")
    else:
        lines.append("- Soft masks do not beat matched no-mask on two or more horizons; treat causal masks mainly as interpretability constraints unless later evidence changes.")

    lines.append("- If beta=1 and beta=2 are close, prefer beta=1 because it is the weaker constraint.")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_root = resolve_path(args.results_root)
    output_dir = resolve_path(args.output_dir) if args.output_dir else results_root / "itransformer_mask_calibration" / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows(results_root, args.pred_lens)
    csv_path = output_dir / "mask_calibration_summary.csv"
    markdown_path = output_dir / "mask_calibration_summary.md"
    write_csv(rows, csv_path)
    write_markdown(rows, markdown_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
