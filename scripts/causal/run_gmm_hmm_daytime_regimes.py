from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from hmmlearn.hmm import GMMHMM
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.project_config import load_project_config

PROJECT_CONFIG = load_project_config()
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_CONFIG.get_path("paths.matplotlib_cache")))

import matplotlib.pyplot as plt
import seaborn as sns

TIME_COLUMN_CANDIDATES = ("timestamp", "datetime", "date", "time", "ds")
DEFAULT_DATA_DIR = "data/processed_long_no_wind_2015_2022"
DEFAULT_OUTPUT_DIR = "results/d1_long_no_wind_2015_2022/regimes/gmm_hmm_daytime"
DEFAULT_DOC_PATH = "docs/d1_long_no_wind_2015_2022/gmm_hmm_daytime_regime_experiment.md"
DEFAULT_FEATURE_COLUMNS = [
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
]
SPLITS = ("train",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run train-only daytime GMM-HMM regime discovery for PV operation states. "
            "Only train_with_regime.csv is written; validation/calibration/test regimes are inferred online "
            "inside downstream forecasting code."
        )
    )
    parser.add_argument(
        "--data_dir",
        default=str(PROJECT_CONFIG.get_path("paths.data_dir", DEFAULT_DATA_DIR)),
        help="Directory containing split CSV files, either directly or under a splits/ subdirectory.",
    )
    parser.add_argument("--time_col", default=None, help="Optional explicit timestamp column name.")
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / DEFAULT_OUTPUT_DIR),
        help="Directory for regime labels, selection metrics, and plots.",
    )
    parser.add_argument(
        "--doc_path",
        default=str(PROJECT_ROOT / DEFAULT_DOC_PATH),
        help="Markdown experiment note to write after the run.",
    )
    parser.add_argument("--feature_cols", nargs="+", default=DEFAULT_FEATURE_COLUMNS)
    parser.add_argument("--day_col", default="day_night_label")
    parser.add_argument("--k_values", nargs="+", type=int, default=[2, 3, 4, 5, 6, 7, 8, 9, 10])
    parser.add_argument(
        "--force_k",
        type=int,
        default=None,
        help="Force the final selected K after scanning. Useful for interpretability comparisons.",
    )
    parser.add_argument("--n_mix", type=int, default=1, help="Gaussian mixtures per hidden state.")
    parser.add_argument("--min_regime_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--min_covar", type=float, default=1e-3)
    parser.add_argument(
        "--covariance_type",
        default="diag",
        choices=["spherical", "diag", "full", "tied"],
        help="GMMHMM covariance type. 'diag' is the default for stable PV regime discovery.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def resolve_split_dir(data_dir: str | Path) -> Path:
    data_dir = Path(data_dir)
    if (data_dir / "train.csv").exists():
        return data_dir
    if (data_dir / "splits" / "train.csv").exists():
        return data_dir / "splits"
    raise FileNotFoundError(f"Cannot find split CSV files under {data_dir}.")


def is_parseable_datetime(series: pd.Series, min_success_ratio: float = 0.95) -> bool:
    if series.empty:
        return False
    parsed = pd.to_datetime(series, errors="coerce")
    return float(parsed.notna().mean()) >= min_success_ratio


def detect_time_column(df: pd.DataFrame, preferred: str | None = None) -> str | None:
    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    candidates.extend([candidate for candidate in TIME_COLUMN_CANDIDATES if candidate not in candidates])
    for column in candidates:
        if column in df.columns and is_parseable_datetime(df[column]):
            return column
    for column in df.columns:
        if column not in candidates and is_parseable_datetime(df[column]):
            return column
    return None


def ensure_datetime_index(df: pd.DataFrame, time_col: str | None = None) -> pd.DataFrame:
    working_df = df.copy()
    if isinstance(working_df.index, pd.DatetimeIndex):
        return working_df
    detected_time_col = detect_time_column(working_df, preferred=time_col)
    if detected_time_col is None:
        raise ValueError("Failed to detect a timestamp column.")
    timestamp = pd.to_datetime(working_df[detected_time_col], errors="raise")
    working_df = working_df.drop(columns=[detected_time_col])
    working_df.index = pd.DatetimeIndex(timestamp, name=detected_time_col)
    return working_df


def load_split_dataframe(path: str | Path, time_col: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = ensure_datetime_index(df, time_col=time_col).sort_index()
    if df.index.has_duplicates:
        duplicate_count = int(df.index.duplicated().sum())
        raise ValueError(f"{path} contains {duplicate_count} duplicated timestamps.")
    return df


def infer_expected_timedelta(index: pd.DatetimeIndex) -> pd.Timedelta:
    diffs = index.to_series().diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        raise ValueError("Cannot infer sampling frequency from timestamps.")
    mode = diffs.mode()
    return mode.iloc[0] if not mode.empty else diffs.iloc[0]


def build_segment_lengths(index: pd.DatetimeIndex, expected_delta: pd.Timedelta) -> list[int]:
    if len(index) == 0:
        return []
    diffs = index.to_series().diff()
    is_new_segment = diffs.isna() | (diffs != expected_delta)
    starts = np.flatnonzero(is_new_segment.to_numpy())
    ends = np.concatenate([starts[1:], np.array([len(index)], dtype=np.int64)])
    return (ends - starts).astype(int).tolist()


def validate_columns(dfs: dict[str, pd.DataFrame], feature_cols: list[str], day_col: str) -> None:
    required = feature_cols + [day_col]
    for split, df in dfs.items():
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise KeyError(f"{split} is missing required columns: {missing}")


def state_proportions(labels: np.ndarray, k: int) -> np.ndarray:
    counts = np.bincount(labels, minlength=k).astype(float)
    return counts / max(counts.sum(), 1.0)


def count_gmmhmm_parameters(
    n_components: int,
    n_mix: int,
    n_features: int,
    covariance_type: str,
) -> int:
    start_params = n_components - 1
    transition_params = n_components * (n_components - 1)
    mixture_weight_params = n_components * (n_mix - 1)
    mean_params = n_components * n_mix * n_features

    if covariance_type == "spherical":
        covariance_params = n_components * n_mix
    elif covariance_type == "diag":
        covariance_params = n_components * n_mix * n_features
    elif covariance_type == "full":
        covariance_params = n_components * n_mix * n_features * (n_features + 1) // 2
    elif covariance_type == "tied":
        covariance_params = n_components * n_features * (n_features + 1) // 2
    else:
        raise ValueError(f"Unsupported covariance_type: {covariance_type}")

    return start_params + transition_params + mixture_weight_params + mean_params + covariance_params


def build_gmmhmm(args: argparse.Namespace, n_components: int) -> GMMHMM:
    return GMMHMM(
        n_components=n_components,
        n_mix=args.n_mix,
        covariance_type=args.covariance_type,
        min_covar=args.min_covar,
        n_iter=args.max_iter,
        tol=args.tol,
        random_state=args.seed + n_components,
        verbose=False,
        implementation="log",
    )


def choose_best_k(selection_df: pd.DataFrame) -> pd.Series:
    stable = selection_df[selection_df["stable"]].copy()
    candidates = stable if not stable.empty else selection_df.copy()
    candidates = candidates.sort_values(["bic", "n_components"], ascending=[True, True])
    return candidates.iloc[0]


def select_k(selection_df: pd.DataFrame, force_k: int | None) -> pd.Series:
    if force_k is None:
        return choose_best_k(selection_df)
    forced = selection_df[selection_df["n_components"] == force_k]
    if forced.empty:
        available = selection_df["n_components"].tolist()
        raise ValueError(f"--force_k {force_k} was not scanned. Available K values: {available}")
    return forced.iloc[0]


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def assign_regimes(
    dfs: dict[str, pd.DataFrame],
    scaler: StandardScaler,
    model: GMMHMM,
    feature_cols: list[str],
    day_col: str,
    expected_delta: pd.Timedelta,
) -> dict[str, pd.DataFrame]:
    labeled: dict[str, pd.DataFrame] = {}
    for split, df in dfs.items():
        output_df = df.copy()
        output_df["regime"] = 0
        day_mask = output_df[day_col] == 1
        day_df = output_df.loc[day_mask]
        if not day_df.empty:
            day_x = scaler.transform(day_df[feature_cols].to_numpy(dtype=float))
            day_lengths = build_segment_lengths(day_df.index, expected_delta)
            output_df.loc[day_mask, "regime"] = model.predict(day_x, lengths=day_lengths) + 1
        labeled[split] = output_df
    return labeled


def regime_summary(labeled: dict[str, pd.DataFrame], feature_cols: list[str], day_col: str) -> pd.DataFrame:
    combined = pd.concat([df.assign(split=split) for split, df in labeled.items()], axis=0)
    total_rows = len(combined)
    day_rows = int((combined[day_col] == 1).sum())
    records: list[dict[str, Any]] = []
    for regime, group in combined.groupby("regime", sort=True):
        day_group = group[group[day_col] == 1]
        record: dict[str, Any] = {
            "regime": int(regime),
            "sample_count": int(len(group)),
            "sample_ratio_all": float(len(group) / total_rows),
            "daytime_sample_count": int(len(day_group)),
            "sample_ratio_daytime": float(len(day_group) / day_rows) if day_rows else 0.0,
        }
        for column in feature_cols:
            record[f"mean_{column}"] = float(group[column].mean())
            record[f"std_{column}"] = float(group[column].std(ddof=0))
        records.append(record)
    return pd.DataFrame(records)


def infer_regime_names(summary: pd.DataFrame) -> dict[int, str]:
    names: dict[int, str] = {0: "night"}
    day_summary = summary[summary["regime"] > 0].copy()
    if day_summary.empty:
        return names

    power = day_summary["mean_Active_Pow"]
    global_rad = day_summary["mean_Radiation_Global_Tilted"]
    diffuse = day_summary["mean_Radiation_Diffuse_Tilted"]
    elevation = day_summary["mean_solar_elevation"]
    diffuse_ratio = diffuse / global_rad.replace(0, np.nan)

    high_power_id = int(power.idxmax())
    low_rad_id = int(global_rad.idxmin())
    transition_id = int(elevation.idxmin())
    cloudy_id = int(diffuse_ratio.fillna(-np.inf).idxmax())

    label_by_index = {
        high_power_id: "sunny_stable_high_output",
        cloudy_id: "cloudy_diffuse_disturbance",
        low_rad_id: "low_irradiance_weak_output",
        transition_id: "morning_evening_transition",
    }
    fallback = [
        "moderate_output_mixed_weather",
        "humid_variable_output",
        "clear_midday_output",
        "low_sun_angle_variable",
        "seasonal_high_output",
        "seasonal_low_output",
        "diffuse_low_output",
        "diffuse_high_output",
        "warm_midday_output",
        "cool_midday_output",
        "early_day_transition",
        "late_day_transition",
    ]
    used = set()
    for row_index, row in day_summary.iterrows():
        regime = int(row["regime"])
        label = label_by_index.get(row_index)
        if label is None or label in used:
            label = next(
                (item for item in fallback if item not in used),
                f"regime_{regime}_unnamed",
            )
        names[regime] = label
        used.add(label)
    return names


def write_labeled_csvs(labeled: dict[str, pd.DataFrame], output_dir: Path) -> None:
    for split, df in labeled.items():
        df.reset_index().to_csv(output_dir / f"{split}_with_regime.csv", index=False)


def save_model_artifact(
    output_dir: Path,
    model: GMMHMM,
    scaler: StandardScaler,
    args: argparse.Namespace,
    selected_row: pd.Series,
    regime_names: dict[int, str],
) -> Path:
    artifact_path = output_dir / "gmm_hmm_regime_model.pkl"
    artifact = {
        "implementation": "hmmlearn.hmm.GMMHMM",
        "model": model,
        "scaler": scaler,
        "feature_columns": list(args.feature_cols),
        "day_column": args.day_col,
        "night_regime": 0,
        "daytime_regime_offset": 1,
        "selected_model": to_jsonable(selected_row.to_dict()),
        "regime_names": {str(key): value for key, value in regime_names.items()},
        "seed": int(args.seed),
    }
    with artifact_path.open("wb") as fp:
        pickle.dump(artifact, fp)
    return artifact_path


def plot_daily_profiles(labeled: dict[str, pd.DataFrame], output_dir: Path) -> None:
    combined = pd.concat(labeled.values()).copy()
    combined = combined[combined["regime"] > 0]
    combined["time_of_day"] = combined.index.strftime("%H:%M")
    profile = (
        combined.groupby(["regime", "time_of_day"], observed=True)["Active_Pow"]
        .mean()
        .reset_index()
    )
    pivot = profile.pivot(index="time_of_day", columns="regime", values="Active_Pow")
    tick_positions = np.arange(0, len(pivot), 24)

    plt.figure(figsize=(12, 6))
    for regime in pivot.columns:
        plt.plot(np.arange(len(pivot)), pivot[regime], label=f"Regime {regime}", linewidth=1.8)
    plt.xticks(tick_positions, pivot.index[tick_positions], rotation=45)
    plt.xlabel("Time of day")
    plt.ylabel("Mean Active_Pow")
    plt.title("Typical Daytime Power Profile by Regime")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "regime_daily_profiles.png", dpi=180)
    plt.close()


def plot_feature_distributions(labeled: dict[str, pd.DataFrame], output_dir: Path, seed: int) -> None:
    combined = pd.concat(labeled.values()).copy()
    combined = combined[combined["regime"] > 0]
    sample = combined.sample(n=min(len(combined), 80_000), random_state=seed)
    features = [
        "Active_Pow",
        "Radiation_Global_Tilted",
        "Radiation_Diffuse_Tilted",
        "Weather_T",
        "Weather_R",
        "solar_elevation",
    ]
    long_df = sample.reset_index().melt(
        id_vars=["regime"],
        value_vars=features,
        var_name="feature",
        value_name="value",
    )

    grid = sns.catplot(
        data=long_df,
        x="regime",
        y="value",
        col="feature",
        kind="box",
        col_wrap=3,
        sharey=False,
        height=3.2,
        aspect=1.2,
        showfliers=False,
    )
    grid.set_axis_labels("Regime", "Value")
    grid.fig.suptitle("Feature Distributions by Daytime Regime", y=1.02)
    grid.tight_layout()
    grid.savefig(output_dir / "regime_feature_distributions.png", dpi=180)
    plt.close(grid.fig)


def plot_time_distribution(labeled: dict[str, pd.DataFrame], output_dir: Path) -> None:
    combined = pd.concat(labeled.values()).copy()
    combined = combined[combined["regime"] > 0]
    combined["hour"] = combined.index.hour + combined.index.minute / 60.0
    combined["month"] = combined.index.month

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(data=combined, x="hour", hue="regime", multiple="stack", bins=48, ax=axes[0])
    axes[0].set_title("Intraday Regime Occurrence")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("Sample count")

    month_counts = combined.groupby(["month", "regime"]).size().reset_index(name="count")
    sns.barplot(data=month_counts, x="month", y="count", hue="regime", ax=axes[1])
    axes[1].set_title("Monthly Regime Occurrence")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Sample count")
    fig.tight_layout()
    fig.savefig(output_dir / "regime_time_distribution.png", dpi=180)
    plt.close(fig)


def plot_power_radiation_scatter(labeled: dict[str, pd.DataFrame], output_dir: Path, seed: int) -> None:
    combined = pd.concat(labeled.values()).copy()
    combined = combined[combined["regime"] > 0]
    sample = combined.sample(n=min(len(combined), 80_000), random_state=seed)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=sample,
        x="Radiation_Global_Tilted",
        y="Active_Pow",
        hue="regime",
        s=8,
        linewidth=0,
        alpha=0.35,
        palette="tab10",
    )
    plt.xlabel("Radiation_Global_Tilted")
    plt.ylabel("Active_Pow")
    plt.title("Power-Radiation Scatter by Regime")
    plt.tight_layout()
    plt.savefig(output_dir / "regime_power_radiation_scatter.png", dpi=180)
    plt.close()


def write_markdown_report(
    doc_path: Path,
    output_dir: Path,
    selected_row: pd.Series,
    summary: pd.DataFrame,
    names: dict[int, str],
    selection_df: pd.DataFrame,
    min_stability_ratio: float,
) -> None:
    best_k = int(selected_row["n_components"])
    selection_lines = [
        "| K | Log likelihood | AIC | BIC | Min regime ratio | Stable |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for _, row in selection_df.iterrows():
        selection_lines.append(
            f"| {int(row['n_components'])} | {row['log_likelihood']:.2f} | "
            f"{row['aic']:.2f} | {row['bic']:.2f} | "
            f"{row['min_regime_ratio']:.2%} | {bool(row['stable'])} |"
        )
    lines = [
        "# GMM-HMM Daytime Regime Experiment",
        "",
        "## Summary",
        "",
        f"- Output directory: `{output_dir}`",
        f"- Selected daytime regime count: K = {best_k}",
        f"- Selection rule: choose the lowest BIC among stable K values; stable means every daytime regime ratio >= {min_stability_ratio:.1%}.",
        f"- Selected model BIC: {selected_row['bic']:.2f}; log likelihood: {selected_row['log_likelihood']:.2f}.",
        "",
        "## K Selection",
        "",
        *selection_lines,
        "",
        "## Regime Interpretation",
        "",
        "| Regime | Name | Daytime ratio | Mean power | Mean global radiation | Mean diffuse radiation | Mean temperature | Mean humidity | Mean solar elevation |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.sort_values("regime").iterrows():
        regime = int(row["regime"])
        lines.append(
            f"| {regime} | {names.get(regime, '')} | "
            f"{row['sample_ratio_daytime']:.2%} | "
            f"{row['mean_Active_Pow']:.3f} | "
            f"{row['mean_Radiation_Global_Tilted']:.3f} | "
            f"{row['mean_Radiation_Diffuse_Tilted']:.3f} | "
            f"{row['mean_Weather_T']:.3f} | "
            f"{row['mean_Weather_R']:.3f} | "
            f"{row['mean_solar_elevation']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "The learned daytime regimes separate samples by output level, irradiance structure, humidity/temperature context, and solar-elevation timing. "
            "Night samples are kept out of HMM training and are labeled as regime 0, so the discovered states focus on daytime PV operating conditions rather than the physically forced night zero-output state.",
            "",
            "These regime labels can support the next stage of split-regime PCMCI by providing mutually exclusive daytime subsets with distinct power-radiation-weather distributions. "
            "Before causal discovery, inspect the saved plots and `regime_summary.csv` to confirm that the chosen K remains interpretable for the paper narrative.",
        ]
    )
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_output_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    output_dir = resolve_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_dir = resolve_split_dir(args.data_dir)
    log(f"Loading train split from {split_dir}")
    dfs = {split: load_split_dataframe(split_dir / f"{split}.csv", time_col=args.time_col) for split in SPLITS}
    validate_columns(dfs, args.feature_cols, args.day_col)

    expected_delta = infer_expected_timedelta(dfs["train"].index)
    train_day = dfs["train"][dfs["train"][args.day_col] == 1]
    train_lengths = build_segment_lengths(train_day.index, expected_delta)
    log(
        f"Train rows={len(dfs['train'])}, train daytime rows={len(train_day)}, "
        f"daytime segments={len(train_lengths)}, expected_delta={expected_delta}"
    )

    scaler = StandardScaler()
    train_x_raw = train_day[args.feature_cols].to_numpy(dtype=float)
    scaler.fit(train_x_raw)
    train_x = scaler.transform(train_x_raw)

    selection_records: list[dict[str, Any]] = []
    fitted_models: dict[int, GMMHMM] = {}
    for k in args.k_values:
        log(f"Fitting daytime GMM-HMM with K={k}, n_mix={args.n_mix}")
        model = build_gmmhmm(args, n_components=k)
        model.fit(train_x, lengths=train_lengths)

        labels = model.predict(train_x, lengths=train_lengths)
        proportions = state_proportions(labels, k)
        n_params = count_gmmhmm_parameters(k, args.n_mix, len(args.feature_cols), args.covariance_type)
        loglik = float(model.score(train_x, lengths=train_lengths))
        aic = 2 * n_params - 2 * loglik
        bic = math.log(len(train_x)) * n_params - 2 * loglik
        stable = bool(np.min(proportions) >= args.min_regime_ratio)
        fitted_models[k] = model
        selection_records.append(
            {
                "n_components": k,
                "n_mix": args.n_mix,
                "log_likelihood": loglik,
                "aic": aic,
                "bic": bic,
                "n_parameters": n_params,
                "min_regime_ratio": float(np.min(proportions)),
                "stable": stable,
                "converged": bool(model.monitor_.converged),
                "n_iter": int(model.monitor_.iter),
                "regime_ratios": json.dumps(
                    {f"state_{i + 1}": float(value) for i, value in enumerate(proportions)},
                    ensure_ascii=False,
                ),
                "transition_matrix": json.dumps(model.transmat_.tolist(), ensure_ascii=False),
            }
        )
        log(
            f"K={k} loglik={loglik:.2f} bic={bic:.2f} "
            f"min_ratio={np.min(proportions):.3f} stable={stable}"
        )

    selection_df = pd.DataFrame(selection_records).sort_values("n_components")
    selection_df.to_csv(output_dir / "regime_selection.csv", index=False)
    selected_row = select_k(selection_df, force_k=args.force_k)
    best_k = int(selected_row["n_components"])
    best_model = fitted_models[best_k]
    log(f"Selected K={best_k}")

    labeled = assign_regimes({"train": dfs["train"]}, scaler, best_model, args.feature_cols, args.day_col, expected_delta)
    summary = regime_summary(labeled, args.feature_cols, args.day_col)
    regime_names = infer_regime_names(summary)
    summary["regime_name"] = summary["regime"].map(regime_names)
    summary.to_csv(output_dir / "regime_summary.csv", index=False)
    model_artifact_path = save_model_artifact(
        output_dir=output_dir,
        model=best_model,
        scaler=scaler,
        args=args,
        selected_row=selected_row,
        regime_names=regime_names,
    )

    transition_df = pd.DataFrame(
        best_model.transmat_,
        index=[f"regime_{i + 1}" for i in range(best_k)],
        columns=[f"regime_{i + 1}" for i in range(best_k)],
    )
    transition_df.to_csv(output_dir / "transition_matrix.csv")

    write_labeled_csvs(labeled, output_dir)
    plot_daily_profiles(labeled, output_dir)
    plot_feature_distributions(labeled, output_dir, seed=args.seed)
    plot_time_distribution(labeled, output_dir)
    plot_power_radiation_scatter(labeled, output_dir, seed=args.seed)

    config = {
        "implementation": "hmmlearn.hmm.GMMHMM",
        "selected_k": best_k,
        "forced_k": args.force_k,
        "n_mix": args.n_mix,
        "covariance_type": args.covariance_type,
        "selection_rule": "lowest BIC among K with every daytime state ratio >= min_regime_ratio; falls back to global lowest BIC if none are stable",
        "sample_policy": (
            "train-only regime discovery; validation/calibration/test full-split offline labels are not written "
            "because downstream forecasting uses online HMM forward filtering"
        ),
        "min_regime_ratio": args.min_regime_ratio,
        "feature_columns": args.feature_cols,
        "day_column": args.day_col,
        "night_regime": 0,
        "daytime_regime_range": [1, best_k],
        "seed": args.seed,
        "expected_delta": str(expected_delta),
        "train_rows": int(len(dfs["train"])),
        "train_daytime_rows": int(len(train_day)),
        "train_daytime_segments": int(len(train_lengths)),
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "var": scaler.var_.tolist(),
        },
        "selected_model": to_jsonable(selected_row.to_dict()),
        "regime_names": {str(key): value for key, value in regime_names.items()},
        "model_artifact_path": str(model_artifact_path),
        "labeled_split_files": {"train": str(output_dir / "train_with_regime.csv")},
    }
    with (output_dir / "best_regime_config.json").open("w", encoding="utf-8") as fp:
        json.dump(config, fp, ensure_ascii=False, indent=2)

    write_markdown_report(
        doc_path=resolve_output_path(args.doc_path),
        output_dir=output_dir,
        selected_row=selected_row,
        summary=summary,
        names=regime_names,
        selection_df=selection_df,
        min_stability_ratio=args.min_regime_ratio,
    )
    log(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
