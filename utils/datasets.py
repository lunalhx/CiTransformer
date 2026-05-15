from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


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
    "day_night_label",
]

DEFAULT_TARGET_COLUMN = "Active_Pow"
DEFAULT_DATA_DIR = "data/processed_long_no_wind_2015_2022"
TIME_COLUMN_CANDIDATES = ("timestamp", "datetime", "date", "time", "ds")


@dataclass
class SplitScalers:
    feature_scaler: StandardScaler
    target_scaler: StandardScaler


def resolve_split_dir(data_dir: str | Path) -> Path:
    data_dir = Path(data_dir)
    direct_train = data_dir / "train.csv"
    nested_train = data_dir / "splits" / "train.csv"

    if direct_train.exists():
        return data_dir
    if nested_train.exists():
        return data_dir / "splits"

    raise FileNotFoundError(
        f"Cannot find split CSV files under {data_dir}. "
        "Expected either data_dir/train.csv or data_dir/splits/train.csv."
    )


def load_split_dataframe(path: str | Path, time_col: str | None = None) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    df = ensure_datetime_index(df, time_col=time_col)
    df = df.sort_index()

    if df.index.has_duplicates:
        duplicate_count = int(df.index.duplicated().sum())
        raise ValueError(f"{path} contains {duplicate_count} duplicated timestamps.")

    return df


def ensure_datetime_index(df: pd.DataFrame, time_col: str | None = None) -> pd.DataFrame:
    working_df = df.copy()

    if isinstance(working_df.index, pd.DatetimeIndex):
        return working_df

    detected_time_col = detect_time_column(working_df, preferred=time_col)
    if detected_time_col is None:
        raise ValueError(
            "Failed to detect a timestamp column. "
            "Please provide --time_col explicitly or keep a parseable timestamp column in the CSV."
        )

    timestamp = pd.to_datetime(working_df[detected_time_col], errors="raise")
    working_df = working_df.drop(columns=[detected_time_col])
    working_df.index = pd.DatetimeIndex(timestamp, name=detected_time_col)
    return working_df


def detect_time_column(df: pd.DataFrame, preferred: str | None = None) -> str | None:
    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    candidates.extend([candidate for candidate in TIME_COLUMN_CANDIDATES if candidate not in candidates])

    for column in candidates:
        if column in df.columns and is_parseable_datetime(df[column]):
            return column

    for column in df.columns:
        if column in candidates:
            continue
        if is_parseable_datetime(df[column]):
            return column

    return None


def is_parseable_datetime(series: pd.Series, min_success_ratio: float = 0.95) -> bool:
    if series.empty:
        return False

    parsed = pd.to_datetime(series, errors="coerce")
    success_ratio = float(parsed.notna().mean())
    return success_ratio >= min_success_ratio


def infer_expected_timedelta(index: pd.DatetimeIndex) -> pd.Timedelta:
    diffs = index.to_series().diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]

    if diffs.empty:
        raise ValueError("Cannot infer sampling frequency because no positive timestamp difference was found.")

    mode = diffs.mode()
    return mode.iloc[0] if not mode.empty else diffs.iloc[0]


def build_segment_boundaries(
    index: pd.DatetimeIndex,
    expected_delta: pd.Timedelta,
) -> tuple[np.ndarray, np.ndarray]:
    if len(index) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    diffs = index.to_series().diff()
    # Any timestamp jump that is not exactly one sampling interval starts a new segment.
    is_new_segment = diffs.isna() | (diffs != expected_delta)
    start_positions = np.flatnonzero(is_new_segment.to_numpy())
    end_positions = np.concatenate([start_positions[1:], np.array([len(index)], dtype=np.int64)])
    return start_positions.astype(np.int64), end_positions.astype(np.int64)


def validate_feature_columns(df: pd.DataFrame, feature_cols: Iterable[str], target_col: str) -> None:
    missing = [column for column in list(feature_cols) + [target_col] if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def fit_split_scalers(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> SplitScalers:
    validate_feature_columns(train_df, feature_cols, target_col)

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # Standardization is fit strictly on train only to avoid leakage.
    feature_scaler.fit(train_df[feature_cols].to_numpy(dtype=np.float32))
    target_scaler.fit(train_df[[target_col]].to_numpy(dtype=np.float32))

    return SplitScalers(feature_scaler=feature_scaler, target_scaler=target_scaler)


class ContinuousSegmentTimeSeriesDataset(Dataset):
    """
    Build sliding windows only inside timestamp-continuous segments.

    If two adjacent rows are not exactly `expected_delta` apart, they belong to
    different segments and no sample is allowed to cross the boundary.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        seq_len: int,
        pred_len: int,
        feature_scaler: StandardScaler,
        target_scaler: StandardScaler,
        expected_delta: pd.Timedelta | None = None,
        regime_col: str = "regime",
        include_regime_labels: bool = True,
        regime_posterior: np.ndarray | None = None,
        regime_transition_matrix: np.ndarray | None = None,
        regime_start_probability: np.ndarray | None = None,
    ) -> None:
        if seq_len <= 0:
            raise ValueError("seq_len must be positive.")
        if pred_len <= 0:
            raise ValueError("pred_len must be positive.")

        self.df = ensure_datetime_index(df)
        self.df = self.df.sort_index()
        validate_feature_columns(self.df, feature_cols, target_col)

        self.feature_cols = feature_cols
        self.target_col = target_col
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.expected_delta = expected_delta or infer_expected_timedelta(self.df.index)
        self.timezone = self.df.index.tz
        self.regime_col = regime_col
        self.has_regime_col = include_regime_labels and regime_col in self.df.columns

        feature_array = self.df[feature_cols].to_numpy(dtype=np.float32)
        raw_target_array = self.df[target_col].to_numpy(dtype=np.float32)

        self.features = feature_scaler.transform(feature_array).astype(np.float32)
        self.targets = (
            target_scaler.transform(raw_target_array.reshape(-1, 1)).astype(np.float32).reshape(-1)
        )
        self.targets_raw = raw_target_array
        self.day_night_label = (
            self.df["day_night_label"].to_numpy(dtype=np.int64)
            if "day_night_label" in self.df.columns
            else np.ones(len(self.df), dtype=np.int64)
        )
        self.regime = (
            pd.to_numeric(self.df[regime_col], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
            if self.has_regime_col
            else np.zeros(len(self.df), dtype=np.int64)
        )
        self.timestamp_ns = self.df.index.asi8.astype(np.int64)
        self.regime_posterior = None
        self.regime_transition_matrix = None
        self.regime_start_probability = None
        self.has_regime_mask_weights = regime_posterior is not None
        if regime_posterior is not None:
            posterior = np.asarray(regime_posterior, dtype=np.float32)
            if posterior.ndim != 2 or posterior.shape[0] != len(self.df) or posterior.shape[1] < 2:
                raise ValueError(
                    "regime_posterior must have shape [rows, K+1] and include night regime 0; "
                    f"got {posterior.shape} for {len(self.df)} rows."
                )
            if regime_transition_matrix is None or regime_start_probability is None:
                raise ValueError(
                    "regime_transition_matrix and regime_start_probability are required with regime_posterior."
                )
            transition = np.asarray(regime_transition_matrix, dtype=np.float32)
            start_probability = np.asarray(regime_start_probability, dtype=np.float32)
            day_regime_count = posterior.shape[1] - 1
            if transition.shape != (day_regime_count, day_regime_count):
                raise ValueError(
                    "regime_transition_matrix must have shape [K, K] for posterior shape [rows, K+1]; "
                    f"got transition={transition.shape}, posterior={posterior.shape}."
                )
            if start_probability.shape != (day_regime_count,):
                raise ValueError(
                    "regime_start_probability must have shape [K] for posterior shape [rows, K+1]; "
                    f"got start={start_probability.shape}, posterior={posterior.shape}."
                )
            self.regime_posterior = posterior
            self.regime_transition_matrix = transition
            self.regime_start_probability = self._normalize_probability_vector(start_probability)

        # Sliding windows are counted per continuous segment only, never across gaps.
        self.segment_starts, segment_ends = build_segment_boundaries(self.df.index, self.expected_delta)
        self.segment_lengths = segment_ends - self.segment_starts

        min_segment_length = self.seq_len + self.pred_len
        self.valid_windows_per_segment = np.maximum(self.segment_lengths - min_segment_length + 1, 0)
        self.valid_segment_mask = self.valid_windows_per_segment > 0
        self.valid_segment_starts = self.segment_starts[self.valid_segment_mask]
        self.valid_segment_lengths = self.segment_lengths[self.valid_segment_mask]
        self.valid_windows_per_segment = self.valid_windows_per_segment[self.valid_segment_mask]
        self.cumulative_windows = np.cumsum(self.valid_windows_per_segment, dtype=np.int64)

    def __len__(self) -> int:
        if len(self.cumulative_windows) == 0:
            return 0
        return int(self.cumulative_windows[-1])

    @staticmethod
    def _normalize_probability_vector(values: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(values, dtype=np.float64).copy()
        probabilities[~np.isfinite(probabilities)] = 0.0
        probabilities = np.maximum(probabilities, 0.0)
        total = float(probabilities.sum())
        if total <= 0.0:
            probabilities = np.full_like(probabilities, 1.0 / max(len(probabilities), 1), dtype=np.float64)
        else:
            probabilities /= total
        return probabilities.astype(np.float32)

    def _compute_regime_mask_weights(self, input_end: int, target_start: int, target_end: int) -> np.ndarray:
        if self.regime_posterior is None or self.regime_transition_matrix is None or self.regime_start_probability is None:
            raise RuntimeError("Regime mask weights requested without regime posterior metadata.")

        posterior_at_input_end = self._normalize_probability_vector(self.regime_posterior[input_end])
        day_state = (
            self._normalize_probability_vector(posterior_at_input_end[1:])
            if float(posterior_at_input_end[1:].sum()) > 0.0
            else None
        )
        transition = self.regime_transition_matrix.astype(np.float64)
        start_probability = self.regime_start_probability.astype(np.float64)
        future_probabilities: list[np.ndarray] = []

        for target_index in range(target_start, target_end):
            full_probability = np.zeros_like(posterior_at_input_end, dtype=np.float64)
            if self.day_night_label[target_index] != 1:
                full_probability[0] = 1.0
                day_state = None
            else:
                if day_state is None:
                    day_state = start_probability.copy()
                else:
                    day_state = self._normalize_probability_vector(day_state @ transition).astype(np.float64)
                full_probability[1:] = day_state
            future_probabilities.append(full_probability)

        weights = np.mean(np.stack(future_probabilities, axis=0), axis=0)
        return self._normalize_probability_vector(weights)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self)}.")

        segment_idx = int(np.searchsorted(self.cumulative_windows, idx, side="right"))
        previous_cumulative = 0 if segment_idx == 0 else int(self.cumulative_windows[segment_idx - 1])
        offset = idx - previous_cumulative

        segment_start = int(self.valid_segment_starts[segment_idx])
        window_start = segment_start + offset
        encoder_end = window_start + self.seq_len
        decoder_end = encoder_end + self.pred_len

        features = self.features[window_start:encoder_end]
        targets = self.targets[encoder_end:decoder_end]
        targets_raw = self.targets_raw[encoder_end:decoder_end]
        target_time_ns = self.timestamp_ns[encoder_end:decoder_end]
        target_day_night = self.day_night_label[encoder_end:decoder_end]
        target_regime = self.regime[encoder_end:decoder_end]

        sample = {
            "x": torch.from_numpy(features),
            "y": torch.from_numpy(targets),
            "y_raw": torch.from_numpy(targets_raw),
            "input_start_ns": torch.tensor(self.timestamp_ns[window_start], dtype=torch.long),
            "input_end_ns": torch.tensor(self.timestamp_ns[encoder_end - 1], dtype=torch.long),
            "target_time_ns": torch.from_numpy(target_time_ns),
            "target_day_night": torch.from_numpy(target_day_night),
        }
        if self.has_regime_col:
            sample["input_end_regime"] = torch.tensor(self.regime[encoder_end - 1], dtype=torch.long)
            sample["target_regime"] = torch.from_numpy(target_regime)
        if self.has_regime_mask_weights:
            regime_mask_weights = self._compute_regime_mask_weights(
                input_end=encoder_end - 1,
                target_start=encoder_end,
                target_end=decoder_end,
            )
            sample["regime_mask_weights"] = torch.from_numpy(regime_mask_weights)
        return sample

    def summary(self) -> dict[str, int | float | str]:
        total_segments = int(len(self.segment_starts))
        usable_segments = int(self.valid_segment_mask.sum())
        dropped_segments = total_segments - usable_segments

        summary = {
            "rows": int(len(self.df)),
            "expected_delta_minutes": float(self.expected_delta / pd.Timedelta(minutes=1)),
            "total_segments": total_segments,
            "usable_segments": usable_segments,
            "dropped_segments": dropped_segments,
            "samples": int(len(self)),
        }

        if total_segments > 0:
            summary["max_segment_length"] = int(self.segment_lengths.max())
            summary["min_segment_length"] = int(self.segment_lengths.min())

        if usable_segments > 0:
            summary["max_usable_segment_length"] = int(self.valid_segment_lengths.max())
            summary["min_usable_segment_length"] = int(self.valid_segment_lengths.min())

        if self.has_regime_col:
            unique, counts = np.unique(self.regime, return_counts=True)
            summary["regime_col"] = self.regime_col
            summary["row_regime_counts"] = {
                str(int(regime)): int(count) for regime, count in zip(unique, counts, strict=True)
            }
        if self.has_regime_mask_weights and self.regime_posterior is not None:
            summary["regime_probability_dim"] = int(self.regime_posterior.shape[1])
            summary["regime_probability_source"] = "online_hmm_forward_filter"

        return summary
