from __future__ import annotations

import torch
from torch import nn


class PersistenceBaseline(nn.Module):
    """
    Standard persistence / naive baseline for multi-step forecasting.

    The predictor copies the last observed target value from the encoder window
    and repeats it for every future horizon:

        y_hat[t+1:t+pred_len] = last_observed_active_power

    Notes
    -----
    The input tensor `x` is expected to be on the *feature-scaler* space used by
    `ContinuousSegmentTimeSeriesDataset`. The output is converted onto the
    *target-scaler* space so it can share the exact same inverse-transform and
    evaluation pipeline as the LSTM baseline.
    """

    def __init__(
        self,
        target_feature_index: int,
        pred_len: int,
        feature_mean: float,
        feature_scale: float,
        target_mean: float,
        target_scale: float,
    ) -> None:
        super().__init__()

        if target_feature_index < 0:
            raise ValueError("target_feature_index must be non-negative.")
        if pred_len <= 0:
            raise ValueError("pred_len must be positive.")
        if feature_scale == 0:
            raise ValueError("feature_scale must be non-zero.")
        if target_scale == 0:
            raise ValueError("target_scale must be non-zero.")

        self.target_feature_index = int(target_feature_index)
        self.pred_len = int(pred_len)

        self.register_buffer("feature_mean", torch.tensor(float(feature_mean), dtype=torch.float32))
        self.register_buffer("feature_scale", torch.tensor(float(feature_scale), dtype=torch.float32))
        self.register_buffer("target_mean", torch.tensor(float(target_mean), dtype=torch.float32))
        self.register_buffer("target_scale", torch.tensor(float(target_scale), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape [batch, seq_len, feature_dim], got {tuple(x.shape)}.")
        if self.target_feature_index >= x.size(-1):
            raise ValueError(
                f"target_feature_index={self.target_feature_index} is out of range for feature_dim={x.size(-1)}."
            )

        last_power_feature_scaled = x[:, -1, self.target_feature_index]
        last_power_raw = last_power_feature_scaled * self.feature_scale + self.feature_mean
        last_power_target_scaled = (last_power_raw - self.target_mean) / self.target_scale
        return last_power_target_scaled.unsqueeze(-1).repeat(1, self.pred_len)

    def extra_repr(self) -> str:
        return f"target_feature_index={self.target_feature_index}, pred_len={self.pred_len}"
