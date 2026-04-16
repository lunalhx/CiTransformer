from __future__ import annotations

import torch
from torch import nn


class LSTMBaseline(nn.Module):
    """
    Standard sequence-to-vector LSTM baseline.

    Input shape:
        [batch_size, seq_len, input_size]

    Output shape:
        [batch_size, pred_len]
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        pred_len: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if input_size <= 0:
            raise ValueError("input_size must be positive.")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if pred_len <= 0:
            raise ValueError("pred_len must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_size, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_output, _ = self.lstm(x)
        last_hidden = lstm_output[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        return self.projection(last_hidden)
