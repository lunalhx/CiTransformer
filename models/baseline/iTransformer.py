from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from layers.Embed import DataEmbedding_inverted
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer


@dataclass
class ITransformerConfig:
    seq_len: int
    pred_len: int
    d_model: int = 128
    n_heads: int = 4
    e_layers: int = 2
    d_ff: int = 256
    factor: int = 5
    dropout: float = 0.1
    activation: str = "gelu"
    output_attention: bool = False
    use_norm: bool = True
    embed: str = "fixed"
    freq: str = "h"
    class_strategy: str = "projection"


class Model(nn.Module):
    """
    Vanilla iTransformer backbone.

    Input:
        x_enc: [batch_size, seq_len, feature_dim]

    Output:
        [batch_size, pred_len, feature_dim]
    """

    def __init__(self, configs: ITransformerConfig) -> None:
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        self.class_strategy = configs.class_strategy
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None,
        x_dec: torch.Tensor | None,
        x_mark_dec: torch.Tensor | None,
    ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        del x_dec, x_mark_dec

        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev
        else:
            means = None
            stdev = None

        _, _, feature_dim = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :feature_dim]

        if self.use_norm and means is not None and stdev is not None:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out, attns

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None,
        x_dec: torch.Tensor | None,
        x_mark_dec: torch.Tensor | None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor | None]]:
        del mask
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

        if self.output_attention:
            return dec_out[:, -self.pred_len :, :], attns
        return dec_out[:, -self.pred_len :, :]


class ITransformerBaseline(nn.Module):
    """
    Wrap vanilla iTransformer so it matches the existing baseline API.

    External interface:
        input  -> x: [batch_size, seq_len, feature_dim]
        output -> y_hat: [batch_size, pred_len]

    The wrapper selects the target variable from the multivariate backbone
    output and converts it from feature-scaler space into target-scaler space,
    so the existing loss / inverse-transform / metrics pipeline can stay
    unchanged.
    """

    def __init__(
        self,
        target_feature_index: int,
        feature_mean: float,
        feature_scale: float,
        target_mean: float,
        target_scale: float,
        *,
        seq_len: int,
        pred_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 256,
        factor: int = 5,
        dropout: float = 0.1,
        activation: str = "gelu",
        output_attention: bool = False,
        use_norm: bool = True,
    ) -> None:
        super().__init__()

        if seq_len <= 0:
            raise ValueError("seq_len must be positive.")
        if pred_len <= 0:
            raise ValueError("pred_len must be positive.")
        if target_feature_index < 0:
            raise ValueError("target_feature_index must be non-negative.")
        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if n_heads <= 0:
            raise ValueError("n_heads must be positive.")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}.")
        if e_layers <= 0:
            raise ValueError("e_layers must be positive.")
        if d_ff <= 0:
            raise ValueError("d_ff must be positive.")
        if feature_scale == 0:
            raise ValueError("feature_scale must be non-zero.")
        if target_scale == 0:
            raise ValueError("target_scale must be non-zero.")

        self.target_feature_index = int(target_feature_index)
        self.pred_len = int(pred_len)
        self.backbone = Model(
            ITransformerConfig(
                seq_len=seq_len,
                pred_len=pred_len,
                d_model=d_model,
                n_heads=n_heads,
                e_layers=e_layers,
                d_ff=d_ff,
                factor=factor,
                dropout=dropout,
                activation=activation,
                output_attention=output_attention,
                use_norm=use_norm,
            )
        )

        self.register_buffer("feature_mean", torch.tensor(float(feature_mean), dtype=torch.float32))
        self.register_buffer("feature_scale", torch.tensor(float(feature_scale), dtype=torch.float32))
        self.register_buffer("target_mean", torch.tensor(float(target_mean), dtype=torch.float32))
        self.register_buffer("target_scale", torch.tensor(float(target_scale), dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape [batch, seq_len, feature_dim], got {tuple(x.shape)}.")
        if self.target_feature_index >= x.size(-1):
            raise ValueError(
                f"target_feature_index={self.target_feature_index} is out of range for feature_dim={x.size(-1)}."
            )

        backbone_output = self.backbone(
            x_enc=x,
            x_mark_enc=x_mark_enc,
            x_dec=x_dec,
            x_mark_dec=x_mark_dec,
            mask=mask,
        )
        if isinstance(backbone_output, tuple):
            multivariate_prediction = backbone_output[0]
        else:
            multivariate_prediction = backbone_output

        target_prediction_feature_scaled = multivariate_prediction[:, :, self.target_feature_index]
        target_prediction_raw = (
            target_prediction_feature_scaled * self.feature_scale + self.feature_mean
        )
        target_prediction_scaled = (target_prediction_raw - self.target_mean) / self.target_scale
        return target_prediction_scaled

    def extra_repr(self) -> str:
        return f"target_feature_index={self.target_feature_index}, pred_len={self.pred_len}"
