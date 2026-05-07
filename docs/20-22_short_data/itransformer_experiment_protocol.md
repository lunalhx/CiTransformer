# iTransformer Baseline 实验文档

更新日期：2026-05-07

本文档是 vanilla iTransformer baseline 的实验说明，已根据最新 `results/itransformer/pred_len_1/12/24/48` 结果重写。后续关于默认 iTransformer baseline 的实现、协议、运行方式和结果汇总，以本文档为准。

本文档从属于统一实验协议：`docs/实验协议.md`。除非本文件特别说明，数据来源、split、`seq_len / pred_len`、指标口径和结果导出结构都遵循统一协议。

## 1. 实验目的

vanilla iTransformer baseline 用作光伏功率预测任务中的 Transformer 类深度学习对照组。它不包含 CiTransformer 的因果图、PCMCI、HMM、regime、causal mask 等额外设计，而是一个标准 iTransformer backbone 加工程适配层。

本次默认 baseline 已经按 `pred_len=1/12/24/48` 全部完成，并生成了对应的 metrics、predictions、plot 和 checkpoint。相比最初 30 epoch 版本，`pred_len=12/24/48` 已经额外加长训练，其中 `pred_len=48` 最新配置为 `EPOCHS=120 / PATIENCE=20`。

## 2. 代码位置

核心文件如下：

- `models/baseline/iTransformer.py`：iTransformer backbone 与目标适配 wrapper
- `scripts/run_iTransformer.py`：单次训练、验证、测试入口
- `scripts/run_itransformer_experiments.sh`：批量运行 `pred_len=1/12/24/48`
- `scripts/tune_itransformer.sh`：validation-only 调参脚本
- `scripts/run_itransformer_tuning_standard.sh`：标准调参入口
- `scripts/summarize_itransformer_tuning.py`：调参结果汇总
- `utils/datasets.py`：连续时间片段数据集、滑窗与标准化
- `results/itransformer/pred_len_*/`：本次默认 baseline 输出目录
- `checkpoints/itransformer/pred_len_*/best_model.pth`：每个预测步长的最优 checkpoint

## 3. 模型定义

当前实现是一个 vanilla iTransformer baseline：

1. 输入 `x` 的形状是 `[batch, seq_len, feature_dim]`。
2. iTransformer backbone 输出未来 `pred_len` 步的多变量预测。
3. 从多变量输出中取出目标列 `Active_Pow`。
4. 将目标输出从 feature-scaler 空间映射到 target-scaler 空间。
5. 复用统一的 loss、inverse-transform 和指标计算流程。

模型输出形状：

```text
[batch, pred_len]
```

这里 `Active_Pow` 既是输入特征之一，也是预测目标，和 Persistence、LSTM baseline 的输入协议保持一致。

## 4. 数据与 Split 用法

默认数据目录：

```text
data/processed_selected_2020_2022
```

默认使用四个 split：

- `train.csv`
- `validation.csv`
- `calibration.csv`
- `test.csv`

各 split 的实际用途如下：

| split | 用途 |
| --- | --- |
| train | 拟合 `feature_scaler` 和 `target_scaler`；训练模型参数；DataLoader 使用 `shuffle=True` |
| validation | 每个 epoch 后计算验证损失；用于 early stopping 和 best checkpoint 选择 |
| calibration | 为协议一致性读取并统计；当前 vanilla iTransformer baseline 不使用它 |
| test | 训练结束后加载最佳 checkpoint，只评估一次；导出 test 指标、预测 CSV 和图像 |

论文中可以明确写：

```text
The calibration split is reserved for protocol consistency but is unused in the current vanilla iTransformer baseline.
```

## 5. 连续时间片段约束

当前实验不会直接对整张表按行号滑窗。样本构造遵循以下规则：

1. 按时间戳排序。
2. 从训练集时间戳推断预期采样间隔，本次为 5 分钟。
3. 如果相邻时间戳间隔不等于预期采样间隔，则切分为新的连续 segment。
4. 每个滑窗必须完全位于同一个连续 segment 内。
5. 长度小于 `seq_len + pred_len` 的 segment 会被丢弃。

每个样本的形式为：

```text
input  = [t-seq_len+1, ..., t]
target = [t+1, ..., t+pred_len]
```

该约束与 Persistence、LSTM baseline 一致，保证三类 baseline 可以公平比较。

## 6. 标准化、Loss 与指标尺度

标准化流程严格避免数据泄漏：

- `feature_scaler` 只在 train split 上拟合
- `target_scaler` 只在 train split 上拟合
- `validation / calibration / test` 只做 transform
- 最终评估前统一 inverse transform 回原始 `Active_Pow` 尺度

需要区分两类数值：

- `history.train_loss` 和 `history.validation_loss`：标准化目标空间上的 `MSELoss`
- `metrics.json` 中的 validation/test 指标：反归一化后的原始功率尺度指标

因此训练 loss 不能直接和最终 MAE/RMSE 等原始量纲指标混着解释。

## 7. 本次运行配置

默认 baseline 运行入口：

```bash
cd "${PROJECT_ROOT:-/path/to/CiTransformer}"
bash scripts/run_itransformer_experiments.sh
```

本次实际保存下来的配置如下：

| pred_len | seq_len | batch_size | d_model | n_heads | e_layers | d_ff | dropout | lr | epochs_cfg | patience | total_epochs | best_epoch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 96 | 256 | 128 | 4 | 2 | 256 | 0.1 | 1e-3 | 30 | 8 | 20 | 12 |
| 12 | 96 | 256 | 128 | 4 | 2 | 256 | 0.1 | 1e-3 | 80 | 15 | 59 | 44 |
| 24 | 96 | 256 | 128 | 4 | 2 | 256 | 0.1 | 1e-3 | 80 | 15 | 59 | 44 |
| 48 | 96 | 256 | 128 | 4 | 2 | 256 | 0.1 | 1e-3 | 120 | 20 | 86 | 66 |

其他固定配置：

| 参数 | 值 |
| --- | --- |
| `DATA_DIR` | `data/processed_selected_2020_2022` |
| `FACTOR` | `5` |
| `ACTIVATION` | `gelu` |
| `WEIGHT_DECAY` | `1e-5` |
| `GRAD_CLIP` | `1.0` |
| `MIN_DELTA` | `1e-5` |
| `NUM_WORKERS` | `0` |
| `SEED` | `42` |
| `DEVICE` | `auto` |
| `DISABLE_NORM` | `False` |
| `OUTPUT_ATTENTION` | `False` |

默认输入特征：

- `Active_Pow`
- `Radiation_Global_Tilted`
- `Radiation_Diffuse_Tilted`
- `Weather_T`
- `Weather_R`
- `solar_elevation`
- `sin_time_of_day`
- `cos_time_of_day`
- `sin_day_of_year`
- `cos_day_of_year`
- `day_night_label`

## 8. 输出文件

本次四组默认 baseline 实验都已完成，并生成以下文件：

| pred_len | metrics | predictions | plot | checkpoint |
| ---: | --- | --- | --- | --- |
| 1 | `results/itransformer/pred_len_1/metrics.json` | `results/itransformer/pred_len_1/predictions.csv` | `results/itransformer/pred_len_1/pred_plot.png` | `checkpoints/itransformer/pred_len_1/best_model.pth` |
| 12 | `results/itransformer/pred_len_12/metrics.json` | `results/itransformer/pred_len_12/predictions.csv` | `results/itransformer/pred_len_12/pred_plot.png` | `checkpoints/itransformer/pred_len_12/best_model.pth` |
| 24 | `results/itransformer/pred_len_24/metrics.json` | `results/itransformer/pred_len_24/predictions.csv` | `results/itransformer/pred_len_24/pred_plot.png` | `checkpoints/itransformer/pred_len_24/best_model.pth` |
| 48 | `results/itransformer/pred_len_48/metrics.json` | `results/itransformer/pred_len_48/predictions.csv` | `results/itransformer/pred_len_48/pred_plot.png` | `checkpoints/itransformer/pred_len_48/best_model.pth` |

`metrics.json` 同时保存：

- `validation_metrics`
- `test_metrics`
- `reported_metrics`，当前等同于 test
- `history`
- `per_horizon_all`
- `per_horizon_daytime`

`predictions.csv` 和 `pred_plot.png` 当前基于 test split 导出。

## 9. 指标口径

本次指标按两个 scope 汇报：

- `all_timestamps`：所有时间点
- `daytime_only`：只统计白天目标点

每个 scope 下的聚合指标包括：

| 指标 | 含义 |
| --- | --- |
| `count` | 参与统计的预测点数量 |
| `mae` | Mean Absolute Error |
| `mse` | Mean Squared Error |
| `rmse` | Root Mean Squared Error |
| `mbe` | Mean Bias Error |
| `median_ae` | Median Absolute Error |
| `p95_ae` | 95th percentile Absolute Error |
| `max_ae` | Maximum Absolute Error |
| `smape` | symmetric MAPE |
| `mape_nonzero` | 排除 `abs(y_true) <= 1e-6` 后的 MAPE |
| `wape` | Weighted Absolute Percentage Error |
| `nmae_by_mean` | 以 `mean(abs(y_true))` 归一化的 MAE |
| `nrmse_by_mean` | 以 `mean(abs(y_true))` 归一化的 RMSE |
| `nmbe_by_mean` | 以 `mean(abs(y_true))` 归一化的 MBE |
| `nmae_by_max` | 以 `max(abs(y_true))` 归一化的 MAE |
| `nrmse_by_max` | 以 `max(abs(y_true))` 归一化的 RMSE |
| `nmbe_by_max` | 以 `max(abs(y_true))` 归一化的 MBE |
| `r2` | R-squared |
| `pearson_r` | Pearson correlation coefficient |
| `mean_true` | 真实值均值 |
| `mean_pred` | 预测值均值 |
| `mean_abs_true` | 真实值绝对值均值 |
| `max_abs_true` | 真实值绝对值最大值 |
| `sum_abs_true` | 真实值绝对值总和 |
| `nonzero_target_count` | 非零目标点数量 |

说明：

- `MAPE` 排除了接近 0 的目标值，避免夜间零功率造成除零问题。
- `nmae_by_mean / nrmse_by_mean / nmbe_by_mean` 使用 `mean(abs(y_true))` 作为分母。
- `nmae_by_max / nrmse_by_max / nmbe_by_max` 使用当前统计范围内的 `max(abs(y_true))` 作为近似容量尺度。
- `per_horizon_all` 和 `per_horizon_daytime` 中对每个 horizon 也保存了同一组指标；完整逐 horizon 数值以各 `metrics.json` 为准。

## 10. 关键结果表

论文主表通常可以优先放 test split 的四个核心指标：

| pred_len | epochs_cfg | patience | total_epochs | best_epoch | best_validation_loss | test all MAE | test all RMSE | test daytime MAE | test daytime RMSE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 30 | 8 | 20 | 12 | 0.026334 | 0.198580 | 0.468537 | 0.382269 | 0.678439 |
| 12 | 80 | 15 | 59 | 44 | 0.068833 | 0.381554 | 0.785343 | 0.720058 | 1.119539 |
| 24 | 80 | 15 | 59 | 44 | 0.088474 | 0.488084 | 0.948665 | 0.853244 | 1.298828 |
| 48 | 120 | 20 | 86 | 66 | 0.118749 | 0.629728 | 1.118483 | 0.999557 | 1.465227 |

本次加长训练后的判断：

- `pred_len=1`：已充分，best epoch 为 12，后续由 early stopping 停在 20。
- `pred_len=12/24`：基本够，best epoch 为 44，实际跑到 59 后 early stopping。
- `pred_len=48`：已从 80 epoch 继续补到 120 epoch 配置，实际跑到 86 后 early stopping，best epoch 仍为 66；继续单纯加 epoch 的收益预计有限。

## 11. 完整聚合指标

下面罗列本次四组默认 iTransformer 实验的完整聚合指标。列名与 `metrics.json` 保持一致。

### 11.1 Validation - all_timestamps

| pred_len | total_epochs | best_epoch | count | mae | mse | rmse | mbe | median_ae | p95_ae | max_ae | smape | mape_nonzero | wape | nmae_by_mean | nrmse_by_mean | nmbe_by_mean | nmae_by_max | nrmse_by_max | nmbe_by_max | r2 | pearson_r | mean_true | mean_pred | mean_abs_true | max_abs_true | sum_abs_true | nonzero_target_count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 20 | 12 | 46109 | 0.187758 | 0.208716 | 0.456854 | 0.026327 | 0.042965 | 0.896129 | 6.440719 | 115.915295 | 15.164853 | 8.764222 | 8.764222 | 21.325172 | 1.228914 | 1.985003 | 4.829925 | 0.278336 | 0.973409 | 0.986873 | 2.142325 | 2.168652 | 2.142325 | 9.458833 | 98780.463220 | 20858 |
| 12 | 59 | 44 | 551856 | 0.348138 | 0.545544 | 0.738609 | 0.011466 | 0.090348 | 1.679537 | 8.501401 | 120.495192 | 32.880339 | 16.224024 | 16.224024 | 34.420839 | 0.534343 | 3.680565 | 7.808674 | 0.121220 | 0.930555 | 0.965544 | 2.145821 | 2.157287 | 2.145821 | 9.458833 | 1184183.965379 | 250032 |
| 24 | 59 | 44 | 1100544 | 0.430832 | 0.701210 | 0.837383 | 0.000953 | 0.147887 | 1.913341 | 11.450414 | 121.502356 | 50.744596 | 20.043039 | 20.043039 | 38.956474 | 0.044338 | 4.554813 | 8.852922 | 0.010076 | 0.910811 | 0.955911 | 2.149535 | 2.150488 | 2.149535 | 9.458833 | 2365657.859945 | 499422 |
| 48 | 86 | 66 | 2188416 | 0.559661 | 0.941159 | 0.970134 | -0.079578 | 0.254545 | 2.207343 | 15.470477 | 122.765458 | 59.191738 | 25.950890 | 25.950890 | 44.984101 | -3.689963 | 5.916805 | 10.256378 | -0.841312 | 0.880460 | 0.941102 | 2.156614 | 2.077036 | 2.156614 | 9.458833 | 4719569.550962 | 996232 |

### 11.2 Validation - daytime_only

| pred_len | total_epochs | best_epoch | count | mae | mse | rmse | mbe | median_ae | p95_ae | max_ae | smape | mape_nonzero | wape | nmae_by_mean | nrmse_by_mean | nmbe_by_mean | nmae_by_max | nrmse_by_max | nmbe_by_max | r2 | pearson_r | mean_true | mean_pred | mean_abs_true | max_abs_true | sum_abs_true | nonzero_target_count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 20 | 12 | 20898 | 0.364949 | 0.441833 | 0.664705 | 0.045862 | 0.161801 | 1.492917 | 6.440719 | 14.548381 | 15.164853 | 7.720854 | 7.720854 | 14.062510 | 0.970252 | 3.858284 | 7.027351 | 0.484857 | 0.913403 | 0.957145 | 4.726790 | 4.772652 | 4.726790 | 9.458833 | 98780.463220 | 20858 |
| 12 | 59 | 44 | 250512 | 0.668582 | 1.144794 | 1.069951 | -0.000456 | 0.348151 | 2.473812 | 8.501401 | 24.980613 | 32.880339 | 14.143731 | 14.143731 | 22.634613 | -0.009641 | 7.068335 | 11.311656 | -0.004818 | 0.775704 | 0.891328 | 4.727055 | 4.726599 | 4.727055 | 9.458833 | 1184183.965379 | 250032 |
| 24 | 59 | 44 | 500382 | 0.768763 | 1.414038 | 1.189133 | 0.024833 | 0.425573 | 2.646954 | 11.450414 | 27.409595 | 50.744596 | 16.260811 | 16.260811 | 25.152453 | 0.525258 | 8.127461 | 12.571672 | 0.262534 | 0.722901 | 0.861545 | 4.727704 | 4.752536 | 4.727704 | 9.458833 | 2365657.859945 | 499422 |
| 48 | 86 | 66 | 998152 | 0.925020 | 1.815249 | 1.347312 | -0.070192 | 0.601317 | 2.928436 | 15.470477 | 30.677205 | 59.191738 | 19.563451 | 19.563451 | 28.494590 | -1.484516 | 9.779432 | 14.243954 | -0.742084 | 0.644201 | 0.821408 | 4.728307 | 4.658115 | 4.728307 | 9.458833 | 4719569.550962 | 996232 |

### 11.3 Test - all_timestamps

| pred_len | total_epochs | best_epoch | count | mae | mse | rmse | mbe | median_ae | p95_ae | max_ae | smape | mape_nonzero | wape | nmae_by_mean | nrmse_by_mean | nmbe_by_mean | nmae_by_max | nrmse_by_max | nmbe_by_max | r2 | pearson_r | mean_true | mean_pred | mean_abs_true | max_abs_true | sum_abs_true | nonzero_target_count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 20 | 12 | 100121 | 0.198580 | 0.219527 | 0.468537 | 0.027190 | 0.049317 | 0.957350 | 6.020705 | 114.530726 | 21.282000 | 9.249129 | 9.249129 | 21.822749 | 1.266433 | 2.011303 | 4.745545 | 0.275397 | 0.972326 | 0.986319 | 2.147013 | 2.174203 | 2.147013 | 9.873200 | 214961.055930 | 46319 |
| 12 | 59 | 44 | 1198944 | 0.381554 | 0.616764 | 0.785343 | 0.012284 | 0.101093 | 1.827213 | 12.439950 | 120.004366 | 66.033316 | 17.759518 | 17.759518 | 36.554006 | 0.571750 | 3.864540 | 7.954293 | 0.124415 | 0.922256 | 0.961548 | 2.148447 | 2.160730 | 2.148447 | 9.873200 | 2575867.119601 | 555185 |
| 24 | 59 | 44 | 2392416 | 0.488084 | 0.899965 | 0.948665 | -0.012724 | 0.158991 | 2.172454 | 23.888534 | 121.521740 | 102.962236 | 22.702634 | 22.702634 | 44.126036 | -0.591860 | 4.943519 | 9.608484 | -0.128878 | 0.886554 | 0.945199 | 2.149898 | 2.137174 | 2.149898 | 9.873200 | 5143450.891159 | 1109063 |
| 48 | 86 | 66 | 4762944 | 0.629728 | 1.251004 | 1.118483 | -0.104629 | 0.265542 | 2.516051 | 49.226705 | 123.012568 | 117.972270 | 29.250767 | 29.250767 | 51.953335 | -4.859984 | 6.378157 | 11.328474 | -1.059724 | 0.842277 | 0.924705 | 2.152861 | 2.048232 | 2.152861 | 9.873200 | 10253954.924974 | 2212895 |

### 11.4 Test - daytime_only

| pred_len | total_epochs | best_epoch | count | mae | mse | rmse | mbe | median_ae | p95_ae | max_ae | smape | mape_nonzero | wape | nmae_by_mean | nrmse_by_mean | nmbe_by_mean | nmae_by_max | nrmse_by_max | nmbe_by_max | r2 | pearson_r | mean_true | mean_pred | mean_abs_true | max_abs_true | sum_abs_true | nonzero_target_count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 20 | 12 | 46500 | 0.382269 | 0.460279 | 0.678439 | 0.042459 | 0.176517 | 1.524007 | 6.020705 | 16.038099 | 21.282000 | 8.269168 | 8.269168 | 14.675871 | 0.918458 | 3.871780 | 6.871519 | 0.430039 | 0.918312 | 0.959576 | 4.622818 | 4.665277 | 4.622818 | 9.873200 | 214961.055930 | 46319 |
| 12 | 59 | 44 | 557357 | 0.720058 | 1.253367 | 1.119539 | 0.009177 | 0.398578 | 2.582681 | 12.439950 | 28.021919 | 66.033316 | 15.580355 | 15.580355 | 24.224180 | 0.198559 | 7.293054 | 11.339166 | 0.092944 | 0.777598 | 0.891785 | 4.621575 | 4.630751 | 4.621575 | 9.873200 | 2575867.119601 | 555185 |
| 24 | 59 | 44 | 1113407 | 0.853244 | 1.686954 | 1.298828 | 0.034830 | 0.499951 | 2.856599 | 23.888534 | 31.430973 | 102.962236 | 18.470233 | 18.470233 | 28.115832 | 0.753964 | 8.642016 | 13.155084 | 0.352771 | 0.700740 | 0.854273 | 4.619560 | 4.654390 | 4.619560 | 9.873200 | 5143450.891159 | 1109063 |
| 48 | 86 | 66 | 2221583 | 0.999557 | 2.146889 | 1.465227 | -0.038852 | 0.646826 | 3.165293 | 49.226705 | 34.955024 | 117.972270 | 21.656017 | 21.656017 | 31.745046 | -0.841751 | 10.123938 | 14.840443 | -0.393509 | 0.619207 | 0.809243 | 4.615607 | 4.576755 | 4.615607 | 9.873200 | 10253954.924974 | 2212895 |

## 12. 调参与导出工作流

除了默认 baseline，仓库仍保留 validation-only 调参与 tuned checkpoint 导出流程。

运行标准调参：

```bash
cd "${PROJECT_ROOT:-/path/to/CiTransformer}"
bash scripts/run_itransformer_tuning_standard.sh
```

调参逻辑：

- `test` 不参与调参
- 单次 run 使用 `--tuning_only --report_split validation`
- 默认主目标是 `pred_len=12/24/48`
- `pred_len=1` 作为参考 run
- 汇总时优先看 validation `daytime_only RMSE`，再看 `daytime_only MAE`

用调参选出的最佳 checkpoint 导出 test：

```bash
cd "${PROJECT_ROOT:-/path/to/CiTransformer}"

PYTHON_BIN=./.venv/bin/python \
MODE=export_tuned_best \
PRED_LENS="1 12 24 48" \
RUN_PRED_LEN1_REF=1 \
RESULTS_BASE_DIR=results/itransformer_tuned \
bash scripts/run_itransformer_experiments.sh
```

这一步不重新训练，而是加载调参 summary 中定位到的 checkpoint，并导出最终 test 指标。第 10-11 节的结果表只汇报本次最新默认 `results/itransformer/` baseline；调参后的 tuned 结果作为独立补充实验记录在第 16 节。

## 13. 运行验证

本次已经完成以下验证：

- `pred_len=1` 运行完成，输出 `metrics.json / predictions.csv / pred_plot.png / best_model.pth`
- `pred_len=12` 运行完成，输出 `metrics.json / predictions.csv / pred_plot.png / best_model.pth`
- `pred_len=24` 运行完成，输出 `metrics.json / predictions.csv / pred_plot.png / best_model.pth`
- `pred_len=48` 运行完成，输出 `metrics.json / predictions.csv / pred_plot.png / best_model.pth`
- 四组结果目录都位于 `results/itransformer/pred_len_*`
- 四组 checkpoint 都位于 `checkpoints/itransformer/pred_len_*`
- `metrics.json` 中的 `config.pred_len` 与目录名一致

## 14. 论文写法建议

可以在方法或实验设置中写：

```text
The vanilla iTransformer baseline is trained only on the training split, with the validation split used for early stopping and checkpoint selection. Feature and target scalers are fit only on the training split. Sliding windows are constructed only within timestamp-continuous segments to avoid crossing temporal gaps. The multivariate iTransformer backbone predicts all encoder variables, after which the Active_Pow channel is extracted as the forecasting target. Evaluation is performed after inverse transformation on the original power scale, and results are reported for both all timestamps and daytime-only timestamps.
```

结果分析时建议同时汇报 `all_timestamps` 和 `daytime_only`，但正文重点关注 `daytime_only`。原因是夜间光伏功率大量接近 0，`all_timestamps` 会让整体误差看起来偏乐观，而白天时段更能反映实际发电预测难度。

需要注意：当前 best checkpoint 的选择依据是标准化目标空间上的全时段 validation MSE，并不是 daytime-only validation MAE/RMSE。如果论文主线更强调白天预测质量，后续可以考虑按 daytime validation 指标选择 checkpoint。

## 15. 当前结论

本次 vanilla iTransformer baseline 已经按 `pred_len=1/12/24/48` 全部跑完，且四组输出完整。

从训练充分性看，单纯增加 epoch 的主要问题已经基本处理：

- `pred_len=12/24` 在 `EPOCHS=80 / PATIENCE=15` 下 early stopping，best epoch 均为 44。
- `pred_len=48` 在 `EPOCHS=120 / PATIENCE=20` 下 early stopping，best epoch 为 66。
- 因此当前 iTransformer 的短板不再主要是 epoch 不够，而更可能是默认超参数尚未调优。

从 test split 看，随着预测步长从 1 增加到 48，误差整体增大，符合多步预测任务的预期。若后续希望进一步提升 vanilla iTransformer baseline，建议优先做 validation-only 调参，重点尝试 `learning_rate`、`dropout`、`e_layers`、`d_model`、`d_ff` 和学习率调度，而不是继续单纯增加 epoch。

## 16. 补充实验：validation-only tuned iTransformer

本节记录标准调参完成后，用 validation 选出的最优共享配置导出 test split 的结果。该实验不覆盖第 7-15 节默认 iTransformer baseline，而是作为 tuned iTransformer 的补充结果单独保存。

调参与 test 导出命令如下：

```bash
cd "${PROJECT_ROOT:-/path/to/CiTransformer}"
EPOCHS=120 PATIENCE=20 PLAN=standard bash scripts/run_itransformer_tuning_standard.sh

MODE=export_tuned_best \
REPORT_SPLIT=test \
PRED_LENS="1 12 24 48" \
TUNING_PLAN=standard \
bash scripts/run_itransformer_experiments.sh
```

调参输出与 tuned test 输出位置如下：

| 类型 | 路径 |
| --- | --- |
| validation 调参根目录 | `results/tuning/itransformer/standard/` |
| final validation summary | `results/tuning/itransformer/standard/summary/final_validation/` |
| 最优共享配置 | `results/tuning/itransformer/standard/summary/final_validation/best_shared_config.json` |
| tuned test 输出 | `results/itransformer_tuned/pred_len_*/` |

### 16.1 最优共享配置

标准调参主目标覆盖 `pred_len=12/24/48`，排序优先级为 validation `daytime_only RMSE`，再看 validation `daytime_only MAE`。本次选出的最优共享配置为：

| 参数 | 值 |
| --- | ---: |
| `seq_len` | `96` |
| `learning_rate` | `0.001` |
| `batch_size` | `128` |
| `d_model` | `192` |
| `d_ff` | `384` |
| `e_layers` | `2` |
| `dropout` | `0.1` |
| `weight_decay` | `1e-5` |
| `activation` | `gelu` |
| `n_heads` | `4` |
| `factor` | `5` |
| `disable_norm` | `False` |
| `output_attention` | `False` |
| `seed` | `42` |

共享配置排名前 5 如下：

| rank | avg val daytime RMSE | max val daytime RMSE | avg val daytime MAE | max val daytime MAE | batch_size | d_model | d_ff | e_layers | dropout |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.189987 | 1.340880 | 0.780358 | 0.909594 | 128 | 192 | 384 | 2 | 0.1 |
| 2 | 1.198146 | 1.362143 | 0.788257 | 0.947346 | 128 | 128 | 512 | 2 | 0.1 |
| 3 | 1.198388 | 1.353383 | 0.796052 | 0.929621 | 128 | 128 | 256 | 3 | 0.1 |
| 4 | 1.202212 | 1.345521 | 0.787032 | 0.911387 | 128 | 128 | 256 | 2 | 0.1 |
| 5 | 1.202999 | 1.336352 | 0.796368 | 0.922764 | 256 | 128 | 256 | 2 | 0.2 |

### 16.2 Tuned Test 结果

下面结果来自 `results/itransformer_tuned/pred_len_*/metrics.json`。这些 run 使用 `--eval_checkpoint_path` 加载调参阶段保存的最优 checkpoint，因此 `history` 为空，`best_epoch` 和 `best_validation_loss` 来自 checkpoint 本身。

| pred_len | source checkpoint | batch_size | d_model | d_ff | best_epoch | best_validation_loss | val daytime MAE | val daytime RMSE | test all MAE | test all RMSE | test daytime MAE | test daytime RMSE | test daytime R2 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `ref1_sharedbest_pl1_lr0p001_bs128_dm192_ff384_el2_do0p1_wd1e-05_actgelu_nh4_fac5` | 128 | 192 | 384 | 13 | 0.026949 | 0.356874 | 0.668531 | 0.208397 | 0.473310 | 0.375503 | 0.681652 | 0.917536 |
| 12 | `s2_wider192_pl12_lr0p001_bs128_dm192_ff384_el2_do0p1_wd1e-05_actgelu_nh4_fac5` | 128 | 192 | 384 | 68 | 0.065223 | 0.654698 | 1.041916 | 0.375859 | 0.776159 | 0.699056 | 1.094358 | 0.787490 |
| 24 | `s2_wider192_pl24_lr0p001_bs128_dm192_ff384_el2_do0p1_wd1e-05_actgelu_nh4_fac5` | 128 | 192 | 384 | 45 | 0.088540 | 0.776782 | 1.187167 | 0.477956 | 0.918328 | 0.830344 | 1.258302 | 0.719124 |
| 48 | `s2_wider192_pl48_lr0p001_bs128_dm192_ff384_el2_do0p1_wd1e-05_actgelu_nh4_fac5` | 128 | 192 | 384 | 80 | 0.117185 | 0.909594 | 1.340880 | 0.628939 | 1.122469 | 1.012052 | 1.507176 | 0.597091 |

### 16.3 与默认 iTransformer 对比

负数表示 tuned iTransformer 误差低于默认 iTransformer；对 `R2` 则正数表示 tuned 更好。

| pred_len | metric | default iTransformer | tuned iTransformer | delta tuned-default | change |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | test all MAE | 0.198580 | 0.208397 | +0.009817 | +4.94% |
| 1 | test all RMSE | 0.468537 | 0.473310 | +0.004773 | +1.02% |
| 1 | test daytime MAE | 0.382269 | 0.375503 | -0.006766 | -1.77% |
| 1 | test daytime RMSE | 0.678439 | 0.681652 | +0.003213 | +0.47% |
| 12 | test all MAE | 0.381554 | 0.375859 | -0.005694 | -1.49% |
| 12 | test all RMSE | 0.785343 | 0.776159 | -0.009185 | -1.17% |
| 12 | test daytime MAE | 0.720058 | 0.699056 | -0.021001 | -2.92% |
| 12 | test daytime RMSE | 1.119539 | 1.094358 | -0.025180 | -2.25% |
| 24 | test all MAE | 0.488084 | 0.477956 | -0.010127 | -2.07% |
| 24 | test all RMSE | 0.948665 | 0.918328 | -0.030337 | -3.20% |
| 24 | test daytime MAE | 0.853244 | 0.830344 | -0.022900 | -2.68% |
| 24 | test daytime RMSE | 1.298828 | 1.258302 | -0.040526 | -3.12% |
| 48 | test all MAE | 0.629728 | 0.628939 | -0.000789 | -0.13% |
| 48 | test all RMSE | 1.118483 | 1.122469 | +0.003987 | +0.36% |
| 48 | test daytime MAE | 0.999557 | 1.012052 | +0.012495 | +1.25% |
| 48 | test daytime RMSE | 1.465227 | 1.507176 | +0.041949 | +2.86% |

本轮 tuned iTransformer 的主要收益集中在 `pred_len=12/24`：

- `pred_len=12`：test daytime MAE 从 `0.720058` 降到 `0.699056`，RMSE 从 `1.119539` 降到 `1.094358`。
- `pred_len=24`：test daytime MAE 从 `0.853244` 降到 `0.830344`，RMSE 从 `1.298828` 降到 `1.258302`。
- `pred_len=48`：test all MAE 几乎持平，但 daytime MAE/RMSE 变差，说明 validation 共享配置没有稳定迁移到最长预测步长的 test daytime 指标。
- `pred_len=1`：该 run 只是共享最优配置的参考测试，不是主调参目标；daytime MAE 略好，但 RMSE 基本持平略差。

### 16.4 与 LSTM baseline 的关系

如果取当前已有 LSTM 两组结果中的较优 daytime 指标作为对照，tuned iTransformer 仍未超过 LSTM：

| pred_len | tuned iTransformer daytime MAE | best LSTM daytime MAE | MAE gap | tuned iTransformer daytime RMSE | best LSTM daytime RMSE | RMSE gap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.375503 | 0.358423 | +0.017080 | 0.681652 | 0.631753 | +0.049899 |
| 12 | 0.699056 | 0.641671 | +0.057385 | 1.094358 | 0.974316 | +0.120043 |
| 24 | 0.830344 | 0.737715 | +0.092629 | 1.258302 | 1.082560 | +0.175741 |
| 48 | 1.012052 | 0.868246 | +0.143806 | 1.507176 | 1.238622 | +0.268554 |

这说明 vanilla iTransformer 经过当前 standard 调参后，虽然相对默认 iTransformer 在中等预测步长上有改善，但仍不是本数据集上最强的神经网络 baseline。论文后续不应把贡献点放在 vanilla iTransformer 本身，而应重点验证 CiTransformer 的因果约束、变量级 attention 约束和可解释性是否能进一步改善 `pred_len=12/24/48`，尤其是 daytime 指标。
