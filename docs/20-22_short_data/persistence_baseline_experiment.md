# Persistence Baseline 实验文档

更新日期：2026-05-07

本文档是 Persistence baseline 的单一权威说明，已经合并原来的实现工作记录和实验流程说明。后续关于 Persistence baseline 的实现、协议、运行方式和结果汇总，以本文档为准。

本文档从属于统一实验协议：`docs/实验协议.md`。除非本文件特别说明，数据来源、split、`seq_len / pred_len`、指标口径和结果导出结构都遵循统一协议。

## 1. 实验目的

Persistence baseline 用来作为光伏功率预测任务中的规则型对照组。它不训练任何神经网络参数，也不做超参数搜索，而是直接复制输入窗口中最后一个已观测的 `Active_Pow`，作为未来所有预测步长的输出。

这类 baseline 对光伏预测很重要：如果复杂模型不能稳定超过它，说明模型还没有真正学到比“最近观测值延续”更多的信息。

## 2. 代码位置

核心文件如下：

- `models/baseline/persistence.py`：Persistence baseline 模型定义
- `scripts/run_persistence.py`：单次评估入口
- `scripts/run_persistence_experiments.sh`：批量运行 `pred_len=1/12/24/48`
- `utils/datasets.py`：连续时间片段数据集、滑窗与标准化
- `results/persistence/pred_len_*/`：本次实验输出目录

原来的 `docs/persistence_baseline_work_summary.md` 已合并到本文档，不再单独维护第二份完整说明。

## 3. Baseline 定义

输入形状：

```text
[batch, seq_len, feature_dim]
```

输出形状：

```text
[batch, pred_len]
```

预测规则：

```text
y_hat[t+1:t+pred_len] = last_observed_active_power
```

也就是：

```text
y_hat = [last_power] * pred_len
```

这里的 `last_power` 来自输入窗口最后一个时间步的 `Active_Pow`。代码不会把 `Active_Pow` 的列索引写死，而是从 `feature_cols` 中动态定位 `target_col`，得到 `target_feature_index`。

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
| train | 拟合 `feature_scaler` 和 `target_scaler`；推断预期采样间隔；不参与参数训练 |
| validation | 执行完整规则型推理；写入 `validation_metrics` |
| calibration | 为协议一致性读取并统计；当前 Persistence baseline 不使用它 |
| test | 执行完整规则型推理；写入 `test_metrics`；导出 `predictions.csv` 和 `pred_plot.png` |

论文中可以明确写：

```text
The calibration split is reserved for protocol consistency but is unused in the current persistence baseline.
```

## 5. 连续时间片段约束

当前实验不会直接对整张表按行号滑窗。样本构造遵循以下规则：

1. 按时间戳排序。
2. 从训练集时间戳推断预期采样间隔，本次为 5 分钟。
3. 如果相邻时间戳间隔不等于预期采样间隔，则切分为新的连续 segment。
4. 每个滑窗必须完全位于同一个连续 segment 内。
5. 长度小于 `seq_len + pred_len` 的 segment 会被丢弃。

这对 Persistence baseline 很关键，因为跨时间断点复制最后观测功率会把断点前的功率错误延续到断点后，造成协议错误。

## 6. 标准化与指标尺度

虽然 Persistence baseline 不训练，但仍然沿用统一的无泄漏标准化流程：

- `feature_scaler` 只在 train split 上拟合
- `target_scaler` 只在 train split 上拟合
- `validation / calibration / test` 只做 transform
- 最终评估前统一 inverse transform 回原始 `Active_Pow` 尺度

需要特别注意的是：

- 输入 `x` 中的最后一个 `Active_Pow` 位于 feature scaler 空间
- 目标 `y` 位于 target scaler 空间

因此模型前向时会先把最后一个输入功率值从 feature scaler 空间还原到原始功率尺度，再映射到 target scaler 空间，最后交给统一的 `collect_predictions(...)` 和 inverse-transform 评估流程。

本文档中所有结果指标都已经是反归一化后的原始功率尺度，可以直接用于论文表格。

## 7. 本次运行配置

本次运行命令：

```bash
cd /home/lunalhx/projects/CiTransformer
bash scripts/run_persistence_experiments.sh
```

批量脚本默认配置：

| 参数 | 值 |
| --- | --- |
| `DATA_DIR` | `data/processed_selected_2020_2022` |
| `SEQ_LEN` | `96` |
| `PRED_LENS` | `1 12 24 48` |
| `BATCH_SIZE` | `256` |
| `NUM_WORKERS` | `0` |
| `SEED` | `42` |
| `DEVICE` | `auto`，本次实际使用 `cpu` |
| `RESULTS_BASE_DIR` | `results/persistence` |

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

本次四组实验都已完成，并生成以下文件：

| pred_len | metrics | predictions | plot |
| ---: | --- | --- | --- |
| 1 | `results/persistence/pred_len_1/metrics.json` | `results/persistence/pred_len_1/predictions.csv` | `results/persistence/pred_len_1/pred_plot.png` |
| 12 | `results/persistence/pred_len_12/metrics.json` | `results/persistence/pred_len_12/predictions.csv` | `results/persistence/pred_len_12/pred_plot.png` |
| 24 | `results/persistence/pred_len_24/metrics.json` | `results/persistence/pred_len_24/predictions.csv` | `results/persistence/pred_len_24/pred_plot.png` |
| 48 | `results/persistence/pred_len_48/metrics.json` | `results/persistence/pred_len_48/predictions.csv` | `results/persistence/pred_len_48/pred_plot.png` |

`metrics.json` 同时保存：

- `validation_metrics`
- `test_metrics`
- `reported_metrics`，当前等同于 test
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

| pred_len | test all MAE | test all RMSE | test daytime MAE | test daytime RMSE |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.173894 | 0.486424 | 0.367969 | 0.706817 |
| 12 | 0.446656 | 0.959829 | 0.881127 | 1.329748 |
| 24 | 0.690850 | 1.366852 | 1.271167 | 1.807145 |
| 48 | 1.180872 | 2.117195 | 1.938048 | 2.623120 |

## 11. 完整聚合指标

下面罗列本次四组 Persistence 实验的完整聚合指标。列名与 `metrics.json` 保持一致。

### 11.1 Validation - all_timestamps

| pred_len | count | mae | mse | rmse | mbe | median_ae | p95_ae | max_ae | smape | mape_nonzero | wape | nmae_by_mean | nrmse_by_mean | nmbe_by_mean | nmae_by_max | nrmse_by_max | nmbe_by_max | r2 | pearson_r | mean_true | mean_pred | mean_abs_true | max_abs_true | sum_abs_true | nonzero_target_count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 46109 | 0.162776 | 0.224123 | 0.473416 | 0.000033 | 0.000000 | 0.900226 | 6.772499 | 33.331304 | 12.842463 | 7.598106 | 7.598106 | 22.098227 | 0.001537 | 1.720890 | 5.005013 | 0.000348 | 0.971447 | 0.985724 | 2.142325 | 2.142358 | 2.142325 | 9.458833 | 98780.463220 | 20858 |
| 12 | 551856 | 0.437563 | 0.919197 | 0.958748 | 0.000592 | 0.000000 | 2.340334 | 8.484600 | 46.283081 | 39.323844 | 20.391387 | 20.391387 | 44.679775 | 0.027588 | 4.625968 | 10.136006 | 0.006259 | 0.882991 | 0.941500 | 2.145821 | 2.146413 | 2.145821 | 9.458833 | 1184183.965379 | 250032 |
| 24 | 1100544 | 0.680424 | 1.878210 | 1.370478 | 0.000822 | 0.000000 | 3.591795 | 8.484600 | 55.983814 | 57.712466 | 31.654494 | 31.654494 | 63.756948 | 0.038252 | 7.193535 | 14.488869 | 0.008693 | 0.761104 | 0.880560 | 2.149535 | 2.150357 | 2.149535 | 9.458833 | 2365657.859945 | 499422 |
| 48 | 2188416 | 1.169023 | 4.490369 | 2.119049 | 0.000348 | 0.149467 | 5.339334 | 9.018867 | 72.551032 | 83.019464 | 54.206380 | 54.206380 | 98.258134 | 0.016125 | 12.359058 | 22.402861 | 0.003677 | 0.429661 | 0.714832 | 2.156614 | 2.156962 | 2.156614 | 9.458833 | 4719569.550962 | 996232 |

### 11.2 Validation - daytime_only

| pred_len | count | mae | mse | rmse | mbe | median_ae | p95_ae | max_ae | smape | mape_nonzero | wape | nmae_by_mean | nrmse_by_mean | nmbe_by_mean | nmae_by_max | nrmse_by_max | nmbe_by_max | r2 | pearson_r | mean_true | mean_pred | mean_abs_true | max_abs_true | sum_abs_true | nonzero_target_count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 20898 | 0.349795 | 0.477631 | 0.691109 | -0.009278 | 0.127600 | 1.520910 | 6.772499 | 12.872555 | 12.842463 | 7.400275 | 7.400275 | 14.621101 | -0.196278 | 3.698083 | 7.306491 | -0.098084 | 0.906387 | 0.953551 | 4.726790 | 4.717513 | 4.726790 | 9.458833 | 98780.463220 | 20858 |
| 12 | 250512 | 0.865160 | 1.743328 | 1.320351 | -0.097447 | 0.510800 | 3.081363 | 8.484600 | 35.148214 | 39.323844 | 18.302313 | 18.302313 | 27.931797 | -2.061470 | 9.146587 | 13.958925 | -1.030220 | 0.658435 | 0.841602 | 4.727055 | 4.629608 | 4.727055 | 9.458833 | 1184183.965379 | 250032 |
| 24 | 500382 | 1.241914 | 3.190195 | 1.786112 | -0.252808 | 0.779000 | 4.142333 | 8.484600 | 49.608645 | 57.712466 | 26.268862 | 26.268862 | 37.779685 | -5.347365 | 13.129675 | 18.883002 | -2.672714 | 0.374840 | 0.735881 | 4.727704 | 4.474896 | 4.727704 | 9.458833 | 2365657.859945 | 499422 |
| 48 | 998152 | 1.880388 | 6.589020 | 2.566909 | -0.681893 | 1.278666 | 5.572267 | 9.018867 | 72.075834 | 83.019464 | 39.768730 | 39.768730 | 54.288108 | -14.421511 | 19.879702 | 27.137689 | -7.209065 | -0.291484 | 0.538687 | 4.728307 | 4.046414 | 4.728307 | 9.458833 | 4719569.550962 | 996232 |

### 11.3 Test - all_timestamps

| pred_len | count | mae | mse | rmse | mbe | median_ae | p95_ae | max_ae | smape | mape_nonzero | wape | nmae_by_mean | nrmse_by_mean | nmbe_by_mean | nmae_by_max | nrmse_by_max | nmbe_by_max | r2 | pearson_r | mean_true | mean_pred | mean_abs_true | max_abs_true | sum_abs_true | nonzero_target_count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 100121 | 0.173894 | 0.236609 | 0.486424 | -0.000091 | 0.000000 | 0.967400 | 6.450733 | 33.370406 | 14.301485 | 8.099348 | 8.099348 | 22.655872 | -0.004255 | 1.761273 | 4.926715 | -0.000925 | 0.970172 | 0.985086 | 2.147013 | 2.146921 | 2.147013 | 9.873200 | 214961.055930 | 46319 |
| 12 | 1198944 | 0.446656 | 0.921272 | 0.959829 | -0.000758 | 0.000000 | 2.341200 | 8.070533 | 46.743873 | 51.826487 | 20.789707 | 20.789707 | 44.675496 | -0.035286 | 4.523921 | 9.721561 | -0.007678 | 0.883872 | 0.941922 | 2.148447 | 2.147688 | 2.148447 | 9.873200 | 2575867.119601 | 555185 |
| 24 | 2392416 | 0.690850 | 1.868285 | 1.366852 | -0.001579 | 0.014933 | 3.510617 | 8.854469 | 56.814545 | 90.374783 | 32.134073 | 32.134073 | 63.577537 | -0.073468 | 6.997223 | 13.844065 | -0.015998 | 0.764491 | 0.882184 | 2.149898 | 2.148319 | 2.149898 | 9.873200 | 5143450.891159 | 1109063 |
| 48 | 4762944 | 1.180872 | 4.482515 | 2.117195 | -0.003266 | 0.175000 | 5.307734 | 9.057400 | 73.779885 | 151.532685 | 54.851274 | 54.851274 | 98.343341 | -0.151702 | 11.960372 | 21.443859 | -0.033079 | 0.434858 | 0.717170 | 2.152861 | 2.149595 | 2.152861 | 9.873200 | 10253954.924974 | 2212895 |

### 11.4 Test - daytime_only

| pred_len | count | mae | mse | rmse | mbe | median_ae | p95_ae | max_ae | smape | mape_nonzero | wape | nmae_by_mean | nrmse_by_mean | nmbe_by_mean | nmae_by_max | nrmse_by_max | nmbe_by_max | r2 | pearson_r | mean_true | mean_pred | mean_abs_true | max_abs_true | sum_abs_true | nonzero_target_count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 46500 | 0.367969 | 0.499590 | 0.706817 | -0.006645 | 0.139867 | 1.592477 | 6.450733 | 13.862269 | 14.301485 | 7.959846 | 7.959846 | 15.289742 | -0.143741 | 3.726950 | 7.158945 | -0.067302 | 0.911335 | 0.955875 | 4.622818 | 4.616174 | 4.622818 | 9.873200 | 214961.055930 | 46319 |
| 12 | 557357 | 0.881127 | 1.768231 | 1.329748 | -0.081315 | 0.522467 | 3.079946 | 8.070533 | 36.635987 | 51.826487 | 19.065506 | 19.065506 | 28.772624 | -1.759471 | 8.924428 | 13.468261 | -0.823596 | 0.686239 | 0.851366 | 4.621575 | 4.540260 | 4.621575 | 9.873200 | 2575867.119601 | 555185 |
| 24 | 1113407 | 1.271167 | 3.265774 | 1.807145 | -0.216679 | 0.817000 | 4.094000 | 8.854469 | 51.638366 | 90.374783 | 27.517055 | 27.517055 | 39.119418 | -4.690471 | 12.874923 | 18.303540 | -2.194619 | 0.420663 | 0.744737 | 4.619560 | 4.402881 | 4.619560 | 9.873200 | 5143450.891159 | 1109063 |
| 48 | 2221583 | 1.938048 | 6.880758 | 2.623120 | -0.600673 | 1.357534 | 5.641900 | 9.057400 | 74.617542 | 151.532685 | 41.989011 | 41.989011 | 56.831519 | -13.013951 | 19.629379 | 26.568080 | -6.083872 | -0.220436 | 0.531224 | 4.615607 | 4.014935 | 4.615607 | 9.873200 | 10253954.924974 | 2212895 |

## 12. 运行验证

本次已经完成以下验证：

- `pred_len=1` 运行完成，输出 `metrics.json / predictions.csv / pred_plot.png`
- `pred_len=12` 运行完成，输出 `metrics.json / predictions.csv / pred_plot.png`
- `pred_len=24` 运行完成，输出 `metrics.json / predictions.csv / pred_plot.png`
- `pred_len=48` 运行完成，输出 `metrics.json / predictions.csv / pred_plot.png`
- 四组结果目录都位于 `results/persistence/pred_len_*`
- `metrics.json` 中的 `config.pred_len` 与目录名一致
- 本次实际评估设备为 CPU

## 13. 论文写法建议

可以在方法或实验设置中写：

```text
The persistence baseline contains no trainable parameters. For each sample, it repeats the last observed Active_Pow in the encoder window for all future prediction horizons. Feature and target scalers are fit only on the training split, and evaluation is performed after inverse transformation on the original power scale. Sliding windows are constructed only within timestamp-continuous segments to avoid crossing temporal gaps. Results are reported for both all timestamps and daytime-only timestamps.
```

结果分析时建议同时汇报 `all_timestamps` 和 `daytime_only`，但正文重点关注 `daytime_only`。原因是夜间光伏功率大量接近 0，`all_timestamps` 会让整体误差看起来偏乐观，而白天时段更能反映实际发电预测难度。

## 14. 当前结论

本次 Persistence baseline 已经按 `pred_len=1/12/24/48` 全部跑完，且四组输出完整。

从 test split 看，随着预测步长从 1 增加到 48，误差稳定增大，符合 Persistence baseline 的预期行为。该 baseline 是一个强 naive baseline，后续 LSTM、vanilla iTransformer 和 Causal-iTransformer 的论文表格都应至少与它并排比较。
