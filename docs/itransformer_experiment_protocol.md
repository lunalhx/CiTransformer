# iTransformer Baseline 实验文档

更新日期：2026-05-07

本文档是 vanilla iTransformer baseline 的实验说明，已根据本次新跑出的 `results/itransformer/pred_len_1/12/24/48` 正式结果重写。后续关于默认 iTransformer baseline 的实现、协议、运行方式和结果汇总，以本文档为准。

本文档从属于统一实验协议：`docs/实验协议.md`。除非本文件特别说明，数据来源、split、`seq_len / pred_len`、指标口径和结果导出结构都遵循统一协议。

## 1. 实验目的

vanilla iTransformer baseline 用作光伏功率预测任务中的 Transformer 类深度学习对照组。它不包含 CiTransformer 的因果图、PCMCI、HMM、regime、causal mask 等额外设计，而是一个标准 iTransformer backbone 加工程适配层。

本次默认 baseline 已经按 `pred_len=1/12/24/48` 全部完成，并生成了对应的 metrics、predictions、plot 和 checkpoint。

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

本次默认 baseline 运行命令：

```bash
cd /home/lunalhx/projects/CiTransformer
bash scripts/run_itransformer_experiments.sh
```

批量脚本默认配置：

| 参数 | 值 |
| --- | --- |
| `DATA_DIR` | `data/processed_selected_2020_2022` |
| `MODE` | `train` |
| `SEQ_LEN` | `96` |
| `PRED_LENS` | `1 12 24 48` |
| `BATCH_SIZE` | `256` |
| `D_MODEL` | `128` |
| `N_HEADS` | `4` |
| `E_LAYERS` | `2` |
| `D_FF` | `256` |
| `FACTOR` | `5` |
| `DROPOUT` | `0.1` |
| `ACTIVATION` | `gelu` |
| `LEARNING_RATE` | `1e-3` |
| `WEIGHT_DECAY` | `1e-5` |
| `GRAD_CLIP` | `1.0` |
| `EPOCHS` | `30` |
| `PATIENCE` | `8` |
| `MIN_DELTA` | `1e-5` |
| `NUM_WORKERS` | `0` |
| `SEED` | `42` |
| `DEVICE` | `auto` |
| `DISABLE_NORM` | `0` |
| `OUTPUT_ATTENTION` | `0` |

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

| pred_len | best_epoch | best_validation_loss | test all MAE | test all RMSE | test daytime MAE | test daytime RMSE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12 | 0.026334 | 0.198580 | 0.468537 | 0.382269 | 0.678439 |
| 12 | 30 | 0.069700 | 0.402836 | 0.792542 | 0.725050 | 1.123624 |
| 24 | 30 | 0.092058 | 0.489028 | 0.948959 | 0.861342 | 1.296482 |
| 48 | 30 | 0.130177 | 0.638291 | 1.179369 | 1.046426 | 1.576175 |

从本次结果看，`pred_len=12/24/48` 的 best epoch 都是 30，说明在当前 `EPOCHS=30` 的固定设置下，长预测步长没有触发 early stopping，训练可能仍受 epoch 上限影响。后续如果要优化 vanilla iTransformer baseline，可以优先检查是否需要更长训练或更细的 learning-rate schedule。

## 11. 完整聚合指标

下面罗列本次四组默认 iTransformer 实验的完整聚合指标。列名与 `metrics.json` 保持一致。

### 11.1 Validation - all_timestamps

| pred_len | best_epoch | count | mae | mse | rmse | mbe | median_ae | p95_ae | max_ae | smape | mape_nonzero | wape | nmae_by_mean | nrmse_by_mean | nmbe_by_mean | nmae_by_max | nrmse_by_max | nmbe_by_max | r2 | pearson_r | mean_true | mean_pred | mean_abs_true | max_abs_true | sum_abs_true | nonzero_target_count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12 | 46109 | 0.187758 | 0.208716 | 0.456854 | 0.026327 | 0.042965 | 0.896129 | 6.440719 | 115.915295 | 15.164853 | 8.764222 | 8.764222 | 21.325172 | 1.228914 | 1.985003 | 4.829925 | 0.278336 | 0.973409 | 0.986873 | 2.142325 | 2.168652 | 2.142325 | 9.458833 | 98780.463220 | 20858 |
| 12 | 30 | 551856 | 0.371780 | 0.552412 | 0.743244 | -0.043121 | 0.134117 | 1.715450 | 8.188909 | 120.878638 | 31.821612 | 17.325785 | 17.325785 | 34.636829 | -2.009534 | 3.930509 | 7.857674 | -0.455881 | 0.929681 | 0.964791 | 2.145821 | 2.102700 | 2.145821 | 9.458833 | 1184183.965379 | 250032 |
| 24 | 30 | 1100544 | 0.438178 | 0.729614 | 0.854175 | -0.070913 | 0.136847 | 2.025096 | 8.642976 | 122.111164 | 46.384478 | 20.384797 | 20.384797 | 39.737647 | -3.298993 | 4.632478 | 9.030445 | -0.749701 | 0.907198 | 0.953731 | 2.149535 | 2.078622 | 2.149535 | 9.458833 | 2365657.859945 | 499422 |
| 48 | 30 | 2188416 | 0.574282 | 1.031730 | 1.015741 | -0.011898 | 0.213487 | 2.342404 | 13.230717 | 123.313310 | 64.562772 | 26.628888 | 26.628888 | 47.098869 | -0.551699 | 6.071388 | 10.738545 | -0.125787 | 0.868956 | 0.932915 | 2.156614 | 2.144716 | 2.156614 | 9.458833 | 4719569.550962 | 996232 |

### 11.2 Validation - daytime_only

| pred_len | best_epoch | count | mae | mse | rmse | mbe | median_ae | p95_ae | max_ae | smape | mape_nonzero | wape | nmae_by_mean | nrmse_by_mean | nmbe_by_mean | nmae_by_max | nrmse_by_max | nmbe_by_max | r2 | pearson_r | mean_true | mean_pred | mean_abs_true | max_abs_true | sum_abs_true | nonzero_target_count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12 | 20898 | 0.364949 | 0.441833 | 0.664705 | 0.045862 | 0.161801 | 1.492917 | 6.440719 | 14.548381 | 15.164853 | 7.720854 | 7.720854 | 14.062510 | 0.970252 | 3.858284 | 7.027351 | 0.484857 | 0.913403 | 0.957145 | 4.726790 | 4.772652 | 4.726790 | 9.458833 | 98780.463220 | 20858 |
| 12 | 30 | 250512 | 0.674559 | 1.145193 | 1.070137 | -0.076862 | 0.354375 | 2.471944 | 8.188909 | 25.726941 | 31.821612 | 14.270182 | 14.270182 | 22.638554 | -1.626006 | 7.131528 | 11.313625 | -0.812597 | 0.775625 | 0.889148 | 4.727055 | 4.650193 | 4.727055 | 9.458833 | 1184183.965379 | 250032 |
| 24 | 30 | 500382 | 0.793696 | 1.468402 | 1.211777 | -0.132358 | 0.441431 | 2.813501 | 8.642976 | 28.732448 | 46.384478 | 16.788186 | 16.788186 | 25.631398 | -2.799623 | 8.391053 | 12.811058 | -1.399305 | 0.712248 | 0.862117 | 4.727704 | 4.595346 | 4.727704 | 9.458833 | 2365657.859945 | 499422 |
| 48 | 30 | 998152 | 0.961589 | 1.952197 | 1.397211 | -0.117014 | 0.604052 | 3.097513 | 13.230717 | 31.898576 | 64.562772 | 20.336864 | 20.336864 | 29.549909 | -2.474748 | 10.166048 | 14.771490 | -1.237084 | 0.617359 | 0.803642 | 4.728307 | 4.611294 | 4.728307 | 9.458833 | 4719569.550962 | 996232 |

### 11.3 Test - all_timestamps

| pred_len | best_epoch | count | mae | mse | rmse | mbe | median_ae | p95_ae | max_ae | smape | mape_nonzero | wape | nmae_by_mean | nrmse_by_mean | nmbe_by_mean | nmae_by_max | nrmse_by_max | nmbe_by_max | r2 | pearson_r | mean_true | mean_pred | mean_abs_true | max_abs_true | sum_abs_true | nonzero_target_count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12 | 100121 | 0.198580 | 0.219527 | 0.468537 | 0.027190 | 0.049317 | 0.957350 | 6.020705 | 114.530726 | 21.282000 | 9.249129 | 9.249129 | 21.822749 | 1.266433 | 2.011303 | 4.745545 | 0.275397 | 0.972326 | 0.986319 | 2.147013 | 2.174203 | 2.147013 | 9.873200 | 214961.055930 | 46319 |
| 12 | 30 | 1198944 | 0.402836 | 0.628123 | 0.792542 | -0.034162 | 0.139116 | 1.854131 | 11.348562 | 120.295073 | 65.370129 | 18.750121 | 18.750121 | 36.889064 | -1.590073 | 4.080099 | 8.027203 | -0.346006 | 0.920824 | 0.960460 | 2.148447 | 2.114285 | 2.148447 | 9.873200 | 2575867.119601 | 555185 |
| 24 | 30 | 2392416 | 0.489028 | 0.900524 | 0.948959 | -0.080026 | 0.150551 | 2.261274 | 12.508714 | 122.085581 | 95.252555 | 22.746584 | 22.746584 | 44.139730 | -3.722320 | 4.953089 | 9.611466 | -0.810538 | 0.886483 | 0.944344 | 2.149898 | 2.069872 | 2.149898 | 9.873200 | 5143450.891159 | 1109063 |
| 48 | 30 | 4762944 | 0.638291 | 1.390912 | 1.179369 | -0.033529 | 0.232380 | 2.600278 | 86.944348 | 123.446268 | 139.004486 | 29.648525 | 29.648525 | 54.781496 | -1.557427 | 6.464889 | 11.945157 | -0.339598 | 0.824638 | 0.912185 | 2.152861 | 2.119331 | 2.152861 | 9.873200 | 10253954.924974 | 2212895 |

### 11.4 Test - daytime_only

| pred_len | best_epoch | count | mae | mse | rmse | mbe | median_ae | p95_ae | max_ae | smape | mape_nonzero | wape | nmae_by_mean | nrmse_by_mean | nmbe_by_mean | nmae_by_max | nrmse_by_max | nmbe_by_max | r2 | pearson_r | mean_true | mean_pred | mean_abs_true | max_abs_true | sum_abs_true | nonzero_target_count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12 | 46500 | 0.382269 | 0.460279 | 0.678439 | 0.042459 | 0.176517 | 1.524007 | 6.020705 | 16.038099 | 21.282000 | 8.269168 | 8.269168 | 14.675871 | 0.918458 | 3.871780 | 6.871519 | 0.430039 | 0.918312 | 0.959576 | 4.622818 | 4.665277 | 4.622818 | 9.873200 | 214961.055930 | 46319 |
| 12 | 30 | 557357 | 0.725050 | 1.262530 | 1.123624 | -0.051524 | 0.404754 | 2.579305 | 11.348562 | 28.568600 | 65.370129 | 15.688366 | 15.688366 | 24.312574 | -1.114848 | 7.343613 | 11.380543 | -0.521853 | 0.775972 | 0.888717 | 4.621575 | 4.570051 | 4.621575 | 9.873200 | 2575867.119601 | 555185 |
| 24 | 30 | 1113407 | 0.861342 | 1.680865 | 1.296482 | -0.112540 | 0.505579 | 2.970470 | 12.499366 | 32.628436 | 95.252555 | 18.645531 | 18.645531 | 28.065043 | -2.436156 | 8.724036 | 13.131321 | -1.139850 | 0.701820 | 0.858162 | 4.619560 | 4.507021 | 4.619560 | 9.873200 | 5143450.891159 | 1109063 |
| 48 | 30 | 2221583 | 1.046426 | 2.484329 | 1.576175 | -0.074862 | 0.679459 | 3.287450 | 86.944348 | 35.908453 | 139.004486 | 22.671479 | 22.671479 | 34.148818 | -1.621935 | 10.598655 | 15.964179 | -0.758236 | 0.559356 | 0.776340 | 4.615607 | 4.540745 | 4.615607 | 9.873200 | 10253954.924974 | 2212895 |

## 12. 调参与导出工作流

除了默认 baseline，仓库仍保留 validation-only 调参与 tuned checkpoint 导出流程。

运行标准调参：

```bash
cd /home/lunalhx/projects/CiTransformer
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
cd /home/lunalhx/projects/CiTransformer

PYTHON_BIN=./.venv/bin/python \
MODE=export_tuned_best \
PRED_LENS="1 12 24 48" \
RUN_PRED_LEN1_REF=1 \
RESULTS_BASE_DIR=results/itransformer_tuned \
bash scripts/run_itransformer_experiments.sh
```

这一步不重新训练，而是加载调参 summary 中定位到的 checkpoint，并导出最终 test 指标。本文档的结果表只汇报本次新跑出的默认 `results/itransformer/` baseline，不混入历史 tuned 结果。

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

从 test split 看，随着预测步长从 1 增加到 48，误差整体增大，符合多步预测任务的预期。`pred_len=12/24/48` 的 best epoch 都达到 30，说明当前默认配置在长预测步长上可能还没有训练到明显收敛平台。后续如果要强化 vanilla iTransformer baseline，建议优先尝试更长 epoch、学习率调度或 validation-only 小规模调参。
