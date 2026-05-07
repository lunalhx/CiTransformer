# LSTM Baseline 实验文档

更新日期：2026-05-07

本文档是 LSTM baseline 的单一权威说明，已经合并原来的实现工作记录和实验流程说明。后续关于 LSTM baseline 的实现、实验协议、运行方式和结果汇总，以本文档为准。

本文档从属于统一实验协议：`docs/实验协议.md`。除非本文件特别说明，数据来源、split、`seq_len / pred_len`、指标口径和结果导出结构都遵循统一协议。

## 1. 实验目的

LSTM baseline 用作光伏功率预测任务中的神经网络序列建模对照组。它比 Persistence baseline 更灵活，可以利用输入窗口内的多变量历史信息，但结构仍然保持朴素、可解释，适合作为后续 vanilla iTransformer 和 Causal-iTransformer 的基础对比模型。

本次实验已经按 `pred_len=1/12/24/48` 全部完成，并生成了对应的 metrics、predictions、plot 和 checkpoint。

## 2. 代码位置

核心文件如下：

- `models/baseline/lstm.py`：LSTM baseline 模型定义
- `scripts/run_lstm.py`：单次训练、验证、测试入口
- `scripts/run_lstm_experiments.sh`：批量运行 `pred_len=1/12/24/48`
- `utils/datasets.py`：连续时间片段数据集、滑窗与标准化
- `results/lstm/pred_len_*/`：本次实验输出目录
- `checkpoints/lstm/pred_len_*/best_model.pth`：每个预测步长的最优 checkpoint

原来的两份文档已经合并到本文档，并删除旧文件：

- `docs/lstm_baseline_work_summary.md`
- `docs/lstm_experiment_protocol.md`

## 3. 模型定义

模型是一个 sequence-to-vector LSTM baseline：

- 输入形状：`[batch, seq_len, feature_dim]`
- 主体结构：多层 `nn.LSTM`
- 输出头：`dropout + linear`
- 输出形状：`[batch, pred_len]`
- 预测目标：未来 `pred_len` 步的 `Active_Pow`

每个 `pred_len` 单独训练一个模型，而不是共享同一个输出头。

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
| calibration | 为协议一致性读取并统计；当前 LSTM baseline 不使用它 |
| test | 训练结束后加载最佳 checkpoint，只评估一次；导出 test 指标、预测 CSV 和图像 |

论文中可以明确写：

```text
The calibration split is reserved for protocol consistency but is unused in the current LSTM baseline.
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

这个约束可以避免窗口跨越清洗后的时间断点，保证 LSTM 不会把不连续数据误认为连续时序。

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

本次运行命令：

```bash
cd "${PROJECT_ROOT:-/path/to/CiTransformer}"
bash scripts/run_lstm_experiments.sh
```

批量脚本默认配置：

| 参数 | 值 |
| --- | --- |
| `DATA_DIR` | `data/processed_selected_2020_2022` |
| `SEQ_LEN` | `96` |
| `PRED_LENS` | `1 12 24 48` |
| `BATCH_SIZE` | `256` |
| `HIDDEN_SIZE` | `128` |
| `NUM_LAYERS` | `2` |
| `DROPOUT` | `0.1` |
| `LEARNING_RATE` | `1e-3` |
| `WEIGHT_DECAY` | `1e-5` |
| `GRAD_CLIP` | `1.0` |
| `EPOCHS` | `30` |
| `PATIENCE` | `8` |
| `MIN_DELTA` | `1e-5` |
| `NUM_WORKERS` | `0` |
| `SEED` | `42` |
| `DEVICE` | `auto` |

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

| pred_len | metrics | predictions | plot | checkpoint |
| ---: | --- | --- | --- | --- |
| 1 | `results/lstm/pred_len_1/metrics.json` | `results/lstm/pred_len_1/predictions.csv` | `results/lstm/pred_len_1/pred_plot.png` | `checkpoints/lstm/pred_len_1/best_model.pth` |
| 12 | `results/lstm/pred_len_12/metrics.json` | `results/lstm/pred_len_12/predictions.csv` | `results/lstm/pred_len_12/pred_plot.png` | `checkpoints/lstm/pred_len_12/best_model.pth` |
| 24 | `results/lstm/pred_len_24/metrics.json` | `results/lstm/pred_len_24/predictions.csv` | `results/lstm/pred_len_24/pred_plot.png` | `checkpoints/lstm/pred_len_24/best_model.pth` |
| 48 | `results/lstm/pred_len_48/metrics.json` | `results/lstm/pred_len_48/predictions.csv` | `results/lstm/pred_len_48/pred_plot.png` | `checkpoints/lstm/pred_len_48/best_model.pth` |

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
| 1 | 23 | 0.021505 | 0.180015 | 0.432572 | 0.358423 | 0.631753 |
| 12 | 6 | 0.049147 | 0.320193 | 0.669547 | 0.656714 | 0.979247 |
| 24 | 3 | 0.060102 | 0.363343 | 0.749933 | 0.737715 | 1.093978 |
| 48 | 1 | 0.077889 | 0.451145 | 0.854190 | 0.871656 | 1.238622 |

从本次结果看，`pred_len=12/24/48` 的 best epoch 较早，尤其 `pred_len=48` 在第 1 个 epoch 达到当前最佳验证损失。这说明这组固定超参数可以作为 baseline，但长预测步长可能仍有调参空间。

## 11. 完整聚合指标

下面罗列本次四组 LSTM 实验的完整聚合指标。列名与 `metrics.json` 保持一致。

### 11.1 Validation - all_timestamps

| pred_len | best_epoch | count | mae | mse | rmse | mbe | median_ae | p95_ae | max_ae | smape | mape_nonzero | wape | nmae_by_mean | nrmse_by_mean | nmbe_by_mean | nmae_by_max | nrmse_by_max | nmbe_by_max | r2 | pearson_r | mean_true | mean_pred | mean_abs_true | max_abs_true | sum_abs_true | nonzero_target_count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 23 | 46109 | 0.165244 | 0.170442 | 0.412846 | 0.024252 | 0.028784 | 0.818245 | 5.565958 | 114.791245 | 15.048012 | 7.713297 | 7.713297 | 19.270941 | 1.132054 | 1.746980 | 4.364663 | 0.256398 | 0.978286 | 0.989121 | 2.142325 | 2.166577 | 2.142325 | 9.458833 | 98780.463220 | 20858 |
| 12 | 6 | 551856 | 0.283702 | 0.389517 | 0.624113 | 0.009623 | 0.039662 | 1.410724 | 7.246202 | 118.791522 | 34.654926 | 13.221129 | 13.221129 | 29.085064 | 0.448432 | 2.999331 | 6.598206 | 0.101731 | 0.950416 | 0.974899 | 2.145821 | 2.155443 | 2.145821 | 9.458833 | 1184183.965379 | 250032 |
| 24 | 3 | 1100544 | 0.325133 | 0.476349 | 0.690180 | 0.036038 | 0.047557 | 1.596384 | 6.593207 | 119.531502 | 41.242538 | 15.125759 | 15.125759 | 32.108343 | 1.676537 | 3.437353 | 7.296673 | 0.380996 | 0.939412 | 0.969443 | 2.149535 | 2.185573 | 2.149535 | 9.458833 | 2365657.859945 | 499422 |
| 48 | 1 | 2188416 | 0.411430 | 0.617321 | 0.785698 | -0.008205 | 0.102111 | 1.857621 | 6.924411 | 120.987912 | 49.882356 | 19.077586 | 19.077586 | 36.432009 | -0.380445 | 4.349691 | 8.306500 | -0.086741 | 0.921592 | 0.960042 | 2.156614 | 2.148410 | 2.156614 | 9.458833 | 4719569.550962 | 996232 |

### 11.2 Validation - daytime_only

| pred_len | best_epoch | count | mae | mse | rmse | mbe | median_ae | p95_ae | max_ae | smape | mape_nonzero | wape | nmae_by_mean | nrmse_by_mean | nmbe_by_mean | nmae_by_max | nrmse_by_max | nmbe_by_max | r2 | pearson_r | mean_true | mean_pred | mean_abs_true | max_abs_true | sum_abs_true | nonzero_target_count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 23 | 20898 | 0.331270 | 0.370733 | 0.608878 | 0.021602 | 0.149807 | 1.353936 | 5.565958 | 12.019361 | 15.048012 | 7.008355 | 7.008355 | 12.881434 | 0.457008 | 3.502232 | 6.437141 | 0.228377 | 0.927339 | 0.963044 | 4.726790 | 4.748392 | 4.726790 | 9.458833 | 98780.463220 | 20858 |
| 12 | 6 | 250512 | 0.589986 | 0.850558 | 0.922257 | 0.000339 | 0.330587 | 2.030873 | 7.246202 | 21.198441 | 34.654926 | 12.481038 | 12.481038 | 19.510187 | 0.007162 | 6.237403 | 9.750222 | 0.003579 | 0.833352 | 0.912945 | 4.727055 | 4.727393 | 4.727055 | 9.458833 | 1184183.965379 | 250032 |
| 24 | 3 | 500382 | 0.663961 | 1.029485 | 1.014635 | 0.064301 | 0.377928 | 2.262048 | 6.593207 | 23.078822 | 41.242538 | 14.044056 | 14.044056 | 21.461486 | 1.360082 | 7.019485 | 10.726857 | 0.679795 | 0.798259 | 0.894231 | 4.727704 | 4.792004 | 4.727704 | 9.458833 | 2365657.859945 | 499422 |
| 48 | 1 | 998152 | 0.797673 | 1.318069 | 1.148072 | -0.080489 | 0.513650 | 2.506945 | 6.924411 | 26.798959 | 49.882356 | 16.870153 | 16.870153 | 24.280824 | -1.702285 | 8.433099 | 12.137565 | -0.850943 | 0.741651 | 0.862578 | 4.728307 | 4.647818 | 4.728307 | 9.458833 | 4719569.550962 | 996232 |

### 11.3 Test - all_timestamps

| pred_len | best_epoch | count | mae | mse | rmse | mbe | median_ae | p95_ae | max_ae | smape | mape_nonzero | wape | nmae_by_mean | nrmse_by_mean | nmbe_by_mean | nmae_by_max | nrmse_by_max | nmbe_by_max | r2 | pearson_r | mean_true | mean_pred | mean_abs_true | max_abs_true | sum_abs_true | nonzero_target_count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 23 | 100121 | 0.180015 | 0.187119 | 0.432572 | 0.011957 | 0.029316 | 0.880817 | 5.474617 | 113.313177 | 23.321710 | 8.384442 | 8.384442 | 20.147640 | 0.556914 | 1.823269 | 4.381278 | 0.121106 | 0.976411 | 0.988150 | 2.147013 | 2.158970 | 2.147013 | 9.873200 | 214961.055930 | 46319 |
| 12 | 6 | 1198944 | 0.320193 | 0.448294 | 0.669547 | -0.003307 | 0.044973 | 1.568229 | 6.517616 | 118.118329 | 79.572050 | 14.903455 | 14.903455 | 31.164251 | -0.153922 | 3.243049 | 6.781461 | -0.033494 | 0.943492 | 0.971398 | 2.148447 | 2.145140 | 2.148447 | 9.873200 | 2575867.119601 | 555185 |
| 24 | 3 | 2392416 | 0.363343 | 0.562400 | 0.749933 | 0.015100 | 0.051447 | 1.829433 | 6.187620 | 119.232512 | 72.820780 | 16.900493 | 16.900493 | 34.882266 | 0.702378 | 3.680097 | 7.595645 | 0.152943 | 0.929106 | 0.963957 | 2.149898 | 2.164999 | 2.149898 | 9.873200 | 5143450.891159 | 1109063 |
| 48 | 1 | 4762944 | 0.451145 | 0.729641 | 0.854190 | -0.001994 | 0.111409 | 2.050605 | 6.326900 | 120.707454 | 111.409148 | 20.955621 | 20.955621 | 39.676986 | -0.092634 | 4.569393 | 8.651604 | -0.020199 | 0.908009 | 0.952921 | 2.152861 | 2.150866 | 2.152861 | 9.873200 | 10253954.924974 | 2212895 |

### 11.4 Test - daytime_only

| pred_len | best_epoch | count | mae | mse | rmse | mbe | median_ae | p95_ae | max_ae | smape | mape_nonzero | wape | nmae_by_mean | nrmse_by_mean | nmbe_by_mean | nmae_by_max | nrmse_by_max | nmbe_by_max | r2 | pearson_r | mean_true | mean_pred | mean_abs_true | max_abs_true | sum_abs_true | nonzero_target_count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 23 | 46500 | 0.358423 | 0.399112 | 0.631753 | -0.002275 | 0.172796 | 1.394779 | 5.474617 | 13.377878 | 23.321710 | 7.753347 | 7.753347 | 13.665967 | -0.049220 | 3.630263 | 6.398663 | -0.023046 | 0.929167 | 0.963934 | 4.622818 | 4.620543 | 4.622818 | 9.873200 | 214961.055930 | 46319 |
| 12 | 6 | 557357 | 0.656714 | 0.958925 | 0.979247 | -0.029634 | 0.403415 | 2.168067 | 6.517616 | 23.946927 | 79.572050 | 14.209745 | 14.209745 | 21.188602 | -0.641211 | 6.651480 | 9.918234 | -0.300146 | 0.829845 | 0.911132 | 4.621575 | 4.591941 | 4.621575 | 9.873200 | 2575867.119601 | 555185 |
| 24 | 3 | 1113407 | 0.737715 | 1.196788 | 1.093978 | 0.022222 | 0.428874 | 2.473783 | 6.187620 | 26.518689 | 72.820780 | 15.969383 | 15.969383 | 23.681433 | 0.481042 | 7.471896 | 11.080278 | 0.225074 | 0.787694 | 0.887792 | 4.619560 | 4.641782 | 4.619560 | 9.873200 | 5143450.891159 | 1109063 |
| 48 | 1 | 2221583 | 0.871656 | 1.534184 | 1.238622 | -0.062133 | 0.569315 | 2.720440 | 6.326900 | 30.030800 | 111.409148 | 18.884961 | 18.884961 | 26.835514 | -1.346156 | 8.828502 | 12.545294 | -0.629312 | 0.727883 | 0.853970 | 4.615607 | 4.553474 | 4.615607 | 9.873200 | 10253954.924974 | 2212895 |

## 12. 运行验证

本次已经完成以下验证：

- `pred_len=1` 运行完成，输出 `metrics.json / predictions.csv / pred_plot.png / best_model.pth`
- `pred_len=12` 运行完成，输出 `metrics.json / predictions.csv / pred_plot.png / best_model.pth`
- `pred_len=24` 运行完成，输出 `metrics.json / predictions.csv / pred_plot.png / best_model.pth`
- `pred_len=48` 运行完成，输出 `metrics.json / predictions.csv / pred_plot.png / best_model.pth`
- 四组结果目录都位于 `results/lstm/pred_len_*`
- 四组 checkpoint 都位于 `checkpoints/lstm/pred_len_*`
- `metrics.json` 中的 `config.pred_len` 与目录名一致

## 13. 论文写法建议

可以在方法或实验设置中写：

```text
The LSTM baseline is trained only on the training split, with the validation split used for early stopping and checkpoint selection. Feature and target scalers are fit only on the training split. Sliding windows are constructed only within timestamp-continuous segments to avoid crossing temporal gaps. The model uses the final hidden state of a multi-layer LSTM and a linear projection head to predict future Active_Pow values. Evaluation is performed after inverse transformation on the original power scale, and results are reported for both all timestamps and daytime-only timestamps.
```

结果分析时建议同时汇报 `all_timestamps` 和 `daytime_only`，但正文重点关注 `daytime_only`。原因是夜间光伏功率大量接近 0，`all_timestamps` 会让整体误差看起来偏乐观，而白天时段更能反映实际发电预测难度。

需要注意：当前 best checkpoint 的选择依据是标准化目标空间上的全时段 validation MSE，并不是 daytime-only validation MAE/RMSE。如果论文主线更强调白天预测质量，后续可以考虑按 daytime validation 指标选择 checkpoint。

## 14. 补充实验：lr=3e-4, epochs=60, patience=12

为避免覆盖第 7-13 节记录的默认参数实验，当前补充实验单独输出到新的结果目录：

```bash
cd "${PROJECT_ROOT:-/path/to/CiTransformer}"
LEARNING_RATE=3e-4 \
EPOCHS=60 \
PATIENCE=12 \
RESULTS_BASE_DIR=results/lstm_lr3e4 \
CHECKPOINT_BASE_DIR=checkpoints/lstm_lr3e4 \
bash scripts/run_lstm_experiments.sh
```

本轮实验与默认 LSTM baseline 保持相同的数据、split、特征、模型宽度和 batch size，只调整训练超参数：

| 参数 | 默认实验 | 补充实验 |
| --- | ---: | ---: |
| `LEARNING_RATE` | `1e-3` | `3e-4` |
| `EPOCHS` | `30` | `60` |
| `PATIENCE` | `8` | `12` |
| `SEQ_LEN` | `96` | `96` |
| `BATCH_SIZE` | `256` | `256` |
| `HIDDEN_SIZE` | `128` | `128` |
| `NUM_LAYERS` | `2` | `2` |
| `DROPOUT` | `0.1` | `0.1` |
| `WEIGHT_DECAY` | `1e-5` | `1e-5` |
| `SEED` | `42` | `42` |

输出文件如下：

| pred_len | metrics | predictions | plot | checkpoint |
| ---: | --- | --- | --- | --- |
| 1 | `results/lstm_lr3e4/pred_len_1/metrics.json` | `results/lstm_lr3e4/pred_len_1/predictions.csv` | `results/lstm_lr3e4/pred_len_1/pred_plot.png` | `checkpoints/lstm_lr3e4/pred_len_1/best_model.pth` |
| 12 | `results/lstm_lr3e4/pred_len_12/metrics.json` | `results/lstm_lr3e4/pred_len_12/predictions.csv` | `results/lstm_lr3e4/pred_len_12/pred_plot.png` | `checkpoints/lstm_lr3e4/pred_len_12/best_model.pth` |
| 24 | `results/lstm_lr3e4/pred_len_24/metrics.json` | `results/lstm_lr3e4/pred_len_24/predictions.csv` | `results/lstm_lr3e4/pred_len_24/pred_plot.png` | `checkpoints/lstm_lr3e4/pred_len_24/best_model.pth` |
| 48 | `results/lstm_lr3e4/pred_len_48/metrics.json` | `results/lstm_lr3e4/pred_len_48/predictions.csv` | `results/lstm_lr3e4/pred_len_48/pred_plot.png` | `checkpoints/lstm_lr3e4/pred_len_48/best_model.pth` |

核心结果如下。完整逐 horizon 和完整聚合指标以各 `metrics.json` 为准。

| pred_len | best_epoch | total_epochs | best_validation_loss | val all MAE | val all RMSE | val daytime MAE | val daytime RMSE | test all MAE | test all RMSE | test daytime MAE | test daytime RMSE | test all R2 | test daytime R2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 20 | 32 | 0.021949 | 0.162750 | 0.417083 | 0.334435 | 0.614583 | 0.177176 | 0.434688 | 0.360306 | 0.634857 | 0.976180 | 0.928470 |
| 12 | 7 | 19 | 0.049505 | 0.281503 | 0.626384 | 0.582932 | 0.924415 | 0.313850 | 0.667406 | 0.641671 | 0.974316 | 0.943852 | 0.831555 |
| 24 | 6 | 18 | 0.060558 | 0.333808 | 0.692793 | 0.677195 | 1.016859 | 0.367143 | 0.743417 | 0.739750 | 1.082560 | 0.930333 | 0.792102 |
| 48 | 4 | 16 | 0.076215 | 0.390428 | 0.777209 | 0.783919 | 1.140289 | 0.435604 | 0.857477 | 0.868246 | 1.247893 | 0.907300 | 0.723794 |

与默认实验相比，本轮 `lr=3e-4` 在 `pred_len=12` 的 test 指标上略有改善，`pred_len=24` 的 test daytime RMSE 也略低；但 `pred_len=1` 和 `pred_len=48` 的 daytime RMSE 没有优于默认实验。整体看，这组参数可以作为补充对照，但不应直接替代默认 LSTM baseline 的全部结果。

## 15. 当前结论

本次 LSTM baseline 已经按 `pred_len=1/12/24/48` 全部跑完，且四组输出完整。

从 test split 看，随着预测步长从 1 增加到 48，误差整体增大，符合多步预测任务的预期。当前结果可作为正式 baseline 的第一版结果，但 `pred_len=12/24/48` 的 best epoch 较早，说明这组固定超参数在长预测步长上仍可能不是最优。后续如果要写最终论文表格，建议至少做一轮小规模 validation 调参。
