# 11 变量 Global PCMCI Masked iTransformer 实验文档

更新日期：2026-05-07

本文档记录本次使用 11 变量全局 PCMCI 因果图约束 iTransformer 变量级 attention 的实验操作与结果。实验遵循 `docs/实验协议.md` 的数据划分、连续时间片段、标准化、训练验证测试流程和指标口径。

## 1. 实验目的

本实验用于验证：在 tuned iTransformer 最优共享超参数基础上，将 train split 生成的 11 变量 Global PCMCI 因果图注入 iTransformer 变量级 attention，是否能改善光伏功率预测表现。

与 vanilla/tuned iTransformer 的区别是：

1. iTransformer 输入仍使用默认 11 个特征。
2. 因果图同样使用这 11 个特征生成，不再使用旧版 8 变量图。
3. attention mask 直接使用 `11 x 11` 邻接矩阵，不做 8 到 11 的扩展。
4. mask 方向为 `adjacency[target, source] = 1`，表示 target query 可以关注 source key/value。
5. 模型从头训练，不加载 tuned checkpoint 直接评估。

## 2. 因果图生成设置

因果图输出目录：

```text
results/causal_graphs/global_pcmci_11vars_train
```

生成命令：

```bash
cd "${PROJECT_ROOT:-/path/to/CiTransformer}"
.venv/bin/python -u causal_algo/run_global_pcmci.py \
  --sample_scope full_train \
  --output_dir results/causal_graphs/global_pcmci_11vars_train
```

核心设置如下：

| 项目 | 数值 |
| --- | ---: |
| 数据来源 | `train.csv` only |
| sample_scope | `full_train` |
| row_filter | no daytime filtering |
| raw train samples | 103619 |
| PCMCI used rows | 103619 |
| continuous segments kept | 20 |
| tau_min / tau_max | 1 / 12 |
| pc_alpha | 0.05 |
| alpha_level | 0.05 |
| fdr_method | `fdr_bh` |
| topk_active / topk_other | 5 / 3 |
| raw significant lag edges | 403 |
| physical-prior filtered lag edges | 309 |
| aggregated edges before / after Top-K | 22 / 15 |
| runtime_seconds | 876.617555 |

11 个 PCMCI 变量：

```text
Active_Pow
Radiation_Global_Tilted
Radiation_Diffuse_Tilted
Weather_T
Weather_R
solar_elevation
sin_time_of_day
cos_time_of_day
sin_day_of_year
cos_day_of_year
day_night_label
```

时间/标签变量包括 `solar_elevation`、`sin_time_of_day`、`cos_time_of_day`、`sin_day_of_year`、`cos_day_of_year`、`day_night_label`。本实验中它们只作为外生 source：允许指向功率、辐照度、气象等非时间变量，但不允许其他变量指向这些时间/标签变量。最终导出的邻接矩阵中，时间/标签变量没有非自环 incoming edge。

mask 统计：

| 项目 | 数值 |
| --- | ---: |
| mask shape | 11 x 11 |
| allowed positions | 21 |
| total positions | 121 |
| mask density | 17.3554% |
| additive allowed value | 0.0 |
| additive blocked value | -1e9 |

`Active_Pow` 的最终父变量：

| source | significant_lags | strongest_lag | max_abs_mci | sign |
| --- | --- | ---: | ---: | --- |
| Active_Pow | 1,2,3,4,5,6,9,10,12 | 1 | 0.630773 | positive |
| Radiation_Global_Tilted | 1,2,3,4,6,12 | 3 | 0.065448 | positive |
| day_night_label | 1,3,9 | 1 | 0.029858 | positive |
| Radiation_Diffuse_Tilted | 2,4,10 | 4 | 0.018287 | negative |
| solar_elevation | 7,8,12 | 7 | 0.011872 | negative |

## 3. Masked iTransformer 设置

批量训练入口：

```bash
cd "${PROJECT_ROOT:-/path/to/CiTransformer}"
bash scripts/run_global_pcmci_itransformer_11vars.sh
```

如果需要强制重建因果图后再训练：

```bash
REBUILD_CAUSAL_GRAPH=1 bash scripts/run_global_pcmci_itransformer_11vars.sh
```

本次模型使用 tuned iTransformer 的最优共享超参数：

| 参数 | 值 |
| --- | ---: |
| seq_len | 96 |
| pred_len | 1, 12, 24, 48 |
| batch_size | 128 |
| d_model | 192 |
| n_heads | 4 |
| e_layers | 2 |
| d_ff | 384 |
| dropout | 0.1 |
| learning_rate | 0.001 |
| weight_decay | 0.00001 |
| factor | 5 |
| activation | gelu |
| epochs | 120 |
| patience | 20 |
| min_delta | 0.00001 |
| grad_clip | 1.0 |
| seed | 42 |

输出目录：

| pred_len | results | checkpoint |
| ---: | --- | --- |
| 1 | `results/itransformer_global_pcmci_11vars/pred_len_1` | `checkpoints/itransformer_global_pcmci_11vars/pred_len_1/best_model.pth` |
| 12 | `results/itransformer_global_pcmci_11vars/pred_len_12` | `checkpoints/itransformer_global_pcmci_11vars/pred_len_12/best_model.pth` |
| 24 | `results/itransformer_global_pcmci_11vars/pred_len_24` | `checkpoints/itransformer_global_pcmci_11vars/pred_len_24/best_model.pth` |
| 48 | `results/itransformer_global_pcmci_11vars/pred_len_48` | `checkpoints/itransformer_global_pcmci_11vars/pred_len_48/best_model.pth` |

## 4. 训练摘要

| pred_len | epochs_cfg | patience | total_epochs | best_epoch | best_validation_loss | trainable params |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 120 | 20 | 36 | 16 | 0.026184 | 613249 |
| 12 | 120 | 20 | 73 | 53 | 0.063138 | 615372 |
| 24 | 120 | 20 | 47 | 27 | 0.085646 | 617688 |
| 48 | 120 | 20 | 78 | 58 | 0.108285 | 622320 |

`history.train_loss` 和 `history.validation_loss` 是标准化目标空间中的 MSELoss；下表中的 validation/test 指标均为 inverse transform 后的原始功率尺度。

## 5. 核心结果

| pred_len | val daytime MAE | val daytime RMSE | test all MAE | test all RMSE | test daytime MAE | test daytime RMSE | test daytime R2 | test daytime Pearson r |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.373235 | 0.663564 | 0.205410 | 0.467686 | 0.387540 | 0.676417 | 0.918798 | 0.959240 |
| 12 | 0.636049 | 1.019579 | 0.381777 | 0.763299 | 0.692715 | 1.077413 | 0.794021 | 0.896672 |
| 24 | 0.771266 | 1.166572 | 0.483833 | 0.899077 | 0.839050 | 1.244879 | 0.725084 | 0.861014 |
| 48 | 0.875864 | 1.295323 | 0.583403 | 1.067931 | 0.996857 | 1.448431 | 0.627888 | 0.807277 |

## 6. 与 tuned iTransformer 对比

下表以 `results/itransformer_tuned/pred_len_*` 为对照。`delta = masked - tuned`，因此负数表示 masked iTransformer 更好。

| pred_len | masked test daytime MAE | tuned test daytime MAE | delta MAE | masked test daytime RMSE | tuned test daytime RMSE | delta RMSE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.387540 | 0.375503 | 0.012038 | 0.676417 | 0.681652 | -0.005235 |
| 12 | 0.692715 | 0.699056 | -0.006341 | 1.077413 | 1.094358 | -0.016946 |
| 24 | 0.839050 | 0.830344 | 0.008707 | 1.244879 | 1.258302 | -0.013423 |
| 48 | 0.996857 | 1.012052 | -0.015196 | 1.448431 | 1.507176 | -0.058745 |

从 test daytime RMSE 看，11 变量 Global PCMCI mask 在四个预测步长上均优于 tuned iTransformer，其中 `pred_len=48` 改善最大，RMSE 降低约 0.058745。MAE 上表现更混合：`pred_len=12` 和 `48` 有改善，`pred_len=1` 和 `24` 略差于 tuned iTransformer。

## 7. 简要结论

本次 11 变量 Global PCMCI masked iTransformer 在长预测步长上更有价值，尤其是 `pred_len=48` 的 test daytime RMSE 明显低于 tuned iTransformer。短预测步长下，mask 对 RMSE 有轻微收益，但 MAE 不一定同步改善。

这说明全局因果 mask 可以在一定程度上提升变量级 attention 的泛化表现，但它不是对所有指标都单调改进。后续如果继续推进，可以重点比较：

- 11 变量 full-train Global PCMCI mask；
- 旧 8 变量 daytime-only Global PCMCI mask；
- 后续按运行态势划分的 situation-aware PCMCI mask。
