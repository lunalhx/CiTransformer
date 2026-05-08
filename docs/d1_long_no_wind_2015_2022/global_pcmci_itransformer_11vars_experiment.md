# D1 2015-2022 11 变量 PCMCI-Derived Masked iTransformer 实验说明

更新日期：2026-05-09

本文档记录 D1 long no-wind 2015-2022 主实验中的 11 变量 PCMCI-derived variable-level attention mask 设置。该 mask 是基于 train split 的 PCMCI 条件依赖结果，经物理先验和 Top-K 过滤后聚合成 iTransformer 变量级 attention permission；它不是 lag-aware attention mask，也不应表述为确定性的真实结构因果图。

## 1. 数据与入口

本实验固定使用：

```text
data/processed_long_no_wind_2015_2022
```

批量入口：

```bash
bash scripts/run_global_pcmci_itransformer_11vars.sh
```

脚本会读取：

```text
data/processed_long_no_wind_2015_2022/splits/train.csv
```

并输出到：

```text
results/d1_long_no_wind_2015_2022/causal_graphs/global_pcmci_11vars_train
```

除非设置 `REBUILD_CAUSAL_GRAPH=1`，脚本会先校验已有 `global_pcmci_config.json` 是否与当前 train path、变量顺序、`sample_scope`、`tau_min/tau_max`、`pc_alpha`、`alpha_level`、`fdr_method`、`freq_minutes` 完全一致；不一致时自动重建，避免复用旧的 2020-2022 图。

## 2. PCMCI 设置

主实验设置：

| 项目 | 值 |
| --- | --- |
| sample_scope | `full_train` |
| tau_min / tau_max | `1 / 12` |
| lag 单位 | 5 分钟 |
| 最大滞后 | 60 分钟 |
| pc_alpha | `0.05` |
| alpha_level | `0.05` |
| fdr_method | `fdr_bh` |
| independence test | `ParCorr(significance="analytic")` |
| topk_active / topk_other | `5 / 3` |

变量顺序固定为：

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

`solar_elevation`、周期时间特征和 `day_night_label` 作为外生 temporal/solar control variables 使用。最终图是 global/static mask，不表达 regime-specific 或 time-varying dependency。

## 3. 输出语义

PCMCI 的 lag-level link 语义是：

```text
source(t-lag) -> target(t)
```

本实验将显著 lag 聚合成变量级邻接矩阵：

```text
adjacency[target, source] = 1
```

含义是 target query 可以在 iTransformer 变量级 attention 中关注 source key/value。由于 iTransformer 当前 token 是变量的整段历史表示，该 mask 不限制具体 lag。

核心输出：

| 文件 | 含义 |
| --- | --- |
| `raw_p_matrix.npy` | Tigramite 原始 p matrix |
| `effective_p_matrix.npy` | 实际用于边筛选的 p/q matrix |
| `raw_val_matrix.npy` | MCI/val matrix |
| `raw_significant_lag_edges.csv` | 显著 lag-level links，未经过物理先验 |
| `prior_allowed_lag_edges.csv` | 通过物理先验的 lag-level links |
| `prior_filtered_aggregated_edges.csv` | 物理先验后按 source-target 聚合的变量边 |
| `topk_final_edges.csv` | Top-K 后实际用于 mask 的变量边 |
| `global_causal_adjacency.csv` | `target x source` 邻接矩阵 |
| `global_causal_mask.npy` | binary mask，`1=allowed, 0=blocked` |
| `global_additive_attention_mask.npy` | additive mask，`0.0=allowed, -1e9=blocked` |

`global_causal_edges.csv` 保留为 `topk_final_edges.csv` 的兼容别名。

## 4. 推荐报告口径

推荐表述：

```text
prior-constrained PCMCI-derived variable-level attention mask
```

或：

```text
Granger-style temporal dependency graph derived from train-only PCMCI
```

不建议表述为“真实因果图”或“已证明的物理因果结构”。结果解释应强调：该图只基于 train split、是全局静态图、使用 ParCorr 线性条件相关检验，并经过物理先验和 Top-K 工程约束。

## 5. 后续敏感性分析

主实验完成后建议补充：

- `tau_max=6/12/24` 的边稳定性对比；
- 保留 vs 去掉 `day_night_label`；
- `full_train` vs `daytime` scope；
- 如计算资源允许，补充非线性 independence test 版本。
