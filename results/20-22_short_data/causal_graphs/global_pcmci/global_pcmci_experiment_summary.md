# Global PCMCI 因果图实验结果总结

## 1. 实验目的

本实验对应论文第一阶段的 **Global PCMCI Causal Graph** 消融实验。目标是在不使用 validation、calibration、test 的前提下，仅基于训练集 daytime-only 数据挖掘全局变量级因果结构，并将其转换为可注入 iTransformer 变量级 attention 的 causal mask。

该阶段用于对比：

- vanilla iTransformer：不使用因果约束；
- Global-PCMCI iTransformer：使用本实验得到的全局因果 mask；
- 后续 Situation-aware PCMCI iTransformer：进一步按运行态势构建专属因果图。

## 2. 数据与运行设置

| 项目 | 数值 |
|---|---:|
| 数据文件 | `/home/lunalhx/projects/CiTransformer/data/processed_selected_2020_2022/splits/train.csv` |
| 时间列 | `timestamp` |
| 原始 train 样本数 | 103619 |
| daytime-only 样本数 | 47460 |
| PCMCI 实际使用样本数 | 47443 |
| 采样间隔 | 5 分钟 |
| daytime 后连续 segment 总数 | 356 |
| 保留连续 segment 数 | 355 |
| 过短丢弃 segment 数 | 1 |
| 保留 segment 长度范围 | 34 - 152 |
| 保留 segment 中位长度 | 136 |
| 缺失值丢弃行数 | 0 |

本实验严格只使用训练集。daytime-only 过滤使用 `day_night_label == 1`。过滤后再按 5 分钟时间戳检查连续性，时间断点前后的数据不会被拼接成同一条时间序列。

## 3. PCMCI 参数

| 参数 | 数值 |
|---|---:|
| 条件独立检验 | ParCorr |
| `tau_min` | 1 |
| `tau_max` | 12 |
| 最大滞后时间 | 60 分钟 |
| `pc_alpha` | 0.05 |
| `alpha_level` | 0.05 |
| `fdr_method` | `fdr_bh` |
| `topk_active` | 5 |
| `topk_other` | 3 |
| 运行时间 | 210.45 秒 |

变量列表：

```text
Active_Pow
Radiation_Global_Tilted
Radiation_Diffuse_Tilted
Weather_T
Weather_R
solar_elevation
sin_day_of_year
cos_day_of_year
```

## 4. 显著边筛选结果

| 阶段 | 数量 |
|---|---:|
| PCMCI 显著 lag-level 边 | 157 |
| 物理先验过滤掉的 lag-level 边 | 102 |
| 聚合后的变量级边，Top-K 前 | 13 |
| 聚合后的变量级边，Top-K 后 | 12 |

物理先验过滤使图明显稀疏化，避免了 `Active_Pow -> 气象/辐照度/天文变量`、变量指向外生时间变量、时间变量互相指向等不符合物理机制的方向。

## 5. 最终变量级因果边

下表为 Top-K 后保留的聚合边。`significant_lags` 的单位是 5 分钟采样步长，例如 lag=12 表示 60 分钟前。

| source | target | significant_lags | strongest_lag | 最强滞后时间 | max_abs_mci | MCI 符号 |
|---|---|---:|---:|---:|---:|---|
| Active_Pow | Active_Pow | 1,3,6,9 | 1 | 5 分钟 | 0.6017 | positive |
| Radiation_Global_Tilted | Active_Pow | 2,4,12 | 2 | 10 分钟 | 0.0484 | negative |
| Radiation_Diffuse_Tilted | Active_Pow | 1,2,4,6 | 4 | 20 分钟 | 0.0285 | negative |
| solar_elevation | Active_Pow | 1,2,7,9,12 | 1 | 5 分钟 | 0.0252 | positive |
| Radiation_Diffuse_Tilted | Radiation_Diffuse_Tilted | 1,2,3,4,5,6,7,8,11 | 1 | 5 分钟 | 0.8583 | positive |
| Weather_T | Radiation_Diffuse_Tilted | 6,8,12 | 12 | 60 分钟 | 0.0268 | negative |
| Weather_R | Radiation_Diffuse_Tilted | 12 | 12 | 60 分钟 | 0.0192 | negative |
| Radiation_Global_Tilted | Radiation_Global_Tilted | 1,2,3,4,6,12 | 1 | 5 分钟 | 0.7406 | positive |
| solar_elevation | Radiation_Global_Tilted | 1,2,7,9,11 | 1 | 5 分钟 | 0.0284 | positive |
| Weather_T | Radiation_Global_Tilted | 3 | 3 | 15 分钟 | 0.0142 | positive |
| Weather_R | Weather_R | 1,2,3,4,5,7,8 | 1 | 5 分钟 | 0.7513 | positive |
| Weather_T | Weather_T | 1,2,3,4,6,8 | 1 | 5 分钟 | 0.7170 | positive |

## 6. Active_Pow 的主要父变量

最终 `Active_Pow` 的父变量为：

| 父变量 | significant_lags | strongest_lag | 最强滞后时间 | max_abs_mci | 解释 |
|---|---:|---:|---:|---:|---|
| Active_Pow | 1,3,6,9 | 1 | 5 分钟 | 0.6017 | 功率自身短时惯性最强，是最主要预测信息来源。 |
| Radiation_Global_Tilted | 2,4,12 | 2 | 10 分钟 | 0.0484 | 总倾斜面辐照度对功率存在显著滞后影响。 |
| Radiation_Diffuse_Tilted | 1,2,4,6 | 4 | 20 分钟 | 0.0285 | 散射辐照度对功率存在显著滞后影响。 |
| solar_elevation | 1,2,7,9,12 | 1 | 5 分钟 | 0.0252 | 太阳高度角对功率有稳定外生驱动作用。 |

结论上看，`Active_Pow` 的全局父变量符合光伏短时预测的基本物理直觉：自身历史功率最强，其次是辐照度变量和太阳高度角。`Weather_T` 与 `Weather_R` 没有直接进入 `Active_Pow` 的 Top-K 父变量，但它们通过影响辐照度变量进入图结构。

需要注意，辐照度到功率的 MCI 符号为负并不等同于“辐照度升高导致功率下降”的边际物理解释。PCMCI 的 MCI 是在控制其他变量和滞后项后的条件关联，受到 `Active_Pow` 强自回归、辐照度变量共线性、太阳高度角控制等因素影响。因此，本实验更适合将边的存在性和强度用于构造 attention mask，而不是直接将符号解释为简单单变量响应方向。

## 7. 变量级邻接矩阵与 attention mask

邻接矩阵方向采用：

```text
adjacency[target, source] = 1
```

含义是：在 iTransformer 变量级 attention 中，`target` 变量作为 query 时允许关注 `source` 变量的 key/value。

最终邻接关系为：

| target | 允许关注的 source |
|---|---|
| Active_Pow | Active_Pow, Radiation_Global_Tilted, Radiation_Diffuse_Tilted, solar_elevation |
| Radiation_Global_Tilted | Radiation_Global_Tilted, Weather_T, solar_elevation |
| Radiation_Diffuse_Tilted | Radiation_Diffuse_Tilted, Weather_T, Weather_R |
| Weather_T | Weather_T |
| Weather_R | Weather_R |
| solar_elevation | solar_elevation |
| sin_day_of_year | sin_day_of_year |
| cos_day_of_year | cos_day_of_year |

mask 统计：

| 项目 | 数值 |
|---|---:|
| mask 形状 | 8 x 8 |
| 允许 attention 的位置数 | 15 |
| 总位置数 | 64 |
| mask 密度 | 23.44% |
| additive mask 允许值 | 0.0 |
| additive mask 屏蔽值 | -1e9 |

对角线全部为 1，因此每个变量至少可以关注自身。

## 8. 图结构解读

本次全局因果图呈现出三个特点：

1. **功率预测主要依赖自身滞后和辐照度/太阳高度角。**  
   `Active_Pow(t)` 保留了来自 `Active_Pow(t-k)`、`Radiation_Global_Tilted(t-k)`、`Radiation_Diffuse_Tilted(t-k)` 和 `solar_elevation(t-k)` 的输入。

2. **辐照度变量具有很强自回归结构。**  
   两个辐照度变量均保留了显著自身滞后，其中 `Radiation_Diffuse_Tilted` 的最大绝对 MCI 为 0.8583，`Radiation_Global_Tilted` 为 0.7406。

3. **气象变量主要作为辐照度的上游调制因素。**  
   `Weather_T` 指向两个辐照度变量，`Weather_R` 指向散射辐照度；二者没有直接指向功率，说明在当前全局图和控制条件下，它们对功率的直接增益弱于辐照度与太阳高度角。

## 9. 输出文件说明

| 文件 | 说明 |
|---|---|
| `raw_p_matrix.npy` | PCMCI 原始 p 矩阵 |
| `raw_val_matrix.npy` | PCMCI 原始 MCI/test statistic 矩阵 |
| `global_causal_edges.csv` | Top-K 后聚合变量级因果边 |
| `global_causal_adjacency.csv` | 变量级邻接矩阵，方向为 `target x source` |
| `global_causal_mask.npy` | binary attention mask，1 表示允许，0 表示屏蔽 |
| `global_additive_attention_mask.npy` | additive attention mask，0.0 表示允许，-1e9 表示屏蔽 |
| `global_causal_graph.png` | 全局因果图可视化 |
| `global_pcmci_config.json` | 参数、样本、segment、物理先验、运行时间等配置记录 |
| `active_pow_parents.csv` | `Active_Pow` 父变量单独结果 |

## 10. 实验结论

本次 Global PCMCI 实验得到了一张较稀疏、物理方向基本合理的变量级因果图。图中 `Active_Pow` 的主要信息来源为自身历史、总/散射辐照度和太阳高度角；气象变量更多表现为辐照度变量的上游影响因素。最终 mask 密度为 23.44%，相比全连接变量 attention 明显减少了可关注关系，有利于后续验证全局因果约束是否能改善 iTransformer 的泛化表现。

下一步建议将 `global_causal_mask.npy` 注入 iTransformer 的变量级 attention，并与 vanilla iTransformer 在相同数据划分、相同预测步长和相同训练设置下对比 MAE、RMSE、R2 等指标。如果 Global-PCMCI iTransformer 相比 vanilla 有稳定提升，则可以进一步推进到 “态势划分 + 态势专属 PCMCI 图” 的第二阶段实验。
