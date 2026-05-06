# iTransformer 实验流程说明

本实验协议从属于统一协议：`docs/实验协议.md`。后续横向对比默认以统一协议中的数据来源、split、`seq_len/pred_len`、指标和导出结构为准；如有偏离，必须在本文件或对应实验记录里单独说明。

## 1. 当前这个实验在做什么

这个仓库里的 iTransformer baseline 目前分成两条相关但用途不同的实验链路：

- `scripts/run_itransformer_experiments.sh`
  - 用于跑 vanilla iTransformer baseline
  - 默认会分别运行 `pred_len=1/12/24/48`
  - 结果默认写入 `results/itransformer/pred_len_x/`
- `scripts/tune_itransformer.sh`
  - 用于做正式的 validation-only 小规模调参
  - 默认主目标是 `pred_len=12/24/48`
  - 会额外补一个 `pred_len=1` 参考 run
  - 结果写入 `results/tuning/itransformer/<plan>/`

为方便执行，仓库里还提供了一个便捷启动脚本：

- `scripts/run_itransformer_tuning_standard.sh`
  - 默认调用 `scripts/tune_itransformer.sh`
  - 默认使用 `PLAN=standard`

核心代码位置如下：

- 模型定义：`models/baseline/iTransformer.py`
- 单次训练 / 评估入口：`scripts/run_iTransformer.py`
- baseline 批量脚本：`scripts/run_itransformer_experiments.sh`
- 正式调参脚本：`scripts/tune_itransformer.sh`
- 调参结果汇总：`scripts/summarize_itransformer_tuning.py`

这套实现明确对应的是 **vanilla iTransformer baseline**，不是 CiTransformer，也没有加入因果图、HMM、PCMCI、regime、causal mask 等额外设计。

## 2. 模型结构是什么

当前实现使用的是一个比较标准的 vanilla iTransformer 结构，再外面包了一层适配器，以便复用现有 baseline 的训练和评估接口。

模型大致流程是：

1. 输入 `x` 的形状是 `[batch, seq_len, feature_dim]`
2. 进入 iTransformer backbone，输出未来 `pred_len` 步的多变量预测
3. 从多变量输出里取出目标列 `Active_Pow`
4. 再把这个目标从 feature-scaler 空间映射到 target-scaler 空间
5. 继续复用现有的 loss、inverse-transform 和指标计算流程

因此它和 LSTM baseline 在工程协议上是对齐的，但模型主体换成了 iTransformer。

对应实现文件：

- backbone 与 wrapper：`models/baseline/iTransformer.py`

这里有一个很重要的点：

- `Active_Pow` 既是预测目标，也是输入特征之一
- 这和当前 LSTM、Persistence baseline 的输入协议保持一致

## 3. train / validation / calibration / test 分别怎么用

当前 iTransformer 代码里这 4 个 split 的用途如下：

- `train`
  - 用来拟合 `feature_scaler` 和 `target_scaler`
  - 用来训练模型参数
  - DataLoader 使用 `shuffle=True`
- `validation`
  - 每个 epoch 后计算验证损失
  - 用来做 early stopping
  - 用来保存 best checkpoint
  - 在正式调参里，也用于不同超参数组合之间的优先级排序
- `test`
  - 在普通 baseline 训练流程里，只在训练完成后做最终导出
  - 在正式调参流程里 **不参与选参数**
  - 在最终导出阶段，会从已选好的 checkpoint 直接做推理
- `calibration`
  - 会被读取并统计样本信息
  - 当前 vanilla iTransformer baseline 不参与训练
  - 不参与 checkpoint 选择
  - 不参与最终主结果导出

因此，正式的论文级实验流程应该理解为：

1. 只在 `train` 上拟合 scaler
2. 只在 `train` 上训练模型
3. 只用 `validation` 做 checkpoint 选择和超参数选择
4. `test` 只在最终配置确定后做导出和汇报

如果你后面写论文，建议明确写一句：

`The calibration split is reserved but unused in the current vanilla iTransformer baseline.`

## 4. 样本构造和无泄漏协议是什么

这套 iTransformer baseline 没有单独实现一套数据协议，而是直接复用了当前 baseline 系列统一的数据集与标准化逻辑。

核心约束如下：

- 样本必须基于 `timestamp` 的连续 segment 构造
- 不能跨越时间断点滑窗
- 不能对整张表直接按 `iloc` 粗暴滑窗
- `feature_scaler` 和 `target_scaler` 只在 `train` 上 fit
- `validation / calibration / test` 只做 transform
- 最终所有指标都在 inverse-transform 之后的真实功率尺度上计算

这几点和 LSTM、Persistence 的协议是一致的，也是这组 baseline 之间能公平比较的前提。

当前 `prepare_datasets(...)` 已支持：

- 调参模式下不加载 `test`
- 导出 test 时再单独加载 `test`

对应实现位置：

- 数据准备：`scripts/run_lstm.py`
- iTransformer 入口：`scripts/run_iTransformer.py`

## 5. 当前默认超参数是什么

普通 baseline 批量脚本 `scripts/run_itransformer_experiments.sh` 的默认超参数如下：

- `DATA_DIR=data/processed_selected_2020_2022`
- `SEQ_LEN=96`
- `PRED_LENS=1 12 24 48`
- `BATCH_SIZE=256`
- `D_MODEL=128`
- `N_HEADS=4`
- `E_LAYERS=2`
- `D_FF=256`
- `FACTOR=5`
- `DROPOUT=0.1`
- `ACTIVATION=gelu`
- `LEARNING_RATE=1e-3`
- `WEIGHT_DECAY=1e-5`
- `GRAD_CLIP=1.0`
- `EPOCHS=30`
- `PATIENCE=8`
- `MIN_DELTA=1e-5`
- `NUM_WORKERS=0`
- `SEED=42`
- `DEVICE=auto`

默认输入特征与其他 baseline 一致：

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

## 6. loss、checkpoint 选择、最终指标分别代表什么

这里有三个容易混淆但必须分开的层次：

- 训练过程里的 `train_loss`
  - 是标准化目标空间上的 `MSELoss`
- 单个 run 内部的 best checkpoint
  - 也是按 `validation loss` 选出来的
  - 换句话说，单个训练 run 内部并不是按 daytime RMSE 选 checkpoint
- 不同超参数组合之间的最终排序
  - 由 `scripts/summarize_itransformer_tuning.py` 完成
  - 优先看 `validation daytime RMSE`
  - 再看 `validation daytime MAE`
  - 然后才看 `all timestamps` 的 RMSE / MAE

因此，正式调参时的逻辑是：

1. 每个配置内部先按 validation loss 训练出自己的 best checkpoint
2. 再把每个配置的 validation 指标汇总起来
3. 用 daytime 优先的排序规则决定哪组超参数最好

最终 `metrics.json` 里的 `validation_metrics`、`reported_metrics` 和 `test_metrics` 都是在 **反归一化后的真实功率尺度** 上计算的，这些数值可以直接用于论文汇报。

当前导出的指标已经扩展到统一协议要求的完整 PV 误差集合，包括：

- `MAE / MSE / RMSE`
- `MBE`
- `MedianAE / P95AE / MaxAE`
- `sMAPE / MAPE(nonzero) / WAPE`
- `nMAE / nRMSE / nMBE`，分别按 `mean(abs(y_true))` 和 `max(abs(y_true))` 两套口径归一化
- `R2 / Pearson_r`

调参排序仍默认沿用 daytime RMSE/MAE 作为主排序口径；新增指标会进入 `metrics.json` 和 tuning summary，供后续筛选模型优势维度时使用。

## 7. 正式调参流程是什么

当前正式调参走的是一个 **小规模、分阶段、validation-only** 的流程。

默认推荐入口：

- `scripts/run_itransformer_tuning_standard.sh`

底层执行逻辑：

- 调参脚本：`scripts/tune_itransformer.sh`
- 单次 run：`scripts/run_iTransformer.py --tuning_only --report_split validation`

### 7.1 调参阶段

第一阶段主要调训练动力学参数：

- `learning_rate`
- `batch_size`
- `dropout`

第二阶段在第一阶段最佳共享配置上调模型容量：

- `d_model`
- `d_ff`
- `e_layers`

默认不动：

- `seq_len`
- `n_heads`
- `disable_norm`
- `output_attention`

### 7.2 调参目标

优先级如下：

1. `pred_len=12/24/48` 的 `daytime_only RMSE`
2. `pred_len=12/24/48` 的 `daytime_only MAE`
3. `all timestamps` 的 RMSE / MAE
4. `pred_len=1` 只作为补充参考

### 7.3 实验数量

`minimal` 计划：

- Stage 1: 4 个配置
- Stage 2: 2 个配置
- 主目标 `pred_len=12/24/48` 共 18 次 validation-only run
- 再补 1 次 `pred_len=1` 参考
- 总计 19 次

`standard` 计划：

- Stage 1: 5 个配置
- Stage 2: 3 个配置
- 主目标 `pred_len=12/24/48` 共 24 次 validation-only run
- 再补 1 次 `pred_len=1` 参考
- 总计 25 次

### 7.4 一个非常重要的约束

当前调参脚本明确保证：

- `test` 不参与调参
- `tune_itransformer.sh` 运行时会传入 `--tuning_only`
- 这时 `run_iTransformer.py` 不会加载 `test` split

这点是整个 iTransformer baseline 论文防御的核心之一。

## 8. 当前调参得到的最佳共享配置是什么

根据当前仓库里已经保存的标准版调参汇总：

- 汇总文件：`results/tuning/itransformer/standard/summary/final_validation/best_shared_config.json`

当前 `pred_len=12/24` 的最佳共享配置是：

- `seq_len=96`
- `learning_rate=1e-3`
- `batch_size=256`
- `d_model=128`
- `d_ff=256`
- `e_layers=3`
- `dropout=0.0`
- `weight_decay=1e-5`
- `activation=gelu`
- `n_heads=4`
- `factor=5`
- `disable_norm=False`
- `output_attention=False`
- `seed=42`

这组配置的含义是：

- 与默认 baseline 相比，真正被 validation 选择出来的主要变化是：
  - `e_layers: 2 -> 3`
  - `dropout: 0.1 -> 0.0`
- 其他主参数基本保持默认值

## 9. 结果目录分别代表什么

这是当前 iTransformer 最容易混淆的一部分，建议按下面理解。

### 9.1 默认 baseline 结果

目录：

- `results/itransformer/pred_len_1/`
- `results/itransformer/pred_len_12/`
- `results/itransformer/pred_len_24/`
- `results/itransformer/pred_len_48/`

含义：

- 普通 baseline 批量脚本 `scripts/run_itransformer_experiments.sh` 在 `MODE=train` 下跑出来的结果
- 这是 vanilla iTransformer 的默认训练结果

### 9.2 正式调参过程结果

目录：

- `results/tuning/itransformer/standard/`

里面包含：

- 每个 validation-only run 的 `metrics.json`
- 每个 run 的 `predictions.csv`
- 每个 run 的 `pred_plot.png`
- 汇总表：
  - `summary/stage1/`
  - `summary/final_validation/`
  - `summary/with_pred_len_1/`

这些目录是“调参过程”的正式记录，也是论文里最应该保留的部分。

### 9.3 调参后最终 test 导出

当前精简后的正式导出路径应当是：

- `results/itransformer_tuned/pred_len_1/`
- `results/itransformer_tuned/pred_len_12/`
- `results/itransformer_tuned/pred_len_24/`
- `results/itransformer_tuned/pred_len_48/`

这一步不重新训练，而是：

1. 从调参 summary 中找出最佳共享配置
2. 定位对应 checkpoint
3. 直接加载 checkpoint
4. 导出 `test` 指标和预测文件

当前这个流程是通过下面的命令触发的：

- `scripts/run_itransformer_experiments.sh` 配合 `MODE=export_tuned_best`

### 9.4 旧目录和 smoke 目录怎么理解

如果你在仓库里看到这些目录：

- `results/itransformer_tuned_from_ckpt/`
- `results/*_smoke/`

它们的含义分别是：

- `itransformer_tuned_from_ckpt`
  - 是更早一版导出脚本留下的历史结果目录
  - 结果本身可以参考
  - 但当前精简后的正式工作流已经不再依赖这个目录
- `*_smoke`
  - 只是脚本修复时的自测产物
  - 不属于正式实验结果

## 10. 当前已经保存的默认 baseline 结果

当前仓库里 `results/itransformer/` 下保存的 vanilla iTransformer 默认 test 结果如下。

注意：

- 当前仓库里已经保存的历史结果仍然是 `pred_len=1/12/24`
- `pred_len=48` 已经补进默认实验协议，但还需要单独跑出正式结果

| pred_len | best_epoch | all MAE | all RMSE | daytime MAE | daytime RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 19 | 0.073155 | 0.197747 | 0.139251 | 0.291766 |
| 12 | 15 | 0.174877 | 0.371234 | 0.310469 | 0.493172 |
| 24 | 17 | 0.298915 | 0.619721 | 0.476455 | 0.715708 |

从这些结果可以得到一个很明确的结论：

- 当前默认 vanilla iTransformer 已经明显强于当前仓库里的 LSTM baseline
- 在 `pred_len=12/24` 上也优于当前 Persistence baseline

因此，它已经是一个合格的强 baseline。

## 11. 当前已经保存的 tuned test 导出结果

仓库里还保留了一份历史导出目录：

- `results/itransformer_tuned_from_ckpt/`

这份目录对应的是：

- 使用 validation 选出来的 tuned checkpoint
- 再单独做 test 导出

其中保存的 tuned test 指标如下：

| pred_len | all MAE | all RMSE | daytime MAE | daytime RMSE |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.072339 | 0.198604 | 0.144980 | 0.293643 |
| 12 | 0.176864 | 0.380445 | 0.330434 | 0.509854 |
| 24 | 0.309678 | 0.621446 | 0.526972 | 0.775725 |

这个结果说明了一个非常值得保留的事实：

- 这次 validation-selected tuned vanilla iTransformer，在 test 上并没有超过默认 baseline
- 尤其在 `pred_len=12/24` 的 daytime 指标上，反而略差

从论文防御角度，这不是坏事，但必须诚实处理：

- 不能再根据 test 结果回头改参数
- 应该保留这次 tuned baseline 的完整调参记录
- 然后如实报告“validation 优选配置并不一定带来更好的 test daytime 指标”

## 12. 当前应该怎么运行

### 12.1 跑默认 vanilla iTransformer baseline

```bash
cd /home/lunalhx/projects/CiTransformer
bash scripts/run_itransformer_experiments.sh
```

### 12.2 跑正式 validation-only 调参

```bash
cd /home/lunalhx/projects/CiTransformer
bash scripts/run_itransformer_tuning_standard.sh
```

### 12.3 用调参选出的最佳 checkpoint 导出最终 test

```bash
cd /home/lunalhx/projects/CiTransformer

PYTHON_BIN=./.venv/bin/python \
DEVICE=cuda \
MODE=export_tuned_best \
PRED_LENS="1 12 24 48" \
RUN_PRED_LEN1_REF=1 \
RESULTS_BASE_DIR=results/itransformer_tuned \
bash scripts/run_itransformer_experiments.sh
```

这条命令的关键点是：

- 不会重新训练
- 会直接加载已经存在的 tuned checkpoint
- 会把正式导出结果写到 `results/itransformer_tuned/`

## 13. 如果论文里只想写关键实验处理，可以写什么

如果你想在论文里用较短但可复现的方式描述 vanilla iTransformer baseline，我建议可以概括成下面几点：

1. Vanilla iTransformer is trained only on the train split, with validation used for early stopping and checkpoint selection, while test is reserved for final evaluation only.
2. Feature scaling and target scaling are fit only on the train split to avoid data leakage.
3. Samples are constructed only within timestamp-continuous segments, and sliding windows never cross temporal discontinuities.
4. The multivariate iTransformer backbone predicts all encoder variables, after which the `Active_Pow` channel is extracted as the forecasting target.
5. All final metrics are computed after inverse transformation on the original power scale.
6. Hyperparameter selection is performed only on validation results, with daytime RMSE and daytime MAE for `pred_len=12/24/48` treated as the primary objectives.

## 14. 我的实际建议

如果你当前的目标是继续推进论文实验，我建议按下面理解这套 iTransformer baseline：

- `results/itransformer/`
  - 作为默认 vanilla baseline 结果保留
- `results/tuning/itransformer/standard/`
  - 作为正式调参记录保留
- `results/itransformer_tuned/`
  - 作为当前工作流下的最终 test 导出目录

然后在论文表格和文字里，建议明确区分三件事：

- 默认 vanilla iTransformer baseline
- validation-selected tuned vanilla iTransformer baseline
- 你的 CiTransformer

如果 tuned vanilla 最终没有优于默认 vanilla，也建议不要删掉这部分结果。因为从论文防御角度，这恰恰说明：

- 你没有让 test 参与调参
- 你的 baseline 调参是受控、可复现、且诚实汇报的
