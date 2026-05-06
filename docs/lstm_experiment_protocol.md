# LSTM 实验流程说明

本实验协议从属于统一协议：`docs/实验协议.md`。后续横向对比默认以统一协议中的数据来源、split、`seq_len/pred_len`、指标和导出结构为准；如有偏离，必须在本文件或对应实验记录里单独说明。

## 1. 当前这个实验在做什么

这个仓库里的 LSTM baseline 是通过 `scripts/run_lstm_experiments.sh` 统一调用 `scripts/run_lstm.py` 来运行的，当前会分别跑 4 组预测步长：

- `pred_len=1`
- `pred_len=12`
- `pred_len=24`
- `pred_len=48`

每个 `pred_len` 都会单独训练一个模型，并把结果分别写到：

- `results/lstm/pred_len_1/`
- `results/lstm/pred_len_12/`
- `results/lstm/pred_len_24/`
- `results/lstm/pred_len_48/`

模型本身是一个比较标准、比较朴素的 sequence-to-vector LSTM baseline：

- 输入形状：`[batch, seq_len, feature_dim]`
- 主体结构：多层 `nn.LSTM`
- 输出头：`dropout + linear`
- 输出形状：`[batch, pred_len]`

对应代码位置：

- 模型定义：`models/baseline/lstm.py`
- 训练与测试入口：`scripts/run_lstm.py`
- 数据集与标准化：`utils/datasets.py`

当前 `scripts/run_lstm.py` 还支持两个协议内的辅助模式：

- `--tuning_only --report_split validation`：只加载 train/validation/calibration，不加载 test，用于 validation-only 调参
- `--eval_checkpoint_path <best_model.pth>`：跳过训练，直接从已有 checkpoint 导出 validation 或 test 结果

默认批量脚本 `scripts/run_lstm_experiments.sh` 不使用这些辅助模式，仍然走正式 baseline 的 train -> validation checkpoint -> test export 流程。

## 2. train / validation / test / calibration 分别怎么用

当前代码里这 4 个 split 的用途如下：

- `train`
  - 用来拟合 `feature_scaler` 和 `target_scaler`
  - 用来训练模型参数
  - DataLoader 使用 `shuffle=True`
- `validation`
  - 每个 epoch 结束后计算一次验证损失
  - 用来选择最优 checkpoint
  - 用来做 early stopping
- `test`
  - 只在训练结束后使用一次
  - 代码会先重新加载“验证集上最优”的 checkpoint，再在 test 上做推理
  - test 指标会保存到 `results/.../metrics.json`
  - 如果显式使用 `--tuning_only --report_split validation`，则不加载 test，也不会导出 `test_metrics`
- `calibration`
  - 会被读取，也会参与数据统计
  - 但当前这版 LSTM baseline 不参与训练
  - 不参与 checkpoint 选择
  - 也不参与最终测试指标汇报

所以这套实验的实际流程是：

1. 只在 `train` 上拟合 scaler
2. 用 `train` 训练模型
3. 每个 epoch 用 `validation` 监控损失
4. 按验证损失保存最优模型
5. 如果验证集长期不提升，则 early stopping
6. 训练结束后重新加载最佳 checkpoint
7. 最后只在 `test` 上评估一次

这是一个标准而且合理的 train / validation / test 流程。

这里有一个需要在论文里说明的小点：

- 仓库里虽然读取了 `calibration.csv`
- 但当前这版 LSTM baseline 实际上并没有使用 calibration split

如果你后面写论文，建议明确写一句：`calibration split is reserved but unused in the current LSTM baseline.`

## 3. 当前实验使用的超参数

当前提交到仓库里的实验脚本，默认参数定义在 `scripts/run_lstm_experiments.sh`，主要设置如下：

- `DATA_DIR=data/processed_selected_2020_2022`
- `SEQ_LEN=96`
- `PRED_LENS=1 12 24 48`
- `BATCH_SIZE=256`
- `HIDDEN_SIZE=128`
- `NUM_LAYERS=2`
- `DROPOUT=0.1`
- `LEARNING_RATE=1e-3`
- `WEIGHT_DECAY=1e-5`
- `GRAD_CLIP=1.0`
- `EPOCHS=30`
- `PATIENCE=8`
- `MIN_DELTA=1e-5`
- `NUM_WORKERS=0`
- `LOG_INTERVAL=0`
- `PROGRESS_MININTERVAL=15`
- `SEED=42`
- `DEVICE=auto`

默认输入特征是：

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

这里有两个值得特别说明的点：

- `Active_Pow` 既是预测目标，也是输入特征之一，所以这是标准时间序列里常见的自回归式用法
- 训练时目标值会先标准化，测试时再把预测值反归一化回原始功率尺度

## 4. 当前代码里的 loss 和 metrics 分别代表什么

当前代码里其实存在两种不同的“数值尺度”：

- 训练和验证阶段的 `loss`
  - 使用的是标准化目标值上的 `nn.MSELoss()`
- `metrics.json` 里的 validation/test 指标
  - 使用的是反归一化后的预测值和原始真实值来计算
  - 因此它们是在原始 `Active_Pow` 功率尺度上汇报的

这意味着：

- `history.train_loss` 和 `history.validation_loss`
- 不能直接和最终 `validation_metrics` / `test_metrics` 里的原始量纲指标做数值比较

如果后面写论文，这两类数不能混着解释成同一个量纲。

当前导出的指标已经扩展到统一协议要求的完整 PV 误差集合，包括：

- `MAE / MSE / RMSE`
- `MBE`
- `MedianAE / P95AE / MaxAE`
- `sMAPE / MAPE(nonzero) / WAPE`
- `nMAE / nRMSE / nMBE`，分别按 `mean(abs(y_true))` 和 `max(abs(y_true))` 两套口径归一化
- `R2 / Pearson_r`

## 5. 当前已经得到的结果

当前仓库里已经保存下来的结果如下。

注意：

- 当前仓库里已经保存的历史结果仍然是 `pred_len=1/12/24`
- `pred_len=48` 已经补进默认实验协议，但还需要单独跑出正式结果

| pred_len | best_epoch | all MAE | all RMSE | daytime MAE | daytime RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 22 | 0.126833 | 0.248860 | 0.256358 | 0.367473 |
| 12 | 4 | 0.259465 | 0.485269 | 0.534752 | 0.717474 |
| 24 | 3 | 0.339347 | 0.637045 | 0.709647 | 0.940426 |

这些结果可以这样理解：

- 预测步长越长，误差越大，这是很正常的现象
- `pred_len=12` 和 `pred_len=24` 的最佳 epoch 分别只有 4 和 3，说明当前这组超参数对于长预测步长大概率还不是很理想

## 6. 这个 baseline 需不需要调参

简短结论：

- 需要做一些小规模调参
- 但不需要一开始就做特别大的网格搜索

是否“必须调很多”取决于你想在论文里把这个 baseline 放到什么位置。

如果你只是需要一个合理、可运行的 baseline：

- 用一组固定参数也是可以接受的
- 但最好在论文里诚实地写成 “single-configuration baseline” 或者 “lightly tuned baseline”

如果你希望它成为一个更有说服力、更公平的对比基线：

- 建议至少做一轮基于 validation 的小规模调参
- 否则审稿人可能会认为 baseline 没有被认真优化

我对当前这版结果的判断是：

- 它已经是一个有效的第一版 baseline
- 但它还不太像“充分优化后的 LSTM baseline”
- 特别是 `pred_len=12` 和 `pred_len=24`，我建议你再做一点调参之后，再把结果放进论文最终表格

我建议优先尝试这些低成本参数：

- `seq_len`：试 `96` 和 `288`
- `hidden_size`：试 `64`、`128`、`256`
- `num_layers`：试 `1` 和 `2`
- `dropout`：试 `0.0`、`0.1`、`0.3`
- `learning_rate`：试 `1e-3` 和 `5e-4`

不建议一开始就做很大的搜索。对 baseline 来说，一轮小而清楚的 validation 调参通常就够用了。

## 7. 你是不是应该更关注 daytime only

对于光伏功率预测这个任务，我的建议是：

- `all_timestamps` 和 `daytime_only` 都要报
- 但如果你的研究重点是实际光伏发电预测质量，那么正文里应该更强调 `daytime_only`

原因是：

- 夜间功率通常为 0 或非常接近 0
- 如果把大量夜间点都放进总体指标，整体误差可能会显得比白天真实预测效果更好
- 真正有预测难度、也更有实际意义的，通常是白天时段

因此比较好的论文呈现方式是：

- `all_timestamps` 保留，作为完整汇报
- `daytime_only` 作为主结果或重点讨论对象

但这里有一个非常重要的细节：

- 当前代码选最优 checkpoint 的依据，是“标准化后的全时段 validation MSE”
- 并不是按 `daytime_only` 的 validation 指标来选模型

如果你论文的核心目标更偏向白天功率预测，那么更合理的实验协议会是：

- 训练过程可以保持不变
- 但 best checkpoint 的选择标准，最好改成更贴近论文目标的验证指标，例如 daytime validation MAE 或 daytime validation RMSE

## 8. 不连续时间段的处理是否正确

总体上看，我认为当前处理方式是正确的，而且作为 baseline 来说是比较稳妥的。

当前代码的做法是：

- 先按时间戳排序
- 从训练集时间戳推断预期采样间隔
- 如果相邻两个时间点的间隔不等于这个预期采样间隔，就认为进入了新的 segment
- 只允许在每个连续 segment 内部做滑窗
- 长度小于 `seq_len + pred_len` 的 segment 会被丢弃

这种处理方式的优点是：

- 避免把缺失段两边的数据错误地拼成一条连续序列
- 避免窗口跨越清洗后留下的断点
- 相比简单的 `iloc` 连续滑窗，这种做法更安全、更符合时序建模逻辑

对你现在这个任务来说，我认为这是一个对的方法。

当然它也有一些可能的代价：

- 如果时间戳存在轻微抖动、时区切换或夏令时问题，用“完全相等”判断可能会切得比较碎
- 不允许跨 gap 会减少可用样本数量
- 某些非常小的 gap，如果理论上可以安全插值，这种做法会显得偏保守

但对于 baseline 实验，我反而认为“保守”通常比“错误跨段”更好。

## 9. 如果论文里只想写关键实验处理，可以写什么

如果你想在论文里用比较简洁、可复现的方式描述这套 LSTM baseline，我建议可以概括成下面几点：

1. LSTM baseline 仅在 train split 上训练，validation split 用于 early stopping 和 checkpoint 选择，test split 仅用于最终评估。
2. 特征标准化和目标标准化都只在 train split 上拟合，以避免数据泄漏。
3. 样本只在时间连续的片段内部构造；任何与预期采样间隔不一致的时间跳变都会开启新的 segment，滑窗不会跨越时间断点。
4. 模型使用多层 LSTM 的最后一个时间步隐状态，并通过线性层直接输出未来 `pred_len` 步的 `Active_Pow` 预测。
5. 最终测试指标是在反归一化后的原始功率尺度上计算的。
6. 结果同时汇报全时段指标和白天时段指标，其中白天指标对于光伏预测解释尤其重要。

## 10. 我的实际建议

如果你当前的目标是先把项目推进下去：

- 这版结果可以继续保留，作为 baseline 原型
- 也可以先用它做内部对比和趋势分析

如果你的目标是论文最终实验表格：

- 保留当前实现方式
- 再做一轮小规模 validation 调参
- 如果论文主线更看重白天预测效果，建议改成按 daytime validation 指标选最优 checkpoint
- 最终同时报告 `all_timestamps` 和 `daytime_only`
