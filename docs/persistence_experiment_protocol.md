# Persistence 实验流程说明

## 1. 当前这个实验在做什么

这个仓库里的 Persistence baseline 是通过 `scripts/run_persistence_experiments.sh` 统一调用 `scripts/run_persistence.py` 来运行的，当前会分别跑 4 组预测步长：

- `pred_len=1`
- `pred_len=12`
- `pred_len=24`
- `pred_len=48`

每个 `pred_len` 都会单独评估一次，并把结果分别写到：

- `results/persistence/pred_len_1/`
- `results/persistence/pred_len_12/`
- `results/persistence/pred_len_24/`
- `results/persistence/pred_len_48/`

模型本身不是神经网络训练基线，而是一个标准、朴素、无参数的 persistence / naive baseline：

- 输入形状：`[batch, seq_len, feature_dim]`
- 预测规则：取输入窗口最后一个时间步的 `Active_Pow`
- 输出形状：`[batch, pred_len]`
- 输出内容：把最后一个已观测功率值重复到未来 `pred_len` 步

对应代码位置：

- 模型定义：`models/baseline/persistence.py`
- 评估入口：`scripts/run_persistence.py`
- 批量运行脚本：`scripts/run_persistence_experiments.sh`
- 数据集与标准化：`utils/datasets.py`

## 2. train / validation / test / calibration 分别怎么用

当前代码里这 4 个 split 的用途如下：

- `train`
  - 用来拟合 `feature_scaler` 和 `target_scaler`
  - 用来推断预期采样间隔 `expected_delta`
  - 不参与任何参数训练，因为 Persistence baseline 没有可学习参数
- `validation`
  - 会执行一次完整推理
  - 用于输出 `validation_metrics`
  - 不用于 early stopping，因为这里不存在训练过程
- `test`
  - 会执行一次完整推理
  - `test_metrics` 会保存到 `results/.../metrics.json`
  - `predictions.csv` 和 `pred_plot.png` 也基于 test 结果导出
- `calibration`
  - 会被读取，也会参与数据统计
  - 但当前这版 Persistence baseline 不参与训练
  - 不参与主评估导出

所以这套实验的实际流程是：

1. 只在 `train` 上拟合 scaler
2. 只在 `train` 上推断采样间隔
3. 分别在 `validation` 和 `test` 上做规则型推理
4. 输出反归一化后的指标
5. 将 test 预测保存为 CSV 和图像

这里有一个需要在论文里说明的小点：

- 仓库里虽然读取了 `calibration.csv`
- 但当前这版 Persistence baseline 实际上并没有使用 calibration split

如果你后面写论文，建议明确写一句：`calibration split is reserved but unused in the current persistence baseline.`

## 3. 当前实验使用的默认参数

当前批量脚本 `scripts/run_persistence_experiments.sh` 的主要默认设置如下：

- `DATA_DIR=data/processed`
- `SEQ_LEN=96`
- `PRED_LENS=1 12 24 48`
- `BATCH_SIZE=256`
- `NUM_WORKERS=0`
- `SEED=42`
- `DEVICE=auto`
- `RESULTS_BASE_DIR=results/persistence`

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

- `Active_Pow` 既是预测目标，也是输入特征之一，因此 Persistence baseline 才能从最后一个观测时刻复制功率值
- 虽然 baseline 不训练，但仍然沿用了 train-only fit 的 scaler 和统一 inverse-transform 评估流程

## 4. 当前代码里的“预测值尺度”和“最终指标尺度”分别代表什么

当前代码里其实存在两种不同的数值空间：

- 模型前向输出阶段
  - 输出位于目标 scaler 的标准化空间
- `metrics.json` 里的 validation/test 指标
  - 使用的是反归一化后的预测值和原始真实值
  - 因此它们是在原始 `Active_Pow` 功率尺度上汇报的

这意味着：

- `metrics.json` 里的数值已经是论文实验里可以直接使用的真实量纲指标
- 不需要再对输出结果做额外反标准化

## 5. 当前已经得到的结果

当前仓库里已经保存下来的 Persistence baseline 结果如下。

注意：

- 当前仓库里已经保存的历史结果仍然是 `pred_len=1/12/24`
- `pred_len=48` 已经补进默认实验协议，但还需要单独跑出正式结果

| pred_len | all MAE | all RMSE | daytime MAE | daytime RMSE |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.064730 | 0.213526 | 0.141273 | 0.316413 |
| 12 | 0.209007 | 0.434182 | 0.434236 | 0.627447 |
| 24 | 0.346674 | 0.661986 | 0.681058 | 0.923872 |

如果把它和当前仓库里的 LSTM baseline 做一个并排观察，可以得到下面这组结果：

| pred_len | baseline | all MAE | all RMSE | daytime MAE | daytime RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Persistence | 0.064730 | 0.213526 | 0.141273 | 0.316413 |
| 1 | LSTM | 0.126833 | 0.248860 | 0.256358 | 0.367473 |
| 12 | Persistence | 0.209007 | 0.434182 | 0.434236 | 0.627447 |
| 12 | LSTM | 0.259465 | 0.485269 | 0.534752 | 0.717474 |
| 24 | Persistence | 0.346674 | 0.661986 | 0.681058 | 0.923872 |
| 24 | LSTM | 0.339347 | 0.637045 | 0.709647 | 0.940426 |

这些结果至少说明两件事：

- 当前数据集上，Persistence baseline 并不是一个很弱的 baseline，尤其在 `pred_len=1` 和 `pred_len=12` 上表现相当强
- 对于 `pred_len=24`，Persistence 与当前 LSTM 的差距已经比较接近，说明这个任务在较长预测步长上并不容易被简单拉开

如果后面写论文，这一点很重要：

- 你的正式模型需要至少显著超过 Persistence，才更有说服力

## 6. 这个 baseline 需不需要调参

简短结论：

- 它几乎没有“训练意义上的调参”
- 但仍然存在“实验协议层面的设置”

具体来说，Persistence 没有下列东西：

- 学习率
- epoch
- optimizer
- hidden size
- dropout

因此它不存在传统意义上的超参数搜索。

但它仍然会受到这些设置影响：

- `seq_len`
- `pred_len`
- 输入特征列定义
- 样本构造规则
- 是否报告 `daytime_only`

其中最核心的是：

- `seq_len` 决定输入窗口最后一个时刻之前能看到多少历史

不过对标准 persistence 而言，真正用于预测的只有“最后一个时间步的 `Active_Pow`”，所以只要 `seq_len >= 1`，预测规则本身并不会因为 `seq_len` 变大而改变。

这也意味着：

- Persistence baseline 的主要价值不是“调到最好”
- 而是给出一个公平、透明、强解释性的参考下界

## 7. 你是不是应该更关注 daytime only

对于光伏功率预测这个任务，我的建议和 LSTM 一样：

- `all_timestamps` 和 `daytime_only` 都要报
- 但如果研究重点是实际光伏发电预测质量，那么正文里应该更强调 `daytime_only`

原因是：

- 夜间功率通常为 0 或非常接近 0
- Persistence 在夜间复制 0 本来就很容易获得较小误差
- 如果把大量夜间点都放进总体指标，整体误差会显得偏乐观
- 真正有预测难度、也更有实际意义的，仍然主要是白天时段

因此比较好的论文呈现方式是：

- `all_timestamps` 保留，作为完整汇报
- `daytime_only` 作为重点分析对象

## 8. 不连续时间段的处理是否正确

总体上看，我认为当前处理方式是正确的，而且对 Persistence baseline 同样重要。

当前代码的做法是：

- 先按时间戳排序
- 从训练集时间戳推断预期采样间隔
- 如果相邻两个时间点的间隔不等于这个预期采样间隔，就认为进入了新的 segment
- 只允许在每个连续 segment 内部做滑窗
- 长度小于 `seq_len + pred_len` 的 segment 会被丢弃

这对 Persistence 尤其重要，因为：

- 如果跨 gap 取最后一个观测功率值，实际上会把断点前的值错误复制到断点后的目标区间
- 这会直接引入不合理的样本，属于协议错误，而不是模型能力问题

因此当前这种严格的 segment 约束是必要的。

## 9. 如果论文里只想写关键实验处理，可以写什么

如果你想在论文里用比较简洁、可复现的方式描述这套 Persistence baseline，我建议可以概括成下面几点：

1. Persistence baseline does not train any model parameters; it repeats the last observed `Active_Pow` in the input window for all future horizons.
2. Feature scaling and target scaling are fit only on the train split to avoid data leakage.
3. Samples are constructed only within timestamp-continuous segments; any timestamp jump inconsistent with the expected sampling interval starts a new segment.
4. The persistence prediction is generated from the last encoder-step `Active_Pow`, while evaluation is performed after inverse transformation on the original power scale.
5. Results are reported for both all timestamps and daytime-only timestamps (`day_night_label == 1`).

## 10. 我的实际建议

如果你当前的目标是先把项目推进下去：

- 这版 Persistence baseline 已经足够作为正式对照组使用
- 可以直接放进后续所有模型比较表里

如果你的目标是论文最终实验表格：

- 保留当前实现方式，不建议再“优化” Persistence 本身
- 重点去提高 LSTM、vanilla iTransformer、Causal-iTransformer 相对于 Persistence 的增益
- 正文里建议明确指出该 baseline 是一个 **strong naive baseline**

对光伏预测这类任务来说，我认为这是一个非常值得保留的结论：

- 如果一个更复杂的模型长期打不过 Persistence，那么问题通常不在 baseline 太强，而在新模型还没有真正学到比“复制最近观测值”更多的信息
