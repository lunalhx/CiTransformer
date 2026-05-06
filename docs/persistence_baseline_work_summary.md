# Persistence Baseline 实现工作记录

## 1. 文档目的

本文档用于记录本次为论文项目补充的 **光伏功率预测 Persistence baseline** 实验代码，包括：

- 我完成了哪些任务
- 新增/修改了哪些文件
- 关键实现逻辑是什么
- 如何运行实验
- 我已经做了哪些验证

---

## 2. 本次完成的工作

本次已经为当前项目补充了一套 **可运行的 Persistence / naive baseline 实验代码**，核心目标是作为后续 `LSTM`、`vanilla iTransformer` 和 `Causal-iTransformer` 的规则型对比基线。

本次完成内容如下：

### 2.1 补充了标准 Persistence baseline

新增了一个无训练参数的 Persistence baseline，严格按照经典定义实现：

- 输入形状：`[batch_size, seq_len, feature_dim]`
- 输出形状：`[batch_size, pred_len]`

预测规则为：

1. 从输入窗口最后一个时间步中取出 `Active_Pow`
2. 将该值重复 `pred_len` 次
3. 作为未来 `pred_len` 步的预测值

即：

`y_hat = [last_power] * pred_len`

这里没有引入任何可学习网络，也没有把 Persistence 伪装成 Linear / MLP / LSTM。

---

### 2.2 复用了“基于连续时间片段”的 Dataset

Persistence baseline 没有重新实现一套数据滑窗逻辑，而是直接复用了当前 LSTM baseline 已经实现好的 `ContinuousSegmentTimeSeriesDataset`。

因此它天然继承了以下约束：

- **不能直接对整张表按 `iloc` 滑窗**
- 必须基于 `timestamp` 检查真实时间连续性
- 只允许在同一个连续片段（segment）内构造样本
- 任何窗口都**不能跨越时间断点**

这保证了 Persistence 与 LSTM 在样本构造口径上保持一致，适合作为论文中的正式 baseline。

---

### 2.3 复用了严格无泄漏的标准化流程

Persistence baseline 虽然不训练，但仍然严格接入了现有 LSTM 的 scaler / inverse-transform 流程，而不是绕开它。

当前流程是：

- `feature_scaler` 只在 `train.csv` 上拟合
- `target_scaler` 只在 `train.csv` 上拟合
- `validation / calibration / test` 只做 `transform`
- 最终评估时再统一 `inverse_transform` 回原始功率尺度

这里特别处理了一个容易被忽略的问题：

- `x` 中最后一个 `Active_Pow` 位于 **feature scaler 空间**
- `y` 位于 **target scaler 空间**

因此 Persistence baseline 在前向过程中，先把最后一个输入功率值从 feature 空间映射回原始尺度，再映射到 target 空间，最后继续复用现有的逆变换评估流程。

这样做的好处是：

- 不破坏当前工程的无泄漏评估口径
- 即使未来输入和目标 scaler 逻辑发生变化，这个 baseline 也不会 silently 出错

---

### 2.4 实现了无训练的评估流程

Persistence baseline 对应的运行脚本中：

- 没有 `optimizer`
- 没有 `backward`
- 没有 `early stopping`
- 没有 checkpoint 训练与保存

它只做以下事情：

1. 读取四个 split
2. 用 train split 拟合 scaler
3. 构造 validation / test 的 DataLoader
4. 直接按 Persistence 规则推理
5. 计算指标并保存结果

因此它是一个真正的规则型 baseline，而不是“无训练外观、内部仍带可学习参数”的伪 baseline。

---

### 2.5 实现了统一的结果保存格式

Persistence baseline 已复用 LSTM 的主要评估与导出逻辑，结果输出保持尽量统一。

当前会保存：

- `results/persistence/pred_len_x/metrics.json`
- `results/persistence/pred_len_x/predictions.csv`
- `results/persistence/pred_len_x/pred_plot.png`

其中：

- `predictions.csv` 当前与 LSTM 一致，只保存 **test split** 的预测结果
- `pred_plot.png` 直接复用了 LSTM 的画图函数
- `metrics.json` 中同时保存：
  - `validation_metrics`
  - `test_metrics`
  - `all_timestamps`
  - `daytime_only`

支持的指标包括：

- `MAE`
- `MSE`
- `RMSE`
- `sMAPE`
- `MAPE(nonzero)`

---

## 3. 新增/修改的文件

本次新增或修改的文件如下：

### 3.1 模型文件

文件：

- `models/baseline/persistence.py`

作用：

- 定义标准 Persistence baseline
- 明确从输入窗口最后一个时间步中提取 `Active_Pow`
- 将最后观测值重复到未来 `pred_len` 步

---

### 3.2 运行入口脚本

文件：

- `scripts/run_persistence.py`

作用：

- 读取数据集
- 复用 LSTM 的 Dataset / DataLoader / 评估逻辑
- 在 validation 和 test 上执行推理
- 保存指标、预测结果和图像

---

### 3.3 批量实验脚本

文件：

- `scripts/run_persistence_experiments.sh`

作用：

- 批量运行 `pred_len=1/12/24/48`
- 自动选择 Python 解释器
- 自动检查 `torch / pandas / matplotlib`
- 自动配置 `MPLCONFIGDIR`

---

### 3.4 模型导出入口

文件：

- `models/baseline/__init__.py`

作用：

- 导出 `PersistenceBaseline`

---

## 4. 数据路径与兼容方式

脚本默认读取 2020-2022 年选定光伏数据集：

- `data/processed_selected_2020_2022/splits/train.csv`
- `data/processed_selected_2020_2022/splits/validation.csv`
- `data/processed_selected_2020_2022/splits/calibration.csv`
- `data/processed_selected_2020_2022/splits/test.csv`

如果 `data_dir/` 下直接存在四个 split 文件，则直接读取；否则自动回退到 `data_dir/splits/`。

因此当前仓库可以直接运行，不需要你手动改数据路径。

---

## 5. 关键实现说明

### 5.1 Persistence 的精确定义

对每个样本：

- 输入窗口为 `[seq_len, feature_dim]`
- 预测目标为未来 `pred_len` 步的 `Active_Pow`

模型实际执行的是：

1. 取输入窗口最后一个时间步
2. 定位其中的 `Active_Pow`
3. 得到 `last_power`
4. 输出 `[last_power] * pred_len`

因此这是最标准的 naive persistence baseline。

---

### 5.2 Active_Pow 列索引识别

代码中没有把 `Active_Pow` 的列索引写死为某个魔法数字。

当前做法是：

- 先从 `feature_cols` 中查找 `target_col`
- 再动态得到 `target_feature_index`

这样即使后续输入特征列顺序发生变化，只要 `Active_Pow` 仍在 `feature_cols` 中，这个 baseline 仍然是正确的。

---

### 5.3 scaler 尺度处理

为了严格复用 LSTM 的评估流程，Persistence baseline 的输出不是直接在原始值上硬写入结果文件，而是：

1. 从 `x` 中取出最后一个 `Active_Pow` 的特征尺度值
2. 反推回原始功率尺度
3. 再投到目标尺度
4. 交给统一的 `collect_predictions(...)`
5. 最终再由 `target_scaler.inverse_transform(...)` 恢复成真实功率尺度

这保证了：

- 预测口径与 LSTM 一致
- 指标口径与 LSTM 一致
- 结果可直接并排比较

---

### 5.4 Calibration 集处理

当前代码会读取 `calibration.csv` 并完成 Dataset 构造，但不会参与训练。

由于 Persistence baseline 本身不存在训练过程，因此这里的实际语义是：

- `calibration` 被保留用于协议兼容与数据统计
- 但不会参与 validation/test 的主评估输出

这样保留了你后续做：

- conformal calibration
- uncertainty calibration
- post-hoc 校准

的扩展接口。

---

## 6. 运行方式

### 6.1 单个实验运行

在项目根目录执行：

```bash
cd /home/lunalhx/projects/CiTransformer
./.venv/bin/python scripts/run_persistence.py --seq_len 96 --pred_len 1 --batch_size 256 --results_dir results/persistence/pred_len_1
```

---

### 6.2 批量运行 1 / 12 / 24 / 48 步实验

在项目根目录执行：

```bash
cd /home/lunalhx/projects/CiTransformer
./scripts/run_persistence_experiments.sh
```

该脚本默认会依次运行：

- `pred_len=1`
- `pred_len=12`
- `pred_len=24`
- `pred_len=48`

并将结果写入：

- `results/persistence/pred_len_1/`
- `results/persistence/pred_len_12/`
- `results/persistence/pred_len_24/`
- `results/persistence/pred_len_48/`

---

## 7. 我已经做过的验证

截至目前，我已经确认以下内容：

- `scripts/run_persistence.py` 可以通过 `compileall`
- `scripts/run_persistence_experiments.sh` 已通过 `bash -n` 语法检查
- `results/persistence/pred_len_1/` 已生成完整结果
- `results/persistence/pred_len_12/` 已生成完整结果
- `results/persistence/pred_len_24/` 已生成完整结果

当前已经存在的输出文件包括：

- `results/persistence/pred_len_1/metrics.json`
- `results/persistence/pred_len_1/predictions.csv`
- `results/persistence/pred_len_1/pred_plot.png`
- `results/persistence/pred_len_12/metrics.json`
- `results/persistence/pred_len_12/predictions.csv`
- `results/persistence/pred_len_12/pred_plot.png`
- `results/persistence/pred_len_24/metrics.json`
- `results/persistence/pred_len_24/predictions.csv`
- `results/persistence/pred_len_24/pred_plot.png`

这说明以下流程均已验证通过：

- 数据读取
- 时间列解析
- segment 划分
- DataLoader 构造
- Persistence 前向推理
- inverse-transform
- validation / test 指标计算
- 结果文件写出

---

## 8. 当前文档生成位置

本说明文档生成在以下位置：

- 相对路径：`docs/persistence_baseline_work_summary.md`
- 绝对路径：`/home/lunalhx/projects/CiTransformer/docs/persistence_baseline_work_summary.md`

---

## 9. 当前结论

截至目前，本项目中的 **Persistence baseline 实验代码已经完成**，并满足以下核心要求：

- 基于连续时间片段构造样本
- 严格避免跨断点滑窗
- 严格避免 scaler 数据泄漏
- 不使用任何可学习网络
- 不存在训练、反向传播和 early stopping
- calibration 集不参与主评估
- 可以直接与当前 LSTM baseline 并排比较

如果后续需要，我可以继续补两类内容：

1. 再单独写一份 Persistence 与 LSTM 的结果对比文档
2. 继续补 `vanilla iTransformer baseline` 的统一训练入口，并保持与当前两套 baseline 相同的数据接口与评估接口
