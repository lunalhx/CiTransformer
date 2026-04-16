# LSTM Baseline 实现工作记录

## 1. 文档目的

本文档用于记录本次为论文项目补充的 **光伏功率预测 LSTM baseline** 实验代码，包括：

- 我完成了哪些任务
- 新增/修改了哪些文件
- 关键实现逻辑是什么
- 如何运行实验
- 我已经做了哪些验证

---

## 2. 本次完成的工作

本次已经为当前项目补充了一套 **可运行的 PyTorch LSTM baseline 实验代码**，核心目标是作为后续 `vanilla iTransformer` 和 `Causal-iTransformer` 的对比基线。

本次完成内容如下：

### 2.1 补充了标准 LSTM 模型

新增了一个标准的多步预测 LSTM 模型，支持以下配置：

- `input_size`
- `hidden_size`
- `pred_len`
- `num_layers`
- `dropout`

模型输入为：

- 形状：`[batch_size, seq_len, feature_dim]`

模型输出为：

- 形状：`[batch_size, pred_len]`

即直接输出未来 `pred_len` 步的 `Active_Pow` 预测值。

---

### 2.2 实现了“基于连续时间片段”的 Dataset

这是本次最关键的部分，已经严格按你的要求实现：

- **不能直接对整张表按 `iloc` 滑窗**
- 必须基于 `timestamp` 检查真实时间连续性
- 只允许在同一个连续片段（segment）内构造样本
- 任何窗口都**不能跨越时间断点**

具体实现逻辑：

1. 自动识别时间列并转成 `DatetimeIndex`
2. 计算相邻时间戳差值
3. 当差值不等于采样频率时，判定为新的连续 segment
4. 每个 segment 内单独计算可用窗口
5. 只有长度满足 `seq_len + pred_len` 的 segment 才参与训练/验证/测试

因此，清洗后因长缺失段删除造成的时间断点，已经被严格隔离，不会错误地被 LSTM 当成连续序列。

---

### 2.3 实现了严格无泄漏的标准化流程

已经按要求实现以下约束：

- `scaler` 只在 `train.csv` 上 `fit`
- `validation.csv / calibration.csv / test.csv` 只做 `transform`
- `calibration` 集不会进入训练
- 测试集不会参与训练或标准化拟合

这里使用了两个 `StandardScaler`：

- `feature_scaler`：用于输入特征
- `target_scaler`：用于目标列 `Active_Pow`

训练时模型学习的是标准化后的目标值，测试输出时再逆变换回原始量纲。

---

### 2.4 实现了完整训练流程

训练脚本已经包含：

- train / validation 循环
- `MSELoss`
- `Adam`
- `early stopping`
- best checkpoint 保存
- 固定随机种子
- CPU / GPU / MPS 自动适配
- 可配置超参数

支持的主要参数包括：

- `seq_len`
- `pred_len`
- `hidden_size`
- `num_layers`
- `dropout`
- `batch_size`
- `epochs`
- `patience`
- `learning_rate`

---

### 2.5 实现了测试与结果保存

测试阶段已支持输出以下指标：

- `MAE`
- `MSE`
- `RMSE`
- `sMAPE`
- `MAPE(nonzero)`

说明：

- 由于夜间 `Active_Pow = 0`，普通 MAPE 会出现除零不稳定问题
- 因此这里实现的是 **只对非零真实值计算的 MAPE**
- 同时保留了 `sMAPE` 作为更稳健的补充指标

另外支持两种评估口径：

- `all timestamps`
- `daytime only`（`day_night_label == 1`）

实验结果会保存到：

- `results/lstm/metrics.json`
- `results/lstm/predictions.csv`
- `results/lstm/pred_plot.png`
- `checkpoints/lstm/best_model.pth`

---

## 3. 新增/修改的文件

本次新增或修改的文件如下：

### 3.1 模型文件

文件：

- `models/baseline/lstm.py`

作用：

- 定义标准 LSTM baseline 模型

---

### 3.2 数据集与数据处理文件

文件：

- `utils/datasets.py`

作用：

- 自动识别时间列
- 转换为 `DatetimeIndex`
- 推断采样频率
- 依据时间连续性切分 segment
- 仅在 segment 内滑窗
- train-only scaler 拟合

---

### 3.3 训练与测试入口脚本

文件：

- `scripts/run_lstm.py`

作用：

- 读取数据集
- 构造 DataLoader
- 训练 LSTM
- 验证并 early stopping
- 加载 best checkpoint
- 在 test 上评估
- 保存指标、预测结果和图像

---

### 3.4 模型导出入口

文件：

- `models/baseline/__init__.py`

作用：

- 导出 `LSTMBaseline`

---

## 4. 数据路径与兼容方式

脚本默认读取：

- `data/processed/train.csv`
- `data/processed/validation.csv`
- `data/processed/calibration.csv`
- `data/processed/test.csv`

但当前仓库实际数据位于：

- `data/processed/splits/train.csv`
- `data/processed/splits/validation.csv`
- `data/processed/splits/calibration.csv`
- `data/processed/splits/test.csv`

我已经在代码里做了兼容：

- 如果 `data/processed/` 下直接存在四个 split 文件，就读取那里
- 否则自动回退到 `data/processed/splits/`

因此当前仓库可以直接运行，不需要你手动改路径。

---

## 5. 关键实现说明

### 5.1 Segment 划分规则

对于按时间排序后的序列：

- 若相邻两行时间差等于采样频率，例如 `5 min`
- 则属于同一连续片段

若相邻两行时间差不等于采样频率：

- 则认定为新的 segment 起点

这样可以保证：

- 缺失段删除后造成的时间断裂不会被当作连续时序

---

### 5.2 段内滑窗规则

对每个连续 segment，只有当长度满足：

`segment_length >= seq_len + pred_len`

时，才允许构造样本。

每个样本形式为：

- 输入：`[t-seq_len+1, ..., t]`
- 目标：`[t+1, ..., t+pred_len]`

且输入与目标都必须完全位于同一个 segment 内。

---

### 5.3 Calibration 集处理

当前代码会读取 `calibration.csv` 并完成 Dataset 构造，但不会参与训练。

这样保留了你后续做：

- conformal calibration
- uncertainty calibration
- post-hoc 校准

的扩展接口。

---

## 6. 运行方式

由于你当前使用的是 Anaconda 环境 `causal_env`，建议用以下两种方式之一运行。

### 6.1 方式一：先激活环境再运行

```bash
conda activate causal_env
cd /Users/lunalhx/Desktop/202509/CiTransformer
python scripts/run_lstm.py --seq_len 96 --pred_len 1 --batch_size 256 --hidden_size 128 --num_layers 2 --dropout 0.1 --epochs 30 --patience 8
```

### 6.2 方式二：直接指定 conda 环境运行

```bash
conda run -n causal_env python /Users/lunalhx/Desktop/202509/CiTransformer/scripts/run_lstm.py --seq_len 96 --pred_len 1 --batch_size 256 --hidden_size 128 --num_layers 2 --dropout 0.1 --epochs 30 --patience 8
```

---

## 7. 我已经做过的验证

我已经在你的 Conda 环境 `causal_env` 中实际执行过一次小规模 smoke test，说明代码不是只写出来而已，而是已经跑通。

验证命令示例：

```bash
conda run -n causal_env python scripts/run_lstm.py --epochs 1 --seq_len 24 --pred_len 1 --batch_size 128 --hidden_size 32 --num_layers 1 --dropout 0.0 --max_train_batches 2 --max_eval_batches 1 --results_dir results/lstm_smoke --checkpoint_path checkpoints/lstm/smoke_model.pth
```

该测试已经成功生成：

- `results/lstm_smoke/metrics.json`
- `results/lstm_smoke/predictions.csv`
- `results/lstm_smoke/pred_plot.png`
- `checkpoints/lstm/smoke_model.pth`

这说明以下流程均已验证通过：

- 数据读取
- 时间列解析
- segment 划分
- DataLoader 构造
- LSTM 前向传播
- 训练与验证
- checkpoint 保存
- test 推理
- 结果文件写出

---

## 8. 当前文档生成位置

本说明文档生成在以下位置：

- 相对路径：`docs/lstm_baseline_work_summary.md`
- 绝对路径：`/Users/lunalhx/Desktop/202509/CiTransformer/docs/lstm_baseline_work_summary.md`

---

## 9. 当前结论

截至目前，本项目中的 **LSTM baseline 实验代码已经完成**，并满足以下核心要求：

- 基于连续时间片段构造样本
- 严格避免跨断点滑窗
- 严格避免 scaler 数据泄漏
- 不使用 HMM / PCMCI / 因果图 / 因果 mask
- 仅使用通用预处理后的 11 个特征
- calibration 集不参与训练
- 能在你的 `causal_env` 环境中运行

如果后续需要，我可以继续补两类内容：

1. 再单独写一份 `LSTM baseline 使用说明` 文档，专门面向论文实验复现
2. 继续补 `vanilla iTransformer baseline` 的统一训练入口，和当前 LSTM 使用相同数据接口与评估接口
