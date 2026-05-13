# Scripts 目录说明

## 统一入口怎么用

实验类流程推荐统一从 `scripts/run_experiment.sh` 进入：

```bash
bash scripts/run_experiment.sh --help
```

基本格式是：

```bash
bash scripts/run_experiment.sh <任务名> [额外参数]
```

例如：

```bash
bash scripts/run_experiment.sh lstm
bash scripts/run_experiment.sh itransformer
bash scripts/run_experiment.sh global-pcmci-mask
```

`run_experiment.sh` 本身不写训练逻辑，它只负责按任务名分发到 `scripts/experiments/`、`scripts/train/`、`scripts/causal/` 下的真实脚本。运行 shell 入口时，Python 环境会按 `scripts/lib/project_config.sh` 自动解析，优先使用 `PYTHON_BIN` 环境变量，其次读取 `configs/local.yaml` 里的 `runtime.python_bin`。

当前本机配置建议使用：

```yaml
runtime:
  python_bin: /opt/anaconda3/envs/causal_env/bin/python
```

## 数据入口在哪里设置

长期固定的数据目录建议写在 `configs/local.yaml`：

```yaml
paths:
  data_dir: /你的/数据目录/processed_long_no_wind_2015_2022
```

脚本默认会在该目录下读取：

```text
splits/train.csv
splits/validation.csv
splits/test.csv
```

临时换一次数据集时，可以不改配置，直接在命令前加环境变量：

```bash
DATA_DIR=/你的/数据目录/processed_long_no_wind_2015_2022 bash scripts/run_experiment.sh lstm
```

也可以用项目配置环境变量：

```bash
CITRANSFORMER_DATA_DIR=/你的/数据目录/processed_long_no_wind_2015_2022 bash scripts/run_experiment.sh lstm
```

常用优先级：

```text
命令行环境变量 DATA_DIR
> CITRANSFORMER_DATA_DIR
> configs/local.yaml
> configs/default.yaml
```

## 批量运行和并行运行的区别

这里的“批量运行”通常指一次跑多个 `pred_len`，例如默认的 `1 12 24 48`。这些普通任务默认是顺序运行，不是并行：

```bash
bash scripts/run_experiment.sh lstm
bash scripts/run_experiment.sh persistence
bash scripts/run_experiment.sh itransformer
```

如果只想临时跑一个预测长度：

```bash
PRED_LENS=12 bash scripts/run_experiment.sh itransformer
```

真正并行运行多个 `pred_len` 的任务是 `parallel-pred-lens`：

```bash
bash scripts/run_experiment.sh parallel-pred-lens \
  --experiment itransformer \
  --pred-lens "1 12 24 48" \
  --max-parallel 4
```

并行运行会把日志写到 `logs/parallel_<experiment>`，具体目录可用 `--log-dir` 覆盖。

## 常用任务

| 我想做什么 | 推荐命令 | 输出位置 |
| --- | --- | --- |
| 顺序运行 LSTM 多个 `pred_len` | `bash scripts/run_experiment.sh lstm` | `paths.results.lstm` |
| 顺序运行 persistence 多个 `pred_len` | `bash scripts/run_experiment.sh persistence` | `paths.results.persistence` |
| 顺序运行 iTransformer 多个 `pred_len` | `bash scripts/run_experiment.sh itransformer` | `paths.results.itransformer` |
| 调参 iTransformer | `bash scripts/run_experiment.sh itransformer-tuning` | `paths.results.tuning_itransformer` |
| 导出 tuned iTransformer 测试结果 | `bash scripts/run_experiment.sh itransformer-tuned-test` | `paths.results.itransformer_tuned` |
| 构建或复用 global PCMCI 因果 mask | `bash scripts/run_experiment.sh global-pcmci-mask` | `paths.results.causal_graphs_global_pcmci_11vars_train` |
| 运行 global PCMCI masked iTransformer | `bash scripts/run_experiment.sh global-pcmci-itransformer` | `paths.results.itransformer_global_pcmci_11vars` |
| 运行 mask calibration 实验 | `bash scripts/run_experiment.sh mask-calibration` | `paths.results_root/itransformer_mask_calibration` |
| 发现 daytime regimes | `bash scripts/run_experiment.sh regime-discovery` | 脚本 `--output_dir` 默认值 |
| 运行 regime-conditioned PCMCI | `bash scripts/run_experiment.sh regime-pcmci` | 脚本 `--output_dir` 默认值 |
| 并行运行多个 `pred_len` 任务 | `bash scripts/run_experiment.sh parallel-pred-lens` | `logs/parallel_<experiment>` |

## 常见示例

```bash
# 查看所有任务
bash scripts/run_experiment.sh --help

# 跑 LSTM baseline，使用 configs/local.yaml/default.yaml 的默认数据目录和 pred_len
bash scripts/run_experiment.sh lstm

# 跑普通 iTransformer，但只跑 pred_len=12
PRED_LENS=12 bash scripts/run_experiment.sh itransformer

# 跑 global PCMCI mask，仅构建或检查因果图，不启动 iTransformer 训练
bash scripts/run_experiment.sh global-pcmci-mask

# 跑 global PCMCI masked iTransformer
bash scripts/run_experiment.sh global-pcmci-itransformer

# 临时换数据目录跑 persistence baseline
DATA_DIR=/你的/数据目录/processed_long_no_wind_2015_2022 bash scripts/run_experiment.sh persistence

# 并行跑 iTransformer 的多个 pred_len
bash scripts/run_experiment.sh parallel-pred-lens \
  --experiment itransformer \
  --pred-lens "1 12 24 48" \
  --max-parallel 4
```

顶层旧兼容 wrapper 已删除。实验类流程统一通过 `scripts/run_experiment.sh` 运行；只有在需要使用底层工具时，才直接调用分组目录下的真实实现。

## 目录结构

| 目录 | 用途 | 对外入口 |
| --- | --- | --- |
| `scripts/lib/` | 共享 shell 工具函数 | 内部 helper |
| `scripts/train/` | 单模型训练和评估 Python 入口 | `run_lstm.py`, `run_iTransformer.py`, `run_persistence.py` |
| `scripts/experiments/` | 批量实验编排 | 该目录下的 shell 脚本 |
| `scripts/causal/` | regime 发现、regime PCMCI、物理后处理 | 该目录下的 Python 脚本 |
| `scripts/reports/` | 结果汇总 | 汇总类 Python 脚本 |
| `scripts/ops/` | 归档、恢复、数据审计辅助工具 | 运维类脚本 |

## 直接入口

需要绕过任务分发器、直接调用底层 Python 或 shell 入口时，使用分组后的真实路径。直接运行 Python 脚本时请使用本机配置指定的环境：

```bash
/opt/anaconda3/envs/causal_env/bin/python scripts/train/run_lstm.py --help
/opt/anaconda3/envs/causal_env/bin/python scripts/train/run_iTransformer.py --help
/opt/anaconda3/envs/causal_env/bin/python scripts/train/run_persistence.py --help
/opt/anaconda3/envs/causal_env/bin/python scripts/causal/run_gmm_hmm_daytime_regimes.py --help
/opt/anaconda3/envs/causal_env/bin/python scripts/causal/run_regime_target_pcmci.py --help
/opt/anaconda3/envs/causal_env/bin/python scripts/reports/summarize_itransformer_tuning.py --help
```

Python import 也应使用分组后的模块路径：

```python
from scripts.train.run_lstm import prepare_datasets
```
