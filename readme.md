# CiTransformer — 因果掩码 iTransformer 时间序列预测

本项目在 **iTransformer** 模型基础上引入 **PCMCI 因果发现算法**，用因果掩码替代原始的全连接注意力机制，实现更具可解释性的时间序列预测。

---

## ⚡ 快速开始 — 环境部署（新电脑上从零配置）

### 第一步：确认 Python 版本

**推荐使用 Python 3.10.x**（例如 3.10.12 或 3.10.15）

> **为什么是 3.10？**
> - PyTorch 2.x 对 3.10 的支持最稳定
> - tigramite (PCMCI) 在 3.10/3.11 上测试最充分
> - 3.12+ 部分科学计算库仍有兼容性问题，不推荐
> - 3.9 太老，3.11 也可以但不如 3.10 稳定

**如何查看当前 Python 版本：**

打开终端（Mac: `Terminal` 或 `iTerm2`），输入：
```bash
python --version
# 或
python3 --version
```

如果版本不对，去官网下载安装: https://www.python.org/downloads/

---

### 第二步：在 PyCharm 中创建虚拟环境

1. 打开 PyCharm，进入 **File → Settings（Mac 是 PyCharm → Preferences）**
2. 左侧菜单点击 **Project: CiTransformer → Python Interpreter**
3. 点击右上角齿轮图标 ⚙ → **Add Interpreter → Add Local Interpreter**
4. 选择 **Virtualenv Environment**
5. 在 `Base interpreter` 处，选择你安装的 **Python 3.10.x** 路径
6. 点击 **OK**，PyCharm 会自动创建虚拟环境（在项目目录下的 `.venv` 文件夹）

**如何确认 PyCharm 当前使用的 Python 版本：**
- 右下角状态栏会显示当前解释器版本，例如 `Python 3.10.12 (venv)`
- 也可以在 PyCharm 的 **Terminal** 里输入 `python --version` 查看

---

### 第三步：安装所有依赖包

在 PyCharm 底部点击 **Terminal** 标签，确保虚拟环境已激活（提示符前面有 `(venv)` 字样），然后运行：

```bash
pip install -r requirements.txt
```

> 这一步会自动安装所有需要的库，耐心等待即可（首次安装约需 5~10 分钟）。

---

### 第四步（可选）：GPU 加速（如果你的电脑有 NVIDIA 显卡）

`requirements.txt` 里默认安装的是 CPU 版 PyTorch，速度较慢。如果你有 NVIDIA GPU：

1. 先卸载 CPU 版：
   ```bash
   pip uninstall torch torchvision
   ```
2. 去 PyTorch 官网获取对应 CUDA 版本的安装命令：
   https://pytorch.org/get-started/locally/
3. 根据你的 CUDA 版本选择对应命令安装（例如 CUDA 12.1）：
   ```bash
   pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
   ```

> **Mac M1/M2/M3 用户**：不需要安装 CUDA，PyTorch 会自动使用 Apple MPS 加速。

---

### 第五步：验证安装是否成功

在 PyCharm Terminal 中输入：

```bash
python -c "import torch; import tigramite; import pandas; print('全部安装成功！'); print('PyTorch版本:', torch.__version__)"
```

如果输出 `全部安装成功！` 说明环境配置完毕。

---

## 跨机器配置

项目运行路径和基础环境通过 YAML + 环境变量统一管理：

```bash
python -m utils.project_config get paths.data_dir
python -m utils.project_config get runtime.device
```

默认配置在 `configs/default.yaml`，每台机器可以复制模板并写自己的本地配置：

```bash
cp configs/local.example.yaml configs/local.yaml
```

`configs/local.yaml` 不提交到 git。也可以用环境变量临时覆盖：

```bash
CITRANSFORMER_DATA_DIR=/mnt/data/pv \
CITRANSFORMER_RESULTS_ROOT=/mnt/experiments/results \
CITRANSFORMER_DEVICE=cuda \
bash scripts/run_lstm_experiments.sh
```

优先级为：CLI 参数 > 环境变量 > `configs/local.yaml` > `configs/default.yaml` > 代码保底默认值。所有相对路径都会按项目根目录解析。

---

## 📁 项目结构

```
CiTransformer/
├── configs/              # 共享默认配置和本机配置模板
├── data/
│   ├── raw/               # 原始 CSV 数据（如 electricity.csv）
│   └── processed/         # 处理后的平稳数据 + 邻接矩阵
│
├── checkpoints/           # 训练好的模型权重（.pth 文件）
│
├── results/
│   ├── causal_graphs/     # PCMCI 生成的四季因果图 + Mask 矩阵
│   └── prediction_plots/  # 预测结果对比图
│
├── models/                # iTransformer 核心架构代码
│   ├── iTransformer.py
│   └── Embed.py
│
├── layers/                # 因果注意力层（核心改动在这里）
│   ├── SelfAttention.py   # 引入因果掩码的 Attention
│   └── Transformer_EncDec.py
│
├── causal_algo/           # 因果发现算法
│   ├── run_seasonal_pcmci.py   # 跑 PCMCI，生成各季节因果 Mask
│   └── run_seasonal_pcmci_v2.py
│
├── preprocessing/         # 数据预处理脚本
│
├── utils/                 # 工具库
│   ├── metrics.py         # 评价指标（MAE, MSE, RMSE）
│   ├── timefeatures.py    # 时间特征编码
│   └── tools.py           # ADF 平稳性检验等工具
│
├── scripts/               # 实验启动脚本
│
├── sandbox/               # 临时测试代码
│
├── run.py                 # 程序主入口
└── requirements.txt       # 环境依赖（pip install -r 安装）
```

---

## 🔧 常见问题

**Q: `pip install` 速度很慢怎么办？**

换国内镜像源：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**Q: 提示 `tigramite` 安装失败怎么办？**

先单独安装：
```bash
pip install tigramite -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**Q: 如何把我的虚拟环境也迁移到新电脑？**

不需要复制 `.venv` 文件夹（体积很大且平台相关），直接在新电脑上重做第一步到第三步即可，`requirements.txt` 保证了版本一致。
