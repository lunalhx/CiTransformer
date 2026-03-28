import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite import data_processing as pp

# ==========================================
# 0. 配置路径与参数
# ==========================================
INPUT_FILE = '../data/processed/spain_electricity_cleaned_lat.csv'
OUTPUT_DIR = '../results/causal_graphs/'
ALPHA_LEVEL = 0.01  # 【极度重要】设为 0.01 或 0.001 以获得稀疏图。默认0.05太宽松！
MAX_LAG = 1  # 因为我们已经手动构造了 lag 特征，这里设为 1 仅用于检测瞬时关系

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# 1. 准备数据
# ==========================================
print("读取清洗后的数据...")
df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)

# 挑选用于因果发现的变量
# 注意：我们不把 sin/cos 时间编码放进去跑因果，因为时间是“公理”，不参与因果竞争。
# 我们只看物理变量和历史变量之间的博弈。
selected_columns = [
    'load_fc',
    'solar_fc',
    'wind_on_fc',
    'price_lag_24',
    'price_lag_168',
    'price'  # Target 必须放最后方便观察
]

data_core = df[selected_columns]
var_names = data_core.columns.tolist()
print(f"参与因果跑分的变量: {var_names}")


# ==========================================
# 2. 定义季节切分函数
# ==========================================
def get_season_data(df_full, season_name):
    # 月份映射
    seasons = {
        'Spring': [3, 4, 5],
        'Summer': [6, 7, 8],
        'Autumn': [9, 10, 11],
        'Winter': [12, 1, 2]
    }
    target_months = seasons[season_name]
    # 筛选月份
    return df_full[df_full.index.month.isin(target_months)]


# ==========================================
# 3. 核心：运行 PCMCI 并施加禁止规则
# ==========================================
def run_pcmci(df_season, season_name):
    print(f"\n>>> 正在分析季节: {season_name} (样本数: {len(df_season)})")

    # 转换为 Tigramite 格式
    dataframe = pp.DataFrame(df_season.values, datatime=np.arange(len(df_season)), var_names=var_names)

    # 初始化条件独立性测试 (ParCorr 适合连续变量)
    parcorr = ParCorr(significance='analytic')
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)

    # ------------------------------------------------
    # 【导师介入】定义禁止连接 (Tiered Constraints)
    # ------------------------------------------------
    # 逻辑：Price (结果) 绝对不能反向导致 Forecast (预测)
    # 我们创建一个 link_assumptions 矩阵
    # N = 变量数.
    # link_assumptions[i, j] = --?--
    # 这里的设置比较复杂，我们使用更直接的方法：
    # 跑完后手动把非法的边 mask 掉，或者只关注指向 Price 的边。

    # 运行 PCMCI
    results = pcmci.run_pcmci(tau_max=MAX_LAG, pc_alpha=ALPHA_LEVEL)

    # 提取 p-values 和 强度矩阵 (val_matrix)
    # matrix 维度: [N, N, tau_max+1]
    # 我们主要关心 lag0 (同期) 的关系，因为 lag 特征已经作为列存在了
    pval_matrix = results['p_matrix'].round(3)
    val_matrix = results['val_matrix'].round(3)

    # ==========================================
    # 4. 生成 Mask 矩阵 (用于 Transformer)
    # ==========================================
    # 规则：如果 p_value < ALPHA, 则置为 1 (有连线)，否则 0
    # 取 lag=0 的切片 (因为我们的数据行已经是针对同一时刻的对齐)
    adj_matrix = (pval_matrix[:, :, 0] < ALPHA_LEVEL).astype(int)

    # 【后处理修正】：
    # 强制将对角线设为 1 (自己对自己有关)
    np.fill_diagonal(adj_matrix, 1)

    # 【强制物理约束】：Price 不能指向 Load/Solar/Wind (因果倒置)
    # var_names 索引:
    # 0:load, 1:solar, 2:wind, 3:lag24, 4:lag168, 5:price
    idx_price = 5
    # 让 Price 指向别人的概率强行归零 (除非你想做博弈论分析)
    adj_matrix[idx_price, 0:5] = 0

    return adj_matrix, val_matrix[:, :, 0]


# ==========================================
# 5. 主循环与画图
# ==========================================
for season in ['Spring', 'Summer', 'Autumn', 'Winter']:
    df_s = get_season_data(data_core, season)
    mask, strength = run_pcmci(df_s, season)

    # 保存 Mask (给 iTransformer 用)
    np.save(os.path.join(OUTPUT_DIR, f'mask_{season}.npy'), mask)

    # 画热力图 (给你写论文用)
    plt.figure(figsize=(10, 8))
    sns.heatmap(mask, annot=True, cmap="Blues", xticklabels=var_names, yticklabels=var_names, cbar=False)
    plt.title(f'Causal Mask - {season} (Alpha={ALPHA_LEVEL})')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'heatmap_{season}.png'))
    plt.close()

    print(f"  -> Mask 已保存: mask_{season}.npy")
    # 打印 Price 的父节点
    price_parents = [var_names[i] for i in range(len(var_names)) if mask[i, 5] == 1 and i != 5]
    print(f"  -> 发现 Price 的直接父节点: {price_parents}")

print("\n全部完成！请检查 results/causal_graphs/ 文件夹。")