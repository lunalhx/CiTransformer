import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite import data_processing as pp

# ==========================================
# 0. 配置
# ==========================================
INPUT_FILE = '../data/processed/spain_electricity_cleaned_lat.csv'
OUTPUT_DIR = '../results/causal_graphs/'
ALPHA_LEVEL = 0.01
MAX_LAG = 1
TOP_K = 3  # 【核心修改】强制每个变量最多只保留 3 个最强父节点

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# 1. 准备数据
# ==========================================
print("读取数据...")
df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
selected_columns = ['load_fc', 'solar_fc', 'wind_on_fc', 'price_lag_24', 'price_lag_168', 'price']
data_core = df[selected_columns]
var_names = data_core.columns.tolist()


# ==========================================
# 2. 辅助函数
# ==========================================
def get_season_data(df_full, season_name):
    seasons = {'Spring': [3, 4, 5], 'Summer': [6, 7, 8],
               'Autumn': [9, 10, 11], 'Winter': [12, 1, 2]}
    return df_full[df_full.index.month.isin(seasons[season_name])]


# ==========================================
# 3. 核心：Top-K 强力筛选
# ==========================================
def run_pcmci_topk(df_season, season_name):
    print(f"\n>>> 正在分析季节: {season_name}")

    # 1. 运行 PCMCI
    dataframe = pp.DataFrame(df_season.values, datatime=np.arange(len(df_season)), var_names=var_names)
    parcorr = ParCorr(significance='analytic')
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)
    results = pcmci.run_pcmci(tau_max=MAX_LAG, pc_alpha=ALPHA_LEVEL)

    # 2. 获取 P值矩阵 和 强度矩阵(val_matrix)
    # val_matrix 代表相关性的绝对强度 (0~1)
    pval_matrix = results['p_matrix'][:, :, 0]
    val_matrix = np.abs(results['val_matrix'][:, :, 0])  # 取绝对值

    # 3. 构建 Top-K Mask
    N = len(var_names)
    adj_matrix = np.zeros((N, N), dtype=int)

    # 对每一个子节点(j)，只保留 val_matrix 最大的 TOP_K 个父节点(i)
    # 且前提是 p_value 必须显著 (< ALPHA)
    for j in range(N):
        # 获取第 j 列的所有父节点强度
        parents_strength = val_matrix[:, j]
        parents_pval = pval_matrix[:, j]

        # 找出显著的索引
        sig_indices = np.where(parents_pval < ALPHA_LEVEL)[0]

        # 如果显著的数量超过 Top-K，则进行裁剪
        if len(sig_indices) > TOP_K:
            # 在显著的里面挑强度最大的 Top-K
            # argsort 返回从小到大的索引，取最后 TOP_K 个
            top_k_idx = parents_strength[sig_indices].argsort()[-TOP_K:]
            final_parents = sig_indices[top_k_idx]
        else:
            final_parents = sig_indices

        # 填充邻接矩阵
        adj_matrix[final_parents, j] = 1

    # 4. 后处理约束
    np.fill_diagonal(adj_matrix, 1)  # 自己连自己
    idx_price = 5
    adj_matrix[idx_price, 0:5] = 0  # 禁止 Price 指向别人

    return adj_matrix


# ==========================================
# 4. 运行并保存
# ==========================================
for season in ['Spring', 'Summer', 'Autumn', 'Winter']:
    df_s = get_season_data(data_core, season)
    mask = run_pcmci_topk(df_s, season)

    # 保存
    np.save(os.path.join(OUTPUT_DIR, f'mask_{season}.npy'), mask)

    # 打印 Price 的 Top-3 父节点
    price_parents = [var_names[i] for i in range(len(var_names)) if mask[i, 5] == 1 and i != 5]
    print(f"  -> {season} Price 的 Top-3 强驱动因子: {price_parents}")

    # 绘图
    plt.figure(figsize=(8, 6))
    sns.heatmap(mask, annot=True, cmap="Blues", xticklabels=var_names, yticklabels=var_names, cbar=False)
    plt.title(f'Top-{TOP_K} Causal Mask - {season}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'heatmap_{season}_topk.png'))
    plt.close()

print("\n优化完成。现在你的图应该是稀疏的了。")