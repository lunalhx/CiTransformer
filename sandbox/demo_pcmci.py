import numpy as np
import pandas as pd
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

# 1. 造一点假数据 (3个变量: var0, var1, var2)
# 假设逻辑：var0 影响 var1 (滞后-1)，var1 影响 var2 (滞后-1)
T = 1000
data = np.random.randn(T, 3)
data[:, 1] += 0.8 * data[:, 0]  # var0 -> var1 (同期相关，模拟强关联)
# 注意：真实因果通常带滞后，为了简单演示先这样写

# 2. 初始化 dataframe
dataframe = pp.DataFrame(data, var_names=["var0", "var1", "var2"])

# 3. 设置因果发现算法 (PCMCI)
# ParCorr 是偏相关系数检验，适合线性关系，速度快
parcorr = ParCorr(significance='analytic')
pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr)

# 4. 运行算法 (最大滞后设为2)
results = pcmci.run_pcmci(tau_max=2, pc_alpha=None)

# 5. 打印 p-value 矩阵 (p值越小，因果越显著)
print("\n=== P-Value Matrix (p_matrix) ===")
print(results['p_matrix'].round(3))

# 6. 打印 邻接矩阵 (Adjacency Matrix)
# 我们设定一个阈值 0.05，小于它的就是有因果 (True/1)，否则是无因果 (False/0)
adj_matrix = results['p_matrix'] < 0.05
print("\n=== Adjacency Matrix (Mask) ===")
# 只看 lag=1 的切片 (假设我们只关心上一时刻的影响)
print(adj_matrix[:, :, 1].astype(int))