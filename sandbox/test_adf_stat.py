import numpy as np
from statsmodels.tsa.stattools import adfuller

# 模拟 1：一个一直增长的数据（不平稳）
data_unstable = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
result1 = adfuller(data_unstable)
print(f"不平稳数据的 p-value: {result1[1]}")

# 模拟 2：一组随机波动的噪声（平稳）
data_stable = np.random.randn(100)
result2 = adfuller(data_stable)
print(f"平稳数据的 p-value: {result2[1]}")