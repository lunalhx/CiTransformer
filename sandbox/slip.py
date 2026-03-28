import pandas as pd
import matplotlib.pyplot as plt
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io

# ===========================
# 1. 读取数据
# ===========================
# 根据你的截图，文件就在 ETT-small 文件夹下
file_path = '../data/raw/ETT-small/ETTh1.csv'
try:
    df = pd.read_csv(file_path)
    print("✅ 数据读取成功！")
except FileNotFoundError:
    # 备用路径，防止你确实有个 dataset 文件夹没截图出来
    file_path = 'dataset/ETT-small/ETTh1.csv'
    df = pd.read_csv(file_path)
    print("✅ 数据读取成功 (使用备用路径)！")

# ===========================
# 2. 聪明的切分：按月份
# ===========================
# 必须把 date 列转成时间格式，这样我们就知道哪个月是夏天，哪个月是冬天
df['date'] = pd.to_datetime(df['date'])

# 定义季节 (北半球)
# 夏季：6月, 7月, 8月
summer_mask = df['date'].dt.month.isin([6, 7, 8])
# 冬季：12月, 1月, 2月
winter_mask = df['date'].dt.month.isin([12, 1, 2])

# 提取数据
df_summer = df[summer_mask].copy()
df_winter = df[winter_mask].copy()

# ===========================
# 3. 清洗数据 (只保留数值列给因果算法用)
# ===========================
# 丢掉 'date' 列，保留后面 7 列数值 (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT)
data_summer = df_summer.iloc[:, 1:]
data_winter = df_winter.iloc[:, 1:]

# 打印一下形状，让你心里有底
print(f"\n📊 夏季数据量: {data_summer.shape} (行, 列)")
print(f"❄️ 冬季数据量: {data_winter.shape} (行, 列)")

# ===========================
# 4. (可选) 导师带你做个快速验证
# ===========================
# 我们画一下 'OT' (油温) 这一列的前200个点
# 理论上夏天的油温应该比冬天高，如果图是对的，说明切分成功
plt.figure(figsize=(10, 4))
plt.plot(data_summer['OT'].values[:200], label='Summer OT (Hot)', color='red', alpha=0.7)
plt.plot(data_winter['OT'].values[:200], label='Winter OT (Cold)', color='blue', alpha=0.7)
plt.title("Check: Oil Temperature (Summer vs Winter)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ===========================
# 5. 下一步准备
# ===========================
# 这些数据 (data_summer, data_winter) 就可以直接喂给 PC 算法了
# 比如: pc(data_summer.to_numpy())

# ===========================
# 1. 定义一个画图函数 (方便复用)
# ===========================
def run_pc_and_draw(data, title, labels):
    print(f"🚀 正在计算 {title} 的因果图...")
    # 运行 PC 算法
    cg = pc(data.to_numpy())

    # 可视化设置
    pyd = GraphUtils.to_pydot(cg.G, labels=labels)
    # 保存为临时图片以便显示
    tmp_png = pyd.create_png(f="png")

    # 在 Notebook 或 IDE 里显示出来
    img = mpimg.imread(io.BytesIO(tmp_png))
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title(title)
    plt.show()
    return cg


# ===========================
# 2. 准备变量名 (给节点起名字)
# ===========================
# 获取列名，比如 ['HUFL', 'HULL', ..., 'OT']
labels = data_summer.columns.tolist()

# ===========================
# 3. 分别跑夏天和冬天
# ===========================
# 跑夏天
cg_summer = run_pc_and_draw(data_summer, "Summer Causal Graph (Hot)", labels)

# 跑冬天
cg_winter = run_pc_and_draw(data_winter, "Winter Causal Graph (Cold)", labels)