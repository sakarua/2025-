import numpy as np
from sklearn.datasets import load_wine
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 加载 Wine 数据集
wine = load_wine()
X = wine.data[:, :3]  # 选择前三个特征
feature_names = wine.feature_names[:3]  # 特征名称

# 2. 数据标准化
def normalize(data):
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

X_normalized = normalize(X)

# 3. 熵权法计算权重
def entropy_weight(data):
    # 计算熵值
    epsilon = 1e-12  # 避免 log(0)
    p = data / np.sum(data, axis=0)  # 计算概率
    entropy = -np.sum(p * np.log(p + epsilon), axis=0) / np.log(len(data))  # 计算熵值
    # 计算权重
    weight = (1 - entropy) / np.sum(1 - entropy)
    return weight

weights = entropy_weight(X_normalized)
print("熵权法计算的权重：", weights)

# 4. TOPSIS法计算得分
def topsis(data, weight):
    # 加权标准化
    weighted_data = data * weight
    # 正理想解和负理想解
    ideal_best = np.max(weighted_data, axis=0)
    ideal_worst = np.min(weighted_data, axis=0)
    # 计算距离
    dist_best = np.sqrt(np.sum((weighted_data - ideal_best) ** 2, axis=1))
    dist_worst = np.sqrt(np.sum((weighted_data - ideal_worst) ** 2, axis=1))
    # 计算 TOPSIS 得分
    score = dist_worst / (dist_best + dist_worst)
    return score

topsis_scores = topsis(X_normalized, weights)

# 5. 多目标规划问题
def multi_objective(x, weights, topsis_weight):
    # 目标 1：最小化特征的平方和
    f1 = np.sum(x**2)
    # 目标 2：最大化酒精含量
    f2 = -x[0]  # 负号表示最大化
    # 加权求和
    return weights[0] * f1 + weights[1] * f2

# 6. 定义约束条件
def constraint(x):
    return 15 - np.sum(x)  # x1 + x2 + x3 <= 15

# 7. 定义变量范围
# 根据酒精含量的实际范围调整变量范围
alcohol_min = np.min(X[:, 0])  # 酒精含量的最小值
alcohol_max = np.max(X[:, 0])  # 酒精含量的最大值
malic_acid_min = np.min(X[:, 1])  # 苹果酸的最小值
malic_acid_max = np.max(X[:, 1])  # 苹果酸的最大值
ash_min = np.min(X[:, 2])  # 灰分的最小值
ash_max = np.max(X[:, 2])  # 灰分的最大值

bounds = [
    (alcohol_min, alcohol_max),  # 酒精含量的范围
    (malic_acid_min, malic_acid_max),  # 苹果酸的范围
    (ash_min, ash_max)  # 灰分的范围
]

# 8. 定义初始猜测
x0 = np.array([alcohol_max, malic_acid_min, ash_min])  # 初始猜测值

# 9. 定义约束字典
cons = {'type': 'ineq', 'fun': constraint}

# 10. 定义权重
topsis_weight = topsis_scores.mean()  # 使用 TOPSIS 得分的均值作为权重

# 11. 求解多目标问题
res = minimize(
    multi_objective,  # 目标函数
    x0,  # 初始猜测
    args=(weights, topsis_weight),  # 权重参数
    method='SLSQP',  # 使用 SLSQP 算法
    bounds=bounds,  # 变量范围
    constraints=cons  # 约束条件
)

# 12. 输出最优解
print("最优解：", res.x)
print("目标函数值：", res.fun)

# 13. 绘制 TOPSIS 得分分布
plt.figure(figsize=(10, 6))
plt.hist(topsis_scores, bins=20, color='blue', alpha=0.7, label='TOPSIS 得分分布')
plt.axvline(topsis_weight, color='red', linestyle='--', label='TOPSIS 权重')
plt.xlabel('TOPSIS 得分')
plt.ylabel('频数')
plt.title('TOPSIS 得分分布及权重')
plt.legend()
plt.grid()
plt.tight_layout()

# 保存图像
plt.savefig('topsis_distribution.png')  # 保存图像为文件
print("图像已保存为 topsis_distribution.png")

# 显示图像（如果环境支持）
plt.show()