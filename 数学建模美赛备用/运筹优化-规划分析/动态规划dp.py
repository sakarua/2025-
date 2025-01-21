import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('TkAgg')
# 加载wine数据集
wine = load_wine()
data = wine.data[:, :4]  # 取前四个特征
features = wine.feature_names[:4]
target = wine.target

# 数据标准化
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

# 熵权法计算权重
def entropy_weight(data):
    # 计算熵值
    epsilon = 1e-10  # 避免log(0)
    p = data / np.sum(data, axis=0)
    entropy = -np.sum(p * np.log(p + epsilon), axis=0)
    # 计算权重
    weight = (1 - entropy) / np.sum(1 - entropy)
    return weight

# 计算权重
weights = entropy_weight(data_normalized)
print("熵权法计算的权重：", weights)

# TOPSIS法
def topsis(data, weights):
    # 加权标准化矩阵
    weighted_matrix = data * weights
    # 理想解和负理想解
    ideal_best = np.max(weighted_matrix, axis=0)
    ideal_worst = np.min(weighted_matrix, axis=0)
    # 计算距离
    distance_best = np.sqrt(np.sum((weighted_matrix - ideal_best) ** 2, axis=1))
    distance_worst = np.sqrt(np.sum((weighted_matrix - ideal_worst) ** 2, axis=1))
    # 计算评分
    score = distance_worst / (distance_best + distance_worst)
    return score

# 计算TOPSIS评分
topsis_scores = topsis(data_normalized, weights)
print("TOPSIS评分：", topsis_scores)

# 动态规划逐个特征求解最优解
def dynamic_programming(data, scores):
    n_samples, n_features = data.shape
    dp = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        for j in range(n_features):
            if j == 0:
                dp[i][j] = data[i][j] * scores[i]
            else:
                dp[i][j] = dp[i][j - 1] + data[i][j] * scores[i]
    return dp

# 动态规划结果
dp_result = dynamic_programming(data_normalized, topsis_scores)
print("动态规划结果：")
print(dp_result)

# 可视化分类结果
plt.figure(figsize=(10, 6))
for i in range(len(features)):
    plt.subplot(2, 2, i + 1)
    plt.scatter(range(len(data)), dp_result[:, i], c=target, cmap='viridis')
    plt.title(f'特征 {features[i]} 的动态规划结果')
    plt.xlabel('样本')
    plt.ylabel('动态规划值')
plt.tight_layout()
plt.show()

# 输出中文内容
print("逐步动态规划的效果：")
for i, feature in enumerate(features):
    print(f"特征 {feature} 的动态规划结果：")
    print(dp_result[:, i])