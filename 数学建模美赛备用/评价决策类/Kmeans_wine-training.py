import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为 SimHei（黑体）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 加载 Wine 数据集
wine = load_wine()
X = wine.data  # 输入特征
y = wine.target  # 真实标签（用于对比）

# 2. 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 使用 KMeans 聚类
n_clusters = 3  # Wine 数据集有 3 个类别
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_scaled)

# 获取聚类标签和簇中心
labels = kmeans.labels_  # 每个点的簇标签
centroids = kmeans.cluster_centers_  # 每个簇的质心

# 将聚类结果加入到原始数据中
data = pd.DataFrame(X, columns=wine.feature_names)
data['Cluster'] = labels
data['True_Label'] = y  # 添加真实标签用于对比

# 4. 可视化聚类结果
# 使用前两个特征进行可视化
plt.figure(figsize=(12, 6))

# 子图1：K-means 聚类结果
plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', marker='o', label='聚类结果')
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='x', label='簇中心')
plt.title('KMeans 聚类结果')
plt.xlabel('特征1（标准化后）')
plt.ylabel('特征2（标准化后）')
plt.legend()

# 子图2：真实标签对比
plt.subplot(1, 2, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis', marker='o', label='真实标签')
plt.title('真实标签分布')
plt.xlabel('特征1（标准化后）')
plt.ylabel('特征2（标准化后）')
plt.legend()

plt.tight_layout()
plt.show()

# 5. 输出聚类结果
print("聚类结果：")
print(data.head())

# 保存聚类结果为 Excel 文件
output_path = 'wine_kmeans_output.xlsx'
data.to_excel(output_path, index=False)
print(f"聚类结果已保存到：{output_path}")