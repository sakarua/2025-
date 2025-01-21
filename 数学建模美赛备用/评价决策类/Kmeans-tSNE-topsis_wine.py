import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('TkAgg')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为 SimHei（黑体）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 加载 Wine 数据集
wine = load_wine()
X = wine.data  # 输入特征
y = wine.target  # 真实标签（用于对比）
feature_names = wine.feature_names  # 特征名称

# 2. 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. KMeans 聚类
n_clusters = 3  # Wine 数据集有 3 个类别
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# 将聚类结果加入到原始数据中
data = pd.DataFrame(X, columns=feature_names)
data['Cluster'] = cluster_labels
data['True_Label'] = y  # 添加真实标签用于对比

# 4. t-SNE 降维并可视化（二维）
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
for i in range(n_clusters):
    plt.scatter(X_tsne[cluster_labels == i, 0], X_tsne[cluster_labels == i, 1], label=f'Cluster {i + 1}')
plt.title('t-SNE 降维结果（二维）')
plt.xlabel('t-SNE 维度 1')
plt.ylabel('t-SNE 维度 2')
plt.legend()
plt.show()

# 5. PCA 降维并可视化（三维）
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
for i in range(n_clusters):
    ax.scatter(X_pca[cluster_labels == i, 0], X_pca[cluster_labels == i, 1], X_pca[cluster_labels == i, 2], label=f'Cluster {i + 1}')
ax.set_title('PCA 降维结果（三维）')
ax.set_xlabel('PCA 维度 1')
ax.set_ylabel('PCA 维度 2')
ax.set_zlabel('PCA 维度 3')
ax.legend()
plt.show()

# 6. 分析酒的种类、度数、杂质度等信息
# 假设特征中：
# - "alcohol" 表示酒的度数
# - "ash" 表示杂质度
# - "magnesium" 表示镁含量

# 提取相关特征
alcohol = data['alcohol']
ash = data['ash']
magnesium = data['magnesium']

# 可视化酒的度数、杂质度和镁含量
plt.figure(figsize=(14, 6))

# 子图1：酒的度数分布
plt.subplot(1, 3, 1)
for i in range(n_clusters):
    plt.hist(alcohol[cluster_labels == i], bins=20, alpha=0.5, label=f'Cluster {i + 1}')
plt.title('酒的度数分布')
plt.xlabel('度数')
plt.ylabel('频数')
plt.legend()

# 子图2：酒的杂质度分布
plt.subplot(1, 3, 2)
for i in range(n_clusters):
    plt.hist(ash[cluster_labels == i], bins=20, alpha=0.5, label=f'Cluster {i + 1}')
plt.title('酒的杂质度分布')
plt.xlabel('杂质度')
plt.ylabel('频数')
plt.legend()

# 子图3：酒的镁含量分布
plt.subplot(1, 3, 3)
for i in range(n_clusters):
    plt.hist(magnesium[cluster_labels == i], bins=20, alpha=0.5, label=f'Cluster {i + 1}')
plt.title('酒的镁含量分布')
plt.xlabel('镁含量')
plt.ylabel('频数')
plt.legend()

plt.tight_layout()
plt.show()

# 7. 输出聚类结果
print("聚类结果：")
print(data.head())

# 保存聚类结果为 Excel 文件
output_dir = 'D:\python\PythonLearning_Basis'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, 'wine_kmeans-tSNE-topsis_output.xlsx')

try:
    data.to_excel(output_path, index=False)
    print(f"聚类结果已保存到：{output_path}")
except PermissionError:
    print(f"无法保存文件，请检查路径权限或文件是否被占用：{output_path}")