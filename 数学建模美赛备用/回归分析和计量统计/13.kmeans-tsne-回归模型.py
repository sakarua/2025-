import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为 SimHei（黑体）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 加载 Wine 数据集
wine = load_wine()
X = wine.data[:, :3]  # 只选择前三个特征
y = wine.target  # 输出标签
feature_names = wine.feature_names[:3]  # 只选择前三个特征名称

# 将数据转换为 DataFrame
data = pd.DataFrame(X, columns=feature_names)
data['Target'] = y  # 添加目标标签

# 2. 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. KMeans 聚类
n_clusters = 3  # Wine 数据集有 3 个类别
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# 将聚类结果加入到原始数据中
data['Cluster'] = cluster_labels

# 4. t-SNE 降维并可视化
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
for i in range(n_clusters):
    plt.scatter(X_tsne[cluster_labels == i, 0], X_tsne[cluster_labels == i, 1], label=f'Cluster {i + 1}')
plt.title('t-SNE 降维结果（前三个特征）')
plt.xlabel('t-SNE 维度 1')
plt.ylabel('t-SNE 维度 2')
plt.legend()
plt.show()

# 5. 线性回归模型
regression_models = {}
mse_scores = {}

for i in range(n_clusters):
    # 获取属于当前簇的数据
    cluster_data = data[data['Cluster'] == i]
    X_cluster = cluster_data.iloc[:, :-2].values  # 排除目标列和聚类列
    y_cluster = cluster_data.iloc[:, -2].values  # 目标标签

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_cluster, y_cluster, test_size=0.2, random_state=42)

    # 训练线性回归模型
    reg = LinearRegression()
    reg.fit(X_train, y_train)

    # 预测和评估
    y_pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # 保存模型和评估结果
    regression_models[f'Cluster_{i + 1}'] = reg
    mse_scores[f'Cluster_{i + 1}'] = mse

    # 输出该类模型的 MSE
    print(f"Cluster {i + 1} 的线性回归模型的 MSE：{mse:.4f}")

# 输出每个类别的 MSE 评分
print("每个类别的 MSE 评分：", mse_scores)

# 6. 输出聚类结果和线性回归模型系数
for i in range(n_clusters):
    print(f"\nCluster {i + 1} 的线性回归模型系数：")
    print(regression_models[f'Cluster_{i + 1}'].coef_)
    print(f"Cluster {i + 1} 的截距：{regression_models[f'Cluster_{i + 1}'].intercept_:.4f}")

# 7. 保存聚类结果为 Excel 文件
output_path = 'wine_kmeans_output.xlsx'
data.to_excel(output_path, index=False)
print(f"\n聚类结果已保存到：{output_path}")

# 8. 生成新数据并预测
# 假设我们生成一些新数据（基于前三个特征的分布）
np.random.seed(42)  # 固定随机种子以确保可重复性
new_data = np.random.rand(10, 3) * 10  # 生成 10 个样本，每个样本有 3 个特征，范围在 [0, 10)
new_data = scaler.transform(new_data)  # 使用之前的标准化器对新数据进行标准化

# 将新数据分配到对应的簇
new_cluster_labels = kmeans.predict(new_data)

# 预测和计算相对误差
predictions = {}
relative_errors = {}

for i in range(n_clusters):
    # 获取属于当前簇的新数据
    cluster_mask = (new_cluster_labels == i)
    if np.sum(cluster_mask) > 0:  # 如果有数据属于当前簇
        X_new_cluster = new_data[cluster_mask]

        # 使用对应的线性回归模型进行预测
        y_pred = regression_models[f'Cluster_{i + 1}'].predict(X_new_cluster)

        # 假设真实值（这里随机生成，实际应用中应从真实数据中获取）
        y_true = np.random.rand(len(X_new_cluster)) * 10  # 随机生成真实值

        # 计算相对误差
        relative_error = np.abs((y_true - y_pred) / y_true) * 100  # 百分比形式

        # 保存预测结果和相对误差
        predictions[f'Cluster_{i + 1}'] = y_pred
        relative_errors[f'Cluster_{i + 1}'] = relative_error

        # 输出预测值和相对误差（中文说明，列表对比方式）
        print(f"\n簇 {i + 1} 的预测结果：")
        print("+-----------+-----------+-----------+----------------+")
        print("|   样本编号 |   预测值   |   真实值   |   相对误差（%）  |")
        print("+-----------+-----------+-----------+----------------+")
        for j in range(len(X_new_cluster)):
            print(
                f"|     {j + 1:2d}     |   {y_pred[j]:.4f}  |   {y_true[j]:.4f}  |      {relative_error[j]:.2f}      |")
        print("+-----------+-----------+-----------+----------------+")

# 输出所有预测结果和相对误差
print("\n所有簇的预测结果：", predictions)
print("所有簇的相对误差：", relative_errors)