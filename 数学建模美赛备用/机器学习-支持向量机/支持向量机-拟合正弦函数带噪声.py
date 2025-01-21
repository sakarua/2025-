import numpy as np
import pylab as plt
from sklearn.svm import SVR
import matplotlib
matplotlib.use('TkAgg')

# 设置随机种子
np.random.seed(123)

# 生成更多的数据点
x = np.linspace(0, 20 * np.pi, 1000).reshape(-1, 1)  # 从 0 到 20π，生成 1000 个点
y = (np.sin(x) + 3 + np.random.uniform(-1, 1, (1000, 1))).ravel()  # 添加噪声

# 创建 SVR 模型
model = SVR(gamma='auto', C=100)  # 调整 C 和 gamma 以更好地拟合数据
print(model)

# 训练模型
model.fit(x, y)

# 预测
pred_y = model.predict(x)

# 输出前 15 个值的对比
print("原始数据与预测值前15个值对比：")
for i in range(15):
    print(y[i], pred_y[i])

# 设置图形尺寸（横向拉长）
plt.figure(figsize=(15, 5))  # 宽度为 15，高度为 5

# 设置字体
plt.rc('font', family='SimHei')
plt.rc('font', size=15)

# 绘制原始数据和预测值
plt.scatter(x, y, s=5, color="blue", label="原始数据")  # 原始数据散点图
plt.plot(x, pred_y, '-r', lw=1.5, label="预测值")  # 预测值曲线

# 显示图例
plt.legend(loc=1)

# 计算模型得分和残差平方和
score = model.score(x, y)
print("score:", score)
ss = ((y - pred_y) ** 2).sum()  # 计算残差平方和
print("残差平方和：", ss)

# 显示图形
plt.show()