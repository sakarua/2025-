from sklearn.neural_network import MLPRegressor
from numpy import array, loadtxt
from pylab import subplot, plot, show, xticks, rc, legend
import matplotlib
matplotlib.use('TkAgg')

# 设置字体大小和字体类型
rc('font', size=10)  # 将字体大小设置为 10
rc('font', family='SimHei')  # 设置字体为 SimHei（黑体）

# 加载数据
a = loadtxt("Pdata17_5.txt")
x0 = a[:, :3]  # 输入特征
y1 = a[:, 3]   # 客运量目标值
y2 = a[:, 4]   # 货运量目标值

# 构建并训练客运量预测模型
md1 = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=10, max_iter=1000)  # 增加 max_iter 避免警告
md1.fit(x0, y1)

# 预测客运量
x = array([[73.39, 3.9635, 0.988], [75.55, 4.0975, 1.0268]])
pred1 = md1.predict(x)
print("客运量模型的拟合优度 (R²):", md1.score(x0, y1))
print("客运量的预测值为：", pred1, '\n----------------')

# 构建并训练货运量预测模型
md2 = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=10, max_iter=1000)  # 增加 max_iter 避免警告
md2.fit(x0, y2)

# 预测货运量
pred2 = md2.predict(x)
print("货运量模型的拟合优度 (R²):", md2.score(x0, y2))
print("货运量的预测值为：", pred2)

# 绘图
yr = range(1990, 2010)  # 横坐标年份
subplot(121)  # 左图：客运量
plot(yr, y1, 'o', label="原始数据")  # 原始数据
plot(yr, md1.predict(x0), '-*', label="网络输出客运量")  # 模型预测结果
xticks(yr, rotation=55)  # 设置横坐标刻度并旋转 55 度
legend()  # 显示图例

subplot(122)  # 右图：货运量
plot(yr, y2, 'o', label="原始数据")  # 原始数据
plot(yr, md2.predict(x0), '-*', label="网络输出货运量")  # 模型预测结果
xticks(yr, rotation=55)  # 设置横坐标刻度并旋转 55 度
legend(loc='upper left')  # 显示图例，位置在左上角

show()  # 显示图形