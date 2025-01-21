import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 设置中文字体和字体大小
plt.rc('font', family='SimHei')
plt.rc('font', size=16)

# 读取数据
d = pd.read_csv('sunspots.csv', usecols=['counts'])

# 使用 ARIMA 模型替代 ARMA 模型
# ARMA(p, q) 等价于 ARIMA(p, 0, q)
md = sm.tsa.ARIMA(d, order=(9, 0, 1)).fit()

# 已知观测值的年代
years = np.arange(1700, 1989)

# 获取预测值
dhat = md.predict()

# 绘制最后 20 年的观测值和预测值
plt.plot(years[-20:], d.values[-20:], 'o-k', label='原始观测值')
plt.plot(years[-20:], dhat.values[-20:], 'P--', label='预测值')
plt.legend()
plt.show()

# 预测下一期的值
dnext = md.predict(start=d.shape[0], end=d.shape[0])
print("下一期的预测值为：", dnext)