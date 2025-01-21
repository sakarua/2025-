import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pylab as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib
matplotlib.use('TkAgg')

# 设置中文字体和字体大小
plt.rc('axes', unicode_minus=False)
plt.rc('font', family='SimHei')
plt.rc('font', size=16)

# 读取数据
d = pd.read_csv('sunspots.csv')
dd = d['counts']
years = d['year'].values.astype(int)

# 绘制时序图
plt.plot(years, dd.values, '-*')
plt.figure()

# 绘制自相关图和偏自相关图
ax1 = plt.subplot(121)
plot_acf(dd, ax=ax1, title='自相关')
ax2 = plt.subplot(122)
plot_pacf(dd, ax=ax2, title='偏自相关')

# 遍历不同的 ARIMA(p, q) 参数组合
for i in range(1, 6):
    for j in range(1, 6):
        md = sm.tsa.ARIMA(dd, order=(i, 0, j)).fit()  # 使用 ARIMA 替代 ARMA
        print([i, j, md.aic, md.bic])

# 选择最优模型并拟合
zmd = sm.tsa.ARIMA(dd, order=(4, 0, 2)).fit()  # 使用 ARIMA 替代 ARMA
print(zmd.summary())  # 显示模型的所有信息

# 残差分析
residuals = pd.DataFrame(zmd.resid)
fig, ax = plt.subplots(1, 2)
residuals.plot(title="残差", ax=ax[0])
residuals.plot(kind='kde', title='密度', ax=ax[1])
plt.legend('')
plt.ylabel('')

# 预测并绘制结果
dhat = zmd.predict()
plt.figure()
plt.plot(years[-20:], dd.values[-20:], 'o-k', label='原始观测值')
plt.plot(years[-20:], dhat.values[-20:], 'P--', label='预测值')
plt.legend()
dnext = zmd.predict(start=d.shape[0], end=d.shape[0])
print("下一期的预测值为：", dnext)
plt.show()