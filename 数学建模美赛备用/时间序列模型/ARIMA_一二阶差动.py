import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
import pylab as plt
from statsmodels.tsa.arima.model import ARIMA  # 使用新的 ARIMA 模型

# 设置中文字体和字体大小
plt.rc('axes', unicode_minus=False)
plt.rc('font', size=12)  # 调整字体大小
plt.rc('font', family='SimHei')

# 读取数据
df = pd.read_csv('austa.csv')

# 绘制一次差分图和自相关图
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(df.value.diff())
plt.title('一次差分', fontsize=14)  # 调整标题字体大小
plt.xlabel('时间', fontsize=12)  # 调整横坐标字体大小
plt.ylabel('差分值', fontsize=12)  # 调整纵坐标字体大小

ax2 = plt.subplot(122)
plot_acf(df.value.diff().dropna(), ax=ax2, title='自相关', lags=20)
ax2.set_xlabel('滞后阶数', fontsize=12)  # 调整横坐标字体大小
ax2.set_ylabel('自相关系数', fontsize=12)  # 调整纵坐标字体大小
plt.tight_layout()
plt.show()

# 使用新的 ARIMA 模型
md = ARIMA(df.value, order=(2, 1, 0))  # 使用新的 ARIMA 模型
mdf = md.fit()  # 拟合模型

# 输出模型摘要（英文版）
print("模型摘要（英文版）：")
print(mdf.summary())

# 输出模型摘要（中文版）
print("\n模型摘要（中文版）：")
summary = mdf.summary().tables[1]
summary_df = pd.DataFrame(summary.data[1:], columns=summary.data[0])

# 根据实际列数调整列名
if len(summary_df.columns) == 7:
    summary_df.columns = ['系数', '标准误差', 'z值', 'P值', '[0.025', '0.975]', '其他']
elif len(summary_df.columns) == 6:
    summary_df.columns = ['系数', '标准误差', 'z值', 'P值', '[0.025', '0.975]']
else:
    summary_df.columns = [f'列{i+1}' for i in range(len(summary_df.columns))]  # 默认列名

print(summary_df)

# 残差分析
residuals = pd.DataFrame(mdf.resid)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
residuals.plot(title="残差", ax=ax[0], fontsize=12)  # 调整字体大小
ax[0].set_xlabel('时间', fontsize=12)  # 调整横坐标字体大小
ax[0].set_ylabel('残差值', fontsize=12)  # 调整纵坐标字体大小

residuals.plot(kind='kde', title='密度', ax=ax[1], fontsize=12)  # 调整字体大小
ax[1].set_xlabel('残差值', fontsize=12)  # 调整横坐标字体大小
ax[1].set_ylabel('密度', fontsize=12)  # 调整纵坐标字体大小
plt.tight_layout()
plt.show()

# 手动绘制原始数据与预测值对比图
plt.figure(figsize=(10, 5))
pred = mdf.get_prediction(start=0, end=len(df) - 1)  # 获取预测值
pred_mean = pred.predicted_mean  # 预测值的均值
plt.plot(df.index, df.value, 'o-k', label='原始数据')  # 原始数据
plt.plot(df.index, pred_mean, 'P--', label='预测值')  # 预测值
plt.xlabel('时间', fontsize=12)  # 调整横坐标字体大小
plt.ylabel('值', fontsize=12)  # 调整纵坐标字体大小
plt.title('原始数据与预测值对比', fontsize=14)  # 调整标题字体大小
plt.legend(fontsize=12)  # 调整图例字体大小
plt.grid(True)
plt.show()