import numpy as np
import statsmodels.api as sm

# 加载数据
a = np.loadtxt("Pdata12_1.txt")  # 加载数据文件
X = sm.add_constant(a[:, :2])  # 增加第一列全部元素为1得到增广矩阵

# 构建并拟合模型
md = sm.OLS(a[:, 2], X).fit()  # 构建并拟合模型

# 提取所有回归系数
print("========== 回归系数 ==========")
print(md.params)
print("------------")

# 求已知自变量值的预测值
y = md.predict(X)

# 输出模型的所有结果（中文）
print("========== 模型详细信息 ==========")
print(f"模型: OLS (普通最小二乘法)")
print(f"因变量: y")
print(f"样本数量: {md.nobs}")
print(f"自变量: const, x1, x2")
print(f"拟合优度 (R²): {md.rsquared:.4f}")
print(f"调整后的拟合优度 (Adj. R²): {md.rsquared_adj:.4f}")
print(f"F 统计量: {md.fvalue:.4f}")
print(f"F 统计量的 p 值: {md.f_pvalue:.4f}")
print("\n========== 回归系数详情 ==========")
print(f"截距: {md.params[0]:.4f}")  # 第一个系数是截距
print(f"x1 的系数: {md.params[1]:.4f}")  # 第二个系数是 x1
print(f"x2 的系数: {md.params[2]:.4f}")  # 第三个系数是 x2
print("\n========== 系数显著性检验 ==========")
print(f"截距的 p 值: {md.pvalues[0]:.4f}")  # 第一个 p 值是截距
print(f"x1 的 p 值: {md.pvalues[1]:.4f}")  # 第二个 p 值是 x1
print(f"x2 的 p 值: {md.pvalues[2]:.4f}")  # 第三个 p 值是 x2
print("\n========== 其他统计量 ==========")
print(f"残差的标准误差: {np.sqrt(md.mse_resid):.4f}")
print(f"对数似然值: {md.llf:.4f}")
print(f"AIC: {md.aic:.4f}")
print(f"BIC: {md.bic:.4f}")
print("\n========== 预测值 ==========")
print(y)