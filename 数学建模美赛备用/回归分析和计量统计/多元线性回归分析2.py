import numpy as np
import statsmodels.api as sm

# 加载数据
a = np.loadtxt("Pdata12_1.txt")  # 加载数据文件
d = {'x1': a[:, 0], 'x2': a[:, 1], 'y': a[:, 2]}  # 构建数据字典

# 构建并拟合模型
md = sm.formula.ols('y ~ x1 + x2', d).fit()  # 使用公式接口构建模型并拟合

# 提取模型信息并输出中文结果
print("========== 模型信息 ==========")
print(f"因变量 (y): {md.model.endog_names}")
print(f"样本数量: {md.nobs}")
print(f"自变量: {md.model.exog_names}")
print(f"拟合优度 (R²): {md.rsquared:.4f}")
print(f"调整后的拟合优度 (Adj. R²): {md.rsquared_adj:.4f}")
print(f"F 统计量: {md.fvalue:.4f}")
print(f"F 统计量的 p 值: {md.f_pvalue:.4f}")
print("\n========== 回归系数 ==========")
print(f"截距 (Intercept): {md.params['Intercept']:.4f}")
print(f"x1 的系数: {md.params['x1']:.4f}")
print(f"x2 的系数: {md.params['x2']:.4f}")
print("\n========== 系数显著性检验 ==========")
print(f"截距的 p 值: {md.pvalues['Intercept']:.4f}")
print(f"x1 的 p 值: {md.pvalues['x1']:.4f}")
print(f"x2 的 p 值: {md.pvalues['x2']:.4f}")
print("\n========== 其他统计量 ==========")
print(f"残差的标准误差: {np.sqrt(md.mse_resid):.4f}")
print(f"对数似然值: {md.llf:.4f}")
print(f"AIC: {md.aic:.4f}")
print(f"BIC: {md.bic:.4f}")

# 计算预测值
ypred = md.predict({'x1': a[:, 0], 'x2': a[:, 1]})  # 计算预测值
print("\n========== 预测值 ==========")
print(ypred)