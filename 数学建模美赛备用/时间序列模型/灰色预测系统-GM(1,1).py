import numpy as np
from sympy import Function, diff, dsolve, symbols, solve, exp

# 原始数据
x0=np.array([41, 49, 61, 78, 96, 104])
n = len(x0)

# 1. 计算级比（用于检验数据是否适合灰色预测）
lamda = x0[:-1] / x0[1:]  # 计算级比
rang = [lamda.min(), lamda.max()]  # 计算级比的范围
theta = [np.exp(-2 / (n + 1)), np.exp(2 / (n + 1))]  # 计算级比容许范围
print("级比范围：", rang)
print("级比容许范围：", theta)

# 2. 累加生成序列（灰色预测模型需要对原始数据进行累加生成）
x1 = np.cumsum(x0)  # 计算1次累加序列
print("累加序列：", x1)

# 3. 计算一次累减序列
ax0 = np.diff(x0)  # 计算一次累减序列
print("累减序列：", ax0)

# 4. 计算均值生成序列（用于构建灰色微分方程）
z = 0.5 * (x1[1:] + x1[:-1])  # 计算均值生成序列
print("均值生成序列：", z)

# 5. 构建灰色微分方程的系数矩阵 B 和常数项矩阵
B = np.c_[-x0[1:], -z, np.ones((n - 1, 1))]  # 构建系数矩阵 B
u = np.linalg.pinv(B).dot(ax0)  # 最小二乘法拟合参数
print("灰色微分方程的参数 u：", u)

# 6. 构造特征多项式并求特征根
p = np.r_[1, u[:-1]]  # 构造特征多项式
r = np.roots(p)  # 求特征根
print("特征根：", r)

# 7. 计算常微分方程的特解
xts = u[2] / u[1]  # 常微分方程的特解
print("常微分方程的特解：", xts)

# 8. 求解微分方程的符号解
c1, c2, t = symbols('c1,c2,t')  # 定义符号变量
eq1 = c1 + c2 + xts - 41  # 初始条件方程 1
eq2 = c1 * np.exp(5 * r[0]) + c2 * np.exp(5 * r[1]) + xts - 429  # 初始条件方程 2
c = solve([eq1, eq2], [c1, c2])  # 求解初始条件
s = c[c1] * exp(r[0] * t) + c[c2] * exp(r[1] * t) + xts  # 微分方程的符号解
print("灰色预测模型的符号解：", s)

# 9. 计算预测值
xt1 = []
for i in range(6):
    xt1.append(float(s.subs({t: i})))  # 将符号解转换为数值
xh0 = np.r_[xt1[0], np.diff(xt1)]  # 还原原始数据的预测值
print("累加序列的预测值：", xt1)
print("原始数据的预测值：", xh0)

# 10. 计算残差和相对误差
cha = x0 - xh0  # 计算残差
delta = np.abs(cha) / x0  # 计算相对误差

# 将 delta 转换为 numpy 可以处理的数值类型
delta_numeric = np.array([float(val) for val in delta])  # 转换为 numpy 数组
print("残差：", cha)
print("相对误差（%）：", np.round(delta_numeric * 100, 2))

# 11. 模型检验
# 11.1 级比偏差检验
lamda_pred = xh0[:-1] / xh0[1:]  # 预测值的级比
lamda_bias = np.abs(lamda - lamda_pred)  # 级比偏差
print("级比偏差：", lamda_bias)

# 11.2 后验差检验
S1 = np.std(x0, ddof=1)  # 原始数据的标准差
S2 = np.std(cha, ddof=1)  # 残差的标准差
C = S2 / S1  # 后验差比值
print("后验差比值 C：", C)

# 小误差概率
P = np.sum(np.abs(cha - np.mean(cha)) < 0.6745 * S1) / n
print("小误差概率 P：", P)

# 11.3 模型精度评估
if C < 0.35 and P > 0.95:
    print("模型精度：优秀")
elif C < 0.5 and P > 0.8:
    print("模型精度：良好")
elif C < 0.65 and P > 0.7:
    print("模型精度：合格")
else:
    print("模型精度：不合格")