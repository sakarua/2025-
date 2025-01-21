import numpy as np
from numpy.random import randint, rand, shuffle
from matplotlib.pyplot import plot, show, rc
import matplotlib
matplotlib.use('TkAgg')

# 加载数据
a = np.loadtxt("Pdata17_2.txt")
xy, d = a[:, :2], a[:, 2:]
N = len(xy)

# 初始化参数
w = 50  # 种群大小
g = 10  # 进化代数

# 初始化种群
J = []
for i in np.arange(w):
    c = np.arange(1, N - 1)
    shuffle(c)
    c1 = np.r_[0, c, N - 1]  # 随机生成一条路径
    J.append(c1)
J = np.array(J) / (N - 1)

# 遗传算法优化
for k in np.arange(g):
    print(f"\n=== 第 {k+1} 代 ===")

    # 交叉操作
    A = J.copy()
    c1 = np.arange(w)
    shuffle(c1)  # 随机配对
    c2 = randint(2, 100, w)  # 随机选择交叉点
    for i in np.arange(0, w, 2):
        temp = A[c1[i], c2[i]:N - 1]
        A[c1[i], c2[i]:N - 1] = A[c1[i + 1], c2[i]:N - 1]
        A[c1[i + 1], c2[i]:N - 1] = temp

    # 变异操作
    B = A.copy()
    by = np.where(rand(w) < 0.1)[0]  # 随机选择变异个体
    B = B[by, :]

    # 选择操作
    G = np.r_[J, A, B]
    ind = np.argsort(G, axis=1)
    NN = G.shape[0]
    L = np.zeros(NN)
    for j in np.arange(NN):
        for i in np.arange(N - 1):
            L[j] = L[j] + d[ind[j, i], ind[j, i + 1]]
    ind2 = np.argsort(L)
    J = G[ind2, :]

    # 输出当前代的最优路径长度和路径
    best_path = ind[ind2[0], :]
    best_length = L[ind2[0]]
    print(f"当前最优路径长度: {best_length}")
    print(f"当前最优路径: {best_path}")

# 输出最终结果
path = ind[ind2[0], :]  # 最优路径
zL = L[ind2[0]]  # 最优路径长度
xx = xy[path, 0]
yy = xy[path, 1]

# 绘制巡航路径
rc('font', size=16)
plot(xx, yy, '-*')  # 画巡航路径
show()

# 输出最终结果
print("\n=== 最终结果 ===")
print("所求的巡航路径长度为：", zL)
print("最优路径：", path)