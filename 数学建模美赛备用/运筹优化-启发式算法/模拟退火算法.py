from numpy import loadtxt, radians, sin, cos, inf, exp
from numpy import array, r_, c_, arange, savetxt
from numpy.lib.scimath import arccos
from numpy.random import shuffle, randint, rand
from matplotlib.pyplot import plot, show, rc
import matplotlib
matplotlib.use('TkAgg')

"""
模拟退火算法通过模拟物理退火过程来优化问题的解。
物理退火是指将材料加热到高温，然后缓慢冷却，以使材料达到低能态（稳定态）。
在模拟退火算法中：
高温阶段：算法以较大的概率接受劣解，从而跳出局部最优解，探索更广泛的解空间。
低温阶段：算法以较小的概率接受劣解，逐渐收敛到全局最优解。
"""


# 加载数据
a = loadtxt("Pdata17_1.txt")
x = a[:, ::2].flatten()
y = a[:, 1::2].flatten()

# 初始化起点和终点
d1 = array([[70, 40]])  # 起点
xy = c_[x, y]  # 城市坐标
xy = r_[d1, xy, d1]  # 添加起点和终点
N = xy.shape[0]  # 城市数量

# 计算城市之间的距离
t = radians(xy)  # 转化为弧度
d = array([[6370 * arccos(cos(t[i, 0] - t[j, 0]) * cos(t[i, 1]) * cos(t[j, 1]) +
            sin(t[i, 1]) * sin(t[j, 1])) for i in range(N)]
           for j in range(N)]).real

# 保存数据
savetxt('Pdata17_2.txt', c_[xy, d])

# 初始化路径
path = arange(N)  # 初始路径
L = inf  # 初始路径长度
for j in range(1000):
    path0 = arange(1, N - 1)
    shuffle(path0)  # 随机打乱路径
    path0 = r_[0, path0, N - 1]  # 添加起点和终点
    L0 = d[0, path0[1]]  # 初始化路径长度
    for i in range(1, N - 1):
        L0 += d[path0[i], path0[i + 1]]
    if L0 < L:
        path = path0
        L = L0
print("初始最优路径:", path)
print("初始最优路径长度:", L)

# 模拟退火算法
e = 0.1**30  # 终止温度
M = 20000  # 迭代次数
at = 0.999  # 温度衰减系数
T = 1  # 初始温度

for k in range(M):
    c = randint(1, 101, 2)  # 随机选择两个城市
    c.sort()
    c1, c2 = c[0], c[1]

    # 计算路径长度变化
    df = d[path[c1 - 1], path[c2]] + d[path[c1], path[c2 + 1]] - \
         d[path[c1 - 1], path[c1]] - d[path[c2], path[c2 + 1]]

    # 接受新解
    if df < 0:
        path = r_[path[0], path[1:c1], path[c2:c1 - 1:-1], path[c2 + 1:102]]
        L += df
    else:
        if exp(-df / T) >= rand(1):
            path = r_[path[0], path[1:c1], path[c2:c1 - 1:-1], path[c2 + 1:102]]
            L += df

    # 降温
    T *= at
    if T < e:
        break

    # 输出当前迭代结果
    if k % 1000 == 0:
        print(f"迭代次数: {k}, 当前温度: {T:.6f}, 当前路径长度: {L:.2f}")

print("最终最优路径:", path)
print("最终最优路径长度:", L)

# 绘制巡航路径
xx = xy[path, 0]
yy = xy[path, 1]
rc('font', size=16)
plot(xx, yy, '-*')  # 画巡航路径
show()