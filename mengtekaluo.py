#-*-coding:GBK -*- 

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

# Ŀ�꺯�������Ŀ�꣬תΪ��С�����⣩
def objective(vars):
    T, P = vars  # �¶Ⱥ�ѹ��
    return -(T - 20)**2 - (P - 30)**2  # ���ű�ʾ�������ת��Ϊ��С��

# ����Լ������
def constraint1(vars):
    T, P = vars
    return 50 - (T + P)  # T + P <= 50

def constraint2(vars):
    T, P = vars
    return T - 10  # T >= 10

def constraint3(vars):
    T, P = vars
    return P - 15  # P >= 15

# ��ʼ�²�
#initial_guess = [25, 25]

#��Ϊʹ�����ؿ��޷����г�ʼֵ����
n = 1000000
x1 = np.random.uniform(-100,100,size = n)
x2 = np.random.uniform(-100,100,size = n)
fmin = 1000000
for i in range (n):
    x = np.array([x1[i],x2[i]])
    if constraint1(x) >= 0 and constraint2(x) >= 0 and constraint3(x) >= 0:
        result = objective(x)
        if result < fmin:
            fmin = result
            x0 = x

print("���ؿ��޷���ʼֵ����Ϊ:",x0)

# ����Լ����������ʽ
constraints = [{'type': 'ineq', 'fun': constraint1},
               {'type': 'ineq', 'fun': constraint2},
               {'type': 'ineq', 'fun': constraint3}]

# ʹ��minimize�����������
solution = minimize(objective, x0, method='SLSQP', constraints=constraints)

# ������
T_opt, P_opt = solution.x
print(f"�����¶�: {T_opt:.2f}")
print(f"����ѹ��: {P_opt:.2f}")
print(f"���������: {-solution.fun:.2f}")  # ע������ȡ���ţ���Ϊ������С���˸�Ŀ�꺯��