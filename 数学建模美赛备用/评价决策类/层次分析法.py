"""
#AHP 层次分析法
1. 条理化层次化 —— 最高层、中间层、最底层
2. 判断重要性矩阵 —— 重要性矩阵是用来评估各个层次之间的相对重要性的。
3. 建立层次网络 —— 层次网络是用来描述各个层次之间的相互关系的。
判断矩阵构造 标度1-9 需要一致性检验避免矛盾
aij=i的重要度/j的重要程度
如果 aij=aik*akj且矩阵各行各列成倍数关系 则说明是一致矩阵 —— 正互反矩阵
一致性指标CI=特征值-n/n-1 如果CI>0.1 则说明层次分析法有效
4. 层次分析法的步骤：
"""
import numpy as np
class AHP:
    """
    #相关信息的传入和准备
    """

    def __init__(self, array):
        ## 记录矩阵相关信息
        self.array = array
        ## 记录矩阵大小
        self.n = array.shape[0]
        # 初始化RI值，用于一致性检验
        self.RI_list = [0, 0, 0.52, 0.89, 1.12, 1.26, 1.36, 1.41, 1.46, 1.49, 1.52, 1.54, 1.56, 1.58,
                        1.59]
        # 矩阵的特征值和特征向量
        self.eig_val, self.eig_vector = np.linalg.eig(self.array)
        # 矩阵的最大特征值
        self.max_eig_val = np.max(self.eig_val)
        # 矩阵最大特征值对应的特征向量
        self.max_eig_vector = self.eig_vector[:, np.argmax(self.eig_val)].real
        # 矩阵的一致性指标CI
        self.CI_val = (self.max_eig_val - self.n) / (self.n - 1)
        # 矩阵的一致性比例CR
        self.CR_val = self.CI_val / (self.RI_list[self.n - 1])

    """
    #一致性判断
    """

    def test_consist(self):
        # 打印矩阵的一致性指标CI和一致性比例CR
        print("判断矩阵的CI值为：" + str(self.CI_val))
        print("判断矩阵的CR值为：" + str(self.CR_val))
        # 进行一致性检验判断
        if self.n == 2:  # 当只有两个子因素的情况
            print("仅包含两个子因素，不存在一致性问题")
        else:
            if self.CR_val < 0.1:  # CR值小于0.1，可以通过一致性检验
                print("判断矩阵的CR值为" + str(self.CR_val) + ",通过一致性检验")
                return True
            else:  # CR值大于0.1, 一致性检验不通过
                print("判断矩阵的CR值为" + str(self.CR_val) + "未通过一致性检验")
                return False

    """
    #算术平均法求权重
    """

    def cal_weight_by_arithmetic_method(self):
        # 求矩阵的每列的和
        col_sum = np.sum(self.array, axis=0)
        # 将判断矩阵按照列归一化
        array_normed = self.array / col_sum
        # 计算权重向量
        array_weight = np.sum(array_normed, axis=1) / self.n
        # 打印权重向量
        print("算术平均法计算得到的权重向量为：\n", array_weight)
        # 返回权重向量的值
        return array_weight

    """
    #几何平均法求权重
    """

    def cal_weight__by_geometric_method(self):
        # 求矩阵的每列的积
        col_product = np.prod(self.array, axis=0)
        # 将得到的积向量的每个分量进行开n次方
        array_power = np.power(col_product, 1 / self.n)
        # 将列向量归一化
        array_weight = array_power / np.sum(array_power)
        # 打印权重向量
        print("几何平均法计算得到的权重向量为：\n", array_weight)
        # 返回权重向量的值
        return array_weight

    """
    #特征值法求权重
    """

    def cal_weight__by_eigenvalue_method(self):
        # 将矩阵最大特征值对应的特征向量进行归一化处理就得到了权重
        array_weight = self.max_eig_vector / np.sum(self.max_eig_vector)
        # 打印权重向量
        print("特征值法计算得到的权重向量为：\n", array_weight)
        # 返回权重向量的值
        return array_weight


#if __name__ == "__main__":
#主要目的是：
#确保测试代码不会在模块被导入时运行。
#提高代码的复用性和模块化设计能力。
#__name__变量的值为当前模块的名称。


if __name__ == "__main__":
    # 给出判断矩阵
    b = np.array([[1, 1 / 3, 1 / 8], [3, 1, 1 / 3], [8, 3, 1]])

    # 算术平均法求权重
    weight1 = AHP(b).cal_weight_by_arithmetic_method()
    # 几何平均法求权重
    weight2 = AHP(b).cal_weight__by_geometric_method()
    # 特征值法求权重
    weight3 = AHP(b).cal_weight__by_eigenvalue_method()
