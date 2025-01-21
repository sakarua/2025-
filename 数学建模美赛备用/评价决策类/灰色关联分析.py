import numpy as np


def grey_relation_coefficient(sequence, reference):
    """
    计算灰色关联系数
    :param sequence: 待比较序列
    :param reference: 参考序列
    :return: 灰色关联系数
    """
    # 计算序列的差值
    delta = np.abs(sequence - reference)

    # 计算最小差值和最大差值
    min_delta = np.min(delta)
    max_delta = np.max(delta)

    # 计算灰色关联系数
    rho = 0.5  # 分辨系数，通常取0.5
    epsilon = 1e-10  # 添加一个很小的值，避免除以零
    grey_coefficient = (min_delta + rho * max_delta) / (delta + rho * max_delta + epsilon)

    return grey_coefficient


def grey_relation_grade(sequences, reference):
    """
    计算灰色关联度
    :param sequences: 待比较序列的列表
    :param reference: 参考序列
    :return: 灰色关联度
    """
    # 计算每个序列的灰色关联系数
    grey_coefficients = np.array([grey_relation_coefficient(seq, reference) for seq in sequences])

    # 计算灰色关联度（对每个序列的关联系数取平均值）
    grey_grades = np.mean(grey_coefficients, axis=1)

    return grey_grades


# 示例数据
reference_sequence = np.array([1, 2, 3, 4, 5])  # 参考序列
sequences = np.array([
    [1.1, 2.2, 3.1, 4.2, 5.1],  # 序列1
    [1.2, 2.1, 3.2, 4.1, 5.2],  # 序列2
    [1.9, 2.6, 3.7, 4.8, 5.9]  # 序列3
])

# 计算灰色关联度
grey_grades = grey_relation_grade(sequences, reference_sequence)
print("灰色关联度:", grey_grades)