import numpy as np
import pandas as pd


def test():
    # 1.扁平化
    n = np.array([[1, 2, 3], [5, 6, 7]])
    print("\n")
    print(n.ravel())

    # 创建一个一维数组
    one_d_array = np.array([1, 2, 3, 4, 5, 6])

    # 指定想要的行数（n）和列数（m）
    n, m = 2, 3

    # 使用reshape方法将一维数组转换为二维矩阵
    two_d_matrix = one_d_array.reshape(n, m)

    print(two_d_matrix)
