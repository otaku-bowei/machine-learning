# 单变量的线性回归,用阅读分数预测写作分数
import numpy as np
import pandas as pd
import tool.matplotlib.draw_matplotlib as tm
import sklearn as sk
import sklearn.linear_model as lm


# read_data 读取数据
def read_data() -> pd.DataFrame:
    df = pd.read_csv('StudentsPerformance.csv')
    print(df.head())
    print(df.describe())
    df = df.loc[:, ['reading score', 'writing score']]
    # df_np = df.to_numpy()
    # tm.draw_2d_point(df_np.T[0], df_np.T[1])
    return df


# compute_cost 计算损失函数 (1/2*m)*Σ(wT*X-y)²
def compute_cost(X, y, w, b):
    # 得到矩阵*学习参数的列矩阵y-hat，减去y然后求平方
    m = int(X.shape[0])
    predictions = X.dot(w) + b
    cost = (1 / (2 * m)) * np.sum(np.power((predictions - y), 2))
    return cost


# pd_data_to_np pd数据转np
def pd_data_to_np():
    data = read_data()
    cols = data.shape[1]
    X = data.iloc[:, 0:cols - 1]  # X是所有行，去掉最后一列
    y = data.iloc[:, cols - 1:cols]  # y是所有行，最后一列
    x_np = X.to_numpy()
    y_np = y.to_numpy()
    return x_np, y_np


# train 训练，整理数据，利用梯度下降，根据损失函数找到凸函数的最点
def train():
    x_np, y_np = pd_data_to_np()
    # 此处可以对比，当一开始的w设置的已经接近的时候，梯度下降的很慢，因为已经很接近 局部最小值
    w = np.matrix(np.array([0]))
    # w = np.matrix(np.array([1]))
    b = 0
    # 体现了学习率的作用，学习率必须根据损失函数的偏导数调整，真实数据根据损失函数的偏导数很大,而w本身局部最小值较小时，容易发生梯度震荡
    w, b, cost = batch_gradient_descent(x_np, y_np, w.T, b, 0.000001, 1000)
    # tm.draw_2d_line(w.ravel()[0], b, x_range=(0, 100))
    print(w)
    print(b)
    print(cost)
    return w, b, cost


# train_by_sklearn 用sklearn训练
def train_by_sklearn():
    x_np, y_np = pd_data_to_np()
    reg = lm.LinearRegression()
    reg.fit(x_np, y_np)
    print(reg.coef_)
    print(reg.intercept_)


# batch_gradient_descent 根据学习率和迭代次数进行批量梯度下降 θ[j]=θ[j]-α(əJ(θ)/əθ[j])
def batch_gradient_descent(X, y, w, b, learning_rate, num_iterations):
    """
    批量梯度下降优化
    参数:
    X -- 输入特征，形状 (m, n)
    y -- 真实标签，形状 (m, 1)
    w -- 权重参数，形状 (n, 1)
    b -- 偏置参数
    learning_rate -- 学习率
    num_iterations -- 迭代次数
    J(w,b) -- [ J(w, b) = \frac{1}{2m} \sum_{i=1}{m} (\hat{y}_i - y_i)2 ] 损失函数，损失函数的偏导数决定学习方向
    返回:
    w -- 更新后的权重参数
    b -- 更新后的偏置参数
    cost_history -- 每次迭代的损失值
    """
    m = int(X.shape[0])
    cost_history = []
    for i in range(num_iterations):
        # 计算y-hat
        predictions = X.dot(w) + b
        # 计算误差
        error = predictions - y
        # 用平均误差求偏导数--当求出的导数很大的时候，会出现梯度震荡，无法收敛
        dw = X.T.dot(error) / m
        db = np.sum(error) / m
        # 根据学习率更新参数
        w = w - learning_rate * dw
        b = b - learning_rate * db
        # 记录损失值
        cost = compute_cost(X, y, w, b)
        cost_history.append(cost)
        # 打印进度（可选）
        if i % 100 == 0:
            print(f"Iteration {i}: Cost {cost}")
    return w, b, cost_history


# main 主函数
def main():
    train()
    train_by_sklearn()


if __name__ == "__main__":
    main()
