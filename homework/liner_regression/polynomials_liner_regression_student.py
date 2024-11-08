# 单变量的线性回归,用阅读分数预测写作分数
import numpy as np
import pandas as pd
import tool.matplotlib.draw_matplotlib as tm
import sklearn as sk
import sklearn.linear_model as lm
import sklearn.preprocessing as pping
import sklearn.pipeline as pl


# read_data 读取数据
def read_data() -> pd.DataFrame:
    df = pd.read_csv('StudentsPerformance.csv')
    df = df.loc[:, ['reading score', 'writing score']]
    # df_np = df.to_numpy()
    # tm.draw_2d_point(df_np.T[0], df_np.T[1])
    return df


# compute_cost 计算损失函数 (1/2*m)*Σ(wT*X-y)²
def compute_cost(X, y, w, b):
    # 得到矩阵*学习参数的列矩阵y-hat，减去y然后求平方
    m = int(X.shape[0])
    predictions = np.power(X, 3).dot(w[0]) + np.power(X, 2).dot(w[1]) + X.dot(w[2]) + b
    cost = (1 / (2 * m)) * np.sum(np.power((predictions - y), 2))
    return cost


# pd_data_to_np pd数据转np
def pd_data_to_np():
    data = read_data()
    cols = data.shape[1]
    # 做特征工程，范围缩小到0-1
    X = data.iloc[:, 0:cols - 1] / 100  # X是所有行，去掉最后一列
    y = data.iloc[:, cols - 1:cols] / 100  # y是所有行，最后一列
    x_np = X.to_numpy()
    y_np = y.to_numpy()
    return x_np, y_np


# train 训练，整理数据，利用梯度下降，根据损失函数找到凸函数的最点
def train():
    x_np, y_np = pd_data_to_np()
    # 此处可以对比，当一开始的w设置的已经接近的时候，梯度下降的很慢，因为已经很接近 局部最小值--对比不用matrix(准确率更低)
    w = np.matrix(np.array([0, 0, 0]))
    # w = np.array([0])
    # w = np.matrix(np.array([1]))
    b = 0
    # 体现了学习率的作用，学习率必须根据损失函数的偏导数调整，真实数据根据损失函数的偏导数很大,而w本身局部最小值较小时，容易发生梯度震荡
    w, b, cost = batch_gradient_descent(x_np, y_np, w.T, b, 0.001, 3600)
    # tm.draw_2d_line(w.ravel()[0], b, x_range=(0, 100))
    print(np.array(w.tolist()))
    print(b)
    print(cost)
    return np.array(w.tolist()), b, cost


# train_by_sklearn 用sklearn训练
def train_by_sklearn():
    x_np, y_np = pd_data_to_np()
    # 3次方
    reg = pping.PolynomialFeatures(degree=3)
    liner = lm.LinearRegression()
    # pipeline = pl.make_pipeline(reg, liner)
    # 使用Pipeline将两个步骤链接起来
    pipeline = pl.Pipeline([("polynomial_features", reg), ("linear_regression", liner)])
    # pipeline = pl.Pipeline([('reg', reg), ('liner', liner)])
    pipeline.fit(x_np, y_np)
    print(pipeline.named_steps.get('polynomial_features'))
    print(liner.coef_)
    print(liner.intercept_)
    # tm.draw_data_line(x_np, y_np, reg.coef_, reg.intercept_, x_range=(0, 100))
    return pipeline.named_steps['linear_regression'].coef_, pipeline.named_steps['linear_regression'].intercept_


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
    w0 = w[0]
    w1 = w[1]
    w2 = w[2]
    for i in range(num_iterations):
        # 计算y-hat，改为多项式函数
        predictions = np.power(X, 3).dot(w0) + np.power(X, 2).dot(w1) + X.dot(w2) + b
        # 计算误差
        error = predictions - y
        # 用平均误差求偏导数--多项式偏导数有变化
        dw0 = np.power(X, 3).T.dot(error) / m
        dw1 = np.power(X, 2).T.dot(error) / m
        dw2 = X.T.dot(error) / m
        db = np.sum(error) / m
        # 根据学习率更新参数
        w0 = w0 - learning_rate * dw0
        w1 = w1 - learning_rate * dw1
        w2 = w2 - learning_rate * dw2
        b = b - learning_rate * db
        # 记录损失值
        cost = compute_cost(X, y, w, b)
        cost_history.append(cost)
        # 打印进度（可选）
        if i % 100 == 0:
            print(f"Iteration {i}: Cost {cost}")
    w = np.array(np.array([w0, w1, w2]).ravel())
    return w, b, cost_history


# main 主函数
def main():
    x_np, y_np = pd_data_to_np()
    w1, b1, c1 = train()
    w2, b2 = train_by_sklearn()
    # 多项式中，自定义的梯度下降函数比sklearn的更准确
    tm.compare_draw_data_line(x_np, y_np, w2[0][:3], b2, w2[0][:3], w2[0][3], x_range=(0, 1))
    # tm.compare_draw_data_line(x_np, y_np, w2[0][:3], b2, w1, b1, x_range=(0, 1))
    # tm.compare_draw_data_line(x_np, y_np, w2[0][:3], w2[0][3], w1, b1, x_range=(0, 1))


if __name__ == "__main__":
    main()
