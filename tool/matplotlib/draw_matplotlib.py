import string

import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import numpy as np


# draw_keras_by_key 描绘训练过程中的某些学习参数
# def draw_keras_by_key(model_fit_history: keras.src.callbacks.history.History, key: string):
#     print("\n")
#     print(model_fit_history.history[key])
#     print("\n")
#     plt.plot(model_fit_history.history[key], label=key)
#     plt.xlabel(key)
#     plt.ylabel(key + "-value")
#     plt.legend()
#     plt.show()


# draw_model 描绘模型
def draw_model(path: string, x_train, y_train: np.float64):
    model = tf.keras.models.load_model(path)
    # 使用模型进行预测
    y_pred = model.predict(x_train)
    # 绘制训练集的数据点
    plt.scatter(x_train, y_train, label='Training Data', color='blue')
    # 为了绘制模型的函数，我们需要生成一个更密集的输入范围
    X_dense = np.linspace(x_train.min(), x_train.max(), 100).reshape(-1, 1)
    y_pred_dense = model.predict(X_dense)
    # 绘制模型的函数（预测输出）
    plt.plot(X_dense, y_pred_dense, label='Model Prediction', color='red')
    # 添加图例和标签
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.legend()

    # 显示图形
    plt.show()


def draw_2d_point(x, y: np.float64, point_size=20, color='blue'):
    plt.scatter(x, y, point_size)
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.plot(x, y, color='green')
    plt.show()


def draw_2d_line(m, b: np.float64, x_range=(-10, 10), color='blue'):
    # 生成x的值
    x = np.arange(x_range[0], x_range[1],  0.1)
    x = x.reshape(1, int(x.shape[0]))
    # 计算y的值
    y = x * m + b
    draw_2d_point(x, y, 1, color=color)


def draw_data_line(x_t, y_t, m, b: np.float64, x_range=(-10, 10), color='blue'):
    plt.scatter(x_t, y_t, 1, 'blue')
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.plot(x, y, color='green')
    # 生成x的值
    x = np.arange(x_range[0], x_range[1],  0.1)
    x = x.reshape(1, int(x.shape[0]))
    # 计算y的值
    y = x * m + b
    plt.scatter(x, y, 1, color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.plot(x, y, color='green')
    plt.show()


def compare_draw_data_line(x_t, y_t, m1, b1, m2, b2: np.float64, x_range=(-10, 10), color='blue'):
    plt.scatter(x_t, y_t, 1, 'blue')
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.plot(x, y, color='green')
    # 生成x的值
    x = np.arange(x_range[0], x_range[1],  0.001)
    x1 = x.reshape(1, int(x.shape[0]))
    # 计算y的值
    # y1 = x1 * m1 + b1
    y1 = x1 * m1[2] + np.power(x1, 2) * m1[1] + np.power(x1, 3) * m1[1] + b1
    # y1 = x1 * m1[4] + np.power(x1, 2) * m1[3] + np.power(x1, 3) * m1[2] + np.power(x1, 4) * m1[1] + np.power(x1, 5) * m1[0] + b1
    plt.scatter(x1, y1, 1, color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.plot(x, y, color='green')
    x2 = x.reshape(1, int(x.shape[0]))
    # 计算y的值
    # y2 = x2 * m2 + b2
    y2 = x2 * m2[2] + np.power(x2, 2) * m2[1] + np.power(x2, 3) * m2[1] + b2
    plt.scatter(x2, y2, 1, color='yellow')
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.plot(x, y, color='green')
    plt.show()