import string

import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import numpy as np


# draw_keras_by_key 描绘训练过程中的某些学习参数
def draw_keras_by_key(model_fit_history: keras.src.callbacks.history.History, key: string):
    print("\n")
    print(model_fit_history.history[key])
    print("\n")
    plt.plot(model_fit_history.history[key], label=key)
    plt.xlabel(key)
    plt.ylabel(key + "-value")
    plt.legend()
    plt.show()


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


def draw_2d_point(x, y: np.float64):
    plt.scatter(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.plot(x, y, color='red')
    plt.show()


def draw_2d_line(m, b: np.float64, x_range=(-10, 10)):
    # 生成x的值
    x = np.linspace(x_range[0], x_range[1],  4000)
    # 计算y的值
    y = x * m + b
    # 创建一个新的图形
    plt.figure()
    # 绘制直线
    plt.plot(x, y, label=f'y = {m}x + {b}', color='red')
    # 添加标题和标签
    plt.title('Plot of the line y = mx + b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    # 显示网格
    plt.grid(True)
    # 显示图形
    plt.show()
