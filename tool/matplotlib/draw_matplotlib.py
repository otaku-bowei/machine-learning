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
