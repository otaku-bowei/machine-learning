import re
import string
import datetime
import numpy as np
import pandas as pd
import sklearn.linear_model
import tensorflow as tf
import keras
from datashader import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import tool.tensorboard.tensor_board as tb


# read_data
def read_data():
    # 读取文件
    train_data = pd.read_csv(".\\train.csv")
    test_data = pd.read_csv(".\\test.csv")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(train_data.describe())
    # 数据统计展示
    # train_data.describe().to_csv('train_des.csv', index=False)
    # test_data.describe().to_csv('test_des.csv', index=False)
    return train_data, test_data


# pd_to_np pd格式转换为np格式
def pd_to_np(train_data, test_data: pd.DataFrame):
    # 乱序
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    test_data = test_data.sample(frac=1).reset_index(drop=True)
    train_data = train_data.drop('ID_code', axis=1)
    train_data_tmp = train_data.loc[:, train_data.columns != 'target']
    train_label_tmp = train_data.loc[:, train_data.columns == 'target']
    test_data_tmp = test_data.loc[:, test_data.columns != 'ID_code']
    test_label_tmp = test_data.loc[:, test_data.columns == 'ID_code']
    return train_data_tmp.to_numpy(), train_label_tmp.to_numpy(), test_data_tmp.to_numpy(), test_label_tmp.to_numpy()



# read_init_model 读取模型，否则初始化模型
def read_init_model(file_name: string) -> keras.src.models.sequential.Sequential:
    model = {}
    try:
        model = tf.keras.models.load_model(file_name)
    except Exception as e:
        model = tf.keras.models.Sequential([
            # 此处加了一层32个神经元，分数提高了一点
            # tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            # tf.keras.layers.Dense(64, activation='relu'),
            # tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=0.01),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    finally:
        return model
    # 1.建立全连接神经网络，relu作为激活函数，dropout率为0.2


# nn 定义一个全连接神经网络
def nn(train_d, train_l: pd.DataFrame):
    callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda batch, logs: [callback.on_train_begin])
    # 1.建立全连接神经网络
    model = read_init_model('my_model.keras')
    # 2.调整数据格式
    train_data = train_d.astype(np.float64)
    train_label = train_l.astype(np.float64)
    # 4.1定义交叉熵损失函数
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    # 4.2定义tensorBoard
    tensorboard_callback = tb.draw_board('bank')
    # 5.进行训练
    train_history = model.fit(train_data, train_label, epochs=16, callbacks=[tensorboard_callback])
    # 6.保存模型
    model.save('my_model.keras')


# predict_test_data 预测测试集
def predict_test_data(test_d, test_l: pd.DataFrame):
    model = tf.keras.models.load_model('my_model.keras')
    test_data = test_d
    pre = model.predict(test_data)
    pre = np.round(pre).astype(float)
    result = np.concatenate((test_l.astype(int), pre), axis=1)
    output = pd.DataFrame(result, columns=['id', 'price'])
    output.to_csv('submission.csv', index=False)


# train_by_sklearn 用sklearn做多项式回归学习
def train_by_sklearn(train_data, train_label: pd.DataFrame):
    model = Pipeline([('poly'), PolynomialFeatures(degree=3), ('linear', LinearRegression(fit_intercept=False))])
    x = train_data
    y = x ** 3 - x ** 2 + 2 * x - 5
    model = model.fit(x[:, np.newaxis], y)
    return model.named_steps['linear'].coef_


# main 主函数
def main():
    # 1.pd读取数据集，提取相关有用信息并做数据预处理
    train_data, test_data = read_data()
    # 2.整理字段
    train_d, train_l, test_d, test_l = pd_to_np(train_data, test_data)
    nn(train_d, train_l)
    # predict_test_data(test_d, test_l)


if __name__ == "__main__":
    main()
