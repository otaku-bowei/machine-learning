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


# deal_data_with_avg_price 特征工程，某个分类标识字段x，根据平均出售价格，分配比重
def deal_data_with_avg_price(train_data, test_date: pd.DataFrame, name: string):
    data_tmp = train_data.loc[:, [name, 'price']]
    column_values = data_tmp[name].value_counts()
    for column in column_values.index:
        data_t = data_tmp.loc[data_tmp[name] == column, :]
        sum = data_t['price'].sum()
        count = data_t['price'].count()
        train_data.loc[train_data[name] == column, name] = sum / count
        test_date.loc[test_date[name] == column, name] = sum / count
    print("成功转换字段:" + name)


# device_column 将某列分裂为多列
def device_column(data: pd.DataFrame):
    d = data.loc[:, ['engine']]
    data['mali'] = data['engine'].apply(lambda x: rep_str(x))
    data['pailiang'] = data['engine'].apply(lambda x: rep_str(x, 1))
    data['gang'] = data['engine'].apply(lambda x: rep_str(x, 2))
    print(data.head())


# rep_str
def rep_str(s: string, index: int = 0):
    ss = str(s).split(' ')
    r_ss = [re.sub(r'[a-zA-Z/]', '', s_) for s_ in ss]
    r_ss = r_ss + [None] * (3 - len(r_ss)) if len(r_ss) < 3 else r_ss
    try:
        f = float(r_ss[index])
        return f
    except Exception as e:
        return None


# read_data
def deal_data():
    # 读取文件
    train_data = pd.read_csv(".\\train.csv")
    test_data = pd.read_csv(".\\test.csv")
    device_column(train_data)
    device_column(test_data)
    # 特征工程：品牌、型号年份、里程、发动机、变速器、额外颜色、初始颜色、事故历史、价格
    # brand,model_year,milage,engine,transmission,ext_col,int_col,accident,price
    # 发动机-engine 需要分裂数据
    deal_data_with_avg_price(train_data, test_data, 'brand')
    # deal_data_with_avg_price(train_data, test_data, 'engine')
    deal_data_with_avg_price(train_data, test_data, 'transmission')
    deal_data_with_avg_price(train_data, test_data, 'ext_col')
    deal_data_with_avg_price(train_data, test_data, 'int_col')
    deal_data_with_avg_price(train_data, test_data, 'accident')
    print(train_data.head())
    print(train_data.describe())
    train_data.to_csv('train_deal.csv', index=False)
    test_data.to_csv('test_deal.csv', index=False)


# read_data
def read_data():
    # 读取文件
    train_data = pd.read_csv(".\\train_deal.csv")
    test_data = pd.read_csv(".\\test_deal.csv")
    # 特征提取
    train_data = train_data.loc[:, ['brand', 'model_year', 'milage', 'mali', 'pailiang', 'gang', 'transmission', 'ext_col', 'int_col', 'accident', 'price']]
    test_data = test_data.loc[:, ['brand', 'model_year', 'milage', 'mali', 'pailiang', 'gang', 'transmission', 'ext_col', 'int_col', 'accident', 'id']]
    # 空缺数据补空值--测试集和训练集都要
    col_mean = train_data.mean()
    for col in train_data.columns:
        train_data.fillna(col_mean[col], inplace=True)
        test_data.fillna(col_mean[col], inplace=True)
    # 特征缩放
    train_data = train_data.div(train_data.max())
    normalized_columns = [col for col in test_data.columns if col != 'id']
    test_data[normalized_columns] = test_data[normalized_columns].div(train_data[normalized_columns].max())
    # test_data = test_data.div(train_data.loc[:, ['brand', 'model_year', 'milage', 'mali', 'pailiang', 'gang', 'transmission', 'ext_col', 'int_col', 'accident']].max())
    return train_data, test_data, train_data['price'].max()


# pd_to_np pd格式转换为np格式
def pd_to_np(train_data, test_data: pd.DataFrame):
    # 乱序
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    test_data = test_data.sample(frac=1).reset_index(drop=True)
    train_data_tmp = train_data.loc[:, train_data.columns != 'price']
    train_label_tmp = train_data.loc[:, train_data.columns == 'price']
    test_data_tmp = test_data.loc[:, test_data.columns != 'id']
    test_label_tmp = test_data.loc[:, test_data.columns == 'id']
    return train_data_tmp.to_numpy(), train_label_tmp.to_numpy(), test_data_tmp.to_numpy(), test_label_tmp.to_numpy()


# read_init_model 读取模型，否则初始化模型
def read_init_model(file_name: string) -> keras.src.models.sequential.Sequential:
    model = {}
    try:
        model = tf.keras.models.load_model(file_name)
    except Exception as e:
        model = tf.keras.models.Sequential([
            # 此处加了一层32个神经元，分数提高了一点
            tf.keras.layers.Dense(128, activation='relu'),
            # tf.keras.layers.Dense(256, activation='relu'),
            # tf.keras.layers.Dense(64, activation='relu'),
            # tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=0.01),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
    finally:
        return model
    # 1.建立全连接神经网络，relu作为激活函数，dropout率为0.2


# nn 定义一个全连接神经网络
def nn(train_d, train_l, test_d: pd.DataFrame):
    callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda batch, logs: [callback.on_train_begin])
    # 1.建立全连接神经网络
    model = read_init_model('my_model.keras')
    # 2.调整数据格式
    train_data = train_d.astype(np.float64)
    train_label = train_l.astype(np.float64)
    # 4.1定义交叉熵损失函数
    loss_fn = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer='adam', loss=loss_fn, metrics=['mse'])
    # 4.2定义tensorBoard
    tensorboard_callback = tb.draw_board('secondhandCycle')
    # 5.进行训练
    train_history = model.fit(train_data, train_label, epochs=16, callbacks=[tensorboard_callback])
    # 6.保存模型
    model.save('my_model.keras')


# predict_test_data 预测测试集
def predict_test_data(test_d, test_l: pd.DataFrame, max_price: float):
    model = tf.keras.models.load_model('my_model.keras')
    test_data = test_d
    pre = model.predict(test_data)
    pre = pre * max_price
    # test_l = tf.reshape(tf.convert_to_tensor(test_l[:, 0:]), [len(test_l), 1])
    # pre = tf.reshape(tf.convert_to_tensor(pre[:, 0:]), [len(pre), 1])
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
    # r = rep_str('177.0HP')
    # print(r)
    # 1.数据特征处理
    # deal_data()
    # 1.pd读取数据集，提取相关有用信息并做数据预处理
    train_data, test_data, max_price = read_data()
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(train_data.describe())
    print(test_data.describe())
    # 2.整理字段
    train_d, train_l, test_d, test_l = pd_to_np(train_data, test_data)
    # nn(train_d, train_l, test_data)
    predict_test_data(test_d, test_l, max_price)


if __name__ == "__main__":
    main()
