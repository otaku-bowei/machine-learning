import string
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
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


# read_data
def deal_data():
    # 读取文件
    train_data = pd.read_csv(".\\train.csv")
    test_data = pd.read_csv(".\\test.csv")
    # 特征工程：品牌、型号年份、里程、发动机、变速器、额外颜色、初始颜色、事故历史、价格
    # brand,model_year,milage,engine,transmission,ext_col,int_col,accident,price
    deal_data_with_avg_price(train_data, test_data, 'brand')
    deal_data_with_avg_price(train_data, test_data, 'engine')
    deal_data_with_avg_price(train_data, test_data, 'transmission')
    deal_data_with_avg_price(train_data, test_data, 'ext_col')
    deal_data_with_avg_price(train_data, test_data, 'int_col')
    deal_data_with_avg_price(train_data, test_data, 'accident')
    print(train_data.head())
    print(train_data.describe())
    train_data.to_csv('train_deal.csv', index=False)
    train_data.to_csv('test_deal.csv', index=False)


# read_data
def read_data():
    # 读取文件
    train_data = pd.read_csv(".\\train_deal.csv")
    test_data = pd.read_csv(".\\test_deal.csv")
    # 空缺数据补空值--测试集和训练集都要
    train_data.loc[test_data['accident'].isnull(), ['accident']] = 49024.80414354319
    test_data.loc[test_data['accident'].isnull(), ['accident']] = 49024.80414354319
    train_data = train_data.loc[:, ['brand', 'model_year', 'milage', 'engine', 'transmission', 'ext_col', 'int_col', 'accident', 'price']]
    test_data = test_data.loc[:, ['brand', 'model_year', 'milage', 'engine', 'transmission', 'ext_col', 'int_col', 'accident', 'id']]
    return train_data, test_data


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
            # 特征缩放
            tf.keras.layers.experimental.preprocessing.Normalization(),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
    finally:
        return model
    # 1.建立全连接神经网络，relu作为激活函数，dropout率为0.2


# nn 定义一个全连接神经网络
def nn(train_d, train_l, test_d: pd.DataFrame):
    callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda batch, logs: [callback.on_train_begin])
    # 1.建立全连接神经网络，relu作为激活函数，dropout率为0.2
    model = read_init_model('my_model.keras')
    # 2.调整数据格式
    train_data = train_d.astype(np.float64)
    train_label = train_l.astype(np.float64)
    # 4.1定义交叉熵损失函数
    loss_fn = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    # 4.2定义tensorBoard
    tensorboard_callback = tb.draw_board('secondhandCycle')
    # 5.进行训练
    train_history = model.fit(train_data, train_label, epochs=10, callbacks=[tensorboard_callback])
    # 6.保存模型
    model.save('my_model.keras')


# predict_test_data 预测测试集
def predict_test_data(test_d, test_l: pd.DataFrame):
    model = tf.keras.models.load_model('my_model.keras')
    test_d = test_d.astype(np.float64)
    test_data = tf.reshape(tf.convert_to_tensor(test_d[:, 0:]), [len(test_d), 1, 6])
    pre = model.predict(test_data)
    test_l = tf.reshape(tf.convert_to_tensor(test_l[:, 0:]), [len(test_l), 1])
    pre = tf.reshape(tf.convert_to_tensor(pre[:, 0:]), [len(pre), 1])
    pre = np.round(pre).astype(int)
    result = np.concatenate((test_l, pre), axis=1)
    output = pd.DataFrame(result, columns=['id', 'price'])
    output.to_csv('submission.csv', index=False)


# def draw_train_function():


# main 主函数
def main():
    # 1.数据特征处理
    # deal_data()
    # 1.pd读取数据集，提取相关有用信息并做数据预处理
    train_data, test_data = read_data()
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(train_data.describe())
    # 2.整理字段
    train_d, train_l, test_d, test_l = pd_to_np(train_data, test_data)
    nn(train_d, train_l, test_data)
    # predict_test_data(test_d, test_l)


if __name__ == "__main__":
    main()
