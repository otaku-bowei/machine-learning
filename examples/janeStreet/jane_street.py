import string
from abc import ABC

import keras
import numpy as np
import pandas as pd
import pyarrow as pa
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pyarrow.parquet as pq
import tensorflow as tf
import tool.tensorboard.tensor_board as tb

'''
目标:预测responder_6
数据:
    (1)date_id和time_id--时间序列id,但顺序时间结构并不具备相同间隔
    (2)symbol_id--金融资产id
    (3)weight--计算得分函数的权重
    (4)feature_{00...78}--79个匿名特征
    (5)responder_{0...8}--9个匿名响应，范围是(-5,5),需要预测responder_6
测试集:
    (1)is_scored--标识此行是否在评估量度计算中,某些日期不计分,这可能包括训练和公开排行榜/公开排行榜到私人排行榜之间的过渡期
    (2)date_id和time_id--时间序列id,但顺序时间结构并不具备相同间隔
    (3)symbol_id--金融资产id
    (4)weight--计算得分函数的权重
    (5)feature_{00...78}--79个匿名特征
需要处理的问题:
    (1)symbol_id按计算编码分类
    (2)回归预测
    (3)lags的使用
    (4)匿名特征的空值处理--添加特征存在标志标识数据
    (5)responder的关系
    (6)权重对得分函数的影响
    (7)is_scored对梯度的影响
要求:
    (1)在线学习--顺序要求
    (2)基于time-series 模块进行预测--顺序要求
    (3)损失函数
'''


# read_org_data 读取数据
def read_org_data():
    train_date_0_path = "C:\Program Files\BusinessFile\新手村\AI培训资料\数据集\Jane Street\\train.parquet\partition_id=0\part-0.parquet"
    train_date_1_path = "C:\Program Files\BusinessFile\新手村\AI培训资料\数据集\Jane Street\\train.parquet\partition_id=1\part-0.parquet"
    lags_path = "C:\Program Files\BusinessFile\新手村\AI培训资料\数据集\Jane Street\lags.parquet\date_id=0\part-0.parquet"
    test_path = "C:\Program Files\BusinessFile\新手村\AI培训资料\数据集\Jane Street\\test.parquet\date_id=0\part-0.parquet"
    train_data_0 = pd.read_parquet(train_date_0_path)
    train_data_1 = pd.read_parquet(train_date_1_path)
    lags = pd.read_parquet(lags_path)
    test_data = pd.read_parquet(test_path)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    # print(train_data_0.head())
    # print(train_data_1.head())
    # print(lags.head())
    # print(test_data.head())
    # return pd.concat([train_data_0, train_data_1], axis=0), lags_path, test_data
    return train_data_0, lags_path, test_data


# read_deal_data 读取数据
def read_deal_data(name : string):
    train_data_1 = pd.read_parquet(name)
    return train_data_1


# deal_data 对数据进行预处理≈1h
def deal_data(data: pd.DataFrame, name : string, index : int):
    # 1.填充空值，添加相关指示值
    # 乱序
    data = data.drop(columns=['symbol_id'])
    # data = data.drop(columns=['is_scored'])
    # concat
    data_len = len(data)
    var_len = len(data.iloc[0])
    # 79个特征需要添加标识列，标识该值是否为空,为空为1*weight,不为空为0
    X_cnt = np.zeros((len(data), var_len + 79))
    for j in tqdm(range(data_len)):
    # for j in tqdm(range(0, 5000)):
        train_row = data.iloc[j]
        weight = train_row['weight']
        for i in range(len(train_row)):
            if i < index:
                X_cnt[j, i] = train_row[i]
            elif i > index + 79 - 1:
                X_cnt[j, i + 79] = train_row[i]
            elif train_row[i] is None or train_row[i] == '' or np.isnan(train_row[i]):
                X_cnt[j, i] = 0
                X_cnt[j, i + 79] = weight
            else:
                X_cnt[j, i] = train_row[i]
                X_cnt[j, i + 79] = 0
    save_data = pd.DataFrame(X_cnt)
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # print(save_data.head(5))
    table = pa.Table.from_pandas(save_data)
    pq.write_table(table, name)


# 数据缩放--去除
def scaling(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

# Cost 损失函数
class Cost(tf.keras.losses.Loss):

    # 默认权重为1
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights
        self.count = 0

    def call(self, y_true, y_pred):
        weight = self.weights[self.count]
        # 确保y_true和y_pred是相同形状的张量
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        # 确保权重与y_true的形状相同，或者能够通过广播匹配
        weight = tf.broadcast_to(weight, y_true.shape)
        # 计算分子：加权残差平方和
        numerator = tf.reduce_sum(weight * tf.square(y_true - y_pred))
        # 计算分母：加权真实值平方和
        denominator = tf.reduce_sum(weight * tf.square(y_true))
        # 计算R²
        r_squared = 1 - (numerator / denominator)
        # 损失函数应该是要最小化的值，所以返回-R²（因为TensorFlow通常是最小化损失）
        # 但是，由于R²本身已经是1减去一个正数，所以直接返回-r_squared是不正确的。
        # 我们应该返回负的R²的分子部分（即残差平方和的加权和），这样当它被最小化时，R²会最大化。
        # 因此，损失实际上是 -R² + 1 的分子部分，即 numerator / denominator 的部分。
        # 但由于我们想要最大化R²，所以我们只返回numerator（因为denominator是常数对于给定的数据集）。
        # 不过，为了保持损失函数的一致性（即损失越小越好），我们仍然返回numerator的正值作为损失。
        # 注意：这里的处理有点微妙，因为通常我们想要最大化R²，但损失函数应该被最小化。
        # 因此，我们返回numerator作为损失，这样当numerator减小时（即预测更准确时），损失也减小。

        # 实际上，我们应该返回一个与R²相反方向的量度，即残差平方和的加权和（numerator）
        self.count += 1
        return - r_squared


# 模拟数据流,将数据切割为10条一组
# def data_div_batch(data : pd.DataFrame, step : int):
#     batch = int(len(data)/step)
#     result = np.arange()
#     for i in tqdm(range(batch)):


# read_init_model 读取模型，否则初始化模型
def read_init_model(file_name: string) -> keras.src.models.sequential.Sequential:
    model = {}
    try:
        model = tf.keras.models.load_model(file_name)
    except Exception as e:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            # tf.keras.layers.Dropout(0.2),
            # 此处激活函数帮我跳出了局部最小值，线性激活函数输出的是z1-z6,用softmax才会输出概率值
            tf.keras.layers.Dense(9, activation='relu')
        ])
    finally:
        return model


# nn 定义一个全连接神经网络
def nn(train_d, train_l: pd.DataFrame):
    callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda batch, logs: [callback.on_train_begin])
    # 1.建立全连接神经网络
    model = read_init_model('my_model_self_cost.keras')
    # 2.调整数据格式
    train_data = train_d.astype(np.float64)
    train_label = train_l.astype(np.float64)
    # 4.1定义损失函数--得分损失函数,用均方误差作为反向传播基准
    loss_fn = Cost(weights=train_data['weights'])
    # loss_fn = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer='adam', loss=loss_fn, metrics=['mae'])
    # 4.2定义tensorBoard
    tensorboard_callback = tb.draw_board('jane_street')
    # 5.进行训练
    """
    1.mae损失函数，简单的nn，mae≈5.8，loss≈9.4
    2.根据得分函数自定义损失函数，nn不变，
    """
    train_history = model.fit(train_data, train_label, epochs=10, callbacks=[tensorboard_callback])
    # 6.保存模型
    model.save('my_model.keras')


# train 开始训练
def train(train_data : pd.DataFrame):
    # 第二列为权重
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    train_label = train_data.iloc[:, -9:]
    train_data = train_data.iloc[:, :161]
    train_data = train_data.rename(columns={2:'weights'})
    nn(train_data, train_label)


# main 主函数
def main():
    # 1.pd读取数据集，提取相关有用信息并做数据预处理
    # train_data, lag_data, test_data = read_org_data()
    # deal_data(train_data, 'deal_train_data_0.parquet', 3)
    train_data = read_deal_data('deal_train_data.parquet')
    # test = read_deal_data('deal_test_data.parquet')
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # print(train.describe())
    # print(test.describe())
    train(train_data)


if __name__ == "__main__":
    main()
