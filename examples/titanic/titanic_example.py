import numpy as np
import pandas as pd
import tensorflow as tf


# read_data
def read_data():
    # 读取文件
    train_data = pd.read_csv(".\\train.csv")
    test_data = pd.read_csv(".\\test.csv")
    test_label = pd.read_csv(".\\gender_submission.csv")
    # 测试集需要做联表
    left = test_data.set_index(['PassengerId'])
    right = test_label.set_index(['PassengerId'])
    test_data = left.join(right, on='PassengerId', rsuffix='_r', lsuffix='_l')
    test_data = test_data.reset_index()
    # 性别需要做映射
    train_data.loc[train_data['Sex'] == 'male', ['Sex']] = 1
    train_data.loc[train_data['Sex'] == 'female', ['Sex']] = 0
    test_data.loc[test_data['Sex'] == 'male', ['Sex']] = 1
    test_data.loc[test_data['Sex'] == 'female', ['Sex']] = 0
    # 年龄为空的填平均值
    age_mean = train_data['Age'].mean()
    train_data.loc[train_data['Age'].isnull(), ['Age']] = age_mean
    test_data.loc[test_data['Age'].isnull(), ['Age']] = age_mean
    return (train_data.loc[:, ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']],
            test_data.loc[:, ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']])


# pd_to_np pd格式转换为np格式
def pd_to_np(train_data, test_data: pd.DataFrame):
    train_data_tmp = train_data.loc[:, train_data.columns != 'Survived']
    train_label_tmp = train_data.loc[:, train_data.columns == 'Survived']
    test_data_tmp = test_data.loc[:, test_data.columns != 'Survived']
    test_label_tmp = test_data.loc[:, test_data.columns == 'Survived']
    return train_data_tmp.to_numpy(), train_label_tmp.to_numpy(), test_data_tmp.to_numpy(), test_label_tmp.to_numpy()


# nn 定义一个全连接神经网络
def nn(train_d, train_l: pd.DataFrame):
    train_d = train_d.astype(np.float64)
    train_l = train_l.astype(np.float64)
    # 1.建立全连接神经网络，relu作为激活函数，dropout率为0.2
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(1, 6)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    # 2.调整数据格式
    train_len = len(train_d)
    train_data = tf.reshape(tf.convert_to_tensor(train_d[:, 0:]), [train_len, 1, 6])
    train_label = tf.reshape(tf.convert_to_tensor(train_l[:, 0:]), [train_len, 1, 1])
    # 3.用softmax进行多分类
    predictions = model(train_data[:1]).numpy()
    tf.nn.sigmoid(predictions).numpy()
    # 4.定义交叉熵损失函数
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    loss_fn(train_label[:1], predictions).numpy()
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    # 5.进行训练
    model.fit(train_data, train_label, epochs=5)

def main():
    # 1.pd读取数据集，提取相关有用信息并做数据预处理
    train_data, test_data = read_data()
    train_d, train_l, test_d, test_l = pd_to_np(train_data, test_data)
    nn(train_d, train_l)


if __name__ == "__main__":
    main()
