import string
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import tool.tensorboard.tensor_board as tb


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
    # 特征缩放：年龄、票价
    print(train_data.describe())
    # 年龄为空的填平均值
    age_mean = train_data['Age'].mean()
    train_data.loc[train_data['Age'].isnull(), ['Age']] = age_mean
    test_data.loc[test_data['Age'].isnull(), ['Age']] = age_mean
    return (train_data.loc[:, ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']],
            test_data.loc[:, ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'PassengerId']])


# pd_to_np pd格式转换为np格式
def pd_to_np(train_data, test_data: pd.DataFrame):
    # 乱序
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    test_data = test_data.sample(frac=1).reset_index(drop=True)
    train_data_tmp = train_data.loc[:, train_data.columns != 'Survived']
    train_label_tmp = train_data.loc[:, train_data.columns == 'Survived']
    test_data_tmp = test_data.loc[:, test_data.columns != 'PassengerId']
    test_label_tmp = test_data.loc[:, test_data.columns == 'PassengerId']
    return train_data_tmp.to_numpy(), train_label_tmp.to_numpy(), test_data_tmp.to_numpy(), test_label_tmp.to_numpy()


# read_init_model 读取模型，否则初始化模型
def read_init_model(file_name: string) -> keras.src.models.sequential.Sequential:
    model = {}
    try:
        model = tf.keras.models.load_model(file_name)
    except Exception as e:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(8, activation='relu'),
            # tf.keras.layers.Dropout(0.2),
            # tf.keras.layers.Dense(16, activation='relu'),
            # tf.keras.layers.Dropout(0.2),
            # tf.keras.layers.Dense(6, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    finally:
        return model
    # 1.建立全连接神经网络，relu作为激活函数，dropout率为0.2


# nn 定义一个全连接神经网络
def nn(train_d, train_l, test_d: pd.DataFrame):
    callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda batch, logs: [callback.on_train_begin])
    train_d = train_d.astype(np.float64)
    train_l = train_l.astype(np.float64)
    # 1.建立全连接神经网络，relu作为激活函数，dropout率为0.2
    model = read_init_model('my_model.keras')
    # 2.调整数据格式
    train_len = len(train_d)
    # print(type(train_d))
    # train_data = tf.reshape(tf.convert_to_tensor(train_d[:, 0:]), [train_len, 1, 6])
    # train_label = tf.reshape(tf.convert_to_tensor(train_l[:, 0:]), [train_len, 1, 1])
    train_data = train_d
    train_label = train_l
    # 3.用logistic进行二分类
    # predictions = model(train_data[:1]).numpy()
    # tf.nn.sigmoid(predictions).numpy()
    # 4.1定义交叉熵损失函数
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    # loss_fn(train_label[:1], predictions).numpy()
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    # 4.2定义tensorBoard
    tensorboard_callback = tb.draw_board('titanic')
    # 5.进行训练
    train_history = model.fit(train_data, train_label, epochs=256, callbacks=[tensorboard_callback])
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
    output = pd.DataFrame(result, columns=['PassengerId', 'Survived'])
    output.to_csv('submission.csv', index=False)


# def draw_train_function():


# main 主函数
def main():
    # 1.pd读取数据集，提取相关有用信息并做数据预处理
    train_data, test_data = read_data()
    # 2.整理字段
    train_d, train_l, test_d, test_l = pd_to_np(train_data, test_data)
    nn(train_d, train_l, test_data)
    # predict_test_data(test_d, test_l)


if __name__ == "__main__":
    main()
