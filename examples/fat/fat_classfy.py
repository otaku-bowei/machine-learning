import re
import string
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import tool.tensorboard.tensor_board as tb

"""
id：未明确说明，但通常此类字段用于唯一标识每条数据记录。
Gender：性别。
Age：年龄。
Height：身高。
Weight：体重。
family_history_with_overweight：家族肥胖史。
FAVC：是否频繁食用高热量食物。
FCVC：食用蔬菜的频次。
NCP：食用主餐的次数。
CAEC：两餐之间的食品消费情况，如总是、经常、有时候。
SMOKE：是否吸烟。
CH2O：每日耗水量。
SCC：高热量饮料消耗量。
FAF：运动频率。
TUE：使用电子设备的时间。
CALC：酒精消耗量，如无、经常、有时候。
MTRANS：日常交通方式，如汽车、自行车、摩托车、公共交通、步行。
NObeyesdad：肥胖水平，包括正常、体重不足、肥胖类型I、肥胖类型II、肥胖类型III、一级超重、二级超重等1。
"""


# deal_data
def deal_data():
    # 读取文件
    train_data = pd.read_csv(".\\train.csv")
    test_data = pd.read_csv(".\\test.csv")
    # 数据预处理：性别、家族肥胖史
    train_data.loc[train_data['Gender'] == 'Male', 'Gender'] = 1
    train_data.loc[train_data['Gender'] == 'Female', 'Gender'] = 0
    test_data.loc[test_data['Gender'] == 'Male', 'Gender'] = 1
    test_data.loc[test_data['Gender'] == 'Female', 'Gender'] = 0

    train_data.loc[train_data['family_history_with_overweight'] == 'yes', 'family_history_with_overweight'] = 1
    train_data.loc[train_data['family_history_with_overweight'] == 'no', 'family_history_with_overweight'] = 0
    test_data.loc[test_data['family_history_with_overweight'] == 'yes', 'family_history_with_overweight'] = 1
    test_data.loc[test_data['family_history_with_overweight'] == 'no', 'family_history_with_overweight'] = 0
    train_data.loc[train_data['FAVC'] == 'yes', 'FAVC'] = 1
    train_data.loc[train_data['FAVC'] == 'no', 'FAVC'] = 0
    test_data.loc[test_data['FAVC'] == 'yes', 'FAVC'] = 1
    test_data.loc[test_data['FAVC'] == 'no', 'FAVC'] = 0
    train_data.loc[train_data['CAEC'] == 'Sometimes', 'CAEC'] = 1
    train_data.loc[train_data['CAEC'] == 'Always', 'CAEC'] = 3
    test_data.loc[test_data['CAEC'] == 'Sometimes', 'CAEC'] = 1
    test_data.loc[test_data['CAEC'] == 'Always', 'CAEC'] = 3
    train_data.loc[train_data['CAEC'] == 'Frequently', 'CAEC'] = 2
    test_data.loc[test_data['CAEC'] == 'Frequently', 'CAEC'] = 2
    train_data.loc[train_data['CAEC'] == 'no', 'CAEC'] = 0
    test_data.loc[test_data['CAEC'] == 'no', 'CAEC'] = 0
    train_data.loc[train_data['SMOKE'] == 'yes', 'SMOKE'] = 1
    train_data.loc[train_data['SMOKE'] == 'no', 'SMOKE'] = 0
    test_data.loc[test_data['SMOKE'] == 'yes', 'SMOKE'] = 1
    test_data.loc[test_data['SMOKE'] == 'no', 'SMOKE'] = 0
    train_data.loc[train_data['SCC'] == 'yes', 'SCC'] = 1
    train_data.loc[train_data['SCC'] == 'no', 'SCC'] = 0
    test_data.loc[test_data['SCC'] == 'yes', 'SCC'] = 1
    test_data.loc[test_data['SCC'] == 'no', 'SCC'] = 0

    train_data.loc[train_data['CALC'] == 'Sometimes', 'CALC'] = 1
    train_data.loc[train_data['CALC'] == 'Always', 'CALC'] = 3
    test_data.loc[test_data['CALC'] == 'Sometimes', 'CALC'] = 1
    test_data.loc[test_data['CALC'] == 'Always', 'CALC'] = 3
    train_data.loc[train_data['CALC'] == 'Frequently', 'CALC'] = 2
    test_data.loc[test_data['CALC'] == 'Frequently', 'CALC'] = 2
    train_data.loc[train_data['CALC'] == 'no', 'CALC'] = 0
    test_data.loc[test_data['CALC'] == 'no', 'CALC'] = 0

    train_data.loc[train_data['MTRANS'] == 'Public_Transportation', 'MTRANS'] = 3
    train_data.loc[train_data['MTRANS'] == 'Automobile', 'MTRANS'] = 0
    train_data.loc[train_data['MTRANS'] == 'Walking', 'MTRANS'] = 5
    train_data.loc[train_data['MTRANS'] == 'Motorbike', 'MTRANS'] = 1
    train_data.loc[train_data['MTRANS'] == 'Bike', 'MTRANS'] = 10
    test_data.loc[test_data['MTRANS'] == 'Public_Transportation', 'MTRANS'] = 3
    test_data.loc[test_data['MTRANS'] == 'Automobile', 'MTRANS'] = 0
    test_data.loc[test_data['MTRANS'] == 'Walking', 'MTRANS'] = 5
    test_data.loc[test_data['MTRANS'] == 'Motorbike', 'MTRANS'] = 1
    test_data.loc[test_data['MTRANS'] == 'Bike', 'MTRANS'] = 10

    # 结果分类
    train_data.loc[train_data['NObeyesdad'] == 'Insufficient_Weight', 'NObeyesdad'] = 0
    train_data.loc[train_data['NObeyesdad'] == 'Normal_Weight', 'NObeyesdad'] = 1
    train_data.loc[train_data['NObeyesdad'] == 'Obesity_Type_I', 'NObeyesdad'] = 2
    train_data.loc[train_data['NObeyesdad'] == 'Obesity_Type_II', 'NObeyesdad'] = 3
    train_data.loc[train_data['NObeyesdad'] == 'Obesity_Type_III', 'NObeyesdad'] = 4
    train_data.loc[train_data['NObeyesdad'] == 'Overweight_Level_I', 'NObeyesdad'] = 5
    train_data.loc[train_data['NObeyesdad'] == 'Overweight_Level_II', 'NObeyesdad'] = 6

    train_data.to_csv('train_deal.csv', index=False)
    test_data.to_csv('test_deal.csv', index=False)


# read_data
def read_data():
    # 读取文件
    train_data = pd.read_csv(".\\train_deal.csv")
    test_data = pd.read_csv(".\\test_deal.csv")
    return train_data, test_data


# pd_to_np pd格式转换为np格式
def pd_to_np(train_data, test_data: pd.DataFrame):
    # 乱序
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    test_data = test_data.sample(frac=1).reset_index(drop=True)
    train_data = train_data.drop('id', axis=1)
    train_data_tmp = train_data.loc[:, train_data.columns != 'NObeyesdad']
    train_label_tmp = train_data.loc[:, train_data.columns == 'NObeyesdad']
    test_data_tmp = test_data.loc[:, test_data.columns != 'id']
    test_label_tmp = test_data.loc[:, test_data.columns == 'id']
    # 注意顺序，先缩放测试集，因为缩放过程需要根据训练集的最大值
    test_data_tmp = test_data_tmp.div(train_data_tmp.max())
    train_data_tmp = train_data_tmp.div(train_data_tmp.max())
    return train_data_tmp.to_numpy(), train_label_tmp.to_numpy(), test_data_tmp.to_numpy(), test_label_tmp.to_numpy()


# read_init_model 读取模型，否则初始化模型
def read_init_model(file_name: string) -> keras.src.models.sequential.Sequential:
    model = {}
    try:
        model = tf.keras.models.load_model(file_name)
    except Exception as e:
        model = tf.keras.models.Sequential([
            # 此处加了一层32个神经元，分数提高了一点
            tf.keras.layers.Dense(112, activation='relu'),
            # tf.keras.layers.Dense(28, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            # 此处激活函数帮我跳出了局部最小值，线性激活函数输出的是z1-z6,用softmax才会输出概率值
            tf.keras.layers.Dense(7, activation='softmax')
        ])
    finally:
        return model


# nn 定义一个全连接神经网络
def nn(train_d, train_l: pd.DataFrame):
    callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda batch, logs: [callback.on_train_begin])
    # 1.建立全连接神经网络
    model = read_init_model('my_model.keras')
    # 2.调整数据格式
    train_data = train_d.astype(np.float64)
    train_label = train_l.astype(np.float64)
    # 3.用softmax进行多分类
    predictions = model(train_data[:1]).numpy()
    tf.nn.softmax(predictions).numpy()
    # 4.1定义交叉熵损失函数
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    # 4.2定义tensorBoard
    tensorboard_callback = tb.draw_board('fat_classfy')
    # 5.进行训练
    train_history = model.fit(train_data, train_label, epochs=100, callbacks=[tensorboard_callback])
    # 6.保存模型
    model.save('my_model.keras')


# predict_test_data 预测测试集
def predict_test_data(test_d, test_l: pd.DataFrame):
    model = tf.keras.models.load_model('my_model.keras')
    test_data = test_d
    pre = model.predict(test_data)
    max_indices = np.matrix(np.argmax(pre, axis=1)).T
    result = np.concatenate((test_l.astype(int), max_indices), axis=1)
    # print(result[0:5])
    output = pd.DataFrame(result, columns=['id', 'NObeyesdad'])
    output.to_csv('submission.csv', index=False)

def deal_result():
    """
    结果分类
    train_data.loc[train_data['NObeyesdad'] == 'Insufficient_Weight', 'NObeyesdad'] = 0
    train_data.loc[train_data['NObeyesdad'] == 'Normal_Weight', 'NObeyesdad'] = 1
    train_data.loc[train_data['NObeyesdad'] == 'Obesity_Type_I', 'NObeyesdad'] = 2
    train_data.loc[train_data['NObeyesdad'] == 'Obesity_Type_II', 'NObeyesdad'] = 3
    train_data.loc[train_data['NObeyesdad'] == 'Obesity_Type_III', 'NObeyesdad'] = 4
    train_data.loc[train_data['NObeyesdad'] == 'Overweight_Level_I', 'NObeyesdad'] = 5
    train_data.loc[train_data['NObeyesdad'] == 'Overweight_Level_II', 'NObeyesdad'] = 6
    """
    r = pd.read_csv('.\\submission.csv')
    r.loc[r['NObeyesdad'] == 0, 'NObeyesdad'] = 'Insufficient_Weight'
    r.loc[r['NObeyesdad'] == 1, 'NObeyesdad'] = 'Normal_Weight'
    r.loc[r['NObeyesdad'] == 2, 'NObeyesdad'] = 'Obesity_Type_I'
    r.loc[r['NObeyesdad'] == 3, 'NObeyesdad'] = 'Obesity_Type_II'
    r.loc[r['NObeyesdad'] == 4, 'NObeyesdad'] = 'Obesity_Type_III'
    r.loc[r['NObeyesdad'] == 5, 'NObeyesdad'] = 'Overweight_Level_I'
    r.loc[r['NObeyesdad'] == 6, 'NObeyesdad'] = 'Overweight_Level_II'
    r.to_csv('submission.csv', index=False)


# main 主函数
def main():
    # 1.数据特征处理
    # deal_data()
    # 1.pd读取数据集，提取相关有用信息并做数据预处理
    train_data, test_data = read_data()
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    # print(train_data.describe())
    # print(test_data.describe())
    # # 2.整理字段
    train_d, train_l, test_d, test_l = pd_to_np(train_data, test_data)
    # nn(train_d, train_l)
    predict_test_data(test_d, test_l)
    deal_result()


if __name__ == "__main__":
    main()


# 649/649 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.1428 - loss: 1.9465
# 训练几次后发现梯度消失了