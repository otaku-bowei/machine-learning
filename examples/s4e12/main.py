'''
id,Age,Gender,Annual Income,Marital Status,Number of Dependents,Education Level,Occupation,Health Score,Location,Policy Type,Previous Claims,Vehicle Age,Credit Score,Insurance Duration,Policy Start Date,Customer Feedback,Smoking Status,Exercise Frequency,Property Type,Premium Amount
身份证，年龄，性别，年收入，婚姻状况，受抚养人数，教育水平，职业，健康评分，地点，保单类型，以前的索赔，车龄，信用评分，保险期限，保单开始日期，客户反馈，吸烟状况，运动频率，财产类型，保费金额
'''


'''
https://lightgbm.readthedocs.io/en/latest/Quick-Start.html#run-lightgbm
https://lightgbm.readthedocs.io/en/latest/Features.html
lightgbm特性
1.支持分类特征--对比one-hot编码效率更快
2.支持加权训练
3.
'''
import string
from multiprocessing import cpu_count

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import tool.tensorboard.tensor_board as tb
from tqdm import tqdm
import lightgbm as lgb


def print_dp(data: pd.DataFrame):
    print("\r\n")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(data.head())


# one-hot向量转换为0-1编码
def one_hot_numpy(data: pd.DataFrame, col_name: string) -> pd.DataFrame:
    # 创建一个OneHotEncoder对象
    encoder = OneHotEncoder(sparse_output=False)  # 设置sparse=False以返回密集数组
    one_hot_encoded_array = encoder.fit_transform(
        data[[col_name]])
    one_hot_encoded_df = pd.DataFrame(one_hot_encoded_array, columns=encoder.get_feature_names_out(
        [col_name]))
    return one_hot_encoded_df


# read_csv 读取csv
def read_csv(file_path: string) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df


TRAIN_PATH = './train.csv'
TEST_PATH = './test.csv'
ONE_HOT_FIELD = ['Gender', 'Marital Status', 'Education Level', 'Occupation', 'Policy Type', 'Property Type']
NP_FIELD = ['Age', 'Annual Income', 'Number of Dependents', 'Health Score', 'Premium Amount', 'Policy Start Date']
TEST_NP_FIELD = ['Age', 'Annual Income', 'Number of Dependents', 'Health Score', 'id', 'Policy Start Date']
CATEGORICAL_FEATURE = ['Gender', 'Marital Status', 'Education Level', 'Occupation', 'Location',  'Policy Type', 'Customer Feedback', 'Smoking Status', 'Exercise Frequency', 'Property Type']
params = {
    'bagging_freq': 5,
    'bagging_fraction': 1.0,
    'boost_from_average': 'false',
    'boost': 'gbdt',
    'feature_fraction': 1.0,
    'learning_rate': 0.005,
    'max_depth': -1,
    # 均方差损失标准
    'metric': 'l2',
    'min_data_in_leaf': 30,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 64,
    'num_threads': cpu_count(),
    'tree_learner': 'serial',
    # 回归任务
    'objective': 'regression',
    # 'objective': 'binary',
    'verbosity': -1
}


def train(train_data, train_label: pd.DataFrame):
    # 将训练集分为训练数据和交叉验证数据
    length = len(train_data)
    # 交叉验证集的比例会稍微影响决策树精度
    index = int(length * 0.5)
    train_d = train_data.loc[:index, :]
    valid_d = train_data.loc[index:, :]
    train_l = train_label.loc[:index, :]
    valid_l = train_label.loc[index:, :]
    data = lgb.Dataset(train_d, label=train_l)
    valid = lgb.Dataset(valid_d, label=valid_l)
    # 添加交叉验证，评分直接 2.81342提升到1.09089
    bst = lgb.train(params, data, num_boost_round=200, valid_sets=valid)
    return bst


def train_with_categorical_feature(train_data, train_label : pd.DataFrame):
    # 将训练集分为训练数据和交叉验证数据
    length = len(train_data)
    index = int(length * 0.8)
    train_d = train_data.loc[:index, :]
    valid_d = train_data.loc[index:, :]
    train_l = train_label.loc[:index, :]
    valid_l = train_label.loc[index:, :]
    # FIXME
    data = lgb.Dataset(train_d, label=train_l, categorical_feature=CATEGORICAL_FEATURE)
    valid = lgb.Dataset(valid_d, label=valid_l, categorical_feature=CATEGORICAL_FEATURE)
    # 添加交叉验证，评分直接 2.81342提升到1.09089
    bst = lgb.train(params, data, num_boost_round=200, valid_sets=valid)
    return bst


def one_hot_fix(data, org_data: pd.DataFrame, clo_names: []) -> pd.DataFrame:
    for field_name in clo_names:
        type_one_hot = one_hot_numpy(org_data, field_name)
        data = pd.concat([data, type_one_hot], axis=1)
    return data


def predict(data: lgb.Dataset, model_name: string):
    # init model
    bst = lgb.Booster(model_file=model_name)
    y_pred = bst.predict(data)
    return y_pred


# precision_fix 精度扩展，对某些字段进行精度扩展字段--对决策树处理回归问题有帮助
def precision_fix():
    print()


# nan_to_mean 某些列的 nan 是有意义的，将其标识为平均值，TODO--并用标识列标注该值
def nan_to_mean(data: pd.DataFrame, field: string, num: int = None) -> pd.DataFrame:
    if num is None:
        data.fillna(data.mean(), inplace=True)
    else:
        data.fillna(num, inplace=True)
    return data


# date_fix 处理日期字段
def date_fix(data: pd.DataFrame, field: string) -> pd.DataFrame:
    df = pd.DataFrame(data.loc[:, [field]])
    df['date'] = pd.to_datetime(df[field])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    data = pd.concat([data, df.drop([field, 'date'], axis=1)], axis=1)
    return data.drop([field], axis=1)


def train_by_nn():
    # 1.预处理数据
    train_org_data = pd.read_csv(TRAIN_PATH)
    test_org_data = pd.read_csv(TEST_PATH)
    train_data = train_org_data.loc[:, NP_FIELD]
    test_data = test_org_data.loc[:, TEST_NP_FIELD]
    # 对'Marital Status', 'Education Level', 'Occupation', 'Policy Type', 'Property Type'字段进行独热编码
    # TODO--lightgbm原生支持分类编码而无需进行one-hot编码——categorical_feature参数
    train_data = one_hot_fix(train_data, train_org_data, CATEGORICAL_FEATURE)
    train_data = date_fix(train_data, 'Policy Start Date')
    test_data = one_hot_fix(test_data, test_org_data, CATEGORICAL_FEATURE)
    test_data = date_fix(test_data, 'Policy Start Date')
    # 乱序
    train_data = pd.DataFrame(train_data)
    # 提取label
    train_label = train_data.loc[:, ['Premium Amount']]
    train_data = train_data.drop(['Premium Amount'], axis=1)
    test_id = test_data.loc[:, ['id']]
    test_data = test_data.drop(['id'], axis=1)
    # 标准归一化--和测试集一起做归一化
    length = len(train_data)
    cols_names = train_data.columns
    # 特征工程，处理nan值-- 1.08912提升到1.08899，说明某些字段的nan值是对预测结果有影响的
    all_data = pd.concat([train_data, test_data])
    # for col in cols_names:
    #     all_data = nan_to_mean(all_data, col, -1)
    scaler = StandardScaler()
    all_data = scaler.fit_transform(all_data)
    # 重新分为训练集和测试集
    train_data = pd.DataFrame(all_data, columns=cols_names).iloc[0:length, :]
    test_data = pd.DataFrame(all_data, columns=cols_names).iloc[length:, :]
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(len(cols_names), )),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        # 丢弃部分神经元梯度下降更稳定，防止梯度消失
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    # 4.2定义tensorBoard
    tensorboard_callback = tb.draw_board('mnist')
    # 5.进行训练
    # model.fit(train_data, train_label, epochs=10, callbacks=[tensorboard_callback])
    # model.save('nn_model.keras')
    return test_data, test_id



def nn_pred(test_data, test_id: pd.DataFrame):
    model = tf.keras.models.load_model('nn_model.keras')
    y_test = model.predict(test_data)
    test_id['Premium Amount'] = y_test
    output = pd.DataFrame(test_id, columns=['id', 'Premium Amount'])
    output.to_csv('output.csv', index=False)



def train_without_one_hot():
    train_org_data = pd.read_csv(TRAIN_PATH)
    test_org_data = pd.read_csv(TEST_PATH)
    train_data = train_org_data.drop(['Policy Start Date'], axis=1)
    test_data = test_org_data.drop(['Policy Start Date'], axis=1)
    # 区分值
    train_label = train_data.loc[:, ['Premium Amount']]
    train_data = train_data.drop(['Premium Amount'], axis=1)
    test_id = test_data.loc[:, ['id']]
    test_data = test_data.drop(['id'], axis=1)
    bst = train_with_categorical_feature(train_data, train_label)
    bst.save_model('model2.txt', num_iteration=bst.best_iteration)
    # 预测
    predictions = predict(lgb.Dataset(test_data), 'model2.txt')
    test_id['Premium Amount'] = predictions
    output = pd.DataFrame(test_id, columns=['id', 'Premium Amount'])
    output.to_csv('output.csv', index=False)


def train_by_lightgbm():
    train_org_data = pd.read_csv(TRAIN_PATH)
    test_org_data = pd.read_csv(TEST_PATH)
    train_data = train_org_data.loc[:, NP_FIELD]
    test_data = test_org_data.loc[:, TEST_NP_FIELD]
    # 对'Marital Status', 'Education Level', 'Occupation', 'Policy Type', 'Property Type'字段进行独热编码
    # TODO--lightgbm原生支持分类编码而无需进行one-hot编码——categorical_feature参数
    train_data = one_hot_fix(train_data, train_org_data, CATEGORICAL_FEATURE)
    train_data = date_fix(train_data, 'Policy Start Date')
    test_data = one_hot_fix(test_data, test_org_data, CATEGORICAL_FEATURE)
    test_data = date_fix(test_data, 'Policy Start Date')
    # 乱序
    train_data = pd.DataFrame(train_data)
    # 提取label
    train_label = train_data.loc[:, ['Premium Amount']]
    train_data = train_data.drop(['Premium Amount'], axis=1)
    test_id = test_data.loc[:, ['id']]
    test_data = test_data.drop(['id'], axis=1)
    # 标准归一化--和测试集一起做归一化--FIXME--去除归一化后由1.08992提升到1.08988
    length = len(train_data)
    cols_names = train_data.columns
    # 特征工程，处理nan值-- 1.08912提升到1.08899，说明某些字段的nan值是对预测结果有影响的
    all_data = pd.concat([train_data, test_data])
    for col in cols_names:
        all_data = nan_to_mean(all_data, col, -1)
    # scaler = StandardScaler()
    # all_data = scaler.fit_transform(all_data)
    # 重新分为训练集和测试集
    train_data = pd.DataFrame(all_data, columns=cols_names).iloc[0:length, :]
    test_data = pd.DataFrame(all_data, columns=cols_names).iloc[length:, :]
    # print_dp(train_data.head())
    # print_dp(test_data.head())
    # 训练
    bst = train(train_data, train_label)
    bst.save_model('model.txt', num_iteration=bst.best_iteration)
    # 预测
    predictions = predict(test_data, 'model.txt')
    test_id['Premium Amount'] = predictions
    output = pd.DataFrame(test_id, columns=['id', 'Premium Amount'])
    output.to_csv('output.csv', index=False)


# main 主函数
def main():
    test_data, test_id = train_by_nn()
    nn_pred(test_data, test_id)



if __name__ == "__main__":
    main()
