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
NP_FIELD = ['Age', 'Annual Income', 'Number of Dependents', 'Health Score', 'Premium Amount']
TEST_NP_FIELD = ['Age', 'Annual Income', 'Number of Dependents', 'Health Score', 'id']
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
    index = int(length * 0.8)
    train_d = train_data.loc[:index, :]
    valid_d = train_data.loc[index:, :]
    train_l = train_label.loc[:index, :]
    valid_l = train_label.loc[index:, :]
    data = lgb.Dataset(train_d, label=train_l)
    valid = lgb.Dataset(valid_d, label=valid_l)
    # 添加交叉验证，评分直接 2.81342提升到1.09089
    bst = lgb.train(params, data, num_boost_round=200, valid_sets=valid)
    return bst


def predict(data: lgb.Dataset):
    # init model
    bst = lgb.Booster(model_file='model.txt')
    y_pred = bst.predict(data)
    return y_pred


# main 主函数
def main():
    train_org_data = pd.read_csv(TRAIN_PATH)
    test_org_data = pd.read_csv(TEST_PATH)
    train_data = train_org_data.loc[:, NP_FIELD]
    test_data = test_org_data.loc[:, TEST_NP_FIELD]
    # 对'Marital Status', 'Education Level', 'Occupation', 'Policy Type', 'Property Type'字段进行独热编码
    # TODO--lightgbm原生支持分类编码而无需进行one-hot编码——categorical_feature参数
    for field_name in ONE_HOT_FIELD:
        type_one_hot = one_hot_numpy(train_org_data, field_name)
        train_data = pd.concat([train_data, type_one_hot], axis=1)
    for field_name in ONE_HOT_FIELD:
        type_one_hot = one_hot_numpy(test_org_data, field_name)
        test_data = pd.concat([test_data, type_one_hot], axis=1)
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
    all_data = pd.concat([train_data, test_data])
    scaler = StandardScaler()
    all_data = scaler.fit_transform(all_data)
    # 重新分为训练集和测试集
    train_data = pd.DataFrame(all_data, columns=cols_names).iloc[0:length, :]
    test_data = pd.DataFrame(all_data, columns=cols_names).iloc[length:, :]
    print_dp(train_data.head())
    print_dp(test_data.head())
    # 训练
    bst = train(train_data, train_label)
    bst.save_model('model.txt', num_iteration=bst.best_iteration)
    # 预测
    predictions = predict(test_data)
    test_id['Premium Amount'] = predictions
    output = pd.DataFrame(test_id, columns=['id', 'Premium Amount'])
    output.to_csv('output.csv', index=False)


if __name__ == "__main__":
    main()
