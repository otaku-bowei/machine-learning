from enum import Enum

import sklearn.metrics
from hillclimbers import climb_hill, partial
from sklearn.compose import ColumnTransformer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
import string
from multiprocessing import cpu_count
import tool.matplotlib.field_analysis as fa
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, MinMaxScaler
import tool.tensorboard.tensor_board as tb
from tqdm import tqdm
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
import holidays

'''
id,date,country,store,product,num_sold
id,日期,国家,商店名,产品名,销售数量
'''

'''
1. 查看数据情况
2. 处理数据特征
'''


# print_dp 分析pandas统计数据
def print_dp(data: pd.DataFrame):
    print("\r\n")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(data.head())
    print("\r\n")
    print(data.describe())
    print("\r\n")
    print(data.info())


# read_csv 读取csv
def read_csv(file_path: string) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df


TRAIN_PATH = './train.csv'
TEST_PATH = './test.csv'

params = {
    'bagging_freq': 5,
    'bagging_fraction': 1.0,
    'boost_from_average': 'false',
    'boost': 'gbdt',
    'feature_fraction': 1.0,
    'learning_rate': 0.005,
    'max_depth': -1,
    # 均方差损失标准
    'metric': 'rmse',
    # 'metric': 'l2',
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


# deal_data 数据特征处理
def deal_data(train_data, test_data: pd.DataFrame):
    print()
    # 1.对销售数据为空的数据补0
    train_data = train_data.fillna(50.0)
    # 2.训练集乱序
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    # 3.特征和标签区分
    train_label = train_data.loc[:, ['num_sold']]
    train_data = train_data.drop(['num_sold', 'id'], axis=1)
    test_id = test_data.loc[:, ['id']]
    test_data = test_data.drop(['id'], axis=1)
    # 4.训练集和测试集进行特征处理
    length = len(train_data)
    all_data = pd.concat([train_data, test_data], axis=0)
    all_data = deal_feature(all_data)
    cols_names = all_data.columns
    print_dp(all_data)
    # 5.再分为训练集和测试集
    train_data = pd.DataFrame(all_data, columns=cols_names).iloc[0:length, :]
    test_data = pd.DataFrame(all_data, columns=cols_names).iloc[length:, :]
    return train_data, train_label, test_data, test_id


# deal_feature 特征处理
def deal_feature(data: pd.DataFrame) -> pd.DataFrame:
    field = 'date'
    date_str = 'date_tmp'
    # 1.将日期分割
    df = pd.DataFrame()
    df[field] = data.loc[:, [field]]
    df[date_str] = pd.to_datetime(df[field])
    # df['year'] = df['date'].dt.year
    df['month'] = df[date_str].dt.month
    df['day'] = df[date_str].dt.day
    df['weekday'] = df[date_str].dt.weekday
    df['day_of_week'] = df[date_str].dt.day_of_week
    data = pd.concat([data, df.drop([field, date_str], axis=1)], axis=1)
    # 2.标注节假日
    # data['holiday'] = data.apply(is_holiday, axis=1)
    data = get_holiday(data)
    data = data.drop(field, axis=1).reset_index(drop=True)
    # 3.将店名和产品名做one-hot向量处理
    col_names = ['country', 'store', 'product']
    for field_name in col_names:
        type_one_hot = one_hot_numpy(data, field_name)
        data = pd.concat([data, type_one_hot], axis=1)
        data = data.drop(field_name, axis=1).reset_index(drop=True)
    return data


def get_holiday(data: pd.DataFrame):
    alpha2 = dict(zip(np.sort(data.country.unique()), ['CA', 'FI', 'IT', 'KE', 'NO', 'SG']))
    h = {c: holidays.country_holidays(a, years=range(2010, 2020)) for c, a in alpha2.items()}
    data['is_holiday'] = 0
    for c in alpha2:
        dates = []
        for date, name in sorted(h[c].items()):
            dates.append(str(date))
        data.loc[(data.country == c) & (data.date.isin(dates)), 'is_holiday'] = 1
    return data


# one-hot向量转换为0-1编码
def one_hot_numpy(data: pd.DataFrame, col_name: string) -> pd.DataFrame:
    # 创建一个OneHotEncoder对象
    encoder = OneHotEncoder(sparse_output=False)  # 设置sparse=False以返回密集数组
    one_hot_encoded_array = encoder.fit_transform(
        data[[col_name]])
    one_hot_encoded_df = pd.DataFrame(one_hot_encoded_array, columns=encoder.get_feature_names_out(
        [col_name]))
    return one_hot_encoded_df


def mape(y_pred, data):
    """
    LightGBM的自定义评估函数格式
    y_pred: 模型的预测值
    data: LightGBM的Dataset对象，包含真实值等信息
    """
    y_true = data.get_label()
    # y_true, y_pred = pd.Series(y_true), pd.Series(y_pred)
    # 计算MAPE
    mape = (np.abs(y_true - y_pred) / y_true).mean() * 100
    return 'MAPE', mape, False


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
    bst = lgb.train(params, data, num_boost_round=100, valid_sets=[valid], feval=mape, )
    # 获取评估指标值
    bst.save_model('model.txt', num_iteration=bst.best_iteration)
    return bst


def predict(data: lgb.Dataset, model_name: string):
    # init model
    bst = lgb.Booster(model_file=model_name)
    # start_iteration 表示取全部训练目标的均值
    y_pred = bst.predict(data)
    return y_pred


def train_by_lightgbm(train_data, train_label, test_data, test_id: pd.DataFrame):
    # 训练
    bst = train(train_data, train_label)
    # 预测
    predictions = predict(test_data, 'model.txt')
    test_id['num_sold'] = predictions
    output = pd.DataFrame(test_id, columns=['id', 'num_sold'])
    output.to_csv('output.csv', index=False)


def sout_nan_data(train_data: pd.DataFrame):
    nan_values = train_data.loc[train_data.num_sold.isna(), :]
    nan_values.to_csv('nan_value.csv', index=False)


# main 主函数
def main():
    train_org_data = read_csv(TRAIN_PATH)
    test_org_data = read_csv(TEST_PATH)
    train_data, train_label, test_data, test_id = deal_data(train_org_data, test_org_data)
    train_by_lightgbm(train_data, train_label, test_data, test_id)
    # sout_nan_data(train_org_data)


if __name__ == "__main__":
    main()
