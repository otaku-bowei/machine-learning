
import pandas as pd
from lightgbm import Booster
from tqdm import tqdm

import lightgbm as lgb


def read_org_data_tmp():
    train_date_0_path = "C:\Program Files\BusinessFile\新手村\AI培训资料\数据集\Jane Street\\train.parquet\partition_id=0\part-0.parquet"
    train_date_1_path = "C:\Program Files\BusinessFile\新手村\AI培训资料\数据集\Jane Street\\train.parquet\partition_id=1\part-0.parquet"
    lags_path = "C:\Program Files\BusinessFile\新手村\AI培训资料\数据集\Jane Street\lags.parquet\date_id=0\part-0.parquet"
    test_path = "C:\Program Files\BusinessFile\新手村\AI培训资料\数据集\Jane Street\\test.parquet\date_id=0\part-0.parquet"
    train_data_0 = pd.read_parquet(train_date_0_path)
    train_data_1 = pd.read_parquet(train_date_1_path)
    lags = pd.read_parquet(lags_path)
    test_data = pd.read_parquet(test_path)
    return train_data_0, train_data_1, test_data, lags


# 初始化模型参数
param = {
    'objective': 'regression',  # 回归问题
    'metric': 'l2',  # 使用L2损失（均方误差）作为评估指标
    'num_leaves': 31,
    'learning_rate': 0.01,
    'feature_fraction': 0.9
}


def init_model() -> Booster:
    train_data_0, train_data_1, test_data, lags = read_org_data_tmp()
    train_d_tmp = train_data_0.loc[0: 38]
    train_d_tmp = train_d_tmp.reset_index(drop=True)
    # 对数据进行区分
    train_data = train_d_tmp.drop(
        columns=['date_id', 'time_id', 'symbol_id', 'responder_0', 'responder_1', 'responder_2', 'responder_3',
                 'responder_4', 'responder_5',
                 'responder_6', 'responder_7', 'responder_8'])
    train_label = train_d_tmp.loc[:, ['responder_6']]
    # 假设我们有一个初始的39*80训练集和一个39*9的目标值向量
    X_initial = train_data.to_numpy()
    y_initial = train_label.to_numpy().ravel()
    # 创建一个新的LightGBM数据集
    train_data_new = lgb.Dataset(X_initial, label=y_initial)
    gbm = lgb.train(param, train_data_new, num_boost_round=10)
    return gbm


def train(gbm : Booster = None):
    train_data_0, train_data_1, test_data, lags = read_org_data_tmp()
    # 初始化变量
    num_iterations = 10  # 迭代次数
    num_samples = 39  # 样本数量
    num_features_original = 80  # 原始特征数量
    train_data_len = len(train_data_0)
    num_features_added = 9  # 每次迭代添加的特征数量（即上一次的预测值）
    lags_np = lags.to_numpy()
    if gbm is None:
        gbm = init_model()
    for batch in tqdm(range(1, int(train_data_len / 39))):
        train_d_tmp = train_data_0.loc[batch * 39: (batch + 1) * 39 - 1]
        train_d_tmp = train_d_tmp.reset_index(drop=True)
        # 对数据进行区分
        train_data = train_d_tmp.drop(
            columns=['date_id', 'time_id', 'symbol_id', 'responder_0', 'responder_1', 'responder_2', 'responder_3',
                     'responder_4', 'responder_5',
                     'responder_6', 'responder_7', 'responder_8'])
        train_label = train_d_tmp.loc[:, ['responder_6']]
        # 假设我们有一个初始的39*80训练集和一个39*9的目标值向量
        X_initial = train_data.to_numpy()
        y_initial = train_label.to_numpy().ravel()
        # 创建一个新的LightGBM数据集
        train_data_new = lgb.Dataset(X_initial, label=y_initial)
        gbm.update(train_set=train_data_new)
    return gbm


def main():
    train()


if __name__ == '__main__':
    main()
