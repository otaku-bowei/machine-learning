import numpy as np
import pandas as pd
import gc, os

import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from multiprocessing import cpu_count
from tqdm import tqdm  # 进度条库


# test 答案测试
def test():
    # 1.从测试集删除fake数据——避免测试集对模型的判断干扰
    # ===== fake samples =====
    te_ = pd.read_csv('./test.csv').drop(['ID_code'], axis=1).values

    unique_samples = []
    unique_count = np.zeros_like(te_)
    for feature in tqdm(range(te_.shape[1])):
        _, index_, count_ = np.unique(te_[:, feature], return_counts=True, return_index=True)
        unique_count[index_[count_ == 1], feature] += 1

    # Samples which have unique values are real the others are fake
    real_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]
    synthetic_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) == 0)[:, 0]

    # =============================================================================
    # setting
    # =============================================================================

    # parameters

    params = {
        'bagging_freq': 5,
        'bagging_fraction': 1.0,
        'boost_from_average': 'false',
        'boost': 'gbdt',
        'feature_fraction': 1.0,
        'learning_rate': 0.005,
        'max_depth': -1,
        'metric': 'binary_logloss',
        'min_data_in_leaf': 30,
        'min_sum_hessian_in_leaf': 10.0,
        'num_leaves': 64,
        'num_threads': cpu_count(),
        'tree_learner': 'serial',
        'objective': 'binary',
        'verbosity': -1
    }

    NFOLD = 10

    NROUND = 1600

    SEED = np.random.randint(99999)
    np.random.seed(SEED)

    SUBMIT_FILE_PATH = f'../output/2nd-place-solution.csv.gz'

    # =============================================================================
    # drop vars
    # =============================================================================

    drop_vars = [7,
                 10,
                 17,
                 27,
                 29,
                 30,
                 38,
                 41,
                 46,
                 96,
                 100,
                 103,
                 126,
                 158,
                 185]

    var_len = 200 - len(drop_vars)
    print("read data success\r\n")
    # 2.concat测试集和训练集
    # =============================================================================
    # load
    # =============================================================================
    train = pd.read_csv("./train.csv.zip")
    test = pd.read_csv("./test.csv.zip").drop(synthetic_samples_indexes)

    X_train = train.iloc[:, 2:].values
    y_train = train.target.values

    X_test = test.iloc[:, 1:].values

    X = np.concatenate([X_train, X_test], axis=0)
    del X_train, X_test;
    gc.collect()

    reverse_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 15, 16, 18, 19, 22, 24, 25, 26,
                    27, 29, 32, 35, 37, 40, 41, 47, 48, 49, 51, 52, 53, 55, 60, 61,
                    62, 65, 66, 67, 69, 70, 71, 74, 78, 79, 82, 84, 89, 90, 91, 94,
                    95, 96, 97, 99, 103, 105, 106, 110, 111, 112, 118, 119, 125, 128,
                    130, 133, 134, 135, 137, 138, 140, 144, 145, 147, 151, 155, 157,
                    159, 161, 162, 163, 164, 167, 168, 170, 171, 173, 175, 176, 179,
                    180, 181, 184, 185, 187, 189, 190, 191, 195, 196, 199,

                    ]
    # 随机列 取反——增强数据
    for j in reverse_list:
        X[:, j] *= -1
    # 随机列 删除——增强数据
    # drop
    X = np.delete(X, drop_vars, 1)
    print("enhanced data success\r\n")
    # 3.数据缩放
    # scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print("scaler data success\r\n")
    # 4.计数编码——用类别出现的次数代表该类别
    # count encoding
    X_cnt = np.zeros((len(X), var_len * 4))
    # 5.将不同精度的数据进行复制，然后进行 技术编码，本质上是将数据放到一个 范围 区间，然后 进行统计归类
    for j in tqdm(range(var_len)):
        for i in range(1, 4):
            x = np.round(X[:, j], i + 1)
            dic = pd.value_counts(x).to_dict()
            X_cnt[:, i + j * 4] = pd.Series(x).map(dic)
        x = X[:, j]
        dic = pd.value_counts(x).to_dict()
        X_cnt[:, j * 4] = pd.Series(x).map(dic)
    print("encode data success\r\n")
    # raw + count feature
    X_raw = X.copy()  # rename for readable
    del X;
    gc.collect()
    # 将缩放后的训练集与技术编码后的数据concat
    X = np.zeros((len(X_raw), var_len * 5))
    for j in tqdm(range(var_len)):
        X[:, 5 * j + 1:5 * j + 5] = X_cnt[:, 4 * j:4 * j + 4]
        X[:, 5 * j] = X_raw[:, j]

    # 6.取消透视所有列
    # 此前，X为一个[400k , var_len * 5] 的数据集，将X按5位步长截取为 [200k, 5] * var_len 份数据集，其中每份都映射 训练集 的 label
    # treat each var as same
    X_train_concat = np.concatenate([
        np.concatenate([
            X[:200000, 5 * cnum:5 * cnum + 5],
            np.ones((len(y_train), 1)).astype("int") * cnum
        ], axis=1) for cnum in range(var_len)], axis=0)
    y_train_concat = np.concatenate([y_train for cnum in range(var_len)], axis=0)

    # 输出透视后的结果
    print(X_train_concat[0:5, :])
    print(y_train_concat[0:5, :])
