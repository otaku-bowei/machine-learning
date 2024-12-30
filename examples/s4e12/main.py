'''
id,Age,Gender,Annual Income,Marital Status,Number of Dependents,Education Level,Occupation,Health Score,Location,Policy Type,Previous Claims,Vehicle Age,Credit Score,Insurance Duration,Policy Start Date,Customer Feedback,Smoking Status,Exercise Frequency,Property Type,Premium Amount
身份证，年龄，性别，年收入，婚姻状况，受抚养人数，教育水平，职业，健康评分，地点，保单类型，以前的索赔，车龄，信用评分，保险期限，保单开始日期，客户反馈，吸烟状况，运动频率，财产类型，保费金额
'''
from enum import Enum

import sklearn.metrics
from hillclimbers import climb_hill, partial
from sklearn.compose import ColumnTransformer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline

'''
1.nn训练后发现loss超高，输出值都一样--https://www.zhihu.com/question/390693207/answer/3197389090--（1）梯度爆炸（2）死亡relu（3）学习率过大（4）过拟合
    --填充了nan值，应该是relu死亡，nan值导致数据一直为负数
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
import tool.matplotlib.field_analysis as fa
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
import tool.tensorboard.tensor_board as tb
from tqdm import tqdm
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor


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
IMPORTAMT_FIELD = ['Health Score', 'Annual Income', 'Age']
ONE_HOT_FIELD = ['Gender', 'Marital Status', 'Education Level', 'Occupation', 'Policy Type', 'Property Type']
NP_FIELD = ['Age', 'Annual Income', 'Number of Dependents', 'Previous Claims', 'Health Score', 'Premium Amount',
            'Policy Start Date', 'nan_count']
TEST_NP_FIELD = ['Age', 'Annual Income', 'Number of Dependents', 'Previous Claims', 'Health Score', 'id',
                 'Policy Start Date', 'nan_count']
CATEGORICAL_FEATURE = ['Gender', 'Marital Status', 'Education Level', 'Occupation', 'Location', 'Policy Type',
                       'Customer Feedback', 'Smoking Status', 'Exercise Frequency', 'Property Type']
FILL_NAN_FIELD = ['Age', 'Annual Income', 'Number of Dependents', 'Health Score']
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


def rmsle(y_true, y_pred):
    """
    计算RMSLE（Root Mean Squared Logarithmic Error）
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


def rmsle_eval(y_pred, data):
    """
    LightGBM的自定义评估函数格式
    y_pred: 模型的预测值
    data: LightGBM的Dataset对象，包含真实值等信息
    """
    y_true = data.get_label()
    return 'RMSLE', rmsle(y_true, y_pred), False


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
    # data = lgb.Dataset(train_data, label=train_label)
    valid = lgb.Dataset(valid_d, label=valid_l)
    # 添加交叉验证，评分直接 2.81342提升到1.09089
    bst = lgb.train(params, data, num_boost_round=100, valid_sets=[valid], feval=rmsle_eval, )
    # 获取评估指标值
    bst.save_model('model.txt', num_iteration=bst.best_iteration)
    # loss计算
    loss_cul(valid_d, valid_l, 'model.txt')
    return bst


class CulType(Enum):
    DIVISE = '/',
    PLUS = '+'


# interaction_features 交互特征
def interaction_features(data: pd.DataFrame, field1, field2: str, type: CulType):
    name = field1 + '-' + field2
    if type == CulType.PLUS:
        data[name] = data[field1] + data[field2]
    elif type == CulType.DIVISE:
        data[name] = data[field1] / data[field2]
    return data




def train_with_categorical_feature(train_data, train_label: pd.DataFrame):
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
    bst.best_score()
    return bst


def one_hot_fix(data, org_data: pd.DataFrame, clo_names: []) -> pd.DataFrame:
    for field_name in clo_names:
        type_one_hot = one_hot_numpy(org_data, field_name)
        data = pd.concat([data, type_one_hot], axis=1)
    return data


def predict(data: lgb.Dataset, model_name: string):
    # init model
    bst = lgb.Booster(model_file=model_name)
    # start_iteration 表示取全部训练目标的均值
    y_pred = bst.predict(data)
    return y_pred


# loss_cul 计算回归的loss值
def loss_cul(data, label: pd.DataFrame, model_name: string):
    # d = lgb.Dataset(data=data, label=label)
    y_pred = predict(data, model_name)
    l_np = label.to_numpy()
    loss = sklearn.metrics.root_mean_squared_log_error(l_np, y_pred)
    print(loss)


def log_data(data: pd.DataFrame, field_name: str):
    data[field_name] = np.log(data[field_name])
    return data


# precision_fix 精度扩展，对某些字段进行精度扩展字段--对决策树处理回归问题有帮助
def precision_fix():
    print()


# nan_to_mean 某些列的 nan 是有意义的，将其标识为平均值，TODO--并用标识列标注该值
def nan_to_mean(data: pd.DataFrame, field: string, num: float = None) -> pd.DataFrame:
    if num is None:
        data.loc[:, field] = data[field].fillna(data[field].mean())
    else:
        data.loc[:, field] = data[field].fillna(num)
    data[field + '_is_nan'] = data[field].isna().astype(int)
    col_names = data.columns
    return data


# date_fix 处理日期字段
def date_fix(data: pd.DataFrame, field: str) -> pd.DataFrame:
    df = pd.DataFrame(data.loc[:, [field]])
    df['date'] = pd.to_datetime(df[field])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    df['day_of_week'] = df['date'].dt.day_of_week
    # df['month_name'] = df['date'].dt.month_name
    data = pd.concat([data, df.drop([field, 'date'], axis=1)], axis=1)
    return data.drop([field], axis=1)


# 自定义对数激活函数
def log_activation(x):
    return tf.math.log1p(x)


def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


def train_by_nn(model: tf.keras.models.Sequential = None):
    # 1.预处理数据
    train_org_data = pd.read_csv(TRAIN_PATH)
    test_org_data = pd.read_csv(TEST_PATH)
    train_org_data['nan_count'] = train_org_data.isna().sum(axis=1)
    test_org_data['nan_count'] = test_org_data.isna().sum(axis=1)
    train_data = train_org_data.loc[:, NP_FIELD]
    test_data = test_org_data.loc[:, TEST_NP_FIELD]
    # 对'Marital Status', 'Education Level', 'Occupation', 'Policy Type', 'Property Type'字段进行独热编码
    # TODO--lightgbm原生支持分类编码而无需进行one-hot编码——categorical_feature参数
    train_data = one_hot_fix(train_data, train_org_data, CATEGORICAL_FEATURE)
    # train_data = date_fix(train_data, 'Policy Start Date')
    train_data = train_data.drop('Policy Start Date', axis=1)
    test_data = one_hot_fix(test_data, test_org_data, CATEGORICAL_FEATURE)
    # test_data = date_fix(test_data, 'Policy Start Date')
    test_data = test_data.drop('Policy Start Date', axis=1)
    # 乱序
    train_data = pd.DataFrame(train_data)
    # 提取label
    train_label = train_data.loc[:, ['Premium Amount']]
    train_data = train_data.drop(['Premium Amount'], axis=1)
    test_id = test_data.loc[:, ['id']]
    test_data = test_data.drop(['id'], axis=1)
    # 标准归一化--和测试集一起做归一化
    length = len(train_data)
    # 特征工程，处理nan值-- 1.08912提升到1.08899，说明某些字段的nan值是对预测结果有影响的
    all_data = pd.concat([train_data, test_data])
    for col in FILL_NAN_FIELD:
        all_data = nan_to_mean(all_data, col, -1)

    # 'Annual Income', 'Number of Dependents', 'Previous Claims', 'Health Score', 'Premium Amount'
    all_data = interaction_features(all_data, 'Annual Income', 'Age', CulType.DIVISE)
    all_data = interaction_features(all_data, 'Health Score', 'Age', CulType.DIVISE)
    all_data = interaction_features(all_data, 'Annual Income', 'Previous Claims', CulType.PLUS)
    cols_names = all_data.columns
    scaler = StandardScaler()
    all_data = scaler.fit_transform(all_data)
    # 重新分为训练集和测试集
    train_data = pd.DataFrame(all_data, columns=cols_names).iloc[0:length, :]
    test_data = pd.DataFrame(all_data, columns=cols_names).iloc[length:, :]

    # 区分训练集和验证集
    length = len(train_data)
    index = int(length * 0.8)
    train_d = train_data.loc[:index, :]
    valid_d = train_data.loc[index:, :]
    train_l = train_label.loc[:index, :]
    valid_l = train_label.loc[index:, :]

    if model is None:
        model = tf.keras.models.Sequential([
            # tf.keras.layers.Flatten(input_shape=(len(cols_names),)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            # 丢弃部分神经元梯度下降更稳定，防止梯度消失--如果没有正确使用dropout（例如，没有关闭dropout），全连接层神经元会全部处于激活状态，导致预测结果相同‌
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1)
        ])
    # loss_fn = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer='adam', loss=RMSLE(), metrics=['mse'])
    # 4.2定义tensorBoard
    tensorboard_callback = tb.draw_board('s4e12')
    # 5.进行训练
    model.fit(train_data, train_label, epochs=5, callbacks=[tensorboard_callback])
    model.save('nn_model.keras')
    # 6.loss计算--抽取0.2的训练集作为验证
    y_pred = model.predict(valid_d)
    loss = sklearn.metrics.root_mean_squared_log_error(valid_l, y_pred)
    print(loss)
    return test_data, test_id


# train_by_nn_symbol_field 根据gbdt训练结果，挑选关键的几个字段进行nn训练
def train_by_nn_symbol_field(model: tf.keras.models.Sequential = None):
    # 1.预处理数据
    train_org_data = pd.read_csv(TRAIN_PATH)
    test_org_data = pd.read_csv(TEST_PATH)
    train_fields = ['Health Score', 'Annual Income', 'Age', 'Premium Amount']
    test_fields = ['Health Score', 'Annual Income', 'Age', 'id']
    train_data = train_org_data.loc[:, train_fields]
    test_data = test_org_data.loc[:, test_fields]
    # 乱序
    train_data = pd.DataFrame(train_data)
    # 提取label
    train_label = train_data.loc[:, ['Premium Amount']]
    train_data = train_data.drop(['Premium Amount'], axis=1)
    test_id = test_data.loc[:, ['id']]
    test_data = test_data.drop(['id'], axis=1)
    # 标准归一化--和测试集一起做归一化
    length = len(train_data)
    all_data = pd.concat([train_data, test_data])
    for col in IMPORTAMT_FIELD:
        all_data = nan_to_mean(all_data, col)
    cols_names = all_data.columns
    scaler = StandardScaler()
    all_data = scaler.fit_transform(all_data)
    # 重新分为训练集和测试集
    train_data = pd.DataFrame(all_data, columns=cols_names).iloc[0:length, :]
    test_data = pd.DataFrame(all_data, columns=cols_names).iloc[length:, :]
    # 区分训练集和验证集
    length = len(train_data)
    index = int(length * 0.8)
    train_d = train_data.loc[:index, :]
    valid_d = train_data.loc[index:, :]
    train_l = train_label.loc[:index, :]
    valid_l = train_label.loc[index:, :]

    if model is None:
        model = tf.keras.models.Sequential([
            # tf.keras.layers.Flatten(input_shape=(len(cols_names),)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            # 丢弃部分神经元梯度下降更稳定，防止梯度消失--如果没有正确使用dropout（例如，没有关闭dropout），全连接层神经元会全部处于激活状态，导致预测结果相同‌
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1)
        ])
    # loss_fn = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer='adam', loss=RMSLE(), metrics=['mse'])
    # 4.2定义tensorBoard
    tensorboard_callback = tb.draw_board('s4e12')
    # 5.进行训练
    model.fit(train_d, train_l, epochs=2, callbacks=[tensorboard_callback])
    model.save('nn2_model.keras')
    # 6.loss计算--抽取0.2的训练集作为验证
    y_pred = model.predict(valid_d)
    loss = sklearn.metrics.root_mean_squared_log_error(valid_l, y_pred)
    print(loss)
    return test_data, test_id


def nn_pred(test_data, test_id: pd.DataFrame):
    model = tf.keras.models.load_model('nn_model.keras')
    y_test = model.predict(test_data)
    test_id['Premium Amount'] = y_test
    output = pd.DataFrame(test_id, columns=['id', 'Premium Amount'])
    output.to_csv('output.csv', index=False)


@tf.keras.saving.register_keras_serializable()
class RMSLE(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        """
        计算RMSLE损失
        :param y_true: 实际值，Tensor类型
        :param y_pred: 预测值，Tensor类型
        :return: RMSLE损失值，Tensor类型
        """
        # 确保y_true和y_pred的dtype一致，并且避免对数运算时的数值问题
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

        # 计算对数误差
        log_y_true = tf.math.log1p(y_true)
        log_y_pred = tf.math.log1p(y_pred)
        log_error = log_y_pred - log_y_true

        # 计算平方误差并求平均
        mse = tf.reduce_mean(tf.square(log_error))

        # 返回RMSLE
        return tf.sqrt(mse)


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
    train_org_data['nan_count'] = train_org_data.isna().sum(axis=1)
    test_org_data['nan_count'] = test_org_data.isna().sum(axis=1)
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
    # train_label = np.log1p(train_label)
    train_data = train_data.drop(['Premium Amount'], axis=1)
    test_id = test_data.loc[:, ['id']]
    test_data = test_data.drop(['id'], axis=1)
    # 标准归一化--和测试集一起做归一化--FIXME--去除归一化后由1.08992提升到1.08988
    length = len(train_data)
    # 特征工程，处理nan值-- 1.08912提升到1.08889，说明某些字段的nan值是对预测结果有影响的
    all_data = pd.concat([train_data, test_data])
    for col in FILL_NAN_FIELD:
        all_data = nan_to_mean(all_data, col, -1)
    # log_data(all_data, 'Annual Income')
    cols_names = all_data.columns
    # scaler = StandardScaler()
    # all_data = scaler.fit_transform(all_data)
    # 重新分为训练集和测试集
    train_data = pd.DataFrame(all_data, columns=cols_names).iloc[0:length, :]
    test_data = pd.DataFrame(all_data, columns=cols_names).iloc[length:, :]
    # print_dp(train_data.head())
    # print_dp(test_data.head())
    # 训练
    bst = train(train_data, train_label)
    # 预测
    predictions = predict(test_data, 'model.txt')
    test_id['Premium Amount'] = predictions
    output = pd.DataFrame(test_id, columns=['id', 'Premium Amount'])
    output.to_csv('output.csv', index=False)
    # loss--1184999.9169039582


def analysis_data(data: pd.DataFrame):
    # fa.nan_compare(data, True, 'Occupation', 'Premium Amount')
    # fa.nan_compare(data, False, 'Age', 'Premium Amount')
    # fa.nan_compare(data, False, 'Health Score', 'Premium Amount')
    fa.nan_compare(data, False, 'Annual Income', 'Premium Amount')
    # fa.nan_compare(data, False, 'Number of Dependents', 'Premium Amount')
    # fa.nan_compare(data, True, 'Customer Feedback', 'Premium Amount')


# deal_nan_data 通过分析后将某列的值赋为合理值
def deal_nan_data(data: pd.DataFrame):
    # NP_FIELD = ['Age', 'Annual Income', 'Number of Dependents', 'Health Score', 'Premium Amount', 'Policy Start Date']
    data['Annual Income'] = data['Annual Income'].fillna(74998.5)
    data['Health Score'] = data['Health Score'].fillna(49.5)


def shikamaru_answer():
    light_model = LGBMRegressor(random_state=42, verbosity=-1, device='gpu')
    cat_model = CatBoostRegressor(random_state=42, logging_level='Silent', task_type='GPU')
    xgb_model = XGBRegressor(random_state=42, tree_method='gpu_hist')
    hist_model = HistGradientBoostingRegressor(random_state=42)
    # 1.读取原始数据
    train_org_data = pd.read_csv(TRAIN_PATH)
    test_org_data = pd.read_csv(TEST_PATH)
    train_data = date_fix(train_org_data, 'Policy Start Date')
    test_data = date_fix(test_org_data, 'Policy Start Date')
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    # handling object columns
    cat_pipeline = make_pipeline(oe)
    col_trans = ColumnTransformer(
        [
            ('cat', cat_pipeline, train_data.drop(columns='Premium Amount').select_dtypes("object").columns),
        ],
        remainder='passthrough'
    )
    X = pd.DataFrame(col_trans.fit_transform(train_data.drop(columns=['Premium Amount'])),
                     columns=col_trans.get_feature_names_out())
    y = train_data['Premium Amount']
    test_p = pd.DataFrame(col_trans.transform(test_data), columns=X.columns)

    # 训练多个model
    models = {'light_model': light_model, 'cat_model': cat_model,
              'xgb_model': xgb_model, 'hist_model': hist_model}
    # Cross Validation and OOF on train
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    # Log-transforming our target so we can use the RMSE metric on it to yield the RMSLE eval metric
    y2 = np.log1p(y)
    # A dataframe to store our OOF predictions from the train
    train_dfs = train_data[['id']]
    train_dfs.set_index('id', inplace=True)
    # Iterating our models, showing CV score per folds
    for model_name, model in models.items():
        print(f"Working on {model_name}")
        values = []
        for i, (t_idx, v_idx) in enumerate(cv.split(X, y2)):
            X_train, X_val = X.iloc[t_idx,], X.iloc[v_idx,]
            y_train, y_val = y2.iloc[t_idx,], y2.iloc[v_idx,]
            model.fit(X_train, y_train)
            yhat = model.predict(X_val)
            rmse_error = sklearn.metrics.root_mean_squared_log_error(y_val, yhat)
            print(f"Fold {i + 1}", rmse_error)
            train_dfs.loc[v_idx, model_name] = yhat
            values.append(rmse_error)
            i += 1
        print(f"{model_name} eval_metric summary: {np.mean(values)} ± {round(np.std(values), 4)}")
        print("\n")

    test_dfs = test_org_data[['id']]
    test_dfs.set_index('id', inplace=True)

    for model_name, model in models.items():
        print(f"Working on {model_name}")
        model.fit(X, y2)
        predictions = model.predict(test_p)
        test_dfs[model_name] = predictions

    train_df = pd.concat([X, np.log1p(y)], axis=1)
    # predictions with hillclimbing
    test_predictions = climb_hill(train=train_df,
                                  oof_pred_df=train_dfs,
                                  test_pred_df=test_dfs,
                                  target='Premium Amount',
                                  eval_metric=partial(root_mean_squared_error),
                                  objective='minimize')
    print(test_predictions)


# main 主函数
def main():
    # test_data, test_id = train_by_nn(tf.keras.models.load_model('nn_model.keras'))
    test_data, test_id = train_by_nn()
    # test_data, test_id = train_by_nn_symbol_field()
    # train_by_lightgbm()
    # train_org_data = pd.read_csv(TRAIN_PATH)
    # analysis_data(train_org_data)
    nn_pred(test_data, test_id)
    # shikamaru_answer()


if __name__ == "__main__":
    main()
