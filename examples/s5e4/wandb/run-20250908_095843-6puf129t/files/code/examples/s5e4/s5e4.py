import pandas as pd
import lightgbm as lgb
import shap
from keras import Sequential
from keras.src.layers import Dense
from sklearn.model_selection import train_test_split
from sympy import false
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import tool.matplotlib.draw_matplotlib as dm

API_KEY = "local-0e045fef2630a4f59fdf629d25206dfc2c6af98e"

# id、播客_名称、剧集_标题、剧集_长度_分钟、类型、主持人_受欢迎程度_百分比、出版物_日期、出版物_时间、访客_受欢迎度_百分比、广告数量_广告、剧集_评论、收听_时间_分钟
# id,Podcast_Name,Episode_Title,Episode_Length_minutes,Genre,Host_Popularity_percentage,Publication_Day,Publication_Time,Guest_Popularity_percentage,Number_of_Ads,Episode_Sentiment,Listening_Time_minutes
# 0,Mystery Matters,Episode 98,,True Crime,74.81,Thursday,Night,,0.0,Positive,31.41998
train_path = "../../playground/playground-series-s5e4/train.csv"
test_path = "../../playground/playground-series-s5e4/test.csv"
str_fields = ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
class_fields = ['Podcast_Name', 'Episode_Title', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment', ]
nan_fields = ['Episode_Length_minutes', 'Guest_Popularity_percentage', ]

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# train_df.dropna(inplace=True)

pd.reset_option('display.max_rows', None)
pd.reset_option('display.max_columns', None)
pd.reset_option('display.max_colwidth', None)
print(train_df.describe().T)

# 查看每个值的枚举数量
print(len(train_df))
for field in str_fields:
    value_counts = train_df[field].value_counts()
    num_unique_values = len(value_counts.index)
    print(f"{field}字段有 {num_unique_values} 个不同的值。")

print("\r\n")
# 统计空值
null_counts = train_df.isnull().sum()
print(null_counts)

import wandb
from datetime import datetime

train_losses = []
valid_losses = []
iterations = []


def log_metrics(env):
    """记录每轮训练的详细参数和结果"""
    if len(env.evaluation_result_list) > 0:
        # 获取验证集RMSE
        valid_rmse = env.evaluation_result_list[0][2]

        # 获取当前模型参数
        current_params = env.params

        # 记录所有关键信息
        wandb.log({
            "iteration": env.iteration,
            "valid_rmse": valid_rmse,
            "learning_rate": current_params.get('learning_rate', 0.09458256238967232),
            "max_depth": current_params.get('max_depth', 10),
            "num_leaves": current_params.get('num_leaves', 256),
            "feature_fraction": current_params.get('feature_fraction', 0.5290646151646433),
            "bagging_fraction": current_params.get('bagging_fraction', 0.8336637608522288),
            "min_data_in_leaf": current_params.get('min_data_in_leaf', 51),
            "best_iteration": env.model.best_iteration if hasattr(env.model, 'best_iteration') else env.iteration
        })

        # 每50轮额外记录特征重要性
        if env.iteration % 50 == 0:
            feature_importance = pd.DataFrame({
                'feature': env.model.feature_name(),
                'importance': env.model.feature_importance(importance_type='gain')
            }).sort_values('importance', ascending=False).head(10)

            wandb.log({
                f"feature_importance_iter_{env.iteration}": wandb.Table(dataframe=feature_importance)
            })


def lgb_train(train_df, test_df: pd.DataFrame):
    # 数据清洗
    train_df = train_df.drop(['id'], axis=1)
    test_id = test_df.loc[:, ['id']]
    test_df = test_df.drop(['id'], axis=1)
    # train_df = train_df.fillna(0.0)
    # for c in nan_fields:
    #     m = train_df[c].mean()
    #     train_df[f'NA_{c}'] = train_df[c].isna().astype('int8')
    #     train_df[c] = train_df[c].fillna(m)
    #     test_df[f'NA_{c}'] = test_df[c].isna().astype('int8')
    #     test_df[c] = test_df[c].fillna(m)
    # 对空值做补充
    # train_df['Episode_Length_minutes'] = train_df['Episode_Length_minutes'].fillna(train_df['Listening_Time_minutes'] * 0.75)

    # 分类字段 明确类型
    for col in str_fields:
        train_df[col] = train_df[col].astype('category')
        test_df[col] = test_df[col].astype('category')

    # 基于Episode_Length_minutes做简单的交互特征
    mi_fields = ['Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']
    for col in mi_fields:
        train_df[f"{col}_Episode_Length_minutes"] = np.where(train_df['Episode_Length_minutes'].notna(),
                                                             train_df['Episode_Length_minutes'], 0.0) * train_df[col]
        test_df[f"{col}_Episode_Length_minutes"] = np.where(test_df['Episode_Length_minutes'].notna(),
                                                            test_df['Episode_Length_minutes'], 0.0) * test_df[col]

    # 根据时间字段分组统计条数,添加一个时间的交互信息
    both_result = train_df.groupby(['Publication_Day', 'Publication_Time'])[
        'Listening_Time_minutes'].mean().reset_index(name='daytime_mi')
    day_result = train_df.groupby(['Publication_Day'])['Listening_Time_minutes'].mean().reset_index(name='day_mi')
    time_result = train_df.groupby(['Publication_Time'])['Listening_Time_minutes'].mean().reset_index(name='time_mi')
    train_df = pd.merge(left=train_df, right=both_result, on=['Publication_Day', 'Publication_Time'], how='left')
    train_df = pd.merge(left=train_df, right=day_result, on=['Publication_Day'], how='left')
    train_df = pd.merge(left=train_df, right=time_result, on=['Publication_Time'], how='left')
    test_df = pd.merge(left=test_df, right=both_result, on=['Publication_Day', 'Publication_Time'], how='left')
    test_df = pd.merge(left=test_df, right=day_result, on=['Publication_Day'], how='left')
    test_df = pd.merge(left=test_df, right=time_result, on=['Publication_Time'], how='left')

    train_df = train_df.drop(['Genre'], axis=1)
    test_df = test_df.drop(['Genre'], axis=1)

    # ... 现有代码 ...
    # train param
    # Besttrial = {'bagging_freq': 8, 'bagging_fraction': 0.8336637608522288, 'feature_fraction': 0.5290646151646433,
    #              'learning_rate': 0.09458256238967232, 'max_depth': 10, 'min_data_in_leaf': 51, 'num_leaves': 255}
    params = {
        'bagging_freq': 8,
        'bagging_fraction': 0.8336637608522288,
        'boost_from_average': 'false',
        'boost': 'gbdt',
        'feature_fraction': 0.5290646151646433,
        'learning_rate': 0.09458256238967232,
        'max_depth': 10,
        # 损失函数
        'metric': 'rmse',
        # 'metric': 'l2',
        'min_data_in_leaf': 51,
        'min_sum_hessian_in_leaf': 10.0,
        'num_leaves': 256,
        'num_threads': 16,
        'tree_learner': 'serial',
        # 回归任务
        'objective': 'regression',
        # 'objective': 'binary',
        'verbosity': -1
    }
    wandb.login(key=API_KEY)
    # run = wandb.init(mode="offline", entity = 's5e4-entity', project="s5e4-project", config={
    #     'rmse': 'rmse',
    #     'num_boost_round': 'num_boost_round'
    # })
    run = wandb.init(
        project="s5e4-project",
        name=f"lgb-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={
            **params,
            "num_boost_round": 500,
            "early_stopping_rounds": 50,
            "train_size": 0.9,
            "random_state": 1,
            "categorical_features": len(class_fields),
            "model_type": "LightGBM",
            "task_type": "regression"
        },
        tags=["lightgbm", "regression", "s5e4", "podcast-listening-time"],
        save_code=True,
        notes="LightGBM model for podcast listening time prediction with comprehensive monitoring"
    )

    # ... 现有代码 ...
    #
    print(train_df.head())
    train_data = train_df.drop(['Listening_Time_minutes'], axis=1)
    train_label = train_df['Listening_Time_minutes']
    # 1. 进行one-hot转换
    # train_df = pd.get_dummies(train_df, columns=['Podcast_Name', 'Episode_Title'], )
    # l = len(train_df)
    # df = pd.concat([train_data, test_df], axis=0)
    # df = pd.get_dummies(df, columns=class_fields)
    # train_data = df.iloc[:l, ]
    # test_df = df.iloc[l:, ]

    train_d, valid_d, train_l, valid_l = train_test_split(train_data, train_label, random_state=1, train_size=0.9)
    # data = lgb.Dataset(train_d, label=train_l, )
    # valid = lgb.Dataset(valid_d, label=valid_l, )
    # bst = lgb.train(params, data, num_boost_round=500, valid_sets=[valid], )
    data = lgb.Dataset(train_d, label=train_l, categorical_feature=class_fields)
    valid = lgb.Dataset(valid_d, label=valid_l, categorical_feature=class_fields)
    bst = lgb.train(
        params,
        data,
        num_boost_round=500,
        valid_sets=[valid],
        valid_names=['valid'],
        callbacks=[
            lgb.early_stopping(50, verbose=True),
            lgb.log_evaluation(50),
            log_metrics
        ]
    )

    # data = lgb.Dataset(train_d, label=train_l, )
    # valid = lgb.Dataset(valid_d, label=valid_l, )
    # bst = lgb.train(params, data, num_boost_round=500, valid_sets=[valid], )
    pre = bst.predict(test_df)
    test_id['Listening_Time_minutes'] = pre
    test_id.to_csv('result.csv', index=False)

    print(f"cv成绩：{bst.best_score['valid_0']['rmse']}")
    run.finish()
    return bst


def nn_train(train_df, test_df: pd.DataFrame):
    print("开始nn训练")
    # 数据预处理
    # 分离特征和标签
    train_data = train_df.drop(['id', 'Listening_Time_minutes'], axis=1)
    train_label = train_df['Listening_Time_minutes']

    test_id = test_df['id']
    # 对分类特征进行独热编码
    categorical_columns = ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day', 'Publication_Time',
                           'Episode_Sentiment']
    # 训练集和测试集统一处理
    l = len(train_data)
    data = pd.concat([train_data, test_df], axis=1)
    data = pd.get_dummies(data, columns=categorical_columns)
    data = data.fillna(0.0)
    train_data = data.loc[:l, :]
    test_data = data.loc[l:, :]

    null_counts = train_data.isnull().sum()
    print(null_counts)

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(train_data, train_label, random_state=1, train_size=0.9)
    # 构建神经网络模型
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        # 丢弃部分神经元梯度下降更稳定，防止梯度消失--如果没有正确使用dropout（例如，没有关闭dropout），全连接层神经元会全部处于激活状态，导致预测结果相同‌
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])

    # 编译模型
    model.compile(optimizer='adam', loss='rmse', metrics=['mae'])
    print(X_train)

    # 训练模型
    history = model.fit(tf.convert_to_tensor(X_train), tf.convert_to_tensor(y_train), epochs=50, batch_size=32,
                        validation_data=(X_val, y_val))

    # 进行预测
    predictions = model.predict(tf.convert_to_tensor(test_data), )

    # 保存结果
    test_id['Listening_Time_minutes'] = predictions
    test_id.to_csv('nn_result.csv', index=False)

    print("神经网络训练完成，结果已保存至 nn_result.csv")
    return model


bst = lgb_train(train_df.copy(), test_df.copy())

# nn_model = nn_train(train_df.copy(), test_df.copy())
# ... 现有代码 ...
# 创建可以计算 SHAP 值的对象
# explainer = shap.TreeExplainer(bst)
# # 计算验证集的 SHAP 值
# shap_values = explainer.shap_values(valid_d)
# # 绘制全局特征重要性图
# shap.summary_plot(shap_values, valid_d)
