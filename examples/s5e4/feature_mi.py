import pandas as pd
import lightgbm as lgb
import shap
from keras import Sequential
from keras.src.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sympy import false
import numpy as np
import tensorflow as tf
import optuna
from sklearn.model_selection import train_test_split

import tool.matplotlib.draw_matplotlib as dm

# id、播客_名称、剧集_标题、剧集_长度_分钟、类型、主持人_受欢迎程度_百分比、出版物_日期、出版物_时间、访客_受欢迎度_百分比、广告数量_广告、剧集_评论、收听_时间_分钟
# id,Podcast_Name,Episode_Title,Episode_Length_minutes,Genre,Host_Popularity_percentage,Publication_Day,Publication_Time,Guest_Popularity_percentage,Number_of_Ads,Episode_Sentiment,Listening_Time_minutes
# 0,Mystery Matters,Episode 98,,True Crime,74.81,Thursday,Night,,0.0,Positive,31.41998
train_path = "./playground-series-s5e4/train.csv"
test_path = "./playground-series-s5e4/test.csv"
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

def engineer_features(X_train, X_test):
    combined = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)

    # 1. Ad Density
    combined['ads_per_minute'] = combined['Number_of_Ads'] / (combined['Episode_Length_minutes'] + 1e-3)

    # 2. Is Weekend
    combined['is_weekend'] = combined['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)

    # 3. Time of Day Features
    combined['is_morning'] = (combined['Publication_Time'] == 'Morning').astype(int)
    combined['is_night'] = (combined['Publication_Time'] == 'Night').astype(int)

    # 4. Episode Length Buckets
    combined['length_bucket'] = pd.cut(combined['Episode_Length_minutes'], bins=[0, 30, 60, 90, 200],
                                       labels=['short', 'medium', 'long', 'very_long'])

    # 5. Sentiment Ordinal Mapping
    sentiment_map = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
    combined['sentiment_score'] = combined['Episode_Sentiment'].map(sentiment_map)

    # 6. Host-Guest Popularity Ratio
    combined['popularity_ratio'] = combined['Guest_Popularity_percentage'] / (
        combined['Host_Popularity_percentage'] + 1e-3)

    # 7. Episode Number from Title
    combined['episode_number'] = combined['Episode_Title'].str.extract(r'(\d+)').astype(float)

    # 8. Genre + Sentiment Interaction
    combined['genre_sentiment'] = combined['Genre'].astype(str) + "_" + combined['Episode_Sentiment'].astype(str)

    # --- Handle Missing Values ---
    # Fill numeric columns using Genre-wise mean
    for col in ['Episode_Length_minutes', 'Guest_Popularity_percentage']:
        combined[col] = combined.groupby('Genre')[col].transform(lambda x: x.fillna(x.mean()))

    # --- Encode Categorical Features ---
    categorical_cols = ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day',
                        'Publication_Time', 'Episode_Sentiment', 'length_bucket', 'genre_sentiment']

    for col in categorical_cols:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col].astype(str))

    # Split back to train and test
    X_train_fe = combined.iloc[:len(X_train)].reset_index(drop=True)
    X_test_fe = combined.iloc[len(X_train):].reset_index(drop=True)

    return X_train_fe, X_test_fe


def n_fe(df):

    df['Is_Weekend'] = df['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
    df['Is_High_Host_Popularity'] = (df['Host_Popularity_percentage'] > 70).astype(int)
    df['Is_High_Guest_Popularity'] = (df['Guest_Popularity_percentage'] > 70).astype(int)
    df['Host_Guest_Popularity_Gap'] = df['Host_Popularity_percentage'] - df['Guest_Popularity_percentage']
    df['Ad_Density'] = df['Number_of_Ads'] / df['Episode_Length_minutes']
    df['Ad_Density'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['Is_Long_Episode'] = (df['Episode_Length_minutes'] > 60).astype(int)

    return df


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

    train_df, test_df = engineer_features(train_df, test_df)
    test_df = test_df.drop(['Listening_Time_minutes'], axis=1)

    # train_df = n_fe(train_df)
    # test_df = n_fe(test_df)
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
    # Besttrial: {'bagging_freq': 2, 'bagging_fraction': 0.9793095720949818, 'feature_fraction': 0.8342793414263366,
    #         'learning_rate': 0.046753011771420085, 'max_depth': 10, 'min_data_in_leaf': 17, 'num_leaves': 256}
    params = {
        'bagging_freq': 2,
        'bagging_fraction': 0.9793095720949818,
        'boost_from_average': 'false',
        'boost': 'gbdt',
        'feature_fraction': 0.8342793414263366,
        'learning_rate': 0.046753011771420085,
        'max_depth': 10,
        # 损失函数
        'metric': 'rmse',
        # 'metric': 'l2',
        'min_data_in_leaf': 17,
        'min_sum_hessian_in_leaf': 10.0,
        'num_leaves': 256,
        'num_threads': 16,
        'tree_learner': 'serial',
        # 回归任务
        'objective': 'regression',
        # 'objective': 'binary',
        'verbosity': -1
    }

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
    bst = lgb.train(params, data, num_boost_round=3200, valid_sets=[valid], categorical_feature=class_fields, )

    # data = lgb.Dataset(train_d, label=train_l, )
    # valid = lgb.Dataset(valid_d, label=valid_l, )
    # bst = lgb.train(params, data, num_boost_round=500, valid_sets=[valid], )
    pre = bst.predict(test_df)
    test_id['Listening_Time_minutes'] = pre
    test_id.to_csv('result.csv', index=False)

    print(f"cv成绩：{bst.best_score['valid_0']['rmse']}")
    return bst


bst = lgb_train(train_df.copy(), test_df.copy())
# 标准分数
restful_score = bst.best_score['valid_0']['rmse']
# for循环添加mi特征进行对比

# nn_model = nn_train(train_df.copy(), test_df.copy())
# ... 现有代码 ...
# 创建可以计算 SHAP 值的对象
# explainer = shap.TreeExplainer(bst)
# # 计算验证集的 SHAP 值
# shap_values = explainer.shap_values(valid_d)
# # 绘制全局特征重要性图
# shap.summary_plot(shap_values, valid_d)
