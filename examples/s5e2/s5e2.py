import pandas as pd
from sklearn.model_selection import train_test_split


'''
id,Brand,Material,Size,Compartments,Laptop Compartment,Waterproof,Style,Color,Weight Capacity (kg),Price
标识，品牌，材料，尺寸，隔间，笔记本电脑隔间，防水，样式，颜色，重量容量 (公斤) ，价格
'''

# 1.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# train_org_data = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
# test_org_data = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
train_org_data = pd.read_csv('training_extra.csv')
test_org_data = pd.read_csv('test.csv')
train_data, valid_data, train_label, valid_label = train_test_split(train_org_data.drop(['id', 'Price'], axis=1), train_org_data['Price'], test_size=0.2)
test_data, test_id = test_org_data.drop(['id'], axis=1), test_org_data['id']

import seaborn as sns

def ana_train_data():
    # sns.lineplot(train_data, y='Price', x='Brand', hue='Laptop Compartment')
    sns.swarmplot(x=train_org_data['Brand'], y=train_org_data['Price'])


# 2.
# print(train_org_data.head())
# print(train_org_data.describe())
# print(train_org_data['Brand'].value_counts())
# print(train_org_data['Material'].value_counts())
# print(train_org_data['Size'].value_counts())
# print(train_org_data['Style'].value_counts())
# print(train_org_data['Color'].value_counts())

# 3.
BOOLEAN_FIELDS = ['Laptop Compartment', 'Waterproof', ]
def exchange_yes_or_no(data: pd.DataFrame, field_name: str):
    data[field_name + '_ex'] = data[field_name].map({'Yes': 1, 'No': 0})
    return data.drop([field_name], axis=1)

ONE_HOT_FIELDS = ['Brand', 'Material', 'Size', 'Style', 'Color', ]

def exchange_one_host(data: pd.DataFrame):
    type_one_hot = pd.get_dummies(data, columns=ONE_HOT_FIELDS, dummy_na=True)
    return type_one_hot

# 4.
# 对数据进行基础的整理
all_feature_data = pd.concat([train_data, valid_data, test_data], axis=0)
# 1.个别字段Yes/No转换为1/0
for field in BOOLEAN_FIELDS:
    train_data = exchange_yes_or_no(train_data, field)
    valid_data = exchange_yes_or_no(valid_data, field)
    test_data = exchange_yes_or_no(test_data, field)



# 5.
# 对某些字段进行one-hot处理
train_data = exchange_one_host(train_data)
valid_data = exchange_one_host(valid_data)
test_data = exchange_one_host(test_data)

# 6.
# 训练一个模型
import lightgbm as lgb

# params = {
#     'bagging_freq': 5,
#     'bagging_fraction': 1.0,
#     'boost_from_average': 'false',
#     'boost': 'gbdt',
#     'feature_fraction': 1.0,
#     'learning_rate': 0.005,
#     'max_depth': -1,
#     # 均方差损失标准
#     'metric': 'rmse',
#     # 'metric': 'l2',
#     'min_data_in_leaf': 30,
#     'min_sum_hessian_in_leaf': 10.0,
#     'num_leaves': 64,
#     'num_threads': 8,
#     'tree_learner': 'serial',
#     # 回归任务
#     'objective': 'regression',
#     # 'objective': 'binary',
#     'verbosity': -1
# }
# data = lgb.Dataset(train_data, label=train_label, )
# valid = lgb.Dataset(valid_data, label=valid_label, )
# bst = lgb.train(params, data, num_boost_round=100, valid_sets=[valid], )


# 6.2
# train_data = train_data.fillna(0)
# valid_data = valid_data.fillna(0)
# test_data = test_data.fillna(0)
#
# from sklearn.ensemble import RandomForestRegressor
# model = RandomForestRegressor(n_estimators=100,
#                                   random_state=0).fit(train_data, train_label)

# 7. 分析重要性


# main 主函数
def main():
    ana_train_data()


if __name__ == "__main__":
    main()