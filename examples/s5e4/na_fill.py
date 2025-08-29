import matplotlib.pyplot as plt
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sympy import false
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import eli5
import seaborn as sns
from eli5.sklearn import PermutationImportance
import shap  # package used to calculate Shap values

import tool.matplotlib.draw_matplotlib as dm
# id、播客_名称、剧集_标题、剧集_长度_分钟、类型、主持人_受欢迎程度_百分比、出版物_日期、出版物_时间、访客_受欢迎度_百分比、广告数量_广告、剧集_评论、收听_时间_分钟
# id,Podcast_Name,Episode_Title,Episode_Length_minutes,Genre,Host_Popularity_percentage,Publication_Day,Publication_Time,Guest_Popularity_percentage,Number_of_Ads,Episode_Sentiment,Listening_Time_minutes
# 0,Mystery Matters,Episode 98,,True Crime,74.81,Thursday,Night,,0.0,Positive,31.41998
train_path = "./playground-series-s5e4/train.csv"
test_path = "./playground-series-s5e4/test.csv"
str_fields = ['Podcast_Name', 'Episode_Title', 'Genre','Publication_Day','Publication_Time', 'Episode_Sentiment']

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

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


# 数据清洗
test_id = test_df.loc[:, ['id']]
test_df = test_df.drop(['id'], axis=1)
for col in str_fields:
    train_df[col] = train_df[col].astype('category')
    test_df[col] = test_df[col].astype('category')

# 筛选空行

# 2. 划分训练集和测试集# ... existing code ...
#
# 筛选 train_df 中含有空值的行
# rows_with_nan = train_df[train_df.isnull().any(axis=1)]
rows_with_nan = train_df[train_df['Guest_Popularity_percentage'].isnull()]
# 打印筛选结果
print("含有空值的行:")
print(rows_with_nan)
sns.scatterplot(data=rows_with_nan, x='id', y='Listening_Time_minutes')
plt.show()
# 做one-hot处理
# train_df = train_df.drop(['id'], axis=1)
#
# train_data = train_df.drop(['Listening_Time_minutes'], axis=1)
# train_label = train_df['Listening_Time_minutes']
#
# l = len(train_df)
# df = pd.concat([train_data, test_df], axis=0)
# df = pd.get_dummies(df, columns=str_fields)
# train_data = df.iloc[:l, ]
# test_df = df.iloc[l:, ]
# train_d, valid_d, train_l, valid_l = train_test_split(train_data, train_label, random_state=1, )



# 1.剧集时长有大量空值，根据MI结果做补充

# 2.Listening_Time_minutes结果值比剧集时长要大2，做数据纠正
