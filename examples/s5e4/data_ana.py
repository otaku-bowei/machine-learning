import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sympy import print_glsl

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

# train_df = train_df.fillna(0.0)
train_df.dropna(inplace=True)

# 看测试集有没有空数据
# null_counts = train_df.isnull().sum()
# print(null_counts)
# null_counts = test_df.isnull().sum()
# print(null_counts)
# rows_with_nan = test_df[test_df.isnull().any(axis=1)]
# print(rows_with_nan)
# 输出MI
# 计算互信息
# train_df.dropna(inplace=True)
class_fields = ['Podcast_Name','Episode_Title','Genre','Publication_Day','Publication_Time','Episode_Sentiment']
regression_fields = ['Episode_Length_minutes','Host_Popularity_percentage','Guest_Popularity_percentage','Number_of_Ads', 'daytime_mi', 'day_mi', 'time_mi']
both_result = train_df.groupby(['Publication_Day', 'Publication_Time'])['Listening_Time_minutes'].mean().reset_index(name='daytime_mi')
day_result = train_df.groupby(['Publication_Day'])['Listening_Time_minutes'].mean().reset_index(name='day_mi')
time_result = train_df.groupby(['Publication_Time'])['Listening_Time_minutes'].mean().reset_index(name='time_mi')
train_df = pd.merge(left=train_df, right=both_result, on=['Publication_Day', 'Publication_Time'], how='left')
train_df = pd.merge(left=train_df, right=day_result, on=['Publication_Day'], how='left')
train_df = pd.merge(left=train_df, right=time_result, on=['Publication_Time'], how='left')
# # 对字符串类型的列进行独热编码
y= train_df.pop('Listening_Time_minutes')
X_class = train_df.loc[:, class_fields]
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
# X_encoded = encoder.fit_transform(X_class)
X_regression = train_df.loc[:, regression_fields]
mi_regression = mutual_info_regression(X_regression, y)
# mi_class = mutual_info_classif(X_encoded, y)
print(mi_regression)
# print("\r\n")
# print(mi_class)
# 分析用户根据日期的记录
# day_group = train_df.groupby(['Publication_Day', 'Publication_Time'])['Listening_Time_minutes'].sum().reset_index()
# day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday','Friday', 'Saturday', 'Sunday']
# day_group['Publication_Day'] = pd.Categorical(day_group['Publication_Day'], categories=day_order, ordered=True)
# sns.lineplot(data=day_group, hue='Publication_Time', x='Publication_Day', y='Listening_Time_minutes')
# plt.show()


# 分析剧集时长相关性
# sns.lmplot(data=train_df, x='Episode_Length_minutes', y='Listening_Time_minutes')
# plt.show()

# y = train_df.pop('Episode_Length_minutes')
# X_regression = train_df.loc[:, ['Host_Popularity_percentage','Guest_Popularity_percentage','Number_of_Ads']]
# mi_regression = mutual_info_regression(X_regression, y)
# print(mi_regression)
# sns.lmplot(data=train_df, x='Guest_Popularity_percentage', y='Episode_Length_minutes')
# plt.show()
# X_class = train_df.loc[:, class_fields]
# from sklearn.preprocessing import OrdinalEncoder
# encoder = OrdinalEncoder()
# X_encoded = encoder.fit_transform(X_class)
# mi_class = mutual_info_classif(X_encoded, y)
# print(mi_class)
# 确认是否有目标值大于剧集时长的数据
# over_time_df = train_df[(train_df['Listening_Time_minutes'] > train_df['Episode_Length_minutes']) & (train_df['Episode_Length_minutes'] > 0)]
# print(over_time_df.loc[:, ['id', 'Episode_Length_minutes', 'Listening_Time_minutes']])
# print(len(over_time_df))
# 判断是否有重复值--TODO
# dd_df = train_df.duplicated(subset=['Podcast_Name', 'Episode_Title', 'Episode_Length_minutes' ], keep='first')
# dd_df = train_df.duplicated(subset=['Podcast_Name', 'Episode_Title'], keep='first')
# print(dd_df)

# 判断类型分组是否多余
# title_group = train_df.groupby(['Podcast_Name', 'Episode_Title'])['Listening_Time_minutes'].sum().reset_index()
# genre_group = train_df.groupby(['Podcast_Name', 'Episode_Title', 'Genre'])['Listening_Time_minutes'].sum().reset_index()
# print(len(title_group))
# print(len(genre_group))
# print(genre_group)
# 对空值处理--TODO

# 分析主持人和收听时间的关系
# sns.scatterplot(data=train_df, x='Host_Popularity_percentage', y='Listening_Time_minutes')
# plt.show()
# sns.scatterplot(data=train_df, x='Guest_Popularity_percentage', y='Listening_Time_minutes')
# plt.show()
# 分析剧集时长和收听时间的关系

# 分析播放类型和收听时长的关系
# sns.scatterplot(data=train_df, hue='Genre', x='id', y='Listening_Time_minutes')
# plt.show()
# 分析剧集评论数和收听时长的关系
# sns.scatterplot(data=train_df, x='Episode_Sentiment', y='Listening_Time_minutes')
# plt.show()
# 分析广告数量和收听时长的关系
# sns.scatterplot(data=train_df, x='Number_of_Ads', y='Listening_Time_minutes')
# plt.show()

# 根据时间字段分组统计条数
# both_result = train_df.groupby(['Publication_Day', 'Publication_Time'])['Listening_Time_minutes'].mean().reset_index(name='mean')
# day_result = train_df.groupby(['Publication_Day'])['id'].count().reset_index(name='count')
# time_result = train_df.groupby(['Publication_Time'])['id'].count().reset_index(name='count')
# print(both_result)
# print('\r\n')
# print(day_result)
# print('\r\n')
# print(time_result)
# plt.show()

