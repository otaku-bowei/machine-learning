import numpy as np
import pandas as pd
import pyarrow as pa
from tqdm import tqdm
import pyarrow.parquet as pq

'''
目标:预测responder_6
数据:
    (1)date_id和time_id--时间序列id,但顺序时间结构并不具备相同间隔
    (2)symbol_id--金融资产id
    (3)weight--计算得分函数的权重
    (4)feature_{00...78}--79个匿名特征
    (5)responder_{0...8}--9个匿名响应，范围是(-5,5),需要预测responder_6
测试集:
    (1)is_scored--标识此行是否在评估量度计算中,某些日期不计分,这可能包括训练和公开排行榜/公开排行榜到私人排行榜之间的过渡期
    (2)date_id和time_id--时间序列id,但顺序时间结构并不具备相同间隔
    (3)symbol_id--金融资产id
    (4)weight--计算得分函数的权重
    (5)feature_{00...78}--79个匿名特征
需要处理的问题:
    (1)symbol_id按计算编码分类
    (2)回归预测
    (3)lags的使用
    (4)匿名特征的空值处理--添加特征存在标志标识数据
    (5)responder的关系
    (6)权重对得分函数的影响
    (7)is_scored对梯度的影响
要求:
    (1)在线学习
    (2)基于time-series 模块进行预测
'''


# read_org_data 读取数据
def read_org_data():
    train_date_0_path = "C:\Program Files\BusinessFile\新手村\AI培训资料\数据集\Jane Street\\train.parquet\partition_id=0\part-0.parquet"
    train_date_1_path = "C:\Program Files\BusinessFile\新手村\AI培训资料\数据集\Jane Street\\train.parquet\partition_id=1\part-0.parquet"
    lags_path = "C:\Program Files\BusinessFile\新手村\AI培训资料\数据集\Jane Street\lags.parquet\date_id=0\part-0.parquet"
    test_path = "C:\Program Files\BusinessFile\新手村\AI培训资料\数据集\Jane Street\\test.parquet\date_id=0\part-0.parquet"
    train_data_0 = pd.read_parquet(train_date_0_path)
    train_data_1 = pd.read_parquet(train_date_1_path)
    lags = pd.read_parquet(lags_path)
    test_data = pd.read_parquet(test_path)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    # print(train_data_0.head())
    # print(train_data_1.head())
    # print(lags.head())
    # print(test_data.head())
    return pd.concat([train_data_0, train_data_1], axis=0), lags_path, test_data


# read_deal_data 读取数据
def read_deal_data():
    train_date_path = "./deal_train_data.parquet"
    train_data_1 = pd.read_parquet(train_date_path)
    return train_data_1


# deal_train_data 对数据进行预处理
def deal_train_data(train_data: pd.DataFrame):
    # 1.填充空值，添加相关指示值
    # 乱序
    train_data = train_data.drop(columns=['symbol_id'])
    # concat
    data_len = len(train_data)
    var_len = len(train_data.iloc[0])
    # 79个特征需要添加标识列，标识该值是否为空,为空为1*weight,不为空为0
    X_cnt = np.zeros((len(train_data), var_len + 79))
    for j in tqdm(range(data_len)):
    # for j in tqdm(range(0, 5000)):
        train_row = train_data.iloc[j]
        weight = train_row['weight']
        for i in range(len(train_row)):
            if i < 3:
                X_cnt[j, i] = train_row[i]
            elif i > 81:
                X_cnt[j, i + 79] = train_row[i]
            elif train_row[i] is None or train_row[i] == '' or np.isnan(train_row[i]):
                X_cnt[j, i] = 0
                X_cnt[j, i + 79] = weight
            else:
                X_cnt[j, i] = train_row[i]
                X_cnt[j, i + 79] = 0
    save_data = pd.DataFrame(X_cnt)
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # print(save_data.head(5))
    table = pa.Table.from_pandas(save_data)
    pq.write_table(table, 'deal_train_data.parquet')


# main 主函数
def main():
    # 1.pd读取数据集，提取相关有用信息并做数据预处理
    train_data, lag_data, test_data = read_org_data()
    deal_train_data(train_data)
    # train = read_deal_data()
    # print(train)


if __name__ == "__main__":
    main()
