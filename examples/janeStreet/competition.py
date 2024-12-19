import os

import pandas as pd
import polars as pl
import numpy as np
import os
from pathlib import Path
import lightgbm_self as lgb
# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDRegressor
from tqdm import tqdm
import tensorflow as tf

# Gather the train data partitions
# DATA_DIR = Path('/kaggle/input/jane-street-real-time-market-data-forecasting')
DATA_DIR = Path('C:\Program Files\BusinessFile\新手村\AI培训资料\数据集\Jane Street\\')
N_PARTITION = len(os.listdir(DATA_DIR / 'train.parquet'))
M_PARTITION = len(os.listdir(DATA_DIR / 'test.parquet'))
train_parquets = [f"{DATA_DIR}/train.parquet/partition_id={i}/part-0.parquet" for i in range(N_PARTITION)]
test_parquets = [f"{DATA_DIR}/test.parquet/date_id={i}/part-0.parquet" for i in range(M_PARTITION)]


# Load a subset of the data, you can choose any partition or multiple
# df = pd.read_parquet(train_parquets[6])

def read_org_data():
    train_ds = pd.concat([pd.read_parquet(file) for file in train_parquets])
    test_ds = pd.concat([pd.read_parquet(file) for file in test_parquets])
    return train_ds, test_ds


def read_org_data_tmp():
    train_date_0_path = "C:\Program Files\BusinessFile\新手村\AI培训资料\数据集\Jane Street\\train.parquet\partition_id=0\part-0.parquet"
    train_date_1_path = "C:\Program Files\BusinessFile\新手村\AI培训资料\数据集\Jane Street\\train.parquet\partition_id=1\part-0.parquet"
    lags_path = "C:\Program Files\BusinessFile\新手村\AI培训资料\数据集\Jane Street\lags.parquet\date_id=0\part-0.parquet"
    test_path = "C:\Program Files\BusinessFile\新手村\AI培训资料\数据集\Jane Street\\test.parquet\date_id=0\part-0.parquet"
    train_data_0 = pd.read_parquet(train_date_0_path)
    train_data_1 = pd.read_parquet(train_date_1_path)
    lags = pd.read_parquet(lags_path)
    test_data = pd.read_parquet(test_path)
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # print(train_data_0.describe())
    # print(train_data_1.head())
    # print(lags.head())
    # print(test_data.head())
    # return pd.concat([train_data_0, train_data_1], axis=0), lags_path, test_data
    # return pd.concat((train_data_0, train_data_1)), test_data
    return train_data_0, test_data

def read_lags_data():
    lags_path = "C:\Program Files\BusinessFile\新手村\AI培训资料\数据集\Jane Street\lags.parquet\date_id=0\part-0.parquet"
    lags = pd.read_parquet(lags_path)
    return lags

def deal_data(data : pd.DataFrame):
    # 1.填充空值，添加相关指示值
    # 使用isna()方法检查NaN值，并将结果转换为整数（True变为1，False变为0）
    nan_flags = data.isna().astype(int)
    # 为了区分原DataFrame和NaN标志DataFrame，可以给后者添加后缀
    nan_flags.columns = [col + '_NaN_flag' for col in data.columns]
    # 将NaN标志列添加到原DataFrame中
    data = pd.concat([data, nan_flags], axis=1)
    # data = data.fillna(0.0)
    return data

# part_data 将每个训练集合切片，并保留交叉校验集合
def read_part_data(train_ds : pd.DataFrame):
    train_pd = train_ds.groupby('date_id')
    return train_pd

def read_part_data_by_symbol_id(train_ds : pd.DataFrame, chunk_size : int):
    train_pd = train_ds.groupby('date_id')
    return train_pd

def online_learning(train_d : pd.DataFrame, model : SGDRegressor() = None):
    if model is None:
        model = SGDRegressor()
    # cost = metrics.MAE()
    train_part_data = read_part_data(train_d)
    sorted_groups = sorted(train_part_data.groups.keys())
    # sorted_groups = sorted_groups[0:2]
    for date_id in tqdm(sorted_groups):
        train_d_tmp = train_part_data.get_group(date_id)
        # 训练数据每次用5条来进行学习
        # print(train_d_tmp.head())
        # print(date_id)
        # 对数据进行区分
        train_data = train_d_tmp.drop(
            columns=['date_id', 'time_id', 'symbol_id', 'responder_0', 'responder_1', 'responder_2', 'responder_3', 'responder_4', 'responder_5',
                     'responder_6', 'responder_7', 'responder_8'])

        train_label = train_d_tmp.loc[:, ['responder_0', 'responder_1', 'responder_2', 'responder_3', 'responder_4', 'responder_5',
                     'responder_6', 'responder_7', 'responder_8']]
        train_data = deal_data(train_data)
        train_data = train_data.fillna(0.0)
        # 训练集逐一训练更新
        # length = len(train_data)
        train_numpy = train_data.to_numpy()
        test_numpy = train_label.to_numpy()
        model.partial_fit(train_numpy, test_numpy)
        # for i in tqdm(range(0, length)):
        #     d, l = train_data.iloc[i:i+1], train_label[i:i+1]
        #     d_dict = d.iloc[0, :].to_dict()
        #     l_num = l.iloc[0, l.columns.get_loc('responder_6')]
        #     weight = d.iloc[0, d.columns.get_loc('weight')]
        #     y_pred = model.predict(d_dict)
            # cost = model.score()
            # model.learn_one(d_dict, l_num)
            # cost.update(l_num, y_pred, weight)
        # print(cost)

def online_learning_by_symbol(train_d : pd.DataFrame, model : SGDRegressor() = None):
    if model is None:
        model = SGDRegressor()
    # sorted_groups = sorted_groups[0:2]
    train_data_len = len(train_d)
    for batch in tqdm(range(0, int(train_data_len / 39))):
        train_d_tmp = train_d.loc[batch * 39 : (batch + 1) * 39 - 1]
        train_d_tmp = train_d_tmp.reset_index(drop=True)
        # 训练数据每次用5条来进行学习
        # print(train_d_tmp.head())
        # print(date_id)
        # 对数据进行区分
        train_data = train_d_tmp.drop(
            columns=['date_id', 'time_id', 'symbol_id', 'responder_0', 'responder_1', 'responder_2', 'responder_3', 'responder_4', 'responder_5',
                     'responder_6', 'responder_7', 'responder_8'])
        train_label = train_d_tmp.loc[:, ['responder_6']]
        # 拼接lags
        train_data = train_data.fillna(0.0)
        # 训练集逐一训练更新
        train_numpy = train_data.to_numpy()
        test_numpy = train_label.to_numpy().ravel()
        model.partial_fit(train_numpy, test_numpy)

model = SGDRegressor()


def online_predict(test_data : pd.DataFrame, model : SGDRegressor()) -> pd.DataFrame:
    test_data_tmp = test_data.drop(columns=['row_id', 'date_id', 'time_id', 'symbol_id', 'is_scored'])
    length = len(test_data) / 39
    result = np.ndarray
    for i in tqdm(range(0, int(length))):
        d = test_data_tmp.loc[i * 39:(i+1) * 39 - 1]
        d = d.reset_index(drop=True)
        d = d.fillna(0.0)
        x = d.to_numpy()
        y_pred = model.predict(x)
        if i == 0:
            result = y_pred
        else:
            result = np.append(result, y_pred)
        # test_data.loc[i : i + 39, 'responder_6'] = y_pred
    test_data['responder_6'] = result
    return test_data.loc[:, ['row_id', 'responder_6']]

def nn(train_d, train_l: pd.DataFrame):
    callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda batch, logs: [callback.on_train_begin])
    # 1.建立全连接神经网络
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        # tf.keras.layers.Dropout(0.2),
        # 此处激活函数帮我跳出了局部最小值，线性激活函数输出的是z1-z6,用softmax才会输出概率值
        tf.keras.layers.Dense(9, activation='relu')
    ])
    # 2.调整数据格式
    train_data = train_d.astype(np.float64)
    train_label = train_l.astype(np.float64)
    # 4.1定义损失函数--得分损失函数,用均方误差作为反向传播基准
    # loss_fn = Cost(weights=train_data['weights'])
    loss_fn = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer='adam', loss=loss_fn, metrics=['mae'])
    # 5.进行训练
    """
    1.mae损失函数，简单的nn，mae≈5.8，loss≈9.4
    2.根据得分函数自定义损失函数，nn不变，
    """
    train_history = model.fit(train_data, train_label, epochs=2)
    # 6.保存模型
    return model


def plot_by_period(df, days, column, symbols=None, time_id_ranges=None, color_by='period'):
    """
    Plots a specified feature or responder against time_id for given days, grouped by trading periods or symbol.

    Parameters:
        df (DataFrame): The dataframe containing the data to plot.
        days (str, int, list or tuple): The days (date_id) to visualize. Tuple will be start and end of a range.
        column (str): The column (feature or responder) to plot.
        symbols (str, int, list, or tuple, optional): The symbol(s) to filter the data by. Tuple will be start
            and end of a range. Default is None (no filtering).
        time_id_ranges (dict, optional): Custom time_id boundaries for trading periods.
            Default is Pre-Market, Regular Trading Hours, and After-Hours.
        color_by (str, optional): Determines the color coding ('period', 'symbol', or None). Default is 'period'.
    """
    # Default trading periods
    if time_id_ranges is None:
        time_id_ranges = {
            'Pre-Market': (0, 332),
            'Regular Trading Hours': (333, 725),
            'After-Hours': (726, 968)
        }

    # Function to identify trading period
    def assign_period(time_id):
        for period, (start, end) in time_id_ranges.items():
            if start <= time_id <= end:
                return period
        return 'Unknown'

    # Function to map hours to time_id ranges
    def hour_to_time_id(hour):
        total_minutes = (hour - 4) * 60
        return int((total_minutes / (16 * 60)) * 968)

    # Prepare hour ticks and labels
    hour_ticks = [hour_to_time_id(hour) for hour in range(4, 21)]
    hour_labels = [f"{hour:02d}:00" for hour in range(4, 21)]

    # Handle days parameter
    if isinstance(days, (str, int)):
        days = [days]
    elif isinstance(days, tuple) and len(days) == 2:
        start, end = days
        days = list(range(start, end + 1))

    # Filter by symbol if provided
    if symbols is not None:
        if isinstance(symbols, tuple) and len(symbols) == 2:
            start, end = symbols
            symbols = list(range(start, end + 1))
            suffix = f" for Symbols {start} - {end}"
        elif isinstance(symbols, (str, int)):
            symbols = [symbols]
            suffix = f" for Symbol {symbols[0]}"
        elif isinstance(symbols, list):
            suffix = f" for Symbols {', '.join(map(str, symbols))}"
        df = df[df['symbol_id'].isin(symbols)]
    elif color_by == 'period':
        suffix = ' by Time Period'
    elif color_by == 'symbol':
        suffix = ' by Symbol'
    else:
        suffix = ''

    # Loop through each day and create a plot
    for day in days:
        df_day = df[df['date_id'] == day].copy()

        # Assign coloring based on the `color_by` parameter
        if color_by == 'period':
            df_day['color_group'] = df_day['time_id'].apply(assign_period)
            color_labels = list(time_id_ranges.keys())
        elif color_by == 'symbol':
            # Assign prefixed values directly to color_group
            df_day['color_group'] = df_day['symbol_id'].apply(lambda symbol: f"Symbol {symbol}")
            color_labels = df_day['color_group'].unique()
        else:
            df_day['color_group'] = 'All Data'
            color_labels = ['All Data']

        fig, ax1 = plt.subplots(figsize=(14, 7))

        # Set vertical grid aligned with trading hours
        ax1.set_xticks(hour_ticks)
        ax1.grid(color='lightgrey', linestyle='-', linewidth=0.5, alpha=1, zorder=0)

        # Plot color-coded data for each group
        for group in color_labels:
            group_data = df_day[df_day['color_group'] == group]
            ax1.plot(group_data['time_id'], group_data[column], label=group, alpha=0.8, zorder=3)

        ax1.set_title(f"{column} on Day {day}{suffix}", fontsize=16, pad=14)
        ax1.set_xlabel("Time ID", fontsize=14, labelpad=10)
        ax1.set_ylabel(column, fontsize=14, labelpad=10)
        ax1.legend()

        # Add secondary x-axis for trading hours
        ax2 = ax1.secondary_xaxis('top')
        ax2.set_xticks(hour_ticks)
        ax2.set_xticklabels(hour_labels)
        ax2.set_xlabel("Trading Hour", fontsize=14, labelpad=10)

        plt.show()


# Replace this function with your inference code.
# You can return either a Pandas or Polars dataframe, though Polars is recommended.
# Each batch of predictions (except the very first) must be returned within 1 minute of the batch features being provided.
def predict(test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame | pd.DataFrame:
    """Make a prediction."""
    # All the responders from the previous day are passed in at time_id == 0. We save them in a global variable for access at every time_id.
    # Use them as extra features, if you like.
    global lags_
    if lags is not None:
        lags_ = lags
    # Replace this section with your own predictions
    # predictions = test.select(
    #     'row_id',
        # 'responder_6'
        # pl.lit('responder_6').alias('responder_6'),
    # )
    data_dict = test.to_dict()
    # 使用字典创建一个 pandas.DataFrame 对象
    pd_df = pd.DataFrame(data_dict)
    predictions = online_predict(pd_df, model)
    if isinstance(predictions, pl.DataFrame):
        assert predictions.columns == ['row_id', 'responder_6']
    elif isinstance(predictions, pd.DataFrame):
        assert (predictions.columns == ['row_id', 'responder_6']).all()
    else:
        raise TypeError('The predict function must return a DataFrame')
    # Confirm has as many rows as the test data.
    assert len(predictions) == len(test)
    return predictions


def main():
    train_data, test_data = read_org_data_tmp()
    lags = read_lags_data()
    # print(test_data.head())
    # train_data, test_data = deal_data(train_data, test_data)
    # train_d = train_data.drop(
    #     columns=['responder_0', 'responder_1', 'responder_2', 'responder_3', 'responder_4', 'responder_5',
    #              'responder_6', 'responder_7', 'responder_8'])
    # train_l = train_data.loc[:,
    #           ['responder_0', 'responder_1', 'responder_2', 'responder_3', 'responder_4', 'responder_5', 'responder_6',
    #            'responder_7', 'responder_8']]
    # model = nn(train_d, train_l)
    #
    # test_d = test_data.drop(columns=['row_id'])
    # test_result = model.predict(test_d)
    # tr = pd.DataFrame(test_result,
    #                   columns=['responder_0', 'responder_1', 'responder_2', 'responder_3', 'responder_4', 'responder_5',
    #                            'responder_6',
    #                            'responder_7', 'responder_8'])
    # tr = pd.concat((test_data, tr), axis=1)
    # output = predict(pl.from_pandas(tr), None).to_pandas()
    # output.to_csv('submission.csv', index=False)
    # online_learning(train_data, model)
    online_learning_by_symbol(train_data, model)
    # online_predict(test_data, model)
    output = predict(pl.from_pandas(test_data), None)
    output.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()

