import pandas as pd
import polars as pl
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDRegressor
from tqdm import tqdm
import tensorflow as tf
import tensorflow_decision_forests as tfd

# Gather the train data partitions
DATA_DIR = Path('/kaggle/input/jane-street-real-time-market-data-forecasting')
N_PARTITION = len(os.listdir(DATA_DIR / 'train.parquet'))
M_PARTITION = len(os.listdir(DATA_DIR / 'test.parquet'))
train_parquets = [f"{DATA_DIR}/train.parquet/partition_id={i}/part-0.parquet" for i in range(N_PARTITION)]
test_parquets = [f"{DATA_DIR}/test.parquet/date_id={i}/part-0.parquet" for i in range(M_PARTITION)]
lags_file = f"{DATA_DIR}/lags.parquet/date_id=0/part-0.parquet"


# Load a subset of the data, you can choose any partition or multiple
# df = pd.read_parquet(train_parquets[6])

def read_org_train_data(i: int):
    train_ds = pd.read_parquet(train_parquets[i])
    return train_ds


def read_org_test_data():
    test_ds = pd.read_parquet(test_parquets[0])
    return test_ds


# part_data 将每个训练集合切片，并保留交叉校验集合
def read_part_data(train_ds: pd.DataFrame):
    train_pd = train_ds.groupby('date_id')
    return train_pd


def online_learning_by_symbol(train_d, lags: pd.DataFrame, model: SGDRegressor() = None):
    if model is None:
        model = SGDRegressor()
    # sorted_groups = sorted_groups[0:2]
    train_data_len = len(train_d)
    history_lags = lags.loc[:, ['responder_0_lag_1', 'responder_1_lag_1', 'responder_2_lag_1', 'responder_3_lag_1',
                                'responder_4_lag_1', 'responder_5_lag_1', 'responder_6_lag_1', 'responder_7_lag_1',
                                'responder_8_lag_1']]
    # history_lags = df = pd.DataFrame(0, index=range(rows), columns=range(cols))
    for batch in tqdm(range(0, int(train_data_len / 39))):
        train_d_tmp = train_d.loc[batch * 39: (batch + 1) * 39 - 1]
        train_d_tmp = train_d_tmp.reset_index(drop=True)
        # 对数据进行区分
        train_data = train_d_tmp.drop(
            columns=['date_id', 'time_id', 'symbol_id', 'responder_0', 'responder_1', 'responder_2', 'responder_3',
                     'responder_4', 'responder_5',
                     'responder_6', 'responder_7', 'responder_8'])
        train_label = train_d_tmp.loc[:, ['responder_6']]
        # train_data = deal_data(train_data)
        # 拼接lags
        train_data = pd.concat([train_data, history_lags], axis=1)
        train_data = train_data.fillna(0.0)
        # 训练集逐一训练更新
        # length = len(train_data)
        train_numpy = train_data.to_numpy()
        test_numpy = train_label.to_numpy().reshape(-1)
        model.partial_fit(train_numpy, test_numpy)
        history_lags = train_d_tmp.loc[:,
                       ['responder_0', 'responder_1', 'responder_2', 'responder_3', 'responder_4', 'responder_5',
                        'responder_6', 'responder_7', 'responder_8']]


def online_predict(test_data, lags : pd.DataFrame, model : SGDRegressor()) -> pd.DataFrame:
    test_data_tmp = test_data.drop(columns=['row_id', 'date_id', 'time_id', 'symbol_id', 'is_scored'])
    history_lags = lags.loc[:, ['responder_0_lag_1', 'responder_1_lag_1', 'responder_2_lag_1', 'responder_3_lag_1',
                                'responder_4_lag_1', 'responder_5_lag_1', 'responder_6_lag_1', 'responder_7_lag_1',
                                'responder_8_lag_1']]
    length = len(test_data) / 39
    for i in tqdm(range(0, int(length))):
        d = test_data_tmp.iloc[i:i + 38]
        d = d.reset_index(drop=True)
        d = pd.concat([d, history_lags], axis=1)
        d = d.fillna(0.0)
        x = d.to_numpy()
        y_pred = model.predict(x)
        print(y_pred)
        test_data.loc[i : i + 38, 'responder_6'] = y_pred


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
    predictions = test.select(
        'row_id',
        'response_6'
        # pl.lit(0.0).alias('responder_6'),
    )
    if isinstance(predictions, pl.DataFrame):
        assert predictions.columns == ['row_id', 'responder_6']
    elif isinstance(predictions, pd.DataFrame):
        assert (predictions.columns == ['row_id', 'responder_6']).all()
    else:
        raise TypeError('The predict function must return a DataFrame')
    # Confirm has as many rows as the test data.
    assert len(predictions) == len(test)
    return predictions


def read_lags_data() -> pd.DataFrame:
    lags = pd.read_parquet(lags_file)
    return lags

lags = read_lags_data()
model = SGDRegressor()
for i in range(N_PARTITION):
    train_data = read_org_train_data(0)
    online_learning_by_symbol(train_data, lags, model)
test_data = read_org_test_data()
online_predict(test_data, lags, model)
output = predict(pl.from_pandas(test_data), None).to_pandas()
output.to_csv('submission.csv', index=False)

import kaggle_evaluation.jane_street_inference_server
import os

inference_server = kaggle_evaluation.jane_street_inference_server.JSInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        (
            '/kaggle/input/jane-street-real-time-market-data-forecasting/test.parquet',
            '/kaggle/input/jane-street-real-time-market-data-forecasting/lags.parquet',
        )
    )