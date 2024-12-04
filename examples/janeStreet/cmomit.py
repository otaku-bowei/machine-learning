import os
import kaggle_evaluation.jane_street_inference_server
import pandas as pd
import polars as pl
import numpy as np
import os
from pathlib import Path

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import tensorflow as tf

# Gather the train data partitions
DATA_DIR = Path('/kaggle/input/jane-street-real-time-market-data-forecasting')
N_PARTITION = len(os.listdir(DATA_DIR / 'train.parquet'))
M_PARTITION = len(os.listdir(DATA_DIR / 'test.parquet'))
train_parquets = [f"{DATA_DIR}/train.parquet/partition_id={i}/part-0.parquet" for i in range(N_PARTITION)]
test_parquets = [f"{DATA_DIR}/test.parquet/date_id={i}/part-0.parquet" for i in range(M_PARTITION)]


# Load a subset of the data, you can choose any partition or multiple
# df = pd.read_parquet(train_parquets[6])

def read_org_data():
    train_ds = pd.read_parquet(train_parquets[6])
    test_ds = pd.read_parquet(test_parquets[0])
    return train_ds, test_ds


def deal_data(train_ds, test_ds: pd.DataFrame):
    # 1.填充空值，添加相关指示值
    train_data = train_ds.drop(columns=['symbol_id'])
    test_ds = test_ds.drop(columns=['symbol_id'])
    test_data = test_ds.drop(columns=['is_scored'])
    return train_data, test_data


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
        pl.lit(0.0).alias('responder_6'),
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

train_data, test_data = read_org_data()

train_data, test_data = deal_data(train_data, test_data)

train_d = train_data.drop(
    columns=['responder_0', 'responder_1', 'responder_2', 'responder_3', 'responder_4', 'responder_5',
            'responder_6', 'responder_7', 'responder_8'])
train_l = train_data.loc[:,
            ['responder_0', 'responder_1', 'responder_2', 'responder_3', 'responder_4', 'responder_5', 'responder_6',
            'responder_7', 'responder_8']]
model = nn(train_d, train_l)

test_d = test_data.drop(columns=['row_id'])
test_result = model.predict(test_d)
tr = pd.DataFrame(test_result,
                columns=['responder_0', 'responder_1', 'responder_2', 'responder_3', 'responder_4', 'responder_5',
                        'responder_6','responder_7', 'responder_8'])
tr = pd.concat((test_data, tr), axis=1)
output = predict(pl.from_pandas(tr), None).to_pandas()
output.to_csv('submission.csv', index=False)