# Import necessary libraries
import numpy as np
import polars as pl
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import os
import joblib
import kaggle_evaluation.jane_street_inference_server

# Set up constants
TARGET = 'responder_6'
FEAT_COLS = [f"feature_{i:02d}" for i in range(79)]


# Function to load data with optional filtering
def load_data(date_id_range=None, time_id_range=None, columns=None, return_type='pl'):
    data_dir = '../input/jane-street-real-time-market-data-forecasting'
    data = pl.scan_parquet(f"{data_dir}/train.parquet")

    if date_id_range is not None:
        start_date, end_date = date_id_range
        data = data.filter((pl.col("date_id") >= start_date) & (pl.col("date_id") <= end_date))

    if time_id_range is not None:
        start_time, end_time = time_id_range
        data = data.filter((pl.col("time_id") >= start_time) & (pl.col("time_id") <= end_time))

    if columns is not None:
        data = data.select(columns)

    if return_type == 'pd':
        return data.collect().to_pandas()
    else:
        return data.collect()


# Function to calculate R² score
def calculate_r2(y_true, y_pred, weights):
    numerator = np.sum(weights * (y_true - y_pred) ** 2)
    denominator = np.sum(weights * (y_true ** 2))
    r2_score = 1 - (numerator / denominator)
    return r2_score


# Function to evaluate the model
def evaluate_model(model, test_data):
    y_pred = model.predict(test_data[FEAT_COLS])
    y_true = test_data[TARGET].to_numpy()
    weights = test_data['weight'].to_numpy()
    r2_score = calculate_r2(y_true, y_pred, weights)
    print(f"Sample weighted zero-mean R-squared score (R2) on test data: {r2_score}")


# Class to manage a group of models
class ModelGroup:
    def __init__(self):
        self.models = []

    def add_model(self, model):
        self.models.append(model)

    def predict(self, test_data):
        preds = []
        for model in self.models:
            if isinstance(model, lgb.Booster):
                pred = model.predict(test_data[FEAT_COLS])
            elif isinstance(model, xgb.Booster):
                pred = model.predict(xgb.DMatrix(test_data[FEAT_COLS]))
            elif hasattr(model, 'predict'):
                pred = model.predict(test_data[FEAT_COLS])
            else:
                raise ValueError("Unsupported model type")
            preds.append(pred)

        avg_pred = np.mean(preds, axis=0)
        return avg_pred

    @classmethod
    def load(cls, file_path):
        model_group = joblib.load(file_path)
        return model_group


# Function to train XGBoost with K-Folds
def train_xgb_kfold(total_days=1498, n_splits=5, save_models=False):
    model_group = ModelGroup()
    fold_size = total_days // n_splits
    folds = [(i * fold_size, min((i + 1) * fold_size - 1, total_days - 1)) for i in range(n_splits)]

    for fold_idx in range(n_splits):
        valid_range = folds[fold_idx]
        train_ranges = [folds[i] for i in range(n_splits) if i != fold_idx]

        print(f"Fold {fold_idx}: validation range {valid_range}, train parts: {train_ranges}")

        valid_data = load_data(date_id_range=valid_range, columns=["date_id", "weight"] + FEAT_COLS + [TARGET],
                               return_type='pl')
        valid_weight = valid_data['weight'].to_pandas()

        train_data = None
        for train_range in train_ranges:
            partial_train_data = load_data(date_id_range=train_range,
                                           columns=["date_id", "weight"] + FEAT_COLS + [TARGET], return_type='pl')
            if train_data is None:
                train_data = partial_train_data
            else:
                train_data = train_data.vstack(partial_train_data)

        train_weight = train_data['weight'].to_pandas()

        dtrain = xgb.DMatrix(train_data.select(FEAT_COLS).to_pandas(), label=train_data[TARGET].to_pandas(),
                             weight=train_weight)
        dvalid = xgb.DMatrix(valid_data.select(FEAT_COLS).to_pandas(), label=valid_data[TARGET].to_pandas(),
                             weight=valid_weight)

        XGB_PARAMS = {
            'eval_metric': 'rmse',
            'learning_rate': 0.5,
            'max_depth': 12,
            'min_child_weight': 1.5,
            'subsample': 0.8555,
            'colsample_bytree': 0.85555555,
            'random_state': 42,
            'tree_method': 'gpu_hist',
        }

        model = xgb.train(XGB_PARAMS, dtrain, num_boost_round=1000, evals=[(dtrain, 'train'), (dvalid, 'valid')],
                          early_stopping_rounds=100, verbose_eval=50)

        y_valid_pred = model.predict(dvalid)
        r2_score = calculate_r2(valid_data[TARGET].to_pandas(), y_valid_pred, valid_weight)
        print(f"Fold {fold_idx} validation R2 score: {r2_score}")

        model_group.add_model(model)

    if save_models:
        joblib.dump(model_group, "xgb_model_group.pkl")
        print("Saved the model group to xgb_model_group.pkl")

    return model_group


# Uncomment to train a new model
# total_days = 1699
# xgb_models = train_xgb_kfold(total_days=total_days, n_splits=5, save_models=False)

# Load pre-trained model group
xgb_models = ModelGroup.load("/kaggle/input/xgb_model/other/default/1/xgb_model_group.pkl")
lags_ = None


# Prediction function for the inference server
def predict(test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame | pd.DataFrame:
    global lags_
    if lags is not None:
        lags_ = lags

    feat = test[FEAT_COLS].to_pandas()
    pred = xgb_models.predict(feat)

    predictions = test.select('row_id').with_columns(pl.Series('responder_6', pred.ravel()))

    assert isinstance(predictions, (pl.DataFrame, pd.DataFrame))
    assert list(predictions.columns) == ['row_id', 'responder_6']
    assert len(predictions) == len(test)

    return predictions


# Set up the inference server
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
