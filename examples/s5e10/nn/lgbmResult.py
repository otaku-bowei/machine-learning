import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from datetime import datetime

from examples.s5e10.nn.data import read_org_data


# 1. 加载LightGBM模型
def load_lgbm_model(model_path):
    """
    加载训练好的LightGBM模型

    Args:
        model_path: 模型文件路径，通常是.txt格式

    Returns:
        model: 加载好的LightGBM模型
    """
    print(f"正在加载LightGBM模型: {model_path}")

    # 检查文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 加载模型
    try:
        model = lgb.Booster(model_file=model_path)
        print(f"模型加载成功！特征数量: {model.num_feature()}")
        return model
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        raise


# 2. 预处理新数据
def preprocess_new_data(data, categorical_columns=None):
    """
    预处理用于预测的新数据

    Args:
        data: pandas DataFrame，需要预测的数据
        categorical_columns: 分类特征列名列表，默认为None

    Returns:
        X: 预处理后的特征数据
    """
    print("\n开始预处理预测数据...")

    # 创建数据副本以避免修改原始数据
    df = data.copy()

    # 如果有id列，删除id列
    if 'id' in df.columns:
        X = df.drop(['id'], axis=1)
    else:
        X = df.copy()

    # 如果提供了分类特征列，将其转换为category类型
    if categorical_columns:
        for col in categorical_columns:
            if col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = pd.Categorical(X[col])
                print(f"已将特征 '{col}' 转换为分类类型")
    else:
        print("未提供分类特征列表，跳过分类特征转换")

    print(f"预处理完成，特征数量: {X.shape[1]}, 样本数量: {X.shape[0]}")
    return X


# 3. 使用模型进行预测
def predict_with_model(model, X, batch_size=None):
    """
    使用加载的模型进行预测

    Args:
        model: 加载好的LightGBM模型
        X: 预处理后的特征数据
        batch_size: 批处理大小，对于大数据集可设置，默认为None（一次性预测）

    Returns:
        predictions: 预测结果数组
    """
    print("\n开始预测...")

    if batch_size and batch_size < X.shape[0]:
        # 分批预测以节省内存
        predictions = []
        for i in range(0, X.shape[0], batch_size):
            batch_end = min(i + batch_size, X.shape[0])
            batch_pred = model.predict(X.iloc[i:batch_end])
            predictions.extend(batch_pred)
            print(f"已预测批次: {i // batch_size + 1}, 处理样本: {batch_end}/{X.shape[0]}")
    else:
        # 一次性预测
        predictions = model.predict(X)

    # 转换为numpy数组
    predictions = np.array(predictions)

    # 显示预测结果的统计信息
    print(f"\n预测结果统计信息:")
    print(f"预测样本数: {len(predictions)}")
    print(f"预测值范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"预测值均值: {predictions.mean():.4f}")
    print(f"预测值标准差: {predictions.std():.4f}")

    return predictions


# 4. 保存预测结果
def save_predictions(data, predictions, output_path='predictions.csv',
                     id_column='id', prediction_column='accident_risk'):
    """
    将预测结果保存为CSV文件

    Args:
        data: 原始输入数据（用于获取id列）
        predictions: 预测结果数组
        output_path: 输出文件路径
        id_column: id列名
        prediction_column: 预测结果列名
    """
    print(f"\n保存预测结果到: {output_path}")

    # 创建结果DataFrame
    result_df = pd.DataFrame({
        prediction_column: predictions
    })

    # 如果数据中有id列，添加到结果中
    if id_column in data.columns:
        result_df[id_column] = data[id_column].values
        # 将id列移到第一列
        cols = [id_column, prediction_column]
        result_df = result_df[cols]

    # 保存为CSV
    try:
        result_df.to_csv(output_path, index=False)
        print(f"预测结果已成功保存！文件路径: {output_path}")
        print(f"预测结果预览:")
        print(result_df.head())
        return result_df
    except Exception as e:
        print(f"保存预测结果失败: {str(e)}")
        raise


# 5. 主预测函数
def lgbm_predict_pipeline(model_path, data_path, output_path=None,
                          categorical_columns=None, batch_size=None):
    """
    完整的LightGBM预测流程

    Args:
        model_path: 模型文件路径
        data_path: 预测数据文件路径
        output_path: 预测结果输出路径，默认为None（自动生成）
        categorical_columns: 分类特征列名列表
        batch_size: 批处理大小

    Returns:
        result_df: 包含预测结果的DataFrame
    """
    print("=== LightGBM 预测流程开始 ===")

    # 如果未提供输出路径，自动生成
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'predictions_{timestamp}.csv'

    # 加载数据
    print(f"正在加载预测数据: {data_path}")
    data = pd.read_csv(data_path)
    print(f"数据加载完成，形状: {data.shape}")

    # 加载模型
    model = load_lgbm_model(model_path)

    # 预处理数据
    X = preprocess_new_data(data, categorical_columns)

    # 进行预测
    predictions = predict_with_model(model, X, batch_size)

    # 保存预测结果
    result_df = save_predictions(data, predictions, output_path)

    print("\n=== 预测流程完成 ===")
    return result_df


# 6. 辅助函数：加载特征重要性
def load_feature_importance(model_path, output_csv=None):
    """
    加载模型的特征重要性

    Args:
        model_path: 模型文件路径
        output_csv: 输出CSV文件路径，默认为None（不保存）

    Returns:
        importance_df: 特征重要性DataFrame
    """
    print(f"加载模型特征重要性: {model_path}")
    model = load_lgbm_model(model_path)

    importance_df = pd.DataFrame({
        'Feature': model.feature_name(),
        'Importance': model.feature_importance(importance_type='split')
    }).sort_values('Importance', ascending=False).reset_index(drop=True)

    print("\n特征重要性（前10个）:")
    print(importance_df.head(10))

    if output_csv:
        importance_df.to_csv(output_csv, index=False)
        print(f"\n特征重要性已保存至: {output_csv}")

    return importance_df


# 1. 加载模型
model = load_lgbm_model('lightgbm_model.txt')

# 2. 加载和预处理数据
_, data = read_org_data()
categorical_columns = ['road_type', 'lighting', 'weather', 'time_of_day',
                      'holiday', 'school_season', 'road_signs_present', 'public_road']
X = preprocess_new_data(data, categorical_columns)

# 3. 进行预测
predictions = predict_with_model(model, X)

# 4. 保存结果
save_predictions(data, predictions, 'predictions.csv')