import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import contant


def read_org_data():
    train_org_data = pd.read_csv(contant.Data.get('train_path'))
    test_org_data = pd.read_csv(contant.Data.get('test_path'))
    print("训练数据形状:", train_org_data.shape)
    print("前5行数据:", train_org_data.head())
    return train_org_data, test_org_data


def analyze_data_info(data: pd.DataFrame, print_details: bool = True) -> dict:
    """
    分析数据信息，包括空值统计、基本统计量和数据类型

    参数:
        data: pandas DataFrame，要分析的数据
        print_details: 是否打印详细信息

    返回:
        dict: 包含分析结果的字典
    """
    analysis_results = {}

    # 1. 基本信息
    basic_info = {
        '总行数': len(data),
        '总列数': len(data.columns),
        '数据类型': data.dtypes.value_counts().to_dict()
    }
    analysis_results['基本信息'] = basic_info

    if print_details:
        print("=== 基本信息 ===")
        print(f"总行数: {basic_info['总行数']}")
        print(f"总列数: {basic_info['总列数']}")
        print(f"数据类型分布: {basic_info['数据类型']}")

    # 2. 空值统计
    null_stats = data.isnull().sum()
    null_percent = (null_stats / len(data) * 100).round(2)
    null_info = pd.DataFrame({
        '空值数量': null_stats,
        '空值百分比': null_percent
    })[null_stats > 0]

    analysis_results['空值统计'] = null_info.to_dict()

    if print_details:
        print("\n=== 空值统计 ===")
        if len(null_info) > 0:
            print(null_info)
        else:
            print("✓ 没有发现空值")

    # 3. 数值特征统计（均值、标准差、最小值、最大值等）
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        numerical_stats = data[numerical_cols].describe().T
        numerical_stats['方差'] = data[numerical_cols].var()
        numerical_stats['偏度'] = data[numerical_cols].skew()
        numerical_stats['峰度'] = data[numerical_cols].kurtosis()

        analysis_results['数值特征统计'] = numerical_stats.to_dict()

        if print_details:
            print("\n=== 数值特征统计 ===")
            # 打印均值、标准差、最小值、最大值
            print(numerical_stats[['mean', 'std', 'min', 'max']].round(4))

    # 4. 分类特征统计（唯一值数量、出现频率最高的值等）
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        categorical_info = {}
        for col in categorical_cols:
            unique_count = data[col].nunique()
            top_value = data[col].mode().iloc[0] if not data[col].mode().empty else None
            top_count = data[col].value_counts().iloc[0] if not data[col].value_counts().empty else 0

            categorical_info[col] = {
                '唯一值数量': unique_count,
                '最常见值': top_value,
                '最常见值出现次数': top_count,
                '最常见值占比': round(top_count / len(data) * 100, 2)
            }

        analysis_results['分类特征统计'] = categorical_info

        if print_details:
            print("\n=== 分类特征统计 ===")
            for col, info in categorical_info.items():
                print(f"\n{col}:")
                print(f"  唯一值数量: {info['唯一值数量']}")
                print(f"  最常见值: {info['最常见值']} (出现 {info['最常见值出现次数']} 次, {info['最常见值占比']}%)")

    # 5. 相关性分析（仅数值特征）
    if len(numerical_cols) > 1 and print_details:
        print("\n=== 数值特征相关性（绝对值大于0.5）===")
        corr = data[numerical_cols].corr()
        high_corr = corr[abs(corr) > 0.5]
        # 只显示上三角矩阵，排除自相关
        high_corr = high_corr.where(np.triu(np.ones(high_corr.shape), k=1).astype(bool))
        high_corr = high_corr.stack().reset_index()
        high_corr.columns = ['特征1', '特征2', '相关系数']
        high_corr = high_corr.sort_values('相关系数', key=abs, ascending=False)
        print(high_corr.round(4))

    return analysis_results

class S5e10Data(Dataset):
    def __init__(self, data: pd.DataFrame, train=True, preprocessor=None):
        # 分离特征和标签
        self.train = train
        self.X = data[contant.Data.get('all_feature')].copy()

        # 处理标签（如果是训练数据）
        # if train and contant.Data.get('label_field') in data.columns:
        if contant.Data.get('label_field') in data.columns:
            self.y = data[contant.Data.get('label_field')].values.reshape(-1, 1)
        else:
            self.y = None

        # 特征工程：处理分类特征和数值特征
        self.categorical_features = contant.Data.get('type_feature')
        self.numerical_features = [f for f in contant.Data.get('all_feature')
                                   if f not in self.categorical_features]

        # 创建预处理器（如果没有提供）
        if preprocessor is None and train:
            # 数值特征处理：标准化
            numerical_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])

            # 分类特征处理：独热编码
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            # 组合预处理器
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, self.numerical_features),
                    ('cat', categorical_transformer, self.categorical_features)
                ])

            # 拟合预处理器并转换数据
            self.X_transformed = self.preprocessor.fit_transform(self.X)
        elif preprocessor is not None:
            # 使用提供的预处理器进行转换（用于测试数据）
            self.preprocessor = preprocessor
            self.X_transformed = self.preprocessor.transform(self.X)
        else:
            # 不进行预处理
            self.preprocessor = None
            self.X_transformed = self.X.values

    def __len__(self):
        # 返回数据集的总长度
        return len(self.X)

    def __getitem__(self, idx):
        # 修复：正确处理稀疏矩阵和数组索引
        if hasattr(self.X_transformed, 'toarray'):
            # 对于稀疏矩阵（如OneHotEncoder的输出）
            x = torch.tensor(self.X_transformed[idx].toarray().flatten(), dtype=torch.float)
        else:
            # 对于普通numpy数组
            x = torch.tensor(self.X_transformed[idx], dtype=torch.float)

        # 确保标签格式正确
        if self.y is not None:
            y = torch.tensor(self.y[idx], dtype=torch.float)
            return x, y  # 确保总是返回元组(x, y)，不会有多余的值
        return x

    def get_input_dim(self):
        # 获取转换后的输入维度
        if hasattr(self.X_transformed, 'shape'):
            return self.X_transformed.shape[1]
        elif hasattr(self.X_transformed, 'toarray'):
            return self.X_transformed.toarray().shape[1]
        else:
            return len(self.X_transformed[0])