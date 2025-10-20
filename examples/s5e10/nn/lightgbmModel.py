import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# 设置随机种子以确保可复现性
np.random.seed(42)


# 1. 数据加载与初步探索
def load_and_explore_data(file_path):
    """加载数据并进行初步探索"""
    print(f"正在加载数据: {file_path}")
    df = pd.read_csv(file_path)

    print(f"\n数据基本信息:")
    print(f"数据集形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    print(f"\n数据类型:")
    print(df.dtypes)
    print(f"\n缺失值统计:")
    print(df.isnull().sum())

    # 显示前几行数据
    print(f"\n前5行数据:")
    print(df.head())

    return df


# 2. 数据预处理
def preprocess_data(df):
    """数据预处理"""
    print("\n开始数据预处理...")

    # 分离特征和目标变量
    X = df.drop(['id', 'accident_risk'], axis=1)
    y = df['accident_risk']

    # 识别分类特征
    cat_features = []
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].nunique() < 10:
            cat_features.append(col)
            # 将分类特征转换为category类型
            if X[col].dtype == 'object':
                X[col] = pd.Categorical(X[col])

    print(f"识别到的分类特征: {cat_features}")

    return X, y, cat_features


importance_types = {
    'split': '分裂次数',
    'gain': '信息增益'
}

# 3. 模型训练
def train_lightgbm_model(X, y, cat_features):
    """训练LightGBM模型"""
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\n训练集大小: {X_train.shape[0]}")
    print(f"验证集大小: {X_val.shape[0]}")

    # 创建LightGBM数据集
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
    val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_features)

    # 设置模型参数
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.01,
        'num_leaves': 31,
        'max_depth': -1,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'verbose': 0,
        # 增加特征重要性计算的参数
        'importance_type': 'split'  # 默认使用分裂次数计算重要性
    }

    print("\n开始训练LightGBM模型...")
    # 训练模型
    evals_result = {}
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        # evals_result=evals_result,
        # early_stopping_rounds=50,
        # verbose_eval=20
    )

    # 预测
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)

    # 评估模型性能
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)

    print(f"\n模型评估结果:")
    print(f"最佳迭代次数: {model.best_iteration}")
    print(f"验证集 RMSE: {rmse:.6f}")
    print(f"验证集 R²: {r2:.6f}")

    # 特征重要性
    print("\n特征重要性 (基于分裂次数):")
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importance(importance_type='split')
    }).sort_values('Importance', ascending=False)
    print(importance_df)

    return model, X_val, y_val, y_pred, evals_result, importance_df


# 5. 保存模型
def save_model(model, output_dir):
    """保存训练好的模型"""
    model_filename = 'lightgbm_model.txt'
    model.save_model(model_filename)
    print(f"\n模型已保存至: {model_filename}")

    # 保存特征重要性
    importance_df = pd.DataFrame({
        'Feature': model.feature_name(),
        'Importance': model.feature_importance(importance_type='split')
    }).sort_values('Importance', ascending=False)
    importance_df.to_csv('feature_importance.csv', index=False)
    print(f"特征重要性已保存至: feature_importance.csv")


# 5. 特征重要性深度分析
def analyze_feature_importance(model, X, output_dir='feature_importance_results'):
    """
    深度分析特征重要性，包括多种重要性计算方法和可视化

    参数:
    - model: 训练好的LightGBM模型
    - X: 特征数据集
    - output_dir: 输出结果目录

    返回:
    - importance_dict: 包含各种重要性方法结果的字典
    """
    print("\n=== 开始特征重要性深度分析 ===")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    importance_dict = {}

    # 1. 计算多种特征重要性
    importance_types = {
        'split': '分裂次数',
        'gain': '信息增益'
    }

    # 存储所有重要性结果
    all_importance_dfs = []

    for importance_type, description in importance_types.items():
        print(f"\n{description}特征重要性:")
        importance_values = model.feature_importance(importance_type=importance_type)
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            f'{importance_type}_importance': importance_values
        }).sort_values(f'{importance_type}_importance', ascending=False)

        importance_dict[importance_type] = importance_df
        all_importance_dfs.append(importance_df)

        # 打印前10个重要特征
        print(importance_df.head(10).to_string(index=False))

        # 保存为CSV
        importance_df.to_csv(f'{output_dir}/{importance_type}_importance.csv', index=False)
        print(f"{description}重要性已保存至: {output_dir}/{importance_type}_importance.csv")

    # 2. 综合重要性分析
    # 合并不同类型的重要性
    if len(all_importance_dfs) > 1:
        combined_df = all_importance_dfs[0].copy()
        for df in all_importance_dfs[1:]:
            combined_df = pd.merge(combined_df, df, on='Feature')

        # 标准化不同类型的重要性值
        for col in combined_df.columns[1:]:
            max_val = combined_df[col].max()
            if max_val > 0:
                combined_df[f'{col}_normalized'] = combined_df[col] / max_val

        # 计算综合重要性得分
        normalized_cols = [col for col in combined_df.columns if '_normalized' in col]
        combined_df['combined_score'] = combined_df[normalized_cols].mean(axis=1)
        combined_df = combined_df.sort_values('combined_score', ascending=False)

        importance_dict['combined'] = combined_df
        combined_df.to_csv(f'{output_dir}/combined_importance.csv', index=False)
        print(f"\n综合特征重要性分析已保存至: {output_dir}/combined_importance.csv")

    # 3. 可视化特征重要性
    visualize_importance(model, X, importance_dict, output_dir)

    # 4. 特征重要性解释
    generate_importance_insights(importance_dict, output_dir)

    print(f"\n特征重要性分析结果已保存在目录: {output_dir}")
    return importance_dict


def visualize_importance(model, X, importance_dict, output_dir):
    """
    可视化不同类型的特征重要性
    """
    print("\n生成特征重要性可视化...")

    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 分裂次数重要性图
    if 'split' in importance_dict:
        split_df = importance_dict['split']
        plt.figure(figsize=(12, 10))
        sns.barplot(x='split_importance', y='Feature', data=split_df.head(15))
        plt.title('基于分裂次数的特征重要性 (前15个)', fontsize=16)
        plt.xlabel('重要性得分', fontsize=14)
        plt.ylabel('特征名称', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/split_importance.png', dpi=300, bbox_inches='tight')

    # 2. 信息增益重要性图
    if 'gain' in importance_dict:
        gain_df = importance_dict['gain']
        plt.figure(figsize=(12, 10))
        sns.barplot(x='gain_importance', y='Feature', data=gain_df.head(15))
        plt.title('基于信息增益的特征重要性 (前15个)', fontsize=16)
        plt.xlabel('重要性得分', fontsize=14)
        plt.ylabel('特征名称', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/gain_importance.png', dpi=300, bbox_inches='tight')

    # 3. 综合重要性雷达图
    if 'combined' in importance_dict:
        combined_df = importance_dict['combined']
        top_features = combined_df.head(8)

        # 雷达图数据准备
        features = top_features['Feature'].tolist()
        split_values = []
        gain_values = []

        if 'split_importance_normalized' in top_features.columns:
            split_values = top_features['split_importance_normalized'].tolist()
        if 'gain_importance_normalized' in top_features.columns:
            gain_values = top_features['gain_importance_normalized'].tolist()

        # 绘制雷达图
        if split_values and gain_values:
            categories = features
            N = len(categories)

            # 角度设置
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # 闭合雷达图

            # 创建图形
            plt.figure(figsize=(12, 10))
            ax = plt.subplot(111, polar=True)

            # 绘制数据
            ax.plot(angles, split_values + split_values[:1], 'o-', linewidth=2, label='分裂次数重要性')
            ax.plot(angles, gain_values + gain_values[:1], 'o-', linewidth=2, label='信息增益重要性')

            # 设置角度标签
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=12, rotation=0, ha='right')

            # 设置图例和标题
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            plt.title('特征重要性雷达图 (前8个特征)', fontsize=16, pad=20)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/radar_importance.png', dpi=300, bbox_inches='tight')

    # 4. 重要性分布箱线图
    plt.figure(figsize=(12, 6))
    all_importance_scores = []
    labels = []

    for importance_type, df in importance_dict.items():
        if importance_type != 'combined':
            col_name = f'{importance_type}_importance'
            if col_name in df.columns:
                all_importance_scores.append(df[col_name].values)
                labels.append(importance_types.get(importance_type, importance_type))

    if all_importance_scores:
        sns.boxplot(data=all_importance_scores)
        plt.title('不同重要性计算方法的分布', fontsize=16)
        plt.xticks(range(len(labels)), labels, fontsize=12)
        plt.ylabel('重要性得分', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/importance_distribution.png', dpi=300, bbox_inches='tight')


def generate_importance_insights(importance_dict, output_dir):
    """
    生成特征重要性的分析洞察
    """
    print("\n生成特征重要性洞察...")

    insights = []
    insights.append("# 特征重要性分析报告\n")
    insights.append("## 1. 总体分析\n")

    # 分析各个特征重要性方法
    if 'split' in importance_dict:
        split_df = importance_dict['split']
        top_split = split_df.iloc[0]['Feature']
        top_split_score = split_df.iloc[0]['split_importance']
        insights.append(f"- 基于分裂次数的最重要特征: {top_split} (得分: {top_split_score:.2f})")

        # 计算重要性分布统计
        split_stats = split_df['split_importance'].describe()
        insights.append(f"- 分裂次数重要性分布: 均值={split_stats['mean']:.2f}, 标准差={split_stats['std']:.2f}")

    if 'gain' in importance_dict:
        gain_df = importance_dict['gain']
        top_gain = gain_df.iloc[0]['Feature']
        top_gain_score = gain_df.iloc[0]['gain_importance']
        insights.append(f"- 基于信息增益的最重要特征: {top_gain} (得分: {top_gain_score:.2f})")

        gain_stats = gain_df['gain_importance'].describe()
        insights.append(f"- 信息增益重要性分布: 均值={gain_stats['mean']:.2f}, 标准差={gain_stats['std']:.2f}")

    # 分析特征一致性
    if 'split' in importance_dict and 'gain' in importance_dict:
        split_top10 = set(importance_dict['split'].head(10)['Feature'])
        gain_top10 = set(importance_dict['gain'].head(10)['Feature'])
        common_features = split_top10.intersection(gain_top10)

        insights.append(f"\n## 2. 特征重要性一致性分析\n")
        insights.append(f"- 两种方法共同认为重要的特征 (前10个中): {len(common_features)}个")
        if common_features:
            insights.append(f"  共同重要特征: {', '.join(sorted(common_features))}")

        # 计算排名相关性
        split_ranks = importance_dict['split'].set_index('Feature').index
        gain_ranks = importance_dict['gain'].set_index('Feature').index

        # 创建排名映射
        split_rank_dict = {feature: i + 1 for i, feature in enumerate(split_ranks)}
        gain_rank_dict = {feature: i + 1 for i, feature in enumerate(gain_ranks)}

        # 计算Spearman相关系数
        common_features_list = list(split_rank_dict.keys() & gain_rank_dict.keys())
        split_rank_values = [split_rank_dict[f] for f in common_features_list]
        gain_rank_values = [gain_rank_dict[f] for f in common_features_list]

        correlation = np.corrcoef(split_rank_values, gain_rank_values)[0, 1]
        insights.append(f"- 特征重要性排名相关性 (Spearman): {correlation:.4f}")

        # 解释相关性强度
        if abs(correlation) > 0.8:
            insights.append("  相关性很强，两种方法对特征重要性的判断高度一致")
        elif abs(correlation) > 0.5:
            insights.append("  相关性中等，两种方法对特征重要性的判断有一定一致性")
        else:
            insights.append("  相关性较弱，两种方法对特征重要性的判断存在差异")

    # 分析不重要的特征
    insights.append("\n## 3. 低重要性特征分析\n")
    for importance_type, df in importance_dict.items():
        if importance_type != 'combined':
            col_name = f'{importance_type}_importance'
            if col_name in df.columns:
                zero_importance = df[df[col_name] == 0]
                if len(zero_importance) > 0:
                    insights.append(
                        f"- 基于{importance_types.get(importance_type, importance_type)}方法，{len(zero_importance)}个特征重要性为0")
                    if len(zero_importance) <= 10:
                        insights.append(f"  这些特征: {', '.join(zero_importance['Feature'].tolist())}")
                    else:
                        insights.append(f"  前5个: {', '.join(zero_importance['Feature'].head().tolist())} 等")

                # 识别重要性极低的特征
                threshold = df[col_name].quantile(0.25)
                low_importance = df[df[col_name] <= threshold]
                insights.append(
                    f"- 基于{importance_types.get(importance_type, importance_type)}方法，{len(low_importance)}个特征重要性位于底部25%")

    # 给出特征工程建议
    insights.append("\n## 4. 特征工程建议\n")

    # 高重要性特征处理建议
    if 'combined' in importance_dict:
        top_features = importance_dict['combined'].head(5)['Feature'].tolist()
        insights.append(f"- 对于高重要性特征 ({', '.join(top_features)})，可以考虑:")
        insights.append("  - 创建特征交互项")
        insights.append("  - 进行特征分箱或离散化")
        insights.append("  - 检查这些特征的缺失值和异常值")

    # 低重要性特征处理建议
    insights.append("- 对于低重要性特征，可以考虑:")
    insights.append("  - 在后续模型中删除这些特征以减少过拟合风险")
    insights.append("  - 对这些特征进行更深入的特征工程")
    insights.append("  - 考虑这些特征是否与其他特征高度相关")

    # 保存洞察报告
    with open(f'{output_dir}/importance_insights.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(insights))

    print(f"特征重要性洞察报告已保存至: {output_dir}/importance_insights.md")


# 主函数
def main():
    print("=== LightGBM 模型训练流程 ===")

    # 数据文件路径
    data_path = '/Volumes/zhitai2/git/python/project/tensorflow-learning/examples/s5e10/playground-series-s5e10/train.csv'

    # 1. 加载和探索数据
    df = load_and_explore_data(data_path)

    # 2. 数据预处理
    X, y, cat_features = preprocess_data(df)

    # 3. 训练模型
    model, X_val, y_val, y_pred, evals_result, importance_df = train_lightgbm_model(X, y, cat_features)

    # 4. 可视化结果
    # visualize_results(model, evals_result, importance_df, X_val, y_val, y_pred)

    # 5. 深度特征重要性分析
    # analyze_feature_importance(model, X)

    # 5. 保存模型
    output_dir = 'lgbm_results_' + datetime.now().strftime('%Y%m%d_%H%M%S')
    save_model(model, output_dir)

    print("\n=== 训练完成 ===")


if __name__ == "__main__":
    main()


'''
重要性 分析
Feature,Importance
curvature,11440
weather,4489
num_reported_accidents,3650
speed_limit,2677
lighting,2482
num_lanes,1163
holiday,1054
public_road,938
time_of_day,874
road_type,702
road_signs_present,400
school_season,131

weather+light


- 1.
id : 数据记录的唯一标识符
- 2.
road_type : 道路类型

- urban: 城市道路
- rural: 乡村道路
- highway: 高速公路
- 3.
num_lanes : 车道数量，整数类型（1-4）
- 4.
curvature : 道路曲率，表示道路的弯曲程度，数值范围约为0-1
- 5.
speed_limit : 限速值，单位可能是km/h或mph
- 6.
lighting : 光照条件

- daylight: 日光充足
- dim: 光线昏暗
- night: 夜晚
- 7.
weather : 天气条件

- rainy: 雨天
- clear: 晴天
- foggy: 雾天
- 8.
road_signs_present : 是否有道路标志（布尔值）
- 9.
public_road : 是否为公共道路（布尔值）
- 10.
time_of_day : 一天中的时间段

- morning: 早晨
- afternoon: 下午
- evening: 晚上
- 11.
holiday : 是否为假日（布尔值）
- 12.
school_season : 是否为学校开学季节（布尔值）
- 13.
num_reported_accidents : 报告的事故数量，整数类型
- 14.
accident_risk : 事故风险评分，这是目标变量，数值范围约为0-0.84
'''