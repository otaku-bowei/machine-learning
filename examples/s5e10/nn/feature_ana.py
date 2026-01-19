import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from examples.s5e10.nn.contant import Data
from examples.s5e10.nn.data import read_org_data


def analyze_feature_impact(importance_file, train_file, output_dir=None):
    """
    根据特征重要性分析各特征对accident_risk的影响

    参数:
        importance_file: 特征重要性CSV文件路径
        train_file: 训练数据CSV文件路径
        output_dir: 分析结果输出目录（可选）

    返回:
        字典，包含分析结果和洞察
    """
    print("开始特征影响分析...")

    # 读取特征重要性数据
    importance_df = pd.read_csv(importance_file)
    print(f"特征重要性数据加载完成，共{len(importance_df)}个特征")

    # 读取训练数据
    train_df = pd.read_csv(train_file)
    print(f"训练数据加载完成，共{len(train_df)}条记录")

    # 确保目标变量存在
    if 'accident_risk' not in train_df.columns:
        raise ValueError("训练数据中不存在'accident_risk'列")

    # 合并并按重要性排序
    all_features = set(importance_df['Feature'].tolist())
    common_features = all_features.intersection(set(train_df.columns))

    if 'id' in train_df.columns:
        train_df = train_df.drop('id', axis=1)

    # 按重要性排序特征
    importance_df = importance_df.sort_values('Importance', ascending=False)

    # 存储分析结果
    analysis_results = {
        'importance_ranking': importance_df.to_dict('records'),
        'feature_impacts': {},
        'key_insights': []
    }

    # 分析每个特征对accident_risk的影响
    print("\n开始详细特征分析:")

    for _, row in importance_df.iterrows():
        feature = row['Feature']
        importance = row['Importance']

        if feature not in train_df.columns:
            print(f"跳过特征 {feature}: 不在训练数据中")
            continue

        print(f"\n分析特征: {feature} (重要性: {importance})")
        feature_data = train_df[feature]
        target_data = train_df['accident_risk']

        # 初始化该特征的分析结果
        feature_analysis = {
            'importance': importance,
            'summary_stats': {},
            'correlation': None,
            'impact_description': '',
            'visualization_data': {}
        }

        # 分析数值型特征
        if pd.api.types.is_numeric_dtype(feature_data):
            # 计算相关系数
            corr = feature_data.corr(target_data)
            feature_analysis['correlation'] = corr

            # 计算统计信息
            feature_analysis['summary_stats'] = {
                'mean': feature_data.mean(),
                'std': feature_data.std(),
                'min': feature_data.min(),
                'max': feature_data.max(),
                'correlation_with_risk': corr
            }

            # 分析影响方向
            if abs(corr) > 0.1:
                direction = "正相关" if corr > 0 else "负相关"
                strength = "强" if abs(corr) > 0.3 else "中等" if abs(corr) > 0.1 else "弱"
                feature_analysis['impact_description'] = f"与事故风险呈{strength}{direction}关系"
                analysis_results['key_insights'].append(
                    f"{feature}与事故风险呈{strength}{direction}关系 (相关系数: {corr:.3f})")
            else:
                feature_analysis['impact_description'] = "与事故风险相关性较弱"

            # 可视化数据准备
            feature_analysis['visualization_data'] = {
                'feature_values': feature_data.tolist()[:1000],  # 限制数据量
                'risk_values': target_data.tolist()[:1000]
            }

        # 分析分类特征
        else:
            # 计算每个类别的平均风险
            category_risks = train_df.groupby(feature)['accident_risk'].agg(['mean', 'std', 'count']).reset_index()
            category_risks = category_risks.sort_values('mean', ascending=False)

            feature_analysis['summary_stats'] = {
                'categories': category_risks.to_dict('records'),
                'category_count': len(category_risks)
            }

            # 找出高风险类别
            if len(category_risks) > 1:
                max_risk = category_risks['mean'].max()
                min_risk = category_risks['mean'].min()
                risk_diff = max_risk - min_risk

                if risk_diff > 0.1:  # 风险差异显著
                    high_risk_cats = category_risks[category_risks['mean'] > (min_risk + risk_diff / 2)][
                        feature].tolist()
                    feature_analysis['impact_description'] = f"高风险类别: {high_risk_cats}"
                    analysis_results['key_insights'].append(
                        f"{feature}特征中，{', '.join(map(str, high_risk_cats))}类别的平均事故风险较高")
                else:
                    feature_analysis['impact_description'] = "不同类别间风险差异较小"

            feature_analysis['visualization_data'] = category_risks.to_dict('records')

        analysis_results['feature_impacts'][feature] = feature_analysis

    # 生成综合洞察
    print("\n生成综合洞察...")

    # 识别最重要的特征
    top_features = importance_df['Feature'].tolist()[:5]
    analysis_results['key_insights'].insert(0, f"最重要的5个特征: {', '.join(top_features)}")

    # 分析特征重要性分布
    importance_mean = importance_df['Importance'].mean()
    importance_std = importance_df['Importance'].std()
    high_importance_count = len(importance_df[importance_df['Importance'] > (importance_mean + importance_std)])

    analysis_results['key_insights'].append(
        f"高度重要特征({importance_mean + importance_std:.1f}以上): {high_importance_count}个")

    # 保存分析结果
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)

        # 保存分析报告
        report_path = os.path.join(output_dir, 'feature_impact_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("特征影响分析报告\n")
            f.write("=" * 50 + "\n\n")

            f.write("一、特征重要性排序\n")
            for i, row in importance_df.iterrows():
                f.write(f"{i + 1}. {row['Feature']}: {row['Importance']}\n")

            f.write("\n二、关键洞察\n")
            for insight in analysis_results['key_insights']:
                f.write(f"- {insight}\n")

            f.write("\n三、详细特征分析\n")
            for feature, analysis in analysis_results['feature_impacts'].items():
                f.write(f"\n{feature} (重要性: {analysis['importance']}):\n")
                f.write(f"  影响描述: {analysis['impact_description']}\n")
                if 'correlation' in analysis and analysis['correlation'] is not None:
                    f.write(f"  相关系数: {analysis['correlation']:.3f}\n")

        print(f"分析报告已保存至: {report_path}")

        # 生成可视化图表
        plt.figure(figsize=(15, 20))

        # 1. 特征重要性条形图
        plt.subplot(3, 1, 1)
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('特征重要性排序')
        plt.xlabel('重要性得分')
        plt.ylabel('特征')

        # 2. 最重要数值特征与风险的关系（如果存在）
        numeric_features = [f for f in top_features if
                            f in train_df.columns and pd.api.types.is_numeric_dtype(train_df[f])]
        if numeric_features:
            plt.subplot(3, 1, 2)
            sns.scatterplot(x=train_df[numeric_features[0]][:1000], y=train_df['accident_risk'][:1000], alpha=0.5)
            plt.title(f'{numeric_features[0]}与事故风险的关系 (前1000个样本)')
            plt.xlabel(numeric_features[0])
            plt.ylabel('accident_risk')

        # 3. 分类特征的风险分布（如果存在）
        categorical_features = [f for f in top_features if
                                f in train_df.columns and not pd.api.types.is_numeric_dtype(train_df[f])]
        if categorical_features:
            plt.subplot(3, 1, 3)
            category_risks = train_df.groupby(categorical_features[0])['accident_risk'].mean().reset_index()
            sns.barplot(x=categorical_features[0], y='accident_risk', data=category_risks)
            plt.title(f'{categorical_features[0]}各类别的平均事故风险')
            plt.xlabel(categorical_features[0])
            plt.ylabel('平均accident_risk')
            plt.xticks(rotation=45)

        plt.tight_layout()
        viz_path = os.path.join(output_dir, 'feature_impact_visualization.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"可视化图表已保存至: {viz_path}")

    print("\n特征影响分析完成！")
    print("\n关键洞察:")
    for i, insight in enumerate(analysis_results['key_insights'], 1):
        print(f"{i}. {insight}")

    return analysis_results


def ana_feature(feature_field, label_field: str, df: pd.DataFrame, fenlei: bool, output_dir='./analysis_results',):
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置matplotlib字体支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 使用英文文件名避免编码问题，但图表标题仍使用中文
    title = f'{feature_field}与{label_field}的关系'
    safe_filename = f'{feature_field}_vs_{label_field}_{"categorical" if fenlei else "scatter"}'
    
    plt.figure(figsize=(10, 6))  # 创建新的图形
    plt.title(title)
    if fenlei:
        sns.barplot(data=df, x='id', y=label_field, hue=feature_field)
    else:
        sns.scatterplot(data=df, x=feature_field, y=label_field,)
    
    # 保存图片
    save_path = os.path.join(output_dir, f'{safe_filename}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存至: {save_path}")
    
    # 关闭当前图形以释放内存
    plt.close()


# 按分类特征分组的统计分析
def analyze_categorical_effect(df, categorical_feature, target):
    # 基本统计
    stats = df.groupby(categorical_feature)[target].agg(['mean', 'median', 'std', 'count'])

    # 可视化
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=categorical_feature, y=target, data=df)
    plt.title(f'{categorical_feature} 对 {target} 的影响')
    plt.xticks(rotation=45)
    plt.tight_layout()

    return stats


'''
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
'''
importance_data = pd.read_csv('feature_importance.csv')
importance_feature = importance_data['Feature']
train_data, _ = read_org_data()
#
# for imf in importance_feature:
#     if imf not in Data.get('type_feature'):
#         ana_feature(imf, Data.get('label_field'), train_data, True if imf in Data.get('type_feature') else False)

analyze_categorical_effect(train_data, Data.get('type_feature'), Data.get('label_field'))

'''
1.curvature大于0.5时会明显增加风险
2.
'''

# 使用示例
# if __name__ == "__main__":
#     # 文件路径
#     importance_file = "./feature_importance.csv"
#     train_file = "../playground-series-s5e10/train.csv"
#     output_dir = "./analysis_results"
#
#     # 运行分析
#     results = analyze_feature_impact(importance_file, train_file, output_dir)