import string

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

'''
某些Tabular Data数据中，nan值有可能对结果的影响很大，哪怕用one-hot的方式进行标注归类，模型对这些值的处理仍然有可能很鸡肋，如果要填补nan值，最好对相关列的nan值进行分析
'''
# nan_compare 数据空值对比，data--数据，type_field--是否类型字段,analysis_field--分析字段, value_field--值字段
def nan_compare(data: pd.DataFrame, type_field: bool, analysis_field, value_field: string):
    data = data.loc[:, [analysis_field, value_field]]
    nan_data = data[data[analysis_field].isnull()]
    not_nan_data = data[data[analysis_field].notnull()]
    print(nan_data)
    print("\r\n")
    print(not_nan_data)
    # 分析字段是类型，用折线图显示，其中折线图的横坐标为结果值，按区间范围分10段，纵坐标为每个区间，该类型的个数
    if type_field:
        bins = np.linspace(data[value_field].min(), data[value_field].max(), 11)
        bin_labels = [f'({bins[i]}-{bins[i + 1]})' for i in range(len(bins) - 1)]
        # 使用pandas的cut函数将结果值分段
        data[analysis_field] = pd.cut(data[value_field], bins=bins, labels=bin_labels)
        # 计算每个区间中每种类型的数量
        type_counts = data.groupby([analysis_field, 'type']).size().unstack(fill_value=0)
        # 绘制折线图
        plt.figure(figsize=(10, 6))
        for type_ in type_counts.columns:
            plt.plot(type_counts.index, type_counts[type_], marker='o', label=type_)
        # 设置图表标题和坐标轴标签
        plt.title('Type Counts by Result Value Interval')
        plt.xlabel('Result Value Interval')
        plt.ylabel('Type Count')
        plt.xticks(rotation=45)  # 旋转x轴标签以便更好地显示
        plt.legend()  # 显示图例
        plt.grid(True)  # 显示网格线
        plt.tight_layout()  # 调整布局以避免标签重叠
        plt.show()
