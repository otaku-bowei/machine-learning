import string

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.python.summary.writer import writer
import tensorflow as tf
import tensorboard as tb

'''
某些Tabular Data数据中，nan值有可能对结果的影响很大，哪怕用one-hot的方式进行标注归类，模型对这些值的处理仍然有可能很鸡肋，如果要填补nan值，最好对相关列的nan值进行分析
'''
# nan_compare 数据空值对比，data--数据，type_field--是否类型字段,analysis_field--分析字段, value_field--值字段
def nan_compare(data_prg: pd.DataFrame, type_field: bool, analysis_field, value_field: string):
    data = data_prg.loc[:, [analysis_field, value_field]]
    # 分析字段是类型，用折线图显示，其中折线图的横坐标为结果值，按区间范围分10段，纵坐标为每个区间，该类型的个数
    if type_field:
        data = data.fillna('Nan')
        bins = np.linspace(data[value_field].min(), data[value_field].max(), 11)
        bin_labels = [f'({bins[i]}-{bins[i + 1]})' for i in range(len(bins) - 1)]
        # 使用pandas的cut函数将结果值分段
        data['bin'] = pd.cut(data[value_field], bins=bins, labels=bin_labels)
        # 分组统计每个区间中每种类型的数量
        grouped = data.groupby(['bin', analysis_field]).size().unstack(fill_value=0)
        print_dp(grouped)
        # 绘制折线图
        plt.figure(figsize=(10, 6))
        for type_ in grouped.columns:
            plt.plot(grouped.index, grouped[type_], marker='o', label=type_)
        # 设置图表标题和坐标轴标签
        plt.title('Type Counts by Result Value Interval')
        plt.xlabel(value_field)
        plt.ylabel(analysis_field)
        plt.xticks(rotation=45)  # 旋转x轴标签以便更好地显示
        plt.legend()  # 显示图例
        plt.grid(True)  # 显示网格线
        plt.tight_layout()  # 调整布局以避免标签重叠
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper center')
        plt.show()
    else:
        max = data[analysis_field].max() + 1
        data = float_field_to_range_type(data, analysis_field, 10, max)
        bins = np.linspace(data[value_field].min(), data[value_field].max(), 11)
        bin_labels = [f'({bins[i]}-{bins[i + 1]})' for i in range(len(bins) - 1)]
        # 使用pandas的cut函数将结果值分段
        data['bin'] = pd.cut(data[value_field], bins=bins, labels=bin_labels)
        # 分组统计每个区间中每种类型的数量
        grouped = data.groupby(['bin', 'interval']).size().unstack(fill_value=0)
        print_dp(grouped)
        # 绘制折线图
        # 设置图表标题和坐标轴标签
        for type_ in grouped.columns:
            plt.plot(grouped.index, grouped[type_], marker='o', label=type_)
        plt.xlabel(value_field)
        plt.ylabel(analysis_field)
        plt.xticks(rotation=45)  # 旋转x轴标签以便更好地显示
        plt.legend()  # 显示图例
        plt.grid(True)  # 显示网格线
        plt.tight_layout()  # 调整布局以避免标签重叠
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper center')
        plt.show()


def float_field_to_range_type(data: pd.DataFrame, value_field: string, num: int, nan_value: float):
    bins = np.linspace(data[value_field].min(), data[value_field].max(), num)
    bins = np.append(bins, nan_value + 1)
    bin_labels = [f'({bins[i]}-{bins[i + 1]})' for i in range(len(bins) - 1)]
    data.fillna(nan_value + 0.5, inplace=True)
    # print_dp(data, 30)
    data['interval'] = pd.cut(data[value_field], bins=bins, labels=bin_labels)
    return data



def save_tensorboard(path: string):
    with writer.as_default():
        tf.summary.image("Sine Wave Plot", ["sine_wave.png"], step=0)
    # 启动TensorBoard以查看结果
    ttb = tb.TensorBoard()
    ttb.configure(argv=[None, "--logdir", './'])
    url = ttb.launch()
    print("TensorBoard at %s" % url)



def print_dp(data: pd.DataFrame, num: int = 5):
    print("\r\n")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(data.head(num))