# 多元线性回归,预测各个分数
import numpy as np
import pandas as pd
import tool.matplotlib.draw_matplotlib as tm


# read_data 读取数据
def read_data() -> pd.DataFrame:
    df = pd.read_csv('StudentsPerformance.csv')
    print(df.head())
    print(df.describe())
    df = df.loc[:, ['gender(female-1,male-0)', 'test preparation course', 'reading score', 'writing score']]
    return df


# main 主函数
def main():
    read_data()


if __name__ == "__main__":
    main()
