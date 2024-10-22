import numpy as np
import pandas as pd


def readDate() :
    #1.读取文件
    train_data = pd.read_csv("C:\Program Files\BusinessFile\新手村\AI培训资料\数据集\\titanic\\train.csv")
    print(train_data.head())
    test_data = pd.read_csv("C:\Program Files\BusinessFile\新手村\AI培训资料\数据集\\titanic\\test.csv")
    print(test_data.head())
    print(train_data.shape)
    #2.按numpy整理数据


def main():
    readDate()

if __name__ == "__main__":
    main()