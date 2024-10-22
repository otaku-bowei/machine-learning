import pandas as pd
import numpy

# 19种分类
def TrainNlp():
    train_data = numpy.loadtxt('C:\Program Files\BusinessFile\新手村\AI培训资料\数据集\达观杯nlp\\train_set.csv')
    train_data = train_data[1:]


def main():
    print("hello")
    TrainNlp()


if __name__ == "__main__":
    main()
