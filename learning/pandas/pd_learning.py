import pandas as pd

def test():
    #0.读取文件
    train_data = pd.read_csv("C:\Program Files\BusinessFile\新手村\AI培训资料\数据集\\titanic\\train.csv")
    print("\n")
    #1.读取数据
    print(train_data.head())
    print("\n")
    print(train_data.shape)
    print("\n")
    #2.数据筛选
    print(train_data['Sex'][0])
    print("\n")
    print(train_data.iloc[:, 0])
    print("\n")
    print(train_data.iloc[1:2, :])
    print("\n")
    print(train_data.iloc[[1,2,3], :])
    print("\n")
    print(train_data.loc[:, ['Sex', 'Fare']])
    print("\n")
    print(train_data.loc[train_data['Sex'] == 'male'])
    print("\n")
    print(train_data.loc[(train_data['Sex'] == 'male') & (train_data.PassengerId > 500) | (train_data.Embarked.isin(['S']))])
    #3.数据赋值
    print("\n")
    train_data.loc[train_data['Sex'] == 'male'] = 1
    print(train_data.loc[train_data['Sex'] == 'female', ['Sex']])
    print("\n")
    print(train_data.loc[train_data['Sex'] == 1, ['Sex']])
    #4.提取摘要、汇聚信息
    print("\n")
    print(train_data.Sex.describe())
    print("\n")
    print(train_data.PassengerId.mean())
    print("\n")
    print(train_data.Sex.unique())
    print("\n")
    print(train_data.Sex.value_counts())
    #5.映射
    print("\n")
    pv_age = train_data.Age.mean()
    train_data.Age.map(lambda a:pv_age)
    print(train_data.loc[train_data.Age == ''])
    #6.运算
    train_data.PassengerId + 1
    print("\n")
    print(train_data.head(2))