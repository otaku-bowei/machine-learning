import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def test():
    one_hot()
    one_hot_numpy()


# one-hot向量转换为0-1编码
def one_hot_numpy():
    # 创建一个示例DataFrame
    data = {'Category': ['A', 'B', 'A', 'C', 'B', 'A']}
    df = pd.DataFrame(data)

    # 创建一个OneHotEncoder对象
    encoder = OneHotEncoder(sparse_output=False)  # 设置sparse=False以返回密集数组

    # 对'Category'列进行独热编码
    # 首先需要将DataFrame的列转换为二维数组，并且只选择需要编码的列
    one_hot_encoded_array = encoder.fit_transform(df[['Category']])

    # 将编码后的数组转换回DataFrame，并设置列名
    one_hot_encoded_df = pd.DataFrame(one_hot_encoded_array, columns=encoder.get_feature_names_out(['Category']))
    print("\r\n")
    print(one_hot_encoded_df)


def one_hot():
    # 创建一个示例DataFrame
    data = {'Category': ['A', 'B', 'A', 'C', 'B', 'A']}
    df = pd.DataFrame(data)

    # 对'Category'列进行独热编码
    one_hot_encoded_df = pd.get_dummies(df, columns=['Category'])
    print("\r\n")
    print(one_hot_encoded_df)