file_name = '20AC9C-CELL-10.244.232.227-1.0-20250911112204-20250911110000-5.csv'

import pandas as pd
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', None)  # 不限制显示宽度
pd.set_option('display.max_colwidth', None)  # 不限制列的显示宽度
data = pd.read_csv(file_name, sep="\\|", dtype=str)
data = data[data['RANK1_DlRp_Num'].notna()]
data = data.loc[:, ['RANK2_DlRp_Num']]
print(data.head(1000))
# data.to_csv('data.csv', index=False, sep="|")