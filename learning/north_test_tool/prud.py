import pandas as pd
import json


def parse_json_file(file_path):
    try:
        # 打开并读取JSON文件
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # 提取deviceInfo中的SN和OUI
        sn = data['deviceInfo']['SN']
        oui = data['deviceInfo']['OUI']

        # 提取dataModels列表中的instance值
        instances = [item['instance'] for item in data['dataModels']]
        values = [item['value'] for item in data['dataModels']]
        functionCodes = [item['functionCode'] for item in data['dataModels']]

        # 创建一个包含所需列的DataFrame
        df = pd.DataFrame({
            'SN': [sn] * len(instances),
            'OUI': [oui] * len(instances),
            'instance': instances,
            'value': values,
            'functionCode': functionCodes,
        })

        return df
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
    except Exception as e:
        print(f"解析文件时出错: {e}")

# 设置pandas显示选项
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', None)  # 不限制显示宽度
pd.set_option('display.max_colwidth', None)  # 不限制列的显示宽度

file_path = 'resource/20AC9C-CTBU022J120070-NR-20250606103000.txt'
all_function_code_path = 'resource/all_function_code.txt'
# dirs = ['2025051207']
def gen_file():
    df = pd.DataFrame()
    # 指定文件夹路径
    # 解析这个文件
    result_df = parse_json_file(file_path)
    df = pd.concat([df, result_df])
    df.to_csv('dataModel.csv', index=False)
    print("输出文件:test.csv")

# gen_file()

def process_config_file():
    """
    读取配置文件并生成去重且无空行的字符串数组
    :param file_path: 配置文件绝对路径
    :return: 处理后的字符串列表（无空行、无重复）
    """
    unique_lines = {}  # 使用字典保持顺序并去重（Python 3.7+支持有序字典）

    with open(all_function_code_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 去除首尾空白字符（包括换行符）
            processed_line = line.strip()

            # 跳过空行
            if not processed_line:
                continue

            # 去重（字典键自动去重）
            unique_lines[processed_line] = None  # 值用None占位

    # 转换字典键为列表（保持文件中的原始顺序）
    return list(unique_lines.keys())

# all_function_codes = process_config_file()

# data_model = pd.read_csv('resource/dataModel.csv')
# print(all_function_codes)
# existing_codes = set(data_model['functionCode'].unique())
# missing_codes = [code for code in all_function_codes if code not in existing_codes]
# print(missing_codes)
# data_model = data_model.fillna(0)
# data_model.to_csv('dataModel.csv', index=False)

performance_param_path = 'performance/all_east_param.csv'
def gen_performance_data():
    all_param = pd.read_csv(performance_param_path)
    all_param['value'].to_csv('performance/all_east_value.csv', index=False)
    print(all_param['value'])
    all_param['param'].to_csv('performance/all_east_param2.csv', index=False)
    print(all_param['param'])

gen_performance_data()