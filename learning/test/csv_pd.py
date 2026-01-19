import pandas as pd

def convert_txt_to_excel(txt_path, excel_path):
    # 读取txt文件（假设文件编码为UTF-8）
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 解析每一行数据（按竖线分割）
    data = []
    for line in lines:
        # 去除换行符并按|分割（处理可能的末尾空列）
        parts = line.strip().split('|')
        # 补全到4列（根据实际数据调整列数）
        while len(parts) < 4:
            parts.append('')
        data.append(parts)

    # 创建DataFrame并设置列名（根据实际数据含义调整）
    df = pd.DataFrame(
        data,
        columns=['区域编码', '区域路径', '区域名称', '备用字段']
    )

    # 写入Excel（使用openpyxl支持xlsx格式）
    df.to_excel(excel_path, index=False, engine='openpyxl')
    print(f"转换完成，文件保存至：{excel_path}")


input_txt = r'c:\Program Files\BusinessFile\java_project\northbound-adapter\src\main\resources\config\area_code.txt'
output_excel = r'c:\Program Files\BusinessFile\java_project\northbound-adapter\src\main\resources\config\area_code.xlsx'

# 执行转换
convert_txt_to_excel(input_txt, output_excel)