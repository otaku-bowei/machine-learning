# 定义文件路径
file_a_path = 'performance_type_config_v4.txt'
file_b_path = 'performance_type_config_v5.txt'
# file_a_path = 'ums.txt'
# file_b_path = 'north.txt'
fields = ['QOS.InitialEstabReq','QOS.InitialEstabSucc','QOS.EstabReq','QOS.EstabSucc','QOS.ModifyReq','QOS.ModifySucc','QOS.Release','QOS.NormalRelease','QOS.ReleasebyAMF','QOS.ReleasebygNB','QOS.RelNormalbygNB','VoNR.UserMean','VoNR.UserMax','QOS.CauseOfEstabFail','VoNR.ConnReq','VoNR.ConnSucc','QOS.CauseOfAbnormRelease','UE.CauseOfAbnormContextReleasebygNB','UE.CauseOfContextReleasebyAMF','VoNR.ReleaseSum','VoNR.NormalRelease','IMS.ReleaseSum','IMS.NormalRelease','VoNR.Time']

# 读取文件 A 中的行（保留原始内容，仅去除首尾空格）
with open(file_a_path, 'r') as file_a:
    lines_a = [line.strip() for line in file_a.readlines()]  # 改为直接存储处理后的行字符串

# 读取文件 B 中的行（保留原始内容，仅去除首尾空格）
with open(file_b_path, 'r') as file_b:
    lines_b = [line.strip() for line in file_b.readlines()]  # 改为直接存储处理后的行字符串

# 提取每行第一个'|'前的前缀（无'|'则取整行）
prefixes_a = [line.split('|', 1)[0] for line in lines_a]  # 新增：生成A文件前缀列表
prefixes_b = [line.split('|', 1)[0] for line in lines_b]  # 新增：生成B文件前缀列表

# 找出文件 A 中比文件 B 多余的行（基于前缀比较）
extra_lines = [line for line, prefix in zip(lines_a, prefixes_a) if prefix not in prefixes_b]  # 修改比较逻辑

# 输出结果（仅打印fields数组中包含的字段对应的行）
for extra_line in extra_lines:
    current_prefix = extra_line.split('|', 1)[0]  # 提取当前行的前缀
    if current_prefix in fields:  # 检查前缀是否在目标字段列表中
        print(extra_line)