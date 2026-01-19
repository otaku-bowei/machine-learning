import datetime


def timestamp_to_time(timestamp):
    """
    将时间戳转换为指定格式的时间字符串
    :param timestamp: 输入的时间戳（秒）
    :return: 格式化后的时间字符串
    """
    timestamp = timestamp / 1000000000
    # 将时间戳转换为 datetime 对象
    dt_object = datetime.datetime.fromtimestamp(timestamp)
    # 格式化时间，这里的格式可以根据你的需求修改
    formatted_time = dt_object.strftime('%Y-%m-%d %H:%M:%S')
    return formatted_time


# 示例时间戳列表
# timestamps = [1609430400000000000, 1742659388000000000, ]
# for timestamp in timestamps:
#     result = timestamp_to_time(timestamp)
#     print(f"时间戳 {timestamp} 转换后的时间是: {result}")

import random
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


def generate_random_time(date, millisecond_suffix):
    """生成指定日期的随机时间（12:00-17:00）"""
    # 生成随机小时（12-16，因为17:00属于区间上限）
    hour = random.randint(12, 16)
    # 生成随机分钟和秒（0-59）
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    # 组合时间字符串（保留原始毫秒后缀：.646或.317）
    return date.strftime(f"%Y-%m-%dT{hour:02d}:{minute:02d}:{second:02d}{millisecond_suffix}+08:00")


def generate_logs(start_date_str, end_date_str, output_path):
    # 原始日志模板（UAA和RAN-UMS两种类型）
    uaa_template = """[{date_time}]--[UAA]--[ACS_OPERATION]--[RECORD]--"""
    uaa_log = """{"args":["httpServletRequest=org.apache.shiro.web.servlet.ShiroHttpServletRequest@12a1ab8b...","authReqVO={\\"userAccountName\\":\\"zhejiang\\",\\"userAccountPassword\\":\\"H+quOtVdJFTOU5dYQbuHdA==\\",\\"validateCode\\":\\"vm92\\",\\"validateCodeKey\\":\\"262cce3a-32cd-4969-9bd0-1a646c9eae87\\"}"],"description":"用户登录","exceptionMsg":"操作成功","ip":"10.244.212.244","methodUri":"/login","operationDetail":"{\\"resultCode\\":0,\\"resultMsg\\":\\"操作成功\\",\\"resultObj\\":{\\"content\\":{\\"authToken\\":\\"eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyQWNjb3VudE5hbWUiOiJxY2pfdGVzdCIsInRpbWVTdGFtcCI6IjIwMjQtMDQtMjMgMTg6MDg6MjQiLCJpc0J1aWx0IjoiMiIsImlwQWRkcmVzcyI6IjEwLjI0NC4yMTIuMjQ0IiwiZXhwIjoxNzE0NDcxNzA0fQ.Uvz05ACfVIkVhP8OSMIGPkBer-X-QOfAz8truBJydt8\\",\\"isBuiltInUserAccount\\":2,\\"passwordEffectiveTime\\":90,\\"passwordModifyTime\\":\\"2024-03-05T14:13:43\\",\\"resetPassword\\":false,\\"securityAccountOverTime\\":7,\\"securityPasswordOverTime\\":15,\\"ssoLoginUser\\":false,\\"tokenExpireTime\\":1800,\\"userAccountName\\":\\"\\",\\"userAccountOverTime\\":\\"2999-12-31T00:00:00\\"}}}","operationMenu":"secure","operationType":"登入","state":0,"userName":"zhejiang","""

    ran_ums_template = """[{date_time}]--[RAN-UMS]--[ACS_OPERATION]--[RECORD]--"""
    ums_log = """{"id":null,"userName":"ums_admin","operationAddress":"10.244.182.130","businessModule":"设备管理/设备列表/设备列表","operationType":"查询","operationTarget":"基站","methodName":"pageDeviceMain","description":"查询设备列表","operationDetail":[{"pageSize":10,"pageNum":1,"sortName":"id","sortOrder":"DESC","dictTypeCode":null,"dictValueList":null,"includeData":null,"deviceNameList":null,"serialNumberList":null,"componentIdentifierList":null,"idList":null,"excludeDeviceIdList":null,"parentDeviceMainIdList":null,"name":null,"nodebId":null,"serialNumber":null,"ouiList":null,"deviceModelIdList":null,"productCategoryIdList":null,"openStationTypeList":null,"openStationPlanList":null,"areaCodeList":null,"networkTypeList":null,"deviceGroupNetworkTypeList":null,"activateStatusList":null,"connectionStatusList":null,"workflowStatusList":null,"zeroParentDeviceMainId":null,"keyword":null,"notEqualActivateStatus":null,"userDataPermissionVO":null,"deviceIdList":null,"commonDeviceListReqVo":null,"searchColumn":null,"searchKeyword":null,"copyValueList":null,"deviceGroupIdList":null}],"state":0,"exceptionMsg":null,"""

    operator_time = """"operationTime":{timestamp}"""
    end_symbol = """}"""

    # 解析日期范围
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    delta = timedelta(days=1)

    logs = []
    current_date = start_date
    while current_date <= end_date:
        # 生成UAA日志（毫秒后缀.646）
        uaa_time_str = generate_random_time(current_date, ".646")
        uaa_dt = datetime.strptime(uaa_time_str, "%Y-%m-%dT%H:%M:%S.%f%z")
        uaa_utc = uaa_dt.astimezone(ZoneInfo("UTC"))
        uaa_timestamp = int(uaa_utc.timestamp()) * 1000  # 转换为毫秒时间戳
        logs.append(uaa_template.format(date_time=uaa_time_str, timestamp=uaa_timestamp))
        logs.append(uaa_log)

        operation_time = operator_time.format(date_time=uaa_time_str, timestamp=uaa_timestamp)
        logs.append(operation_time)
        logs.append(end_symbol + "\n")


        # 生成RAN-UMS日志（毫秒后缀.317）
        ran_time_str = generate_random_time(current_date, ".317")
        ran_dt = datetime.strptime(ran_time_str, "%Y-%m-%dT%H:%M:%S.%f%z")
        ran_utc = ran_dt.astimezone(ZoneInfo("UTC"))
        ran_timestamp = int(ran_utc.timestamp()) * 1000
        logs.append(ran_ums_template.format(date_time=ran_time_str, timestamp=ran_timestamp))
        logs.append(ums_log)
        operation_time = operator_time.format(date_time=ran_time_str, timestamp=ran_timestamp)
        logs.append(operation_time)
        logs.append(end_symbol + "\n")

        current_date += delta

    # 写入文件（覆盖模式，如需追加改为"a"）
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(logs))


start_date = "2025-02-02"
end_date = "2025-06-25"
output_file = r"c:\Program Files\BusinessFile\python_project\tensorflow-learning\learning\test\ACS_OPERATION_log_0.log"

# 执行生成（设置随机种子可复现结果，注释则完全随机）
# random.seed(42)  # 可选：固定随机种子
generate_logs(start_date, end_date, output_file)
print(f"日志已生成至：{output_file}")
