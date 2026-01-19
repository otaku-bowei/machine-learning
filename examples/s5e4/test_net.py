import wandb
import requests
import time

# 首先测试网络连通性
try:
    response = requests.get("https://api.wandb.ai", timeout=10)
    print(f"网络连通性测试: 状态码 {response.status_code}")
except requests.exceptions.RequestException as e:
    print(f"网络连接问题: {e}")
    print("请检查你的网络连接，特别是防火墙和代理设置")