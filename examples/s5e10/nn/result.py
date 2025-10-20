# 加载模型函数
from examples.s5e10.nn import contant
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import contant
from data import read_org_data, S5e10Data
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from examples.s5e10.nn.model import Net


# 添加在文件末尾

# 加载模型函数
# 修改load_model函数中的torch.load调用
def load_model(model_path='road_accident_model.pth'):
    """
    加载训练好的模型及其预处理器

    Args:
        model_path: 模型文件路径

    Returns:
        model: 加载好的模型
        preprocessor: 数据预处理器
        input_dim: 模型输入维度
    """
    # 修复：添加weights_only=False参数以支持加载包含预处理器的完整模型
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

    # 创建模型实例
    input_dim = checkpoint['input_dim']
    model = Net(input_dim=input_dim)

    # 加载模型参数
    model.load_state_dict(checkpoint['model_state_dict'])

    # 获取预处理器
    preprocessor = checkpoint['preprocessor']

    # 设置为评估模式
    model.eval()

    return model, preprocessor, input_dim


# 预测函数
def predict(model, data_loader, device='cpu'):
    """
    使用模型进行预测

    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        device: 运行设备

    Returns:
        predictions: 预测结果列表
    """
    model.to(device)
    model.eval()

    predictions = []

    with torch.no_grad():
        for batch_data in data_loader:
            # 处理输入数据
            if isinstance(batch_data, tuple) and len(batch_data) == 2:
                inputs, _ = batch_data  # 忽略标签
            else:
                inputs = batch_data

            inputs = inputs.to(device)
            outputs = model(inputs)

            # 将预测结果转换为numpy数组并添加到列表中
            predictions.extend(outputs.cpu().numpy().flatten().tolist())

    return predictions


# 完整的预测流程函数
def make_predictions(test_data, model_path='road_accident_model.pth', output_csv='predictions.csv'):
    """
    使用训练好的模型对测试数据进行预测，并将结果保存为CSV文件

    Args:
        test_data: 测试数据（pandas DataFrame）
        model_path: 模型文件路径
        output_csv: 输出CSV文件路径

    Returns:
        predictions_df: 包含预测结果的DataFrame
    """
    # 加载模型
    print(f"正在加载模型: {model_path}")
    model, preprocessor, input_dim = load_model(model_path)

    # 创建测试数据集
    test_dataset = S5e10Data(test_data, train=False, preprocessor=preprocessor)

    # 创建数据加载器
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=contant.Model.get('batch_size'),
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备进行预测: {device}")

    # 进行预测
    print("开始预测...")
    predictions = predict(model, test_loader, device)

    # 将预测结果转换为pandas DataFrame
    predictions_df = pd.DataFrame({
        'id': test_data['id'].values if 'id' in test_data.columns else range(len(test_data)),
        'accident_risk': predictions  # 假设预测目标是accident_risk
    })

    # 保存为CSV文件
    predictions_df.to_csv(output_csv, index=False)
    print(f"预测结果已保存至: {output_csv}")

    # 返回预测结果DataFrame
    return predictions_df



# 方式2：分步使用
# 1. 加载模型
model, preprocessor, input_dim = load_model('road_accident_model.pth')
# 2. 读取数据
_, test_data = read_org_data()
# 3. 进行预测
predictions_df = make_predictions(
    test_data=test_data,
    model_path='road_accident_model.pth',
    output_csv='my_predictions.csv'  # 可以自定义输出文件名
)
predictions_df.to_csv('result.csv', index=False)
