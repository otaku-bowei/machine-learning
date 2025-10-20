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

from examples.s5e10.nn.data import analyze_data_info


# 设置随机种子，确保可复现性
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


set_seed()


# 定义神经网络模型
class Net(nn.Module):
    def __init__(self, input_dim=30, output_dim=1):
        super(Net, self).__init__()
        # 构建神经网络
        self.model = nn.Sequential(
            # 第一层更宽
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            # 残差连接块1
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),

            # 第二个隐藏层
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            # 残差连接块2
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),

            # 最终层
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.model(x)


# 定义RMSE损失函数
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        return torch.sqrt(self.mse(output, target))


# 训练模型函数
def train_model(net, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device):
    # 将模型移至设备
    net.to(device)

    # 记录训练和验证损失
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None

    print(f"开始训练，使用设备: {device}")

    # 训练循环
    for epoch in range(epochs):
        # 训练模式
        net.train()
        running_train_loss = 0.0

        # 修复：确保正确解包数据
        for i, batch_data in enumerate(train_loader):
            # 明确处理输入和标签
            inputs, labels = batch_data
            # if isinstance(batch_data, tuple) and len(batch_data) == 2:
            #     inputs, labels = batch_data
            # else:
            #     # 如果解包失败，打印错误信息并跳过
            #     print(f"警告：批次 {i} 数据格式错误，跳过处理")
            #     continue

            # 将数据移至设备
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 累计损失
            running_train_loss += loss.item() * inputs.size(0)

        # 计算训练损失
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # 验证模式
        net.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            # 修复：确保正确解包验证数据
            for batch_data in val_loader:
                # 明确处理输入和标签
                inputs, labels = batch_data
                # if isinstance(batch_data, tuple) and len(batch_data) == 2:
                #     inputs, labels = batch_data
                # else:
                #     print(f"警告：验证批次数据格式错误，跳过处理")
                #     continue

                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * inputs.size(0)

        # 计算验证损失
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        # 学习率调度器步进
        if scheduler:
            scheduler.step(epoch_val_loss)

        # 保存最佳模型
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = net.state_dict().copy()

        # 打印进度
        if (epoch + 1) % contant.Model.get('print_range') == 0 or epoch == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], '
                  f'Train Loss: {epoch_train_loss:.6f}, '
                  f'Val Loss: {epoch_val_loss:.6f}')

    # 加载最佳模型
    if best_model_state:
        net.load_state_dict(best_model_state)
        print(f"训练完成！最佳验证损失: {best_val_loss:.6f}")

    return net, train_losses, val_losses


# 主函数：数据准备、模型训练和评估
def train(train_org_data):
    # 准备数据

    # 分割训练集和验证集
    train_df, val_df = train_test_split(
        train_org_data,
        test_size=0.2,
        random_state=42
    )

    # 创建数据集
    train_dataset = S5e10Data(train_df, train=True)
    val_dataset = S5e10Data(val_df, train=False, preprocessor=train_dataset.preprocessor)

    # 获取输入维度
    input_dim = train_dataset.get_input_dim()
    print(f"转换后的输入维度: {input_dim}")

    # 创建数据加载器 - 修复：简化配置以避免潜在问题
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=contant.Model.get('batch_size'),
        shuffle=True,
        num_workers=0,  # 使用0个工作进程避免多进程问题
        pin_memory=False  # 简化内存配置
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=contant.Model.get('batch_size'),
        # batch_size=contant.Model.get('batch_size') * 2,
        shuffle=False,
        num_workers=0,  # 使用0个工作进程避免多进程问题
        pin_memory=False  # 简化内存配置
    )

    # 创建模型
    model = Net(input_dim=input_dim)

    # 损失函数和优化器
    criterion = RMSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=contant.Model.get('lr'),
        weight_decay=1e-4
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        # verbose=True
    )

    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 简化：使用较少的epochs进行快速测试
    test_epochs = min(10, contant.Model.get('epochs')) if contant.Model.get('debug') else contant.Model.get('epochs')  # 先用10个epoch测试

    # 训练模型
    model, train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        epochs=test_epochs,
        device=device
    )

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    plt.show()

    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'preprocessor': train_dataset.preprocessor,
        'input_dim': input_dim
    }, 'road_accident_model.pth')

    print("模型已保存为 'road_accident_model.pth'")


train_data, test_data = read_org_data()
analyze_data_info(train_data)
debug_data = train_data[contant.Model.get('debug_line')] if contant.Model.get('debug') else train_data.copy()
# train(train_data)



