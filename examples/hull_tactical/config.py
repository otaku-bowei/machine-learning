import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

File_config = {
    'train_path': '/kaggle/input/hull-tactical-market-prediction/train.csv',
    'test_path': '/kaggle/input/hull-tactical-market-prediction/test.csv',
    'model_path': '/kaggle/working/model.params',

}
Data_config = {
    'feature_fields': ['date_id', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'E1', 'E10', 'E11', 'E12',
                       'E13', 'E14', 'E15', 'E16', 'E17', 'E18', 'E19', 'E2', 'E20', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8',
                       'E9', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'M1', 'M10', 'M11', 'M12', 'M13',
                       'M14', 'M15', 'M16', 'M17', 'M18', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'P1', 'P10',
                       'P11', 'P12', 'P13', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'S1', 'S10', 'S11', 'S12',
                       'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'V1', 'V10', 'V11', 'V12', 'V13', 'V2', 'V3',
                       'V4', 'V5', 'V6', 'V7', 'V8', 'V9'],
    'label_fields': ['forward_returns', 'risk_free_rate', 'market_forward_excess_returns'],
    'input_dim': 95,
    'output_dim': 3,
    'standard_fields': ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'E1', 'E10', 'E11', 'E12', 'E13', 'E14',
                        'E15', 'E16', 'E17', 'E18', 'E19', 'E2', 'E20', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'I1',
                        'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'M1', 'M10', 'M11', 'M12', 'M13', 'M14', 'M15',
                        'M16', 'M17', 'M18', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'P1', 'P10', 'P11', 'P12',
                        'P13', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'S1', 'S10', 'S11', 'S12', 'S2', 'S3',
                        'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'V1', 'V10', 'V11', 'V12', 'V13', 'V2', 'V3', 'V4', 'V5',
                        'V6', 'V7', 'V8', 'V9'],
}

Model_config = {
    'lr': 0.001,
    'batch_size': 64,
    'num_workers': 4,
    'epochs': 200,
    'debug': False,
    'debug_line': 1000,
    'debug_epochs': 10,
    'print_range': 1,
    'weight_decay': 1e-5,
}


class Net(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(Net, self).__init__()
        # 第一层：输入层到隐藏层
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(0.2)

        # 第二层：隐藏层到隐藏层
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(0.1)

        # 第三层：隐藏层到隐藏层
        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.do3 = nn.Dropout(0.1)

        # 第三层：隐藏层到输出层
        self.fc4 = nn.Linear(32, output_dim)
        # 注意：根据任务类型决定是否保留输出层激活函数
        # self.relu3 = nn.ReLU()  # 回归任务建议移除

    def forward(self, x):
        # 第一层前向传播
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.do1(x)

        # 第二层前向传播
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.do2(x)

        # 第三层前向传播
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.do3(x)
        # 输出层
        x = self.fc4(x)
        # 如果保留了relu3，则添加：x = self.relu3(x)
        return x

    def init_normal(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, mean=0, std=0)
            nn.init.zeros_(m.bias)


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        return torch.sqrt(self.mse(output, target))


def initialize_components(input_dim, output_dim):
    model = Net(input_dim, output_dim)
    # 选择损失函数
    # loss_fn = nn.MSELoss()  # 内置MSE
    # loss_fn = CustomHuberLoss(delta=1.5)  # 自定义Huber
    loss_fn = RMSELoss()  # 自定义RMSE
    optimizer = torch.optim.AdamW(model.parameters(), lr=Model_config.get('lr'),
                                  weight_decay=Model_config.get('weight_decay'))
    return model, loss_fn, optimizer


def train_model(model, train_loader, val_loader, debug: bool = False, batch_size: int = 10):
    num_epochs = Model_config.get('debug_epochs') if debug else Model_config.get('epochs')
    """
    基础训练函数
    """
    # 定义损失函数和优化器
    net_model, criterion, optimizer = initialize_components(Data_config.get('input_dim'),
                                                            Data_config.get('output_dim'))
    # 训练历史记录
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            output = model(data)
            loss = criterion(output, target)
            # 反向传播
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # 每n个batch打印一次
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch + 1}/{num_epochs}, Batch: {batch_idx}, Loss: {loss.item():.6f}')

        # 计算平均训练损失
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证模式
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                val_loss += criterion(output, target).item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')

    return train_losses, val_losses



def setup_training(train_feature_tensor, train_label_tensor, test_feature_tensor, test_label_tensor,
                   batch_size=32, learning_rate=0.001, validation_ratio=0.2):
    """
    设置训练环境，从训练集划分验证集

    参数:
        train_feature_tensor: 训练特征tensor
        train_label_tensor: 训练标签tensor
        test_feature_tensor: 测试特征tensor
        test_label_tensor: 测试标签tensor
        batch_size: 批次大小
        learning_rate: 学习率
        validation_ratio: 验证集比例

    返回:
        train_loader, val_loader, test_loader, optimizer, criterion
    """
    # 1. 从训练集划分验证集
    train_size = int((1 - validation_ratio) * len(train_feature_tensor))

    train_indices = train_feature_tensor[:train_size]
    val_indices = train_feature_tensor[train_size:]

    # 划分训练集和验证集
    train_features = train_feature_tensor[train_indices]
    train_labels = train_label_tensor[train_indices]
    val_features = train_feature_tensor[val_indices]
    val_labels = train_label_tensor[val_indices]

    # 2. 创建数据集
    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    test_dataset = TensorDataset(test_feature_tensor, test_label_tensor)

    # 3. 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 4. 定义优化器和损失函数
    # 注意：这里需要你先定义好模型，然后传入setup_training函数
    # 或者将模型创建移到setup_training内部

    print(f"数据划分完成:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")

    return train_loader, val_loader, test_loader


def train_loop(train_loader, val_loader, input_dim, output_dim, debug : bool = False,
               patience=10, model_save_path='best_model.pth'):
    """
    完整的训练循环，包含早停和模型保存

    参数:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        criterion: 损失函数
        num_epochs: 最大训练轮数
        patience: 早停耐心值
        model_save_path: 模型保存路径

    返回:
        train_losses, val_losses, best_val_loss
    """
    model, criterion, optimizer = initialize_components(input_dim, output_dim)
    num_epochs = Model_config.get('debug_epochs') if debug else Model_config.get('epochs')
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    print("开始训练...")
    print("-" * 60)

    for epoch in range(num_epochs):
        # ===== 训练阶段 =====
        model.train()
        train_loss = 0.0

        for batch_idx, (features, labels) in enumerate(train_loader):
            # 前向传播
            outputs = model(features)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # 打印批次信息（可选）
            if batch_idx % 50 == 0:
                print(f'  批次 {batch_idx}/{len(train_loader)}, 损失: {loss.item():.6f}')

        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ===== 验证阶段 =====
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                val_loss += criterion(outputs, labels).item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # ===== 打印训练信息 =====
        print(f'轮次 [{epoch + 1:03d}/{num_epochs}] | '
              f'训练损失: {avg_train_loss:.6f} | '
              f'验证损失: {avg_val_loss:.6f}')

        # ===== 早停和模型保存 =====
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, model_save_path)
            print(f'  ↳ 保存最佳模型! 验证损失: {avg_val_loss:.6f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'  ↳ 早停! 在轮次 {epoch + 1} 停止训练')
                break

    print("-" * 60)
    print(f"训练完成! 最佳验证损失: {best_val_loss:.6f}")

    # 加载最佳模型
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, train_losses, val_losses, best_val_loss