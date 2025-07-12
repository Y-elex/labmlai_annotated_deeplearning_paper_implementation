import os
from typing import List, Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from labml import experiment
import sys
sys.path.append('/content/labmlai_annotated_deeplearning_paper_implementation/labml_nn')
sys.path.append('/content/labmlai_annotated_deeplearning_paper_implementation')
from resnet import ResNetBase
from labml.configs import option
from experiments.cifar10 import CIFAR10Configs
import matplotlib.pyplot as plt
import pandas as pd

# 数据加载器
def create_data_loaders(train_dir: str, valid_dir: str, batch_size: int):
    """
    创建训练和验证数据加载器
    :param train_dir: 训练数据集目录
    :param valid_dir: 验证数据集目录
    :param batch_size: 批次大小
    :return: 训练和验证数据加载器
    """
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # 假设输入图像尺寸为32x32
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet标准化
    ])

    # 训练数据集
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 验证数据集
    valid_dataset = datasets.ImageFolder(root=valid_dir, transform=transform)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader

# 配置类
class Configs(CIFAR10Configs):
    """
    配置类，用于设置模型和训练参数
    """
    n_blocks: List[int] = [6, 6, 6]  # ResNet的块数
    n_channels: List[int] = [16, 32, 64]  # ResNet的每个卷积块的通道数
    bottlenecks: Optional[List[int]] = [8, 16, 16]  # 每个卷积块的瓶颈层大小
    first_kernel_size: int = 3  # 第一层卷积核大小
    train_batch_size: int = 32  # 批次大小
    epochs: int = 100  # 训练的总轮数
    learning_rate: float = 2.5e-4  # 学习率
    optimizer: str = 'Adam'  # 优化器
    train_dataset: str = ''  # 训练数据集路径
    valid_dataset: str = ''  # 验证数据集路径
    pretrained_model_path: str = 'best_model.pth'  # 预训练模型路径

# 创建ResNet模型
@option(Configs.model)
def _resnet(c: Configs):
    base = ResNetBase(c.n_blocks, c.n_channels, c.bottlenecks, img_channels=3, first_kernel_size=c.first_kernel_size)
    classification = nn.Linear(c.n_channels[-1], 8)  # 输出7个类别，假设有7个情感类别
    model = nn.Sequential(base, classification)
    return model.to(c.device)

# 训练和验证函数
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()  # 设置为训练模式
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()  # 清除旧的梯度
        outputs = model(images)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)  # 获取预测结果
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, valid_loader, criterion, device):
    model.eval()  # 设置为评估模式
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # 禁用梯度计算
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)  # 获取预测结果
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(valid_loader.dataset)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

# 检查并加载预训练模型
def load_pretrained_model(model, model_path, device):
    if os.path.exists(model_path):
        print(f"Loading pretrained model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model = model.to(device)
    else:
        print(f"No pretrained model found at {model_path}. Starting from scratch.")
    return model

def main():
    # 创建实验
    experiment.create(name='resnet', comment='ckplus')

    # 配置训练参数
    conf = Configs()
    train_acc_list = []
    val_acc_list = []
    train_loss_list = []
    val_loss_list = []
    epoch_list = []


    # 设置数据集路径
    train_dir = '/content/labmlai_annotated_deeplearning_paper_implementation/labml_nn/resnet/data/FERPlus/train'
    valid_dir = '/content/labmlai_annotated_deeplearning_paper_implementation/labml_nn/resnet/data/FERPlus/val'

    # 创建数据加载器
    train_loader, valid_loader = create_data_loaders(train_dir, valid_dir, batch_size=conf.train_batch_size)

    # 设置模型、损失函数和优化器
    model = conf.model
    criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=conf.learning_rate)

    # 设置训练和验证的设备（GPU或CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 如果有预训练模型，加载它
    model = load_pretrained_model(model, conf.pretrained_model_path, device)

    # 训练过程
    best_acc = 0.0
    for epoch in range(conf.epochs):
        print(f'Epoch {epoch+1}/{conf.epochs}')

        # 训练一个epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f'Training loss: {train_loss:.4f}, Training accuracy: {train_acc:.2f}%')

        # 验证一个epoch
        val_loss, val_acc = evaluate(model, valid_loader, criterion, device)
        print(f'Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.2f}%')

        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        epoch_list.append(epoch)


        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model_FERPlus.pth')  # 保存最佳模型
    
    # 创建一个 DataFrame
    df = pd.DataFrame({
        'epoch': epoch_list,
        'train_acc': train_acc_list,
        'val_acc': val_acc_list,
        'train_loss': train_loss_list,
        'val_loss': val_loss_list
    })

    # 保存为 Excel 文件
    df.to_excel('training_log.xlsx', index=False)

    plt.figure(figsize=(12, 5))

    # 准确率图
    plt.subplot(1, 2, 1)
    plt.plot(epoch_list, train_acc_list, label='Train Accuracy')
    plt.plot(epoch_list, val_acc_list, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Epoch vs Accuracy')
    plt.legend()
    plt.grid(True)
    
    # 损失图
    plt.subplot(1, 2, 2)
    plt.plot(epoch_list, train_loss_list, label='Train Loss')
    plt.plot(epoch_list, val_loss_list, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch vs Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/accuracy_loss_plot.png')
    plt.show()


    print('Training finished.')

if __name__ == '__main__':
    main()
