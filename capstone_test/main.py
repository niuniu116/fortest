# main.py
import torch
from torch import optim
from model.VGG19 import VGG19
from dataloader import get_data_loaders
from train import train_model
from utils import evaluate_model

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 获取数据加载器
    train_loader, val_loader, test_loader = get_data_loaders(
        train_dir="/Users/yihaoniu/Desktop/data_result/train",
        val_dir="/Users/yihaoniu/Desktop/data_result/val",
        test_dir="/Users/yihaoniu/Desktop/data_result/test"
    )

    # 创建模型
    model = VGG19(num_classes=5)  # 五分类任务
    model.to(device)

    # 定义损失函数和优化器
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    # 训练模型
    train_model(model, train_loader, criterion, optimizer, device, num_epochs=10)

    # 测试模型
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()
