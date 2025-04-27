# train.py
import torch
import torch.nn.functional as F
import time
from model import VGG19
from utils import evaluate_model


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10, val_loader=None):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()  # 设置为训练模式
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        start_time = time.time()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # 清除之前的梯度

            outputs = model(inputs)  # 模型前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            running_loss += loss.item() * inputs.size(0)

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_preds / total_preds * 100

        end_time = time.time()
        epoch_time = end_time - start_time
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.2f}% Time: {epoch_time:.2f}s")

        # 进行验证
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f} Accuracy: {val_acc:.2f}%")
