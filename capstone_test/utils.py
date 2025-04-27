# utils.py
import torch

def evaluate_model(model, data_loader, criterion, device):
    model.eval()  # 设置为评估模式
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():  # 关闭梯度计算，节省内存和计算
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)  # 模型前向传播
            loss = criterion(outputs, labels)  # 计算损失
            running_loss += loss.item() * inputs.size(0)

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

    loss = running_loss / len(data_loader.dataset)
    accuracy = correct_preds / total_preds * 100
    return loss, accuracy
