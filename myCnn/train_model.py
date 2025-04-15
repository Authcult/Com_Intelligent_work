import torch
from tensorboardX import SummaryWriter
import logging

# 配置日志记录器
logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10) :
    """
    训练模型（同时计算Top-1和Top-3准确率）

    参数:
    - model: 模型实例
    - train_loader: 训练集DataLoader
    - val_loader: 验证集DataLoader
    - criterion: 损失函数
    - optimizer: 优化器
    - num_epochs: 训练轮数

    返回:
    - model: 训练后的模型
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    writer = SummaryWriter()

    for epoch in range(num_epochs) :
        model.train()
        running_loss = 0.0
        top1_correct_train = 0
        top3_correct_train = 0
        total_train = 0

        for images, labels in train_loader :
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 计算Top-1和Top-3准确率
            _, top1_pred = torch.max(outputs, 1)
            top1_correct_train += (top1_pred == labels).sum().item()

            _, top3_pred = torch.topk(outputs, 3, dim=1)
            top3_correct_train += torch.sum(top3_pred == labels.view(-1, 1)).item()
            total_train += labels.size(0)

        # 计算准确率
        train_top1_acc = 100 * top1_correct_train / total_train
        train_top3_acc = 100 * top3_correct_train / total_train

        # 验证阶段
        model.eval()
        top1_correct_val = 0
        top3_correct_val = 0
        total_val = 0
        val_loss = 0.0

        with torch.no_grad() :
            for images, labels in val_loader :
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()

                # 计算Top-1和Top-3准确率
                _, top1_pred = torch.max(outputs, 1)
                top1_correct_val += (top1_pred == labels).sum().item()

                _, top3_pred = torch.topk(outputs, 3, dim=1)
                top3_correct_val += torch.sum(top3_pred == labels.view(-1, 1)).item()
                total_val += labels.size(0)

        # 计算验证集准确率
        val_top1_acc = 100 * top1_correct_val / total_val
        val_top3_acc = 100 * top3_correct_val / total_val

        # 日志记录
        log_message = (
            f'Epoch [{epoch + 1}/{num_epochs}], '
            f'Train Loss: {running_loss / len(train_loader):.4f}, '
            f'Train Top-1 Acc: {train_top1_acc:.2f}%, '
            f'Train Top-3 Acc: {train_top3_acc:.2f}%, '
            f'Val Loss: {val_loss / len(val_loader):.4f}, '
            f'Val Top-1 Acc: {val_top1_acc:.2f}%, '
            f'Val Top-3 Acc: {val_top3_acc:.2f}%'
        )
        print(log_message)
        logging.info(log_message)

        # TensorBoard记录
        writer.add_scalars('Train Metrics', {
            'Loss' : running_loss / len(train_loader),
            'Top-1 Acc' : train_top1_acc,
            'Top-3 Acc' : train_top3_acc
        }, epoch)

        writer.add_scalars('Validation Metrics', {
            'Loss' : val_loss / len(val_loader),
            'Top-1 Acc' : val_top1_acc,
            'Top-3 Acc' : val_top3_acc
        }, epoch)

    writer.close()
    return model