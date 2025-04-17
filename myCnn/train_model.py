import torch
from tensorboardX import SummaryWriter
import logging
import os # 引入 os 模块

# 配置日志记录器
# (确保日志文件路径有效，或者去掉 filename 参数在控制台输出)
log_file = 'training.log'
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
# 添加控制台处理器，以便在屏幕上也能看到日志
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)


def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs=10, save_path='best_model.pth', save_best_only=False):
    """
    训练模型（同时计算Top-1和Top-3准确率，并根据参数决定是否保存最佳权重）

    参数:
    - model: 模型实例
    - train_loader: 训练集DataLoader
    - val_loader: 验证集DataLoader
    - criterion: 损失函数
    - optimizer: 优化器
    - num_epochs: 训练轮数
    - save_path (str): 保存最佳模型权重的路径
    - save_best_only (bool): 是否仅保存最佳模型权重，默认为 True

    返回:
    - model: 训练结束时的模型 (最佳权重已保存在 save_path)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    model.to(device)
    writer = SummaryWriter()  # 默认会在 runs/ 目录下创建日志

    best_val_top1_acc = 0.0  # 用于追踪最佳验证Top-1准确率

    logging.info("Starting training...")
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        top1_correct_train = 0
        top3_correct_train = 0
        total_train = 0

        # --- Training Phase ---
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)  # 乘以batch size计算总损失

            # 计算Top-1和Top-3准确率
            _, top1_pred = torch.max(outputs, 1)
            top1_correct_train += (top1_pred == labels).sum().item()

            _, top3_pred_indices = torch.topk(outputs, 3, dim=1)
            top3_correct_train += torch.sum(top3_pred_indices == labels.view(-1, 1).expand_as(top3_pred_indices)).item()

            total_train += labels.size(0)

        # 计算当前 epoch 的训练指标
        epoch_train_loss = running_loss / total_train
        train_top1_acc = 100 * top1_correct_train / total_train
        train_top3_acc = 100 * top3_correct_train / total_train

        # --- Validation Phase ---
        model.eval()  # 设置模型为评估模式
        top1_correct_val = 0
        top3_correct_val = 0
        total_val = 0
        val_loss = 0.0

        with torch.no_grad():  # 在验证阶段不计算梯度
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)  # 乘以batch size计算总损失

                # 计算Top-1和Top-3准确率
                _, top1_pred = torch.max(outputs, 1)
                top1_correct_val += (top1_pred == labels).sum().item()

                _, top3_pred_indices = torch.topk(outputs, 3, dim=1)
                top3_correct_val += torch.sum(top3_pred_indices == labels.view(-1, 1).expand_as(top3_pred_indices)).item()

                total_val += labels.size(0)

        # 计算当前 epoch 的验证指标
        epoch_val_loss = val_loss / total_val
        val_top1_acc = 100 * top1_correct_val / total_val
        val_top3_acc = 100 * top3_correct_val / total_val

        # --- 记录与保存 ---
        log_message = (
            f'Epoch [{epoch + 1}/{num_epochs}], '
            f'Train Loss: {epoch_train_loss:.4f}, '
            f'Train Top-1: {train_top1_acc:.2f}%, '
            f'Train Top-3: {train_top3_acc:.2f}%, | '
            f'Val Loss: {epoch_val_loss:.4f}, '
            f'Val Top-1: {val_top1_acc:.2f}%, '
            f'Val Top-3: {val_top3_acc:.2f}%'
        )
        logging.info(log_message)  # 使用 logging 记录

        # TensorBoard记录
        writer.add_scalar('Loss/Train', epoch_train_loss, epoch)
        writer.add_scalar('Accuracy/Train_Top1', train_top1_acc, epoch)
        writer.add_scalar('Accuracy/Train_Top3', train_top3_acc, epoch)
        writer.add_scalar('Loss/Validation', epoch_val_loss, epoch)
        writer.add_scalar('Accuracy/Validation_Top1', val_top1_acc, epoch)
        writer.add_scalar('Accuracy/Validation_Top3', val_top3_acc, epoch)

        # --- 保存最佳模型 ---
        if save_best_only and val_top1_acc > best_val_top1_acc:
            best_val_top1_acc = val_top1_acc
            torch.save(model.state_dict(), save_path)
            logging.info(f"Epoch {epoch + 1}: New best model saved to {save_path} with Val Top-1 Acc: {best_val_top1_acc:.2f}%")

    logging.info("Training finished.")
    writer.close()  # 关闭 TensorBoard writer

    # 如果未启用 save_best_only，则返回最终模型状态
    if not save_best_only:
        torch.save(model.state_dict(), save_path)
        logging.info(f"Final model saved to {save_path}")

    return model
