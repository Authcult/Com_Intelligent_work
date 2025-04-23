import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import sys
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np


# 训练与验证函数
def train_and_validate(model, train_loader, val_loader, epochs, device='cpu',
                       save_path='cnn_attention_best.pth', save_best_only=True, patience=15, auto_lr = False, lr = 1e-4, LOG_DIR = f"runs/handwriting_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_func = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=LOG_DIR)

    best_val_top1_acc = 0.0  # 用于追踪验证集 Top-1 准确率的最佳值
    no_improve_count = 0

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # 选择一个中间层特征提取钩子
    def hook_fn(module, input, output):
        model.features_map = output.detach()

    # 注册钩子
    model.conv2[0].register_forward_hook(hook_fn)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        top1_correct = 0
        top3_correct = 0
        total = 0

        # --- 训练阶段 ---
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            outputs = model(batch_x)
            loss = loss_func(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)

            # Top-1
            _, top1_pred = torch.max(outputs, 1)
            top1_correct += (top1_pred == batch_y).sum().item()

            # Top-3
            _, top3_pred_indices = torch.topk(outputs, 3, dim=1)
            top3_correct += torch.sum(top3_pred_indices == batch_y.view(-1, 1).expand_as(top3_pred_indices)).item()

            total += batch_y.size(0)

        avg_loss = total_loss / total
        train_top1_acc = top1_correct / total
        train_top3_acc = top3_correct / total

        # --- 验证阶段 ---
        model.eval()
        val_top1_correct = 0
        val_top3_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x, val_y = val_x.to(device), val_y.to(device)
                val_outputs = model(val_x)
                val_loss += loss_func(val_outputs, val_y).item() * val_x.size(0)

                # Top-1
                _, val_top1_pred = torch.max(val_outputs, 1)
                val_top1_correct += (val_top1_pred == val_y).sum().item()

                # Top-3
                _, val_top3_pred_indices = torch.topk(val_outputs, 3, dim=1)
                val_top3_correct += torch.sum(val_top3_pred_indices == val_y.view(-1, 1).expand_as(val_top3_pred_indices)).item()

                val_total += val_y.size(0)

        avg_val_loss = val_loss / val_total
        val_top1_acc = val_top1_correct / val_total
        val_top3_acc = val_top3_correct / val_total

        if auto_lr:
            scheduler.step(val_top1_acc)

        # --- TensorBoard日志记录 ---
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Accuracy/train_top1", train_top1_acc, epoch)
        writer.add_scalar("Accuracy/train_top3", train_top3_acc, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/val_top1", val_top1_acc, epoch)
        writer.add_scalar("Accuracy/val_top3", val_top3_acc, epoch)

        # 可视化特征热力图
        if epoch % 10 == 0:  # 每10个epoch进行一次特征图可视化
            feature_map = model.features_map
            # 假设特征图的大小是 [batch_size, channels, height, width]
            feature_map = feature_map[0]  # 选择第一张图像
            feature_map = feature_map.mean(dim=0)  # 在通道维度上求平均，生成一张2D图像
            feature_map = feature_map.cpu().numpy()
            # 归一化到[0, 1]
            feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
            # 可视化
            plt.imshow(feature_map, cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.title(f"Feature Map Epoch {epoch}")
            plt.savefig(f"{LOG_DIR}/feature_map_epoch_{epoch}.png")
            plt.show()
            plt.close()
            # 将热力图添加到TensorBoard
            writer.add_image(f"Feature Map/Epoch {epoch}", np.expand_dims(feature_map, axis=0), epoch)

        # --- 控制台输出 ---
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}, "
              f"Train Top-1 Acc: {train_top1_acc:.4f}, Train Top-3 Acc: {train_top3_acc:.4f}, "
              f"Val Top-1 Acc: {val_top1_acc:.4f}, Val Top-3 Acc: {val_top3_acc:.4f}")

        # --- 保存最佳模型 ---
        if val_top1_acc > best_val_top1_acc:
            best_val_top1_acc = val_top1_acc
            no_improve_count = 0
            if save_best_only:
                torch.save(model.state_dict(), save_path)
                print(f"✅ 新的最佳模型已保存，Val Top-1 Acc: {val_top1_acc:.4f}")
        else:
            no_improve_count += 1

        # --- Early Stopping ---
        if no_improve_count >= patience:
            print(f"⏳ 验证集Top-1准确率在连续 {patience} 轮未提升，训练提前终止。")
            break

        sys.stdout.flush()

    writer.close()