import torch
from tensorboardX import SummaryWriter
import os
import matplotlib.pyplot as plt
import datetime
import sys


# --- 绘图功能函数 ---

def setup_plotting() :
    """
    初始化matplotlib图形和坐标轴用于绘图
    返回图形对象、坐标轴对象、线条对象（用于更新数据）
    以及存储指标历史的列表
    """
    # 使用非交互模式以适应Jupyter环境
    plt.ioff()

    # 创建包含3个子图的图形
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    ax_loss, ax_top1, ax_top3 = axes

    # 初始化存储训练指标的列表
    epochs_list = []
    train_losses = []
    val_losses = []
    train_top1_accs = []
    val_top1_accs = []
    train_top3_accs = []
    val_top3_accs = []

    # 初始化绘图线条对象
    lines = {
        'train_loss' : ax_loss.plot(epochs_list, train_losses, label='Train Loss', marker='o', linestyle='-')[0],
        'val_loss' : ax_loss.plot(epochs_list, val_losses, label='Val Loss', marker='o', linestyle='-')[0],
        'train_top1' : ax_top1.plot(epochs_list, train_top1_accs, label='Train Top-1 Acc', marker='o', linestyle='-')[
            0],
        'val_top1' : ax_top1.plot(epochs_list, val_top1_accs, label='Val Top-1 Acc', marker='o', linestyle='-')[0],
        'train_top3' : ax_top3.plot(epochs_list, train_top3_accs, label='Train Top-3 Acc', marker='o', linestyle='-')[
            0],
        'val_top3' : ax_top3.plot(epochs_list, val_top3_accs, label='Val Top-3 Acc', marker='o', linestyle='-')[0],
    }

    # 配置图表标签和标题
    ax_loss.set_ylabel('Loss')
    ax_loss.legend()
    ax_top1.set_ylabel('Accuracy (%)')
    ax_top1.legend()
    ax_top3.set_ylabel('Accuracy (%)')
    ax_top3.set_xlabel('Epoch')
    ax_top3.legend()

    fig.suptitle('Training and Validation Metrics Over Epochs')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig, axes, lines, epochs_list, train_losses, val_losses, \
        train_top1_accs, val_top1_accs, train_top3_accs, val_top3_accs


def update_plot(epoch, train_loss, val_loss, train_top1, val_top1, train_top3, val_top3,
                epochs_list, train_losses, val_losses, train_top1_accs, val_top1_accs,
                train_top3_accs, val_top3_accs, lines, axes) :
    """
    更新绘图数据但不实时显示
    """
    # 添加新数据到列表
    epochs_list.append(epoch + 1)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_top1_accs.append(train_top1)
    val_top1_accs.append(val_top1)
    train_top3_accs.append(train_top3)
    val_top3_accs.append(val_top3)

    # 更新线条数据
    lines['train_loss'].set_data(epochs_list, train_losses)
    lines['val_loss'].set_data(epochs_list, val_losses)
    lines['train_top1'].set_data(epochs_list, train_top1_accs)
    lines['val_top1'].set_data(epochs_list, val_top1_accs)
    lines['train_top3'].set_data(epochs_list, train_top3_accs)
    lines['val_top3'].set_data(epochs_list, val_top3_accs)

    # 自动调整坐标轴范围
    for ax in axes :
        ax.relim()
        ax.autoscale_view()


def finalize_plotting(fig) :
    """
    训练完成后显示并保存最终图表
    """
    # 在Jupyter Notebook中显示图表
    plt.show()

    # 保存图表到文件
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_save_dir = "training_plots"
    os.makedirs(plot_save_dir, exist_ok=True)

    plot_save_path = os.path.join(plot_save_dir, f'{timestamp}_training_metrics.png')
    fig.savefig(plot_save_path)
    print(f"Saved final training metrics plot to {plot_save_path}")

    plt.close(fig)


# --- 主训练函数 ---

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None,
                num_epochs=10, save_path='best_model.pth', save_best_only=False) :
    # 设置计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # 初始化TensorBoard记录器
    writer = SummaryWriter()

    best_val_top1_acc = 0.0
    print("Starting training...")

    # 初始化绘图相关对象
    fig, axes, lines, epochs_list, train_losses, val_losses, \
        train_top1_accs, val_top1_accs, train_top3_accs, val_top3_accs = setup_plotting()

    for epoch in range(num_epochs) :
        # 训练阶段
        model.train()
        running_loss = 0.0
        top1_correct_train = 0
        top3_correct_train = 0
        total_train = 0

        for i, (images, labels) in enumerate(train_loader) :
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 按batch更新的学习率调度器
            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR) :
                scheduler.step()

            # 累计统计量
            running_loss += loss.item() * images.size(0)
            _, top1_pred = torch.max(outputs, 1)
            top1_correct_train += (top1_pred == labels).sum().item()

            _, top3_pred_indices = torch.topk(outputs, 3, dim=1)
            top3_correct_train += torch.sum(torch.eq(top3_pred_indices, labels.unsqueeze(1)).any(dim=1)).item()

            total_train += labels.size(0)

        # 计算epoch指标
        epoch_train_loss = running_loss / total_train if total_train > 0 else 0.0
        train_top1_acc = 100 * top1_correct_train / total_train if total_train > 0 else 0.0
        train_top3_acc = 100 * top3_correct_train / total_train if total_train > 0 else 0.0

        # 按epoch更新的学习率调度器
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR) :
            scheduler.step()

        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        # 验证阶段
        model.eval()
        top1_correct_val = 0
        top3_correct_val = 0
        total_val = 0
        val_running_loss = 0.0

        with torch.no_grad() :
            for images, labels in val_loader :
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)

                _, top1_pred = torch.max(outputs, 1)
                top1_correct_val += (top1_pred == labels).sum().item()

                _, top3_pred_indices = torch.topk(outputs, 3, dim=1)
                top3_correct_val += torch.sum(torch.eq(top3_pred_indices, labels.unsqueeze(1)).any(dim=1)).item()

                total_val += labels.size(0)

        # 计算验证集指标
        epoch_val_loss = val_running_loss / total_val if total_val > 0 else 0.0
        val_top1_acc = 100 * top1_correct_val / total_val if total_val > 0 else 0.0
        val_top3_acc = 100 * top3_correct_val / total_val if total_val > 0 else 0.0

        # 输出epoch结果
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], '
            f'Train Loss: {epoch_train_loss:.4f}, '
            f'Train Top-1: {train_top1_acc:.2f}%, '
            f'Train Top-3: {train_top3_acc:.2f}%, | '
            f'Val Loss: {epoch_val_loss:.4f}, '
            f'Val Top-1: {val_top1_acc:.2f}%, '
            f'Val Top-3: {val_top3_acc:.2f}%'
        )

        # 记录到TensorBoard
        writer.add_scalar('Loss/Train', epoch_train_loss, epoch)
        writer.add_scalar('Accuracy/Train_Top1', train_top1_acc, epoch)
        writer.add_scalar('Accuracy/Train_Top3', train_top3_acc, epoch)
        writer.add_scalar('Loss/Validation', epoch_val_loss, epoch)
        writer.add_scalar('Accuracy/Validation_Top1', val_top1_acc, epoch)
        writer.add_scalar('Accuracy/Validation_Top3', val_top3_acc, epoch)

        # 更新绘图数据
        update_plot(epoch, epoch_train_loss, epoch_val_loss, train_top1_acc, val_top1_acc,
                    train_top3_acc, val_top3_acc, epochs_list, train_losses, val_losses,
                    train_top1_accs, val_top1_accs, train_top3_accs, val_top3_accs, lines, axes)

        # 保存最佳模型
        if save_best_only and val_top1_acc > best_val_top1_acc :
            best_val_top1_acc = val_top1_acc
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir) :
                os.makedirs(save_dir)
            torch.save(model.state_dict(), save_path)
            print(
                f"Epoch {epoch + 1}: New best model saved to {save_path} with Val Top-1 Acc: {best_val_top1_acc:.2f}%")

    # 训练结束处理
    print("Training finished.")
    writer.close()

    # 显示并保存最终图表
    finalize_plotting(fig)

    # 保存最终模型
    if not save_best_only :
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir) :
            os.makedirs(save_dir)
        torch.save(model.state_dict(), save_path)
        print(f"Final model saved to {save_path}")

    return model