import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
import sys
import datetime


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


def train_and_validate(model, train_loader, val_loader, epochs, device='cpu',
                       save_path='cnn_attention_best.pth', save_best_only=True,
                       patience=15, auto_lr=False, lr=1e-4,
                       LOG_DIR=f"runs/handwriting_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                       plot_results=True) :
    """
    训练和验证模型，可选地绘制训练曲线

    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        epochs: 训练轮数
        device: 训练设备 ('cpu' 或 'cuda')
        save_path: 模型保存路径
        save_best_only: 是否只保存最佳模型
        patience: 早停等待轮数
        auto_lr: 是否自动调整学习率
        lr: 初始学习率
        LOG_DIR: TensorBoard日志目录
        plot_results: 是否绘制训练曲线
    """
    # 初始化绘图
    if plot_results :
        fig, axes, lines, epochs_list, train_losses, val_losses, \
            train_top1_accs, val_top1_accs, train_top3_accs, val_top3_accs = setup_plotting()

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=LOG_DIR)

    best_val_top1_acc = 0.0
    no_improve_count = 0

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    for epoch in range(epochs) :
        model.train()
        total_loss = 0.0
        top1_correct = 0
        top3_correct = 0
        total = 0

        # --- 训练阶段 ---
        for batch_x, batch_y in train_loader :
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

        with torch.no_grad() :
            for val_x, val_y in val_loader :
                val_x, val_y = val_x.to(device), val_y.to(device)
                val_outputs = model(val_x)
                val_loss += loss_func(val_outputs, val_y).item() * val_x.size(0)

                # Top-1
                _, val_top1_pred = torch.max(val_outputs, 1)
                val_top1_correct += (val_top1_pred == val_y).sum().item()

                # Top-3
                _, val_top3_pred_indices = torch.topk(val_outputs, 3, dim=1)
                val_top3_correct += torch.sum(
                    val_top3_pred_indices == val_y.view(-1, 1).expand_as(val_top3_pred_indices)).item()

                val_total += val_y.size(0)

        avg_val_loss = val_loss / val_total
        val_top1_acc = val_top1_correct / val_total
        val_top3_acc = val_top3_correct / val_total

        if auto_lr :
            scheduler.step(val_top1_acc)

        # --- 更新绘图数据 ---
        if plot_results :
            update_plot(epoch, avg_loss, avg_val_loss, train_top1_acc, val_top1_acc,
                        train_top3_acc, val_top3_acc, epochs_list, train_losses,
                        val_losses, train_top1_accs, val_top1_accs,
                        train_top3_accs, val_top3_accs, lines, axes)

        # --- TensorBoard日志记录 ---
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Accuracy/train_top1", train_top1_acc, epoch)
        writer.add_scalar("Accuracy/train_top3", train_top3_acc, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/val_top1", val_top1_acc, epoch)
        writer.add_scalar("Accuracy/val_top3", val_top3_acc, epoch)

        # --- 控制台输出 ---
        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.4f}, "
              f"Train Top-1 Acc: {train_top1_acc:.4f}, Train Top-3 Acc: {train_top3_acc:.4f}, "
              f"Val Top-1 Acc: {val_top1_acc:.4f}, Val Top-3 Acc: {val_top3_acc:.4f}")

        # --- 保存最佳模型 ---
        if val_top1_acc > best_val_top1_acc :
            best_val_top1_acc = val_top1_acc
            no_improve_count = 0
            if save_best_only :
                torch.save(model.state_dict(), save_path)
                print(f"✅ 新的最佳模型已保存，Val Top-1 Acc: {val_top1_acc:.4f}")
        else :
            no_improve_count += 1

        # --- Early Stopping ---
        if no_improve_count >= patience :
            print(f"⏳ 验证集Top-1准确率在连续 {patience} 轮未提升，训练提前终止。")
            break

        sys.stdout.flush()

    # --- 训练完成后的处理 ---
    writer.close()

    if plot_results :
        finalize_plotting(fig)

    return model