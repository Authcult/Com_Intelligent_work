import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from IPython import get_ipython
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
import sys
import datetime
from IPython.display import display, clear_output  # 添加clear_output用于实时更新
# 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# --- 绘图功能函数 ---

def setup_plotting() :
    """初始化matplotlib图形和坐标轴"""
    # 自动检测环境
    if 'ipykernel' in sys.modules :
        get_ipython().run_line_magic('matplotlib', 'inline')
        plt.ion()
    else :
        plt.ioff()

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    ax_loss, ax_top1, ax_top3 = axes

    # 初始化数据存储
    metrics = {
        'epochs' : [],
        'train_loss' : [], 'val_loss' : [],
        'train_top1' : [], 'val_top1' : [],
        'train_top3' : [], 'val_top3' : []
    }

    # 创建绘图线条
    lines = {
        'train_loss' : ax_loss.plot([], [], label='Train Loss', color='tab:blue', marker='o')[0],
        'val_loss' : ax_loss.plot([], [], label='Val Loss', color='tab:orange', marker='s')[0],
        'train_top1' : ax_top1.plot([], [], label='Train Top-1', color='tab:blue', marker='o')[0],
        'val_top1' : ax_top1.plot([], [], label='Val Top-1', color='tab:orange', marker='s')[0],
        'train_top3' : ax_top3.plot([], [], label='Train Top-3', color='tab:blue', marker='o')[0],
        'val_top3' : ax_top3.plot([], [], label='Val Top-3', color='tab:orange', marker='s')[0]
    }

    # 配置图表
    ax_loss.set_title('Training & Validation Loss')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend()

    ax_top1.set_title('Top-1 Accuracy')
    ax_top1.set_ylabel('Accuracy (%)')
    ax_top1.set_ylim(0, 1)
    ax_top1.legend()

    ax_top3.set_title('Top-3 Accuracy')
    ax_top3.set_ylabel('Accuracy (%)')
    ax_top3.set_xlabel('Epoch')
    ax_top3.set_ylim(0, 1)
    ax_top3.legend()

    plt.tight_layout()
    return fig, axes, lines, metrics


def update_plot(fig, axes, lines, metrics, epoch,
                train_loss, val_loss, train_top1, val_top1, train_top3, val_top3) :
    """更新绘图数据"""
    # 更新数据
    metrics['epochs'].append(epoch + 1)
    metrics['train_loss'].append(train_loss)
    metrics['val_loss'].append(val_loss)
    metrics['train_top1'].append(train_top1)
    metrics['val_top1'].append(val_top1)
    metrics['train_top3'].append(train_top3)
    metrics['val_top3'].append(val_top3)

    # 更新线条
    for key in lines :
        lines[key].set_data(metrics['epochs'], metrics[key])

    # 调整坐标轴
    for ax in axes :
        ax.relim()
        ax.autoscale_view()






def save_plot(fig, save_dir="training_plots") :
    """保存最终图表"""
    display(fig)
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f'training_metrics_{timestamp}.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 训练图表已保存到: {save_path}")
    plt.close(fig)


# --- 特征图可视化 ---

class FeatureMapVisualizer :
    def __init__(self, model, layer_name=None) :
        self.model = model
        self.feature_maps = []
        self.hook_handle = None
        self.setup_hook(layer_name)

    def setup_hook(self, layer_name=None) :
        """自动或手动设置特征图钩子"""
        # 查找目标层
        target_layer = None
        for name, module in self.model.named_modules() :
            if isinstance(module, nn.Conv2d) :
                if layer_name is None or name == layer_name :
                    target_layer = module
                    break

        if target_layer is None :
            raise ValueError("未找到合适的卷积层")

        # 注册钩子
        def hook_fn(module, input, output) :
            if output.dim() == 4 :  # [batch, channel, height, width]
                feat_map = output[0].detach().cpu()  # 取第一个样本
                self.feature_maps.append(feat_map.mean(dim=0))  # 通道平均

        self.hook_handle = target_layer.register_forward_hook(hook_fn)
        print(f"🔍 特征图钩子已注册到: {target_layer}")

    def visualize(self, epoch, log_dir=None) :
        """可视化最新特征图"""
        if not self.feature_maps :
            print("⚠️ 无特征图数据")
            return

        feat_map = self.feature_maps[-1].numpy()
        feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min() + 1e-8)

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # 原始特征图
        im0 = ax[0].imshow(feat_map, cmap='viridis')
        ax[0].set_title(f"Epoch {epoch} - 原始特征图")
        plt.colorbar(im0, ax=ax[0])

        # 热力图
        im1 = ax[1].imshow(feat_map, cmap='hot')
        ax[1].set_title(f"Epoch {epoch} - 热力图")
        plt.colorbar(im1, ax=ax[1])

        plt.tight_layout()

        # 显示和保存
        if 'ipykernel' in sys.modules :
            display(fig)

        if log_dir :
            os.makedirs(log_dir, exist_ok=True)
            save_path = os.path.join(log_dir, f'feature_map_epoch_{epoch}.png')
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"🖼️ 特征图已保存到: {save_path}")

        # 移除 plt.close(fig)，确保图表保持打开状态
        # plt.close(fig)
        # self.feature_maps.clear()  # 清空缓存

    def remove_hook(self) :
        """移除钩子"""
        if self.hook_handle is not None :
            self.hook_handle.remove()
            print("✅ 特征图钩子已移除")
        else :
            print("⚠️ 没有可移除的钩子")



# --- 主训练函数 ---

def train_and_validate(model, train_loader, val_loader, epochs, device='cpu',
                       save_path='best_model.pth', save_best_only=True,
                       patience=10, lr=1e-4, auto_lr=False,
                       log_dir=None, plot_results=True) :
    """完整的训练和验证流程"""

    # 初始化
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 日志记录
    if log_dir is None :
        log_dir = f"runs/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)

    # 特征图可视化
    fm_visualizer = FeatureMapVisualizer(model)

    # 绘图初始化
    if plot_results :
        fig, axes, lines, metrics = setup_plotting()

    # 训练循环
    best_val_acc = 0.0
    no_improve = 0

    for epoch in range(epochs) :
        model.train()
        train_loss, train_top1, train_top3 = 0.0, 0.0, 0.0
        total = 0

        # 训练阶段
        for inputs, targets in train_loader :
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            # 计算指标
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_top1 += (preds == targets).sum().item()

            _, top3_preds = torch.topk(outputs, 3, dim=1)
            train_top3 += torch.sum(top3_preds == targets.view(-1, 1)).item()

            total += targets.size(0)

        # 验证阶段
        val_loss, val_top1, val_top3 = 0.0, 0.0, 0.0
        val_total = 0

        model.eval()
        with torch.no_grad() :
            for inputs, targets in val_loader :
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_top1 += (preds == targets).sum().item()

                _, top3_preds = torch.topk(outputs, 3, dim=1)
                val_top3 += torch.sum(top3_preds == targets.view(-1, 1)).item()

                val_total += targets.size(0)

        # 计算平均指标
        train_loss /= total
        train_top1 /= total
        train_top3 /= total
        val_loss /= val_total
        val_top1 /= val_total
        val_top3 /= val_total

        # 更新图表
        if plot_results :
            update_plot(fig, axes, lines, metrics, epoch,
                        train_loss, val_loss, train_top1, val_top1, train_top3, val_top3)

        # 特征图可视化
        if epoch % 10 == 0 :  # 每5个epoch可视化一次
            fm_visualizer.visualize(epoch, log_dir)

        # 保存最佳模型
        if val_top1 > best_val_acc :
            best_val_acc = val_top1
            no_improve = 0
            if save_best_only :
                torch.save(model.state_dict(), save_path)
                print(f"🎯 新最佳模型 (准确率: {val_top1:.4f}) 已保存到 {save_path}")
        else :
            no_improve += 1

        # 早停检查
        if no_improve >= patience :
            print(f"⏹️ 早停触发: 验证准确率连续 {patience} 轮未提升")
            break

        # 打印日志
        print(f"Epoch {epoch + 1}/{epochs}: "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train Top-1: {train_top1:.4f} | Val Top-1: {val_top1:.4f} | "
              f"Train Top-3: {train_top3:.4f} | Val Top-3: {val_top3:.4f}")

    # 清理
    fm_visualizer.remove_hook()
    writer.close()

    if plot_results :
        save_plot(fig)

    return model

