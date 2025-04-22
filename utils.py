from torchvision.datasets import EMNIST
from PIL import Image
import os
from IPython.display import display

def save_emnist_images_with_class_previews(Edataset = "balanced", save_dir="emnist_png_balanced", max_images_per_class=1000, preview=True, preview_limit_per_class=10):
    digits = [str(i) for i in range(10)]
    letters = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
        'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
    ]
    if Edataset == "letters":
         digits = []  # 字母数据集不需要数字
    all_labels = digits + letters

    # 修改后的函数，区分大小写文件夹名
    def label_char_to_folder_name(label_char):
        if label_char.islower():
            return label_char + "_"   # 小写加下划线后缀
        else:
            return label_char         # 大写和数字保持不变

    def label_to_char(label):
        if 0 <= label < len(all_labels):
            return all_labels[label]
        else:
            return "UNK"

    # 先处理预览部分：读取时用转换后的文件夹名
    if os.path.exists(save_dir) and os.path.isdir(save_dir):
        print(f"检测到目录 {save_dir} 已存在，跳过下载，直接展示预览图...")
        class_preview_images = {}
        for label_char in all_labels:
            folder_name = label_char_to_folder_name(label_char)
            label_dir = os.path.join(save_dir, folder_name)
            imgs = []
            if os.path.exists(label_dir) and os.path.isdir(label_dir):
                files = sorted(os.listdir(label_dir))
                for f in files[:preview_limit_per_class]:
                    try:
                        im = Image.open(os.path.join(label_dir, f)).convert('L')
                        imgs.append(im)
                    except:
                        pass
            if imgs:
                class_preview_images[label_char] = imgs
        if preview:
            for label_char, imgs in class_preview_images.items():
                width = 28 * len(imgs)
                height = 28
                big_img = Image.new('L', (width, height))
                for idx, im in enumerate(imgs):
                    big_img.paste(im, (28*idx, 0))
                print(f"类别 {label_char} 预览图 (共 {len(imgs)} 张):")
                display(big_img)

        print("预览完毕。")
        return

    # 下载并保存图像
    print(f"目录 {save_dir} 不存在，开始下载数据并保存...")
    train_set = EMNIST(root="emnist_data", split=Edataset, train=True, download=True)
    test_set = EMNIST(root="emnist_data", split=Edataset, train=False, download=True)
    full_dataset = train_set + test_set

    os.makedirs(save_dir, exist_ok=True)

    images_per_class = {label_char: 0 for label_char in all_labels}
    saved_images_count = 0

    class_preview_images = {label_char: [] for label_char in all_labels} if preview else None

    for i, (img, label) in enumerate(full_dataset):
        label_char = label_to_char(label)
        if label_char in digits:
            continue
        if label_char == "UNK":
            continue
        if images_per_class[label_char] >= max_images_per_class:
            continue

        folder_name = label_char_to_folder_name(label_char)
        label_dir = os.path.join(save_dir, folder_name)
        os.makedirs(label_dir, exist_ok=True)

        img = img.resize((28, 28))
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT).rotate(90)
        img.save(os.path.join(label_dir, f"{images_per_class[label_char]}.png"))

        if preview and len(class_preview_images[label_char]) < preview_limit_per_class:
            class_preview_images[label_char].append(img)

        images_per_class[label_char] += 1
        saved_images_count += 1

        if saved_images_count % 500 == 0:
            print(f"已保存 {saved_images_count} 张图像...")

    print("所有图像已保存到", save_dir)

    if preview:
        for label_char, imgs in class_preview_images.items():
            if len(imgs) == 0:
                continue
            width = 28 * len(imgs)
            height = 28
            big_img = Image.new('L', (width, height))
            for idx, im in enumerate(imgs):
                big_img.paste(im, (28*idx, 0))
            print(f"类别 {label_char} 预览图 (共 {len(imgs)} 张):")
            display(big_img)

import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
import torchvision.models as models
from torchvision import utils
import torchvision.transforms as T
import torch.utils.data as Data
from PIL import Image
import numpy as np
import torch.optim as optim
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter
from torch.utils.data import Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
#使用tensorboardX进行可视化
from tensorboardX import SummaryWriter
from torchvision import datasets
from torch.utils.data import random_split, DataLoader
#数据增强
from albumentations.pytorch import ToTensorV2
from torchvision.transforms.autoaugment import AutoAugmentPolicy #自动数据增强库
import albumentations as A


class AlbumentationsTransform:
    def __init__(self):
        self.transform=A.Compose([
            A.Resize(28, 28),
            A.Rotate(limit=15, p=0.5),
            A.Affine(translate_percent=(0.1,0.1),p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=(0.5,),std=(0.5,)),
            ToTensorV2()
        ])
    def __call__(self, img):
        img=np.array(img.convert('L'))
        return self.transform(image=img)['image']

class AlbumentationsTransformBase:
    def __init__(self):
        self.transform = A.Compose([
            A.Resize(28, 28),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2()
        ])
    def __call__(self, img):
        img = np.array(img.convert('L'))
        return self.transform(image=img)['image']


def split_dataset(root_dir, transform, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, batch_size=128, shuffle=True,
                  random_seed=42):
    """
    划分数据集为训练集、验证集和测试集，并返回对应的DataLoader。

    参数:
    - root_dir: 数据集根目录
    - transform: 数据预处理变换
    - train_ratio: 训练集比例
    - val_ratio: 验证集比例
    - test_ratio: 测试集比例
    - batch_size: 批次大小
    - shuffle: 是否打乱数据
    - random_seed: 随机种子，用于保证结果可重复

    返回:
    - train_loader: 训练集DataLoader
    - val_loader: 验证集DataLoader
    - test_loader: 测试集DataLoader
    - full_dataset: 原始的ImageFolder数据集
    """
    # 确保比例之和为1
    assert train_ratio + val_ratio + test_ratio == 1, "比例之和必须为1"

    full_dataset = datasets.ImageFolder(root=root_dir, transform=transform)
    targets = full_dataset.targets

    class_indices = {}
    for idx, label in enumerate(targets):
        class_indices.setdefault(label, []).append(idx)

    train_indices = []
    val_indices = []
    test_indices = []

    generator = torch.Generator().manual_seed(random_seed)

    for label, indices in class_indices.items():
        indices = torch.tensor(indices)
        indices = indices[torch.randperm(len(indices), generator=generator)]

        n_total = len(indices)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)
        n_test = n_total - n_train - n_val

        train_indices.extend(indices[:n_train].tolist())
        val_indices.extend(indices[n_train:n_train + n_val].tolist())
        test_indices.extend(indices[n_train + n_val:].tolist())

    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    test_subset = Subset(full_dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    # 统计每个类别数量的函数
    def count_classes(indices, targets):
        label_counts = Counter()
        for idx in indices:
            label_counts[targets[idx]] += 1
        return label_counts

        # 打印每个类别样本数

    train_counts = count_classes(train_indices, targets)
    val_counts = count_classes(val_indices, targets)
    test_counts = count_classes(test_indices, targets)

    classes = full_dataset.classes  # 类别名称列表

    print("训练集每个类别样本数：")
    for label_idx, count in sorted(train_counts.items()):
        print(f"类别 {classes[label_idx]}: {count}")

    print("\n验证集每个类别样本数：")
    for label_idx, count in sorted(val_counts.items()):
        print(f"类别 {classes[label_idx]}: {count}")

    print("\n测试集每个类别样本数：")
    for label_idx, count in sorted(test_counts.items()):
        print(f"类别 {classes[label_idx]}: {count}")

    return train_loader, val_loader, test_loader, full_dataset

# 轻量CNN结构？
class SimpleAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        # Squeeze操作：全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation操作：两个全连接层
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()  # 输入形状 [B, C, H, W]
        # Squeeze
        y = self.avg_pool(x).view(b, c)  # [B, C]
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)  # [B, C, 1, 1]
        # Reweight
        return x * y.expand_as(x)  # 广播乘法 [B, C, H, W]

# 自注意力模块
class SelfAttention2D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key   = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习的缩放系数

    def forward(self, x):
        B, C, H, W = x.shape
        proj_q = self.query(x).view(B, -1, H * W)          # B x C1 x N
        proj_k = self.key(x).view(B, -1, H * W)            # B x C1 x N
        proj_v = self.value(x).view(B, -1, H * W)          # B x C  x N

        attention = torch.bmm(proj_q.permute(0, 2, 1), proj_k)  # B x N x N
        attention = torch.softmax(attention, dim=-1)

        out = torch.bmm(proj_v, attention.permute(0, 2, 1))     # B x C x N
        out = out.view(B, C, H, W)

        return self.gamma * out + x

# 添加残差模块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 3, padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(),
            nn.Conv2d(in_channels//2, in_channels//2, 3, padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(),
            nn.Conv2d(in_channels//2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        #调用注意力机制
        self.attn = SimpleAttention(out_channels) if use_attention else nn.Identity()
        # self.attn = CBAM(out_channels) if use_attention else nn.Identity()

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        res = self.shortcut(x)
        out = self.conv(x)
        out = self.attn(out)
        return self.relu(out + res)

# 加入倒残差模块
class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_ratio=6, stride=1, use_attention=False):
        super().__init__()
        hidden_dim = in_channels * expansion_ratio
        self.use_res_connect = (stride == 1 and in_channels == out_channels)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        # 调用注意力模块
        # self.attn = CBAM(out_channels) if use_attention else nn.Identity()
        self.attn = SimpleAttention(out_channels) if use_attention else nn.Identity()

    def forward(self, x):
        out = self.block(x)
        out = self.attn(out)
        if self.use_res_connect:
            return x + out
        else:
            return out


class CNNWithAttention(nn.Module):
    def __init__(self, num_classes, use_attention=True):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )

        # 插入残差和倒残差模块
        if use_attention:
            self.residual = ResidualBlock(64, 64, use_attention=True)
            self.inverted_residual = InvertedResidualBlock(64, 64, expansion_ratio=6, use_attention=True)
            self.attention = SelfAttention2D(64)
        else:
            self.residual = ResidualBlock(64, 64, use_attention=False)
            self.inverted_residual = InvertedResidualBlock(64, 64, expansion_ratio=6, use_attention=False)
            self.attention = nn.Identity()

        self.fcon1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 输出形状 [B, 64, 1, 1]
            nn.Flatten(),  # 展平为 [B, 64]
            nn.Dropout(0.4),
            nn.Linear(64, 148),  # 输入维度64
            nn.LeakyReLU(),
        )
        self.fcon2 = nn.Linear(148, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.residual(x)
        # x = self.inverted_residual(x)
        x = self.attention(x)
        x = self.fcon1(x)
        x = self.fcon2(x)
        return x

# 训练与验证函数
def train_and_validate(model, train_loader, val_loader, epochs, device='cpu',
                       save_path='cnn_attention_best.pth', save_best_only=True, patience=15, auto_lr = False, lr = 1e-4, LOG_DIR = f"runs/handwriting_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=LOG_DIR)

    best_val_top1_acc = 0.0  # 用于追踪验证集 Top-1 准确率的最佳值
    no_improve_count = 0

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

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
