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
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from collections import Counter


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




