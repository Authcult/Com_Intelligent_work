import torch
import torch.utils.data as Data
from torchvision import datasets
import matplotlib.pyplot as plt


# 定义划分数据集的函数
def split_dataset(root_dir, transform, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, batch_size=128, shuffle=True, random_seed=42):
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

    # 加载整个数据集
    full_dataset = datasets.ImageFolder(root=root_dir, transform=transform)

    # 计算每个子集的大小
    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    # 随机划分数据集
    train_dataset, val_dataset, test_dataset = Data.random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(random_seed))

    # 创建DataLoader
    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = Data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, val_loader, test_loader, full_dataset

# 绘制图像对比top1、top3准确率、推理速度、模型大小
def plot_comparison(model_names, top1_accs, top3_accs, inference_speeds, model_sizes):
    """
    绘制图像对比top1、top3准确率、推理速度、模型大小

    参数:
    model_names (list): 模型名称列表
    top1_accs (list): Top-1 准确率列表
    top3_accs (list): Top-3 准确率列表
    inference_speeds (list): 推理速度列表
    model_sizes (list): 模型大小列表
    """
    # 创建一个包含4个子图的图形
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # 绘制Top-1 准确率
    axs[0, 0].bar(model_names, top1_accs, color='skyblue')
    axs[0, 0].set_title('Top-1 准确率')
    axs[0, 0].set_ylabel('准确率 (%)')
    axs[0, 0].set_xlabel('模型名称')
    axs[0, 0].tick_params(axis='x', rotation=45)

    # 绘制Top-3 准确率
    axs[0, 1].bar(model_names, top3_accs, color='salmon')
    axs[0, 1].set_title('Top-3 准确率')
    axs[0, 1].set_ylabel('准确率 (%)')
    axs[0, 1].set_xlabel('模型名称')
    axs[0, 1].tick_params(axis='x', rotation=45)

    # 绘制推理速度
    axs[1, 0].bar(model_names, inference_speeds, color='lightgreen')
    axs[1, 0].set_title('推理速度 (ms)')
    axs[1, 0].set_ylabel('时间 (ms)')
    axs[1, 0].set_xlabel('模型名称')
    axs[1, 0].tick_params(axis='x', rotation=45)

    # 绘制模型大小
    axs[1, 1].bar(model_names, model_sizes, color='gold')
    axs[1, 1].set_title('模型大小 (MB)')
    axs[1, 1].set_ylabel('大小 (MB)')
    axs[1, 1].set_xlabel('模型名称')
    axs[1, 1].tick_params(axis='x', rotation=45)

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示图形
    plt.show()

