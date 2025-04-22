import time
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# 显示中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子以确保结果可复现
torch.manual_seed(42)  # 你可以选择任何你喜欢的整数作为种子

def evaluate_model(model, test_loader, device, topk=(1, 3)):
    """
    在测试数据集上评估模型，并计算 top-k 准确率和推理速度。

    参数:
        model (torch.nn.Module): 训练好的模型。
        test_loader (torch.utils.data.DataLoader): 测试数据集的 DataLoader。
        device (torch.device): 运行模型的设备（CPU 或 GPU）。
        topk (tuple): 表示要计算的 top-k 准确率的整数元组。

    返回:
        tuple: 包含 top-1 准确率、top-3 准确率和每个样本的推理速度（秒）的元组。
    """
    model.eval()
    correct = {k: 0 for k in topk}
    total = 0
    total_time = 0.0
    top1_correct_test = 0
    top3_correct_test = 0
    total_test = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            start_time = time.time()
            outputs = model(images)
            end_time = time.time()
            total_time += (end_time - start_time)

            _, top1_pred = torch.max(outputs, 1)
            top1_correct_test += (top1_pred == labels).sum().item()

            _, top3_pred_indices = torch.topk(outputs, 3, dim=1)
            top3_correct_test += torch.sum(top3_pred_indices == labels.view(-1, 1).expand_as(top3_pred_indices)).item()

            total_test += labels.size(0)

            all_preds.extend(top1_pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_top1_acc = 100 * top1_correct_test / total_test
    val_top3_acc = 100 * top3_correct_test / total_test
    inference_speed = total_time / total_test

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 获取类别信息
    if isinstance(test_loader.dataset, torch.utils.data.Subset):
        classes = test_loader.dataset.dataset.classes
    else:
        classes = test_loader.dataset.classes

    # 绘制混淆矩阵
    plot_confusion_matrix(cm, classes)

    return val_top1_acc, val_top3_acc, inference_speed

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
