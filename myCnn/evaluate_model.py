import time
from tqdm import tqdm
import torch

# 设置随机种子以确保结果可复现
torch.manual_seed(42)  # 你可以选择任何你喜欢的整数作为种子

def evaluate_model(model, test_loader, device, topk=(1, 3)):
    """
    Evaluate the model on the test dataset and calculate top-k accuracy and inference speed.

    Args:
        model (torch.nn.Module): The trained model.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to run the model on (CPU or GPU).
        topk (tuple): Tuple of integers representing the top-k accuracy to calculate.

    Returns:
        tuple: A tuple containing top-1 accuracy, top-3 accuracy, and inference speed (in seconds per sample).
    """
    model.eval()
    correct = {k: 0 for k in topk}
    total = 0
    total_time = 0.0
    top1_correct_test = 0
    top3_correct_test = 0
    total_test = 0

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

    val_top1_acc = 100 * top1_correct_test / total_test
    val_top3_acc = 100 * top3_correct_test / total_test
    inference_speed = total_time / total_test

    return val_top1_acc, val_top3_acc, inference_speed
