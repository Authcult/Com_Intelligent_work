import torch
from tensorboardX import SummaryWriter
import logging
import os

# 配置日志记录器
log_file = 'training.log'
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None,
                num_epochs=10, save_path='best_model.pth', save_best_only=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    model.to(device)
    writer = SummaryWriter()

    best_val_top1_acc = 0.0
    logging.info("Starting training...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        top1_correct_train = 0
        top3_correct_train = 0
        total_train = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 按 batch 更新的调度器（如 OneCycleLR）
            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

            running_loss += loss.item() * images.size(0)

            _, top1_pred = torch.max(outputs, 1)
            top1_correct_train += (top1_pred == labels).sum().item()

            _, top3_pred_indices = torch.topk(outputs, 3, dim=1)
            top3_correct_train += torch.sum(top3_pred_indices == labels.view(-1, 1).expand_as(top3_pred_indices)).item()

            total_train += labels.size(0)

        epoch_train_loss = running_loss / total_train
        train_top1_acc = 100 * top1_correct_train / total_train
        train_top3_acc = 100 * top3_correct_train / total_train

        # 按 epoch 更新的调度器（如 CosineAnnealingLR）
        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        # --- Validation Phase ---
        model.eval()
        top1_correct_val = 0
        top3_correct_val = 0
        total_val = 0
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                _, top1_pred = torch.max(outputs, 1)
                top1_correct_val += (top1_pred == labels).sum().item()

                _, top3_pred_indices = torch.topk(outputs, 3, dim=1)
                top3_correct_val += torch.sum(top3_pred_indices == labels.view(-1, 1).expand_as(top3_pred_indices)).item()

                total_val += labels.size(0)

        epoch_val_loss = val_loss / total_val
        val_top1_acc = 100 * top1_correct_val / total_val
        val_top3_acc = 100 * top3_correct_val / total_val

        log_message = (
            f'Epoch [{epoch + 1}/{num_epochs}], '
            f'Train Loss: {epoch_train_loss:.4f}, '
            f'Train Top-1: {train_top1_acc:.2f}%, '
            f'Train Top-3: {train_top3_acc:.2f}%, | '
            f'Val Loss: {epoch_val_loss:.4f}, '
            f'Val Top-1: {val_top1_acc:.2f}%, '
            f'Val Top-3: {val_top3_acc:.2f}%'
        )
        logging.info(log_message)

        writer.add_scalar('Loss/Train', epoch_train_loss, epoch)
        writer.add_scalar('Accuracy/Train_Top1', train_top1_acc, epoch)
        writer.add_scalar('Accuracy/Train_Top3', train_top3_acc, epoch)
        writer.add_scalar('Loss/Validation', epoch_val_loss, epoch)
        writer.add_scalar('Accuracy/Validation_Top1', val_top1_acc, epoch)
        writer.add_scalar('Accuracy/Validation_Top3', val_top3_acc, epoch)

        if save_best_only and val_top1_acc > best_val_top1_acc:
            best_val_top1_acc = val_top1_acc
            torch.save(model.state_dict(), save_path)
            logging.info(f"Epoch {epoch + 1}: New best model saved to {save_path} with Val Top-1 Acc: {best_val_top1_acc:.2f}%")

    logging.info("Training finished.")
    writer.close()

    if not save_best_only:
        torch.save(model.state_dict(), save_path)
        logging.info(f"Final model saved to {save_path}")

    return model
