import torch
import torch.nn as nn
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, device, epoch_idx=0):
    model.train()
    ce_loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    correct_s = 0  # secondary 예측 정확도

    # for batch in tqdm(dataloader, desc=f"Train (Epoch {epoch_idx})", leave=False):
    for batch in dataloader:
        # 데이터셋에서 (img, primary_label, pid, secondary_label_local) 형태라면:
        imgs, _, secondary_labels_in_local = batch
        secondary_labels_in_local = secondary_labels_in_local - 1  

        imgs = imgs.to(device)
        secondary_labels_in_local = secondary_labels_in_local.to(device)

        # SingleStageCNN은 단일 logits만 반환
        logits = model(imgs)  # shape: (B, num_secondary_classes)

        # CrossEntropyLoss (2차 라벨에 대해서만 계산)
        loss = ce_loss_fn(logits, secondary_labels_in_local)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 통계 계산
        batch_size = imgs.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        preds = logits.argmax(dim=1)
        correct_s += (preds == secondary_labels_in_local).sum().item()

    avg_loss = total_loss / total_samples
    secondary_acc = correct_s / total_samples

    return avg_loss, secondary_acc


def evaluate(model, dataloader, device, epoch_idx=0):
    model.eval()
    ce_loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    correct_s = 0

    with torch.no_grad():
        # for batch in tqdm(dataloader, desc=f"Eval (Epoch {epoch_idx})", leave=False):
        for batch in dataloader:
            imgs, _, secondary_labels_in_local = batch
            secondary_labels_in_local = secondary_labels_in_local - 1  

            imgs = imgs.to(device)
            secondary_labels_in_local = secondary_labels_in_local.to(device)

            logits = model(imgs)
            loss = ce_loss_fn(logits, secondary_labels_in_local)

            batch_size = imgs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = logits.argmax(dim=1)
            correct_s += (preds == secondary_labels_in_local).sum().item()

    avg_loss = total_loss / total_samples
    secondary_acc = correct_s / total_samples

    return avg_loss, secondary_acc
