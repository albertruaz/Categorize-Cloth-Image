import torch
import torch.nn as nn
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, device, epoch_idx=0):
    model.train()
    ce_loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    correct_s = 0
    
    # 혼동 행렬 등 추가 지표용
    all_preds = []
    all_labels = []

    # for batch in tqdm(dataloader, desc=f"Train (Epoch {epoch_idx})", leave=False):
    for batch in dataloader:
        # product_id, img, sid
        _, imgs, sid = batch
        sid = sid - 1 # 0~32로 변환

        imgs = imgs.to(device)
        sid = sid.to(device)

        # SingleStageCNN은 단일 logits만 반환
        logits = model(imgs)  # shape: (B, num_secondary_classes)

        # CrossEntropyLoss (2차 라벨에 대해서만 계산)
        loss = ce_loss_fn(logits, sid)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 통계 계산
        batch_size = imgs.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        preds = logits.argmax(dim=1)
        correct_s += (preds == sid).sum().item()

        # 혼동 행렬용 데이터 수집
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(sid.cpu().numpy().tolist())

    avg_loss = total_loss / total_samples
    secondary_acc = correct_s / total_samples

    return avg_loss, secondary_acc, all_preds, all_labels


def evaluate(model, dataloader, device, epoch_idx=0):
    model.eval()
    ce_loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    correct_s = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        # for batch in tqdm(dataloader, desc=f"Eval (Epoch {epoch_idx})", leave=False):
        for batch in dataloader:
            _, imgs, sid = batch
            sid = sid - 1  

            imgs = imgs.to(device)
            sid = sid.to(device)

            logits = model(imgs)
            loss = ce_loss_fn(logits, sid)

            batch_size = imgs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = logits.argmax(dim=1)
            correct_s += (preds == sid).sum().item()

            # 혼동 행렬용 데이터 수집
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(sid.cpu().numpy().tolist())

    avg_loss = total_loss / total_samples
    secondary_acc = correct_s / total_samples

    return avg_loss, secondary_acc, all_preds, all_labels
