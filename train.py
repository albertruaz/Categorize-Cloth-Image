import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms

from data.db_connector import ProductImageDataset, DBConnector
from utils.trainer import train_one_epoch, evaluate
from models.hierarchical_cnn import HierarchicalCNN
from config.config import PRIMARY_TO_SECONDARY

def main():
    # GPU 사용 가능하면 GPU로, 아니면 CPU로 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # DB 연결 및 데이터 가져오기
    db = DBConnector()
    train_rows = db.get_product_data(where_condition="1=1", limit=6000, offset=0)
    val_rows   = db.get_product_data(where_condition="1=1", limit=1500, offset=6000)

    # 데이터 증강/전처리 설정
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Dataset
    train_dataset = ProductImageDataset(train_rows, transform=train_transform)
    val_dataset   = ProductImageDataset(val_rows,   transform=val_transform)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val   dataset size: {len(val_dataset)}")

    # DataLoader
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    # 모델 생성
    num_primary_classes = train_dataset.num_primary_classes
    model = HierarchicalCNN(PRIMARY_TO_SECONDARY, num_primary_classes).to(device)

    # 처음에는 backbone 파라미터 동결
    for param in model.backbone.parameters():
        param.requires_grad = False

    # primary_head & secondary_heads만 학습
    params_to_update = list(model.primary_head.parameters())
    for key in model.secondary_heads.keys():
        params_to_update += list(model.secondary_heads[key].parameters())

    optimizer = optim.Adam(params_to_update, lr=1e-3)
    epochs = 5
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # 특정 에포크부터 backbone도 학습
        if epoch == 2:
            print("==> Unfreezing the backbone and lowering LR to 1e-4")
            for param in model.backbone.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=1e-4)

        # 한 epoch 학습
        train_loss, train_p_acc, train_s_acc = train_one_epoch(model, train_loader, optimizer, device, epoch_idx=epoch)

        # 검증
        val_loss, val_p_acc, val_s_acc = evaluate(model, val_loader, device, epoch_idx=epoch)

        # 콘솔 출력
        print(f"[Epoch {epoch+1}/{epochs}]")
        print(f"  Train loss: {train_loss:.4f} | P-acc: {train_p_acc:.3f}, S-acc: {train_s_acc:.3f}")
        print(f"  Val   loss: {val_loss:.4f}   | P-acc: {val_p_acc:.3f}, S-acc: {val_s_acc:.3f}")

        # 베스트 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_hierarchical_cnn.pth")
            print("  [Best model saved]")

    print("Training done!")

    # 예시 예측 (검증 데이터 중 일부)
    model.eval()
    test_samples = min(5, len(val_dataset))

    # primary 인덱스 -> 실제 primary_id
    idx_to_primary = {v: k for k, v in train_dataset.primary_to_idx.items()}

    for i in range(test_samples):
        img, primary_label, pid, secondary_label_local = val_dataset[i]
        with torch.no_grad():
            # 이미지도 GPU로 옮겨 예측
            p_logits, feat = model(img.unsqueeze(0).to(device))
            pred_p_idx = torch.argmax(p_logits, dim=1).item()
            pred_pid = idx_to_primary[pred_p_idx]

            sub_head = model.secondary_heads[str(pred_pid)]
            logit_s = sub_head(feat)
            pred_s_idx = torch.argmax(logit_s, dim=1).item()

        print(f"[Sample {i}]")
        print(f"   GT primary={pid}, GT secondary(local)={secondary_label_local}")
        print(f"   Pred primary={pred_pid}, Pred secondary(local)={pred_s_idx}")

if __name__ == '__main__':
    main()
