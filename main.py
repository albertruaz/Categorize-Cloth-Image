import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pprint

from data.db_connector import ProductImageDataset, DBConnector
from utils.trainer import train_one_epoch, evaluate
from models.hierarchical_cnn import HierarchicalCNN, SingleStageCNN, ClothingClassifierCNN
# from config.config import PRIMARY_TO_SECONDARY

import json

def load_config(config_path="config/config.json"):
    with open(config_path, "r") as file:
        config = json.load(file)

    # PRIMARY_TO_SECONDARY의 key를 int로 변환
    config["PRIMARY_TO_SECONDARY"] = {int(k): v for k, v in config["PRIMARY_TO_SECONDARY"].items()}
    config["unfreeze_schedule"] = {int(k): v for k, v in config["unfreeze_schedule"].items()}
    
    return config

def main():
    config = load_config()
    print("Config Setting:")
    pprint.pprint(config, width=80, compact=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    

    db = DBConnector()
    # train_rows = db.get_product_data(where_condition="1=1", limit=6000, offset=0)
    # val_rows   = db.get_product_data(where_condition="1=1", limit=1500, offset=6000)
    train_rows = db.get_product_data(where_condition="1=1", limit=config["train_num"], offset=0)
    val_rows   = db.get_product_data(where_condition="1=1", limit=config["val_num"], offset=config["train_num"])

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = ProductImageDataset(train_rows, transform=train_transform)
    val_dataset   = ProductImageDataset(val_rows,   transform=val_transform)

    # DataLoader
    batch_size = config["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=True)

    # 모델 생성
    # model = HierarchicalCNN(PRIMARY_TO_SECONDARY, len(PRIMARY_TO_SECONDARY)).to(device)
    # secondary_all = []
    # for sub_list in config["PRIMARY_TO_SECONDARY"].values():
    #     secondary_all.extend(sub_list)
    # secondary_all = sorted(set(secondary_all))
    # num_secondary_classes = len(secondary_all)

    model = ClothingClassifierCNN().to(device)

    # 처음에는 backbone 파라미터 동결
    for param in model.backbone.parameters():
        param.requires_grad = False


    optimizer = optim.AdamW(
        model.classifier.parameters(), 
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )
    
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config["milestones"],
        gamma=config["gamma"]
    )


    # 학습 설정
    epochs = config["epochs"]
    unfreeze_schedule = config["unfreeze_schedule"]
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # epoch==5에서 backbone 언프리징 + LR 감소
        if epoch in unfreeze_schedule:
            unfreeze_ratio = unfreeze_schedule[epoch]
            print(f"==> Unfreezing {unfreeze_ratio*100:.0f}% of the backbone and lowering LR to 1e-4")
            
            total_layers = len(list(model.backbone.parameters()))
            num_layers_to_unfreeze = int(total_layers * unfreeze_ratio)

            for i, param in enumerate(model.backbone.parameters()):
                if i < num_layers_to_unfreeze:
                    param.requires_grad = True
            
            # Optimizer 재설정 (필요 시)
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                    lr=1e-4, weight_decay=1e-4)

            # Scheduler도 필요하면 변경
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        train_loss, train_s_acc = train_one_epoch(model, train_loader, optimizer, device, epoch_idx=epoch)
        val_loss, val_s_acc = evaluate(model, val_loader, device, epoch_idx=epoch)

        scheduler.step()

        print(f"[Epoch {epoch+1}/{epochs}]")
        print(f"  Train loss: {train_loss:.4f} | S-acc: {train_s_acc:.3f}")
        print(f"  Val   loss: {val_loss:.4f}   | S-acc: {val_s_acc:.3f}")

        # best model 갱신
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_single_stage.pth")
            print("  [Best model saved]")

    # 추후 inference 시에도 model(img.unsqueeze(0)) → (1, num_secondary_classes)
    # 로짓 argmax 해서 2차 클래스 직접 예측
    print("Training completed!")


    model.eval()
    test_samples = min(5, len(val_dataset))

    # 기존 코드
    # idx_to_primary = {v: k for k, v in train_dataset.primary_to_idx.items()}

    for i in range(test_samples):
        # val_dataset에서 product_id도 함께 반환하도록 수정된 상태
        img, product_id, secondary_label_local = val_dataset[i]
        with torch.no_grad():
            logits = model(img.unsqueeze(0).to(device))
            pred_s_idx = torch.argmax(logits, dim=1).item()
            pred_s_idx += 1 # 보정 0~32 출력

        print(f"[Sample {i}]")
        print(f"   Product ID: {product_id}")  # product_id 출력 추가
        print(f"   secondary(local)={secondary_label_local}")
        print(f"   Pred secondary(local)={pred_s_idx}")

if __name__ == '__main__':
    main()
