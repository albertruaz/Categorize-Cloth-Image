import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.load_config import load_config
import pprint

import wandb
from utils.logger import log_confusion_matrix, log_epoch_metrics, log_val_check
from sklearn.metrics import confusion_matrix

from data.db_connector import DBConnector, split_train_val
from data.dataset import ProductImageDataset

from utils.trainer import train_one_epoch, evaluate
from models.cnn import ClothingClassifierCNN


def main():
    config = load_config()
    print("Config Setting:")
    pprint.pprint(config, width=80, compact=True)
    
    # wandb 초기화
    wandb.init(
        project="clothing-classifier",
        entity="vingle",
        config=config,
        name=f"batch{config['batch_size']}_lr{config['lr']}"
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nUsing device:", device,"\n")
    
    # DB에서 데이터 추출
    db = DBConnector()
    datas = db.get_product_data(where_condition="1=1", x=config["data_num"])
    train_rows, val_rows = split_train_val(datas)
    
    print("Number of Train data",len(train_rows))
    print("Number of Val data",len(val_rows),"\n")

    # 추출 데이터 DataSet화
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])
    train_dataset = ProductImageDataset(train_rows, transform=transform)
    val_dataset   = ProductImageDataset(val_rows,   transform=transform)

    batch_size = config["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)

    # 모델 세팅
    model = ClothingClassifierCNN().to(device)

    for param in model.backbone.parameters():
        param.requires_grad = False

    # Hyper parameter
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

    epochs = config["epochs"]
    unfreeze_schedule = config["unfreeze_schedule"]
    best_val_loss = float('inf')


    # Train
    for epoch in range(epochs):
        if epoch in unfreeze_schedule:
            unfreeze_ratio = unfreeze_schedule[epoch]
            print(f"==> Unfreezing {unfreeze_ratio*100:.0f}% of the backbone and lowering LR to 1e-4")
            
            total_layers = len(list(model.backbone.parameters()))
            num_layers_to_unfreeze = int(total_layers * unfreeze_ratio)

            for i, param in enumerate(model.backbone.parameters()):
                if i < num_layers_to_unfreeze:
                    param.requires_grad = True
            
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                    lr=1e-4, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        # 학습 및 평가
        train_loss, train_s_acc, train_preds, train_labels = train_one_epoch(
            model, train_loader, optimizer, device, epoch_idx=epoch
        )
        val_loss, val_s_acc, val_preds, val_labels = evaluate(
            model, val_loader, device, epoch_idx=epoch
        )

        scheduler.step()

        # wandb 로깅
        log_epoch_metrics(epoch, train_loss, train_s_acc, val_loss, val_s_acc)
        
        # 5 에폭마다 혼동 행렬 로깅
        if (epoch + 1) % 5 == 0:
            log_confusion_matrix(
                val_preds, 
                val_labels, 
                class_names=[str(i) for i in range(33)]
            )

        print(f"[Epoch {epoch+1}/{epochs}]")
        print(f"  Train loss: {train_loss:.4f} | S-acc: {train_s_acc:.3f}")
        print(f"  Val   loss: {val_loss:.4f}   | S-acc: {val_s_acc:.3f}")

        # best model 갱신
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_single_stage.pth")
            print("  [Best model saved]")

    # 학습 완료 후 최종 혼동 행렬 로깅
    log_confusion_matrix(
        val_preds, 
        val_labels, 
        class_names=[str(i) for i in range(33)]
    )
    
    wandb.finish()
    
    log_val_check(model, val_dataset, config, device)

if __name__ == '__main__':
    main()
