import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.load_config import load_config
import pprint

import wandb
from utils.logger import log_confusion_matrix, log_epoch_metrics, log_val_check, save_image
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
        name=config['name']
    )

    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\nUsing device:", device,"\n")
    
    # DB에서 데이터 추출
    db = DBConnector()
    datas = db.get_product_data(where_condition="1=1", x=config["data_num"])
    train_rows, val_rows, num_for_class, train_num_for_class, val_num_for_class = split_train_val(datas)
    
    print("Number of Train data",len(train_rows))
    print("Number of Val data",len(val_rows))
    print("Number of classes",num_for_class)
    print("Number of Train data for each class",train_num_for_class)
    print("Number of Val data for each class",val_num_for_class,"\n")

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # 모델 세팅
    model = ClothingClassifierCNN(config["hidden_layer"]).to(device)
    
    backbone_params = list(model.backbone.named_parameters())
    classifier_params = list(model.classifier.parameters())
    
    for name, param in backbone_params:
        param.requires_grad = False

    # Initially only train classifier params
    optimizer = optim.AdamW(
        classifier_params,
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    epochs = config["epochs"]
    best_val_loss = float('inf')
    total_backbone_layers = len(backbone_params)

    print("Training Start")
    for epoch in range(epochs):
        if epoch in config["unfreeze"]:
            unfreeze_ratio = config["unfreeze"][epoch]
            num_layers_to_unfreeze = int(total_backbone_layers * unfreeze_ratio)
            newly_unfrozen_layers = backbone_params[total_backbone_layers - num_layers_to_unfreeze : total_backbone_layers - frozen_until]
            if newly_unfrozen_layers:
                print(f"\n[Epoch {epoch}] Unfreezing ratio {unfreeze_ratio:.1f} =>"
                    f" {len(newly_unfrozen_layers)} layers 새로 unfreeze")

                for name, param in newly_unfrozen_layers:
                    param.requires_grad = True
                current_lr = optimizer.param_groups[0]["lr"]
                new_lr = current_lr * 0.1  # 예: 기존의 10분의 1로
                print(f"새로 unfrozen된 레이어에는 lr={new_lr:.6f} 적용")
                optimizer.add_param_group({
                    "params": [param for _, param in newly_unfrozen_layers],
                    "lr": new_lr,
                    "weight_decay": config["weight_decay"]
                })
                new_unfrozen_count = total_backbone_layers - (total_backbone_layers - num_layers_to_unfreeze)
                frozen_until = max(frozen_until, new_unfrozen_count)

        # 학습 및 평가
        train_loss, train_s_acc, train_preds, train_labels = train_one_epoch(
            model, train_loader, optimizer, device, num_for_class, epoch_idx=epoch
        )
        val_loss, val_s_acc, val_preds, val_labels = evaluate(
            model, val_loader, device, num_for_class, epoch_idx=epoch
        )

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
