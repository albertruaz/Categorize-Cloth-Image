import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from torchvision.utils import save_image

import wandb  # 필요 시
from utils.load_config import load_config
from utils.logger import log_confusion_matrix, log_val_check, save_mismatch_cases, save_image, download_images_from_list
from data.db_connector import DBConnector, split_train_val
from data.dataset import ProductImageDataset
from models.cnn import ClothingClassifierCNN
from utils.trainer import evaluate  # train_one_epoch는 test에는 필요 없음

def main():
    config = load_config()
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    db = DBConnector()
    datas = db.get_product_data(where_condition="1=1", x=10)
    _, val_rows, num_for_class, _, _ = split_train_val(datas)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std =[0.229, 0.224, 0.225]
        # )
    ])

    val_dataset = ProductImageDataset(val_rows, transform=transform)
    val_loader  = DataLoader(
        val_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True,
        num_workers=2, 
        pin_memory=True
    )

    model = ClothingClassifierCNN(config["hidden_layer"]).to(device)
    model.load_state_dict(torch.load("best_single_stage.pth"))
    model.eval()
    print("\nLoaded best model weights. Start evaluating...")

    val_loss, val_s_acc, val_ids, val_preds, val_labels = evaluate(
        model, 
        val_loader, 
        device, 
        num_for_class, 
        epoch_idx=0
    )

    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_s_acc:.3f}")

    # 9. 예측-라벨이 불일치하는 ID 리스트 뽑기
    #    (preds, labels)가 (2, 3)처럼 다른 경우, 해당 sample의 product_id를 뽑는다.
    mismatch_dict = {}
    for pid, pred, label in zip(val_ids, val_preds, val_labels):
        if pred != label:
            mismatch_key = (pred, label)
            if mismatch_key not in mismatch_dict:
                mismatch_dict[mismatch_key] = []
            mismatch_dict[mismatch_key].append(pid)

    sorted_mismatches = sorted(mismatch_dict.items(), key=lambda x: len(x[1]), reverse=True)
    save_mismatch_cases(sorted_mismatches, config["SECONDARY_TO_KOREAN"])
    download_list = sorted_mismatches[0][1]
    download_images_from_list(val_dataset, download_list, config)


if __name__ == "__main__":
    main()
