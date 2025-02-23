import wandb
from sklearn.metrics import confusion_matrix
import torch
from utils.load_config import load_config

def map_secondary_to_primary(labels, primary_to_secondary):
    """1~33 라벨을 1~9 PRIMARY 라벨로 변환하는 함수"""
    secondary_to_primary = {}
    for primary, secondaries in primary_to_secondary.items():
        for sec in secondaries:
            secondary_to_primary[sec] = primary
    return [secondary_to_primary[label] for label in labels]

def log_confusion_matrix(preds, labels, class_names=None):
    config = load_config()
    PRIMARY_TO_SECONDARY = config["PRIMARY_TO_SECONDARY"]

    mapped_preds = map_secondary_to_primary(preds, PRIMARY_TO_SECONDARY)
    mapped_labels = map_secondary_to_primary(labels, PRIMARY_TO_SECONDARY)

    wandb.log({
        "primary_conf_mat": wandb.plot.confusion_matrix(
            probs=None,
            y_true=mapped_labels,
            preds=mapped_preds
        )
    })
    wandb.log({
        "secondary_conf_mat": wandb.plot.confusion_matrix(
            probs=None,
            y_true=labels,
            preds=preds,
            class_names=class_names
        )
    })
    
    
def log_epoch_metrics(epoch, train_loss, train_acc, val_loss, val_acc):
    wandb.log({
        "Epoch": epoch,
        "Loss/Train": train_loss,
        "Loss/Val": val_loss,
        "Accuracy/Train": train_acc,
        "Accuracy/Val": val_acc,
    }, step=epoch)

def log_val_check(model, val_dataset, config, device):
    model.eval()

    test_samples = min(config["check_val"], len(val_dataset))
    results = []

    for i in range(test_samples):
        product_id, img, sid = val_dataset[i]
        with torch.no_grad():
            logits = model(img.unsqueeze(0).to(device))
            pred = torch.argmax(logits, dim=1).item()
            pred += 1  # 보정 0~32 출력

        print(f"[Sample {i}]")
        print(f"   Product ID: {product_id}")
        print(f"   Real Category={real_category}")
        print(f"   Pred Category={pred_category}")
