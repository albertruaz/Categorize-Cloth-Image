import wandb
from sklearn.metrics import confusion_matrix
import torch
from utils.load_config import load_config
from torchvision.transforms import ToTensor, ToPILImage
import json
import os
import os
from torchvision.utils import save_image

def map_secondary_to_primary(labels, primary_to_secondary):
    """1~33 라벨을 1~9 PRIMARY 라벨로 변환하는 함수"""
    secondary_to_primary = {}
    for primary, secondaries in primary_to_secondary.items():
        for sec in secondaries:
            secondary_to_primary[sec-1] = primary
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


# def save_image(img, name):
#     img_pil = ToPILImage()(img)
#     img_pil.save(f"sample_{name}.png")
#     print(f"Saved image: sample_{name}.png")


def download_images_from_list(train_dataset, download_list, config, save_dir="sample_downloaded_images"):
    
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert download_list to set for faster lookup
    download_set = set(download_list)
    
    # Iterate through dataset
    for idx in range(len(train_dataset)):
        product_id, img, sid = train_dataset[idx]
        
        if product_id in download_set:
            # Create filename with product ID and category (확장자 중복 제거)
            filename = f"{product_id}_Category-{config['SECONDARY_TO_KOREAN'][sid]}"
            filepath = os.path.join(save_dir, filename + ".png")  # 확장자는 여기서 한 번만 추가
            
            # Save the image
            save_image(img, filepath)
            print(f"Saved image: {filename}")
            
            # Remove from set to track progress
            download_set.remove(product_id)
            
            # If we've found all images, we can stop
            if not download_set:
                break
    
    # Report any products that weren't found
    if download_set:
        print("\nCouldn't find images for following product IDs:")
        for remaining_id in download_set:
            print(remaining_id)


def save_mismatch_cases(sorted_mismatches, secondary_to_korean, json_filename="mismatch_cases.json", txt_filename="mismatch_cases.txt"):
    
    # JSON 형식으로 변환
    json_data = {
        f"예측: {secondary_to_korean.get(p+1, p+1)}, 실제: {secondary_to_korean.get(l+1, l+1)}": ids
        for (p, l), ids in sorted_mismatches
    }

    # JSON 저장
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
    
    # TXT 저장
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write("\n[Mismatch Cases] (리스트 크기순 정렬)\n")
        for (p, l), ids in sorted_mismatches:
            pred_korean = secondary_to_korean.get(p, f"ID {p+1}")
            label_korean = secondary_to_korean.get(l, f"ID {l+1}")
            f.write(f"  예측: {pred_korean}, 실제 라벨: {label_korean} => ID 리스트({len(ids)}개): {ids}\n")

    print(f"Mismatch cases saved to {json_filename} and {txt_filename}")