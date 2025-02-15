import torch
from torch.utils.data import Dataset
import requests
from PIL import Image
from io import BytesIO
from torchvision import transforms
from config.config import PRIMARY_TO_SECONDARY

class ProductImageDataset(Dataset):
    def __init__(self, data_rows, transform=None):
        super().__init__()
        self.transform = transform

        valid_rows = []
        for row in data_rows:
            pid = row.get('primary_category_id')
            sid = row.get('secondary_category_id')
            img_url = row.get('image_url')
            # 기본 null check
            if pid is None or sid is None or img_url is None:
                continue
            # dictionary에 존재하는 primary/secondary 조합만 사용
            if pid not in PRIMARY_TO_SECONDARY:
                continue
            if sid not in PRIMARY_TO_SECONDARY[pid]:
                continue
            valid_rows.append(row)

        self.data = valid_rows

        # Primary 카테고리 ID를 0~N-1 로 매핑
        primary_ids = sorted(list(set(d['primary_category_id'] for d in self.data)))
        self.primary_to_idx = {pid: i for i, pid in enumerate(primary_ids)} # category 숫자 목록 리스트
        self.num_primary_classes = len(primary_ids)

        # secondary_mapping: 각 pid마다 subcat list & index 매핑
        self.secondary_mapping = {}
        for pid in primary_ids:
            valid_subcats = PRIMARY_TO_SECONDARY[pid]
            self.secondary_mapping[pid] = {
                'subcats': valid_subcats,
                'subcat_to_idx': {sc: idx for idx, sc in enumerate(valid_subcats)}
            }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        image_url = row['image_url']

        # 이미지를 다운로드해서 PIL Image로 변환
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content)).convert("RGB")

        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        pid = row['primary_category_id']
        sid = row['secondary_category_id']

        primary_label = self.primary_to_idx[pid]
        sub_info = self.secondary_mapping[pid]
        secondary_label_local = sub_info['subcat_to_idx'][sid]

        return img, primary_label, pid, secondary_label_local

def get_data_transforms():
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
    
    return train_transform, val_transform 