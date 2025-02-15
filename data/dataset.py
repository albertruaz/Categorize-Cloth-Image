import torch
from torch.utils.data import Dataset
import requests
from PIL import Image
from io import BytesIO
from torchvision import transforms
from config.config import PRIMARY_TO_SECONDARY

class ProductImageDataset(Dataset):
    # ... (기존 ProductImageDataset 클래스 코드)
    pass

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