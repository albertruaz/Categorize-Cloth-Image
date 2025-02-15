import torch
from torch.utils.data import DataLoader
from data.db_connector import DBConnector
from data.dataset import ProductImageDataset, get_data_transforms
from models.hierarchical_cnn import HierarchicalCNN
from utils.trainer import Trainer
from config.config import TRAIN_CONFIG, DATA_CONFIG

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # DB 연결 및 데이터 로드
    db = DBConnector()
    train_rows = db.get_product_data(
        where_condition="1=1", 
        limit=DATA_CONFIG['train_limit'], 
        offset=DATA_CONFIG['train_offset']
    )
    val_rows = db.get_product_data(
        where_condition="1=1", 
        limit=DATA_CONFIG['val_limit'], 
        offset=DATA_CONFIG['val_offset']
    )

    # Transform 및 Dataset 생성
    train_transform, val_transform = get_data_transforms()
    train_dataset = ProductImageDataset(train_rows, transform=train_transform)
    val_dataset = ProductImageDataset(val_rows, transform=val_transform)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val   dataset size: {len(val_dataset)}")

    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset, 
        batch_size=TRAIN_CONFIG['batch_size'], 
        shuffle=True,
        num_workers=TRAIN_CONFIG['num_workers']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=False,
        num_workers=TRAIN_CONFIG['num_workers']
    )

    # 모델 생성
    model = HierarchicalCNN(
        PRIMARY_TO_SECONDARY, 
        train_dataset.num_primary_classes
    ).to(device)

    # 학습 실행
    trainer = Trainer(model, train_loader, val_loader, TRAIN_CONFIG, device)
    trainer.train()

if __name__ == '__main__':
    main() 