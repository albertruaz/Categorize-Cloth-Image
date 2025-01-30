# main.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np

# 같은 디렉토리에 있는 multi_task_model.py에서 import
from multi_task_model import MultiTaskModel

# 1) 예시용 Dataset 정의 (DB에서 직접 불러온다고 가정해도 동일함)
#    실제 환경에서는 DB에서 데이터를 로드해서 X, y_style, y_primary, y_secondary를 구성하면 됩니다.
class CustomFashionDataset(Dataset):
    def __init__(self, num_samples=60000, input_dim=1024, 
                 style_dim=8, primary_dim=9, secondary_dim=33):
        super().__init__()
        
        self.num_samples = num_samples
        
        # (1) 1024차원 임베딩 벡터 (랜덤 생성 예시)
        self.X = np.random.rand(num_samples, input_dim).astype(np.float32)

        # (2) 스타일 라벨 (multi-label, 0/1)
        #     실제로는 styleIds를 multi-hot 벡터로 변환해야 함
        self.y_style = np.random.randint(0, 2, size=(num_samples, style_dim)).astype(np.float32)

        # (3) 대분류 라벨 (single-label, one-hot 대신 CrossEntropy를 쓰므로 정수 클래스)
        #     예: [0..(primary_dim-1)] 범위의 정수
        self.y_primary = np.random.randint(0, primary_dim, size=(num_samples,)).astype(np.int64)

        # (4) 세분류 라벨 (single-label, 정수 클래스)
        self.y_secondary = np.random.randint(0, secondary_dim, size=(num_samples,)).astype(np.int64)

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # PyTorch 텐서 변환
        x = torch.from_numpy(self.X[idx])  # shape: (1024,)
        
        style_label = torch.from_numpy(self.y_style[idx]) # shape: (style_dim,)
        primary_label = torch.tensor(self.y_primary[idx]) # shape: (,)
        secondary_label = torch.tensor(self.y_secondary[idx]) # shape: (,)
        
        return x, style_label, primary_label, secondary_label


def train_one_epoch(model, dataloader, optimizer, device):
    """
    한 epoch 동안 학습을 진행하는 함수
    """
    model.train()
    
    # 손실함수 정의
    # - 스타일: 다중 라벨이므로 BCEWithLogitsLoss (시그모이드 내부 적용)
    # - 대분류, 세분류: single-label이므로 CrossEntropyLoss
    bce_loss_fn = nn.BCEWithLogitsLoss()      # 스타일용
    ce_loss_fn = nn.CrossEntropyLoss()        # 대분류, 세분류용

    total_loss = 0.0
    
    for batch in dataloader:
        x, style_label, primary_label, secondary_label = batch
        
        # GPU나 CPU로 이동
        x = x.to(device)
        style_label = style_label.to(device)
        primary_label = primary_label.to(device)
        secondary_label = secondary_label.to(device)
        
        # Forward
        style_logits, primary_logits, secondary_logits = model(x)
        
        # Loss 계산
        loss_style = bce_loss_fn(style_logits, style_label)  # BCEWithLogits
        loss_primary = ce_loss_fn(primary_logits, primary_label)   # CrossEntropy
        loss_secondary = ce_loss_fn(secondary_logits, secondary_label)
        
        loss = loss_style + loss_primary + loss_secondary
        
        # Backward & Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """
    검증/평가용 함수 (간단한 Loss만 계산 예시)
    """
    model.eval()
    bce_loss_fn = nn.BCEWithLogitsLoss()
    ce_loss_fn = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            x, style_label, primary_label, secondary_label = batch
            x = x.to(device)
            style_label = style_label.to(device)
            primary_label = primary_label.to(device)
            secondary_label = secondary_label.to(device)
            
            style_logits, primary_logits, secondary_logits = model(x)
            
            loss_style = bce_loss_fn(style_logits, style_label)
            loss_primary = ce_loss_fn(primary_logits, primary_label)
            loss_secondary = ce_loss_fn(secondary_logits, secondary_label)
            
            loss = loss_style + loss_primary + loss_secondary
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    # 하이퍼파라미터
    input_dim = 1024
    style_dim = 8
    primary_dim = 9
    secondary_dim = 33
    batch_size = 256
    epochs = 5
    lr = 1e-3
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    # (1) Dataset & DataLoader 생성
    train_dataset = CustomFashionDataset(num_samples=60000, 
                                         input_dim=input_dim, 
                                         style_dim=style_dim, 
                                         primary_dim=primary_dim, 
                                         secondary_dim=secondary_dim)
    val_dataset = CustomFashionDataset(num_samples=5000,   # 검증용
                                       input_dim=input_dim, 
                                       style_dim=style_dim, 
                                       primary_dim=primary_dim, 
                                       secondary_dim=secondary_dim)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # (2) 모델 초기화
    model = MultiTaskModel(input_dim=input_dim, 
                           style_dim=style_dim, 
                           primary_dim=primary_dim, 
                           secondary_dim=secondary_dim)
    model.to(device)

    # (3) 옵티마이저
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # (4) 학습 루프
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        
        print(f"[Epoch {epoch+1}/{epochs}] train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    
    # (5) 모델 저장
    torch.save(model.state_dict(), "multi_task_model.pth")
    print("Model saved to multi_task_model.pth")
    
    # (6) 모델 로드 후 예측 예시
    loaded_model = MultiTaskModel(input_dim, style_dim, primary_dim, secondary_dim)
    loaded_model.load_state_dict(torch.load("multi_task_model.pth", map_location=device))
    loaded_model.to(device)
    loaded_model.eval()
    
    # 테스트용 임의 입력 5개
    X_test = torch.rand((5, input_dim)).to(device)  # shape: (5, 1024)
    with torch.no_grad():
        style_logits, primary_logits, secondary_logits = loaded_model(X_test)
    
    # (7) 예측 결과 해석
    #     - 스타일: sigmoid -> 0.5 이상이면 예측 라벨 ON
    #     - 대분류: softmax -> argmax
    #     - 세분류: softmax -> argmax
    style_probs = torch.sigmoid(style_logits)  # (5, 8)
    primary_preds = torch.argmax(primary_logits, dim=1)  # (5,)
    secondary_preds = torch.argmax(secondary_logits, dim=1)  # (5,)

    print("\n=== 예측 결과 예시 ===")
    for i in range(5):
        style_vec = style_probs[i].cpu().numpy()
        style_labels = [idx for idx, prob in enumerate(style_vec) if prob > 0.5]
        print(f"[Sample {i}]")
        print(f"  - style={style_labels}")
        print(f"  - primary={primary_preds[i].item()}")
        print(f"  - secondary={secondary_preds[i].item()}")


if __name__ == "__main__":
    main()
