# multi_task_model.py

import torch
import torch.nn as nn

class MultiTaskModel(nn.Module):
    """
    1024차원 벡터 입력 -> (스타일, 대분류, 세분류) 예측
    - 스타일: multi-label -> BCEWithLogitsLoss
    - 대분류: single-label -> CrossEntropyLoss
    - 세분류: single-label -> CrossEntropyLoss
    """
    def __init__(self, 
                 input_dim=1024, 
                 style_dim=8, 
                 primary_dim=9, 
                 secondary_dim=33):
        super(MultiTaskModel, self).__init__()
        
        # 간단한 MLP 예시
        self.fc1 = nn.Linear(input_dim, 512)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.2)
        
        # 출력 레이어 3개 (멀티태스크)
        self.style_head = nn.Linear(256, style_dim)       # multi-label -> sigmoid, BCEWithLogitsLoss
        self.primary_head = nn.Linear(256, primary_dim)   # single-label -> softmax, CrossEntropyLoss
        self.secondary_head = nn.Linear(256, secondary_dim) # single-label -> softmax, CrossEntropyLoss

    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim=1024) 형태의 텐서
        Returns:
            style_logits, primary_logits, secondary_logits
            - style_logits: (batch_size, style_dim)
            - primary_logits: (batch_size, primary_dim)
            - secondary_logits: (batch_size, secondary_dim)
        """
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        
        style_logits = self.style_head(x)
        primary_logits = self.primary_head(x)
        secondary_logits = self.secondary_head(x)
        
        return style_logits, primary_logits, secondary_logits
