import torch.nn as nn
import timm

  
class SingleStageCNN(nn.Module):
    def __init__(self, num_secondary_classes):
        super().__init__()
        # self.backbone = timm.create_model(
        #     "swin_base_patch4_window7_224",
        #     pretrained=True,
        #     num_classes=0,        # FC 제거
        #     global_pool="avg"
        # )
        self.backbone = timm.create_model(
            "swin_small_patch4_window7_224",
            pretrained=True,
            num_classes=0,
            global_pool="avg"
        )
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_secondary_classes)
        )
        # 전체 Secondary 클래스를 한 번에 분류하는 헤드
        self.classifier = nn.Linear(in_features, num_secondary_classes)

    def forward(self, x):
        features = self.backbone(x)   # (B, in_features)
        logits = self.classifier(features)
        return logits  # shape: (B, num_secondary_classes)\
