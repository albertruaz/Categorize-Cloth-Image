import torch.nn as nn
import timm

class ClothingClassifierCNN(nn.Module):
    def __init__(self, num_secondary_classes=33, freeze_backbone=False):
        super().__init__()
        # ECAResNet50d Backbone
        # self.backbone = timm.create_model(
        #     'ecaresnet50d',
        #     pretrained=True,
        #     num_classes=0,     # 최종 FC 제거
        #     global_pool='avg'
        # )
        # self.backbone = timm.create_model('efficientnet_lite0', pretrained=True, num_classes=0, global_pool='avg')
        self.backbone = timm.create_model('regnetx_002', pretrained=True, num_classes=0, global_pool='avg')

        in_features = self.backbone.num_features

        # 필요하다면 백본 일부를 Freeze
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Custom Head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_secondary_classes)
        )
        
        print(self.classifier)


    def forward(self, x):
        features = self.backbone(x)  # (B, in_features)
        logits = self.classifier(features)
        return logits