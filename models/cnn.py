import torch.nn as nn
import timm

class ClothingClassifierCNN(nn.Module):
    def __init__(self, hidden_layer, num_secondary_classes=33):
        super().__init__()
        
        self.backbone = timm.create_model(
            'regnetx_002', 
            pretrained=True, 
            num_classes=0, 
            global_pool='avg'
        )
        in_features = self.backbone.num_features
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_layer),
            nn.BatchNorm1d(hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, num_secondary_classes)
        )
        
        print(self.classifier)

    def forward(self, x):
        features = self.backbone(x)  # (B, in_features)
        logits = self.classifier(features)
        return logits
