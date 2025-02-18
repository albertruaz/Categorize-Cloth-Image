import torch.nn as nn
import timm

class HierarchicalCNN(nn.Module):
    def __init__(self, primary_to_secondary_map, num_primary_classes):
        super().__init__()
        # ecaresnet50d Backbone
        # self.backbone = timm.create_model(
        #     "ecaresnet50d",
        #     pretrained=True,
        #     num_classes=0,        # fc 제거
        #     global_pool="avg"
        # )
        self.backbone = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=True,
            num_classes=0,
            global_pool="avg"
        )
        in_features = self.backbone.num_features

        # Primary Head
        self.primary_head = nn.Linear(in_features, num_primary_classes)

        # Secondary Heads: 각 primary_id별 별도 Linear
        self.secondary_heads = nn.ModuleDict()
        for pid, sub_list in primary_to_secondary_map.items():
            out_dim = len(sub_list)
            self.secondary_heads[str(pid)] = nn.Linear(in_features, out_dim)

    def forward(self, x):
        feat = self.backbone(x)  # shape: (B, in_features)
        primary_logits = self.primary_head(feat)
        # secondary는 각 샘플별로 pid가 다를 수 있어 한 번에 못 뽑음 -> 학습루프에서 처리
        return primary_logits, feat
    
class SingleStageCNN(nn.Module):
    def __init__(self, num_secondary_classes):
        super().__init__()
        # 기존과 동일한 swin_base_patch4_window7_224 백본 사용
        self.backbone = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=True,
            num_classes=0,        # FC 제거
            global_pool="avg"
        )
        in_features = self.backbone.num_features

        # 전체 Secondary 클래스를 한 번에 분류하는 헤드
        self.classifier = nn.Linear(in_features, num_secondary_classes)

    def forward(self, x):
        features = self.backbone(x)   # (B, in_features)
        logits = self.classifier(features)
        return logits  # shape: (B, num_secondary_classes)
