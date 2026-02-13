import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from typing import List, Tuple, Optional


class AttributeHead(nn.Module):

    def __init__(
        self,
        num_attrs=26,
        in_channels=(128, 256, 512),
        roi_size=7,
        hidden_dim=512,
        dropout=0.3,
        strides=(8, 16, 32)
    ):
        super().__init__()
        self.num_attrs = num_attrs
        self.roi_size = roi_size
        self.strides = strides
        self.in_channels = in_channels

        self.channel_align = nn.ModuleList([
            nn.Conv2d(ch, 256, 1, 1, 0) for ch in in_channels
        ])

        roi_feat_dim = 256 * roi_size * roi_size

        self.classifier = nn.Sequential(
            nn.Linear(roi_feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_attrs)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _select_feature_level(self, rois):
        w = rois[:, 3] - rois[:, 1]
        h = rois[:, 4] - rois[:, 2]
        areas = w * h

        k0 = 4
        k = torch.floor(k0 + torch.log2(torch.sqrt(areas) / 224.0))
        k = torch.clamp(k, min=3, max=5)  # P3, P4, P5

        level_assignments = (k - 3).long()

        return level_assignments

    def forward(self, features: List[torch.Tensor], person_rois: Optional[torch.Tensor]):
        if person_rois is None or person_rois.numel() == 0:
            return None

        num_rois = person_rois.size(0)
        if num_rois == 0:
            return None

        features_aligned = [
            self.channel_align[i](feat)
            for i, feat in enumerate(features)
        ]  # 每个: [B, 256, H, W]

        level_assignments = self._select_feature_level(person_rois)

        roi_features_list = []

        for level in range(3):  # P3, P4, P5
            level_mask = level_assignments == level
            if not level_mask.any():
                continue

            level_rois = person_rois[level_mask]

            spatial_scale = 1.0 / self.strides[level]

            roi_feats = roi_align(
                input=features_aligned[level],
                boxes=level_rois,
                output_size=self.roi_size,
                spatial_scale=spatial_scale,
                sampling_ratio=2,
                aligned=True
            )  # [N_level, 256, roi_size, roi_size]

            roi_features_list.append(roi_feats)

        if len(roi_features_list) == 0:
            return None

        all_roi_features = torch.cat(roi_features_list, dim=0)  # [N, 256, roi_size, roi_size]

        roi_indices = []
        for level in range(3):
            level_mask = level_assignments == level
            level_indices = torch.where(level_mask)[0]
            roi_indices.append(level_indices)

        roi_indices = torch.cat(roi_indices, dim=0)
        sorted_indices = torch.argsort(roi_indices)
        all_roi_features = all_roi_features[sorted_indices]

        roi_features_flat = all_roi_features.view(num_rois, -1)  # [N, 256*roi_size*roi_size]

        attr_preds = self.classifier(roi_features_flat)  # [N, num_attrs]

        return attr_preds


def build_attribute_head(
    num_attrs=26,
    in_channels=(128, 256, 512),
    roi_size=7,
    hidden_dim=512,
    dropout=0.3
):
    return AttributeHead(num_attrs, in_channels, roi_size, hidden_dim, dropout)


if __name__ == '__main__':
    attr_head = build_attribute_head(num_attrs=26)
    attr_head.eval()

    p3 = torch.randn(2, 128, 80, 80)   # B=2, stride=8
    p4 = torch.randn(2, 256, 40, 40)   # B=2, stride=16
    p5 = torch.randn(2, 512, 20, 20)   # B=2, stride=32

    person_rois = torch.tensor([
        [0, 100, 100, 300, 400],  # batch 0, person 1
        [0, 350, 120, 500, 450],  # batch 0, person 2
        [1, 80, 90, 250, 380],    # batch 1, person 1
        [1, 400, 150, 550, 480],  # batch 1, person 2
        [1, 200, 200, 350, 400],  # batch 1, person 3
    ], dtype=torch.float32)

    attr_preds = attr_head([p3, p4, p5], person_rois)

    print("Attribute Head Test:")
    print(f"Number of Person RoIs: {person_rois.size(0)}")
    print(f"Attribute predictions shape: {attr_preds.shape}")
    print(f"Expected: [5, 26]")

    print("\nTest with empty RoIs:")
    empty_rois = torch.empty(0, 5)
    attr_preds_empty = attr_head([p3, p4, p5], empty_rois)
    print(f"Result with empty RoIs: {attr_preds_empty}")
    print(f"Expected: None")

    total_params = sum(p.numel() for p in attr_head.parameters())
    print(f"\nTotal params: {total_params / 1e6:.2f}M")
