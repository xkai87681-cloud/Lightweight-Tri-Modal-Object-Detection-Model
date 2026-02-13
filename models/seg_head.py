import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class SegmentationHead(nn.Module):

    def __init__(
        self,
        num_classes=2,
        in_channels_p3=128,
        in_channels_p4=256,
        hidden_dim=128,
        use_p3=True,
        use_p4=True,
        use_p5=False  # 强制禁用
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_p3 = use_p3
        self.use_p4 = use_p4

        if use_p5:
            raise ValueError("Segmentation Head 禁止使用 P5!")
        self.use_p5 = False

        if self.use_p3:
            self.reduce_p3 = nn.Sequential(
                nn.Conv2d(in_channels_p3, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            )

        if self.use_p4:
            self.reduce_p4 = nn.Sequential(
                nn.Conv2d(in_channels_p4, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            )

        fusion_in_channels = 0
        if self.use_p3:
            fusion_in_channels += hidden_dim
        if self.use_p4:
            fusion_in_channels += hidden_dim

        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in_channels, hidden_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),  # 🚀 降低Dropout（0.6 → 0.3），BDD100K数据充足
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3)  # 🚀 降低Dropout（0.6 → 0.3）
        )

        self.classifier = nn.Conv2d(hidden_dim, num_classes, 1, 1, 0)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features: List[torch.Tensor]):
        p3, p4, _ = features  # 故意忽略 P5

        B, _, H_p3, W_p3 = p3.shape
        target_h = H_p3 * 8  # P3 是 stride=8
        target_w = W_p3 * 8

        multi_scale_features = []

        if self.use_p3:
            p3_reduced = self.reduce_p3(p3)  # [B, hidden_dim, H/8, W/8]
            multi_scale_features.append(p3_reduced)

        if self.use_p4:
            p4_reduced = self.reduce_p4(p4)  # [B, hidden_dim, H/16, W/16]
            p4_upsampled = F.interpolate(
                p4_reduced,
                size=(H_p3, W_p3),
                mode='bilinear',
                align_corners=False
            )
            multi_scale_features.append(p4_upsampled)

        fused = torch.cat(multi_scale_features, dim=1)  # [B, hidden_dim*2, H/8, W/8]
        fused = self.fusion(fused)  # [B, hidden_dim, H/8, W/8]

        seg_logits = self.classifier(fused)  # [B, num_classes, H/8, W/8]

        seg_logits = F.interpolate(
            seg_logits,
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        )  # [B, num_classes, H, W]

        return seg_logits


def build_segmentation_head(
    num_classes=2,
    in_channels_p3=128,
    in_channels_p4=256,
    hidden_dim=128
):
    return SegmentationHead(num_classes, in_channels_p3, in_channels_p4, hidden_dim)


if __name__ == '__main__':
    seg_head = build_segmentation_head(num_classes=2)
    seg_head.eval()

    p3 = torch.randn(2, 128, 80, 80)   # stride=8
    p4 = torch.randn(2, 256, 40, 40)   # stride=16
    p5 = torch.randn(2, 512, 20, 20)   # stride=32 (不使用)

    seg_logits = seg_head([p3, p4, p5])

    print("Segmentation Head Test:")
    print(f"Input P3: {p3.shape}")
    print(f"Input P4: {p4.shape}")
    print(f"Input P5: {p5.shape} (ignored)")
    print(f"Output seg_logits: {seg_logits.shape}")
    print(f"Expected: [2, 2, 640, 640]")

    print("\nTest P5 enforcement:")
    try:
        bad_head = SegmentationHead(num_classes=2, use_p5=True)
        print("ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"Correctly raised ValueError: {e}")

    total_params = sum(p.numel() for p in seg_head.parameters())
    print(f"\nTotal params: {total_params / 1e6:.2f}M")
