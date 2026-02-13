import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DFL(nn.Module):

    def __init__(self, c1=16):
        super().__init__()
        self.c1 = c1
        x = torch.arange(c1, dtype=torch.float)
        self.register_buffer('project', x.view(1, c1, 1, 1))

    def forward(self, x):
        b, _, h, w = x.shape
        x = x.view(b, 4, self.c1, h * w)
        x = F.softmax(x, dim=2)
        x = (x * self.project.view(1, 1, self.c1, 1)).sum(dim=2)
        return x.view(b, 4, h, w)


class DetectionHead(nn.Module):

    def __init__(self, num_classes=2, in_channels=(128, 256, 512), num_levels=3, reg_max=16):
        super().__init__()
        self.num_classes = num_classes
        self.num_levels = num_levels
        self.reg_max = reg_max

        self.stems = nn.ModuleList()
        self.bbox_heads = nn.ModuleList()
        self.cls_heads = nn.ModuleList()

        for in_ch in in_channels:
            stem = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, 1, 1, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.SiLU(inplace=True),
                nn.Conv2d(in_ch, in_ch, 3, 1, 1, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.SiLU(inplace=True)
            )
            self.stems.append(stem)

            bbox_head = nn.Sequential(
                nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1, bias=False),
                nn.BatchNorm2d(in_ch // 2),
                nn.SiLU(inplace=True),
                nn.Conv2d(in_ch // 2, 4 * reg_max, 1, 1, 0, bias=True),  # 4个坐标 * reg_max bins
            )
            self.bbox_heads.append(bbox_head)

            cls_head = nn.Sequential(
                nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1, bias=False),
                nn.BatchNorm2d(in_ch // 2),
                nn.SiLU(inplace=True),
                nn.Conv2d(in_ch // 2, num_classes, 1, 1, 0, bias=True),  # num_classes
            )
            self.cls_heads.append(cls_head)

        self.dfl = DFL(reg_max)

        self._initialize_biases()

    def _initialize_biases(self):
        for cls_head in self.cls_heads:
            conv = cls_head[-1]
            nn.init.constant_(conv.bias, -4.6)  # ln(0.01 / 0.99)

    def forward(self, features):
        bbox_preds = []
        cls_preds = []
        bbox_dist = []

        for i, feat in enumerate(features):
            x = self.stems[i](feat)

            bbox_raw = self.bbox_heads[i](x)  # [B, 4*reg_max, H, W]
            bbox = self.dfl(bbox_raw)  # [B, 4, H, W]
            bbox_preds.append(bbox)
            bbox_dist.append(bbox_raw)

            cls = self.cls_heads[i](x)  # [B, num_classes, H, W]
            cls_preds.append(cls)

        return bbox_preds, cls_preds, bbox_dist


def build_detection_head(num_classes=2, in_channels=(128, 256, 512), reg_max=16):
    return DetectionHead(num_classes, in_channels, reg_max=reg_max)


if __name__ == '__main__':
    det_head = build_detection_head(num_classes=2)
    det_head.eval()

    p3 = torch.randn(2, 128, 80, 80)   # stride=8
    p4 = torch.randn(2, 256, 40, 40)   # stride=16
    p5 = torch.randn(2, 512, 20, 20)   # stride=32

    bbox_preds, cls_preds, bbox_dist = det_head([p3, p4, p5])

    print("Detection Head Test:")
    print(f"Input P3: {p3.shape}")
    print(f"Input P4: {p4.shape}")
    print(f"Input P5: {p5.shape}")
    print()
    for i, (bbox, cls, dist) in enumerate(zip(bbox_preds, cls_preds, bbox_dist)):
        print(f"Level {i}:")
        print(f"  BBox pred: {bbox.shape}")
        print(f"  Cls pred: {cls.shape}")
        print(f"  BBox dist: {dist.shape}")

    total_params = sum(p.numel() for p in det_head.parameters())
    print(f"\nTotal params: {total_params / 1e6:.2f}M")
