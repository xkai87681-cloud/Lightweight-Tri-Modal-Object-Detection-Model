import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from .backbone import build_mobilenetv3_backbone
from .neck import build_yolov8_neck
from .det_head import build_detection_head
from .attr_head import build_attribute_head
from .seg_head import build_segmentation_head
from .attention import build_task_attention


class MTLModel(nn.Module):

    def __init__(
        self,
        num_det_classes=2,
        num_attr_classes=26,
        num_seg_classes=2,
        backbone_pretrained=True,
        backbone_pretrained_type='detection',
        neck_depth=3,
        reg_max=16,
        use_task_attention=False,
        attention_reduction=16,
        use_attention_fusion=True,
    ):
        super().__init__()

        self.num_det_classes = num_det_classes
        self.num_attr_classes = num_attr_classes
        self.num_seg_classes = num_seg_classes
        self.use_task_attention = use_task_attention

        self.backbone = build_mobilenetv3_backbone(
            pretrained=backbone_pretrained,
            pretrained_type=backbone_pretrained_type
        )
        backbone_out_channels = self.backbone.out_channels  # [40, 112, 960]

        neck_out_channels = (128, 256, 512)  # [P3, P4, P5]
        self.neck = build_yolov8_neck(
            in_channels=backbone_out_channels,
            out_channels=neck_out_channels,
            depth=neck_depth
        )

        self.det_head = build_detection_head(
            num_classes=num_det_classes,
            in_channels=neck_out_channels,
            reg_max=reg_max
        )

        self.attr_head = build_attribute_head(
            num_attrs=num_attr_classes,
            in_channels=neck_out_channels,
            roi_size=7,
            hidden_dim=512,
            dropout=0.3
        )

        self.seg_head = build_segmentation_head(
            num_classes=num_seg_classes,
            in_channels_p3=neck_out_channels[0],
            in_channels_p4=neck_out_channels[1],
            hidden_dim=128
        )

        self.task_attention = None
        self.attention_fusion = None
        if use_task_attention:
            self.task_attention, self.attention_fusion = build_task_attention(
                in_channels_list=list(neck_out_channels),
                num_tasks=3,  # detection, attribute, segmentation
                reduction=attention_reduction,
                use_fusion=use_attention_fusion
            )

    def forward(
        self,
        images: torch.Tensor,
        person_rois: Optional[torch.Tensor] = None,
        task: str = 'all'
    ) -> Dict[str, any]:
        c3, c4, c5 = self.backbone(images)

        p3, p4, p5 = self.neck([c3, c4, c5])
        features = [p3, p4, p5]

        outputs = {}

        if task in ['all', 'detection', 'det+seg']:
            if self.use_task_attention:
                det_features = self.task_attention(features, task_id=0)
                if self.attention_fusion:
                    det_features = [
                        fusion(orig, att)
                        for orig, att, fusion in zip(features, det_features, self.attention_fusion)
                    ]
            else:
                det_features = features

            bbox_preds, cls_preds, bbox_dist = self.det_head(det_features)
            outputs['det_bbox'] = bbox_preds
            outputs['det_cls'] = cls_preds
            outputs['det_dist'] = bbox_dist

        if task in ['all', 'attribute']:
            if self.use_task_attention:
                attr_features = self.task_attention(features, task_id=1)
                if self.attention_fusion:
                    attr_features = [
                        fusion(orig, att)
                        for orig, att, fusion in zip(features, attr_features, self.attention_fusion)
                    ]
            else:
                attr_features = features

            attr_preds = self.attr_head(attr_features, person_rois)
            outputs['attr'] = attr_preds  # None if no person_rois
        else:
            outputs['attr'] = None

        if task in ['all', 'segmentation', 'det+seg']:
            if self.use_task_attention:
                seg_features = self.task_attention(features, task_id=2)
                if self.attention_fusion:
                    seg_features = [
                        fusion(orig, att)
                        for orig, att, fusion in zip(features, seg_features, self.attention_fusion)
                    ]
            else:
                seg_features = features

            seg_logits = self.seg_head(seg_features)
            outputs['seg'] = seg_logits

        return outputs

    def get_param_groups(self, lr_backbone=1e-4, lr_neck=1e-3, lr_heads=1e-3, lr_attention=1e-3):
        param_groups = [
            {'params': self.backbone.parameters(), 'lr': lr_backbone, 'name': 'backbone'},
            {'params': self.neck.parameters(), 'lr': lr_neck, 'name': 'neck'},
            {'params': self.det_head.parameters(), 'lr': lr_heads, 'name': 'det_head'},
            {'params': self.attr_head.parameters(), 'lr': lr_heads, 'name': 'attr_head'},
            {'params': self.seg_head.parameters(), 'lr': lr_heads, 'name': 'seg_head'},
        ]

        if self.use_task_attention:
            param_groups.append({
                'params': self.task_attention.parameters(),
                'lr': lr_attention,
                'name': 'task_attention'
            })
            if self.attention_fusion:
                param_groups.append({
                    'params': self.attention_fusion.parameters(),
                    'lr': lr_attention,
                    'name': 'attention_fusion'
                })

        return param_groups


def build_mtl_model(
    num_det_classes=2,
    num_attr_classes=26,
    num_seg_classes=2,
    backbone_pretrained=True,
    backbone_pretrained_type='detection',
    neck_depth=3,
    reg_max=16,
    use_task_attention=False,
    attention_reduction=16,
    use_attention_fusion=True,
):
    return MTLModel(
        num_det_classes=num_det_classes,
        num_attr_classes=num_attr_classes,
        num_seg_classes=num_seg_classes,
        backbone_pretrained=backbone_pretrained,
        backbone_pretrained_type=backbone_pretrained_type,
        neck_depth=neck_depth,
        reg_max=reg_max,
        use_task_attention=use_task_attention,
        attention_reduction=attention_reduction,
        use_attention_fusion=use_attention_fusion,
    )


if __name__ == '__main__':
    print("Building MTL Model...")
    model = build_mtl_model()
    model.eval()

    images = torch.randn(2, 3, 640, 640)

    person_rois = torch.tensor([
        [0, 100, 100, 300, 400],
        [0, 350, 120, 500, 450],
        [1, 80, 90, 250, 380],
    ], dtype=torch.float32)

    print("\n========== Test 1: All Tasks ==========")
    outputs = model(images, person_rois, task='all')
    print(f"Detection bbox levels: {len(outputs['det_bbox'])}")
    for i, bbox in enumerate(outputs['det_bbox']):
        print(f"  Level {i}: {bbox.shape}")
    print(f"Attribute predictions: {outputs['attr'].shape if outputs['attr'] is not None else None}")
    print(f"Segmentation logits: {outputs['seg'].shape}")

    print("\n========== Test 2: Det + Seg Only (Curriculum Learning) ==========")
    outputs = model(images, task='det+seg')
    print(f"Detection: {len(outputs['det_bbox'])} levels")
    print(f"Attribute: {outputs['attr']}")
    print(f"Segmentation: {outputs['seg'].shape}")

    print("\n========== Test 3: No Person RoIs ==========")
    outputs = model(images, person_rois=None, task='all')
    print(f"Attribute with no RoIs: {outputs['attr']}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n========== Model Statistics ==========")
    print(f"Total params: {total_params / 1e6:.2f}M")
    print(f"Trainable params: {trainable_params / 1e6:.2f}M")

    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    neck_params = sum(p.numel() for p in model.neck.parameters())
    det_params = sum(p.numel() for p in model.det_head.parameters())
    attr_params = sum(p.numel() for p in model.attr_head.parameters())
    seg_params = sum(p.numel() for p in model.seg_head.parameters())

    print(f"\nBackbone: {backbone_params / 1e6:.2f}M")
    print(f"Neck: {neck_params / 1e6:.2f}M")
    print(f"Det Head: {det_params / 1e6:.2f}M")
    print(f"Attr Head: {attr_params / 1e6:.2f}M")
    print(f"Seg Head: {seg_params / 1e6:.2f}M")
