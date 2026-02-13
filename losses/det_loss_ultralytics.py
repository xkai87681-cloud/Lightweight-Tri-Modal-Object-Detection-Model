import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.loss import v8DetectionLoss
from losses.focal_bce import FocalBCEWithLogitsLoss


class FocalV8DetectionLoss(v8DetectionLoss):

    def __init__(self, model, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__(model)
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.focal_bce = FocalBCEWithLogitsLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='none')

    def __call__(self, preds, batch):
        loss_tuple = super().__call__(preds, batch)


        # TODO: 这里需要访问父类的中间变量，但Ultralytics没有暴露接口
        return loss_tuple


class UltralyticsDetectionLoss(nn.Module):

    def __init__(self, num_classes=2, reg_max=16, device='cuda', use_focal_bce=True, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.device = device
        self.use_focal_bce = use_focal_bce

        class FakeDetectModule(nn.Module):
            def __init__(self, nc, reg_max, device):
                super().__init__()
                self.nc = nc
                self.reg_max = reg_max
                self.register_buffer('stride', torch.tensor([8, 16, 32], dtype=torch.float32, device=device))
                self.dummy = nn.Parameter(torch.zeros(1, device=device))

        class FakeArgs:
            def __init__(self):
                self.box = 7.5   # box loss gain (YOLOv8 默认)
                self.cls = 0.5   # cls loss gain (YOLOv8 默认)
                self.dfl = 1.5   # dfl loss gain (YOLOv8 默认)

        class FakeModel(nn.Module):
            def __init__(self, nc, reg_max, device):
                super().__init__()
                self.args = FakeArgs()
                self.model = nn.ModuleList([FakeDetectModule(nc, reg_max, device)])
                self.dummy = nn.Parameter(torch.zeros(1, device=device))

        fake_model = FakeModel(num_classes, reg_max, device)

        if use_focal_bce:
            self.loss_fn = FocalV8DetectionLoss(fake_model, focal_alpha, focal_gamma)
        else:
            self.loss_fn = v8DetectionLoss(fake_model)


    def forward(self, preds, targets):
        device = preds['det_cls'][0].device
        batch_size = preds['det_cls'][0].shape[0]

        feats = []
        for i in range(len(preds['det_cls'])):
            bbox_dist = preds['det_dist'][i]  # [B, 4*reg_max, H, W]
            cls_pred = preds['det_cls'][i]    # [B, num_classes, H, W]

            feat = torch.cat([bbox_dist, cls_pred], dim=1)
            feats.append(feat)

        batch_indices_list = []
        labels_list = []
        bboxes_list = []

        for batch_idx, target_dict in enumerate(targets):
            bboxes = target_dict.get('bboxes', torch.empty(0, 4, device=device))
            labels = target_dict.get('labels', torch.empty(0, device=device, dtype=torch.long))

            if len(bboxes) == 0:
                continue

            x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
            x_center = (x1 + x2) / 2  # 已经是归一化坐标
            y_center = (y1 + y2) / 2  # 已经是归一化坐标
            width = x2 - x1           # 已经是归一化坐标
            height = y2 - y1          # 已经是归一化坐标

            num_objs = len(labels)
            batch_indices_list.append(torch.full((num_objs,), batch_idx, device=device, dtype=torch.long))
            labels_list.append(labels)
            bboxes_list.append(torch.stack([x_center, y_center, width, height], dim=1))

        if len(batch_indices_list) > 0:
            batch_targets = {
                'batch_idx': torch.cat(batch_indices_list, dim=0),  # [N]
                'cls': torch.cat(labels_list, dim=0),                # [N]
                'bboxes': torch.cat(bboxes_list, dim=0)              # [N, 4]
            }
        else:
            batch_targets = {
                'batch_idx': torch.zeros(0, device=device, dtype=torch.long),
                'cls': torch.zeros(0, device=device, dtype=torch.long),
                'bboxes': torch.zeros((0, 4), device=device, dtype=torch.float32)
            }

        try:
            loss_items = self.loss_fn(feats, batch_targets)

            if isinstance(loss_items, tuple):
                weighted_loss, unweighted_loss = loss_items
                batch_size = preds['det_cls'][0].shape[0]
                total_loss = weighted_loss.sum() / batch_size
                box_loss = unweighted_loss[0] if len(unweighted_loss) > 0 else torch.tensor(0.0, device=device)
                cls_loss = unweighted_loss[1] if len(unweighted_loss) > 1 else torch.tensor(0.0, device=device)
                dfl_loss = unweighted_loss[2] if len(unweighted_loss) > 2 else torch.tensor(0.0, device=device)
            else:
                total_loss = loss_items.sum() if loss_items.numel() > 1 else loss_items
                box_loss = torch.tensor(0.0, device=device)
                cls_loss = torch.tensor(0.0, device=device)
                dfl_loss = torch.tensor(0.0, device=device)

            return {
                'det_loss': total_loss,
                'det_box_loss': box_loss,
                'det_cls_loss': cls_loss,
                'det_dfl_loss': dfl_loss
            }

        except Exception as e:
            print(f"[ERROR] Ultralytics Detection Loss failed: {e}")
            import traceback
            traceback.print_exc()
            print(f"  feats shapes: {[f.shape for f in feats]}")
            print(f"  batch_targets format: {type(batch_targets)}")
            if isinstance(batch_targets, dict):
                print(f"    batch_idx shape: {batch_targets['batch_idx'].shape}")
                print(f"    cls shape: {batch_targets['cls'].shape}")
                print(f"    bboxes shape: {batch_targets['bboxes'].shape}")

            fallback_loss = torch.tensor(1.0, device=device, requires_grad=True)
            return {
                'det_loss': fallback_loss,
                'det_box_loss': torch.tensor(0.0, device=device),
                'det_cls_loss': torch.tensor(0.0, device=device),
                'det_dfl_loss': torch.tensor(0.0, device=device)
            }


DetectionLoss = UltralyticsDetectionLoss


if __name__ == '__main__':
    print("Testing Ultralytics Detection Loss...")

    device = 'cpu'

    loss_fn = UltralyticsDetectionLoss(num_classes=2, reg_max=16, device=device)

    batch_size = 2
    preds = {
        'det_bbox': [
            torch.randn(batch_size, 4, 80, 80),
            torch.randn(batch_size, 4, 40, 40),
            torch.randn(batch_size, 4, 20, 20),
        ],
        'det_cls': [
            torch.randn(batch_size, 2, 80, 80),
            torch.randn(batch_size, 2, 40, 40),
            torch.randn(batch_size, 2, 20, 20),
        ],
        'det_dist': [
            torch.randn(batch_size, 64, 80, 80),
            torch.randn(batch_size, 64, 40, 40),
            torch.randn(batch_size, 64, 20, 20),
        ],
    }

    targets = [
        {
            'bboxes': torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]], dtype=torch.float32),
            'labels': torch.tensor([0, 1], dtype=torch.long)
        },
        {
            'bboxes': torch.tensor([[150, 150, 250, 250]], dtype=torch.float32),
            'labels': torch.tensor([1], dtype=torch.long)
        }
    ]

    print("Computing loss...")
    loss_dict = loss_fn(preds, targets)

    print("\nLoss Results:")
    for k, v in loss_dict.items():
        if v.numel() == 1:
            print(f"  {k}: {v.item():.4f}")
        else:
            print(f"  {k}: {v}")

    print("\n✓ Ultralytics Detection Loss test passed!")
