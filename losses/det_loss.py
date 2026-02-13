import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FocalLoss(nn.Module):

    def __init__(self, gamma=1.5, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)

        p_t = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = -alpha_t * ((1 - p_t) ** self.gamma) * p_t.log()

        return loss.mean()


def bbox_iou(box1, box2, xywh=False, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1 / 2, x1 + w1 / 2, y1 - h1 / 2, y1 + h1 / 2
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2 / 2, x2 + w2 / 2, y2 - h2 / 2, y2 + h2 / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)

        if CIoU or DIoU:
            c2 = cw ** 2 + ch ** 2 + eps
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4

            if CIoU:
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)) ** 2
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
            else:  # DIoU
                return iou - rho2 / c2

        else:  # GIoU
            c_area = cw * ch + eps
            return iou - (c_area - union) / c_area

    return iou


class DetectionLoss(nn.Module):

    def __init__(
        self,
        num_classes=2,
        reg_max=16,
        gamma=1.5,
        alpha=0.25,
        box_weight=7.5,
        cls_weight=0.5,
        dfl_weight=1.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.dfl_weight = dfl_weight

        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha)

        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, preds, targets):
        device = preds['det_bbox'][0].device

        if targets is None or len(targets) == 0:
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return {
                'det_loss': zero_loss,
                'det_cls_loss': zero_loss,
                'det_bbox_loss': zero_loss
            }

        has_targets = any(len(t.get('bboxes', [])) > 0 for t in targets)

        if not has_targets:
            cls_loss = 0.0
            num_levels = len(preds['det_cls'])

            for i in range(num_levels):
                cls_pred = preds['det_cls'][i]  # [B, num_classes, H, W]
                cls_target = torch.zeros_like(cls_pred)
                cls_loss += self.bce_loss(cls_pred, cls_target).mean()

            cls_loss = cls_loss / num_levels
            bbox_loss = torch.tensor(0.0, device=device)

            total_loss = self.cls_weight * cls_loss

            return {
                'det_loss': total_loss,
                'det_cls_loss': cls_loss,
                'det_bbox_loss': bbox_loss
            }

        cls_pred = preds['det_cls'][0]   # [B, num_classes, H, W] - P3
        bbox_pred = preds['det_bbox'][0]  # [B, 4, H, W]

        B, C, H, W = cls_pred.shape

        pos_cls_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        neg_cls_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        bbox_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        num_pos = 0

        for batch_idx in range(B):
            target_dict = targets[batch_idx]
            gt_bboxes = target_dict.get('bboxes', torch.empty(0, 4, device=device))
            gt_labels = target_dict.get('labels', torch.empty(0, device=device, dtype=torch.long))

            if len(gt_bboxes) == 0:
                cls_target = torch.zeros_like(cls_pred[batch_idx])
                neg_cls_loss += self.bce_loss(cls_pred[batch_idx], cls_target).mean()
                continue

            gt_bboxes_scaled = gt_bboxes.clone()  # [N, 4]
            gt_bboxes_scaled[:, [0, 2]] *= W  # x坐标
            gt_bboxes_scaled[:, [1, 3]] *= H  # y坐标

            gt_centers_x = (gt_bboxes_scaled[:, 0] + gt_bboxes_scaled[:, 2]) / 2  # [N]
            gt_centers_y = (gt_bboxes_scaled[:, 1] + gt_bboxes_scaled[:, 3]) / 2  # [N]

            gt_centers_x = gt_centers_x.clamp(0, W - 1)
            gt_centers_y = gt_centers_y.clamp(0, H - 1)

            grid_x = gt_centers_x.long()  # [N]
            grid_y = gt_centers_y.long()  # [N]

            cls_target = torch.zeros_like(cls_pred[batch_idx])  # [C, H, W]

            for obj_idx in range(len(gt_labels)):
                label = gt_labels[obj_idx].item()
                gx, gy = grid_x[obj_idx].item(), grid_y[obj_idx].item()

                if 0 <= gx < W and 0 <= gy < H and 0 <= label < C:
                    cls_target[label, gy, gx] = 1.0
                    num_pos += 1

            pos_mask = (cls_target > 0)
            neg_mask = ~pos_mask

            if pos_mask.sum() > 0:
                pos_cls_loss += self.bce_loss(cls_pred[batch_idx][pos_mask], cls_target[pos_mask]).mean()

            if neg_mask.sum() > 0:
                neg_cls_loss += self.bce_loss(cls_pred[batch_idx][neg_mask], cls_target[neg_mask]).mean()

            for obj_idx in range(len(gt_labels)):
                gx, gy = grid_x[obj_idx].item(), grid_y[obj_idx].item()
                if 0 <= gx < W and 0 <= gy < H:
                    pred_bbox = bbox_pred[batch_idx, :, gy, gx]  # [4]
                    target_bbox = gt_bboxes_scaled[obj_idx]       # [4]
                    bbox_loss += F.l1_loss(pred_bbox, target_bbox, reduction='mean')

        if num_pos > 0:
            pos_cls_loss = pos_cls_loss / max(num_pos, 1)
            bbox_loss = bbox_loss / max(num_pos, 1)

        if B > 0:
            neg_cls_loss = neg_cls_loss / B

        cls_loss = pos_cls_loss + 0.25 * neg_cls_loss

        total_loss = (
            self.cls_weight * cls_loss +
            self.box_weight * bbox_loss
        )

        total_loss = total_loss + 1e-6

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"[WARNING] Detection Loss is NaN/Inf! pos_cls={pos_cls_loss.item():.4f}, "
                  f"neg_cls={neg_cls_loss.item():.4f}, bbox={bbox_loss.item():.4f}, num_pos={num_pos}")
            total_loss = torch.tensor(1e-4, device=device, requires_grad=True)

        return {
            'det_loss': total_loss,
            'det_cls_loss': cls_loss.detach() if isinstance(cls_loss, torch.Tensor) else torch.tensor(cls_loss, device=device),
            'det_bbox_loss': bbox_loss.detach() if isinstance(bbox_loss, torch.Tensor) else torch.tensor(bbox_loss, device=device)
        }


if __name__ == '__main__':
    det_loss_fn = DetectionLoss(num_classes=2, reg_max=16)

    preds = {
        'det_bbox': [
            torch.randn(2, 4, 80, 80),
            torch.randn(2, 4, 40, 40),
            torch.randn(2, 4, 20, 20),
        ],
        'det_cls': [
            torch.randn(2, 2, 80, 80),
            torch.randn(2, 2, 40, 40),
            torch.randn(2, 2, 20, 20),
        ],
        'det_dist': [
            torch.randn(2, 64, 80, 80),  # 4*16
            torch.randn(2, 64, 40, 40),
            torch.randn(2, 64, 20, 20),
        ]
    }

    losses = det_loss_fn(preds, None)

    print("Detection Loss Test:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
