import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationLoss(nn.Module):
    def __init__(self, num_classes=2, ce_weight=1.0, dice_weight=1.0,
                 class_weights=None, ignore_index=255,
                 use_focal_loss=False, focal_alpha=None, focal_gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        self.use_focal_loss = use_focal_loss

        if use_focal_loss:
            if focal_alpha is None:
                focal_alpha = 0.25  # 默认值
            self.ce_loss = FocalLoss(
                num_classes=num_classes,
                alpha=focal_alpha,
                gamma=focal_gamma,
                ignore_index=ignore_index
            )
            print(f"[SegmentationLoss] Using Focal Loss with alpha={focal_alpha}, gamma={focal_gamma}")
        else:
            if class_weights is not None:
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
            print(f"[SegmentationLoss] Using Cross Entropy Loss with class_weights={class_weights}")

    def dice_loss(self, pred, target, smooth=0.01):
        pred_probs = F.softmax(pred, dim=1)  # (B, C, H, W)

        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index)  # (B, H, W)
            target_clamped = target.clone()
            target_clamped[~valid_mask] = 0
        else:
            valid_mask = torch.ones_like(target, dtype=torch.bool)
            target_clamped = target

        target_one_hot = F.one_hot(target_clamped, num_classes=self.num_classes)  # (B, H, W, C)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        mask = valid_mask.unsqueeze(1).float()  # (B, 1, H, W)
        pred_probs = pred_probs * mask
        target_one_hot = target_one_hot * mask

        intersection = (pred_probs * target_one_hot).sum(dim=(2, 3))  # (B, C)
        union = pred_probs.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))  # (B, C)

        dice = (2.0 * intersection + smooth) / (union + smooth)  # (B, C)

        dice = torch.clamp(dice, min=0.0, max=1.0)

        return 1.0 - dice.mean()

    def forward(self, pred, target):
        pred = torch.clamp(pred, min=-100, max=100)

        ce_loss = self.ce_loss(pred, target)

        dice_loss = self.dice_loss(pred, target)

        if torch.isnan(ce_loss) or torch.isinf(ce_loss):
            print(f"[WARNING] Segmentation CE Loss is NaN/Inf! Returning fallback loss.")
            ce_loss = torch.tensor(1.0, device=pred.device, dtype=pred.dtype, requires_grad=True)

        if torch.isnan(dice_loss) or torch.isinf(dice_loss):
            print(f"[WARNING] Segmentation Dice Loss is NaN/Inf! Returning fallback loss.")
            dice_loss = torch.tensor(1.0, device=pred.device, dtype=pred.dtype, requires_grad=True)

        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"[ERROR] Segmentation Total Loss is NaN/Inf! ce={ce_loss.item():.4f}, dice={dice_loss.item():.4f}")
            total_loss = torch.tensor(1.0, device=pred.device, dtype=pred.dtype, requires_grad=True)

        return {
            'seg_loss': total_loss,
            'seg_ce_loss': ce_loss,
            'seg_dice_loss': dice_loss
        }


class FocalLoss(nn.Module):
    def __init__(self, num_classes=2, alpha=0.25, gamma=2.0, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.ignore_index = ignore_index

        if isinstance(alpha, (list, tuple)):
            if len(alpha) != num_classes:
                raise ValueError(f"alpha长度({len(alpha)})必须等于num_classes({num_classes})")
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = torch.tensor([alpha] * num_classes, dtype=torch.float32)

        self.register_buffer('alpha_buffer', self.alpha)

    def forward(self, pred, target):
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index)
            if valid_mask.any():
                valid_targets = target[valid_mask]
                if (valid_targets < 0).any() or (valid_targets >= self.num_classes).any():
                    raise ValueError(
                        f"Target contains invalid values. Expected [0, {self.num_classes-1}] or {self.ignore_index}, "
                        f"but got min={target.min().item()}, max={target.max().item()}"
                    )

        probs = F.softmax(pred, dim=1)  # (B, C, H, W)

        target_clamped = target.clamp(0, self.num_classes - 1)
        target_probs = probs.gather(1, target_clamped.unsqueeze(1)).squeeze(1)  # (B, H, W)

        focal_weight = (1 - target_probs) ** self.gamma

        alpha_t = self.alpha_buffer[target_clamped]  # (B, H, W)

        ce = F.cross_entropy(pred, target, reduction='none', ignore_index=self.ignore_index)

        loss = alpha_t * focal_weight * ce

        return loss.mean()


class IoULoss(nn.Module):
    def __init__(self, num_classes=2, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, pred, target):
        pred_probs = F.softmax(pred, dim=1)  # (B, C, H, W)

        target_clamped = target.clamp(0, self.num_classes - 1)

        target_one_hot = F.one_hot(target_clamped, num_classes=self.num_classes)  # (B, H, W, C)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        intersection = (pred_probs * target_one_hot).sum(dim=(2, 3))  # (B, C)
        union = pred_probs.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3)) - intersection  # (B, C)

        iou = (intersection + self.smooth) / (union + self.smooth)  # (B, C)

        return 1.0 - iou.mean()


if __name__ == "__main__":
    print("Testing SegmentationLoss:")

    seg_loss_fn = SegmentationLoss(num_classes=2, ce_weight=1.0, dice_weight=1.0)

    pred = torch.randn(2, 2, 320, 320)  # (B, C, H, W)
    target = torch.randint(0, 2, (2, 320, 320))  # (B, H, W)

    losses = seg_loss_fn(pred, target)
    print("Losses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")

    print("\nTesting Focal Loss:")
    focal_loss_fn = FocalLoss(num_classes=2, alpha=0.25, gamma=2.0)
    focal_loss = focal_loss_fn(pred, target)
    print(f"Focal Loss: {focal_loss.item():.4f}")

    print("\nTesting IoU Loss:")
    iou_loss_fn = IoULoss(num_classes=2)
    iou_loss = iou_loss_fn(pred, target)
    print(f"IoU Loss: {iou_loss.item():.4f}")
