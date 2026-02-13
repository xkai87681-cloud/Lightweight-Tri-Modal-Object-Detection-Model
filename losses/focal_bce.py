import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalBCELoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        p = torch.sigmoid(pred)

        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        p_t = p * target + (1 - p) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma

        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)

        focal_loss = alpha_t * focal_weight * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class FocalBCEWithLogitsLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):

        p = torch.sigmoid(pred)

        p_t = p * target + (1 - p) * (1 - target)

        focal_weight = (1 - p_t) ** self.gamma

        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)

        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        focal_loss = alpha_t * focal_weight * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


if __name__ == '__main__':
    pred = torch.randn(10, 5)  # [N, C]
    target = torch.randint(0, 2, (10, 5)).float()  # [N, C]

    bce_loss = F.binary_cross_entropy_with_logits(pred, target)
    print(f"BCE Loss: {bce_loss.item():.4f}")

    focal_bce = FocalBCEWithLogitsLoss(alpha=0.25, gamma=2.0)
    focal_loss = focal_bce(pred, target)
    print(f"Focal BCE Loss: {focal_loss.item():.4f}")

    print(f"Ratio (Focal/BCE): {focal_loss.item() / bce_loss.item():.4f}")
