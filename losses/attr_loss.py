import torch
import torch.nn as nn
import torch.nn.functional as F


class AttributeLoss(nn.Module):
    def __init__(self, num_attrs=26, pos_weight=None, label_smoothing=0.0):
        super().__init__()
        self.num_attrs = num_attrs
        self.label_smoothing = label_smoothing

        if pos_weight is not None:
            if isinstance(pos_weight, (list, tuple)):
                pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
            elif isinstance(pos_weight, torch.Tensor):
                pos_weight = pos_weight.float()
            self.register_buffer('pos_weight', pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits, targets):
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        if self.pos_weight is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits, targets,
                pos_weight=self.pos_weight,
                reduction='mean'
            )
        else:
            loss = F.binary_cross_entropy_with_logits(
                logits, targets,
                reduction='mean'
            )

        return {
            'attr_loss': loss
        }


class FocalBCELoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)

        focal_weight = (1 - pt) ** self.gamma

        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        loss = alpha_weight * focal_weight * bce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class AsymmetricFocalLoss(nn.Module):

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        if self.clip is not None and self.clip > 0:
            probs = (probs + self.clip).clamp(max=1)

        probs_pos = probs
        probs_neg = 1 - probs

        loss_pos = targets * torch.log(probs_pos.clamp(min=self.eps))
        loss_pos = loss_pos * ((1 - probs_pos) ** self.gamma_pos)

        loss_neg = (1 - targets) * torch.log(probs_neg.clamp(min=self.eps))
        loss_neg = loss_neg * (probs_pos ** self.gamma_neg)

        loss = -(loss_pos + loss_neg).mean()

        return loss


class WeightedAttributeLoss(nn.Module):
    def __init__(self, num_attrs=26, attr_weights=None):
        super().__init__()
        self.num_attrs = num_attrs

        if attr_weights is None:
            attr_weights = torch.ones(num_attrs)
        else:
            attr_weights = torch.tensor(attr_weights, dtype=torch.float32)

        self.register_buffer('attr_weights', attr_weights)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce = self.bce_loss(logits, targets)  # (B, num_attrs)

        weighted_bce = bce * self.attr_weights.unsqueeze(0)  # (B, num_attrs)

        loss = weighted_bce.mean()

        return {
            'attr_loss': loss
        }


if __name__ == "__main__":
    print("Testing AttributeLoss:")

    attr_loss_fn = AttributeLoss(num_attrs=26)

    logits = torch.randn(4, 26)
    targets = torch.randint(0, 2, (4, 26)).float()

    losses = attr_loss_fn(logits, targets)
    print(f"Attribute Loss: {losses['attr_loss'].item():.4f}")

    print("\nTesting Focal BCE:")
    focal_loss_fn = FocalBCELoss(alpha=0.25, gamma=2.0)
    focal_loss = focal_loss_fn(logits, targets)
    print(f"Focal BCE Loss: {focal_loss.item():.4f}")

    print("\nTesting Asymmetric Focal Loss:")
    asl_fn = AsymmetricFocalLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
    asl_loss = asl_fn(logits, targets)
    print(f"Asymmetric Focal Loss: {asl_loss.item():.4f}")

    print("\nTesting Weighted Attribute Loss:")
    weights = torch.rand(26)  # 随机权重
    weighted_loss_fn = WeightedAttributeLoss(num_attrs=26, attr_weights=weights)
    weighted_losses = weighted_loss_fn(logits, targets)
    print(f"Weighted Attribute Loss: {weighted_losses['attr_loss'].item():.4f}")
