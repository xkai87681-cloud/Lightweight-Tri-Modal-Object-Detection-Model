import torch
import torch.nn as nn
import math


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1  # 确保是奇数
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)  # (B, C, 1, 1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))  # (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class DWConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.dwconv = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pwconv = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.dwconv(x)))
        x = self.act(self.bn2(self.pwconv(x)))
        return x


class TaskAdapter(nn.Module):
    def __init__(self, channels, reduction=16, attention='SE', use_residual=True, use_layernorm=True):
        super().__init__()
        self.use_residual = use_residual
        self.use_layernorm = use_layernorm

        self.dwconv = DWConv(channels, channels)

        if attention == 'SE':
            self.attn = SEBlock(channels, reduction)
        elif attention == 'ECA':
            self.attn = ECABlock(channels)
        else:
            self.attn = nn.Identity()

        if use_layernorm:
            self.norm = nn.GroupNorm(32, channels)  # GroupNorm作为LayerNorm的替代
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        identity = x

        x = self.dwconv(x)

        x = self.attn(x)

        x = self.norm(x)

        if self.use_residual:
            x = x + identity

        return x


class MultiScaleTaskAdapter(nn.Module):
    def __init__(self, channels=256, reduction=16, attention='SE', num_scales=3):
        super().__init__()
        self.adapters = nn.ModuleList([
            TaskAdapter(channels, reduction, attention) for _ in range(num_scales)
        ])

    def forward(self, features):
        return [adapter(feat) for adapter, feat in zip(self.adapters, features)]


if __name__ == "__main__":
    print("Testing Single TaskAdapter:")
    adapter = TaskAdapter(channels=256, attention='SE')
    x = torch.randn(2, 256, 80, 80)
    out = adapter(x)
    print(f"  Input: {x.shape}, Output: {out.shape}")

    print("\nTesting MultiScaleTaskAdapter:")
    ms_adapter = MultiScaleTaskAdapter(channels=256, attention='ECA', num_scales=3)
    features = [
        torch.randn(2, 256, 80, 80),  # P3
        torch.randn(2, 256, 40, 40),  # P4
        torch.randn(2, 256, 20, 20),  # P5
    ]
    adapted = ms_adapter(features)
    for i, feat in enumerate(adapted):
        print(f"  P{i+3}: {feat.shape}")
