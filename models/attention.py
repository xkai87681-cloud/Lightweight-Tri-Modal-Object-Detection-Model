import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        pool_out = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        attention = self.sigmoid(self.conv(pool_out))  # [B, 1, H, W]
        return x * attention


class CBAM(nn.Module):

    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class TaskSpecificAttention(nn.Module):

    def __init__(self, in_channels, num_tasks=3, reduction=16):
        super().__init__()
        self.num_tasks = num_tasks
        self.in_channels = in_channels

        self.task_attentions = nn.ModuleList([
            CBAM(in_channels, reduction=reduction)
            for _ in range(num_tasks)
        ])

    def forward(self, x, task_id):
        if task_id < 0 or task_id >= self.num_tasks:
            raise ValueError(f"Invalid task_id: {task_id}, must be in [0, {self.num_tasks})")

        return self.task_attentions[task_id](x)


class MultiScaleTaskAttention(nn.Module):

    def __init__(self, in_channels_list, num_tasks=3, reduction=16):
        super().__init__()
        self.num_scales = len(in_channels_list)
        self.num_tasks = num_tasks

        self.scale_attentions = nn.ModuleList([
            TaskSpecificAttention(in_channels, num_tasks, reduction)
            for in_channels in in_channels_list
        ])

    def forward(self, features, task_id):
        attended_features = []
        for i, (feat, attention) in enumerate(zip(features, self.scale_attentions)):
            attended_feat = attention(feat, task_id)
            attended_features.append(attended_feat)

        return attended_features


class FeatureFusion(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x_original, x_attended):
        return self.alpha * x_original + self.beta * x_attended


def build_task_attention(in_channels_list, num_tasks=3, reduction=16, use_fusion=True):
    attention = MultiScaleTaskAttention(in_channels_list, num_tasks, reduction)

    fusion = None
    if use_fusion:
        fusion = nn.ModuleList([
            FeatureFusion(in_channels)
            for in_channels in in_channels_list
        ])

    return attention, fusion


if __name__ == '__main__':
    print("========== Testing Attention Modules ==========")

    print("\n1. Testing ChannelAttention...")
    ca = ChannelAttention(in_channels=256, reduction=16)
    x = torch.randn(2, 256, 32, 32)
    out = ca(x)
    print(f"Input: {x.shape}, Output: {out.shape}")
    assert out.shape == x.shape, "Shape mismatch!"

    print("\n2. Testing SpatialAttention...")
    sa = SpatialAttention(kernel_size=7)
    out = sa(x)
    print(f"Input: {x.shape}, Output: {out.shape}")
    assert out.shape == x.shape, "Shape mismatch!"

    print("\n3. Testing CBAM...")
    cbam = CBAM(in_channels=256, reduction=16)
    out = cbam(x)
    print(f"Input: {x.shape}, Output: {out.shape}")
    assert out.shape == x.shape, "Shape mismatch!"

    print("\n4. Testing TaskSpecificAttention...")
    tsa = TaskSpecificAttention(in_channels=256, num_tasks=3, reduction=16)
    for task_id in range(3):
        out = tsa(x, task_id=task_id)
        print(f"Task {task_id} - Input: {x.shape}, Output: {out.shape}")
        assert out.shape == x.shape, "Shape mismatch!"

    print("\n5. Testing MultiScaleTaskAttention...")
    in_channels_list = [128, 256, 512]  # P3, P4, P5
    msta = MultiScaleTaskAttention(in_channels_list, num_tasks=3, reduction=16)

    p3 = torch.randn(2, 128, 80, 80)
    p4 = torch.randn(2, 256, 40, 40)
    p5 = torch.randn(2, 512, 20, 20)
    features = [p3, p4, p5]

    for task_id in range(3):
        attended_features = msta(features, task_id=task_id)
        print(f"\nTask {task_id}:")
        for i, (orig, att) in enumerate(zip(features, attended_features)):
            print(f"  P{i+3}: {orig.shape} -> {att.shape}")
            assert att.shape == orig.shape, "Shape mismatch!"

    print("\n6. Testing FeatureFusion...")
    ff = FeatureFusion(in_channels=256)
    x_orig = torch.randn(2, 256, 32, 32)
    x_att = torch.randn(2, 256, 32, 32)
    out = ff(x_orig, x_att)
    print(f"Original: {x_orig.shape}, Attended: {x_att.shape}, Fused: {out.shape}")
    assert out.shape == x_orig.shape, "Shape mismatch!"

    print("\n7. Testing build_task_attention...")
    attention, fusion = build_task_attention(
        in_channels_list=[128, 256, 512],
        num_tasks=3,
        reduction=16,
        use_fusion=True
    )
    print(f"Attention: {type(attention)}")
    print(f"Fusion: {type(fusion)}, Length: {len(fusion) if fusion else 0}")

    print("\n========== Parameter Statistics ==========")
    total_params = sum(p.numel() for p in attention.parameters())
    print(f"MultiScaleTaskAttention params: {total_params / 1e3:.2f}K")

    if fusion:
        fusion_params = sum(p.numel() for p in fusion.parameters())
        print(f"FeatureFusion params: {fusion_params}")

    print("\nAll tests passed!")
