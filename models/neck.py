import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.cv2 = Conv(hidden_channels, out_channels, 3, 1)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):

    def __init__(self, in_channels, out_channels, n=1, shortcut=False, expansion=0.5):
        super().__init__()
        self.c = int(out_channels * expansion)
        self.cv1 = Conv(in_channels, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, out_channels, 1, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, expansion=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):

    def __init__(self, in_channels, out_channels, k=5):
        super().__init__()
        c_ = in_channels // 2
        self.cv1 = Conv(in_channels, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))


class YOLOv8Neck(nn.Module):

    def __init__(self, in_channels=(40, 112, 960), out_channels=(128, 256, 512), depth=3):
        super().__init__()
        c3_in, c4_in, c5_in = in_channels
        p3_out, p4_out, p5_out = out_channels

        self.reduce_c5 = Conv(c5_in, p5_out, 1, 1)
        self.sppf = SPPF(p5_out, p5_out, k=5)

        self.upsample_p5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.reduce_c4 = Conv(c4_in, p4_out, 1, 1)
        self.c2f_p4 = C2f(p4_out + p5_out, p4_out, n=depth, shortcut=False)

        self.upsample_p4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.reduce_c3 = Conv(c3_in, p3_out, 1, 1)
        self.c2f_p3 = C2f(p3_out + p4_out, p3_out, n=depth, shortcut=False)

        self.downsample_p3 = Conv(p3_out, p3_out, 3, 2)
        self.c2f_p4_out = C2f(p3_out + p4_out, p4_out, n=depth, shortcut=False)

        self.downsample_p4 = Conv(p4_out, p4_out, 3, 2)
        self.c2f_p5_out = C2f(p4_out + p5_out, p5_out, n=depth, shortcut=False)

        self.out_channels = out_channels

    def forward(self, features):
        c3, c4, c5 = features

        p5 = self.sppf(self.reduce_c5(c5))

        p5_up = self.upsample_p5(p5)
        c4_reduced = self.reduce_c4(c4)
        p4 = self.c2f_p4(torch.cat([c4_reduced, p5_up], dim=1))

        p4_up = self.upsample_p4(p4)
        c3_reduced = self.reduce_c3(c3)
        p3 = self.c2f_p3(torch.cat([c3_reduced, p4_up], dim=1))

        p3_down = self.downsample_p3(p3)
        p4_out = self.c2f_p4_out(torch.cat([p3_down, p4], dim=1))

        p4_down = self.downsample_p4(p4_out)
        p5_out = self.c2f_p5_out(torch.cat([p4_down, p5], dim=1))

        return [p3, p4_out, p5_out]


def build_yolov8_neck(in_channels=(40, 112, 960), out_channels=(128, 256, 512), depth=3):
    return YOLOv8Neck(in_channels, out_channels, depth)


if __name__ == '__main__':
    neck = build_yolov8_neck()
    neck.eval()

    c3 = torch.randn(2, 40, 80, 80)    # stride=8
    c4 = torch.randn(2, 112, 40, 40)   # stride=16
    c5 = torch.randn(2, 960, 20, 20)   # stride=32

    p3, p4, p5 = neck([c3, c4, c5])

    print("YOLOv8 Neck Test:")
    print(f"C3 shape: {c3.shape} -> P3 shape: {p3.shape}")
    print(f"C4 shape: {c4.shape} -> P4 shape: {p4.shape}")
    print(f"C5 shape: {c5.shape} -> P5 shape: {p5.shape}")

    total_params = sum(p.numel() for p in neck.parameters())
    print(f"\nTotal params: {total_params / 1e6:.2f}M")
