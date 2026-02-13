import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from pathlib import Path
import os


class MobileNetV3Backbone(nn.Module):

    def __init__(self, pretrained=True, pretrained_type='detection', out_indices=(6, 12, 16)):
        super().__init__()

        project_root = Path(__file__).parent.parent

        if pretrained:
            print(f"[Backbone] Loading pretrained weights (type={pretrained_type})...")

            if pretrained_type == 'detection':
                mobilenet = self._load_detection_pretrained(project_root)
            else:
                mobilenet = self._load_imagenet_pretrained(project_root)
        else:
            mobilenet = mobilenet_v3_large(weights=None)
            print(f"[Backbone] Using randomly initialized weights (pretrained=False)")

        self.features = mobilenet.features
        self.out_indices = out_indices

        self.out_channels = [40, 112, 960]  # C3, C4, C5

        self._freeze_bn()

    def _load_detection_pretrained(self, project_root):
        try:
            from torchvision.models.detection import ssdlite320_mobilenet_v3_large
            from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights

            local_weight_path = project_root / 'ssdlite320_mobilenet_v3_large_coco.pth'

            if local_weight_path.exists():
                print(f"[Backbone] ✅ Found local COCO detection weights: {local_weight_path}")
                ssd_model = ssdlite320_mobilenet_v3_large(weights=None)
                state_dict = torch.load(local_weight_path, map_location='cpu')
                ssd_model.load_state_dict(state_dict)
            else:
                print(f"[Backbone] 📥 Downloading COCO detection pretrained weights...")
                weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
                ssd_model = ssdlite320_mobilenet_v3_large(weights=weights)
                print(f"[Backbone] ✅ Loaded COCO detection pretrained weights from torchvision")

                try:
                    torch.save(ssd_model.state_dict(), local_weight_path)
                    print(f"[Backbone] 💾 Saved weights to {local_weight_path}")
                except Exception as e:
                    print(f"[Backbone] ⚠️  Could not save weights locally: {e}")

            mobilenet = mobilenet_v3_large(weights=None)

            ssd_backbone_state = ssd_model.backbone.state_dict()

            mobilenet_state = mobilenet.state_dict()
            new_state = {}

            for key in mobilenet_state.keys():
                if key.startswith('features.'):
                    ssd_key = key  # SSDLite backbone 也使用 features.x 格式
                    if ssd_key in ssd_backbone_state:
                        new_state[key] = ssd_backbone_state[ssd_key]
                    else:
                        new_state[key] = mobilenet_state[key]
                else:
                    new_state[key] = mobilenet_state[key]

            mobilenet.load_state_dict(new_state, strict=False)
            print(f"[Backbone] ✅ Extracted COCO detection pretrained backbone")
            print(f"[Backbone] 🎯 This backbone has been optimized for detection tasks!")

            return mobilenet

        except Exception as e:
            print(f"[Backbone] ⚠️  Failed to load detection weights: {e}")
            print(f"[Backbone] Falling back to ImageNet pretrained...")
            return self._load_imagenet_pretrained(project_root)

    def _load_imagenet_pretrained(self, project_root):
        local_weight_path = project_root / 'mobilenet_v3_large-8738ca79.pth'

        if local_weight_path.exists():
            print(f"[Backbone] ✅ Found local ImageNet weights: {local_weight_path}")
            mobilenet = mobilenet_v3_large(weights=None)
            state_dict = torch.load(local_weight_path, map_location='cpu')
            mobilenet.load_state_dict(state_dict)
            print(f"[Backbone] ✅ Loaded ImageNet pretrained weights from local file")
        else:
            print(f"[Backbone] ⚠️  Local weights not found at {local_weight_path}")
            print(f"[Backbone] Trying to load from torchvision (may download)...")
            try:
                weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
                mobilenet = mobilenet_v3_large(weights=weights)
                print(f"[Backbone] ✅ Loaded ImageNet pretrained weights from torchvision")
            except Exception as e:
                print(f"[Backbone] ❌ Failed to load pretrained weights: {e}")
                mobilenet = mobilenet_v3_large(weights=None)
                print(f"[Backbone] ⚠️  Using randomly initialized weights!")

        return mobilenet

    def _freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward(self, x):
        outputs = []

        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.out_indices:
                outputs.append(x)

        return outputs

    def train(self, mode=True):
        super().train(mode)
        self._freeze_bn()
        return self


def build_mobilenetv3_backbone(pretrained=True, pretrained_type='detection'):
    return MobileNetV3Backbone(pretrained=pretrained, pretrained_type=pretrained_type)


if __name__ == '__main__':
    print("=" * 60)
    print("Testing MobileNetV3 Backbone with Detection Pretrained")
    print("=" * 60)

    backbone = build_mobilenetv3_backbone(pretrained=True, pretrained_type='detection')
    backbone.eval()

    x = torch.randn(2, 3, 640, 640)
    c3, c4, c5 = backbone(x)

    print(f"\nInput shape: {x.shape}")
    print(f"C3 shape: {c3.shape}  (stride=8,  channels=40)")
    print(f"C4 shape: {c4.shape}  (stride=16, channels=112)")
    print(f"C5 shape: {c5.shape}  (stride=32, channels=960)")

    total_params = sum(p.numel() for p in backbone.parameters())
    trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"\nTotal params: {total_params / 1e6:.2f}M")
    print(f"Trainable params: {trainable_params / 1e6:.2f}M")
