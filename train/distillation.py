import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class DetectionDistillationLoss(nn.Module):

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        feature_weight: float = 0.1
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.feature_weight = feature_weight

    def forward(
        self,
        student_cls: torch.Tensor,
        teacher_cls: torch.Tensor,
        student_features: Optional[torch.Tensor] = None,
        teacher_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        losses = {}

        student_soft = F.log_softmax(student_cls / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_cls / self.temperature, dim=1)

        kd_loss = F.kl_div(
            student_soft,
            teacher_soft,
            reduction='batchmean'
        ) * (self.temperature ** 2)

        losses['kd_cls'] = kd_loss * self.alpha

        if student_features is not None and teacher_features is not None:
            if student_features.shape[1] != teacher_features.shape[1]:
                pass
            else:
                feature_loss = F.mse_loss(student_features, teacher_features)
                losses['kd_feature'] = feature_loss * self.feature_weight

        return losses


class YOLOv8Teacher(nn.Module):

    def __init__(self, model_name: str = 'yolov8n.pt', device: str = 'cuda'):
        super().__init__()
        self.device = device

        try:
            from ultralytics import YOLO
            self.yolo = YOLO(model_name)
            self.yolo.to(device)
            self.yolo.model.eval()
            print(f"[Teacher] ✅ Loaded YOLOv8 teacher: {model_name}")

            for param in self.yolo.model.parameters():
                param.requires_grad = False

            self.available = True

        except Exception as e:
            print(f"[Teacher] ⚠️  Failed to load YOLOv8: {e}")
            print(f"[Teacher] Knowledge distillation will be disabled")
            self.available = False
            self.yolo = None

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not self.available:
            return {}

        try:
            outputs = {}

            x = images
            features = []

            for i, layer in enumerate(self.yolo.model.model):
                x = layer(x)
                if i in [15, 18, 21]:
                    features.append(x)

            if len(features) == 3:
                outputs['features'] = features

            return outputs

        except Exception as e:
            print(f"[Teacher] Forward error: {e}")
            return {}

    def get_soft_labels(
        self,
        images: torch.Tensor,
        conf_threshold: float = 0.001
    ) -> Optional[torch.Tensor]:
        if not self.available:
            return None

        try:
            results = self.yolo.predict(
                images,
                conf=conf_threshold,
                verbose=False
            )
            return results
        except Exception as e:
            print(f"[Teacher] Prediction error: {e}")
            return None


def create_distillation_trainer(
    student_model: nn.Module,
    teacher_model_name: str = 'yolov8n.pt',
    temperature: float = 4.0,
    alpha: float = 0.3,
    device: str = 'cuda'
) -> Tuple[Optional[YOLOv8Teacher], Optional[DetectionDistillationLoss]]:
    teacher = YOLOv8Teacher(teacher_model_name, device)

    if not teacher.available:
        return None, None

    distill_loss = DetectionDistillationLoss(
        temperature=temperature,
        alpha=alpha
    )

    return teacher, distill_loss


if __name__ == '__main__':
    print("=" * 60)
    print("Testing Knowledge Distillation Module")
    print("=" * 60)

    distill_loss = DetectionDistillationLoss()

    student_cls = torch.randn(2, 80, 20, 20)
    teacher_cls = torch.randn(2, 80, 20, 20)

    losses = distill_loss(student_cls, teacher_cls)
    print(f"Distillation losses: {losses}")

    try:
        teacher = YOLOv8Teacher('yolov8n.pt', device='cpu')
        if teacher.available:
            images = torch.randn(1, 3, 640, 640)
            outputs = teacher(images)
            print(f"Teacher outputs: {outputs.keys() if outputs else 'None'}")
    except Exception as e:
        print(f"Teacher test skipped: {e}")
