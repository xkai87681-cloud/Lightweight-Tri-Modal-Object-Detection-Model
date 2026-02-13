import numpy as np
import torch
from typing import Dict


class SegmentationMetrics:

    def __init__(self, num_classes: int = 2, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, preds: Dict, targets: list):
        if 'seg' in preds:
            seg_logits = preds['seg']
            if isinstance(seg_logits, torch.Tensor):
                seg_preds = torch.argmax(seg_logits, dim=1).cpu().numpy()  # [B, H, W]
            else:
                seg_preds = np.argmax(seg_logits, axis=1)
        elif 'seg_logits' in preds:
            seg_logits = preds['seg_logits']
            if isinstance(seg_logits, torch.Tensor):
                seg_preds = torch.argmax(seg_logits, dim=1).cpu().numpy()  # [B, H, W]
            else:
                seg_preds = np.argmax(seg_logits, axis=1)
        elif 'seg_mask' in preds:
            seg_preds = preds['seg_mask']
            if isinstance(seg_preds, torch.Tensor):
                seg_preds = seg_preds.cpu().numpy()
        else:
            raise ValueError("preds must contain 'seg', 'seg_logits' or 'seg_mask'")

        batch_size = len(targets)
        for i in range(batch_size):
            pred_mask = seg_preds[i]  # [H, W]

            target_mask = targets[i]['seg_mask']
            if isinstance(target_mask, torch.Tensor):
                target_mask = target_mask.cpu().numpy()

            self._update_confusion_matrix(pred_mask, target_mask)

    def _update_confusion_matrix(self, pred: np.ndarray, target: np.ndarray):
        mask = (target != self.ignore_index)
        pred = pred[mask]
        target = target[mask]

        pred = np.clip(pred, 0, self.num_classes - 1)
        target = np.clip(target, 0, self.num_classes - 1)

        indices = self.num_classes * target + pred

        bincount = np.bincount(
            indices.astype(np.int64),
            minlength=self.num_classes ** 2
        )
        self.confusion_matrix += bincount.reshape(self.num_classes, self.num_classes)

    def compute(self) -> Dict[str, float]:
        iou_per_class = []
        for class_id in range(self.num_classes):
            tp = self.confusion_matrix[class_id, class_id]

            fp_fn = (
                self.confusion_matrix[class_id, :].sum() +
                self.confusion_matrix[:, class_id].sum() -
                tp
            )

            if fp_fn == 0:
                iou = 0.0
            else:
                iou = tp / fp_fn

            iou_per_class.append(iou)

        miou = np.mean(iou_per_class)

        total_pixels = self.confusion_matrix.sum()
        correct_pixels = np.diag(self.confusion_matrix).sum()

        if total_pixels > 0:
            pixel_acc = correct_pixels / total_pixels
        else:
            pixel_acc = 0.0

        class_acc = []
        for class_id in range(self.num_classes):
            class_total = self.confusion_matrix[class_id, :].sum()
            class_correct = self.confusion_matrix[class_id, class_id]

            if class_total > 0:
                acc = class_correct / class_total
            else:
                acc = 0.0

            class_acc.append(acc)

        mean_acc = np.mean(class_acc)

        return {
            'mIoU': miou,
            'IoU_per_class': iou_per_class,
            'pixel_acc': pixel_acc,
            'mean_acc': mean_acc,
            'class_acc': class_acc
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)


if __name__ == "__main__":
    print("Testing SegmentationMetrics...")

    metrics = SegmentationMetrics(num_classes=3, ignore_index=255)

    preds = {
        'seg_logits': torch.tensor([
            [
                [[2.0, 1.0, 0.5, 0.0],
                 [1.5, 2.0, 1.0, 0.5],
                 [1.0, 1.5, 2.0, 1.0],
                 [0.5, 1.0, 1.5, 2.0]],

                [[1.0, 2.0, 1.5, 1.0],
                 [2.0, 1.0, 2.0, 1.5],
                 [1.5, 1.0, 1.0, 2.0],
                 [1.0, 0.5, 1.0, 1.5]],

                [[0.5, 0.5, 2.0, 2.0],
                 [0.5, 0.5, 0.5, 2.0],
                 [2.0, 2.0, 0.5, 0.5],
                 [2.0, 2.0, 0.5, 0.5]]
            ],
            [
                [[2.0, 2.0, 1.0, 1.0],
                 [2.0, 2.0, 1.0, 1.0],
                 [1.0, 1.0, 0.5, 0.5],
                 [1.0, 1.0, 0.5, 0.5]],

                [[1.0, 1.0, 2.0, 2.0],
                 [1.0, 1.0, 2.0, 2.0],
                 [2.0, 2.0, 1.0, 1.0],
                 [2.0, 2.0, 1.0, 1.0]],

                [[0.5, 0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5, 0.5],
                 [0.5, 0.5, 2.0, 2.0],
                 [0.5, 0.5, 2.0, 2.0]]
            ]
        ])
    }

    targets = [
        {
            'seg_mask': torch.tensor([
                [0, 1, 2, 2],
                [0, 0, 1, 2],
                [2, 2, 0, 1],
                [2, 2, 0, 0]
            ])
        },
        {
            'seg_mask': torch.tensor([
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [1, 1, 2, 2],
                [1, 1, 2, 2]
            ])
        }
    ]

    metrics.update(preds, targets)

    results = metrics.compute()
    print("\nResults:")
    print(f"  mIoU: {results['mIoU']:.4f}")
    print(f"  Pixel Acc: {results['pixel_acc']:.4f}")
    print(f"  Mean Acc: {results['mean_acc']:.4f}")
    print(f"  IoU per class: {[f'{iou:.4f}' for iou in results['IoU_per_class']]}")
    print(f"  Class acc: {[f'{acc:.4f}' for acc in results['class_acc']]}")

    print("\n✅ SegmentationMetrics test passed!")
