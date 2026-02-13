import numpy as np
import torch
from typing import Dict
from sklearn.metrics import average_precision_score, accuracy_score, precision_recall_fscore_support


class AttributeMetrics:

    def __init__(self, num_attrs: int = 26):
        self.num_attrs = num_attrs

        self.all_preds = []  # List of [num_attrs] predictions (probabilities)
        self.all_targets = []  # List of [num_attrs] targets (binary)

    def update(self, preds: Dict, targets: list):
        if 'attr' in preds:
            attr_logits = preds['attr']
            if attr_logits is None:
                return  # 跳过None的情况
            if isinstance(attr_logits, torch.Tensor):
                attr_probs = torch.sigmoid(attr_logits).cpu().numpy()
            else:
                attr_probs = 1.0 / (1.0 + np.exp(-attr_logits))
        elif 'attr_logits' in preds:
            attr_logits = preds['attr_logits']
            if attr_logits is None:
                return
            if isinstance(attr_logits, torch.Tensor):
                attr_probs = torch.sigmoid(attr_logits).cpu().numpy()
            else:
                attr_probs = 1.0 / (1.0 + np.exp(-attr_logits))
        elif 'attr_probs' in preds:
            attr_probs = preds['attr_probs']
            if attr_probs is None:
                return
            if isinstance(attr_probs, torch.Tensor):
                attr_probs = attr_probs.cpu().numpy()
        else:
            raise ValueError("preds must contain 'attr', 'attr_logits' or 'attr_probs'")

        batch_size = len(targets)
        for i in range(batch_size):
            pred_attrs = attr_probs[i]
            self.all_preds.append(pred_attrs)

            target_attrs = targets[i]['attributes']
            if isinstance(target_attrs, torch.Tensor):
                target_attrs = target_attrs.cpu().numpy()

            self.all_targets.append(target_attrs)

    def compute(self) -> Dict[str, float]:
        if len(self.all_preds) == 0:
            return {
                'mAP': 0.0,
                'AP_per_attr': [0.0] * self.num_attrs,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }

        all_preds = np.array(self.all_preds)  # [N, num_attrs]
        all_targets = np.array(self.all_targets)  # [N, num_attrs]

        ap_per_attr = []
        for attr_idx in range(self.num_attrs):
            pred_attr = all_preds[:, attr_idx]
            target_attr = all_targets[:, attr_idx]

            if len(np.unique(target_attr)) < 2:
                ap_per_attr.append(0.0)
                continue

            try:
                ap = average_precision_score(target_attr, pred_attr)
                ap_per_attr.append(ap)
            except Exception as e:
                print(f"Warning: Failed to compute AP for attribute {attr_idx}: {e}")
                ap_per_attr.append(0.0)

        map_value = np.mean(ap_per_attr)

        binary_preds = (all_preds > 0.5).astype(int)

        sample_accuracy = (binary_preds == all_targets).all(axis=1).mean()

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets.flatten(),
            binary_preds.flatten(),
            average='binary',
            zero_division=0
        )

        return {
            'mAP': map_value,
            'AP_per_attr': ap_per_attr,
            'sample_accuracy': sample_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def reset(self):
        self.all_preds = []
        self.all_targets = []


if __name__ == "__main__":
    print("Testing AttributeMetrics...")

    metrics = AttributeMetrics(num_attrs=5)

    preds = {
        'attr_logits': torch.tensor([
            [2.0, -1.0, 0.5, -0.5, 1.5],
            [1.0, 0.5, -1.5, 1.0, -0.5],
            [-0.5, 2.0, 1.0, 0.0, 0.5]
        ])
    }

    targets = [
        {'attributes': torch.tensor([1, 0, 1, 0, 1])},
        {'attributes': torch.tensor([1, 1, 0, 1, 0])},
        {'attributes': torch.tensor([0, 1, 1, 0, 1])}
    ]

    metrics.update(preds, targets)

    preds2 = {
        'attr_logits': torch.tensor([
            [1.5, -0.5, 1.0, 0.5, -1.0],
            [0.0, 1.0, -1.0, 0.5, 1.5]
        ])
    }

    targets2 = [
        {'attributes': torch.tensor([1, 0, 1, 1, 0])},
        {'attributes': torch.tensor([0, 1, 0, 1, 1])}
    ]

    metrics.update(preds2, targets2)

    results = metrics.compute()
    print("\nResults:")
    print(f"  mAP: {results['mAP']:.4f}")
    print(f"  Sample accuracy: {results['sample_accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  F1: {results['f1']:.4f}")
    print(f"  AP per attr: {[f'{ap:.4f}' for ap in results['AP_per_attr']]}")

    print("\n✅ AttributeMetrics test passed!")
