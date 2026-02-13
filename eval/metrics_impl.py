import torch
import numpy as np


def calculate_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / (union_area + 1e-6)
    return iou


def calculate_detection_ap(predictions, targets, iou_threshold=0.5):
    all_scores = []
    all_tp = []
    n_gt = 0  # 总的ground truth数量

    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes']  # (N, 4)
        pred_scores = pred['scores']  # (N,)
        gt_boxes = target['boxes']  # (M, 4)

        n_gt += len(gt_boxes)

        if len(pred_boxes) == 0:
            continue

        if len(gt_boxes) == 0:
            all_scores.extend(pred_scores.tolist())
            all_tp.extend([0] * len(pred_boxes))
            continue

        gt_matched = np.zeros(len(gt_boxes), dtype=bool)

        for i in range(len(pred_boxes)):
            best_iou = 0
            best_gt_idx = -1

            for j in range(len(gt_boxes)):
                if gt_matched[j]:
                    continue
                iou = calculate_iou(pred_boxes[i], gt_boxes[j])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            all_scores.append(pred_scores[i])
            if best_iou >= iou_threshold:
                all_tp.append(1)
                gt_matched[best_gt_idx] = True
            else:
                all_tp.append(0)

    if n_gt == 0:
        return 0.0

    indices = np.argsort(all_scores)[::-1]
    all_tp = np.array(all_tp)[indices]

    tp_cumsum = np.cumsum(all_tp)
    fp_cumsum = np.cumsum(1 - all_tp)

    recalls = tp_cumsum / n_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    recalls = np.concatenate([[0], recalls, [1]])
    precisions = np.concatenate([[1], precisions, [0]])

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])

    return ap


def calculate_detection_metrics(predictions, targets, num_classes=2, class_names=None):
    if class_names is None:
        class_names = [f'class_{i}' for i in range(num_classes)]

    predictions_by_class = [[] for _ in range(num_classes)]
    targets_by_class = [[] for _ in range(num_classes)]

    for pred, target in zip(predictions, targets):
        for c in range(num_classes):
            pred_mask = pred['labels'] == c
            pred_cls = {
                'boxes': pred['boxes'][pred_mask].cpu().numpy(),
                'scores': pred['scores'][pred_mask].cpu().numpy()
            }
            predictions_by_class[c].append(pred_cls)

            target_mask = target['labels'] == c
            target_cls = {
                'boxes': target['boxes'][target_mask].cpu().numpy()
            }
            targets_by_class[c].append(target_cls)

    iou_thresholds = [0.5, 0.75] + list(np.arange(0.5, 1.0, 0.05))

    aps_by_iou = {iou: [] for iou in iou_thresholds}
    aps_by_class_at_50 = []

    for c in range(num_classes):
        for iou_thresh in iou_thresholds:
            ap = calculate_detection_ap(
                predictions_by_class[c],
                targets_by_class[c],
                iou_threshold=iou_thresh
            )
            aps_by_iou[iou_thresh].append(ap)

            if iou_thresh == 0.5:
                aps_by_class_at_50.append(ap)

    mAP_50 = np.mean(aps_by_iou[0.5])
    mAP_75 = np.mean(aps_by_iou[0.75])
    mAP = np.mean([np.mean(aps) for aps in aps_by_iou.values()])

    metrics = {
        'mAP_50': float(mAP_50),
        'mAP_75': float(mAP_75),
        'mAP': float(mAP),
    }

    for i, (name, ap) in enumerate(zip(class_names, aps_by_class_at_50)):
        metrics[f'AP_50_{name}'] = float(ap)

    return metrics


def calculate_precision_recall_f1(predictions, targets):
    tp = np.sum((predictions == 1) & (targets == 1))
    fp = np.sum((predictions == 1) & (targets == 0))
    fn = np.sum((predictions == 0) & (targets == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def calculate_ap_from_pr(predictions, targets):
    indices = np.argsort(predictions)[::-1]
    targets_sorted = targets[indices]

    tp_cumsum = np.cumsum(targets_sorted)
    fp_cumsum = np.cumsum(1 - targets_sorted)

    n_pos = np.sum(targets)

    if n_pos == 0:
        return 0.0

    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    recalls = tp_cumsum / n_pos

    precisions = np.concatenate([[1], precisions])
    recalls = np.concatenate([[0], recalls])

    ap = 0.0
    for i in range(len(recalls) - 1):
        ap += (recalls[i + 1] - recalls[i]) * precisions[i + 1]

    return ap


def calculate_attribute_metrics(predictions, targets, threshold=0.5):
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()

    pred_binary = (predictions >= threshold).astype(int)

    accuracy = (pred_binary == targets).mean()

    per_attr_acc = (pred_binary == targets).mean(axis=0)
    mean_accuracy = per_attr_acc.mean()

    pred_flat = pred_binary.reshape(-1)
    target_flat = targets.reshape(-1)

    precision, recall, f1 = calculate_precision_recall_f1(pred_flat, target_flat)

    aps = []
    for i in range(targets.shape[1]):
        try:
            ap = calculate_ap_from_pr(predictions[:, i], targets[:, i])
            aps.append(ap)
        except:
            aps.append(0.0)

    mean_ap = np.mean(aps)

    metrics = {
        'accuracy': float(accuracy),
        'mean_accuracy': float(mean_accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'mAP': float(mean_ap),
        'per_attribute_acc': per_attr_acc.tolist(),
        'per_attribute_ap': aps
    }

    return metrics


def calculate_segmentation_metrics(predictions, targets, num_classes, ignore_index=255):
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()

    valid_mask = (targets != ignore_index)

    correct = (predictions == targets) & valid_mask
    pixel_acc = correct.sum() / valid_mask.sum()

    ious = []
    for c in range(num_classes):
        pred_c = (predictions == c)
        target_c = (targets == c)

        pred_c = pred_c & valid_mask
        target_c = target_c & valid_mask

        intersection = (pred_c & target_c).sum()
        union = (pred_c | target_c).sum()

        if union == 0:
            iou = float('nan')
        else:
            iou = intersection / union

        ious.append(iou)

    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    miou = np.mean(valid_ious) if valid_ious else 0.0

    metrics = {
        'mIoU': float(miou),
        'pixel_accuracy': float(pixel_acc),
        'per_class_iou': [float(iou) if not np.isnan(iou) else 0.0 for iou in ious],
        'num_valid_classes': len(valid_ious)
    }

    return metrics


if __name__ == "__main__":
    print("Testing metrics implementation...")

    print("\n1. Testing Detection Metrics:")
    pred = [{
        'boxes': torch.tensor([[10, 10, 50, 50], [60, 60, 100, 100]]),
        'scores': torch.tensor([0.9, 0.8]),
        'labels': torch.tensor([0, 1])
    }]
    target = [{
        'boxes': torch.tensor([[12, 12, 48, 48], [65, 65, 95, 95]]),
        'labels': torch.tensor([0, 1])
    }]
    det_metrics = calculate_detection_metrics(pred, target, num_classes=2)
    print(f"  mAP@0.5: {det_metrics['mAP_50']:.4f}")
    print(f"  mAP@0.75: {det_metrics['mAP_75']:.4f}")

    print("\n2. Testing Attribute Metrics:")
    pred_attr = torch.rand(100, 26)  # 100 samples, 26 attributes
    target_attr = (torch.rand(100, 26) > 0.5).float()
    attr_metrics = calculate_attribute_metrics(pred_attr, target_attr)
    print(f"  Accuracy: {attr_metrics['accuracy']:.4f}")
    print(f"  mAP: {attr_metrics['mAP']:.4f}")
    print(f"  F1: {attr_metrics['f1']:.4f}")

    print("\n3. Testing Segmentation Metrics:")
    pred_seg = torch.randint(0, 2, (10, 640, 640))
    target_seg = torch.randint(0, 2, (10, 640, 640))
    seg_metrics = calculate_segmentation_metrics(pred_seg, target_seg, num_classes=2)
    print(f"  mIoU: {seg_metrics['mIoU']:.4f}")
    print(f"  Pixel Acc: {seg_metrics['pixel_accuracy']:.4f}")

    print("\nAll tests passed!")
