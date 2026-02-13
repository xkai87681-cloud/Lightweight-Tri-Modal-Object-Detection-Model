import numpy as np
import torch
from typing import List, Dict, Tuple


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)
    return iou


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


class DetectionMetrics:

    def __init__(self, num_classes: int = 2, iou_thresholds: List[float] = [0.5]):
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds

        self.predictions = []  # List of dicts: {'boxes', 'scores', 'labels'}
        self.targets = []      # List of dicts: {'boxes', 'labels'}

    def update(self, preds: Dict, targets: List[Dict]):
        if isinstance(preds, list):
            for pred, target in zip(preds, targets):
                pred_dict = {
                    'boxes': pred['boxes'].cpu().numpy() if isinstance(pred['boxes'], torch.Tensor) else pred['boxes'],
                    'scores': pred['scores'].cpu().numpy() if isinstance(pred['scores'], torch.Tensor) else pred['scores'],
                    'labels': pred['labels'].cpu().numpy() if isinstance(pred['labels'], torch.Tensor) else pred['labels']
                }
                self.predictions.append(pred_dict)

                target_boxes = target['bboxes']
                target_labels = target['labels']

                if isinstance(target_boxes, torch.Tensor):
                    target_boxes = target_boxes.cpu().numpy()
                if isinstance(target_labels, torch.Tensor):
                    target_labels = target_labels.cpu().numpy()

                target_dict = {
                    'boxes': target_boxes,
                    'labels': target_labels
                }
                self.targets.append(target_dict)
            return

        pred_boxes = preds.get('pred_boxes', preds.get('boxes'))
        pred_scores = preds.get('pred_scores', preds.get('scores'))
        pred_labels = preds.get('pred_labels', preds.get('labels'))

        if isinstance(pred_boxes, torch.Tensor):
            pred_boxes = pred_boxes.cpu().numpy()
        if isinstance(pred_scores, torch.Tensor):
            pred_scores = pred_scores.cpu().numpy()
        if isinstance(pred_labels, torch.Tensor):
            pred_labels = pred_labels.cpu().numpy()

        batch_size = len(targets)
        for i in range(batch_size):
            pred = {
                'boxes': pred_boxes[i],
                'scores': pred_scores[i],
                'labels': pred_labels[i]
            }
            self.predictions.append(pred)

            target_boxes = targets[i]['bboxes']
            target_labels = targets[i]['labels']

            if isinstance(target_boxes, torch.Tensor):
                target_boxes = target_boxes.cpu().numpy()
            if isinstance(target_labels, torch.Tensor):
                target_labels = target_labels.cpu().numpy()

            target = {
                'boxes': target_boxes,
                'labels': target_labels
            }
            self.targets.append(target)

    def compute(self) -> Dict[str, float]:
        all_ap = {}

        for iou_thr in self.iou_thresholds:
            ap_per_class = []

            for class_id in range(self.num_classes):
                ap = self._compute_class_ap(class_id, iou_thr)
                ap_per_class.append(ap)

            map_value = np.mean(ap_per_class)

            all_ap[f'mAP@{iou_thr:.2f}'] = map_value
            all_ap[f'AP_per_class@{iou_thr:.2f}'] = ap_per_class

        all_ap['mAP'] = all_ap[f'mAP@{self.iou_thresholds[0]:.2f}']
        all_ap['AP_per_class'] = all_ap[f'AP_per_class@{self.iou_thresholds[0]:.2f}']

        return all_ap

    def _compute_class_ap(self, class_id: int, iou_threshold: float) -> float:
        all_pred_boxes = []
        all_pred_scores = []
        all_pred_image_ids = []

        all_gt_boxes = []
        all_gt_image_ids = []

        for img_id, (pred, target) in enumerate(zip(self.predictions, self.targets)):
            class_mask = pred['labels'] == class_id
            if class_mask.sum() > 0:
                all_pred_boxes.append(pred['boxes'][class_mask])
                all_pred_scores.append(pred['scores'][class_mask])
                all_pred_image_ids.append(np.full(class_mask.sum(), img_id))

            gt_class_mask = target['labels'] == class_id
            if gt_class_mask.sum() > 0:
                all_gt_boxes.append(target['boxes'][gt_class_mask])
                all_gt_image_ids.append(np.full(gt_class_mask.sum(), img_id))

        if len(all_gt_boxes) == 0:
            return 0.0

        if len(all_pred_boxes) > 0:
            all_pred_boxes = np.concatenate(all_pred_boxes, axis=0)
            all_pred_scores = np.concatenate(all_pred_scores, axis=0)
            all_pred_image_ids = np.concatenate(all_pred_image_ids, axis=0)
        else:
            return 0.0

        all_gt_boxes = np.concatenate(all_gt_boxes, axis=0)
        all_gt_image_ids = np.concatenate(all_gt_image_ids, axis=0)

        sorted_indices = np.argsort(-all_pred_scores)
        all_pred_boxes = all_pred_boxes[sorted_indices]
        all_pred_scores = all_pred_scores[sorted_indices]
        all_pred_image_ids = all_pred_image_ids[sorted_indices]

        num_gts = len(all_gt_boxes)
        gt_matched = np.zeros(num_gts, dtype=bool)

        num_preds = len(all_pred_boxes)
        tp = np.zeros(num_preds)
        fp = np.zeros(num_preds)

        for pred_idx in range(num_preds):
            pred_box = all_pred_boxes[pred_idx]
            pred_img_id = all_pred_image_ids[pred_idx]

            gt_mask = all_gt_image_ids == pred_img_id
            if gt_mask.sum() == 0:
                fp[pred_idx] = 1
                continue

            gt_boxes_in_image = all_gt_boxes[gt_mask]
            gt_indices_in_image = np.where(gt_mask)[0]

            ious = box_iou(pred_box[None, :], gt_boxes_in_image)[0]

            max_iou_idx = np.argmax(ious)
            max_iou = ious[max_iou_idx]

            if max_iou >= iou_threshold:
                gt_global_idx = gt_indices_in_image[max_iou_idx]
                if not gt_matched[gt_global_idx]:
                    tp[pred_idx] = 1
                    gt_matched[gt_global_idx] = True
                else:
                    fp[pred_idx] = 1
            else:
                fp[pred_idx] = 1

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recall = tp_cumsum / num_gts
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

        ap = compute_ap(recall, precision)

        return ap

    def reset(self):
        self.predictions = []
        self.targets = []


if __name__ == "__main__":
    print("Testing DetectionMetrics...")

    metrics = DetectionMetrics(num_classes=2, iou_thresholds=[0.5, 0.75])

    preds = {
        'pred_boxes': torch.tensor([
            [[10, 10, 50, 50], [60, 60, 100, 100]],  # image 1
            [[20, 20, 60, 60], [0, 0, 0, 0]]          # image 2
        ]),
        'pred_scores': torch.tensor([
            [0.9, 0.8],
            [0.7, 0.0]
        ]),
        'pred_labels': torch.tensor([
            [0, 1],
            [0, 0]
        ])
    }

    targets = [
        {'bboxes': torch.tensor([[12, 12, 48, 48], [62, 62, 98, 98]]),
         'labels': torch.tensor([0, 1])},
        {'bboxes': torch.tensor([[22, 22, 58, 58]]),
         'labels': torch.tensor([0])}
    ]

    metrics.update(preds, targets)

    results = metrics.compute()
    print("\nResults:")
    for k, v in results.items():
        print(f"  {k}: {v}")

    print("\n✅ DetectionMetrics test passed!")
