import torch
import torch.nn.functional as F
import torchvision
from typing import List, Dict


def dist2bbox(distance, anchor_points):
    lt = anchor_points - distance[:, :2]  # left-top
    rb = anchor_points + distance[:, 2:]  # right-bottom
    return torch.cat([lt, rb], dim=-1)


def make_anchors(feats, strides, grid_cell_offset=0.5):
    anchor_points, stride_tensor = [], []

    for i, (feat, stride) in enumerate(zip(feats, strides)):
        _, _, h, w = feat.shape

        sy = torch.arange(h, dtype=torch.float32, device=feat.device)
        sx = torch.arange(w, dtype=torch.float32, device=feat.device)
        grid_y, grid_x = torch.meshgrid(sy, sx, indexing='ij')

        anchor_point = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)
        anchor_point = (anchor_point + grid_cell_offset) * stride

        anchor_points.append(anchor_point)
        stride_tensor.append(torch.full((h * w,), stride, dtype=torch.float32, device=feat.device))

    anchor_points = torch.cat(anchor_points, dim=0)
    stride_tensor = torch.cat(stride_tensor, dim=0)

    return anchor_points, stride_tensor


def postprocess_detections(
    bbox_preds: List[torch.Tensor],
    cls_preds: List[torch.Tensor],
    img_size: int = 640,
    conf_threshold: float = 0.25,
    nms_threshold: float = 0.45,
    max_det: int = 300
) -> List[Dict]:
    batch_size = bbox_preds[0].shape[0]
    device = bbox_preds[0].device

    strides = [8, 16, 32]

    anchor_points, stride_tensor = make_anchors(bbox_preds, strides)

    all_detections = []

    for batch_idx in range(batch_size):
        all_boxes = []
        all_scores = []
        all_labels = []

        for level_idx, (bbox_pred, cls_pred) in enumerate(zip(bbox_preds, cls_preds)):

            b, c, h, w = cls_pred.shape

            bbox = bbox_pred[batch_idx]  # [4, H, W]
            cls = cls_pred[batch_idx]    # [num_classes, H, W]

            bbox = bbox.permute(1, 2, 0).reshape(-1, 4)
            cls = cls.permute(1, 2, 0).reshape(-1, c)

            scores, labels = torch.max(torch.sigmoid(cls), dim=1)

            conf_mask = scores > conf_threshold
            if conf_mask.sum() == 0:
                continue

            bbox = bbox[conf_mask]
            scores = scores[conf_mask]
            labels = labels[conf_mask]

            start_idx = sum([bbox_preds[i].shape[2] * bbox_preds[i].shape[3] for i in range(level_idx)])
            end_idx = start_idx + h * w
            level_anchors = anchor_points[start_idx:end_idx][conf_mask]
            level_strides = stride_tensor[start_idx:end_idx][conf_mask]

            bbox = bbox * level_strides.view(-1, 1)  # 乘以stride
            boxes = dist2bbox(bbox, level_anchors)

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        if len(all_boxes) == 0:
            all_detections.append({
                'boxes': torch.empty(0, 4, device=device),
                'scores': torch.empty(0, device=device),
                'labels': torch.empty(0, dtype=torch.long, device=device)
            })
            continue

        all_boxes = torch.cat(all_boxes, dim=0)
        all_scores = torch.cat(all_scores, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        all_boxes = all_boxes / img_size
        all_boxes = all_boxes.clamp(0, 1)

        keep_indices = []
        num_classes = int(all_labels.max().item()) + 1 if len(all_labels) > 0 else 0

        for c in range(num_classes):
            class_mask = all_labels == c
            if class_mask.sum() == 0:
                continue

            class_boxes = all_boxes[class_mask]
            class_scores = all_scores[class_mask]

            keep = torchvision.ops.nms(class_boxes * img_size, class_scores, nms_threshold)

            class_indices = torch.where(class_mask)[0]
            keep_indices.append(class_indices[keep])

        if len(keep_indices) > 0:
            keep_indices = torch.cat(keep_indices)

            if len(keep_indices) > max_det:
                top_scores, top_indices = torch.topk(all_scores[keep_indices], max_det)
                keep_indices = keep_indices[top_indices]

            final_boxes = all_boxes[keep_indices]
            final_scores = all_scores[keep_indices]
            final_labels = all_labels[keep_indices]
        else:
            final_boxes = torch.empty(0, 4, device=device)
            final_scores = torch.empty(0, device=device)
            final_labels = torch.empty(0, dtype=torch.long, device=device)

        all_detections.append({
            'boxes': final_boxes,
            'scores': final_scores,
            'labels': final_labels
        })

    return all_detections


if __name__ == "__main__":
    print("Testing detection post-processing...")

    batch_size = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    bbox_preds = [
        torch.randn(batch_size, 4, 80, 80).to(device),   # P3: stride=8
        torch.randn(batch_size, 4, 40, 40).to(device),   # P4: stride=16
        torch.randn(batch_size, 4, 20, 20).to(device),   # P5: stride=32
    ]

    cls_preds = [
        torch.randn(batch_size, 2, 80, 80).to(device),
        torch.randn(batch_size, 2, 40, 40).to(device),
        torch.randn(batch_size, 2, 20, 20).to(device),
    ]

    detections = postprocess_detections(bbox_preds, cls_preds, conf_threshold=0.5)

    print(f"\nBatch size: {len(detections)}")
    for i, det in enumerate(detections):
        print(f"Image {i}: {len(det['boxes'])} detections")
        print(f"  Boxes shape: {det['boxes'].shape}")
        print(f"  Scores shape: {det['scores'].shape}")
        print(f"  Labels shape: {det['labels'].shape}")
        if len(det['boxes']) > 0:
            print(f"  Box range: [{det['boxes'].min():.3f}, {det['boxes'].max():.3f}]")

    print("\n✅ Detection post-processing test passed!")
