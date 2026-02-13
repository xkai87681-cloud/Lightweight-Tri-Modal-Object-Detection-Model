import torch
import torchvision


def nms(boxes, scores, iou_threshold=0.5):
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long)

    keep = torchvision.ops.nms(boxes, scores, iou_threshold)
    return keep


def decode_detections(det_outputs, img_size=640, conf_threshold=0.25, nms_threshold=0.5):

    if not isinstance(det_outputs, list) or len(det_outputs) == 0:
        return []

    batch_size = det_outputs[0][0].size(0)

    all_detections = []

    for batch_idx in range(batch_size):
        all_boxes = []
        all_scores = []
        all_labels = []

        device = det_outputs[0][0].device

        for boxes, scores, labels in det_outputs:

            batch_boxes = boxes[batch_idx]  # (N, 4)
            batch_scores = scores[batch_idx]  # (N,)
            batch_labels = labels[batch_idx]  # (N,)

            conf_mask = batch_scores > conf_threshold
            batch_boxes = batch_boxes[conf_mask]
            batch_scores = batch_scores[conf_mask]
            batch_labels = batch_labels[conf_mask]

            if len(batch_boxes) > 0:
                all_boxes.append(batch_boxes)
                all_scores.append(batch_scores)
                all_labels.append(batch_labels)

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
        all_boxes = all_boxes.clamp(0, 1)  # 确保在[0, 1]范围内

        keep_indices = []
        num_classes = int(all_labels.max().item()) + 1 if len(all_labels) > 0 else 0

        for c in range(num_classes):
            class_mask = all_labels == c
            if class_mask.sum() == 0:
                continue

            class_boxes = all_boxes[class_mask]
            class_scores = all_scores[class_mask]

            keep = nms(class_boxes * img_size, class_scores, nms_threshold)

            class_indices = torch.where(class_mask)[0]
            keep_indices.append(class_indices[keep])

        if len(keep_indices) > 0:
            keep_indices = torch.cat(keep_indices)
            final_boxes = all_boxes[keep_indices]
            final_scores = all_scores[keep_indices]
            final_labels = all_labels[keep_indices]
        else:
            device = all_boxes.device
            final_boxes = torch.empty(0, 4, device=device)
            final_scores = torch.empty(0, device=device)
            final_labels = torch.empty(0, dtype=torch.long, device=device)

        all_detections.append({
            'boxes': final_boxes,  # 归一化坐标 [0, 1]
            'scores': final_scores,
            'labels': final_labels
        })

    return all_detections


if __name__ == "__main__":
    print("Testing detection post-processing...")

    batch_size = 2
    det_outputs = [
        (
            torch.rand(batch_size, 100, 4) * 640,  # boxes
            torch.rand(batch_size, 100),            # scores
            torch.randint(0, 2, (batch_size, 100))  # labels
        ),
        (
            torch.rand(batch_size, 50, 4) * 640,
            torch.rand(batch_size, 50),
            torch.randint(0, 2, (batch_size, 50))
        )
    ]

    detections = decode_detections(det_outputs, conf_threshold=0.5)

    print(f"Batch size: {len(detections)}")
    for i, det in enumerate(detections):
        print(f"Image {i}: {len(det['boxes'])} detections")
        print(f"  Boxes shape: {det['boxes'].shape}")
        print(f"  Scores shape: {det['scores'].shape}")
        print(f"  Labels shape: {det['labels'].shape}")
