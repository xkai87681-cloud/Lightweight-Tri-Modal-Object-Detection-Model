import numpy as np
import cv2
import torch


def visualize_detection(image, boxes, labels, class_names=None):
    img = image.copy()
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        if class_names:
            cv2.putText(img, class_names[label], (int(x1), int(y1)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img


def visualize_segmentation(image, mask, num_classes=2):
    colors = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]])
    colored_mask = colors[mask % len(colors)]
    vis = cv2.addWeighted(image, 0.6, colored_mask.astype(np.uint8), 0.4, 0)
    return vis
