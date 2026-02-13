
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from configs.config import cfg


class BDD100KDetection(Dataset):

    CLASS_NAMES = ['person', 'vehicle']

    def __init__(
        self,
        split: str = 'train',
        transform=None,
        image_size: int = None,
        min_bbox_size: float = 0.01  # 归一化坐标下的最小框尺寸
    ):
        super().__init__()
        self.split = split
        self.transform = transform
        self.image_size = image_size or cfg.IMAGE_SIZE
        self.min_bbox_size = min_bbox_size

        if split == 'train':
            self.img_dir = cfg.BDD100K_IMAGES_TRAIN
            self.anno_dir = cfg.BDD100K_LABELS_DET_TRAIN
        else:
            self.img_dir = cfg.BDD100K_IMAGES_VAL
            self.anno_dir = cfg.BDD100K_LABELS_DET_VAL

        if not self.img_dir.exists():
            raise FileNotFoundError(
                f"BDD100K images not found at {self.img_dir}\n"
                f"Please check your dataset structure."
            )

        if not self.anno_dir.exists():
            raise FileNotFoundError(
                f"BDD100K detection labels not found at {self.anno_dir}\n"
                f"Please run the conversion script first."
            )

        self.samples = self._collect_samples()

        print(f"BDD100K Detection [{split}]: {len(self.samples)} images loaded")

    def _collect_samples(self) -> List[Dict]:
        samples = []

        anno_files = sorted(self.anno_dir.glob('*.txt'))

        for anno_path in anno_files:
            img_name = anno_path.stem + '.jpg'
            img_path = self.img_dir / img_name

            if img_path.is_symlink():
                img_path = img_path.resolve()

            if not img_path.exists():
                print(f"Warning: image not found for {anno_path.name}")
                continue

            samples.append({
                'image_path': img_path,
                'anno_path': anno_path,
                'image_name': img_name
            })

        return samples

    def _load_yolo_annotation(self, anno_path: Path, img_width: int, img_height: int) -> Tuple[np.ndarray, np.ndarray]:
        bboxes = []
        labels = []

        with open(anno_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            if width < self.min_bbox_size or height < self.min_bbox_size:
                continue

            x_center_abs = x_center * img_width
            y_center_abs = y_center * img_height
            width_abs = width * img_width
            height_abs = height * img_height

            x1 = x_center_abs - width_abs / 2
            y1 = y_center_abs - height_abs / 2
            x2 = x_center_abs + width_abs / 2
            y2 = y_center_abs + height_abs / 2

            x1 = max(0, min(img_width, x1))
            y1 = max(0, min(img_height, y1))
            x2 = max(0, min(img_width, x2))
            y2 = max(0, min(img_height, y2))

            if x2 > x1 and y2 > y1:
                bboxes.append([x1, y1, x2, y2])
                labels.append(class_id)

        if len(bboxes) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        return np.array(bboxes, dtype=np.float32), np.array(labels, dtype=np.int64)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = Image.open(sample['image_path']).convert('RGB')
        img_width, img_height = image.size
        orig_h, orig_w = img_height, img_width

        bboxes, labels = self._load_yolo_annotation(sample['anno_path'], img_width, img_height)

        if self.transform is not None:
            transformed = self.transform(
                image=np.array(image),
                bboxes=bboxes,
                labels=labels
            )
            image = transformed['image']
            bboxes = transformed.get('bboxes', np.zeros((0, 4), dtype=np.float32))
            labels = transformed.get('labels', np.zeros((0,), dtype=np.int64))
        else:
            image = np.array(image)

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        if len(bboxes) > 0:
            bboxes = np.array(bboxes, dtype=np.float32)
            h, w = image.shape[1:]
            bboxes[:, [0, 2]] /= w
            bboxes[:, [1, 3]] /= h
            bboxes = torch.from_numpy(bboxes).float()
            labels = torch.from_numpy(np.array(labels, dtype=np.int64))
        else:
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        return {
            'task': 'detection',
            'image': image,
            'bboxes': bboxes,
            'labels': labels,
            'orig_size': (orig_h, orig_w),
            'image_id': sample['image_name']
        }


def collate_fn_det(batch):
    images = []
    targets = []
    orig_sizes = []
    image_ids = []

    for item in batch:
        images.append(item['image'])
        targets.append({
            'bboxes': item['bboxes'],
            'labels': item['labels']
        })
        orig_sizes.append(item['orig_size'])
        image_ids.append(item['image_id'])

    images = torch.stack(images, dim=0)

    return {
        'task': 'detection',
        'images': images,
        'targets': targets,
        'orig_sizes': orig_sizes,
        'image_ids': image_ids
    }
