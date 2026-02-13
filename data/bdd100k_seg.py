
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset

from configs.config import cfg


class BDD100KSegmentation(Dataset):

    CLASS_NAMES = ['road', 'lane']

    def __init__(
        self,
        split: str = 'train',
        transform=None,
        image_size: int = None,
        ignore_index: int = 255
    ):
        super().__init__()
        self.split = split
        self.transform = transform
        self.image_size = image_size or cfg.IMAGE_SIZE
        self.ignore_index = ignore_index

        if split == 'train':
            self.img_dir = cfg.BDD100K_IMAGES_TRAIN
            self.anno_dir = cfg.BDD100K_LABELS_SEG_TRAIN
        else:
            self.img_dir = cfg.BDD100K_IMAGES_TRAIN  # 注意：仍使用 train 图像
            self.anno_dir = cfg.BDD100K_LABELS_SEG_VAL

        if not self.img_dir.exists():
            raise FileNotFoundError(
                f"BDD100K images not found at {self.img_dir}\n"
                f"Please check your dataset structure."
            )

        if not self.anno_dir.exists():
            raise FileNotFoundError(
                f"BDD100K segmentation labels not found at {self.anno_dir}\n"
                f"Please run the conversion script first."
            )

        self.samples = self._collect_samples()

        print(f"BDD100K Segmentation [{split}]: {len(self.samples)} images loaded")

    def _collect_samples(self) -> List[Dict]:
        samples = []

        # 这样可以避免大量"图像不存在"的警告
        img_files = sorted(self.img_dir.glob('*.jpg'))

        for img_path in img_files:
            if img_path.is_symlink():
                img_path = img_path.resolve()

            if not img_path.exists():
                continue

            img_stem = img_path.stem  # 例如: a9acd883-5421d727


            anno_path = None

            possible_suffixes = ['_train_id', '_val_id', '']

            for suffix in possible_suffixes:
                if suffix:
                    anno_candidate = self.anno_dir / f"{img_stem}{suffix}.txt"
                else:
                    anno_candidate = self.anno_dir / f"{img_stem}.txt"

                if anno_candidate.exists():
                    anno_path = anno_candidate
                    break

            if anno_path is not None:
                samples.append({
                    'image_path': img_path,
                    'anno_path': anno_path,
                    'image_name': img_path.name
                })

        return samples

    def _load_yolo_polygon_annotation(self, anno_path: Path, img_width: int, img_height: int) -> np.ndarray:
        mask = np.zeros((img_height, img_width), dtype=np.uint8)  # 背景=0

        with open(anno_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 7:  # 至少需要 class_id + 3个点（6个坐标）
                continue

            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]

            if len(coords) % 2 != 0:
                continue

            points = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * img_width)
                y = int(coords[i+1] * img_height)
                points.append([x, y])

            points = np.array(points, dtype=np.int32)

            new_class_id = class_id + 1

            cv2.fillPoly(mask, [points], color=new_class_id)

        return mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = Image.open(sample['image_path']).convert('RGB')
        img_width, img_height = image.size
        orig_h, orig_w = img_height, img_width

        mask = self._load_yolo_polygon_annotation(sample['anno_path'], img_width, img_height)

        if self.transform is not None:
            transformed = self.transform(
                image=np.array(image),
                mask=mask
            )
            image = transformed['image']
            mask = transformed['mask']
        else:
            image = np.array(image)

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()

        return {
            'task': 'segmentation',
            'image': image,
            'seg_mask': mask,  # 注意：使用 seg_mask
            'orig_size': (orig_h, orig_w),
            'image_id': sample['image_name']
        }


def collate_fn_seg(batch):
    images = []
    targets = []
    orig_sizes = []
    image_ids = []

    for item in batch:
        images.append(item['image'])
        targets.append({
            'seg_mask': item['seg_mask']
        })
        orig_sizes.append(item['orig_size'])
        image_ids.append(item['image_id'])

    images = torch.stack(images, dim=0)

    return {
        'task': 'segmentation',
        'images': images,
        'targets': targets,
        'orig_sizes': orig_sizes,
        'image_ids': image_ids
    }
