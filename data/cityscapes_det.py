
import json
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from configs.config import cfg
from utils.dataset_finder import find_cityscapes_structure


class CityscapesDetection(Dataset):

    CITYSCAPES_TO_OURS = {
        24: 0,  # person -> 0
        26: 1,  # car -> 1
    }

    CLASS_NAMES = ['person', 'car']

    def __init__(
        self,
        split: str = 'train',
        transform=None,
        image_size: int = None,
        min_bbox_size: int = 10
    ):
        super().__init__()
        self.split = split
        self.transform = transform
        self.image_size = image_size or cfg.IMAGE_SIZE
        self.min_bbox_size = min_bbox_size

        cityscapes_root = cfg.CITYSCAPES_ROOT
        structure = find_cityscapes_structure(cityscapes_root)

        if split == 'train':
            self.img_dir = structure['images_train']
            self.anno_dir = structure['labels_train']
        else:
            self.img_dir = structure['images_val']
            self.anno_dir = structure['labels_val']

        if self.img_dir is None or not self.img_dir.exists():
            print(f"\n{'='*80}")
            print(f"❌ Cityscapes Dataset Not Found")
            print(f"{'='*80}")
            print(f"Looking for: {cityscapes_root}")
            print(f"\nExpected structure (any of these):")
            print(f"  Option 1 (Standard):")
            print(f"    cityscapes/leftImg8bit/train/")
            print(f"    cityscapes/gtFine/train/")
            print(f"  Option 2 (Simple):")
            print(f"    cityscapes/images/train/")
            print(f"    cityscapes/labels/train/")
            print(f"  Option 3 (Flat):")
            print(f"    cityscapes/train/  (images)")
            print(f"    cityscapes/annotations/train/  (labels)")
            print(f"\nPlease organize your dataset in one of these formats.")
            print(f"{'='*80}\n")
            raise FileNotFoundError(
                f"Cityscapes images not found. Searched in {cityscapes_root}\n"
                f"Please check your dataset structure."
            )

        self.samples = self._collect_samples()

        print(f"Cityscapes Detection [{split}]: {len(self.samples)} images loaded")

    def _collect_samples(self) -> List[Dict]:
        samples = []

        for city_dir in sorted(self.img_dir.iterdir()):
            if not city_dir.is_dir():
                continue

            for img_path in sorted(city_dir.glob('*.png')):
                img_name = img_path.stem.replace('_leftImg8bit', '')
                anno_path = self.anno_dir / city_dir.name / f"{img_name}_gtFine_polygons.json"

                if not anno_path.exists():
                    print(f"Warning: annotation not found for {img_path.name}")
                    continue

                samples.append({
                    'image_path': img_path,
                    'anno_path': anno_path,
                    'city': city_dir.name,
                    'image_name': img_path.name
                })

        return samples

    def _load_annotation(self, anno_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        with open(anno_path, 'r') as f:
            anno = json.load(f)

        img_width = anno['imgWidth']
        img_height = anno['imgHeight']

        bboxes = []
        labels = []

        for obj in anno['objects']:
            label_id = obj['label']

            if label_id not in self.CITYSCAPES_TO_OURS:
                continue

            polygon = np.array(obj['polygon'])
            x_min = polygon[:, 0].min()
            y_min = polygon[:, 1].min()
            x_max = polygon[:, 0].max()
            y_max = polygon[:, 1].max()

            w, h = x_max - x_min, y_max - y_min
            if w < self.min_bbox_size or h < self.min_bbox_size:
                continue

            bboxes.append([x_min, y_min, x_max, y_max])
            labels.append(self.CITYSCAPES_TO_OURS[label_id])

        if len(bboxes) == 0:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
        else:
            bboxes = np.array(bboxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)

        return bboxes, labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]

        image = Image.open(sample['image_path']).convert('RGB')
        image = np.array(image)
        orig_h, orig_w = image.shape[:2]

        bboxes, labels = self._load_annotation(sample['anno_path'])

        if self.transform is not None:
            transformed = self.transform(
                image=image,
                bboxes=bboxes,
                labels=labels
            )
            image = transformed['image']
            bboxes = transformed.get('bboxes', np.zeros((0, 4), dtype=np.float32))
            labels = transformed.get('labels', np.zeros((0,), dtype=np.int64))
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            bboxes = torch.from_numpy(bboxes).float()
            labels = torch.from_numpy(labels).long()

        if isinstance(bboxes, np.ndarray):
            bboxes = torch.from_numpy(bboxes).float()
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).long()

        return {
            'task': 'detection',
            'image': image,
            'bboxes': bboxes,
            'labels': labels,
            'orig_size': (orig_h, orig_w),
            'image_id': sample['image_name']
        }


def collate_fn_det(batch: List[Dict]) -> Dict:
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
