
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from configs.config import cfg
from utils.dataset_finder import find_cityscapes_structure


class CityscapesSegmentation(Dataset):

    CITYSCAPES_TO_OURS = {
        0: 0,  # road -> 0
        -1: 255,  # unlabeled -> 255
        255: 255,  # ignore -> 255
    }

    CLASS_NAMES = ['road', 'background']  # background 作为非 road 的区域

    def __init__(
        self,
        split: str = 'train',
        transform=None,
        image_size: int = None,
        ignore_index: int = 255,
        two_class_mode: bool = True
    ):
        super().__init__()
        self.split = split
        self.transform = transform
        self.image_size = image_size or cfg.IMAGE_SIZE
        self.ignore_index = ignore_index
        self.two_class_mode = two_class_mode

        cityscapes_root = cfg.CITYSCAPES_ROOT
        structure = find_cityscapes_structure(cityscapes_root)

        if split == 'train':
            self.img_dir = structure['images_train']
            self.label_dir = structure['labels_train']
        else:
            self.img_dir = structure['images_val']
            self.label_dir = structure['labels_val']

        if self.img_dir is None or not self.img_dir.exists():
            raise FileNotFoundError(
                f"Cityscapes images not found. Searched in {cityscapes_root}\n"
                f"Please check your dataset structure."
            )

        self.samples = self._collect_samples()

        print(f"Cityscapes Segmentation [{split}]: {len(self.samples)} images loaded")

    def _collect_samples(self) -> List[Dict]:
        samples = []

        for city_dir in sorted(self.img_dir.iterdir()):
            if not city_dir.is_dir():
                continue

            for img_path in sorted(city_dir.glob('*.png')):
                img_name = img_path.stem.replace('_leftImg8bit', '')
                label_path = self.label_dir / city_dir.name / f"{img_name}_gtFine_labelTrainIds.png"

                if not label_path.exists():
                    label_path = self.label_dir / city_dir.name / f"{img_name}_gtFine_labelIds.png"
                    if not label_path.exists():
                        print(f"Warning: label not found for {img_path.name}")
                        continue

                samples.append({
                    'image_path': img_path,
                    'label_path': label_path,
                    'city': city_dir.name,
                    'image_name': img_path.name
                })

        return samples

    def _remap_labels(self, label_map: np.ndarray) -> np.ndarray:
        if self.two_class_mode:
            remapped = np.full_like(label_map, 1, dtype=np.uint8)  # 默认 background

            remapped[label_map == 0] = 0

            remapped[label_map == 255] = 255

        else:
            remapped = label_map.copy()

        return remapped

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]

        image = Image.open(sample['image_path']).convert('RGB')
        image = np.array(image)
        orig_h, orig_w = image.shape[:2]

        label_map = Image.open(sample['label_path'])
        label_map = np.array(label_map, dtype=np.uint8)

        label_map = self._remap_labels(label_map)

        if self.transform is not None:
            transformed = self.transform(
                image=image,
                mask=label_map
            )
            image = transformed['image']
            label_map = transformed.get('mask', label_map)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            label_map = torch.from_numpy(label_map).long()

        if isinstance(label_map, np.ndarray):
            label_map = torch.from_numpy(label_map).long()

        return {
            'task': 'segmentation',
            'image': image,
            'seg_mask': label_map,
            'orig_size': (orig_h, orig_w),
            'image_id': sample['image_name']
        }


def collate_fn_seg(batch: List[Dict]) -> Dict:
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


if __name__ == '__main__':
    print("Testing Cityscapes Segmentation Dataset...")

    from data.transforms import MTLTransform

    transform = MTLTransform(image_size=640, is_train=True)
    dataset = CityscapesSegmentation(split='train', transform=transform)

    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Seg mask shape: {sample['seg_mask'].shape}")
    print(f"  Seg mask unique values: {torch.unique(sample['seg_mask'])}")
    print(f"  Orig size: {sample['orig_size']}")
    print(f"  Image ID: {sample['image_id']}")

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn_seg)
    batch = next(iter(loader))

    print(f"\nBatch:")
    print(f"  Images shape: {batch['images'].shape}")
    print(f"  Num targets: {len(batch['targets'])}")
    print(f"  Target 0 seg_mask: {batch['targets'][0]['seg_mask'].shape}")

    print("\n✅ Cityscapes Segmentation test passed!")
