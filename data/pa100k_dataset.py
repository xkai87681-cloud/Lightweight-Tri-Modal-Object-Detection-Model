
import os
import pandas as pd
import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from PIL import Image
import torch
from torch.utils.data import Dataset

from configs.config import cfg
from utils.dataset_finder import find_pa100k_structure


class PA100KDataset(Dataset):

    def __init__(
        self,
        split: str = 'train',
        transform=None,
        image_size: int = None,
        num_attrs: int = 26
    ):
        super().__init__()
        self.split = split
        self.transform = transform
        self.image_size = image_size or cfg.IMAGE_SIZE
        self.num_attrs = num_attrs

        pa100k_root = cfg.PA100K_ROOT
        structure = find_pa100k_structure(pa100k_root)

        self.img_dir = structure['images_dir']
        self.use_csv = structure['use_csv']

        if self.use_csv:
            if split == 'train':
                self.anno_path = structure['train_csv']
            elif split == 'val':
                self.anno_path = structure['val_csv']
            else:
                csv_path = pa100k_root / 'annotations' / f'{split}.csv'
                if not csv_path.exists():
                    csv_path = pa100k_root / f'{split}.csv'
                self.anno_path = csv_path
        else:
            self.anno_path = structure['annotation_mat']

        if self.img_dir is None or not self.img_dir.exists():
            print(f"\n{'='*80}")
            print(f"❌ PA100K Dataset Not Found")
            print(f"{'='*80}")
            print(f"Looking for: {pa100k_root}")
            print(f"\nExpected structure (any of these):")
            print(f"  Option 1 (MAT format):")
            print(f"    PA100K/annotation.mat")
            print(f"    PA100K/release_data/release_data/train/")
            print(f"  Option 2 (CSV format):")
            print(f"    PA100K/annotations/train.csv")
            print(f"    PA100K/images/train/  or  PA100K/data/train/")
            print(f"\nPlease organize your dataset in one of these formats.")
            print(f"{'='*80}\n")
            raise FileNotFoundError(
                f"PA100K images not found. Searched in {pa100k_root}\n"
                f"Please check your dataset structure."
            )

        if self.anno_path is None or not self.anno_path.exists():
            print(f"\n{'='*80}")
            print(f"❌ PA100K Annotation Not Found")
            print(f"{'='*80}")
            print(f"Looking for: {pa100k_root}")
            print(f"Format detected: {'CSV' if self.use_csv else 'MAT'}")
            if self.use_csv:
                print(f"Expected: {pa100k_root}/annotations/train.csv")
            else:
                print(f"Expected: {pa100k_root}/annotation.mat")
            print(f"{'='*80}\n")
            raise FileNotFoundError(
                f"PA100K annotation not found at {self.anno_path}. "
                f"Please check your dataset structure."
            )

        self._load_annotations()

        print(f"PA100K Attribute [{split}]: {len(self.image_names)} images loaded")

    def _load_annotations(self):
        if self.use_csv:
            df = pd.read_csv(str(self.anno_path))

            self.image_names = df['image_path'].tolist()
            attr_columns = [col for col in df.columns if col.startswith('attr_')]
            self.attributes = df[attr_columns].values.astype(np.float32)
            self.attrs = self.attributes  # 兼容attr使用

        else:
            mat_data = sio.loadmat(str(self.anno_path))

            attr_key_suffix = '_attr' if f'{self.split}_attr' in mat_data else '_label'

            images_name_key = f'{self.split}_images_name'
            attr_key = f'{self.split}{attr_key_suffix}'

            names_array = mat_data[images_name_key]
            self.image_names = []

            if names_array.shape[0] == 1:
                for item in names_array[0]:
                    self.image_names.append(self._extract_string(item))
            else:
                for item in names_array:
                    self.image_names.append(self._extract_string(item))

            self.attrs = mat_data[attr_key].astype(np.float32)  # [N, 26]

        if len(self.image_names) > 0:
            sample_name = self.image_names[0]
            sample_path = self.img_dir / sample_name

            if not sample_path.exists():
                if sample_name.endswith('.png'):
                    jpg_name = sample_name.replace('.png', '.jpg')
                    jpg_path = self.img_dir / jpg_name
                    if jpg_path.exists():
                        self.image_names = [name.replace('.png', '.jpg') for name in self.image_names]
                        print(f"  ℹ️ 图像格式: JPG (已自动转换文件名)")
                elif sample_name.endswith('.jpg'):
                    png_name = sample_name.replace('.jpg', '.png')
                    png_path = self.img_dir / png_name
                    if png_path.exists():
                        self.image_names = [name.replace('.jpg', '.png') for name in self.image_names]
                        print(f"  ℹ️ 图像格式: PNG (已自动转换文件名)")

        if self.attrs.shape[1] != self.num_attrs:
            if self.attrs.shape[1] > self.num_attrs:
                self.attrs = self.attrs[:, :self.num_attrs]
            else:
                pad = np.zeros((self.attrs.shape[0], self.num_attrs - self.attrs.shape[1]), dtype=np.float32)
                self.attrs = np.concatenate([self.attrs, pad], axis=1)

    def _extract_string(self, item) -> str:
        if isinstance(item, np.ndarray):
            if item.size == 1:
                item = item.item()  # 转换为Python标量
            elif item.size == 0:
                return ""
            else:
                item = item.flat[0]

        if isinstance(item, (np.str_, np.unicode_)):
            return str(item)

        if hasattr(item, 'item'):
            item = item.item()

        if isinstance(item, str):
            return item

        if isinstance(item, bytes):
            return item.decode('utf-8', errors='ignore')

        if hasattr(item, '__iter__') and not isinstance(item, (str, bytes)):
            try:
                if hasattr(item, '__len__') and len(item) > 0:
                    result = self._extract_string(item[0])
                    return str(result) if result is not None else ""
                else:
                    return ""
            except Exception:
                pass

        result = str(item)
        if isinstance(result, (np.str_, np.unicode_)):
            return str(result)
        return result

    def compute_pos_weight(self) -> torch.Tensor:
        num_positive = self.attrs.sum(axis=0)  # [26]
        num_negative = len(self.attrs) - num_positive

        num_positive = np.maximum(num_positive, 1.0)

        pos_weight = num_negative / num_positive
        return torch.from_numpy(pos_weight).float()

    def get_attr_statistics(self) -> Dict:
        num_samples = len(self.attrs)
        num_positive = self.attrs.sum(axis=0)
        num_negative = num_samples - num_positive
        pos_ratio = num_positive / num_samples

        stats = {
            'num_positive': num_positive,
            'num_negative': num_negative,
            'pos_ratio': pos_ratio,
            'pos_weight': (num_negative / np.maximum(num_positive, 1.0))
        }

        return stats

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> Dict:
        img_name = self.image_names[index]
        img_path = self.img_dir / img_name

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))

        image = np.array(image)
        orig_h, orig_w = image.shape[:2]

        person_bbox = np.array([[0, 0, orig_w, orig_h]], dtype=np.float32)

        attributes = self.attrs[index]

        if self.transform is not None:
            transformed = self.transform(
                image=image
            )
            image = transformed['image']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        if isinstance(person_bbox, np.ndarray):
            person_bbox = torch.from_numpy(person_bbox).float()
        if isinstance(attributes, np.ndarray):
            attributes = torch.from_numpy(attributes).float()

        return {
            'task': 'attribute',
            'image': image,
            'person_bbox': person_bbox,
            'attributes': attributes,
            'orig_size': (orig_h, orig_w),
            'image_id': img_name
        }


def collate_fn_attr(batch: List[Dict]) -> Dict:
    images = []
    targets = []
    orig_sizes = []
    image_ids = []

    for item in batch:
        images.append(item['image'])
        targets.append({
            'person_bbox': item['person_bbox'],
            'attributes': item['attributes']
        })
        orig_sizes.append(item['orig_size'])
        image_ids.append(item['image_id'])

    images = torch.stack(images, dim=0)

    return {
        'task': 'attribute',
        'images': images,
        'targets': targets,
        'orig_sizes': orig_sizes,
        'image_ids': image_ids
    }


if __name__ == '__main__':
    print("Testing PA100K Attribute Dataset...")

    from data.transforms import MTLTransform

    transform = MTLTransform(image_size=640, is_train=True)
    dataset = PA100KDataset(split='train', transform=transform)

    print(f"Dataset size: {len(dataset)}")

    pos_weight = dataset.compute_pos_weight()
    print(f"\nPos weight: {pos_weight}")

    stats = dataset.get_attr_statistics()
    print(f"\nAttribute statistics:")
    print(f"  Positive ratio (first 5): {stats['pos_ratio'][:5]}")
    print(f"  Pos weight (first 5): {stats['pos_weight'][:5]}")

    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Person bbox shape: {sample['person_bbox'].shape}")
    print(f"  Attributes shape: {sample['attributes'].shape}")
    print(f"  Attributes sum: {sample['attributes'].sum()}")
    print(f"  Orig size: {sample['orig_size']}")
    print(f"  Image ID: {sample['image_id']}")

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn_attr)
    batch = next(iter(loader))

    print(f"\nBatch:")
    print(f"  Images shape: {batch['images'].shape}")
    print(f"  Num targets: {len(batch['targets'])}")
    print(f"  Target 0 person_bbox: {batch['targets'][0]['person_bbox'].shape}")
    print(f"  Target 0 attributes: {batch['targets'][0]['attributes'].shape}")

    print("\n✅ PA100K Attribute test passed!")
