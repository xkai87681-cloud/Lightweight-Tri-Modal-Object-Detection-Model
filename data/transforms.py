
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Tuple, Optional


class MTLTransform:

    def __init__(
            self,
            image_size: int = 640,
            is_train: bool = True,
            use_mosaic: bool = False,
            mosaic_prob: float = 0.0
    ):
        self.image_size = image_size
        self.is_train = is_train
        self.use_mosaic = use_mosaic
        self.mosaic_prob = mosaic_prob

    def __call__(
            self,
            image: np.ndarray,
            bboxes: Optional[np.ndarray] = None,
            labels: Optional[np.ndarray] = None,
            mask: Optional[np.ndarray] = None,
            **kwargs
    ) -> Dict:
        if self.is_train:
            transform = get_train_transform(self.image_size)
        else:
            transform = get_val_transform(self.image_size)

        transform_input = {'image': image}

        if bboxes is not None and len(bboxes) > 0:
            transform_input['bboxes'] = bboxes.tolist()
            transform_input['labels'] = labels.tolist()

        if mask is not None:
            transform_input['mask'] = mask

        transformed = transform(**transform_input)

        output = {'image': transformed['image']}

        if 'bboxes' in transformed and len(transformed['bboxes']) > 0:
            output['bboxes'] = np.array(transformed['bboxes'], dtype=np.float32)
            output['labels'] = np.array(transformed['labels'], dtype=np.int64)
        else:
            if bboxes is not None:
                output['bboxes'] = np.zeros((0, 4), dtype=np.float32)
                output['labels'] = np.zeros((0,), dtype=np.int64)

        if 'mask' in transformed:
            output['mask'] = transformed['mask']

        return output


def get_train_transform(image_size: int = 640, task: str = None) -> A.Compose:
    from configs.config import cfg

    transform_list = [
        A.LongestMaxSize(max_size=image_size, interpolation=cv2.INTER_LINEAR),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=114  # YOLO 风格填充值
        ),

        A.ShiftScaleRotate(
            shift_limit=0.1,     # 平移10%
            scale_limit=0.3,     # 缩放±30%
            rotate_limit=15,     # 旋转±15度
            border_mode=cv2.BORDER_CONSTANT,
            value=114,
            p=0.7
        ),
        A.HorizontalFlip(p=0.5),
        A.RandomResizedCrop(
            height=image_size,
            width=image_size,
            scale=(0.8, 1.0),    # 裁剪80-100%（提高下限，保留小目标）
            ratio=(0.9, 1.1),    # 宽高比
            p=0.5  # 提高概率（0.3→0.5）增强小目标
        ),

        A.ColorJitter(
            brightness=cfg.AUG_BRIGHTNESS_LIMIT,
            contrast=cfg.AUG_CONTRAST_LIMIT,
            saturation=cfg.AUG_SAT_SHIFT_LIMIT / 255.0,
            hue=cfg.AUG_HUE_SHIFT_LIMIT / 360.0,
            p=0.6
        ),

        A.OneOf([
            A.MotionBlur(blur_limit=(3, cfg.AUG_BLUR_LIMIT), p=1.0),
            A.MedianBlur(blur_limit=cfg.AUG_BLUR_LIMIT, p=1.0),
            A.GaussianBlur(blur_limit=(3, cfg.AUG_BLUR_LIMIT), p=1.0),
        ], p=0.3),

        A.GaussNoise(var_limit=cfg.AUG_NOISE_VAR_LIMIT, p=0.2),

        A.OneOf([
            A.RandomRain(p=1.0),
            A.RandomFog(p=1.0),
            A.RandomShadow(p=1.0),
        ], p=0.2),

        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]

    if task == 'detection':
        bbox_params = A.BboxParams(
            format='pascal_voc',  # [x_min, y_min, x_max, y_max]
            label_fields=['labels'],
            min_area=0,
            min_visibility=0.1  # 至少保留 10% 可见的 bbox
        )
        return A.Compose(transform_list, bbox_params=bbox_params)
    else:
        return A.Compose(transform_list)


def get_val_transform(image_size: int = 640, task: str = None) -> A.Compose:
    transform_list = [
        A.LongestMaxSize(max_size=image_size, interpolation=cv2.INTER_LINEAR),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=114
        ),

        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]

    if task == 'detection':
        bbox_params = A.BboxParams(
            format='pascal_voc',
            label_fields=['labels'],
            min_area=0,
            min_visibility=0.0
        )
        return A.Compose(transform_list, bbox_params=bbox_params)
    else:
        return A.Compose(transform_list)


def denormalize(tensor: np.ndarray, mean: List[float] = None, std: List[float] = None) -> np.ndarray:
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    mean = np.array(mean).reshape(-1, 1, 1)
    std = np.array(std).reshape(-1, 1, 1)

    if tensor.shape[0] == 3:
        tensor = tensor.transpose(1, 2, 0)
        mean = mean.reshape(1, 1, -1)
        std = std.reshape(1, 1, -1)

    img = tensor * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img



def mosaic_augmentation(
        images: List[np.ndarray],
        bboxes_list: List[np.ndarray],
        labels_list: List[np.ndarray],
        image_size: int = 640
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert len(images) == 4, "Mosaic requires exactly 4 images"

    cx = np.random.randint(image_size // 2, image_size // 2 + image_size // 4)
    cy = np.random.randint(image_size // 2, image_size // 2 + image_size // 4)

    mosaic_image = np.full((image_size, image_size, 3), 114, dtype=np.uint8)
    mosaic_bboxes = []
    mosaic_labels = []

    positions = [
        (0, 0, cx, cy),  # 左上
        (cx, 0, image_size, cy),  # 右上
        (0, cy, cx, image_size),  # 左下
        (cx, cy, image_size, image_size)  # 右下
    ]

    for i, (img, bboxes, labels) in enumerate(zip(images, bboxes_list, labels_list)):
        h, w = img.shape[:2]
        x1, y1, x2, y2 = positions[i]
        pw, ph = x2 - x1, y2 - y1

        scale = min(pw / w, ph / h)
        nw, nh = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

        mosaic_image[y1:y1 + nh, x1:x1 + nw] = img_resized

        if len(bboxes) > 0:
            bboxes_scaled = bboxes.copy()
            bboxes_scaled[:, [0, 2]] = bboxes[:, [0, 2]] * scale + x1
            bboxes_scaled[:, [1, 3]] = bboxes[:, [1, 3]] * scale + y1

            bboxes_scaled[:, [0, 2]] = np.clip(bboxes_scaled[:, [0, 2]], 0, image_size)
            bboxes_scaled[:, [1, 3]] = np.clip(bboxes_scaled[:, [1, 3]], 0, image_size)

            valid = (bboxes_scaled[:, 2] > bboxes_scaled[:, 0]) & (bboxes_scaled[:, 3] > bboxes_scaled[:, 1])
            bboxes_scaled = bboxes_scaled[valid]
            labels_scaled = labels[valid]

            mosaic_bboxes.append(bboxes_scaled)
            mosaic_labels.append(labels_scaled)

    if len(mosaic_bboxes) > 0:
        mosaic_bboxes = np.concatenate(mosaic_bboxes, axis=0)
        mosaic_labels = np.concatenate(mosaic_labels, axis=0)
    else:
        mosaic_bboxes = np.zeros((0, 4), dtype=np.float32)
        mosaic_labels = np.zeros((0,), dtype=np.int64)

    return mosaic_image, mosaic_bboxes, mosaic_labels

