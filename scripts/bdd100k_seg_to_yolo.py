import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


BDD100K_SEG_CATEGORIES = {
    0: 'road',
    1: 'sidewalk',
    2: 'building',
    3: 'wall',
    4: 'fence',
    5: 'pole',
    6: 'traffic light',
    7: 'traffic sign',
    8: 'vegetation',
    9: 'terrain',
    10: 'sky',
    11: 'person',
    12: 'rider',
    13: 'car',
    14: 'truck',
    15: 'bus',
    16: 'train',
    17: 'motorcycle',
    18: 'bicycle',
}

SIMPLE_SEG_MAP = {
    0: 0,   # road -> road
    1: 1,   # sidewalk -> lane
}


def mask_to_polygons(mask, class_id, min_area=100):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    polygons = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) < 3:
            continue

        points = approx.reshape(-1, 2)
        h, w = mask.shape
        points_norm = points / np.array([w, h])

        points_norm = np.clip(points_norm, 0.0, 1.0)

        coords_str = ' '.join([f'{x:.6f} {y:.6f}' for x, y in points_norm])
        polygon_str = f"{class_id} {coords_str}"

        polygons.append(polygon_str)

    return polygons


def convert_bdd100k_seg_to_yolo(
    mask_dir,
    output_dir,
    category_map=None,
    min_area=100
):
    if category_map is None:
        category_map = SIMPLE_SEG_MAP

    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_files = list(mask_dir.glob('*.png'))
    print(f"Found {len(mask_files)} mask files")

    converted_count = 0
    skipped_count = 0

    for mask_path in tqdm(mask_files):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print(f"Warning: Failed to read {mask_path}")
            skipped_count += 1
            continue

        all_polygons = []

        for original_class_id, new_class_id in category_map.items():
            class_mask = (mask == original_class_id).astype(np.uint8)

            polygons = mask_to_polygons(class_mask, new_class_id, min_area)
            all_polygons.extend(polygons)

        if len(all_polygons) > 0:
            txt_filename = mask_path.stem + '.txt'
            txt_path = output_dir / txt_filename

            with open(txt_path, 'w') as f:
                f.write('\n'.join(all_polygons))

            converted_count += 1
        else:
            skipped_count += 1

    print(f"✅ Conversion complete!")
    print(f"   Converted: {converted_count} images")
    print(f"   Skipped (no valid masks): {skipped_count} images")


def main():
    parser = argparse.ArgumentParser(description='Convert BDD100K segmentation to YOLO format')
    parser.add_argument('--mask-dir', required=True, help='BDD100K mask directory')
    parser.add_argument('--output', required=True, help='Output directory for YOLO labels')
    parser.add_argument('--min-area', type=int, default=100, help='Minimum contour area')

    args = parser.parse_args()

    convert_bdd100k_seg_to_yolo(
        mask_dir=args.mask_dir,
        output_dir=args.output,
        min_area=args.min_area
    )


if __name__ == '__main__':
    main()
