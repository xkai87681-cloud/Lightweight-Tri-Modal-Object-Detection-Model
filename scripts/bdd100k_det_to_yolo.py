import json
import os
from pathlib import Path
from tqdm import tqdm
import argparse


BDD100K_CATEGORIES = {
    'pedestrian': 0,  # person
    'rider': 0,       # person (骑行者也算人)
    'car': 1,
    'truck': 1,       # 合并到car
    'bus': 1,         # 合并到car
    'train': 1,       # 合并到car
    'motorcycle': 1,  # 合并到car
    'bicycle': 1,     # 合并到car
}

BDD100K_CATEGORIES_FULL = {
    'pedestrian': 0,
    'rider': 1,
    'car': 2,
    'truck': 3,
    'bus': 4,
    'train': 5,
    'motorcycle': 6,
    'bicycle': 7,
    'traffic light': 8,
    'traffic sign': 9,
}


def convert_bbox_to_yolo(bbox, img_width, img_height):
    x1, y1 = bbox['x1'], bbox['y1']
    x2, y2 = bbox['x2'], bbox['y2']

    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1

    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height

    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))

    return x_center, y_center, width, height


def convert_bdd100k_to_yolo(
    json_file,
    output_dir,
    category_map=None,
    image_width=1280,
    image_height=720
):
    if category_map is None:
        category_map = BDD100K_CATEGORIES

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)

    print(f"Converting {len(data)} images...")

    converted_count = 0
    skipped_count = 0

    for item in tqdm(data):
        image_name = item['name']
        labels = item.get('labels', [])

        yolo_lines = []

        for label in labels:
            category = label.get('category', '')

            if category not in category_map:
                continue

            class_id = category_map[category]

            box2d = label.get('box2d', None)
            if box2d is None:
                continue

            x_center, y_center, width, height = convert_bbox_to_yolo(
                box2d, image_width, image_height
            )

            if width < 0.01 or height < 0.01:
                continue

            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            yolo_lines.append(yolo_line)

        if len(yolo_lines) > 0:
            txt_filename = image_name.replace('.jpg', '.txt')
            txt_path = output_dir / txt_filename

            with open(txt_path, 'w') as f:
                f.write('\n'.join(yolo_lines))

            converted_count += 1
        else:
            skipped_count += 1

    print(f"✅ Conversion complete!")
    print(f"   Converted: {converted_count} images")
    print(f"   Skipped (no valid labels): {skipped_count} images")


def main():
    parser = argparse.ArgumentParser(description='Convert BDD100K detection to YOLO format')
    parser.add_argument('--json', required=True, help='BDD100K JSON file')
    parser.add_argument('--output', required=True, help='Output directory for YOLO labels')
    parser.add_argument('--width', type=int, default=1280, help='Image width')
    parser.add_argument('--height', type=int, default=720, help='Image height')
    parser.add_argument('--full-classes', action='store_true', help='Use 10 classes instead of 2')

    args = parser.parse_args()

    category_map = BDD100K_CATEGORIES_FULL if args.full_classes else BDD100K_CATEGORIES

    convert_bdd100k_to_yolo(
        json_file=args.json,
        output_dir=args.output,
        category_map=category_map,
        image_width=args.width,
        image_height=args.height
    )


if __name__ == '__main__':
    main()
