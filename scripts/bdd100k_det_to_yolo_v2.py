import json
import os
from pathlib import Path
from tqdm import tqdm
import argparse


BDD100K_CATEGORIES = {
    'person': 0,       # person (BDD100K tracking 格式)
    'pedestrian': 0,   # person (BDD100K detection 格式，兼容)
    'rider': 0,        # person
    'car': 1,
    'truck': 1,
    'bus': 1,
    'train': 1,
    'motorcycle': 1,
    'bicycle': 1,
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


def convert_single_json_to_yolo(json_file, output_dir, category_map, image_width=1280, image_height=720):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_file}: {e}")
        return False


    image_name = data.get('name', '')
    if not image_name:
        image_name = Path(json_file).stem + '.jpg'

    if 'labels' in data:
        labels = data.get('labels', [])
    elif 'frames' in data:
        frames = data.get('frames', [])
        if len(frames) > 0:
            labels = frames[0].get('objects', [])
        else:
            labels = []
    else:
        labels = []

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
        txt_filename = Path(image_name).stem + '.txt'
        txt_path = Path(output_dir) / txt_filename

        with open(txt_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

        return True

    return False


def convert_bdd100k_folder_to_yolo(
    json_dir,
    output_dir,
    category_map=None,
    image_width=1280,
    image_height=720
):
    if category_map is None:
        category_map = BDD100K_CATEGORIES

    json_dir = Path(json_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(json_dir.glob('*.json'))

    if len(json_files) == 0:
        print(f"Warning: No JSON files found in {json_dir}")
        return

    print(f"Found {len(json_files)} JSON files in {json_dir}")

    converted_count = 0
    skipped_count = 0

    for json_file in tqdm(json_files, desc=f"Converting {json_dir.name}"):
        if convert_single_json_to_yolo(json_file, output_dir, category_map, image_width, image_height):
            converted_count += 1
        else:
            skipped_count += 1

    print(f"✅ Conversion complete!")
    print(f"   Converted: {converted_count} files")
    print(f"   Skipped (no valid labels): {skipped_count} files")


def main():
    parser = argparse.ArgumentParser(description='Convert BDD100K detection (multi-JSON) to YOLO format')
    parser.add_argument('--json-dir', required=True, help='BDD100K JSON directory (e.g., labels/train/)')
    parser.add_argument('--output', required=True, help='Output directory for YOLO labels')
    parser.add_argument('--width', type=int, default=1280, help='Image width')
    parser.add_argument('--height', type=int, default=720, help='Image height')

    args = parser.parse_args()

    convert_bdd100k_folder_to_yolo(
        json_dir=args.json_dir,
        output_dir=args.output,
        image_width=args.width,
        image_height=args.height
    )


if __name__ == '__main__':
    main()
