import json
import os
from pathlib import Path
from tqdm import tqdm
import argparse
from collections import Counter


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


def convert_single_json_to_yolo_debug(json_file, output_dir, category_map, image_width=1280, image_height=720, debug=False):
    stats = {
        'success': False,
        'error': None,
        'total_labels': 0,
        'valid_boxes': 0,
        'wrong_category': 0,
        'no_box2d': 0,
        'too_small': 0,
    }

    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        stats['error'] = f"JSON read error: {e}"
        if debug:
            print(f"❌ {json_file.name}: {stats['error']}")
        return stats

    image_name = data.get('name', '')
    if not image_name:
        image_name = Path(json_file).stem + '.jpg'

    if 'labels' in data:
        labels = data.get('labels', [])
        if debug:
            print(f"  Format: Detection (labels)")
    elif 'frames' in data:
        frames = data.get('frames', [])
        if len(frames) > 0:
            labels = frames[0].get('objects', [])
            if debug:
                print(f"  Format: Tracking (frames[0].objects), {len(labels)} objects in first frame")
        else:
            labels = []
            stats['error'] = "No frames in JSON"
            if debug:
                print(f"❌ {json_file.name}: Has 'frames' key but frames list is empty")
            return stats
    else:
        stats['error'] = f"Unknown JSON format. Keys: {list(data.keys())}"
        if debug:
            print(f"❌ {json_file.name}: JSON keys are {list(data.keys())}, expected 'labels' or 'frames'")
        return stats
    stats['total_labels'] = len(labels)

    if len(labels) == 0:
        stats['error'] = "No labels in JSON"
        return stats

    yolo_lines = []

    for label in labels:
        category = label.get('category', '')

        if category not in category_map:
            stats['wrong_category'] += 1
            if debug and stats['wrong_category'] == 1:  # 只打印第一次
                print(f"  Category '{category}' not in mapping")
            continue

        class_id = category_map[category]

        box2d = label.get('box2d', None)
        if box2d is None:
            stats['no_box2d'] += 1
            continue

        try:
            x_center, y_center, width, height = convert_bbox_to_yolo(
                box2d, image_width, image_height
            )
        except Exception as e:
            stats['error'] = f"Bbox conversion error: {e}"
            continue

        if width < 0.01 or height < 0.01:
            stats['too_small'] += 1
            continue

        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        yolo_lines.append(yolo_line)
        stats['valid_boxes'] += 1

    if len(yolo_lines) > 0:
        txt_filename = Path(image_name).stem + '.txt'
        txt_path = Path(output_dir) / txt_filename

        with open(txt_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

        stats['success'] = True

    return stats


def convert_bdd100k_folder_to_yolo_debug(
    json_dir,
    output_dir,
    category_map=None,
    image_width=1280,
    image_height=720,
    debug_sample_size=10
):
    if category_map is None:
        category_map = BDD100K_CATEGORIES

    json_dir = Path(json_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(json_dir.glob('*.json'))

    if len(json_files) == 0:
        print(f"❌ No JSON files found in {json_dir}")
        return

    print(f"Found {len(json_files)} JSON files in {json_dir}")
    print(f"Category mapping: {category_map}")
    print(f"\n{'='*80}")
    print(f"Starting conversion with debug mode...")
    print(f"{'='*80}\n")

    converted_count = 0
    skipped_count = 0
    total_stats = Counter()
    all_categories = Counter()

    print(f"Detailed debug for first {debug_sample_size} files:")
    print("-" * 80)

    for i, json_file in enumerate(tqdm(json_files, desc=f"Converting {json_dir.name}")):
        debug = (i < debug_sample_size)

        if debug:
            print(f"\n[{i+1}] Processing: {json_file.name}")

        stats = convert_single_json_to_yolo_debug(
            json_file, output_dir, category_map, image_width, image_height, debug=debug
        )

        if stats['success']:
            converted_count += 1
        else:
            skipped_count += 1

        total_stats['total_labels'] += stats['total_labels']
        total_stats['valid_boxes'] += stats['valid_boxes']
        total_stats['wrong_category'] += stats['wrong_category']
        total_stats['no_box2d'] += stats['no_box2d']
        total_stats['too_small'] += stats['too_small']

        if debug and stats['error']:
            print(f"  Error: {stats['error']}")
        if debug and stats['total_labels'] > 0:
            print(f"  Labels: {stats['total_labels']}, Valid: {stats['valid_boxes']}, "
                  f"Wrong cat: {stats['wrong_category']}, No box: {stats['no_box2d']}, "
                  f"Too small: {stats['too_small']}")

    print(f"\n{'='*80}")
    print(f"Conversion Summary:")
    print(f"{'='*80}")
    print(f"✅ Successfully converted: {converted_count} files")
    print(f"❌ Skipped (no valid labels): {skipped_count} files")
    print(f"\nDetailed Statistics:")
    print(f"  Total labels processed: {total_stats['total_labels']}")
    print(f"  Valid boxes converted: {total_stats['valid_boxes']}")
    print(f"  Skipped - wrong category: {total_stats['wrong_category']}")
    print(f"  Skipped - no box2d: {total_stats['no_box2d']}")
    print(f"  Skipped - too small: {total_stats['too_small']}")

    print(f"\n{'='*80}")
    print(f"Diagnosis:")
    print(f"{'='*80}")

    if converted_count == 0:
        print("❌ CRITICAL: No files were converted successfully!")
        print("\nPossible causes:")

        if total_stats['total_labels'] == 0:
            print("  1. JSON files don't have 'labels' key")
            print("     → Check JSON structure with diagnose_bdd100k_json.py")

        if total_stats['wrong_category'] > 0:
            print(f"  2. {total_stats['wrong_category']} labels have unmapped categories")
            print("     → Run check_bdd100k_categories.py to see all categories")

        if total_stats['no_box2d'] > 0:
            print(f"  3. {total_stats['no_box2d']} labels don't have 'box2d' field")
            print("     → These might be other annotation types (lane, drivable area, etc.)")

        if total_stats['too_small'] > 0:
            print(f"  4. {total_stats['too_small']} boxes are too small (<1% of image size)")
            print("     → Consider reducing min_size threshold")

    else:
        print(f"✅ Conversion succeeded for {converted_count} files!")


def main():
    parser = argparse.ArgumentParser(description='Convert BDD100K detection (debug version)')
    parser.add_argument('--json-dir', required=True, help='BDD100K JSON directory')
    parser.add_argument('--output', required=True, help='Output directory for YOLO labels')
    parser.add_argument('--width', type=int, default=1280, help='Image width')
    parser.add_argument('--height', type=int, default=720, help='Image height')
    parser.add_argument('--debug-samples', type=int, default=10, help='Number of files to debug in detail')

    args = parser.parse_args()

    convert_bdd100k_folder_to_yolo_debug(
        json_dir=args.json_dir,
        output_dir=args.output,
        image_width=args.width,
        image_height=args.height,
        debug_sample_size=args.debug_samples
    )


if __name__ == '__main__':
    main()
