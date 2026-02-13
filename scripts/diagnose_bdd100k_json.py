import json
import sys
from pathlib import Path
from collections import Counter


def diagnose_json_file(json_file):
    print(f"\n{'='*80}")
    print(f"Diagnosing: {json_file}")
    print(f"{'='*80}\n")

    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading JSON: {e}")
        return

    print("1. Top-level keys in JSON:")
    print(f"   {list(data.keys())}\n")

    print("2. Basic fields:")
    print(f"   'name': {data.get('name', 'NOT FOUND')}")
    print(f"   'attributes': {data.get('attributes', 'NOT FOUND')}")

    labels = data.get('labels', [])
    print(f"\n3. Labels field:")
    print(f"   Total labels: {len(labels)}")

    if len(labels) == 0:
        print("   ⚠️  WARNING: No labels found in this JSON!")
        return

    print(f"\n4. First label structure:")
    first_label = labels[0]
    print(f"   Keys: {list(first_label.keys())}")
    print(f"   Full content:")
    print(f"   {json.dumps(first_label, indent=4)}\n")

    categories = [label.get('category', 'UNKNOWN') for label in labels]
    category_counts = Counter(categories)

    print(f"5. Category statistics:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"   {cat}: {count}")

    print(f"\n6. Bbox format check:")
    has_box2d = 0
    has_box3d = 0
    no_box = 0

    for label in labels:
        if 'box2d' in label:
            has_box2d += 1
            if has_box2d == 1:  # 打印第一个box2d的示例
                print(f"   Example box2d: {label['box2d']}")
        elif 'box3d' in label:
            has_box3d += 1
        else:
            no_box += 1

    print(f"   Labels with box2d: {has_box2d}")
    print(f"   Labels with box3d: {has_box3d}")
    print(f"   Labels without box: {no_box}")

    has_poly2d = sum(1 for label in labels if 'poly2d' in label)
    print(f"   Labels with poly2d: {has_poly2d}")

    print(f"\n7. Category mapping check:")
    BDD100K_CATEGORIES = {
        'pedestrian': 0,
        'rider': 0,
        'car': 1,
        'truck': 1,
        'bus': 1,
        'train': 1,
        'motorcycle': 1,
        'bicycle': 1,
    }

    print(f"   Categories in conversion script: {list(BDD100K_CATEGORIES.keys())}")
    print(f"   Categories in this JSON: {list(category_counts.keys())}")

    unmapped = [cat for cat in category_counts.keys() if cat not in BDD100K_CATEGORIES]
    if unmapped:
        print(f"   ⚠️  Unmapped categories: {unmapped}")
    else:
        print(f"   ✅ All categories are mapped!")

    print(f"\n8. Simulating conversion:")
    valid_boxes = 0
    too_small = 0
    wrong_category = 0
    no_bbox = 0

    for label in labels:
        category = label.get('category', '')

        if category not in BDD100K_CATEGORIES:
            wrong_category += 1
            continue

        box2d = label.get('box2d', None)
        if box2d is None:
            no_bbox += 1
            continue

        x1, y1 = box2d.get('x1', 0), box2d.get('y1', 0)
        x2, y2 = box2d.get('x2', 0), box2d.get('y2', 0)
        width = abs(x2 - x1) / 1280.0
        height = abs(y2 - y1) / 720.0

        if width < 0.01 or height < 0.01:
            too_small += 1
            continue

        valid_boxes += 1

    print(f"   Valid boxes: {valid_boxes}")
    print(f"   Skipped - wrong category: {wrong_category}")
    print(f"   Skipped - no bbox: {no_bbox}")
    print(f"   Skipped - too small: {too_small}")

    if valid_boxes == 0:
        print(f"\n   ❌ No valid boxes would be converted!")
        print(f"   This explains why the conversion resulted in 0 files.")
    else:
        print(f"\n   ✅ This file should produce valid YOLO labels!")


def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_bdd100k_json.py <json_dir>")
        print("Example: python diagnose_bdd100k_json.py /root/autodl-tmp/bdd100k/labels/train")
        sys.exit(1)

    json_dir = Path(sys.argv[1])

    if not json_dir.exists():
        print(f"❌ Directory not found: {json_dir}")
        sys.exit(1)

    json_files = list(json_dir.glob('*.json'))

    if len(json_files) == 0:
        print(f"❌ No JSON files found in {json_dir}")
        sys.exit(1)

    print(f"Found {len(json_files)} JSON files in {json_dir}")
    print(f"Analyzing the first 3 files...\n")

    for json_file in json_files[:3]:
        diagnose_json_file(json_file)

    print(f"\n{'='*80}")
    print("Diagnosis complete!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
