import json
import sys
from pathlib import Path
from collections import Counter


def check_categories(json_dir, max_files=1000):
    json_dir = Path(json_dir)
    json_files = list(json_dir.glob('*.json'))[:max_files]

    print(f"Checking {len(json_files)} JSON files from {json_dir}")
    print(f"{'='*80}\n")

    all_categories = []
    files_with_labels = 0
    files_without_labels = 0
    total_labels = 0

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            labels = data.get('labels', [])
            if len(labels) > 0:
                files_with_labels += 1
                total_labels += len(labels)
                for label in labels:
                    category = label.get('category', 'UNKNOWN')
                    all_categories.append(category)
            else:
                files_without_labels += 1

        except Exception as e:
            print(f"Error reading {json_file}: {e}")

    category_counts = Counter(all_categories)

    print(f"Statistics:")
    print(f"  Files with labels: {files_with_labels}")
    print(f"  Files without labels: {files_without_labels}")
    print(f"  Total labels: {total_labels}")
    print(f"\n{'='*80}")
    print(f"Categories found (sorted by frequency):")
    print(f"{'='*80}\n")

    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:20s}: {count:6d} instances")

    print(f"\n{'='*80}")
    print(f"\nConversion script mapping:")
    print(f"{'='*80}\n")

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

    print(f"Mapped categories (will be converted):")
    for cat in BDD100K_CATEGORIES.keys():
        count = category_counts.get(cat, 0)
        print(f"  {cat:20s}: {count:6d} instances {'✅' if count > 0 else '❌ NOT FOUND'}")

    unmapped = [cat for cat in category_counts.keys() if cat not in BDD100K_CATEGORIES]
    if unmapped:
        print(f"\nUnmapped categories (will be SKIPPED):")
        for cat in unmapped:
            count = category_counts.get(cat, 0)
            print(f"  {cat:20s}: {count:6d} instances")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python check_bdd100k_categories.py <json_dir>")
        print("Example: python check_bdd100k_categories.py /root/autodl-tmp/bdd100k/labels/train")
        sys.exit(1)

    check_categories(sys.argv[1])
