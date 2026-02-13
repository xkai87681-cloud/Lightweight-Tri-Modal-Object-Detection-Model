import os
from pathlib import Path
from typing import Dict, Optional, Tuple


def find_cityscapes_structure(cityscapes_root: Path) -> Dict[str, Path]:
    result = {
        'images_train': None,
        'images_val': None,
        'labels_train': None,
        'labels_val': None
    }

    if not cityscapes_root.exists():
        return result

    if (cityscapes_root / 'leftImg8bit' / 'train').exists():
        result['images_train'] = cityscapes_root / 'leftImg8bit' / 'train'
        result['images_val'] = cityscapes_root / 'leftImg8bit' / 'val'
    elif (cityscapes_root / 'images' / 'train').exists():
        result['images_train'] = cityscapes_root / 'images' / 'train'
        result['images_val'] = cityscapes_root / 'images' / 'val'
    elif (cityscapes_root / 'train').exists():
        result['images_train'] = cityscapes_root / 'train'
        result['images_val'] = cityscapes_root / 'val'
    else:
        for subdir in cityscapes_root.rglob('*train*'):
            if subdir.is_dir():
                image_files = list(subdir.glob('*.png')) + list(subdir.glob('*.jpg'))
                if len(image_files) > 0:
                    result['images_train'] = subdir
                    parent = subdir.parent
                    for val_dir in [parent / 'val', parent / 'validation']:
                        if val_dir.exists():
                            result['images_val'] = val_dir
                            break
                    break

    if (cityscapes_root / 'gtFine' / 'train').exists():
        result['labels_train'] = cityscapes_root / 'gtFine' / 'train'
        result['labels_val'] = cityscapes_root / 'gtFine' / 'val'
    elif (cityscapes_root / 'labels' / 'train').exists():
        result['labels_train'] = cityscapes_root / 'labels' / 'train'
        result['labels_val'] = cityscapes_root / 'labels' / 'val'
    elif (cityscapes_root / 'annotations' / 'train').exists():
        result['labels_train'] = cityscapes_root / 'annotations' / 'train'
        result['labels_val'] = cityscapes_root / 'annotations' / 'val'
    elif result['images_train'] is not None:
        parent = result['images_train'].parent
        for label_dir in ['gtFine', 'labels', 'annotations']:
            if (parent / label_dir / 'train').exists():
                result['labels_train'] = parent / label_dir / 'train'
                result['labels_val'] = parent / label_dir / 'val'
                break

    return result


def find_pa100k_structure(pa100k_root: Path) -> Dict[str, Optional[Path]]:
    result = {
        'images_dir': None,
        'annotation_mat': None,
        'train_csv': None,
        'val_csv': None,
        'use_csv': False
    }

    if not pa100k_root.exists():
        return result

    if (pa100k_root / 'release_data' / 'release_data').exists():
        result['images_dir'] = pa100k_root / 'release_data' / 'release_data'
    elif (pa100k_root / 'images').exists():
        result['images_dir'] = pa100k_root / 'images'
    elif (pa100k_root / 'data').exists():
        result['images_dir'] = pa100k_root / 'data'
    elif (pa100k_root / 'train').exists():
        result['images_dir'] = pa100k_root
    else:
        for subdir in pa100k_root.rglob('*'):
            if subdir.is_dir() and (subdir / 'train').exists():
                train_dir = subdir / 'train'
                image_files = list(train_dir.glob('*.png')) + list(train_dir.glob('*.jpg'))
                if len(image_files) > 0:
                    result['images_dir'] = subdir
                    break

    if (pa100k_root / 'annotations' / 'train.csv').exists():
        result['train_csv'] = pa100k_root / 'annotations' / 'train.csv'
        result['val_csv'] = pa100k_root / 'annotations' / 'val.csv'
        result['use_csv'] = True
    elif (pa100k_root / 'annotation.mat').exists():
        result['annotation_mat'] = pa100k_root / 'annotation.mat'
        result['use_csv'] = False
    else:
        for mat_file in pa100k_root.rglob('*.mat'):
            if 'annotation' in mat_file.stem.lower():
                result['annotation_mat'] = mat_file
                result['use_csv'] = False
                break
        if result['annotation_mat'] is None:
            for csv_file in pa100k_root.rglob('*train*.csv'):
                result['train_csv'] = csv_file
                val_csv = csv_file.parent / 'val.csv'
                if not val_csv.exists():
                    val_csv = csv_file.parent / 'validation.csv'
                if val_csv.exists():
                    result['val_csv'] = val_csv
                result['use_csv'] = True
                break

    return result


def auto_configure_datasets(project_root: Path) -> Tuple[bool, str]:
    messages = []

    cityscapes_root = project_root / 'cityscapes'
    if cityscapes_root.exists():
        cs_structure = find_cityscapes_structure(cityscapes_root)
        if cs_structure['images_train'] and cs_structure['labels_train']:
            messages.append(f"✅ Found Cityscapes:")
            messages.append(f"   Images (train): {cs_structure['images_train']}")
            messages.append(f"   Labels (train): {cs_structure['labels_train']}")
        else:
            messages.append(f"⚠️  Cityscapes directory found but structure incomplete")
            messages.append(f"   Root: {cityscapes_root}")
    else:
        messages.append(f"❌ Cityscapes not found at {cityscapes_root}")

    pa100k_root = project_root / 'PA100K'
    if pa100k_root.exists():
        pa_structure = find_pa100k_structure(pa100k_root)
        if pa_structure['images_dir']:
            messages.append(f"✅ Found PA100K:")
            messages.append(f"   Images: {pa_structure['images_dir']}")
            if pa_structure['use_csv']:
                messages.append(f"   Format: CSV")
                messages.append(f"   Train CSV: {pa_structure['train_csv']}")
            else:
                messages.append(f"   Format: MAT")
                messages.append(f"   Annotation: {pa_structure['annotation_mat']}")
        else:
            messages.append(f"⚠️  PA100K directory found but structure incomplete")
            messages.append(f"   Root: {pa100k_root}")
    else:
        messages.append(f"❌ PA100K not found at {pa100k_root}")

    return True, "\n".join(messages)


if __name__ == '__main__':
    from configs.config import cfg

    print("=" * 80)
    print("Dataset Auto-Discovery Test")
    print("=" * 80)

    success, message = auto_configure_datasets(cfg.PROJECT_ROOT)
    print(message)
