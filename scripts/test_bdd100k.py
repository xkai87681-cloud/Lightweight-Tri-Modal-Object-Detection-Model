import sys
sys.path.insert(0, '/root/autodl-tmp/new_task')

from data.bdd100k_det import BDD100KDetection
from data.bdd100k_seg import BDD100KSegmentation
from configs.config import cfg

print("=" * 80)
print("BDD100K Dataset Loading Test")
print("=" * 80)
print()

print("1. Testing Detection Dataset")
print("-" * 80)
try:
    det_train = BDD100KDetection(split='train')
    print(f"✅ Detection train loaded: {len(det_train)} images")

    det_val = BDD100KDetection(split='val')
    print(f"✅ Detection val loaded: {len(det_val)} images")

    sample = det_train[0]
    print(f"✅ Sample loaded:")
    print(f"   Image: {sample['image'].shape}")
    print(f"   Bboxes: {sample['bboxes'].shape}")
    print(f"   Labels: {sample['labels'].shape}")
    print()
except Exception as e:
    print(f"❌ Detection dataset failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("2. Testing Segmentation Dataset")
print("-" * 80)
try:
    seg_train = BDD100KSegmentation(split='train')
    print(f"✅ Segmentation train loaded: {len(seg_train)} images")

    seg_val = BDD100KSegmentation(split='val')
    print(f"✅ Segmentation val loaded: {len(seg_val)} images")

    sample = seg_train[0]
    print(f"✅ Sample loaded:")
    print(f"   Image: {sample['image'].shape}")
    print(f"   Mask: {sample['mask'].shape}")
    print(f"   Unique classes in mask: {sample['mask'].unique().tolist()}")
    print()
except Exception as e:
    print(f"❌ Segmentation dataset failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 80)
print("✅ All tests passed!")
print("=" * 80)
print()
print("BDD100K datasets are ready for training!")
print()
print("Expected dataset sizes:")
print(f"  Detection train: {len(det_train)} (expected: ~69,500)")
print(f"  Detection val: {len(det_val)} (expected: ~9,900)")
print(f"  Segmentation train: {len(seg_train)} (expected: ~6,876)")
print(f"  Segmentation val: {len(seg_val)} (expected: ~991)")
