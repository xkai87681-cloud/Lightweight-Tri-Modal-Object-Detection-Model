import sys
sys.path.insert(0, '/root/autodl-tmp/new_task')

from data.pa100k_dataset import PA100KDataset
from configs.config import cfg

print("=" * 80)
print("PA100K Dataset Loading Test")
print("=" * 80)
print()

print("Configuration:")
print(f"  PA100K_ROOT: {cfg.PA100K_ROOT}")
print(f"  PA100K_ANNOTATION: {cfg.PA100K_ANNOTATION}")
print(f"  USE_PA100K_CSV: {cfg.USE_PA100K_CSV}")
print()

try:
    print("Loading train set...")
    train_dataset = PA100KDataset(split='train')
    print(f"✅ Train set loaded: {len(train_dataset)} images")

    print("\nLoading val set...")
    val_dataset = PA100KDataset(split='val')
    print(f"✅ Val set loaded: {len(val_dataset)} images")

    print("\nTesting sample access...")
    sample = train_dataset[0]
    print(f"✅ Sample loaded:")
    print(f"   Image: {sample['image'].shape}")
    print(f"   Attributes: {sample['attributes'].shape}")
    print(f"   Sample keys: {list(sample.keys())}")
    if 'image_name' in sample:
        print(f"   Image name: {sample['image_name']}")
    elif 'img_name' in sample:
        print(f"   Image name: {sample['img_name']}")

    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)
    print("\nPA100K dataset is ready for training!")

except Exception as e:
    print("\n" + "=" * 80)
    print("❌ Test failed!")
    print("=" * 80)
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
