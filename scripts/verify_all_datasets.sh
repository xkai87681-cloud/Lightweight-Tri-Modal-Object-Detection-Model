#!/bin/bash
# 完整数据集验证脚本
# 验证 Detection, Segmentation, Attribute 三个任务的数据集

echo "=========================================="
echo "Multi-Task Dataset Verification"
echo "=========================================="
echo ""

cd /root/autodl-tmp/new_task

echo "Step 1: Verifying BDD100K (Detection + Segmentation)"
echo "====================================================="
bash scripts/verify_conversion.sh

echo ""
echo ""
echo "Step 2: Verifying PA100K (Attribute)"
echo "====================================="
python scripts/test_pa100k.py

echo ""
echo ""
echo "=========================================="
echo "Final Summary"
echo "=========================================="
echo ""
echo "If all checks passed above, your dataset is ready!"
echo ""
echo "Next step:"
echo "  python train/train.py"
echo ""
