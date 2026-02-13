#!/bin/bash
# 验证 BDD100K 转换结果

echo "=========================================="
echo "BDD100K Conversion Verification"
echo "=========================================="
echo ""

OUTPUT_ROOT="/root/autodl-tmp/new_task/data/bdd100k"

echo "Counting files (this may take a moment)..."
echo ""

# 使用 find 命令统计，避免参数列表太长的问题
DET_TRAIN_COUNT=$(find ${OUTPUT_ROOT}/labels/detection/train -name "*.txt" -type f 2>/dev/null | wc -l)
DET_VAL_COUNT=$(find ${OUTPUT_ROOT}/labels/detection/val -name "*.txt" -type f 2>/dev/null | wc -l)
SEG_TRAIN_COUNT=$(find ${OUTPUT_ROOT}/labels/segmentation/train -name "*.txt" -type f 2>/dev/null | wc -l)
SEG_VAL_COUNT=$(find ${OUTPUT_ROOT}/labels/segmentation/val -name "*.txt" -type f 2>/dev/null | wc -l)
# 图像可能是软链接，不指定 -type
IMG_TRAIN_COUNT=$(find ${OUTPUT_ROOT}/images/train -name "*.jpg" 2>/dev/null | wc -l)
IMG_VAL_COUNT=$(find ${OUTPUT_ROOT}/images/val -name "*.jpg" 2>/dev/null | wc -l)

echo "=========================================="
echo "Final Dataset Statistics"
echo "=========================================="
echo ""

echo "Detection Labels:"
echo "  Train: ${DET_TRAIN_COUNT} files (expected: ~69,500)"
echo "  Val:   ${DET_VAL_COUNT} files (expected: ~9,900)"
echo ""

echo "Segmentation Labels:"
echo "  Train: ${SEG_TRAIN_COUNT} files (expected: ~6,876)"
echo "  Val:   ${SEG_VAL_COUNT} files (expected: ~991)"
echo ""

echo "Images:"
echo "  Train: ${IMG_TRAIN_COUNT} files (expected: 70,000)"
echo "  Val:   ${IMG_VAL_COUNT} files (expected: 10,000)"
echo ""

# 检查结果
ERRORS=0

if [ $DET_TRAIN_COUNT -lt 60000 ]; then
    echo "❌ Detection train labels too few!"
    ERRORS=$((ERRORS + 1))
fi

if [ $DET_VAL_COUNT -lt 9000 ]; then
    echo "❌ Detection val labels too few!"
    ERRORS=$((ERRORS + 1))
fi

if [ $IMG_TRAIN_COUNT -lt 60000 ]; then
    echo "❌ Train images missing!"
    ERRORS=$((ERRORS + 1))
fi

if [ $IMG_VAL_COUNT -lt 9000 ]; then
    echo "❌ Val images missing!"
    ERRORS=$((ERRORS + 1))
fi

echo "=========================================="
if [ $ERRORS -eq 0 ]; then
    echo "✅ All checks passed!"
    echo "=========================================="
    echo ""
    echo "Dataset is ready for training!"
    echo ""
    echo "Next step:"
    echo "  cd /root/autodl-tmp/new_task"
    echo "  python train/train.py"
else
    echo "❌ Found $ERRORS error(s)!"
    echo "=========================================="
fi
