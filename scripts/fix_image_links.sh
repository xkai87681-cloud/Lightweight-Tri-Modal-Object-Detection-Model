#!/bin/bash
# 修复图像软链接问题
# 使用循环方式避免参数列表过长

set -e

echo "=========================================="
echo "Fixing BDD100K Image Links"
echo "=========================================="
echo ""

BDD100K_ROOT="/root/autodl-tmp/bdd100k"
OUTPUT_ROOT="/root/autodl-tmp/new_task/data/bdd100k"

# 检查源目录
if [ ! -d "${BDD100K_ROOT}/images/100k/train" ]; then
    echo "❌ Error: Source train images not found at ${BDD100K_ROOT}/images/100k/train"
    exit 1
fi

if [ ! -d "${BDD100K_ROOT}/images/100k/val" ]; then
    echo "❌ Error: Source val images not found at ${BDD100K_ROOT}/images/100k/val"
    exit 1
fi

# 创建目标目录
mkdir -p ${OUTPUT_ROOT}/images/train
mkdir -p ${OUTPUT_ROOT}/images/val

echo "Method 1: Attempting to create directory-level symbolic link..."
echo "================================================================"

# 先尝试删除旧的链接（如果存在）
rm -rf ${OUTPUT_ROOT}/images/train 2>/dev/null || true
rm -rf ${OUTPUT_ROOT}/images/val 2>/dev/null || true

# 方法1：直接链接整个目录（最快）
ln -sf ${BDD100K_ROOT}/images/100k/train ${OUTPUT_ROOT}/images/train
ln -sf ${BDD100K_ROOT}/images/100k/val ${OUTPUT_ROOT}/images/val

# 验证
TRAIN_COUNT=$(find ${OUTPUT_ROOT}/images/train -name "*.jpg" -type f 2>/dev/null | wc -l)
VAL_COUNT=$(find ${OUTPUT_ROOT}/images/val -name "*.jpg" -type f 2>/dev/null | wc -l)

echo "Train images: $TRAIN_COUNT"
echo "Val images: $VAL_COUNT"

if [ $TRAIN_COUNT -gt 60000 ] && [ $VAL_COUNT -gt 9000 ]; then
    echo ""
    echo "✅ Directory-level link successful!"
    echo "=========================================="
    exit 0
fi

echo ""
echo "Method 1 failed. Trying Method 2: Individual file links..."
echo "==========================================================="

# 方法2：如果方法1失败，创建独立目录并逐个链接文件
rm -rf ${OUTPUT_ROOT}/images/train 2>/dev/null || true
rm -rf ${OUTPUT_ROOT}/images/val 2>/dev/null || true
mkdir -p ${OUTPUT_ROOT}/images/train
mkdir -p ${OUTPUT_ROOT}/images/val

echo "Linking train images (this may take a few minutes)..."
cd ${OUTPUT_ROOT}/images/train
find ${BDD100K_ROOT}/images/100k/train -name "*.jpg" -type f | while read img; do
    ln -sf "$img" .
done

echo "Linking val images..."
cd ${OUTPUT_ROOT}/images/val
find ${BDD100K_ROOT}/images/100k/val -name "*.jpg" -type f | while read img; do
    ln -sf "$img" .
done

# 验证
cd /root/autodl-tmp/new_task
TRAIN_COUNT=$(find ${OUTPUT_ROOT}/images/train -name "*.jpg" 2>/dev/null | wc -l)
VAL_COUNT=$(find ${OUTPUT_ROOT}/images/val -name "*.jpg" 2>/dev/null | wc -l)

echo ""
echo "=========================================="
echo "Final count:"
echo "  Train: $TRAIN_COUNT images"
echo "  Val: $VAL_COUNT images"
echo "=========================================="

if [ $TRAIN_COUNT -gt 60000 ] && [ $VAL_COUNT -gt 9000 ]; then
    echo "✅ Success!"
else
    echo "❌ Still failed. Please check source image paths."
    exit 1
fi
