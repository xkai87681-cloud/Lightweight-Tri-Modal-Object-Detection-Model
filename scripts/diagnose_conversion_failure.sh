#!/bin/bash
# BDD100K 检测标注转换失败诊断脚本
# 自动识别问题原因

set -e

echo "=========================================="
echo "BDD100K Detection Conversion Diagnosis"
echo "=========================================="
echo ""

BDD100K_ROOT="/root/autodl-tmp/bdd100k"
TRAIN_LABELS="${BDD100K_ROOT}/labels/train"

# 检查目录是否存在
if [ ! -d "$TRAIN_LABELS" ]; then
    echo "❌ Error: $TRAIN_LABELS not found!"
    exit 1
fi

echo "Step 1: Checking JSON file structure..."
echo "========================================="

# 获取第一个JSON文件
FIRST_JSON=$(find "$TRAIN_LABELS" -name "*.json" -type f | head -1)

if [ -z "$FIRST_JSON" ]; then
    echo "❌ No JSON files found in $TRAIN_LABELS"
    exit 1
fi

echo "Sample file: $FIRST_JSON"
echo ""
echo "JSON content preview:"
echo "--------------------"
head -50 "$FIRST_JSON"
echo ""
echo "--------------------"
echo ""

echo "Step 2: Checking categories in dataset..."
echo "=========================================="
python scripts/check_bdd100k_categories.py "$TRAIN_LABELS"

echo ""
echo "Step 3: Detailed diagnosis..."
echo "============================="
python scripts/diagnose_bdd100k_json.py "$TRAIN_LABELS"

echo ""
echo "=========================================="
echo "Diagnosis complete!"
echo "=========================================="
echo ""
echo "Please share the output above to identify the issue."
