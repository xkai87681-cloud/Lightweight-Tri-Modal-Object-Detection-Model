#!/bin/bash
# 快速调试脚本 - 只转换前100个文件并显示详细信息
# 用于快速定位问题

set -e

echo "=========================================="
echo "Quick Debug Test - BDD100K Detection"
echo "=========================================="
echo ""

BDD100K_ROOT="/root/autodl-tmp/bdd100k"
OUTPUT_ROOT="/root/autodl-tmp/new_task/data/bdd100k"
TRAIN_JSON="${BDD100K_ROOT}/labels/train"

# 检查目录
if [ ! -d "$TRAIN_JSON" ]; then
    echo "❌ Error: $TRAIN_JSON not found!"
    exit 1
fi

# 创建临时输出目录
TEMP_OUTPUT="${OUTPUT_ROOT}/labels/detection/train_debug_test"
mkdir -p "$TEMP_OUTPUT"

echo "Step 1: Inspecting first JSON file..."
echo "======================================="
FIRST_JSON=$(find "$TRAIN_JSON" -name "*.json" | head -1)
echo "File: $FIRST_JSON"
echo ""
echo "Content preview:"
head -50 "$FIRST_JSON"

echo ""
echo ""
echo "Step 2: Running debug conversion on first 100 files..."
echo "========================================================"
python scripts/bdd100k_det_to_yolo_debug.py \
    --json-dir "$TRAIN_JSON" \
    --output "$TEMP_OUTPUT" \
    --width 1280 \
    --height 720 \
    --debug-samples 5

echo ""
echo "Step 3: Checking results..."
echo "==========================="
CONVERTED_COUNT=$(ls "$TEMP_OUTPUT"/*.txt 2>/dev/null | wc -l)
echo "Files converted: $CONVERTED_COUNT"

if [ $CONVERTED_COUNT -gt 0 ]; then
    echo ""
    echo "✅ Conversion worked! Sample output:"
    ls "$TEMP_OUTPUT"/*.txt | head -3 | while read f; do
        echo ""
        echo "File: $(basename $f)"
        head -5 "$f"
    done
else
    echo ""
    echo "❌ No files converted - check debug output above for details"
fi

echo ""
echo "=========================================="
echo "Test complete!"
echo "=========================================="
