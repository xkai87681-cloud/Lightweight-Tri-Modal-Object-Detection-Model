#!/bin/bash
# BDD100K 数据集完整性检查脚本
# 验证所有必需文件是否存在

echo "=========================================="
echo "BDD100K Dataset Integrity Check"
echo "=========================================="
echo ""

BDD100K_ROOT="/root/autodl-tmp/bdd100k"
ERRORS=0

# 检查函数
check_exists() {
    local path=$1
    local description=$2

    if [ -e "$path" ]; then
        echo "✅ $description"
        return 0
    else
        echo "❌ $description - NOT FOUND!"
        ERRORS=$((ERRORS + 1))
        return 1
    fi
}

check_count() {
    local path=$1
    local pattern=$2
    local expected=$3
    local description=$4

    if [ -d "$path" ]; then
        local count=$(find "$path" -name "$pattern" -type f 2>/dev/null | wc -l)
        echo "📊 $description: $count files (expected: ~$expected)"

        if [ $count -lt $((expected / 2)) ]; then
            echo "   ⚠️  WARNING: File count seems too low!"
            ERRORS=$((ERRORS + 1))
        fi
    else
        echo "❌ $description - Directory not found!"
        ERRORS=$((ERRORS + 1))
    fi
}

echo "Step 1: Checking BDD100K root directory..."
echo "==========================================="
check_exists "$BDD100K_ROOT" "BDD100K root directory"
echo ""

echo "Step 2: Checking image directories..."
echo "======================================"
check_exists "$BDD100K_ROOT/images/100k/train" "Train images directory"
check_exists "$BDD100K_ROOT/images/100k/val" "Val images directory"
check_count "$BDD100K_ROOT/images/100k/train" "*.jpg" 70000 "Train images"
check_count "$BDD100K_ROOT/images/100k/val" "*.jpg" 10000 "Val images"
echo ""

echo "Step 3: Checking detection labels..."
echo "====================================="
check_exists "$BDD100K_ROOT/labels/det_20/det_train.json" "Detection train JSON"
check_exists "$BDD100K_ROOT/labels/det_20/det_val.json" "Detection val JSON"
echo ""

echo "Step 4: Checking segmentation masks..."
echo "======================================="
check_exists "$BDD100K_ROOT/seg/masks/train" "Segmentation train masks directory"
check_exists "$BDD100K_ROOT/seg/masks/val" "Segmentation val masks directory"
check_count "$BDD100K_ROOT/seg/masks/train" "*_train_id.png" 70000 "Train segmentation masks"
check_count "$BDD100K_ROOT/seg/masks/val" "*_val_id.png" 10000 "Val segmentation masks"
echo ""

echo "=========================================="
if [ $ERRORS -eq 0 ]; then
    echo "✅ All checks passed!"
    echo "=========================================="
    echo ""
    echo "Next step:"
    echo "  cd /root/autodl-tmp/new_task"
    echo "  bash scripts/convert_bdd100k_all.sh"
    exit 0
else
    echo "❌ Found $ERRORS error(s)!"
    echo "=========================================="
    echo ""
    echo "Troubleshooting:"
    echo "1. Make sure you downloaded ALL 3 files:"
    echo "   - bdd100k_images_100k.zip (36GB)"
    echo "   - bdd100k_labels_release.zip (2GB)"
    echo "   - bdd100k_seg_maps.zip (1.2GB) ⚠️ REQUIRED!"
    echo ""
    echo "2. Unzip all files:"
    echo "   cd /root/autodl-tmp"
    echo "   unzip bdd100k_images_100k.zip -d bdd100k/"
    echo "   unzip bdd100k_labels_release.zip -d bdd100k/"
    echo "   unzip bdd100k_seg_maps.zip -d bdd100k/"
    exit 1
fi
