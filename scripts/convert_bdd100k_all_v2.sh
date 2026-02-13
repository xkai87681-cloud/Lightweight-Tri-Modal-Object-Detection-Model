#!/bin/bash
# BDD100K 数据集完整转换脚本（适配 labels/train/, labels/val/ 结构）
# 版本：V2 - 适用于每个图像一个JSON文件的结构

set -e  # 遇到错误立即退出

echo "========================================"
echo "BDD100K to YOLO Conversion Script V2"
echo "Adapted for labels/train/, labels/val/ structure"
echo "========================================"

# 配置路径
BDD100K_ROOT="/root/autodl-tmp/bdd100k"
OUTPUT_ROOT="/root/autodl-tmp/new_task/data/bdd100k"

# 检查必要的目录是否存在
if [ ! -d "${BDD100K_ROOT}/labels/train" ]; then
    echo "❌ Error: ${BDD100K_ROOT}/labels/train not found!"
    echo "Please check your BDD100K directory structure."
    exit 1
fi

if [ ! -d "${BDD100K_ROOT}/labels/val" ]; then
    echo "❌ Error: ${BDD100K_ROOT}/labels/val not found!"
    echo "Please check your BDD100K directory structure."
    exit 1
fi

# 创建输出目录
mkdir -p ${OUTPUT_ROOT}/images/train
mkdir -p ${OUTPUT_ROOT}/images/val
mkdir -p ${OUTPUT_ROOT}/labels/detection/train
mkdir -p ${OUTPUT_ROOT}/labels/detection/val
mkdir -p ${OUTPUT_ROOT}/labels/segmentation/train
mkdir -p ${OUTPUT_ROOT}/labels/segmentation/val

echo ""
echo "Step 1: Converting Detection annotations..."
echo "============================================"

# 转换训练集检测标注（使用V2脚本）
echo "Converting train set..."
python scripts/bdd100k_det_to_yolo_v2.py \
    --json-dir ${BDD100K_ROOT}/labels/train \
    --output ${OUTPUT_ROOT}/labels/detection/train \
    --width 1280 \
    --height 720

# 转换验证集检测标注
echo ""
echo "Converting val set..."
python scripts/bdd100k_det_to_yolo_v2.py \
    --json-dir ${BDD100K_ROOT}/labels/val \
    --output ${OUTPUT_ROOT}/labels/detection/val \
    --width 1280 \
    --height 720

echo ""
echo "Step 2: Converting Segmentation annotations..."
echo "================================================"

# 检查分割标注是否存在
if [ -d "${BDD100K_ROOT}/seg/masks/train" ]; then
    echo "Converting train segmentation masks..."
    python scripts/bdd100k_seg_to_yolo.py \
        --mask-dir ${BDD100K_ROOT}/seg/masks/train \
        --output ${OUTPUT_ROOT}/labels/segmentation/train \
        --min-area 100

    echo ""
    echo "Converting val segmentation masks..."
    python scripts/bdd100k_seg_to_yolo.py \
        --mask-dir ${BDD100K_ROOT}/seg/masks/val \
        --output ${OUTPUT_ROOT}/labels/segmentation/val \
        --min-area 100
else
    echo "⚠️  Warning: Segmentation masks not found at ${BDD100K_ROOT}/seg/masks/"
    echo "Please unzip bdd100k_seg_maps.zip if you haven't done so."
    echo "Skipping segmentation conversion..."
fi

echo ""
echo "Step 3: Creating symbolic links for images..."
echo "==============================================="

# 创建图像软链接（节省空间）
echo "Linking train images..."
ln -sf ${BDD100K_ROOT}/images/100k/train/*.jpg ${OUTPUT_ROOT}/images/train/ 2>/dev/null || true

echo "Linking val images..."
ln -sf ${BDD100K_ROOT}/images/100k/val/*.jpg ${OUTPUT_ROOT}/images/val/ 2>/dev/null || true

echo ""
echo "Step 4: Creating dataset.yaml..."
echo "=================================="

# 创建 YOLO dataset.yaml
cat > ${OUTPUT_ROOT}/detection.yaml <<EOL
# BDD100K Detection Dataset (YOLO format)
path: ${OUTPUT_ROOT}
train: images/train
val: images/val

# Classes (simplified to 2)
nc: 2
names: ['person', 'vehicle']
EOL

cat > ${OUTPUT_ROOT}/segmentation.yaml <<EOL
# BDD100K Segmentation Dataset (YOLO format)
path: ${OUTPUT_ROOT}
train: images/train
val: images/val

# Classes (simplified to 2)
nc: 2
names: ['road', 'lane']
EOL

echo ""
echo "✅ Conversion Complete!"
echo "======================="
echo "Output directory: ${OUTPUT_ROOT}"
echo ""

# 统计文件数量
echo "Dataset Statistics:"
echo "==================="

DET_TRAIN_COUNT=$(ls ${OUTPUT_ROOT}/labels/detection/train/*.txt 2>/dev/null | wc -l)
DET_VAL_COUNT=$(ls ${OUTPUT_ROOT}/labels/detection/val/*.txt 2>/dev/null | wc -l)

echo "Detection:"
echo "  Train: ${DET_TRAIN_COUNT} files"
echo "  Val:   ${DET_VAL_COUNT} files"

if [ -d "${BDD100K_ROOT}/seg/masks/train" ]; then
    SEG_TRAIN_COUNT=$(ls ${OUTPUT_ROOT}/labels/segmentation/train/*.txt 2>/dev/null | wc -l)
    SEG_VAL_COUNT=$(ls ${OUTPUT_ROOT}/labels/segmentation/val/*.txt 2>/dev/null | wc -l)

    echo "Segmentation:"
    echo "  Train: ${SEG_TRAIN_COUNT} files"
    echo "  Val:   ${SEG_VAL_COUNT} files"
fi

IMG_TRAIN_COUNT=$(ls ${OUTPUT_ROOT}/images/train/*.jpg 2>/dev/null | wc -l)
IMG_VAL_COUNT=$(ls ${OUTPUT_ROOT}/images/val/*.jpg 2>/dev/null | wc -l)

echo "Images:"
echo "  Train: ${IMG_TRAIN_COUNT} files"
echo "  Val:   ${IMG_VAL_COUNT} files"

echo ""
echo "Next steps:"
echo "==========="
echo "1. Verify the counts above match expected values (~70,000 train, ~10,000 val)"
echo "2. Update configs/config.py (already done automatically)"
echo "3. Run training: python train/train.py"
