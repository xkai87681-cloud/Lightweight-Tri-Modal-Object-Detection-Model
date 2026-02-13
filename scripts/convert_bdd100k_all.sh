#!/bin/bash
# BDD100K 数据集完整转换脚本
# 将 BDD100K 转换为 YOLO 格式并组织目录结构

set -e  # 遇到错误立即退出

echo "========================================"
echo "BDD100K to YOLO Conversion Script"
echo "========================================"

# 配置路径
BDD100K_ROOT="/root/autodl-tmp/bdd100k"
OUTPUT_ROOT="/root/autodl-tmp/new_task/data/bdd100k"

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

# 转换训练集检测标注
python scripts/bdd100k_det_to_yolo.py \
    --json ${BDD100K_ROOT}/labels/det_20/det_train.json \
    --output ${OUTPUT_ROOT}/labels/detection/train \
    --width 1280 \
    --height 720

# 转换验证集检测标注
python scripts/bdd100k_det_to_yolo.py \
    --json ${BDD100K_ROOT}/labels/det_20/det_val.json \
    --output ${OUTPUT_ROOT}/labels/detection/val \
    --width 1280 \
    --height 720

echo ""
echo "Step 2: Converting Segmentation annotations..."
echo "================================================"

# 转换训练集分割标注（seg_maps.zip解压后的路径）
python scripts/bdd100k_seg_to_yolo.py \
    --mask-dir ${BDD100K_ROOT}/seg/masks/train \
    --output ${OUTPUT_ROOT}/labels/segmentation/train \
    --min-area 100

# 转换验证集分割标注
python scripts/bdd100k_seg_to_yolo.py \
    --mask-dir ${BDD100K_ROOT}/seg/masks/val \
    --output ${OUTPUT_ROOT}/labels/segmentation/val \
    --min-area 100

echo ""
echo "Step 3: Creating symbolic links for images..."
echo "==============================================="

# 创建图像软链接（节省空间）
ln -sf ${BDD100K_ROOT}/images/100k/train/*.jpg ${OUTPUT_ROOT}/images/train/
ln -sf ${BDD100K_ROOT}/images/100k/val/*.jpg ${OUTPUT_ROOT}/images/val/

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
echo "Dataset structure:"
tree -L 3 ${OUTPUT_ROOT}

echo ""
echo "Training samples:"
echo "  Detection train: $(ls ${OUTPUT_ROOT}/labels/detection/train/*.txt | wc -l)"
echo "  Detection val:   $(ls ${OUTPUT_ROOT}/labels/detection/val/*.txt | wc -l)"
echo "  Segmentation train: $(ls ${OUTPUT_ROOT}/labels/segmentation/train/*.txt | wc -l)"
echo "  Segmentation val:   $(ls ${OUTPUT_ROOT}/labels/segmentation/val/*.txt | wc -l)"

echo ""
echo "Next steps:"
echo "1. Update configs/config.py to point to ${OUTPUT_ROOT}"
echo "2. Run training: python train/train.py"
