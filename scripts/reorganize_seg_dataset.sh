#!/bin/bash
# 重新组织 BDD100K 分割数据集
# 只保留与 100k 图像集匹配的标注，并从 train 中划分 val

set -e

OUTPUT_ROOT="/root/autodl-tmp/new_task/data/bdd100k"
BDD100K_ROOT="/root/autodl-tmp/bdd100k"

echo "=========================================="
echo "BDD100K Segmentation Dataset Reorganization"
echo "=========================================="
echo ""

# 创建临时目录
TEMP_DIR="/tmp/bdd100k_seg_reorganize"
mkdir -p $TEMP_DIR/matched_train
mkdir -p $TEMP_DIR/new_train
mkdir -p $TEMP_DIR/new_val

echo "Step 1: 收集匹配的标注..."
MATCHED_COUNT=0

for label in ${OUTPUT_ROOT}/labels/segmentation/train/*.txt; do
    label_name=$(basename $label)
    img_name=${label_name/_train_id.txt/.jpg}

    if [ -f "${BDD100K_ROOT}/images/100k/train/$img_name" ]; then
        cp "$label" "$TEMP_DIR/matched_train/"
        MATCHED_COUNT=$((MATCHED_COUNT + 1))
    fi
done

echo "✅ 找到 $MATCHED_COUNT 个匹配的标注文件"

echo ""
echo "Step 2: 划分 train/val (90%/10%)..."

# 计算划分点
TOTAL=$MATCHED_COUNT
VAL_COUNT=$((TOTAL / 10))
TRAIN_COUNT=$((TOTAL - VAL_COUNT))

echo "  Total: $TOTAL"
echo "  Train: $TRAIN_COUNT (90%)"
echo "  Val: $VAL_COUNT (10%)"

# 随机打乱并划分
cd $TEMP_DIR/matched_train
ls *.txt | shuf > /tmp/shuffled_list.txt

# 前 90% 作为 train
head -n $TRAIN_COUNT /tmp/shuffled_list.txt | while read f; do
    cp "$f" "$TEMP_DIR/new_train/"
done

# 后 10% 作为 val
tail -n $VAL_COUNT /tmp/shuffled_list.txt | while read f; do
    cp "$f" "$TEMP_DIR/new_val/"
done

echo ""
echo "Step 3: 替换原标注目录..."

# 备份原目录
mv ${OUTPUT_ROOT}/labels/segmentation/train ${OUTPUT_ROOT}/labels/segmentation/train_backup
mv ${OUTPUT_ROOT}/labels/segmentation/val ${OUTPUT_ROOT}/labels/segmentation/val_backup

# 使用新目录
mkdir -p ${OUTPUT_ROOT}/labels/segmentation/train
mkdir -p ${OUTPUT_ROOT}/labels/segmentation/val

cp $TEMP_DIR/new_train/*.txt ${OUTPUT_ROOT}/labels/segmentation/train/
cp $TEMP_DIR/new_val/*.txt ${OUTPUT_ROOT}/labels/segmentation/val/

echo ""
echo "Step 4: 清理临时文件..."
rm -rf $TEMP_DIR
rm -f /tmp/shuffled_list.txt

echo ""
echo "=========================================="
echo "✅ 重组完成！"
echo "=========================================="
echo ""

FINAL_TRAIN=$(ls ${OUTPUT_ROOT}/labels/segmentation/train/*.txt | wc -l)
FINAL_VAL=$(ls ${OUTPUT_ROOT}/labels/segmentation/val/*.txt | wc -l)

echo "最终数据集："
echo "  Train: $FINAL_TRAIN 张"
echo "  Val: $FINAL_VAL 张"
echo ""
echo "备份目录（如需恢复）："
echo "  ${OUTPUT_ROOT}/labels/segmentation/train_backup"
echo "  ${OUTPUT_ROOT}/labels/segmentation/val_backup"
