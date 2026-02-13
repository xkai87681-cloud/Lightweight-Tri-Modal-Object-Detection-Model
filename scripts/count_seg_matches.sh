#!/bin/bash
# 统计分割标注与 100k 图像的匹配情况

echo "=== 统计 Train 分割标注匹配情况 ==="
TOTAL_TRAIN_LABELS=$(ls /root/autodl-tmp/new_task/data/bdd100k/labels/segmentation/train/*.txt | wc -l)
echo "总分割标注文件: $TOTAL_TRAIN_LABELS"

MATCHED=0
for label in /root/autodl-tmp/new_task/data/bdd100k/labels/segmentation/train/*.txt; do
    label_name=$(basename $label)
    img_name=${label_name/_train_id.txt/.jpg}

    if [ -f "/root/autodl-tmp/bdd100k/images/100k/train/$img_name" ]; then
        MATCHED=$((MATCHED + 1))
    fi
done

echo "匹配的图像: $MATCHED"
echo "匹配率: $(echo "scale=2; $MATCHED * 100 / $TOTAL_TRAIN_LABELS" | bc)%"

echo ""
echo "=== 统计 Val 分割标注匹配情况 ==="
TOTAL_VAL_LABELS=$(ls /root/autodl-tmp/new_task/data/bdd100k/labels/segmentation/val/*.txt | wc -l)
echo "总分割标注文件: $TOTAL_VAL_LABELS"

MATCHED_VAL=0
for label in /root/autodl-tmp/new_task/data/bdd100k/labels/segmentation/val/*.txt; do
    label_name=$(basename $label)
    img_name=${label_name/_train_id.txt/.jpg}

    if [ -f "/root/autodl-tmp/bdd100k/images/100k/val/$img_name" ]; then
        MATCHED_VAL=$((MATCHED_VAL + 1))
    fi
done

echo "匹配的图像: $MATCHED_VAL"
echo "匹配率: $(echo "scale=2; $MATCHED_VAL * 100 / $TOTAL_VAL_LABELS" | bc)%"

echo ""
echo "=== 结论 ==="
echo "Train: $MATCHED / $TOTAL_TRAIN_LABELS 可用"
echo "Val: $MATCHED_VAL / $TOTAL_VAL_LABELS 可用"
