#!/bin/bash
# 检查重组后的 val 目录

echo "=== 检查 val 标注目录 ==="
VAL_DIR="/root/autodl-tmp/new_task/data/bdd100k/labels/segmentation/val"

echo "Val 标注文件数量:"
ls $VAL_DIR/*.txt 2>/dev/null | wc -l

echo ""
echo "前5个 val 标注文件:"
ls $VAL_DIR/*.txt 2>/dev/null | head -5 | xargs -I {} basename {}

echo ""
echo "=== 检查这些标注对应的图像是否存在 ==="
for label in $(ls $VAL_DIR/*.txt 2>/dev/null | head -3); do
    label_name=$(basename $label)

    # 尝试所有可能的后缀
    for suffix in "_train_id" "_val_id" ""; do
        if [ -n "$suffix" ]; then
            img_name=${label_name/${suffix}.txt/.jpg}
        else
            img_name=${label_name/.txt/.jpg}
        fi

        img_path="/root/autodl-tmp/new_task/data/bdd100k/images/val/${img_name}"

        if [ -e "$img_path" ]; then
            echo "  ✅ $label_name -> $img_name (exists)"
            break
        fi
    done

    if [ ! -e "$img_path" ]; then
        echo "  ❌ $label_name -> 无法找到对应图像"
    fi
done

echo ""
echo "=== 检查标注来源 ==="
echo "这些标注是从 train 还是 val 划分的？"
# 检查标注文件名中的后缀
ls $VAL_DIR/*.txt 2>/dev/null | head -3 | xargs -I {} basename {} | while read f; do
    if echo "$f" | grep -q "_train_id"; then
        echo "  $f - 来自train (图像在 /images/train)"
    elif echo "$f" | grep -q "_val_id"; then
        echo "  $f - 来自val (图像在 /images/val)"
    fi
done
