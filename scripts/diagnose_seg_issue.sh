#!/bin/bash
# BDD100K 分割数据集精确诊断脚本

echo "=== 图像目录检查 ==="
echo "Train images dir:"
ls -ld /root/autodl-tmp/new_task/data/bdd100k/images/train | head -1
echo ""
echo "Val images dir:"
ls -ld /root/autodl-tmp/new_task/data/bdd100k/images/val | head -1

echo ""
echo "=== 检查图像文件数量 ==="
echo "Train images:"
find /root/autodl-tmp/new_task/data/bdd100k/images/train -name "*.jpg" | wc -l
echo "Val images:"
find /root/autodl-tmp/new_task/data/bdd100k/images/val -name "*.jpg" | wc -l

echo ""
echo "=== 检查具体的标注文件对应的图像是否存在 ==="
echo "Train segmentation label -> image match:"
for label in $(ls /root/autodl-tmp/new_task/data/bdd100k/labels/segmentation/train/*.txt | head -3); do
    label_name=$(basename $label)
    img_name=${label_name/_train_id.txt/.jpg}
    img_path="/root/autodl-tmp/new_task/data/bdd100k/images/train/${img_name}"

    if [ -e "$img_path" ]; then
        echo "  ✅ $label_name -> $img_name (exists)"
    else
        echo "  ❌ $label_name -> $img_name (NOT FOUND)"
    fi
done

echo ""
echo "Val segmentation label -> image match:"
for label in $(ls /root/autodl-tmp/new_task/data/bdd100k/labels/segmentation/val/*.txt | head -3); do
    label_name=$(basename $label)
    img_name=${label_name/_train_id.txt/.jpg}
    img_path="/root/autodl-tmp/new_task/data/bdd100k/images/val/${img_name}"

    if [ -e "$img_path" ]; then
        echo "  ✅ $label_name -> $img_name (exists)"
    else
        echo "  ❌ $label_name -> $img_name (NOT FOUND)"
    fi
done

echo ""
echo "=== 检查软链接是否有效 ==="
echo "Sample train image (first one):"
ls -lh /root/autodl-tmp/new_task/data/bdd100k/images/train/*.jpg 2>/dev/null | head -1

echo ""
echo "Sample val image (first one):"
ls -lh /root/autodl-tmp/new_task/data/bdd100k/images/val/*.jpg 2>/dev/null | head -1

echo ""
echo "=== 检查图像文件名示例 ==="
echo "First 5 train images:"
find /root/autodl-tmp/new_task/data/bdd100k/images/train -name "*.jpg" | head -5 | xargs -I {} basename {}

echo ""
echo "First 5 val images:"
find /root/autodl-tmp/new_task/data/bdd100k/images/val -name "*.jpg" | head -5 | xargs -I {} basename {}
