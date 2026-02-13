#!/bin/bash
# 检查 BDD100K 分割图像位置

echo "=== 检查 BDD100K 图像目录结构 ==="
ls -lh /root/autodl-tmp/bdd100k/images/

echo ""
echo "=== 检查是否有 10k 子目录（分割图像）==="
ls /root/autodl-tmp/bdd100k/images/ | grep -E "10k|seg"

echo ""
echo "=== 检查分割标注对应的图像是否在 100k 中 ==="
echo "检查第1个分割图像（不匹配的）："
test -f /root/autodl-tmp/bdd100k/images/100k/train/0004a4c0-d4dff0ad.jpg && echo "  ✅ Found in 100k/train" || echo "  ❌ Not in 100k/train"
test -f /root/autodl-tmp/bdd100k/images/100k/val/0004a4c0-d4dff0ad.jpg && echo "  ✅ Found in 100k/val" || echo "  ❌ Not in 100k/val"

echo ""
echo "检查第2个分割图像（匹配的）："
test -f /root/autodl-tmp/bdd100k/images/100k/train/00054602-3bf57337.jpg && echo "  ✅ 00054602-3bf57337.jpg found in 100k/train" || echo "  ❌ Not found in 100k/train"

echo ""
echo "=== 检查所有图像子目录 ==="
find /root/autodl-tmp/bdd100k/images -maxdepth 2 -type d
