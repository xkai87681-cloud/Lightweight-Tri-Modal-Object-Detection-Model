#!/bin/bash
# BDD100K 目录结构检查脚本
# 帮助诊断实际的目录结构

echo "=========================================="
echo "BDD100K Directory Structure Inspection"
echo "=========================================="
echo ""

BDD100K_ROOT="/root/autodl-tmp/bdd100k"

echo "Checking root directory..."
ls -lh $BDD100K_ROOT/

echo ""
echo "========================================"
echo "Checking labels/ structure..."
echo "========================================"
if [ -d "$BDD100K_ROOT/labels" ]; then
    echo "Top-level directories in labels/:"
    ls -lh $BDD100K_ROOT/labels/

    echo ""
    echo "Checking for train/ directory..."
    if [ -d "$BDD100K_ROOT/labels/train" ]; then
        echo "✅ Found labels/train/"
        echo "   Sample files:"
        ls $BDD100K_ROOT/labels/train/ | head -5
        echo "   Total files:"
        ls $BDD100K_ROOT/labels/train/ | wc -l
    fi

    echo ""
    echo "Checking for val/ directory..."
    if [ -d "$BDD100K_ROOT/labels/val" ]; then
        echo "✅ Found labels/val/"
        echo "   Sample files:"
        ls $BDD100K_ROOT/labels/val/ | head -5
        echo "   Total files:"
        ls $BDD100K_ROOT/labels/val/ | wc -l
    fi

    echo ""
    echo "Checking for det_20/ directory..."
    if [ -d "$BDD100K_ROOT/labels/det_20" ]; then
        echo "✅ Found labels/det_20/"
        echo "   Files:"
        ls -lh $BDD100K_ROOT/labels/det_20/
    fi

    echo ""
    echo "Sample JSON file content (first file in train/):"
    if [ -d "$BDD100K_ROOT/labels/train" ]; then
        FIRST_JSON=$(ls $BDD100K_ROOT/labels/train/*.json | head -1)
        if [ -f "$FIRST_JSON" ]; then
            echo "File: $FIRST_JSON"
            head -30 "$FIRST_JSON"
        fi
    fi
else
    echo "❌ labels/ directory not found!"
fi

echo ""
echo "========================================"
echo "Checking seg/ structure..."
echo "========================================"
if [ -d "$BDD100K_ROOT/seg" ]; then
    echo "Structure of seg/:"
    ls -lh $BDD100K_ROOT/seg/

    if [ -d "$BDD100K_ROOT/seg/masks" ]; then
        echo ""
        echo "Checking seg/masks/:"
        ls -lh $BDD100K_ROOT/seg/masks/

        if [ -d "$BDD100K_ROOT/seg/masks/train" ]; then
            echo ""
            echo "Sample mask files in train/:"
            ls $BDD100K_ROOT/seg/masks/train/ | head -5
            echo "Total mask files:"
            ls $BDD100K_ROOT/seg/masks/train/ | wc -l
        fi
    fi
else
    echo "⚠️  seg/ directory not found!"
    echo "Did you unzip bdd100k_seg_maps.zip?"
fi

echo ""
echo "========================================"
echo "Summary"
echo "========================================"
echo "Please share the output above so I can:"
echo "1. Understand your exact directory structure"
echo "2. Update the conversion scripts accordingly"
echo "3. Fix any path mismatches"
