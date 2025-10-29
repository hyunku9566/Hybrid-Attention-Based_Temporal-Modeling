#!/bin/bash

# Setup script to prepare original ADL data for baseline project

set -e

echo "======================================================================"
echo "📦 Baseline ADL Recognition - Data Setup"
echo "======================================================================"
echo ""

# Configuration
SOURCE_DIR="/home/lee/research-hub/hyunku/iot/iot-data"
TARGET_DIR="/home/lee/research-hub/hyunku/iot/baseline-adl-recognition/data"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "${BLUE}Source data:${NC} $SOURCE_DIR"
echo "${BLUE}Target directory:${NC} $TARGET_DIR"
echo ""

# Create necessary directories
echo "${YELLOW}Creating directories...${NC}"
mkdir -p "$TARGET_DIR/raw"
mkdir -p "$TARGET_DIR/processed"
mkdir -p "$TARGET_DIR/embeddings"

# Method 1: Symbolic links (recommended - no duplication)
echo ""
echo "${YELLOW}Method 1: Creating symbolic links (recommended)${NC}"
echo "This creates links to original data without copying."
echo ""

# Link original raw data folders
if [ -d "$SOURCE_DIR/adl_noerror" ]; then
    ln -sfn "$SOURCE_DIR/adl_noerror" "$TARGET_DIR/raw/adl_noerror"
    echo "${GREEN}✅ Linked adl_noerror (120 CSV files)${NC}"
fi

if [ -d "$SOURCE_DIR/adl_error" ]; then
    ln -sfn "$SOURCE_DIR/adl_error" "$TARGET_DIR/raw/adl_error"
    echo "${GREEN}✅ Linked adl_error (90 CSV files)${NC}"
fi

# Link processed data (if exists)
if [ -d "$SOURCE_DIR/processed_all" ]; then
    ln -sfn "$SOURCE_DIR/processed_all" "$TARGET_DIR/raw/processed_all"
    echo "${GREEN}✅ Linked processed_all (preprocessed CSVs)${NC}"
fi

# Link embeddings (if exists)
if [ -d "$SOURCE_DIR/embeddings_all_augmented" ]; then
    ln -sfn "$SOURCE_DIR/embeddings_all_augmented" "$TARGET_DIR/embeddings/embeddings_all_augmented"
    echo "${GREEN}✅ Linked embeddings_all_augmented${NC}"
fi

# Link existing preprocessed datasets (optional)
if [ -f "$SOURCE_DIR/dataset_with_lengths_v3.npz" ]; then
    ln -sfn "$SOURCE_DIR/dataset_with_lengths_v3.npz" "$TARGET_DIR/processed/dataset_with_lengths_v3.npz"
    echo "${GREEN}✅ Linked dataset_with_lengths_v3.npz (ready to use!)${NC}"
fi

echo ""
echo "${BLUE}Verifying setup...${NC}"
echo ""
echo "📁 Raw data:"
ls -lh "$TARGET_DIR/raw/" 2>/dev/null || echo "  (empty)"

echo ""
echo "📁 Processed data:"
ls -lh "$TARGET_DIR/processed/" 2>/dev/null || echo "  (empty)"

echo ""
echo "📁 Embeddings:"
ls -lh "$TARGET_DIR/embeddings/" 2>/dev/null || echo "  (empty)"

echo ""
echo "======================================================================"
echo "${GREEN}✅ Data setup complete!${NC}"
echo "======================================================================"
echo ""
echo "📊 Available data:"
echo ""
echo "  1️⃣  Original raw data (event logs):"
echo "     - data/raw/adl_noerror/*.csv  (120 files)"
echo "     - data/raw/adl_error/*.csv    (90 files)"
echo ""
echo "  2️⃣  Preprocessed data (binary matrices):"
echo "     - data/raw/processed_all/*.csv (440 files)"
echo ""
echo "  3️⃣  Pre-built dataset (ready to use):"
echo "     - data/processed/dataset_with_lengths_v3.npz"
echo ""
echo "======================================================================"
echo "🚀 Next Steps:"
echo "======================================================================"
echo ""
echo "Option A: Use existing preprocessed dataset (FASTEST)"
echo "---------------------------------------------------------"
echo "  python train/train.py \\"
echo "    --data_path data/processed/dataset_with_lengths_v3.npz \\"
echo "    --epochs 30 \\"
echo "    --batch_size 64"
echo ""
echo "Option B: Build dataset from processed CSVs"
echo "---------------------------------------------------------"
echo "  python data/build_features.py \\"
echo "    --data_dir data/raw/processed_all \\"
echo "    --emb_dir data/embeddings/embeddings_all_augmented \\"
echo "    --output data/processed/dataset.npz \\"
echo "    --T 100 \\"
echo "    --stride 5"
echo ""
echo "Option C: Preprocess raw data first (if needed)"
echo "---------------------------------------------------------"
echo "  # This requires the preprocessing pipeline from iot-data"
echo "  # You can copy preprocess_pipeline.py if needed"
echo ""
echo "======================================================================"
echo "💡 Recommended: Use Option A (existing preprocessed dataset)"
echo "   It's the fastest and already tested!"
echo "======================================================================"
