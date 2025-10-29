#!/bin/bash

# Quick Start Script for Baseline ADL Recognition
# This script demonstrates the complete workflow from data preprocessing to evaluation

set -e  # Exit on error

echo "======================================================================"
echo "🚀 Baseline ADL Recognition - Quick Start"
echo "======================================================================"
echo ""

# Configuration
DATA_DIR="../data/raw"
PROCESSED_DATA="../data/processed/dataset.npz"
CHECKPOINT_DIR="../checkpoints"
RESULTS_DIR="../results"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Step 1: Data Preprocessing
echo "${BLUE}Step 1: Preprocessing data...${NC}"
if [ ! -f "$PROCESSED_DATA" ]; then
    python data/build_features.py \
        --data_dir "$DATA_DIR" \
        --output "$PROCESSED_DATA" \
        --T 100 \
        --stride 5
    echo "${GREEN}✅ Data preprocessing complete${NC}"
else
    echo "⚠️  Processed data already exists: $PROCESSED_DATA"
fi
echo ""

# Step 2: Training
echo "${BLUE}Step 2: Training model...${NC}"
python train/train.py \
    --data_path "$PROCESSED_DATA" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --batch_size 64 \
    --lr 3e-4 \
    --epochs 30 \
    --patience 7 \
    --use_weighted_sampler

echo "${GREEN}✅ Training complete${NC}"
echo ""

# Step 3: Evaluation
echo "${BLUE}Step 3: Evaluating model...${NC}"
python evaluate/evaluate.py \
    --checkpoint "$CHECKPOINT_DIR/best_baseline.pt" \
    --data_path "$PROCESSED_DATA" \
    --split test \
    --output_dir "$RESULTS_DIR"

echo "${GREEN}✅ Evaluation complete${NC}"
echo ""

# Step 4: Visualization
echo "${BLUE}Step 4: Generating visualizations...${NC}"
python evaluate/visualize.py \
    --checkpoint "$CHECKPOINT_DIR/best_baseline.pt" \
    --data_path "$PROCESSED_DATA" \
    --output_dir "$RESULTS_DIR/visualizations" \
    --n_samples 10

echo "${GREEN}✅ Visualizations complete${NC}"
echo ""

# Summary
echo "======================================================================"
echo "🎉 Quick Start Complete!"
echo "======================================================================"
echo "📁 Results saved to: $RESULTS_DIR"
echo "💾 Model checkpoint: $CHECKPOINT_DIR/best_baseline.pt"
echo ""
echo "Next steps:"
echo "  - Check training history: $CHECKPOINT_DIR/training_history.png"
echo "  - View confusion matrix: $CHECKPOINT_DIR/confusion_matrix.png"
echo "  - Explore attention visualizations: $RESULTS_DIR/visualizations/"
echo "  - Review test results: $RESULTS_DIR/test_results.json"
echo ""
echo "======================================================================"
