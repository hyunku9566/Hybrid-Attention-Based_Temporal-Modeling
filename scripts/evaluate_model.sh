#!/bin/bash

# Evaluate a trained model on test set

CHECKPOINT="$1"
DATA_PATH="${2:-../data/processed/dataset.npz}"
OUTPUT_DIR="${3:-../results}"

if [ -z "$CHECKPOINT" ]; then
    echo "Usage: ./evaluate_model.sh <checkpoint_path> [data_path] [output_dir]"
    echo ""
    echo "Example:"
    echo "  ./evaluate_model.sh ../checkpoints/best_baseline.pt"
    exit 1
fi

echo "======================================================================"
echo "📊 Evaluating Model"
echo "======================================================================"
echo "Checkpoint: $CHECKPOINT"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo ""

python ../evaluate/evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --data_path "$DATA_PATH" \
    --split test \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "======================================================================"
echo "✅ Evaluation Complete!"
echo "======================================================================"
echo "Results saved to: $OUTPUT_DIR"
