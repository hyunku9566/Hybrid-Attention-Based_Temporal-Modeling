#!/bin/bash
# Train all 5 models on raw binary data (no embeddings)
# This will create comparison results showing the value of embeddings

echo "================================================================================"
echo "Training All Models on Raw Binary Data (NO Embeddings)"
echo "================================================================================"
echo ""
echo "This experiment demonstrates the contribution of sensor embeddings by"
echo "comparing performance with embeddings (F=114) vs without (F=28)"
echo ""

DATASET="data/processed/dataset_raw_binary.npz"

# Check if dataset exists
if [ ! -f "$DATASET" ]; then
    echo "❌ Dataset not found: $DATASET"
    echo "Please run: python create_raw_binary_dataset.py"
    exit 1
fi

echo "📊 Dataset: $DATASET"
echo "🔬 Models to train: baseline, deep_tcn, local_hybrid, conformer, transformer"
echo ""

# Function to train a model
train_model() {
    local model_name=$1
    local checkpoint_dir="checkpoints_${model_name}_raw"
    local log_file="training_${model_name}_raw.log"
    
    echo "🚀 Training $model_name on raw binary data..."
    echo "   Checkpoint: $checkpoint_dir"
    echo "   Log: $log_file"
    
    python run_train_all.py \
        --model $model_name \
        --data_path $DATASET \
        --checkpoint_dir $checkpoint_dir \
        --epochs 30 \
        --batch_size 32 \
        --hidden_dim 256 \
        --dropout 0.1 \
        --lr 0.001 \
        > $log_file 2>&1 &
    
    local pid=$!
    echo "   PID: $pid"
    echo ""
}

# Train all models in parallel
echo "Starting training jobs..."
echo ""

train_model "baseline"
sleep 2
train_model "deep_tcn"
sleep 2
train_model "local_hybrid"
sleep 2
train_model "conformer"
sleep 2
train_model "transformer"

echo "================================================================================"
echo "✅ All 5 training jobs launched!"
echo "================================================================================"
echo ""
echo "Monitor progress with:"
echo "  tail -f training_*_raw.log"
echo "  ./monitor_training.sh"
echo ""
echo "Check running processes:"
echo "  ps aux | grep run_train_all"
echo ""
