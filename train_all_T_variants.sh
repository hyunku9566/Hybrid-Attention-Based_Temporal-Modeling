#!/bin/bash
# Train all models with different T values

# Configuration
MODELS=("baseline" "deep_tcn" "local_hybrid" "conformer" "transformer")
T_VALUES=(50 150)  # T=100 already done
HIDDEN_DIM=256
DROPOUT=0.1
BATCH_SIZE=32
LR=3e-4
GAMMA=1.5
EPOCHS=50
PATIENCE=15

# Activate virtual environment
source .venv/bin/activate

echo "=================================="
echo "Training All Models with Different T Values"
echo "=================================="
echo "Models: ${MODELS[@]}"
echo "T values: ${T_VALUES[@]}"
echo "=================================="

# Train each model with each T value
for T in "${T_VALUES[@]}"; do
    echo ""
    echo "===================================="
    echo "T=$T - Starting All Models"
    echo "===================================="
    
    DATA_PATH="data/processed/T_variants/dataset_T${T}.npz"
    
    for MODEL in "${MODELS[@]}"; do
        echo ""
        echo "------------------------------------"
        echo "T=$T | Model: $MODEL"
        echo "------------------------------------"
        
        CHECKPOINT_DIR="checkpoints_${MODEL}_T${T}"
        LOG_FILE="training_${MODEL}_T${T}.log"
        
        echo "Starting training..."
        echo "  Data: $DATA_PATH"
        echo "  Checkpoint: $CHECKPOINT_DIR"
        echo "  Log: $LOG_FILE"
        
        python run_train_all.py \
            --model $MODEL \
            --data_path $DATA_PATH \
            --checkpoint_dir $CHECKPOINT_DIR \
            --hidden_dim $HIDDEN_DIM \
            --dropout $DROPOUT \
            --batch_size $BATCH_SIZE \
            --lr $LR \
            --gamma $GAMMA \
            --epochs $EPOCHS \
            --patience $PATIENCE \
            > $LOG_FILE 2>&1 &
        
        PID=$!
        echo "  Started with PID: $PID"
        
        # Wait a bit before starting next model to avoid resource contention
        sleep 5
    done
    
    echo ""
    echo "All models for T=$T started in background"
    echo "Check progress with: tail -f training_*_T${T}.log"
done

echo ""
echo "===================================="
echo "All Training Jobs Launched!"
echo "===================================="
echo ""
echo "Running processes:"
ps aux | grep "python run_train_all" | grep -v grep

echo ""
echo "📊 To monitor progress:"
echo "  watch -n 10 'ps aux | grep run_train_all | grep -v grep'"
echo ""
echo "📁 Log files:"
ls -lh training_*_T*.log 2>/dev/null

echo ""
echo "⏳ Estimated time per model: 15-30 minutes"
echo "   Total models: ${#MODELS[@]} × ${#T_VALUES[@]} = $((${#MODELS[@]} * ${#T_VALUES[@]})) models"
echo "   Estimated total time: 2.5-5 hours (parallel execution)"
