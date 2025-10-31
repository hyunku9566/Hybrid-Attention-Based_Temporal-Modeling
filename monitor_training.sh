#!/bin/bash
# Monitor training progress for T variant experiments

echo "=================================="
echo "Training Progress Monitor"
echo "=================================="
echo ""

# Count running processes
RUNNING=$(ps aux | grep "python run_train_all" | grep -v grep | wc -l)
echo "🚀 Running processes: $RUNNING / 10"
echo ""

# Check which models are done
echo "📊 Completion Status:"
echo ""

for T in 50 150; do
    echo "T=$T:"
    for MODEL in baseline deep_tcn local_hybrid conformer transformer; do
        CHECKPOINT_DIR="checkpoints_${MODEL}_T${T}"
        if [ -f "${CHECKPOINT_DIR}/test_results.json" ]; then
            ACC=$(python3 -c "import json; print(f\"{json.load(open('${CHECKPOINT_DIR}/test_results.json'))['test_acc']*100:.2f}%\")" 2>/dev/null || echo "N/A")
            echo "  ✅ $MODEL: $ACC"
        else
            echo "  ⏳ $MODEL: Training..."
        fi
    done
    echo ""
done

# Show recent log activity
echo "📝 Recent Log Activity (last 5 lines of each):"
echo ""
for LOG in training_*_T*.log; do
    if [ -f "$LOG" ]; then
        SIZE=$(wc -l < "$LOG" 2>/dev/null || echo "0")
        if [ "$SIZE" -gt "0" ]; then
            echo "--- $LOG (${SIZE} lines) ---"
            tail -3 "$LOG" 2>/dev/null | grep -E "Epoch|✅|Test Accuracy" || echo "  (still initializing...)"
            echo ""
        fi
    fi
done

# Show checkpoint sizes
echo "💾 Checkpoint Sizes:"
du -sh checkpoints_*_T* 2>/dev/null | sort -h

echo ""
echo "🔄 To refresh: ./monitor_training.sh"
echo "⏹️  To stop all: pkill -f 'python run_train_all'"
