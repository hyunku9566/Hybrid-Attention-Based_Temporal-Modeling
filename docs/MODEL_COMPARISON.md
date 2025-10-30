# 🔬 Model Comparison: Baseline vs Transformer

## 📊 Performance Comparison

### Overall Results

| Model | Architecture | Accuracy | Macro F1 | Parameters | Training Time |
|-------|--------------|----------|----------|------------|---------------|
| **Baseline (TCN-BiGRU-Attention)** | TCN + BiGRU + Additive Attention | **95.40%** ⭐ | **0.9468** ⭐ | 2.26M | ~15 min |
| **Transformer** | 4-layer Encoder, 8 heads | 83.66% | 0.8042 | 3.22M | ~12 min |

### Performance Gap
- **Accuracy Difference**: Baseline is **11.74%p better** than Transformer
- **Macro F1 Difference**: Baseline is **0.1426 better** than Transformer
- **Parameters**: Transformer has 42% more parameters (3.22M vs 2.26M)

---

## 🔍 Per-Class Performance Comparison

| Activity | Baseline F1 | Transformer F1 | Difference | Winner |
|----------|-------------|----------------|------------|---------|
| **Cooking (t1)** | 0.9363 | 0.8051 | +0.1312 | 🏆 Baseline |
| **Hand Washing (t2)** | 0.9722 | 0.7785 | +0.1937 | 🏆 Baseline |
| **Sleeping (t3)** | 0.9862 | 0.9567 | +0.0295 | 🏆 Baseline |
| **Medicine (t4)** | 0.9190 | 0.8051 | +0.1139 | 🏆 Baseline |
| **Eating (t5)** | 0.9202 | 0.6753 | +0.2449 | 🏆 Baseline |

### Key Observations
- **Baseline dominates** across all classes
- **Biggest gap**: Eating activity (t5) - Baseline is 24.49%p better
- **Smallest gap**: Sleeping activity (t3) - Baseline is 2.95%p better
- **Short activities**: Baseline excels at hand washing (0.9722 vs 0.7785)

---

## 🎯 Detailed Analysis

### Why Baseline Outperforms Transformer?

#### 1. **Architecture Suitability**
- **Baseline**: TCN captures local temporal patterns, BiGRU models long-term dependencies, attention focuses on important timesteps
- **Transformer**: Global self-attention may over-smooth short sequences (T=100)
- **Verdict**: Baseline's hybrid approach is better suited for ADL recognition

#### 2. **Inductive Bias**
- **Baseline**: Strong inductive bias with causal TCN (respects temporal order)
- **Transformer**: Relies on positional encoding, weaker temporal bias
- **Verdict**: Explicit temporal modeling (TCN+BiGRU) wins

#### 3. **Data Efficiency**
- **Baseline**: More parameter-efficient (2.26M params, 95.40% accuracy)
- **Transformer**: Less efficient (3.22M params, 83.66% accuracy)
- **Verdict**: Baseline achieves better performance with fewer parameters

#### 4. **Short Sequence Handling**
- **Baseline**: Designed for sequences with T=100, receptive field=13
- **Transformer**: Global attention on short sequences may cause overfitting
- **Verdict**: Baseline is optimized for this sequence length

#### 5. **Activity-Specific Performance**
- **Short activities (t2, t5)**: Baseline's local temporal modeling excels
- **Long activities (t3)**: Both models perform well, Baseline slightly better
- **Verdict**: Baseline better handles diverse activity durations

---

## 📈 Training Characteristics

### Convergence Speed
| Metric | Baseline | Transformer |
|--------|----------|-------------|
| **Epochs to Best Val** | 36 | 15 |
| **Early Stopping** | No (completed 50) | Yes (at epoch 30) |
| **Best Val Acc** | 94.96% | 84.42% |
| **Final Test Acc** | 95.40% | 83.66% |

### Observations
- **Transformer converges faster** but to a **worse optimum**
- **Baseline takes longer** but achieves **significantly better** performance
- **Early stopping**: Transformer stopped early, suggesting limited capacity for this task

---

## 💡 Key Insights

### 1. Task-Specific Architecture Matters
- ADL recognition benefits from **explicit temporal modeling** (TCN+BiGRU)
- Pure attention mechanisms (Transformer) are **less effective** for short sensor sequences

### 2. Hybrid Approaches Win
- Combining **convolution (TCN)**, **recurrence (BiGRU)**, and **attention** provides best results
- Each component contributes: TCN (local patterns), BiGRU (long-term), Attention (focus)

### 3. Parameter Efficiency
- **More parameters ≠ Better performance**
- Baseline achieves 95.40% with 2.26M params
- Transformer achieves 83.66% with 3.22M params
- **Efficiency ratio**: Baseline is 42% more efficient

### 4. Short Activity Recognition
- **Critical for ADL**: Hand washing (10.8 steps), eating (21.7 steps)
- **Baseline excels**: F1 scores > 0.97 for hand washing
- **Transformer struggles**: F1 scores < 0.78 for hand washing

### 5. Generalization
- **Baseline**: Small gap between validation (94.96%) and test (95.40%)
- **Transformer**: Slight degradation from validation (84.42%) to test (83.66%)
- **Verdict**: Both generalize well, Baseline performs better overall

---

## 🚀 Recommendations

### When to Use Baseline (TCN-BiGRU-Attention)
✅ **Short sensor sequences** (T < 200)  
✅ **Activity recognition** with diverse durations  
✅ **Resource-constrained** environments (2.26M params)  
✅ **High accuracy required** (95%+)  
✅ **Interpretability** needed (attention weights)

### When to Use Transformer
⚠️ **Long sequences** (T > 500) where global attention helps  
⚠️ **Large datasets** (>100k samples) for better training  
⚠️ **Transfer learning** from pretrained models  
⚠️ **Multi-modal data** requiring cross-attention

### For ADL Recognition Specifically
🏆 **Winner: Baseline (TCN-BiGRU-Attention)**
- 11.74% higher accuracy
- Better short activity recognition
- More parameter efficient
- Stronger temporal inductive bias

---

## 📋 Configuration Details

### Baseline Model
```python
Architecture: TCN + BiGRU + Additive Attention
Hidden Dim: 256
TCN Blocks: 3 (dilation: 1, 2, 4)
BiGRU Layers: 1 (bidirectional)
Dropout: 0.1
Parameters: 2,264,326

Training:
Batch Size: 32
Learning Rate: 3e-4
Focal Gamma: 1.5
Epochs: 50 (completed all)
Best Val Acc: 94.96% (epoch 36)
```

### Transformer Model
```python
Architecture: Transformer Encoder
Hidden Dim: 256
Layers: 4
Attention Heads: 8
FFN Dim: 1024 (4x hidden)
Dropout: 0.1
Parameters: 3,222,021

Training:
Batch Size: 32
Learning Rate: 3e-4
Focal Gamma: 1.5
Epochs: 30 (early stopped)
Best Val Acc: 84.42% (epoch 15)
```

---

## 🎯 Conclusion

**For ADL recognition with smart home sensors:**

1. **Baseline (TCN-BiGRU-Attention) is the clear winner**
   - 95.40% accuracy vs 83.66%
   - Better on all activity classes
   - More parameter efficient

2. **Transformers are not optimal for this task**
   - Short sequences don't benefit from global attention
   - Weaker temporal inductive bias hurts performance
   - Higher computational cost for worse results

3. **Key Success Factor: Hybrid Architecture**
   - TCN: Local temporal patterns
   - BiGRU: Long-term dependencies
   - Attention: Dynamic focus
   - This combination outperforms pure attention mechanisms

**Recommendation**: **Use Baseline model** for production ADL recognition systems.

---

**Last Updated**: October 30, 2025  
**Models Compared**: Baseline v2.0, Transformer v1.0  
**Dataset**: Smart Home ADL (7,066 sequences)
