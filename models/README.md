# Models Package

## Architecture Overview

The baseline model consists of three main components:

### 1. Temporal Convolutional Network (TCN)
- **Purpose**: Extract local temporal patterns
- **Configuration**: 3 blocks with dilations [1, 2, 4]
- **Receptive field**: 13 timesteps
- **Key feature**: Causal dilated convolutions with residual connections

### 2. Bidirectional GRU (BiGRU)
- **Purpose**: Capture global sequential context
- **Configuration**: 128 hidden units, 1 layer, bidirectional
- **Output**: Concatenated forward and backward states (256-dim)
- **Advantage**: 25% fewer parameters than BiLSTM

### 3. Additive Attention
- **Purpose**: Focus on critical time steps
- **Type**: Bahdanau-style attention with single focus
- **Output**: Weighted context vector (256-dim) + attention weights
- **Interpretability**: Attention weights can be visualized

## File Descriptions

### `baseline_model.py`
Main model definition combining all components.

**Key classes**:
- `BaselineModel`: Complete ADL recognition model

**Usage**:
```python
from models import BaselineModel

model = BaselineModel(in_dim=114, hidden=128, classes=5)
logits = model(input_tensor)  # [batch, 5]
logits, attn = model(input_tensor, return_attention=True)
```

### `components.py`
Reusable building blocks for the model.

**Key classes**:
- `TCNBlock`: Single TCN block with dilated convolutions
- `AdditiveAttention`: Bahdanau-style attention mechanism
- `FocalLoss`: Loss function for handling class imbalance

**Usage**:
```python
from models.components import TCNBlock, AdditiveAttention, FocalLoss

# TCN block
tcn = TCNBlock(in_ch=128, out_ch=128, kernel=3, dilation=2)
out = tcn(input)  # [batch, channels, time]

# Attention
attn = AdditiveAttention(hidden_dim=128)
context, weights = attn(gru_output)  # context: [batch, 256], weights: [batch, time]

# Focal loss
loss_fn = FocalLoss(gamma=2.0, weight=class_weights)
loss = loss_fn(logits, targets)
```

## Model Specifications

### Parameters
- **Total**: 575,110 parameters (0.58M)
- **Model size**: ~2.3 MB (float32)

### Receptive Field Calculation
```
TCN Block 1 (dilation=1): kernel=3 → RF = 5
TCN Block 2 (dilation=2): kernel=3 → RF = 9
TCN Block 3 (dilation=4): kernel=3 → RF = 13
```

### Memory Requirements
- **Training**: ~4 GB GPU memory (batch_size=64)
- **Inference**: <100 MB per sample

## Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 93.7% |
| Macro F1 | 0.924 |
| Parameters | 0.58M |
| Inference Time | 2.66ms (GPU) |
| Model Size | 2.3 MB |

## Design Rationale

### Why TCN?
- ✅ Exponential receptive field growth with depth
- ✅ Parameter efficient (shared kernels)
- ✅ Causal structure for real-time inference
- ❌ Standard CNN: requires many layers for long-range dependencies

### Why BiGRU over BiLSTM?
- ✅ 25% fewer parameters (3 gates vs 4)
- ✅ Faster training and inference
- ✅ Equal performance on ADL tasks
- ❌ BiLSTM: unnecessary complexity for this task

### Why Additive over Multi-Head Attention?
- ✅ Single attention focus (no over-smoothing)
- ✅ 40% fewer parameters
- ✅ Better for short sequences (<15 timesteps)
- ✅ More interpretable weights
- ❌ Multi-head: dilutes attention across heads, harms t2 F1 (0.848 vs 0.956)

## Ablation Study Results

| Variant | Accuracy | Macro F1 | t2 F1 | Note |
|---------|----------|----------|-------|------|
| **Baseline** | **93.7%** | **0.924** | **0.956** | ✅ Optimal |
| TCN-5 (deeper) | 90.9% | 0.856 | 0.821 | Over-smoothing |
| Multi-Head 4h | 92.8% | 0.895 | 0.848 | Feature dilution |
| Multi-Head 8h | 93.0% | 0.889 | 0.857 | Worse dilution |

## References

1. **TCN**: Bai et al. "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling" (2018)
2. **Attention**: Bahdanau et al. "Neural Machine Translation by Jointly Learning to Align and Translate" (2015)
3. **Focal Loss**: Lin et al. "Focal Loss for Dense Object Detection" (2017)
