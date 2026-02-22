# SASRec: Self-Attentive Sequential Recommendation

**SASRec** (Self-Attentive Sequential Recommendation) applies the Transformer architecture to sequential recommendation problems. It models the entire user sequence using self-attention mechanisms to capture long-term semantics.

## Overview

Unlike RNNs/CNNs, SASRec uses an attention mechanism to look at the entire history of a user's interactions simultaneously. This allows it to:
1.  **Capture Long-Range Dependencies**: Remember items clicked long ago.
2.  **Parallelize Training**: Significantly faster than RNN-based models.

## Usage

```python
from corerec.engines import SASRec

# 1. Initialize
model = SASRec(
    item_num=10000,
    hidden_units=128,
    num_blocks=2,
    num_heads=2,
    dropout_rate=0.2
)

# 2. Train
# (Requires sequential data loader)
```

## API Reference

::: corerec.engines.sasrec.SASRec
    options:
      show_root_heading: true
      show_source: true

## Key Hyperparameters

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `hidden_units` | 64 | Dimension of the embedding and attention layers. |
| `num_blocks` | 2 | Number of Transformer blocks (layers). |
| `num_heads` | 1 | Number of attention heads. |
| `maxlen` | 50 | Maximum sequence length to consider. |

## Theory

SASRec minimizes the binary cross-entropy loss:

$$
L = -\sum_{u \in U} \sum_{t=1}^{n} \left[ \log(\sigma(\hat{r}_{u, t, i^+})) + \sum_{j \in S^-} \log(1 - \sigma(\hat{r}_{u, t, j})) \right]
$$

where $i^+$ is the ground truth next item and $S^-$ are negative samples.
