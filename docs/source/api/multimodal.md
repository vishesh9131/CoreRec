# Multimodal Fusion

Strategies for combining embeddings from different data modalities (text, image, audio, user behavior).

## Quick Start

Modalities usually come in different widths. `MultiModalFusion` projects each
one to `output_dim` and then fuses them; inputs are a dict keyed by modality:

```python
import torch
from corerec.multimodal.fusion_strategies import MultiModalFusion

fusion = MultiModalFusion({"text": 768, "image": 512, "audio": 128}, output_dim=256,
                          strategy="attention")   # or "concat", "weighted", "gated"
batch = {"text": torch.randn(4, 768), "image": torch.randn(4, 512), "audio": torch.randn(4, 128)}
fused = fusion(batch)            # [4, 256]
```

The strategy classes work on inputs that already share a width, e.g.
`AttentionFusion(embedding_dim=256, num_modalities=3)` called with a dict of
`[batch, 256]` tensors.

## Fusion Strategies

```{eval-rst}
.. automodule:: corerec.multimodal.fusion_strategies
   :members:
   :show-inheritance:
```
