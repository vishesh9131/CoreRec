# Multimodal Fusion

Strategies for combining embeddings from different data modalities (text, image, audio, user behavior).

## Quick Start

```python
from corerec.multimodal.fusion_strategies import AttentionFusion

fusion = AttentionFusion(modality_dims=[768, 512, 128], output_dim=256)
fused = fusion(text_emb, image_emb, audio_emb)
```

## Fusion Strategies

```{eval-rst}
.. automodule:: corerec.multimodal.fusion_strategies
   :members:
   :show-inheritance:
```
