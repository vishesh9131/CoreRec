# Quick Start: Modern RecSys with CoreRec

This guide gets you up and running with modern deep learning recommendation systems in 5 minutes.

## Installation

```bash
pip install corerec

# Optional: for production-scale vector search
pip install faiss-cpu  # or faiss-gpu
```

## Example 1: Two-Tower Retrieval (Fast & Scalable)

Perfect for: Large item catalogs (millions), real-time serving, first stage of pipeline

```python
from corerec.engines import TwoTower
from corerec.retrieval.vector_store import create_index
import numpy as np

# Your data
user_ids = ['user_1', 'user_2', 'user_3']
item_ids = ['item_1', 'item_2', 'item_3', 'item_4']
interactions = np.array([
    [1, 1, 0, 0],  # user_1 interacted with item_1, item_2
    [0, 1, 1, 0],  # user_2 interacted with item_2, item_3
    [1, 0, 0, 1],  # user_3 interacted with item_1, item_4
])

# Train model
model = TwoTower(
    embedding_dim=128,
    hidden_dims=[256, 128],
    loss_type="bce",
    num_epochs=10
)

model.fit(user_ids, item_ids, interactions)

# Build fast index
item_embs = model.get_item_embeddings()
index = create_index("numpy", dim=item_embs.shape[1])  # use "faiss" for scale
index.add(item_embs, item_ids)

# Get recommendations
recs = model.recommend('user_1', top_k=5)
print(f"Recommendations: {recs}")
```

## Example 2: Sequential Recommendation (Time-Aware)

Perfect for: Session-based, "next item" prediction, time-sensitive content

```python
from corerec.engines import BERT4Rec
import numpy as np

# Your sequential data
user_ids = ['user_1', 'user_2']
item_ids = ['item_1', 'item_2', 'item_3', 'item_4', 'item_5']
interactions = np.array([
    [1, 1, 1, 0, 0],  # user_1: viewed items 1,2,3 in order
    [0, 1, 1, 1, 0],  # user_2: viewed items 2,3,4 in order
])

# Train
model = BERT4Rec(
    hidden_dim=128,
    num_layers=2,
    num_heads=4,
    num_epochs=10
)

model.fit(user_ids, item_ids, interactions)

# Predict next item
next_items = model.recommend('user_1', top_k=3)
print(f"Next items: {next_items}")
```

## Example 3: Multi-Modal Fusion

Perfect for: Rich item representations (text + images), content-based filtering

```python
from corerec.multimodal.fusion_strategies import MultiModalFusion
import torch

# Your multi-modal features
text_embeddings = torch.randn(10, 768)      # 10 items, BERT embeddings
image_embeddings = torch.randn(10, 2048)    # ResNet features
metadata = torch.randn(10, 32)              # categories, price, etc

# Create fusion
fusion = MultiModalFusion(
    modality_dims={
        'text': 768,
        'image': 2048,
        'metadata': 32
    },
    output_dim=256,
    strategy='attention'  # smart weighting
)

# Fuse into single representation
item_embeddings = fusion({
    'text': text_embeddings,
    'image': image_embeddings,
    'metadata': metadata
})

print(f"Fused embeddings shape: {item_embeddings.shape}")
# Use these embeddings in Two-Tower or other models
```

## Example 4: Complete Pipeline

Perfect for: Production systems, multi-stage ranking

```python
from corerec.pipelines.recommendation_pipeline import (
    RecommendationPipeline,
    RetrievalStage,
    RankingStage,
    RerankingStage,
    DiversityRule
)
from corerec.engines import TwoTower, DCN

# Assume models are trained...
retriever = TwoTower(...)
ranker = DCN(...)

# Build pipeline
pipeline = RecommendationPipeline(config={})

pipeline.add_stage(RetrievalStage(
    model=retriever,
    index=vector_index,
    config={'num_candidates': 1000}
))

pipeline.add_stage(RankingStage(
    model=ranker,
    config={'top_k': 100}
))

pipeline.add_stage(RerankingStage(
    rules=[DiversityRule(max_similar=3)],
    config={}
))

# Get recommendations
user_data = {'user_id': 'user_1', 'features': user_vector}
recs = pipeline.recommend(user_data, item_pool=None, top_k=10)
```

## Key Concepts

### Embeddings

Everything becomes a vector:
- User → 128D vector
- Item → 128D vector
- Similarity = dot product

### Two-Tower Architecture

```
User Features → User Tower → User Embedding (128D)
                                   ↓
                             Dot Product → Score
                                   ↑
Item Features → Item Tower → Item Embedding (128D)
```

Advantages:
- Pre-compute item embeddings
- Fast similarity search with FAISS
- Scales to millions of items

### Pipeline Stages

1. **Retrieval**: Fast, broad (millions → thousands)
2. **Ranking**: Slow, precise (thousands → hundreds)
3. **Reranking**: Rules, diversity (hundreds → tens)

Each stage filters progressively.

## Common Patterns

### Pattern 1: Collaborative Filtering with Deep Learning

Replace matrix factorization with Two-Tower:

```python
# Old way
from corerec.cf_engine import MatrixFactorization
model = MatrixFactorization(k=50)

# New way (better)
from corerec.engines import TwoTower
model = TwoTower(embedding_dim=128)
```

### Pattern 2: Content + Collaborative

Use multi-modal fusion:

```python
# Fuse item content features
item_embs = fusion({
    'title': title_embs,
    'description': desc_embs,
    'image': img_embs
})

# Use in Two-Tower for collaborative signal
model = TwoTower(...)
model.fit(..., item_features=item_embs)
```

### Pattern 3: Session-Based Recommendations

Use sequential models:

```python
model = BERT4Rec(...)
model.fit(user_sequences)
next_item = model.recommend(user_id)
```

## Performance Tips

1. **Use FAISS for large catalogs**: 10x-100x faster than brute force
2. **Cache embeddings**: Pre-compute item vectors offline
3. **Batch operations**: Score multiple users/items at once
4. **Progressive filtering**: Use pipeline stages
5. **Monitor latency**: Set SLA for each stage

## Troubleshooting

### Out of Memory?
- Reduce `batch_size`
- Lower `embedding_dim`
- Use gradient checkpointing

### Slow training?
- Use GPU (`device='cuda'`)
- Reduce `num_epochs`
- Subsample data for prototyping

### Poor quality?
- Try `loss_type="infonce"` (contrastive learning)
- Increase `embedding_dim`
- Add more features (multi-modal)

## Next Steps

1. Read [MODERN_RECSYS_GUIDE.md](MODERN_RECSYS_GUIDE.md) for theory
2. Study [examples/modern_pipeline_example.py](examples/modern_pipeline_example.py)
3. Try with your own data
4. Deploy with FAISS for production scale

## Help

- GitHub: https://github.com/vishesh9131/CoreRec
- Issues: https://github.com/vishesh9131/CoreRec/issues
- Docs: See `/docs` folder

---

**That's it!** You're now ready to build modern recommendation systems with CoreRec.

