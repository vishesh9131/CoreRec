# Modern RecSys Architecture in CoreRec

## The Paradigm Shift

Recommendation systems have evolved significantly. The old matrix factorization approach has been replaced by **embedding-based deep learning** with a multi-stage pipeline.

### Old vs New

| Aspect | Traditional (Pre-2017) | Modern (2020+) |
|--------|----------------------|----------------|
| **Data Format** | Sparse matrices (users × items) | Dense vectors (embeddings) |
| **Features** | Hard to mix IDs with content | Native multi-modal support |
| **Scale** | Struggles beyond 100K items | Handles millions via ANN search |
| **Temporal** | Ignores order | Sequential models (transformers) |
| **Architecture** | Single model | Multi-stage pipeline |

---

## The Modern Pipeline

Production recsys use a three-stage funnel:

```
1. RETRIEVAL    →  2. RANKING    →  3. RERANKING
(millions)         (thousands)        (tens)
   ↓                   ↓                 ↓
Fast ANN           Heavy DL          Business logic
Two-Tower          DLRM/DCN         Diversity/Fresh
```

### Stage 1: Retrieval (Candidate Generation)

**Goal**: Narrow down from millions to ~1000 candidates quickly.

**Technology**: 
- Two-Tower models
- Vector databases (FAISS, Milvus)
- Approximate Nearest Neighbors

**CoreRec Implementation**:

```python
from corerec.engines import TwoTower
from corerec.retrieval.vector_store import create_index

# Train two-tower model
model = TwoTower(
    user_input_dim=64,
    item_input_dim=128,
    embedding_dim=256,
    strategy="infonce"  # contrastive learning
)

model.fit(user_ids, item_ids, interactions)

# Build vector index for fast retrieval
item_embeddings = model.get_item_embeddings()
index = create_index("faiss", dim=256, index_type="hnsw")
index.add(item_embeddings, item_ids)

# Fast retrieval at inference
user_emb = model.get_user_embedding(user_id)
candidates, scores = index.search(user_emb, k=1000)
```

### Stage 2: Ranking (Scoring)

**Goal**: Score candidates precisely using complex features.

**Technology**:
- DCN (Deep & Cross Network)
- DeepFM
- DLRM (Deep Learning Recommendation Model)

**CoreRec Implementation**:

```python
from corerec.engines import DCN

# Heavy model with feature crossing
ranker = DCN(
    embedding_dim=128,
    cross_layers=3,
    deep_layers=[512, 256, 128]
)

ranker.fit(
    user_features=user_feats,
    item_features=item_feats,
    interactions=labels
)

# Score the candidates
top_100 = ranker.rank(candidates, user_id, top_k=100)
```

### Stage 3: Reranking (Final Polish)

**Goal**: Apply business rules, diversity, freshness.

**Technology**:
- Rule-based systems
- Reinforcement learning
- LLM-based reranking

**CoreRec Implementation**:

```python
from corerec.pipelines.recommendation_pipeline import (
    RecommendationPipeline, 
    DiversityRule, 
    FreshnessRule
)

pipeline = RecommendationPipeline(config={})
pipeline.add_stage(retrieval_stage)
pipeline.add_stage(ranking_stage)
pipeline.add_stage(reranking_stage)

final_recs = pipeline.recommend(
    user_data={'user_id': 123, 'features': user_vec},
    item_pool=all_items,
    top_k=10
)
```

---

## Key Architectures

### 1. Two-Tower Model

The industry standard for retrieval. Used by YouTube, Netflix, Pinterest.

**How it works**:
- Separate neural networks encode users and items
- Match via dot product in embedding space
- Pre-compute item embeddings for speed

```python
from corerec.engines import TwoTower

model = TwoTower(
    user_input_dim=64,
    item_input_dim=128,
    embedding_dim=256,
    hidden_dims=[512, 256],
    loss_type="infonce"  # contrastive learning
)
```

**Advantages**:
- Blazing fast (ANN search)
- Scales to millions of items
- Easy to update (just recompute embeddings)

### 2. Sequential Models (Transformers)

Capture temporal patterns in user behavior.

**Models**:
- **SASRec**: Causal transformer (predict next item)
- **BERT4Rec**: Bidirectional transformer (mask & predict)

```python
from corerec.engines import BERT4Rec

model = BERT4Rec(
    hidden_dim=256,
    num_layers=4,
    num_heads=8,
    max_len=200
)

model.fit(user_ids, item_ids, interaction_matrix)
```

**Use cases**:
- Session-based recommendations
- "Next best action"
- Time-sensitive content (news, videos)

### 3. Graph Neural Networks

Leverage user-item-user connections.

```python
from corerec.engines import GNNRec

model = GNNRec(
    num_layers=3,
    embedding_dim=128
)
```

**Use cases**:
- Social recommendations
- Related items
- Collaborative filtering with graph structure

---

## Multi-Modal Fusion

Modern items have multiple features: text, images, metadata. We need to combine them smartly.

```python
from corerec.multimodal.fusion_strategies import MultiModalFusion

# Item = title (text) + thumbnail (image) + metadata
item_fusion = MultiModalFusion(
    modality_dims={
        'text': 768,      # BERT embeddings
        'image': 2048,    # ResNet features
        'metadata': 32    # category, price, etc
    },
    output_dim=256,
    strategy='attention'  # attention-based fusion
)

item_embedding = item_fusion({
    'text': text_emb,
    'image': img_emb,
    'metadata': meta_features
})
```

**Fusion strategies**:
- `concat`: Simple concatenation
- `weighted`: Learned weights per modality
- `attention`: Dynamic attention (best for most cases)
- `gated`: VQA-style gating

---

## Vector Databases

Essential for fast retrieval at scale.

```python
from corerec.retrieval.vector_store import create_index

# Choose backend based on scale
index = create_index(
    backend="faiss",     # or "annoy", "numpy"
    dim=256,
    metric="cosine",
    index_type="hnsw"   # graph-based ANN
)

# Add item embeddings
index.add(embeddings, item_ids)
index.save("item_index.faiss")

# Fast search
scores, items = index.search(query_vec, k=1000)
```

**Backends**:
- **numpy**: < 100K items, no deps
- **FAISS**: Production scale, Facebook's library
- **Annoy**: Medium scale, Spotify's library

---

## Complete Example

```python
from corerec.engines import TwoTower, DCN
from corerec.retrieval.vector_store import create_index
from corerec.pipelines.recommendation_pipeline import (
    RecommendationPipeline,
    RetrievalStage,
    RankingStage,
    RerankingStage,
    DiversityRule
)

# Step 1: Train retrieval model
retriever = TwoTower(embedding_dim=256, loss_type="infonce")
retriever.fit(user_ids, item_ids, interactions)

# Build vector index
item_embs = retriever.get_item_embeddings()
vec_index = create_index("faiss", dim=256, index_type="hnsw")
vec_index.add(item_embs, item_ids)

# Step 2: Train ranking model
ranker = DCN(embedding_dim=128, cross_layers=3)
ranker.fit(user_features, item_features, labels)

# Step 3: Build pipeline
pipeline = RecommendationPipeline(config={})

pipeline.add_stage(RetrievalStage(
    model=retriever,
    index=vec_index,
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
recs = pipeline.recommend(
    user_data={'user_id': 123, 'features': user_vec},
    item_pool=None,  # uses indexed items
    top_k=10
)
```

---

## Migration Guide

### If you're using old collaborative filtering:

**Before**:
```python
from corerec.cf_engine import MatrixFactorization
model = MatrixFactorization(k=50)
model.fit(interaction_matrix)
```

**After**:
```python
from corerec.engines import TwoTower
model = TwoTower(embedding_dim=128)
model.fit(user_ids, item_ids, interaction_matrix)
```

### If you're using content-based:

**Before**:
```python
from corerec.engines.content import TFIDFRecommender
model = TFIDFRecommender()
model.fit(item_texts)
```

**After** (multi-modal):
```python
from corerec.multimodal.fusion_strategies import MultiModalFusion
fusion = MultiModalFusion(
    modality_dims={'text': 768, 'image': 2048},
    output_dim=256,
    strategy='attention'
)
```

---

## Performance Tips

1. **Use ANN search**: Don't brute-force similarity at scale
2. **Cache embeddings**: Pre-compute item vectors
3. **Batch operations**: Score candidates in batches
4. **Progressive filtering**: Start broad, refine gradually
5. **Monitor latency**: Each stage should have SLA

---

## Further Reading

- [Two Towers paper](https://dl.acm.org/doi/10.1145/3298689.3346996) (YouTube)
- [BERT4Rec](https://arxiv.org/abs/1904.06690)
- [DLRM paper](https://arxiv.org/abs/1906.00091) (Facebook)
- [Vector databases comparison](https://arxiv.org/abs/2101.12631)

---

## Summary

Modern RecSys = **Embeddings** + **Multi-stage Pipeline** + **Vector Search**

CoreRec now supports all of this out of the box. The old matrix factorization APIs still work but use the new models for production systems.

