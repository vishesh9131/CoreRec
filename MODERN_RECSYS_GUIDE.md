# Modern Recommendation Systems Guide

A practical guide to building production-grade recommendation systems with CoreRec — Two-Tower retrieval, vector stores, multi-stage pipelines, and multi-modal fusion.

---

## Overview

Modern recommendation systems at scale (Netflix, YouTube, Amazon) follow a common architecture:

1. **Retrieval** — quickly narrow millions of items to hundreds of candidates (ANN, Two-Tower)
2. **Ranking** — score candidates with a complex model (DCN, DeepFM)
3. **Reranking** — apply business rules, diversity, fairness

CoreRec implements this pattern end-to-end. This guide walks through each piece.

---

## 1. Two-Tower Retrieval

The Two-Tower model uses separate encoders for users and items. Embeddings live in a shared space; similarity is computed via dot product. This enables **fast retrieval** over large catalogs using approximate nearest neighbor (ANN) search.

### Why Two-Tower?

- **Scalability**: Precompute item embeddings once; serve user embeddings at query time
- **Latency**: ANN search (FAISS, Annoy) returns top-K in milliseconds
- **Industry standard**: Used by YouTube, Netflix, Pinterest, and others

### Basic Usage

```python
from corerec.engines import TwoTower
from cr_learn import ml_1m

data = ml_1m.load()
ratings = data['ratings']

model = TwoTower(
    user_input_dim=64,
    item_input_dim=128,
    embedding_dim=256,
    epochs=10,
)

model.fit(
    user_ids=ratings['user_id'].values,
    item_ids=ratings['movie_id'].values,
    interactions=ratings['rating'].values,
)

# Get item embeddings for indexing
item_embs = model.get_item_embeddings()
```

### With Optional User/Item Features

```python
model = TwoTower(
    user_input_dim=64,
    item_input_dim=128,
    embedding_dim=256,
    user_features=user_feature_matrix,  # optional
    item_features=item_feature_matrix,  # optional
)
model.fit(user_ids, item_ids, interactions)
```

---

## 2. Vector Store for Fast Retrieval

For catalogs with 100K+ items, brute-force scoring is too slow. Use a vector index (FAISS, Annoy, or NumPy fallback) to retrieve candidates in milliseconds.

### Create an Index

```python
from corerec.retrieval.vector_store import create_index

# FAISS (recommended for large catalogs)
index = create_index("faiss", dim=256, metric="cosine")

# Add item embeddings
item_embeddings = model.get_item_embeddings()  # shape: (n_items, 256)
item_ids = list(range(n_items))
index.add(item_embeddings, ids=item_ids)

# Search
user_emb = model.get_user_embedding(user_id=42)
scores, ids = index.search(user_emb, k=100)
```

### Index Backends

| Backend | Use Case | Install |
|---------|----------|---------|
| `numpy` | Small catalogs (<50K), exact search | built-in |
| `faiss` | Large catalogs, production | `pip install faiss-cpu` |
| `annoy` | Medium catalogs, memory-efficient | `pip install annoy` |

### FAISS Index Types

```python
# Exact search (flat)
index = create_index("faiss", dim=256, index_type="flat")

# Approximate search (IVF) — faster for millions
index = create_index("faiss", dim=256, index_type="ivf")

# HNSW — good quality/speed tradeoff
index = create_index("faiss", dim=256, index_type="hnsw")
```

### Save and Load Index

```python
index.save("item_index.faiss")
# Later:
index.load("item_index.faiss")
```

---

## 3. End-to-End Two-Tower + Vector Store

```python
from corerec.engines import TwoTower
from corerec.retrieval.vector_store import create_index
from cr_learn import ml_1m

# 1. Load data
data = ml_1m.load()
ratings = data['ratings']

# 2. Train Two-Tower
model = TwoTower(
    user_input_dim=64,
    item_input_dim=128,
    embedding_dim=256,
    epochs=10,
)
model.fit(
    user_ids=ratings['user_id'].values,
    item_ids=ratings['movie_id'].values,
    interactions=ratings['rating'].values,
)

# 3. Build vector index
item_embs = model.get_item_embeddings()
item_ids = [model.reverse_item_map[i] for i in range(len(item_embs))]
index = create_index("faiss", dim=256)
index.add(item_embs, ids=item_ids)

# 4. Retrieve top candidates
user_id = 42
candidates = model.recommend(user_id=user_id, top_k=100)
# Or use index directly for custom logic:
# user_emb = model.get_user_embedding(user_id)
# scores, ids = index.search(user_emb, k=100)
```

---

## 4. Multi-Stage Pipeline

Production systems chain Retrieval → Ranking → Reranking. CoreRec provides `RecommendationPipeline` for this.

### Quick Start

```python
from corerec.pipelines import RecommendationPipeline, PipelineConfig
from corerec.retrieval import CollaborativeRetriever, PopularityRetriever
from corerec.ranking import PointwiseRanker
from corerec.reranking import DiversityReranker

# Configure stages
pipeline = RecommendationPipeline(
    config=PipelineConfig(
        retrieval_k=500,   # retrieve 500 candidates
        ranking_k=100,     # rank to 100
        final_k=10,       # rerank to 10
    )
)

# Add retrievers (recall-focused)
pipeline.add_retriever(collab_retriever, weight=1.0)
pipeline.add_retriever(pop_retriever, weight=0.3)

# Set ranker (precision-focused)
pipeline.set_ranker(PointwiseRanker(model=dcn_model))

# Add rerankers (business logic)
pipeline.add_reranker(DiversityReranker(lambda_=0.7))

# Serve
result = pipeline.recommend(user_id=123, top_k=10)
```

### Pipeline Stages

| Stage | Purpose | Typical K |
|-------|---------|-----------|
| **Retrieval** | Fast recall from large catalog | 200–1000 |
| **Ranking** | Complex model scoring | 50–200 |
| **Reranking** | Diversity, fairness, rules | 10–50 |

---

## 5. Multi-Modal Fusion

When items have text, images, and metadata, fuse embeddings from multiple modalities before scoring.

### MultiModalFusion

```python
from corerec.multimodal.fusion_strategies import MultiModalFusion

fusion = MultiModalFusion(
    modality_dims={'text': 768, 'image': 2048, 'meta': 32},
    output_dim=256,
    strategy='attention',  # or 'concat', 'weighted', 'gated'
)

# Fuse embeddings (dict of modality name -> tensor)
item_embedding = fusion({
    'text': text_emb,   # [batch, 768]
    'image': img_emb,   # [batch, 2048]
    'meta': meta_emb,   # [batch, 32]
})
# Result: [batch, 256]
```

### Using with RecommendationPipeline

Multi-modal item embeddings can be fed into a semantic retriever or used as features in a ranker.

---

## 6. Sequential Models (BERT4Rec)

For session-based or sequential behavior, use BERT4Rec (bidirectional transformer).

```python
from corerec.engines.content_based import BERT4Rec

model = BERT4Rec(
    hidden_dim=256,
    num_layers=4,
    num_heads=4,
    max_len=50,
)

model.fit(user_ids, item_ids, interactions)
next_items = model.recommend(user_id=1, top_k=10)
```

---

## 7. Choosing the Right Architecture

| Scenario | Recommended Approach |
|----------|----------------------|
| Large catalog (>100K items) | Two-Tower + FAISS |
| Feature-rich data | DCN, DeepFM |
| Sequential / session data | SASRec, BERT4Rec |
| Graph / social data | GNNRec, LightGCN |
| Multi-modal (text + image) | MultiModalFusion + SemanticRetriever |
| Production at scale | Pipeline (Retrieval → Ranking → Reranking) |

---

## 8. Further Reading

- [API: Retrieval](https://corerec.online/docs/api/retrieval.html)
- [API: Pipeline](https://corerec.online/docs/api/pipeline.html)
- [API: Multimodal](https://corerec.online/docs/api/multimodal.html)
- [API: Embeddings](https://corerec.online/docs/api/embeddings.html)
- [User Guide: Best Practices](https://corerec.online/docs/user_guide/best_practices.html)
- [Towers (Two-Tower building blocks)](https://corerec.online/docs/core/towers/index.html)

---

*CoreRec — Production-grade recommendation systems from research to deployment.*
