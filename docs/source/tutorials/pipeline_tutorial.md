# Multi-Stage Pipeline Tutorial

This tutorial walks through building a production-style recommendation pipeline using CoreRec's three-stage architecture: Retrieval → Ranking → Reranking.

## Overview

Real-world recommendation systems don't just run one model. They use a pipeline:

1. **Retrieval** (Stage 1): Fast candidate generation from millions of items → ~500 candidates
2. **Ranking** (Stage 2): Precise scoring of candidates → ~50 items
3. **Reranking** (Stage 3): Business rules, diversity, fairness → final 10 items

## Step 1: Setup

```python
import numpy as np
import pandas as pd

# Generate sample interaction data
np.random.seed(42)
interactions = []
for user_id in range(1000):
    n_items = np.random.randint(10, 50)
    items = np.random.choice(5000, n_items, replace=False)
    for item_id in items:
        interactions.append({
            'user_id': user_id,
            'item_id': item_id,
            'rating': np.random.randint(1, 6),
        })

train_df = pd.DataFrame(interactions)
```

## Step 2: Create Retrievers

```python
from corerec.engines.collaborative import SAR
from corerec.retrieval import CollaborativeRetriever, PopularityRetriever

# Collaborative filtering retriever
sar = SAR(col_user='user_id', col_item='item_id', col_rating='rating')
sar.fit(train_df)
collab_retriever = CollaborativeRetriever(model=sar, name="collab")

# Popularity-based retriever for cold-start coverage
item_counts = train_df.groupby('item_id').size()
pop_retriever = PopularityRetriever(name="popularity")
pop_retriever.fit(
    item_ids=list(item_counts.index),
    interaction_counts=list(item_counts.values),
)
```

## Step 3: Create Ranker

```python
from corerec.ranking import PointwiseRanker

def score_fn(features):
    return 0.7 * features.get('retrieval_score', 0) + 0.3 * features.get('popularity', 0)

ranker = PointwiseRanker(score_fn=score_fn, name="blended")
ranker.fit()
```

## Step 4: Create Rerankers

```python
from corerec.reranking import DiversityReranker, BusinessRulesReranker

diversity = DiversityReranker(lambda_=0.7)

business = BusinessRulesReranker(name="business")
business.add_boost(item_id=42, multiplier=2.0)
business.add_blocklist([999, 998])
```

## Step 5: Build Pipeline

```python
from corerec.pipelines import RecommendationPipeline, PipelineConfig

pipeline = RecommendationPipeline(
    config=PipelineConfig(
        retrieval_k=200,
        ranking_k=50,
        final_k=10,
        fusion_strategy='rrf',
    ),
    name="tutorial_pipeline",
)

pipeline.add_retriever(collab_retriever, weight=1.0)
pipeline.add_retriever(pop_retriever, weight=0.3)
pipeline.set_ranker(ranker)
pipeline.add_reranker(diversity)
pipeline.add_reranker(business)
```

## Step 6: Generate Recommendations

```python
result = pipeline.recommend(query=5, top_k=10)

print(f"Retrieved: {result.retrieval_candidates} candidates")
print(f"Ranked: {result.ranking_candidates} candidates")
print(f"Final: {result.final_candidates} items")
print(f"Total time: {result.total_ms:.1f}ms")

for item_id, score in result:
    print(f"  Item {item_id}: {score:.4f}")
```

## Step 7: Configuration from YAML

```python
from corerec.pipelines import load_pipeline_config, build_pipeline_from_config

# Load from YAML file
config = load_pipeline_config('pipeline.yaml')
pipeline = build_pipeline_from_config(config)
```

Example `pipeline.yaml`:

```yaml
pipeline:
  name: production_v1
  retrieval:
    k: 500
    fusion: rrf
    sources:
      - type: collaborative
        weight: 1.0
      - type: popularity
        weight: 0.3
  ranking:
    k: 100
    type: pointwise
  reranking:
    - type: diversity
      lambda: 0.7
    - type: business
      boost:
        - item: 42
          multiplier: 2.0
```
