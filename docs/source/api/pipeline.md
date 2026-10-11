# Pipeline System

The pipeline module provides multi-stage recommendation orchestration following the industry-standard Retrieval → Ranking → Reranking pattern.

## Quick Start

```python
import numpy as np
from corerec.engines import ALS
from corerec.pipelines import RecommendationPipeline, PipelineConfig
from corerec.ranking import PointwiseRanker
from corerec.reranking import DiversityReranker
from corerec.retrieval import CollaborativeRetriever

rng = np.random.default_rng(0)
users = rng.integers(0, 200, 5000).tolist()
items = rng.integers(0, 500, 5000).tolist()
als = ALS(factors=32, iterations=5).fit(users, items)

pipeline = RecommendationPipeline(
    config=PipelineConfig(retrieval_k=200, ranking_k=50, final_k=10)
)
pipeline.add_retriever(CollaborativeRetriever(model=als), weight=1.0)
pipeline.set_ranker(PointwiseRanker(score_fn=lambda f: f.get("retrieval_score", 0)).fit())
pipeline.add_reranker(DiversityReranker(lambda_=0.7))

# query is what the retrievers take: a user id for CollaborativeRetriever
result = pipeline.recommend(query=users[0], top_k=10)
print(result.items, result.total_ms)
```

The [pipeline tutorial](../tutorials/pipeline_tutorial.md) builds a fuller one
(several retrievers, business rules, a YAML config).

## Orchestrator

```{eval-rst}
.. automodule:: corerec.pipelines.orchestrator
   :members:
   :show-inheritance:
```

## Stage Abstractions

```{eval-rst}
.. automodule:: corerec.pipelines.recommendation_pipeline
   :members:
   :show-inheritance:
```

## Configuration

```{eval-rst}
.. automodule:: corerec.pipelines.config
   :members:
```

## Data Pipeline

```{eval-rst}
.. automodule:: corerec.pipelines.data_pipeline
   :members:
   :show-inheritance:
```
