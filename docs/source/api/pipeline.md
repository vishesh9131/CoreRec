# Pipeline System

The pipeline module provides multi-stage recommendation orchestration following the industry-standard Retrieval → Ranking → Reranking pattern.

## Quick Start

```python
from corerec.pipelines import RecommendationPipeline, PipelineConfig

pipeline = RecommendationPipeline(
    config=PipelineConfig(retrieval_k=200, ranking_k=50, final_k=10)
)
pipeline.add_retriever(my_retriever, weight=1.0)
pipeline.set_ranker(my_ranker)
pipeline.add_reranker(my_reranker)

result = pipeline.recommend(user_id=123, top_k=10)
```

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
