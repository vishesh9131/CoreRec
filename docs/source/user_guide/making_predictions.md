# Making Predictions

## Single Predictions

Predict a specific user-item pair:

```python
score = model.predict(user_id=1, item_id=100)
print(f"Predicted rating: {score:.2f}")
```

## Recommendations

Get top-K items for a user:

```python
recommendations = model.recommend(user_id=1, top_k=10)
# Returns: list of item IDs ordered by predicted relevance
```

Exclude already-seen items:

```python
seen_items = [101, 102, 103]
recommendations = model.recommend(
    user_id=1, top_k=10, exclude_items=seen_items
)
```

## Batch Operations

For efficiency with multiple users:

```python
# Batch predictions
pairs = [(1, 100), (1, 200), (2, 100)]
scores = model.batch_predict(pairs)

# Batch recommendations
user_ids = [1, 2, 3, 4, 5]
all_recs = model.batch_recommend(user_ids, top_k=10)
# Returns: dict mapping user_id -> list of item_ids
```

## Pipeline-Based Recommendations

For production systems, use the multi-stage pipeline:

```python
from corerec.pipelines import RecommendationPipeline, PipelineConfig

pipeline = RecommendationPipeline(
    config=PipelineConfig(retrieval_k=200, ranking_k=50, final_k=10)
)

result = pipeline.recommend(query=user_id, top_k=10)
print(f"Items: {result.items}")
print(f"Scores: {result.scores}")
print(f"Total time: {result.total_ms:.1f}ms")
```
