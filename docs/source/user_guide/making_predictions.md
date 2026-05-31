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

For multi-stage production serving (retrieval → ranking → reranking), wire retrievers before calling `recommend()`:

```python
from corerec.engines.collaborative import SAR
from corerec.retrieval import CollaborativeRetriever, PopularityRetriever
from corerec.ranking import PointwiseRanker
from corerec.pipelines import RecommendationPipeline, PipelineConfig
import pandas as pd

train_df = pd.DataFrame({
    "userID": [0, 0, 1, 1],
    "itemID": [10, 11, 10, 12],
    "rating": [5, 4, 3, 5],
})

sar = SAR()
sar.fit(train_df)

pipeline = RecommendationPipeline(
    config=PipelineConfig(retrieval_k=50, ranking_k=20, final_k=10)
)
pipeline.add_retriever(CollaborativeRetriever(model=sar, name="collab"), weight=1.0)
pipeline.add_retriever(
    PopularityRetriever(name="pop").fit(
        item_ids=[10, 11, 12], interaction_counts=[2, 2, 1]
    ),
    weight=0.5,
)
pipeline.set_ranker(PointwiseRanker(name="ranker").fit())

result = pipeline.recommend(query=0, top_k=10)
print(f"Items: {result.items}")
print(f"Scores: {result.scores}")
print(f"Total time: {result.total_ms:.1f}ms")
```

For a full walkthrough, see the [Pipeline Tutorial](../tutorials/pipeline_tutorial.md).
