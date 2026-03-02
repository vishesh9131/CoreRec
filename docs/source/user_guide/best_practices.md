# Best Practices

## Choosing a Model

| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| Quick baseline | SAR | Fast, no deep learning needed |
| Large catalog retrieval | TwoTower | Efficient dual-encoder architecture |
| User interaction graph | LightGCN | Captures graph connectivity |
| Sequential behavior | SASRec, BERT4Rec | Temporal pattern modeling |
| Feature-rich data | DeepFM, DCN | Automatic feature interactions |
| Multi-modal content | Multimodal Fusion | Combines text, image, audio |

## Production Pipeline Pattern

For production systems, use the three-stage pipeline:

```python
from corerec.pipelines import RecommendationPipeline, PipelineConfig
from corerec.retrieval import CollaborativeRetriever, PopularityRetriever
from corerec.ranking import PointwiseRanker
from corerec.reranking import DiversityReranker, BusinessRulesReranker

# 1. Configure pipeline
pipeline = RecommendationPipeline(
    config=PipelineConfig(retrieval_k=500, ranking_k=100, final_k=10)
)

# 2. Add retrievers (recall-focused, fast)
pipeline.add_retriever(collab_retriever, weight=1.0)
pipeline.add_retriever(pop_retriever, weight=0.3)

# 3. Set ranker (precision-focused, complex)
pipeline.set_ranker(ranker)

# 4. Add rerankers (business logic)
pipeline.add_reranker(DiversityReranker(lambda_=0.7))
pipeline.add_reranker(business_rules)

# 5. Serve
result = pipeline.recommend(query=user_id, top_k=10)
```

## Error Handling

CoreRec provides specific exceptions:

```python
from corerec.api.exceptions import (
    ModelNotFittedError,    # Model not trained yet
    InvalidParameterError,  # Bad parameter value
    InvalidDataError,       # Malformed input data
)

try:
    recs = model.recommend(user_id=1, top_k=10)
except ModelNotFittedError:
    model.fit(train_data)
    recs = model.recommend(user_id=1, top_k=10)
```

## Model Serving

Deploy models as REST APIs:

```python
from corerec.serving import ModelServer

server = ModelServer(model=my_model, port=8000)
server.start()

# POST /predict     -> single prediction
# POST /recommend   -> recommendations
# GET  /health      -> health check
```

## Demo Frontends

Quickly demo your model with themed UIs:

```python
from corerec.imshow import connector

demo = connector(my_recommender, frontend="spotify")
demo.run()  # Opens browser with Spotify-themed UI
```
