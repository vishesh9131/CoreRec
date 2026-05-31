# Content-Based Models

Recommendations from item/user text and metadata rather than (or in addition to) collaborative signals.

## Production models (CI-tested)

### TFIDFRecommender

TF-IDF similarity over item text documents.

| | |
|---|---|
| **Import** | `from corerec.engines.content_based import TFIDFRecommender` |
| **Tutorial** | [TF-IDF Tutorial](../tutorials/tfidf_tutorial.md) |
| **Fit API** | Item IDs + document dict or corpus |

```python
from corerec.engines.content_based import TFIDFRecommender

docs = {
    101: "wireless bluetooth headphones noise cancelling",
    102: "running shoes lightweight marathon",
    103: "python machine learning tutorial",
}
model = TFIDFRecommender()
model.fit(items=list(docs.keys()), docs=docs)

# Similar items by content
similar = model.recommend(item_id=101, top_k=5)
```

**Use cases:** article/product catalogs, cold-start retrieval, hybrid pipelines (TF-IDF retrieval → ranker).

## Sandbox models (experimental)

| Model | Tutorial |
|-------|----------|
| **MIND-Content** | [MIND Content](../tutorials/mind_content_tutorial.md) |

Import path is documented in the tutorial (`corerec.sandbox.*`).

## When to use

- Rich text/metadata per item
- New items with no interaction history
- Candidate generation stage before a ranker

## See also

- [Retrieval API](../api/retrieval.md) (semantic / popularity retrievers)
- [Model index](models_index.md)
