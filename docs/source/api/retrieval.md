# Retrieval

Stage 1 of the recommendation pipeline. Retrievers generate candidate items from large catalogs efficiently, prioritizing recall over precision.

## Quick Start

```python
import numpy as np
from corerec.retrieval import SemanticRetriever

item_ids = ["a", "b", "c", "d"]
item_vectors = np.random.default_rng(0).normal(size=(4, 384)).astype("float32")

retriever = SemanticRetriever(embedding_dim=384)
retriever.fit(item_ids, item_embeddings=item_vectors)

# the query is an embedding, or text when the retriever has an encoder
result = retriever.retrieve(item_vectors[0], top_k=2)
print(result.item_ids(), result.scores())
```

With `pip install sentence-transformers`, pass texts instead of vectors:
`SemanticRetriever(encoder="all-MiniLM-L6-v2").fit(item_ids, item_texts=texts)`,
then `retrieve("wireless headphones")`. To retrieve for a user, wrap a fitted
model: `CollaborativeRetriever(model=als).retrieve(user_id, top_k=100)`.

## API Reference

```{eval-rst}
.. automodule:: corerec.retrieval
   :members:
   :show-inheritance:
   :no-index:
```

## Base Classes

```{eval-rst}
.. automodule:: corerec.retrieval.base
   :members:
   :show-inheritance:
```

## Implementations

### Collaborative Retriever

```{eval-rst}
.. automodule:: corerec.retrieval.collaborative
   :members:
   :show-inheritance:
```

### Semantic Retriever

```{eval-rst}
.. automodule:: corerec.retrieval.semantic
   :members:
   :show-inheritance:
```

### Popularity Retriever

```{eval-rst}
.. automodule:: corerec.retrieval.popularity
   :members:
   :show-inheritance:
```

### Ensemble Retriever

```{eval-rst}
.. automodule:: corerec.retrieval.ensemble
   :members:
   :show-inheritance:
```

### Vector Index

```{eval-rst}
.. automodule:: corerec.retrieval.index
   :members:
   :show-inheritance:
```
