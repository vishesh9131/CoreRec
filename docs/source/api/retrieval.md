# Retrieval

Stage 1 of the recommendation pipeline. Retrievers generate candidate items from large catalogs efficiently, prioritizing recall over precision.

## Quick Start

```python
from corerec.retrieval import SemanticRetriever, CollaborativeRetriever, EnsembleRetriever

retriever = SemanticRetriever(encoder="all-MiniLM-L6-v2")
retriever.fit(items=item_catalog)
candidates = retriever.retrieve(query=user_id, top_k=100)
```

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
