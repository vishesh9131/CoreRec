# Engines

CoreRec provides multiple recommendation engine categories.

## Deep Learning Models

Top-level models available via `from corerec.engines import <Model>`.

### DCN (Deep & Cross Network)

```{eval-rst}
.. automodule:: corerec.engines.dcn
   :members:
   :show-inheritance:
```

### DeepFM

```{eval-rst}
.. automodule:: corerec.engines.deepfm
   :members:
   :show-inheritance:
```

### GNNRec

```{eval-rst}
.. automodule:: corerec.engines.gnnrec
   :members:
   :show-inheritance:
```

### TwoTower

```{eval-rst}
.. automodule:: corerec.engines.two_tower
   :members:
   :show-inheritance:
```

### BERT4Rec

```{eval-rst}
.. automodule:: corerec.engines.bert4rec
   :members:
   :show-inheritance:
```

### SASRec

```{eval-rst}
.. automodule:: corerec.engines.sasrec
   :members:
   :show-inheritance:
```

### MIND

```{eval-rst}
.. automodule:: corerec.engines.mind
   :members:
   :show-inheritance:
```

### NASRec

```{eval-rst}
.. automodule:: corerec.engines.nasrec
   :members:
   :show-inheritance:
```

## Collaborative Filtering

Available via `from corerec.engines.collaborative import <Model>` or `from corerec import collaborative`.

### SAR (Simple Algorithm for Recommendation)

```{eval-rst}
.. automodule:: corerec.engines.collaborative.sar
   :members:
   :show-inheritance:
```

## Content-Based Filtering

Available via `from corerec.engines.content_based import <Model>` or `from corerec import content_based`.

### TFIDFRecommender

```{eval-rst}
.. autoclass:: corerec.engines.content_based.TFIDFRecommender
   :members:
   :show-inheritance:
```
