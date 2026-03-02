# CoreRec

<div class="hero-banner">
<h1>CoreRec</h1>
<p class="tagline">Production-grade recommendation systems framework. 57+ models, unified API, multi-stage pipelines -- from research to deployment.</p>
<code class="install-cmd">pip install corerec</code>
</div>

<div class="stats-row">
<div class="stat"><span class="stat-num">57+</span><span class="stat-label">Models</span></div>
<div class="stat"><span class="stat-num">5</span><span class="stat-label">Categories</span></div>
<div class="stat"><span class="stat-num">3</span><span class="stat-label">Pipeline Stages</span></div>
<div class="stat"><span class="stat-num">100%</span><span class="stat-label">Type-Hinted</span></div>
</div>

<p class="landing-section-title">Get Started</p>

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} Installation
:link: installation
:link-type: doc

System requirements, pip install, optional dependencies, and verifying your setup.
:::

:::{grid-item-card} QuickStart Guide
:link: quickstart
:link-type: doc

Train your first model in under 5 minutes with a worked DCN example on MovieLens.
:::

:::{grid-item-card} Core Concepts
:link: concepts
:link-type: doc

Architecture overview -- engines, pipelines, retrieval/ranking/reranking stages.
:::

::::

<p class="landing-section-title">Capabilities</p>

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} Unified Model API
:link: api/base_recommender
:link-type: doc

Every model shares `fit()`, `predict()`, `recommend()`, `save()`, `load()`. Swap models without changing your code.
:::

:::{grid-item-card} Multi-Stage Pipeline
:link: api/pipeline
:link-type: doc

Retrieval, ranking, reranking -- compose retrieval sources, scoring models, and post-processing in a single pipeline.
:::

:::{grid-item-card} Retrieval Layer
:link: api/retrieval
:link-type: doc

Collaborative, semantic, and popularity retrievers with ensemble fusion (RRF, weighted, union).
:::

:::{grid-item-card} Ranking & Reranking
:link: api/ranking
:link-type: doc

Pointwise, pairwise, and feature-cross rankers. Diversity, fairness, and business-rule rerankers.
:::

:::{grid-item-card} Visualization (imshow)
:link: api/imshow
:link-type: doc

Plug any recommender into Spotify, YouTube, or Netflix-style frontends for interactive demos.
:::

:::{grid-item-card} Model Serving
:link: api/serving
:link-type: doc

Batch inference, model loading, and a lightweight model server for production deployment.
:::

::::

<p class="landing-section-title">Model Categories</p>

::::{grid} 1 2 3 5
:gutter: 3

:::{grid-item-card} Deep Learning
:link: models/deep_learning
:link-type: doc

**29 models** -- DCN, DeepFM, DIEN, DIN, DLRM, Wide&Deep, and more.
:::

:::{grid-item-card} Matrix Factorization
:link: models/matrix_factorization
:link-type: doc

**9 models** -- SVD, ALS, NMF, and advanced variants.
:::

:::{grid-item-card} Graph-Based
:link: models/graph_based
:link-type: doc

**6 models** -- GNNRec, LightGCN, graph neural networks.
:::

:::{grid-item-card} Sequential
:link: models/sequential
:link-type: doc

**6 models** -- SASRec, MIND, Caser, NextItNet.
:::

:::{grid-item-card} Bayesian
:link: models/bayesian
:link-type: doc

**3 models** -- BPR, BPRMF, VMF.
:::

::::

<p class="landing-section-title">Quick Example</p>

```python
from corerec.engines.dcn import DCN
import cr_learn

data = cr_learn.load_dataset('movielens-100k')

model = DCN(embedding_dim=64, epochs=20, verbose=True)
model.fit(user_ids=data.user_ids, item_ids=data.item_ids, ratings=data.ratings)

recs = model.recommend(user_id=1, top_k=10)
model.save('dcn_model.pkl')
```

<p class="landing-section-title">Learn More</p>

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc

Step-by-step tutorials for every model -- DCN, DeepFM, SASRec, LightGCN, and 50+ more.
:::

:::{grid-item-card} User Guide
:link: user_guide/data_preparation
:link-type: doc

Data preparation, training, predictions, persistence, and best practices.
:::

:::{grid-item-card} API Reference
:link: api/engines
:link-type: doc

Full API documentation for all modules -- engines, pipelines, retrieval, ranking, evaluation.
:::

::::

---

*{ref}`genindex`  --  {ref}`modindex`  --  {ref}`search`*

```{toctree}
---
hidden: true
maxdepth: 2
caption: Getting Started
---
installation
quickstart
concepts
```

```{toctree}
---
hidden: true
maxdepth: 2
caption: User Guide
---
user_guide/data_preparation
user_guide/model_training
user_guide/making_predictions
user_guide/model_persistence
user_guide/best_practices
```

```{toctree}
---
hidden: true
maxdepth: 2
caption: Model Documentation
---
models/index
models/models_index
```

```{toctree}
---
hidden: true
maxdepth: 2
caption: Tutorials & Learning
---
tutorials/index
```

```{toctree}
---
hidden: true
maxdepth: 2
caption: API Reference
---
api/base_recommender
api/exceptions
api/mixins
api/engines
api/pipeline
api/retrieval
api/ranking
api/reranking
api/multimodal
api/embeddings
api/explanation
api/evaluation
api/serving
api/imshow
api/constants
```

```{toctree}
---
hidden: true
maxdepth: 2
caption: Examples
---
examples/basic_usage
examples/advanced_usage
examples/production_deployment
```

```{toctree}
---
hidden: true
maxdepth: 1
caption: Development
---
contributing
changelog
license
```
