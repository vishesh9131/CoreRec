# Tutorial Index

Tutorials for the models CoreRec ships.

## Getting Started

Before diving into specific models, we recommend:
1. Read the [Installation Guide](../installation.md)
2. Follow the [QuickStart](../quickstart.md)
3. Understand [Core Concepts](../concepts.md)

## Production Models (Tested & Stable)

These models are **production-ready** — fully tested, CI-enforced, and implement the complete `BaseRecommender` interface. Start here.

### Core Engine Models

```{toctree}
---
maxdepth: 1
---
dcn_tutorial
deepfm_tutorial
sasrec_tutorial
two_tower_tutorial
```

### Collaborative Filtering Models

```{toctree}
---
maxdepth: 1
---
sar_tutorial
lightgcn_tutorial
```

### Content-Based Models

```{toctree}
---
maxdepth: 1
---
tfidf_tutorial
```

## Removed Models

CoreRec 0.7.0 removed the experimental sandbox. Its 51 tutorials are replaced
by one page listing each removed model and the closest one that ships today.

```{toctree}
---
maxdepth: 1
---
removed_models
```

## Pipeline & System Tutorials

End-to-end system tutorials:

```{toctree}
---
maxdepth: 1
---
pipeline_tutorial
```

## Tutorial Structure

### Production Model Tutorials
Full working examples with tested code you can copy-paste and run.


## Learning Path

### Beginners
1. Start with [DCN Tutorial](dcn_tutorial.md) (Production)
2. Explore [SAR Tutorial](sar_tutorial.md) (Production)

### Intermediate
1. Deep dive into [DeepFM](deepfm_tutorial.md) (Production)
2. Learn [Graph Methods with LightGCN](lightgcn_tutorial.md) (Production)

### Advanced
1. See [Removed Models](removed_models.md) if you are coming from 0.6
2. Deploy to [Production](../examples/production_deployment.md)
