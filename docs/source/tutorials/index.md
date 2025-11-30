# Tutorial Index

Comprehensive tutorials for all 57 CoreRec models with cr_learn examples.

## Getting Started

Before diving into specific models, we recommend:
1. Read the [Installation Guide](../installation.md)
2. Follow the [QuickStart](../quickstart.md)
3. Understand [Core Concepts](../concepts.md)

## Core Engine Models

These 6 models are production-ready and cover most use cases:

```{toctree}
---
maxdepth: 1
---
dcn_tutorial
deepfm_tutorial
gnnrec_tutorial
mind_tutorial
nasrec_tutorial
sasrec_tutorial
```

## Neural Network Models (29)

Deep learning models for complex pattern recognition:

```{toctree}
---
maxdepth: 1
---
afm_tutorial
autofi_tutorial
autoint_tutorial
bert4rec_tutorial
bst_tutorial
bivae_tutorial
caser_tutorial
dcn_base_tutorial
deepcrossing_tutorial
deepfm_base_tutorial
deeprec_tutorial
dien_tutorial
difm_tutorial
din_tutorial
dlrm_tutorial
ensfm_tutorial
escmm_tutorial
esmm_tutorial
fgcnn_tutorial
ffm_tutorial
fibinet_tutorial
flen_tutorial
fm_tutorial
gan_tutorial
gatenet_tutorial
gnn_base_tutorial
gru_cf_tutorial
ncf_tutorial
nfm_tutorial
nextitnet_tutorial
widedeep_tutorial
youtubednn_tutorial
pnn_tutorial
mmoe_tutorial
ple_tutorial
monolith_tutorial
tdm_tutorial
```

## Matrix Factorization Models (9)

Classic collaborative filtering approaches:

```{toctree}
---
maxdepth: 1
---
a2svd_tutorial
als_tutorial
fm_base_tutorial
matrixfactorization_tutorial
mf_base_tutorial
svd_tutorial
userbased_tutorial
```

## Graph-Based Models (6)

Models leveraging graph structure:

```{toctree}
---
maxdepth: 1
---
geoimc_tutorial
lightgcn_tutorial
lightgcn_base_tutorial
```

## Sequential Models (6)

Time-aware recommendation models:

```{toctree}
---
maxdepth: 1
---
rbm_tutorial
rlrmc_tutorial
sar_tutorial
slirec_tutorial
sum_tutorial
```

## Bayesian Models (3)

Probabilistic approaches:

```{toctree}
---
maxdepth: 1
---
bpr_tutorial
bprmf_tutorial
vmf_tutorial
```

## Content-Based Models (3)

Feature-based recommendations:

```{toctree}
---
maxdepth: 1
---
mind_content_tutorial
tfidf_tutorial
```

## Tutorial Structure

Each tutorial follows a consistent structure:

1. **Introduction** - What the model does
2. **How It Works** - Architecture and math
3. **Step-by-Step Tutorial** - Complete example with cr_learn
4. **Advanced Usage** - Customization and tuning
5. **Key Takeaways** - When to use, best practices
6. **Further Reading** - Papers and references

## Learning Path

### Beginners
1. Start with [DCN Tutorial](dcn_tutorial.md)
2. Try [Matrix Factorization](svd_tutorial.md)
3. Explore [Sequential Models](sasrec_tutorial.md)

### Intermediate
1. Deep dive into [DeepFM](deepfm_tutorial.md)
2. Learn [Graph Methods](gnnrec_tutorial.md)
3. Master [Multi-Interest](mind_tutorial.md)

### Advanced
1. Study [Neural Architecture Search](nasrec_tutorial.md)
2. Implement [Custom Models](../examples/advanced_usage.md)
3. Deploy to [Production](../examples/production_deployment.md)
