# Tutorial Index

Comprehensive tutorials for all CoreRec models.

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
gnnrec_tutorial
mind_tutorial
nasrec_tutorial
sasrec_tutorial
bert4rec_tutorial
```

### Collaborative Filtering Models

```{toctree}
---
maxdepth: 1
---
ncf_tutorial
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

---

## Sandbox Models (Experimental)

```{admonition} Sandbox Notice
:class: warning
The models below are **experimental**. They are included for research and learning purposes. Sandbox models may have incomplete implementations and are not covered by production CI tests. See [Model Tiers](../models/index.md#model-tiers) for details.
```

Each sandbox tutorial explains the model's architecture, mathematical foundations, use cases, and scaling considerations. Code examples are provided as **reference implementations** — they may require additional validation before production use.

### Neural Network Models (Sandbox)

```{toctree}
---
maxdepth: 1
---
afm_tutorial
autofi_tutorial
autoint_tutorial
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

### Matrix Factorization Models (Sandbox)

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

### Graph-Based Models (Sandbox)

```{toctree}
---
maxdepth: 1
---
geoimc_tutorial
lightgcn_base_tutorial
```

### Sequential Models (Sandbox)

```{toctree}
---
maxdepth: 1
---
rbm_tutorial
rlrmc_tutorial
slirec_tutorial
sum_tutorial
```

### Bayesian Models (Sandbox)

```{toctree}
---
maxdepth: 1
---
bpr_tutorial
bprmf_tutorial
vmf_tutorial
```

### Content-Based Models (Sandbox)

```{toctree}
---
maxdepth: 1
---
mind_content_tutorial
```

## Pipeline & System Tutorials

End-to-end system tutorials:

```{toctree}
---
maxdepth: 1
---
pipeline_tutorial
imshow_tutorial
```

## Tutorial Structure

### Production Model Tutorials
Full working examples with tested code you can copy-paste and run.

### Sandbox Model Tutorials
Each sandbox tutorial covers:
1. **Introduction** — What the model does and its original paper
2. **Architecture & Theory** — How it works, with diagrams and math
3. **Reference Implementation** — Code examples (informational, not production-tested)
4. **Use Cases & Scaling** — When to use it, how to scale, and production considerations
5. **Key Takeaways** — Best practices and further reading

## Learning Path

### Beginners
1. Start with [DCN Tutorial](dcn_tutorial.md) (Production)
2. Try [NCF Tutorial](ncf_tutorial.md) (Production)
3. Explore [SAR Tutorial](sar_tutorial.md) (Production)

### Intermediate
1. Deep dive into [DeepFM](deepfm_tutorial.md) (Production)
2. Learn [Graph Methods with LightGCN](lightgcn_tutorial.md) (Production)
3. Master [Multi-Interest with MIND](mind_tutorial.md) (Production)

### Advanced
1. Study [Neural Architecture Search](nasrec_tutorial.md) (Production)
2. Explore [Sandbox Models](../models/index.md#sandbox-models-experimental) for research
3. Deploy to [Production](../examples/production_deployment.md)
