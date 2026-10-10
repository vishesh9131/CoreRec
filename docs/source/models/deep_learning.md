# Deep Learning Models

Neural network models for rating prediction, ranking, and top-N recommendation.

## Production models (CI-tested)

These models live under `corerec.engines.*`, inherit `BaseRecommender`, and pass automated tests on every commit.

| Model | Import | Tutorial |
|-------|--------|----------|
| **DCN** | `from corerec.engines.dcn import DCN` | [DCN Tutorial](../tutorials/dcn_tutorial.md) |
| **DeepFM** | `from corerec.engines.deepfm import DeepFM` | [DeepFM Tutorial](../tutorials/deepfm_tutorial.md) |
| **SASRec** | `from corerec.engines.sasrec import SASRec` | [SASRec Tutorial](../tutorials/sasrec_tutorial.md) |
| **TwoTower** | `from corerec.engines.two_tower import TwoTower` | [TwoTower Tutorial](../tutorials/two_tower_tutorial.md) |

### Example (triplet-based models)

Most production deep models accept `(user_ids, item_ids, ratings)` triplets. Use **binary (0/1) or normalized ratings** for models trained with BCE loss (e.g. GNNRec).

```python
from corerec.engines.dcn import DCN

model = DCN(embedding_dim=64, epochs=20, verbose=True)
model.fit(user_ids=user_ids, item_ids=item_ids, ratings=ratings)

score = model.predict(user_id=1, item_id=100)
recs = model.recommend(user_id=1, top_k=10)
model.save("artifacts/dcn")  # safe bundle by default
```

**SASRec** and **BERT4Rec** use an **interaction matrix** instead of raw triplets — see their tutorials.

## Sandbox models (experimental)

Implementations under `corerec/sandbox/`. Not production-tested.

| Model | Import path | Tutorial |
|-------|-------------|----------|
| AFM | `corerec.sandbox.collaborative_full.nn_base.AFM_base` | [AFM](../tutorials/removed_models.md) |
| AutoInt | `corerec.sandbox.collaborative_full.nn_base.AutoInt_base` | [AutoInt](../tutorials/removed_models.md) |
| AutoFI | `corerec.sandbox.collaborative_full.nn_base.AutoFI_base` | [AutoFI](../tutorials/removed_models.md) |
| BST | `corerec.sandbox.collaborative_full.nn_base.BST_base` | [BST](../tutorials/removed_models.md) |
| BiVAE | `corerec.sandbox.collaborative_full.variational_encoder_base.bivae_base` | [BiVAE](../tutorials/removed_models.md) |
| Caser | `corerec.sandbox.collaborative_full.nn_base.caser` | [Caser](../tutorials/removed_models.md) |
| DeepCrossing | sandbox nn_base | [DeepCrossing](../tutorials/removed_models.md) |
| DeepRec | `corerec.sandbox.collaborative_full.nn_base.DeepRec_base` | [DeepRec](../tutorials/removed_models.md) |
| DIEN | `corerec.sandbox.collaborative_full.nn_base.DIEN_base` | [DIEN](../tutorials/removed_models.md) |
| DiFM | sandbox nn_base | [DiFM](../tutorials/removed_models.md) |
| DIN | `corerec.sandbox.collaborative_full.nn_base.DIN_base` | [DIN](../tutorials/removed_models.md) |
| DLRM | `corerec.sandbox.collaborative_full.nn_base.DLRM_base` | [DLRM](../tutorials/removed_models.md) |
| ENSFM | `corerec.sandbox.collaborative_full.nn_base.ENSFM_base` | [ENSFM](../tutorials/removed_models.md) |
| ESCM2 | `corerec.sandbox.collaborative_full.nn_base.ESCMM_base` | [ESCMM](../tutorials/removed_models.md) |
| ESMM | `corerec.sandbox.collaborative_full.nn_base.ESMM_base` | [ESMM](../tutorials/removed_models.md) |
| FGCNN | `corerec.sandbox.collaborative_full.nn_base.FGCNN_base` | [FGCNN](../tutorials/removed_models.md) |
| FFM | `corerec.sandbox.collaborative_full.nn_base.FFM_base` | [FFM](../tutorials/removed_models.md) |
| FiBiNet | `corerec.sandbox.collaborative_full.nn_base.Fibinet_base` | [FiBiNet](../tutorials/removed_models.md) |
| FLEN | `corerec.sandbox.collaborative_full.nn_base.FLEN_base` | [FLEN](../tutorials/removed_models.md) |
| FM | `corerec.sandbox.collaborative_full.nn_base.FM_base` | [FM](../tutorials/removed_models.md) |
| GAN-Rec | `corerec.sandbox.collaborative_full.nn_base.gan_ufilter_base` | [GAN](../tutorials/removed_models.md) |
| GateNet | sandbox nn_base | [GateNet](../tutorials/removed_models.md) |
| GRU-CF | `corerec.sandbox.collaborative_full.nn_base.gru_ufilter_base` | [GRU-CF](../tutorials/removed_models.md) |
| NFM | `corerec.sandbox.collaborative_full.nn_base.NFM_base` | [NFM](../tutorials/removed_models.md) |
| NextItNet | `corerec.sandbox.collaborative_full.sequential_model_base.nextitnet_base` | [NextItNet](../tutorials/removed_models.md) |
| Wide&Deep | `corerec.sandbox.collaborative_full.nn_base.WideDeep_base` | [Wide&Deep](../tutorials/removed_models.md) |
| YouTubeDNN | sandbox content nn | [YouTubeDNN](../tutorials/removed_models.md) |
| PNN | sandbox nn_base | [PNN](../tutorials/removed_models.md) |
| MMoE | `corerec.sandbox.collaborative_full.nn_base.MMoE_base` | [MMoE](../tutorials/removed_models.md) |
| PLE | `corerec.sandbox.collaborative_full.nn_base.PLE_base` | [PLE](../tutorials/removed_models.md) |
| TDM | `corerec.sandbox.collaborative_full.nn_base.TDM_base` | [TDM](../tutorials/removed_models.md) |
| DCN-Base | `corerec.sandbox.collaborative_full.nn_base.DCN` | [DCN Base](../tutorials/removed_models.md) |
| DeepFM-Base | `corerec.sandbox.collaborative_full.nn_base.DeepFM_base` | [DeepFM Base](../tutorials/removed_models.md) |

```{admonition} Sandbox warning
:class: warning
Always import sandbox models from `corerec.sandbox.*`, not `corerec.engines.*`. Validate thoroughly before any production use.
```

## When to use deep learning models

- Large interaction datasets with non-linear patterns
- Rich side features (DCN, DeepFM)
- Sequential behavior (SASRec, BERT4Rec)
- Graph structure (GNNRec)
- Multi-interest sessions (MIND)
- Dual-tower retrieval at scale (TwoTower)

## See also

- [Model tiers overview](index.md#model-tiers)
- [Full model index](models_index.md)
- [Tutorials](../tutorials/index.md)
