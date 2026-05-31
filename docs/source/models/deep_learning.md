# Deep Learning Models

Neural network models for rating prediction, ranking, and top-N recommendation.

## Production models (CI-tested)

These models live under `corerec.engines.*`, inherit `BaseRecommender`, and pass automated tests on every commit.

| Model | Import | Tutorial |
|-------|--------|----------|
| **DCN** | `from corerec.engines.dcn import DCN` | [DCN Tutorial](../tutorials/dcn_tutorial.md) |
| **DeepFM** | `from corerec.engines.deepfm import DeepFM` | [DeepFM Tutorial](../tutorials/deepfm_tutorial.md) |
| **GNNRec** | `from corerec.engines.gnnrec import GNNRec` | [GNNRec Tutorial](../tutorials/gnnrec_tutorial.md) |
| **MIND** | `from corerec.engines.mind import MIND` | [MIND Tutorial](../tutorials/mind_tutorial.md) |
| **NASRec** | `from corerec.engines.nasrec import NASRec` | [NASRec Tutorial](../tutorials/nasrec_tutorial.md) |
| **SASRec** | `from corerec.engines.sasrec import SASRec` | [SASRec Tutorial](../tutorials/sasrec_tutorial.md) |
| **TwoTower** | `from corerec.engines.two_tower import TwoTower` | [TwoTower Tutorial](../tutorials/two_tower_tutorial.md) |
| **BERT4Rec** | `from corerec.engines.bert4rec import BERT4Rec` | [BERT4Rec Tutorial](../tutorials/bert4rec_tutorial.md) |

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
| AFM | `corerec.sandbox.collaborative_full.nn_base.AFM_base` | [AFM](../tutorials/afm_tutorial.md) |
| AutoInt | `corerec.sandbox.collaborative_full.nn_base.AutoInt_base` | [AutoInt](../tutorials/autoint_tutorial.md) |
| AutoFI | `corerec.sandbox.collaborative_full.nn_base.AutoFI_base` | [AutoFI](../tutorials/autofi_tutorial.md) |
| BST | `corerec.sandbox.collaborative_full.nn_base.BST_base` | [BST](../tutorials/bst_tutorial.md) |
| BiVAE | `corerec.sandbox.collaborative_full.variational_encoder_base.bivae_base` | [BiVAE](../tutorials/bivae_tutorial.md) |
| Caser | `corerec.sandbox.collaborative_full.nn_base.caser` | [Caser](../tutorials/caser_tutorial.md) |
| DeepCrossing | sandbox nn_base | [DeepCrossing](../tutorials/deepcrossing_tutorial.md) |
| DeepRec | `corerec.sandbox.collaborative_full.nn_base.DeepRec_base` | [DeepRec](../tutorials/deeprec_tutorial.md) |
| DIEN | `corerec.sandbox.collaborative_full.nn_base.DIEN_base` | [DIEN](../tutorials/dien_tutorial.md) |
| DiFM | sandbox nn_base | [DiFM](../tutorials/difm_tutorial.md) |
| DIN | `corerec.sandbox.collaborative_full.nn_base.DIN_base` | [DIN](../tutorials/din_tutorial.md) |
| DLRM | `corerec.sandbox.collaborative_full.nn_base.DLRM_base` | [DLRM](../tutorials/dlrm_tutorial.md) |
| ENSFM | `corerec.sandbox.collaborative_full.nn_base.ENSFM_base` | [ENSFM](../tutorials/ensfm_tutorial.md) |
| ESCM2 | `corerec.sandbox.collaborative_full.nn_base.ESCMM_base` | [ESCMM](../tutorials/escmm_tutorial.md) |
| ESMM | `corerec.sandbox.collaborative_full.nn_base.ESMM_base` | [ESMM](../tutorials/esmm_tutorial.md) |
| FGCNN | `corerec.sandbox.collaborative_full.nn_base.FGCNN_base` | [FGCNN](../tutorials/fgcnn_tutorial.md) |
| FFM | `corerec.sandbox.collaborative_full.nn_base.FFM_base` | [FFM](../tutorials/ffm_tutorial.md) |
| FiBiNet | `corerec.sandbox.collaborative_full.nn_base.Fibinet_base` | [FiBiNet](../tutorials/fibinet_tutorial.md) |
| FLEN | `corerec.sandbox.collaborative_full.nn_base.FLEN_base` | [FLEN](../tutorials/flen_tutorial.md) |
| FM | `corerec.sandbox.collaborative_full.nn_base.FM_base` | [FM](../tutorials/fm_tutorial.md) |
| GAN-Rec | `corerec.sandbox.collaborative_full.nn_base.gan_ufilter_base` | [GAN](../tutorials/gan_tutorial.md) |
| GateNet | sandbox nn_base | [GateNet](../tutorials/gatenet_tutorial.md) |
| GRU-CF | `corerec.sandbox.collaborative_full.nn_base.gru_ufilter_base` | [GRU-CF](../tutorials/gru_cf_tutorial.md) |
| NFM | `corerec.sandbox.collaborative_full.nn_base.NFM_base` | [NFM](../tutorials/nfm_tutorial.md) |
| NextItNet | `corerec.sandbox.collaborative_full.sequential_model_base.nextitnet_base` | [NextItNet](../tutorials/nextitnet_tutorial.md) |
| Wide&Deep | `corerec.sandbox.collaborative_full.nn_base.WideDeep_base` | [Wide&Deep](../tutorials/widedeep_tutorial.md) |
| YouTubeDNN | sandbox content nn | [YouTubeDNN](../tutorials/youtubednn_tutorial.md) |
| PNN | sandbox nn_base | [PNN](../tutorials/pnn_tutorial.md) |
| MMoE | `corerec.sandbox.collaborative_full.nn_base.MMoE_base` | [MMoE](../tutorials/mmoe_tutorial.md) |
| PLE | `corerec.sandbox.collaborative_full.nn_base.PLE_base` | [PLE](../tutorials/ple_tutorial.md) |
| Monolith | `corerec.sandbox.collaborative_full.nn_base.Monolith_base` | [Monolith](../tutorials/monolith_tutorial.md) |
| TDM | `corerec.sandbox.collaborative_full.nn_base.TDM_base` | [TDM](../tutorials/tdm_tutorial.md) |
| DCN-Base | `corerec.sandbox.collaborative_full.nn_base.DCN` | [DCN Base](../tutorials/dcn_base_tutorial.md) |
| DeepFM-Base | `corerec.sandbox.collaborative_full.nn_base.DeepFM_base` | [DeepFM Base](../tutorials/deepfm_base_tutorial.md) |

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
