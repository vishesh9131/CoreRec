# CoreRec benchmarking study

The findings are written up as a LaTeX manuscript:

- **[manuscript/corerec_benchmark.tex](manuscript/corerec_benchmark.tex)** -> compiled **[manuscript/corerec_benchmark.pdf](manuscript/corerec_benchmark.pdf)**

Rebuild the PDF with `pdflatex corerec_benchmark.tex` (run twice for references).

## What the study covers

CoreRec 0.5.3 vs Cornac 2.4.0, RecBole 1.2.1, implicit 0.7.3, LightFM 1.17,
Surprise 1.1.5, on MovieLens-100K and MovieLens-1M, with four parts:

1. **Base benchmark** (ML-100K, holdout full ranking).
2. **Same model, different framework** (ML-1M@100k): CoreRec's NeuMF/LightGCN/
   neighborhood/DCN/DeepFM/GNNRec vs other frameworks' implementations of the
   same architecture, scored by an identical metric harness (RecBole via its own
   evaluator on the same split).
3. **Scalability sweep** ($10^4 \rightarrow 10^6$ interactions).
4. **Large-scale public datasets**: Gowalla (~1.0M) and Yelp2018 (~1.6M), the
   canonical graph-recsys benchmarks.
5. **Leave-one-out + 99 negatives** (the NeuMF-paper protocol) as a second
   standard methodology.
6. **Production-ecosystem case study**: framed as a bank's Instagram-style feed —
   where CoreRec is strong vs weak as a deployable stack.
7. **Online serving engine** (`corerec.serving.OnlineRecommender`): closes the
   online/high-scale/freshness gaps — ANN retrieval (FAISS HNSW), incremental
   add-items + user fold-in without retrain, graceful cold-start. Demo on Gowalla:
   ~0.09 ms/user (10k req/s), 1.5 GB (vs SAR's 38 GB), 1M-item ANN at 0.02 ms/query,
   99.8% of exact accuracy. Run: `python bench/demo_online.py --dataset gowalla`.

Plus a per-model "why CoreRec loses" analysis, a capability/interpretability
matrix, an "After remediation" section showing the deep-model fixes, and the
online-serving section showing the new engine.

## Model zoo: 14 -> 41 production models, leading 3 of 4 benchmarks

The production catalogue was grown from 14 to 41 models (all unified-contract,
CI-gated): deep CTR (FM, AFM, NFM, DeepFM, DCN, AutoInt, xDeepFM, FiBiNet, PNN,
WideDeep, GMF, MLP), sequential (GRU4Rec, Caser, BST, DIN, DIEN, NARM), classic CF
(ItemKNN, UserKNN, EASE, SLIM), auto-encoder (MultVAE, MultiDAE), graph (NGCF) and
native MF (ALS, Item2Vec). Best CoreRec model vs best competitor per dataset:

| Dataset | Best competitor | Best CoreRec | Result |
|---|---|---|---|
| ML-100K (NDCG@10) | WARP 0.435 | **SLIM 0.460** | CoreRec #1 |
| Gowalla (NDCG@20) | WARP 0.117 | **LightGCN 0.145** | CoreRec #1 |
| Yelp2018 (NDCG@20) | ALS 0.045 | **LightGCN 0.046** | CoreRec #1 |
| MovieLens-1M (NDCG@20) | ALS 0.360 | UserKNN 0.339 | #2 (close) |

## Native trainers now lead the large-scale benchmarks

With the new GPU sparse trainers (`OnlineRecommender.from_interactions(df, model="lightgcn")`),
**CoreRec's LightGCN is #1 on both million-scale graph benchmarks** at the standard
NDCG@20 cutoff, beating every competitor:

| Dataset | CoreRec LightGCN | best competitor |
|---|---|---|
| Gowalla (~1.0M) | **0.1451** (#1) | lightfm WARP 0.1171 |
| Yelp2018 (~1.6M) | **0.0460** (#1) | implicit ALS 0.0448 |
| MovieLens-1M | 0.3115 (#3) | implicit ALS 0.3600 |

(ALS, a specialized MF, still leads the small dense ML-1M.) The same embeddings
feed the sub-ms ANN serving engine. See manuscript §"A native training engine that
wins the large-scale benchmarks".

## Headline result

CoreRec's well-implemented models are competitive or better than reference
implementations of the same architecture (its NeuMF, LightGCN and neighborhood
model win their families), but — *as originally benchmarked* — its CTR/graph
models (DCN, DeepFM, GNNRec) collapsed to a near-constant output under the
documented usage and lost.

**Update (remediation):** those failures were then fixed in CoreRec. A `task`
contract with negative sampling restores the deep models to parity with the
reference implementations (DCN NDCG@10 0.005→0.296, DeepFM 0.018→0.285, GNNRec
0.000→0.358), vectorized inference cuts recommend latency ~150× (DCN 236→1.55
ms/user), and a lazy-import fix drops the SAR footprint 754→124 MB (lightest in
the comparison). See Section "After remediation" in the manuscript for the
before/after table.

## Reproducing

Code and raw results live in [`bench/`](bench/):

- `datautil.py` - data, splits (ML-100K canonical, ML-1M subsample, leave-one-out)
- `metrics.py` - shared Recall/NDCG/HitRate/RMSE + LOO metrics
- `runner.py` - holdout adapters (CoreRec, Cornac, implicit, LightFM, Surprise)
- `recbole_runner.py` - RecBole on the same split (atomic files)
- `loo_runner.py` - leave-one-out protocol
- `run_*.sh` - drivers; `aggregate*.py` - build CSV/markdown tables
- `results/` - raw per-run JSON, `*_results.csv`, `*_table.md`

```bash
pip install --user scikit-surprise cornac implicit lightfm recbole
cd bench
sh run_scale.sh        # scalability sweep
sh run_same.sh         # same-model matrix
sh run_recbole_deep.sh # RecBole reference implementations
sh run_loo.sh          # leave-one-out study
python aggregate.py && python aggregate_scale.py && python aggregate_same.py
```
