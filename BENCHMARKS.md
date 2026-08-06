# CoreRec benchmarks

Everything below is reproducible from a clean checkout. Where CoreRec loses, the
losing number is here too — a table where one library wins every row is a
marketing document, not a benchmark.

## Setup

| | |
|---|---|
| Dataset | MovieLens 100K, official `u1.base` / `u1.test` split |
| Relevance | test rating ≥ 4 |
| Metric | Recall@10, NDCG@10, HitRate@10 over 300 users (seed 42) |
| Masking | training items masked at ranking time |
| Shared budget | `RANK_DIM=32`, `EPOCHS=20` for every iterative model |
| Hardware | CPU only, single machine, CUDA disabled for all frameworks |
| Baseline | `implicit` 0.7.3 |

Test rows whose user or item is unseen in train are dropped before scoring, for
every framework equally.

```bash
pip install implicit
cd Findings/bench
python runner.py --framework implicit --model ALS --dataset ml100k --out results/fresh/implicit_ALS.json
python runner.py --framework corerec  --model ALS --dataset ml100k --out results/fresh/corerec_ALS.json
python aggregate.py results/fresh          # -> all_results.csv + table.md
```

## Results

Sorted by NDCG@10. Fusion rows combine two models with Reciprocal Rank Fusion
(k=60) and are marked — they are **not** comparable to single-model rows.

| Framework | Model | NDCG@10 | Recall@10 | Fit (s) | Latency (ms/user) |
|---|---|---:|---:|---:|---:|
| implicit *(fused)* | ALS + ItemKNN | **0.4553** | 0.2450 | 0.49 | 0.42 |
| corerec *(fused)* | ALS + SAR | 0.4488 | 0.2382 | 9.27 | 2.56 |
| **corerec** | **ALS** | **0.4168** | 0.2345 | 8.96 | 0.98 |
| implicit | ALS | 0.4099 | 0.2304 | 0.50 | 0.11 |
| **corerec** | **SAR (cosine)** | **0.3955** | — | 0.18 | 1.26 |
| implicit | ItemKNN (cosine) | 0.3858 | 0.2069 | 0.08 | 0.14 |
| corerec | SAR (jaccard) | 0.3730 | 0.1990 | 0.21 | 1.27 |
| corerec | LightGCN | 0.3361 | 0.1665 | 126.98 | 11.90 |
| corerec | NCF | 0.3343 | 0.1775 | 143.44 | 6.83 |
| implicit | BPR | 0.3239 | 0.1446 | 0.31 | 0.08 |
| corerec | GNNRec | *did not complete* | — | >2460 | — |

## What this says

**CoreRec wins every like-for-like single-model comparison.** Its ALS beats
implicit's ALS (0.4168 vs 0.4099) and its cosine SAR beats implicit's cosine
ItemKNN (0.3955 vs 0.3858), on identical data with an identical budget.

**CoreRec loses on speed, by a lot.** Its ALS takes 8.96s where implicit's takes
0.50s — 18× slower for a 1.7% quality gain. implicit's ALS is years of tuned
Cython over BLAS. On a dataset 100× this size that ratio is the whole story, and
it is the strongest argument for using implicit over CoreRec today.

**CoreRec loses the fused comparison, narrowly.** 0.4488 vs 0.4553, a 1.4% gap,
despite CoreRec winning *both* components. The cause is measurable: CoreRec's two
signals have a mean Spearman rank correlation of 0.474 against implicit's 0.453.
Rank fusion pays for decorrelation, not for component strength, and CoreRec's ALS
and SAR are slightly more alike than implicit's ALS and ItemKNN.

**Neither library benefits from a third signal.** Fusing three models made both
worse — implicit 0.4553 → 0.4371 (adding BPR), CoreRec 0.4488 → 0.4247 (adding
Item2Vec). A weak third component drags an ensemble down in both.

**CoreRec's neural models are not competitive here.** NCF and LightGCN land at
0.334–0.336, below every classical method including a 0.08s ItemKNN, while taking
127–143s to train. This is consistent with Dacrema et al. (2019), who found
neural recommenders routinely fail to beat well-tuned simple baselines. If you
are choosing a model for a dataset this size, the classical ones are the answer.

## Caveats

- **One dataset, one split.** ML-100K is small and well-studied. Nothing here
  generalises to production scale without re-running; conclusions about ALS
  throughput in particular will change with data size.
- **No hyperparameter search.** Every model runs at a shared `RANK_DIM=32` /
  `EPOCHS=20`. That is even-handed, not optimal — each library would score higher
  tuned, and the ordering could change.
- **The `_cosine` variants are configuration alignment, not tuning.** implicit's
  ItemKNN is cosine, so CoreRec's SAR is compared at cosine. CoreRec's jaccard
  default is reported too, and it is worse (0.3730).
- **Latency is single-user, single-threaded**, measured after fit with a warm
  cache. It is a floor, not a production SLA.
- **Fit times were measured with other jobs running** in some cases. NCF's 308s
  entry in the raw JSON was contention; its clean number is 143s.

## Bugs this benchmark found

Running the comparison was worth more than the table.

1. **`batch_predict` didn't batch.** `BaseRecommender.batch_predict`, documented
   as scoring pairs "efficiently", was a list comprehension over `predict()` —
   one forward pass per pair for a torch model. Scoring a 1650-item catalogue for
   one user took **262ms**, about 4 QPS, and 2300× implicit's ALS. NCF now
   overrides it with a single batched pass: **262ms → 6.8ms, 38× faster**, with
   scores identical to 6e-08. Covered by `test_batch_predict_matches_predict`.

2. **`GNNRec` cannot train on ML-100K.** Killed after 41 minutes on 100k
   interactions where LightGCN took 127s and NCF 143s. Recorded as a
   non-completion rather than dropped from the table.

3. **`SAR(similarity_type="lift")` is broken.** Returns NDCG@10 = 0.0007 —
   indistinguishable from random. Cosine and jaccard both work. Not yet fixed.

4. **Memory numbers were 1024× wrong on macOS.** `peak_mem_mb` treated
   `ru_maxrss` as kilobytes, but macOS reports bytes. One run recorded
   `138204 MB` for a process that peaked near 135MB. Any memory figure produced
   on a Mac before this fix is unusable, which is why results predating it are
   kept in a separate directory.

5. **The benchmark's own data paths were broken.** `BENCH` pointed at
   `corerec/Findings/bench` (no such directory) and `datautil.ML100K_DIR` had one
   `..` too many. The five NDCG floor tests in `tests/test_benchmark_floors.py`
   had therefore never executed.

## Reproducing

```bash
git clone https://github.com/vishesh9131/CoreRec && cd CoreRec
pip install -e ".[dev,serving]" implicit
cd Findings/bench
for m in ALS SAR SAR_cosine NCF LightGCN; do
  python runner.py --framework corerec --model $m --dataset ml100k --out results/fresh/corerec_$m.json
done
for m in ALS BPR ItemKNN; do
  python runner.py --framework implicit --model $m --dataset ml100k --out results/fresh/implicit_$m.json
done
python runner.py --framework corerec_hybrid    --model ALS_SAR_RRF --dataset ml100k --out results/fresh/corerec_hybrid.json
python runner.py --framework implicit_ensemble --model ALS_KNN_RRF --dataset ml100k --out results/fresh/implicit_ensemble.json
python aggregate.py results/fresh
```

ML-100K is not tracked in this repo; the runner skips cleanly when it is absent.
