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
| Hardware | CPU only, single-threaded per job, CUDA disabled for all frameworks |
| Baseline | `implicit` 0.7.3 |

Numbers below are from a 96-core Linux box (Ubuntu 22.04, Python 3.10, torch
2.10). Each job is pinned to one BLAS thread so parallel jobs can't
oversubscribe each other and distort `fit_time_s` — so fit times are
**single-core** figures, not wall-clock on a big machine.

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
| implicit *(fused)* | ALS + ItemKNN | **0.4547** | 0.2450 | 0.99 | 0.39 |
| corerec *(fused)* | ALS + SAR | 0.4493 | 0.2382 | 5.38 | 3.51 |
| **corerec** | **ALS** | **0.4168** | 0.2345 | 5.13 | 1.32 |
| implicit | ALS | 0.4100 | 0.2304 | 0.56 | 0.07 |
| **corerec** | **SAR (cosine)** | **0.3955** | 0.2077 | 0.35 | 1.93 |
| implicit | ItemKNN (cosine) | 0.3858 | 0.2069 | 0.08 | 0.27 |
| corerec | SAR (jaccard) | 0.3730 | 0.1990 | 0.54 | 3.73 |
| corerec | LightGCN † | 0.3360 ± 0.0077 | 0.1739 | 150.97 | 12.57 |
| corerec | NCF ‡ | 0.3359 | 0.1766 | 162.10 | 4.46 |
| corerec | NCF_binary † | 0.3091 ± 0.0178 | 0.1606 | 83.50 | 3.91 |
| implicit | BPR † | 0.3204 ± 0.0069 | 0.1360 | 1.48 | 0.07 |
| corerec | GNNRec | *did not complete* | — | >3720 | — |

† Sampled-negative model: mean ± std over 5 seeds, because a single run of these
lands up to 13% away from the mean. Deterministic rows are single runs that
reproduce exactly. See "Which of these numbers are stable".

‡ Also samples negatives, but only a single run was done; treat it the same way
as the † rows and expect a comparable spread. NCF_binary, which differs only in
its positive set, has a std of 0.0178.

GNNRec was stopped after 62 minutes on one BLAS thread, where LightGCN finishes
the same data in 151s and NCF in 162s. It is listed rather than dropped: a model
that cannot train on the smallest standard benchmark within an hour is a result.
Under looser conditions it has scored 0.334 — below a 0.08-second ItemKNN.

## Which of these numbers are stable

Two checks, because a benchmark nobody can reproduce is a press release.

**Across machines.** The whole table was run on an M-series macOS laptop
(Python 3.12, torch 2.2, numpy 1.x) and a 96-core Ubuntu box (Python 3.10,
torch 2.10, numpy 2.2). Every deterministic model reproduced exactly:

| Model | laptop | 96-core box |
|---|---:|---:|
| corerec ALS | 0.4168 | 0.4168 |
| corerec SAR (cosine) | 0.3955 | 0.3955 |
| implicit ItemKNN | 0.3858 | 0.3858 |
| corerec SAR (jaccard) | 0.3730 | 0.3730 |
| implicit ALS | 0.4099 | 0.4100 |

**Across seeds.** Five seeds each, evaluation cohort held fixed so only the
model's own randomness moves:

| Framework | Model | mean NDCG@10 | std | spread |
|---|---|---:|---:|---:|
| implicit | ItemKNN | 0.3858 | **0.0000** | 0.0% |
| implicit | ALS | 0.4061 | 0.0048 | 2.4% |
| corerec | LightGCN | 0.3360 | 0.0077 | 6.0% |
| implicit | BPR | 0.3204 | 0.0069 | 5.1% |
| corerec | NCF_binary | 0.3091 | **0.0178** | **13.1%** |

ItemKNN has no randomness anywhere and lands on 0.0000 std across all five
runs, which is what makes the rest of the column trustworthy: the spread on the
others is the models, not the harness.

Three consequences, and they cut against the single-run table above:

- **NCF_binary's 0.3236 and LightGCN's 0.3444 were lucky draws.** Their means
  are 0.3091 and 0.3360. Any single run of a sampled-negative model can land
  most of a standard deviation from where it belongs.
- **A gap smaller than the relevant std is not a result.** NCF at 0.3359 and
  NCF_binary at 0.3236 differ by less than NCF_binary's own std, so the
  binarisation experiment proved nothing either way.
- **Even ALS is seed-sensitive at 2.4%,** so the 1.7% quality margin CoreRec's
  ALS holds over implicit's is inside the noise of a single run. It survived
  because it reproduced exactly on two machines at a fixed seed, not because
  0.4168 beats 0.4100 once.

Reproduce with `Findings/bench/run_seeds.sh` (`SEEDS="1 2 3 4 5"`); every run's
JSON records the seed it used.

## What this says

**CoreRec wins every like-for-like single-model comparison.** Its ALS beats
implicit's ALS (0.4168 vs 0.4099) and its cosine SAR beats implicit's cosine
ItemKNN (0.3955 vs 0.3858), on identical data with an identical budget.

**CoreRec loses on speed, by a lot.** Its ALS takes 5.13s where implicit's takes
0.56s — 9× slower for a 1.7% quality gain. implicit's ALS is years of tuned
Cython over BLAS. On a dataset 100× this size that ratio is the whole story, and
it is the strongest argument for using implicit over CoreRec today. The gap is
partly single-core-performance-bound rather than purely algorithmic: the same
comparison on a slower laptop core measured 18×, so the ratio itself moves with
hardware.

**CoreRec loses the fused comparison, narrowly.** 0.4493 vs 0.4547, a 1.2% gap,
despite CoreRec winning *both* components. The cause is measurable: CoreRec's two
signals have a mean Spearman rank correlation of 0.474 against implicit's 0.453.
Rank fusion pays for decorrelation, not for component strength, and CoreRec's ALS
and SAR are slightly more alike than implicit's ALS and ItemKNN.

**Neither library benefits from a third signal.** Fusing three models made both
worse — implicit 0.4553 → 0.4371 (adding BPR), CoreRec 0.4488 → 0.4247 (adding
Item2Vec). A weak third component drags an ensemble down in both. (Measured on
the laptop, before the reproducibility check above; BPR's instability means the
implicit figure in particular should be treated as approximate.)

**CoreRec's neural models are not competitive here.** NCF and LightGCN land
around 0.32–0.34, below every classical method including a 0.08s ItemKNN, while
taking 84–162s to train. The gap to the classical models is an order of
magnitude larger than their run-to-run drift, so this conclusion survives the
instability noted above. This is consistent with Dacrema et al. (2019), who found
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
- **Fit times are single-core.** Each job is pinned to one BLAS thread so
  parallel runs don't distort each other. Multi-threaded numbers would be lower
  for every framework, and not necessarily by the same factor.
- **The three sampled-negative models are single-run.** See the stability
  section: those rows move by up to 5% between machines and need multi-seed
  runs before small differences mean anything.

## Bugs this benchmark found

Running the comparison was worth more than the table.

1. **`batch_predict` didn't batch.** `BaseRecommender.batch_predict`, documented
   as scoring pairs "efficiently", was a list comprehension over `predict()` —
   one forward pass per pair for a torch model. Scoring a 1650-item catalogue for
   one user took **262ms**, about 4 QPS, and 2300× implicit's ALS. NCF now
   overrides it with a single batched pass: **262ms → 6.8ms, 38× faster**, with
   scores identical to 6e-08. Covered by `test_batch_predict_matches_predict`.

2. **`GNNRec` is impractically slow, and the obvious fix made it worse.**
   It was first killed after 41 minutes on 100k interactions where LightGCN
   took 127s. `EmbeddingPropagationLayer.forward` was rebuilding a dense
   identity matrix on every mini-batch (~7,800 times per run), so the graph —
   fixed for the life of the model — is now built once and cached.

   The tempting second half of that fix was wrong. The graph is 97.6% zeros, so
   `torch.sparse.mm` looks like a 42× FLOP saving; measured on identical data
   and seed it was nearly 2× *slower*, and caching alone accounts for only ~7%:

   | variant | 3-epoch fit |
   |---|---:|
   | dense + cached (shipped) | 7.5s |
   | dense + uncached (original) | 8.1s |
   | sparse + cached | 14.3s |

   PyTorch's CPU sparse kernels, the backward pass especially, lose to BLAS at
   this size; FLOP counts do not predict wall-clock. What actually dominates is
   the *number* of propagations, not the cost of each: `forward()` propagates
   the whole graph and then gathers the batch, so `batch_size` does not reduce
   work, and `recommend()` re-propagates per item batch per user. Caching
   embeddings in eval mode is the fix that would matter for serving, and has
   not been attempted.

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
for m in ALS SAR SAR_cosine NCF NCF_binary LightGCN GNNRec; do
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

To reproduce the whole table in parallel (each job single-threaded so the timings
stay meaningful):

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# ... launch the runner.py invocations above concurrently, then:
python aggregate.py results/fresh
```
