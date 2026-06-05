#!/bin/sh
# Leave-one-out + 99 negatives protocol (HR@10, NDCG@10) on ML-1M@100k.
# Classic + cornac neural score cheaply (2000 users); CoreRec's own neural models
# score per-pair (slow) so they use a smaller, clearly-labeled user cap.
set -u
cd "$(dirname "$0")"
SIZE=100000
mkdir -p results/loo
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

run() {  # framework model device eval_users
  out="results/loo/${1}_${2}.json"
  if [ -f "$out" ]; then echo "skip $out"; return; fi
  echo "### LOO $1 $2 ($3, $4 users)"
  if [ "$3" = "cpu" ]; then export CUDA_VISIBLE_DEVICES=""; else unset CUDA_VISIBLE_DEVICES; fi
  python -W ignore loo_runner.py --framework "$1" --model "$2" \
      --size "$SIZE" --device "$3" --eval_users "$4" --out "$out" 2>/dev/null \
      && echo "  ok" || echo "  FAILED $1 $2"
}

# cheap scorers: 2000 users (cornac LightGCN/NGCF need dgl; cornac NeuMF is
# CPU-only and prohibitively slow here, so the deep same-model check uses RecBole)
run corerec  SAR      cpu  2000
run cornac   ItemKNN  cpu  2000
run cornac   BPR      cpu  2000
run cornac   MF       cpu  2000
run cornac   VAECF    cpu  2000
run implicit ALS      cpu  2000
run implicit BPR      cpu  2000
run lightfm  WARP     cpu  2000
# CoreRec neural: per-pair scoring is slow, smaller cap (labeled)
run corerec  NCF      cuda 200
run corerec  LightGCN cuda 200
run corerec  DCN      cuda 200

echo "LOO STUDY DONE"
