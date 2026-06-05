#!/bin/sh
# Same-model, cross-framework matrix on a fixed ML-1M subsample (100k interactions).
# Classic models on CPU; neural models on GPU (fair: torch-vs-torch).
# All scored by the identical harness (datautil split + metrics.py).
set -u
cd "$(dirname "$0")"
SIZE=100000
mkdir -p results/same
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

run() {  # framework model device
  out="results/same/${1}_${2}.json"
  if [ -f "$out" ]; then echo "skip $out"; return; fi
  echo "### $1 $2 ($3)"
  if [ "$3" = "cpu" ]; then export CUDA_VISIBLE_DEVICES=""; else unset CUDA_VISIBLE_DEVICES; fi
  python -W ignore runner.py --framework "$1" --model "$2" \
      --dataset ml1m --size "$SIZE" --device "$3" --eval_users 100 --out "$out" 2>/dev/null \
      && echo "  ok" || echo "  FAILED $1 $2"
}

# classic CF (CPU)
run corerec  SAR      cpu
run cornac   ItemKNN  cpu
run cornac   UserKNN  cpu
run cornac   BPR      cpu
run cornac   MF       cpu
run cornac   GMF      cpu
run cornac   WMF      cpu
run implicit ALS      cpu
run implicit BPR      cpu
run lightfm  WARP     cpu
run surprise SVD      cpu

# neural (GPU)
run corerec  NCF       cuda
run corerec  LightGCN  cuda
run corerec  GNNRec    cuda
run corerec  DCN       cuda
run corerec  DeepFM    cuda
run cornac   NeuMF     cuda
run cornac   LightGCN  cuda
run cornac   NGCF      cuda
run cornac   VAECF     cuda

echo "SAME-MODEL MATRIX DONE"
