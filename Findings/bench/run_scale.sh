#!/bin/sh
# Scalability sweep: subsample ML-1M to increasing interaction counts and run
# every CPU framework + CoreRec SAR at each scale. CPU-only, 4 threads, fair.
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
mkdir -p results/scale

SIZES="10000 50000 100000 500000 1000000"
SPECS="implicit:ALS implicit:BPR lightfm:WARP cornac:BPR cornac:MF surprise:SVD corerec:SAR"

for size in $SIZES; do
  for spec in $SPECS; do
    fw=$(echo "$spec" | cut -d: -f1)
    md=$(echo "$spec" | cut -d: -f2)
    out="results/scale/${fw}_${md}_${size}.json"
    if [ -f "$out" ]; then echo "skip $out"; continue; fi
    echo "### size=$size $fw $md"
    python -W ignore runner.py --framework "$fw" --model "$md" \
        --dataset ml1m --size "$size" --out "$out" 2>/dev/null \
        && echo "  ok" || echo "  FAILED $fw $md @ $size"
  done
done
echo "SWEEP DONE"
