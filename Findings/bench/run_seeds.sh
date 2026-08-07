#!/usr/bin/env bash
# Run the sampled-negative models over several seeds.
#
# A single run of these is not meaningful: benchmarking the same code on two
# machines moved implicit's BPR by 5.2%, corerec's NCF_binary by 3.4% and
# LightGCN by 2.5%. ALS and ItemKNN are included as controls -- they are
# deterministic, so their spread should be zero, which is what makes the spread
# on the others attributable to negative sampling rather than the harness.
#
# The evaluation cohort is fixed at seed 42 inside runner.py regardless of
# --seed, so every run scores the same 300 users.
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python}"
OUT="$HERE/results/seeds"
SEEDS="${SEEDS:-1 2 3 4 5}"

mkdir -p "$OUT"
cd "$HERE" || exit 1

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

JOBS=(
  "implicit:BPR"
  "corerec:NCF_binary"
  "corerec:LightGCN"
  "implicit:ALS"          # control: deterministic
  "implicit:ItemKNN"      # control: deterministic
)

for spec in "${JOBS[@]}"; do
  fw="${spec%%:*}"; m="${spec##*:}"
  for s in $SEEDS; do
    out="$OUT/${fw}_${m}_seed${s}.json"
    if [ -s "$out" ]; then echo "skip $fw/$m seed=$s"; continue; fi
    start=$(date +%s)
    if "$PY" runner.py --framework "$fw" --model "$m" --dataset ml100k \
         --seed "$s" --out "$out" > "$OUT/${fw}_${m}_seed${s}.log" 2>&1; then
      echo "OK   $fw/$m seed=$s ($(( $(date +%s) - start ))s)"
    else
      echo "FAIL $fw/$m seed=$s"
    fi
  done
done
echo "SEEDS COMPLETE"
