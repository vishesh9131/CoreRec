#!/bin/sh
# RecBole reference implementations of the deep models CoreRec also ships, on the
# SAME ML-1M@100k split (atomic files). Establishes whether CoreRec's deep models
# match a reference framework's. GPU.
set -u
cd "$(dirname "$0")"
SIZE=100000
mkdir -p results/same
unset CUDA_VISIBLE_DEVICES

# DCN/DeepFM are CTR models in RecBole (AUC/LogLoss eval, not top-K ranking),
# so they are not directly comparable on Recall/NDCG and are excluded here.
for model in NeuMF LightGCN NGCF SASRec BERT4Rec; do
  out="results/same/recbole_${model}.json"
  if [ -f "$out" ]; then echo "skip $out"; continue; fi
  echo "### recbole $model"
  python -W ignore recbole_runner.py --model "$model" --size "$SIZE" \
      --device cuda --out "$out" 2>/dev/null \
      && echo "  ok" || echo "  FAILED recbole $model"
done
echo "RECBOLE DEEP DONE"
