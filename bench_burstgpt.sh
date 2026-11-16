#!/bin/bash
# BurstGPT serving benchmark: request / response token counts from a production
# GPT-4 trace (synthetic prompt text). Short on average with a long tail
# (request p50 526, p99 4170, max 31407 tokens). --ignore-eos so every request
# generates its traced response length, same sample for every run.
# Dataset: wget https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_without_fails_2.csv
# Usage: bash bench_burstgpt.sh [model_path] [port] [csv_path]
set -euo pipefail

MODEL_PATH="${1:-/data/models/Kimi-K2.6-MXFP4}"
PORT="${2:-8002}"
DATASET_PATH="${3:-/triton_dev/kimi-k2.6/BurstGPT_without_fails_2.csv}"
OUT_DIR="results_burstgpt_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

for CONC in 16 32 64; do
    NAME="burstgpt_c${CONC}"
    vllm bench serve \
        --model="$MODEL_PATH" \
        --backend=vllm \
        --trust-remote-code \
        --base-url="http://localhost:$PORT" \
        --dataset-name=burstgpt \
        --dataset-path="$DATASET_PATH" \
        --num-prompts=1000 \
        --max-concurrency="$CONC" \
        --num-warmups="$CONC" \
        --request-rate=inf \
        --ignore-eos \
        --temperature=0 \
        --seed=0 \
        --percentile-metrics="ttft,tpot,itl,e2el" \
        --save-result \
        --result-dir="$OUT_DIR" \
        --result-filename="${NAME}.json" 2>&1 | tee "$OUT_DIR/${NAME}.txt"
done
