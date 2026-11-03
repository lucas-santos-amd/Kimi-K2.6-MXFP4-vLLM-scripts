#!/bin/bash
set -euo pipefail

# needs the server started with serve_kimi_trace.sh (--profiler-config must be
# set on the SERVER; --profile here only triggers start/stop around the run)

##################### need changed #######################
MODEL_PATH="${1:-/data/models/Kimi-K2.6-MXFP4}"
ISL_LIST=(1024)
CONC_LIST=(64)
OSL=1024
RANGE_RATIO=0
PORT=8002
##########################################################

LOG_FILE="result_profile.txt"
if [ -f "$LOG_FILE" ]; then
    rm "$LOG_FILE"
fi

curl -sf "http://localhost:$PORT/health" > /dev/null || {
    echo "ERROR: Server not running on port $PORT"
    exit 1
}

for ISL in "${ISL_LIST[@]}"; do

    for CONC in "${CONC_LIST[@]}"; do
    echo "=========================================" | tee -a "$LOG_FILE"
    echo "Start Test ISL=$ISL, OSL=$OSL, CONC=$CONC" | tee -a "$LOG_FILE"
    echo "=========================================" | tee -a "$LOG_FILE"

        vllm bench serve \
            --model="$MODEL_PATH" \
            --backend=vllm \
            --trust-remote-code \
            --base-url="http://localhost:$PORT" \
            --dataset-name=random \
            --random-input-len="$ISL" \
            --random-output-len="$OSL" \
            --random-range-ratio "$RANGE_RATIO" \
            --num-prompts="$CONC" \
            --max-concurrency="$CONC" \
            --request-rate=inf \
            --ignore-eos \
            --temperature=0 \
            --profile \
            --percentile-metrics="ttft,tpot,itl,e2el"  2>&1 | tee -a "$LOG_FILE"
			
		# vllm bench serve \
			# --model="$MODEL_PATH" \
			# --backend=vllm \
			# --trust-remote-code \
			# --base-url="http://localhost:$PORT" \
			# --dataset-name=sharegpt \
			# --dataset-path=/triton_dev/kimi-k2.6/ShareGPT_V3_unfiltered_cleaned_split.json \
			# --sharegpt-output-len="$OSL" \
			# --num-prompts="$CONC" \
			# --max-concurrency="$CONC" \
			# --request-rate=inf \
			# --ignore-eos \
			# --temperature=0 \
			# --profile \
			# --percentile-metrics="ttft,tpot,itl,e2el"  2>&1 | tee -a "$LOG_FILE"

        echo -e "\n" | tee -a "$LOG_FILE"
    done
done

ls -lh "${TRACE_DIR:-/triton_dev/hhome/traces}"
