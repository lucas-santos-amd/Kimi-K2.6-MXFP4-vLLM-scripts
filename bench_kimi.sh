#!/bin/bash
set -euo pipefail

##################### need changed #######################
MODEL_PATH="${1:-/data/models/Kimi-K2.6-MXFP4}"
ISL_LIST=(1024)
CONC_LIST=(64)
OSL=1024
PORT=8002
##########################################################

# NOTE: vLLM reads --random-range-ratio as [len*(1-r), len*(1+r)]; ATOM reads it
# as [len*r, len]. To compare against ATOM, run ATOM's client against this server.
LOG_FILE="result.txt"
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
            --random-range-ratio 0 \
            --num-prompts=$(( CONC * 10 )) \
            --max-concurrency="$CONC" \
            --request-rate=inf \
            --ignore-eos \
            --temperature=0 \
            --save-result --result-dir . \
            --percentile-metrics="ttft,tpot,itl,e2el"  2>&1 | tee -a "$LOG_FILE"

        echo -e "\n" | tee -a "$LOG_FILE"
    done
done
