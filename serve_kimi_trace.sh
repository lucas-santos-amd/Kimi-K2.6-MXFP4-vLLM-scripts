#!/bin/bash

MODEL_PATH="${1:-/data/models/Kimi-K2.6-MXFP4}"
TRACE_DIR="${TRACE_DIR:-/triton_dev/hhome/traces_mla_11102026}"
# TRACE_DIR="${TRACE_DIR:-/triton_dev/hhome/traces_bf16_prefill_triton_bmm_10232026}"
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1
# export VLLM_ROCM_USE_AITER_FP4BMM=0

USE_INDUCTOR_GRAPH_PARTITION="-cc.use_inductor_graph_partition=True"
FUSE_ROPE_KVCACHE="-cc.pass_config.fuse_rope_kvcache_cat_mla=True"
# USE_AITER_MLA_FP8_PREFILL="-ac.use_aiter_mla_fp8_prefill=False"
	# $USE_AITER_MLA_FP8_PREFILL \
	
		# --attention-backend ROCM_AITER_MLA \
rm -rf ~/.cache/vllm /tmp/torchinductor*
vllm serve $MODEL_PATH \
    --tensor-parallel-size=8 --gpu-memory-utilization=0.9 \
    --dtype=auto --kv-cache-dtype=fp8 \
    --max-num-batched-tokens=32768 \
    $USE_INDUCTOR_GRAPH_PARTITION \
    $FUSE_ROPE_KVCACHE \
    --no-enable-prefix-caching \
    --disable-uvicorn-access-log \
    --max-model-len 32768 \
    --max-num-seqs 512 \
    --trust-remote-code \
    --port 8002 \
    --profiler-config "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"$TRACE_DIR\", \"torch_profiler_with_stack\": false}"
