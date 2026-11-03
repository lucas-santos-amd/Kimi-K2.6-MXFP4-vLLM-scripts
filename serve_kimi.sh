MODEL_PATH="${1:-/data/models/Kimi-K2.6-MXFP4}"
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1

USE_INDUCTOR_GRAPH_PARTITION="-cc.use_inductor_graph_partition=True"
FUSE_ROPE_KVCACHE="-cc.pass_config.fuse_rope_kvcache_cat_mla=True"

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
    --trust-remote-code

# -cc.pass_config.fuse_allreduce_rms=False
