#!/bin/bash
# ASM (a8w8) vs Gluon (a16w8) MLA decode microbench on Kimi-K2.6 TP8 shapes:
# batch 64, 8 heads per rank, kv_lora 512 + rope 64, fp8 KV cache, block size 64,
# 3 GiB cache per layer (Kimi serves ~2.93 GiB/layer, so Gluon takes its >2 GiB path).
# KV lengths per request ~ U[mean*(1-r), mean*(1+r)]: r = 0 balanced, r = 0.8 imbalanced.
# Gluon runs today's split (default) and the split rebalance with its partition
# built once outside the timed graph (meta; its metadata kernel is timed alone).
# Needs AITER branch mla_gluon_split_rebalance.
# Usage: bash run_mla_microbench.sh [gpu] [heads]   (heads 16 = DeepSeek-R1 TP8)
set -euo pipefail

export ROCR_VISIBLE_DEVICES="${1:-7}"
HEADS="${2:-8}"
OUT_DIR="mla_microbench_h${HEADS}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

for mean in 1024 8192; do
    for ratio in 0 0.8; do
        python bench_mla_decode_kernels.py \
            --batch 64 --heads "$HEADS" --kv-mean "$mean" --kv-range-ratio "$ratio" \
            --kv-pool-gib 3 --kernels asm gluon --gluon-split-meta \
            --profile --out "$OUT_DIR/kv${mean}_r${ratio}.json" 2>&1 \
            | grep -vE "queue-intercept|USDT|^\[aiter\]" | tee "$OUT_DIR/kv${mean}_r${ratio}.txt"
    done
done

python - "$OUT_DIR" <<'EOF'
import glob, json, sys
rows = []
for path in sorted(glob.glob(f"{sys.argv[1]}/kv*.json")):
    d = json.load(open(path))
    a = d["args"]
    res = {(r["kernel"], r["config"]): r for r in d["results"]}
    rows.append((a["kv_mean"], a["kv_range_ratio"], a["kv_min"], a["kv_max"], d["mean_kv_len"], res))

cols = sorted({k for *_, res in rows for k in res if k[0] == "gluon"},
              key=lambda k: (k[1] != "default", k[1].startswith("meta"), k[1]))
names = [k[1] for k in cols]
print()
print("CUDA-graph us per call (Gluon vs ASM in brackets), then summed kernel us per call;")
print("meta excludes its metadata kernel, which runs once per decode step (last column)")
print("| mean kv | ratio | kv range | asm | " + " | ".join(f"gluon {n}" for n in names)
      + " | asm kern | " + " | ".join(f"{n} kern" for n in names) + " | metadata us |")
print("|---" * (5 + 2 * len(names) + 1) + "|")
for mean, ratio, lo, hi, real_mean, res in sorted(rows):
    asm = res[("asm", "default")]
    graph, kern = [], []
    for key in cols:
        r = res.get(key)
        if r is None:
            graph.append("-"); kern.append("-")
            continue
        delta = 100 * (r["us_per_call"] - asm["us_per_call"]) / asm["us_per_call"]
        graph.append(f"{r['us_per_call']:.2f} ({delta:+.1f}%)")
        kern.append(f"{r.get('kernel_us_per_call', 0):.2f}")
    meta = res.get(("gluon_split_metadata", "meta"))
    meta_us = f"{meta['us_per_call']:.2f}" if meta else "-"
    print(f"| {mean} ({real_mean:.0f}) | {ratio} | {lo}..{hi} | {asm['us_per_call']:.2f} | "
          + " | ".join(graph) + f" | {asm.get('kernel_us_per_call', 0):.2f} | "
          + " | ".join(kern) + f" | {meta_us} |")
EOF
