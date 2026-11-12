#!/bin/bash
# ASM (a8w8) vs Gluon (a16w8) MLA decode microbench on Kimi-K2.6 TP8 shapes:
# batch 64, 8 heads per rank, kv_lora 512 + rope 64, fp8 KV cache, block size 64,
# 3 GiB cache per layer (Kimi serves ~2.93 GiB/layer, so Gluon takes its >2 GiB path).
# KV lengths per request ~ U[mean*(1-r), mean*(1+r)]: r = 0 balanced, r = 0.8 imbalanced.
# Usage: bash run_mla_microbench.sh [gpu]
set -euo pipefail

export ROCR_VISIBLE_DEVICES="${1:-7}"
OUT_DIR="mla_microbench_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

for mean in 1024 8192; do
    for ratio in 0 0.8; do
        python bench_mla_decode_kernels.py \
            --batch 64 --kv-mean "$mean" --kv-range-ratio "$ratio" \
            --kv-pool-gib 3 --kernels asm gluon --profile \
            --out "$OUT_DIR/kv${mean}_r${ratio}.json" 2>&1 \
            | grep -vE "queue-intercept|USDT|^\[aiter\]" | tee "$OUT_DIR/kv${mean}_r${ratio}.txt"
    done
done

python - "$OUT_DIR" <<'EOF'
import glob, json, sys
rows = []
for path in sorted(glob.glob(f"{sys.argv[1]}/kv*.json")):
    d = json.load(open(path))
    a = d["args"]
    res = {r["kernel"]: r for r in d["results"] if r["config"] == "default"}
    rows.append((a["kv_mean"], a["kv_range_ratio"], a["kv_min"], a["kv_max"], d["mean_kv_len"], res))
print()
print("| mean kv | ratio | kv range | asm graph us | gluon graph us | gluon vs asm "
      "| asm kernels us | gluon kernels us | asm max err | gluon max err |")
print("|---|---|---|---|---|---|---|---|---|---|")
for mean, ratio, lo, hi, real_mean, res in sorted(rows):
    a, g = res["asm"], res["gluon"]
    delta = 100 * (g["us_per_call"] - a["us_per_call"]) / a["us_per_call"]
    print(f"| {mean} ({real_mean:.0f}) | {ratio} | {lo}..{hi} | {a['us_per_call']:.2f} | "
          f"{g['us_per_call']:.2f} | {delta:+.1f}% | {a.get('kernel_us_per_call', 0):.2f} | "
          f"{g.get('kernel_us_per_call', 0):.2f} | {a['max_abs']:.1e} | {g['max_abs']:.1e} |")
EOF
