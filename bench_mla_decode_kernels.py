"""Kimi-K2.6 TP8 MLA decode on MI355X: asm a8w8 vs AITER Triton vs AITER Gluon.

All three read the same fp8 KV cache (block_size-token blocks, random block
assignment, shared block table across layers like vLLM):

- asm:    aiter.mla.mla_decode_fwd, persistent metadata, fp8 q padded 8 -> 16
          heads (repeat_interleave), page size 1 token indices. Same call as
          vLLM's ROCM_AITER_MLA decode today.
- triton: aiter.ops.triton.attention.mla.mla_decode_fwd, fp8 q (8 or 16 heads),
          TILE_SIZE = block size, block table.
- gluon:  aiter.ops.triton.gluon.mla_gluon.mla_gluon, bf16 q (8 heads),
          regime bh16bn128, kv_indices + kv_indptr.

Timing: one CUDA graph runs the op on --layers distinct caches (so the working
set is larger than the 256 MB MALL); reported time = graph time / layers, which
includes the gaps between launches. --profile also prints the summed kernel
time per call (what the trace rows report), profiled on eager calls.

Sweeps (local monkeypatches, no AITER change):
- --triton-sweep: num_warps x waves_per_eu x num_stages x segments.
- --gluon-splits: NUM_KV_SPLITS values.
- --asm-sweep: kv_granularity x max_split_per_batch x fast_mode of the asm
  persistent metadata.

--kv-pool-gib sizes each layer's cache (blocks spread over the whole pool);
above 2 GiB Gluon takes its 64-bit global_load path.
"""

import argparse
import importlib
import inspect
import itertools
import json
import math

import torch

from aiter import dtypes, get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.mla import mla_decode_fwd as asm_mla_decode_fwd

triton_mla = importlib.import_module("aiter.ops.triton.attention.mla")
gluon_mod = importlib.import_module("aiter.ops.triton.gluon.mla_gluon")

KV_LORA = 512
ROPE = 64
D = KV_LORA + ROPE
ASM_HEADS = 16
FP8 = dtypes.fp8
FP8_MAX = torch.finfo(FP8).max
DEV = "cuda"


def build_inputs(args):
    g = torch.Generator(device=DEV).manual_seed(args.seed)
    B, P = args.batch, args.block_size
    seq_lens = torch.randint(
        args.kv_min, args.kv_max + 1, (B,), generator=g, device=DEV, dtype=torch.int32
    )
    blocks_per_req = (seq_lens + P - 1) // P
    max_blocks = int(blocks_per_req.max())
    num_blocks = max(B * max_blocks + 16, math.ceil(args.kv_pool_gib * 2**30 / (P * D)))
    perm = torch.randperm(num_blocks, generator=g, device=DEV).to(torch.int32)
    block_table = perm[: B * max_blocks].view(B, max_blocks).contiguous()

    kv_indptr = torch.zeros(B + 1, dtype=torch.int32, device=DEV)
    kv_indptr[1:] = torch.cumsum(seq_lens, 0)
    tok = torch.arange(int(seq_lens.max()), device=DEV, dtype=torch.int32)
    kv_indices = torch.cat(
        [
            block_table[i, tok[:L] // P] * P + tok[:L] % P
            for i, L in enumerate(seq_lens.tolist())
        ]
    ).contiguous()

    k_scale = torch.tensor([args.k_scale], dtype=torch.float32, device=DEV)
    q_scale = torch.tensor([args.q_scale], dtype=torch.float32, device=DEV)
    layers = []
    chunk = max(1, 2**28 // (P * D * 4))
    for _ in range(args.layers):
        kv_fp8 = torch.empty(num_blocks, P, D, dtype=FP8, device=DEV)
        for s in range(0, num_blocks, chunk):
            n = min(chunk, num_blocks - s)
            kv = torch.randn(n, P, D, generator=g, device=DEV) * 0.5
            kv_fp8[s : s + n] = (kv / args.k_scale).clamp(-FP8_MAX, FP8_MAX).to(FP8)
        q = (torch.randn(B, args.heads, D, generator=g, device=DEV) * 0.5).to(
            torch.bfloat16
        )
        q_fp8 = (q.float() / args.q_scale).clamp(-FP8_MAX, FP8_MAX).to(FP8)
        layers.append(dict(kv=kv_fp8, q_bf16=q, q_fp8=q_fp8))

    qo_indptr = torch.arange(B + 1, dtype=torch.int32, device=DEV)
    return dict(
        seq_lens=seq_lens,
        block_table=block_table,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        qo_indptr=qo_indptr,
        last_page_len=torch.ones(B, dtype=torch.int32, device=DEV),
        k_scale=k_scale,
        q_scale=q_scale,
        num_blocks=num_blocks,
        layers=layers,
        max_seqlen_kv=args.max_seqlen_kv or int(seq_lens.max()),
    )


def reference(inp, layer, sm_scale, q_kind, args):
    """fp32 attention on the dequantized cache; q as the kernel sees it."""
    kv = layer["kv"].view(-1, D)
    if q_kind == "fp8":
        q = layer["q_fp8"].float() * args.q_scale
    else:
        q = layer["q_bf16"].float()
    out = torch.empty(q.shape[0], q.shape[1], KV_LORA, device=DEV)
    ip = inp["kv_indptr"].tolist()
    for i in range(q.shape[0]):
        k = kv[inp["kv_indices"][ip[i] : ip[i + 1]].long()].float() * args.k_scale
        p = torch.softmax((q[i] @ k.T) * sm_scale, dim=-1)
        out[i] = p @ k[:, :KV_LORA]
    return out


def make_asm(inp, args, sm_scale, kv_granularity=16, max_split=-1, fast_mode=True):
    """Defaults are what vLLM passes to get_mla_metadata_v1 today."""
    B = args.batch
    info = get_mla_metadata_info_v1(
        B,
        1,
        ASM_HEADS,
        FP8,
        FP8,
        is_sparse=False,
        fast_mode=fast_mode,
        max_split_per_batch=max_split,
    )
    work = [torch.empty(size, dtype=dt, device=DEV) for size, dt in info]
    get_mla_metadata_v1(
        inp["qo_indptr"],
        inp["kv_indptr"],
        inp["last_page_len"],
        ASM_HEADS,
        1,
        True,
        work[0],
        work[2],
        work[1],
        work[3],
        work[4],
        work[5],
        page_size=1,
        kv_granularity=kv_granularity,
        max_seqlen_qo=1,
        uni_seqlen_qo=1,
        fast_mode=fast_mode,
        max_split_per_batch=max_split,
        dtype_q=FP8,
        dtype_kv=FP8,
    )
    rep = ASM_HEADS // args.heads
    qs = [
        layer["q_fp8"].repeat_interleave(rep, dim=1).contiguous()
        for layer in inp["layers"]
    ]
    outs = [
        torch.empty(B, ASM_HEADS, KV_LORA, dtype=torch.bfloat16, device=DEV)
        for _ in inp["layers"]
    ]

    def run():
        for layer, q, o in zip(inp["layers"], qs, outs):
            asm_mla_decode_fwd(
                q,
                layer["kv"].view(-1, 1, 1, D),
                o,
                inp["qo_indptr"],
                inp["kv_indptr"],
                inp["kv_indices"],
                inp["last_page_len"],
                1,
                sm_scale=sm_scale,
                logit_cap=0.0,
                q_scale=inp["q_scale"],
                kv_scale=inp["k_scale"],
                causal=True,
                work_meta_data=work[0],
                work_indptr=work[1],
                work_info_set=work[2],
                reduce_indptr=work[3],
                reduce_final_map=work[4],
                reduce_partial_map=work[5],
            )

    return run, lambda: outs[0][:, ::rep].float()


def make_triton(inp, args, sm_scale):
    B, P = args.batch, args.block_size
    heads = args.triton_heads
    rep = heads // args.heads
    qs = [
        layer["q_fp8"].repeat_interleave(rep, dim=1).contiguous()
        for layer in inp["layers"]
    ]
    outs = [
        torch.empty(B, heads, KV_LORA, dtype=torch.bfloat16, device=DEV)
        for _ in inp["layers"]
    ]

    def run():
        for layer, q, o in zip(inp["layers"], qs, outs):
            triton_mla.mla_decode_fwd(
                q,
                layer["kv"].view(inp["num_blocks"], P, 1, D),
                o,
                cu_seqlens_q=inp["qo_indptr"],
                seqused_k=inp["seq_lens"],
                max_seqlen_kv=inp["max_seqlen_kv"],
                block_tables=inp["block_table"],
                softmax_scale=sm_scale,
                kv_lora_rank=KV_LORA,
                qk_rope_head_dim=ROPE,
                causal=True,
                q_descale=inp["q_scale"],
                kv_descale=inp["k_scale"],
            )

    return run, lambda: outs[0][:, ::rep].float()


def make_gluon(inp, args, sm_scale):
    B = args.batch
    outs = [
        torch.empty(B, args.heads, KV_LORA, dtype=torch.bfloat16, device=DEV)
        for _ in inp["layers"]
    ]
    min_len = int(inp["seq_lens"].min())

    def run():
        for layer, o in zip(inp["layers"], outs):
            q_nope, q_pe = torch.split(layer["q_bf16"], [KV_LORA, ROPE], dim=-1)
            gluon_mod.mla_gluon(
                q_nope=q_nope,
                q_pe=q_pe,
                kv_c=layer["kv"].view(-1, D),
                o=o,
                page_table=inp["kv_indices"],
                seq_info=inp["kv_indptr"],
                sm_scale=sm_scale,
                k_pe=None,
                kv_pe_offset=KV_LORA,
                use_2d_view=False,
                kv_scale=args.k_scale,
                min_kv_seq_len=min_len,
            )

    return run, lambda: outs[0].float()


def time_graph(run, layers, iters, reps):
    run()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        run()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000 / (iters * layers))
    times.sort()
    return times[len(times) // 2], graph


def kernel_time(run, layers, iters):
    """Summed device time per call, profiled eagerly (same kernels as the graph)."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            run()
        torch.cuda.synchronize()
    rows = []
    for e in prof.key_averages():
        t = getattr(e, "self_device_time_total", None)
        if t is None:
            t = getattr(e, "self_cuda_time_total", 0)
        if t > 0:
            rows.append((e.key, t / (iters * layers)))
    rows.sort(key=lambda r: -r[1])
    return sum(t for _, t in rows), rows


def check(name, out, ref):
    err = (out - ref).abs()
    return dict(max_abs=float(err.max()), mean_abs=float(err.mean()))


def run_one(name, maker, inp, args, sm_scale, refs, tag=""):
    run, first_out = maker(inp, args, sm_scale)
    us, _ = time_graph(run, args.layers, args.iters, args.reps)
    res = dict(kernel=name, config=tag, us_per_call=us)
    res.update(check(name, first_out(), refs["bf16" if name == "gluon" else "fp8"]))
    line = (
        f"{name:7s} {tag:48s} {us:8.2f} us/call (graph)  "
        f"max|err| {res['max_abs']:.2e}  mean|err| {res['mean_abs']:.2e}"
    )
    if args.profile:
        k_us, rows = kernel_time(run, args.layers, args.iters)
        res["kernel_us_per_call"] = k_us
        res["kernels"] = rows
        line += f"  kernels {k_us:7.2f} us/call"
        for k, t in rows:
            line += f"\n          {t:7.2f} us  {k[:110]}"
    print(line, flush=True)
    return res


def triton_sweep(inp, args, sm_scale, refs):
    orig = triton_mla.select_3d_config
    results = []
    max_segments = max(1, math.ceil(inp["max_seqlen_kv"] / args.block_size))
    segs = [s for s in (1, 2, 4, 8, 16, 32) if s <= triton_next_pow2(max_segments)]
    for nw, wpe, ns, seg in itertools.product((2, 4, 8), (1, 2), (1, 2, 3), segs):

        def patched(*a, _o=(nw, wpe, ns, seg), **k):
            attn, red = orig(*a, **k)
            attn.update(
                num_warps=_o[0],
                waves_per_eu=_o[1],
                num_stages=_o[2],
                NUM_SEGMENTS_PER_SEQ=_o[3],
            )
            red["NUM_SEGMENTS_PER_SEQ"] = _o[3]
            return attn, red

        triton_mla.select_3d_config = patched
        tag = f"warps{nw} wpe{wpe} stages{ns} segs{seg}"
        try:
            results.append(
                run_one("triton", make_triton, inp, args, sm_scale, refs, tag)
            )
        except Exception as exc:  # noqa: BLE001
            print(f"triton  {tag:48s} FAILED: {type(exc).__name__}: {exc}"[:200])
        finally:
            triton_mla.select_3d_config = orig
    return results


def triton_next_pow2(n):
    return 1 << (n - 1).bit_length()


def asm_sweep(inp, args, sm_scale, refs):
    """Persistent-metadata knobs of the existing asm decode (no kernel change)."""
    results = []
    for gran, max_split, fast in itertools.product(
        (16, 32, 64, 128, 256), (-1, 1, 2, 4, 8, 16), (True, False)
    ):

        def maker(i, a, s, _o=(gran, max_split, fast)):
            return make_asm(
                i, a, s, kv_granularity=_o[0], max_split=_o[1], fast_mode=_o[2]
            )

        tag = f"gran{gran} max_split{max_split} fast{int(fast)}"
        try:
            results.append(run_one("asm", maker, inp, args, sm_scale, refs, tag))
        except Exception as exc:  # noqa: BLE001
            print(f"asm     {tag:48s} FAILED: {type(exc).__name__}: {exc}"[:200])
    return results


_GLUON_BH16_SPLITS = (
    "256 // (batch_size * qlen * NUM_M_BLOCKS)",
    "get_num_sms() // (batch_size * qlen * NUM_M_BLOCKS)",
)


def gluon_with_splits(splits):
    """mla_gluon with the bh16 split count pinned (older AITER hardcodes 256 CUs)."""
    src = inspect.getsource(gluon_mod.mla_gluon)
    old = next((s for s in _GLUON_BH16_SPLITS if s in src), None)
    if old is None:
        raise RuntimeError("mla_gluon bh16 split formula not found")
    namespace = dict(vars(gluon_mod))
    exec(src.replace(old, str(splits)), namespace)  # noqa: S102
    return namespace["mla_gluon"]


def gluon_splits(inp, args, sm_scale, refs):
    orig = gluon_mod.mla_gluon
    results = []
    for splits in args.gluon_splits:
        try:
            gluon_mod.mla_gluon = gluon_with_splits(splits)
            results.append(
                run_one(
                    "gluon", make_gluon, inp, args, sm_scale, refs, f"splits{splits}"
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"gluon   splits{splits} FAILED: {type(exc).__name__}: {exc}"[:200])
        finally:
            gluon_mod.mla_gluon = orig
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--heads", type=int, default=8, help="real q heads per rank")
    p.add_argument("--triton-heads", type=int, default=8, choices=(8, 16))
    p.add_argument("--kv-min", type=int, default=1024)
    p.add_argument("--kv-max", type=int, default=2048)
    p.add_argument(
        "--kv-mean",
        type=int,
        default=0,
        help="if set, overrides --kv-min/--kv-max with "
        "[mean*(1-r), mean*(1+r)], r = --kv-range-ratio",
    )
    p.add_argument(
        "--kv-range-ratio",
        type=float,
        default=0.0,
        help="spread around --kv-mean, as vllm bench --random-range-ratio",
    )
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument(
        "--max-seqlen-kv",
        type=int,
        default=0,
        help="host max_seqlen_kv for Triton (0 = batch max); vLLM's graph value",
    )
    p.add_argument(
        "--kv-pool-gib",
        type=float,
        default=0.0,
        help="per-layer cache size (0 = just what the batch needs)",
    )
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--k-scale", type=float, default=1.0)
    p.add_argument("--q-scale", type=float, default=1.0)
    p.add_argument("--sm-scale", type=float, default=192**-0.5)
    p.add_argument("--kernels", nargs="+", default=["asm", "triton", "gluon"])
    p.add_argument("--triton-sweep", action="store_true")
    p.add_argument("--asm-sweep", action="store_true")
    p.add_argument("--gluon-splits", type=int, nargs="*", default=[])
    p.add_argument("--profile", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="")
    args = p.parse_args()
    if args.kv_mean:
        args.kv_min = max(1, int(args.kv_mean * (1 - args.kv_range_ratio)))
        args.kv_max = max(args.kv_min, int(args.kv_mean * (1 + args.kv_range_ratio)))

    inp = build_inputs(args)
    sm_scale = args.sm_scale
    refs = {
        "fp8": reference(inp, inp["layers"][0], sm_scale, "fp8", args),
        "bf16": reference(inp, inp["layers"][0], sm_scale, "bf16", args),
    }
    mean_len = float(inp["seq_lens"].float().mean())
    print(
        f"B={args.batch} heads={args.heads} kv {args.kv_min}..{args.kv_max} "
        f"(mean {mean_len:.0f}, max {int(inp['seq_lens'].max())}) "
        f"block_size={args.block_size} layers={args.layers} "
        f"cache/layer {inp['num_blocks'] * args.block_size * D / 2**30:.2f} GiB "
        f"k_scale={args.k_scale} q_scale={args.q_scale}",
        flush=True,
    )
    makers = dict(asm=make_asm, triton=make_triton, gluon=make_gluon)
    results = [
        run_one(k, makers[k], inp, args, sm_scale, refs, "default")
        for k in args.kernels
    ]
    if args.triton_sweep:
        results += triton_sweep(inp, args, sm_scale, refs)
    if args.asm_sweep:
        results += asm_sweep(inp, args, sm_scale, refs)
    if args.gluon_splits:
        results += gluon_splits(inp, args, sm_scale, refs)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                dict(args=vars(args), mean_kv_len=mean_len, results=results),
                f,
                indent=1,
            )


if __name__ == "__main__":
    main()
