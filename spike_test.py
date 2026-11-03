"""Spike: does a Triton rope + KV-cache write reproduce vLLM's
concat_and_cache_mla_rope_fused bit for bit?

Run on the MI355X inside the vLLM environment:
    python spike_rope_kv_numerics.py
"""

import torch
import triton
import triton.language as tl

from vllm import _custom_ops as ops
from vllm.platforms import current_platform


MODES = {
    0: "bf16 after every op, default cast",
    1: "bf16 after every op, explicit RNE",
    2: "fp32, one rounding, no FMA",
    3: "fp32 FMA: fma(x,c,-y*s), fma(y,c,x*s)",
    4: "fp32 FMA: fma(-y,s,x*c), fma(x,s,y*c)",
    5: "fma(x,c,-bf16(y*s)), fma(y,c,bf16(x*s))",
    6: "fma(x,c,-bf16(y*s)), fma(x,s,bf16(y*c))",
}


@triton.jit
def _rne(v):
    return v.to(tl.bfloat16, fp_downcast_rounding="rtne")


@triton.jit
def _rope_pair_bf16(x, y, c, s, MODE: tl.constexpr):
    f32 = tl.float32
    x = x.to(f32)
    y = y.to(f32)
    c = c.to(f32)
    s = s.to(f32)
    if MODE == 0:
        xc = (x * c).to(tl.bfloat16).to(f32)
        ys = (y * s).to(tl.bfloat16).to(f32)
        yc = (y * c).to(tl.bfloat16).to(f32)
        xs = (x * s).to(tl.bfloat16).to(f32)
        x_out = (xc - ys).to(tl.bfloat16)
        y_out = (yc + xs).to(tl.bfloat16)
    elif MODE == 1:
        x_out = _rne(_rne(x * c).to(f32) - _rne(y * s).to(f32))
        y_out = _rne(_rne(y * c).to(f32) + _rne(x * s).to(f32))
    elif MODE == 2:
        x_out = _rne(x * c - y * s)
        y_out = _rne(y * c + x * s)
    elif MODE == 3:
        x_out = _rne(tl.fma(x, c, -(y * s)))
        y_out = _rne(tl.fma(y, c, x * s))
    elif MODE == 4:
        x_out = _rne(tl.fma(-y, s, x * c))
        y_out = _rne(tl.fma(x, s, y * c))
    elif MODE == 5:
        x_out = _rne(tl.fma(x, c, -_rne(y * s).to(f32)))
        y_out = _rne(tl.fma(y, c, _rne(x * s).to(f32)))
    else:
        x_out = _rne(tl.fma(x, c, -_rne(y * s).to(f32)))
        y_out = _rne(tl.fma(x, s, _rne(y * c).to(f32)))
    return x_out, y_out


@triton.jit
def _to_cache_fp8(v, scale, CACHE_MAX: tl.constexpr, out_ty: tl.constexpr):
    v = v.to(tl.float32)
    q = tl.math.div_rn(v, tl.zeros_like(v) + scale)
    return tl.clamp(q, -CACHE_MAX, CACHE_MAX).to(out_ty)


@triton.jit
def _rope_cache_kernel(
    pos_ptr,
    q_pe_ptr,
    k_pe_ptr,
    kv_c_ptr,
    cos_sin_ptr,
    slot_ptr,
    cache_ptr,
    scale_ptr,
    q_out_ptr,
    num_heads,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_ct,
    stride_slot,
    R: tl.constexpr,
    L: tl.constexpr,
    IS_NEOX: tl.constexpr,
    BLOCK_H: tl.constexpr,
    CACHE_MAX: tl.constexpr,
    MODE: tl.constexpr,
):
    t = tl.program_id(0)
    slot = tl.load(slot_ptr + t)
    if slot < 0:
        return  # vLLM's kernel leaves a padded token's q_pe and cache untouched
    pos = tl.load(pos_ptr + t)
    half: tl.constexpr = R // 2
    offs = tl.arange(0, half)
    c = tl.load(cos_sin_ptr + pos * R + offs)
    s = tl.load(cos_sin_ptr + pos * R + half + offs)
    if IS_NEOX:
        xi = offs
        yi = offs + half
    else:
        xi = 2 * offs
        yi = 2 * offs + 1

    h = tl.arange(0, BLOCK_H)
    hmask = (h < num_heads)[:, None]
    q_base = q_pe_ptr + t * stride_qt + h[:, None] * stride_qh
    o_base = q_out_ptr + t * stride_qt + h[:, None] * stride_qh
    qx = tl.load(q_base + xi[None, :], mask=hmask)
    qy = tl.load(q_base + yi[None, :], mask=hmask)
    qx, qy = _rope_pair_bf16(qx, qy, c[None, :], s[None, :], MODE)
    tl.store(o_base + xi[None, :], qx, mask=hmask)
    tl.store(o_base + yi[None, :], qy, mask=hmask)

    scale = tl.load(scale_ptr)
    out_ty = cache_ptr.dtype.element_ty
    kx = tl.load(k_pe_ptr + t * stride_kt + xi)
    ky = tl.load(k_pe_ptr + t * stride_kt + yi)
    kx, ky = _rope_pair_bf16(kx, ky, c, s, MODE)
    entry = cache_ptr + slot * stride_slot
    tl.store(entry + L + xi, _to_cache_fp8(kx, scale, CACHE_MAX, out_ty))
    tl.store(entry + L + yi, _to_cache_fp8(ky, scale, CACHE_MAX, out_ty))
    lo = tl.arange(0, L)
    kv = tl.load(kv_c_ptr + t * stride_ct + lo)
    tl.store(entry + lo, _to_cache_fp8(kv, scale, CACHE_MAX, out_ty))


def run(k_scale: float, is_neox: bool, num_tokens: int = 64, num_heads: int = 8):
    """Per mode, (q_pe bf16 mismatches, cache byte mismatches) against vLLM."""
    torch.manual_seed(0)
    dev, bf16 = "cuda", torch.bfloat16
    R, L, block_size, num_blocks, max_pos = 64, 512, 16, 256, 8192
    fp8 = current_platform.fp8_dtype()

    q = torch.randn(num_tokens, num_heads, 128 + R, dtype=bf16, device=dev)
    q_pe = q[..., 128:]  # strided view, as the MLA wrapper passes it
    kv_lora = torch.randn(num_tokens, L + R, dtype=bf16, device=dev)
    kv_c, k_pe = kv_lora.split([L, R], dim=-1)
    angles = torch.rand(max_pos, R // 2, device=dev) * 6.3
    cos_sin = torch.cat([angles.cos(), angles.sin()], dim=-1).to(bf16)
    positions = torch.randint(0, max_pos, (num_tokens,), device=dev)
    slots = torch.randperm(num_blocks * block_size, device=dev)[:num_tokens]
    slots[::7] = -1  # padded tokens are skipped
    scale = torch.tensor([k_scale], dtype=torch.float32, device=dev)

    # vLLM's HIP kernel: ropes q_pe / k_pe in place and fills the cache.
    q_ref = q.clone()
    k_pe_ref = k_pe.clone()
    cache_ref = torch.zeros(num_blocks, block_size, L + R, dtype=torch.uint8, device=dev)
    ops.concat_and_cache_mla_rope_fused(
        positions,
        q_ref[..., 128:],
        k_pe_ref,
        kv_c.contiguous(),
        cos_sin,
        is_neox,
        slots,
        cache_ref,
        "fp8",
        scale,
    )

    results = {}
    for mode in MODES:
        q_out = q.clone()  # padded tokens keep their unrotated q_pe, as in vLLM
        cache_tri = torch.zeros_like(cache_ref)
        cache_view = cache_tri.view(fp8).view(-1, L + R)
        _rope_cache_kernel[(num_tokens,)](
            positions,
            q_pe,
            k_pe,
            kv_c,
            cos_sin,
            slots,
            cache_view,
            scale,
            q_out[..., 128:],
            num_heads,
            q_pe.stride(0),
            q_pe.stride(1),
            k_pe.stride(0),
            kv_c.stride(0),
            cache_view.stride(0),
            R=R,
            L=L,
            IS_NEOX=is_neox,
            BLOCK_H=triton.next_power_of_2(num_heads),
            CACHE_MAX=torch.finfo(fp8).max,
            MODE=mode,
            enable_fp_fusion=False,
        )
        q_mism = (
            q_out[..., 128:].view(torch.int16) != q_ref[..., 128:].view(torch.int16)
        )
        cache_diff = (cache_tri != cache_ref).sum()
        results[mode] = (q_mism.sum().item(), cache_diff.item())
        if mode == 5 and k_scale == 1.0:
            per_token = q_mism.flatten(1).sum(1)
            bad = per_token.nonzero().flatten().tolist()
            print(f"    mode 5: {len(bad)} tokens with q_pe mismatches")
            for t in bad[:8]:
                head_rows = q_mism[t].any(-1).nonzero().flatten().tolist()
                ref_vs_input = (
                    q_ref[t, :, 128:].view(torch.int16) != q[t, :, 128:].view(torch.int16)
                ).sum().item()
                print(
                    f"      token {t:2d}: slot={slots[t].item():5d} "
                    f"pos={positions[t].item():5d} mismatches={per_token[t].item():3d}/512 "
                    f"heads={head_rows} vLLM changed {ref_vs_input}/512 of the input"
                )
    return results


def _rtz(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> bf16 rounding toward zero (drop the low 16 bits)."""
    return (x.view(torch.int32) & -65536).view(torch.float32).to(torch.bfloat16)


def _ulp_hist(a: torch.Tensor, b: torch.Tensor) -> dict[str, int]:
    da = a.view(torch.int16).to(torch.int32)
    db = b.view(torch.int16).to(torch.int32)
    same_sign = (da < 0) == (db < 0)
    gap = (da - db).abs()
    return {
        "0": int((gap == 0).sum()),
        "1": int(((gap == 1) & same_sign).sum()),
        "2": int(((gap == 2) & same_sign).sum()),
        ">2 or sign": int((((gap > 2) & same_sign) | ((~same_sign) & (gap > 0))).sum()),
    }


def diagnose(is_neox: bool, num_tokens: int = 64, num_heads: int = 8):
    """Which torch-emulated rope recipe does vLLM's q_pe match, and by how many ulps?"""
    torch.manual_seed(0)
    dev, bf16, f32 = "cuda", torch.bfloat16, torch.float32
    R, L, max_pos = 64, 512, 8192
    q = torch.randn(num_tokens, num_heads, 128 + R, dtype=bf16, device=dev)
    kv_lora = torch.randn(num_tokens, L + R, dtype=bf16, device=dev)
    kv_c, k_pe = kv_lora.split([L, R], dim=-1)
    angles = torch.rand(max_pos, R // 2, device=dev) * 6.3
    cos_sin = torch.cat([angles.cos(), angles.sin()], dim=-1).to(bf16)
    positions = torch.randint(0, max_pos, (num_tokens,), device=dev)
    slots = torch.arange(num_tokens, device=dev)
    cache = torch.zeros(num_tokens, 1, L + R, dtype=torch.uint8, device=dev)
    q_ref = q.clone()
    ops.concat_and_cache_mla_rope_fused(
        positions, q_ref[..., 128:], k_pe.clone(), kv_c.contiguous(), cos_sin,
        is_neox, slots, cache, "fp8", torch.ones(1, dtype=f32, device=dev),
    )
    pe_in, pe_out = q[..., 128:], q_ref[..., 128:]
    cs = cos_sin[positions].to(f32)
    c, s = cs[:, None, : R // 2], cs[:, None, R // 2 :]
    if is_neox:
        xi, yi = slice(0, R // 2), slice(R // 2, R)
    else:
        xi, yi = slice(0, R, 2), slice(1, R, 2)
    x, y = pe_in[..., xi].to(f32), pe_in[..., yi].to(f32)
    got = torch.cat([pe_out[..., xi], pe_out[..., yi]], dim=-1)

    def rne(v):
        return v.to(bf16)

    candidates = {
        "per-op RNE": (
            rne(rne(x * c).to(f32) - rne(y * s).to(f32)),
            rne(rne(y * c).to(f32) + rne(x * s).to(f32)),
        ),
        "fp32, single RNE": (rne(x * c - y * s), rne(y * c + x * s)),
        "per-op RTZ": (
            _rtz(_rtz(x * c).to(f32) - _rtz(y * s).to(f32)),
            _rtz(_rtz(y * c).to(f32) + _rtz(x * s).to(f32)),
        ),
        "fp32, single RTZ": (_rtz(x * c - y * s), _rtz(y * c + x * s)),
        "products RNE, sum RTZ": (
            _rtz(rne(x * c).to(f32) - rne(y * s).to(f32)),
            _rtz(rne(y * c).to(f32) + rne(x * s).to(f32)),
        ),
    }
    print(f"diagnose is_neox={is_neox}: ulp gap vs vLLM q_pe ({got.numel()} values)")
    for name, (xo, yo) in candidates.items():
        print(f"  {name:<22} {_ulp_hist(torch.cat([xo, yo], dim=-1), got)}")

    # Mismatching x outputs, with everything needed to rebuild them by hand.
    per_op = candidates["per-op RNE"][0]
    single = candidates["fp32, single RNE"][0]
    got_x = pe_out[..., xi]
    exact = x.double() * c.double() - y.double() * s.double()
    c_b, s_b = c.expand_as(x), s.expand_as(x)
    idx = (got_x.view(torch.int16) != single.view(torch.int16)).nonzero()[:12]
    print("  samples (x_out = x*cos - y*sin), token head pair:")
    for t, h, p in idx.tolist():
        print(
            f"    t={t:2d} h={h} p={p:2d} x={x[t, h, p].item():+.6e} y={y[t, h, p].item():+.6e} "
            f"cos={c_b[t, h, p].item():+.6e} sin={s_b[t, h, p].item():+.6e} | "
            f"vLLM={got_x[t, h, p].float().item():+.6e} "
            f"per-op={per_op[t, h, p].float().item():+.6e} "
            f"single={single[t, h, p].float().item():+.6e} "
            f"fp64={exact[t, h, p].item():+.6e}"
        )


if __name__ == "__main__":
    exact = set(MODES)
    for k_scale in (1.0, 0.37, 0.004):  # 0.004 forces fp8 saturation
        for is_neox in (False, True):
            print(f"k_scale={k_scale} is_neox={is_neox}")
            for mode, (q_diff, cache_diff) in run(k_scale, is_neox).items():
                print(
                    f"  mode {mode} ({MODES[mode]:<40}) "
                    f"q_pe mismatches={q_diff:6d}/32768  "
                    f"cache byte mismatches={cache_diff:6d}"
                )
                if q_diff or cache_diff:
                    exact.discard(mode)
    print("bit-exact in every case:", [MODES[m] for m in sorted(exact)] or "none")
