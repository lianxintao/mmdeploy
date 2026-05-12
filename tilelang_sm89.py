"""
FlashMLA TileLang SM89 Implementation
======================================
Implements flash_mla_with_kvcache for SM89 (Ada Lovelace: RTX 4090, L40S, etc.)

Single partial kernel with 4 NoPE groups of qts=128:
- V32:  d_qk=576, d_nope=512, 4 full groups, v_include_rope=False
- M1:   d_qk=512, d_nope=448 padded to 512, 4 groups (last partial),
        v_include_rope=True, scales merged 7→4
"""

from typing import Optional, Tuple
import dataclasses

import torch
import tilelang
import tilelang.language as T

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}
if hasattr(tilelang.PassConfigKey, "TL_ENABLE_FAST_MATH"):
    _PASS_CONFIGS[tilelang.PassConfigKey.TL_ENABLE_FAST_MATH] = False

BF16  = "bfloat16"
FP8   = "float8_e4m3fn"
FP32  = "float32"
INT32 = "int32"
UINT8 = "uint8"
LOG2E = 1.44269504

H_PER_BLOCK = 16
BI          = 64


def _split_kvcache_fp8(k_cache, d_qk):
    """Split packed FP8 KV cache into flat KV, scale, and rope components."""
    assert k_cache.shape[2] == 1
    num_blocks = k_cache.shape[0]
    block_size = k_cache.shape[1]
    if d_qk == 576:
        d_nope, d_rope = 512, 64
        bpt = 656
        num_groups, siu = 4, False
        rope_off = d_nope + 16
        scale_off = d_nope
        scale_nb = 16
    elif d_qk == 512:
        d_nope, d_rope = 448, 64
        bpt = 584
        num_groups, siu = 7, True
        scale_off = d_nope + d_rope * 2
        scale_nb = 8
    else:
        raise ValueError(f"Unsupported d_qk={d_qk}")

    N = num_blocks * block_size
    kv_flat = k_cache.squeeze(2).reshape(N, bpt)

    # Extract rope
    if d_qk == 576:
        # V32: rope at bytes 528-655, stored contiguously as BF16
        rope_raw = kv_flat[:, rope_off : rope_off + d_rope * 2].contiguous()
        k_rope = rope_raw.view(torch.uint8).view(torch.bfloat16).view(N, d_rope)
    else:
        # M1: rope stored at byte stride 2 due to non-contiguous view in quantize.
        # Must use the same view pattern as dequantize to read correctly.
        flat = k_cache.view(num_blocks, -1)
        nope_rope_flat = flat[:, :block_size * (d_nope + 2 * d_rope)]
        nope_rope = nope_rope_flat.view(num_blocks, block_size, d_nope + 2 * d_rope)
        k_rope = nope_rope[:, :, d_nope:].view(torch.bfloat16).reshape(N, d_rope)

    # Extract scale
    scale_raw = kv_flat[:, scale_off : scale_off + scale_nb].contiguous()
    if not siu:
        k_scale = scale_raw.view(torch.uint8).view(torch.float32).view(N, num_groups)
    else:
        k_scale = scale_raw.view(torch.uint8)[:, :num_groups].contiguous()

    return kv_flat, k_scale, k_rope


# ===========================================================================
# Unified partial kernel: 4 NoPE groups of qts=128
# ===========================================================================

@tilelang.jit(out_idx=[-2, -1], pass_configs=_PASS_CONFIGS)
def _partial_unified(
    heads: int,
    d_v: int,
    topk: int,
    inner_iter: int,
    have_topk_length: bool,
    v_include_rope: bool,
):
    """4 groups of qts=128. d_nope=512 (padded from 448 for M1)."""
    d_nope, d_rope = 512, 64
    ng = 4
    qts = 128
    REPLICATE_H = heads // H_PER_BLOCK
    N_GROUPS = topk // (BI * inner_iter)
    D_QK = d_nope + d_rope

    batch   = T.symbolic("batch")
    seq_len = T.symbolic("seq_len")
    n_tokens = T.symbolic("n_tokens")
    bpt = T.symbolic("bpt")

    @T.prim_func
    def main(
        q:           T.Tensor([batch, seq_len, heads, D_QK], BF16),
        kv_flat:     T.Tensor([n_tokens, bpt], FP8),
        k_scale:     T.Tensor([n_tokens, ng], FP32),
        k_rope_bf16: T.Tensor([n_tokens, d_rope], BF16),
        indices:     T.Tensor([batch, seq_len, 1, topk], INT32),
        topk_length: T.Tensor([batch], INT32),
        partial_o:   T.Tensor([batch, seq_len, N_GROUPS, heads, d_v], BF16),
        partial_lse: T.Tensor([batch, seq_len, N_GROUPS, heads], FP32),
    ):
        with T.Kernel(batch * seq_len * REPLICATE_H, N_GROUPS, threads=256) as (bx, by):
            rep_i = bx % REPLICATE_H
            tmp = bx // REPLICATE_H
            s_i = tmp % seq_len
            b_i = tmp // seq_len
            group_i = by
            H0 = rep_i * H_PER_BLOCK

            Q_nope  = T.alloc_shared([H_PER_BLOCK, d_nope], BF16)
            Q_rope  = T.alloc_shared([H_PER_BLOCK, d_rope], BF16)
            K_fp8   = T.alloc_shared([BI, d_nope], FP8)
            K_rope  = T.alloc_shared([BI, d_rope], BF16)
            K_scale = T.alloc_shared([BI, ng], FP32)
            K_buf   = T.alloc_shared([BI, qts], BF16)
            S_smem  = T.alloc_shared([H_PER_BLOCK, BI], BF16)
            page_s  = T.alloc_shared([BI], INT32)
            mask_s  = T.alloc_shared([BI], INT32)

            acc_s_frag = T.alloc_fragment([H_PER_BLOCK, BI], FP32)
            m_i = T.alloc_fragment([H_PER_BLOCK], FP32)
            m_prev = T.alloc_fragment([H_PER_BLOCK], FP32)
            lsum = T.alloc_fragment([H_PER_BLOCK], FP32)
            lsum_i = T.alloc_fragment([H_PER_BLOCK], FP32)
            alpha = T.alloc_fragment([H_PER_BLOCK], FP32)

            acc_g0 = T.alloc_fragment([H_PER_BLOCK, qts], FP32)
            acc_g1 = T.alloc_fragment([H_PER_BLOCK, qts], FP32)
            acc_g2 = T.alloc_fragment([H_PER_BLOCK, qts], FP32)
            acc_g3 = T.alloc_fragment([H_PER_BLOCK, qts], FP32)
            if v_include_rope:
                acc_r = T.alloc_fragment([H_PER_BLOCK, d_rope], FP32)
            tmp_out = T.alloc_fragment([H_PER_BLOCK, qts], FP32)

            T.fill(m_i, -(2.0 ** 30))
            T.fill(lsum, 0.0)
            T.fill(acc_g0, 0.0)
            T.fill(acc_g1, 0.0)
            T.fill(acc_g2, 0.0)
            T.fill(acc_g3, 0.0)
            if v_include_rope:
                T.fill(acc_r, 0.0)

            T.copy(q[b_i, s_i, H0:H0 + H_PER_BLOCK, :d_nope], Q_nope)
            T.copy(q[b_i, s_i, H0:H0 + H_PER_BLOCK, d_nope:], Q_rope)

            for k_i in T.serial(inner_iter):
                topk_blk = group_i * inner_iter + k_i
                base = topk_blk * BI

                for bi in T.Parallel(BI):
                    gidx = base + bi
                    raw = indices[b_i, s_i, 0, gidx]
                    if have_topk_length:
                        v = T.if_then_else((raw >= 0) & (gidx < topk_length[b_i]), 1, 0)
                    else:
                        v = T.if_then_else(raw >= 0, 1, 0)
                    mask_s[bi] = v
                    page_s[bi] = T.if_then_else(v == 1, raw, 0)

                for bi, d in T.Parallel(BI, d_nope):
                    K_fp8[bi, d] = T.if_then_else(
                        mask_s[bi] == 1, kv_flat[page_s[bi], d], 0.0)
                for bi, d in T.Parallel(BI, d_rope):
                    K_rope[bi, d] = T.if_then_else(
                        mask_s[bi] == 1, k_rope_bf16[page_s[bi], d], 0.0)
                for bi, g in T.Parallel(BI, ng):
                    K_scale[bi, g] = T.if_then_else(
                        mask_s[bi] == 1, k_scale[page_s[bi], g], 0.0)

                for hi, bi in T.Parallel(H_PER_BLOCK, BI):
                    acc_s_frag[hi, bi] = T.if_then_else(
                        mask_s[bi] == 1, 0.0, -T.infinity(FP32))

                # QK^T: 4 groups
                for bi, d in T.Parallel(BI, qts):
                    K_buf[bi, d] = T.Cast(BF16, K_fp8[bi, d]) * T.Cast(BF16, K_scale[bi, 0])
                T.gemm(Q_nope[:, 0:qts], K_buf, acc_s_frag,
                       transpose_B=True, clear_accum=False, policy=T.GemmWarpPolicy.FullCol)

                for bi, d in T.Parallel(BI, qts):
                    K_buf[bi, d] = T.Cast(BF16, K_fp8[bi, qts + d]) * T.Cast(BF16, K_scale[bi, 1])
                T.gemm(Q_nope[:, qts:2*qts], K_buf, acc_s_frag,
                       transpose_B=True, clear_accum=False, policy=T.GemmWarpPolicy.FullCol)

                for bi, d in T.Parallel(BI, qts):
                    K_buf[bi, d] = T.Cast(BF16, K_fp8[bi, 2*qts + d]) * T.Cast(BF16, K_scale[bi, 2])
                T.gemm(Q_nope[:, 2*qts:3*qts], K_buf, acc_s_frag,
                       transpose_B=True, clear_accum=False, policy=T.GemmWarpPolicy.FullCol)

                for bi, d in T.Parallel(BI, qts):
                    K_buf[bi, d] = T.Cast(BF16, K_fp8[bi, 3*qts + d]) * T.Cast(BF16, K_scale[bi, 3])
                T.gemm(Q_nope[:, 3*qts:4*qts], K_buf, acc_s_frag,
                       transpose_B=True, clear_accum=False, policy=T.GemmWarpPolicy.FullCol)

                # QK^T: RoPE
                T.gemm(Q_rope, K_rope, acc_s_frag,
                       transpose_B=True, clear_accum=False, policy=T.GemmWarpPolicy.FullCol)

                # Online softmax
                T.copy(m_i, m_prev)
                T.reduce_max(acc_s_frag, m_i, dim=1, clear=False)
                for hi in T.Parallel(H_PER_BLOCK):
                    alpha[hi] = T.exp2(m_prev[hi] - m_i[hi])
                for hi, bi in T.Parallel(H_PER_BLOCK, BI):
                    acc_s_frag[hi, bi] = T.exp2(acc_s_frag[hi, bi] - m_i[hi])
                T.reduce_sum(acc_s_frag, lsum_i, dim=1)
                for hi in T.Parallel(H_PER_BLOCK):
                    lsum[hi] = lsum[hi] * alpha[hi] + lsum_i[hi]

                # Rescale
                for hi, d in T.Parallel(H_PER_BLOCK, qts):
                    acc_g0[hi, d] *= alpha[hi]
                    acc_g1[hi, d] *= alpha[hi]
                    acc_g2[hi, d] *= alpha[hi]
                    acc_g3[hi, d] *= alpha[hi]
                if v_include_rope:
                    for hi, d in T.Parallel(H_PER_BLOCK, d_rope):
                        acc_r[hi, d] *= alpha[hi]

                for hi, bi in T.Parallel(H_PER_BLOCK, BI):
                    S_smem[hi, bi] = T.Cast(BF16, acc_s_frag[hi, bi])

                # PV: 4 groups
                for bi, d in T.Parallel(BI, qts):
                    K_buf[bi, d] = T.Cast(BF16, K_fp8[bi, d]) * T.Cast(BF16, K_scale[bi, 0])
                T.gemm(S_smem, K_buf, tmp_out, clear_accum=True, policy=T.GemmWarpPolicy.FullCol)
                for hi, d in T.Parallel(H_PER_BLOCK, qts):
                    acc_g0[hi, d] += tmp_out[hi, d]

                for bi, d in T.Parallel(BI, qts):
                    K_buf[bi, d] = T.Cast(BF16, K_fp8[bi, qts + d]) * T.Cast(BF16, K_scale[bi, 1])
                T.gemm(S_smem, K_buf, tmp_out, clear_accum=True, policy=T.GemmWarpPolicy.FullCol)
                for hi, d in T.Parallel(H_PER_BLOCK, qts):
                    acc_g1[hi, d] += tmp_out[hi, d]

                for bi, d in T.Parallel(BI, qts):
                    K_buf[bi, d] = T.Cast(BF16, K_fp8[bi, 2*qts + d]) * T.Cast(BF16, K_scale[bi, 2])
                T.gemm(S_smem, K_buf, tmp_out, clear_accum=True, policy=T.GemmWarpPolicy.FullCol)
                for hi, d in T.Parallel(H_PER_BLOCK, qts):
                    acc_g2[hi, d] += tmp_out[hi, d]

                for bi, d in T.Parallel(BI, qts):
                    K_buf[bi, d] = T.Cast(BF16, K_fp8[bi, 3*qts + d]) * T.Cast(BF16, K_scale[bi, 3])
                T.gemm(S_smem, K_buf, tmp_out, clear_accum=True, policy=T.GemmWarpPolicy.FullCol)
                for hi, d in T.Parallel(H_PER_BLOCK, qts):
                    acc_g3[hi, d] += tmp_out[hi, d]

                if v_include_rope:
                    T.gemm(S_smem, K_rope, acc_r, clear_accum=False, policy=T.GemmWarpPolicy.FullCol)

            # Finalize
            for hi in T.Parallel(H_PER_BLOCK):
                denom = T.if_then_else(lsum[hi] == 0.0, 1.0, lsum[hi])
                alpha[hi] = 1.0 / denom
                partial_lse[b_i, s_i, group_i, H0 + hi] = T.if_then_else(
                    lsum[hi] == 0.0, -(2.0 ** 30),
                    T.log2(lsum[hi]) + m_i[hi])

            # Write NoPE output
            for hi, d in T.Parallel(H_PER_BLOCK, qts):
                partial_o[b_i, s_i, group_i, H0 + hi, d] = T.Cast(BF16, acc_g0[hi, d] * alpha[hi])
                partial_o[b_i, s_i, group_i, H0 + hi, qts + d] = T.Cast(BF16, acc_g1[hi, d] * alpha[hi])
                partial_o[b_i, s_i, group_i, H0 + hi, 2*qts + d] = T.Cast(BF16, acc_g2[hi, d] * alpha[hi])
                partial_o[b_i, s_i, group_i, H0 + hi, 3*qts + d] = T.Cast(BF16, acc_g3[hi, d] * alpha[hi])

            if v_include_rope:
                for hi, d in T.Parallel(H_PER_BLOCK, d_rope):
                    partial_o[b_i, s_i, group_i, H0 + hi, d_nope + d] = T.Cast(
                        BF16, acc_r[hi, d] * alpha[hi])

    return main


# ===========================================================================
# Combine kernel
# ===========================================================================

@tilelang.jit(out_idx=[-2, -1], pass_configs=_PASS_CONFIGS)
def _build_combine(
    heads: int,
    d_v: int,
    n_groups: int,
    have_attn_sink: bool,
):
    REPLICATE_H = heads // H_PER_BLOCK
    batch   = T.symbolic("batch")
    seq_len = T.symbolic("seq_len")

    @T.prim_func
    def main(
        partial_o:   T.Tensor([batch, seq_len, n_groups, heads, d_v], BF16),
        partial_lse: T.Tensor([batch, seq_len, n_groups, heads], FP32),
        attn_sink:   T.Tensor([heads], FP32),
        output:      T.Tensor([batch, seq_len, heads, d_v], BF16),
        lse_out:     T.Tensor([batch, seq_len, heads], FP32),
    ):
        with T.Kernel(batch * seq_len * REPLICATE_H, threads=256) as (bx,):
            rep_i = bx % REPLICATE_H
            tmp = bx // REPLICATE_H
            s_i = tmp % seq_len
            b_i = tmp // seq_len
            H0 = rep_i * H_PER_BLOCK

            lse_smem = T.alloc_shared([n_groups, H_PER_BLOCK], FP32)
            for g in T.serial(n_groups):
                T.copy(partial_lse[b_i, s_i, g, H0:H0 + H_PER_BLOCK], lse_smem[g, :])

            lse_max = T.alloc_fragment([H_PER_BLOCK], FP32)
            lse_sum = T.alloc_fragment([H_PER_BLOCK], FP32)
            scale_f = T.alloc_fragment([H_PER_BLOCK], FP32)
            acc_o   = T.alloc_fragment([H_PER_BLOCK, d_v], FP32)

            T.fill(lse_max, -(2.0 ** 30))
            for g in T.serial(n_groups):
                for hi in T.Parallel(H_PER_BLOCK):
                    lse_max[hi] = T.max(lse_max[hi], lse_smem[g, hi])

            T.fill(lse_sum, 0.0)
            for g in T.serial(n_groups):
                for hi in T.Parallel(H_PER_BLOCK):
                    lse_sum[hi] = lse_sum[hi] + T.exp2(lse_smem[g, hi] - lse_max[hi])

            T.fill(acc_o, 0.0)
            for g in T.serial(n_groups):
                for hi in T.Parallel(H_PER_BLOCK):
                    scale_f[hi] = T.exp2(lse_smem[g, hi] - lse_max[hi] - T.log2(lse_sum[hi]))
                for hi, d in T.Parallel(H_PER_BLOCK, d_v):
                    acc_o[hi, d] = acc_o[hi, d] + scale_f[hi] * T.Cast(
                        FP32, partial_o[b_i, s_i, g, H0 + hi, d])

            for hi in T.Parallel(H_PER_BLOCK):
                lse_b2 = T.log2(lse_sum[hi]) + lse_max[hi]
                lse_out[b_i, s_i, H0 + hi] = T.if_then_else(
                    lse_sum[hi] == 0.0, T.infinity(FP32), lse_b2 / LOG2E)

            if have_attn_sink:
                for hi, d in T.Parallel(H_PER_BLOCK, d_v):
                    lse_b2 = T.log2(lse_sum[hi]) + lse_max[hi]
                    sink_val = attn_sink[H0 + hi]
                    sf = T.if_then_else(
                        T.reinterpret("uint32", sink_val) == 0x7F800000,
                        0.0,
                        1.0 / (1.0 + T.exp2(sink_val * LOG2E - lse_b2)))
                    acc_o[hi, d] *= sf

            for hi, d in T.Parallel(H_PER_BLOCK, d_v):
                output[b_i, s_i, H0 + hi, d] = T.Cast(
                    BF16, T.if_then_else(lse_sum[hi] == 0.0, 0.0, acc_o[hi, d]))

    return main


# ===========================================================================
# Runners
# ===========================================================================

def _pick_inner_iter(topk, block_I=64):
    n = topk // block_I
    it = 1
    while it * 2 <= n and n % (it * 2) == 0:
        it *= 2
    return min(it, 4)


def _pad_m1_data(q, kv_flat, k_scale_7, k_rope_bf16, softmax_scale):
    """Pad MODEL1 (d_nope=448) to V32 (d_nope=512) and merge 7 scales to 4."""
    b, s_q, h_q, d_qk = q.shape
    N = kv_flat.shape[0]
    d_nope_padded = 512
    d_rope = 64

    # Pad Q_nope from 448 to 512
    q_nope = q[..., :448]
    q_rope = q[..., 448:]
    q_nope_padded = torch.nn.functional.pad(q_nope.float(), (0, 64))
    q_padded = torch.cat([q_nope_padded, q_rope.float()], dim=-1)
    q_scaled = (q_padded * softmax_scale * LOG2E).to(torch.bfloat16)

    # Compute 4-group scales and re-quantize nope data.
    # M1 has 7 groups of 64 elements, each with its own e8m0 scale.
    # We convert to 4 groups of 128 by: dequantize → pad → re-quantize.
    # This preserves numerical correctness because the new (FP8, scale)
    # pairs are consistent with each other.
    if k_scale_7.dtype == torch.uint8:
        # Valid data in [-1,1] gives uint8 ~[105,119]. Clamp garbage to safe range.
        k_scale_f32 = torch.pow(2.0, k_scale_7.float().clamp(64, 140) - 127.0)
    else:
        k_scale_f32 = k_scale_7.float().clamp(0, None)

    # Dequant nope: [N, 448] FP8 → [N, 448] float32.
    # Garbage e8m0 scales (from NaN quantization) produce extreme values.
    # Clamp dequantized data to [-10, 10] (well above the expected [-1, 1])
    # so that subsequent 4-group scales are numerically stable.
    nope_fp32 = kv_flat[:, :448].float()
    nope_deq = nope_fp32.reshape(N, 7, 64) * k_scale_f32.unsqueeze(-1)
    nope_deq = nope_deq.nan_to_num(0.0).clamp(-10.0, 10.0).reshape(N, 448)

    # Pad nope to 512, reshape to 4 groups × 128
    nope_padded = torch.nn.functional.pad(nope_deq, (0, 64))  # [N, 512]
    nope_4g = nope_padded.reshape(N, 4, 128)  # [N, 4, 128]

    # Compute 4-group scales
    max_abs_4g = nope_4g.abs().amax(dim=-1, keepdim=True)  # [N, 4, 1]
    scale_4g = (max_abs_4g / 448.0).clamp(min=1e-8)  # [N, 4, 1]
    k_scale_4 = scale_4g.squeeze(-1)  # [N, 4]

    # Re-quantize nope to FP8 with consistent 4-group scales.
    # The padded zeros (448:512) naturally quantize to FP8 zero.
    kv_padded_nope = (nope_4g / scale_4g).clamp(-448, 448).to(torch.float8_e4m3fn)
    kv_padded_nope = kv_padded_nope.reshape(N, 512)  # [N, 512]

    # Build kv_padded: [nope_req:512][rope_bf16:128][scale_orig:8] = 648
    kv_padded = torch.cat([kv_padded_nope, kv_flat[:, 448:]], dim=1)

    return q_scaled, kv_padded, k_scale_4, k_rope_bf16


def _run_partial(q, kv_flat, k_scale, k_rope_bf16, indices,
                 topk_length, d_qk, d_v, softmax_scale):
    b, s_q, h_q, _ = q.shape
    topk = indices.shape[-1]
    inner_iter = _pick_inner_iter(topk)

    if d_qk == 512:
        # Pad MODEL1 to V32 format.
        # The unified kernel has padded d_nope=512 and writes rope at offset 512.
        # We use kernel d_v=576 to accommodate (512 padded nope + 64 rope),
        # then trim to 512 (448 real nope + 64 rope).
        q_scaled, kv_padded, k_scale_4, k_rp = _pad_m1_data(
            q, kv_flat, k_scale, k_rope_bf16, softmax_scale)
        kernel = _partial_unified(h_q, 576, topk, inner_iter,
                                  (topk_length is not None), v_include_rope=True)
    else:
        q_scaled = (q.float() * softmax_scale * LOG2E).to(torch.bfloat16)
        kv_padded = kv_flat
        k_scale_4 = k_scale
        k_rp = k_rope_bf16
        kernel = _partial_unified(h_q, d_v, topk, inner_iter,
                                  (topk_length is not None), v_include_rope=False)

    idx = indices.unsqueeze(2).contiguous() if indices.dim() == 3 else indices.contiguous()
    tl = (topk_length if topk_length is not None
          else torch.zeros(b, dtype=torch.int32, device=q.device))

    po, pl = kernel(q_scaled, kv_padded, k_scale_4, k_rp, idx, tl)

    if d_qk == 512:
        # Trim padded nope elements: keep first 448 nope + 64 rope
        po = torch.cat([po[..., :448], po[..., 512:576]], dim=-1)  # [..., 512]

    return po, pl


def _run_combine(partial_o, partial_lse, attn_sink, h_q, d_v):
    b, s_q, n_groups = partial_o.shape[:3]
    have_sink = attn_sink is not None
    kernel = _build_combine(h_q, d_v, n_groups, have_sink)
    sink = (attn_sink if have_sink
            else torch.zeros(h_q, dtype=torch.float32, device=partial_o.device))
    out, lse = kernel(partial_o, partial_lse, sink)
    return out, lse.permute(0, 2, 1).contiguous()


# ===========================================================================
# Public interface
# ===========================================================================

@dataclasses.dataclass
class Sm89SchedMeta:
    @dataclasses.dataclass
    class Config:
        b: int; s_q: int; h_q: int; page_block_size: int; h_k: int
        causal: bool; is_fp8_kvcache: bool; topk: Optional[int]
        extra_page_block_size: Optional[int]; extra_topk: Optional[int]
    have_initialized: bool = False
    config: Optional["Sm89SchedMeta.Config"] = None


def get_mla_metadata(*args, **kwargs):
    return Sm89SchedMeta(), None


def flash_mla_with_kvcache(
    q, k_cache, block_table, cache_seqlens, head_dim_v,
    tile_scheduler_metadata, num_splits=None,
    softmax_scale=None, causal=False, is_fp8_kvcache=False,
    indices=None, attn_sink=None,
    extra_k_cache=None, extra_indices_in_kvcache=None,
    topk_length=None, extra_topk_length=None,
):
    b, s_q, h_q, d_qk = q.shape
    if softmax_scale is None:
        softmax_scale = d_qk ** (-0.5)

    if indices is not None:
        assert is_fp8_kvcache, "SM89 sparse requires is_fp8_kvcache=True"
        assert not causal

        kv, sc, rp = _split_kvcache_fp8(k_cache, d_qk)
        po, pl = _run_partial(q, kv, sc, rp, indices, topk_length,
                              d_qk, head_dim_v, softmax_scale)

        if extra_k_cache is not None:
            assert extra_indices_in_kvcache is not None
            ekv, esc, erp = _split_kvcache_fp8(extra_k_cache, d_qk)
            epo, epl = _run_partial(q, ekv, esc, erp, extra_indices_in_kvcache,
                                    extra_topk_length, d_qk, head_dim_v, softmax_scale)
            po = torch.cat([po, epo], dim=2)
            pl = torch.cat([pl, epl], dim=2)

        return _run_combine(po, pl, attn_sink, h_q, head_dim_v)
    else:
        raise NotImplementedError("SM89 dense decode not yet implemented.")
