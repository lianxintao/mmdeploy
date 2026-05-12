from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from lib import TestParam, Testcase, TestcaseForDecode, KVScope

# Triton kernel for accelerated KV gather in reference decode.
# Grid tiles across BOTH (batch,seq) AND token dimensions for full GPU utilization.
@triton.jit
def _process_kv_scope_kernel(
    blocked_k_ptr,              # [num_kv, d_qk] bf16
    indices_in_kvcache_ptr,     # [b, s_q, topk] int32
    topk_length_ptr,            # [b] int32 or empty
    gathered_kv_ptr,            # [b, s_q, topk, d_qk] bf16
    invalid_mask_ptr,           # [b, s_q, topk] bool
    d_qk: tl.constexpr,
    d_qk_rnd: tl.constexpr,     # next_power_of_2(d_qk) for tl.arange
    topk: tl.constexpr,
    BLOCK_TK: tl.constexpr,     # tokens per block (must divide topk)
    HAS_TOPK_LENGTH: tl.constexpr,
    stride_bk_n: tl.int32,
    stride_idx_b: tl.int32, stride_idx_s: tl.int32,
    stride_out_b: tl.int32, stride_out_s: tl.int32, stride_out_t: tl.int32,
    stride_mask_b: tl.int32, stride_mask_s: tl.int32,
    B: tl.int32, S_Q: tl.int32,
):
    """Tiled KV gather. Grid: (B * S_Q * topk // BLOCK_TK)."""
    pid = tl.program_id(0)
    tiles_per_seq = topk // BLOCK_TK   # integer, both constexpr
    bid_sq = pid // tiles_per_seq
    tile_id = pid % tiles_per_seq
    bid = bid_sq // S_Q
    sid = bid_sq % S_Q

    t_start = tile_id * BLOCK_TK
    offs_tk = t_start + tl.arange(0, BLOCK_TK)
    t_in_bounds = offs_tk < topk
    offs_d = tl.arange(0, d_qk_rnd)
    d_in_bounds = offs_d < d_qk

    # Load indices
    idx_ptr = (indices_in_kvcache_ptr + bid * stride_idx_b
               + sid * stride_idx_s + offs_tk)
    kv_idx = tl.load(idx_ptr, mask=t_in_bounds, other=-1)
    invalid = (kv_idx < 0) | ~t_in_bounds
    kv_idx_safe = tl.maximum(kv_idx, 0)

    if HAS_TOPK_LENGTH:
        tk_len = tl.load(topk_length_ptr + bid)
        invalid = invalid | (offs_tk >= tk_len)

    # Gather KV
    kv_ptr = (blocked_k_ptr + kv_idx_safe[:, None] * stride_bk_n
              + offs_d[None, :])
    load_mask = (~invalid[:, None]) & d_in_bounds[None, :]
    kv = tl.load(kv_ptr, mask=load_mask, other=0.0)

    # Store gathered KV
    out_ptr = (gathered_kv_ptr + bid * stride_out_b
               + sid * stride_out_s
               + offs_tk[:, None] * stride_out_t
               + offs_d[None, :])
    store_mask = t_in_bounds[:, None] & d_in_bounds[None, :]
    tl.store(out_ptr, kv, mask=store_mask)

    # Store mask
    mask_ptr = (invalid_mask_ptr + bid * stride_mask_b
                + sid * stride_mask_s + offs_tk)
    tl.store(mask_ptr, invalid.to(tl.int8), mask=t_in_bounds)

def _merge_two_lse(lse0: torch.Tensor, lse1: Optional[torch.Tensor], s_q: int, h_q: int) -> torch.Tensor:
    if lse1 is None:
        return lse0
    else:
        return torch.logsumexp(
            torch.stack([
                lse0.view(s_q, h_q),
                lse1.broadcast_to(s_q, h_q)
            ], dim=0),
            dim=0
        )
        
def ref_sparse_attn_fwd(p: TestParam, t: Testcase) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
    - o: [s_q, h_q, dv]
    - o_fp32: [s_q, h_q, dv]
    - max_logits: [s_q, h_q]
    - lse: [s_q, h_q]
    """
    indices = t.indices.clone().squeeze(1)
    if t.topk_length is not None:
        mask = torch.arange(p.topk, device=t.topk_length.device).unsqueeze(0).broadcast_to(p.s_q, p.topk) >= t.topk_length.unsqueeze(1)   # [s_q, topk]
        indices[mask] = -1
    invalid_mask = (indices < 0) | (indices >= p.s_kv)    # [s_q, topk]
    indices[invalid_mask] = 0

    q = t.q.float()
    gathered_kv = t.kv.index_select(dim=0, index=indices.flatten()).reshape(p.s_q, p.topk, p.d_qk).float()   # [s_q, topk, d_qk]
    P = (q @ gathered_kv.transpose(1, 2))   # [s_q, h_q, topk]
    P *= t.sm_scale
    P[invalid_mask.unsqueeze(1).broadcast_to(P.shape)] = float("-inf")

    orig_lse = torch.logsumexp(P, dim=-1)   # [s_q, h_q]
    max_logits = P.max(dim=-1).values   # [s_q, h_q]

    lse_for_o = _merge_two_lse(orig_lse, t.attn_sink, p.s_q, p.h_q)
    if not torch.is_inference_mode_enabled():
        lse_for_o = lse_for_o.clone()
    lse_for_o[lse_for_o == float("-inf")] = float("+inf")   # So that corresponding O will be 0
    s_for_o = torch.exp(P - lse_for_o.unsqueeze(-1))
    out = s_for_o @ gathered_kv[..., :p.d_v]   # [s_q, h_q, dv]

    lonely_q_mask = orig_lse == float("-inf")   # [s_q, h_q]
    orig_lse[lonely_q_mask] = float("+inf")
    return (out.to(torch.bfloat16), out, max_logits, orig_lse)


def ref_sparse_attn_decode(
    p: TestParam,
    t: TestcaseForDecode
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    A reference implementation of sparse decoding attention in PyTorch
    """
    assert p.h_kv == 1
    assert p.decode is not None
    b = p.decode.b

    def process_kv_scope(kv_scope: KVScope) -> Tuple[torch.Tensor, torch.Tensor]:
        assert kv_scope.indices_in_kvcache is not None
        topk = kv_scope.indices_in_kvcache.size(-1)
        blocked_k_flat = kv_scope.blocked_k.view(-1, p.d_qk)

        gathered_kv = torch.empty(b, p.s_q, topk, p.d_qk, dtype=torch.bfloat16, device=blocked_k_flat.device)
        invalid_mask = torch.empty(b, p.s_q, topk, dtype=torch.bool, device=blocked_k_flat.device)

        topk_len_tensor = kv_scope.topk_length if kv_scope.topk_length is not None else torch.empty(0, dtype=torch.int32, device=blocked_k_flat.device)

        d_qk_rnd = triton.next_power_of_2(p.d_qk)
        # Pick largest BLOCK_TK that divides topk (64 preferred, 32 or 128 fallback)
        for blk in [64, 32, 128]:
            if topk % blk == 0:
                BLOCK_TK = blk
                break
        else:
            BLOCK_TK = 64  # should not happen for typical topk values
        tiles_per_seq = topk // BLOCK_TK
        grid = (b * p.s_q * tiles_per_seq,)

        _process_kv_scope_kernel[grid](
            blocked_k_flat,
            kv_scope.indices_in_kvcache,
            topk_len_tensor,
            gathered_kv,
            invalid_mask,
            d_qk=p.d_qk,
            d_qk_rnd=d_qk_rnd,
            topk=topk,
            BLOCK_TK=BLOCK_TK,
            HAS_TOPK_LENGTH=kv_scope.topk_length is not None,
            stride_bk_n=blocked_k_flat.stride(0),
            stride_idx_b=kv_scope.indices_in_kvcache.stride(0),
            stride_idx_s=kv_scope.indices_in_kvcache.stride(1),
            stride_out_b=gathered_kv.stride(0),
            stride_out_s=gathered_kv.stride(1),
            stride_out_t=gathered_kv.stride(2),
            stride_mask_b=invalid_mask.stride(0),
            stride_mask_s=invalid_mask.stride(1),
            B=b, S_Q=p.s_q,
        )
        return gathered_kv, invalid_mask

    gathered_kv, invalid_mask = process_kv_scope(t.kv_scope)
    if t.extra_kv_scope is not None:
        gathered_kv1, invalid_mask1 = process_kv_scope(t.extra_kv_scope)
        gathered_kv = torch.cat([gathered_kv, gathered_kv1], dim=2)  # [b, s_q, topk+extra_topk, d]
        invalid_mask = torch.cat([invalid_mask, invalid_mask1], dim=2)   # [b, s_q, topk+extra_topk]

    gathered_kv = gathered_kv.view(b*p.s_q, -1, p.d_qk).float()
    gathered_kv[gathered_kv != gathered_kv] = 0.0
    q = t.q.float().view(b*p.s_q, p.h_q, p.d_qk)
    attn_weight = q @ gathered_kv.transpose(-1, -2)  # [t.b*t.s_q, t.h_q, topk+extra_topk]
    attn_weight *= t.sm_scale
    attn_weight[invalid_mask.view(b*p.s_q, 1, -1).broadcast_to(b*p.s_q, p.h_q, invalid_mask.size(-1))] = float("-inf")
    lse = attn_weight.logsumexp(dim=-1)  # [t.b*t.s_q, t.h_q]
    attn_weight = torch.exp(attn_weight - lse.unsqueeze(-1))
    output = attn_weight @ gathered_kv[..., :p.d_v]    # [t.b*t.s_q, t.h_q, t.dv]
    output = output.view(b, p.s_q, p.h_q, p.d_v)
    lse = lse.view(b, p.s_q, p.h_q)

    # Attention sink
    if t.attn_sink is not None:
        output *= (1.0 / (1.0 + torch.exp(t.attn_sink.view(1, 1, p.h_q) - lse))).unsqueeze(-1)

    # Correct for q tokens which has no attendable k
    lonely_q_mask = (lse == float("-inf"))
    output[lonely_q_mask.unsqueeze(-1).broadcast_to(b, p.s_q, p.h_q, p.d_v)] = 0.0
    lse[lonely_q_mask] = float("+inf")

    return output.to(torch.bfloat16), lse.transpose(1, 2)