"""
Fused dequant + gather reference for FlashMLA sparse decoding (MODEL1 only).
Zero-copy: as_strided views share k_cache memory, kernel uses page-level addressing.
Scale is raw uint8 e8m0, kernel converts inline via tl.math.exp2.

FIXES vs. original:
  - All batch/seq strides upgraded to tl.int64 to avoid int32 overflow when s_q is
    large (e.g. 128k prefill). Previously `sid * stride_out_s` overflowed int32
    when s_q * topk * d_qk exceeded 2^31, causing illegal memory access.
  - PAGE_BYTES already int64; BLOCK_SIZE/BPT promoted to int64 to remove reliance
    on Triton's auto-promotion when num_blocks reaches multi-million scale
    (cache > 2 GB).
  - All constexpr byte offsets are explicitly promoted to int64 inside address
    expressions, removing any dependence on compiler-version-specific promotion
    rules for `int64 + constexpr_int`.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# FP8 KV cache views: zero-copy as_strided views (no copies)
# ---------------------------------------------------------------------------

def _get_cache_views(k_cache: torch.Tensor):
    """
    Zero-copy overlapping fp8/bf16/uint8 views of the packed MODEL1 KV cache.
    NO copies performed. Scale bytes are read directly from k_cache memory in the kernel.

    Returns (kv_fp8, kv_bf16, k_u8, d_nope, d_rope, num_tiles, tile_size,
             rope_bf16_off, page_bytes, block_size, bpt).
    """
    assert k_cache.ndim == 4 and k_cache.shape[2] == 1
    num_blocks, block_size = k_cache.shape[:2]

    d_nope, d_rope, tile_size, num_tiles = 448, 64, 64, 7
    bpt = 576  # bytes per token in data section (nope=448 fp8 + rope=64 bf16=128 bytes)

    page_bytes = int(k_cache.stride(0))  # float8 elements (= bytes) per block, includes padding
    total_bytes = num_blocks * page_bytes
    k_u8 = k_cache.view(torch.uint8).as_strided((total_bytes,), (1,))
    kv_fp8 = k_u8.view(torch.float8_e4m3fn)    # 1 element = 1 byte
    kv_bf16 = k_u8.view(torch.bfloat16)         # 1 element = 2 bytes

    rope_bf16_off = d_nope // 2                 # 448 / 2 = 224

    return (kv_fp8, kv_bf16, k_u8, d_nope, d_rope, num_tiles, tile_size,
            rope_bf16_off, page_bytes, block_size, bpt)


# ---------------------------------------------------------------------------
# Fused dequantize + gather kernel  (page-level addressing, zero-copy views)
# ---------------------------------------------------------------------------

@triton.jit
def _fused_dequant_gather_kernel(
    kv_fp8_ptr,            # float8e4nv flat  (1-byte elems, shares k_cache memory)
    kv_bf16_ptr,           # bf16 flat        (2-byte elems, same memory)
    k_u8_ptr,              # uint8 flat        (1-byte elems, same memory — for scale access)
    indices_ptr,           # int32            [b, s_q, topk]
    topk_length_ptr,       # int32            [b] or empty
    gathered_kv_ptr,       # bf16             [b, s_q, total_topk, d_qk]
    invalid_mask_ptr,      # int8             [b, s_q, total_topk]
    #
    d_nope: tl.constexpr,
    d_rope: tl.constexpr,
    d_qk: tl.constexpr,
    num_tiles: tl.constexpr,
    tile_size: tl.constexpr,
    rope_bf16_off: tl.constexpr,
    ROPE_BYTE_OFF: tl.constexpr,           # byte offset of rope within token (448)
    topk: tl.constexpr,
    BLOCK_TK: tl.constexpr,
    TILES_PER_SEQ: tl.constexpr,           # cdiv(topk, BLOCK_TK) — computed at launch
    HAS_TOPK_LENGTH: tl.constexpr,
    SCALE_SECTION_OFFSET: tl.constexpr,    # byte offset of scale section within block (BLOCK_SIZE * BPT)
    WRITE_OFFSET_TK: tl.constexpr,         # offset in total_topk dim for this kernel launch
    #
    PAGE_BYTES: tl.int64,                  # stride_0 of k_cache (bytes per block, includes padding)
    BLOCK_SIZE: tl.int64,                  # ← FIX: int32 → int64 (avoid auto-promotion reliance)
    BPT: tl.int64,                         # ← FIX: int32 → int64 (avoid auto-promotion reliance)
    # ← FIX: ALL batch/seq strides upgraded to int64.
    #        Originally int32; `sid * stride_out_s` overflowed when
    #        s_q * topk * d_qk > 2^31 (e.g. 128k prefill).
    stride_idx_b: tl.int64,
    stride_idx_s: tl.int64,
    stride_out_b: tl.int64,
    stride_out_s: tl.int64,
    stride_out_t: tl.int64,
    stride_mask_b: tl.int64,
    stride_mask_s: tl.int64,
    B: tl.int32,
    S_Q: tl.int32,
):
    # ---- 1D grid: each program handles BLOCK_TK tokens × ALL nope tiles ----
    pid = tl.program_id(0)
    bid_sq = pid // TILES_PER_SEQ
    tile_id = pid % TILES_PER_SEQ
    bid = bid_sq // S_Q
    sid = bid_sq % S_Q

    # ← FIX: pre-cast bid/sid to int64 for use in all address arithmetic below
    bid64 = bid.to(tl.int64)
    sid64 = sid.to(tl.int64)

    t_start = tile_id * BLOCK_TK
    offs_tk = t_start + tl.arange(0, BLOCK_TK)
    t_in_bounds = offs_tk < topk

    # ---- load indices ONCE for all tiles ----
    # ← FIX: use int64 base offsets for indices addressing
    idx_ptr = (indices_ptr
               + bid64 * stride_idx_b
               + sid64 * stride_idx_s
               + offs_tk.to(tl.int64))
    kv_idx = tl.load(idx_ptr, mask=t_in_bounds, other=-1)
    invalid = (kv_idx < 0) | ~t_in_bounds
    kv_idx_safe = tl.maximum(kv_idx, 0)

    if HAS_TOPK_LENGTH:
        tk_len = tl.load(topk_length_ptr + bid)
        invalid = invalid | (offs_tk >= tk_len)

    # Write invalid mask ONCE
    # ← FIX: int64 address arithmetic + explicit int64 WRITE_OFFSET
    mask_ptr = (invalid_mask_ptr
                + bid64 * stride_mask_b
                + sid64 * stride_mask_s
                + tl.full([], WRITE_OFFSET_TK, tl.int64)
                + offs_tk.to(tl.int64))
    tl.store(mask_ptr, invalid.to(tl.int8), mask=t_in_bounds)

    valid_1d = ~invalid

    # ---- page-level addressing ONCE ----
    # ← FIX: page_id / page_off promoted to int64 before any multiplication.
    #        Even though kv_idx fits in int32, downstream products do not.
    page_id = (kv_idx_safe // BLOCK_SIZE.to(tl.int32)).to(tl.int64)
    page_off = (kv_idx_safe % BLOCK_SIZE.to(tl.int32)).to(tl.int64)
    token_byte_base = page_id * PAGE_BYTES + page_off * BPT

    # ---- dequant ALL nope tiles (looped) ----
    for tile_y in tl.static_range(num_tiles):
        t_off = tile_y * tile_size
        offs_d_nope = t_off + tl.arange(0, tile_size)

        # ← FIX: cast small offset to int64 to keep the whole address in int64
        nope_ptr = (
            kv_fp8_ptr
            + token_byte_base[:, None]
            + offs_d_nope[None, :].to(tl.int64)
        )
        nope_fp8 = tl.load(nope_ptr, mask=valid_1d[:, None], other=0.0)

        # Load e8m0 scale inline from k_cache memory (no pre-copied tensor)
        # ← FIX: explicit int64 promotion for all literal/constexpr terms
        scale_byte_off = (
            page_id * PAGE_BYTES
            + tl.full([], SCALE_SECTION_OFFSET, tl.int64)
            + page_off * tl.full([], 8, tl.int64)
            + tl.full([], tile_y, tl.int64)
        )
        scale_raw = tl.load(k_u8_ptr + scale_byte_off, mask=valid_1d, other=127)
        scale = tl.math.exp2(scale_raw.to(tl.float32) - 127.0)

        nope_f32 = nope_fp8.to(tl.float32) * scale[:, None]
        nope_bf16 = nope_f32.to(tl.bfloat16)

        # ← FIX: every term in the output address is int64
        out_ptr = (
            gathered_kv_ptr
            + bid64 * stride_out_b
            + sid64 * stride_out_s
            + (tl.full([], WRITE_OFFSET_TK, tl.int64)
               + offs_tk[:, None].to(tl.int64)) * stride_out_t
            + offs_d_nope[None, :].to(tl.int64)
        )
        tl.store(out_ptr, nope_bf16, mask=t_in_bounds[:, None])

    # ---- load + store rope ONCE after all tiles ----
    offs_rope = tl.arange(0, d_rope)
    # ← FIX: explicit int64 promotion of constexpr ROPE_BYTE_OFF
    rope_byte_base = token_byte_base + tl.full([], ROPE_BYTE_OFF, tl.int64)
    rope_elem_base = rope_byte_base // 2       # int64 // 2 = int64
    rope_ptr = (
        kv_bf16_ptr
        + rope_elem_base[:, None]
        + offs_rope[None, :].to(tl.int64)
    )
    rope = tl.load(rope_ptr, mask=valid_1d[:, None], other=0.0)

    # ← FIX: int64 output address
    out_rope_ptr = (
        gathered_kv_ptr
        + bid64 * stride_out_b
        + sid64 * stride_out_s
        + (tl.full([], WRITE_OFFSET_TK, tl.int64)
           + offs_tk[:, None].to(tl.int64)) * stride_out_t
        + (tl.full([], d_nope, tl.int64) + offs_rope[None, :].to(tl.int64))
    )
    tl.store(out_rope_ptr, rope, mask=t_in_bounds[:, None])


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def _fused_gather_kv(
    kv_fp8: torch.Tensor,
    kv_bf16: torch.Tensor,
    k_mem: torch.Tensor,
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    d_nope: int,
    d_rope: int,
    num_tiles: int,
    tile_size: int,
    rope_bf16_off: int,
    page_bytes: int,
    block_size: int,
    bpt: int,
    gathered_kv_out: Optional[torch.Tensor] = None,
    invalid_mask_out: Optional[torch.Tensor] = None,
    write_offset_tk: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    b, s_q, topk = indices.shape
    d_qk = d_nope + d_rope
    device = kv_fp8.device

    if gathered_kv_out is None:
        gathered_kv = torch.empty(b, s_q, topk, d_qk, dtype=torch.bfloat16, device=device)
    else:
        gathered_kv = gathered_kv_out

    if invalid_mask_out is None:
        invalid_mask = torch.empty(b, s_q, topk, dtype=torch.bool, device=device)
    else:
        invalid_mask = invalid_mask_out

    topk_len_tensor = (
        topk_length if topk_length is not None
        else torch.empty(0, dtype=torch.int32, device=device)
    )

    # Choose BLOCK_TK: prefer large blocks, ensure enough grid_x parallelism
    BLOCK_TK = 64
    for blk in [256, 128, 64, 32, 16, 8]:
        if topk >= blk:
            BLOCK_TK = blk
            break
    tiles_per_seq = (topk + BLOCK_TK - 1) // BLOCK_TK
    grid_x = b * s_q * tiles_per_seq
    # Shrink BLOCK_TK if too few programs to fill GPU
    while grid_x < 32 and BLOCK_TK > 8:
        BLOCK_TK //= 2
        tiles_per_seq = (topk + BLOCK_TK - 1) // BLOCK_TK
        grid_x = b * s_q * tiles_per_seq

    grid = (grid_x, 1)  # 1D grid — kernel loops over all nope tiles internally

    _fused_dequant_gather_kernel[grid](
        kv_fp8, kv_bf16, k_mem,
        indices, topk_len_tensor,
        gathered_kv, invalid_mask,
        d_nope=d_nope,
        d_rope=d_rope,
        d_qk=d_qk,
        num_tiles=num_tiles,
        tile_size=tile_size,
        rope_bf16_off=rope_bf16_off,
        ROPE_BYTE_OFF=d_nope,
        topk=topk,
        BLOCK_TK=BLOCK_TK,
        TILES_PER_SEQ=tiles_per_seq,
        HAS_TOPK_LENGTH=topk_length is not None,
        SCALE_SECTION_OFFSET=block_size * bpt,
        WRITE_OFFSET_TK=write_offset_tk,
        PAGE_BYTES=int(page_bytes),
        BLOCK_SIZE=int(block_size),     # ← FIX: ensure Python int (will bind to int64 param)
        BPT=int(bpt),                   # ← FIX: ensure Python int (will bind to int64 param)
        # ← FIX: pass strides as Python ints; kernel param type (tl.int64) governs binding
        stride_idx_b=int(indices.stride(0)),
        stride_idx_s=int(indices.stride(1)),
        stride_out_b=int(gathered_kv.stride(0)),
        stride_out_s=int(gathered_kv.stride(1)),
        stride_out_t=int(gathered_kv.stride(2)),
        stride_mask_b=int(invalid_mask.stride(0)),
        stride_mask_s=int(invalid_mask.stride(1)),
        B=b, S_Q=s_q,
    )
    return gathered_kv, invalid_mask


# ---------------------------------------------------------------------------
# Main decode function
# ---------------------------------------------------------------------------

def flash_mla_decode_torch(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    head_dim_v: int,
    softmax_scale: float,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    b, s_q, h_q, d_qk = q.shape
    d_v = head_dim_v

    # ---------- zero-copy views + unified dequant/gather ----------
    (kv_fp8, kv_bf16, k_u8, d_nope, d_rope,
     n_tiles, t_size, rope_off, pg_bytes, bs, bpt) = _get_cache_views(k_cache)

    main_topk = indices.shape[2]
    total_topk = main_topk + (extra_indices.shape[2] if extra_indices is not None else 0)

    # Pre-allocate unified output — both main and extra writes target this tensor
    gathered_kv = torch.empty(b, s_q, total_topk, d_qk, dtype=torch.bfloat16, device=q.device)
    invalid_mask = torch.empty(b, s_q, total_topk, dtype=torch.bool, device=q.device)

    # Gather main → region [0, main_topk)
    _fused_gather_kv(
        kv_fp8, kv_bf16, k_u8, indices, topk_length,
        d_nope, d_rope, n_tiles, t_size, rope_off, pg_bytes, bs, bpt,
        gathered_kv_out=gathered_kv, invalid_mask_out=invalid_mask, write_offset_tk=0,
    )

    # Gather extra → region [main_topk, total_topk) if present
    if extra_k_cache is not None:
        assert extra_indices is not None
        (ek_fp8, ek_bf16, ek_u8, _, _, _, _, er_off,
         epg_bytes, ebs, ebpt) = _get_cache_views(extra_k_cache)
        _fused_gather_kv(
            ek_fp8, ek_bf16, ek_u8, extra_indices, extra_topk_length,
            d_nope, d_rope, n_tiles, t_size, er_off, epg_bytes, ebs, ebpt,
            gathered_kv_out=gathered_kv, invalid_mask_out=invalid_mask,
            write_offset_tk=main_topk,
        )

    # ---------- mixed-precision attention (keep KV/Q in bf16, softmax in f32) ----------
    total_topk = gathered_kv.shape[2]

    # QK^T in bf16 → bf16 result (saves memory: no gathered_kv.float())
    q_flat = q.reshape(b * s_q, h_q, d_qk)
    kv_flat = gathered_kv.reshape(b * s_q, total_topk, d_qk)
    # QK^T in bf16, logsumexp handles bf16 natively (no .float() needed)
    attn_weight = (q_flat @ kv_flat.transpose(-1, -2)) * softmax_scale
    attn_weight[
        invalid_mask.reshape(b * s_q, 1, total_topk).broadcast_to(
            b * s_q, h_q, total_topk
        )
    ] = float("-inf")

    lse = attn_weight.logsumexp(dim=-1)
    # exp auto-upcasts bf16→f32, PV uses bf16 matmul
    attn_weight = torch.exp(attn_weight - lse.unsqueeze(-1))
    output = attn_weight.to(torch.bfloat16) @ kv_flat[..., :d_v]
    output = output.reshape(b, s_q, h_q, d_v)
    lse = lse.reshape(b, s_q, h_q)

    if attn_sink is not None:
        output *= (
            1.0 / (1.0 + torch.exp(attn_sink.view(1, 1, h_q) - lse))
        ).unsqueeze(-1)

    lonely_q_mask = lse == float("-inf")
    output[lonely_q_mask.unsqueeze(-1).broadcast_to(b, s_q, h_q, d_v)] = 0.0
    lse[lonely_q_mask] = float("+inf")

    return output.to(torch.bfloat16), lse.transpose(1, 2)
