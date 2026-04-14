"""
SM89 (Ada Lovelace) compatible Triton kernels for FP8 MQA logits computation.

Drop-in replacements for:
  - aiter.ops.triton.pa_mqa_logits.deepgemm_fp8_paged_mqa_logits  (paged attention)
  - aiter.ops.triton.fp8_mqa_logits.fp8_mqa_logits                (ragged attention)

These kernels use standard Triton operations (no AMD gluon/CDNA intrinsics)
and target NVIDIA SM89 FP8 tensor cores.
"""

import torch
import triton
import triton.language as tl


# ============================================================
# Kernel 1: Paged FP8 MQA Logits
# ============================================================


@triton.jit
def _fp8_paged_mqa_logits_kernel(
    batch_size,
    next_n,
    heads_num,
    # Q pointers and strides
    Q_buffer,
    stride_q_batch,
    stride_q_next_n,
    stride_q_heads,
    # KV cache: two views of the same packed buffer
    KV_buf_fp8,  # fp8 view   (for K data)
    KV_buf_fp32,  # fp32 view  (for scale data)
    BUF_STRIDE_FP8,  # page stride in fp8  elements
    BUF_STRIDE_FP32,  # page stride in fp32 elements
    S_OFFSET_FP32,  # scale region offset within page (fp32 elements)
    # Metadata
    context_len_ptr,
    kv_indices,
    stride_kvi_batch,
    # Weights and output
    weights,
    stride_w_batch,
    OutLogits_buffer,
    stride_out_batch,
    max_model_len,
    # Parallelism
    SplitKV,
    # Compile-time constants
    KV_BLOCK_SIZE: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    CHUNK_Q: tl.constexpr,
    CHUNK_K: tl.constexpr,
):
    pid = tl.program_id(0)

    pid_next_n = pid % next_n
    remain = pid // next_n
    pid_batch = remain % batch_size
    pid_split_kv = remain // batch_size

    context_length = tl.load(context_len_ptr + pid_batch)

    context_chunk_num = tl.cdiv(context_length, CHUNK_K)
    split_chunk_num = tl.cdiv(context_chunk_num, SplitKV)

    split_start = pid_split_kv * split_chunk_num * CHUNK_K
    split_length = tl.minimum(
        context_length - split_start,
        split_chunk_num * CHUNK_K,
    )

    if split_length <= 0:
        return

    # Load Q: [CHUNK_Q, HIDDEN_DIM] fp8
    q = tl.load(
        Q_buffer
        + pid_batch * stride_q_batch
        + pid_next_n * stride_q_next_n
        + (tl.arange(0, CHUNK_Q) * stride_q_heads)[:, None]
        + tl.arange(0, HIDDEN_DIM)[None, :],
    )

    # Load per-head weights: [CHUNK_Q] fp32
    scale_weight = tl.load(
        weights
        + (pid_batch * next_n + pid_next_n) * stride_w_batch
        + tl.arange(0, CHUNK_Q),
    )

    for ctx_offset in range(split_start, split_start + split_length, CHUNK_K):
        positions = ctx_offset + tl.arange(0, CHUNK_K)
        mask_kv = positions < context_length

        # --- page-table lookup ---
        logical_block = positions // KV_BLOCK_SIZE
        offset_in_block = positions % KV_BLOCK_SIZE

        physical_block = tl.load(
            kv_indices + pid_batch * stride_kvi_batch + logical_block,
            mask=mask_kv,
            other=0,
        )

        # --- Load K fp8 [CHUNK_K, HIDDEN_DIM] ---
        # Packed layout: K at page_base + token * HIDDEN_DIM
        k_base = physical_block * BUF_STRIDE_FP8 + offset_in_block * HIDDEN_DIM
        k = tl.load(
            KV_buf_fp8
            + k_base[:, None]
            + tl.arange(0, HIDDEN_DIM)[None, :],
            mask=mask_kv[:, None],
            other=0.0,
        )

        # --- Load scale fp32 [CHUNK_K] ---
        # Packed layout: scale at page_base_fp32 + S_OFFSET_FP32 + token
        s_addr = physical_block * BUF_STRIDE_FP32 + S_OFFSET_FP32 + offset_in_block
        k_scale = tl.load(KV_buf_fp32 + s_addr, mask=mask_kv, other=0.0)

        # --- QK^T  →  [CHUNK_Q, CHUNK_K] fp32 ---
        o = tl.dot(q, tl.trans(k))
        o = o * k_scale[None, :]
        o = tl.maximum(o, 0.0)
        o = o * scale_weight[:, None]

        # causal mask
        causal_mask = positions <= context_length - next_n + pid_next_n
        o = tl.where(causal_mask[None, :], o, float("-inf"))

        # reduce across heads → [CHUNK_K]
        logits = tl.sum(o, axis=0)

        out_base = (pid_batch * next_n + pid_next_n) * stride_out_batch
        tl.store(
            OutLogits_buffer + out_base + positions,
            logits,
            mask=mask_kv & (positions < max_model_len),
        )


def deepgemm_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    out_logits: torch.Tensor,
    context_lens: torch.Tensor,
    kv_indices: torch.Tensor,
    max_model_len: int,
    Preshuffle: bool = False,
    KVBlockSize: int = 64,
    ChunkK: int = 128,
    TotalCuCount: int = 128,
    WavePerEU: int = 2,
    VarCtxSchedule=None,
):
    """
    SM89-compatible paged FP8 MQA logits.

    Parameters match ``aiter.ops.triton.pa_mqa_logits.deepgemm_fp8_paged_mqa_logits``.
    ``out_logits`` should be pre-filled with ``-inf``.
    """
    batch_size, next_n, heads, hidden_dim = q_fp8.size()
    _, max_block_len = kv_indices.size()

    # Packed buffer constants (K data | scale data per page)
    buf_numel_per_page = KVBlockSize * (hidden_dim + hidden_dim // 128 * 4)
    BUF_STRIDE_FP8 = buf_numel_per_page
    BUF_STRIDE_FP32 = buf_numel_per_page // 4
    S_OFFSET_FP32 = KVBlockSize * hidden_dim // 4

    # Flatten and create fp8 / fp32 views of the same memory
    kv_flat = kv_cache.reshape(-1)
    kv_buf_fp8 = kv_flat.view(torch.float8_e4m3fn)
    kv_buf_fp32 = kv_flat.view(torch.float32)

    # SplitKV heuristic
    tile_q_count = max(batch_size * next_n, 1)
    SplitKV = max(1, (max(1, TotalCuCount // tile_q_count) + 4) // 5 * 5 * WavePerEU)

    grid = (batch_size * next_n * SplitKV,)

    _fp8_paged_mqa_logits_kernel[grid](
        batch_size,
        next_n,
        heads,
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        q_fp8.stride(2),
        kv_buf_fp8,
        kv_buf_fp32,
        BUF_STRIDE_FP8,
        BUF_STRIDE_FP32,
        S_OFFSET_FP32,
        context_lens,
        kv_indices,
        kv_indices.stride(0),
        weights,
        weights.stride(0),
        out_logits,
        out_logits.stride(0),
        max_model_len,
        SplitKV,
        KV_BLOCK_SIZE=KVBlockSize,
        HIDDEN_DIM=hidden_dim,
        CHUNK_Q=heads,
        CHUNK_K=ChunkK,
        num_warps=4,
        num_stages=2,
    )


# ============================================================
# Kernel 2: Ragged (variable-length) FP8 MQA Logits
# ============================================================


@triton.jit
def _fp8_mqa_logits_kernel(
    num_tokens,
    # Q
    Q_buffer,
    stride_q_token,
    stride_q_heads,
    # KV (flat, already gathered from pages)
    KV_buffer,
    stride_kv,
    scale_buffer,
    # Weights
    weights,
    stride_w,
    # Range per token
    ks_ptr,
    ke_ptr,
    # Output
    OutLogits,
    stride_out,
    total_kv,
    # Constants
    CHUNK_Q: tl.constexpr,
    CHUNK_K: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    ks_val = tl.load(ks_ptr + pid)
    ke_val = tl.load(ke_ptr + pid)

    # Load Q: [CHUNK_Q, HIDDEN_DIM] fp8
    q = tl.load(
        Q_buffer
        + pid * stride_q_token
        + (tl.arange(0, CHUNK_Q) * stride_q_heads)[:, None]
        + tl.arange(0, HIDDEN_DIM)[None, :],
    )

    # Load per-head weights: [CHUNK_Q] fp32
    w = tl.load(weights + pid * stride_w + tl.arange(0, CHUNK_Q))

    # Iterate only over the valid KV range, chunk-aligned
    ks_aligned = (ks_val // CHUNK_K) * CHUNK_K

    for chunk_start in range(ks_aligned, ke_val, CHUNK_K):
        positions = chunk_start + tl.arange(0, CHUNK_K)
        in_range = (positions >= ks_val) & (positions < ke_val)
        valid = in_range & (positions < total_kv)

        # Load K fp8: [CHUNK_K, HIDDEN_DIM]
        k = tl.load(
            KV_buffer
            + positions[:, None] * stride_kv
            + tl.arange(0, HIDDEN_DIM)[None, :],
            mask=valid[:, None],
            other=0.0,
        )

        # Load scale fp32: [CHUNK_K]
        k_scale = tl.load(scale_buffer + positions, mask=valid, other=0.0)

        # QK^T → [CHUNK_Q, CHUNK_K] fp32
        o = tl.dot(q, tl.trans(k))
        o = o * k_scale[None, :]
        o = tl.maximum(o, 0.0)
        o = o * w[:, None]

        # Mask out-of-range
        o = tl.where(in_range[None, :], o, 0.0)

        # Sum across heads → [CHUNK_K]
        logits = tl.sum(o, axis=0)

        # Write valid logits; out-of-range stays as pre-filled -inf
        tl.store(
            OutLogits + pid * stride_out + positions,
            logits,
            mask=in_range & (positions < total_kv),
        )


def fp8_mqa_logits(
    q_fp8: torch.Tensor,
    kv_fp8: torch.Tensor,
    kv_scale: torch.Tensor,
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
) -> torch.Tensor:
    """
    SM89-compatible ragged FP8 MQA logits.

    Parameters match ``aiter.ops.triton.fp8_mqa_logits.fp8_mqa_logits``.

    Args:
        q_fp8:    [num_tokens, heads, hidden_dim]  fp8
        kv_fp8:   [total_kv, hidden_dim]           fp8
        kv_scale: [total_kv]                       fp32
        weights:  [num_tokens, heads]               fp32
        ks:       [num_tokens]                      int32  (start of valid KV range)
        ke:       [num_tokens]                      int32  (end   of valid KV range)

    Returns:
        logits: [num_tokens, total_kv] fp32
    """
    num_tokens, heads, hidden_dim = q_fp8.shape
    total_kv = kv_fp8.shape[0]

    # Pre-fill with -inf so topk ignores out-of-range positions
    out = torch.full(
        (num_tokens, total_kv),
        float("-inf"),
        dtype=torch.float32,
        device=q_fp8.device,
    )

    if num_tokens == 0 or total_kv == 0:
        return out

    CHUNK_K = min(64, triton.next_power_of_2(total_kv))

    grid = (num_tokens,)
    _fp8_mqa_logits_kernel[grid](
        num_tokens,
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        kv_fp8,
        kv_fp8.stride(0),
        kv_scale,
        weights,
        weights.stride(0),
        ks,
        ke,
        out,
        out.stride(0),
        total_kv,
        CHUNK_Q=heads,
        CHUNK_K=CHUNK_K,
        HIDDEN_DIM=hidden_dim,
        num_warps=4,
        num_stages=2,
    )

    return out
