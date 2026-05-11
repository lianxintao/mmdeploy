"""SM89-optimized TileLang 0.1.9 sparse MLA decode kernel.

之前版本错误地混用了 Triton/TVM 的 API（T.program_id / T.prim_func 内 T.alloc_var 等），
这些在 TileLang 0.1.9 中根本不存在。

TileLang 0.1.9 正确 API：
    with T.Kernel(gx, gy, gz, threads=N) as (bx, by, bz):  ← grid index 由此而来
        T.alloc_shared([M, N], dtype)
        T.alloc_fragment([M, N], dtype)
        T.copy(src[i:i+M, j:j+N], dst)           ← tile 级异步拷贝
        for k in T.Pipelined(range, num_stages=N): ← 软件流水线
        for i, j in T.Parallel(M, N):             ← 并行元素操作
        T.gemm(A, B, C, transpose_B=True, ...)    ← tensor core gemm
        T.reduce_max / T.reduce_sum / T.fill / T.clear / T.copy
        T.if_then_else(cond, true_val, false_val)  ← 标量三元
        T.infinity(dtype) / T.exp2 / T.log

对比原始 Triton V2 在 SM89 上的优化：
  1. Grid (B, H, num_split): split-KV 充分利用 128 SMs
  2. T.Pipelined 流水线: 隐藏 gather 延迟
  3. T.gemm: 利用 tensor core（SM89 支持 bf16 tc）
  4. block_N=64: 更大 tile 摊薄 page 地址计算开销

参照官方：
  tilelang/examples/deepseek_v32/sparse_mla_fwd.py
  tilelang/examples/deepseek_mla/example_mla_decode.py
"""

import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# DSv4 layout constants
_NOPE_DIM          = 448
_ROPE_DIM          = 64
_D                 = _NOPE_DIM + _ROPE_DIM   # 512
_NOPE_PAD          = 512   # pad nope to power-of-2 for tensor core
_TOKEN_DATA_STRIDE = 576   # bytes per token: 448 FP8 + 128 BF16
_SCALE_STRIDE      = 8     # bytes per token in scale section

_kernel_cache: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
#  Core prim_func builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_kernel(
    B:          int,
    H:          int,
    topk:       int,
    num_split:  int,
    block_N:    int,   # tokens per split tile, e.g. 64
    page_size:  int,
    page_bytes: int,   # stride(0) of fp8 flat buffer = bytes/page
    scale_off:  int,   # byte offset of scale section = page_size * 576
    sm_scale:   float,
):
    """Return a compiled TileLang kernel for given (B, H, topk, ...) config."""
    import tilelang
    import tilelang.language as T

    nope_pad    = _NOPE_PAD   # 512
    rope_dim    = _ROPE_DIM   # 64
    nope_dim    = _NOPE_DIM   # 448
    num_threads = 256         # 8 warps; SM89 preferred

    @T.prim_func
    def sparse_decode_fwd(
        # Q split into nope (padded) and rope parts for cleaner gemm
        Q_nope:  T.Buffer((B, H, nope_pad), "bfloat16"),   # [B, H, 512]
        Q_rope:  T.Buffer((B, H, rope_dim), "bfloat16"),   # [B, H, 64]
        # Flat KV cache views (same physical memory, three dtypes)
        KV_fp8:  T.Buffer((B * topk * _TOKEN_DATA_STRIDE,),          "float8_e4m3fn"),
        KV_u8:   T.Buffer((B * topk * _TOKEN_DATA_STRIDE,),          "uint8"),
        KV_bf16: T.Buffer((B * topk * _TOKEN_DATA_STRIDE // 2,),     "bfloat16"),
        # Gather indices [B, topk]; -1 = invalid/padding
        Indices: T.Buffer((B, topk), "int32"),
        # Split-KV partial outputs
        Out:     T.Buffer((B, H, num_split, nope_pad + rope_dim), "float32"),
        LSE:     T.Buffer((B, H, num_split),                       "float32"),
    ):
        # ── Grid: batch × heads × split-KV ──────────────────────────────
        # bx=batch, by=head, bz=split tile index
        with T.Kernel(B, H, num_split, threads=num_threads) as (bx, by, bz):

            # ── Shared memory ────────────────────────────────────────────
            # Q: one query token per head (decode mode)
            Q_nope_s = T.alloc_shared((1, nope_pad), "bfloat16")
            Q_rope_s = T.alloc_shared((1, rope_dim),  "bfloat16")
            # KV tile: block_N tokens × dims (float32 after dequant)
            KV_nope_s = T.alloc_shared((block_N, nope_pad), "float32")
            KV_rope_s = T.alloc_shared((block_N, rope_dim), "float32")

            # ── Fragments ────────────────────────────────────────────────
            acc_s         = T.alloc_fragment((1, block_N), "float32")
            acc_s_cast    = T.alloc_fragment((1, block_N), "bfloat16")
            acc_o_nope    = T.alloc_fragment((1, nope_pad), "float32")
            acc_o_rope    = T.alloc_fragment((1, rope_dim),  "float32")
            m_i           = T.alloc_fragment((1,), "float32")  # running max
            m_prev        = T.alloc_fragment((1,), "float32")
            l_i           = T.alloc_fragment((1,), "float32")  # running sum
            rescale       = T.alloc_fragment((1,), "float32")

            T.fill(acc_o_nope, 0.0)
            T.fill(acc_o_rope, 0.0)
            T.fill(l_i,   0.0)
            T.fill(m_i,   -T.infinity("float32"))

            # ── Load Q (one decode query, head=by) ───────────────────────
            # Q_nope: [B, H, 512]; slice [bx, by, :] → [1, 512]
            for d in T.Parallel(nope_pad):
                Q_nope_s[0, d] = Q_nope[bx, by, d]
            for d in T.Parallel(rope_dim):
                Q_rope_s[0, d] = Q_rope[bx, by, d]

            # ── Gather KV tile for this split (bz) ───────────────────────
            # bz covers token range [bz*block_N, (bz+1)*block_N)
            # Use T.Pipelined to overlap the gather loop with the next
            # block's compute. num_stages=2 = double-buffer.
            #
            # The inner T.Parallel(block_N) body is the "load" stage;
            # gemm/softmax below is the "compute" stage — TileLang's
            # pipeline pass schedules them automatically.

            for i in T.Pipelined(block_N, num_stages=2):
                t = bz * block_N + i
                # Clamp to valid range; invalid rows will be masked later
                raw_idx = T.if_then_else(
                    t < topk,
                    Indices[bx, t],
                    T.int32(0)
                )
                valid = T.if_then_else(t < topk, True, False)
                # Avoid page index OOB even for invalid tokens
                safe_idx   = T.if_then_else(raw_idx >= 0, raw_idx, T.int32(0))
                pg         = safe_idx // page_size
                pg_off     = safe_idx %  page_size
                tok_base   = pg * page_bytes + pg_off * _TOKEN_DATA_STRIDE
                scale_base = pg * page_bytes + scale_off + pg_off * _SCALE_STRIDE

                # FP8 nope → float32 with UE8M0 dequant
                for d in T.Parallel(nope_pad):
                    fp8_v      = KV_fp8[tok_base + d]
                    grp        = d // 64
                    sc_raw     = KV_u8[scale_base + grp]
                    sc_f32     = T.exp2(
                        T.cast(sc_raw, "float32") - T.float32(127.0)
                    )
                    dq = T.cast(fp8_v, "float32") * sc_f32
                    KV_nope_s[i, d] = T.if_then_else(
                        (t < topk) & (raw_idx >= 0) & (d < nope_dim),
                        dq,
                        T.float32(0.0)
                    )

                # BF16 rope
                rope_elem_base = (tok_base + nope_dim) // 2
                for d in T.Parallel(rope_dim):
                    val = KV_bf16[rope_elem_base + d]
                    KV_rope_s[i, d] = T.if_then_else(
                        (t < topk) & (raw_idx >= 0),
                        T.cast(val, "float32"),
                        T.float32(0.0)
                    )

            # ── QK scores via gemm ────────────────────────────────────────
            # acc_s[1, block_N] = Q_nope_s[1,512] × KV_nope_s^T[block_N,512]
            #                   + Q_rope_s[1,64]  × KV_rope_s^T[block_N,64]
            #
            # gemm requires matching dtype; cast Q to float32 in-place.
            Q_nope_f32 = T.alloc_shared((1, nope_pad), "float32")
            Q_rope_f32 = T.alloc_shared((1, rope_dim),  "float32")
            for d in T.Parallel(nope_pad):
                Q_nope_f32[0, d] = T.cast(Q_nope_s[0, d], "float32")
            for d in T.Parallel(rope_dim):
                Q_rope_f32[0, d] = T.cast(Q_rope_s[0, d], "float32")

            T.gemm(Q_nope_f32, KV_nope_s, acc_s,
                   transpose_B=True,
                   policy=T.GemmWarpPolicy.FullCol,
                   clear_accum=True)
            T.gemm(Q_rope_f32, KV_rope_s, acc_s,
                   transpose_B=True,
                   policy=T.GemmWarpPolicy.FullCol,
                   clear_accum=False)

            # Scale + mask padding tokens with -inf
            for n in T.Parallel(block_N):
                t_n = bz * block_N + n
                raw_n = T.if_then_else(t_n < topk, Indices[bx, t_n], T.int32(-1))
                acc_s[0, n] = T.if_then_else(
                    (t_n < topk) & (raw_n >= 0),
                    acc_s[0, n] * T.float32(sm_scale),
                    -T.infinity("float32")
                )

            # ── Online softmax (base-2 for SM89 efficiency) ───────────────
            T.copy(m_i, m_prev)
            T.fill(m_i, -T.infinity("float32"))
            T.reduce_max(acc_s, m_i, dim=1, clear=False)

            # rescale = exp2(m_prev - m_new)
            rescale[0] = T.exp2(m_prev[0] - m_i[0])

            # p = exp2(score - m_new)
            for n in T.Parallel(block_N):
                acc_s[0, n] = T.exp2(acc_s[0, n] - m_i[0])

            # sum_p
            p_sum = T.alloc_fragment((1,), "float32")
            T.fill(p_sum, 0.0)
            T.reduce_sum(acc_s, p_sum, dim=1)

            # Update running stats
            l_i[0] = l_i[0] * rescale[0] + p_sum[0]

            # Rescale old accumulator
            for d in T.Parallel(nope_pad):
                acc_o_nope[0, d] *= rescale[0]
            for d in T.Parallel(rope_dim):
                acc_o_rope[0, d] *= rescale[0]

            # Cast p to bf16 for gemm with KV
            for n in T.Parallel(block_N):
                acc_s_cast[0, n] = T.cast(acc_s[0, n], "bfloat16")

            # acc_o += p @ KV (nope + rope)
            # KV_nope_s is float32; need cast for bf16 gemm
            KV_nope_cast = T.alloc_shared((block_N, nope_pad), "bfloat16")
            KV_rope_cast = T.alloc_shared((block_N, rope_dim),  "bfloat16")
            for i, d in T.Parallel(block_N, nope_pad):
                KV_nope_cast[i, d] = T.cast(KV_nope_s[i, d], "bfloat16")
            for i, d in T.Parallel(block_N, rope_dim):
                KV_rope_cast[i, d] = T.cast(KV_rope_s[i, d], "bfloat16")

            T.gemm(acc_s_cast, KV_nope_cast, acc_o_nope,
                   policy=T.GemmWarpPolicy.FullCol, clear_accum=False)
            T.gemm(acc_s_cast, KV_rope_cast, acc_o_rope,
                   policy=T.GemmWarpPolicy.FullCol, clear_accum=False)

            # ── Write partial output ──────────────────────────────────────
            safe_l = T.if_then_else(l_i[0] > 0.0, l_i[0], T.float32(1.0))
            for d in T.Parallel(nope_pad):
                Out[bx, by, bz, d]          = acc_o_nope[0, d] / safe_l
            for d in T.Parallel(rope_dim):
                Out[bx, by, bz, nope_pad + d] = acc_o_rope[0, d] / safe_l

            # LSE (natural log convention)
            LSE[bx, by, bz] = T.if_then_else(
                l_i[0] > 0.0,
                m_i[0] / T.float32(1.4426950408889634) + T.log(l_i[0]),
                -T.infinity("float32")
            )

    # Compile to CUDA
    return tilelang.compile(
        sparse_decode_fwd,
        out_idx=[-2, -1],
        target="cuda",
        execution_backend="cython",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Split-KV reduce (PyTorch side)
# ─────────────────────────────────────────────────────────────────────────────

def _splitk_reduce(
    out_partial: torch.Tensor,   # [B, H, num_split, D]  float32
    lse_partial: torch.Tensor,   # [B, H, num_split]     float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_lse, _ = lse_partial.max(dim=2, keepdim=True)
    w = torch.where(
        lse_partial > -1e20,
        torch.exp(lse_partial - max_lse.squeeze(2)),
        torch.zeros_like(lse_partial),
    )                                                          # [B, H, num_split]
    total  = w.sum(dim=2, keepdim=True).clamp(min=1e-20)
    merged = (w.unsqueeze(-1) * out_partial).sum(dim=2) / total
    merged_lse = max_lse.squeeze(2) + torch.log(total.squeeze(2))
    return merged.to(torch.bfloat16), merged_lse


# ─────────────────────────────────────────────────────────────────────────────
#  _run_tilelang_sparse_decode — mirrors _run_triton_sparse_decode interface
# ─────────────────────────────────────────────────────────────────────────────

def _run_tilelang_sparse_decode(
    q:            torch.Tensor,              # [B, 1, H, D] bfloat16
    k_cache:      torch.Tensor,              # [pages, page_sz, 1, bpt] float8_e4m3fn
    indices:      torch.Tensor,              # [B, ...] int32
    topk_length:  Optional[torch.Tensor],
    softmax_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:

    B, _, H, D = q.shape
    page_size  = int(k_cache.shape[1])
    page_bytes = int(k_cache.stride(0))      # fp8 stride = bytes per page

    flat_idx = indices.reshape(B, -1).contiguous()
    topk     = int(flat_idx.shape[1])

    # Mask out-of-range indices with -1
    if topk_length is not None:
        rng  = torch.arange(topk, device=q.device).unsqueeze(0)
        mask = rng >= topk_length.unsqueeze(1)
        flat_idx = flat_idx.clone()
        flat_idx[mask] = -1

    # Three flat views of the KV cache
    num_pages   = int(k_cache.shape[0])
    total_bytes = num_pages * page_bytes
    raw_fp8  = k_cache.as_strided((total_bytes,), (1,))
    raw_u8   = raw_fp8.view(torch.uint8)
    raw_bf16 = raw_u8.view(torch.bfloat16)

    # Q: squeeze seq dim → [B, H, D]; split nope+rope; pad nope to 512
    q3      = q.squeeze(1).contiguous()                           # [B, H, 512]
    q_nope  = torch.zeros(B, H, _NOPE_PAD, dtype=q.dtype, device=q.device)
    q_nope[:, :, :_NOPE_DIM] = q3[:, :, :_NOPE_DIM]
    q_rope  = q3[:, :, _NOPE_DIM:].contiguous()                   # [B, H, 64]

    scale_off = page_size * _TOKEN_DATA_STRIDE
    block_N   = 64
    num_split = max(1, (topk + block_N - 1) // block_N)

    # JIT compile / retrieve
    cache_key = (B, H, topk, page_size, page_bytes, num_split)
    if cache_key not in _kernel_cache:
        logger.info(f"[TileLang] JIT compile key={cache_key}")
        _kernel_cache[cache_key] = _build_kernel(
            B=B, H=H, topk=topk, num_split=num_split,
            block_N=block_N, page_size=page_size,
            page_bytes=page_bytes, scale_off=scale_off,
            sm_scale=softmax_scale,
        )
    kernel = _kernel_cache[cache_key]

    out_partial = torch.empty(
        B, H, num_split, _NOPE_PAD + _ROPE_DIM,
        dtype=torch.float32, device=q.device
    )
    lse_partial = torch.full(
        (B, H, num_split), float("-inf"),
        dtype=torch.float32, device=q.device
    )

    kernel(q_nope, q_rope, raw_fp8, raw_u8, raw_bf16, flat_idx,
           out_partial, lse_partial)

    # Reduce split-KV dimension; only keep the first _D=512 dims of output
    # (nope_pad=512 and rope_dim=64 are stored separately in Out,
    #  but D=512 total head dim = nope 448 + rope 64 → need to combine)
    out_nope = out_partial[..., :_NOPE_DIM]          # [B, H, num_split, 448]
    out_rope = out_partial[..., _NOPE_PAD:_NOPE_PAD + _ROPE_DIM]  # [B,H,ns,64]
    out_combined = torch.cat([out_nope, out_rope], dim=-1)          # [B,H,ns,512]

    out, lse = _splitk_reduce(out_combined, lse_partial)
    return out.unsqueeze(1), lse.unsqueeze(1)    # [B,1,H,D], [B,1,H]


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers (identical to Triton version)
# ─────────────────────────────────────────────────────────────────────────────

def _merge_partial_attn(out1, lse1, out2, lse2):
    max_lse = torch.maximum(lse1, lse2)
    w1 = torch.where(lse1 > -1e20, torch.exp(lse1 - max_lse), torch.zeros_like(lse1))
    w2 = torch.where(lse2 > -1e20, torch.exp(lse2 - max_lse), torch.zeros_like(lse2))
    total  = (w1 + w2).clamp(min=1e-20)
    merged = (w1.unsqueeze(-1) * out1.float() + w2.unsqueeze(-1) * out2.float()) \
             / total.unsqueeze(-1)
    return merged.to(torch.bfloat16), max_lse + torch.log(total)


def _apply_attn_sink(out, lse, attn_sink):
    sink_lse     = attn_sink.view(1, 1, -1).expand_as(lse)
    combined_lse = torch.logaddexp(lse, sink_lse)
    w = torch.where(lse > -1e20, torch.exp(lse - combined_lse), torch.zeros_like(lse))
    return (out.float() * w.unsqueeze(-1)).to(torch.bfloat16), combined_lse


# ─────────────────────────────────────────────────────────────────────────────
#  Public API — identical signature to flash_mla_sparse_decode_triton
# ─────────────────────────────────────────────────────────────────────────────

def flash_mla_sparse_decode_tilelang(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    attn_sink: Optional[torch.Tensor],
    head_dim_v: int,
    softmax_scale: float,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SM89 TileLang sparse MLA decode. Drop-in for flash_mla_sparse_decode_triton.

    Returns:
        out: [B, 1, H, D]  bfloat16
        lse: [B, H, 1]     float32
    """
    if softmax_scale is None:
        softmax_scale = float(q.shape[-1] ** -0.5)

    out, lse = _run_tilelang_sparse_decode(
        q, k_cache, indices, topk_length, softmax_scale
    )

    if extra_k_cache is not None and extra_indices is not None:
        out_ex, lse_ex = _run_tilelang_sparse_decode(
            q, extra_k_cache, extra_indices, extra_topk_length, softmax_scale
        )
        out, lse = _merge_partial_attn(out, lse, out_ex, lse_ex)

    if attn_sink is not None:
        out, lse = _apply_attn_sink(out, lse, attn_sink)

    return out, lse.permute(0, 2, 1)
