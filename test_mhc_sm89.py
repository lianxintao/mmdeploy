import itertools

import pytest
import tilelang
import torch
import triton
import triton.testing


# =============================================================================
# PyTorch reference implementation of the full mhc_pre pipeline
# =============================================================================
def _sinkhorn_normalize(
    comb_logits: torch.Tensor, eps: float, iters: int
) -> torch.Tensor:
    """Sinkhorn normalization to produce a doubly-stochastic matrix.

    Args:
        comb_logits: [N, hc, hc] logits for the combination matrix
        eps: small epsilon for numerical stability
        iters: number of Sinkhorn iterations
    Returns:
        comb: [N, hc, hc] doubly-stochastic matrix
    """
    # Subtract row max for numerical stability (log-sum-exp trick)
    row_max = comb_logits.max(dim=-1, keepdim=True).values
    cm = torch.exp(comb_logits - row_max)

    # First row normalization
    row_sum = cm.sum(dim=-1, keepdim=True)
    cm = cm / row_sum + eps

    # First column normalization
    col_sum = cm.sum(dim=-2, keepdim=True)
    cm = cm / (col_sum + eps)

    # Remaining iterations
    for _ in range(iters - 1):
        row_sum = cm.sum(dim=-1, keepdim=True)
        cm = cm / (row_sum + eps)
        col_sum = cm.sum(dim=-2, keepdim=True)
        cm = cm / (col_sum + eps)

    return cm


def mhc_pre_torch_ref(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure PyTorch reference for mhc_pre.

    Matches the algorithmic flow of the tilelang kernel exactly.
    """
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]

    # Flatten residual
    residual_flat = residual.float().view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]

    # Step 1: GEMM + squared-sum + RMSNorm
    x_flat = residual_flat.flatten(1)  # [N, hc * hidden]
    sqrsum = x_flat.square().sum(dim=-1)  # [N]
    mixes = torch.mm(x_flat, fn.T)  # [N, hc_mult3]
    rms = torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)  # [N]
    mixes = mixes * rms.unsqueeze(-1)  # [N, hc_mult3]

    # Step 2: Split mixes into pre, post, comb logits
    # pre: sigmoid(mixes[:hc] * scale[0] + base[:hc]) + eps
    pre = (
        torch.sigmoid(mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult])
        + hc_pre_eps
    )
    # post: sigmoid(mixes[hc:2*hc] * scale[1] + base[hc:2*hc]) * post_mult
    post = (
        torch.sigmoid(
            mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1]
            + hc_base[hc_mult : 2 * hc_mult]
        )
        * hc_post_mult_value
    )
    # comb logits: mixes[2*hc:] * scale[2] + base[2*hc:]
    comb_logits = (
        mixes[:, 2 * hc_mult :] * hc_scale[2] + hc_base[2 * hc_mult :]
    ).view(num_tokens, hc_mult, hc_mult)

    # Step 3: Sinkhorn normalize the comb matrix
    comb = _sinkhorn_normalize(comb_logits, hc_sinkhorn_eps, sinkhorn_repeat)

    # Step 4: Pre-mixing: y = sum_hc(pre[:, hc] * residual[:, hc, :])
    layer_input = (pre.unsqueeze(-1) * residual_flat).sum(dim=1)

    # Reshape to original
    post = post.view(*outer_shape, hc_mult, 1)
    comb = comb.view(*outer_shape, hc_mult, hc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size).bfloat16()

    return post, comb, layer_input


# =============================================================================
# Test parameters
# =============================================================================
DEVICE = "cuda"

# Typical DeepSeek V4 shapes
# hc_mult=4, hidden_size=4096 -> hc_hidden_size=16384
# hc_mult=4, hidden_size=7168 -> hc_hidden_size=28672
STANDARD_CONFIGS = [
    # (hc_mult, hidden_size, hc_hidden_size)
    (4, 4096, 16384),
    (4, 7168, 28672),
]

# Batch sizes to test
BATCH_SIZES = [
    1,           # single token
    7,           # odd small batch
    32,          # one warp
    64,          # two warps
    128,         # medium
    256,         # medium-large
    512,         # large (split-K range)
    1024,        # larger
    2048,        # boundary
    4096,        # beyond boundary (uses simple kernel)
]

# Parameters
HC_SINKHORN_ITERS = 20
HC_EPS = 1e-6
RMS_EPS = 1e-6
POST_MULT_VALUE = 2.0


def _make_params(hc_mult, hidden_size):
    """Create consistent random parameters."""
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    hc_hidden_size = hc_mult * hidden_size
    fn = torch.randn(hc_mult3, hc_hidden_size, device=DEVICE, dtype=torch.float32) * 0.02
    hc_scale = torch.ones(3, device=DEVICE, dtype=torch.float32) * 0.1
    hc_base = torch.zeros(hc_mult3, device=DEVICE, dtype=torch.float32)
    return fn, hc_scale, hc_base


# =============================================================================
# Test: individual kernels
# =============================================================================
class TestMhcPreGemmSqrsumSM89:
    """Test the GEMM + squared-sum kernels."""

    @pytest.mark.parametrize(
        "batch_size, hc_mult, hidden_size",
        [
            (1, 4, 4096),
            (32, 4, 4096),
            (128, 4, 4096),
            (1024, 4, 4096),
            (4096, 4, 4096),
        ],
    )
    def test_gemm_sqrsum_simple(
        self, batch_size, hc_mult, hidden_size
    ):
        """Test the simple (non-split-K) GEMM kernel."""
        from sglang.srt.layers.mhc_sm89 import mhc_pre_gemm_sqrsum_tilelang_sm89

        hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
        hc_hidden_size = hc_mult * hidden_size

        fn, _, _ = _make_params(hc_mult, hidden_size)

        residual = torch.randn(
            batch_size, hc_mult, hidden_size, device=DEVICE, dtype=torch.bfloat16
        )
        x_flat = residual.view(batch_size, hc_hidden_size)

        # Tilelang output
        out_tl = torch.zeros(batch_size, hc_mult3, device=DEVICE, dtype=torch.float32)
        sqrsum_tl = torch.zeros(batch_size, device=DEVICE, dtype=torch.float32)
        mhc_pre_gemm_sqrsum_tilelang_sm89(
            x_flat, fn, out_tl, sqrsum_tl, hc_mult3, hc_hidden_size
        )

        # Torch reference
        x_f32 = x_flat.float()
        out_ref = torch.mm(x_f32, fn.T)
        sqrsum_ref = x_f32.square().sum(dim=-1)

        triton.testing.assert_close(out_tl, out_ref, atol=5e-2, rtol=5e-2)
        triton.testing.assert_close(sqrsum_tl, sqrsum_ref, atol=5e-2, rtol=5e-2)

    @pytest.mark.parametrize(
        "batch_size, hc_mult, hidden_size",
        [
            (1, 4, 4096),
            (32, 4, 4096),
            (128, 4, 4096),
            (512, 4, 4096),
            (1024, 4, 4096),
            (2048, 4, 4096),
            (1, 4, 7168),
            (32, 4, 7168),
            (128, 4, 7168),
        ],
    )
    def test_gemm_sqrsum_splitk(
        self, batch_size, hc_mult, hidden_size
    ):
        """Test the split-K GEMM kernel."""
        from sglang.srt.layers.mhc_sm89 import mhc_pre_gemm_sqrsum_splitk_kernel_sm89

        hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
        hc_hidden_size = hc_mult * hidden_size

        fn, _, _ = _make_params(hc_mult, hidden_size)

        residual = torch.randn(
            batch_size, hc_mult, hidden_size, device=DEVICE, dtype=torch.bfloat16
        )
        x_flat = residual.view(batch_size, hc_hidden_size)

        split_k = 8
        if hc_hidden_size == 16384:
            hidden_block = 256
        elif hc_hidden_size == 28672:
            hidden_block = 128
        else:
            return

        kernel_0, kernel_1 = mhc_pre_gemm_sqrsum_splitk_kernel_sm89(
            hc_mult3, hc_hidden_size, split_k=split_k, hidden_block=hidden_block
        )

        partial_out = torch.zeros(
            split_k, batch_size, 32, device=DEVICE, dtype=torch.float32
        )
        partial_sqrsum = torch.zeros(
            split_k, batch_size, device=DEVICE, dtype=torch.float32
        )
        kernel_0(x_flat, fn, partial_out, partial_sqrsum)

        out_tl = torch.zeros(batch_size, hc_mult3, device=DEVICE, dtype=torch.float32)
        sqrsum_tl = torch.zeros(batch_size, device=DEVICE, dtype=torch.float32)
        kernel_1(partial_out, partial_sqrsum, out_tl, sqrsum_tl)

        # Torch reference
        x_f32 = x_flat.float()
        out_ref = torch.mm(x_f32, fn.T)
        sqrsum_ref = x_f32.square().sum(dim=-1)

        triton.testing.assert_close(out_tl, out_ref, atol=1e-2, rtol=1e-2)
        triton.testing.assert_close(sqrsum_tl, sqrsum_ref, atol=1e-2, rtol=1e-2)


# =============================================================================
# Test: full mhc_pre pipeline
# =============================================================================
class TestMhcPreSM89:
    """Test the full mhc_pre pipeline end-to-end."""

    @pytest.mark.parametrize(
        "batch_size, hc_mult, hidden_size",
        [
            # Small hidden (4096)
            (1, 4, 4096),
            (32, 4, 4096),
            (128, 4, 4096),
            (256, 4, 4096),
            (512, 4, 4096),
            (1024, 4, 4096),
            (2048, 4, 4096),
            (4096, 4, 4096),
            # Large hidden (7168)
            (1, 4, 7168),
            (32, 4, 7168),
            (128, 4, 7168),
            (512, 4, 7168),
            (2048, 4, 7168),
            (4096, 4, 7168),
            # Edge cases
            (3, 4, 4096),
            (17, 4, 4096),
            (127, 4, 4096),
        ],
    )
    def test_mhc_pre_full(self, batch_size, hc_mult, hidden_size):
        """Full end-to-end test of mhc_pre."""
        from sglang.srt.layers.mhc_sm89 import mhc_pre

        fn, hc_scale, hc_base = _make_params(hc_mult, hidden_size)

        residual = torch.randn(
            batch_size, hc_mult, hidden_size, device=DEVICE, dtype=torch.bfloat16
        )

        # Tilelang SM89 output
        post_tl, comb_tl, layer_input_tl = mhc_pre(
            residual=residual,
            fn=fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            rms_eps=RMS_EPS,
            hc_pre_eps=HC_EPS,
            hc_sinkhorn_eps=HC_EPS,
            hc_post_mult_value=POST_MULT_VALUE,
            sinkhorn_repeat=HC_SINKHORN_ITERS,
        )

        # PyTorch reference
        post_ref, comb_ref, layer_input_ref = mhc_pre_torch_ref(
            residual=residual,
            fn=fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            rms_eps=RMS_EPS,
            hc_pre_eps=HC_EPS,
            hc_sinkhorn_eps=HC_EPS,
            hc_post_mult_value=POST_MULT_VALUE,
            sinkhorn_repeat=HC_SINKHORN_ITERS,
        )

        # Validate shapes
        expected_post_shape = (batch_size, hc_mult, 1)
        expected_comb_shape = (batch_size, hc_mult, hc_mult)
        expected_layer_shape = (batch_size, hidden_size)

        assert post_tl.shape == expected_post_shape, f"{post_tl.shape} != {expected_post_shape}"
        assert comb_tl.shape == expected_comb_shape, f"{comb_tl.shape} != {expected_comb_shape}"
        assert layer_input_tl.shape == expected_layer_shape, f"{layer_input_tl.shape} != {expected_layer_shape}"

        # Validate dtypes
        assert post_tl.dtype == torch.float32
        assert comb_tl.dtype == torch.float32
        assert layer_input_tl.dtype == torch.bfloat16

        # Numerical comparison
        triton.testing.assert_close(post_tl, post_ref, atol=1e-2, rtol=1e-2)
        triton.testing.assert_close(comb_tl, comb_ref, atol=1e-2, rtol=1e-2)
        triton.testing.assert_close(
            layer_input_tl.float(), layer_input_ref.float(), atol=1e-2, rtol=1e-2
        )

    def test_mhc_pre_empty_batch(self):
        """Test that mhc_pre handles zero-token batch gracefully."""
        from sglang.srt.layers.mhc_sm89 import mhc_pre

        hc_mult, hidden_size = 4, 4096
        fn, hc_scale, hc_base = _make_params(hc_mult, hidden_size)
        residual = torch.randn(
            0, hc_mult, hidden_size, device=DEVICE, dtype=torch.bfloat16
        )

        # Empty batch should complete without error
        post, comb, layer_input = mhc_pre(
            residual=residual,
            fn=fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            rms_eps=RMS_EPS,
            hc_pre_eps=HC_EPS,
            hc_sinkhorn_eps=HC_EPS,
            hc_post_mult_value=POST_MULT_VALUE,
            sinkhorn_repeat=HC_SINKHORN_ITERS,
        )
        assert post.shape == (0, hc_mult, 1)
        assert comb.shape == (0, hc_mult, hc_mult)
        assert layer_input.shape == (0, hidden_size)

    def test_mhc_pre_extra_batch_dims(self):
        """Test that mhc_pre preserves extra batch dimensions."""
        from sglang.srt.layers.mhc_sm89 import mhc_pre

        hc_mult, hidden_size = 4, 4096
        fn, hc_scale, hc_base = _make_params(hc_mult, hidden_size)

        # Multi-dimensional batch: [2, 3, hc, hidden]
        residual = torch.randn(
            2, 3, hc_mult, hidden_size, device=DEVICE, dtype=torch.bfloat16
        )

        post, comb, layer_input = mhc_pre(
            residual=residual,
            fn=fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            rms_eps=RMS_EPS,
            hc_pre_eps=HC_EPS,
            hc_sinkhorn_eps=HC_EPS,
            hc_post_mult_value=POST_MULT_VALUE,
            sinkhorn_repeat=HC_SINKHORN_ITERS,
        )

        assert post.shape == (2, 3, hc_mult, 1)
        assert comb.shape == (2, 3, hc_mult, hc_mult)
        assert layer_input.shape == (2, 3, hidden_size)

        # Also verify against reference
        post_ref, comb_ref, layer_input_ref = mhc_pre_torch_ref(
            residual=residual,
            fn=fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            rms_eps=RMS_EPS,
            hc_pre_eps=HC_EPS,
            hc_sinkhorn_eps=HC_EPS,
            hc_post_mult_value=POST_MULT_VALUE,
            sinkhorn_repeat=HC_SINKHORN_ITERS,
        )
        triton.testing.assert_close(post, post_ref, atol=1e-2, rtol=1e-2)
        triton.testing.assert_close(comb, comb_ref, atol=1e-2, rtol=1e-2)
        triton.testing.assert_close(
            layer_input.float(), layer_input_ref.float(), atol=1e-2, rtol=1e-2
        )

    def test_mhc_pre_dtype_checks(self):
        """Test that mhc_pre enforces correct input dtypes."""
        from sglang.srt.layers.mhc_sm89 import mhc_pre

        hc_mult, hidden_size = 4, 4096
        fn, hc_scale, hc_base = _make_params(hc_mult, hidden_size)

        # Wrong residual dtype
        residual_f16 = torch.randn(
            8, hc_mult, hidden_size, device=DEVICE, dtype=torch.float16
        )
        with pytest.raises(AssertionError):
            mhc_pre(
                residual=residual_f16,
                fn=fn,
                hc_scale=hc_scale,
                hc_base=hc_base,
                rms_eps=RMS_EPS,
                hc_pre_eps=HC_EPS,
                hc_sinkhorn_eps=HC_EPS,
                hc_post_mult_value=POST_MULT_VALUE,
                sinkhorn_repeat=HC_SINKHORN_ITERS,
            )

    def test_mhc_pre_shape_checks(self):
        """Test that mhc_pre validates input shapes."""
        from sglang.srt.layers.mhc_sm89 import mhc_pre

        hc_mult, hidden_size = 4, 4096
        fn, hc_scale, hc_base = _make_params(hc_mult, hidden_size)

        residual = torch.randn(
            8, hc_mult, hidden_size, device=DEVICE, dtype=torch.bfloat16
        )

        # Wrong fn shape
        wrong_fn = torch.randn(
            10, hc_mult * hidden_size, device=DEVICE, dtype=torch.float32
        )
        with pytest.raises(AssertionError):
            mhc_pre(
                residual=residual,
                fn=wrong_fn,
                hc_scale=hc_scale,
                hc_base=hc_base,
                rms_eps=RMS_EPS,
                hc_pre_eps=HC_EPS,
                hc_sinkhorn_eps=HC_EPS,
                hc_post_mult_value=POST_MULT_VALUE,
                sinkhorn_repeat=HC_SINKHORN_ITERS,
            )


# =============================================================================
# Numeric consistency: compare SM89 vs original SM90+ implementation
# =============================================================================
class TestMhcPreSM89vsOriginal:
    """Verify SM89 version produces consistent results with original on SM90+."""

    @pytest.mark.parametrize(
        "batch_size, hc_mult, hidden_size",
        [
            (32, 4, 4096),
            (128, 4, 4096),
            (1024, 4, 4096),
            (32, 4, 7168),
        ],
    )
    def test_consistency_with_original(self, batch_size, hc_mult, hidden_size):
        """Compare SM89 and original implementations when available."""
        import inspect

        import tilelang.language as T_lang

        # Check if original mhc can compile (requires wg_wait support)
        gemm_params = set(inspect.signature(T_lang.gemm).parameters.keys())
        if "wg_wait" not in gemm_params:
            pytest.skip(
                "Original mhc_pre requires tilelang with wg_wait support "
                "(tilelang >= 0.1.10). This test runs on tilelang "
                f"{tilelang.__version__}."
            )

        from sglang.srt.layers.mhc import (
            mhc_pre as mhc_pre_orig,
        )
        from sglang.srt.layers.mhc_sm89 import (
            mhc_pre as mhc_pre_sm89,
        )

        fn, hc_scale, hc_base = _make_params(hc_mult, hidden_size)

        residual = torch.randn(
            batch_size, hc_mult, hidden_size, device=DEVICE, dtype=torch.bfloat16
        )

        r1 = residual.clone()
        r2 = residual.clone()

        post_orig, comb_orig, layer_orig = mhc_pre_orig(
            residual=r1,
            fn=fn.clone(),
            hc_scale=hc_scale.clone(),
            hc_base=hc_base.clone(),
            rms_eps=RMS_EPS,
            hc_pre_eps=HC_EPS,
            hc_sinkhorn_eps=HC_EPS,
            hc_post_mult_value=POST_MULT_VALUE,
            sinkhorn_repeat=HC_SINKHORN_ITERS,
        )

        post_sm89, comb_sm89, layer_sm89 = mhc_pre_sm89(
            residual=r2,
            fn=fn.clone(),
            hc_scale=hc_scale.clone(),
            hc_base=hc_base.clone(),
            rms_eps=RMS_EPS,
            hc_pre_eps=HC_EPS,
            hc_sinkhorn_eps=HC_EPS,
            hc_post_mult_value=POST_MULT_VALUE,
            sinkhorn_repeat=HC_SINKHORN_ITERS,
        )

        triton.testing.assert_close(post_sm89, post_orig, atol=1e-2, rtol=1e-2)
        triton.testing.assert_close(comb_sm89, comb_orig, atol=1e-2, rtol=1e-2)
        triton.testing.assert_close(
            layer_sm89.float(), layer_orig.float(), atol=1e-2, rtol=1e-2
        )


if __name__ == "__main__":
    pytest.main([__file__])
