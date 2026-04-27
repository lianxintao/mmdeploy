
    # Mirror the dispatch logic used by fused_moe.py's CUDA reduce path:
    # - topk == 1 + scale == 1.0: directly copy intermediate to output
    # - topk == 2 + scale == 1.0: use torch.add to avoid an extra reduce kernel
    # - otherwise: torch.compile variant is faster for small batches per
    #   micro benchmark, native sgl_kernel.moe_sum_reduce wins for larger ones
    if topk == 1 and routed_scaling_factor == 1.0:
        output.copy_(intermediate_cache3.view(M, K))
    elif topk == 2 and routed_scaling_factor == 1.0:
        torch.add(
            intermediate_cache3[:, 0],
            intermediate_cache3[:, 1],
            out=output,
        )
    else:
        # According to micro benchmark results, torch.compile can get better
        # performance for small token.
        if M <= 32:
            moe_sum_reduce_torch_compile(
                intermediate_cache3.view(*intermediate_cache3.shape),
                output,
                routed_scaling_factor,
            )
        else:
            moe_sum_reduce(
                intermediate_cache3.view(*intermediate_cache3.shape),
                output,
                routed_scaling_factor,
            )
    return output



--mamba-scheduler-strategy extra_buffer \
    --page-size 64 \
    --mamba-track-interval 256  # 必须能被page_size整除
--mamba-full-memory-ratio 
--mamba-ssm-dtype bfloat16


https://app.warp.dev/conversation/a14fd851-85a5-4d53-82bc-961da9724d1e
**Grep Results:**

- **/mnt/ssd_data/ssd_projects/warp/sglang/python/sglang/jit_kernel/benchmark/bench_hicache.py**
- **/mnt/ssd_data/ssd_projects/warp/sglang/sgl-kernel/include/sgl_kernel_ops.h**
- **/mnt/ssd_data/ssd_projects/warp/sglang/python/sglang/srt/hardware_backend/npu/utils.py**
- **/mnt/ssd_data/ssd_projects/warp/sglang/benchmark/hicache/perf.py**
- **/mnt/ssd_data/ssd_projects/warp/sglang/sgl-kernel/csrc/common_extension.cc**
- **/mnt/ssd_data/ssd_projects/warp/sglang/sgl-kernel/python/sgl_kernel/__init__.py**
- **/mnt/ssd_data/ssd_projects/warp/sglang/sgl-kernel/csrc/kvcacheio/transfer.cu**
- **/mnt/ssd_data/ssd_projects/warp/sglang/sgl-kernel/python/sgl_kernel/kvcacheio.py**
- **/mnt/ssd_data/ssd_projects/warp/sglang/test/registered/unit/server_args/test_server_args.py**
- **/mnt/ssd_data/ssd_projects/warp/sglang/python/sglang/srt/mem_cache/hi_mamba_radix_cache.py**
- **/mnt/ssd_data/ssd_projects/warp/sglang/python/sglang/jit_kernel/hicache.py**
- **/mnt/ssd_data/ssd_projects/warp/sglang/benchmark/hf3fs/bench_zerocopy.py**
- **/mnt/ssd_data/ssd_projects/warp/sglang/python/sglang/srt/mem_cache/hiradix_cache.py**
- **/mnt/ssd_data/ssd_projects/warp/sglang/python/sglang/srt/mem_cache/memory_pool_host.py**
- **/mnt/ssd_data/ssd_projects/warp/sglang/sgl-kernel/tests/test_kvcacheio.py**
- **/mnt/ssd_data/ssd_projects/warp/sglang/python/sglang/srt/server_args.py**
- **/mnt/ssd_data/ssd_projects/warp/sglang/python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py**
- **/mnt/ssd_data/ssd_projects/warp/sglang/python/sglang/jit_kernel/csrc/hicache.cuh**
- **/mnt/ssd_data/ssd_projects/warp/sglang/sgl-kernel/csrc/common_extension_rocm.cc**




**Files Read:**

**/mnt/ssd_data/ssd_projects/warp/sglang/benchmark/hicache/perf.py**
```
from __future__ import annotations

from typing import Any, Callable, NamedTuple

import torch


def jit_hicache_impl(
    k_cache_dst: torch.Tensor,
    v_cache_dst: torch.Tensor,
    indices_dst: torch.Tensor,
    k_cache_src: torch.Tensor,
    v_cache_src: torch.Tensor,
    indices_src: torch.Tensor,
    item_bytes: int,
    block_quota: int,
) -> None:
    from sglang.jit_kernel.hicache import transfer_hicache_one_layer

    _ = item_bytes

    transfer_hicache_one_layer(
        k_cache_dst=k_cache_dst,
        v_cache_dst=v_cache_dst,
        indices_dst=indices_dst,
        k_cache_src=k_cache_src,
        v_cache_src=v_cache_src,
        indices_src=indices_src,
        block_quota=block_quota,
    )


def ref_hicache_impl(
    k_cache_dst: torch.Tensor,
    v_cache_dst: torch.Tensor,
    indices_dst: torch.Tensor,
    k_cache_src: torch.Tensor,
    v_cache_src: torch.Tensor,
    indices_src: torch.Tensor,
    item_bytes: int,
    block_quota: int,
) -> None:
    from sgl_kernel import transfer_kv_per_layer

    transfer_kv_per_layer(
        src_k=k_cache_src,
        src_v=v_cache_src,
        dst_k=k_cache_dst,
        dst_v=v_cache_dst,
        src_indices=indices_src,
        dst_indices=indices_dst,
        item_size=item_bytes,
        block_quota=block_quota,
    )


class HicacheBenchArgs(NamedTuple):
    cache_item_size: int
    dtype: torch.dtype
    block_quota: int


def perf(f: Callable[[], Any], loop: int = 100) -> float:
    tic = torch.cuda.Event(enable_timing=True)
    toc = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    # warm up
    f()
    torch.cuda._sleep(10**8)
    tic.record()
    for _ in range(loop):
        f()
    toc.record()
    toc.synchronize()
    return tic.elapsed_time(toc) / loop


@torch.inference_mode()
def test_hicache_kernel(args: HicacheBenchArgs) -> None:
    CACHE_ITEM_SIZE, DTYPE, BLOCK_QUOTA = args

    CUDA_CACHE_SIZE = 1024 * 1024
    HOST_CACHE_SIZE = CUDA_CACHE_SIZE * 2

    cuda_cache = torch.randn(
        (2, CUDA_CACHE_SIZE, CACHE_ITEM_SIZE),
        dtype=DTYPE,
        device="cuda",
    )
    host_cache = torch.empty(
        (2, HOST_CACHE_SIZE, CACHE_ITEM_SIZE),
        dtype=DTYPE,
        device="cpu",
        pin_memory=True,
    )

    ITEM_BYTES = cuda_cache.element_size() * CACHE_ITEM_SIZE

    def _gen_indices(size: int, bs: int) -> torch.Tensor:
        assert bs <= size
        result = (
            (torch.randperm(size, dtype=torch.int64, device="cuda")[:bs]).sort().values
        )
        if not (torch.all(result >= 0) and torch.all(result < size)):
            where = (result < 0) | (result >= size)
            place = where.nonzero(as_tuple=False)
            print("Invalid indices at positions:", place)
            print("Invalid indices values:", result[place])
            raise ValueError("Generated invalid indices")
        return result

    def _calc_tput(dur: float) -> float:
        return (MEM / (1024**3)) / (dur / 1000)  # GB/s

    def _gain_str(aot_dur: float, jit_dur: float) -> str:
        gain = 100 * (aot_dur / jit_dur - 1)
        if gain >= 0:
            return f"+{gain:>6.2f}%"
        else:
            return f"-{-gain:>6.2f}%"

    print(f"{CACHE_ITEM_SIZE = }, {DTYPE = }, {BLOCK_QUOTA = }")

    def _fast_test_correctness(bs: int):
        src_indices = _gen_indices(CUDA_CACHE_SIZE, bs)
        dst_indices = _gen_indices(HOST_CACHE_SIZE, bs)
        host_cache_cuda = torch.randn_like(host_cache, device="cuda")
        host_cache.copy_(host_cache_cuda, non_blocking=True)

        # copy from cuda to host
        jit_hicache_impl(
            k_cache_dst=host_cache[0],
            v_cache_dst=host_cache[1],
            indices_dst=dst_indices,
            k_cache_src=cuda_cache[0],
            v_cache_src=cuda_cache[1],
            indices_src=src_indices,
            item_bytes=ITEM_BYTES,
            block_quota=BLOCK_QUOTA,
        )
        dst_indices = dst_indices.cpu()
        assert torch.all(
            host_cache[0][dst_indices].cuda() == cuda_cache[0][src_indices]
        )

    BS_RANGE = [2**n for n in range(8, 18)]
    for bs in BS_RANGE:
        _fast_test_correctness(bs)

    print("Correctness passed! Start HiCache kernel performance test...")
    print("=" * 70)

    for bs in BS_RANGE:
        indices_dst = _gen_indices(CUDA_CACHE_SIZE, bs)
        indices_src = _gen_indices(HOST_CACHE_SIZE, bs)
        MEM = 2 * bs * ITEM_BYTES

        def _run_kernel_h2d(impl):
            return impl(
                k_cache_dst=cuda_cache[0],
                v_cache_dst=cuda_cache[1],
                indices_dst=indices_dst,
                k_cache_src=host_cache[0],
                v_cache_src=host_cache[1],
                indices_src=indices_src,
                item_bytes=ITEM_BYTES,
                block_quota=BLOCK_QUOTA,
            )

        our_h2d_dur = perf(lambda: _run_kernel_h2d(jit_hicache_impl))
        ref_h2d_dur = perf(lambda: _run_kernel_h2d(ref_hicache_impl))
        print(
            f"{bs = :6d}, H->D",
            f"| aot {_calc_tput(ref_h2d_dur):<6.2f} GB/s",
            f"| jit {_calc_tput(our_h2d_dur):<6.2f} GB/s",
            f"| {_gain_str(ref_h2d_dur, our_h2d_dur)}",
        )

    print("=" * 70)

    for bs in BS_RANGE:
        indices_dst = _gen_indices(HOST_CACHE_SIZE, bs)
        indices_src = _gen_indices(CUDA_CACHE_SIZE, bs)
        MEM = 2 * bs * ITEM_BYTES

        def _run_kernel_d2h(impl):
            return impl(
                k_cache_dst=host_cache[0],
                v_cache_dst=host_cache[1],
                indices_dst=indices_dst,
                k_cache_src=cuda_cache[0],
                v_cache_src=cuda_cache[1],
                indices_src=indices_src,
                item_bytes=ITEM_BYTES,
                block_quota=BLOCK_QUOTA,
            )

        our_d2h_dur = perf(lambda: _run_kernel_d2h(jit_hicache_impl))
        ref_d2h_dur = perf(lambda: _run_kernel_d2h(ref_hicache_impl))
        print(
            f"{bs = :6d}, D->H",
            f"| aot {_calc_tput(ref_d2h_dur):<6.2f} GB/s",
            f"| jit {_calc_tput(our_d2h_dur):<6.2f} GB/s",
            f"| {_gain_str(ref_d2h_dur, our_d2h_dur)}",
        )

    print("=" * 70)


def main() -> None:
    torch.cuda.set_device(0)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    tic = torch.cuda.Event(enable_timing=True)
    toc = torch.cuda.Event(enable_timing=True)

    BUF_SIZE = 1024 * 1024 * 1024
    cuda_mem = torch.empty(BUF_SIZE, dtype=torch.uint8, device="cuda")
    host_mem = torch.empty(BUF_SIZE, dtype=torch.uint8, device="cpu", pin_memory=True)

    # test peak bandwidth
    tic.record()
    cuda_mem.copy_(host_mem, non_blocking=True)
    toc.record()
    toc.synchronize()
    dur = tic.elapsed_time(toc)
    print(f"Peak H->D Bandwidth: {(BUF_SIZE / (1024**3)) / (dur / 1000):.2f} GB/s")

    tic.record()
    host_mem.copy_(cuda_mem, non_blocking=True)
    toc.record()
    toc.synchronize()
    dur = tic.elapsed_time(toc)
    print(f"Peak D->H Bandwidth: {(BUF_SIZE / (1024**3)) / (dur / 1000):.2f} GB/s")

    for block_quota in [1, 2, 3, 4]:
        for cache_item_size in [128, 256, 512, 1024]:
            args = HicacheBenchArgs(
                cache_item_size=cache_item_size,
                dtype=torch.float16,
                block_quota=block_quota,
            )
            test_hicache_kernel(args)


if __name__ == "__main__":
    main()
```

**/mnt/ssd_data/ssd_projects/warp/sglang/python/sglang/jit_kernel/benchmark/bench_hicache.py**
```
"""Benchmark for HiCache JIT kernel performance.

This benchmark tests the performance of KV cache transfer operations
between GPU and CPU (host pinned memory), comparing:
- SGL AOT Kernel: Pre-compiled transfer_kv kernels from sgl_kernel
- SGL JIT Kernel: JIT-compiled hicache kernels
- PyTorch Indexing: Plain PyTorch index copy
- PyTorch 2 Stream: PyTorch implementation using 2 CUDA streams

Tests cover:
- One Layer: CPU->GPU
- All Layer: GPU->CPU

Note: Uses do_bench instead of do_bench_cudagraph since CUDA graph
capture doesn't support CPU-GPU memory transfers.
"""

import itertools
import os
from dataclasses import dataclass
from typing import Tuple

import torch
import triton
import triton.testing
from sgl_kernel import transfer_kv_all_layer, transfer_kv_per_layer

from sglang.jit_kernel.benchmark.utils import DEFAULT_QUANTILES, get_benchmark_range
from sglang.jit_kernel.hicache import (
    can_use_hicache_jit_kernel,
    transfer_hicache_all_layer,
    transfer_hicache_one_layer,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=29, suite="stage-b-kernel-benchmark-1-gpu-large")

DISABLE_TORCH = os.environ.get("DISABLE_TORCH", "0") == "1"
PAGE_SIZE = 1
ENABLE_SORT = True
GPU_CACHE_SIZE = 256 * 1024  # 256K tokens on GPU
HOST_CACHE_SIZE = 512 * 1024  # 512K tokens on CPU
NUM_LAYERS = 8


@dataclass(frozen=True)
class HiCacheCache:
    k_cache_cuda: torch.Tensor
    v_cache_cuda: torch.Tensor
    k_cache_host: torch.Tensor
    v_cache_host: torch.Tensor

    def get_slice(self, num_layers: int, element_size: int) -> "HiCacheCache":
        def slice_cuda(t: torch.Tensor) -> torch.Tensor:
            needed_cuda = num_layers * GPU_CACHE_SIZE
            return t.view(-1, element_size)[:needed_cuda].unflatten(0, (num_layers, -1))

        def slice_host(t: torch.Tensor) -> torch.Tensor:
            needed_host = num_layers * HOST_CACHE_SIZE
            return t.view(-1, element_size)[:needed_host].unflatten(0, (num_layers, -1))

        return HiCacheCache(
            k_cache_cuda=slice_cuda(self.k_cache_cuda),
            v_cache_cuda=slice_cuda(self.v_cache_cuda),
            k_cache_host=slice_host(self.k_cache_host),
            v_cache_host=slice_host(self.v_cache_host),
        )


def gen_indices(
    size: int, max_size: int, *, page_size: int = PAGE_SIZE
) -> torch.Tensor:
    def align(x: int) -> int:
        return (x + page_size - 1) // page_size

    assert size <= max_size and max_size % page_size == 0
    indices = torch.randperm(align(max_size))[: align(size)]
    offsets = torch.arange(page_size)
    return (indices[:, None] * page_size + offsets).flatten().cuda()[:size]


def sglang_aot_transfer_one(
    k_cache_dst: torch.Tensor,
    v_cache_dst: torch.Tensor,
    indices_dst: torch.Tensor,
    k_cache_src: torch.Tensor,
    v_cache_src: torch.Tensor,
    indices_src: torch.Tensor,
    item_size: int,
) -> None:
    """SGL AOT Kernel for single layer transfer."""
    transfer_kv_per_layer(
        k_cache_src,
        k_cache_dst,
        v_cache_src,
        v_cache_dst,
        indices_src,
        indices_dst,
        item_size,
    )


def sglang_jit_transfer_one(
    k_cache_dst: torch.Tensor,
    v_cache_dst: torch.Tensor,
    indices_dst: torch.Tensor,
    k_cache_src: torch.Tensor,
    v_cache_src: torch.Tensor,
    indices_src: torch.Tensor,
    element_dim: int,
) -> None:
    """SGL JIT Kernel for single layer transfer."""
    transfer_hicache_one_layer(
        k_cache_dst,
        v_cache_dst,
        indices_dst,
        k_cache_src,
        v_cache_src,
        indices_src,
        element_dim=element_dim,
    )


def sglang_aot_transfer_all(
    k_ptrs_dst: torch.Tensor,
    v_ptrs_dst: torch.Tensor,
    indices_dst: torch.Tensor,
    k_ptrs_src: torch.Tensor,
    v_ptrs_src: torch.Tensor,
    indices_src: torch.Tensor,
    item_size: int,
    num_layers: int,
) -> None:
    """SGL AOT Kernel for all layer transfer."""
    transfer_kv_all_layer(
        k_ptrs_src,
        k_ptrs_dst,
        v_ptrs_src,
        v_ptrs_dst,
        indices_src,
        indices_dst,
        item_size,
        num_layers,
    )


def sglang_jit_transfer_all(
    k_ptrs_dst: torch.Tensor,
    v_ptrs_dst: torch.Tensor,
    indices_dst: torch.Tensor,
    k_ptrs_src: torch.Tensor,
    v_ptrs_src: torch.Tensor,
    indices_src: torch.Tensor,
    stride_bytes: int,
    element_size: int,
) -> None:
    """SGL JIT Kernel for all layer transfer."""
    transfer_hicache_all_layer(
        k_ptrs_dst,
        v_ptrs_dst,
        indices_dst,
        k_ptrs_src,
        v_ptrs_src,
        indices_src,
        kv_cache_src_stride_bytes=stride_bytes,
        kv_cache_dst_stride_bytes=stride_bytes,
        element_size=element_size,
    )


def pytorch_transfer(
    k_cache_dst: torch.Tensor,
    v_cache_dst: torch.Tensor,
    indices_dst_on_dst: torch.Tensor,
    k_cache_src: torch.Tensor,
    v_cache_src: torch.Tensor,
    indices_src_on_src: torch.Tensor,
) -> None:
    """PyTorch indexing baseline."""
    dst_device = k_cache_dst.device
    k_cache_dst[indices_dst_on_dst] = k_cache_src[indices_src_on_src].to(dst_device)
    v_cache_dst[indices_dst_on_dst] = v_cache_src[indices_src_on_src].to(dst_device)


# Benchmark configuration

BS_RANGE = get_benchmark_range(
    full_range=[2**n for n in range(0, 16)],
    ci_range=[16],
)
ELEMENT_SIZE_RANGE = get_benchmark_range(
    full_range=[64, 128, 256, 512, 1024],
    ci_range=[1024],
)

LINE_VALS = ["aot", "jit", "torch"]
LINE_NAMES = ["SGL AOT Kernel", "SGL JIT Kernel", "PyTorch"]
STYLES = [("orange", "-"), ("blue", "--"), ("red", ":")]

CONFIGS = list(itertools.product(ELEMENT_SIZE_RANGE, BS_RANGE))


# =============================================================================
# One Layer Benchmarks
# =============================================================================


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["element_size", "batch_size"],
        x_vals=CONFIGS,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="hicache-one-layer-h2d",
        args={},
    )
)
def benchmark_one_layer_h2d(
    element_size: int, batch_size: int, provider: str
) -> Tuple[float, float, float]:
    """One Layer: Host (CPU) -> Device (GPU)."""
    global cache
    cache_local = cache.get_slice(num_layers=NUM_LAYERS, element_size=element_size)
    k_cache_src = cache_local.k_cache_host
    v_cache_src = cache_local.v_cache_host
    k_cache_dst = cache_local.k_cache_cuda
    v_cache_dst = cache_local.v_cache_cuda
    torch.manual_seed(batch_size * 65536 + element_size)
    indices_src_gpu = gen_indices(batch_size, HOST_CACHE_SIZE)
    indices_dst_gpu = gen_indices(batch_size, GPU_CACHE_SIZE)

    if ENABLE_SORT:
        indices_src_gpu, mapping = indices_src_gpu.sort()
        indices_dst_gpu = indices_dst_gpu[mapping]
    indices_src_cpu = indices_src_gpu.cpu()
    torch.cuda.synchronize()

    element_bytes = element_size * k_cache_src.element_size()

    FN_MAP = {
        "aot": lambda: [
            sglang_aot_transfer_one(
                k_cache_dst[i],
                v_cache_dst[i],
                indices_dst_gpu,
                k_cache_src[i],
                v_cache_src[i],
                indices_src_gpu,
                element_bytes,
            )
            for i in range(NUM_LAYERS)
        ],
        "jit": lambda: [
            sglang_jit_transfer_one(
                k_cache_dst[i],
                v_cache_dst[i],
                indices_dst_gpu,
                k_cache_src[i],
                v_cache_src[i],
                indices_src_gpu,
                element_size,
            )
            for i in range(NUM_LAYERS)
        ],
        "torch": lambda: [
            pytorch_transfer(
                k_cache_dst[i],
                v_cache_dst[i],
                indices_dst_gpu,
                k_cache_src[i],
                v_cache_src[i],
                indices_src_cpu,
            )
            for i in range(NUM_LAYERS)
        ],
    }

    if provider == "jit" and not can_use_hicache_jit_kernel(element_size=element_bytes):
        return (float("nan"), float("nan"), float("nan"))

    if DISABLE_TORCH and provider in ["torch"]:
        return (float("nan"), float("nan"), float("nan"))

    ms, min_ms, max_ms = triton.testing.do_bench(  # type: ignore
        FN_MAP[provider], quantiles=DEFAULT_QUANTILES, warmup=5, rep=25
    )
    return (
        1000 * ms / NUM_LAYERS,
        1000 * max_ms / NUM_LAYERS,
        1000 * min_ms / NUM_LAYERS,
    )


# =============================================================================
# All Layer Benchmarks
# =============================================================================


def _create_ptr_tensor(tensors, device="cuda"):
    """Create a tensor of data pointers."""
    return torch.tensor(
        [t.data_ptr() for t in tensors],
        dtype=torch.uint64,
        device=device,
    )


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["element_size", "batch_size"],
        x_vals=CONFIGS,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="hicache-all-layer-d2h",
        args={},
    )
)
def benchmark_all_layer_d2h(
    element_size: int, batch_size: int, provider: str
) -> Tuple[float, float, float]:
    """All Layer: Device (GPU) -> Host (CPU)."""
    global cache
    cache_local = cache.get_slice(num_layers=NUM_LAYERS, element_size=element_size)
    k_caches_src = cache_local.k_cache_cuda
    v_caches_src = cache_local.v_cache_cuda
    k_caches_dst = cache_local.k_cache_host
    v_caches_dst = cache_local.v_cache_host
    torch.manual_seed(batch_size * 65536 + element_size)

    indices_src_gpu = gen_indices(batch_size, GPU_CACHE_SIZE)
    indices_dst_gpu = gen_indices(batch_size, HOST_CACHE_SIZE)
    if ENABLE_SORT:
        indices_dst_gpu, mapping = indices_dst_gpu.sort()
        indices_src_gpu = indices_src_gpu[mapping]
    indices_dst_cpu = indices_dst_gpu.cpu()
    torch.cuda.synchronize()

    element_bytes = element_size * k_caches_src.element_size()

    k_ptrs_src = _create_ptr_tensor([k_caches_src[i] for i in range(NUM_LAYERS)])
    v_ptrs_src = _create_ptr_tensor([v_caches_src[i] for i in range(NUM_LAYERS)])
    k_ptrs_dst = _create_ptr_tensor([k_caches_dst[i] for i in range(NUM_LAYERS)])
    v_ptrs_dst = _create_ptr_tensor([v_caches_dst[i] for i in range(NUM_LAYERS)])

    FN_MAP = {
        "aot": lambda: sglang_aot_transfer_all(
            k_ptrs_dst,
            v_ptrs_dst,
            indices_dst_gpu,
            k_ptrs_src,
            v_ptrs_src,
            indices_src_gpu,
            element_bytes,
            NUM_LAYERS,
        ),
        "jit": lambda: sglang_jit_transfer_all(
            k_ptrs_dst,
            v_ptrs_dst,
            indices_dst_gpu,
            k_ptrs_src,
            v_ptrs_src,
            indices_src_gpu,
            element_bytes,
            element_bytes,
        ),
        "torch": lambda: [
            pytorch_transfer(
                k_caches_dst[i],
                v_caches_dst[i],
                indices_dst_cpu,
                k_caches_src[i],
                v_caches_src[i],
                indices_src_gpu,
            )
            for i in range(NUM_LAYERS)
        ],
    }

    if provider == "jit" and not can_use_hicache_jit_kernel(element_size=element_bytes):
        return (float("nan"), float("nan"), float("nan"))

    if DISABLE_TORCH and provider in ["torch"]:
        return (float("nan"), float("nan"), float("nan"))

    ms, min_ms, max_ms = triton.testing.do_bench(  # type: ignore
        FN_MAP[provider], quantiles=DEFAULT_QUANTILES, warmup=5, rep=25
    )
    return (
        1000 * ms / NUM_LAYERS,
        1000 * max_ms / NUM_LAYERS,
        1000 * min_ms / NUM_LAYERS,
    )


if __name__ == "__main__":
    MAX_SIZE = max(ELEMENT_SIZE_RANGE)
    DEVICE_SHAPE = (NUM_LAYERS * GPU_CACHE_SIZE, MAX_SIZE)
    HOST_SHAPE = (NUM_LAYERS * HOST_CACHE_SIZE, MAX_SIZE)

    cache = HiCacheCache(
        k_cache_cuda=torch.empty(DEVICE_SHAPE, dtype=torch.bfloat16, device="cuda"),
        v_cache_cuda=torch.empty(DEVICE_SHAPE, dtype=torch.bfloat16, device="cuda"),
        k_cache_host=torch.empty(HOST_SHAPE, dtype=torch.bfloat16, pin_memory=True),
        v_cache_host=torch.empty(HOST_SHAPE, dtype=torch.bfloat16, pin_memory=True),
    )

    print("=" * 60)
    print("One Layer: Host -> Device (CPU -> GPU)")
    print("=" * 60)
    benchmark_one_layer_h2d.run(print_data=True)

    print("\n" + "=" * 60)
    print("All Layer: Device -> Host (GPU -> CPU) [per-layer avg]")
    print("=" * 60)
    benchmark_all_layer_d2h.run(print_data=True)
```





**Files Read:**

**/mnt/ssd_data/ssd_projects/warp/sglang/sgl-kernel/tests/test_kvcacheio.py**
```
import sys

import pytest
import torch
from sgl_kernel.kvcacheio import (
    transfer_kv_all_layer,
    transfer_kv_all_layer_direct_lf_pf,
    transfer_kv_all_layer_lf_ph,
    transfer_kv_all_layer_mla,
    transfer_kv_direct,
    transfer_kv_per_layer,
    transfer_kv_per_layer_direct_pf_lf,
    transfer_kv_per_layer_mla,
)

from sglang.srt.utils import is_hip


def ref_copy_with_indices(src_pool, dst_pool, src_indices, dst_indices):
    dst_pool[dst_indices] = src_pool[src_indices].to(dst_pool.device)


def ref_copy_with_indices_pf_direct(
    src_pool, dst_pool, src_indices, dst_indices, page_size, layer_id, lf_to_pf=False
):
    if lf_to_pf:
        for i in range(0, len(src_indices), page_size):
            dst_pool[dst_indices[i] // page_size][layer_id] = src_pool[layer_id][
                src_indices[i : i + page_size]
            ].to(dst_pool.device)
    else:
        for i in range(0, len(src_indices), page_size):
            dst_pool[layer_id][dst_indices[i : i + page_size]] = src_pool[
                src_indices[i] // page_size
            ][layer_id].to(dst_pool.device)


def ref_copy_with_indices_page_head(
    src_pool,
    dst_pool,
    src_indices,
    dst_indices,
    page_size,
    layer_id,
    head_num,
    lf_to_ph=False,
):
    if lf_to_ph:
        for head_id in range(head_num):
            for i in range(0, len(src_indices)):
                dst_pool[dst_indices[i] // page_size][head_id][
                    dst_indices[i] % page_size
                ][layer_id] = src_pool[layer_id][src_indices[i]][head_id].to(
                    dst_pool.device
                )
    else:
        for head_id in range(head_num):
            for i in range(0, len(src_indices)):
                dst_pool[layer_id][dst_indices[i]][head_id] = src_pool[
                    src_indices[i] // page_size
                ][head_id][src_indices[i] % page_size][layer_id].to(dst_pool.device)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_items_to_transfer", [1, 128, 1024])
@pytest.mark.parametrize("page_size", [1, 16, 64])
@pytest.mark.parametrize("item_size", [256])
@pytest.mark.parametrize("total_items_in_pool", [10240])
@pytest.mark.parametrize("is_mla", [False, True])
@pytest.mark.parametrize("all_layers", [False, True])
def test_transfer_kv(
    dtype: torch.dtype,
    num_items_to_transfer: int,
    item_size: int,
    page_size: int,
    total_items_in_pool: int,
    is_mla: bool,
    all_layers: bool,
):
    """
    Tests the per-layer transfer functions, treating tensors as memory pools.
    """

    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    device = "cuda"
    torch.cuda.manual_seed(42)

    num_layers = 4  # A small number of layers for pool creation

    total_pages_in_pool = total_items_in_pool // page_size
    num_pages_to_transfer = num_items_to_transfer // page_size
    if num_pages_to_transfer == 0:
        torch.set_default_dtype(original_dtype)
        return
    page_indices = torch.randperm(total_pages_in_pool, dtype=torch.int64)
    src_indices_host = torch.cat(
        [
            torch.arange(p * page_size, (p + 1) * page_size)
            for p in page_indices[:num_pages_to_transfer]
        ]
    )
    src_indices_device = src_indices_host.to(device)
    dst_indices_host = torch.cat(
        [
            torch.arange(p * page_size, (p + 1) * page_size)
            for p in page_indices[num_pages_to_transfer : 2 * num_pages_to_transfer]
        ]
    )
    dst_indices_device = dst_indices_host.to(device)

    # Prepare memory pools based on whether it's an MLA case.
    if is_mla:
        src_pool_host = torch.randn(
            num_layers, total_items_in_pool, item_size
        ).pin_memory()
        dst_pool_ref = torch.zeros_like(src_pool_host).to(device)
        dst_pool_kernel = torch.zeros_like(dst_pool_ref)
        dst_pool_direct = torch.zeros_like(dst_pool_ref)
    else:
        src_k_pool = torch.randn(
            num_layers, total_items_in_pool, item_size
        ).pin_memory()
        src_v_pool = torch.randn(
            num_layers, total_items_in_pool, item_size
        ).pin_memory()
        dst_k_pool_ref = torch.zeros_like(src_k_pool).to(device)
        dst_v_pool_ref = torch.zeros_like(src_v_pool).to(device)
        dst_k_pool_kernel = torch.zeros_like(dst_k_pool_ref)
        dst_v_pool_kernel = torch.zeros_like(dst_v_pool_ref)
        dst_k_pool_direct = torch.zeros_like(dst_k_pool_ref)
        dst_v_pool_direct = torch.zeros_like(dst_v_pool_ref)

    torch.cuda.synchronize()

    # We will test the per-layer function on the first layer (index 0) of the pool.
    layer_idx_to_test = 0

    if is_mla:
        if not all_layers:
            ref_copy_with_indices(
                src_pool_host[layer_idx_to_test],
                dst_pool_ref[layer_idx_to_test],
                src_indices_host,
                dst_indices_device,
            )
            transfer_kv_per_layer_mla(
                src_pool_host[layer_idx_to_test],
                dst_pool_kernel[layer_idx_to_test],
                src_indices_device,
                dst_indices_device,
                item_size=item_size * dtype.itemsize,
            )
            transfer_kv_direct(
                [src_pool_host[layer_idx_to_test]],
                [dst_pool_direct[layer_idx_to_test]],
                src_indices_host,
                dst_indices_device,
                page_size=page_size,
            )
        else:
            for layer_id in range(num_layers):
                ref_copy_with_indices(
                    src_pool_host[layer_id],
                    dst_pool_ref[layer_id],
                    src_indices_host,
                    dst_indices_device,
                )
            src_layers_device = torch.tensor(
                [src_pool_host[layer_id].data_ptr() for layer_id in range(num_layers)],
                dtype=torch.uint64,
                device=device,
            )
            dst_layers_device = torch.tensor(
                [
                    dst_pool_kernel[layer_id].data_ptr()
                    for layer_id in range(num_layers)
                ],
                dtype=torch.uint64,
                device=device,
            )
            transfer_kv_all_layer_mla(
                src_layers_device,
                dst_layers_device,
                src_indices_device,
                dst_indices_device,
                item_size=item_size * dtype.itemsize,
                num_layers=num_layers,
            )
            transfer_kv_direct(
                [src_pool_host[layer_id] for layer_id in range(num_layers)],
                [dst_pool_direct[layer_id] for layer_id in range(num_layers)],
                src_indices_host,
                dst_indices_device,
                page_size=page_size,
            )
        torch.cuda.synchronize()
        torch.testing.assert_close(dst_pool_kernel, dst_pool_ref)
        torch.testing.assert_close(dst_pool_direct, dst_pool_ref)
```





**Files Read:**

**/mnt/ssd_data/ssd_projects/warp/sglang/sgl-kernel/python/sgl_kernel/kvcacheio.py**
```
from typing import List

import torch


def is_hip() -> bool:
    return torch.version.hip is not None


_is_hip = is_hip()


def transfer_kv_per_layer(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_per_layer.default(
        src_k,
        dst_k,
        src_v,
        dst_v,
        src_indices,
        dst_indices,
        item_size,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_per_layer_pf_lf(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    item_size: int,
    src_layout_dim: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_per_layer_pf_lf.default(
        src_k,
        dst_k,
        src_v,
        dst_v,
        src_indices,
        dst_indices,
        layer_id,
        item_size,
        src_layout_dim,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_per_layer_ph_lf(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    item_size: int,
    src_layout_dim: int,
    page_size: int,
    head_num: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_per_layer_ph_lf.default(
        src_k,
        dst_k,
        src_v,
        dst_v,
        src_indices,
        dst_indices,
        layer_id,
        item_size,
        src_layout_dim,
        page_size,
        head_num,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_all_layer(
    src_k_layers: torch.Tensor,
    dst_k_layers: torch.Tensor,
    src_v_layers: torch.Tensor,
    dst_v_layers: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    num_layers: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_all_layer.default(
        src_k_layers,
        dst_k_layers,
        src_v_layers,
        dst_v_layers,
        src_indices,
        dst_indices,
        item_size,
        num_layers,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_all_layer_lf_pf(
    src_k_layers: torch.Tensor,
    dst_k: torch.Tensor,
    src_v_layers: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    dst_layout_dim: int,
    num_layers: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_all_layer_lf_pf.default(
        src_k_layers,
        dst_k,
        src_v_layers,
        dst_v,
        src_indices,
        dst_indices,
        item_size,
        dst_layout_dim,
        num_layers,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_all_layer_lf_ph(
    src_k_layers: torch.Tensor,
    dst_k: torch.Tensor,
    src_v_layers: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    dst_layout_dim: int,
    num_layers: int,
    page_size: int,
    head_num: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_all_layer_lf_ph.default(
        src_k_layers,
        dst_k,
        src_v_layers,
        dst_v,
        src_indices,
        dst_indices,
        item_size,
        dst_layout_dim,
        num_layers,
        page_size,
        head_num,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_direct(
    src_layers: List[torch.Tensor],
    dst_layers: List[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    page_size: int,
):
    torch.ops.sgl_kernel.transfer_kv_direct.default(
        src_layers, dst_layers, src_indices, dst_indices, page_size
    )


def transfer_kv_per_layer_direct_pf_lf(
    src_ptrs: List[torch.Tensor],
    dst_ptrs: List[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    page_size: int,
):
    torch.ops.sgl_kernel.transfer_kv_per_layer_direct_pf_lf.default(
        src_ptrs, dst_ptrs, src_indices, dst_indices, layer_id, page_size
    )


def transfer_kv_all_layer_direct_lf_pf(
    src_ptrs: List[torch.Tensor],
    dst_ptrs: List[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    page_size: int,
):
    torch.ops.sgl_kernel.transfer_kv_all_layer_direct_lf_pf.default(
        src_ptrs, dst_ptrs, src_indices, dst_indices, page_size
    )


def transfer_kv_per_layer_mla(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_per_layer_mla.default(
        src,
        dst,
        src_indices,
        dst_indices,
        item_size,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_per_layer_mla_pf_lf(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    item_size: int,
    src_layout_dim: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_per_layer_mla_pf_lf.default(
        src,
        dst,
        src_indices,
        dst_indices,
        layer_id,
        item_size,
        src_layout_dim,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_all_layer_mla(
    src_layers: torch.Tensor,
    dst_layers: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    num_layers: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_all_layer_mla.default(
        src_layers,
        dst_layers,
        src_indices,
        dst_indices,
        item_size,
        num_layers,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_all_layer_mla_lf_pf(
    src_layers: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    dst_layout_dim: int,
    num_layers: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_all_layer_mla_lf_pf.default(
        src_layers,
        dst,
        src_indices,
        dst_indices,
        item_size,
        dst_layout_dim,
        num_layers,
        block_quota,
        num_warps_per_block,
    )
```

**/mnt/ssd_data/ssd_projects/warp/sglang/python/sglang/srt/mem_cache/memory_pool_host.py**
```
from __future__ import annotations

import abc
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from sglang.srt.mem_cache.hicache_storage import PoolName

import numpy as np
import psutil
import torch

from sglang.jit_kernel.hicache import (
    can_use_hicache_jit_kernel,
)
from sglang.jit_kernel.hicache import (
    transfer_hicache_all_layer as jit_transfer_hicache_all_layer,
)
from sglang.jit_kernel.hicache import (
    transfer_hicache_all_layer_mla as jit_transfer_hicache_all_layer_mla,
)
from sglang.jit_kernel.hicache import (
    transfer_hicache_one_layer as jit_transfer_hicache_one_layer,
)
from sglang.jit_kernel.hicache import (
    transfer_hicache_one_layer_mla as jit_transfer_hicache_one_layer_mla,
)
from sglang.srt.mem_cache.memory_pool import (
    KVCache,
    MambaPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
    NSATokenToKVPool,
)
from sglang.srt.utils import is_cuda, is_mps, is_npu, is_xpu

_is_cuda = is_cuda()
_is_npu = is_npu()
_is_xpu = is_xpu()
_is_mps = is_mps()
if not (_is_npu or _is_xpu or _is_mps):
    from sgl_kernel.kvcacheio import (
        transfer_kv_all_layer,
        transfer_kv_all_layer_direct_lf_pf,
        transfer_kv_all_layer_lf_pf,
        transfer_kv_all_layer_lf_ph,
        transfer_kv_all_layer_mla,
        transfer_kv_all_layer_mla_lf_pf,
        transfer_kv_direct,
        transfer_kv_per_layer,
        transfer_kv_per_layer_direct_pf_lf,
        transfer_kv_per_layer_mla,
        transfer_kv_per_layer_mla_pf_lf,
        transfer_kv_per_layer_pf_lf,
        transfer_kv_per_layer_ph_lf,
    )
if _is_npu:
    from sgl_kernel_npu.kvcacheio import TransferDirection, transfer_kv_dim_exchange

logger = logging.getLogger(__name__)


def synchronized(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return func(self, *args, **kwargs)

    return wrapper


class HostTensorAllocator(abc.ABC):
    def __init__(self):
        """Initialize the HostTensorAllocator."""
        self.dtype = None
        self.dims = None

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        """Allocate a tensor of given dims and dtype on the memory."""
        self.dtype = dtype
        self.dims = dims
        tensor = torch.empty(dims, dtype=dtype, device=device)
        return tensor


def get_allocator_from_storage(allocator_type):
    if allocator_type == "mooncake":
        try:
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeHostTensorAllocator,
            )

            return MooncakeHostTensorAllocator()
        except ImportError:
            logger.warning(
                "Mooncake's tensor allocator requires mooncake >= 0.3.8.post1. "
                "Please upgrade Mooncake by 'pip install mooncake-transfer-engine --upgrade'. "
                "Fallback to use default allocator."
            )
            return HostTensorAllocator()
    else:
        return HostTensorAllocator()


def alloc_with_host_register(
    dims,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: HostTensorAllocator,
) -> torch.Tensor:
    """
    Allocate tensor and register host memory with cudaHostRegister.
    CudaHostRegister only applies when pin_memory=True.
    """
    buffer = allocator.allocate(dims, dtype=dtype, device=device)
    if pin_memory:
        torch.cuda.cudart().cudaHostRegister(
            buffer.data_ptr(), buffer.numel() * buffer.element_size(), 0
        )
    return buffer


def alloc_with_pin_memory(
    dims,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: None,
) -> torch.Tensor:
    """
    Allocate tensor using PyTorch's built-in pin_memory flag.
    """
    buffer = torch.empty(dims, dtype=dtype, device=device, pin_memory=pin_memory)
    return buffer


ALLOC_MEMORY_FUNCS = defaultdict(
    lambda: alloc_with_host_register,
    {
        "npu": alloc_with_pin_memory,
    },
)


class HostKVCache(abc.ABC):

    def __init__(
        self,
        device_pool: KVCache,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool,
        device: str,
        allocator_type: str = "default",
    ):
        self.device_pool = device_pool
        self.page_size = page_size
        self.layout = layout
        self.pin_memory = pin_memory
        self.device = device
        self.allocator = get_allocator_from_storage(allocator_type)

        self.dtype = device_pool.store_dtype
        self.size_per_token = self.get_size_per_token()
        if host_size > 0:
            self.size = int(host_size * 1e9 // self.size_per_token)
        else:
            self.size = int(device_pool.size * host_to_device_ratio)
        # Align up the host memory pool size to the page size
        self.page_num = self.size // self.page_size + 1
        self.size = self.page_num * self.page_size
        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer

        assert (
            self.size > device_pool.size
        ), "The host memory should be larger than the device memory with the current protocol"

        # Verify there is enough available host memory.
        host_mem = psutil.virtual_memory()
        requested_bytes = self.size * self.size_per_token
        # preserve at least 10GB for other usage
        ten_gb = 10 * (1024**3)
        available_bytes = host_mem.available - ten_gb
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory available. Requesting "
                f"{requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free. Please reduce the "
                f"size of the hierarchical cache."
            )
```

**/mnt/ssd_data/ssd_projects/warp/sglang/python/sglang/srt/mem_cache/memory_pool_host.py**
```
            dims = (2, self.layer_num, self.size, self.head_num, self.head_dim)
        elif self.layout == "page_first":
            dims = (2, self.size, self.layer_num, self.head_num, self.head_dim)
        elif self.layout == "page_first_direct":
            dims = (
                2,
                self.page_num,
                self.layer_num,
                self.page_size,
                self.head_num,
                self.head_dim,
            )
        elif self.layout == "page_head":
            dims = (
                2,
                self.page_num,
                self.head_num,
                self.page_size,
                self.layer_num,
                self.head_dim,
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        self.token_stride_size = self.head_num * self.head_dim * self.dtype.itemsize
        self.layout_dim = self.token_stride_size * self.layer_num

        alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
        buffer = alloc_func(
            dims,
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
        )
        return buffer

    @property
    def k_buffer(self):
        return self.kv_buffer[0]

    @property
    def v_buffer(self):
        return self.kv_buffer[1]

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
    ):
        if io_backend == "kernel":
            if self.layout == "layer_first":
                if self.can_use_jit:
                    jit_transfer_hicache_one_layer(
                        k_cache_dst=device_pool.k_buffer[layer_id],
                        v_cache_dst=device_pool.v_buffer[layer_id],
                        k_cache_src=self.k_buffer[layer_id],
                        v_cache_src=self.v_buffer[layer_id],
                        indices_dst=device_indices,
                        indices_src=host_indices,
                        element_dim=self.element_dim,
                    )
                else:
                    transfer_kv_per_layer(
                        src_k=self.k_buffer[layer_id],
                        dst_k=device_pool.k_buffer[layer_id],
                        src_v=self.v_buffer[layer_id],
                        dst_v=device_pool.v_buffer[layer_id],
                        src_indices=host_indices,
                        dst_indices=device_indices,
                        item_size=self.token_stride_size,
                    )
            elif self.layout == "page_first":
                if self.can_use_jit:
                    # Transpose [page, layer, ...] -> [layer, page, ...] then
                    # index by layer_id to get a per-layer view with strided layout.
                    # The kernel handles different src/dst strides automatically.
                    jit_transfer_hicache_one_layer(
                        k_cache_dst=device_pool.k_buffer[layer_id],
                        v_cache_dst=device_pool.v_buffer[layer_id],
                        k_cache_src=self.k_data_refs[layer_id],
                        v_cache_src=self.v_data_refs[layer_id],
                        indices_dst=device_indices,
                        indices_src=host_indices,
                        element_dim=self.element_dim,
                    )
                else:
                    transfer_kv_per_layer_pf_lf(
                        src_k=self.k_buffer,
                        dst_k=device_pool.k_buffer[layer_id],
                        src_v=self.v_buffer,
                        dst_v=device_pool.v_buffer[layer_id],
                        src_indices=host_indices,
                        dst_indices=device_indices,
                        layer_id=layer_id,
                        item_size=self.token_stride_size,
                        src_layout_dim=self.layout_dim,
                    )
            elif self.layout == "page_head":
                transfer_kv_per_layer_ph_lf(
                    src_k=self.k_buffer,
                    dst_k=device_pool.k_buffer[layer_id],
                    src_v=self.v_buffer,
                    dst_v=device_pool.v_buffer[layer_id],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=layer_id,
                    item_size=self.token_stride_size,
                    src_layout_dim=self.layout_dim,
                    page_size=self.page_size,
                    head_num=self.head_num,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=[self.k_buffer[layer_id], self.v_buffer[layer_id]],
                    dst_layers=[
                        device_pool.k_buffer[layer_id],
                        device_pool.v_buffer[layer_id],
                    ],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    page_size=self.page_size,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_per_layer_direct_pf_lf(
                    src_ptrs=[self.k_buffer, self.v_buffer],
                    dst_ptrs=[
                        device_pool.k_buffer[layer_id],
                        device_pool.v_buffer[layer_id],
                    ],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=layer_id,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "kernel_ascend":
            if self.layout == "page_first_direct":
                # Ascend-specific: transfer KV data for all layers when layer_id == 0
                if layer_id == 0:
                    transfer_kv_dim_exchange(
                        device_indices=device_indices,
                        host_indices=host_indices,
                        device_k=device_pool.k_buffer,
                        host_k=self.k_buffer,
                        device_v=device_pool.v_buffer,
                        host_v=self.v_buffer,
                        page_size=self.page_size,
                        direction=TransferDirection.H2D,
                    )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if io_backend == "kernel":
            if self.layout == "layer_first":
                if self.can_use_jit:
                    jit_transfer_hicache_all_layer(
                        k_ptr_dst=self.k_data_ptrs,
                        v_ptr_dst=self.v_data_ptrs,
                        indices_dst=host_indices,
                        k_ptr_src=device_pool.k_data_ptrs,
                        v_ptr_src=device_pool.v_data_ptrs,
                        indices_src=device_indices,
                        kv_cache_dst_stride_bytes=self.token_stride_size,
                        kv_cache_src_stride_bytes=self.token_stride_size,
                        element_size=self.element_dim * self.dtype.itemsize,
                    )
                else:
                    transfer_kv_all_layer(
                        src_k_layers=device_pool.k_data_ptrs,
                        dst_k_layers=self.k_data_ptrs,
                        src_v_layers=device_pool.v_data_ptrs,
                        dst_v_layers=self.v_data_ptrs,
                        src_indices=device_indices,
                        dst_indices=host_indices,
                        item_size=self.token_stride_size,
                        num_layers=self.layer_num,
                    )
            elif self.layout == "page_first":
                if self.can_use_jit:
                    # Use transposed data ptrs so the kernel writes to
                    # [layer, page, item] view with stride layout_dim per token.
                    jit_transfer_hicache_all_layer(
                        k_ptr_dst=self.k_data_ptrs,
                        v_ptr_dst=self.v_data_ptrs,
                        indices_dst=host_indices,
                        k_ptr_src=device_pool.k_data_ptrs,
                        v_ptr_src=device_pool.v_data_ptrs,
```

**/mnt/ssd_data/ssd_projects/warp/sglang/python/sglang/srt/mem_cache/memory_pool_host.py**
```
            dims = (
                self.layer_num,
                self.size,
                1,
                self.kv_cache_dim,
            )
        elif self.layout == "page_first":
            dims = (
                self.size,
                self.layer_num,
                1,
                self.kv_cache_dim,
            )
        elif self.layout == "page_first_direct":
            dims = (
                self.page_num,
                self.layer_num,
                self.page_size,
                1,
                self.kv_cache_dim,
            )
        # Ascend-specific: Aligns with NPUMLATokenToKVPool layout
        # Separately allocate k_buffer and v_buffer for easier data transfer.
        elif self.layout == "page_first_kv_split":
            base_dims = (
                self.page_num,
                self.layer_num,
                self.page_size,
                1,
            )
            alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
            self.k_buffer = alloc_func(
                (*base_dims, self.kv_lora_rank),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
            self.v_buffer = alloc_func(
                (*base_dims, self.qk_rope_head_dim),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
            self.index_k_buffer = None
            if self.device_pool.index_head_dim is not None:
                self.index_k_buffer = alloc_func(
                    (*base_dims, self.device_pool.index_head_dim),
                    dtype=self.dtype,
                    device=self.device,
                    pin_memory=self.pin_memory,
                    allocator=self.allocator,
                )
            # Return k_buffer to preserve original kv_buffer and data_refs init logic,
            # though Ascend doesn't use these parameters.
            return self.k_buffer
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        self.token_stride_size = self.kv_cache_dim * self.dtype.itemsize
        self.layout_dim = self.token_stride_size * self.layer_num

        alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
        buffer = alloc_func(
            dims,
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
        )
        return buffer

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        if io_backend == "kernel":
            if self.layout == "layer_first":
                if self.can_use_jit:
                    jit_transfer_hicache_one_layer_mla(
                        cache_dst=device_pool.kv_buffer[layer_id],
                        cache_src=self.kv_buffer[layer_id],
                        indices_dst=device_indices,
                        indices_src=host_indices,
                        element_dim=self.kv_cache_dim,
                    )
                else:
                    transfer_kv_per_layer_mla(
                        src=self.kv_buffer[layer_id],
                        dst=device_pool.kv_buffer[layer_id],
                        src_indices=host_indices,
                        dst_indices=device_indices,
                        item_size=self.token_stride_size,
                    )
            elif self.layout == "page_first":
                if self.can_use_jit:
                    jit_transfer_hicache_one_layer_mla(
                        cache_dst=device_pool.kv_buffer[layer_id],
                        cache_src=self.data_refs[layer_id],
                        indices_dst=device_indices,
                        indices_src=host_indices,
                        element_dim=self.kv_cache_dim,
                    )
                else:
                    transfer_kv_per_layer_mla_pf_lf(
                        src=self.kv_buffer,
                        dst=device_pool.kv_buffer[layer_id],
                        src_indices=host_indices,
                        dst_indices=device_indices,
                        layer_id=layer_id,
                        item_size=self.token_stride_size,
                        src_layout_dim=self.layout_dim,
                    )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=[self.kv_buffer[layer_id]],
                    dst_layers=[device_pool.kv_buffer[layer_id]],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    page_size=self.page_size,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_per_layer_direct_pf_lf(
                    src_ptrs=[self.kv_buffer],
                    dst_ptrs=[device_pool.kv_buffer[layer_id]],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=layer_id,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "kernel_ascend":
            if self.layout == "page_first_kv_split":
                # Ascend-specific: transfer KV data for all layers when layer_id == 0
                if layer_id == 0:
                    transfer_kv_dim_exchange(
                        device_indices=device_indices,
                        host_indices=host_indices,
                        device_k=device_pool.k_buffer,
                        host_k=self.k_buffer,
                        device_v=device_pool.v_buffer,
                        host_v=self.v_buffer,
                        device_index_k=device_pool.index_k_buffer,
                        host_index_k=self.index_k_buffer,
                        page_size=self.page_size,
                        direction=TransferDirection.H2D,
                    )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if io_backend == "kernel":
            if self.layout == "layer_first":
                if self.can_use_jit:
                    jit_transfer_hicache_all_layer_mla(
                        ptr_dst=self.data_ptrs,
                        indices_dst=host_indices,
                        ptr_src=device_pool.data_ptrs,
                        indices_src=device_indices,
                        cache_dst_stride_bytes=self.token_stride_size,
                        cache_src_stride_bytes=self.token_stride_size,
                        element_size=self.kv_cache_dim * self.dtype.itemsize,
                    )
                else:
                    transfer_kv_all_layer_mla(
                        src_layers=device_pool.data_ptrs,
                        dst_layers=self.data_ptrs,
                        src_indices=device_indices,
                        dst_indices=host_indices,
                        item_size=self.token_stride_size,
                        num_layers=self.layer_num,
                    )
            elif self.layout == "page_first":
                if self.can_use_jit:
                    jit_transfer_hicache_all_layer_mla(
                        ptr_dst=self.data_ptrs,
                        indices_dst=host_indices,
                        ptr_src=device_pool.data_ptrs,
                        indices_src=device_indices,
                        cache_src_stride_bytes=self.token_stride_size,
                        cache_dst_stride_bytes=self.layout_dim,
                        element_size=self.kv_cache_dim * self.dtype.itemsize,
                    )
                else:
                    transfer_kv_all_layer_mla_lf_pf(
                        src_layers=device_pool.data_ptrs,
                        dst=self.kv_buffer,
                        src_indices=device_indices,
                        dst_indices=host_indices,
                        item_size=self.token_stride_size,
                        dst_layout_dim=self.layout_dim,
                        num_layers=self.layer_num,
```

**/mnt/ssd_data/ssd_projects/warp/sglang/python/sglang/srt/mem_cache/memory_pool_host.py**
```
        return tensor.contiguous().view(torch.uint8).reshape(-1)

    @synchronized
    def clear(self):
        self.mem_state = torch.zeros(
            (self.size,), dtype=torch.uint8, device=self.device
        )
        self.free_slots = torch.arange(self.size, dtype=torch.int64)

    def available_size(self):
        return len(self.free_slots)

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        assert (
            need_size % self.page_size == 0
        ), "The requested size should be a multiple of the page size."
        if need_size > self.available_size():
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        self.free_slots = torch.cat([self.free_slots, indices])
        return len(indices)

    def get_size_per_token(self):
        conv_total_size = 0
        for conv_shape in self.conv_state_shapes:
            conv_total_size += int(np.prod(conv_shape)) * self.conv_dtype.itemsize
        temporal_size = (
            int(np.prod(self.temporal_state_shape)) * self.temporal_dtype.itemsize
        )
        return (conv_total_size + temporal_size) * self.num_mamba_layers

    def get_ksize_per_token(self):
        return self.get_size_per_token()

    @staticmethod
    def _item_size_per_index(tensor: torch.Tensor) -> int:
        if tensor.shape[0] == 0:
            return 0
        return int(tensor[0].numel() * tensor.element_size())

    @staticmethod
    def _copy_tensor(
        src: torch.Tensor,
        dst: torch.Tensor,
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
        io_backend: str,
    ) -> None:
        if src_indices.numel() == 0:
            return
        if io_backend == "kernel":
            # TODO: Rename the interface for clarity.
            # Here, transfer_kv_per_layer_mla is reused to transfer the Mamba state.
            # This has nothing to do with MLA; it's only reused because this interface happens to transfer a single Pool.
            transfer_kv_per_layer_mla(
                src=src,
                dst=dst,
                src_indices=src_indices,
                dst_indices=dst_indices,
                item_size=MambaPoolHost._item_size_per_index(src),
            )
        elif io_backend == "direct":
            transfer_kv_direct(
                src_layers=[src],
                dst_layers=[dst],
                src_indices=src_indices,
                dst_indices=dst_indices,
                page_size=1,
            )
        else:
            raise ValueError(f"Unsupported io_backend: {io_backend}")

    @staticmethod
    def _copy_tensor_pf_lf(
        src: torch.Tensor,
        dst: torch.Tensor,
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
        layer_id: int,
        num_layers: int,
        io_backend: str,
    ) -> None:
        if src_indices.numel() == 0:
            return
        if io_backend == "kernel":
            item_size = MambaPoolHost._item_size_per_index(dst)
            transfer_kv_per_layer_mla_pf_lf(
                src=src,
                dst=dst,
                src_indices=src_indices,
                dst_indices=dst_indices,
                layer_id=layer_id,
                item_size=item_size,
                src_layout_dim=item_size * num_layers,
            )
        elif io_backend == "direct":
            transfer_kv_per_layer_direct_pf_lf(
                src_ptrs=[src],
                dst_ptrs=[dst],
                src_indices=src_indices,
                dst_indices=dst_indices,
                layer_id=layer_id,
                page_size=1,
            )
        else:
            raise ValueError(f"Unsupported io_backend: {io_backend}")

    @staticmethod
    def _copy_tensor_all_layers_lf_pf(
        src_layers: torch.Tensor,
        dst: torch.Tensor,
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
        num_layers: int,
        device: str,
        io_backend: str,
    ) -> None:
        if src_indices.numel() == 0:
            return
        if io_backend == "kernel":
            item_size = MambaPoolHost._item_size_per_index(src_layers[0])
            src_ptrs = torch.tensor(
                [src_layers[i].data_ptr() for i in range(num_layers)],
                dtype=torch.uint64,
                device=device,
            )
            transfer_kv_all_layer_mla_lf_pf(
                src_layers=src_ptrs,
                dst=dst,
                src_indices=src_indices,
                dst_indices=dst_indices,
                item_size=item_size,
                dst_layout_dim=item_size * num_layers,
                num_layers=num_layers,
            )
        elif io_backend == "direct":
            src_ptrs = [src_layers[i] for i in range(num_layers)]
            transfer_kv_all_layer_direct_lf_pf(
                src_ptrs=src_ptrs,
                dst_ptrs=[dst],
                src_indices=src_indices,
                dst_indices=dst_indices,
                page_size=1,
            )
        else:
            raise ValueError(f"Unsupported io_backend: {io_backend}")

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend="kernel",
    ):
        if self.layout in ["page_first", "page_first_direct"]:
            self._copy_tensor_pf_lf(
                src=self.temporal_buffer,
                dst=device_pool.mamba_cache.temporal[layer_id],
                src_indices=host_indices,
                dst_indices=device_indices,
                layer_id=layer_id,
                num_layers=self.num_mamba_layers,
                io_backend=io_backend,
            )
            for conv_idx in range(len(self.conv_state_shapes)):
                self._copy_tensor_pf_lf(
                    src=self.conv_buffer[conv_idx],
                    dst=device_pool.mamba_cache.conv[conv_idx][layer_id],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=layer_id,
                    num_layers=self.num_mamba_layers,
                    io_backend=io_backend,
                )
        else:
            self._copy_tensor(
                self.temporal_buffer[layer_id],
                device_pool.mamba_cache.temporal[layer_id],
                host_indices,
                device_indices,
                io_backend,
            )
            for conv_idx in range(len(self.conv_state_shapes)):
                self._copy_tensor(
                    self.conv_buffer[conv_idx][layer_id],
                    device_pool.mamba_cache.conv[conv_idx][layer_id],
                    host_indices,
                    device_indices,
                    io_backend,
                )

    def backup_from_device_all_layer(
```

**/mnt/ssd_data/ssd_projects/warp/sglang/python/sglang/srt/mem_cache/memory_pool_host.py**
```
        )
        self.indexer_layout_dim = self.indexer_page_stride_size * self.layer_num
        self.indexer_page_num = (self.size + self.page_size + 1) // self.page_size
        self._init_indexer_buffers()
        logger.info(
            f"NSATokenToKVPoolHost initialized with indexer page stride size: {self.indexer_page_stride_size}, page num: {self.indexer_page_num}"
        )

    def get_size_per_token(self):
        base = super().get_size_per_token()
        return (
            base
            + self.indexer_size_per_token * self.layer_num * self.indexer_dtype.itemsize
        )

    def _init_indexer_buffers(self):
        alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
        self.index_k_device_ptrs = torch.tensor(
            [x.data_ptr() for x in self.device_pool.index_k_with_scale_buffer],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        if self.layout == "layer_first":
            self.index_k_with_scale_buffer = [
                alloc_func(
                    (self.indexer_page_num, self.indexer_page_stride_size),
                    dtype=self.indexer_dtype,
                    device=self.device,
                    pin_memory=self.pin_memory,
                    allocator=self.allocator,
                )
                for _ in range(self.layer_num)
            ]
            self.index_k_data_refs = [
                self.index_k_with_scale_buffer[i] for i in range(self.layer_num)
            ]
            self.index_k_data_ptrs = torch.tensor(
                [x.data_ptr() for x in self.index_k_data_refs],
                dtype=torch.uint64,
                device=self.device_pool.device,
            )
        elif self.layout in ["page_first", "page_first_direct"]:
            self.index_k_with_scale_buffer = alloc_func(
                (
                    self.indexer_page_num,
                    self.layer_num,
                    1,
                    self.indexer_page_stride_size,
                ),
                dtype=self.indexer_dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def _get_indexer_page_indices(self, host_indices, device_indices):
        if host_indices.numel() == 0:
            return host_indices, device_indices
        if host_indices.numel() % self.page_size != 0:
            raise ValueError(
                "Index buffer transfer expects page-aligned indices for NSA."
            )
        host_page_indices = (
            host_indices.reshape(-1, self.page_size)[:, 0] // self.page_size
        )
        device_page_indices = (
            device_indices.reshape(-1, self.page_size)[:, 0] // self.page_size
        )
        return host_page_indices, device_page_indices

    def _load_indexer_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        host_page_indices, device_page_indices = self._get_indexer_page_indices(
            host_indices, device_indices
        )
        use_kernel = io_backend == "kernel" and self.indexer_page_stride_size % 8 == 0
        if use_kernel:
            if self.layout == "layer_first":
                transfer_kv_per_layer_mla(
                    src=self.index_k_with_scale_buffer[layer_id],
                    dst=device_pool.index_k_with_scale_buffer[layer_id],
                    src_indices=host_page_indices,
                    dst_indices=device_page_indices,
                    item_size=self.indexer_page_stride_size,
                )
            elif self.layout == "page_first":
                transfer_kv_per_layer_mla_pf_lf(
                    src=self.index_k_with_scale_buffer,
                    dst=device_pool.index_k_with_scale_buffer[layer_id],
                    src_indices=host_page_indices,
                    dst_indices=device_page_indices,
                    layer_id=layer_id,
                    item_size=self.indexer_page_stride_size,
                    src_layout_dim=self.indexer_layout_dim,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=[self.index_k_with_scale_buffer[layer_id]],
                    dst_layers=[device_pool.index_k_with_scale_buffer[layer_id]],
                    src_indices=host_page_indices,
                    dst_indices=device_page_indices,
                    page_size=1,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_per_layer_direct_pf_lf(
                    src_ptrs=[self.index_k_with_scale_buffer],
                    dst_ptrs=[device_pool.index_k_with_scale_buffer[layer_id]],
                    src_indices=host_page_indices,
                    dst_indices=device_page_indices,
                    layer_id=layer_id,
                    page_size=1,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def _backup_indexer_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        host_page_indices, device_page_indices = self._get_indexer_page_indices(
            host_indices, device_indices
        )
        use_kernel = io_backend == "kernel" and self.indexer_page_stride_size % 8 == 0
        if use_kernel:
            if self.layout == "layer_first":
                transfer_kv_all_layer_mla(
                    src_layers=self.index_k_device_ptrs,
                    dst_layers=self.index_k_data_ptrs,
                    src_indices=device_page_indices,
                    dst_indices=host_page_indices,
                    item_size=self.indexer_page_stride_size,
                    num_layers=self.layer_num,
                )
            elif self.layout == "page_first":
                transfer_kv_all_layer_mla_lf_pf(
                    src_layers=self.index_k_device_ptrs,
                    dst=self.index_k_with_scale_buffer,
                    src_indices=device_page_indices,
                    dst_indices=host_page_indices,
                    item_size=self.indexer_page_stride_size,
                    dst_layout_dim=self.indexer_layout_dim,
                    num_layers=self.layer_num,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=device_pool.index_k_with_scale_buffer,
                    dst_layers=self.index_k_with_scale_buffer,
                    src_indices=device_page_indices,
                    dst_indices=host_page_indices,
                    page_size=1,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_all_layer_direct_lf_pf(
                    src_ptrs=device_pool.index_k_with_scale_buffer,
                    dst_ptrs=[self.index_k_with_scale_buffer],
                    src_indices=device_page_indices,
                    dst_indices=host_page_indices,
                    page_size=1,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
    ):
        super().load_to_device_per_layer(
            device_pool, host_indices, device_indices, layer_id, io_backend
        )
        self._load_indexer_to_device_per_layer(
            device_pool, host_indices, device_indices, layer_id, io_backend
        )

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        super().backup_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend
        )
        self._backup_indexer_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend
        )
```





**Files Read:**

**/mnt/ssd_data/ssd_projects/warp/sglang/python/sglang/srt/mem_cache/hiradix_cache.py**
```
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from sglang.srt.managers.cache_controller import HiCacheController, PrefetchOperation
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InitLoadBackParams,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    NSATokenToKVPool,
)
from sglang.srt.mem_cache.memory_pool_host import (
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
    NSATokenToKVPoolHost,
)
from sglang.srt.mem_cache.radix_cache import (
    RadixCache,
    RadixKey,
    TreeNode,
    compute_node_hash_values,
    split_node_hash_value,
)
from sglang.srt.mem_cache.utils import convert_to_bigram_key
from sglang.srt.observability.metrics_collector import StorageMetricsCollector

if TYPE_CHECKING:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class HiRadixCache(RadixCache):

    def __init__(self, params: CacheInitParams, server_args: ServerArgs):
        self._enable_metrics_flag = params.enable_metrics

        self.page_size = params.page_size
        self.kv_cache = params.token_to_kv_pool_allocator.get_kvcache()

        if isinstance(self.kv_cache, MHATokenToKVPool):
            self.token_to_kv_pool_host = MHATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            )
        elif isinstance(self.kv_cache, NSATokenToKVPool):
            self.token_to_kv_pool_host = NSATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            )
        elif isinstance(self.kv_cache, MLATokenToKVPool):
            self.token_to_kv_pool_host = MLATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            )
        else:
            raise ValueError(f"HiRadixCache only supports MHA and MLA yet")

        self.tp_group = params.tp_cache_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)
        self.pp_rank = params.pp_rank
        self.pp_size = params.pp_size
        self.attn_cp_rank = params.attn_cp_rank
        self.attn_cp_size = params.attn_cp_size
        self.enable_storage = server_args.hicache_storage_backend is not None
        self.enable_storage_metrics = self.enable_storage and params.enable_metrics
        self.extra_metric_labels = server_args.extra_metric_labels

        (
            extra_config,
            prefetch_threshold,
            prefetch_timeout_base,
            prefetch_timeout_per_ki_token,
            hicache_storage_pass_prefix_keys,
        ) = self._parse_storage_backend_extra_config(
            server_args.hicache_storage_backend_extra_config
        )
        # TODO: support more timeout check functions
        self.is_prefetch_timeout = self._prefetch_timeout_check_linear_func
        self.prefetch_stop_policy = server_args.hicache_storage_prefetch_policy

        self.load_cache_event = threading.Event()
        self.cache_controller = HiCacheController(
            params.token_to_kv_pool_allocator,
            self.token_to_kv_pool_host,
            self.page_size,
            self.tp_group,
            load_cache_event=self.load_cache_event,
            write_policy=server_args.hicache_write_policy,
            io_backend=server_args.hicache_io_backend,
            storage_backend=server_args.hicache_storage_backend,
            prefetch_threshold=prefetch_threshold,
            model_name=server_args.served_model_name,
            storage_backend_extra_config=extra_config,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            attn_cp_rank=self.attn_cp_rank,
            attn_cp_size=self.attn_cp_size,
            enable_storage_metrics=self.enable_storage_metrics,
        )
        self._apply_storage_runtime_config(
            storage_backend=server_args.hicache_storage_backend,
            prefetch_threshold=prefetch_threshold,
            prefetch_timeout_base=prefetch_timeout_base,
            prefetch_timeout_per_ki_token=prefetch_timeout_per_ki_token,
            hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
            enable_storage=self.enable_storage,
            enable_storage_metrics=self.enable_storage_metrics,
            extra_metric_labels=self.extra_metric_labels,
        )

        # record the nodes with ongoing write through
        self.ongoing_write_through = {}
        # record the node segments with ongoing load back
        self.ongoing_load_back = {}
        # record the ongoing prefetch requests
        self.ongoing_prefetch = {}
        self.ongoing_backup = {}
        # track per-request tokens loaded from storage (L3 hits)
        # key: request_id, value: number of tokens actually loaded from storage
        self.prefetch_loaded_tokens_by_reqid: dict[str, int] = {}
        # todo: dynamically adjust the threshold
        self.write_through_threshold = (
            1 if server_args.hicache_write_policy == "write_through" else 2
        )
        self.load_back_threshold = 10

        # Detach storage backend automatically on process shutdown
        atexit.register(self.shutdown)

        self.evictable_host_leaves = set()

        super().__init__(params=params)

    def shutdown(self):
        """Best-effort auto-detach of storage backend on process shutdown.

        This keeps startup and runtime behavior consistent: if a backend was attached
        (either via CLI args or via admin API), we attempt to detach it on exit.
        """
        try:
            if self.enable_storage:
                self.detach_storage_backend()
        except Exception:
            logger.exception("Failed to detach storage backend on process shutdown.")

    def _apply_storage_runtime_config(
        self,
        *,
        storage_backend: Optional[str],
        prefetch_threshold: int,
        prefetch_timeout_base: float,
        prefetch_timeout_per_ki_token: float,
        hicache_storage_pass_prefix_keys: bool,
        enable_storage: bool,
        enable_storage_metrics: bool,
        extra_metric_labels: Optional[Dict[str, str]],
    ) -> None:
        prefetch_timeout_per_page = (
            self.page_size / 1024 * prefetch_timeout_per_ki_token
        )

        self.enable_storage = enable_storage
        self.prefetch_threshold = prefetch_threshold
        self.prefetch_timeout_base = prefetch_timeout_base
        self.prefetch_timeout_per_page = prefetch_timeout_per_page
        self.hicache_storage_pass_prefix_keys = hicache_storage_pass_prefix_keys
        self.enable_storage_metrics = enable_storage_metrics

        if self.enable_storage_metrics:
            labels = {
                "storage_backend": storage_backend,
                "tp_rank": self.cache_controller.tp_rank,
                "dp_rank": self.cache_controller.dp_rank,
                "pp_rank": self.cache_controller.pp_rank,
```



Now I have enough context. Based on the user's request, they want to know:

1. Which parts of the inference process are affected when enabling `--hicache-io-backend kernel` (i.e., the GPU-assisted I/O kernels)
2. How to directly test the kernel-level speed change

## 开启 `--hicache-io-backend kernel` 后受影响的部分

当 `enable_hierarchical_cache=True` 且 `hicache_io_backend="kernel"` 时，以下路径会改走 **GPU-assisted I/O kernels**（自定义 CUDA kernel），而不是 `cudaMemcpyAsync` 或 `torch index copy`。

### 1. L1 ↔ L2 的两条核心数据路径
从 `python/sglang/srt/mem_cache/memory_pool_host.py` 看，kernel backend 只影响两件事：

- **`load_to_device_per_layer`**（**L2 → L1，per-layer**）
  - 在 prefill 开始时，上一轮匹配到的 host-side KV 要搬到 GPU。这一步是在 `attention layer forward` 之前 **逐层触发** 的，用来做 **compute-transfer overlap**：layer N+1 的拷贝与 layer N 的计算重叠（`hicache_design.md` 中的 "Compute-Transfer Overlap"）。
  - `kernel` 路径：`transfer_kv_per_layer` / `transfer_kv_per_layer_pf_lf` / `transfer_kv_per_layer_ph_lf`（MHA）、`transfer_kv_per_layer_mla*`（MLA）、NSA 还有一个 indexer 的版本。
  - `direct` 路径则是 `transfer_kv_direct` / `transfer_kv_per_layer_direct_pf_lf`，本质是 `cudaMemcpyAsync` 包装。
- **`backup_from_device_all_layer`**（**L1 → L2，all-layer**）
  - prefill 算完后，把新生成的 KV 写回 host pool（受 write-policy 控制）。
  - `kernel` 路径：`transfer_kv_all_layer` / `transfer_kv_all_layer_lf_pf` / `transfer_kv_all_layer_lf_ph`、`transfer_kv_all_layer_mla*`。
  - `direct` 路径：`transfer_kv_direct` / `transfer_kv_all_layer_direct_lf_pf`。

### 2. 推理阶段的具体影响点
- **Prefill TTFT**：`load_to_device_per_layer` 被 cache controller 在每层 attention 前调用；kernel 版本可以发小 message size、异步 launch，使 H2D 和 compute 真正重叠。`direct`(`cudaMemcpyAsync`) 在 small-page/少 token 场景下延迟显著更高，等同于每层做一次 PCIe stall。
- **L1 write-back 吞吐**（`write_through`/`write_through_selective`/`write_back`）：`backup_from_device_all_layer` 走 kernel 后单次聚合所有 layer 的 D2H 拷贝，比多次 per-layer `cudaMemcpyAsync` 更省 launch 开销。
- **Decode 阶段**：kernel backend **不影响** decode 本身；它只在 prefill 初始 load 与 write-back 时生效。所以 "开/关 kernel" 的差异只应在 **TTFT / prefill throughput / write-back 时长** 上体现。
- **Attention backend 的副作用**（`server_args.py:3059-3087` `_resolve_io_decode_attention_compatibility`）：选 `kernel` 时若 decode attention backend 是 FA3，会被**自动改成 `direct`**。所以测 kernel 时必须确保 `--attention-backend` 不是 `fa3`（常用 `triton` 或 `flashinfer`）。
- **Layout 的副作用**：`kernel` 不支持 `page_first_direct` 布局（会被降级成 `direct` IO）。所以想真正跑 kernel 路径，布局必须是 `layer_first`、`page_first`、或 `page_head`。

## 具体如何测试 kernel 速度变化

仓库里已经提供了 **三层测试手段**，从 micro-kernel 到 end-to-end，全部都可以直接对比 kernel vs direct。

### 方法 A：micro-benchmark（最直接，只测拷贝 kernel 本身）

**A1. `benchmark/hicache/perf.py`** — 对比 JIT kernel vs AOT `transfer_kv_per_layer` 的 H2D / D2H 吞吐（GB/s）：

```bash
python3 benchmark/hicache/perf.py
```
输出示例（每组 `cache_item_size × block_quota`）：
```
bs = 256 , H->D | aot 12.34 GB/s | jit 18.56 GB/s | +50.41%
bs = 256 , D->H | aot 10.11 GB/s | jit 13.70 GB/s | +35.51%
```
脚本自带 peak H↔D 带宽基线打印，便于看你离 PCIe 极限多远。

**A2. `python/sglang/jit_kernel/benchmark/bench_hicache.py`** — triton `do_bench` 风格，同时对比 **AOT kernel / JIT kernel / PyTorch indexing**，覆盖 `one-layer H2D` 与 `all-layer D2H`：

```bash
python3 python/sglang/jit_kernel/benchmark/bench_hicache.py
```
想只看 kernel 之间的差异、去掉 torch 基线：
```bash
DISABLE_TORCH=1 python3 python/sglang/jit_kernel/benchmark/bench_hicache.py
```
**A3. 正确性测试** — `sgl-kernel/tests/test_kvcacheio.py` 交叉校验 `kernel` 与 `direct` 输出一致（用在 PR 改 kernel 时）：
```bash
pytest sgl-kernel/tests/test_kvcacheio.py -v
```
### 方法 B：手写最小对比脚本（想只测 `kernel` vs `direct` 这一个 axis）

```python
# bench_hicache_io_backend.py
import torch
from sgl_kernel.kvcacheio import transfer_kv_per_layer, transfer_kv_direct

N_HOST, N_DEV, D, PAGE = 1<<18, 1<<17, 512, 64
dtype = torch.bfloat16

host_k = torch.empty((N_HOST, D), dtype=dtype, pin_memory=True)
host_v = torch.empty_like(host_k)
dev_k  = torch.empty((N_DEV, D), dtype=dtype, device="cuda")
dev_v  = torch.empty_like(dev_k)

def bench(fn, loops=100):
    torch.cuda.synchronize(); t0=torch.cuda.Event(True); t1=torch.cuda.Event(True)
    fn(); torch.cuda.synchronize()
    t0.record()
    for _ in range(loops): fn()
    t1.record(); t1.synchronize()
    return t0.elapsed_time(t1)/loops

for bs in [256, 1024, 4096, 16384]:
    si = torch.randperm(N_HOST, device="cuda")[:bs].sort().values
    di = torch.randperm(N_DEV , device="cuda")[:bs].sort().values
    item_bytes = D * dtype.itemsize

    kernel_ms = bench(lambda: transfer_kv_per_layer(
        src_k=host_k, dst_k=dev_k, src_v=host_v, dst_v=dev_v,
        src_indices=si, dst_indices=di, item_size=item_bytes))

    direct_ms = bench(lambda: transfer_kv_direct(
        src_layers=[host_k, host_v], dst_layers=[dev_k, dev_v],
        src_indices=si.cpu(), dst_indices=di, page_size=PAGE))

    mem_gb = 2*bs*item_bytes/(1024**3)
    print(f"bs={bs:<6d} kernel={mem_gb/(kernel_ms/1000):5.2f}GB/s  "
          f"direct={mem_gb/(direct_ms/1000):5.2f}GB/s  "
          f"speedup={direct_ms/kernel_ms:.2f}x")
```
### 方法 C：端到端 profiling（看 kernel 在真实推理里的收益）

启两套 server，**只改 `--hicache-io-backend`**，其他全部一致：

```bash
# ① kernel 组
python3 -m sglang.launch_server \
  --model-path /path/to/Llama-3.1-8B-Instruct \
  --attention-backend triton \
  --page-size 64 \
  --enable-hierarchical-cache --hicache-ratio 2 \
  --hicache-io-backend kernel \
  --hicache-mem-layout layer_first \
  --hicache-write-policy write_through \
  --port 30000

# ② direct 组（对照）
python3 -m sglang.launch_server \
  --model-path /path/to/Llama-3.1-8B-Instruct \
  --attention-backend triton \
  --page-size 64 \
  --enable-hierarchical-cache --hicache-ratio 2 \
  --hicache-io-backend direct \
  --hicache-mem-layout layer_first \
  --hicache-write-policy write_through \
  --port 30000
```
用 torch-profiler 把 kernel 粒度 timeline 抓下来：

```bash
# 1) 触发 server 开始 profile
curl http://localhost:30000/start_profile -d '{"output_dir":"/tmp/hicache_prof"}'
# 2) 跑一轮多轮对话（命中 L2，才会触发 load_to_device_per_layer）
python3 benchmark/hicache/bench_multiturn.py --port 30000 \
  --num-clients 32 --num-rounds 5 --request-length 2048
# 3) 停止
curl http://localhost:30000/stop_profile
```
trace 里需要看的 kernel 名：
- `transfer_kv_per_layer_kernel` / `transfer_kv_all_layer_kernel`（kernel backend 专有）
- MemcpyAsync `HtoD` / `DtoH` + per-layer gap（direct backend 的典型模式）

对比两份 trace 中：
- **Prefill 区间总时长**（TTFT 差异的根因）
- 每层 attention 前 H2D 拷贝的 **launch 数量和总耗时**
- 是否有 `overlap`：kernel 模式下 H2D copy 应当和 GEMM 同时出现在不同 stream；direct 模式常看到 stream 串行化。

### 方法 D：直接看指标

Server 开启 `--enable-metrics --enable-cache-report` 后，Prometheus `/metrics` 暴露了 HiCache 相关计数/直方图（如 host-pool hit、write-back latency）。对同一批 benchmark 请求下两种 backend 的 **prefill latency histogram** 做分位对比，能定量看到 kernel 带来的 TTFT 下降。

# def inorderTraversal(root):
    res = []
    stack = []
    curr = root
    
    while curr or stack:
        # 一直往左走
        while curr:
            stack.append(curr)
            curr = curr.left
        
        curr = stack.pop()
        res.append(curr.val)
        
        curr = curr.right
    
    return res

def mySqrt(x: int) -> int:
    if x < 2:
        return x

    left, right = 1, x // 2
    ans = 0

    while left <= right:
        mid = (left + right) // 2

        if mid * mid == x:
            return mid
        elif mid * mid < x:
            ans = mid
            left = mid + 1
        else:
            right = mid - 1

    return ans

def longestValidParentheses(s: str) -> int:
    stack = [-1]
    max_len = 0

    for i in range(len(s)):
        if s[i] == '(':
            stack.append(i)
        else:
            stack.pop()
            if not stack:
                stack.append(i)
            else:
                max_len = max(max_len, i - stack[-1])

    return max_len


--mamba-full-memory-ratio 2.0 

--mamba-scheduler-strategy extra_buffer \
  --page-size 64 \


from sglang.srt.layers.attention.nsa.sm89_mqa_logits import (
                deepgemm_fp8_paged_mqa_logits as triton_paged_mqa_logits,
            )

            batch_size, next_n, heads, _ = q_fp8.shape
            logits = torch.full(
                (batch_size * next_n, max_seq_len),
                float("-inf"),
                device=q_fp8.device,
                dtype=torch.float32,
            )
            triton_paged_mqa_logits(
                q_fp8,
                kv_cache_fp8,
                weights,
                logits,
                seqlens_32,
                block_tables,
                max_seq_len,
                KVBlockSize=block_kv,
                ChunkK=128,
                TotalCuCount=self.sm_count,
                WavePerEU=2,
            )
        else:


# SM < 90 (e.g. SM89/Ada): both v1 and v2 exceed the 100 KB per-block
            # shared-memory limit.  Use the two-stage partial + combine path
            # with inner_iter >= 2 so that Q stays in registers.
            block_I = 64
            inner_iter = 2  # keeps Q in registers, shared mem ≈ 80 KB
            ni = topk // block_I

            # Replicate the H_per_block logic from sparse_mla_fwd_decode_partial
            # so we can derive a safe thread count here.
            head_kv = num_heads  # kv_group = 1
            padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
            replicate_h = (head_kv // 64) if head_kv > 64 else 1
            h_per_block = padded_H if replicate_h == 1 else 64

            # The partial kernel uses T.gemm with FullCol policy, which places
            # all warps along the M dimension (= h_per_block).  Each warp's
            # M-tile must be >= 16 (MMA instruction minimum), so:
            #   threads = (h_per_block // 16) * 32
            partial_threads = max(1, h_per_block // 16) * 32

            kernel_partial = sparse_mla_fwd_decode_partial(
                num_heads,
                d_v,
                tail_dim,
                topk,
                sm_scale=sm_scale,
                block_I=block_I,
                inner_iter=inner_iter,
                threads=partial_threads,
            )
            partial_o, partial_lse = kernel_partial(
                q.unsqueeze(0), kv.unsqueeze(0), indices.unsqueeze(0)
            )
            n_groups = ni // inner_iter
            # Combine kernel has no T.gemm, so thread count is unconstrained.
            kernel_combine = sparse_mla_fwd_decode_combine(
                num_heads,
                d_v,
                n_groups * block_I,
                head_per_block=4,
                block_I=block_I,
                threads=128,
            )
            out = kernel_combine(partial_o, partial_lse)
    return out




export SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR=0.65
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.1 --trust-remote-code \
  --nnodes 4 --node-rank 0 --tp 8 --pp-size 4 \
  --port 30000 --dist-init-addr <MASTER_NODE_IP> \
  --mem-fraction-static 0.8 --attention-backend fa3 \
  --host 0.0.0.0 --watchdog-timeout 3600 \
  --max-running-requests 128 --chunked-prefill-size 12288 --enable-dynamic-chunking
https://arxiv.org/pdf/1811.08309
https://zhuanlan.zhihu.com/p/1929593678864847592
https://github.com/gty111/GEMM_MMA/blob/vector/gemm.cu


ancFyyI87muqvtIvO8rJziPkOJSWs1Dq1kTroXtTFqo=                                           

https://github.com/NVIDIA/TransformerEngine
https://github.com/NVIDIA/TileGym/tree/main/src/tilegym/ops/cutile
https://github.com/NVIDIA/cutile-python
https://github.com/NVIDIA/accelerated-computing-hub/blob/main/Accelerated_Python_User_Guide/notebooks/Chapter_01_GPU_Computing_Basics.ipynb
https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html
https://zhuanlan.zhihu.com/p/639297098
https://zhuanlan.zhihu.com/p/5750410146
https://zhuanlan.zhihu.com/p/518857175
https://developer.download.nvidia.cn/video/gputechconf/gtc/2019/presentation/s9593-cutensor-high-performance-tensor-operations-in-cuda-v2.pdf
https://zhuanlan.zhihu.com/p/555339335
https://github.com/xlite-dev/HGEMM
https://zhuanlan.zhihu.com/p/555339335
https://zhuanlan.zhihu.com/p/669926191
https://zhuanlan.zhihu.com/p/4496065391
https://zhuanlan.zhihu.com/p/441146275
https://zhuanlan.zhihu.com/p/584236348
https://github.com/AyakaGEMM/Hands-on-GEMM/tree/main/src/cuda
https://github.com/ifromeast/cuda_learning/blob/main/03_gemm/sgemm_v1.cu
https://github.com/AyakaGEMM/Hands-on-GEMM/blob/main/src/cuda/i8tc_ptx_cutlass_k32_gemm.cu
https://zhuanlan.zhihu.com/p/703256080
marlin
https://zhuanlan.zhihu.com/p/707470647

42.194.203.117:21116
42.194.203.117:21117
+Fa1jUWPsB2QxKC4+Ivk51ZswxliJjtWNTTPZY7J0tU=

110.40.200.222:21116
110.40.200.222:21117
ancFyyI87muqvtIvO8rJziPkOJSWs1Dq1kTroXtTFqo=
https://triton-lang.cn/main/getting-started/tutorials/index.html

@torch.no_grad()
def event_loop_pp_disagg_prefill(self: Scheduler) -> None:
    """A pipeline parallel scheduler loop for prefill worker in disaggregation mode."""
    
    # Initialize microbatch structures for pipeline parallelism
    mbs = [None] * self.pp_size
    last_mbs = [None] * self.pp_size
    self.running_mbs = [
        ScheduleBatch(reqs=[], batch_is_full=False) for _ in range(self.pp_size)
    ]
    bids = [None] * self.pp_size
    pp_outputs: Optional[PPProxyTensors] = None
    
    while True:
        server_is_idle = True
        
        for mb_id in range(self.pp_size):
            self.running_batch = self.running_mbs[mb_id]
            self.last_batch = last_mbs[mb_id]

            # Process incoming requests - similar to disagg_prefill
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            
            # Handle bootstrapped requests from disagg queue
            self.waiting_queue.extend(
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
            )
            
            # Process prefill chunks for current microbatch
            self.process_prefill_chunk()
            
            # Get new batch for prefill processing
            mbs[mb_id] = self.get_new_batch_prefill()
            
            # Handle MLP sync if required
            if require_mlp_sync(self.server_args):
                mbs[mb_id], _ = self.prepare_mlp_sync_batch(mbs[mb_id])
                
            self.running_mbs[mb_id] = self.running_batch
            self.cur_batch = mbs[mb_id]
            
            if self.cur_batch:
                server_is_idle = False
                result = self.run_batch(self.cur_batch)

                # (last rank) send the outputs to the next step
                if self.pp_group.is_last_rank:
                    next_token_ids, bids[mb_id] = (
                        result.next_token_ids,
                        result.bid,
                    )
                    if self.cur_batch.return_logprob:
                        pp_outputs = PPProxyTensors(
                            {
                                "next_token_ids": next_token_ids,
                                "extend_input_len_per_req": result.extend_input_len_per_req,
                                "extend_logprob_start_len_per_req": result.extend_logprob_start_len_per_req,
                            }
                            | (
                                {
                                    f"logits_output.{k}": v
                                    for k, v in result.logits_output.__dict__.items()
                                }
                                if result.logits_output is not None
                                else {}
                            )
                        )
                    else:
                        pp_outputs = PPProxyTensors(
                            {
                                "next_token_ids": next_token_ids,
                            }
                        )
                    # send the output from the last round to let the next stage worker run post processing
                    self.pp_group.send_tensor_dict(
                        pp_outputs.tensors,
                        all_gather_group=self.attn_tp_group,
                    )

            # receive outputs and post-process (filter finished reqs) the coming microbatch
            next_mb_id = (mb_id + 1) % self.pp_size
            next_pp_outputs = None
            if mbs[next_mb_id] is not None:
                next_pp_outputs: Optional[PPProxyTensors] = PPProxyTensors(
                    self.pp_group.recv_tensor_dict(
                        all_gather_group=self.attn_tp_group
                    )
                )
                mbs[next_mb_id].output_ids = next_pp_outputs["next_token_ids"]
                logits_output_args = {
                    k[len("logits_output.") :]: v
                    for k, v in next_pp_outputs.tensors.items()
                    if k.startswith("logits_output.")
                }
                if len(logits_output_args) > 0:
                    logits_output = LogitsProcessorOutput(**logits_output_args)
                else:
                    logits_output = None
                output_result = GenerationBatchResult(
                    logits_output=logits_output,
                    pp_hidden_states_proxy_tensors=None,
                    next_token_ids=next_pp_outputs["next_token_ids"],
                    extend_input_len_per_req=next_pp_outputs.tensors.get(
                        "extend_input_len_per_req", None
                    ),
                    extend_logprob_start_len_per_req=next_pp_outputs.tensors.get(
                        "extend_logprob_start_len_per_req", None
                    ),
                    bid=bids[next_mb_id],
                    can_run_cuda_graph=result.can_run_cuda_graph,
                )
                # Use disagg_prefill specific batch result processing
                self.process_batch_result_disagg_prefill(mbs[next_mb_id], output_result)
                last_mbs[next_mb_id] = mbs[next_mb_id]

            # (not last rank)
            if not self.pp_group.is_last_rank:
                if self.cur_batch:
                    bids[mb_id] = result.bid
                # carry the outputs to the next stage
                # send the outputs from the last round to let the next stage worker run post processing
                if pp_outputs:
                    self.pp_group.send_tensor_dict(
                        pp_outputs.tensors,
                        all_gather_group=self.attn_tp_group,
                    )

                # send out reqs to the next stage
                dp_offset = self.attn_dp_rank * self.attn_tp_size
                if self.attn_tp_rank == 0:
                    point_to_point_pyobj(
                        recv_reqs,
                        self.pp_rank * self.tp_size + dp_offset,
                        self.world_group.cpu_group,
                        self.pp_rank * self.tp_size + dp_offset,
                        (self.pp_rank + 1) * self.tp_size + dp_offset,
                    )

                # send out proxy tensors to the next stage
                if self.cur_batch:
                    self.pp_group.send_tensor_dict(
                        result.pp_hidden_states_proxy_tensors,
                        all_gather_group=self.attn_tp_group,
                    )

            pp_outputs = next_pp_outputs
            
            # Process inflight disagg prefill queue for current microbatch
            if len(self.disagg_prefill_inflight_queue) > 0:
                self.process_disagg_prefill_inflight_queue()
            
            # Reset batch_is_full flag for each microbatch to prevent hanging under high concurrency
            # This is critical for PD separation mode
            self.running_mbs[mb_id].batch_is_full = False

        # When the server is idle, self-check and re-init some states
        # Only do this when all microbatches are None and inflight queue is empty
        all_mbs_none = all(mb is None for mb in mbs)
        if server_is_idle and all_mbs_none and len(self.disagg_prefill_inflight_queue) == 0:
            self.check_memory()
            self.new_token_ratio = self.init_new_token_ratio
            self.maybe_sleep_on_idle()


from sglang.srt.model_executor.logits_processor import LogitsProcessorOutput
from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils import require_mlp_sync, point_to_point_pyobj
from sglang.srt.managers.scheduler import PPProxyTensors



   @torch.no_grad()
    def event_loop_pp_disagg_decode(self: Scheduler):
        """A pipeline parallelism scheduler loop for decode worker in disaggregation mode."""
        mbs = [None] * self.pp_size
        last_mbs = [None] * self.pp_size
        self.running_mbs = [
            ScheduleBatch(reqs=[], batch_is_full=False) for _ in range(self.pp_size)
        ]
        bids = [None] * self.pp_size
        pp_outputs: Optional[PPProxyTensors] = None
        
        while True:
            server_is_idle = True
            for mb_id in range(self.pp_size):
                self.running_batch = self.running_mbs[mb_id]
                self.last_batch = last_mbs[mb_id]

                recv_reqs = self.recv_requests()
                self.process_input_requests(recv_reqs)
                # polling and allocating kv cache
                self.process_decode_queue()
                mbs[mb_id] = self.get_next_disagg_decode_batch_to_run()
                self.running_mbs[mb_id] = self.running_batch

                self.cur_batch = mbs[mb_id]
                prepare_mlp_sync_flag = require_mlp_sync(self.server_args)

                if self.cur_batch:
                    server_is_idle = False
                    # Generate fake extend output.
                    if self.cur_batch.forward_mode.is_extend():
                        # Note: Logprobs should be handled on the prefill engine.
                        self.stream_output(
                            self.cur_batch.reqs, any(req.return_logprob for req in self.cur_batch.reqs)
                        )
                        if prepare_mlp_sync_flag:
                            self._prepare_idle_batch_and_run(None)
                    else:
                        if prepare_mlp_sync_flag:
                            self.prepare_mlp_sync_batch(self.cur_batch)
                        result = self.run_batch(self.cur_batch)

                # (last rank) send the outputs to the next step
                if self.pp_group.is_last_rank:
                    if self.cur_batch and not self.cur_batch.forward_mode.is_extend():
                        next_token_ids, bids[mb_id] = (
                            result.next_token_ids,
                            result.bid,
                        )
                        if self.cur_batch.return_logprob:
                            pp_outputs = PPProxyTensors(
                                {
                                    "next_token_ids": next_token_ids,
                                    "extend_input_len_per_req": result.extend_input_len_per_req,
                                    "extend_logprob_start_len_per_req": result.extend_logprob_start_len_per_req,
                                }
                                | (
                                    {
                                        f"logits_output.{k}": v
                                        for k, v in result.logits_output.__dict__.items()
                                    }
                                    if result.logits_output is not None
                                    else {}
                                )
                            )
                        else:
                            pp_outputs = PPProxyTensors(
                                {
                                    "next_token_ids": next_token_ids,
                                }
                            )
                        # send the output from the last round to let the next stage worker run post processing
                        self.pp_group.send_tensor_dict(
                            pp_outputs.tensors,
                            all_gather_group=self.attn_tp_group,
                        )

                # receive outputs and post-process (filter finished reqs) the coming microbatch
                next_mb_id = (mb_id + 1) % self.pp_size
                next_pp_outputs = None
                if mbs[next_mb_id] is not None:
                    next_pp_outputs: Optional[PPProxyTensors] = PPProxyTensors(
                        self.pp_group.recv_tensor_dict(
                            all_gather_group=self.attn_tp_group
                        )
                    )
                    mbs[next_mb_id].output_ids = next_pp_outputs["next_token_ids"]
                    logits_output_args = {
                        k[len("logits_output.") :]: v
                        for k, v in next_pp_outputs.tensors.items()
                        if k.startswith("logits_output.")
                    }
                    if len(logits_output_args) > 0:
                        logits_output = LogitsProcessorOutput(**logits_output_args)
                    else:
                        logits_output = None
                    output_result = GenerationBatchResult(
                        logits_output=logits_output,
                        pp_hidden_states_proxy_tensors=None,
                        next_token_ids=next_pp_outputs["next_token_ids"],
                        extend_input_len_per_req=next_pp_outputs.tensors.get(
                            "extend_input_len_per_req", None
                        ),
                        extend_logprob_start_len_per_req=next_pp_outputs.tensors.get(
                            "extend_logprob_start_len_per_req", None
                        ),
                        bid=bids[next_mb_id],
                        can_run_cuda_graph=result.can_run_cuda_graph,
                    )
                    self.process_batch_result(mbs[next_mb_id], output_result)
                    last_mbs[next_mb_id] = mbs[next_mb_id]

                # (not last rank)
                if not self.pp_group.is_last_rank:
                    if self.cur_batch and not self.cur_batch.forward_mode.is_extend():
                        bids[mb_id] = result.bid
                    # carry the outputs to the next stage
                    # send the outputs from the last round to let the next stage worker run post processing
                    if pp_outputs:
                        self.pp_group.send_tensor_dict(
                            pp_outputs.tensors,
                            all_gather_group=self.attn_tp_group,
                        )

                    # send out reqs to the next stage
                    dp_offset = self.attn_dp_rank * self.attn_tp_size
                    if self.attn_tp_rank == 0:
                        point_to_point_pyobj(
                            recv_reqs,
                            self.pp_rank * self.tp_size + dp_offset,
                            self.world_group.cpu_group,
                            self.pp_rank * self.tp_size + dp_offset,
                            (self.pp_rank + 1) * self.tp_size + dp_offset,
                        )

                    # send out proxy tensors to the next stage
                    if self.cur_batch and not self.cur_batch.forward_mode.is_extend():
                        self.pp_group.send_tensor_dict(
                            result.pp_hidden_states_proxy_tensors,
                            all_gather_group=self.attn_tp_group,
                        )

                pp_outputs = next_pp_outputs

            # When the server is idle, self-check and re-init some states
            if server_is_idle and (
                len(self.waiting_queue)
                + len(self.disagg_decode_transfer_queue.queue)
                + len(self.disagg_decode_prealloc_queue.queue)
                == 0
            ):
                self.check_memory()
                self.new_token_ratio = self.init_new_token_ratio
                self.maybe_sleep_on_idle()
现在的disa
        logger.info(
            f"decode rank: engine_rank={self.kv_mgr.kv_args.engine_rank}, "
            f"target_tp_ranks={self.target_tp_ranks}, target_dp_group={self.target_dp_group}, "
            f"bootstrap_addr={self.bootstrap_addr}"
        )
ggregation 不支持 pipline 并行，原因是在prefill，decode 的文件中并未像scheduler文件中那样单独实现了适应与P/D disaggregation脚骨的 event_loop_pp函数，并排查其他相关问题，给我增加PD分离模式的pipline 并行功能，保宁完善



        # PP stage之间同一个req的kv_receiver状态同步
        if hasattr(self.scheduler, 'pp_group') and self.scheduler.pp_group.world_size > 1:
            self._sync_kv_receiver_states_across_pp_stages(polls)

    def _sync_kv_receiver_states_across_pp_stages(self, polls):
        """
        在PP stage之间同步kv_receiver状态，使用all_reduce MIN操作确保状态一致性
        类似于 dist.all_reduce(tensor_to_reduce, op=dist.ReduceOp.MIN, group=pp_group)
        """
        try:
            current_rank = self.scheduler.pp_group.rank_in_group
            pp_world_size = self.scheduler.pp_group.world_size
            
            if pp_world_size <= 1:
                return  # 只有一个PP stage，无需同步
            
            # 构建request ID到本地队列索引的映射
            req_id_to_local_idx = {}
            for i, decode_req in enumerate(self.queue):
                req_id_to_local_idx[decode_req.req.rid] = i
            
            # 收集所有PP stages的kv_receiver状态
            all_stages_states = self._collect_all_pp_stages_states(polls)
            
            # 对于每个请求，使用MIN操作同步状态
            sync_updates = self._apply_min_reduce_logic(all_stages_states, req_id_to_local_idx, polls)
            
            # 输出同步日志
            if sync_updates:
                logger.info(
                    f"PP Stage {current_rank} kv_receiver sync updates:\n" +
                    "\n".join(sync_updates)
                )
            else:
                logger.debug(f"PP Stage {current_rank} kv_receiver states already consistent")
            
            # 最终barrier同步，确保所有stages完成状态同步
            self.scheduler.pp_group.barrier()
            
        except Exception as e:
            logger.warning(f"PP kv_receiver state sync failed on stage {current_rank}: {e}")
            # 同步失败时继续执行，避免阻塞整个流程
            pass

    def _collect_all_pp_stages_states(self, polls):
        """收集所有PP stages的kv_receiver状态"""
        current_rank = self.scheduler.pp_group.rank_in_group
        pp_world_size = self.scheduler.pp_group.world_size
        
        # 构建当前stage的状态
        current_stage_requests = {}
        for i, (decode_req, poll) in enumerate(zip(self.queue, polls)):
            current_stage_requests[decode_req.req.rid] = {
                "poll_status": int(poll),
                "bootstrap_room": decode_req.req.bootstrap_room,
                "local_idx": i
            }
        
        # 收集所有stages的状态
        all_stages_states = [None] * pp_world_size
        all_stages_states[current_rank] = current_stage_requests
        
        # 与其他stages交换状态信息
        for other_rank in range(pp_world_size):
            if other_rank == current_rank:
                continue
                
            # 使用rank顺序避免死锁
            if current_rank < other_rank:
                self.scheduler.pp_group.send_object(current_stage_requests, dst=other_rank)
                other_stage_requests = self.scheduler.pp_group.recv_object(src=other_rank)
            else:
                other_stage_requests = self.scheduler.pp_group.recv_object(src=other_rank)
                self.scheduler.pp_group.send_object(current_stage_requests, dst=other_rank)
            
            all_stages_states[other_rank] = other_stage_requests
        
        return all_stages_states

    def _apply_min_reduce_logic(self, all_stages_states, req_id_to_local_idx, polls):
        """
        应用MIN reduce逻辑，对于相同的kv_receiver取最小状态值
        类似于: tensor_to_reduce = min(tensor_to_reduce) across all PP stages
        """
        current_rank = self.scheduler.pp_group.rank_in_group
        sync_updates = []
        
        # 收集所有stages中出现的请求ID
        all_req_ids = set()
        for stage_requests in all_stages_states:
            if stage_requests is not None:
                all_req_ids.update(stage_requests.keys())
        
        # 对每个请求ID进行MIN reduce操作
        for req_id in all_req_ids:
            if req_id not in req_id_to_local_idx:
                continue  # 当前stage没有这个请求，跳过
                
            local_idx = req_id_to_local_idx[req_id]
            current_status = int(polls[local_idx])
            
            # 收集所有stages中这个请求的状态
            req_statuses = []
            stages_with_req = []
            
            for stage_rank, stage_requests in enumerate(all_stages_states):
                if stage_requests is not None and req_id in stage_requests:
                    req_statuses.append(stage_requests[req_id]["poll_status"])
                    stages_with_req.append(stage_rank)
            
            if len(req_statuses) > 1:
                # 使用MIN操作找到最小状态值（最保守的状态）
                min_status = min(req_statuses)
                
                if min_status != current_status:
                    # 更新本地状态为最小值
                    polls[local_idx] = min_status
                    sync_updates.append(
                        f"Req {req_id}: MIN reduce updated poll_status from {current_status} "
                        f"to {min_status} (stages: {stages_with_req}, statuses: {req_statuses})"
                    )
                else:
                    logger.debug(
                        f"Req {req_id}: Already at MIN status {current_status} "
                        f"(stages: {stages_with_req}, statuses: {req_statuses})"
                    )
        
        return sync_updates

# DeepSeek 32B 多机部署配置
# 需要至少2台8卡机器
# 使用 hostNetwork 直接通过宿主机IP访问，无需Service

---
# Prefill Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deepseek32b-prefill
  labels:
    app: deepseek32b-prefill
    yice: "true"
    environment: test
    release: test
spec:
  replicas: 1
  selector:
    matchLabels:
      app: deepseek32b-prefill
      role: prefill
  template:
    metadata:
      labels:
        app: deepseek32b-prefill
        role: prefill
        component: prefill
        yice: "true"
        environment: test
        release: test
    spec:
      containers:
      - name: sglang-prefill
        image: aicr.byd.com/docker.io/lmsysorg/sglang:v0.4.7-cu124-post1
        command:
        - python3
        - -m
        - sglang.launch_server
        - --port
        - "30000"
        - --host
        - "0.0.0.0"
        - --model-path
        - /models/DeepSeek-R1-Distill-Qwen-32B
        - --page-size
        - "64"
        - --disaggregation-mode
        - prefill
        - --mem-fraction-static
        - "0.85"
        - --tp-size
        - "8"  # 使用全部8张GPU
        - --disaggregation-ib-device
        - mlx5_bond_0
        - --trust-remote-code
        - --quantization
        - fp8
        - --kv-cache-dtype
        - fp8_e5m2
        - --attention-backend
        - flashinfer
        env:
        - name: NVSHMEM_HCA_PE_MAPPING
          value: "mlx5_bond_0:1:2"
        - name: NVSHMEM_IB_GID_INDEX
          value: "3"
        - name: NVSHMEM_ENABLE_NIC_PE_MAPPING
          value: "1"
        - name: SGLANG_SET_CPU_AFFINITY
          value: "true"
        - name: SGL_ENABLE_JIT_DEEPGEMM
          value: "1"
        - name: NCCL_IB_QPS_PER_CONNECTION
          value: "8"
        - name: NCCL_IB_SPLIT_DATA_ON_QPS
          value: "1"
        - name: NCCL_NET_PLUGIN
          value: none
        - name: NCCL_IB_TC
          value: "136"
        - name: NCCL_MIN_NCHANNELS
          value: "4"
        - name: MC_TE_METRIC
          value: "false"
        - name: NCCL_IB_SL
          value: "5"
        - name: NCCL_IB_HCA
          value: "mlx5_bond_0"
        - name: NCCL_SOCKET_IFNAME
          value: "bond1"
        readinessProbe:
          periodSeconds: 30
          tcpSocket:
            port: 30000
        resources:
          limits:
            nvidia.com/gpu: "8"  # 使用全部8张GPU
        securityContext:
          capabilities:
            add:
            - IPC_LOCK
          privileged: true
        volumeMounts:
        - mountPath: /dev/shm
          name: dshm
        - mountPath: /models
          name: host-models
        - mountPath: /dev/infiniband
          name: ib

      # 使用 affinity 选择匹配的机器
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: org
                operator: In
                values:
                - "yiceai"
              - key: yiceai
                operator: In
                values:
                - "true"
              - key: deploy
                operator: In
                values:
                - "deepseekr1-32b-pd-p"
        # 反亲和性确保 prefill 和 decode 不在同一节点
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - deepseek32b-decode
            topologyKey: kubernetes.io/hostname
      volumes:
      - emptyDir:
          medium: Memory
        name: dshm
      - hostPath:
          path: /export/models
        name: host-models
      - hostPath:
          path: /dev/infiniband
        name: ib
      dnsPolicy: Default
      hostIPC: true
      hostNetwork: true  # 使用宿主机网络

---
# Decode Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deepseek32b-decode
  labels:
    app: deepseek32b-decode
    yice: "true"
    environment: test
    release: test
spec:
  replicas: 1
  selector:
    matchLabels:
      app: deepseek32b-decode
      role: decode
  template:
    metadata:
      labels:
        app: deepseek32b-decode
        role: decode
        component: decode
        yice: "true"
        environment: test
        release: test
    spec:
      containers:
      - name: sglang-decode
        image: aicr.byd.com/docker.io/lmsysorg/sglang:v0.4.7-cu124-post1
        command:
        - python3
        - -m
        - sglang.launch_server
        - --port
        - "30001"
        - --host
        - "0.0.0.0"
        - --model-path
        - /models/DeepSeek-R1-Distill-Qwen-32B
        - --page-size
        - "64"
        - --disaggregation-mode
        - decode
        - --mem-fraction-static
        - "0.85"
        - --tp-size
        - "8"  # 使用全部8张GPU
        - --disaggregation-ib-device
        - mlx5_bond_0
        - --trust-remote-code
        - --quantization
        - fp8
        - --kv-cache-dtype
        - fp8_e5m2
        - --attention-backend
        - flashinfer
        env:
        - name: NVSHMEM_HCA_PE_MAPPING
          value: "mlx5_bond_0:1:2"
        - name: NVSHMEM_IB_GID_INDEX
          value: "3"
        - name: NVSHMEM_ENABLE_NIC_PE_MAPPING
          value: "1"
        - name: SGLANG_SET_CPU_AFFINITY
          value: "true"
        - name: SGL_ENABLE_JIT_DEEPGEMM
          value: "1"
        - name: NCCL_IB_QPS_PER_CONNECTION
          value: "8"
        - name: NCCL_IB_SPLIT_DATA_ON_QPS
          value: "1"
        - name: NCCL_NET_PLUGIN
          value: none
        - name: NCCL_IB_TC
          value: "136"
        - name: NCCL_MIN_NCHANNELS
          value: "4"
        - name: MC_TE_METRIC
          value: "false"
        - name: NCCL_IB_SL
          value: "5"
        - name: NCCL_IB_HCA
          value: "mlx5_bond_0"
        - name: NCCL_SOCKET_IFNAME
          value: "bond1"
        readinessProbe:
          periodSeconds: 30
          tcpSocket:
            port: 30001
        resources:
          limits:
            nvidia.com/gpu: "8"  # 使用全部8张GPU
        securityContext:
          capabilities:
            add:
            - IPC_LOCK
          privileged: true
        volumeMounts:
        - mountPath: /dev/shm
          name: dshm
        - mountPath: /models
          name: host-models
        - mountPath: /dev/infiniband
          name: ib

      # 使用 affinity 选择匹配的机器
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: org
                operator: In
                values:
                - "yiceai"
              - key: yiceai
                operator: In
                values:
                - "true"
              - key: deploy
                operator: In
                values:
                - "deepseekr1-32b-pd-p"
        # 反亲和性确保 prefill 和 decode 不在同一节点
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - deepseek32b-prefill
            topologyKey: kubernetes.io/hostname
      volumes:
      - emptyDir:
          medium: Memory
        name: dshm
      - hostPath:
          path: /export/models
        name: host-models
      - hostPath:
          path: /dev/infiniband
        name: ib
      dnsPolicy: Default
      hostIPC: true
      hostNetwork: true  # 使用宿主机网络

---
# Load Balancer Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deepseek32b-loadbalancer
  labels:
    app: deepseek32b-loadbalancer
    yice: "true"
    environment: test
    release: test
spec:
  replicas: 1
  selector:
    matchLabels:
      app: deepseek32b-loadbalancer
      role: loadbalancer
  template:
    metadata:
      labels:
        app: deepseek32b-loadbalancer
        role: loadbalancer
        component: loadbalancer
        yice: "true"
        environment: test
        release: test
    spec:
      containers:
      - name: sgl-loadbalancer
        image: lmsysorg/sglang:latest
        command:
        - python
        - -m
        - sglang.srt.disaggregation.mini_lb
        - --prefill
        - $(PREFILL_HOST_URL)
        - --decode
        - $(DECODE_HOST_URL)
        - --host
        - 0.0.0.0
        - --port
        - "8000"
        env:
        - name: PREFILL_HOST_URL
          value: "http://192.168.1.100:30000"  # 替换为实际的 prefill 宿主机IP
        - name: DECODE_HOST_URL
          value: "http://192.168.1.101:30001"  # 替换为实际的 decode 宿主机IP
        readinessProbe:
          periodSeconds: 30
          tcpSocket:
            port: 8000
        resources:
          limits:
            cpu: "2"
            memory: "4Gi"

      # 使用 affinity 选择匹配的机器，建议与 prefill/decode 分离
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: org
                operator: In
                values:
                - "yiceai"
              - key: yiceai
                operator: In
                values:
                - "true"
              - key: deploy
                operator: In
                values:
                - "deepseekr1-32b-pd-p"
        # 反亲和性确保 LB 不与 prefill/decode 在同一节点（可选）
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - deepseek32b-prefill
                  - deepseek32b-decode
              topologyKey: kubernetes.io/hostname
      hostNetwork: true  # 使用宿主机网络
      dnsPolicy: Default


#!/bin/bash

set -e  # 遇到错误时退出脚本

# 安装 git 和 git-lfs（适配 Debian/Ubuntu，其他系统需手动替换）
if ! command -v git &> /dev/null; then
    echo "安装 git..."
    sudo apt-get update
    sudo apt-get install -y git
else
    echo "git 已安装"
fi

if ! command -v git-lfs &> /dev/null; then
    echo "安装 git-lfs..."
    sudo apt-get install -y git-lfs
else
    echo "git-lfs 已安装"
fi

# 激活 git-lfs
echo "激活 git-lfs..."
git lfs install

# 创建并进入目标目录
mkdir -p ms-models
cd ms-models

# Clone 模型仓库
echo "Cloning Qwen3-235B-A22B-Instruct-2507-FP8..."
git clone https://www.modelscope.cn/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8.git

echo "Cloning Qwen3-Coder-480B-A35B-Instruct-FP8..."
git clone https://www.modelscope.cn/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8.git

MOE
https://pan.baidu.com/s/1gcjUMDz_vcmLS6yZAeH35Q?pwd=pbo8
https://zhuanlan.zhihu.com/p/719466709
https://triton.csdn.net/6719f57fcd8b2677c3d4ab94.html
https://zhuanlan.zhihu.com/p/21251657579
https://zhuanlan.zhihu.com/p/21251657579
https://zhuanlan.zhihu.com/p/25401744621
https://zhuanlan.zhihu.com/p/1895178845830771205
https://blog.csdn.net/zpp13hao1/article/details/147891337
https://zhuanlan.zhihu.com/p/1911059432953061899
https://mmssai.com/archives/33850
https://www.cnblogs.com/cavalier-chen/p/18937098
https://zhuanlan.zhihu.com/p/26436168971
https://zhuanlan.zhihu.com/p/1895178845830771205
https://blog.csdn.net/zpp13hao1/article/details/147891337
https://mmssai.com/archives/33850
echo "所有操作完成！"



nvshmem
https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/abstract.html
https://zhuanlan.zhihu.com/p/26575363781
https://blog.csdn.net/jkjgj/article/details/142066495
https://zhuanlan.zhihu.com/p/1952325639211315297
https://zhuanlan.zhihu.com/p/1941562551407187752
https://zhuanlan.zhihu.com/p/1933899894881489690
https://blog.csdn.net/jkjgj/article/details/142066495
https://zhuanlan.zhihu.com/p/26082845081
https://www.51cto.com/aigc/6550.html
https://blog.csdn.net/gitblog_00341/article/details/151640319



__device__ __forceinline__ float gelu(const float& val) {
  constexpr float kAlpha = M_SQRT1_2;
  return val * 0.5f * (1.0f + ::erf(val * kAlpha));
}

__device__ __forceinline__ float gelu_tanh(const float& val) {
  const float cdf =
      0.5f * (1.0f + math::tanh((0.7978845608028654f * (val + 0.044715f * val * val * val))));
  return val * cdf;
}

void silu_and_mul(at::Tensor& out, at::Tensor& input, bool enable_pdl) {
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);

  const c10::cuda::OptionalCUDAGuard device_guard(out.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), c_type, [&] {
    uint32_t vec_size = 16 / sizeof(c_type);
    cudaLaunchConfig_t config;
    config.gridDim = num_tokens;
    config.blockDim = std::min(d / vec_size, 1024U);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
    config.numAttrs = 1;
    config.attrs = attrs;
    auto kernel = flashinfer::activation::act_and_mul_kernel<c_type, silu>;

    cudaLaunchKernelEx(&config, kernel, static_cast<c_type*>(out.data_ptr()),
                       static_cast<c_type*>(input.data_ptr()), d);

namespace activation {
template <typename T, float (*Activation)(const float&)>
__global__ void act_and_mul_kernel(T* __restrict__ out, const T* __restrict__ input, const int d) {
  constexpr uint32_t vec_size = 16 / sizeof(T);
  const int64_t token_idx = blockIdx.x;
  const int64_t thread_idx = threadIdx.x;
  const int64_t stride = blockDim.x;
  const int64_t offset = token_idx * 2 * d;

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

#pragma unroll 1
  for (uint32_t idx = thread_idx; idx < d / vec_size; idx += stride) {
    vec_t<float, vec_size> x_vec, y_vec, out_vec;
    x_vec.cast_load(input + offset + idx * vec_size);
    y_vec.cast_load(input + offset + d + idx * vec_size);
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      out_vec[i] = Activation(x_vec[i]) * y_vec[i];
    }
    out_vec.cast_store(out + token_idx * d + idx * vec_size);
  }

  const int64_t remaining_offset = d - d % (stride * vec_size);
  // process the remaining elements
#pragma unroll 1
  for (int64_t idx = thread_idx; idx < d % (stride * vec_size); idx += stride) {
    float x = input[offset + remaining_offset + idx],
          y = input[offset + remaining_offset + d + idx];
    out[token_idx * d + remaining_offset + idx] = Activation(x) * y;
  }

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

__device__ __forceinline__ __nv_bfloat162 silu_mul_half2(const __nv_bfloat162& val, const __nv_bfloat162& factor ) {
      return __hmul2(__h2div(val, __hadd2(__float2bfloat162_rn(1.0f), h2exp(__hneg2(val)))), factor);
}
__global__ void act_mul_f32_kernel(float* __restrict__ out, const float* __restrict__ input, const int d) {
        const int64_t vec_size = 4;
    const int64_t token_idx = blockIdx.x;
    const int64_t thread_idx = threadIdx.x;
    const int64_t stride = blockDim.x;
    const int64_t offset = token_idx * 2 * d;

    cudaDeviceSynchronize(); 
    cudaError_t err = cudaGetLastError(); 
    if (err != cudaSuccess) { 
        printf("CUDA Error: %s\n",cudaGetErrorString(err)); 
    }



# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Convert TRT-LLM GPTQ format checkpoint to Kimi-K2-Thinking compressed-tensors format.
The source GPTQ checkpoint uses:
- int32 packing: 8 int4 values per int32
- Requires .qweight, .scales, .qzeros tensors
TRT-LLM compressed-tensors format uses:
- int32 packing: 8 int4 values per int32 (same)
- group_size: 32
- symmetric quantization
"""

import argparse
import json
from pathlib import Path

import safetensors
import safetensors.torch
import torch
from tqdm import tqdm


def unpack_int32_to_int4_gptq(weight_packed: torch.Tensor) -> torch.Tensor:
    """
    Unpack GPTQ int32 tensor containing 8 int4 values into int8 tensor.
    Args:
        weight_packed: Shape (K/8, N) dtype int32
    Returns:
        unpacked: Shape (K, N) dtype int8 with values in range [-8, 7]
    """
    # Convert int32 to uint8 view to extract nibbles
    w_packed_uint8 = weight_packed.contiguous().view(torch.uint8)

    # Each int32 = 4 bytes, each byte has 2 int4 values
    k_div_8, n = weight_packed.shape
    w_packed_uint8 = w_packed_uint8.view(k_div_8 * 4, n)

    # Allocate output: (K, N) where K = K_div_8 * 8
    k = k_div_8 * 8
    w_unpacked = torch.zeros(k, n, dtype=torch.int8)

    # Extract low and high nibbles
    w_unpacked[0::2, :] = (w_packed_uint8 & 0x0F).to(torch.int8)
    w_unpacked[1::2, :] = (w_packed_uint8 >> 4).to(torch.int8)

    # Convert from uint4 [0, 15] to int4 [-8, 7]
    # Values > 7 should be interpreted as negative
    w_unpacked[w_unpacked > 7] -= 16

    return w_unpacked.contiguous()


def pack_int4_to_int32_compressed(weight_unpacked: torch.Tensor) -> torch.Tensor:
    """
    Pack int8 tensor (with int4 values) into int32 compressed-tensors format.
    Args:
        weight_unpacked: Shape (N, K) dtype int8 with values in range [-8, 7]
    Returns:
        packed: Shape (N, K/8) dtype int32
    """
    n, k = weight_unpacked.shape
    assert k % 8 == 0, "K must be divisible by 8"

    # Convert int4 [-8, 7] to uint4 [0, 15]
    w_uint4 = weight_unpacked.clone()
    w_uint4[w_uint4 < 0] += 16
    w_uint4 = w_uint4.to(torch.uint8)

    # Pack 2 uint4 into 1 uint8
    w_packed_uint8 = torch.zeros(n, k // 2, dtype=torch.uint8)
    w_packed_uint8 = (w_uint4[:, 1::2] << 4) | (w_uint4[:, 0::2])

    # Reshape to int32: (N, K/2) uint8 -> (N, K/8, 4) uint8 -> (N, K/8) int32
    w_packed_int32 = (
        w_packed_uint8.view(n, k // 8, 4).contiguous().view(torch.uint8).view(n, k // 8).view(torch.int32)
    )

    return w_packed_int32.contiguous()


def convert_gptq_to_compressed_tensor(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int = 32,
) -> dict[str, torch.Tensor]:
    """
    Convert GPTQ format to compressed-tensors format.
    Args:
        qweight: Shape (K/8, N) dtype int32 (GPTQ format is transposed)
        scales: Shape (K/group_size, N) dtype fp16 (GPTQ format is transposed)
        qzeros: Shape (K/group_size, N/8) dtype int32 (not used for symmetric)
        group_size: Quantization group size
    Returns:
        Dictionary with:
            - weight_packed: Shape (N, K/8) dtype int32
            - weight_scale: Shape (N, K/group_size) dtype bfloat16
            - weight_shape: Shape (2,) dtype int64 containing [N, K]
    """
    k_div_8, n = qweight.shape
    k = k_div_8 * 8

    # GPTQ weights are transposed: (K/8, N) -> Need to transpose back to (N, K/8)
    # First unpack to (K, N), then transpose to (N, K), then repack to (N, K/8)
    
    # Unpack GPTQ format
    weight_unpacked = unpack_int32_to_int4_gptq(qweight)  # Shape: (K, N)
    
    # Transpose to compressed-tensors layout
    weight_unpacked = weight_unpacked.t().contiguous()  # Shape: (N, K)
    
    # Repack in compressed-tensors format
    weight_packed = pack_int4_to_int32_compressed(weight_unpacked)  # Shape: (N, K/8)

    # Transpose scales: (K/group_size, N) -> (N, K/group_size)
    weight_scale = scales.t().contiguous().to(torch.bfloat16)

    # Store original weight shape
    weight_shape = torch.tensor([n, k], dtype=torch.int64)

    return {
        "weight_packed": weight_packed,
        "weight_scale": weight_scale,
        "weight_shape": weight_shape,
    }


def convert_checkpoint(
    input_dir: str,
    output_dir: str,
    num_shards: int | None = None,
    skip_existing: bool = True,
):
    """
    Convert all shards from GPTQ to compressed-tensors format.
    Args:
        input_dir: Source checkpoint directory (GPTQ format)
        output_dir: Output checkpoint directory (compressed-tensors format)
        num_shards: Number of shards to process (None = all)
        skip_existing: Skip conversion if output shard already exists
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all safetensors files
    shard_files = sorted(input_path.glob("model-*.safetensors"))
    if not shard_files:
        raise ValueError(f"No model shards found in {input_dir}")

    if num_shards is not None:
        shard_files = shard_files[:num_shards]

    print(f"Found {len(shard_files)} shards to process")

    # Track weight mapping for index file
    new_weight_map = {}

    # Load and convert each shard
    for shard_idx, shard_file in enumerate(tqdm(shard_files, desc="Processing shards")):
        shard_name = shard_file.name
        output_file = output_path / shard_name

        # Check if output already exists
        if skip_existing and output_file.exists():
            print(f"\n⏭️  Skipping {shard_name} (already exists)")
            # Still need to build the weight_map from existing file
            with safetensors.safe_open(str(output_file), framework="pt", device="cpu") as f:
                for key in f:
                    new_weight_map[key] = shard_name
            continue

        print(f"\n🔄 Converting {shard_file.name}...")

        # Load source shard
        source_tensors = {}
        with safetensors.safe_open(str(shard_file), framework="pt", device="cpu") as f:
            for key in f:
                source_tensors[key] = f.get_tensor(key)

        # Convert tensors
        output_tensors = {}

        # Group GPTQ tensors by base key
        processed_keys = set()
        
        for key in tqdm(source_tensors.keys(), desc="Converting tensors", leave=False):
            if key in processed_keys:
                continue
                
            if key.endswith(".qweight"):
                # This is a quantized weight - convert to compressed-tensors format
                base_key = key[: -len(".qweight")]
                scales_key = base_key + ".scales"
                qzeros_key = base_key + ".qzeros"

                if scales_key in source_tensors and qzeros_key in source_tensors:
                    # Convert to compressed-tensors format
                    compressed_tensors = convert_gptq_to_compressed_tensor(
                        qweight=source_tensors[key],
                        scales=source_tensors[scales_key],
                        qzeros=source_tensors[qzeros_key],
                        group_size=32,
                    )

                    # Save with compressed-tensors naming convention
                    packed_key = base_key + ".weight_packed"
                    scale_key = base_key + ".weight_scale"
                    shape_key = base_key + ".weight_shape"

                    output_tensors[packed_key] = compressed_tensors["weight_packed"]
                    output_tensors[scale_key] = compressed_tensors["weight_scale"]
                    output_tensors[shape_key] = compressed_tensors["weight_shape"]

                    new_weight_map[packed_key] = shard_name
                    new_weight_map[scale_key] = shard_name
                    new_weight_map[shape_key] = shard_name

                    # Mark related keys as processed
                    processed_keys.add(key)
                    processed_keys.add(scales_key)
                    processed_keys.add(qzeros_key)
                else:
                    print(f"Warning: Missing scales or qzeros for {key}")

            elif key.endswith((".scales", ".qzeros")):
                # Skip these as they're handled above
                processed_keys.add(key)
                continue
            else:
                # Keep non-quantized tensors as-is
                output_tensors[key] = source_tensors[key]
                new_weight_map[key] = shard_name
                processed_keys.add(key)

        # Save converted shard
        safetensors.torch.save_file(output_tensors, str(output_file))
        print(f"✅ Saved to {output_file}")

    # Copy config.json and update quantization settings
    config_file = input_path / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)

        # Remove TRT-LLM quantization config if present
        config.pop("quantization", None)

        # Add HuggingFace compressed-tensors quantization config
        config["quantization_config"] = {
            "quant_method": "compressed-tensors",
            "format": "int-quantized",
            "group_size": 32,
            "num_bits": 4,
            "strategy": "channel",
            "type": "symmetric",
        }

        output_config_file = output_path / "config.json"
        with open(output_config_file, "w") as f:
            json.dump(config, f, indent=2)
        print(f"\nSaved config to {output_config_file}")

    # Generate new safetensors index file
    index_data = {
        "metadata": {
            "total_size": sum(
                (output_path / shard_file.name).stat().st_size for shard_file in shard_files
            )
        },
        "weight_map": new_weight_map,
    }

    index_file = output_path / "model.safetensors.index.json"
    with open(index_file, "w") as f:
        json.dump(index_data, f, indent=2)
    print(f"\nGenerated index file: {index_file}")
    print(f"  Total tensors: {len(new_weight_map)}")

    # Copy other necessary files
    import shutil

    # JSON files (tokenizer and generation config)
    for file in [
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]:
        src = input_path / file
        if src.exists():
            shutil.copy(src, output_path / file)
            print(f"Copied: {file}")

    # Python files (model architecture, custom tokenizers)
    for file in ["configuration_deepseek.py", "modeling_deepseek.py", "tokenization_kimi.py"]:
        src = input_path / file
        if src.exists():
            shutil.copy(src, output_path / file)
            print(f"Copied: {file}")

    # Tokenizer model files
    for file in ["tiktoken.model", "tokenizer.model", "sentencepiece.model"]:
        src = input_path / file
        if src.exists():
            shutil.copy(src, output_path / file)
            print(f"Copied: {file}")

    # Template files
    for file in ["chat_template.jinja", "chat_template.json"]:
        src = input_path / file
        if src.exists():
            shutil.copy(src, output_path / file)
            print(f"Copied: {file}")

    print(f"\n✓ Conversion complete! Output saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert GPTQ checkpoint to Kimi-K2 compressed-tensors format"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/home/scratch.omniml_data_2/zhiyuc/checkpoints/Kimi-K2-Thinking-GPTQ",
        help="Input checkpoint directory with GPTQ format",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/scratch.omniml_data_1/models/Kimi-K2-Thinking-Compressed",
        help="Output directory for compressed-tensors format checkpoint",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Number of shards to convert (default: all)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-convert shards even if they already exist (default: skip existing)",
    )

    args = parser.parse_args()

    convert_checkpoint(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_shards=args.num_shards,
        skip_existing=not args.no_skip_existing,
    )


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
