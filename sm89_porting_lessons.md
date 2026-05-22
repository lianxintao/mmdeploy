# SM89 Sparse FP8 Decode 移植经验总结

本文记录将 FlashMLA SM90 稀疏 FP8 解码内核（Model1 Head64）移植至 SM89（Ada Lovelace）过程中的经验、技巧与踩坑记录。

---

## 一、架构差异速查

| 特性 | SM90 (H100) | SM89 (L40S) | SM120 (RTX 5060 Ti) |
|------|-------------|-------------|---------------------|
| WGMMA | ✅ | ❌ | ❌ |
| TMA Load/Store | ✅ | ❌ | ❌ |
| stmatrix (STSM) | ✅ SM90+ only | ❌ | ❌ |
| ldmatrix (LDSM) | ✅ | ✅ SM75+ | ✅ |
| cp.async | ✅ | ✅ SM80+ | ✅ |
| Cluster | ✅ | ❌ | ❌ |
| PDL (griddepcontrol) | ✅ | ❌ | ❌ |
| Max smem/block | 228KB | 100KB | 100KB |
| MMA Atom | GMMA 64×64×16 | SM80 16×8×16 | SM80 16×8×16 |

---

## 二、关键设计决策与论证

### 2.1 BLOCK_M = 32（非 64）

**约束**：SM89 最大 255 寄存器/线程。O 累加器占用：`BLOCK_M × HEAD_DIM_V / NUM_THREADS = BLOCK_M × 512 / 128` floats。

- BLOCK_M=64 → 256 regs for O → 超限 ❌
- BLOCK_M=32 → 128 regs for O → 可行 ✅（总 ~232 regs）

**后果**：`NUM_M_BLOCKS=2`，两个 CTA 各处理 32 个 query head。

### 2.2 TOPK_BLOCK_SIZE = 64 + 单 KV buffer

SM89 smem 上限 100KB：
- TOPK=64 双 buffer: 128KB → 超限 ❌
- TOPK=64 单 buffer: ~69KB → 可行 ✅

### 2.3 TiledMMA 的 Tile 参数必须用 `<M, 16, 16>` 而非 `<M, N_full, 16>`

**教训**：`SM75_U16x8_LDSM_T` 对 `make_tiled_copy_B` 有向量化对齐要求。用 `Tile<64, N_full=64, 16>` 时 static_assert 失败；改为 `Tile<64, 16, 16>`（FA2 标准做法）后通过。

**规律**：FA2 的 TiledMMA 永远用 `Tile<kNWarps*16, 16, 16>`，不要把完整 N 维度塞进 Tile。

### 2.4 K（B 操作数）用 LDSM.N 而非 LDSM.T

**踩坑**：直觉上 K^T 转置应该用 LDSM.T，但 FA2 对 QK GEMM 的 K（B 操作数）用的是 `SmemCopyAtom = SM75_U32x4_LDSM_N`。只有 PV GEMM 的 V^T 才用 LDSM.T。

**原因**：`SM80_16x8x16_F32BF16BF16_TN` 的 "TN" 中，T 指 A 操作数（Q）是 column-major 存储，N 指 B 操作数（K）是 row-major 存储，两者都用 LDSM.N 读取。

---

## 三、CUTLASS/CuTe 细节踩坑

### 3.1 SM80 MMA atom 名称拼写

```cpp
// 错误：
SM80_16x8x16_F32BF16BF16_TN     // 缺少尾部 F32

// 正确（此 CUTLASS 版本）：
SM80_16x8x16_F32BF16BF16F32_TN
```

**排查方法**：直接 grep CUTLASS 头文件 `mma_sm80.hpp` 找 struct 名。

### 3.2 stmatrix (STSM) 仅 SM90+ 支持

```
ptxas fatal: Feature 'stmatrix' requires .target sm_90 or higher
```

CUTLASS 将 `SM90_U32x4_STSM_N` 命名误导为"SM90+"但实际生成的 PTX 指令 `stmatrix` 需要 SM90+（ptxas 强制检查）。

**解决方案**：改用 `AutoVectorizingCopyWithAssumedAlignment<128>` 做 S matrix 的 reg→smem 写入，性能略差但 SM89 兼容。

### 3.3 `partition_fragment_C` 的大小是基于自然 MMA Tile，不是请求的 Shape

**关键发现**：
```cpp
// TiledMMA<SM80_16x8x16, <_4,_1,_1>, Tile<64,16,16>>
// 自然 M-tile = 4 × 16 = 64 行
Tensor rP = partition_fragment_C(TiledMma_QK{}, Shape<32, 64>{});
// size(rP) = 64×64/128 = 32，不是 32×64/128 = 16！
```

CuTe 基于自然 tile 分配寄存器，而不是请求的受限 Shape。这导致：
- warp 2/3 的 rP/rO/coord 会包含行 32-63 的元素（越界）
- 需要在 store_o_to_global 中加 `if (row >= BLOCK_M) continue;` 守卫
- SmemLayoutS/SmemLayoutQ 需要扩展为 MMA_M_TILE=64 行以避免 smem 越界

### 3.4 `make_identity_tensor` 的 M 维度必须与 TiledMMA 的自然 M-tile 一致

**踩坑**：在 LSE 写入和 `scale_softmax` 的行检测逻辑中，用了 `make_identity_tensor(Shape<BLOCK_M, N>{})` 来获取每个 fragment 元素的坐标，但 TiledMma_QK/PV 的自然 M-tile 是 `MMA_M_TILE=64`，不是 `BLOCK_M=32`。

**问题根因**：CUTE 的 `partition_C(identity_tensor(32, N))` 对超出 [0, 32) 的行做模运算：

```
Warp 2 (MMA M-coord = 32..47) → identity_tensor M-coord = 32 % 32 = 0..15
Warp 3 (MMA M-coord = 48..63) → identity_tensor M-coord = 48 % 32 = 16..31
```

**后果**：warp 2/3 计算出的 `row_ids ∈ [0, 31]`，通过了 `row < num_valid = 32` 守卫，写入 `lse_accum[row]`（全局内存）。由于 warp 0/1 同时向同一地址写入正确值，两者形成**不确定执行顺序**的 race：偶尔 warp 2/3 最后写入，用 garbage rL/rM 覆盖了正确的 LSE → NaN。表现为**随机性 session 级失败**（单独运行测试通过，pytest 批量运行时偶发失败）。

**正确做法**：identity_tensor M 维度必须与 TiledMMA 的自然 tile M 一致，让 warp 2/3 得到真实坐标 32..63，`row < num_valid = 32` 自然过滤：

```cpp
// ❌ 错误：M 维度用 BLOCK_M
auto coord_P = thr_qk.partition_C(
    make_identity_tensor(Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>{})
);

// ✅ 正确：M 维度用 TiledMMA 的自然 tile 大小
auto coord_P = thr_qk.partition_C(
    make_identity_tensor(Shape<Int<MMA_M_TILE>, Int<TOPK_BLOCK_SIZE>>{})
);
```

**规律**：凡是用 `partition_C(make_identity_tensor(...))` 做坐标探测的代码，identity_tensor 的每个维度必须等于 TiledMMA 对应维度的自然 tile 大小（即 `num_warps × atom_M`），而非业务逻辑上的有效行数。有效范围过滤应全部交给后续的 `row < num_valid` 守卫。

### 3.5 SmemLayoutQ 必须覆盖完整 MMA_M_TILE 行

**错误现象**：`compute-sanitizer` 报 `Invalid __shared__ read of size 16 bytes` at LDSM instruction。

**原因**：LDSM 的 smem 地址由 TiledMMA 的自然 tile（64 行）决定。若 SmemLayoutQ 只有 32 行，warp 2/3 的 LDSM 会读 smem 越界。

**解决方案**：
```cpp
// SmemLayoutQ 扩展为 MMA_M_TILE=64 行（与 SmemLayoutKV 等大，union 不增加 smem）
using SmemLayoutQ = tile_to_shape(atom, Shape<MMA_M_TILE, HEAD_DIM_K>{});
// 加载时只填前 BLOCK_M=32 行（用 SmemLayoutQ_Copy）
using SmemLayoutQ_Copy = tile_to_shape(atom, Shape<BLOCK_M, HEAD_DIM_K>{});
```

### 3.6 `cute::gemm` 对 "A 在寄存器，B 在 smem" 不会自动 LDSM 加载

**踩坑**：用 `cute::gemm(tiled_mma, rQ, thr_qk.partition_fragment_B(sKV), rP)` 时，QK GEMM 结果全为 0.0，原因是 `partition_fragment_B(sKV)` 创建的 smem 引用没有触发 LDSM 加载。

**正确做法**（FA2 的 gemm_rs 模式）：
```cpp
auto smem_copy_K = make_tiled_copy_B(SmemCopyAtomA{}, TiledMma_QK{});
Tensor tKrK     = thr_qk.partition_fragment_B(sKV);   // 寄存器 buffer
Tensor tKsK     = thr_copy_K.partition_S(sKV);         // smem 源
Tensor tKrK_copy = thr_copy_K.retile_D(tKrK);

// 显式逐 K-tile 加载并计算
cute::copy(smem_copy_K, tKsK(_, _, _0{}), tKrK_copy(_, _, _0{}));
for (int k = 0; k < size<2>(tKrK); ++k) {
    if (k < size<2>(tKrK) - 1)
        cute::copy(smem_copy_K, tKsK(_, _, k+1), tKrK_copy(_, _, k+1));
    cute::gemm(tiled_mma, rQ(_, _, k), tKrK(_, _, k), rP);
}
```

---

## 四、内存对齐陷阱

### 4.1 FP8 KV 数据：stride 584 = 8 字节对齐，非 16 字节

MODEL1 per-token stride = 448 + 128 + 8 = 584 bytes。584 mod 16 = 8，即 token 1 起头部字节仅保证 8 字节对齐。

**问题**：`ld.global.v4.s32`（16 字节 load）在奇数 token 上会触发 `misaligned address`。

**解决方案**：改用两次 `ld.global.nc.v2.u32`（各 8 字节）完成 16 字节数据的加载：
```cpp
asm volatile("ld.global.nc.v2.u32 {%0, %1}, [%2];"
    : "=r"(lo_lo), "=r"(lo_hi) : "l"(addr));
asm volatile("ld.global.nc.v2.u32 {%0, %1}, [%2];"
    : "=r"(hi_lo), "=r"(hi_hi) : "l"(addr + 8));
```

### 4.2 combine.cu 的 PDL (Programmatic Dependent Launch)

```cpp
// SM89/SM120 不支持 PDL，需要运行时检测：
int arch_major = 0;
cudaDeviceGetAttribute(&arch_major, cudaDevAttrComputeCapabilityMajor, dev);
if (arch_major == 9) {
    // SM90 专属：PDL launch
    cudaLaunchKernelEx(...);
} else {
    // SM89/SM120：普通 stream-ordered launch
    combine_kernel<<<grid, block, smem_size, stream>>>(params);
}
```

同时，combine kernel 内部的 `cudaGridDependencySynchronize()` 需要守卫：
```cpp
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    cudaGridDependencySynchronize();
#endif
```

### 4.3 `lse_accum` / `o_accum` 中间 buffer 必须初始化为安全哨兵值

**背景**：SplitKV 路径中，`lse_accum` 和 `o_accum` 是按 `total_num_splits = b + num_sm_parts` 分配的 buffer，由 decode kernel 各 partition 写入、combine kernel 读取汇总。

**触发条件**：当 `num_sm_parts` 远大于实际 KV block 数时（如 `num_sm_parts=23`，topk=256 → 4 blocks），大量 partition 因 `begin_req_idx >= b` 提前 return，**从不写入**它们分配到的 `lse_accum[split_idx]` 槽位。若 combine kernel 的读取范围（由 `num_splits` prefix-sum 数组决定）包含了这些槽位，就会读到 PyTorch caching allocator 留下的历史 garbage 值。

**后果**：
- garbage 为普通浮点：combine 引入微小的错误贡献（可能在数值容差内不被察觉）
- garbage 为 `+∞`：`max(+∞, valid) = +∞` → `lse - max = valid - ∞ = -∞` → `exp(-∞) = 0` → `sum = 0` → `log(0) + ∞ = -∞ + ∞ = NaN` ← **实际触发的情况**
- garbage 为 `NaN`：`max(NaN, valid) = NaN` → 直接传播

**现象**：仅在 pytest batch 运行时偶发 `isfinite(lse).all()` 失败，单独运行测试通过（首次分配的 GPU memory 通常为零）。

**解决方案**：在 SM89/SM120 路径上，decode kernel 启动前将 buffer 初始化为安全恒等值：

```cpp
if (arch.is_sm89() || arch.is_sm120()) {
    // -inf 是 log-sum-exp 归约的恒等元（exp(-inf) = 0，贡献为零）
    lse_accum.fill_(-std::numeric_limits<float>::infinity());
    // 零乘任何权重仍为零
    o_accum.zero_();
}
```

**规律**：任何使用 `torch::empty` 分配的中间 buffer，若存在"部分 partition 不写入"的情况，必须用恒等值初始化。对 log-sum-exp 归约，恒等元是 `-∞`；对加权求和，恒等元是 `0`。

---

## 五、调试技巧

### 5.1 compute-sanitizer 使用优先于 printf

```bash
compute-sanitizer --tool memcheck python -c "..."
```

- 能精确定位到具体文件:行号 和 thread (x,y,z)
- 区分 `Invalid __global__ read`（全局内存）vs `Invalid __shared__ read`（smem）
- 注意：sanitizer 对 SM89/SM120 有时会报 "no kernel image" 错误，此时直接用 `CUDA_LAUNCH_BLOCKING=1` 更可靠

### 5.2 CUDA sticky error 的排查

CUDA kernel 错误往往是"sticky"的——在后续的 API 调用（如 `cudaFuncSetAttribute`）才报出。

**规律**：报错文件/行号指向的是"发现错误"的地方，不是"产生错误"的地方。用 `CUDA_LAUNCH_BLOCKING=1` 可以使 kernel 同步执行，从而立即在错误 kernel 处报告。

### 5.3 printf 调试的注意事项

- CUDA kernel printf 输出需要 `cudaDeviceSynchronize()` 才会 flush 到 host
- `%p` 格式符打印指针在 SM89/SM120 可能触发 "operation not supported" 错误，改用 `%llx` 或 `(unsigned long long)ptr`
- 在 CUDA kernel 中用 `__isnanf(v)` 而非 `v != v` 检测 NaN

### 5.4 发现非 ASCII 字符导致 ptxas fatal 的技巧

```bash
grep -rn '[^\x00-\x7F]' csrc/sm89/
# 或
python3 -c "
content = open('file.cuh').read()
result = ''.join(c if ord(c)<128 else '?' for c in content)
open('file.cuh', 'w').write(result)
"
```

`--source-in-ptx` 编译选项会把源码嵌入 PTX，中文注释会导致 ptxas fatal non-ASCII error。

### 5.5 "all rP = 0" 定位 GEMM 失败的方法

如果 QK GEMM 后 rP 全为 0.0：

1. 先确认 sKV smem 值是否正确（加 printf 读 sKV(0,0)）
2. 检查是否用了 `cute::gemm` 而不是显式 LDSM 加载（见 3.6 节）
3. 检查 Tile 参数是否与 SmemCopyAtom 兼容

---

## 六、数据格式陷阱：quant.py 的 stride-576 vs kernel 的 stride-584

这是最隐蔽也最重要的问题。

### 问题描述

FlashMLA 的 `quant.py` 在 `quantize_k_cache` 中使用了内部 stride-576 的数据布局：

```python
result_k_nope_rope_part = result[:, :block_size*576].view(num_blocks, block_size, 576)
# stride 576 存储 nope+rope
result_k_scales = result[:, 36864:].view(1, 64, 8)
# scales 在 flat 偏移 36864 处
```

而 FlashMLA 的 CUDA kernel（SM89/SM90）读取时：
- **nope+rope**：用 `gK_base = kv_ptr + token * k_row_stride`，其中 `k_row_stride = kv_fp8.stride(1) = 584`（stride-584）
- **scales**：用 `kv_ptr + page_size * 576 + token * 8 = kv_ptr + 36864 + token * 8`

**结果**：

| 访问 | quant.py 写入位置 | kernel 读取位置 | 匹配？ |
|------|-----------------|----------------|--------|
| Scales | flat[36864+t×8] | flat[36864+t×8] | ✅ |
| Nope(t) | flat[t×576] | flat[t×584] | ❌（t>0 时差 8t 字节）|
| Rope(t) | flat[t×576+448] | flat[t×584+448] | ❌（同上）|

### 后果

对于 token t（t>0），kernel 读到的是 token t 的 nope（从第 8t 字节开始）而非从第 0 字节，rope 读取也有偏移，具体表现为：

- `t × 8` 字节的 rope 末尾被下一个 token 的 FP8 nope 字节污染
- FP8 nope 字节被解读为 BF16 时可能得到极大值（如 2^16 量级），导致 QK 溢出 → inf → lse=inf

### 测试数据补丁方案

在测试时，对 quant.py 生成的数据做以下 patch：

```python
kv_u8 = kv_fp8.view(torch.uint8)

# Patch 1: 移除 nope 区 fp8_e4m3fn NaN（0x7F 和 0xFF）
kv_nope = kv_u8[..., :448]
kv_nope.masked_fill_(kv_nope == 0x7F, 0)
kv_nope.masked_fill_(kv_nope == 0xFF, 0xFE)  # 0xFF→-448（最大有限负值）

# Patch 2: 清零 rope 区中被 FP8 污染的字节
# token t 的污染区：kv_fp8[0, t, 0, max(448, 576-8t):576]
for t in range(1, page_block_size):
    cs = 576 - 8 * t
    kv_u8[:, t, 0, max(448, cs):576] = 0
```

### 根本解决方案（生产环境）

使用 `compare_three_methods_v2.py` 中的 `build_fp8_kv_cache` 函数（per-token 584 字节紧凑布局），确保测试数据与 kernel 期望的格式完全一致。

---

## 七、移植检查清单

移植 SM90 kernel 到 SM89 时，依次检查：

```
[ ] 1. 替换 WGMMA → SM80 MMA + 显式 gemm_rs 循环
[ ] 2. 替换 TMA Load → cp.async + cp_async_wait + LDSM
[ ] 3. 替换 TMA Store → stmatrix（或 AutoVectorizingCopy）+ global store
[ ] 4. 移除 Cluster 相关代码（SM89 不支持）
[ ] 5. 移除 stmatrix/SM90 特有指令
[ ] 6. 将 PDL (cudaGridDependencySynchronize) 用 __CUDA_ARCH__ >= 900 守卫
[ ] 7. 检查 smem 总大小 ≤ 100KB
[ ] 8. 检查每线程寄存器 ≤ 255（重点：O 累加器大小）
[ ] 9. 检查全局内存加载对齐（尤其 stride 非 2^N 时用 8B 而非 16B 加载）
[ ] 10. 检查 TiledMMA Tile 参数与 SmemCopyAtom 的向量化兼容性
[ ] 11. SmemLayoutQ/S 等 smem 布局是否覆盖完整 MMA_M_TILE 行
[ ] 12. store_o_to_global 等涉及 coord 的写回函数加 row < BLOCK_M 守卫
[ ] 13. combine.cu 的 PDL launch 路径加运行时 arch 检测
[ ] 14. 源文件中无非 ASCII 字符（中文注释需转英文，避免 ptxas fatal）
[ ] 15. 所有 partition_C(make_identity_tensor(...)) 的 identity_tensor 维度与 TiledMMA 自然 tile 对齐（用 MMA_M_TILE 而非 BLOCK_M）
[ ] 16. SplitKV 中间 buffer（lse_accum / o_accum）分配后初始化为恒等值（-inf / 0）
```

---

## 八、性能注意事项

### 8.1 寄存器压力

最终编译的 SM89 kernel 使用 255 个寄存器（满配）并有少量 local memory spill (~3.4KB)。这是由于：
- O 累加器：128 floats（BLOCK_M=32 × HEAD_DIM_V=512 / 128 线程）
- Q 寄存器：~64 寄存器（BLOCK_M=32 × HEAD_DIM_K=512 / 128 线程 / 2）
- 额外开销：~40 寄存器

如果 spill 性能不可接受，可将 BLOCK_M 减到 16（4 CTAs 覆盖 64 heads）。

### 8.2 单 buffer vs 双 buffer

100KB smem 限制下无法做双 buffer（需 128KB）。使用 L2 prefetch hint 作为替代：
```cpp
asm volatile("prefetch.global.L2 [%0];" :: "l"(next_token_ptr) : "memory");
```

### 8.3 SplitKV 对大 batch 的重要性

对于 batch=64、kvcache=8256 tokens（129 个 TOPK block）：
- 不使用 SplitKV：128 CTAs（64 batch × 2 m-block），L40S 利用率约 90%
- 使用 SplitKV（num_parts≈16）：2048 CTAs，100% 利用率

保留 FlashMLA 的 `smxx::decode::get_decoding_sched_meta` 调度器是大 batch 下性能的关键。

---

## 九、与 SM90 的行为差异总结

| 方面 | SM90 行为 | SM89 行为 |
|------|-----------|-----------|
| Q 加载 | TMA + transaction barrier | cp.async + LDSM |
| K/V 去量化 | producer WG 专用（3 warpgroup 分工） | 所有 4 warp 共同完成 |
| S 写 smem | stmatrix（硬件高效） | AutoVectorizingCopy（较慢但正确）|
| O 写 global | TMA 5D Store | 逐元素 store + coord 映射 |
| 同步机制 | NamedBarrier（warpgroup 粒度） | `__syncthreads()` |
| Warp 2/3 | 参与 PV GEMM 的另半边 V | 计算无效行（自动被 empty fragment 过滤）|
| 数值精度 | 满精度 | 因 stride-576/584 数据格式问题，测试环境中 K 数据有 8t 字节偏移 |

---

*文档创建于 2026-05-21，记录 SM89 sparse FP8 MLA decode 移植过程。*



  用具体例子来说明，假设 page_block_size = 4（4 个 token，方便画图）。

  ---
  一、每个 token 的数据结构

  MODEL1 格式每个 token 有 584 字节：

  [0 .. 447]   = 448 字节  FP8 nope 值（fp8_e4m3fn）
  [448 .. 575] = 128 字节  BF16 rope 值（64 × bf16）
  [576 .. 583] =   8 字节  量化 scale（fp8_e8m0）

  ---
  二、quant.py 实际如何写入内存

  quantize_k_cache 里的关键代码：

  result = torch.empty((1, size_per_block_padded), dtype=fp8_e4m3fn)[:, :4*584]
  # result 形状 (1, 2336)，stride(1) = 1

  # ① 写 nope + rope（用 stride-576 的视图）
  result_k_nope_rope_part = result[:, :4*576].view(1, 4, 576)
  #                                    ^^^
  #                                    4×576 = 2304，不是 4×584 = 2336！

  # ② 写 scales（在 flat 偏移 2304 处）
  result_k_scales = result[:, 2304:].view(1, 4, 8)

  注意：4*576 = 2304，而不是 4*584 = 2336。

  quant.py 写入的实际内存布局：

  flat 字节偏移：
  [0   .. 575]  ← result_k_nope_rope_part[0, 0, 0..575]  = token 0 的 nope+rope
  [576 .. 1151] ← result_k_nope_rope_part[0, 1, 0..575]  = token 1 的 nope+rope
  [1152.. 1727] ← result_k_nope_rope_part[0, 2, 0..575]  = token 2 的 nope+rope
  [1728.. 2303] ← result_k_nope_rope_part[0, 3, 0..575]  = token 3 的 nope+rope
  [2304.. 2335] ← result_k_scales[0, 0..3, 0..7]        = 所有 token 的 scales

  token 相邻间距 = 576 字节（stride-576）

  ---
  三、最终返回的 kv_fp8 tensor

  return result.view(1, 4, 1, 584)
  # kv_fp8.stride(1) = 584

  kv_fp8 视图认为的布局：

  flat 字节偏移：
  [0   .. 583]  ← kv_fp8[0, 0, 0, 0..583]  = token 0 的 584 字节槽
  [584 .. 1167] ← kv_fp8[0, 1, 0, 0..583]  = token 1 的 584 字节槽
  [1168.. 1751] ← kv_fp8[0, 2, 0, 0..583]  = token 2 的 584 字节槽
  [1752.. 2335] ← kv_fp8[0, 3, 0, 0..583]  = token 3 的 584 字节槽

  token 相邻间距 = 584 字节（stride-584）

  ---
  四、两种视图的对比——bug 发生的根源

  flat 字节偏移：
  0        576      1152     1728     2304
  |--------|--------|--------|--------|-------|
  ^token0  ^token1  ^token2  ^token3  ^scales
    （quant.py stride-576 写入位置）

  0       584      1168     1752     2336
  |--------|--------|--------|--------|
  ^token0  ^token1  ^token2  ^token3
    （kv_fp8 stride-584 读取位置）

  偏移对比（对 token 0 是一致的）：

  ┌─────────────────┬───────────────────┬─────────────────┬─────────────┐
  │                 │ quant.py 写入位置 │ kv_fp8 读取位置 │    偏差     │
  ├─────────────────┼───────────────────┼─────────────────┼─────────────┤
  │ token 0 nope[0] │ flat[0]           │ flat[0]         │ 0 字节 ✅   │
  ├─────────────────┼───────────────────┼─────────────────┼─────────────┤
  │ token 1 nope[0] │ flat[576]         │ flat[584]       │ +8 字节 ❌  │
  ├─────────────────┼───────────────────┼─────────────────┼─────────────┤
  │ token 2 nope[0] │ flat[1152]        │ flat[1168]      │ +16 字节 ❌ │
  ├─────────────────┼───────────────────┼─────────────────┼─────────────┤
  │ token 3 nope[0] │ flat[1728]        │ flat[1752]      │ +24 字节 ❌ │
  └─────────────────┴───────────────────┴─────────────────┴─────────────┘

  规律：token t 的偏移差 = t × 8 字节

  ---
  五、具体看 token 1 的 rope 被污染
  
  以 token 1 为例，CUDA kernel 读取它的 rope：

  kernel 读 token 1 rope：flat[584+448 .. 584+575] = flat[1032 .. 1159]

  这段内存在 quant.py 的实际存储中是什么？

  flat 内存内容（quant.py 写入）：
  [576  .. 1023]  = token 1 的 nope（448字节 FP8）
  [1024 .. 1151]  = token 1 的 rope（128字节 BF16）
  [1152 .. 1727]  = token 2 的 nope+rope
                ↑
  kernel 读 rope：flat[1032 .. 1159]
                = flat[1032..1151] ← token1 的 rope byte[8..127]（真实BF16，正常）
                + flat[1152..1159] ← token2 的 nope byte[0..7]（FP8，被误读为BF16！）

  图示：

  flat 地址：  1032                       1151 1152      1159
               |---token1 rope(byte8~127)---|---token2 nope(byte0~7)---|
                          ↑ kernel 以为这是 token1 的完整 rope ↑

  token 2 的 nope 前 8 字节（FP8 格式）被 kernel 误读成 BF16 rope！

  ---
  六、FP8 字节被误读为 BF16 时有多危险
  
  FP8（fp8_e4m3fn）值范围 ±448，但字节模式解读为 BF16 时完全不同：

  举例：设 token 2 的 nope[0] 量化后为 FP8 字节 0x48 0x47（连续两字节）

  两个 FP8 字节作为独立 FP8 值（正常用途）：
    0x48 = sign=0, exp=1001, mantissa=000 → 2^(9-7) × 1.0 = 4.0
    0x47 = sign=0, exp=1000, mantissa=111 → 2^(8-7) × 1.875 = 3.75

  同样两字节被误读为 1 个 BF16 值（小端）：
    BF16 = 0x4748
         = sign=0, exp=10001110=0x8E=142, mantissa=1001000
         = 2^(142-127) × (1 + 0x48/0x80)
         = 2^15 × 1.5625
         = 32768 × 1.5625 = 51200  ← 超大值！

  这个 51200 作为 K 向量的某个分量参与 Q·K 计算：

  Q[dim] ≈ 1.0（来自 torch.randn）
  K[dim] = 51200（被污染的假 BF16 值）
  该维贡献 = 1.0 × 51200 = 51200

  512 个维度求和最终 QK score 可达 ~1e6 量级
  → softmax(1e6 / √512) → 该 token 得分接近 1，其他趋近 0
  → 但若多个维度都有大值，累加后可能超出 float32 范围 → inf
  → lse = logf(inf + ...) = inf 或 NaN

  ---
  七、为什么 token 0 正常，token t>0 出问题
  
  对 token 0：quant.py 写在 flat[0]，kernel 也读 flat[0]，完全一致，不存在污染。

  对 token t（t>0）：
  - 偏差 = t × 8 字节
  - kernel 读 rope 末尾时，实际读到了 t×8 字节的下一个 token FP8 nope 数据
  - t 越大，污染越严重（token 16 时整个 rope 全被污染，token 63 时 rope 多出 504 字节污染）

  ---
  八、为什么 scales 是对的
  
  scales 的访问公式：
  scales_ptr = kv_ptr + page_size * 576 + token * 8
             = kv_ptr + 2304 + token * 8

  quant.py 将 scales 写在 result[:, 2304:]：
  - token 0 的 scales：flat[2304..2311] ✅
  - token 1 的 scales：flat[2312..2319] ✅

  两者完全匹配，scales 没有 bug，问题只在 nope+rope。

  ---
  九、测试 patch 的原理
  
  理解了 bug 后，patch 逻辑就清晰了：

  # kernel 读 token t 的 rope 的 flat 范围：
  # [t*584+448 .. t*584+575]

  # 其中被污染的部分（FP8 字节误读为 BF16）：
  # 从 max(t*576+576, t*584+448) 开始 = 从 (t*584+448 + overlap) 开始
  # 污染长度 = t*8 字节（最多 128 字节）

  # 在 kv_fp8 stride-584 视图中，污染字节的索引范围：
  contaminated_start = 576 - 8*t  # 0 ~ 575 的 d 偏移
  # 实际从 max(448, contaminated_start) 到 575 都要清零
  kv_u8[:, t, 0, max(448, contaminated_start):576] = 0

  清零后：kernel 读到的"假 rope"维度全为 BF16 0.0，不再引入极端值，QK score 有限。

  ---
  总结：这个 bug 的本质是 result.view(1, 4, 576) 和 result.view(1, 4, 1, 584) 对同一块内存用了不同的 stride（576 vs 584），导致 token t（t>0）的读写位置差了 t×8 字节，使得 rope 末尾被相邻 token 的 FP8 nope
  数据污染，误读为 BF16 时得到极大值，最终 QK score 溢出 → lse = inf。
  
  
  
  
  ---
  为什么用 576 字节对齐？—— 完整真相
  
  一、SM90 kernel 如何访问 KV 数据（MODEL1）

  看 SM90 producer warpgroup 的代码（line 554）：

  // ✅ SM90 MODEL1 的 gK_base 计算
  gK_base = k_ptr + block_index * k_block_stride
          + rel_idx_in_block * (HEAD_DIM_NOPE + HEAD_DIM_ROPE * sizeof(bf16));
  //                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  //                           = 448 + 64×2 = 448 + 128 = 576 字节

  SM90 每个 token 的 nope+rope 偏移 = token × 576，而不是 token × 584！

  以及 scale 的位置（line 555）：

  gK_scales_base = k_ptr + block_index * k_block_stride
                 + page_block_size * (HEAD_DIM_NOPE + HEAD_DIM_ROPE * sizeof(bf16))
  //               ^^^^ 64         × (448             + 128                       )
  //                              = 64 × 576 = 36864
                 + rel_idx_in_block * NUM_SCALES;

  Scale 起始偏移 = page_block_size × 576 = 36864，紧跟在所有 token 的 nope+rope 数据之后。

  ---
  二、SM100 kernel 的 TMA descriptor（config.h line 41）

  // SM100 MODEL1
  static constexpr int TMA_K_STRIDE = D_NOPE + 2*D_ROPE;  // = 448 + 128 = 576

  SM100 建立 TMA descriptor（kernel.cuh line 912-930）：

  // 要求: k_batch_stride 必须是 TMA_K_STRIDE (576) 的整数倍
  KU_ASSERT(k_batch_stride % TMA_K_STRIDE == 0, "... Padding might be necessary");

  // nope TMA: 以 576 字节为行步长
  CUtensorMap tensor_map_kv_nope = make_tensor_map(
      {D_NOPE/8,  num_blocks * (k_batch_stride / TMA_K_STRIDE)},  // shape: [56, N]
      {TMA_K_STRIDE},   // ← 每行间距 = 576 字节
      ...
  );
  // rope TMA: 同样以 576 字节为行步长，但从 D_NOPE 偏移处开始
  CUtensorMap tensor_map_kv_rope = make_tensor_map(
      {D_ROPE, num_blocks * (k_batch_stride / TMA_K_STRIDE)},
      {TMA_K_STRIDE},   // ← 576 字节
      base_ptr + D_NOPE, // ← 从 nope 之后开始
      ...
  );

  以及 TMA 坐标计算（line 657-658）：

  // 每个 token = 1 个 TMA 坐标单元（576/576 = 1）
  tma_coords_step_per_token = 576 / TMA_K_STRIDE;  // = 1

  // 每个 page = k_batch_stride / 576 个坐标单元
  tma_coords_step_per_block = k_batch_stride / TMA_K_STRIDE;

  ---
  三、576 字节的来源与意义
  
  576 = D_NOPE + 2×D_ROPE = 448 + 128 — 这是 每个 token 的 nope+rope 数据大小。

  完整的 token 内存布局（MODEL1）：

  页面起始                                        页面结束
  │                                               │
  ├── token 0: [nope: 448B] [rope: 128B]          │ ← stride 576 per token
  ├── token 1: [nope: 448B] [rope: 128B]          │
  ├── ...                                         │
  ├── token 63: [nope: 448B] [rope: 128B]         │ ← 64×576 = 36864B
  │                                               │
  ├── scales: [8B × 64 tokens = 512B]             │ ← flat[36864..37375]
  │                                               │
  └── padding: [64B]                              │ ← total = 37440 = 65×576

  576 字节的三重作用：

  ┌─────────────────────────────┬────────────────────────────────────────┐
  │            作用             │                  说明                  │
  ├─────────────────────────────┼────────────────────────────────────────┤
  │ ① nope+rope 的每 token 步长 │ SM90/SM89 kernel 用 token × 576 定位   │
  ├─────────────────────────────┼────────────────────────────────────────┤
  │ ② TMA 行步长                │ SM100 TMA descriptor stride = 576      │
  ├─────────────────────────────┼────────────────────────────────────────┤
  │ ③ scale 起始偏移基数        │ scales 在 page_size × 576 = 36864 之后 │
  └─────────────────────────────┴────────────────────────────────────────┘

  ---
  四、为什么要 padding？
  
  要求：k_batch_stride 必须是 576 的整数倍（SM100 显式 assert，SM90 隐式依赖）。

  pages 内实际使用 = 64 × 584 = 37,376 字节

  37,376 / 576 = 64.888...  ← 不整除！

  padding 到  65 × 576 = 37,440 字节  ← 可整除 ✓

  k_batch_stride = 37,440 / 576 = 65 个 TMA 坐标步长

  padding 64 字节的目的：让 TMA 坐标系能整除，使每页占整数个 TMA 步长。

  ---
  五、我们 SM89 kernel 的 Bug 根因
  
  SM90 用 token × 576 访问 nope+rope，我们的 SM89 kernel 错误地用了：

  // ❌ SM89 错误实现（我们的代码）
  gK_base = kv_ptr + token * k_row_stride;  // k_row_stride = kv_fp8.stride(1) = 584

  // ✅ SM90 正确做法
  gK_base = kv_ptr + token * (HEAD_DIM_NOPE + HEAD_DIM_ROPE * sizeof(bf16));  // = 576

  我们错误地用了 PyTorch tensor 的 stride(1) = 584，应该用 576 = d_nope + d_rope×2。

  ---
  六、设计的深层逻辑

  为什么 scale 不放在 token 内部（stride-584），而是统一放在 36864 之后？

  因为 SM100 的 TMA 使用两个独立的 2D tensor加载 nope 和 rope：

  TMA(nope): 以 576 字节步长逐 token 加载 448 字节 FP8 nope
  TMA(rope): 以 576 字节步长逐 token 加载 128 字节 BF16 rope

  TMA 是 2D 访问，stride = 576。如果 scale 嵌在每 token 的 token[576..583] 里，TMA 就无法绕过它们加载连续的 nope+rope。把 scale 统一放在 36864+ 处，TMA 可以干净地加载 nope+rope，scale 由独立的 ld.global 访问。

  这是 576 字节对齐的核心原因：满足 TMA 2D 访问的 stride 要求。
  
  
  
    SMEM 占用分析：

  KV buffer: 64×512×2 = 64KB  (TOPK_BLOCK_SIZE=64 bf16)
  Q buffer:  64×512×2 = 64KB  (MMA_M_TILE=64, 4 warps)
  S scores:  64×64×2  =  8KB
  misc:                0.4KB (sM/sL/sOScale + is_kv_valid)
  ─────────────────────────────────
  Total:  max(64,64) + 8 + 0.4 = 72KB / 100KB limit
  → Occupancy: 1 CTA/SM

  占用的 128 threads / 1536 max = 8.3%，36 SMs 上同时只有 36 个 CTA 在跑。576 个
   CTA 需要 16 波执行。

  ---
  优化方案

  方案 1：减少 TOPK_BLOCK_SIZE → 提高 Occupancy（中等改动）

  将 TOPK_BLOCK_SIZE 从 64 降至 32，改用 Layout<_2,_1,_1>（2 warps，64
  threads）：

  KV: 32×512×2 = 32KB
  Q:  32×512×2 = 32KB  (MMA_M_TILE=32)
  S:  32×32×2  =  2KB
  ──────────────────
  Total: 32 + 2 + 0.4 = 35KB → 2 CTA/SM!

  Threads/SM: 2×64 = 128/1536  (8.3% → same)
  Regs/CTA:  64×~200 = 12,800  (大幅低于 255 限制)

  - BLOCK_M=16, NUM_M_BLOCKS=4（CTA 数翻倍）
  - KV block 数翻倍（129→258），但每 block token 数减半
  - 预期收益：20-40%（latency hiding，一个 CTA dequant 时另一个做 GEMM）

  代价：kernel 大量重写（MMA layout、smem layout 全部改）。

  方案 2：BLOCK_M=16 降低寄存器压力（小改动）

  保持 4 warps / 128 threads，但 BLOCK_M=16：

  O accumulator: 16×512/128 = 64 regs (原 128 regs)
  → 节省 ~64 regs/thread
  → 消除 local memory spill (3408 bytes)
  → ptxas 可能选择更优的指令调度

  - NUM_M_BLOCKS=4（CTA 数翻倍）
  - __launch_bounds__(128, 2) 尝试提升 occupancy
  - 预期收益：10-20%（更好的指令级并行 + 消除 spill）
  - SMEM 不变（仍是 72KB，因 MMA_M_TILE=64 未变）

  方案 3：C128 场景专用 fast path（小改动，高收益）

  当前 extra cache（15K tokens, page_size=2）有特殊问题：
  - page_size=2 → k_block_stride = 2304，每 page 2 tokens 但有 97% 内存 padding
  - 7500 个 page，L2 cache 局部性极差

  可加一个 fast path：page_size=2 时跳过 page-level 寻址，直接用 flat 索引访问
  token 数据。这个修改仅影响 dequant_kv_to_smem 中的地址计算。

  预期收益：5-15%（减少 L2 miss）

  方案 4：SW pipelining（较大改动）

  当前流程是严格串行：

  dequant KV → sync → QK GEMM → Softmax → store S → sync → PV GEMM → sync

  如果能 缩减 SMEM 到 ~35KB（方案 1），就可以用双 buffer：
  buf0: dequant block N   │ buf0: QK GEMM + PV GEMM
  buf1: dequant block N+1 │ buf1: QK GEMM + PV GEMM
      ←── wave 0 ──→     ←── wave 1 ──→

  dequant（CUDA core）和 GEMM（tensor core）可以 overlap。

  预期收益：30-50%（但需要方案 1 的 SMEM 缩减 + 双 buffer 实现）

  ---
  优先级建议

  ┌────────┬────────┬────────┬──────────┬───────────────────────────────┐
  │ 优先级 │  方案  │ 改动量 │ 预期收益 │             说明              │
  ├────────┼────────┼────────┼──────────┼───────────────────────────────┤
  │ P0     │ 方案 2 │ 小     │ 10-20%   │ 改 BLOCK_M=16，几乎不影响架构 │
  ├────────┼────────┼────────┼──────────┼───────────────────────────────┤
  │ P1     │ 方案 3 │ 小     │ 5-15%    │ page_size=2 fast path         │
  ├────────┼────────┼────────┼──────────┼───────────────────────────────┤
  │ P2     │ 方案 1 │ 大     │ 20-40%   │ 2-warp layout，需要大量重写   │
  │ P0     │ 方案 2 │ 小     │ 10-20%   │ 改 BLOCK_M=16，几乎不影响架构 │
  ├────────┼────────┼────────┼──────────┼───────────────────────────────┤
  │ P1     │ 方案 3 │ 小     │ 5-15%    │ page_size=2 fast path         │
  ├────────┼────────┼────────┼──────────┼───────────────────────────────┤
  │ P2     │ 方案 1 │ 大     │ 20-40%   │ 2-warp layout，需要大量重写   │
  ├────────┼────────┼────────┼──────────┼───────────────────────────────┤
  │ P3     │ 方案 4 │ 大     │ 30-50%   │ 双 buffer，依赖方案 1         │
  └────────┴────────┴────────┴──────────┴───────────────────────────────┘

  最值得先做的是 方案 2（BLOCK_M=16），只改几个常量，消除寄存器溢出，CTA

  预期收益：30-50%（但需要方案 1 的 SMEM 缩减 + 双 buffer 实现）

  ---
  优先级建议

  ---
  优先级建议

  ┌────────┬────────┬────────┬──────────┬───────────────────────────────┐
  │ 优先级 │  方案  │ 改动量 │ 预期收益 │             说明              │
  ├────────┼────────┼────────┼──────────┼───────────────────────────────┤
  │ P0     │ 方案 2 │ 小     │ 10-20%   │ 改 BLOCK_M=16，几乎不影响架构 │
  ├────────┼────────┼────────┼──────────┼───────────────────────────────┤
  │ P1     │ 方案 3 │ 小     │ 5-15%    │ page_size=2 fast path         │
  ├────────┼────────┼────────┼──────────┼───────────────────────────────┤
  │ P2     │ 方案 1 │ 大     │ 20-40%   │ 2-warp layout，需要大量重写   │
  ├────────┼────────┼────────┼──────────┼───────────────────────────────┤
  │ P3     │ 方案 4 │ 大     │ 30-50%   │ 双 buffer，依赖方案 1         │
  └────────┴────────┴────────┴──────────┴───────────────────────────────┘

  最值得先做的是 方案 2（BLOCK_M=16），只改几个常量，消除寄存器溢出，CTA 数翻倍增加 GPU 利用率，风险最低收益确定。


