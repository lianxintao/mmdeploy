"""
高效的 FP8 专用 MoE 内核实现
只支持块级量化，分块计算，内存效率高
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Dict, Any, List

# 导入 FP8 量化相关函数
from sglang.srt.layers.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
    per_token_group_quant_fp8,
)
from sglang.srt.utils import is_cuda, is_hip

_is_cuda = is_cuda()
_is_hip = is_hip()

if _is_cuda:
    from sgl_kernel import sgl_per_token_group_quant_fp8

# 常量定义
padding_size = 128


@triton.jit
def fp8_moe_gate_kernel(
    # 输入矩阵指针 (必须是 FP8)
    a_ptr,           # 输入激活 [M, K] - FP8
    w1_ptr,          # 第一层权重 [E, N1, K] - FP8，N1 = gate_size + up_size
    b1_ptr,          # 第一层偏置 [E, N1] (可选)
    intermediate_ptr, # 中间结果 [M, topk, N1//2] (SiLU后的结果)
    # FP8 量化参数 (必须提供)
    a_scale_ptr,     # 激活量化缩放因子 [M, num_blocks_k]
    w1_scale_ptr,    # W1权重量化缩放因子 [E, num_blocks_n, num_blocks_k]
    # 路由相关
    topk_weights_ptr,      # topk权重 [M, topk]
    sorted_token_ids_ptr,  # 排序后的token ID
    expert_ids_ptr,        # 专家ID
    num_tokens_post_padded_ptr,
    # 矩阵维度
    N1: tl.constexpr,      # w1的输出维度 (gate_size + up_size)
    K: tl.constexpr,       # 输入维度
    EM,                    # 有效的M维度
    num_valid_tokens,
    # 步长参数
    stride_am, stride_ak,
    stride_w1e, stride_w1k, stride_w1n,
    stride_b1e, stride_b1n,
    stride_im, stride_in,
    stride_asm, stride_ask,  # A量化缩放因子步长
    stride_w1se, stride_w1sk, stride_w1sn,  # W1量化缩放因子步长
    # 块大小参数
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.dtype,
    HAS_BIAS1: tl.constexpr,
    # FP8 块量化参数
    group_n: tl.constexpr,  # 块量化的N维度块大小
    group_k: tl.constexpr,  # 块量化的K维度块大小
):
    """
    高效的 FP8 MoE 第一阶段内核：计算 silu(A @ W1_gate) * (A @ W1_up)
    分块计算，每次处理 (BLOCK_SIZE_M, BLOCK_SIZE_N) 的输出块
    """
    # 程序ID映射
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N1 // 2, BLOCK_SIZE_N)  # 输出维度是 N1//2
    
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # 检查有效性
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    # Token 索引设置
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    # 专家ID
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # 写入零输出
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        i_ptrs = intermediate_ptr + stride_im * offs_token[:, None] + stride_in * offs_cn[None, :]
        i_mask = token_mask[:, None] & (offs_cn[None, :] < N1 // 2)
        tl.store(i_ptrs, accumulator, mask=i_mask)
        return

    # ====== 计算输出块的列索引 ======
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # gate 和 up 的列索引
    offs_n_gate = offs_n  # gate: [0, N1//2)
    offs_n_up = offs_n + N1 // 2  # up: [N1//2, N1)
    
    # 初始化累加器：gate 和 up 分别计算
    gate_accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    up_accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # 预计算量化缩放因子指针
    a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
    offs_w1sn_gate = offs_n_gate // group_n
    offs_w1sn_up = offs_n_up // group_n
    w1_gate_scale_ptrs = (
        w1_scale_ptr + off_experts * stride_w1se + offs_w1sn_gate * stride_w1sn
    )
    w1_up_scale_ptrs = (
        w1_scale_ptr + off_experts * stride_w1se + offs_w1sn_up * stride_w1sn
    )
    
    # K维度循环
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        offs_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        k_mask = offs_k < K
        
        # 加载A块
        a_ptrs = (
            a_ptr
            + (offs_token // top_k)[:, None] * stride_am
            + offs_k[None, :] * stride_ak
        )
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        
        # 加载W1_gate块
        w1_gate_ptrs = (
            w1_ptr
            + off_experts * stride_w1e
            + offs_k[:, None] * stride_w1k
            + offs_n_gate[None, :] * stride_w1n
        )
        w1_gate = tl.load(
            w1_gate_ptrs,
            mask=k_mask[:, None] & (offs_n_gate[None, :] < N1 // 2),
            other=0.0,
        )
        
        # 加载W1_up块
        w1_up_ptrs = (
            w1_ptr
            + off_experts * stride_w1e
            + offs_k[:, None] * stride_w1k
            + offs_n_up[None, :] * stride_w1n
        )
        w1_up = tl.load(
            w1_up_ptrs,
            mask=k_mask[:, None] & (offs_n_up[None, :] < N1),
            other=0.0,
        )
        
        # FP8 块量化缩放
        k_group = (k * BLOCK_SIZE_K) // group_k
        a_scale = tl.load(
            a_scale_ptrs + k_group * stride_ask, mask=token_mask, other=0.0
        )
        w1_gate_scale = tl.load(w1_gate_scale_ptrs + k_group * stride_w1sk)
        w1_up_scale = tl.load(w1_up_scale_ptrs + k_group * stride_w1sk)
        
        # 计算 A @ W1_gate 和 A @ W1_up，应用 FP8 缩放
        gate_accumulator += tl.dot(a, w1_gate) * a_scale[:, None] * w1_gate_scale[None, :]
        up_accumulator += tl.dot(a, w1_up) * a_scale[:, None] * w1_up_scale[None, :]
    
    # 添加偏置（如果存在）
    if HAS_BIAS1:
        bias1_gate = tl.load(
            b1_ptr + off_experts * stride_b1e + offs_n_gate * stride_b1n,
            mask=offs_n_gate < N1 // 2,
            other=0.0,
        )
        bias1_up = tl.load(
            b1_ptr + off_experts * stride_b1e + offs_n_up * stride_b1n,
            mask=offs_n_up < N1,
            other=0.0,
        )
        gate_accumulator += bias1_gate[None, :]
        up_accumulator += bias1_up[None, :]
    
    # ====== SiLU激活和门控乘法 ======
    # 应用SiLU激活到gate: SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    gate_silu = gate_accumulator / (1.0 + tl.exp(-gate_accumulator))
    
    # 门控乘法: silu(A @ W1_gate) * (A @ W1_up)
    activated_result = gate_silu * up_accumulator
    
    # 应用topk权重（如果需要）
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        activated_result *= moe_weight[:, None]

    activated_result = activated_result.to(compute_type)
    
    # ====== 写回结果 ======
    i_ptrs = intermediate_ptr + stride_im * offs_token[:, None] + stride_in * offs_n[None, :]
    i_mask = token_mask[:, None] & (offs_n[None, :] < N1 // 2)
    tl.store(i_ptrs, activated_result, mask=i_mask)


@triton.jit
def fp8_moe_final_kernel(
    # 输入矩阵指针 (必须是 FP8)
    intermediate_ptr, # 中间结果 [M, topk, N1//2] - FP8
    w2_ptr,          # 第二层权重 [E, N2, N1//2] - FP8
    b2_ptr,          # 第二层偏置 [E, N2] (可选)
    c_ptr,           # 输出 [M, topk, N2]
    # FP8 量化参数 (必须提供)
    a2_scale_ptr,    # 中间激活量化缩放因子 [M, num_blocks_k]
    w2_scale_ptr,    # W2权重量化缩放因子 [E, num_blocks_n, num_blocks_k]
    # 路由相关
    sorted_token_ids_ptr,  # 排序后的token ID
    expert_ids_ptr,        # 专家ID
    num_tokens_post_padded_ptr,
    # 矩阵维度
    N1_half: tl.constexpr, # w1的输出维度的一半
    N2: tl.constexpr,      # w2的输出维度（最终输出维度）
    EM,                    # 有效的M维度
    num_valid_tokens,
    # 步长参数
    stride_im, stride_in,
    stride_w2e, stride_w2k, stride_w2n,
    stride_b2e, stride_b2n,
    stride_cm, stride_cn,
    stride_a2sm, stride_a2sk,  # A2量化缩放因子步长
    stride_w2se, stride_w2sk, stride_w2sn,  # W2量化缩放因子步长
    # 块大小参数
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.dtype,
    HAS_BIAS2: tl.constexpr,
    # FP8 块量化参数
    group_n: tl.constexpr,  # 块量化的N维度块大小
    group_k: tl.constexpr,  # 块量化的K维度块大小
):
    """
    高效的 FP8 MoE 第二阶段内核：计算 intermediate @ W2 + bias2
    """
    # 程序ID映射
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N2, BLOCK_SIZE_N)
    
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # 检查有效性
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    # Token 索引设置
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    # 专家ID
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # 写入零输出
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N2)
        tl.store(c_ptrs, accumulator, mask=c_mask)
        return

    # ====== 计算输出块的列索引 ======
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # 初始化累加器
    final_accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # 预计算量化缩放因子指针
    a2_scale_ptrs = a2_scale_ptr + offs_token * stride_a2sm
    offs_w2sn = offs_n // group_n
    w2_scale_ptrs = (
        w2_scale_ptr + off_experts * stride_w2se + offs_w2sn * stride_w2sn
    )
    
    # K维度循环
    for k in range(0, tl.cdiv(N1_half, BLOCK_SIZE_K)):
        offs_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        k_mask = offs_k < N1_half
        
        # 加载intermediate块
        i_ptrs = (
            intermediate_ptr
            + offs_token[:, None] * stride_im
            + offs_k[None, :] * stride_in
        )
        intermediate = tl.load(
            i_ptrs,
            mask=token_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        
        # 加载W2块
        w2_ptrs = (
            w2_ptr
            + off_experts * stride_w2e
            + offs_k[:, None] * stride_w2k
            + offs_n[None, :] * stride_w2n
        )
        w2 = tl.load(
            w2_ptrs,
            mask=k_mask[:, None] & (offs_n[None, :] < N2),
            other=0.0,
        )
        
        # FP8 块量化缩放
        k_group = (k * BLOCK_SIZE_K) // group_k
        a2_scale = tl.load(
            a2_scale_ptrs + k_group * stride_a2sk, mask=token_mask, other=0.0
        )
        w2_scale = tl.load(w2_scale_ptrs + k_group * stride_w2sk)
        
        # 计算 intermediate @ W2，应用 FP8 缩放
        final_accumulator += tl.dot(intermediate, w2) * a2_scale[:, None] * w2_scale[None, :]
    
    # 添加偏置（如果存在）
    if HAS_BIAS2:
        bias2 = tl.load(
            b2_ptr + off_experts * stride_b2e + offs_n * stride_b2n,
            mask=offs_n < N2,
            other=0.0,
        )
        final_accumulator += bias2[None, :]
    
    final_accumulator = final_accumulator.to(compute_type)
    
    # ====== 写回最终结果 ======
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_n[None, :]
    c_mask = token_mask[:, None] & (offs_n[None, :] < N2)
    tl.store(c_ptrs, final_accumulator, mask=c_mask)


def invoke_fp8_moe_kernel(
    A: torch.Tensor,           # 输入 [M, K]
    W1: torch.Tensor,          # 第一层权重 [E, N1, K] - FP8
    W2: torch.Tensor,          # 第二层权重 [E, N2, N1//2] - FP8
    C: torch.Tensor,           # 输出 [M, topk, N2]
    bias1: Optional[torch.Tensor] = None,  # 第一层偏置 [E, N1]
    bias2: Optional[torch.Tensor] = None,  # 第二层偏置 [E, N2]
    # FP8 量化相关参数 (必须提供)
    A_scale: torch.Tensor = None,  # 输入激活缩放因子
    W1_scale: torch.Tensor = None,  # W1权重缩放因子
    W2_scale: torch.Tensor = None,  # W2权重缩放因子
    A2_scale: torch.Tensor = None,  # 中间激活缩放因子
    # 路由参数
    topk_weights: torch.Tensor = None,
    topk_ids: torch.Tensor = None,
    sorted_token_ids: torch.Tensor = None,
    expert_ids: torch.Tensor = None,
    num_tokens_post_padded: torch.Tensor = None,
    mul_routed_weight: bool = True,
    top_k: int = 1,
    config: Dict[str, Any] = None,
    compute_type: tl.dtype = tl.float16,
    # FP8 块量化参数 (必须提供)
    block_shape: List[int] = None,
) -> None:
    """
    调用高效的 FP8 专用 MoE 内核
    
    Args:
        A: 输入张量 [M, K] - 必须已经量化为 FP8
        W1: 第一层权重 [E, N1, K] - 必须已经量化为 FP8
        W2: 第二层权重 [E, N2, N1//2] - 必须已经量化为 FP8
        C: 输出张量 [M, topk, N2]
        bias1, bias2: 偏置项（可选）
        A_scale, W1_scale, W2_scale, A2_scale: FP8 量化缩放因子（必须提供）
        block_shape: [block_n, block_k] 块量化参数（必须提供）
        其他参数: MoE 路由相关参数
    """
    
    assert A.dtype == torch.float8_e4m3fn, "A must be FP8"
    assert W1.dtype == torch.float8_e4m3fn, "W1 must be FP8"
    assert W2.dtype == torch.float8_e4m3fn, "W2 must be FP8"
    assert A_scale is not None, "A_scale is required for FP8"
    assert W1_scale is not None, "W1_scale is required for FP8"
    assert W2_scale is not None, "W2_scale is required for FP8"
    assert A2_scale is not None, "A2_scale is required for FP8"
    assert block_shape is not None and len(block_shape) == 2, "block_shape [block_n, block_k] is required"
    
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1
    
    # 获取维度
    M = A.shape[0]
    K = A.shape[1]
    E, N1, _ = W1.shape
    _, N2, N1_half = W2.shape
    
    # 验证维度一致性
    assert N1_half == N1 // 2, f"W2的输入维度{N1_half}应该等于W1输出维度的一半{N1//2}"
    assert W1.shape[2] == K, f"W1的K维度{W1.shape[2]}应该等于输入K维度{K}"
    
    # 获取块量化参数
    block_n, block_k = block_shape
    
    # 默认配置
    if config is None:
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "num_warps": 4,
            "num_stages": 2,
        }
    
    # 创建中间缓存 - 存储SiLU激活后的结果
    intermediate_cache = torch.empty(
        (M, top_k, N1_half),
        device=A.device,
        dtype=A.dtype,  # 保持 FP8 类型
    )
    
    # 第一步：计算W1并应用SiLU + 门控
    grid1 = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(N1_half, META["BLOCK_SIZE_N"]),
    )
    
    fp8_moe_gate_kernel[grid1](
        A, W1, bias1, intermediate_cache,
        A_scale, W1_scale,
        topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
        N1=N1,
        K=K,
        EM=sorted_token_ids.shape[0],
        num_valid_tokens=topk_ids.numel(),
        stride_am=A.stride(0),
        stride_ak=A.stride(1),
        stride_w1e=W1.stride(0),
        stride_w1k=W1.stride(2),
        stride_w1n=W1.stride(1),
        stride_b1e=bias1.stride(0) if bias1 is not None else 0,
        stride_b1n=bias1.stride(1) if bias1 is not None else 0,
        stride_im=intermediate_cache.stride(0),
        stride_in=intermediate_cache.stride(2),
        stride_asm=A_scale.stride(0) if A_scale.ndim == 2 else 0,
        stride_ask=A_scale.stride(1) if A_scale.ndim == 2 else 0,
        stride_w1se=W1_scale.stride(0),
        stride_w1sk=W1_scale.stride(2),
        stride_w1sn=W1_scale.stride(1),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        HAS_BIAS1=bias1 is not None,
        group_n=block_n,
        group_k=block_k,
        **config,
    )
    
    # 第二步：计算W2
    grid2 = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(N2, META["BLOCK_SIZE_N"]),
    )
    
    fp8_moe_final_kernel[grid2](
        intermediate_cache, W2, bias2, C,
        A2_scale, W2_scale,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        N1_half=N1_half,
        N2=N2,
        EM=sorted_token_ids.shape[0],
        num_valid_tokens=topk_ids.numel(),
        stride_im=intermediate_cache.stride(0),
        stride_in=intermediate_cache.stride(2),
        stride_w2e=W2.stride(0),
        stride_w2k=W2.stride(2),
        stride_w2n=W2.stride(1),
        stride_b2e=bias2.stride(0) if bias2 is not None else 0,
        stride_b2n=bias2.stride(1) if bias2 is not None else 0,
        stride_cm=C.stride(1),
        stride_cn=C.stride(2),
        stride_a2sm=A2_scale.stride(0) if A2_scale.ndim == 2 else 0,
        stride_a2sk=A2_scale.stride(1) if A2_scale.ndim == 2 else 0,
        stride_w2se=W2_scale.stride(0),
        stride_w2sk=W2_scale.stride(2),
        stride_w2sn=W2_scale.stride(1),
        top_k=top_k,
        compute_type=compute_type,
        HAS_BIAS2=bias2 is not None,
        group_n=block_n,
        group_k=block_k,
        **config,
    )


def fp8_moe_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    # FP8 量化参数 (必须提供)
    w1_scale: torch.Tensor = None,
    w2_scale: torch.Tensor = None,
    a1_scale: torch.Tensor = None,
    a2_scale: torch.Tensor = None,
    block_shape: List[int] = None,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    compute_type: tl.dtype = tl.float16,
    config: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    使用高效 FP8 专用内核的 MoE 实现
    """
    # 检查约束
    assert activation == "silu", "FP8 MoE内核只支持SiLU激活"
    assert w1_scale is not None, "FP8量化需要W1缩放因子"
    assert w2_scale is not None, "FP8量化需要W2缩放因子"
    assert a1_scale is not None, "FP8量化需要A1缩放因子"
    assert a2_scale is not None, "FP8量化需要A2缩放因子"
    assert block_shape is not None and len(block_shape) == 2, "FP8量化需要指定block_shape [block_n, block_k]"
    
    # 输入量化为 FP8
    block_n, block_k = block_shape
    if _is_cuda:
        hidden_states_fp8, a1_scale = sglang_per_token_group_quant_fp8(hidden_states, block_k)
    else:
        hidden_states_fp8, a1_scale = per_token_group_quant_fp8(hidden_states, block_k)
    
    num_tokens, _ = hidden_states.shape
    E, N1, _ = w1.shape
    _, N2, _ = w2.shape
    
    # 输出张量
    if inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty(
            (num_tokens, N2), device=hidden_states.device, dtype=hidden_states.dtype
        )
    
    # 使用 moe_align_block_size 处理路由
    from .fused_moe import moe_align_block_size
    
    (
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    ) = moe_align_block_size(topk_ids, config["BLOCK_SIZE_M"], E)
    
    # 分块处理
    max_workspace = out_hidden_states.numel() * out_hidden_states.element_size()
    max_workspace //= config.get("BLOCK_SIZE_M", 64)
    
    for chunk_idx in range(0, topk_ids.shape[0], max_workspace):
        begin_chunk_idx = chunk_idx
        end_chunk_idx = min(chunk_idx + max_workspace, topk_ids.shape[0])
        tokens_in_chunk = end_chunk_idx - begin_chunk_idx
        
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]
        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_hidden_states = hidden_states_fp8[begin_chunk_idx:end_chunk_idx]
        
        # 为当前块创建输出缓存
        output_cache = torch.empty(
            (tokens_in_chunk, topk_ids.shape[1], N2),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        
        # 调用高效的FP8内核
        invoke_fp8_moe_kernel(
            curr_hidden_states,
            w1, w2, output_cache,
            b1, b2,
            a1_scale, w1_scale, w2_scale, a2_scale,
            curr_topk_weights, curr_topk_ids,
            sorted_token_ids, expert_ids, num_tokens_post_padded,
            apply_router_weight_on_input,
            topk_ids.shape[1],
            config, compute_type,
            block_shape,
        )
        
        # 合并topk结果
        if topk_ids.shape[1] == 1:
            # 单个专家的情况
            out_hidden_states[begin_chunk_idx:end_chunk_idx] = output_cache.squeeze(1)
        else:
            # 多个专家需要求和
            out_hidden_states[begin_chunk_idx:end_chunk_idx] = output_cache.sum(dim=1)
    
    return out_hidden_states
