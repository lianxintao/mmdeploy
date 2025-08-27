"""
Merged MoE Triton Kernel - 将三个内核合并成一个
将 invoke_fused_moe_kernel + silu_and_mul + invoke_fused_moe_kernel 合并为单个内核
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Dict, Any, List


@triton.jit
def merged_moe_gate_kernel(
    # 输入矩阵指针
    a_ptr,           # 输入激活 [M, K]
    w1_ptr,          # 第一层权重 [E, N1, K] 
    b1_ptr,          # 第一层偏置 [E, N1] (可选)
    intermediate_ptr, # 中间结果 [M, topk, N1//2] (SiLU后的结果)
    # 路由相关
    topk_weights_ptr,      # topk权重 [M, topk]
    sorted_token_ids_ptr,  # 排序后的token ID
    expert_ids_ptr,        # 专家ID
    num_tokens_post_padded_ptr,
    # 矩阵维度
    N1: tl.constexpr,      # w1的输出维度
    K: tl.constexpr,       # 输入维度
    EM,                    # 有效的M维度
    num_valid_tokens,
    # 步长参数
    stride_am, stride_ak,
    stride_w1e, stride_w1k, stride_w1n,
    stride_b1e, stride_b1n,
    stride_im, stride_in,
    # 块大小参数
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    HAS_BIAS1: tl.constexpr,
):
    """
    第一步：计算 A @ W1 + bias1，然后应用 SiLU 和门控
    """
    # 程序ID映射
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N1 // 2, BLOCK_SIZE_N)  # 注意：输出是N1//2
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

    # ====== 计算 A @ W1 + bias1 ======
    offs_bn1 = tl.arange(0, N1)  # W1的完整输出维度
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # A的指针设置
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )
    
    # W1的指针设置
    w1_ptrs = (
        w1_ptr
        + off_experts * stride_w1e
        + offs_k[:, None] * stride_w1k
        + offs_bn1[None, :] * stride_w1n
    )

    # 加载bias1 (如果存在)
    if HAS_BIAS1:
        bias1 = tl.load(
            b1_ptr + off_experts * stride_b1e + offs_bn1[None, :] * stride_b1n
        )
    
    # 第一次矩阵乘法累加器
    intermediate_result = tl.zeros((BLOCK_SIZE_M, N1), dtype=tl.float32)
    
    # K维度循环进行A @ W1
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # 加载A块
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        # 加载W1块
        w1 = tl.load(
            w1_ptrs,
            mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
            other=0.0,
        )
        
        # 累加
        intermediate_result += tl.dot(a, w1)
        
        # 更新指针
        a_ptrs += BLOCK_SIZE_K * stride_ak
        w1_ptrs += BLOCK_SIZE_K * stride_w1k

    # 添加bias1
    if HAS_BIAS1:
        intermediate_result += bias1

    # ====== SiLU激活和门控机制 ======
    # 将intermediate_result分成gate和up两部分
    N1_half = N1 // 2
    gate = intermediate_result[:, :N1_half]  # 前半部分作为gate
    up = intermediate_result[:, N1_half:]    # 后半部分作为up
    
    # 应用SiLU激活: SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    gate_silu = gate / (1.0 + tl.exp(-gate))
    
    # 门控乘法: gate_silu * up
    activated_result = gate_silu * up  # shape: [BLOCK_SIZE_M, N1_half]

    # 应用topk权重（如果需要）
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        activated_result *= moe_weight[:, None]

    activated_result = activated_result.to(compute_type)
    
    # ====== 写回中间结果 ======
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    i_ptrs = intermediate_ptr + stride_im * offs_token[:, None] + stride_in * offs_cn[None, :]
    i_mask = token_mask[:, None] & (offs_cn[None, :] < N1_half)
    tl.store(i_ptrs, activated_result[:, offs_cn], mask=i_mask)


@triton.jit
def merged_moe_final_kernel(
    # 输入矩阵指针
    intermediate_ptr, # 中间结果 [M, topk, N1//2]
    w2_ptr,          # 第二层权重 [E, N2, N1//2]
    b2_ptr,          # 第二层偏置 [E, N2] (可选)
    c_ptr,           # 输出 [M, topk, N2]
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
    # 块大小参数
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    HAS_BIAS2: tl.constexpr,
):
    """
    第二步：计算 intermediate @ W2 + bias2
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

    # ====== 计算 intermediate @ W2 + bias2 ======
    offs_bn2 = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # intermediate的指针设置
    i_ptrs = intermediate_ptr + (
        offs_token[:, None] * stride_im + offs_k[None, :] * stride_in
    )
    
    # W2的指针设置
    w2_ptrs = (
        w2_ptr
        + off_experts * stride_w2e
        + offs_k[:, None] * stride_w2k
        + offs_bn2[None, :] * stride_w2n
    )

    # 加载bias2 (如果存在)
    if HAS_BIAS2:
        bias2 = tl.load(
            b2_ptr + off_experts * stride_b2e + offs_bn2[None, :] * stride_b2n,
            mask=offs_bn2[None, :] < N2,
            other=0.0,
        )

    # 最终输出累加器
    final_accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # K维度循环进行 intermediate @ W2
    for k in range(0, tl.cdiv(N1_half, BLOCK_SIZE_K)):
        # 加载intermediate块
        intermediate = tl.load(
            i_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < N1_half - k * BLOCK_SIZE_K),
            other=0.0,
        )
        # 加载W2块
        w2 = tl.load(
            w2_ptrs,
            mask=(offs_k[:, None] < N1_half - k * BLOCK_SIZE_K) & (offs_bn2[None, :] < N2),
            other=0.0,
        )
        
        # 累加
        final_accumulator += tl.dot(intermediate, w2)
        
        # 更新指针
        i_ptrs += BLOCK_SIZE_K * stride_in
        w2_ptrs += BLOCK_SIZE_K * stride_w2k

    # 添加bias2
    if HAS_BIAS2:
        final_accumulator += bias2

    final_accumulator = final_accumulator.to(compute_type)
    
    # ====== 写回最终结果 ======
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N2)
    tl.store(c_ptrs, final_accumulator, mask=c_mask)


def invoke_merged_moe_kernel(
    A: torch.Tensor,           # 输入 [M, K]
    W1: torch.Tensor,          # 第一层权重 [E, N1, K]
    W2: torch.Tensor,          # 第二层权重 [E, N2, N1//2]
    C: torch.Tensor,           # 输出 [M, topk, N2]
    bias1: Optional[torch.Tensor] = None,  # 第一层偏置 [E, N1]
    bias2: Optional[torch.Tensor] = None,  # 第二层偏置 [E, N2]
    topk_weights: torch.Tensor = None,
    topk_ids: torch.Tensor = None,
    sorted_token_ids: torch.Tensor = None,
    expert_ids: torch.Tensor = None,
    num_tokens_post_padded: torch.Tensor = None,
    mul_routed_weight: bool = True,
    top_k: int = 1,
    config: Dict[str, Any] = None,
    compute_type: tl.dtype = tl.float16,
) -> None:
    """
    调用合并的MoE内核 - 使用两步法
    
    Args:
        A: 输入张量 [M, K]
        W1: 第一层专家权重 [E, N1, K]，其中N1是中间维度
        W2: 第二层专家权重 [E, N2, N1//2]，注意输入维度是N1//2因为SiLU门控
        C: 输出张量 [M, topk, N2]
        bias1: 第一层偏置（可选）
        bias2: 第二层偏置（可选）
        其他参数: MoE路由相关参数
    """
    
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
        dtype=A.dtype,
    )
    
    # 第一步：计算W1并应用SiLU + 门控
    grid1 = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(N1_half, META["BLOCK_SIZE_N"]),
    )
    
    merged_moe_gate_kernel[grid1](
        A, W1, bias1, intermediate_cache,
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
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        HAS_BIAS1=bias1 is not None,
        **config,
    )
    
    # 第二步：计算W2
    grid2 = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(N2, META["BLOCK_SIZE_N"]),
    )
    
    merged_moe_final_kernel[grid2](
        intermediate_cache, W2, bias2, C,
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
        top_k=top_k,
        compute_type=compute_type,
        HAS_BIAS2=bias2 is not None,
        **config,
    )


def merged_fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    compute_type: tl.dtype = tl.float16,
    config: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    使用合并内核的MoE实现
    """
    # 检查约束
    assert activation == "silu", "合并内核目前只支持SiLU激活"
    assert hidden_states.shape[1] == w1.shape[2], "隐藏状态维度与W1不匹配"
    assert w1.shape[1] == 2 * w2.shape[2], "W1输出维度应该是W2输入维度的2倍"
    assert topk_weights.shape == topk_ids.shape, "topk形状不匹配"
    assert hidden_states.is_contiguous(), "hidden_states必须是连续的"
    assert w1.is_contiguous(), "w1必须是连续的"
    assert w2.is_contiguous(), "w2必须是连续的"
    
    num_tokens, _ = hidden_states.shape
    E, N1, _ = w1.shape
    _, N2, _ = w2.shape
    
    # 对于大批次，我们仍然使用分块处理
    CHUNK_SIZE = 64 * 1024
    M = min(num_tokens, CHUNK_SIZE)
    
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
    
    # 创建输出张量
    if inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)
    
    # 导入对齐函数
    from .fused_moe import moe_align_block_size
    
    # 分块处理
    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (
            chunk * CHUNK_SIZE,
            min((chunk + 1) * CHUNK_SIZE, num_tokens),
        )
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.shape
        
        if tokens_in_chunk == 0:
            break
        
        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]
        
        # 对齐块大小
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            curr_topk_ids, config["BLOCK_SIZE_M"], E
        )
        
        # 创建输出缓存
        output_cache = torch.empty(
            (tokens_in_chunk, topk_ids.shape[1], N2),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        
        # 调用合并的内核
        invoke_merged_moe_kernel(
            curr_hidden_states,
            w1,
            w2,
            output_cache,
            b1,
            b2,
            curr_topk_weights,
            curr_topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            apply_router_weight_on_input,
            topk_ids.shape[1],
            config,
            compute_type,
        )
        
        # 合并topk结果
        if topk_ids.shape[1] == 1:
            # 单个专家的情况
            out_hidden_states[begin_chunk_idx:end_chunk_idx] = output_cache.squeeze(1)
        else:
            # 多个专家需要求和
            out_hidden_states[begin_chunk_idx:end_chunk_idx] = output_cache.sum(dim=1)
    
    return out_hidden_states
