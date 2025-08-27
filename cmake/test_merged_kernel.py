"""
测试合并MoE内核的正确性和性能
"""

import torch
import numpy as np
import time
from typing import Tuple, Optional

# 导入原始实现和合并内核实现
from .fused_moe import (
    fused_experts_impl,
    moe_align_block_size,
    try_get_optimal_moe_config,
)
from .merged_moe_kernel import merged_fused_experts_impl
import triton.language as tl


def create_test_data(
    M: int = 128,           # token数量
    K: int = 4096,          # 输入维度
    N1: int = 8192,         # 中间维度
    N2: int = 4096,         # 输出维度
    E: int = 8,             # 专家数量
    top_k: int = 2,         # 每个token选择的专家数
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, ...]:
    """
    创建测试数据
    """
    # 输入激活
    hidden_states = torch.randn(M, K, device=device, dtype=dtype)
    
    # 专家权重
    w1 = torch.randn(E, N1, K, device=device, dtype=dtype)
    w2 = torch.randn(E, N2, N1 // 2, device=device, dtype=dtype)  # 注意：N1//2因为SiLU门控
    
    # 偏置（可选）
    b1 = torch.randn(E, N1, device=device, dtype=dtype)
    b2 = torch.randn(E, N2, device=device, dtype=dtype)
    
    # 生成topk路由信息
    # 为每个token随机选择top_k个专家
    topk_ids = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32)
    topk_weights = torch.rand(M, top_k, device=device, dtype=dtype)
    # 归一化权重
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    
    return hidden_states, w1, w2, b1, b2, topk_weights, topk_ids


def test_correctness():
    """
    测试合并内核的正确性，与原始三步实现对比
    """
    print("=== 测试合并内核正确性 ===")
    
    # 创建测试数据
    M, K, N1, N2, E, top_k = 64, 512, 1024, 512, 4, 2
    hidden_states, w1, w2, b1, b2, topk_weights, topk_ids = create_test_data(
        M, K, N1, N2, E, top_k
    )
    
    # 使用原始三步实现
    print("运行原始三步实现...")
    original_result = fused_experts_impl(
        hidden_states.clone(),
        w1, w2,
        topk_weights, topk_ids,
        b1, b2,
        inplace=False,
        activation="silu",
        apply_router_weight_on_input=False,
    )
    
    # 使用合并内核实现
    print("运行合并内核实现...")
    try:
        merged_result = merged_fused_experts_impl(
            hidden_states.clone(),
            w1, w2,
            topk_weights, topk_ids,
            b1, b2,
            inplace=False,
            activation="silu",
            apply_router_weight_on_input=False,
        )
        
        # 比较结果
        diff = torch.abs(original_result - merged_result)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"最大差异: {max_diff:.6f}")
        print(f"平均差异: {mean_diff:.6f}")
        
        # 检查相对误差
        relative_error = diff / (torch.abs(original_result) + 1e-8)
        max_rel_error = relative_error.max().item()
        mean_rel_error = relative_error.mean().item()
        
        print(f"最大相对误差: {max_rel_error:.6f}")
        print(f"平均相对误差: {mean_rel_error:.6f}")
        
        # 判断是否通过测试
        tolerance = 1e-3  # 考虑到float16精度
        if max_rel_error < tolerance:
            print("✅ 正确性测试通过！")
            return True
        else:
            print("❌ 正确性测试失败！")
            return False
            
    except Exception as e:
        print(f"❌ 合并内核执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def benchmark_performance():
    """
    基准测试性能对比
    """
    print("\n=== 性能基准测试 ===")
    
    # 不同的测试配置
    test_configs = [
        (128, 4096, 8192, 4096, 8, 2),    # 小批次
        (512, 4096, 8192, 4096, 8, 2),    # 中等批次
        (1024, 4096, 8192, 4096, 8, 2),   # 大批次
    ]
    
    warmup_iters = 5
    benchmark_iters = 20
    
    for M, K, N1, N2, E, top_k in test_configs:
        print(f"\n配置: M={M}, K={K}, N1={N1}, N2={N2}, E={E}, top_k={top_k}")
        
        # 创建测试数据
        hidden_states, w1, w2, b1, b2, topk_weights, topk_ids = create_test_data(
            M, K, N1, N2, E, top_k
        )
        
        # 预热
        for _ in range(warmup_iters):
            _ = fused_experts_impl(
                hidden_states.clone(),
                w1, w2, topk_weights, topk_ids, b1, b2,
                inplace=False, activation="silu"
            )
            try:
                _ = merged_fused_experts_impl(
                    hidden_states.clone(),
                    w1, w2, topk_weights, topk_ids, b1, b2,
                    inplace=False, activation="silu"
                )
            except:
                pass
        
        torch.cuda.synchronize()
        
        # 测试原始实现
        start_time = time.time()
        for _ in range(benchmark_iters):
            _ = fused_experts_impl(
                hidden_states.clone(),
                w1, w2, topk_weights, topk_ids, b1, b2,
                inplace=False, activation="silu"
            )
        torch.cuda.synchronize()
        original_time = (time.time() - start_time) / benchmark_iters
        
        # 测试合并内核实现
        try:
            torch.cuda.synchronize()
            start_time = time.time()
            for _ in range(benchmark_iters):
                _ = merged_fused_experts_impl(
                    hidden_states.clone(),
                    w1, w2, topk_weights, topk_ids, b1, b2,
                    inplace=False, activation="silu"
                )
            torch.cuda.synchronize()
            merged_time = (time.time() - start_time) / benchmark_iters
            
            speedup = original_time / merged_time
            print(f"原始实现时间: {original_time*1000:.3f} ms")
            print(f"合并内核时间: {merged_time*1000:.3f} ms")
            print(f"加速比: {speedup:.2f}x")
        except Exception as e:
            print(f"合并内核测试失败: {e}")


def test_different_shapes():
    """
    测试不同形状的兼容性
    """
    print("\n=== 测试不同形状兼容性 ===")
    
    test_shapes = [
        (32, 256, 512, 256, 4, 1),      # 小模型
        (64, 1024, 2048, 1024, 8, 2),   # 中等模型
        (128, 2048, 4096, 2048, 16, 4), # 大模型
    ]
    
    for i, (M, K, N1, N2, E, top_k) in enumerate(test_shapes):
        print(f"\n测试形状 {i+1}: M={M}, K={K}, N1={N1}, N2={N2}, E={E}, top_k={top_k}")
        
        try:
            hidden_states, w1, w2, b1, b2, topk_weights, topk_ids = create_test_data(
                M, K, N1, N2, E, top_k
            )
            
            result = merged_fused_experts_impl(
                hidden_states,
                w1, w2, topk_weights, topk_ids, b1, b2,
                inplace=False, activation="silu"
            )
            
            expected_shape = (M, N2)
            if result.shape == expected_shape:
                print(f"✅ 形状正确: {result.shape}")
            else:
                print(f"❌ 形状错误: 期望 {expected_shape}, 得到 {result.shape}")
                
        except Exception as e:
            print(f"❌ 测试失败: {e}")


def test_edge_cases():
    """
    测试边界情况
    """
    print("\n=== 测试边界情况 ===")
    
    # 测试无偏置的情况
    print("测试无偏置...")
    try:
        M, K, N1, N2, E, top_k = 32, 256, 512, 256, 4, 2
        hidden_states, w1, w2, _, _, topk_weights, topk_ids = create_test_data(
            M, K, N1, N2, E, top_k
        )
        
        result = merged_fused_experts_impl(
            hidden_states, w1, w2, topk_weights, topk_ids,
            b1=None, b2=None,
            inplace=False, activation="silu"
        )
        print("✅ 无偏置测试通过")
    except Exception as e:
        print(f"❌ 无偏置测试失败: {e}")
    
    # 测试top_k=1的情况
    print("测试top_k=1...")
    try:
        M, K, N1, N2, E, top_k = 32, 256, 512, 256, 4, 1
        hidden_states, w1, w2, b1, b2, topk_weights, topk_ids = create_test_data(
            M, K, N1, N2, E, top_k
        )
        
        result = merged_fused_experts_impl(
            hidden_states, w1, w2, topk_weights, topk_ids, b1, b2,
            inplace=False, activation="silu"
        )
        print("✅ top_k=1测试通过")
    except Exception as e:
        print(f"❌ top_k=1测试失败: {e}")


def main():
    """
    主测试函数
    """
    print("开始测试合并MoE内核...\n")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        return
    
    # 设置设备
    torch.cuda.set_device(0)
    
    # 运行所有测试
    success = True
    
    # 正确性测试
    success &= test_correctness()
    
    # 形状兼容性测试
    test_different_shapes()
    
    # 边界情况测试
    test_edge_cases()
    
    # 性能测试（如果正确性通过）
    if success:
        benchmark_performance()
    
    print(f"\n测试完成！")


if __name__ == "__main__":
    main()
