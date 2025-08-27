"""
简单测试合并MoE内核
"""

import torch
import triton.language as tl
import sys
import os

# 添加路径以便导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_kernel_basic():
    """
    基本的kernel测试
    """
    print("=== 测试合并MoE内核基本功能 ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        return False
    
    try:
        from merged_moe_kernel import invoke_merged_moe_kernel
        from fused_moe import moe_align_block_size
        print("✅ 成功导入模块")
    except Exception as e:
        print(f"❌ 导入模块失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 创建简单的测试数据
    device = "cuda"
    dtype = torch.float16
    
    # 小规模测试
    M, K, N1, N2, E, top_k = 32, 128, 256, 128, 4, 2
    
    try:
        # 输入数据
        A = torch.randn(M, K, device=device, dtype=dtype)
        W1 = torch.randn(E, N1, K, device=device, dtype=dtype)
        W2 = torch.randn(E, N2, N1 // 2, device=device, dtype=dtype)
        bias1 = torch.randn(E, N1, device=device, dtype=dtype)
        bias2 = torch.randn(E, N2, device=device, dtype=dtype)
        
        # 路由信息
        topk_ids = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32)
        topk_weights = torch.rand(M, top_k, device=device, dtype=dtype)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        print("✅ 成功创建测试数据")
        
        # 配置
        config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 2,
            "num_warps": 2,
            "num_stages": 2,
        }
        
        # 对齐块大小
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, config["BLOCK_SIZE_M"], E
        )
        
        print("✅ 成功完成块对齐")
        
        # 创建输出张量
        C = torch.empty(M, top_k, N2, device=device, dtype=dtype)
        
        # 调用合并内核
        invoke_merged_moe_kernel(
            A=A,
            W1=W1,
            W2=W2,
            C=C,
            bias1=bias1,
            bias2=bias2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            mul_routed_weight=False,  # 先测试不应用路由权重
            top_k=top_k,
            config=config,
            compute_type=tl.float16,
        )
        
        print("✅ 成功执行合并内核")
        
        # 检查输出是否有效
        if torch.isnan(C).any():
            print("❌ 输出包含NaN")
            return False
        
        if torch.isinf(C).any():
            print("❌ 输出包含Inf")
            return False
        
        print(f"✅ 输出形状: {C.shape}")
        print(f"✅ 输出范围: [{C.min().item():.4f}, {C.max().item():.4f}]")
        print(f"✅ 输出均值: {C.mean().item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 内核执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_reference():
    """
    与参考实现对比测试
    """
    print("\n=== 与参考实现对比测试 ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        return False
    
    try:
        from merged_moe_kernel import invoke_merged_moe_kernel
        from fused_moe import invoke_fused_moe_kernel, moe_align_block_size
        print("✅ 成功导入所有模块")
    except Exception as e:
        print(f"❌ 导入模块失败: {e}")
        return False
    
    # 创建测试数据
    device = "cuda"
    dtype = torch.float16
    
    M, K, N1, N2, E, top_k = 16, 64, 128, 64, 2, 1  # 非常小的规模
    
    try:
        # 输入数据
        A = torch.randn(M, K, device=device, dtype=dtype)
        W1 = torch.randn(E, N1, K, device=device, dtype=dtype)
        W2 = torch.randn(E, N2, N1 // 2, device=device, dtype=dtype)
        bias1 = torch.randn(E, N1, device=device, dtype=dtype)
        bias2 = torch.randn(E, N2, device=device, dtype=dtype)
        
        # 路由信息
        topk_ids = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32)
        topk_weights = torch.ones(M, top_k, device=device, dtype=dtype)  # 全为1，简化测试
        
        print("✅ 成功创建测试数据")
        
        # 配置
        config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 2,
            "num_warps": 2,
            "num_stages": 2,
        }
        
        # 对齐块大小
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, config["BLOCK_SIZE_M"], E
        )
        
        # ======= 参考实现：三步法 =======
        print("运行参考实现（三步法）...")
        
        # 第一步：A @ W1 + bias1
        intermediate1 = torch.empty(M, top_k, N1, device=device, dtype=dtype)
        invoke_fused_moe_kernel(
            A, W1, bias1, intermediate1,
            None, None, None,  # 量化相关参数
            topk_weights, topk_ids,
            sorted_token_ids, expert_ids, num_tokens_post_padded,
            False,  # mul_routed_weight
            top_k, config, tl.float16,
            False, False, False, False, False,  # 量化标志
            None  # block_shape
        )
        
        # 第二步：SiLU和门控
        # 模拟silu_and_mul的功能
        intermediate1_reshaped = intermediate1.view(-1, N1)
        gate = intermediate1_reshaped[:, :N1//2]
        up = intermediate1_reshaped[:, N1//2:]
        gate_silu = gate / (1.0 + torch.exp(-gate))
        intermediate2 = (gate_silu * up).view(M, top_k, N1//2)
        
        # 第三步：intermediate2 @ W2 + bias2
        reference_output = torch.empty(M, top_k, N2, device=device, dtype=dtype)
        invoke_fused_moe_kernel(
            intermediate2.view(M * top_k, N1//2), W2, bias2, reference_output,
            None, None, None,  # 量化相关参数
            topk_weights, topk_ids,
            sorted_token_ids, expert_ids, num_tokens_post_padded,
            False,  # mul_routed_weight
            1, config, tl.float16,  # 注意：这里top_k=1因为已经是展开的
            False, False, False, False, False,  # 量化标志
            None  # block_shape
        )
        
        print("✅ 参考实现完成")
        
        # ======= 合并内核实现 =======
        print("运行合并内核实现...")
        
        merged_output = torch.empty(M, top_k, N2, device=device, dtype=dtype)
        invoke_merged_moe_kernel(
            A=A,
            W1=W1,
            W2=W2,
            C=merged_output,
            bias1=bias1,
            bias2=bias2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            mul_routed_weight=False,
            top_k=top_k,
            config=config,
            compute_type=tl.float16,
        )
        
        print("✅ 合并内核完成")
        
        # 比较结果
        diff = torch.abs(reference_output - merged_output)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"最大差异: {max_diff:.6f}")
        print(f"平均差异: {mean_diff:.6f}")
        
        # 相对误差
        relative_error = diff / (torch.abs(reference_output) + 1e-8)
        max_rel_error = relative_error.max().item()
        mean_rel_error = relative_error.mean().item()
        
        print(f"最大相对误差: {max_rel_error:.6f}")
        print(f"平均相对误差: {mean_rel_error:.6f}")
        
        # 判断是否通过
        tolerance = 1e-2  # 相对宽松的阈值
        if max_rel_error < tolerance:
            print("✅ 对比测试通过！")
            return True
        else:
            print("❌ 对比测试失败！")
            print(f"参考输出样本: {reference_output[0, 0, :5]}")
            print(f"合并输出样本: {merged_output[0, 0, :5]}")
            return False
            
    except Exception as e:
        print(f"❌ 对比测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    主测试函数
    """
    print("开始测试合并MoE内核...\n")
    
    success = True
    
    # 基本功能测试
    success &= test_kernel_basic()
    
    # 与参考实现对比测试
    success &= test_with_reference()
    
    if success:
        print("\n🎉 所有测试通过！")
    else:
        print("\n❌ 部分测试失败")
    
    return success


if __name__ == "__main__":
    main()
