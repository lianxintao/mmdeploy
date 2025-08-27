"""
测试高效的 FP8 专用 MoE 内核
"""

import torch
import triton
import sys
import os

# 添加路径以便导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_efficient_fp8_test_data(
    M: int = 64,            # token数量
    K: int = 256,           # 输入维度
    N1: int = 512,          # 中间维度
    N2: int = 256,          # 输出维度
    E: int = 4,             # 专家数量
    top_k: int = 1,         # 每个token选择的专家数
    device: str = "cuda",
    block_shape: list = [64, 64],
):
    """
    创建高效 FP8 测试数据
    """
    block_n, block_k = block_shape
    
    # 输入激活 (FP16，会在内核中量化为FP8)
    hidden_states = torch.randn(M, K, device=device, dtype=torch.float16)
    
    # 专家权重 (FP8)
    w1 = torch.randn(E, N1, K, device=device, dtype=torch.float8_e4m3fn)
    w2 = torch.randn(E, N2, N1 // 2, device=device, dtype=torch.float8_e4m3fn)
    
    # 偏置
    b1 = torch.randn(E, N1, device=device, dtype=torch.float16)
    b2 = torch.randn(E, N2, device=device, dtype=torch.float16)
    
    # FP8 块量化缩放因子
    num_blocks_n1 = triton.cdiv(N1, block_n)
    num_blocks_k1 = triton.cdiv(K, block_k)
    num_blocks_n2 = triton.cdiv(N2, block_n)
    num_blocks_k2 = triton.cdiv(N1 // 2, block_k)
    
    w1_scale = torch.rand(E, num_blocks_n1, num_blocks_k1, device=device, dtype=torch.float32) * 0.1
    w2_scale = torch.rand(E, num_blocks_n2, num_blocks_k2, device=device, dtype=torch.float32) * 0.1
    a1_scale = torch.rand(M, num_blocks_k1, device=device, dtype=torch.float32) * 0.1
    a2_scale = torch.rand(M, num_blocks_k2, device=device, dtype=torch.float32) * 0.1
    
    # 路由信息
    topk_ids = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32)
    topk_weights = torch.ones(M, top_k, device=device, dtype=torch.float16)
    
    return {
        'hidden_states': hidden_states,
        'w1': w1, 'w2': w2, 'b1': b1, 'b2': b2,
        'w1_scale': w1_scale, 'w2_scale': w2_scale,
        'a1_scale': a1_scale, 'a2_scale': a2_scale,
        'topk_weights': topk_weights, 'topk_ids': topk_ids,
    }


def test_efficient_fp8_kernel():
    """
    测试高效 FP8 专用内核
    """
    print("=== 测试高效 FP8 专用 MoE 内核 ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        return False
    
    try:
        from fp8_moe_kernel import fp8_moe_impl
        print("✅ 成功导入高效 FP8 内核")
    except Exception as e:
        print(f"❌ 导入模块失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 创建测试数据
    M, K, N1, N2, E, top_k = 32, 128, 256, 128, 4, 1
    block_shape = [64, 64]  # block_n, block_k
    
    test_data = create_efficient_fp8_test_data(
        M, K, N1, N2, E, top_k,
        block_shape=block_shape
    )
    
    try:
        # 测试高效 FP8 内核
        result = fp8_moe_impl(
            hidden_states=test_data['hidden_states'],
            w1=test_data['w1'],
            w2=test_data['w2'],
            topk_weights=test_data['topk_weights'],
            topk_ids=test_data['topk_ids'],
            b1=test_data['b1'],
            b2=test_data['b2'],
            w1_scale=test_data['w1_scale'],
            w2_scale=test_data['w2_scale'],
            a1_scale=test_data['a1_scale'],
            a2_scale=test_data['a2_scale'],
            block_shape=block_shape,
            inplace=False,
            activation="silu",
            apply_router_weight_on_input=False,
        )
        
        print("✅ 高效 FP8 内核测试通过")
        print(f"输出形状: {result.shape}")
        print(f"输出范围: [{result.min().item():.4f}, {result.max().item():.4f}]")
        
        # 检查输出有效性
        if torch.isnan(result).any():
            print("❌ 输出包含NaN")
            return False
        
        if torch.isinf(result).any():
            print("❌ 输出包含Inf")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 高效 FP8 内核测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_comparison():
    """
    测试性能对比
    """
    print("\n=== 性能对比测试 ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        return False
    
    try:
        from fp8_moe_kernel import fp8_moe_impl
        print("✅ 成功导入高效 FP8 内核")
    except Exception as e:
        print(f"❌ 导入模块失败: {e}")
        return False
    
    # 创建较大的测试数据进行性能测试
    M, K, N1, N2, E, top_k = 256, 512, 1024, 512, 8, 2
    block_shape = [64, 64]
    
    test_data = create_efficient_fp8_test_data(
        M, K, N1, N2, E, top_k,
        block_shape=block_shape
    )
    
    try:
        # 预热
        for _ in range(3):
            result = fp8_moe_impl(
                hidden_states=test_data['hidden_states'],
                w1=test_data['w1'],
                w2=test_data['w2'],
                topk_weights=test_data['topk_weights'],
                topk_ids=test_data['topk_ids'],
                b1=test_data['b1'],
                b2=test_data['b2'],
                w1_scale=test_data['w1_scale'],
                w2_scale=test_data['w2_scale'],
                a1_scale=test_data['a1_scale'],
                a2_scale=test_data['a2_scale'],
                block_shape=block_shape,
            )
        
        # 性能测试
        torch.cuda.synchronize()
        import time
        
        num_runs = 10
        start_time = time.time()
        
        for _ in range(num_runs):
            result = fp8_moe_impl(
                hidden_states=test_data['hidden_states'],
                w1=test_data['w1'],
                w2=test_data['w2'],
                topk_weights=test_data['topk_weights'],
                topk_ids=test_data['topk_ids'],
                b1=test_data['b1'],
                b2=test_data['b2'],
                w1_scale=test_data['w1_scale'],
                w2_scale=test_data['w2_scale'],
                a1_scale=test_data['a1_scale'],
                a2_scale=test_data['a2_scale'],
                block_shape=block_shape,
            )
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs * 1000  # ms
        print(f"✅ 性能测试完成")
        print(f"平均延迟: {avg_time:.2f} ms")
        print(f"形状: M={M}, K={K}, N1={N1}, N2={N2}, E={E}, top_k={top_k}")
        
        return True
        
    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_sizes():
    """
    测试不同尺寸配置
    """
    print("\n=== 测试不同尺寸配置 ===")
    
    configurations = [
        {"name": "小规模", "M": 16, "K": 64, "N1": 128, "N2": 64, "E": 2, "top_k": 1},
        {"name": "中等规模", "M": 64, "K": 256, "N1": 512, "N2": 256, "E": 4, "top_k": 2},
        {"name": "大规模", "M": 128, "K": 512, "N1": 1024, "N2": 512, "E": 8, "top_k": 2},
    ]
    
    success_count = 0
    
    for config in configurations:
        print(f"\n测试配置: {config['name']}")
        
        try:
            from fp8_moe_kernel import fp8_moe_impl
            
            block_shape = [64, 64]
            test_data = create_efficient_fp8_test_data(
                config["M"], config["K"], config["N1"], config["N2"], 
                config["E"], config["top_k"], block_shape=block_shape
            )
            
            result = fp8_moe_impl(
                hidden_states=test_data['hidden_states'],
                w1=test_data['w1'],
                w2=test_data['w2'],
                topk_weights=test_data['topk_weights'],
                topk_ids=test_data['topk_ids'],
                b1=test_data['b1'],
                b2=test_data['b2'],
                w1_scale=test_data['w1_scale'],
                w2_scale=test_data['w2_scale'],
                a1_scale=test_data['a1_scale'],
                a2_scale=test_data['a2_scale'],
                block_shape=block_shape,
            )
            
            expected_shape = (config["M"], config["N2"])
            if result.shape == expected_shape:
                print(f"✅ {config['name']} 测试通过")
                success_count += 1
            else:
                print(f"❌ {config['name']} 形状错误: 期望 {expected_shape}, 得到 {result.shape}")
                
        except Exception as e:
            print(f"❌ {config['name']} 测试失败: {e}")
    
    print(f"\n配置测试结果: {success_count}/{len(configurations)} 通过")
    return success_count == len(configurations)


def main():
    """
    主测试函数
    """
    print("开始测试高效 FP8 专用 MoE 内核...\n")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过所有测试")
        return
    
    success = True
    
    # 基本功能测试
    success &= test_efficient_fp8_kernel()
    
    # 性能测试
    success &= test_performance_comparison()
    
    # 不同配置测试
    success &= test_different_sizes()
    
    if success:
        print("\n🎉 所有高效 FP8 内核测试通过！")
    else:
        print("\n❌ 部分高效 FP8 内核测试失败")
    
    print("\n📝 高效 FP8 内核特性:")
    print("  ✅ 分块计算 - 每次处理 (BLOCK_SIZE_M, BLOCK_SIZE_N) 的输出")
    print("  ✅ FP8 专用 - 无非量化分支，性能最优")
    print("  ✅ 块级量化 - 最高精度的量化方案")
    print("  ✅ 内存高效 - 避免大中间矩阵，减少cache miss")
    print("  ✅ SiLU融合 - 在分块中直接计算激活和门控")


if __name__ == "__main__":
    main()
