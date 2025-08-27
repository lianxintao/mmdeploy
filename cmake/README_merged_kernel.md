# 合并 MoE Kernel 实现

## 概述

本文档描述了将三个 MoE 内核合并为两个内核的实现，以提高性能并减少内存访问开销。

## 背景

原始的 fused MoE 实现使用三个分离的内核：
1. **第一次 `invoke_fused_moe_kernel`**: 计算 `A @ W1 + bias1` → `intermediate_cache1`
2. **`silu_and_mul`**: 对 `intermediate_cache1` 进行 SiLU 激活和门控 → `intermediate_cache2`  
3. **第二次 `invoke_fused_moe_kernel`**: 计算 `intermediate_cache2 @ W2 + bias2` → 最终输出

这种方法的问题：
- 需要多次内核启动开销
- 中间结果需要写入全局内存然后重新读取
- 内存带宽利用效率不高

## 新实现方案

### 两内核方案

我们将三个内核合并为两个专门优化的内核：

#### 1. `merged_moe_gate_kernel`
- **功能**: 计算 `A @ W1 + bias1`，然后直接应用 SiLU 激活和门控
- **输入**: 
  - `a_ptr`: 输入激活 [M, K]
  - `w1_ptr`: 第一层权重 [E, N1, K]
  - `b1_ptr`: 第一层偏置 [E, N1]（可选）
- **输出**: 
  - `intermediate_ptr`: SiLU门控后的结果 [M, topk, N1//2]
- **优势**: 避免写入完整的中间结果，直接输出门控后的压缩结果

#### 2. `merged_moe_final_kernel`
- **功能**: 计算 `intermediate @ W2 + bias2`
- **输入**:
  - `intermediate_ptr`: SiLU门控后的结果 [M, topk, N1//2]
  - `w2_ptr`: 第二层权重 [E, N2, N1//2]
  - `b2_ptr`: 第二层偏置 [E, N2]（可选）
- **输出**:
  - `c_ptr`: 最终输出 [M, topk, N2]

### 关键优化

1. **内存访问优化**: 
   - 中间结果维度减半（N1 → N1/2）
   - 减少一次完整的全局内存读写循环

2. **计算融合**:
   - 在第一个内核中直接融合 SiLU 激活和门控操作
   - 避免存储未激活的中间结果

3. **向量化操作**:
   - 使用 Triton 的向量化指令进行高效的逐元素操作
   - SiLU 激活使用优化的数学函数

## 代码结构

### 文件说明

- `merged_moe_kernel.py`: 主要实现文件
  - `merged_moe_gate_kernel`: 第一阶段内核
  - `merged_moe_final_kernel`: 第二阶段内核
  - `invoke_merged_moe_kernel`: 高级调用接口
  - `merged_fused_experts_impl`: 与原始接口兼容的实现

- `test_merged_kernel.py`: 全面的测试套件
- `simple_test.py`: 简化的快速测试

### 关键参数

```python
# 内核配置参数
config = {
    "BLOCK_SIZE_M": 64,    # M维度的块大小
    "BLOCK_SIZE_N": 64,    # N维度的块大小  
    "BLOCK_SIZE_K": 32,    # K维度的块大小
    "GROUP_SIZE_M": 8,     # M维度的分组大小
    "num_warps": 4,        # 每个CTA的warp数量
    "num_stages": 2,       # 流水线阶段数
}
```

### 使用示例

```python
from merged_moe_kernel import merged_fused_experts_impl

# 与原始接口完全兼容
result = merged_fused_experts_impl(
    hidden_states=input_tensor,
    w1=expert_weights_1,
    w2=expert_weights_2,
    topk_weights=routing_weights,
    topk_ids=routing_indices,
    b1=bias_1,          # 可选
    b2=bias_2,          # 可选
    activation="silu",  # 目前只支持 SiLU
    inplace=False,
)
```

## 性能预期

### 理论分析

1. **内核启动开销**: 从 3 次减少到 2 次 (33% 减少)
2. **内存访问**: 
   - 减少一次完整的 N1 维度中间结果读写
   - 中间结果大小减半 (N1 → N1/2)
3. **计算效率**: 
   - 更好的局部性，SiLU 计算直接使用寄存器中的数据
   - 减少全局内存往返

### 预期加速比

- **小批次** (M <= 256): 1.2-1.5x
- **中等批次** (256 < M <= 1024): 1.3-1.8x  
- **大批次** (M > 1024): 1.4-2.0x

性能提升主要来自内存访问优化，在内存带宽受限的场景下效果更明显。

## 兼容性

### 支持的配置
- ✅ 所有标准的 MoE 配置
- ✅ 可选的偏置项 (b1, b2)
- ✅ 不同的 top-k 值
- ✅ 各种批次大小和维度
- ✅ float16 和 bfloat16 数据类型

### 当前限制
- 🔴 目前只支持 SiLU 激活函数
- 🔴 不支持量化 (FP8, INT8 等)
- 🔴 不支持专家并行 (EP)

### 未来扩展
- 🟡 添加 GELU 激活函数支持
- 🟡 添加量化支持
- 🟡 优化专家并行场景

## 测试和验证

### 正确性测试
```bash
# 快速测试
python simple_test.py

# 全面测试
python test_merged_kernel.py
```

### 性能基准测试
```bash
# 在 test_merged_kernel.py 中包含性能对比
python test_merged_kernel.py
```

测试涵盖：
- 不同形状和配置的正确性验证
- 与原始三内核实现的数值对比
- 边界条件和异常情况处理
- 性能基准测试

## 集成指南

### 替换原始实现

要使用新的合并内核替换原始实现：

1. 将 `merged_moe_kernel.py` 添加到项目中
2. 在 `fused_moe.py` 中导入新的实现
3. 修改 `fused_experts_impl` 函数调用新的实现：

```python
# 在 fused_moe.py 中
from .merged_moe_kernel import merged_fused_experts_impl

def fused_experts_impl(...):
    # 如果满足条件，使用合并内核
    if activation == "silu" and not use_quantization:
        return merged_fused_experts_impl(...)
    else:
        # 回退到原始实现
        return original_fused_experts_impl(...)
```

### 配置优化

根据硬件和工作负载调整配置参数：

```python
# 针对 A100/H100 优化
config_a100 = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128, 
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 16,
    "num_warps": 8,
    "num_stages": 4,
}

# 针对 V100 优化
config_v100 = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 32, 
    "GROUP_SIZE_M": 8,
    "num_warps": 4,
    "num_stages": 2,
}
```

## 总结

合并 MoE 内核提供了显著的性能改进，特别是在内存带宽受限的场景下。通过减少内核启动开销和优化内存访问模式，我们实现了：

- **更少的内核调用**: 3 → 2
- **更少的内存访问**: 减少中间结果的完整读写
- **更好的缓存局部性**: SiLU 计算使用寄存器中的数据
- **向后兼容**: 与现有接口完全兼容

这个实现为大规模 MoE 模型的高效推理提供了重要的优化。
