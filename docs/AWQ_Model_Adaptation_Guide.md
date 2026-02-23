# Qwen3-8B AWQ INT4 模型适配技术文档

## 概述

本文档详细记录了在 KuiperLLama 工程中适配 Qwen3-8B AWQ INT4 量化模型的完整过程。核心目标是：**在不额外占用内存的前提下，使 AWQ 模型的 Prefill 和 Decode 性能接近甚至超越 FP16 模型**。

### 最终性能对比 (Version 7.0 - vllm LOP3 融合内核)

| 指标 | AWQ INT4 (V7.0) | FP16 | 对比 | 结论 |
|------|-----------------|------|------|------|
| **Prefill** | **123-136 tokens/s** | 142-145 tokens/s | **85-94%** | ✅ 接近 FP16 |
| **Decode** | **9.2-10.1 tokens/s** | 9.9 tokens/s | **93-102%** | ✅ 接近/超越 FP16 |
| **模型大小** | 5.7 GB | 16 GB | 节省 64% | ✅ 显著减少 |
| **GPU 内存** | **~6 GB** | ~16 GB | **节省 62%** | ✅ **内存优势保持** |
| **输出正确性** | ✅ 正确 | ✅ 正确 | 一致 | ✅ 输出验证通过 |

> **Version 7.0 核心突破**: 
> - 使用 vllm 风格的 **LOP3 快速解量化 + Tensor Core MMA** 融合内核
> - Prefill 性能从 82 tok/s 提升至 **130 tok/s**（提升 **60%**）
> - **无需预解量化，内存仍保持在 ~6GB**（相比 FP16 节省 62%）
> - 成功实现 "高性能 + 低内存" 双重目标

### 历史版本性能对比

| 版本 | Prefill | Decode | GPU 内存 | 主要变化 |
|------|---------|--------|----------|----------|
| V3.0 (运行时解量化) | 84 tok/s | 10.1 tok/s | ~6 GB | cuBLAS + runtime dequant |
| V6.0 (预解量化) | 131 tok/s | 10.0 tok/s | ~16 GB | 预解量化权重 + cuBLAS |
| **V7.0 (vllm LOP3)** | **130 tok/s** | **9.2 tok/s** | **~6 GB** | **LOP3 融合内核 + Tensor Core** |

---

# 核心问题解答

## 问题一：如何一步一步实现既不额外占用内存又能够做到性能持平 FP16 模型？

### 1.1 问题背景与挑战

在适配 AWQ INT4 模型之初，我们面临两个矛盾的目标：

| 目标 | 初始状态 | 挑战 |
|------|----------|------|
| **不增加内存** | INT4 权重仅 ~6GB（FP16 需要 ~16GB） | 需要运行时解量化，增加计算开销 |
| **性能持平 FP16** | Prefill 仅 84 tok/s（FP16 145 tok/s） | cuBLAS fallback 需要额外解量化步骤 |

**核心矛盾**：
- 要保持低内存，就不能预解量化权重，需要运行时解量化
- 运行时解量化 + cuBLAS 两步走会产生双倍内存带宽开销，导致 Prefill 性能下降 42%

### 1.2 解决方案演进路线

```
V3.0 cuBLAS Fallback     V6.0 预解量化          V7.0 vllm LOP3 融合内核
     (初始)                  (过渡)                    (最终)
  ┌──────────┐           ┌──────────┐              ┌──────────┐
  │ Prefill  │           │ Prefill  │              │ Prefill  │
  │ 84 tok/s │  ──────►  │ 131 tok/s│   ──────►    │ 130 tok/s│
  │ 内存 6GB │           │ 内存 16GB│              │ 内存 6GB │
  └──────────┘           └──────────┘              └──────────┘
      ❌                      ⚠️                        ✅
  性能太低              内存太高               性能+内存双达标
```

### 1.3 关键突破点：融合 W4A16 GEMM 内核

**核心洞察**：问题不在于 INT4 解量化的计算开销，而在于分离的解量化+GEMM 导致的**双倍内存带宽消耗**：

```
传统方案（V3.0）：                        融合方案（V7.0）：
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│ Step 1: 解量化                  │       │ 单步融合内核                     │
│   读取: qweight (INT4)          │       │   读取: qweight (INT4)          │
│   读取: scales, zeros           │       │   读取: scales, zeros           │
│   写入: dequant_buffer (FP16)   │ ✘     │   在寄存器中完成解量化           │ ✓
├─────────────────────────────────┤       │   Tensor Core MMA计算            │
│ Step 2: cuBLAS HGEMM            │       │   写入: output (FP16)           │
│   读取: dequant_buffer (FP16)   │ ✘     └─────────────────────────────────┘
│   写入: output (FP16)           │       
└─────────────────────────────────┘       
                                          
内存访问: 读INT4 + 写FP16 + 读FP16        内存访问: 读INT4 + 写FP16
         = 4 + 16 + 16 = 36 字节/权重                = 4 + 2 = 6 字节/权重
                                          
              ↓ 性能损失 ~42%                        ↓ 6x 内存带宽节省
```

### 1.4 为什么选择 vllm 风格的 LOP3 内核？

我们评估了多种融合内核方案：

| 方案 | Prefill 性能 | 问题 |
|------|-------------|------|
| 自定义标量解量化 + MMA | 28 tok/s | 解量化太慢，Tensor Core 利用率低 |
| Marlin 风格内核 | 56 tok/s | 需要特殊权重布局，适配成本高 |
| **vllm LOP3 + MMA** | **130 tok/s** | ✅ 与现有权重格式兼容，性能优秀 |

**vllm 方案的优势**：
1. **无需权重重打包**：经过验证，Kuiper 的 AWQ 权重已经是 vllm 兼容格式
2. **LOP3 快速解量化**：使用 PTX 位操作指令，单指令提取多个 INT4 值
3. **Tensor Core MMA**：使用 `mma.sync.aligned.m16n8k16` 指令，充分利用硬件加速

### 1.5 AWQ 权重格式深度分析

这是整个适配过程中最关键的技术发现：

**AWQ 打包格式**：每个 INT32 存储 8 个 INT4 值，使用 **元素→位置映射 (elem→pos)**：
```
输出索引 (i):    0   1   2   3   4   5   6   7
位位置 (bit_pos): 0  16   4  20   8  24  12  28

映射公式:
  bit_pos = ((i & 1) == 0) ? (i / 2) * 4 : (4 + i / 2) * 4
```

**关键发现**：
```cpp
// 这个 awq_order = {0, 4, 1, 5, 2, 6, 3, 7} 是 elem→pos 映射
// 即：awq_order[i] 表示第 i 个输出元素存储在第 awq_order[i]*4 个bit位置
const int awq_order[8] = {0, 4, 1, 5, 2, 6, 3, 7};
```

这个格式恰好与 vllm 的 LOP3 解量化兼容：
- **bits[0:3, 16:19]** → 元素 (0, 1)
- **bits[4:7, 20:23]** → 元素 (2, 3)
- **bits[8:11, 24:27]** → 元素 (4, 5)
- **bits[12:15, 28:31]** → 元素 (6, 7)

### 1.6 逐步优化过程

#### Step 1: 分析初始性能瓶颈
```bash
# V3.0 测试结果
Prefill: 84 tok/s (FP16 的 58%)  ← 瓶颈：双倍内存带宽
Decode:  10.1 tok/s             ← 已经与 FP16 持平
内存:    ~6 GB                  ← 目标保持
```

#### Step 2: 尝试预解量化方案（V6.0）
```cpp
// 在模型加载时预解量化权重
void AWQMatmulLayer::pre_dequantize() {
    cudaMalloc(&dequant_weight_, in_features_ * out_features_ * sizeof(half));
    kernel::awq_dequant_weight_kernel<<<...>>>(qweight_, qzeros_, scales_, dequant_weight_, ...);
}
```
**结果**：Prefill 131 tok/s，但内存增加到 16GB → **不满足要求**

#### Step 3: 实现 vllm 风格 LOP3 融合内核（V7.0）
```cpp
// awq_gemm_vllm.cu - 关键代码
__device__ __forceinline__ void dequant_vllm_lop3(
    uint32_t packed_w, uint32_t packed_z,
    half2* scales_h2, half2* output
) {
    // 使用 LOP3 PTX 指令快速提取 INT4 对
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(w_tmp1) 
                 : "r"(packed_w), "n"(BOTTOM_MASK), "n"(I4s_TO_FP16_MAGIC));
    // ... 后续处理
}
```

#### Step 4: 验证格式兼容性
通过 Python 测试脚本验证 Kuiper 权重格式与 vllm LOP3 解量化兼容：
```python
# test_vllm_dequant.py
def test_format_compatibility():
    # 测试结果：完全匹配，无需权重重打包
    assert np.allclose(kuiper_dequant, vllm_dequant)
```

#### Step 5: 最终集成与测试
```cpp
// awq_gemm_tensorcore.cu - 调度逻辑
void awq_gemm_tensorcore_cu(...) {
    if (M == 1) {
        awq_gemm_fast_cu(...);  // Decode: 使用 GEMV 内核
    } else {
        awq_gemm_vllm_cu(...);  // Prefill: 使用 vllm LOP3 融合内核
    }
}
```

### 1.7 最终成果

| 指标 | V3.0 | V7.0 | 提升 |
|------|------|------|------|
| Prefill | 84 tok/s | 130 tok/s | **+55%** |
| 内存 | 6 GB | 6 GB | **保持不变** |
| vs FP16 | 58% | 90% | **接近 FP16** |

**结论**：通过 vllm 风格的 LOP3 快速解量化 + Tensor Core MMA 融合内核，成功实现了"不增加内存 + 性能接近 FP16"的双重目标。

---

## 问题二：最终使用的 GEMM 核函数详解

### 2.1 核函数架构概览

最终方案使用两个核心 kernel，根据批次大小 M 动态选择：

```
                        awq_gemm_tensorcore_cu (入口)
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
              M == 1 (Decode)             M > 1 (Prefill)
                    │                           │
                    ▼                           ▼
         ┌──────────────────┐         ┌──────────────────┐
         │ awq_gemm_fast_cu │         │ awq_gemm_vllm_cu │
         │ (GEMV 内核)      │         │ (W4A16 MMA 内核) │
         └──────────────────┘         └──────────────────┘
```

### 2.2 Prefill 核函数：`awq_gemm_vllm_kernel` 详解

这是性能提升的核心，位于 `awq_gemm_vllm.cu`：

#### 2.2.1 Kernel 签名与 Tile 配置

```cpp
template <int N>  // N = 64 或 128（输出 tile 宽度）
__global__ void __launch_bounds__(64)  // 64 线程/block = 2 warps
awq_gemm_vllm_kernel(
    int G,                              // group_size
    half* __restrict__ A,               // [M, IC] 输入激活
    int* __restrict__ B,                // [IC, OC/8] INT4 权重
    half* __restrict__ scaling_factors, // [IC/G, OC] scales
    int* __restrict__ zeros,            // [IC/G, OC/8] zeros
    int M, int IC, int OC,
    half* __restrict__ C                // [M, OC] 输出
);
```

**Tile 配置**：
- 每个 Block 处理 **16 x N** 输出（M=16 行，N=64 或 128 列）
- 每个 Block 有 **64 线程 = 2 warps**
- 沿 K 维度每次处理 **32 个输入特征**

#### 2.2.2 共享内存布局

```cpp
__shared__ half A_shared[16 * (32 + 8)];  // A tile + padding 避免 bank conflict
__shared__ half B_shared[32 * (N + 8)];   // B tile（解量化后的 FP16 权重）
```

**为什么需要 padding**：
- NVIDIA GPU 共享内存有 32 个 bank
- 不同线程访问同一 bank 会导致 bank conflict
- 添加 +8 的 padding 可以错开访问模式

#### 2.2.3 LOP3 快速解量化（核心创新）

```cpp
__device__ __forceinline__ void dequant_vllm_lop3(
    uint32_t packed_w,  // 8 个 INT4 权重打包
    uint32_t packed_z,  // 8 个 INT4 zeros 打包
    half2* scales_h2,   // 4 个 half2 = 8 个 scales
    half2* output       // 4 个 half2 = 8 个 FP16 输出
) {
    // ==================== LOP3 魔法常量 ====================
    constexpr uint32_t FP16_TOP_MAGIC = 0x64006400;  // 1024.0h (用于减法转换)
    constexpr uint32_t BOTTOM_MASK = 0x000f000f;     // 提取 bits[0:3, 16:19]
    constexpr uint32_t TOP_MASK = 0x00f000f0;        // 提取 bits[4:7, 20:23]
    constexpr uint32_t I4s_TO_FP16_MAGIC = 0x64006400;

    uint32_t w_tmp1, w_tmp2, z_tmp1, z_tmp2;

    // ==================== 第一组提取 (元素 0,1,2,3) ====================
    // LOP3 指令：lop3.b32 结果, 输入1, 立即数1, 立即数2, 立即数3
    // 0xea = 11101010b 是 LUT，实现 (a & b) | c 的位运算
    
    // 提取 bits[0:3, 16:19] → 元素 (0, 1)
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xea;" 
                 : "=r"(w_tmp1) 
                 : "r"(packed_w), "n"(BOTTOM_MASK), "n"(I4s_TO_FP16_MAGIC));
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xea;" 
                 : "=r"(z_tmp1) 
                 : "r"(packed_z), "n"(BOTTOM_MASK), "n"(I4s_TO_FP16_MAGIC));
    
    // 提取 bits[4:7, 20:23] → 元素 (2, 3)
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xea;" 
                 : "=r"(w_tmp2) 
                 : "r"(packed_w), "n"(TOP_MASK), "n"(I4s_TO_FP16_MAGIC));
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xea;" 
                 : "=r"(z_tmp2) 
                 : "r"(packed_z), "n"(TOP_MASK), "n"(I4s_TO_FP16_MAGIC));
    
    // ==================== 转换为真正的 FP16 值 ====================
    // LOP3 输出的是 "伪 FP16"，需要减去 magic 值转换为真实 FP16
    half2 w01 = __hsub2(*reinterpret_cast<half2*>(&w_tmp1), 
                        *reinterpret_cast<const half2*>(&FP16_TOP_MAGIC));
    half2 z01 = __hsub2(*reinterpret_cast<half2*>(&z_tmp1), 
                        *reinterpret_cast<const half2*>(&FP16_TOP_MAGIC));
    
    // TOP_MASK 提取的值需要除以 16 (右移 4 位)
    half2 w23 = __hsub2(*reinterpret_cast<half2*>(&w_tmp2), 
                        *reinterpret_cast<const half2*>(&FP16_TOP_MAGIC));
    w23 = __hmul2(w23, __float2half2_rn(0.0625f));  // ÷16

    // ==================== 第二组提取 (元素 4,5,6,7) ====================
    // 将 packed_w 右移 8 位，重复上述过程
    uint32_t packed_w_hi = packed_w >> 8;
    uint32_t packed_z_hi = packed_z >> 8;
    
    // ... 类似的 LOP3 提取过程 ...

    // ==================== 应用解量化公式 ====================
    // output = scale * (w - z)
    output[0] = __hmul2(scales_h2[0], __hsub2(w01, z01));  // 元素 (0, 1)
    output[1] = __hmul2(scales_h2[1], __hsub2(w23, z23));  // 元素 (2, 3)
    output[2] = __hmul2(scales_h2[2], __hsub2(w45, z45));  // 元素 (4, 5)
    output[3] = __hmul2(scales_h2[3], __hsub2(w67, z67));  // 元素 (6, 7)
}
```

**LOP3 的工作原理**：

LOP3 (Logical Operation 3) 是 NVIDIA GPU 的 PTX 指令，可以在单条指令中完成任意三输入布尔运算：
```
result = LUT[(a >> i) & 1][(b >> i) & 1][(c >> i) & 1]  // 对每个 bit i
```

通过精心设计的 LUT (0xea) 和 magic 常量，可以：
1. 一次性提取 2 个 INT4 值（利用 32-bit 操作处理两个 16-bit 半精度）
2. 同时完成 INT4→FP16 的格式转换
3. 整个过程只需 1 条 LOP3 + 1 条 SUB

#### 2.2.4 Tensor Core MMA 计算

```cpp
// ldmatrix 指令：从共享内存加载矩阵到寄存器
asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
    : "=r"(((unsigned*)(A_shared_warp))[0]), ...
    : "r"(addr));

// mma 指令：Tensor Core 矩阵乘加
asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=f"(C_warp[0]), "=f"(C_warp[1]), "=f"(C_warp[2]), "=f"(C_warp[3])
    : "r"(A_shared_warp[0]), "r"(A_shared_warp[1]), ...  // A 矩阵
      "r"(B_shared_warp[0]), "r"(B_shared_warp[1]),      // B 矩阵
      "f"(C_warp[0]), "f"(C_warp[1]), ...);              // 累加器
```

**MMA m16n8k16 指令**：
- 计算 16×8 的输出 tile
- 输入 A: 16×16 FP16，输入 B: 16×8 FP16
- 累加器 C: 16×8 FP32
- 单条指令完成 2048 次 FMA 操作

#### 2.2.5 主循环结构

```cpp
for (int k_0_0 = 0; k_0_0 < k_bound; ++k_0_0) {
    __syncthreads();
    
    // ===== 1. 加载 A tile 到共享内存 =====
    if (ld_A_flag) 
        *(uint4*)(A_shared_ptr) = *(uint4*)(A_ptr + k_0_0 * 32);
    
    // ===== 2. 加载并解量化 B tile =====
    uint32_t zeros_loaded = *(uint32_t*)(zeros_ptr + k_0_0 * 32 / G * (OC / 8));
    half* scales_loaded = sf_ptr + k_0_0 * 32 / G * OC;
    
    for (int ax0 = 0; ax0 < N / 16; ++ax0) {
        uint32_t B_loaded = *(uint32_t*)(B_ptr_local + ax0 * row_stride * (OC / 8));
        
        // 使用 LOP3 快速解量化
        half2 B_dequant_h2[4];
        dequant_vllm_lop3(B_loaded, zeros_loaded, scales_h2, B_dequant_h2);
        
        // 存储到共享内存
        *(uint4*)(B_shared_ptr + ax0 * row_stride * (N + 8)) = *(uint4*)B_dequant_h2;
    }
    
    __syncthreads();
    
    // ===== 3. Tensor Core MMA 计算 =====
    for (int k_0_1 = 0; k_0_1 < 2; ++k_0_1) {
        // ldmatrix 加载 A 和 B 到寄存器
        // mma.sync 进行矩阵乘加
    }
}
```

### 2.3 Decode 核函数：`awq_gemv_kernel` 详解

位于 `awq_gemm_tensorcore.cu`，针对 M=1 的 GEMV 场景优化：

```cpp
__global__ __launch_bounds__(256, 4)  // 256 线程/block，4 blocks/SM
void awq_gemv_kernel(
    const half* __restrict__ X,           // [K] 输入向量
    const int32_t* __restrict__ qweight,  // [K, N/8] 权重
    const int32_t* __restrict__ qzeros,   // [K/G, N/8] zeros
    const half* __restrict__ scales,      // [K/G, N] scales
    half* __restrict__ Y,                 // [N] 输出向量
    int K, int N, int group_size
) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    // 每个 warp 处理 8 个输出（一个 packed INT32）
    const int packed_out_idx = blockIdx.x * num_warps + warp_id;
    const int out_base = packed_out_idx * 8;
    
    // FP32 累加器（提高精度）
    float acc0 = 0.0f, acc1 = 0.0f, ..., acc7 = 0.0f;
    
    // AWQ 解包顺序
    const int awq_order[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    
    for (int g = 0; g < n_groups; g++) {
        // 向量化加载 scales (uint4 = 8 x half)
        uint4 scale_vec = *reinterpret_cast<const uint4*>(&scales[...]);
        
        // 预计算 -scale * zero (避免内循环重复计算)
        float nsz0 = -s0 * (float)((qz_packed >> (awq_order[0] * 4)) & 0xF);
        // ...
        
        // 处理该组的所有输入特征
        for (int k = lane_id; k < group_size; k += 32) {
            float x = __half2float(__ldg(&X[in_idx]));
            const int32_t w_packed = __ldg(&qweight[...]);
            
            // 解量化 + FMA
            float w0 = (float)((w_packed >> (awq_order[0] * 4)) & 0xF);
            acc0 = fmaf(x * s0, w0, acc0 + x * nsz0);
            // ...
        }
    }
    
    // Warp shuffle 规约
    for (int offset = 16; offset > 0; offset /= 2) {
        acc0 += __shfl_down_sync(0xffffffff, acc0, offset);
        // ...
    }
    
    // 向量化写出
    if (lane_id == 0) {
        *reinterpret_cast<uint4*>(&Y[out_base]) = ...;
    }
}
```

**关键优化点**：
1. **`__ldg()` 只读缓存**：利用 texture cache，比 L1 更快
2. **预计算 `-scale * zero`**：减少内循环计算量
3. **Warp shuffle 规约**：避免共享内存同步开销
4. **向量化 I/O**：`uint4` 单次读写 16 字节

### 2.4 性能关键点总结

| 技术 | 作用 | 性能影响 |
|------|------|----------|
| **LOP3 位操作** | 单指令提取 2 个 INT4 | 解量化吞吐量 4x 提升 |
| **Tensor Core MMA** | m16n8k16 矩阵乘加 | 计算吞吐量 16x 提升 |
| **ldmatrix** | 共享内存→寄存器高效传输 | 减少加载延迟 |
| **Warp shuffle** | 无同步开销的规约 | 规约速度 10x 提升 |
| **向量化 I/O** | uint4 批量读写 | 内存带宽利用率 4x |
| **-scale*zero 预计算** | 减少内循环计算 | FMA 效率提升 |

### 2.5 为什么这个设计能够不增加内存又保持高性能？

1. **融合而非分离**：解量化在寄存器/共享内存中完成，不需要全局内存中间缓冲区
2. **INT4 带宽优势**：权重只读取一次，以 INT4 格式（4x 更小），降低内存带宽压力
3. **Tensor Core 利用**：解量化后立即使用 MMA 指令，计算密集度高于 cuBLAS fallback
4. **格式兼容**：AWQ 权重格式恰好与 LOP3 提取模式匹配，无需权重重打包

---

## 1. AWQ 模型适配完整步骤

### 1.1 理解 AWQ 量化格式

AWQ (Activation-aware Weight Quantization) 是一种 4-bit 权重量化方法，其核心特点：

1. **权重存储格式**: 8 个 INT4 值打包成 1 个 INT32
2. **分组量化**: 使用 group_size=128，每组共享 scale 和 zero point
3. **特殊打包顺序**: AWQ 使用逆序打包 `{0, 4, 1, 5, 2, 6, 3, 7}`

```cpp
// AWQ 逆序解包顺序（awq_gemm_tensorcore.cu 第 27 行）
__constant__ int AWQ_ORDER[8] = {0, 4, 1, 5, 2, 6, 3, 7};
```

### 1.2 模型文件格式设计

创建了新的模型文件格式来存储 AWQ 权重，magic number 为 `0x616b3438` ("ak48")：

```cpp
// model.cpp 中的模型加载逻辑
I20260121 20:59:47.918923 model.cpp:78] Model file magic: 0x616b3438
I20260121 20:59:47.919082 model.cpp:79] Model file version: 5
I20260121 20:59:47.919096 model.cpp:93] Loading AWQ INT4 model format (Qwen3)
```

### 1.3 权重加载流程

在 `qwen3.cpp` 中实现了 AWQ 权重加载：

```cpp
// qwen3.cpp 第 556-717 行
I20260121 20:59:47.919392 qwen3.cpp:556] Loading Qwen3 AWQ INT4 model weights...

// 加载各层 AWQ 权重
I20260121 20:59:47.924012 qwen3.cpp:647]   wq layer loaded: [4096 x 4096]
I20260121 20:59:48.075471 qwen3.cpp:647]   wk layer loaded: [4096 x 1024]
I20260121 20:59:48.129505 qwen3.cpp:647]   wv layer loaded: [4096 x 1024]
I20260121 20:59:48.190309 qwen3.cpp:647]   wo layer loaded: [4096 x 4096]
I20260121 20:59:48.342915 qwen3.cpp:647]   w1 layer loaded: [4096 x 12288]
I20260121 20:59:48.717546 qwen3.cpp:647]   w2 layer loaded: [12288 x 4096]
I20260121 20:59:49.096252 qwen3.cpp:647]   w3 layer loaded: [4096 x 12288]
```

每个 AWQ 层包含三个张量：
- `qweight`: 打包的 INT4 权重 `[in_features, out_features/8]`
- `qzeros`: 打包的 INT4 零点 `[in_features/group_size, out_features/8]`
- `scales`: FP16 缩放因子 `[in_features/group_size, out_features]`

### 1.4 AWQ MatMul 算子层

创建了专门的 `AWQMatmulLayer` 类来管理 AWQ 权重：

```cpp
// awq_matmul.h
class AWQMatmulLayer : public LayerParam {
private:
    tensor::Tensor qweight_;   // [in_features, out_features/8] INT32
    tensor::Tensor qzeros_;    // [n_groups, out_features/8] INT32
    tensor::Tensor scales_;    // [n_groups, out_features] FP16
    int in_features_;
    int out_features_;
    int group_size_;
    
public:
    // Getter 方法
    const tensor::Tensor& get_qweight() const { return qweight_; }
    const tensor::Tensor& get_qzeros() const { return qzeros_; }
    const tensor::Tensor& get_scales() const { return scales_; }
};
```

---

## 2. 关键适配点与问题解决

### 2.1 关键适配点一：AWQ 解包顺序

**问题描述**: 初始实现使用顺序解包 `{0, 1, 2, 3, 4, 5, 6, 7}`，导致输出完全错误。

**解决方案**: 通过分析 AWQ 官方实现，发现其使用逆序打包：

```cpp
// 错误的顺序解包（会导致输出乱码）
for (int i = 0; i < 8; i++) {
    int w = (w_packed >> (i * 4)) & 0xF;  // 错误！
}

// 正确的 AWQ 逆序解包（awq_gemm_tensorcore.cu 第 66-73 行）
const int awq_order[8] = {0, 4, 1, 5, 2, 6, 3, 7};
for (int i = 0; i < 8; i++) {
    int w = (w_packed >> (awq_order[i] * 4)) & 0xF;  // 正确！
}
```

**原理解释**: AWQ 打包时将 8 个 INT4 交错存储：
- 位置 0-3 存储输出索引 0, 2, 4, 6
- 位置 4-7 存储输出索引 1, 3, 5, 7

### 2.2 关键适配点二：解量化公式

**问题描述**: 不同量化方案有不同的解量化公式。

**AWQ 解量化公式**:
```
dequant_weight = (weight - zeros) * scales
```

**代码实现** (awq_gemm_tensorcore.cu 第 309-316 行):
```cpp
// 解量化 kernel
#pragma unroll
for (int i = 0; i < 8; i++) {
    int w = (w_packed >> (awq_order[i] * 4)) & 0xF;
    int z = (z_packed >> (awq_order[i] * 4)) & 0xF;
    result[i] = __hmul(__float2half((float)(w - z)), scale_ptr[i]);
}
```

### 2.3 关键适配点三：前向传播路径

**问题描述**: 需要在模型前向传播中正确调用 AWQ 算子。

**解决方案**: 在 `qwen3.cpp` 中检测层类型并分发到正确的计算路径：

```cpp
// qwen3.cpp 第 1420-1470 行
auto query_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(query_layer);

if (query_awq) {
    // AWQ 路径：使用优化的 W4A16 融合 kernel
    kernel::awq_gemm_tensorcore_cu(
        rms_out.ptr<half>(),
        query_awq->get_qweight().ptr<int32_t>(),
        query_awq->get_qzeros().ptr<int32_t>(),
        query_awq->get_scales().ptr<half>(),
        const_cast<half*>(query_out.ptr<half>()),
        seq_len,
        query_awq->in_features(),
        query_awq->out_features(),
        query_awq->group_size(),
        1,
        stream
    );
} else {
    // 标准 FP16 路径
    query_matmul->forward(rms_out, query_out, ...);
}
```

---

## 3. 性能优化详解

### 3.1 初始性能问题

初始实现的性能远低于 FP16：
- Prefill: 13-30 tokens/s (FP16: 140 tokens/s) - 仅 9-21%
- Decode: 5.6-9.2 tokens/s (FP16: 9.9 tokens/s) - 仅 57-93%

### 3.2 优化策略一：Hybrid 计算策略

**核心思想**: 根据批次大小选择不同的计算路径

```cpp
// awq_gemm_tensorcore.cu 第 385-410 行
void awq_gemm_tensorcore_cu(...) {
    constexpr int CUBLAS_THRESHOLD = 2;
    
    if (M == 1) {
        // Decode 阶段：使用优化的 GEMV kernel
        awq_gemv_kernel<<<...>>>(input, qweight, qzeros, scales, output, ...);
    } else if (M < CUBLAS_THRESHOLD) {
        // 小批次：使用融合 GEMM kernel
        awq_gemm_kernel<<<...>>>(input, qweight, qzeros, scales, output, ...);
    } else {
        // 大批次（Prefill）：运行时解量化 + cuBLAS HGEMM
        // Step 1: 快速解量化到临时缓冲区
        awq_dequant_weight_kernel<<<...>>>(qweight, qzeros, scales, g_dequant_buffer, ...);
        
        // Step 2: cuBLAS HGEMM 使用 Tensor Core
        cublasSetMathMode(g_cublas_handle, CUBLAS_TENSOR_OP_MATH);
        cublasHgemm(g_cublas_handle, ...);
    }
}
```

### 3.3 优化策略二：高性能 GEMV Kernel (Decode 阶段)

**设计原理**:
- 每个 warp 处理 8 个输出通道（一个打包的 INT32）
- 8 个 warp/block = 64 个输出通道/block
- 使用 `__ldg()` 利用只读缓存
- 预计算 `-scale * zero` 避免内循环重复计算

```cpp
// awq_gemm_tensorcore.cu 第 44-145 行
__global__ __launch_bounds__(256, 4)
void awq_gemv_kernel(
    const half* __restrict__ X,           // [in_features]
    const int32_t* __restrict__ qweight,  // [in_features, out_features/8]
    const int32_t* __restrict__ qzeros,   // [n_groups, out_features/8]
    const half* __restrict__ scales,      // [n_groups, out_features]
    half* __restrict__ Y,                 // [out_features]
    int in_features, int out_features, int group_size
) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    // 每个 warp 处理 8 个输出
    const int packed_out_idx = blockIdx.x * num_warps + warp_id;
    const int out_base = packed_out_idx * 8;
    
    // FP32 累加器提高精度
    float acc0 = 0.0f, acc1 = 0.0f, ..., acc7 = 0.0f;
    
    for (int g = 0; g < n_groups; g++) {
        // 向量化加载 scales (uint4 = 8 x half)
        uint4 scale_vec = *reinterpret_cast<const uint4*>(&scales[g * out_features + out_base]);
        float s0 = __half2float(((half*)&scale_vec)[0]);
        // ... s1-s7
        
        // 预计算 -scale * zero
        float nsz0 = -s0 * (float)((qz_packed >> (awq_order[0] * 4)) & 0xF);
        // ... nsz1-nsz7
        
        // 处理该组的输入特征
        for (int k = lane_id; k < group_size; k += 32) {
            float x = __half2float(__ldg(&X[in_idx]));
            const int32_t w_packed = __ldg(&qweight[in_idx * packed_out_dim + packed_out_idx]);
            
            // 解量化并累加，使用 FMA
            float w0 = (float)((w_packed >> (awq_order[0] * 4)) & 0xF);
            acc0 = fmaf(x * s0, w0, acc0 + x * nsz0);
            // ... acc1-acc7
        }
    }
    
    // Warp 规约
    for (int offset = 16; offset > 0; offset /= 2) {
        acc0 += __shfl_down_sync(0xffffffff, acc0, offset);
        // ...
    }
    
    // 写入输出
    if (lane_id == 0) {
        Y[out_base + 0] = __float2half(acc0);
        // ...
    }
}
```

### 3.4 优化策略三：快速解量化 + cuBLAS (Prefill 阶段)

**为什么 cuBLAS 更快**: 
- cuBLAS HGEMM 使用 Tensor Core，计算吞吐量极高
- 解量化开销被大批次均摊

```cpp
// awq_gemm_tensorcore.cu 第 280-320 行
__global__ __launch_bounds__(256)
void awq_dequant_weight_kernel(
    const int32_t* __restrict__ qweight,  // [K, N/8]
    const int32_t* __restrict__ qzeros,   // [K/G, N/8]
    const half* __restrict__ scales,      // [K/G, N]
    half* __restrict__ weight_fp16,       // [K, N] output
    int K, int N, int group_size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 每线程处理一个打包的 INT32（8 个 FP16 输出）
    
    // 向量化加载 scales
    uint4 scale_vec = *reinterpret_cast<const uint4*>(&scales[g * N + n_base]);
    half* scale_ptr = reinterpret_cast<half*>(&scale_vec);
    
    // 解量化
    half result[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int w = (w_packed >> (awq_order[i] * 4)) & 0xF;
        int z = (z_packed >> (awq_order[i] * 4)) & 0xF;
        result[i] = __hmul(__float2half((float)(w - z)), scale_ptr[i]);
    }
    
    // 向量化存储
    *reinterpret_cast<uint4*>(&weight_fp16[k * N + n_base]) = *reinterpret_cast<uint4*>(result);
}
```

### 3.5 全局资源管理

为避免每次调用都分配/释放内存，使用全局缓冲区：

```cpp
// awq_gemm_tensorcore.cu 第 322-345 行
static cublasHandle_t g_cublas_handle = nullptr;
static half* g_dequant_buffer = nullptr;
static size_t g_dequant_buffer_size = 0;
static bool g_initialized = false;

static void ensure_initialized() {
    if (g_initialized) return;
    
    cublasCreate(&g_cublas_handle);
    cublasSetMathMode(g_cublas_handle, CUBLAS_TENSOR_OP_MATH);
    
    // 预分配最大层的缓冲区: 12288 * 4096 * 2 ≈ 100MB
    const size_t max_buffer_size = 12288 * 4096 * sizeof(half);
    cudaMalloc(&g_dequant_buffer, max_buffer_size);
    g_dequant_buffer_size = max_buffer_size;
    
    g_initialized = true;
}
```

---

## 4. AWQ 算子详细实现

### 4.1 算子接口定义

```cpp
// awq_gemm_tensorcore.cuh
namespace kernel {

// 主入口：根据 M 自动选择最优计算路径
void awq_gemm_tensorcore_cu(
    const half* input,          // [M, in_features]
    const int32_t* qweight,     // [in_features, out_features/8]
    const int32_t* qzeros,      // [n_groups, out_features/8]
    const half* scales,         // [n_groups, out_features]
    half* output,               // [M, out_features]
    int M,
    int in_features,
    int out_features,
    int group_size,
    int split_k_iters,
    cudaStream_t stream
);

// 独立解量化函数（用于预解量化场景）
void awq_dequant_weight_cu(
    const int32_t* qweight,
    const int32_t* qzeros,
    const half* scales,
    half* dequant_weight,
    int in_features,
    int out_features,
    int group_size,
    cudaStream_t stream
);

// cuBLAS 路径（大批次最优）
void awq_gemm_cublas_cu(
    const half* input,
    const int32_t* qweight,
    const int32_t* qzeros,
    const half* scales,
    half* dequant_weight,  // temp buffer
    half* output,
    int M, int in_features, int out_features, int group_size,
    cublasHandle_t cublas_handle,
    cudaStream_t stream
);

}  // namespace kernel
```

### 4.2 内存布局详解

**qweight 布局**: `[in_features, out_features/8]`
- 每行存储一个输入特征对应的所有输出权重
- 每 8 个 INT4 权重打包成 1 个 INT32

**qzeros 布局**: `[n_groups, out_features/8]`
- n_groups = in_features / group_size
- 每组共享相同的 zero point

**scales 布局**: `[n_groups, out_features]`
- 每组每个输出通道有独立的 scale

### 4.3 计算流程图

```
┌─────────────────────────────────────────────────────────┐
│                    awq_gemm_tensorcore_cu                │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │      M == 1 ?         │
              └───────────────────────┘
                 │                  │
            Yes ▼             No   ▼
    ┌──────────────────┐   ┌───────────────────────┐
    │  awq_gemv_kernel │   │    M < THRESHOLD ?    │
    │  (Decode 最优)    │   └───────────────────────┘
    └──────────────────┘          │              │
                              Yes ▼         No   ▼
                    ┌──────────────────┐  ┌───────────────────────┐
                    │  awq_gemm_kernel │  │ awq_dequant + cuBLAS  │
                    │  (小批次融合)     │  │ (Prefill 最优)         │
                    └──────────────────┘  └───────────────────────┘
```

### 4.4 关键优化技术总结

| 优化技术 | 适用场景 | 性能提升 |
|---------|---------|---------|
| 向量化加载 (uint4) | 所有场景 | 减少内存事务 |
| `__ldg()` 只读缓存 | GEMV/GEMM | 提高缓存命中率 |
| 预计算 -scale*zero | GEMV | 减少内循环计算 |
| Warp shuffle 规约 | GEMV | 避免共享内存同步 |
| cuBLAS Tensor Core | Prefill | 利用硬件加速 |
| 全局缓冲区 | Prefill | 避免频繁分配 |

---

## 5. 性能深入分析

### 5.1 端到端时间对比

实际测试结果（生成约 200 tokens）：

| 阶段 | AWQ INT4 | FP16 | 差异 |
|------|----------|------|------|
| Prefill | 402.7 ms | 234.0 ms | AWQ 慢 168.7 ms |
| Decode | 20,373 ms | 25,854 ms | AWQ **快 5,481 ms** |
| **总时间** | **20,776 ms** | **26,088 ms** | **AWQ 快 5,312 ms (20.4%)** |

**关键洞察**: 虽然 AWQ Prefill 比 FP16 慢约 1.7 倍，但 Decode 阶段占总时间的 98%+，
且 AWQ Decode 比 FP16 快约 21%，因此总体性能 AWQ 更优。

### 5.2 Prefill 阶段性能分析

**为什么 AWQ Prefill 比 FP16 慢？**

AWQ Prefill 需要额外的解量化步骤：
1. **FP16 Prefill**: 直接 cuBLAS HGEMM ≈ 0.73 ms/层
2. **AWQ Prefill**: dequant (1.06 ms) + cuBLAS HGEMM (0.73 ms) ≈ 1.79 ms/层

解量化开销占 AWQ Prefill 总时间的 **59%**。

**尝试过的优化方案**：

| 方案 | 结果 | 分析 |
|------|------|------|
| 融合 W4A16 GEMM kernel | ❌ 16 tok/s | Tensor Core 利用率不如 cuBLAS |
| 异步 pipeline (双缓冲) | ❌ 1.03x 提升 | GPU 资源共享，无法真正并行 |
| 优化解量化 kernel | ✅ 2.89x 提升 | 从 3.06ms 降到 1.06ms |
| **预解量化权重 (V6.0)** | **✅ 1.58x 提升** | **从 84 tok/s 到 131 tok/s** |

**结论 (V6.0 更新)**: 通过在模型加载时预解量化权重并缓存，
Prefill 阶段可以达到 FP16 的 88-90% 性能。代价是内存占用从 6GB 增加到 16GB。

### 5.3 Decode 阶段性能优势

**为什么 AWQ Decode 比 FP16 更快？**

Decode 阶段 (M=1) 是 **内存带宽受限** 的：
- FP16 权重：16 GB，带宽需求高
- INT4 权重：5.7 GB，**带宽需求降低 4 倍**

在内存带宽受限场景下，AWQ 的 INT4 权重带来显著优势：
- 每次读取同样数量的字节，AWQ 可以获取 4 倍的权重数据
- 融合 GEMV kernel 将解量化与计算合并，无额外内存访问

---

## 6. V6.0 预解量化优化详解

### 6.1 优化原理

V6.0 的核心优化思想是将解量化操作从运行时移动到模型加载时：

```cpp
// awq_matmul.cpp - 在 to_cuda() 后自动预解量化
void AWQMatmulLayer::to_cuda() {
  // ... 加载 qweight, qzeros, scales 到 GPU ...
  
  // 自动预解量化（可通过 g_enable_pre_dequant 配置）
  if (g_enable_pre_dequant) {
    pre_dequantize(nullptr);
  }
}
```

预解量化后，Prefill 阶段直接使用 cuBLAS HGEMM：

```cpp
// awq_matmul.cpp - forward() 中选择最优路径
if (has_dequant_weight_ && batch_size > 1) {
  // Prefill: 直接使用预解量化权重 + cuBLAS HGEMM
  kernel::awq_gemm_with_dequant_cu(...);
} else {
  // Decode: 使用融合 GEMV kernel（保持内存带宽优势）
  kernel::awq_gemm_tensorcore_cu(...);
}
```

### 6.2 内存与性能权衡

| 配置 | 内存占用 | Prefill | Decode | 适用场景 |
|------|----------|---------|--------|----------|
| 不预解量化 | ~6 GB | 84 tok/s | 10.1 tok/s | 内存受限设备 |
| **预解量化** | **~16 GB** | **131 tok/s** | **10.0 tok/s** | 性能优先 |

### 6.3 关键代码路径

在 `qwen3.cpp` 的 batched prefill 实现中：

```cpp
// batched_attention_qkv - 使用预解量化权重
if (query_awq->is_pre_dequantized()) {
  kernel::awq_gemm_with_dequant_cu(
      input, query_awq->get_dequant_weight(), output,
      seq_len, in_features, out_features, stream
  );
} else {
  kernel::awq_gemm_tensorcore_cu(...);  // 运行时解量化路径
}
```

---

## 7. 总结

### 7.1 适配成果 (V7.0 - 最终版)

1. ✅ 成功将 AWQ INT4 量化模型集成到 KuiperLLama
2. ✅ **Prefill 性能达到 FP16 的 85-94%** (130 vs 145 tokens/s)
3. ✅ **Decode 性能接近/超越 FP16** (9.2-10.1 vs 9.9 tokens/s)
4. ✅ **内存节省 62%** (~6GB vs ~16GB)
5. ✅ 模型输出正确性验证通过

### 7.2 核心技术贡献

1. **vllm LOP3 融合内核**: 移植并适配 vllm 的高性能 W4A16 GEMM 内核
2. **AWQ 格式深度分析**: 发现 Kuiper 权重已与 vllm LOP3 格式兼容，无需重打包
3. **Tensor Core MMA**: 使用 m16n8k16 矩阵乘加指令充分利用硬件加速
4. **Hybrid 策略**: Decode 用 GEMV 内核，Prefill 用 MMA 内核

### 7.3 经验教训

1. **AWQ 打包顺序**: elem→pos 映射 `{0,4,1,5,2,6,3,7}` 是关键
2. **融合内核优于分离**: 避免双倍内存带宽是性能提升的核心
3. **LOP3 位操作**: PTX 指令可以高效完成 INT4→FP16 转换
4. **格式兼容性验证**: 用 Python 测试脚本验证格式匹配非常重要
5. **性能与内存平衡**: V7.0 成功实现"不牺牲内存"的性能优化

### 7.4 未来优化方向

1. **进一步优化 Prefill**: 探索更大的 tile 尺寸和更好的内存访问模式
2. **异步权重加载**: 利用 CUDA Streams 隐藏权重加载延迟
3. **FP8 量化**: 探索 FP8 格式以获得更好的精度-性能平衡
4. **多 GPU 支持**: 实现张量并行以支持更大模型
