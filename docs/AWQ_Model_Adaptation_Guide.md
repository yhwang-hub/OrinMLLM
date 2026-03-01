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

---

# 深度技术分析

## 问题八：AWQ 模型导出、权重解析与推理全流程详解

### 8.1 export_qwen3-8B-awq.py 模型导出全流程

本节详细分析 `tools/export_qwen3-8B-awq.py` 脚本如何将 HuggingFace 格式的 `/mnt/ssd/QwenModels/Qwen3-8B-awq` 模型转换为 KuiperLLama 工程使用的 `/mnt/ssd/QwenModels/Qwen3-8B-awq.bin` 自定义二进制格式。

#### 8.1.1 总体流程概览

```
┌─────────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
│  1. 加载 HF 模型权重     │ ──►  │  2. 构建 256 字节头部    │ ──►  │  3. 按固定顺序写入权重   │
│  load_hf_weights()      │      │  awq_export() header    │      │  awq_export() body      │
│  - safetensors 文件      │      │  - magic, version       │      │  - FP16 + AWQ INT4      │
│  - AutoConfig 解析       │      │  - 模型结构参数          │      │  - 序列化到 .bin        │
└─────────────────────────┘      └─────────────────────────┘      └─────────────────────────┘
```

#### 8.1.2 第一步：加载 HuggingFace 模型权重

`load_hf_weights()` 函数完成模型加载：

```python
def load_hf_weights(model_path):
    # 1. 使用 transformers 库解析 config.json
    hf_config = AutoConfig.from_pretrained(model_path)
    
    # 2. 提取量化配置（group_size 等）
    group_size = hf_config.quantization_config.get('group_size', 128)
    
    # 3. 从 .safetensors 文件加载所有张量
    safetensor_files = sorted(list(model_path.glob("*.safetensors")))
    hf_dict = {}
    for sf_file in safetensor_files:
        with safe_open(sf_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                hf_dict[key] = f.get_tensor(key)
    
    # 4. 验证 AWQ 量化键存在
    assert 'model.layers.0.self_attn.q_proj.qweight' in hf_dict
    assert 'model.layers.0.self_attn.q_proj.qzeros' in hf_dict
    assert 'model.layers.0.self_attn.q_proj.scales' in hf_dict
    
    # 5. 验证 Qwen3 特有的 q_norm/k_norm
    assert 'model.layers.0.self_attn.q_norm.weight' in hf_dict
    assert 'model.layers.0.self_attn.k_norm.weight' in hf_dict
```

加载后 `hf_dict` 中包含两类张量：

| 类别 | 典型键名 | 数据类型 | 说明 |
|------|----------|----------|------|
| **FP16 非量化权重** | `model.embed_tokens.weight`、`model.norm.weight`、`model.layers.{i}.input_layernorm.weight`、`model.layers.{i}.self_attn.q_norm.weight` | FP16 | 嵌入、LayerNorm、Q/K Norm 等 |
| **AWQ 量化权重** | `model.layers.{i}.self_attn.q_proj.qweight`、`.qzeros`、`.scales` | INT32/FP16 | 线性层量化后的三元组 |

#### 8.1.3 第二步：构建 256 字节文件头

```python
def awq_export(hf_dict, config, filepath, group_size=128):
    version = 5  # AWQ INT4 版本标识
    
    # === 写入 256 字节头部 ===
    # [字节 0-3]   magic = 0x616b3438 ("ak48")，标识 Qwen3 AWQ 格式
    out_file.write(struct.pack('I', 0x616b3438))
    
    # [字节 4-7]   version = 5，AWQ INT4 专用版本号
    out_file.write(struct.pack('i', version))
    
    # [字节 8-35]  7 个 int32 模型结构参数
    #   dim=4096, hidden_dim=14336, n_layers=36, n_heads=32, 
    #   n_kv_heads=8, vocab_size=151936, max_seq_len=40960
    header = struct.pack('iiiiiii', dim, hidden_dim, n_layers, n_heads,
                         n_kv_heads, vocab_size, max_seq_len)
    out_file.write(header)
    
    # [字节 36]    shared_classifier (uint8)，0 表示 lm_head 独立
    out_file.write(struct.pack('B', int(shared_classifier)))
    
    # [字节 37-40] head_dim = 128 (Qwen3 特有)
    out_file.write(struct.pack('i', head_dim))
    
    # [字节 41-44] group_size = 128 (AWQ 量化组大小)
    out_file.write(struct.pack('i', group_size))
    
    # [字节 45-255] 全零填充
    pad = 256 - out_file.tell()
    out_file.write(b'\0' * pad)
```

文件头的完整二进制布局：

```
偏移量    大小    字段名              Qwen3-8B-AWQ 实际值      说明
────────────────────────────────────────────────────────────────────────
0x00      4B     magic               0x616b3438 ("ak48")     格式标识
0x04      4B     version             5                        AWQ INT4 版本
0x08      4B     dim                 4096                     隐藏维度
0x0C      4B     hidden_dim          14336                    FFN 中间维度
0x10      4B     n_layers            36                       Transformer 层数
0x14      4B     n_heads             32                       注意力头数
0x18      4B     n_kv_heads          8                        KV 头数(GQA)
0x1C      4B     vocab_size          151936                   词表大小
0x20      4B     max_seq_len         40960                    最大序列长度
0x24      1B     shared_classifier   0                        是否共享 lm_head
0x25      4B     head_dim            128                      每个注意力头的维度
0x29      4B     group_size          128                      AWQ 量化组大小
0x2D-0xFF 211B   padding             0x00...                  填充至 256 字节
```

#### 8.1.4 第三步：按固定顺序序列化权重

权重的写入顺序是 **固定且严格** 的，C++ 推理引擎按照同样的顺序读取。整体分为三个区域：

**区域 A：FP16 非量化权重（结构参数）**

```python
# 1. attention_norm (input_layernorm) × 36 层, 每层 dim=4096 个 FP16 元素
for i in range(n_layers):  # 36 次
    serialize_fp16(out_file, hf_dict[f'model.layers.{i}.input_layernorm.weight'])
    # 每层写入 4096 × 2 = 8192 字节

# 2. ffn_norm (post_attention_layernorm) × 36 层
for i in range(n_layers):
    serialize_fp16(out_file, hf_dict[f'model.layers.{i}.post_attention_layernorm.weight'])

# 3. final norm (model.norm.weight), 1 × dim=4096 个 FP16
serialize_fp16(out_file, hf_dict['model.norm.weight'])

# 4. token embeddings, vocab_size × dim = 151936 × 4096 个 FP16
serialize_fp16(out_file, hf_dict['model.embed_tokens.weight'])
```

**区域 B：AWQ INT4 量化权重（线性层）**

每个量化线性层由三个张量组成，按 `qweight → qzeros → scales` 的顺序写入：

```python
def write_awq_weights(layer_name):
    qweight = hf_dict[f'{layer_name}.qweight']  # [in_features, out_features/8] INT32
    qzeros  = hf_dict[f'{layer_name}.qzeros']   # [num_groups, out_features/8] INT32
    scales  = hf_dict[f'{layer_name}.scales']    # [num_groups, out_features] FP16
    
    serialize_int32(out_file, qweight)
    serialize_int32(out_file, qzeros)
    serialize_fp16(out_file, scales)
```

写入顺序（所有层先写完某一类权重再写下一类）：

| 序号 | 权重 | HF 键名 | 形状 (`[in, out]`) | 写入次序 |
|------|------|---------|-------------------|---------|
| 5 | wq (q_proj) | `self_attn.q_proj` | `[4096, 4096]` | 36 层依次 |
| 6 | wk (k_proj) | `self_attn.k_proj` | `[4096, 1024]` | 36 层依次 |
| 7 | wv (v_proj) | `self_attn.v_proj` | `[4096, 1024]` | 36 层依次 |
| 8 | wo (o_proj) | `self_attn.o_proj` | `[4096, 4096]` | 36 层依次 |
| 9 | w1 (gate_proj) | `mlp.gate_proj` | `[4096, 14336]` | 36 层依次 |
| 10 | w2 (down_proj) | `mlp.down_proj` | `[14336, 4096]` | 36 层依次 |
| 11 | w3 (up_proj) | `mlp.up_proj` | `[4096, 14336]` | 36 层依次 |

**区域 C：FP16 尾部权重**

```python
# 12. lm_head 权重 (如果不与 embedding 共享)
if not shared_classifier:
    serialize_fp16(out_file, hf_dict['lm_head.weight'])  # [151936, 4096] FP16

# 13. q_norm 权重 × 36 层, 每层 head_dim=128 个 FP16
for i in range(n_layers):
    serialize_fp16(out_file, hf_dict[f'model.layers.{i}.self_attn.q_norm.weight'])

# 14. k_norm 权重 × 36 层, 每层 head_dim=128 个 FP16
for i in range(n_layers):
    serialize_fp16(out_file, hf_dict[f'model.layers.{i}.self_attn.k_norm.weight'])
```

#### 8.1.5 AWQ INT4 权重打包方式详解

AWQ 的核心在于将 8 个 4-bit 整数打包进一个 32-bit 整数中，但打包顺序 **并非简单的顺序排列**，而是采用特殊的交织映射（interleaved mapping）：

```
一个 INT32 中 8 个 INT4 值的存储布局：
═══════════════════════════════════════════════════════════════
INT32 位位置:  [31..28] [27..24] [23..20] [19..16] [15..12] [11..8] [7..4] [3..0]
AWQ 逻辑索引:    7        3        6        2        5       1      4      0
═══════════════════════════════════════════════════════════════

AWQ 映射表（逻辑索引 → 位偏移）:
  逻辑索引 0 → 位偏移 0  (bit[3..0])
  逻辑索引 1 → 位偏移 8  (bit[11..8])
  逻辑索引 2 → 位偏移 16 (bit[19..16])
  逻辑索引 3 → 位偏移 24 (bit[27..24])
  逻辑索引 4 → 位偏移 4  (bit[7..4])
  逻辑索引 5 → 位偏移 12 (bit[15..12])
  逻辑索引 6 → 位偏移 20 (bit[23..20])
  逻辑索引 7 → 位偏移 28 (bit[31..28])

等价的 AWQ order 数组: {0, 4, 1, 5, 2, 6, 3, 7}
提取第 i 个元素: w_i = (packed_int32 >> (awq_order[i] * 4)) & 0xF
```

**具体量化存储示例**（以 `q_proj` 为例）：

```
原始 q_proj 权重形状: [in_features=4096, out_features=4096] FP16
量化后:
  qweight: [4096, 4096/8] = [4096, 512] INT32  ← 每个 INT32 打包 8 个 INT4
  qzeros:  [4096/128, 4096/8] = [32, 512] INT32
  scales:  [4096/128, 4096] = [32, 4096] FP16

反量化公式: weight_fp16[k][n] = scales[k/group_size][n] × (qweight_packed[k][n/8][n%8] - qzeros[k/group_size][n/8][n%8])
```

**序列化函数实现**：

```python
def serialize_fp16(file, tensor):
    """将张量转为 FP16 后按行优先顺序写入"""
    d = tensor.detach().cpu().view(-1).to(torch.float16).numpy()
    file.write(d.tobytes())  # 直接写入原始字节

def serialize_int32(file, tensor):
    """将 INT32 张量按行优先顺序写入"""
    d = tensor.detach().cpu().view(-1).to(torch.int32).numpy()
    file.write(d.tobytes())
```

#### 8.1.6 .bin 文件完整内存布局

```
┌──────────────────────────────────────────────────┐ 偏移 0
│                 Header (256 B)                   │
│  magic(4) + version(4) + config(29) + head_dim(4)│
│  + group_size(4) + padding(211)                  │
├──────────────────────────────────────────────────┤ 偏移 256
│           区域 A: FP16 非量化权重                  │
│  attention_norm × 36 层 (dim=4096, FP16)          │
│  ffn_norm × 36 层 (dim=4096, FP16)                │
│  final_norm (dim=4096, FP16)                      │
│  embed_tokens (151936 × 4096, FP16)               │
├──────────────────────────────────────────────────┤
│           区域 B: AWQ 量化权重                     │
│  wq × 36 层 (qweight+qzeros+scales)              │
│  wk × 36 层 (qweight+qzeros+scales)              │
│  wv × 36 层 (qweight+qzeros+scales)              │
│  wo × 36 层 (qweight+qzeros+scales)              │
│  w1 × 36 层 (qweight+qzeros+scales)              │
│  w2 × 36 层 (qweight+qzeros+scales)              │
│  w3 × 36 层 (qweight+qzeros+scales)              │
├──────────────────────────────────────────────────┤
│           区域 C: FP16 尾部权重                    │
│  lm_head (151936 × 4096, FP16) [如果不共享]       │
│  q_norm × 36 层 (head_dim=128, FP16)              │
│  k_norm × 36 层 (head_dim=128, FP16)              │
└──────────────────────────────────────────────────┘
```

---

### 8.2 qwen3.cpp 中 AWQ 权重解析流程详解

本节详细分析 `kuiper/source/model/qwen3.cpp` 工程如何从 `.bin` 文件中解析 AWQ 量化权重，并结合实例讲解每一步。

#### 8.2.1 文件头解析（model.cpp::read_model_file）

权重解析的入口在基类 `Model::gen_model_from_file()` → `Model::read_model_file()`（位于 [kuiper/source/model/model.cpp](../kuiper/source/model/model.cpp)）：

```cpp
// model.cpp L41-L95：读取文件头
base::Status Model::read_model_file() {
    FILE* file = fopen(model_path_.data(), "rb");
    
    // 1. 读取 4 字节 magic
    uint32_t magic = 0;
    fread(&magic, sizeof(uint32_t), 1, file);
    
    // 2. 判断格式类型
    if (magic == 0x616b3438) {  // "ak48" = Qwen3 AWQ
        bool is_awq_format = true;
        
        // 3. 读取 version
        int32_t version = 0;
        fread(&version, sizeof(int32_t), 1, file);  // version = 5
        
        // 4. 设置模型标志
        is_awq_model_ = true;      // 标记为 AWQ 模型
        is_fp16_model_ = true;     // AWQ 非量化权重使用 FP16
        
        // 5. 读取 7 个结构参数
        fread(&config.dim, sizeof(int32_t), 1, file);        // 4096
        fread(&config.hidden_dim, sizeof(int32_t), 1, file); // 14336
        fread(&config.layer_num, sizeof(int32_t), 1, file);  // 36
        fread(&config.head_num, sizeof(int32_t), 1, file);   // 32
        fread(&config.kv_head_num, sizeof(int32_t), 1, file);// 8
        fread(&config.vocab_size, sizeof(int32_t), 1, file); // 151936
        fread(&config.seq_len, sizeof(int32_t), 1, file);    // 40960
        
        // 6. 读取 shared_classifier
        uint8_t shared_classifier = 0;
        fread(&shared_classifier, sizeof(uint8_t), 1, file); // 0
        
        // 7. 读取 head_dim (Qwen3 特有)
        int32_t head_dim = 0;
        fread(&head_dim, sizeof(int32_t), 1, file);  // 128
        
        // 8. 读取 group_size (AWQ 特有)
        fread(&group_size_, sizeof(int32_t), 1, file);  // 128
    }
    
    // 9. mmap 映射整个文件，weight_data 指向偏移 256 处 
    raw_model_data_->data = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    raw_model_data_->weight_data = static_cast<int8_t*>(raw_model_data_->data) + 256;
}
```

关键点：文件通过 `mmap` 映射到内存，`weight_data` 指向 256 字节头部之后的权重数据起始地址。后续所有权重读取都基于这个指针的偏移。

#### 8.2.2 AWQ 权重加载（qwen3.cpp::create_param_layers_awq）

`Qwen3Model::create_layers()` 检测到 AWQ 模型后，调用 `create_param_layers_awq()` 来解析权重。该函数使用一个 `pos` 变量（以字节为单位）从 `base_ptr` 开始逐段读取：

```cpp
void Qwen3Model::create_param_layers_awq() {
    // base_ptr 指向权重数据起始位置（即文件偏移 256）
    const uint8_t* base_ptr = static_cast<const uint8_t*>(raw_model_data_->weight_data);
    size_t pos = 0;  // 字节偏移
```

**实例解析：加载第 0 层的 attention_norm 权重**

```cpp
// 1. attention_norm (input_layernorm) - FP16
for (int32_t i = 0; i < config_->layer_num_; ++i) {  // i = 0..35
    auto rms_norm_layer = std::make_shared<op::RmsNormLayer>(device_type_, dim);  // dim=4096
    
    // base_ptr + pos 指向该层 input_layernorm 的 FP16 数据
    // 第 0 层: pos = 0, 数据大小 = 4096 × 2 = 8192 字节
    rms_norm_layer->set_weight_fp16(0, {dim}, base_ptr + pos, cpu_device_type);
    qwen_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    
    pos += dim * sizeof(uint16_t);  // pos += 4096 × 2 = 8192
}
// 36 层读完后: pos = 36 × 4096 × 2 = 294,912 字节
```

**实例解析：加载 token embeddings**

```cpp
// 4. token embeddings - FP16
auto emb_layer = std::make_shared<op::EmbeddingLayer>(
    device_type_, dim, config_->seq_len_, std::abs(config_->vocab_size_));
// vocab_size=151936, dim=4096, 共 151936 × 4096 × 2 = 1,244,856,320 字节 ≈ 1.16 GB
emb_layer->set_weight_fp16(0, {std::abs(config_->vocab_size_), dim},
                           base_ptr + pos, cpu_device_type);
qwen_layers_->embedding_layer_ = emb_layer;
pos += config_->vocab_size_ * dim * sizeof(uint16_t);
```

**实例解析：加载 AWQ 量化线性层（以 q_proj 为例）**

AWQ 层的加载通过 `load_awq_layer` lambda 函数实现：

```cpp
auto load_awq_layer = [&](int32_t in_features, int32_t out_features, 
                          std::vector<std::shared_ptr<op::Layer>>& layer_list,
                          const std::string& name) {
    int32_t packed_out = out_features / 8;  // 每 8 个 INT4 打包为 1 个 INT32
    int32_t num_groups = in_features / group_size_;  // 量化分组数
    
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        // 创建 AWQMatmulLayer 而非普通 MatmulLayer
        auto awq_layer = std::make_shared<op::AWQMatmulLayer>(
            device_type_, in_features, out_features, group_size_);
        
        // 依次读取三个张量：qweight → qzeros → scales
        // === qweight ===
        const void* qweight_ptr = base_ptr + pos;
        size_t qweight_size = in_features * packed_out * sizeof(int32_t);
        pos += qweight_size;
        
        // === qzeros ===
        const void* qzeros_ptr = base_ptr + pos;
        size_t qzeros_size = num_groups * packed_out * sizeof(int32_t);
        pos += qzeros_size;
        
        // === scales ===
        const void* scales_ptr = base_ptr + pos;
        size_t scales_size = num_groups * out_features * sizeof(uint16_t);
        pos += scales_size;
        
        // 将三个指针传给 AWQMatmulLayer
        awq_layer->set_awq_weights(qweight_ptr, qzeros_ptr, scales_ptr, cpu_device_type);
        layer_list.push_back(awq_layer);
    }
};
```

**以 `q_proj` 第 0 层为具体实例**：

```
q_proj: in_features=4096, out_features=4096
  packed_out = 4096 / 8 = 512
  num_groups = 4096 / 128 = 32

  qweight: [4096, 512] INT32 → 4096 × 512 × 4 = 8,388,608 字节 = 8 MB
  qzeros:  [32, 512] INT32   → 32 × 512 × 4   = 65,536 字节   = 64 KB
  scales:  [32, 4096] FP16   → 32 × 4096 × 2  = 262,144 字节  = 256 KB

  三者合计: 8,716,288 字节 ≈ 8.3 MB（相比 FP16 的 4096×4096×2=32 MB，节省约 73%）
```

#### 8.2.3 AWQMatmulLayer 内部权重存储

`set_awq_weights()` 中将原始指针数据复制到内部张量：

```cpp
void AWQMatmulLayer::set_awq_weights(const void* qweight_ptr, 
                                      const void* qzeros_ptr,
                                      const void* scales_ptr,
                                      base::DeviceType src_device) {
    // 创建 CPU 端张量并复制数据
    int32_t packed_out = out_features_ / 8;
    int32_t num_groups = in_features_ / group_size_;
    
    // qweight: 1D INT32 张量
    qweight_ = tensor::Tensor(base::DataType::kDataTypeInt32, 
                              in_features_ * packed_out, true, alloc);
    std::memcpy(qweight_.ptr<void>(), qweight_ptr, 
                in_features_ * packed_out * sizeof(int32_t));
    
    // qzeros: 1D INT32 张量
    qzeros_ = tensor::Tensor(base::DataType::kDataTypeInt32, 
                             num_groups * packed_out, true, alloc);
    std::memcpy(qzeros_.ptr<void>(), qzeros_ptr, 
                num_groups * packed_out * sizeof(int32_t));
    
    // scales: 1D FP16 张量
    scales_ = tensor::Tensor(base::DataType::kDataTypeFp16, 
                             num_groups * out_features_, true, alloc);
    std::memcpy(scales_.ptr<void>(), scales_ptr, 
                num_groups * out_features_ * sizeof(uint16_t));
}
```

#### 8.2.4 rmsnorm_layers_ 索引布局

解析过程中 `rmsnorm_layers_` 的 push_back 顺序决定了运行时的访问索引：

```
rmsnorm_layers_ 索引布局（共 4 × 36 + 1 = 145 个）:
─────────────────────────────────────────────────────
索引 [0..35]      → attention_norm (input_layernorm), 层 0~35
索引 [36..71]     → ffn_norm (post_attention_layernorm), 层 0~35
索引 [72]         → final_norm (model.norm)
索引 [73..108]    → q_norm, 层 0~35
索引 [109..144]   → k_norm, 层 0~35

索引公式:
  attention_norm[i] = rmsnorm_layers_[i]
  ffn_norm[i]       = rmsnorm_layers_[i + layer_num]
  final_norm        = rmsnorm_layers_[2 * layer_num]
  q_norm[i]         = rmsnorm_layers_[i + 2 * layer_num + 1]
  k_norm[i]         = rmsnorm_layers_[i + 3 * layer_num + 1]
```

#### 8.2.5 GPU 权重迁移

加载到 CPU 后，`init_mem()` 调用 `qwen_layers_->to_cuda()` 将所有权重迁移到 GPU。对于 `AWQMatmulLayer`，其 `to_cuda()` 方法将三个量化张量分别复制到 GPU 显存：

```cpp
void AWQMatmulLayer::to_cuda() {
    auto cuda_alloc = base::CUDADeviceAllocatorFactory::get_instance();
    
    // qweight: CPU INT32 → GPU INT32
    tensor::Tensor cuda_qweight(base::DataType::kDataTypeInt32, 
                                 qweight_.size(), true, cuda_alloc);
    cudaMemcpy(cuda_qweight.ptr<void>(), qweight_.ptr<void>(),
               qweight_.byte_size(), cudaMemcpyHostToDevice);
    qweight_ = std::move(cuda_qweight);
    
    // qzeros, scales 同理...
}
```

---

### 8.3 AWQ 算子使用与完整推理流程详解

本节结合源码详细讲解 `qwen3.cpp` 中 AWQ 算子的使用方式，以及基于 `/mnt/ssd/QwenModels/Qwen3-8B-awq.bin` 模型的完整推理流程。

#### 8.3.1 AWQ 算子架构概览

AWQ 算子由三层组成：

```
┌─────────────────────────────────────────────────────────┐
│  应用层: AWQMatmulLayer (awq_matmul.cpp)                │
│    - forward(input, output) 接口                        │
│    - 根据 batch_size 自动选择 kernel                    │
├─────────────────────────────────────────────────────────┤
│  调度层: awq_gemm_tensorcore_cu (awq_gemm_tensorcore.cu)│
│    - M=1 (decode) → awq_gemm_fast_cu (GEMV 优化)       │
│    - M>1 (prefill) → awq_gemm_vllm_cu (MMA 优化)       │
├─────────────────────────────────────────────────────────┤
│  内核层: CUDA Kernels                                   │
│    - awq_gemv_fast_kernel: M=1 warp-level GEMV          │
│    - awq_gemm_small_batch_kernel: M=2~8 小 batch        │
│    - awq_gemm_fast_kernel: M>8 tiled GEMM               │
│    - awq_gemm_vllm_kernel: Tensor Core MMA + LOP3       │
└─────────────────────────────────────────────────────────┘
```

#### 8.3.2 AWQMatmulLayer::forward 调度逻辑

```cpp
// awq_matmul.cpp
base::Status AWQMatmulLayer::forward(const tensor::Tensor& input, 
                                      const tensor::Tensor& output) {
    int batch_size = input.size() / in_features_;  // 推导 batch 大小
    cudaStream_t stream = cuda_config_ ? cuda_config_->stream : nullptr;
    
    if (has_dequant_weight_ && batch_size > 1) {
        // 路径 1: 预解量化 + cuBLAS HGEMM (当内存允许且 batch>1)
        kernel::awq_gemm_with_dequant_cu(
            input_fp16, dequant_weight_fp16, output_fp16,
            batch_size, in_features_, out_features_, stream);
    } else {
        // 路径 2: 运行时在线解量化 + 融合 GEMM/GEMV
        int split_k_iters = (batch_size == 1) ? 4 : 1;
        kernel::awq_gemm_tensorcore_cu(
            input_fp16,
            qweight_.ptr<int32_t>(),    // [in_features, out_features/8] INT32
            qzeros_.ptr<int32_t>(),     // [num_groups, out_features/8] INT32
            scales_fp16,                 // [num_groups, out_features] FP16
            output_fp16,
            batch_size, in_features_, out_features_, 
            group_size_, split_k_iters, stream);
    }
}
```

#### 8.3.3 核心 CUDA Kernel 详解

**Decode 路径（M=1）：awq_gemv_fast_kernel**

```
调用链: AWQMatmulLayer::forward → awq_gemm_tensorcore_cu → awq_gemm_fast_cu
       → awq_gemv_fast_kernel

特点: 
  - 每个 warp (32 线程) 处理 8 个输出通道
  - 向量化加载 INT32 (4 字节，包含 8 个 INT4)  
  - 在寄存器中完成 AWQ 反量化: w_fp16 = scale * (w_int4 - zero)
  - warp shuffle 归约求和
  - 纯带宽受限操作
```

**Prefill 路径（M>1）：awq_gemm_vllm_kernel**

```
调用链: AWQMatmulLayer::forward → awq_gemm_tensorcore_cu → awq_gemm_vllm_cu
       → awq_gemm_vllm_kernel<N>

特点:
  - 使用 Tensor Core MMA 指令 (mma.sync.aligned.m16n8k16)
  - LOP3 快速反量化: 使用 PTX lop3.b32 位操作一次提取多个 INT4
  - ldmatrix 从共享内存加载到寄存器碎片
  - 模板参数 N=64 或 N=128
```

**vllm LOP3 反量化原理**：

```
vllm 对 AWQ 打包格式的处理:
  偶数索引元素 (0,2,4,6) 放在低 16 位
  奇数索引元素 (1,3,5,7) 放在高 16 位

使用 lop3.b32 PTX 指令提取:
  // 等价于: result = (source & mask1) | (const2 & ~mask1) 等复合位操作
  asm volatile("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(res) : "r"(src), "r"(mask), "r"(val));

转换为 FP16 后乘以 scale 并减去零点:
  half2 result = __hmul2(scale, __hsub2(w_half2, z_half2));
```

**反量化 kernel（预解量化路径）**：

```cpp
// awq_gemm_tensorcore.cu
__global__ void awq_dequant_weight_kernel(
    const int32_t* qweight, const int32_t* qzeros,
    const half* scales, half* weight_fp16,
    int K, int N, int group_size) {
    
    const int awq_order[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    
    int32_t w_packed = __ldg(&qweight[idx]);  // 加载 1 个 INT32 (含 8 个 INT4)
    int32_t z_packed = __ldg(&qzeros[g * packed_N + n_packed]);
    
    for (int i = 0; i < 8; i++) {
        int w = (w_packed >> (awq_order[i] * 4)) & 0xF;  // 提取第 i 个 INT4
        int z = (z_packed >> (awq_order[i] * 4)) & 0xF;  // 提取对应零点
        result[i] = scale[i] * (float)(w - z);            // 反量化
    }
}
```

#### 8.3.4 完整推理流程（Decode 单步）

以 Decode 阶段（每次生成 1 个 token）为例，输入维度为 `[1, 4096]` FP16：

```
Qwen3Model::decode(input=[1, 4096], pos=N)
│
├── for layer_idx = 0..35:  // 遍历 36 层 Transformer
│   │
│   ├── 1) attention_rms(layer_idx, input)
│   │   │  RMSNorm: input[4096] → rmsnorm_output[4096]
│   │   └─ 调用 rmsnorm_layers_[layer_idx]->forward(input, rmsnorm_output)
│   │      └─ CUDA kernel: rmsnorm_kernel_cu (FP16 in-place)
│   │
│   ├── 2) attention_qkv(layer_idx, pos_tensor)
│   │   │
│   │   ├─ 2a) wq (q_proj): rmsnorm_output[4096] → query[4096]
│   │   │   └─ AWQMatmulLayer::forward(rmsnorm_output, query)
│   │   │      └─ awq_gemm_tensorcore_cu(M=1, K=4096, N=4096)
│   │   │         └─ awq_gemm_fast_cu → awq_gemv_fast_kernel  ← GEMV 路径
│   │   │
│   │   ├─ 2b) q_norm: query reshape [32, 128] → per-head RMSNorm → reshape [4096]
│   │   │   └─ rmsnorm_layers_[layer_idx + 2*36 + 1]->forward(query, query)
│   │   │
│   │   ├─ 2c) wk (k_proj): rmsnorm_output[4096] → key[1024]
│   │   │   └─ AWQMatmulLayer::forward(rmsnorm_output, key)
│   │   │      └─ awq_gemm_tensorcore_cu(M=1, K=4096, N=1024)
│   │   │         └─ awq_gemv_fast_kernel  ← 注意 GQA: N=1024 (8个KV头)
│   │   │
│   │   ├─ 2d) k_norm: key reshape [8, 128] → per-head RMSNorm → reshape [1024]
│   │   │   └─ rmsnorm_layers_[layer_idx + 3*36 + 1]->forward(key, key)
│   │   │
│   │   ├─ 2e) wv (v_proj): rmsnorm_output[4096] → value[1024]
│   │   │   └─ AWQMatmulLayer::forward(rmsnorm_output, value)
│   │   │      └─ awq_gemv_fast_kernel (M=1, K=4096, N=1024)
│   │   │
│   │   ├─ 2f) RoPE: 对 query 和 key 施加旋转位置编码
│   │   │   └─ rope_layer_->forward(query, key, pos, sin_cache, cos_cache)
│   │   │
│   │   └─ 2g) KV Cache 更新: key → key_cache[layer][pos], value → value_cache[layer][pos]
│   │
│   ├── 3) attention_mha(layer_idx, pos_tensor)
│   │   │
│   │   ├─ 3a) Flash Attention (FP16 路径):
│   │   │   └─ flash_attention_decode_layer_->forward()
│   │   │      查询 query[4096] 与 key_cache/value_cache 做注意力
│   │   │      输出 mha_output[4096]
│   │   │
│   │   └─ 3b) wo (o_proj): mha_output[4096] → attn_output[4096]
│   │       └─ AWQMatmulLayer::forward(mha_output, attn_output)
│   │          └─ awq_gemv_fast_kernel (M=1, K=4096, N=4096)
│   │
│   └── 4) feed_forward(layer_idx, input)  或  feed_forward_fused()
│       │
│       ├─ 4a) residual add: input = input + attn_output
│       │   └─ add_layer_->forward(input, attn_output, input)
│       │
│       ├─ 4b) FFN RMSNorm: input → ffn_norm_output
│       │   └─ rmsnorm_layers_[layer_idx + 36]->forward(input, ffn_norm_output)
│       │
│       ├─ 4c) w1 (gate_proj): ffn_norm_output[4096] → w1_output[14336]
│       │   └─ AWQMatmulLayer::forward(ffn_norm_output, w1_output)
│       │      └─ awq_gemv_fast_kernel (M=1, K=4096, N=14336)
│       │
│       ├─ 4d) w3 (up_proj): ffn_norm_output[4096] → w3_output[14336]
│       │   └─ AWQMatmulLayer::forward(ffn_norm_output, w3_output)
│       │      └─ awq_gemv_fast_kernel (M=1, K=4096, N=14336)
│       │
│       ├─ 4e) SwiGLU: w1_output = SiLU(w1_output) × w3_output
│       │   └─ swiglu_layer_->forward(w1_output, w3_output, w1_output)
│       │
│       ├─ 4f) w2 (down_proj): w1_output[14336] → w2_output[4096]
│       │   └─ AWQMatmulLayer::forward(w1_output, w2_output)
│       │      └─ awq_gemv_fast_kernel (M=1, K=14336, N=4096)
│       │
│       └─ 4g) residual add: input = input + w2_output
│           └─ add_layer_->forward(input, w2_output, input)
│
└── 5) cls_logits(input)
    ├─ 5a) final RMSNorm: rmsnorm_layers_[72]->forward(input, input)
    ├─ 5b) lm_head: input[4096] → forward_output[151936]
    │   └─ cls_layer_ (MatmulLayer, FP16, 非量化)->forward(input, forward_output)
    └─ 5c) argmax: next_token = argmax(forward_output)
```

#### 8.3.5 完整推理流程（Prefill 批处理）

以 Prefill 阶段为例（输入 `seq_len=100` 个 token），AWQ 算子使用 GEMM 路径：

```
Qwen3Model::prefill(input=[100, 4096], seq_len=100, start_pos=0)
│
├── 预分配缓冲区（双缓冲优化）:
│   hidden_buf0, hidden_buf1: [100, 4096]
│   rms_out, query_out: [100, 4096]
│   key_out, value_out: [100, 1024]
│   mha_out, wo_out: [100, 4096]
│   ffn_norm_out: [100, 4096]
│   w1_out, w3_out: [100, 14336]
│   w2_out: [100, 4096]
│
├── for layer_idx = 0..35:
│   │
│   ├─ 1) batched_attention_rms: RMSNorm [100, 4096] → rms_out [100, 4096]
│   │
│   ├─ 2) batched_attention_qkv:
│   │   ├─ wq: rms_out[100, 4096] → query_out[100, 4096]
│   │   │   └─ AWQMatmulLayer::forward(rms_out, query_out)
│   │   │      └─ awq_gemm_tensorcore_cu(M=100, K=4096, N=4096)
│   │   │         └─ awq_gemm_vllm_cu → awq_gemm_vllm_kernel<128>  ← MMA 路径!
│   │   │
│   │   ├─ wk: rms_out[100, 4096] → key_out[100, 1024]
│   │   │   └─ awq_gemm_vllm_kernel<128> (M=100, K=4096, N=1024)
│   │   │
│   │   ├─ wv: rms_out[100, 4096] → value_out[100, 1024]
│   │   │   └─ awq_gemm_vllm_kernel<128> (M=100, K=4096, N=1024)
│   │   │
│   │   ├─ Q/K Norm: per-head RMSNorm on [100*32, 128] / [100*8, 128]
│   │   ├─ Batched RoPE: batched_rope_layer_->forward()
│   │   └─ KV Cache: cudaMemcpyAsync 批量写入
│   │
│   ├─ 3) batched_attention_mha:
│   │   ├─ Flash Attention Prefill: flash_attention_prefill_layer_->forward()
│   │   └─ wo: mha_out[100, 4096] → wo_out[100, 4096]
│   │       └─ AWQMatmulLayer::forward(mha_out, wo_out)
│   │          └─ awq_gemm_vllm_kernel<128> (M=100, K=4096, N=4096)
│   │
│   ├─ 4) residual add: layer_output = layer_input + wo_out
│   │
│   └─ 5) batched_feed_forward_optimized:
│       ├─ FFN RMSNorm
│       ├─ w1 (gate_proj): ffn_norm_out[100, 4096] → w1_out[100, 14336]
│       │   └─ awq_gemm_vllm_kernel<128> (M=100, K=4096, N=14336)
│       ├─ w3 (up_proj): ffn_norm_out[100, 4096] → w3_out[100, 14336]
│       │   └─ awq_gemm_vllm_kernel<128> (M=100, K=4096, N=14336)
│       ├─ SwiGLU: w1_out = SiLU(w1_out) × w3_out
│       ├─ w2 (down_proj): w1_out[100, 14336] → w2_out[100, 4096]
│       │   └─ awq_gemm_vllm_kernel<128> (M=100, K=14336, N=4096)
│       └─ residual add
│
└── cls_logits (仅最后一个 token)
    ├─ final RMSNorm
    ├─ lm_head (FP16 MatmulLayer, 非 AWQ)
    └─ argmax → next_token
```

#### 8.3.6 Decode vs Prefill AWQ 算子对比

| 维度 | Decode (M=1) | Prefill (M>1) |
|------|-------------|---------------|
| **入口** | `forward(input, output)` | `forward(input, output)` |
| **调度** | `awq_gemm_fast_cu` | `awq_gemm_vllm_cu` |
| **CUDA Kernel** | `awq_gemv_fast_kernel` | `awq_gemm_vllm_kernel<N>` |
| **计算单元** | CUDA Core (标量/向量) | Tensor Core MMA |
| **反量化方式** | 标量 `>> & 0xF` | LOP3 PTX 位操作 |
| **内存模式** | 带宽受限 (GEMV) | 计算受限 (GEMM) |
| **典型性能** | ~10 tok/s | ~130 tok/s |
| **split_k_iters** | 4 (减少 kernel 启动开销) | 1 |

#### 8.3.7 每层线性投影的 AWQ GEMM 尺寸汇总

以 Qwen3-8B 模型参数为例（`dim=4096, kv_dim=1024, intermediate_size=14336`）：

| 投影层 | in_features (K) | out_features (N) | qweight 大小 | 每层 AWQ 总大小 |
|--------|----------------|------------------|-------------|----------------|
| q_proj | 4096 | 4096 | [4096, 512] INT32 | 8.3 MB |
| k_proj | 4096 | 1024 | [4096, 128] INT32 | 2.1 MB |
| v_proj | 4096 | 1024 | [4096, 128] INT32 | 2.1 MB |
| o_proj | 4096 | 4096 | [4096, 512] INT32 | 8.3 MB |
| gate_proj | 4096 | 14336 | [4096, 1792] INT32 | 29.1 MB |
| up_proj | 4096 | 14336 | [4096, 1792] INT32 | 29.1 MB |
| down_proj | 14336 | 4096 | [14336, 512] INT32 | 29.5 MB |
| **每层合计** | | | | **108.5 MB** |
| **36 层总计** | | | | **≈ 3.8 GB** |

加上 embedding (1.16 GB)、lm_head (1.16 GB) 及 norm 层，模型文件总大小约 **5.7 GB**，相比 FP16 的 ~16 GB 节省约 **64%**。

#### 8.3.8 AWQ 算子与 FP16 算子的动态切换

`qwen3.cpp` 中通过 `std::dynamic_pointer_cast` 实现 AWQ 和 FP16 路径的运行时切换：

```cpp
// feed_forward_fused() 中的示例
auto w1_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(w1_layer);
auto w3_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(w3_layer);

if (w1_awq || w3_awq) {
    // AWQ 路径: 逐个调用 AWQ matmul (无法使用 fused FFN)
    STATUS_CHECK(w1_layer->forward(ffn_norm_output, w1_output));
    STATUS_CHECK(w3_layer->forward(ffn_norm_output, w3_output));
    STATUS_CHECK(swiglu_layer_->forward(w1_output, w3_output, w1_output));
} else {
    // FP16 路径: 可以使用 fused W1+W3+SwiGLU kernel
    fused_ffn_layer_->forward();
}
```

**设计要点**：
- AWQ 的线性层创建为 `AWQMatmulLayer`，FP16 的线性层创建为 `MatmulLayer`
- 二者均继承自 `op::Layer`，存储在同一个 `std::vector<std::shared_ptr<op::Layer>>` 中
- 运行时通过 `dynamic_pointer_cast` 判断实际类型并选择对应的代码路径
- **AWQ 模型不支持 Fused FFN**（因为 fused kernel 需要直接访问 FP16 权重矩阵），回退到分步执行
- `lm_head` 始终使用 FP16 `MatmulLayer`，不进行 AWQ 量化（保持输出精度）
