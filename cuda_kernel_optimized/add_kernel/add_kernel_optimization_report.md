# CUDA Add Kernel 优化报告

> **目标平台**: NVIDIA Orin (SM 8.7, Ampere架构)
> **CUDA Toolkit**: v12.6.68
> **日期**: 2026-02-12
> **工程路径**: `/mnt/ssd/workspace/OrinMLLM`
> **源文件**: `kuiper/source/op/kernels/cuda/add_kernel.cu`
> **NCU 报告**: `cuda_kernel_optimized/ncu_report.ncu-rep`

---

## 目录

1. [NCU 性能指标对比](#1-ncu-性能指标对比)
2. [每个 CUDA Kernel 的优化原理](#2-每个-cuda-kernel-的优化原理)
3. [工程中的使用方式与调用链](#3-工程中的使用方式与调用链)
4. [Global Memory / Threads / Block / Shared Memory 层面运行原理](#4-global-memory--threads--block--shared-memory-层面运行原理)

---

## 1. NCU 性能指标对比

### 测试条件

- **数据规模**: 4096 × 512 = 2,097,152 元素（模拟 Transformer 中 `dim=4096, seq_len=512` 的残差加法）
- **Broadcast Bias**: rows=512, cols=4096（模拟 QKV 投影 bias 加法）
- **采集命令**:
```bash
sudo /usr/local/cuda-12.6/bin/ncu --set full \
  --kernel-name "regex:.*add.*|.*broadcast.*" \
  --launch-skip 0 --launch-count 8 \
  -o ncu_report ./bench_add
```

### 1.1 add_kernel_cu_fp32 — FP32 逐元素加法

| 指标 | 优化前 | 优化后 | 变化 |
|------|--------|--------|------|
| **Block Size** | 512 | 256 | ↓ 减半 |
| **Grid Size** | 4,096 | 2,048 | ↓ **减半** (4x向量化) |
| **Registers/Thread** | 16 | 22 | ↑ +6 (需存储 float4) |
| **Static Shared Memory** | 0 B | 0 B | 不变 |
| **L2 Cache Throughput** | 87.54% | 88.60% | ↑ +1.06pp |
| **SM Compute Throughput** | 15.57% | 5.09% | ↓ (计算更高效) |
| **Achieved Active Warps/SM** | 38.58 | 43.58 | ↑ **+13.0%** |
| **Achieved Occupancy** | 80.4% | 90.7%* | ↑ **+10.3pp** |
| **Issue Slot Utilization** | 每6.3 cycles 发射1指令 | 每19.3 cycles 发射1指令 | 每线程工作量更多 |
| **L1TEX Stall** | 49.7 cycles (82.9%) | 193.4 cycles (92.4%) | 单次加载更宽→等待更长 |
| **FP32 Instruction Count** | 65,536 | 65,536 | 不变 (相同计算量) |

**分析**: 优化后 Grid 减半（从 4096 block 降至 2048），每线程处理 4 个 float (float4 向量化加载)。L2 Cache 吞吐量维持在 88.6% 的峰值利用率，说明已接近内存带宽上限。SM 计算吞吐从 15.57% 降至 5.09%，意味着计算指令更紧凑，浪费更少的指令槽在地址计算上。Occupancy 从 80.4% 提升至 90.7%。

### 1.2 add_kernel_cu_fp16_impl — FP16 逐元素加法

| 指标 | 优化前 | 优化后 | 变化 |
|------|--------|--------|------|
| **Block Size** | 256 | 256 | 不变 |
| **Grid Size** | 4,096 | **1,024** | ↓ **减少 75%** (8x向量化) |
| **Registers/Thread** | 16 | 16 | 不变 |
| **Static Shared Memory** | 0 B | 0 B | 不变 |
| **L2 Cache Throughput** | 86.31% | 86.63% | ↑ +0.32pp |
| **SM Compute Throughput** | 17.00% | 5.17% | ↓ (更高效) |
| **Achieved Active Warps/SM** | 40.39 | 42.88 | ↑ +6.2% |
| **Achieved Occupancy** | 84.2% | **89.3%** | ↑ **+5.1pp** |
| **Issue Slot Utilization** | 每5.6 cycles | 每19.0 cycles | 每线程工作量 4x |
| **L1TEX Stall** | 47.2 cycles (83.7%) | 185.9 cycles (91.7%) | 更宽加载→更长等待 |

**分析**: 优化前使用 `half2` (32-bit) 每线程处理 2 个 half 元素；优化后使用 `float4` (128-bit) 每线程处理 8 个 half 元素。Grid 从 4096 降至 1024（减少 75%），显著降低 block 调度开销。L2 吞吐维持 86.6%，Occupancy 提升至 89.3%。

### 1.3 broadcast_add_bias_fp16_kernel — FP16 广播 Bias 加法

| 指标 | 优化前 | 优化后 | 变化 |
|------|--------|--------|------|
| **Block Size** | 256 | 256 | 不变 |
| **Grid Size** | 8,192 (1D) | **1,024** (2D: 2×512) | ↓ **减少 87.5%** |
| **Registers/Thread** | 16 | 32 | ↑ +16 (存储 float4 + 中间值) |
| **Static Shared Memory** | 0 B | 0 B | 不变 |
| **L2 Cache Throughput** | 41.53% | **83.57%** | ↑ **+42.04pp (翻倍!)** |
| **SM Compute Throughput** | 52.00% | 9.05% | ↓ (消除了取模计算) |
| **Achieved Active Warps/SM** | 38.84 | 40.93 | ↑ +5.4% |
| **Achieved Occupancy** | 80.9% | **85.3%** | ↑ **+4.4pp** |
| **Issue Slot Utilization** | 每1.9 cycles | 每10.3 cycles | 工作量大幅增加 |
| **L1TEX Stall** | 9.1 cycles (51.1%) | 83.8 cycles (78.9%) | 更宽访问模式 |
| **ALU Pipeline** | 33.8% (整数运算瓶颈) | 远低于此 | ✅ **消除取模瓶颈** |

**分析**: **这是优化提升最大的 kernel**。
- **优化前**: 使用 1D grid + `idx % cols` 取模运算，ALU 管线利用率高达 33.8%（几乎全部消耗在整数取模上），L2 吞吐仅 41.5%——**计算瓶颈严重拖慢了内存访问**
- **优化后**: 使用 2D grid (`blockIdx.y=row, blockIdx.x=col_block`)，完全消除整数取模；加上 float4 向量化，L2 吞吐翻倍至 83.6%，Grid 减少 87.5%

### 1.4 add_vec_fp16_kernel — FP16 简单向量加法

| 指标 | 优化前 | 优化后 | 变化 |
|------|--------|--------|------|
| **Block Size** | 256 | 256 | 不变 |
| **Grid Size** | 8,192 | **1,024** | ↓ **减少 87.5%** (8x向量化) |
| **Registers/Thread** | 16 | 16 | 不变 |
| **Static Shared Memory** | 0 B | 0 B | 不变 |
| **L2 Cache Throughput** | 61.50% | **86.41%** | ↑ **+24.91pp (+40.5%)** |
| **SM Compute Throughput** | 24.51% | 5.17% | ↓ (更高效) |
| **Achieved Active Warps/SM** | 38.02 | 42.75 | ↑ **+12.4%** |
| **Achieved Occupancy** | 79.2% | **89.1%** | ↑ **+9.9pp** |
| **Issue Slot Utilization** | 每4.0 cycles | 每19.0 cycles | 每线程工作量 8x |
| **L1TEX Stall** | 26.7 cycles (73.0%) | 186.6 cycles (92.0%) | 更宽加载 |

**分析**: 从逐元素标量 (每线程 1 个 half) 升级到 float4 向量化 (每线程 8 个 half)。L2 吞吐从 61.5% 跃升至 86.4%（+40.5%），Grid 减少 87.5%，Occupancy 从 79.2% 提升至 89.1%。

### 1.5 NCU 指标总览图

```
L2 Cache Throughput (% of Peak)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fp32_add   原: ████████████████████████████████████████████▏ 87.5%
           优: █████████████████████████████████████████████▎ 88.6%  (+1.1pp)

fp16_add   原: ███████████████████████████████████████████▏  86.3%
           优: ███████████████████████████████████████████▎  86.6%  (+0.3pp)

bias_add   原: ████████████████████▊                         41.5%
           优: █████████████████████████████████████████▊    83.6%  (+42.0pp ★)

vec_add    原: ██████████████████████████████▊               61.5%
           优: ███████████████████████████████████████████▏  86.4%  (+24.9pp ★)


Grid Size (# of Blocks)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fp32_add   原: ████████████████████████████████████████████████ 4,096
           优: ████████████████████████                         2,048  (-50%)

fp16_add   原: ████████████████████████████████████████████████ 4,096
           优: ████████████                                     1,024  (-75%)

bias_add   原: ████████████████████████████████████████████████████████████████████████████████████████████████ 8,192
           优: ████████████                                     1,024  (-87.5% ★)

vec_add    原: ████████████████████████████████████████████████████████████████████████████████████████████████ 8,192
           优: ████████████                                     1,024  (-87.5% ★)
```

---

## 2. 每个 CUDA Kernel 的优化原理

### 2.1 add_kernel_cu_fp32 — FP32 向量化加法

#### 优化前代码
```cuda
__global__ void add_kernel_cu_fp32(int32_t size, const float* in1, const float* in2, float* out) {
  int32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= size) return;
  out[tid] = in1[tid] + in2[tid];
}
```

#### 优化后代码
```cuda
__global__ void add_kernel_cu_fp32(int32_t size, const float* __restrict__ in1,
                                   const float* __restrict__ in2, float* __restrict__ out) {
  const int VEC = 4;
  int32_t idx = (threadIdx.x + blockDim.x * blockIdx.x) * VEC;
  if (idx + (VEC - 1) < size) {
    float4 a = __ldg(reinterpret_cast<const float4*>(in1 + idx));
    float4 b = __ldg(reinterpret_cast<const float4*>(in2 + idx));
    float4 c;
    c.x = a.x + b.x; c.y = a.y + b.y; c.z = a.z + b.z; c.w = a.w + b.w;
    *reinterpret_cast<float4*>(out + idx) = c;
  } else {
    for (int32_t i = idx; i < size; i++) out[i] = __ldg(in1 + i) + __ldg(in2 + i);
  }
}
```

#### 优化技术

| 技术 | 说明 |
|------|------|
| **float4 向量化加载 (128-bit)** | 每次全局内存事务加载 4 个 float (16 bytes)，而非 1 个 float (4 bytes)。Orin 的全局内存访问以 32-byte 为粒度，使用 128-bit 加载能将内存事务数减少到原来的 1/4，大幅减少 L1 cache line 浪费 |
| **`__ldg()` 只读缓存** | 通过 texture cache 路径加载只读数据，不经过 L1 数据缓存，避免与写操作争用 L1 缓存行。在 SM 8.7 上 `__ldg` 走 uniform data path，减少 cache thrashing |
| **`__restrict__` 指针别名提示** | 告知编译器三个指针互不重叠，允许编译器生成更积极的指令调度和寄存器分配 |
| **Grid 减半** | 从 4096 blocks 降至 2048 blocks，减少 block 调度器的调度开销和尾部 wave 不均的概率 |

### 2.2 add_kernel_cu_fp16_impl — FP16 高度向量化加法

#### 优化前代码
```cuda
__global__ void add_kernel_cu_fp16_impl(int32_t size, const half* in1, const half* in2, half* out) {
  int32_t idx = (threadIdx.x + blockDim.x * blockIdx.x) * 2;
  if (idx + 1 < size) {
    half2 val1 = *reinterpret_cast<const half2*>(in1 + idx);  // 32-bit 加载
    half2 val2 = *reinterpret_cast<const half2*>(in2 + idx);  // 32-bit 加载
    *reinterpret_cast<half2*>(out + idx) = __hadd2(val1, val2);
  } else if (idx < size) {
    out[idx] = __hadd(in1[idx], in2[idx]);
  }
}
```

#### 优化后代码
```cuda
__global__ void add_kernel_cu_fp16_impl(int32_t size, const half* __restrict__ in1,
                                        const half* __restrict__ in2, half* __restrict__ out) {
  const int VEC = 8;
  int32_t idx = (threadIdx.x + blockDim.x * blockIdx.x) * VEC;
  if (idx + (VEC - 1) < size) {
    float4 a4 = __ldg(reinterpret_cast<const float4*>(in1 + idx));  // 128-bit
    float4 b4 = __ldg(reinterpret_cast<const float4*>(in2 + idx));  // 128-bit
    half2* a = reinterpret_cast<half2*>(&a4);
    half2* b = reinterpret_cast<half2*>(&b4);
    float4 c4;
    half2* c = reinterpret_cast<half2*>(&c4);
    #pragma unroll
    for (int i = 0; i < 4; i++) c[i] = __hadd2(a[i], b[i]);
    *reinterpret_cast<float4*>(out + idx) = c4;                     // 128-bit
  } else { /* scalar tail */ }
}
```

#### 优化技术

| 技术 | 说明 |
|------|------|
| **128-bit → 8 half/thread** | 从 `half2` (32-bit, 2 元素/线程) 升级到 `float4` (128-bit, 8 元素/线程)。单次全局内存事务处理 8 个 FP16 值，提升 4 倍 |
| **寄存器内 reinterpret** | 将 `float4` 在寄存器中 reinterpret 为 4 个 `half2`，零开销类型转换，充分利用 `__hadd2` SIMD 指令 |
| **`#pragma unroll`** | 循环 4 次的 `__hadd2` 被完全展开，消除分支预测和循环变量开销 |
| **Grid 减少 75%** | 从 4096 → 1024 blocks，block 调度开销大幅降低 |

### 2.3 broadcast_add_bias_fp16_kernel — FP16 广播 Bias 加法 (★最大优化)

#### 优化前代码
```cuda
__global__ void broadcast_add_bias_fp16_kernel(
    const half* matrix, const half* bias, half* output,
    int32_t rows, int32_t cols) {
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t total = rows * cols;
  if (idx < total) {
    int32_t col = idx % cols;  // ← 昂贵的整数取模!
    output[idx] = __hadd(matrix[idx], bias[col]);
  }
}
```

#### 优化后代码
```cuda
__global__ void broadcast_add_bias_fp16_kernel(
    const half* __restrict__ matrix, const half* __restrict__ bias,
    half* __restrict__ output, int32_t rows, int32_t cols) {
  const int VEC = 8;
  int32_t col_base = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
  int32_t row = blockIdx.y;  // ← 2D grid 直接得到行号
  if (row >= rows || col_base >= cols) return;
  int32_t idx = row * cols + col_base;
  if (col_base + VEC <= cols) {
    float4 m = __ldg(reinterpret_cast<const float4*>(matrix + idx));
    float4 b = __ldg(reinterpret_cast<const float4*>(bias + col_base));
    // ... __hadd2 x4 ...
    *reinterpret_cast<float4*>(output + idx) = result;
  } else { /* scalar tail */ }
}
```

#### 优化技术

| 技术 | 说明 |
|------|------|
| **2D Grid 消除整数取模** | 原版使用 `col = idx % cols` 计算列索引——在 GPU 上整数取模非常昂贵（约 20+ cycles），NCU 显示 ALU 管线高达 33.8%（几乎全是整数运算）。优化后使用 `dim3 grid(col_blocks, rows)` 的 2D 网格，`blockIdx.y` 直接给出行号，`blockIdx.x * blockDim.x + threadIdx.x` 直接给出列，无需取模 |
| **float4 向量化** | 逐元素标量变为 8 half/线程，内存事务减少 8x |
| **`__ldg()` bias 缓存** | bias 向量被多行重复读取，`__ldg()` 通过只读 texture cache 路径实现高效缓存复用 |
| **Grid 减少 87.5%** | 从 8192 → 1024 blocks |
| **L2 吞吐翻倍** | 从 41.5% → 83.6%，因为消除了计算瓶颈，内存管线可以全速运行 |

### 2.4 add_vec_fp16_kernel — FP16 向量加法

#### 优化前代码
```cuda
__global__ void add_vec_fp16_kernel(half* a, const half* b, half* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) output[idx] = __hadd(a[idx], b[idx]);  // 每线程1个half
}
```

#### 优化后代码
```cuda
__global__ void add_vec_fp16_kernel(half* __restrict__ a, const half* __restrict__ b,
                                    half* __restrict__ output, int n) {
  const int VEC = 8;
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
  if (idx + (VEC - 1) < n) {
    float4 av = __ldg(reinterpret_cast<const float4*>(a + idx));   // 128-bit
    float4 bv = __ldg(reinterpret_cast<const float4*>(b + idx));   // 128-bit
    half2* ah = reinterpret_cast<half2*>(&av);
    half2* bh = reinterpret_cast<half2*>(&bv);
    float4 cv;
    half2* ch = reinterpret_cast<half2*>(&cv);
    #pragma unroll
    for (int i = 0; i < 4; i++) ch[i] = __hadd2(ah[i], bh[i]);
    *reinterpret_cast<float4*>(output + idx) = cv;                  // 128-bit
  } else { /* scalar tail */ }
}
```

#### 优化技术

| 技术 | 说明 |
|------|------|
| **8x 向量化** | 从每线程 1 个 half (16-bit) 升级到 8 个 half (128-bit float4)，减少内存事务 8x |
| **`__ldg()` + `__restrict__`** | a 参数虽然不是 const（因为接口需要），但通过 `__ldg` 强制走只读缓存 |
| **Grid 减少 87.5%** | 从 8192 → 1024 blocks |
| **L2 吞吐 +40.5%** | 从 61.5% → 86.4%，接近带宽峰值 |

---

## 3. 工程中的使用方式与调用链

### 3.1 整体架构

```
                           add_kernel.cu (CUDA kernels)
                                    │
                           add_kernel.cuh (头文件声明)
                                    │
                    ┌───────────────┼───────────────────┐
                    ▼               ▼                   ▼
         kernels_interfaces.cpp  batched_add.cpp   batched_add.cpp
         (注册函数指针)          (BatchedAddLayer)  (BiasAddLayer)
                    │               │                   │
                    └───────┬───────┘                   │
                            ▼                           ▼
              ┌─────────────┼──────────────┐     Qwen2 QKV bias
              ▼             ▼              ▼
          qwen2.cpp     qwen3.cpp    qwen3_vl.cpp
         (残差连接)     (残差连接)    (残差连接 + DeepStack)
```

### 3.2 各 Kernel 对应的模型操作

#### `add_kernel_cu()` → Transformer 残差连接

**调用路径**: `add_kernel_cu` → `kernels_interfaces.cpp::get_add_kernel(kDeviceCUDA)` → `BatchedAddLayer::forward()` → 模型层

**用途**: 实现 Transformer 中最核心的残差连接 (Residual Connection):

```
                    ┌──────────────────────────┐
   input ──────────►│   Attention / FFN Layer   │──► layer_output
     │              └──────────────────────────┘        │
     │                                                  │
     └──────────────── add_kernel_cu ──────────────────►│ output = input + layer_output
```

**模型调用站点** (12+ 处):

| 模型 | 组件 | 源码位置 |
|------|------|----------|
| Qwen2 | Attention 残差 | `qwen2.cpp:1868` — `layer_output = layer_input + mha_out` |
| Qwen2 | FFN 残差 | `qwen2.cpp:1757, 1797` — `input = input + w2_out` |
| Qwen3 | Attention 残差 | `qwen3.cpp:1842` — `layer_output = layer_input + mha_out` |
| Qwen3 | FFN 残差 | `qwen3.cpp:1706, 1768` — `input = input + w2_out` |
| Qwen3-VL | Attention 残差 | `qwen3_vl.cpp:2276` — `layer_output = layer_input + mha_out` |
| Qwen3-VL | FFN 残差 | `qwen3_vl.cpp:2959, 3028` — `input = input + w2_out` |

> **频率分析**: 对于 36 层的 Qwen3-8B，每个 token 的前向传播调用该 kernel **72 次** (每层 2 次：1次 Attention 残差 + 1次 FFN 残差)

#### `broadcast_add_bias_fp16_cu()` → Qwen2 QKV 投影 Bias

**调用路径**: `broadcast_add_bias_fp16_cu` → `BiasAddLayer::forward()` → Qwen2 Attention

```
                 ┌──── Wq(x) ───► query  + bias_q ◄── broadcast_add_bias
input ──────────►├──── Wk(x) ───► key    + bias_k ◄── broadcast_add_bias
                 └──── Wv(x) ───► value  + bias_v ◄── broadcast_add_bias
```

**调用站点**:
| 模型 | 组件 | 源码位置 |
|------|------|----------|
| Qwen2 | Query bias | `qwen2.cpp:1498-1502` — `query_out += query_bias` |
| Qwen2 | Key bias | `qwen2.cpp:1509-1513` — `key_out += key_bias` |
| Qwen2 | Value bias | `qwen2.cpp:1520-1524` — `value_out += value_bias` |

> **注意**: 仅 Qwen2 模型使用（Qwen3/Qwen3-VL 的 QKV 投影无 bias）

#### `add_cu()` → Qwen3-VL DeepStack 视觉特征注入

**调用路径**: `add_cu` → `BatchedAddLayer::forward_raw()` → Qwen3-VL

```
hidden_states:  [token_0] [token_1] ... [vis_start] ... [vis_end] ... [token_N]
                                            │                │
                                            ▼                ▼
ds_features:                         [ds_feat_0]  ...  [ds_feat_K]
                                            │                │
                                     add_cu (in-place slice add)
                                            ▼                ▼
hidden_states:  [token_0] [token_1] ... [vis+ds_0] ... [vis+ds_K] ... [token_N]
```

- **源码位置**: `qwen3_vl.cpp:2292` 
- **用途**: 将视觉编码器 (ViT) 提取的 DeepStack 特征，注入到隐藏层中对应视觉 token 的位置
- **为什么用原始指针接口**: 需要对 hidden states tensor 的**子切片**执行 in-place 加法，而非整个 tensor

#### `add_kernel_cu_pure_fp16()` → 未使用 (Dead Code)

此函数在项目中**没有被任何代码调用**。功能上与 `add_kernel_cu()` 的 FP16 分支完全重复。

### 3.3 调用频率估算 (Qwen3-8B, seq_len=512)

| Kernel | 每 token 调用次数 | 数据量/次 |
|--------|-------------------|-----------|
| `add_kernel_cu` (FP16 残差) | 72 (36层 × 2) | 4096 elements = 8 KB |
| `broadcast_add_bias_fp16_cu` | 0 (Qwen3无bias) | — |
| `add_cu` (DeepStack) | 0 (仅VL) | 4096 × num_vis_tokens |

---

## 4. Global Memory / Threads / Block / Shared Memory 层面运行原理

### 4.1 Global Memory 访问模式

#### Orin (SM 8.7) 内存层次结构
```
                ┌─────────────────────────────────────────────────┐
                │         DRAM (LPDDR5, ~102.4 GB/s)             │
                └────────────────────┬────────────────────────────┘
                                     │
                ┌────────────────────┴────────────────────────────┐
                │            L2 Cache (4 MB, ~800 GB/s)           │
                └────────────────────┬────────────────────────────┘
                                     │
          ┌──────────────────────────┼──────────────────────────────┐
          │                          │                              │
   ┌──────┴──────┐           ┌───────┴──────┐              ┌───────┴──────┐
   │ L1/Tex Cache │           │ L1/Tex Cache │              │ L1/Tex Cache │
   │ (128 KB/SM) │           │ (128 KB/SM)  │              │ (128 KB/SM)  │
   └──────┬──────┘           └───────┬──────┘              └───────┴──────┘
          │                          │                              │
   ┌──────┴──────┐           ┌───────┴──────┐              ┌───────┴──────┐
   │  SM 0       │           │  SM 1        │              │  SM N        │
   │  (Warps)    │           │  (Warps)     │              │  (Warps)     │
   └─────────────┘           └──────────────┘              └──────────────┘
```

#### 优化前 vs 优化后的内存事务

**以 `add_vec_fp16_kernel` 为例 (N = 2M elements = 4 MB FP16 数据)**:

**优化前** — 每线程加载 1 个 half (2 bytes):
```
Warp 0 (32 threads): 读 in1[0..31] = 64 bytes → 2个 32-byte transaction
                      读 in2[0..31] = 64 bytes → 2个 32-byte transaction
                      写 out[0..31] = 64 bytes → 2个 32-byte transaction
                                                  ─── 共 6 transactions / warp
需要 2M/32 = 65,536 warps → 65,536 × 6 = 393,216 total transactions
```

**优化后** — 每线程加载 8 个 half (16 bytes) via float4:
```
Warp 0 (32 threads): 读 in1[0..255] = 512 bytes → 16个 32-byte txn (完美合并!)
                      读 in2[0..255] = 512 bytes → 16个 32-byte txn
                      写 out[0..255] = 512 bytes → 16个 32-byte txn
                                                    ─── 共 48 transactions / warp
需要 2M/256 = 8,192 warps → 8,192 × 48 = 393,216 total transactions (相同!)
```

**关键差异**: 虽然总事务数相同，但优化版的**warp 数量减少了 8x** (65536 → 8192)。这意味着：
- **Block 调度开销大幅降低**: GPU block scheduler 调度 1024 blocks vs 8192 blocks
- **单 warp 的 memory-level parallelism 更高**: 每个 warp 一次请求 16 个事务（而非 2 个），使内存控制器能更高效地合并和流水线化请求
- **L2 cache 行利用率更高**: 128-bit 对齐的加载正好占满 cache line 的有效部分

#### `__ldg()` 的内存路径

```
普通 load (LDG):     Global Memory → L2 → L1 Data Cache → Register
__ldg() load (LDG.E): Global Memory → L2 → L1 Texture Cache → Register
                                              │
                                     (独立于 L1 Data Cache,
                                      不受 store 操作污染,
                                      有专门的 read-only 缓存行)
```

在 add kernel 中，输入数据永远不会被写回，使用 `__ldg()` 将输入走只读纹理缓存路径，避免与 output 的 store 操作争用 L1 Data Cache。

### 4.2 Thread 组织与执行

#### 线程-元素映射

以 **`add_kernel_cu_fp16_impl`** (优化后) 处理 N=2M FP16 元素为例:

```
Block 0:  Thread 0 → elements [0,   1,  2,  3,  4,  5,  6,  7]   (float4 load)
          Thread 1 → elements [8,   9, 10, 11, 12, 13, 14, 15]
          ...
          Thread 255→ elements [2040, 2041, ..., 2047]
          
Block 1:  Thread 0 → elements [2048, 2049, ..., 2055]
          ...
Block 1023: Thread 255 → elements [2,097,144 .. 2,097,151]
```

#### Warp 级别的指令执行

每个 Warp (32 threads) 内的指令执行流:
```
Cycle 0:   IADD   idx = (threadIdx + blockDim * blockIdx) * 8    // 地址计算
Cycle 1:   ISETP  idx + 7 < size ?                                // 边界检查
Cycle 2:   LDG.E.128  a4 = [in1 + idx * 2]                       // 128-bit __ldg load
Cycle 3-N: (stall waiting for L1TEX response, ~185 cycles)        // 等待数据返回
Cycle N+1: LDG.E.128  b4 = [in2 + idx * 2]                       // 第二次 128-bit load
Cycle N+2: (stall or interleaved with other warps)
Cycle M:   HADD2  c[0] = a[0] + b[0]                             // 计算 (1 cycle)
Cycle M+1: HADD2  c[1] = a[1] + b[1]
Cycle M+2: HADD2  c[2] = a[2] + b[2]
Cycle M+3: HADD2  c[3] = a[3] + b[3]
Cycle M+4: STG.E.128 [out + idx * 2] = c4                        // 128-bit store
```

- **计算仅占 4 cycles**，其余全是内存等待 → 典型的 **memory-bound kernel**
- SM 调度器在等待内存时切换到其他活跃 warp 执行，实现 latency hiding

### 4.3 Block 组织与 SM 调度

#### Orin GPU 的 SM 配置
```
Orin GPU: 16 SMs, 每 SM 最多 48 warps (1536 threads), 最多 16 blocks/SM
```

#### 各 Kernel 的 Block 调度

| Kernel | Block数 | Block/SM | 活跃线程/SM | Occupancy |
|--------|---------|----------|-------------|-----------|
| fp32_add (优化前) | 4096 | 4096/16=256→截断16 | 16×512=8192→截断1536 | 80.4% |
| fp32_add (优化后) | 2048 | 2048/16=128→截断16 | 16×256=4096→截断1536 | 90.7% |
| fp16_add (优化前) | 4096 | 256→截断16 | 16×256=4096→截断1536 | 84.2% |
| fp16_add (优化后) | 1024 | 64→截断16 | 16×256=4096→截断1536 | 89.3% |
| bias_add (优化前) | 8192 | 512→截断16 | 16×256=4096→截断1536 | 80.9% |
| bias_add (优化后) | 1024 (2D) | 64→截断16 | 16×256=4096→截断1536 | 85.3% |
| vec_add (优化前) | 8192 | 512→截断16 | 16×256=4096→截断1536 | 79.2% |
| vec_add (优化后) | 1024 | 64→截断16 | 16×256=4096→截断1536 | 89.1% |

**Grid 减小的好处**:
1. **减少尾部 wave 不均**: 当 block 数不是 SM 数整数倍时，最后一个 wave 的 SM 利用率不满。block 越少，尾部浪费比例越小
2. **block 调度开销降低**: GPU 的 GPC (Graphics Processing Cluster) 中的 block 调度器需要分配每个 block 到合适的 SM，block 数越少，调度延迟越低
3. **减少同步点**: 每个 block 完成后需要释放 SM 资源并启动新 block，减少 block 切换次数

#### broadcast_add_bias_fp16_kernel 的 2D Grid

```
优化前: 1D Grid (8192, 1, 1)
        Block 0:   elements [0..255]     → row 0, cols [0..255]
        Block 1:   elements [256..511]   → row 0, cols [256..511] ... row 0, cols [3840..4095]
        Block 16:  elements [4096..4351] → row 1, cols [0..255]
        ...
        每线程需要: col = idx % 4096  ← 整数除法指令!

优化后: 2D Grid (2, 512)
        Block(0,0):   row=0, cols [0..2047]    ← blockIdx.y=row, 无需取模
        Block(1,0):   row=0, cols [2048..4095]
        Block(0,1):   row=1, cols [0..2047]
        Block(1,1):   row=1, cols [2048..4095]
        ...
        Block(0,511): row=511, cols [0..2047]
        Block(1,511): row=511, cols [2048..4095]
        
        row = blockIdx.y;                      ← 零开销!
        col = (blockIdx.x * blockDim.x + threadIdx.x) * 8;  ← 简单乘加
```

### 4.4 Shared Memory 使用

**所有 kernel 均不使用 Shared Memory** (Static Shared Memory = 0 bytes)。

原因分析:

| 是否适合用 Shared Memory? | 分析 |
|--------------------------|------|
| **逐元素 add kernel** | ❌ 不需要。每个输出元素仅依赖两个输入元素的对应位置，**无数据复用**。使用 shared memory 只会增加一次多余的 load→store→load 往返，反而降低性能 |
| **broadcast bias add** | ⚠️ bias 被多行重复读取，理论上可以先 load 到 shared mem。但优化后使用 2D grid + `__ldg()` 走 L1 texture cache，bias 数据被自动缓存，效果等价且编程更简单。对于 dim=4096 的 bias (8 KB FP16)，L1 cache (128 KB/SM) 完全容纳 |

**结论**: 对于加法这类 element-wise 操作，最有效的优化手段是**增大访存粒度** (向量化)和**减少冗余计算** (消除取模)，而非引入 shared memory。

### 4.5 完整数据流图 (以一次 Prefill 中的残差加法为例)

```
                  DRAM (Global Memory)
                  ┌─────────────────────────────────────────────────────┐
                  │  input[4096 × 512]     mha_out[4096 × 512]         │
                  │  (残差输入)              (Attention输出)              │
                  └──────┬──────────────────────┬────────────────────────┘
                         │ __ldg 128-bit load   │ __ldg 128-bit load
                         ▼                      ▼
                  ┌──────────────────────────────────────────────┐
                  │           L2 Cache (4 MB)                    │
                  │  input cacheline ← 热数据，命中率高           │
                  │  mha_out cacheline ← 刚写入，可能在 L2       │
                  └──────┬──────────────────────┬────────────────┘
                         │                      │
                         ▼                      ▼
                  ┌──────────────────────────────────────────────┐
                  │      L1 Texture Cache (128 KB/SM)            │
                  │  (只读路径, 不受 store 污染)                   │
                  └──────┬──────────────────────┬────────────────┘
                         │                      │
                         ▼                      ▼
                  ┌──────────────────────────────────────────────┐
                  │              SM Registers                     │
                  │  float4 a4 = {h0h1, h2h3, h4h5, h6h7}       │
                  │  float4 b4 = {h0h1, h2h3, h4h5, h6h7}       │
                  │                                              │
                  │  __hadd2 × 4 → float4 c4                    │  ← 4 cycles 计算
                  │                                              │
                  └──────────────────┬───────────────────────────┘
                                     │ 128-bit store
                                     ▼
                  ┌─────────────────────────────────────────────┐
                  │  L2 Cache → DRAM                            │
                  │  output[4096 × 512]                         │
                  │  (残差加法结果, 下一层的输入)                  │
                  └─────────────────────────────────────────────┘
```

---

## 附录: NCU 采集命令参考

```bash
# 1. 编译 benchmark (含优化前/优化后两版 kernel)
cd /mnt/ssd/workspace/OrinMLLM/cuda_kernel_optimized
nvcc -O3 -arch=sm_87 -o bench_add bench_add_kernels.cu

# 2. 采集完整 NCU 报告 (需要 sudo)
sudo /usr/local/cuda-12.6/bin/ncu --set full \
  --kernel-name "regex:.*add.*|.*broadcast.*" \
  --launch-skip 0 --launch-count 8 \
  -o ncu_report ./bench_add

# 3. 在 Nsight Compute GUI 中打开报告
ncu-ui ncu_report.ncu-rep

# 4. 导出 CSV 格式关键指标
sudo /usr/local/cuda-12.6/bin/ncu -i ncu_report.ncu-rep --csv \
  --metrics "gpu__time_duration.avg,sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  dram__throughput.avg.pct_of_peak_sustained_elapsed,launch__grid_size,launch__block_size,\
  launch__registers_per_thread" > ncu_metrics.csv
```

---

> **报告生成**: 2026-02-12 | **NCU 版本**: CUDA 12.6 | **目标架构**: SM 8.7 (Orin)
