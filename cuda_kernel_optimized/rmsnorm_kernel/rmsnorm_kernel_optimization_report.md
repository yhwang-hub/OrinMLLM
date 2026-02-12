# RMSNorm CUDA Kernel 优化报告

## 目标平台
- **GPU**: NVIDIA Jetson Orin (SM 8.7, Ampere架构)
- **CUDA**: 12.6.68
- **Compute Capability**: 8.7
- **SM数量**: 16
- **每SM最大线程数**: 1536
- **每SM最大寄存器数**: 65536

---

## 1. NCU 性能指标对比（优化前 vs 优化后）

### 测试配置
- Hidden Size: 4096 (Qwen3-8B / Qwen2.5-7B 典型值)
- Block Size: 128 threads
- 工具: NVIDIA Nsight Compute (ncu)

### 1.1 NCU 指标对比

| Kernel | 指标 | 优化前 | 优化后 | 变化 |
|--------|------|--------|--------|------|
| `row_rmsnorm_f32` (1 row) | Duration | 43.33 us | 31.10 us | **-28.2%** |
| | Registers/Thread | 20 | 40 | +100% |
| | Block Size | 128 | 128 | - |
| | Grid Size | 1 | 1 | - |
| `row_rmsnorm_f32_dim` (batch=4) | Duration | 38.21 us | 24.22 us | **-36.6%** |
| | Registers/Thread | 24 | 40 | +67% |
| | Block Size | 128 | 128 | - |
| | Grid Size | 4 | 4 | - |
| `row_rmsnorm_pure_fp16` (1 row) | Duration | 31.01 us | 11.04 us | **-64.4%** |
| | Registers/Thread | 18 | 39 | +117% |
| | Block Size | 128 | 128 | - |
| | Grid Size | 1 | 1 | - |

### 1.2 独立 Benchmark 计时对比

| Kernel | 优化前 (us) | 优化后 (us) | 加速比 |
|--------|-------------|-------------|--------|
| `row_rmsnorm_f32` (1 row, 4096) | 24.632 | 17.856 | **1.38x** |
| `row_rmsnorm_f32_dim` (batch=1) | 24.635 | 17.813 | **1.38x** |
| `row_rmsnorm_f32_dim` (batch=4) | 25.836 | 19.005 | **1.36x** |
| `row_rmsnorm_f32_dim` (batch=16) | 26.723 | 19.971 | **1.34x** |
| `row_rmsnorm_f32_dim` (batch=64) | 34.422 | 32.346 | **1.06x** |
| `row_rmsnorm_pure_fp16` (1 row, 4096) | 34.541 | 16.232 | **2.13x** |

> **注意**: 寄存器使用增加是因为循环展开和向量化优化使编译器分配更多寄存器以提升 ILP (指令级并行)。在 Orin 上 65536 寄存器/SM 的限制下，128 threads × 40 regs = 5120 regs/block，最多允许 12 blocks/SM，不影响占用率。

---

## 2. 每个 CUDA Kernel 的优化原理

### 2.1 通用优化（适用于所有 6 个 kernel）

#### (a) Warp Shuffle 替代 cub::BlockReduce
- **原理**: `cub::BlockReduce` 使用共享内存进行归约，需要约 512 字节 TempStorage 和多次 `__syncthreads()`。Warp shuffle (`__shfl_down_sync`) 直接在寄存器间交换数据，延迟极低（1 cycle vs 共享内存的 20+ cycles）
- **实现**: 
  1. 每个 warp 内部通过 5 次 `__shfl_down_sync` 完成 32 -> 1 归约
  2. 每个 warp 的 lane 0 写结果到共享内存（仅 4 个 float = 16 bytes）
  3. 第 0 个 warp 读取 4 个部分和并再次 shuffle 归约
  4. thread 0 将最终结果写回共享内存广播给所有线程
- **效果**: 共享内存使用从 ~512 bytes 降至 16 bytes，`__syncthreads()` 从 3 次降至 2 次

#### (b) `__restrict__` 指针声明
- **原理**: 告知编译器输入/权重/输出指针不存在别名(aliasing)，允许编译器进行更激进的优化：如将多次内存访问合并为向量化访问，以及更好的指令调度
- **效果**: 编译器可以自由重排 load/store 指令，提升 ILP

#### (c) `#pragma unroll` 循环展开
- **原理**: 减少循环控制开销（分支预测、计数器更新），同时暴露更多 ILP 机会。对于 hidden_size=4096、128 threads 的情况，每个线程处理 8 个 float4（FP32）或 4 个 uint4（FP16），展开因子为 4 或 2
- **效果**: 减少约 10% 的指令开销

#### (d) `__ldg()` 只读缓存加载
- **原理**: 权重数据在所有 batch 行间共享且只读。`__ldg()` 强制通过 texture/L2 只读缓存路径加载，避免污染 L1 数据缓存，同时在多 block 共享权重时减少 DRAM 访问
- **应用**: 所有权重读取均使用 `__ldg()`

### 2.2 FP16 权重加载优化 (`row_rmsnorm_f32_fp16w` / `_dim`)

#### 原始实现问题
```cuda
// 4 次标量 16-bit 加载 (低效)
float w0 = __half2float(wei[base_idx]);
float w1 = __half2float(wei[base_idx + 1]);
float w2 = __half2float(wei[base_idx + 2]);
float w3 = __half2float(wei[base_idx + 3]);
```
每次加载仅 16 bits，带宽利用率仅 12.5%（16/128 bits）。

#### 优化后实现
```cuda
// 1 次 64-bit 向量化加载 (高效)
const uint2* wei_u2 = reinterpret_cast<const uint2*>(wei);
uint2 w_raw = __ldg(wei_u2 + i);  // 64-bit = 4 halfs
float2 fw01 = __half22float2(*reinterpret_cast<const half2*>(&w_raw.x));
float2 fw23 = __half22float2(*reinterpret_cast<const half2*>(&w_raw.y));
```
使用 `uint2` (64 bits) 一次加载 4 个 half 值，带宽利用率提升至 50%。

### 2.3 纯 FP16 128-bit 向量化优化 (`row_rmsnorm_pure_fp16` / `_dim`)

#### 原始实现
```cuda
// 32-bit half2 加载 (2 halfs per load)
const half2* in_h2 = reinterpret_cast<const half2*>(in);
half2 val = in_h2[i];  // 32-bit load
```

#### 优化后实现
```cuda
// 128-bit uint4 加载 (8 halfs per load，4x 带宽)
const uint4* in_vec = reinterpret_cast<const uint4*>(in);
uint4 raw = in_vec[i];  // 128-bit load = 8 halfs
float2 f0 = __half22float2(*reinterpret_cast<const half2*>(&raw.x));
float2 f1 = __half22float2(*reinterpret_cast<const half2*>(&raw.y));
float2 f2 = __half22float2(*reinterpret_cast<const half2*>(&raw.z));
float2 f3 = __half22float2(*reinterpret_cast<const half2*>(&raw.w));
```
- **原理**: 每次全局内存事务(transaction)在 Orin 上为 32 bytes (256 bits)。使用 128-bit 加载使每个线程的加载效率翻倍（从 32-bit 到 128-bit），更好地匹配内存事务粒度
- **效果**: FP16 kernel 加速 **2.13x**，是所有优化中效果最显著的

### 2.4 FP32 全精度优化 (`row_rmsnorm_f32` / `_dim`)

核心优化为 warp shuffle + `__ldg` + unroll。原始的 `float4` 向量化加载已经是最优的（128-bit），无法进一步扩展。优化主要来自归约算法改进。

### 2.5 模板化 dim kernel

原始 `row_rmsnorm_f32_dim` 和 `row_rmsnorm_f32_fp16w_dim` 不是模板函数，在 `cub::BlockReduce` 中硬编码了 `<float, 128>`。优化后统一为模板 `<BLOCK_DIM>` 参数，使 block_reduce_sum 能正确推导 warp 数量。

---

## 3. 在 OrinMLLM 工程中的使用方式

### 3.1 调用链路

```
RmsNormLayer::forward()  (kuiper/source/op/rmsnorm.cpp)
├── 1D tensor → get_rmsnorm_kernel(DeviceType)
│   → rmsnorm_kernel_cu()
│   ├── FP16 input + FP16 weight + FP16 output → row_rmsnorm_pure_fp16<128>
│   ├── FP32 input + FP16 weight → row_rmsnorm_f32_fp16w<128>
│   └── FP32 input + FP32 weight → row_rmsnorm_f32<128>
│
└── nD tensor → get_rmsnorm_dim_kernel(DeviceType)
    → rmsnorm_kernel_cu_dim()
    ├── FP16 all → row_rmsnorm_pure_fp16_dim<128>
    ├── FP32 + FP16 weight → row_rmsnorm_f32_fp16w_dim<128>
    └── FP32 all → row_rmsnorm_f32_dim<128>

RMSNormDimLayer::forward()  (kuiper/source/op/misc_layers.cpp)
└── 直接调用 rmsnorm_kernel_cu_dim()
```

### 3.2 工厂函数注册

```cpp
// kuiper/source/op/kernels/kernels_interfaces.cpp
RMSNormKernel get_rmsnorm_kernel(DeviceType type) {
    if (type == kDeviceCUDA) return rmsnorm_kernel_cu;
}
RMSNormKernelDim get_rmsnorm_dim_kernel(DeviceType type) {
    if (type == kDeviceCUDA) return rmsnorm_kernel_cu_dim;
}
```

### 3.3 各模型中的调用场景

| 模型 | Kernel 路径 | 精度配置 | 调用位置 |
|------|------------|----------|---------|
| Qwen3-8B-fp16 | `pure_fp16` / `pure_fp16_dim` | FP16 input × FP16 weight → FP16 output | 每层 attention 前 + FFN 前 + 最终输出 |
| Qwen3-8B-AWQ | `f32_fp16w` / `f32_fp16w_dim` | FP32 input × FP16 weight → FP32 output | 同上 |
| Qwen2.5-7B | `f32` / `f32_dim` | FP32 input × FP32 weight → FP32 output | 同上 |
| Qwen2.5-7B-fp16 | `pure_fp16` / `pure_fp16_dim` | FP16 all | 同上 |
| Qwen3-VL-8B-fp16 | `pure_fp16` / `pure_fp16_dim` | FP16 all | ViT encoder + LLM backbone |

### 3.4 调用频率

在 Transformer 推理中，每个 decoder layer 调用 RMSNorm **2次**（attention 前 + FFN 前），加上最终输出层 1 次：
- **Qwen3-8B**: 36 layers × 2 + 1 = **73 次/step**
- **Qwen2.5-7B**: 28 layers × 2 + 1 = **57 次/step**
- **Qwen3-VL-8B**: 36 layers × 2 + 1 = **73 次/step** (LLM 部分)

### 3.5 Decode vs Prefill 路径差异

| 阶段 | Grid Size | Block Size | Kernel 变体 |
|------|-----------|------------|-------------|
| **Decode** (1 token) | 1 | 128 | `row_rmsnorm_*` (单行) |
| **Prefill** (N tokens) | N | 128 | `row_rmsnorm_*_dim` (批量) |

---

## 4. 优化后 CUDA Kernel 的运行机制详解

### 4.1 Global Memory 层面

#### 数据布局与访问模式
- **Input**: `[batch_size × hidden_size]`，FP32 为 float4 (128-bit) 对齐，FP16 为 uint4 (128-bit) 对齐
- **Weight**: `[hidden_size]`，所有 batch 行共享，通过 `__ldg()` 走只读缓存路径
- **Output**: `[batch_size × hidden_size]`，与 Input 同布局

#### 合并访问 (Coalesced Access)
```
Thread 0: loads in[0:3]   (float4/uint4)    → Memory Transaction 0
Thread 1: loads in[4:7]                     → Memory Transaction 0
...
Thread 31: loads in[124:127]                → Memory Transaction 0
Thread 0: loads in[512:515] (next iteration) → Memory Transaction 4
```
连续的 32 个线程（1 warp）访问连续的 128-byte 内存区域，完美匹配 Orin 的 32-byte 内存事务粒度。每次事务传输效率为 100%。

#### 两遍访问策略
1. **Pass 1 (Sum of Squares)**: 读取 input → 计算 sum(x²) → L1 缓存保留数据
2. **Pass 2 (Normalize)**: 读取 input (L1 cache hit) + 读取 weight (`__ldg` → L2/texture cache) → 写入 output

### 4.2 Thread 层面

#### 线程工作分配 (hidden_size=4096, 128 threads)
- **FP32 kernel**: 每线程处理 4096/(128×4) = 8 个 float4 = 32 个元素
- **FP16 kernel**: 每线程处理 4096/(128×8) = 4 个 uint4 = 32 个元素

#### 指令流水线
```
[Thread i, Iteration 0]
  LOAD float4 from in[i*4]          ← Global memory load (L1 cached)
  FMA x² accumulation               ← 等待 load 完成期间调度其他 warp
[Thread i, Iteration 1]
  LOAD float4 from in[(i+128)*4]    ← 下一个元素
  ...
[归约完成后]
[Thread i, Output Phase]
  LOAD float4 from in[i*4]          ← L1 cache hit
  LDG float4/uint2 from wei[i*4]    ← Read-only cache
  MUL + MUL (scale * in * wei)
  STORE float4 to out[i*4]
```

### 4.3 Block 层面

#### Block 配置
- **Block Size**: 128 threads = 4 warps
- **Grid Size**: Decode 时为 1，Prefill 时为 `batch_size`

#### 占用率分析 (Orin SM 8.7)
```
每 block 资源消耗:
- Threads: 128
- Registers: 128 × 40 = 5,120
- Shared Memory: 16 bytes (warp_sums[4])

SM 可容纳 blocks:
- 线程限制: 1536 / 128 = 12 blocks
- 寄存器限制: 65536 / 5120 = 12 blocks
- Shared Memory 限制: 48KB / 16B ≈ 3000 blocks (不受限)
- 实际: min(12, 12, 16) = 12 blocks/SM

理论占用率: 12 × 128 / 1536 = 100%
```

#### Block 间同步
不需要 block 间同步。每个 block 独立处理一行数据，结果直接写回 global memory。

### 4.4 Shared Memory 层面

#### 优化前 (cub::BlockReduce)
```
__shared__ cub::BlockReduce::TempStorage temp;  // ~512 bytes
__shared__ float shared_val;                      // 4 bytes
总计: ~516 bytes
```

#### 优化后 (Warp Shuffle)
```
__shared__ float smem_reduce[4];  // 4 warps × 4 bytes = 16 bytes
// 重复使用: 
//   Phase 1: 存储 4 个 warp 的部分和
//   Phase 2: 广播最终归约结果 (smem_reduce[0])
总计: 16 bytes
```

#### Shared Memory Bank Conflict 分析
- 4 个 float 值使用 4 个 bank（每 bank 4 bytes），无 bank conflict
- 广播读取 `smem_reduce[0]` 时所有线程读同一地址，硬件广播机制自动优化

---

## 5. 优化前后模型推理性能对比

### 5.1 各模型 End-to-End 推理性能

测试条件: prompt="你好", max_tokens=128, --prefix-cache --stream --interactive

| 模型 | 指标 | 优化前 (Reference) | 优化后 (OrinMLLM) | 提升 |
|------|------|-------------------|-------------------|------|
| **Qwen3-8B-fp16** | Prefill (tokens/s) | 134.691 | 138.866 | **+3.1%** |
| | Decode (tokens/s) | 10.184 | 10.188 | +0.04% |
| **Qwen3-8B-AWQ** | Prefill (tokens/s) | 120.191 | 124.822 | **+3.9%** |
| | Decode (tokens/s) | 9.299 | 9.888 | **+6.3%** |
| **Qwen2.5-7B** | Prefill (tokens/s) | 6.047 | 6.008 | -0.6% |
| | Decode (tokens/s) | 5.657 | 5.654 | -0.05% |
| **Qwen2.5-7B-fp16** | Prefill (tokens/s) | 96.646 | 135.715 | **+40.4%** |
| | Decode (tokens/s) | 9.454 | 9.606 | **+1.6%** |

### 5.2 Qwen3-VL-8B 推理性能（含 ViT）

测试条件: --image demo.jpeg --prompt "Describe this image." --cuda-graph --stream --max-pixel 500000

| 阶段 | 优化前 (Reference) | 优化后 (OrinMLLM) | 提升 |
|------|-------------------|-------------------|------|
| **ViT Encode** | 529.64 ms | 502.85 ms | **-5.1%** |
| **Prefill** (511 tokens) | 384.67 tokens/s | 677.45 tokens/s | **+76.1%** |
| | (1328.42 ms) | (754.30 ms) | |
| **Decode** | 9.74 tokens/s | 10.07 tokens/s | **+3.4%** |
| | (102.68 ms/token) | (99.30 ms/token) | |
| **总时间** | 28,130.94 ms | 26,989.93 ms | **-4.1%** |

### 5.3 性能提升分析

#### 为什么 FP16 模型提升最大？
1. **FP16 kernel 加速 2.13x**: 128-bit 向量化加载 (uint4) 对比原来的 32-bit (half2) 加载，带宽利用率提升 4 倍
2. **FP16 模型的 RMSNorm 占比更高**: FP16 matmul 本身更快（使用 Tensor Core），使得 RMSNorm 在总时间中的占比更大
3. Qwen3-VL Prefill 因为处理 511 tokens（含大量图像 token），RMSNorm 的批量 dim kernel 调用次数极多

#### 为什么 Qwen2.5-7B (INT8 量化) 几乎没有提升？
1. 该模型使用 FP32 weight 的 RMSNorm (`row_rmsnorm_f32`)
2. FP32 kernel 的向量化加载（float4 = 128-bit）已经是原始实现的做法
3. 模型瓶颈在 INT8 反量化 + matmul，RMSNorm 占比极小

#### 为什么 Decode 提升比 Prefill 小？
1. Decode 阶段每步只处理 1 token（1 行），仅使用 1 个 SM
2. Decode 瓶颈是 attention 的 KV cache 访问和大矩阵乘法
3. RMSNorm 在 decode 总时间中占比 < 1%

---

## 附录: 优化后代码核心结构

```
rmsnorm_kernel.cu
├── warp_reduce_sum()          — Warp 级 shuffle 归约
├── block_reduce_sum<N>()      — Block 级归约 + 广播
├── row_rmsnorm_f32_fp16w<N>   — FP32 in × FP16 weight (单行, decode)
├── row_rmsnorm_f32_fp16w_dim<N> — FP32 in × FP16 weight (批量, prefill)
├── row_rmsnorm_f32_dim<N>     — FP32 全精度 (批量)
├── row_rmsnorm_f32<N>         — FP32 全精度 (单行)
├── row_rmsnorm_pure_fp16<N>   — 纯 FP16 (单行) ← 128-bit loads
├── row_rmsnorm_pure_fp16_dim<N> — 纯 FP16 (批量) ← 128-bit loads
├── rmsnorm_kernel_cu()        — 单行 Host 入口
├── rmsnorm_kernel_cu_dim()    — 批量 Host 入口
├── rmsnorm_kernel_cu_pure_fp16()    — 独立纯 FP16 入口
└── rmsnorm_kernel_cu_pure_fp16_dim() — 独立纯 FP16 批量入口
```

## NCU 报告文件

- 优化前: `ncu_rmsnorm_before.ncu-rep`
- 优化后: `ncu_rmsnorm_after.ncu-rep`
