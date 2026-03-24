# OrinMLLM AWQ GEMM/GEMV、VIT 优化与 CUDA Graph 深度分析报告

> 分析日期：2026-03-23  
> 分析对象：`kuiper/source/model/qwen3_awq.cpp`、`kuiper/source/model/qwen3_vl.cpp` 及相关 CUDA kernel  
> 平台：NVIDIA Jetson AGX Orin 64GB (SM87, LPDDR5 204.8 GB/s)

---

## 目录

1. [Prefill 阶段 LOP3 解包 + 反量化 + ldmatrix + mma 完成 GEMM 计算的伪代码](#1-prefill-阶段-awq-gemm-伪代码lop3--ldmatrix--mma)
2. [Decode 阶段 LOP3 解包 + 反量化完成 GEMV 计算的伪代码](#2-decode-阶段-awq-gemv-伪代码lop3--反量化)
3. [VIT 阶段优化手段分析](#3-qwen3_vlcpp-中-vit-阶段优化手段详解)
4. [Decode 阶段 CUDA Graph 10% 提升是否达到 Orin 硬件上限](#4-decode-阶段-cuda-graph-10-性能提升是否为-orin-硬件上限)

---

## 1. Prefill 阶段 AWQ GEMM 伪代码（LOP3 + ldmatrix + mma）

### 1.1 调用链

```
qwen3_awq.cpp: batched_qkv_projection() / batched_matmul_forward()
  └─ AWQMatmulLayer::forward()                      [awq_matmul.cpp]
      └─ awq_gemm_tensorcore_cu()                   [awq_gemm_tensorcore.cu]
          └─ (M > 1) awq_gemm_vllm_cu()             [awq_gemm_vllm.cu]
              └─ awq_gemm_vllm_kernel<N>()           N ∈ {64, 128}
```

当 `batch_size > 1`（即 prefill 阶段，M = seq_len），`awq_gemm_tensorcore_cu` 调度至 Tensor Core MMA 路径。

### 1.2 数据布局

| 张量 | 形状 | 数据类型 | 说明 |
|------|------|----------|------|
| **A (input)** | `[M, K]` | FP16 | 输入激活值，M = 序列长度，K = in_features |
| **B (qweight)** | `[K, N/8]` | INT32 | 每个 INT32 打包 8 个 INT4 权重 |
| **qzeros** | `[K/G, N/8]` | INT32 | 每 group 的零点，同样 8 个 INT4 打包 |
| **scales** | `[K/G, N]` | FP16 | 每 group 每 output channel 的缩放因子 |
| **C (output)** | `[M, N]` | FP16 | 输出矩阵 |

### 1.3 vllm AWQ 打包格式

```
INT32 内的 bit 布局 (vllm 重排后):
  bits[0:3,  16:19]  → 元素 (0, 1)    // 低位对
  bits[4:7,  20:23]  → 元素 (2, 3)    // 中低位对
  bits[8:11, 24:27]  → 元素 (4, 5)    // 中高位对
  bits[12:15,28:31]  → 元素 (6, 7)    // 高位对
  
偶数索引 (0,2,4,6) 在低16位，奇数索引 (1,3,5,7) 在高16位
→ 天然适配 half2 LOP3 提取
```

### 1.4 完整伪代码

```
=================================================================
AWQ GEMM Tensor Core Kernel (Prefill, M>1)
模板参数: N ∈ {64, 128}
=================================================================

// ──────────── 线程块 / 网格配置 ────────────
threads_per_block = (32, 2)              // 32 × 2 = 64 threads/block
j_factors = ceil(OC / N)                 // N 方向的 tile 数
num_blocks = ceil(M / 16) × j_factors    // M 方向每 16 行一个 tile

// ──────────── Shared Memory 布局 ────────────
A_shared[16 × (32 + 8)]    // 16 行 × 40 列 FP16 (padding 8 防 bank conflict)
B_shared[32 × (N + 8)]     // 32 行 × (N+8) 列 FP16 (padding 8 防 bank conflict)

// ──────────── 寄存器文件 ────────────
C_warp[32]                  // FP32 累加器，每线程 32 个 float
A_shared_warp[8]            // 从 A_shared 加载的 8 个 half (via ldmatrix)
B_shared_warp[N/4]          // 从 B_shared 加载的 N/4 个 half (via ldmatrix trans)

// ──────────── 初始化累加器 ────────────
for j in 0 .. N/32:
    for i in 0 .. 8:
        C_warp[j*8 + i] = 0.0f

// ──────────── 主循环：沿 K 维度迭代，步长 32 ────────────
for k_0_0 in 0 .. K/32:

    // =========== Stage 1: 加载 A tile 到 Shared Memory ===========
    __syncthreads()
    
    // 每线程加载 8 个 FP16 (uint4 = 16 bytes, 向量化)
    if (当前行 < M):
        *(uint4*)(A_shared_ptr) = *(uint4*)(A_ptr + k_0_0 * 32)
    else:
        *(uint4*)(A_shared_ptr) = {0, 0, 0, 0}   // 零填充
    
    // =========== Stage 2: LOP3 解包 + 反量化 → B_shared ===========
    
    // 加载当前 K group 的 zeros 和 scales
    zeros_loaded = *(uint32*)(zeros_ptr + k_0_0 * 32 / G * (OC/8))
    scales_loaded = sf_ptr + k_0_0 * 32 / G * OC
    
    for ax0 in 0 .. N/16:
        // 从 Global Memory 加载 1 个 packed INT32 权重
        B_loaded = *(uint32*)(B_ptr_local + ax0 * row_stride * (OC/8))
        
        // ─── LOP3 反量化核心 (dequant_vllm_lop3) ───
        // 
        // 输入: packed_w (INT32), packed_z (INT32), scales (4×half2)
        // 输出: 4 × half2 = 8 个 FP16 反量化值
        //
        // 常量定义:
        //   FP16_TOP_MAGIC = 0x64006400     // half2{1024.0, 1024.0}
        //   BOTTOM_MASK    = 0x000f000f     // 提取 bits[0:3] 和 bits[16:19]
        //   TOP_MASK       = 0x00f000f0     // 提取 bits[4:7] 和 bits[20:23]
        
        // Step 1: LOP3 提取低位 nibble 对 (元素 0,1)
        //   lop3.b32 d = (packed_w & BOTTOM_MASK) | FP16_MAGIC
        //   → 等效: d.lo16 = 0x6400 | (w & 0x000f)   // FP16 编码
        //   → 等效: d.hi16 = 0x6400 | (w>>16 & 0x000f)
        asm("lop3.b32 w_tmp1, packed_w, BOTTOM_MASK, FP16_MAGIC, 0xea")
        asm("lop3.b32 z_tmp1, packed_z, BOTTOM_MASK, FP16_MAGIC, 0xea")
        
        // Step 2: LOP3 提取中低位 nibble 对 (元素 2,3)
        asm("lop3.b32 w_tmp2, packed_w, TOP_MASK, FP16_MAGIC, 0xea")
        asm("lop3.b32 z_tmp2, packed_z, TOP_MASK, FP16_MAGIC, 0xea")
        
        // Step 3: 转换为正确的 FP16 整数值
        w01 = half2(w_tmp1) - half2(1024.0, 1024.0)    // 减去偏移
        z01 = half2(z_tmp1) - half2(1024.0, 1024.0)
        w23 = half2(w_tmp2) * half2(1/16, 1/16)        // 右移4位修正
                            - half2(64.0, 64.0)          // 修正偏移
        z23 = half2(z_tmp2) * half2(1/16, 1/16) - half2(64.0, 64.0)
        
        // Step 4: 处理高 8 位 (元素 4,5,6,7)
        packed_w_hi = packed_w >> 8
        packed_z_hi = packed_z >> 8
        // 重复 Step 1-3 得到 w45, z45, w67, z67
        
        // Step 5: 反量化公式 output = scale × (weight - zero)
        output[0] = scales_h2[0] × (w01 - z01)   // half2 FMA
        output[1] = scales_h2[1] × (w23 - z23)
        output[2] = scales_h2[2] × (w45 - z45)
        output[3] = scales_h2[3] × (w67 - z67)
        // ─── 结束 LOP3 反量化 ───
        
        // 写入 B_shared (直接 uint4 存储 = 4 × half2 = 8 × FP16)
        *(uint4*)(B_shared_ptr + ax0 * stride) = *(uint4*)(output)
    
    __syncthreads()

    // =========== Stage 3: ldmatrix + mma 计算 ===========
    
    // 内层循环：K 维度步长 16 (32 / 16 = 2 次迭代)
    for k_0_1 in {0, 1}:

        // ── ldmatrix 加载 A tile ──
        // 将共享内存地址转换为 .shared 空间指针
        addr_A = cvta.to.shared(A_shared[k_0_1 * 16] + lane_offset)
        
        // ldmatrix.sync.aligned.m8n8.x4.shared.b16
        //   从 shared memory 加载 4 个 8×8 FP16 子矩阵到寄存器
        //   输出: A_shared_warp[0..3] = 4 × uint32 (每个含 2 × FP16)
        asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 
             {A0, A1, A2, A3}, [addr_A]")
        
        // ── ldmatrix 加载 B tile (转置) ──
        for ax1_0 in 0 .. N/32:
            addr_B = cvta.to.shared(B_shared[...] + lane_offset)
            
            // ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16
            //   从 shared memory 加载 4 个 8×8 FP16 子矩阵 (带转置)
            asm("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16
                 {B0, B1, B2, B3}, [addr_B]")

        // ── Tensor Core MMA 计算 ──
        for j_0_4 in 0 .. N/32:

            // mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
            //   计算: C[16×8] += A[16×16] × B[16×8]
            //   A: row-major FP16, B: col-major FP16, C: FP32
            
            // 第一个 8 列
            asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
                 {C0, C1, C2, C3},         // 输出 4 × float
                 {A0, A1, A2, A3},          // A 寄存器: 4 × uint32
                 {B0, B1},                  // B 寄存器: 2 × uint32
                 {C0, C1, C2, C3}")         // 累加输入 (in-place)
            
            // 第二个 8 列
            asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
                 {C4, C5, C6, C7},
                 {A0, A1, A2, A3},
                 {B4, B5},                  // B 的后半部分
                 {C4, C5, C6, C7}")

// ──────────── 写回结果 ────────────
for ax1 in 0 .. N/32:
    for local_id in 0 .. 8:
        row = (blockIdx / j_factors) * 16 + 行偏移
        if row < M:
            C[row, col] = float2half(C_warp[ax1*8 + local_id])
```

### 1.5 关键指标

| 指标 | 值 |
|------|-----|
| 总 LOP3 指令数 / INT32 解包 | 8 条 (4 对 weight + 4 对 zero) |
| MMA 指令 | `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` |
| 每次 MMA 计算规模 | 16×8 输出 from 16×16 × 16×8 |
| 累加精度 | FP32 |
| Shared Memory 总量 | `16×40 + 32×(N+8)` ×2 bytes (N=128 时 ≈ 5.6 KB) |
| Thread Block | 64 threads (32×2) |
| 网格规模 | `ceil(M/16) × ceil(N/N_tile)` |

---

## 2. Decode 阶段 AWQ GEMV 伪代码（LOP3 + 反量化）

> Decode 阶段 M=1，运算退化为 GEMV（矩阵-向量乘）。由于算术强度极低（~1 FLOP/Byte），Tensor Core MMA 的 m16n8k16 指令需将 M=1 填充至 M=16，浪费 93.75% 算力且无带宽增益。因此 AWQ 在 M=1 时走专用 GEMV 路径，使用 LOP3 INT4→FP16 解包 + 标量 half2 FMA 完成反量化计算。

### 2.1 调用链

```
qwen3_awq.cpp: batched_qkv_projection()
  └─ (M==1 && qweight_t 有效) kernel::awq_fused_qkv_cu()     [awq_gemm_fast.cu]
  └─ (fallback)              AWQMatmulLayer::forward()
      └─ awq_gemm_tensorcore_cu()                              [awq_gemm_tensorcore.cu]
          └─ (M==1 && has qweight_t) awq_gemv_coalesced_cu()   [awq_gemm_fast.cu]
          └─ (M==1 && no qweight_t)  awq_gemm_fast_cu()        [awq_gemm_fast.cu]
```

### 2.2 Coalesced GEMV 数据布局

| 张量 | 形状 | 说明 |
|------|------|------|
| **X** | `[K]` | 输入向量，FP16 |
| **qweight_t** | `[N/8, K]` | **转置后**的权重，coalesced 读取 |
| **qzeros** | `[K/G, N/8]` | 零点 |
| **scales** | `[K/G, N]` | 缩放因子 |
| **Y** | `[N]` | 输出向量，FP16 |

### 2.3 转置优化

```
原始布局: qweight[K, N/8]
  → 32 lanes 读取 stride = N/8 × 4 bytes (2048~6144 bytes)
  → 每 warp 触发 32 条 cache line → 严重非合并

转置布局: qweight_t[N/8, K]
  → 32 lanes × uint4 (16 bytes) = 512 bytes / 迭代 → 4 条 cache line
  → 完全合并访问，带宽利用率最大化
```

### 2.4 LOP3 解包 + 反量化完整数据流

以下图示展示了从 INT32 打包权重到最终 FP16 反量化值的完整数据流管线：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     LOP3 INT4→FP16 解包 + 反量化管线                        │
│                                                                             │
│  输入: packed_w (INT32)                                                     │
│  ┌───────────────────────────────────────────────────────────┐              │
│  │ bit31 ... bit16 │ bit15 ... bit0                         │              │
│  │ elem7 elem5 elem3 elem1 │ elem6 elem4 elem2 elem0       │              │
│  │  [28:31][24:27][20:23][16:19] │ [12:15][8:11][4:7][0:3]  │              │
│  └───────────────────────────────────────────────────────────┘              │
│                                                                             │
│  ┌─────── Stage 1: LOP3 位提取 (4 条 PTX 指令) ──────────┐                │
│  │                                                         │                │
│  │  lop3.b32 out0, packed_w,      0x000f000f, 0x64006400   │                │
│  │    → 提取 bits[0:3] 和 bits[16:19]                      │                │
│  │    → out0 = {0x6400|elem0_nibble, 0x6400|elem1_nibble}  │                │
│  │    → 编码为 half2{1024+elem0, 1024+elem1}               │                │
│  │                                                         │                │
│  │  lop3.b32 out1, packed_w,      0x00f000f0, 0x64006400   │                │
│  │    → 提取 bits[4:7] 和 bits[20:23]                      │                │
│  │    → out1 = half2{1024+elem2×16, 1024+elem3×16}         │                │
│  │                                                         │                │
│  │  lop3.b32 out2, packed_w>>8,   0x000f000f, 0x64006400   │                │
│  │    → 提取 bits[8:11] 和 bits[24:27]                     │                │
│  │    → out2 = half2{1024+elem4, 1024+elem5}               │                │
│  │                                                         │                │
│  │  lop3.b32 out3, packed_w>>8,   0x00f000f0, 0x64006400   │                │
│  │    → 提取 bits[12:15] 和 bits[28:31]                    │                │
│  │    → out3 = half2{1024+elem6×16, 1024+elem7×16}         │                │
│  └─────────────────────────────────────────────────────────┘                │
│                                                                             │
│  ┌─────── Stage 2: FP16 值修正 (4 条 PTX 指令) ───────────┐               │
│  │                                                         │                │
│  │  sub.f16x2 out0, out0, 0x64006400                       │                │
│  │    → half2{1024+elem0, 1024+elem1} - {1024, 1024}       │                │
│  │    → half2{elem0, elem1}     ← 正确 INT4 值 [0..15]    │                │
│  │                                                         │                │
│  │  fma.rn.f16x2 out1, out1, 0x2c002c00, 0xd400d400        │                │
│  │    → (1024+elem2×16) × (1/16) + (-64) = elem2           │                │
│  │    → half2{elem2, elem3}     ← 正确 INT4 值 [0..15]    │                │
│  │                                                         │                │
│  │  sub.f16x2 out2, out2, 0x64006400                       │                │
│  │    → half2{elem4, elem5}                                │                │
│  │                                                         │                │
│  │  fma.rn.f16x2 out3, out3, 0x2c002c00, 0xd400d400        │                │
│  │    → half2{elem6, elem7}                                │                │
│  └─────────────────────────────────────────────────────────┘                │
│                                                                             │
│  至此，8 条 PTX 指令完成 1 个 INT32 → 4 × half2 (8 × FP16) 的提取         │
│  对比标量方式 (shift+mask+cast) 需要 32 条指令 → 4× 加速                   │
│                                                                             │
│  ┌─────── Stage 3: 反量化公式 (4 × half2 FMA) ───────────┐                │
│  │                                                         │                │
│  │  // 预计算: neg_sz = -(scale × zero)                    │                │
│  │  // 反量化: dequant = scale × weight + neg_sz           │                │
│  │  //        = scale × (weight - zero)                    │                │
│  │                                                         │                │
│  │  dq01 = hfma2(scale01, w_half2{elem0,elem1}, neg_sz01) │                │
│  │  dq23 = hfma2(scale23, w_half2{elem2,elem3}, neg_sz23) │                │
│  │  dq45 = hfma2(scale45, w_half2{elem4,elem5}, neg_sz45) │                │
│  │  dq67 = hfma2(scale67, w_half2{elem6,elem7}, neg_sz67) │                │
│  └─────────────────────────────────────────────────────────┘                │
│                                                                             │
│  ┌─────── Stage 4: GEMV 点积累加 (4 × half2 MUL + FP32 累加) ──┐          │
│  │                                                               │          │
│  │  x_h2 = half2(x_val, x_val)    // 输入标量广播为 half2       │          │
│  │                                                               │          │
│  │  prod01 = hmul2(x_h2, dq01)                                  │          │
│  │  acc[0] += __low2float(prod01)   // half → float 累加        │          │
│  │  acc[1] += __high2float(prod01)                               │          │
│  │                                                               │          │
│  │  prod23 = hmul2(x_h2, dq23)                                  │          │
│  │  acc[2] += __low2float(prod23)                                │          │
│  │  acc[3] += __high2float(prod23)                               │          │
│  │                                                               │          │
│  │  // ... prod45, prod67 同理                                   │          │
│  │                                                               │          │
│  │  每个 K 位置: 4 条 hfma2 (反量化) + 4 条 hmul2 (点积)        │          │
│  │             + 8 条 FP32 累加 = 16 条浮点指令处理 8 个输出     │          │
│  └───────────────────────────────────────────────────────────────┘          │
│                                                                             │
│  ┌─────── Stage 5: Warp Shuffle 归约 ────────────────────────┐             │
│  │                                                            │             │
│  │  // 32 个 lane 各持有部分和，需要归约为最终值              │             │
│  │  for offset in {16, 8, 4, 2, 1}:                           │             │
│  │      for i in 0..8:                                        │             │
│  │          acc[i] += __shfl_down_sync(0xffffffff, acc[i], offset) │        │
│  │  // 5 轮 × 8 个值 = 40 条指令，~5 cycles (寄存器→寄存器)  │             │
│  │                                                            │             │
│  │  if lane_id == 0:                                          │             │
│  │      *(uint4*)(&Y[out_base]) = pack_to_half8(acc)          │             │
│  │      // 16 bytes 向量化写回                                │             │
│  └────────────────────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 指令统计

| 阶段 | 指令数 | 说明 |
|------|--------|------|
| LOP3 位提取 | 4 条 `lop3.b32` | 同时完成 mask + OR 两步操作 |
| FP16 值修正 | 2 条 `sub.f16x2` + 2 条 `fma.rn.f16x2` | 去除 magic number 偏移 |
| 反量化 | 4 条 `hfma2` | scale × w + neg_sz |
| 点积 | 4 条 `hmul2` + 8 条 FP32 add | x × dequant 后累加 |
| **总计/INT32** | **8 + 4 + 12 = 24 条** | **标量方式需 ~56 条** |

### 2.5 非合并 GEMV 伪代码 (awq_gemv_fast_kernel)

这是最基础的 decode GEMV kernel，使用原始 `qweight[K, N/8]` 布局（未转置），每 lane 逐 K 标量访问：

```
=================================================================
AWQ Non-Coalesced GEMV Kernel (Decode, M=1)
源文件: awq_gemm_fast.cu → awq_gemv_fast_kernel()
=================================================================

// ──────────── 线程块 / 网格配置 ────────────
threads_per_block = 256                    // 8 warps
launch_bounds(256, 4)                      // 每 SM 最多 4 个 block
num_blocks = ceil(N / 64)                  // 每 block 64 个输出通道

// ──────────── 线程分配 ────────────
warp_id = threadIdx.x / 32                // 0..7
lane_id = threadIdx.x % 32                // 0..31
packed_out_idx = blockIdx.x * 8 + warp_id // 该 warp 的 packed 列
out_base = packed_out_idx * 8             // 8 个输出通道起始

if out_base >= N: return

// ──────────── 寄存器初始化 ────────────
float acc[8] = {0, ..., 0}               // FP32 累加器

// ──────────── 外层循环：按 group 迭代 ────────────
for g in 0 .. K/group_size:
    
    // ── Per-group: LOP3 提取零点 ──
    qz = __ldg(qzeros[g * packed_N + packed_out_idx])
    z_h[4] = lop3_extract_int4_to_fp16x2(qz)       // 8 条 PTX 指令
    
    // ── Per-group: 加载 scales + 预计算 neg_sz ──
    scale_vec = *(uint4*)(&scales[g * N + out_base]) // 16 bytes 向量化
    s_h2[4] = reinterpret<half2[4]>(scale_vec)
    for j in 0..4:
        neg_sz_h2[j] = -(s_h2[j] × z_h2[j])
    
    group_start = g * group_size

    // ── 内层循环：K 维度标量迭代 ──
    // 每 lane 步长 32 → 32 个 lane 覆盖连续 32 个 K 位置
    // group_size=128 → 每 group 需 4 次迭代
    for k = lane_id; k < group_size; k += 32:       // ★ 步长 32 (非向量化)
        k_idx = group_start + k

        // 标量加载输入 + 广播
        x_val = __ldg(X[k_idx])
        x_h2 = half2(x_val, x_val)

        // 标量加载 packed 权重 (非合并访问!)
        // 地址: qweight[k_idx * packed_N + packed_out_idx]
        // 同一 warp 的 32 lanes 访问 K 维度连续但 N 维度相同 → stride = packed_N
        w_packed = __ldg(qweight[k_idx * packed_N + packed_out_idx])  // ★ 非合并
        
        // LOP3 解包
        w_h[4] = lop3_extract_int4_to_fp16x2(w_packed)
        
        // 反量化 + 点积累加
        for j in 0..4:
            dq_h2 = hfma2(s_h2[j], w_h[j], neg_sz_h2[j])
            prod = hmul2(x_h2, dq_h2)
            acc[j*2]   += low2float(prod)
            acc[j*2+1] += high2float(prod)

// ──────────── Warp Shuffle 归约 ────────────
for offset in {16, 8, 4, 2, 1}:
    for i in 0..8:
        acc[i] += __shfl_down_sync(0xffffffff, acc[i], offset)

// ──────────── 写回 (仅 lane 0) ────────────
if lane_id == 0:
    *(uint4*)(&Y[out_base]) = float8_to_half8(acc)
```

#### 非合并 vs 合并访问对比

```
非合并 (awq_gemv_fast_kernel):                   合并 (awq_gemv_coalesced_kernel):
  qweight 布局: [K, N/8]                           qweight_t 布局: [N/8, K]
  同 warp 32 lanes 访问:                            同 warp 32 lanes 访问:
    lane0: qweight[k  , packed_out_idx]               lane0: qweight_t[packed_out_idx, k*4+0..3]
    lane1: qweight[k+1, packed_out_idx]               lane1: qweight_t[packed_out_idx, k*4+4..7]
    ...                                               ...
    lane31: qweight[k+31, packed_out_idx]              lane31: qweight_t[packed_out_idx, k*4+124..127]
                                                    
  地址间距 = packed_N × 4 bytes                     地址连续 (uint4 = 16 bytes × 32 = 512 B)
  例: N=4096 → packed_N=512 → 2048 B/lane          → 仅 4 条 cache line (完全合并)
  → 每 warp 触发 32 条 cache line                 
                                                    向量化: 每 lane 处理 4 个 K 位置/迭代
  标量: 每 lane 处理 1 个 K 位置/迭代                 → group_size=128 时只需 1 次迭代
    → group_size=128 时需 4 次迭代                 
```

### 2.6 合并 GEMV 伪代码 (awq_gemv_coalesced_kernel)

```
=================================================================
AWQ Coalesced GEMV Kernel (Decode, M=1)
=================================================================

// ──────────── 线程块 / 网格配置 ────────────
threads_per_block = 256                    // 8 warps
launch_bounds(256, 4)                      // 每 SM 最多 4 个 block
num_blocks = ceil(N / 64)                  // 每 block 处理 64 个输出通道
// 每 block: 8 warps × 8 outputs/warp = 64 outputs

// ──────────── 线程分配 ────────────
warp_id = threadIdx.x / 32                // 0..7
lane_id = threadIdx.x % 32                // 0..31
packed_out_idx = blockIdx.x * 8 + warp_id // 该 warp 负责的 packed INT32 列
out_base = packed_out_idx * 8             // 该 warp 负责的 8 个输出通道起始

if out_base >= N: return                  // 边界检查

// ──────────── 寄存器初始化 ────────────
float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0}  // FP32 累加器 (数值稳定性)

// 该 warp 在转置权重中的行基址
warp_qweight = qweight_t + packed_out_idx * K

// ──────────── 外层循环：按 group 迭代 ────────────
for g in 0 .. K/group_size:
    
    // ── Per-group 预处理 (group_size 次迭代摊销) ──
    
    // 1. LOP3 提取零点 → 4 × half2
    qz = __ldg(qzeros[g * packed_N + packed_out_idx])
    z_h[4] = lop3_extract_int4_to_fp16x2(qz)
    //   LOP3 指令序列:
    //   lop3.b32 z_h[0], qz,      0x000f000f, 0x64006400, 0xea  // bits[0:3,16:19]
    //   lop3.b32 z_h[1], qz,      0x00f000f0, 0x64006400, 0xea  // bits[4:7,20:23]
    //   lop3.b32 z_h[2], qz>>8,   0x000f000f, 0x64006400, 0xea  // bits[8:11,24:27]
    //   lop3.b32 z_h[3], qz>>8,   0x00f000f0, 0x64006400, 0xea  // bits[12:15,28:31]
    //   sub.f16x2 z_h[0], z_h[0], 0x64006400   // 减去 1024.0 偏移
    //   fma.f16x2 z_h[1], z_h[1], 1/16, -64    // 右移 4 位修正
    //   sub.f16x2 z_h[2], z_h[2], 0x64006400
    //   fma.f16x2 z_h[3], z_h[3], 1/16, -64
    
    // 2. 加载 scales 为 4 × half2 (uint4 = 16 bytes 向量化)
    scale_vec = *(uint4*)(&scales[g * N + out_base])
    s_h2[4] = reinterpret<half2[4]>(scale_vec)
    
    // 3. 预计算 neg_scale_zero = -(scale × zero)
    //    用于后续 FMA: scale × w + neg_sz = scale × (w - zero)
    for j in 0..4:
        neg_sz_h2[j] = -( s_h2[j] × z_h2[j] )   // half2 乘法 + 取反
    
    group_start = g * group_size

    // ── 内层循环：向量化 K 维度迭代 ──
    // 每 lane 处理 4 个 K 位置 (uint4 = 4×INT32 = 32 个 INT4)
    // 32 lanes × 4 = 128 = group_size → 每 group 仅 1 次迭代
    for k = lane_id * 4; k < group_size; k += 128:
        k_idx = group_start + k

        // 向量化加载 4 个 packed INT32 权重 (16 bytes, 完全合并)
        w4 = *(uint4*)(&warp_qweight[k_idx])
        
        // 向量化加载 4 个 FP16 输入 (8 bytes)
        x2 = *(uint2*)(&X[k_idx])
        x_ptr = reinterpret<half[4]>(x2)
        
        // 4 个 K 位置依次处理 (ILP)
        for v in 0..4:
            x_h2 = half2(x_ptr[v], x_ptr[v])   // 广播为 half2
            
            // LOP3 提取权重 → 4 × half2
            w_h[4] = lop3_extract_int4_to_fp16x2(w4[v])
            
            // 反量化 + 累加: 4 个 half2 FMA
            for j in 0..4:
                w_h2 = reinterpret<half2>(w_h[j])
                // dequant = scale × w + neg_scale_zero = scale × (w - zero)
                dq_h2 = hfma2(s_h2[j], w_h2, neg_sz_h2[j])
                // prod = input × dequant
                prod = hmul2(x_h2, dq_h2)
                acc[j*2]   += low_float(prod)    // 半精度→单精度
                acc[j*2+1] += high_float(prod)

// ──────────── Warp Shuffle 归约 ────────────
// 5 轮 butterfly reduction: offset = 16, 8, 4, 2, 1
for offset in {16, 8, 4, 2, 1}:
    for i in 0..8:
        acc[i] += __shfl_down_sync(0xffffffff, acc[i], offset)

// ──────────── 写回输出 (仅 lane 0) ────────────
if lane_id == 0:
    for i in 0..8:
        out_half[i] = float2half(acc[i])
    // uint4 向量化写回 (16 bytes = 8 × FP16)
    *(uint4*)(&Y[out_base]) = *(uint4*)(out_half)
```

### 2.7 Fused QKV GEMV 变体

在 decode 阶段，`awq_fused_qkv_kernel` 将 Q、K、V 三个 GEMV 融合为单次 kernel launch：

```
=================================================================
Fused QKV GEMV (单 Kernel 完成 Q+K+V 投影)
=================================================================

// 总 block 数 = ceil(Q_N/64) + ceil(K_N/64) + ceil(V_N/64)
// 例如 Qwen3-8B: ceil(4096/64) + ceil(1024/64) + ceil(1024/64) = 64 + 16 + 16 = 96 blocks

q_blocks = ceil(Q_N / 64)
k_blocks = ceil(K_N / 64)

// 通过 blockIdx.x 判断当前 block 属于哪个投影
if blockIdx.x < q_blocks:
    → 执行 Q 投影 GEMV (使用 q_qweight_t, q_qzeros, q_scales)
else if blockIdx.x < q_blocks + k_blocks:
    → 执行 K 投影 GEMV (使用 k_qweight_t, k_qzeros, k_scales)
else:
    → 执行 V 投影 GEMV (使用 v_qweight_t, v_qzeros, v_scales)

// 三个投影共享相同输入 X，内部 GEMV 逻辑与上述完全相同
// 优势: 3 次 kernel launch → 1 次，消除 2 次 launch overhead (~6 μs)
```

### 2.8 Fused Gate+Up+SwiGLU GEMV 变体

```
=================================================================
Fused Gate + Up + SwiGLU (单 Kernel 完成 FFN 前半段)
=================================================================

launch_bounds(256, 2)  // 降低到每 SM 2 个 block (寄存器压力: gate_acc[8] + up_acc[8])

Phase 1: gate = W1 * x        // GEMV, 结果存入 gate_acc[8]
Phase 2: up   = W3 * x        // GEMV, x 从 L2 Cache 复用 (仅约 7 KB)
Phase 3: output = SiLU(gate) * up
    for i in 0..8:
        silu_gate = gate_acc[i] / (1 + exp(-gate_acc[i]))
        out[i] = silu_gate * up_acc[i]

// 优势: 
//   1. 消除 2 个 kernel launch + 1 个 SwiGLU kernel
//   2. 消除 intermediate_dim × 2 × sizeof(FP16) 的中间 buffer 显存访问
//   3. 输入向量 x 的第二次读取命中 L2 Cache
```

### 2.9 GEMM vs GEMV 一览

| 特性 | Prefill GEMM (M>1) | Decode GEMV (M=1) |
|------|---------------------|---------------------|
| **Kernel** | `awq_gemm_vllm_kernel<N>` | `awq_gemv_coalesced_kernel` |
| **计算核心** | Tensor Core MMA (`m16n8k16`) | Scalar FMA (half2) |
| **数据搬运** | ldmatrix from shared memory | 直接 Global → Register |
| **反量化** | LOP3 inline → Shared Memory | LOP3 inline → Register |
| **Thread Block** | 64 threads (32×2) | 256 threads (8 warps) |
| **输出 Tile** | 16×N (N∈{64,128}) | 64 outputs/block |
| **瓶颈类型** | 计算密集 (compute-bound) | 带宽受限 (memory-bound) |
| **weight 布局** | `[K, N/8]` 原始布局 | `[N/8, K]` 转置布局 |

---

## 3. Qwen3_VL.cpp 中 VIT 阶段优化手段详解

### 3.1 优化手段总览

Qwen3-VL 的 Vision Encoder (ViT) 在 OrinMLLM 中经过系统性优化，主要包含以下几大类手段：

| 编号 | 优化类别 | 具体手段 | 核心原理 |
|------|----------|----------|----------|
| O1 | **算子融合** | Fused Split + RoPE + Transpose | 5 → 1 kernel，消除中间 buffer |
| O2 | **算子融合** | Fused Bias + GELU (FP16 Round-trip) | 2 → 1 kernel，保持数值精度 |
| O3 | **算子融合** | Fused Bias + Residual | 2 → 1 kernel，消除一次全局读写 |
| O4 | **算子融合** | Fused Normalize + Patch Extract | CPU→GPU 管线融合 |
| O5 | **向量化** | Float4 (128-bit) 向量化 | 4× 内存吞吐量提升 |
| O6 | **向量化** | Half2 (32-bit) 向量化 | 2× 吞吐量 + 天然 SIMD |
| O7 | **归约优化** | Warp Shuffle 替代 Shared Memory | ~80% 归约延迟降低 |
| O8 | **内存访问** | Double Buffering | 消除 cudaMemcpy，零拷贝残差 |
| O9 | **GEMM 策略** | cuBLAS HGEMM 替代 Flash Attention | Orin 上 cuBLAS 18× 快于手写 FA |
| O10 | **内存预分配** | Workspace 复用 | 消除 ViT 推理中的 cudaMalloc |
| O11 | **预处理下沉** | GPU 端归一化 + Patch 提取 | 消除 CPU FP32 中间张量 |
| O12 | **位置编码** | CPU 计算 + 异步 H2D | 保证 bit-exact 精度 |

### 3.2 O1: Fused Split + RoPE + Transpose (核心优化)

#### 原理

传统实现需要 5 个独立 kernel：
```
朴素流程 (5 个 kernel):
  ① split_qkv:    [N, 3H] → Q[N,H], K[N,H], V[N,H]
  ② rope_q:       Q[N,H] → Q_roped[N,H]  
  ③ rope_k:       K[N,H] → K_roped[N,H]
  ④ transpose_q:  Q_roped[N,H] → Q_t[heads, N, d]
  ⑤ transpose_kv: K_roped[N,H] + V[N,H] → K_t[heads,N,d], V_t[heads,N,d]
```

融合后仅需 1 个 kernel：
```
融合流程 (1 个 kernel):
  fused_split_rope_transpose:
    [N, 3H] + cos/sin → Q_t[heads,N,d], K_t[heads,N,d], V_t[heads,N,d]
```

#### 实现细节

```
Kernel 分两个 Phase:

Phase 1 — RoPE (Q + K)
  线程索引 = (head_idx, token_idx, dim_idx_h2) 的扁平映射
  每线程处理 1 个 half2 对 (即 2 个 FP16 值)
  
  for idx = global_thread_id; idx < rope_total; idx += stride:
      // 解码线程坐标
      dim_h2 = idx % half_head_dim_h2      // 哪个 half2 位置
      token  = (idx / half_head_dim_h2) % num_tokens
      head   = idx / (half_head_dim_h2 * num_tokens)
      
      // 一次性从 QKV 中读取 Q 和 K (连续内存)
      q_pair = __ldg(qkv + token * 3H + head * d + dim_h2 * 2)   // half2
      k_pair = __ldg(qkv + token * 3H + H + head * d + dim_h2 * 2)
      
      // 加载旋转角度
      cos_pair = __ldg(cos_cache + token * d + dim_h2 * 2)
      sin_pair = __ldg(sin_cache + token * d + dim_h2 * 2)
      
      // RoPE 旋转 (FMA 优化)
      // q_rot = q * cos - q_swap * sin  (互换实虚部)
      q_rotated = fma(q, cos, -(q_swap × sin))
      k_rotated = fma(k, cos, -(k_swap × sin))
      
      // 直接写入转置后的布局 [heads, tokens, d]
      Q_t[head * N * d + token * d + dim_h2] = q_rotated
      K_t[head * N * d + token * d + dim_h2] = k_rotated

Phase 2 — V Copy (Float4 向量化)
  V 不需要 RoPE，直接执行转置拷贝
  每线程处理 1 个 float4 = 8 个 half 值
  
  for idx = global_thread_id; idx < v_total; idx += stride:
      f4_idx = idx % head_dim_f4
      token  = (idx / head_dim_f4) % num_tokens
      head   = idx / (head_dim_f4 * num_tokens)
      
      // Float4 向量化加载 (16 bytes)
      v_f4 = __ldg(qkv + token * 3H + 2H + head * d + f4_idx * 8)
      
      // Float4 向量化写入转置布局
      V_t[head * N * d + token * d + f4_idx * 8] = v_f4
```

#### 优化效果

- **Kernel launch 减少**: 5 → 1（节省 ~4 × 26μs = 104μs / block）
- **Global Memory 访问减少**: 消除 3 个中间 buffer（Q_tmp, K_tmp, V_tmp），节省 `3 × N × H × 2` 字节读写
- **27 个 Transformer 层总计**: 节省 108 次 kernel launch

### 3.3 O2: Fused Bias + GELU (FP16 Round-trip 精度保持)

#### 原理

MLP 中的 bias add + GELU 激活本是两个独立操作：
```
朴素:  ① y = x + bias    → 写回 Global Memory
       ② z = GELU(y)     → 从 Global Memory 重新加载
```

融合后：
```
融合:  z = GELU(x + bias)  → 一次读取，一次写入
```

#### FP16 Round-trip 精度技术

关键细节：直接在 FP32 中计算 `x + bias` 再算 GELU 会与分步执行的结果不同（因为分步方案中间结果会被截断为 FP16）。为保持 bit-exact 一致性，融合 kernel 模拟了 FP16 round-trip：

```cuda
// 正确的 FP16 Round-trip 实现
half sum_h = __float2half(__half2float(input[i]) + __half2float(bias[i]));  // 截断到 FP16
float gelu_input = __half2float(sum_h);  // 再转回 FP32 计算 GELU
output[i] = __float2half(gelu_approx(gelu_input));
```

#### GELU 近似计算

```cuda
float gelu_approx(float x) {
    // tanh 近似: GELU(x) = 0.5x × [1 + tanh(√(2/π) × (x + 0.044715 x³))]
    float inner = 0.7978845608f * fmaf(0.044715f, x * x * x, x);  // FMA
    return 0.5f * x * (1.0f + tanhf(inner));
}
```

#### Float4 向量化

每线程处理 8 个 FP16 元素（1 个 float4 = 16 bytes）：
```
Grid size = ceil(numel/8 / 256)   // 比标量版减小 8×
每线程: float4 load → 8× (FP16→FP32 + bias + round-trip + GELU + FP32→FP16) → float4 store
```

### 3.4 O3: Fused Bias + Residual

#### 原理

Transformer 中的 "bias add + residual connection" 组合：
```
朴素:  ① y = proj_out + bias    → 写回
       ② z = y + residual       → 再次读写

融合:  z = proj_out + bias + residual  → 一次读三路，一次写
```

Float4 向量化加载三个输入张量，单次计算后 float4 写回：
```
有残差:  output[i] = input[i] + bias[i % hidden_size] + residual[i]
无残差:  output[i] = input[i] + bias[i % hidden_size]
```

每次 ViT forward 调用 ~89 次（27 层 × 3 + merger 等额外调用）。

### 3.5 O4: Fused Normalize + Patch Extract (GPU 预处理)

#### 原理

传统图像预处理在 CPU 执行：
```
CPU 朴素流程:
  ① 加载 uint8 [H, W, 3]
  ② 归一化: float32 pixel = (uint8 - mean*255) / (std*255)        // CPU 密集循环
  ③ 转换: fp32 → fp16                                              // CPU 密集循环
  ④ 重排: HWC → CHW [3, H, W]                                     // CPU 密集循环
  ⑤ 提取 patches: [3, H, W] → [num_patches, patch_dim]           // CPU 密集循环
  ⑥ cudaMemcpy H2D: fp16 patches                                  // PCI/LPDDR5
```

优化后：
```
GPU 融合流程:
  ① 加载 uint8 [H, W, 3]                                          // CPU (stb)
  ② cudaMemcpy H2D: raw uint8 (仅原始像素)                        // 极小数据量
  ③ GPU kernel: uint8→fp16 归一化 + HWC→CHW + patch 提取          // 全部在 GPU 完成
```

#### 优势

- 消除 CPU 浮点计算循环
- H2D 数据量从 `N×patch_dim×2 bytes (FP16)` 降至 `H×W×3 bytes (uint8)`，减少约 6×
- GPU 端利用 32 MB 预分配 buffer，零 cudaMalloc 开销
- 672×672 图像：仅需传输 ~1.35 MB (uint8) vs ~8.1 MB (FP16 patches)

### 3.6 O5 + O6: Float4 / Half2 向量化

#### 原理

NVIDIA GPU 内存子系统一次事务处理 32/128 bytes。小于此粒度的访问浪费带宽：

| 向量化级别 | 每线程单次加载 | 有效带宽利用 |
|-----------|---------------|-------------|
| Scalar (half) | 2 bytes | 低 |
| Half2 | 4 bytes | 2× |
| Float4 | 16 bytes | 8× (接近最优) |

#### 在 ViT 中的应用

| Kernel | 向量化方式 | 效果 |
|--------|-----------|------|
| `bias_gelu_fp16_kernel` | Float4 | 每线程 8 个 half，grid 缩小 8× |
| `gelu_fp16_kernel` | Float4 | 同上 |
| `bias_add_residual_fp16_kernel` | Float4 | 三路输入 float4 加载 |
| `fused_split_rope_transpose` (V copy) | Float4 | 16 bytes/thread |
| `transpose_head_token_kernel` | Float4 / Half2 自适应 | head_dim 对齐时用 float4 |
| `spatial_merge_fp16_kernel` | Float4 (via cudaMemcpy) | 块级传输 |
| `layernorm_with_bias_fp16_kernel` | Half2 | 归约和归一化 |
| `pos_embed_interpolate_fp16_kernel` | Half2 | 双线性插值 |
| `fused_split_rope_transpose` (RoPE) | Half2 | 旋转计算 |
| `vision_softmax_fp16_kernel` | Half2 | 最大值/求和/归一化 |

### 3.7 O7: Warp Shuffle 归约

#### 原理

LayerNorm 需要对 hidden_size=1152 个元素求 mean 和 variance，传统方法使用 Shared Memory：

```
Shared Memory 归约:               Warp Shuffle 归约:
  for stride = blockDim/2..1:       for offset = 16..1:
      if tid < stride:                  val += __shfl_xor_sync(mask, val, offset)
          smem[tid] += smem[tid+stride] // ~5 cycles/round (register<->register)
      __syncthreads()                 // ~1 cycle/round, 无 sync barrier
  // 8+ rounds, ~50+ cycles          // 5 rounds, ~5 cycles
```

#### 实现 (三阶段)

```
Stage 1: 线程局部累加 (Half2 向量化)
  for i = tid; i < hidden_size/2; i += blockDim:
      f2 = __half22float2(input_h2[i])
      local_sum += f2.x + f2.y
      local_sum_sq += f2.x² + f2.y²

Stage 2: Warp 内 Shuffle 归约
  for offset in {16, 8, 4, 2, 1}:
      local_sum    += __shfl_xor_sync(0xffffffff, local_sum, offset)
      local_sum_sq += __shfl_xor_sync(0xffffffff, local_sum_sq, offset)

Stage 3: 跨 Warp 归约 (Shared Memory, 仅需 1 次)
  每 warp 的 lane 0 写入 s_sum[warp_id]
  __syncthreads()
  warp 0 再做一次 shuffle 归约
  广播最终 mean / variance
```

延迟估算：5 (shuffle) + 10 (1×syncthreads + 1×cross-warp) ≈ **15 cycles**，vs 原来 50+ cycles，降低约 **70%**。

### 3.8 O8: Double Buffering (零拷贝残差)

#### 原理

Transformer 每层需要残差连接 `output = layer(x) + x`。朴素实现需要先拷贝输入：
```
朴素: 
  x_copy = cudaMemcpy(x)          // 额外 cudaMemcpy
  tmp = transformer_block(x)
  output = tmp + x_copy
```

Double buffering 使用两个交替 buffer：
```
优化:
  Layer 0: input=hidden_states, output=buf[0]     // buf[0] = block(hidden) + hidden
  Layer 1: input=buf[0],        output=buf[1]     // buf[1] = block(buf[0]) + buf[0]
  Layer 2: input=buf[1],        output=buf[0]     // buf[0] = block(buf[1]) + buf[1]
  ...
```

input 和 output 始终是不同 buffer，residual 直接引用 input，**零次额外 cudaMemcpy**。

### 3.9 O9: cuBLAS HGEMM 替代 Flash Attention

#### 背景

ViT 中的 Self-Attention 涉及三个矩阵运算：
```
① scores = Q × K^T          [heads, N, N]     N=1764 patches
② probs  = softmax(scores)  [heads, N, N]
③ output = probs × V        [heads, N, d]     d=72
```

#### 为什么 cuBLAS 比 Flash Attention 快 18×？

| 因素 | Flash Attention | cuBLAS batched HGEMM |
|------|-----------------|---------------------|
| **硬件利用** | 手写 shared memory tiling | NVIDIA 深度优化 Tensor Core 调度 |
| **适用场景** | 长序列 (seq_len >> 1K) | 中短序列 (seq_len < 2K) |
| **ViT 特点** | N=1764, d=72 | 正方形矩阵，Tensor Core 友好 |
| **SM 利用率** | 低 (手写 kernel 占用大) | 高 (库级优化) |

实际测试：cuBLAS batched HGEMM 在 N=1764, d=72 下比手写 Flash Attention kernel 快 **18×**。

#### 实现

```
// Attention 计算流程
① cublasHgemmStridedBatched: Q × K^T → scores     [16, 1764, 1764]
② vision_softmax_fp16_kernel: softmax(scores / √d)  // 自定义 kernel
③ cublasHgemmStridedBatched: probs × V → output    [16, 1764, 72]
④ transpose_head_token_kernel: [16, 1764, 72] → [1764, 1152]  // 还原布局
```

### 3.10 O10 + O11: Workspace 预分配 + GPU 预处理

#### Workspace 预分配

ViT 的所有中间 buffer 在首次调用时一次性分配：
```cpp
// 首次调用或图像尺寸变化时分配
vision_workspace_->normed1 = Tensor(FP16, [num_patches, 1152], alloc_gpu);
vision_workspace_->qkv     = Tensor(FP16, [num_patches, 3456], alloc_gpu);
vision_workspace_->query   = Tensor(FP16, [num_patches, 1152], alloc_gpu);
// ... (共 ~15 个 buffer)
```

后续推理复用已有 workspace，**零 cudaMalloc 开销**。

### 3.11 综合性能效果

| 阶段 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| ViT 编码 (672×672 图) | 552.44 ms | 474.16 ms | **-14.2%** |
| ViT Kernel 调用次数 | ~400+ | ~250 | **-37.5%** |
| 代码行数 | 2043 行 | 1211 行 | **-40.7%** |

每次 ViT Forward 的 kernel 调用统计：

| Kernel | 调用次数 | 主要优化 |
|--------|---------|----------|
| `layernorm_with_bias` | 58 | Warp Shuffle + Half2 |
| `bias_add_residual` | 89 | Float4 融合 |
| `fused_split_rope_transpose` | 27 | 5→1 融合 |
| `bias_gelu` | 27 | Float4 + FP16 Round-trip |
| `vision_softmax` | 27 | Half2 + Warp Shuffle |
| `transpose_head_token` | 27 | Float4 自适应 |
| cuBLAS HGEMM | 108 | Tensor Core 调度 |
| `pos_embed_interpolate` | 1 | Half2 双线性插值 |
| `fused_normalize_patches` | 1 | CPU→GPU 管线融合 |

---

## 4. Decode 阶段 CUDA Graph 10% 性能提升是否为 Orin 硬件上限？

### 4.1 CUDA Graph 的作用机制

CUDA Graph 通过将多次独立的 kernel launch 录制为一张有向无环图（DAG），然后作为整体一次性提交给 GPU 执行，从而消除逐 kernel 的 CPU 端 launch overhead：

```
不使用 CUDA Graph:
  每步 decode:
    506 个 kernel × ~26 μs/launch = ~13.2 ms CPU 端开销
    + ~88 ms GPU 计算 = ~101 ms/step

使用 CUDA Graph:
  每步 decode:
    1 次 cudaGraphLaunch × ~250 μs = ~0.25 ms CPU 端开销  
    + ~88 ms GPU 计算 = ~88.25 ms/step → 实际约 92 ms/step (含其他开销)
```

### 4.2 实际测量数据

| 指标 | 无 CUDA Graph | 有 CUDA Graph | 变化 |
|------|---------------|---------------|------|
| Decode 延迟 (FP16) | ~112 ms/tok | ~101 ms/tok | **-9.8%** |
| Kernel launch 开销 | ~13.2 ms | ~0.25 ms | **-98%** |
| GPU 执行时间 | ~88 ms | ~88 ms | 不变 |
| 吞吐量 (FP16) | ~8.9 tok/s | ~9.87 tok/s | **+10.9%** |

### 4.3 核心结论：10% 不是 Orin 的硬件上限，而是 CUDA Graph 这一优化手段的天花板

CUDA Graph 只能消除 **CPU 端 kernel launch overhead**，对于占比 ~87% 的 GPU 计算时间无能为力。下面从 Roofline 模型出发分析真正的硬件上限：

#### 4.3.1 Decode 阶段的带宽受限本质

Decode 阶段 (M=1) 是典型的 **Memory-Bandwidth Bound** 场景：

$$
\text{算术强度} = \frac{\text{FLOPs}}{\text{Bytes}} = \frac{2 \times K}{K \times 2} = 1 \text{ FLOP/Byte (FP16)}
$$

而 Orin 的 Roofline 转折点：

$$
I^* = \frac{\text{Peak FLOPS}}{\text{Peak Bandwidth}} = \frac{137 \text{ TFLOPS (FP16 Tensor Core)}}{204.8 \text{ GB/s}} \approx 669 \text{ FLOPs/Byte}
$$

$$
I_{\text{decode}} = 1 \ll 669 = I^* \quad \Rightarrow \quad \text{极度带宽受限}
$$

#### 4.3.2 各精度方案的理论吞吐上限

| 精度方案 | 模型权重大小 | 理论最低延迟 | 理论吞吐上限 | 实际吞吐 | 利用率 |
|---------|-------------|-------------|-------------|---------|--------|
| **FP16** | 16.38 GB | 80 ms | 12.5 tok/s | 9.87 tok/s | 79% |
| **AWQ W4A16** | 4.59 GB | 22.4 ms | 44.6 tok/s | ~18 tok/s | ~40% |
| **SQ INT8** | 8.19 GB | 40 ms | 25 tok/s | ~18 tok/s | 72% |

FP16 下实际 9.87 tok/s ÷ 理论 12.5 tok/s ≈ **79% 带宽利用率**。

#### 4.3.3 为什么 CUDA Graph 只能带来 ~10% 提升？

```
单步 decode 耗时分解 (FP16):

┌─────────────────────────────────────────────────────────┐
│ GPU 计算 (GEMV + Attention + RMSNorm + ...)   ~88 ms   │  ← 87% (带宽受限，无法优化)
├─────────────────────────────────────────────────────────┤
│ Kernel launch overhead                        ~13 ms   │  ← 13% (CUDA Graph 消除)
├─────────────────────────────────────────────────────────┤
│ 其他 (argmax, H2D, sync)                      ~1 ms    │  ← <1%
└─────────────────────────────────────────────────────────┘
  总计                                          ~102 ms

CUDA Graph 后:
  88 + 0.25 + 1 ≈ 89.25 ms → 实际 ~92 ms (含 cuBLAS workspace 等额外开销)
  提升 = (102 - 92) / 102 ≈ 10%
```

### 4.4 突破 10% 限制的方向

CUDA Graph 的 10% 是 **launch overhead 占比** 的上限，不是 Orin 硬件的上限。要进一步提升 decode 性能，需要从**降低带宽需求**入手：

| 优化方向 | 预期效果 | 原理 |
|---------|---------|------|
| **AWQ INT4 量化** | FP16 → AWQ: 9.87 → **~33 tok/s** (+234%) | 权重从 16 bit 降至 4 bit，带宽需求降 4× |
| **KV Cache INT8/FP8** | 长序列场景 1-2 ms/step | 注意力阶段 KV 读取量减半 |
| **Speculative Decoding** | 理论 2-3× 吞吐 | 小模型草稿 + 大模型验证，平摊权重加载 |
| **算子融合** | 1-2 ms/step | RMSNorm + Linear 融合，减少 kernel 数量 |
| **INT8 权重 + FP16 激活** | FP16 → INT8: 9.87 → **~18 tok/s** (+82%) | 权重从 16 bit 降至 8 bit |

### 4.5 最终回答

> **CUDA Graph 带来的 ~10% decode 性能提升不是 Orin 的硬件上限。** 这 10% 仅反映了 kernel launch overhead 在总延迟中的占比。Decode 阶段的真正瓶颈是 **LPDDR5 内存带宽 (204.8 GB/s)**，每步必须读取全部模型权重。FP16 模型 (16.38 GB) 的理论吞吐上限为 ~12.5 tok/s，当前 9.87 tok/s 已达 79% 利用率。
>
> 要实现更大幅度的性能提升，核心手段是**模型量化**：
> - AWQ INT4 可将吞吐提升至 ~33 tok/s (+234%)
> - SmoothQuant INT8 可提升至 ~18 tok/s (+82%)
>
> 这些提升来自于从根本上减少每步需要从 LPDDR5 读取的数据量，而非消除 CPU 端开销。CUDA Graph 已经将 launch overhead 从 13 ms 压缩到 0.25 ms（-98%），在其目标领域已近最优。

---

## 附录 A: LOP3 指令详解

### A.1 LOP3 指令语义

```
lop3.b32 d, a, b, c, immLut;

// immLut 是 8-bit 真值表，定义 d 的每一位如何由 a, b, c 的对应位决定
// 对于 immLut = 0xea:
//   d[bit_i] = (a[bit_i] & b[bit_i]) | c[bit_i]
//
// 等效: d = (a & b) | c
//
// 在 AWQ 中应用:
//   a = packed INT32 (待提取)
//   b = MASK (选择哪些 nibble)
//   c = FP16_MAGIC (构造合法 FP16 编码)
//   d = 把指定 nibble 嵌入到 FP16 尾数位中
```

### A.2 为什么 LOP3 比标量方式快？

```
标量方式 (每个 INT4 元素):
  shift → mask → cast_to_float → subtract  // 4 条指令 × 8 元素 = 32 条

LOP3 方式 (每对 half2):
  lop3 → [sub|fma]                          // 2 条指令 × 4 对 = 8 条
  
加速比: 32 / 8 = 4× 
```

### A.3 FP16 Magic Number 原理

```
FP16 编码: s(1) | e(5) | m(10)

0x6400 = 0 | 11001 | 0000000000  = +1024.0

当执行 (nibble & 0x000f) | 0x6400 时:
  结果 = 0 | 11001 | 000000xxxx  = 1024.0 + nibble_value

减去 1024.0 → 得到 nibble_value 的 FP16 表示 (0~15)

对于 TOP_MASK (0x00f0) 提取的 nibble:
  位置偏移了 4 bit → 值被放大 16×
  通过 × (1/16) + (-64) 修正:
    fma(result, 1/16, -64) = (1024 + nibble×16) / 16 - 64 = nibble
```

---

## 附录 B: MMA 指令格式

```ptx
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    {d0, d1, d2, d3},         // 输出: 4×float (16×8 子矩阵的部分结果)
    {a0, a1, a2, a3},         // A 矩阵: 4×uint32 (含 8×FP16 = 16×8 布局)
    {b0, b1},                 // B 矩阵: 2×uint32 (含 4×FP16 = 16×8 布局)
    {c0, c1, c2, c3};         // 累加器: 4×float (与输出共用)

// m16n8k16 含义:
//   每次 MMA 计算 16×8 的输出块
//   消耗 A 的 16×16 子矩阵和 B 的 16×8 子矩阵
//   K 维度步长 = 16
//
// 在 AWQ GEMM 中:
//   主循环 K 步长 = 32 → 每次迭代做 2 轮 MMA (k_0_1 ∈ {0, 1})
//   每轮 MMA 中对 N 方向做 N/32 个 16×8 tile
//   每个 16×8 tile 需要 2 次 mma 指令 (前 8 列 + 后 8 列)
```
