# Flash Attention Decode FP16 Online Softmax Kernel 深度分析报告

> **文件**: `kuiper/source/op/kernels/cuda/flash_attention_kernel.cu`  
> **核心 Kernel**: `flash_attention_decode_kernel_fp16_online_softmax`  
> **启动函数**: `flash_attention_decode_fp16_cu`  
> **目标平台**: NVIDIA Jetson Orin (SM 8.7, Ampere 架构)  
> **应用场景**: LLM 推理 Decode 阶段的单 token Attention 计算

---

## 目录

1. [算子背景与问题定义](#1-算子背景与问题定义)
2. [常量定义与模型参数](#2-常量定义与模型参数)
3. [启动函数 flash_attention_decode_fp16_cu 分析](#3-启动函数-flash_attention_decode_fp16_cu-分析)
4. [Grid / Block / Thread 层面设计详解](#4-grid--block--thread-层面设计详解)
5. [Shared Memory 布局与使用](#5-shared-memory-布局与使用)
6. [Kernel 分块（Tiling）策略详解](#6-kernel-分块tiling策略详解)
7. [Q/K/V 分块与维度变化详解 — Qwen3-8B Decode 实例](#7-qkv-分块与维度变化详解--qwen3-8b-decode-实例)
8. [Kernel 逐阶段代码解读](#8-kernel-逐阶段代码解读)
9. [网格图示例](#9-网格图示例)
10. [开源 FlashAttention 与 FlashDecoding 实现详解](#10-开源-flashattention-与-flashdecoding-实现详解)
11. [相对于开源实现的优化点](#11-相对于开源实现的优化点)
12. [优化原理详解](#12-优化原理详解)
13. [Nsight Compute 性能分析](#13-nsight-compute-性能分析)
14. [总结](#14-总结)

---

## 1. 算子背景与问题定义

### 1.1 Decode 阶段的 Attention 计算特点

在 LLM 推理的 **Decode 阶段**，每次只生成一个新 token，因此查询 Q 的序列长度为 **1**，而 KV Cache 的长度随生成逐步递增。Attention 的计算公式为：

$$
O = \text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d}}\right) \cdot V
$$

其中：
- $Q \in \mathbb{R}^{1 \times d}$（单个 query 向量）
- $K \in \mathbb{R}^{n \times d}$（KV Cache 中的所有 key）
- $V \in \mathbb{R}^{n \times d}$（KV Cache 中的所有 value）
- $n$ = `kv_len = pos + 1`，即到当前位置为止的所有 KV 对
- $d$ = `head_size`，每个注意力头的维度

### 1.2 核心挑战

| 挑战 | 说明 |
|------|------|
| **访存瓶颈** | Decode 是典型的 memory-bound 操作，每步只做 O(n·d) 的计算，但需读取 O(n·d) 的 KV Cache |
| **序列长度动态变化** | kv_len 随推理逐步增长，从几十到几千不等 |
| **数值稳定性** | Softmax 需要减去最大值来避免溢出，传统方法需要多次遍历数据 |
| **CUDA Graph 兼容性** | 为了减少 CPU-GPU 同步开销，kernel 参数不能包含动态值 |

---

## 2. 常量定义与模型参数

### 2.1 Kernel 编译期常量

```cpp
constexpr int ONLINE_TILE_K = 512;       // 每个 tile 处理 512 个 KV 位置
constexpr int ONLINE_BLOCK_SIZE = 256;   // 256 线程/block (8 warps)
constexpr int ONLINE_NUM_WARPS = 8;      // 8 个 warp
constexpr float SOFTMAX_FTZ = -20.0f;    // Flush-to-zero 阈值
```

### 2.2 Qwen3-VL 8B 模型典型参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `head_num` | 32 | Query 注意力头数 |
| `kv_head_num` | 8 | KV 头数 (GQA) |
| `head_size` | 128 | 每个头的维度 |
| `kv_mul` | 4 | head_num / kv_head_num |
| `dim` | 4096 | 32 × 128，总维度 |
| `kv_dim` | 1024 | 8 × 128，KV 维度 |

---

## 3. 启动函数 flash_attention_decode_fp16_cu 分析

```cpp
void flash_attention_decode_fp16_cu(...) {
    // ...
    if (kv_len > 256) {
        // 长序列：使用 online softmax tiled kernel
        static int32_t* d_pos_scratch = nullptr;
        if (!d_pos_scratch) cudaMalloc(&d_pos_scratch, sizeof(int32_t));
        cudaMemcpyAsync(d_pos_scratch, &pos, sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream);

        dim3 grid(head_num);           // Grid: 32 个 block
        dim3 block(ONLINE_BLOCK_SIZE); // Block: 256 个线程

        const int smem_size = head_size * sizeof(half)           // 256 bytes (query)
                            + ONLINE_TILE_K * sizeof(float)      // 2048 bytes (scores)
                            + ONLINE_NUM_WARPS * sizeof(float);  // 32 bytes (reduce)
        // 总计: 2336 bytes

        flash_attention_decode_kernel_fp16_online_softmax<<<grid, block, smem_size, stream>>>(
            Q, K, V, O, d_pos_scratch, head_num, kv_head_num,
            head_size, kv_mul, kv_dim, scale);
    } else {
        // 短序列 (≤256)：使用非 tiled 的 optimized kernel（更低开销）
        // ...
    }
}
```

### 3.1 关键设计决策

1. **自适应分支**：`kv_len > 256` 时使用 tiled online softmax（减少 shared memory 消耗，更好的 L2 利用率），短序列用非 tiled kernel 避免 tiling 开销。

2. **GPU 侧存储 pos**：通过 `d_pos_scratch`（device memory）传递 position 而非作为 kernel 参数，使得 kernel 的参数签名保持固定——这是 **CUDA Graph 兼容**的关键。CUDA Graph capture 时 kernel 参数必须在 graph 构建时确定，而 `pos` 每步变化，因此通过指针间接读取。

3. **固定 shared memory 大小**：`smem_size` 仅依赖编译期常量（`head_size`, `ONLINE_TILE_K`, `ONLINE_NUM_WARPS`），不依赖动态的 `kv_len`，确保 CUDA Graph 可以重放。

---

## 4. Grid / Block / Thread 层面设计详解

### 4.1 Grid 维度

```
Grid: dim3(head_num) = dim3(32)
```

- **gridDim.x = 32**（对应 32 个 Query Attention Head）
- 每个 block 负责 **一个 Attention Head** 的完整 Decode Attention 计算
- 通过 GQA 映射：`kv_head = head / kv_mul`，4 个 query head 共享 1 个 KV head

```
Block 0  → head=0  → kv_head=0
Block 1  → head=1  → kv_head=0
Block 2  → head=2  → kv_head=0
Block 3  → head=3  → kv_head=0
Block 4  → head=4  → kv_head=1
 ...
Block 31 → head=31 → kv_head=7
```

### 4.2 Block 维度

```
Block: dim3(256)
```

- 256 线程 = **8 个 Warp**（每个 Warp 32 线程）
- 所有 256 线程在 Q·K 打分阶段**完全并行**（每个线程处理不同的 KV 位置）
- 在 V 累加阶段，线程按 `my_dim = tid % head_size` 映射到输出维度

### 4.3 Thread 到任务的映射

| 阶段 | Thread 职责 | 说明 |
|------|------------|------|
| Q 加载 | `tid < head_size/2` 的线程加载 `half2` | 64 线程即可加载 128 个 half |
| Q·K 打分 | 所有 256 线程并行 | `k_idx = tid, tid+256, tid+512, ...` |
| Max/Sum 规约 | 全部 256 线程参与 warp → block 两级规约 | |
| V 累加 | `my_dim = tid % 128` | 线程 0 和 128 都写 dim=0，但因 `tid < head_size` 条件，只有 0-127 写出 |

**关键公式**：当 `head_size = 128, ONLINE_BLOCK_SIZE = 256` 时：
- 线程 0-127 映射到输出维度 0-127，参与 V 累加和最终输出
- 线程 128-255 在 Q·K 和 softmax 阶段贡献算力，但**不参与 V 累加的写出**

这是一个精妙的设计：用"多余"的线程来加速计算密集型的 Q·K 打分阶段，而不浪费在内存受限的 V 累加阶段。

---

## 5. Shared Memory 布局与使用

### 5.1 内存布局

```
smem_raw (总计 2336 bytes)
├── s_query:  [0, 256)       -- half[128]  = 256 bytes  -- 当前 head 的 Q 向量
├── s_scores: [256, 2304)    -- float[512] = 2048 bytes -- 当前 tile 的注意力分数
└── s_reduce: [2304, 2336)   -- float[8]   = 32 bytes   -- warp 级规约临时空间
```

**物理地址计算**：
```cpp
extern __shared__ char smem_raw[];
half* s_query   = reinterpret_cast<half*>(smem_raw);                           // offset 0
float* s_scores = reinterpret_cast<float*>(smem_raw + head_size * sizeof(half)); // offset 256
float* s_reduce = s_scores + ONLINE_TILE_K;                                     // offset 2304
```

### 5.2 各区域用途

#### s_query（256 bytes）
- **写入**：kernel 启动时，由前 64 个线程以 `half2` 方式协作加载
- **读取**：在每个 tile 的 Q·K 点积阶段被所有 256 线程反复读取
- **生命周期**：整个 kernel 执行期间不变（write-once, read-many）
- **优化效果**：避免 256 线程每次都从 global memory 读 Q，利用 shared memory 的低延迟

```cpp
// 加载 Q 到 shared memory (half2 向量化)
if (tid < head_size / 2) {  // tid < 64
    reinterpret_cast<half2*>(s_query)[tid] = reinterpret_cast<const half2*>(q_ptr)[tid];
}
```

#### s_scores（2048 bytes）
- **写入**：Q·K 打分阶段，每线程写一个或多个 score 到 `s_scores[k_idx]`
- **读取**：
  1. Max 规约前读到寄存器
  2. 写回 `exp(score - m_new)` 后在 V 累加阶段读取
- **生命周期**：每个 tile 内重复使用，tile 之间被覆盖
- **关键**：固定为 `ONLINE_TILE_K=512` 个 float，不随 kv_len 变化

#### s_reduce（32 bytes）
- **用途**：8 个 warp 的局部最大值/求和结果暂存
- **流程**：各 warp lane_id=0 写 → `__syncthreads()` → thread 0 做最终规约 → 写回 `s_reduce[0]` → 所有线程读取
- 复用存放 max 和 sum（不同步骤中不同时使用）

### 5.3 Shared Memory 不随 kv_len 变化的意义

传统实现中 `s_scores` 的大小等于 `kv_len`，随推理步进不断增长，这导致：
1. Shared memory 大小动态变化，无法与 CUDA Graph 兼容
2. 长序列时 shared memory 超出限制（Orin 上 SM shared memory 为 48KB）

该实现将 `s_scores` 固定为 512 个 float（2KB），通过 tiling 处理任意长度的 kv_len。

---

## 6. Kernel 分块（Tiling）策略详解

### 6.1 Tiling 思想

将 KV Cache 的长度维度（`kv_len`）分成多个大小为 `ONLINE_TILE_K=512` 的 tile 进行处理。每个 tile 内完成：

1. Q·K 点积计算
2. Online softmax 更新（max, sum, correction）
3. V 加权累加

```
kv_len = 2000 的分块方式:
┌─────────┐┌─────────┐┌─────────┐┌──────────┐
│ Tile 0  ││ Tile 1  ││ Tile 2  ││ Tile 3   │
│ [0,512) ││[512,1024)││[1024,1536)││[1536,2000)│
│ len=512 ││ len=512 ││ len=512 ││ len=464  │
└─────────┘└─────────┘└─────────┘└──────────┘
```

### 6.2 Tile 内的线程分工

以 Tile 0（长度 512）为例，256 线程的每个线程处理：

```
Thread 0:   k_idx = 0,   256
Thread 1:   k_idx = 1,   257
Thread 2:   k_idx = 2,   258
...
Thread 255: k_idx = 255, 511
```

即每个线程在一个 tile 内处理 $\lceil 512 / 256 \rceil = 2$ 个 KV 位置的 Q·K 点积。

### 6.3 Online Softmax 跨 Tile 更新

传统 softmax 需要两次遍历数据（第一次找 max，第二次计算 exp 和归一化），而 **online softmax** 只需一次遍历，在每个 tile 处理完后更新全局统计量：

$$
m_{\text{new}} = \max(m_{\text{old}}, m_j)
$$
$$
\text{correction} = e^{m_{\text{old}} - m_{\text{new}}}
$$
$$
l_{\text{new}} = \text{correction} \times l_{\text{old}} + l_j
$$
$$
O_{\text{new}} = \text{correction} \times O_{\text{old}} + \sum_{k \in \text{tile}} \text{softmax}(k) \times V_k
$$

每处理一个新 tile，通过 `correction` 因子对之前累积的结果进行缩放补偿，确保最终结果等价于全局 softmax。

### 6.4 Tile 大小选择 (512) 的考量

| 方面 | 分析 |
|------|------|
| **Shared Memory** | 512 × 4 bytes = 2048 bytes，远低于 48KB 限制 |
| **Occupancy** | 小 shared memory → 更多 block 可并发，提升 SM 利用率 |
| **L2 Cache** | 512 个 KV 位置 × 128 × 2 bytes = 128KB 的 K 数据可放入 L2 |
| **Tile 开销** | 每 tile 需 2 次 `__syncthreads()` + 1 次 warp reduce，512 均摊后开销低 |
| **线程利用率** | 512 / 256 = 2 次迭代/线程，充分利用 ILP |

---

## 7. Q/K/V 分块与维度变化详解 — Qwen3-8B Decode 实例

本节以 **Qwen3-8B** 模型 Decode 阶段为例（假设当前已生成 `pos=999`，即 `kv_len=1000`），详细讲解 `flash_attention_decode_kernel_fp16_online_softmax` 核函数中 Query、Key、Value 是如何分块的，以及 Flash Attention 计算过程中各张量的维度变化。

### 7.1 Qwen3-8B Decode 阶段参数一览

| 参数 | 值 | 含义 |
|------|-----|------|
| `head_num` | 32 | Query 注意力头数 |
| `kv_head_num` | 8 | KV 缓存头数 (GQA) |
| `head_size` | 128 | 每个头的维度 |
| `kv_mul` | 4 | `head_num / kv_head_num`，4 个 Q 头共享 1 个 KV 头 |
| `dim` | 4096 | 模型隐藏维度 = 32 × 128 |
| `kv_dim` | 1024 | KV 缓存维度 = 8 × 128 |
| `pos` | 999 | 当前生成位置（0-indexed） |
| `kv_len` | 1000 | 需要计算 attention 的 KV 长度 = pos + 1 |
| `ONLINE_TILE_K` | 512 | 每个 tile 处理的 KV 位置数 |
| `ONLINE_BLOCK_SIZE` | 256 | 每个 block 的线程数 |

### 7.2 全局张量形状与 GQA 映射

#### 7.2.1 输入/输出张量全局形状

```
         ┌──────────────────────────────────────────────────────┐
  Q      │ [dim] = [4096]                                      │  ← 当前 token 的 query（FP16）
         │  = [head_num × head_size] = [32 × 128]               │
         └──────────────────────────────────────────────────────┘

         ┌──────────────────────────────────────────────────────┐
  K_cache│ [max_seq_len, kv_dim] = [40960, 1024]               │  ← 全部 KV 缓存（FP16）
         │  每行 = [kv_head_num × head_size] = [8 × 128]        │
         └──────────────────────────────────────────────────────┘

         ┌──────────────────────────────────────────────────────┐
  V_cache│ [max_seq_len, kv_dim] = [40960, 1024]               │  ← 全部 KV 缓存（FP16）
         │  每行 = [kv_head_num × head_size] = [8 × 128]        │
         └──────────────────────────────────────────────────────┘

         ┌──────────────────────────────────────────────────────┐
  O      │ [dim] = [4096]                                      │  ← 输出向量（FP16）
         │  = [head_num × head_size] = [32 × 128]               │
         └──────────────────────────────────────────────────────┘
```

#### 7.2.2 GQA 分组映射关系

Qwen3-8B 使用 **Grouped Query Attention (GQA)**，32 个 Q 头共享 8 个 KV 头，每 4 个 Q 头对应 1 个 KV 头：

```
  Q heads:    [  0  1  2  3 ] [  4  5  6  7 ] ... [ 28 29 30 31 ]
               ─────┬─────    ─────┬─────          ─────┬─────
  KV heads:        0                 1          ...       7

  映射公式:  kv_head = head / kv_mul = head / 4
  偏移计算:  head_offset = kv_head × head_size = kv_head × 128
```

每个 block 处理一个 Q head，通过 `head_offset` 索引到对应 KV head 的数据段：

```
  K_cache 行布局 (kv_dim=1024):
  ┌───────────┬───────────┬───────────┬─────┬───────────┐
  │ kv_head 0 │ kv_head 1 │ kv_head 2 │ ... │ kv_head 7 │
  │ [0..127]  │[128..255] │[256..383] │     │[896..1023]│
  └───────────┴───────────┴───────────┴─────┴───────────┘
       ↑                                          ↑
  Block 0-3 读取此段                         Block 28-31 读取此段
  (head 0-3, kv_head=0)                    (head 28-31, kv_head=7)
```

### 7.3 单个 Block 视角下的 Q/K/V 有效形状

以 **Block 0（head=0, kv_head=0）** 为例，该 block 实际参与计算的数据切片：

```
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                        单个 Block 的有效数据                             │
  ├──────────────────────────────────────────────────────────────────────────┤
  │                                                                        │
  │   Q_head:  Q[head × head_size : (head+1) × head_size]                  │
  │            = Q[0 : 128]                                                │
  │            有效形状: [1, 128]   ← 单个 query 向量                       │
  │                                                                        │
  │   K_head:  K_cache[0 : kv_len, head_offset : head_offset + head_size]  │
  │            = K_cache[0:1000, 0:128]                                    │
  │            有效形状: [1000, 128] ← 1000 个 key 向量                     │
  │                                                                        │
  │   V_head:  V_cache[0 : kv_len, head_offset : head_offset + head_size]  │
  │            = V_cache[0:1000, 0:128]                                    │
  │            有效形状: [1000, 128] ← 1000 个 value 向量                   │
  │                                                                        │
  │   O_head:  O[head × head_size : (head+1) × head_size]                  │
  │            = O[0 : 128]                                                │
  │            有效形状: [1, 128]   ← 单个输出向量                           │
  │                                                                        │
  └──────────────────────────────────────────────────────────────────────────┘
```

### 7.4 Flash Attention 计算全流程维度变化

#### 7.4.1 标准 Attention 维度变化（无分块参考）

$$
O = \text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d}}\right) \cdot V
$$

```
  ┌─────────┐   ┌──────────┐T     ┌──────────┐     ┌──────────┐     ┌─────────┐
  │    Q    │ × │    K     │  =   │  scores  │ →   │ attn_wt  │  ×  │    V    │ = O
  │[1, 128] │   │[1000,128]│      │ [1,1000] │     │ [1,1000] │     │[1000,128│   [1,128]
  └─────────┘   └──────────┘      └──────────┘     └──────────┘     └─────────┘
                     K^T                softmax
                 [128, 1000]          归一化
```

维度变化逐步标注：

| 步骤 | 计算 | 输入维度 | 输出维度 | FLOPs |
|------|------|---------|---------|-------|
| ① Q·K^T | 矩阵乘 | Q:[1,128] × K^T:[128,1000] | scores:[1,1000] | 2×128×1000 = 256K |
| ② ×scale | 逐元素乘 | scores:[1,1000] | scores:[1,1000] | 1000 |
| ③ softmax | max → exp → sum → div | scores:[1,1000] | attn_weights:[1,1000] | ~4×1000 |
| ④ attn×V | 矩阵乘 | attn_weights:[1,1000] × V:[1000,128] | O:[1,128] | 2×1000×128 = 256K |

#### 7.4.2 Tiled Online Softmax 分块后的维度变化

将 `kv_len=1000` 沿 KV 位置轴分成 `⌈1000/512⌉ = 2` 个 tile：

```
  KV 位置轴:   0                              512                        1000
               ├────────────── Tile 0 ─────────┤──────── Tile 1 ──────────┤
               │           len = 512           │        len = 488         │
               └───────────────────────────────┴──────────────────────────┘
```

**每个 Tile 内的分块计算维度**：

```
  ══════════════════════════════════════════════════════════════════════════
  Tile 0 (tile_start=0, tile_len=512):
  ══════════════════════════════════════════════════════════════════════════

  ┌─────────┐   ┌──────────┐T     ┌──────────┐        ┌──────────┐
  │  Q_head │ × │ K_tile0  │  =   │ scores_0 │  ×     │ V_tile0  │ → partial O
  │[1, 128] │   │[512, 128]│      │ [1, 512] │        │[512, 128]│   [1, 128]
  └─────────┘   └──────────┘      └──────────┘        └──────────┘
                     K^T_tile0     softmax(scores_0)
                  [128, 512]       → attn_wt_0 [1, 512]

  ── 状态更新 ──
  row_max_0:   max(scores_0)                 ← 标量
  row_sum_0:   Σ exp(scores_0 - row_max_0)   ← 标量
  acc_o_0:     attn_wt_0 × V_tile0           ← [1, 128]（未归一化）

  ══════════════════════════════════════════════════════════════════════════
  Tile 1 (tile_start=512, tile_len=488):
  ══════════════════════════════════════════════════════════════════════════

  ┌─────────┐   ┌──────────┐T     ┌──────────┐        ┌──────────┐
  │  Q_head │ × │ K_tile1  │  =   │ scores_1 │  ×     │ V_tile1  │ → partial O
  │[1, 128] │   │[488, 128]│      │ [1, 488] │        │[488, 128]│   [1, 128]
  └─────────┘   └──────────┘      └──────────┘        └──────────┘
                     K^T_tile1     softmax + correction
                  [128, 488]       → attn_wt_1 [1, 488]

  ── Online Softmax 校正更新 ──
  m_new       = max(row_max_0, max(scores_1))              ← 全局 max
  correction  = exp(row_max_0 - m_new)                     ← 校正因子
  acc_o       = correction × acc_o_0 + attn_wt_1 × V_tile1 ← [1, 128]
  row_sum     = correction × row_sum_0 + Σ exp(scores_1 - m_new)

  ══════════════════════════════════════════════════════════════════════════
  最终归一化:
  ══════════════════════════════════════════════════════════════════════════
  O_head = acc_o / row_sum                                  ← [1, 128]
```

### 7.5 Q 的分块策略：不分块，共享加载

**Q 不沿任何维度分块**，因为 Decode 阶段 Q 只有一个向量 [1, 128]，足够小到完全放入 shared memory。

```
  Q_head [1, 128] (256 bytes FP16)
  ┌─────────────────────────────────────────────────────────────┐
  │ q_0  q_1  q_2  q_3 ... q_62  q_63 │ q_64 ... q_126 q_127  │
  │         half2[0..63]               │     (未使用的线程等待)  │
  └────────────────┬───────────────────┴────────────────────────┘
                   │
         Warp 0-1 (Thread 0-63) 加载
         使用 half2 向量化: 每线程加载 4 bytes

  加载代码:
  ┌──────────────────────────────────────────────────────────────┐
  │ if (tid < head_size / 2) {  // tid < 64                     │
  │     s_query_h2[tid] = q_ptr_h2[tid];  // half2 load         │
  │ }                                                           │
  │ __syncthreads();  // 确保 Q 对所有 256 线程可见               │
  └──────────────────────────────────────────────────────────────┘

  加载后 Q 驻留在 shared memory，整个 kernel 生命周期内不变:
  - 每个 tile 的 Q·K 点积阶段被 256 线程反复读取
  - Write-once，Read-many 访问模式
  - 避免 256 线程 × ⌈1000/512⌉ tiles = 512 次 global memory Q 读取
```

### 7.6 K 的分块策略：沿序列维度 Tiling

K 是计算量最大的部分，沿 KV 位置轴（序列维度）按 `ONLINE_TILE_K=512` 分块。

```
  K_head 全局有效形状: [kv_len, head_size] = [1000, 128]

  分块方式 (沿行/序列维度):
  ┌──────────────────────┐  ┌──────────────────────┐
  │      K_tile0         │  │      K_tile1         │
  │  [512, 128]          │  │  [488, 128]          │
  │  KV位置 0 ~ 511      │  │  KV位置 512 ~ 999    │
  │                      │  │                      │
  │  k_0   = K[0,   0:128] │  │  k_512 = K[512, 0:128] │
  │  k_1   = K[1,   0:128] │  │  k_513 = K[513, 0:128] │
  │  ...                 │  │  ...                 │
  │  k_511 = K[511, 0:128] │  │  k_999 = K[999, 0:128] │
  └──────────────────────┘  └──────────────────────┘

  Tile 内线程分工 (以 Tile 0, tile_len=512 为例):
  ┌───────────────────────────────────────────────────────────────────┐
  │ 256 线程处理 512 个 KV 位置，每线程 2 个位置:                      │
  │                                                                   │
  │ Thread 0:   dot(Q, K[0])   → s_scores[0]     ← 迭代 1            │
  │             dot(Q, K[256]) → s_scores[256]   ← 迭代 2            │
  │                                                                   │
  │ Thread 1:   dot(Q, K[1])   → s_scores[1]                          │
  │             dot(Q, K[257]) → s_scores[257]                        │
  │                                                                   │
  │ Thread t:   dot(Q, K[t])       → s_scores[t]                      │
  │             dot(Q, K[t+256])   → s_scores[t+256]                  │
  │                                                                   │
  │ Thread 255: dot(Q, K[255]) → s_scores[255]                        │
  │             dot(Q, K[511]) → s_scores[511]                        │
  └───────────────────────────────────────────────────────────────────┘

  每个 dot(Q, K[k]) 的向量化计算过程:
  ┌───────────────────────────────────────────────────────────────────┐
  │ Q 和 K[k] 都是 128 个 half = 256 bytes                           │
  │                                                                   │
  │ 使用 float4 加载 (128-bit = 8 个 half / 次)                       │
  │ 循环 head_size/8 = 16 次:                                         │
  │                                                                   │
  │   iter 0: q_packed = float4(s_query[0:7])     ← shared mem       │
  │           k_packed = float4(K[k, 0:7])        ← global mem __ldg │
  │           → 拆为 4 个 half2 → 转 float2 → fmaf 累加               │
  │                                                                   │
  │   iter 1: q_packed = float4(s_query[8:15])                        │
  │           k_packed = float4(K[k, 8:15])                           │
  │           → 同上                                                   │
  │                                                                   │
  │   ...                                                             │
  │   iter 15: q_packed = float4(s_query[120:127])                    │
  │            k_packed = float4(K[k, 120:127])                       │
  │            → 同上                                                  │
  │                                                                   │
  │ 最终: score = (dot.x + dot.y) × scale                            │
  │        score 写入 s_scores[k_idx]（shared memory）                │
  └───────────────────────────────────────────────────────────────────┘
```

**K 不在 head_size 维度分块的原因**：`head_size=128` 足够小，单个线程即可在 16 次 float4 迭代中完成整个 128 维的点积计算，无需跨线程协作 partial sum。

### 7.7 V 的分块策略：与 K 相同的序列分块，不同的线程映射

V 沿序列维度使用与 K 完全相同的 tile 边界，但线程到数据的映射方式完全不同：

```
  V_head 全局有效形状: [kv_len, head_size] = [1000, 128]

  与 K 相同的分块:
  ┌──────────────────────┐  ┌──────────────────────┐
  │      V_tile0         │  │      V_tile1         │
  │  [512, 128]          │  │  [488, 128]          │
  │  KV位置 0 ~ 511      │  │  KV位置 512 ~ 999    │
  └──────────────────────┘  └──────────────────────┘

  但 V 的线程映射与 K 截然不同（按列而非按行分配）:
  ┌───────────────────────────────────────────────────────────────────┐
  │ K: 每线程处理不同的 KV 位置 (行)，计算完整 128 维点积              │
  │ V: 每线程负责一个固定的输出维度 (列)，遍历 tile 内所有 KV 位置     │
  └───────────────────────────────────────────────────────────────────┘
```

**V 累加的线程分工图解**：

```
  V_tile0 [512, 128] — 矩阵视角 (行=KV位置, 列=head维度)
  ┌────────────────────────────────────────────────────────┐
  │ pos=0:   v[0,0]   v[0,1]   v[0,2]   ...   v[0,127]   │
  │ pos=1:   v[1,0]   v[1,1]   v[1,2]   ...   v[1,127]   │
  │ pos=2:   v[2,0]   v[2,1]   v[2,2]   ...   v[2,127]   │
  │ ...                                                    │
  │ pos=511: v[511,0] v[511,1] v[511,2] ... v[511,127]    │
  └────────────────────────────────────────────────────────┘
     ↕          ↕        ↕                      ↕
   Thr 0     Thr 1    Thr 2    ...           Thr 127
   (dim=0)   (dim=1)  (dim=2)               (dim=127)

  Thread 0 的工作: 遍历列 0，计算 acc_o += Σ attn_wt[k] × v[k, 0]
  Thread 1 的工作: 遍历列 1，计算 acc_o += Σ attn_wt[k] × v[k, 1]
  ...
  Thread 127 的工作: 遍历列 127，计算 acc_o += Σ attn_wt[k] × v[k, 127]
  Thread 128-255:  不参与 V 累加（已在 Q·K/softmax 阶段完成使命）

  内存访问模式 (对于某个 KV 位置 k):
  128 线程同时读取 V[k, head_offset+0..127]  ← 连续 256 bytes，完美 coalesced
```

**V 累加的 4 路展开**：每线程在 tile_len 维度上每次处理 4 个连续 KV 位置，充分利用指令级并行 (ILP)：

```
  Thread 0 (dim=0) 在 Tile 0 中的访问模式:

  迭代 1: 加载 V[0, 0], V[1, 0], V[2, 0], V[3, 0]     ← 4 路展开
          acc_o = fmaf(s[0], v0, fmaf(s[1], v1, fmaf(s[2], v2, fmaf(s[3], v3, acc_o))))

  迭代 2: 加载 V[4, 0], V[5, 0], V[6, 0], V[7, 0]
          acc_o = fmaf(s[4], v0, fmaf(s[5], v1, fmaf(s[6], v2, fmaf(s[7], v3, acc_o))))

  ... 共 512/4 = 128 次迭代
```

### 7.8 完整计算流程维度变化总图 — Qwen3-8B 实例

以 Block 0（head=0, kv_head=0, kv_len=1000）为例，展示完整的维度变化：

```
  ╔══════════════════════════════════════════════════════════════════════════════╗
  ║                    Flash Attention Decode 完整流程                          ║
  ║               Block 0 | head=0 | kv_head=0 | kv_len=1000                   ║
  ╠══════════════════════════════════════════════════════════════════════════════╣
  ║                                                                            ║
  ║  ① Q 加载 (一次性)                                                         ║
  ║  ─────────────────                                                         ║
  ║  Global Memory Q[0:128] ──half2──→ Shared Memory s_query[128]              ║
  ║  维度: [4096] 中取 [128]  →  s_query: [128] (half)                         ║
  ║                                                                            ║
  ║  ② 初始化 Online Softmax 状态                                              ║
  ║  ────────────────────────────                                              ║
  ║  row_max = -FLT_MAX     (标量)                                             ║
  ║  row_sum = 0.0          (标量)                                             ║
  ║  acc_o   = 0.0          (标量 × 128 线程 = 等效 [128] 向量)                 ║
  ║                                                                            ║
  ╠═══════════════════════ Tile 0: [0, 512) ═══════════════════════════════════╣
  ║                                                                            ║
  ║  ③ Q · K_tile0^T (256 线程并行)                                            ║
  ║  ──────────────────────────────                                            ║
  ║                                                                            ║
  ║    s_query   × K_cache[0:512, 0:128]^T  =  s_scores                       ║
  ║    [1, 128]    [512, 128]^T=[128,512]      [1, 512]                        ║
  ║                                                                            ║
  ║    每线程：dot(s_query[128], K[k, 0:128]) × scale → s_scores[k]           ║
  ║    Thread t 处理 k = t 和 k = t+256 (共 2 个 KV 位置)                      ║
  ║    每个 dot: 16 次 float4 load × 4 次 half2→float2 fmaf = 128 次 fmaf     ║
  ║                                                                            ║
  ║  ④ Max 规约                                                                ║
  ║  ────────────                                                              ║
  ║    s_scores[512] ──warp shuffle──→ s_reduce[8] ──thread 0──→ m_j (标量)    ║
  ║    m_new = max(-FLT_MAX, m_j) = m_j                                        ║
  ║                                                                            ║
  ║  ⑤ Exp + Sum                                                               ║
  ║  ────────────                                                              ║
  ║    s_scores[k] = exp(s_scores[k] - m_new)  或 0 (若 < FTZ)                ║
  ║    维度不变: [1, 512]                                                       ║
  ║    规约得 l_j = Σ s_scores[k]  (标量)                                       ║
  ║                                                                            ║
  ║  ⑥ V 累加 (128 线程参与)                                                    ║
  ║  ─────────────────────────                                                 ║
  ║    correction = exp(-FLT_MAX - m_new) ≈ 0                                  ║
  ║    acc_o = 0 × 0 + Σ s_scores[k] × V[k, my_dim]  (每线程一个标量)          ║
  ║                                                                            ║
  ║    等效矩阵运算:  attn_wt_0 [1,512] × V_tile0 [512,128] = partial_O [1,128]║
  ║                                                                            ║
  ║  ⑦ 状态更新                                                                ║
  ║    row_max = m_j, row_sum = 0 × 0 + l_j = l_j                             ║
  ║                                                                            ║
  ╠═══════════════════════ Tile 1: [512, 1000) ════════════════════════════════╣
  ║                                                                            ║
  ║  ⑧ Q · K_tile1^T (256 线程并行)                                            ║
  ║  ──────────────────────────────                                            ║
  ║                                                                            ║
  ║    s_query   × K_cache[512:1000, 0:128]^T = s_scores                      ║
  ║    [1, 128]    [488, 128]^T=[128, 488]       [1, 488]                      ║
  ║                                                                            ║
  ║    Thread t 处理 k = t (若 t<488) 和 k = t+256 (若 t+256<488)             ║
  ║    Thread 0-231: 处理 2 个位置; Thread 232-255: 处理 1 个位置              ║
  ║    s_scores[488..511] 未被写入（tile_len=488 < TILE_K=512）                ║
  ║                                                                            ║
  ║  ⑨ Max 规约 → m_j'                                                        ║
  ║    m_new = max(row_max, m_j')    ← 跨两个 tile 的全局最大值                ║
  ║                                                                            ║
  ║  ⑩ Exp + Sum → l_j'                                                       ║
  ║    s_scores[k] = exp(s_scores[k] - m_new) 或 0                            ║
  ║    维度: [1, 488]                                                           ║
  ║                                                                            ║
  ║  ⑪ Online Softmax 校正 + V 累加                                            ║
  ║  ──────────────────────────────                                            ║
  ║    correction = exp(row_max_old - m_new)  ← 补偿之前 tile 的 max 偏差      ║
  ║    acc_o = correction × acc_o_old + Σ s_scores[k] × V[k+512, my_dim]      ║
  ║                                                                            ║
  ║    等效矩阵: correction × partial_O_0 [1,128]                              ║
  ║            + attn_wt_1 [1,488] × V_tile1 [488,128]                        ║
  ║            = acc_o [1, 128]                                                ║
  ║                                                                            ║
  ║  ⑫ 状态更新                                                                ║
  ║    row_max = m_new                                                         ║
  ║    row_sum = correction × row_sum_old + l_j'                               ║
  ║                                                                            ║
  ╠═══════════════════════ 最终归一化 ═════════════════════════════════════════╣
  ║                                                                            ║
  ║  ⑬ O = acc_o / row_sum                                                     ║
  ║     Thread tid (tid < 128): O[head×128 + tid] = acc_o / row_sum            ║
  ║     输出维度: [1, 128] (half)                                               ║
  ║                                                                            ║
  ║     等价于标准 Attention:                                                   ║
  ║     O = softmax(Q [1,128] × K [1000,128]^T / √128) × V [1000,128]         ║
  ║       = [1, 128]                                                            ║
  ║                                                                            ║
  ╚══════════════════════════════════════════════════════════════════════════════╝
```

### 7.9 32 个 Block 并行处理全部 Head 的全局视图

```
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                  Qwen3-8B Decode: 32 Blocks 并行全局视图                     │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │                                                                             │
  │  Q 全局输入 [4096] = [32 heads × 128 dims]                                  │
  │  ┌──────┬──────┬──────┬─────── ... ──────┬──────┐                           │
  │  │head 0│head 1│head 2│                  │head31│                           │
  │  │[128] │[128] │[128] │                  │[128] │                           │
  │  └──┬───┴──┬───┴──┬───┴─────── ... ──────┴──┬───┘                           │
  │     │      │      │                         │                               │
  │     ↓      ↓      ↓                         ↓                               │
  │  Block 0 Block 1 Block 2    ...          Block 31                           │
  │     │      │      │                         │                               │
  │     │   GQA: 每 4 个 block 共享 KV head      │                               │
  │     ↓      ↓      ↓                         ↓                               │
  │  ┌──────────────────┐        ┌──────────────────┐                           │
  │  │ KV head 0        │  ...   │ KV head 7        │                           │
  │  │ K[1000,128]      │        │ K[1000,128]      │                           │
  │  │ V[1000,128]      │        │ V[1000,128]      │                           │
  │  │ ← Block 0,1,2,3  │        │ ← Block 28-31    │                           │
  │  └──────────────────┘        └──────────────────┘                           │
  │     │      │      │                         │                               │
  │     ↓      ↓      ↓                         ↓                               │
  │  ┌──────┬──────┬──────┬─────── ... ──────┬──────┐                           │
  │  │O_h 0 │O_h 1 │O_h 2 │                  │O_h31 │                           │
  │  │[128] │[128] │[128] │                  │[128] │                           │
  │  └──────┴──────┴──────┴─────── ... ──────┴──────┘                           │
  │  O 全局输出 [4096] = [32 heads × 128 dims]                                  │
  │                                                                             │
  │  数据量汇总:                                                                 │
  │  ┌─────────────────────────────────────────────────────────────────────┐     │
  │  │ Q  读取:  32 × 128 × 2 bytes  = 8 KB                              │     │
  │  │ K  读取:  8 × 1000 × 128 × 2  = 2 MB  (8 个 KV head，各被读 4 次) │     │
  │  │ V  读取:  8 × 1000 × 128 × 2  = 2 MB  (8 个 KV head，各被读 4 次) │     │
  │  │ O  写出:  32 × 128 × 2 bytes  = 8 KB                              │     │
  │  │ 实际 global mem 访问 ≈ 4 MB + 16 KB（K/V 主导）                    │     │
  │  └─────────────────────────────────────────────────────────────────────┘     │
  └─────────────────────────────────────────────────────────────────────────────┘
```

### 7.10 数据流维度变化汇总表

以单个 Block（单个 head）为例，汇总每个计算阶段的输入/输出维度：

| 阶段 | 操作 | 输入形状 | 输出形状 | 存储位置 | 线程参与数 |
|------|------|---------|---------|---------|----------|
| Q 加载 | Global → Shared | Q_global: [4096] 中 [128] | s_query: [128] half | Shared Memory | 64 |
| Q·K^T (per tile) | 向量点积 | s_query: [128], K_tile: [tile_len, 128] | s_scores: [tile_len] float | Shared Memory | 256 |
| ×scale | 逐元素 | s_scores: [tile_len] | s_scores: [tile_len] | Shared Memory | 256 |
| Max 规约 | reduce-max | s_scores: [tile_len] | m_j: 标量 | Register → Shared | 256→8→1 |
| Exp | 逐元素 | s_scores: [tile_len], m_new: 标量 | s_scores: [tile_len] | Shared Memory | 256 |
| Sum 规约 | reduce-sum | s_scores: [tile_len] | l_j: 标量 | Register → Shared | 256→8→1 |
| V 累加 | 加权求和 | s_scores: [tile_len], V_tile: [tile_len, 128] | acc_o: 标量×128线程=[128] | Register | 128 |
| 校正 | ×correction | acc_o: [128], row_sum: 标量 | acc_o: [128], row_sum: 标量 | Register | 128 |
| 归一化 | ÷row_sum | acc_o: [128] | O_head: [128] half | Register → Global | 128 |

---

## 8. Kernel 逐阶段代码解读

### 8.1 初始化阶段

```cpp
const int head = blockIdx.x;         // 当前处理第几个 attention head
const int tid = threadIdx.x;         // 线程索引 [0, 255]
const int lane_id = tid & 31;        // warp 内的 lane (位运算优化 % 32)
const int warp_id = tid >> 5;        // warp 索引 (位运算优化 / 32)

// 从 GPU memory 读取 pos（CUDA Graph 兼容）
const int pos = *reinterpret_cast<const volatile int32_t*>(pos_ptr);
const int kv_len = pos + 1;

// GQA 映射：4 个 query head 共享一个 KV head
const int kv_head = head / kv_mul;
const int head_offset = kv_head * head_size;
```

**`volatile` 关键字**：防止编译器将 GPU memory 读取优化掉或缓存旧值，确保每次 kernel 执行时读到最新的 pos。

### 8.2 Q 向量加载

```cpp
if (tid < head_size / 2) {  // 当 head_size=128 时，tid < 64
    reinterpret_cast<half2*>(s_query)[tid] = reinterpret_cast<const half2*>(q_ptr)[tid];
}
__syncthreads();
```

- 只需 64 线程即可加载 128 个 half 值（每线程加载一个 half2 = 4 bytes）
- 全部 256 线程中只有前 64 个工作，其余等待
- 加载完后 `__syncthreads()` 确保所有线程可见

### 8.3 Phase 1: Q·K 点积计算

```cpp
for (int k_idx = tid; k_idx < tile_len; k_idx += ONLINE_BLOCK_SIZE) {
    const float4* k_ptr_f4 = reinterpret_cast<const float4*>(
        K_cache + (tile_start + k_idx) * kv_dim + head_offset);

    float2 dot = make_float2(0.0f, 0.0f);
    #pragma unroll
    for (int d = 0; d < head_size / 8; d++) {  // 128/8 = 16 iterations
        float4 q_packed = q_ptr_f4[d];           // 128-bit load from shared mem
        float4 k_packed = __ldg(k_ptr_f4 + d);   // 128-bit load from global mem

        const half2* qh = reinterpret_cast<const half2*>(&q_packed);
        const half2* kh = reinterpret_cast<const half2*>(&k_packed);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float2 qf = __half22float2(qh[i]);
            float2 kf = __half22float2(kh[i]);
            dot.x = fmaf(qf.x, kf.x, dot.x);
            dot.y = fmaf(qf.y, kf.y, dot.y);
        }
    }

    float score = (dot.x + dot.y) * scale;
    s_scores[k_idx] = score;
    tile_max_local = fmaxf(tile_max_local, score);
}
```

**逐行解析**：

1. **float4 向量化加载**：每次读取 128 bits = 8 个 half 值
   - `float4 q_packed`：从 shared memory 读 Q 的 8 个连续 half
   - `float4 k_packed = __ldg(...)`：从 global memory 通过只读缓存读 K 的 8 个连续 half
   
2. **half2 → float2 转换 + fmaf**：
   - 将 8 个 half 拆为 4 个 `half2`
   - 每个 `half2` 转为 `float2` 后做 fused multiply-add
   - 使用两个独立累加器 `dot.x` 和 `dot.y` 提升 ILP

3. **循环次数**：`head_size/8 = 128/8 = 16` 次外层 × 4 次内层 = 64 次 fmaf

4. **__ldg 内联函数**：通过只读纹理缓存路径加载，对 read-only 数据有更好的缓存行为

### 8.4 Phase 1.5: Max 规约

```cpp
// Warp 内规约 (Butterfly pattern)
for (int offset = 16; offset > 0; offset >>= 1)
    tile_max_local = fmaxf(tile_max_local, __shfl_xor_sync(0xffffffff, tile_max_local, offset));
if (lane_id == 0) s_reduce[warp_id] = tile_max_local;
__syncthreads();

// Thread 0 做跨 warp 最终规约
float m_j;
if (tid == 0) {
    m_j = s_reduce[0];
    for (int w = 1; w < ONLINE_NUM_WARPS; w++)
        m_j = fmaxf(m_j, s_reduce[w]);
    s_reduce[0] = m_j;
}
__syncthreads();
m_j = s_reduce[0];  // 所有线程读取全局最大值
```

**两级规约流程**：

```
Level 1 (Warp 内 - 无共享内存, 使用 shuffle):
  Warp 0: 32 threads → s_reduce[0]
  Warp 1: 32 threads → s_reduce[1]
  ...
  Warp 7: 32 threads → s_reduce[7]

Level 2 (跨 Warp - Thread 0 串行):
  Thread 0: max(s_reduce[0..7]) → s_reduce[0]

Broadcast:
  所有线程从 s_reduce[0] 读取
```

### 8.5 Phase 2: Exp 计算 + Sum 规约

```cpp
float tile_sum_local = 0.0f;
for (int k_idx = tid; k_idx < tile_len; k_idx += ONLINE_BLOCK_SIZE) {
    float val = s_scores[k_idx] - m_new;
    float exp_val = (val > SOFTMAX_FTZ) ? expf(val) : 0.0f;  // Flush-to-zero
    s_scores[k_idx] = exp_val;
    tile_sum_local += exp_val;
}
```

- **Flush-to-zero (FTZ)**：当 `score - max < -20.0` 时直接置零，避免计算极小的 exp 值
  - $e^{-20} \approx 2 \times 10^{-9}$，对 FP16 精度已无意义
  - 节省 `expf()` 调用开销（exp 是昂贵的特殊函数）

Sum 规约与 Max 规约结构完全相同（warp shuffle → shared memory → thread 0）。

### 8.6 Phase 3: Online Softmax 校正 + V 累加

```cpp
// 校正之前累积的输出
float correction = expf(row_max - m_new);
acc_o *= correction;

// V 累加 (4-路展开)
if (my_dim < head_size) {
    const half* v_base = V_cache + head_offset + my_dim;
    int k = 0;
    for (; k + 3 < tile_len; k += 4) {
        const int base_pos = tile_start + k;
        float s0 = s_scores[k];
        float s1 = s_scores[k + 1];
        float s2 = s_scores[k + 2];
        float s3 = s_scores[k + 3];
        float v0 = __half2float(__ldg(v_base + (int64_t)(base_pos) * kv_dim));
        float v1 = __half2float(__ldg(v_base + (int64_t)(base_pos + 1) * kv_dim));
        float v2 = __half2float(__ldg(v_base + (int64_t)(base_pos + 2) * kv_dim));
        float v3 = __half2float(__ldg(v_base + (int64_t)(base_pos + 3) * kv_dim));
        acc_o = fmaf(s0, v0, acc_o);
        acc_o = fmaf(s1, v1, acc_o);
        acc_o = fmaf(s2, v2, acc_o);
        acc_o = fmaf(s3, v3, acc_o);
    }
    // 尾部处理...
}

row_max = m_new;
row_sum = fmaf(correction, row_sum, l_j);  // correction * row_sum + l_j
```

**V 累加的内存访问模式**：
- 每个线程负责一个固定的输出维度 `my_dim`
- 对每个 KV 位置，读取 `V[pos, head_offset + my_dim]`
- 这意味着 128 个线程同时读取同一行的连续 128 个 half 值 → **完美 coalesced** 访问

**correction 校正的数学原理**：
- 当处理到新 tile 时，之前的 max（`row_max`）可能小于新 tile 的 max（`m_j`）
- 需要将之前的累积结果乘以 $e^{m_{\text{old}} - m_{\text{new}}}$ 来补偿
- 这确保了最终结果等价于使用全局 max 的标准 softmax

### 8.7 最终归一化与输出

```cpp
if (my_dim < head_size && tid < head_size) {
    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    half* o_ptr = O + head * head_size;
    o_ptr[my_dim] = __float2half(acc_o * inv_sum);
}
```

- 只有 `tid < 128` 的线程写入输出（`tid >= 128` 的线程只在 Q·K 阶段做了贡献）
- 最终除以总和 `row_sum` 完成 softmax 归一化

---

## 9. 网格图示例

### 9.1 整体 Grid 映射

以 Qwen3-VL 8B 为例（`head_num=32, kv_head_num=8, head_size=128, pos=1999, kv_len=2000`）：

```
                        GPU Grid: dim3(32)
    ┌───────────────────────────────────────────────────────────┐
    │                                                           │
    │   Block 0    Block 1    Block 2    Block 3                │
    │  (head=0)   (head=1)   (head=2)   (head=3)               │
    │  kv_head=0  kv_head=0  kv_head=0  kv_head=0              │
    │     │          │          │          │                    │
    │     └──────────┴──────────┴──────────┘                    │
    │                    │                                      │
    │         共享 KV Head 0 的 K/V Cache                        │
    │                                                           │
    │   Block 4    Block 5    Block 6    Block 7                │
    │  (head=4)   (head=5)   (head=6)   (head=7)               │
    │  kv_head=1  kv_head=1  kv_head=1  kv_head=1              │
    │     │          │          │          │                    │
    │     └──────────┴──────────┴──────────┘                    │
    │                    │                                      │
    │         共享 KV Head 1 的 K/V Cache                        │
    │                                                           │
    │          ....... (类推) .......                            │
    │                                                           │
    │   Block 28   Block 29   Block 30   Block 31               │
    │  (head=28)  (head=29)  (head=30)  (head=31)              │
    │  kv_head=7  kv_head=7  kv_head=7  kv_head=7              │
    │     │          │          │          │                    │
    │     └──────────┴──────────┴──────────┘                    │
    │                    │                                      │
    │         共享 KV Head 7 的 K/V Cache                        │
    │                                                           │
    └───────────────────────────────────────────────────────────┘
```

### 9.2 单个 Block 内部结构 (以 Block 0 为例)

```
                    Block 0: 256 threads (8 warps)
    ┌──────────────────────────────────────────────────────────┐
    │                                                          │
    │   Warp 0: Thread[0..31]        dim 映射: [0..31]         │
    │   Warp 1: Thread[32..63]       dim 映射: [32..63]        │
    │   Warp 2: Thread[64..95]       dim 映射: [64..95]        │
    │   Warp 3: Thread[96..127]      dim 映射: [96..127]       │
    │   ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
    │   Warp 4: Thread[128..159]     Q·K打分辅助(不写出V)       │
    │   Warp 5: Thread[160..191]     Q·K打分辅助(不写出V)       │
    │   Warp 6: Thread[192..223]     Q·K打分辅助(不写出V)       │
    │   Warp 7: Thread[224..255]     Q·K打分辅助(不写出V)       │
    │                                                          │
    │   Shared Memory:                                         │
    │   ┌──────────┬──────────────────────┬─────────┐          │
    │   │ s_query  │      s_scores        │s_reduce │          │
    │   │ 256 B    │      2048 B          │  32 B   │          │
    │   │half[128] │     float[512]       │float[8] │          │
    │   └──────────┴──────────────────────┴─────────┘          │
    │                                                          │
    └──────────────────────────────────────────────────────────┘
```

### 9.3 Tiling 过程示意 (kv_len=2000)

```
KV Cache 位置轴:  0                    512                  1024                 1536       2000
                  ├────────────────────┼────────────────────┼────────────────────┼──────────┤
                  │     Tile 0         │     Tile 1         │     Tile 2         │  Tile 3  │
                  │    len = 512       │    len = 512       │    len = 512       │ len=464  │
                  └────────────────────┴────────────────────┴────────────────────┴──────────┘

处理 Tile 0:
═══════════════════════════════════════════════════════════════════════════════════
  Phase 1: Q·K 打分
  ┌────────────────────────────────────────────────────────────┐
  │  Thread 0:   dot(Q, K[0])   → s_scores[0]                 │
  │  Thread 0:   dot(Q, K[256]) → s_scores[256]               │
  │  Thread 1:   dot(Q, K[1])   → s_scores[1]                 │
  │  Thread 1:   dot(Q, K[257]) → s_scores[257]               │
  │  ...                                                       │
  │  Thread 255: dot(Q, K[255]) → s_scores[255]               │
  │  Thread 255: dot(Q, K[511]) → s_scores[511]               │
  └────────────────────────────────────────────────────────────┘
      ↓ __syncthreads()
  Phase 1.5: Max 规约
  ┌────────────────────────────────────────────────────────────┐
  │  Warp 0  ──(shuffle)──→ s_reduce[0]                       │
  │  Warp 1  ──(shuffle)──→ s_reduce[1]                       │
  │  ...                                                       │
  │  Warp 7  ──(shuffle)──→ s_reduce[7]                       │
  │      ↓ __syncthreads()                                    │
  │  Thread 0: max(s_reduce[0..7]) → s_reduce[0] = m_j       │
  │      ↓ __syncthreads()                                    │
  │  All threads: m_new = max(row_max, m_j)                   │
  └────────────────────────────────────────────────────────────┘
      ↓
  Phase 2: Exp + Sum
  ┌────────────────────────────────────────────────────────────┐
  │  Thread i: s_scores[i] = exp(s_scores[i] - m_new)        │
  │  Thread i: s_scores[i+256] = exp(s_scores[i+256] - m_new)│
  │  → warp shuffle sum → s_reduce → thread 0 sum → l_j      │
  └────────────────────────────────────────────────────────────┘
      ↓ __syncthreads()
  Phase 3: V 累加
  ┌────────────────────────────────────────────────────────────┐
  │  acc_o *= correction   (校正之前 tile 的结果)               │
  │                                                            │
  │  Thread 0 (dim=0):   Σ s_scores[k] * V[k, offset+0]      │
  │  Thread 1 (dim=1):   Σ s_scores[k] * V[k, offset+1]      │
  │  ...                                                       │
  │  Thread 127(dim=127): Σ s_scores[k] * V[k, offset+127]   │
  │  Thread 128-255: (不参与 V 累加)                            │
  └────────────────────────────────────────────────────────────┘
      ↓
  更新: row_max = m_new, row_sum = correction * row_sum + l_j
═══════════════════════════════════════════════════════════════════════════════════
处理 Tile 1 (重复上述流程)...
处理 Tile 2 (重复上述流程)...
处理 Tile 3 (最后一个 tile, tile_len=464, 部分线程空闲)...

最终: O[my_dim] = acc_o / row_sum
```

### 9.4 V 累加阶段的内存访问模式

```
                    V_cache 布局 (KV Cache, 行主序)
                    ┌─── kv_dim = 1024 ─────────────────────┐
                    │ kv_h0      kv_h1  ... kv_h7            │
                    │[0..127]  [128..255]   [896..1023]      │
          pos=0  ───┤  ▲         ▲             ▲             │
          pos=1  ───┤  │         │             │             │
          pos=2  ───┤  │         │             │             │
          ...       │  │         │             │             │
          pos=511───┤  │         │             │             │
                    └──┼─────────┼─────────────┼─────────────┘
                       │         │             │
                       │         │             │
                  Block 0-3   Block 4-7    Block 28-31
                  (head 0-3)  (head 4-7)   (head 28-31)
                  kv_head=0   kv_head=1    kv_head=7

  Block 0 内, Thread 0-127 同时访问同一行的 head_offset 到 head_offset+127:
  ┌──────────────────────────────────────────────────────────┐
  │ Thr0    Thr1    Thr2    ...    Thr127                    │
  │  ↓       ↓       ↓              ↓                        │
  │ V[k,0]  V[k,1]  V[k,2]  ...   V[k,127]  ← Coalesced!  │
  └──────────────────────────────────────────────────────────┘
```

---

## 10. 开源 FlashAttention 与 FlashDecoding 实现详解

本节深入讲解 [FlashAttention](https://github.com/Dao-AILab/flash-attention)（Dao et al.）和 [FlashDecoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)（Tri Dao, Daniel Haziza, Francisco Massa, Grigory Sizov）的开源实现原理，作为理解本项目 kernel 优化决策的背景知识。

### 10.1 FlashAttention 核心思想：IO-Aware Tiling

#### 10.1.1 传统 Attention 的内存瓶颈

标准 Attention 实现的计算流程：

```
传统实现 (PyTorch 标准路径):

  ① S = Q × K^T            写出 S ∈ R^{N×N} 到 HBM    ← O(N²) 显存
  ② P = softmax(S)          从 HBM 读 S, 写回 P 到 HBM  ← O(N²) 读 + O(N²) 写
  ③ O = P × V               从 HBM 读 P, 写出 O 到 HBM  ← O(N²) 读

  总 HBM 访问量 ≈ 4 × N² × d bytes (对 N=4096, d=128: ~8 GB 读写)
  中间矩阵 S, P 各占 N² 内存, 对长序列极其浪费
```

**核心问题**：Attention 是 **memory-bound** 操作。计算量是 $O(N^2 d)$，但中间矩阵 $S$ 和 $P$ 需要 $O(N^2)$ 的 HBM 读写。在 A100 上 HBM 带宽 2TB/s，但计算能力 312 TFLOPS，算术强度远超实际需要，瓶颈在 IO。

#### 10.1.2 FlashAttention 的 Tiling 算法

**核心洞察**：利用 GPU 的 SRAM（Shared Memory，~20MB 总量，19TB/s 带宽）代替 HBM 存储中间结果，将 Attention 分成小 tile 在 SRAM 中完成，**永不将完整的 $N \times N$ 矩阵写入 HBM**。

```
FlashAttention IO-Aware Tiling:

  GPU 内存层次:
  ┌─────────────────────────────────────────────────────────────────────┐
  │                         HBM (80GB, 2TB/s)                         │
  │  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐                      │
  │  │   Q   │  │   K   │  │   V   │  │   O   │  ← 输入/输出常驻 HBM  │
  │  │ N×d   │  │ N×d   │  │ N×d   │  │ N×d   │                      │
  │  └───┬───┘  └───┬───┘  └───┬───┘  └───┬───┘                      │
  │      │          │          │          ↑                           │
  │      │    Tile  │   Tile   │          │                           │
  │      ↓    Load  ↓   Load   ↓          │                           │
  │  ┌──────────────────────────────────────┐                         │
  │  │          SRAM / Shared Memory        │  ← 每 SM 48-192KB      │
  │  │         (19 TB/s 带宽)                │                         │
  │  │                                      │                         │
  │  │  Q_tile [B_r, d]                     │  ← 分块加载             │
  │  │  K_tile [B_c, d]                     │                         │
  │  │  V_tile [B_c, d]                     │                         │
  │  │  S_tile [B_r, B_c]  ← 在 SRAM 计算   │  ← 不写回 HBM!         │
  │  │  O_tile [B_r, d]    ← 在 SRAM 累加   │                         │
  │  │                                      │                         │
  │  └──────────────────────────────────────┘                         │
  └─────────────────────────────────────────────────────────────────────┘

  B_r = Q 的 tile 行数 (如 128)
  B_c = K/V 的 tile 列数 (如 128)
  d   = head_size (如 128)

  关键: S_tile = Q_tile × K_tile^T 在 SRAM 中计算并就地做 softmax，
        结果直接乘以 V_tile 累加到 O_tile，S_tile 从不写回 HBM。
```

#### 10.1.3 FlashAttention Forward Pass 伪代码

```
Algorithm: FlashAttention Forward (单个 head)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入: Q [N, d], K [N, d], V [N, d] in HBM
输出: O [N, d] in HBM

1. 设 B_r, B_c 为 tile 大小, T_r = ⌈N/B_r⌉, T_c = ⌈N/B_c⌉

2. 初始化 O = 0, ℓ = 0, m = -∞  (都在 HBM 中)

3. 将 K, V 分成 T_c 个 block: K_1,...,K_{T_c} 和 V_1,...,V_{T_c}
   将 Q 分成 T_r 个 block: Q_1,...,Q_{T_r}

4. 外层循环: for j = 1 to T_c:           ← 遍历 KV blocks
     从 HBM 加载 K_j, V_j 到 SRAM

   5. 内层循环: for i = 1 to T_r:         ← 遍历 Q blocks
        从 HBM 加载 Q_i, O_i, ℓ_i, m_i 到 SRAM

        a. S_ij = Q_i × K_j^T               ← [B_r, B_c] 在 SRAM 中计算
        b. m̃_ij = rowmax(S_ij)              ← 当前 tile 的行最大值
        c. P̃_ij = exp(S_ij - m̃_ij)         ← 在 SRAM 中就地计算
        d. ℓ̃_ij = rowsum(P̃_ij)             ← 行求和

        e. m_new = max(m_i, m̃_ij)           ← 更新全局 max
        f. ℓ_new = exp(m_i - m_new)·ℓ_i + exp(m̃_ij - m_new)·ℓ̃_ij
        g. O_i = diag(exp(m_i - m_new))^{-1} · (diag(ℓ_i)·exp(m_i-m_new)·O_i
                 + exp(m̃_ij - m_new) · P̃_ij × V_j)
           O_i = diag(ℓ_new)^{-1} · O_i     ← Online softmax 校正

        h. 将 O_i, ℓ_new, m_new 写回 HBM

6. 返回 O
```

**IO 复杂度对比**：

| 实现 | HBM 读写量 | 额外内存 |
|------|-----------|----------|
| 标准 Attention | $O(N^2 d + N^2)$ | $O(N^2)$（S, P 矩阵）|
| FlashAttention | $O(N^2 d^2 / M)$ | $O(N)$（ℓ, m 向量）|

其中 $M$ 是 SRAM 大小。当 $M \gg d^2$ 时，FlashAttention IO 近似 $O(N^2 d / \sqrt{M})$，相比标准实现减少 $\sqrt{M}/d$ 倍。

#### 10.1.4 FlashAttention 的并行化策略

FlashAttention v1/v2 的 GPU **并行维度**：

```
  FlashAttention 并行化 (训练/Prefill):
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Grid 维度:  (batch_size × head_num, T_r)     ← 跨 batch 和 Q block 并行
                                                  每个 block 处理一个 Q tile
  Block 维度: 128/256 线程

  ┌─────────────────────────────────────────────────────────────────┐
  │ batch 0, head 0:                                               │
  │   Block(0,0)→Q_tile0  Block(0,1)→Q_tile1  ... Block(0,T_r-1) │
  │   每个 block 遍历所有 K/V tiles (内层循环)                       │
  │                                                                │
  │ batch 0, head 1:                                               │
  │   Block(1,0)→Q_tile0  Block(1,1)→Q_tile1  ... Block(1,T_r-1) │
  │   ...                                                          │
  └─────────────────────────────────────────────────────────────────┘

  并行度 = batch_size × head_num × T_r
  例: batch=16, heads=32, N=2048, B_r=128 → T_r=16
      并行度 = 16 × 32 × 16 = 8192 blocks
```

**关键限制**：在 **Decode 阶段**（seq_len=1），$T_r = 1$，并行度降为 $\text{batch\_size} \times \text{head\_num}$。当 batch_size=1 时仅 32 个 block，A100 的 108 个 SM **利用率不足 30%**。这正是 FlashDecoding 要解决的问题。

### 10.2 FlashAttention 版本演进

| 版本 | 年份 | 关键改进 | 目标硬件 |
|------|------|---------|----------|
| **FlashAttention v1** | 2022 | IO-aware tiling + online softmax，首次实现无需 $O(N^2)$ HBM 访问 | A100 |
| **FlashAttention v2** | 2023 | 重写 kernel，优化并行策略和 work partitioning，前向 2x 加速 | A100/H100 |
| **v2.2** | 2023 | **集成 FlashDecoding**——decode 场景 split-K 并行，`flash_attn_with_kvcache` 接口 | A100/H100 |
| **v2.3** | 2023 | 滑动窗口注意力 (Sliding Window)，用于 Mistral 7B | A100/H100 |
| **v2.5** | 2024 | Paged KV Cache (PagedAttention)，与 vLLM 类似的分页管理 | A100/H100 |
| **FlashAttention v3** | 2024 | Hopper 专用优化: WGMMA、异步 TMA、FP8 支持 | H100 |
| **FlashAttention v4** | 2025 | CuTeDSL 实现，Hopper + Blackwell 两代 GPU 支持 | H100/B200 |

FlashAttention v2 相比 v1 的主要改进：

```
  FlashAttention v1 → v2 的变化:
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ① 循环顺序调换:
     v1: 外层遍历 KV blocks, 内层遍历 Q blocks
         → 每处理一个 KV block，需要读写所有 Q 的 O、ℓ、m
         → 对 O 的 HBM 写回次数 = T_c × T_r ≈ O(N²/B_r·B_c)

     v2: 外层遍历 Q blocks, 内层遍历 KV blocks
         → 每个 Q block 只在最后写回一次 O
         → 对 O 的 HBM 写回次数 = T_r ≈ O(N/B_r)
         → 减少约 T_c 倍 HBM 写入!

  ② 并行策略优化:
     v1: 并行度 = batch × heads
     v2: 并行度 = batch × heads × T_r (跨 Q blocks 并行)
         → 序列越长，并行度越高

  ③ Warp 内部工作分配:
     v1: 所有 warp 共同处理 Q×K^T 和 P×V
     v2: 不同 warp 分别处理 K/V 的不同部分，减少 warp 间同步
         4 warps: warp 0-3 各处理 K/V 的 1/4
         然后 warp 间通过 shared memory 交换结果
```

### 10.3 FlashAttention 的 Decode 接口：`flash_attn_with_kvcache`

从 v2.2 开始，FlashAttention 提供了专门的 KV Cache 推理接口：

```python
# FlashAttention KV Cache 推理接口
flash_attn_with_kvcache(
    q,                    # (batch_size, seqlen_q, nheads, headdim)
    k_cache,              # (batch_size, max_seqlen, nheads_k, headdim)  或 paged 格式
    v_cache,              # (batch_size, max_seqlen, nheads_k, headdim)  或 paged 格式
    k=None, v=None,       # 可选: 新 K/V 就地写入 cache
    cache_seqlens=None,   # (batch_size,) 每条序列的当前长度
    block_table=None,     # Paged KV Cache 的 block 映射表
    softmax_scale=None,   # 默认 1/√d
    causal=False,
    rotary_cos=None,      # 可选: 就地 RoPE 旋转
    rotary_sin=None,
)
```

**设计特点**：
- **KV Cache 原位更新**：新的 K/V 直接写入 cache，避免额外拷贝
- **RoPE 融合**：旋转位置编码在 kernel 内就地完成
- **GQA/MQA 原生支持**：Q 和 KV 头数不同时自动分组
- **Paged KV Cache**：支持 vLLM 风格的分页内存管理
- **内部自动选择**：根据 `seqlen_q` 大小自动切换 Prefill kernel 或 FlashDecoding (split-K) kernel

### 10.4 FlashDecoding：解决 Decode 阶段 GPU 利用率问题

#### 10.4.1 问题背景

FlashAttention 在 Decode 阶段（query_len=1）的 GPU 利用率极低：

```
  FlashAttention Decode 的 GPU 利用率问题:
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  训练/Prefill:
    Grid = (batch × heads, T_r)   T_r = ⌈seq_len / B_r⌉
    seq_len=2048, B_r=128 → T_r=16
    batch=16, heads=32 → 16×32×16 = 8192 blocks
    A100 (108 SMs): 每 SM 运行 ~76 blocks → 完全饱和 ✓

  Decode (query_len=1):
    Grid = (batch × heads, 1)     T_r = 1 (只有一个 query)
    batch=1, heads=32 → 1×32×1 = 32 blocks
    A100 (108 SMs): 只用了 32/108 = 30% 的 SM! ✗

    即使 batch=4: 4×32=128 blocks → 勉强填满
    但长序列需要小 batch (内存限制)，矛盾!

  ┌─────────────────────────────────────────────────────┐
  │  A100: 108 SMs                                      │
  │                                                     │
  │  FlashAttention Decode (batch=1, heads=32):         │
  │  [■][■][■][■][■][■][■][■][■][■][■][■][■][■][■][■]  │
  │  [■][■][■][■][■][■][■][■][■][■][■][■][■][■][■][■]  │
  │  [□][□][□][□][□][□][□][□][□][□][□][□][□][□][□][□]  │
  │  [□][□][□][□][□][□][□][□][□][□][□][□][□][□][□][□]  │
  │  [□][□][□][□][□][□][□][□][□][□][□][□][□][□][□][□]  │
  │  [□][□][□][□][□][□][□][□][□][□][□][□][□][□][□][□]  │
  │  [□][□][□][□][□][□][□][□][□][□]                    │
  │                                                     │
  │  ■ = 活跃 SM (32)    □ = 空闲 SM (76)               │
  │  GPU 利用率: 30%                                     │
  └─────────────────────────────────────────────────────┘
```

#### 10.4.2 FlashDecoding 的 Split-K 并行策略

FlashDecoding 的核心思想：**在 KV 序列维度上额外并行**，将长序列的 KV Cache 分配给多个 block 同时处理。

```
  FlashDecoding Split-K 策略:
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━

  原始 FlashAttention (单 block per head):
  ┌─────────────────────────────────────────────────────┐
  │  Q [1, d]                                           │
  │   ↓                                                 │
  │  Block 0: 顺序处理所有 K/V [0 → kv_len]              │
  │  ┌──────────────────────────────────────┐            │
  │  │ K[0..kv_len] → score → softmax → ×V │ → O        │
  │  └──────────────────────────────────────┘            │
  └─────────────────────────────────────────────────────┘

  FlashDecoding (多 block per head, split-K):
  ┌─────────────────────────────────────────────────────┐
  │  Q [1, d]  (每个 split 都读取完整的 Q)                │
  │   ↓  ↓  ↓  ↓                                       │
  │  Split 0     Split 1     Split 2     Split 3        │
  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐    │
  │  │K[0,S)  │  │K[S,2S) │  │K[2S,3S)│  │K[3S,N) │    │
  │  │partial │  │partial │  │partial │  │partial │    │
  │  │ O_0    │  │ O_1    │  │ O_2    │  │ O_3    │    │
  │  │ lse_0  │  │ lse_1  │  │ lse_2  │  │ lse_3  │    │
  │  └───┬────┘  └───┬────┘  └───┬────┘  └───┬────┘    │
  │      │           │           │           │          │
  │      └─────────┬─┴───────────┴───┬───────┘          │
  │                ↓   Reduce Kernel  ↓                  │
  │       ┌────────────────────────────────┐             │
  │       │  O = Σ exp(lse_i - lse_max)   │             │
  │       │      × O_i / Σ exp(lse_i-max) │             │
  │       └───────────────┬────────────────┘             │
  │                       ↓                              │
  │                    O [1, d]                          │
  └─────────────────────────────────────────────────────┘

  S = ⌈kv_len / num_splits⌉  (每个 split 处理的 KV 长度)
```

#### 10.4.3 FlashDecoding 三步算法详解

**Step 1: 分割 KV Cache（逻辑操作，无 GPU 计算）**

将 K/V Cache 沿序列维度等分为 `num_splits` 份，每份是原始 tensor 的一个 view，不涉及数据拷贝。

```
  KV Cache [kv_len, kv_dim] 分割为 num_splits=4 份:

  原始:    K[0 ─────────────────────────── kv_len]
  Split 0: K[0 ──── S)           S = ⌈kv_len/4⌉
  Split 1: K[S ──── 2S)
  Split 2: K[2S ─── 3S)
  Split 3: K[3S ─── kv_len)
```

**Step 2: Compute Kernel — 每个 split 独立计算 partial attention**

每个 split 作为一个独立的 FlashAttention 子问题：

```
  Grid: (batch × heads × num_splits)
  每个 block 输出:
    - partial_O [1, d]:    该 split 内的加权 V 求和 (未归一化)
    - lse (标量):          log-sum-exp = log(Σ exp(score_k - max)) + max
                           用于后续跨 split 的重新归一化

  例: batch=1, heads=32, num_splits=4
      Grid = 1 × 32 × 4 = 128 blocks → A100 利用率显著提升!
```

每个 split 内部使用标准 FlashAttention 的 online softmax tiling，与完整 kernel 完全相同，只是 KV 范围被限制在 `[split_start, split_end)` 内。最终每个 split 记录 `lse` 值用于后续校正。

**Step 3: Reduce Kernel — 跨 split 合并结果**

使用一个独立的 reduce kernel 将各 split 的 partial 结果合并为最终输出：

```
  Reduce 算法:
  ━━━━━━━━━━━━
  输入: partial_O[i] ∈ R^d, lse[i] ∈ R  (i = 0..num_splits-1)

  1. lse_max = max(lse[0], lse[1], ..., lse[num_splits-1])

  2. O = Σ_i  exp(lse[i] - lse_max) × partial_O[i]
         ─────────────────────────────────────────────
                Σ_i  exp(lse[i] - lse_max)

  数学等价性:
  因为 lse[i] = log(Σ_{k∈split_i} exp(s_k)) (包含了 max 校正)
  所以 exp(lse[i]) = Σ_{k∈split_i} exp(s_k)
  合并后恢复全局 softmax 结果
```

#### 10.4.4 FlashDecoding 的 GPU 利用率改善

```
  对比: batch=1, heads=32, kv_len=8192
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  FlashAttention:          32 blocks  → A100 利用率 30%
  FlashDecoding (splits=4): 128 blocks → A100 利用率 ~100% ✓
  FlashDecoding (splits=8): 256 blocks → A100 利用率 ~100% + 更好的隐藏延迟 ✓

  ┌─────────────────────────────────────────────────────┐
  │  A100: 108 SMs                                      │
  │                                                     │
  │  FlashDecoding (batch=1, heads=32, splits=4):       │
  │  [■][■][■][■][■][■][■][■][■][■][■][■][■][■][■][■]  │
  │  [■][■][■][■][■][■][■][■][■][■][■][■][■][■][■][■]  │
  │  [■][■][■][■][■][■][■][■][■][■][■][■][■][■][■][■]  │
  │  [■][■][■][■][■][■][■][■][■][■][■][■][■][■][■][■]  │
  │  [■][■][■][■][■][■][■][■][■][■][■][■][■][■][■][■]  │
  │  [■][■][■][■][■][■][■][■][■][■][■][■][■][■][■][■]  │
  │  [■][■][■][■][■][■][■][■][■][■][■][■]              │
  │                                                     │
  │  ■ = 活跃 SM (108)   □ = 空闲 SM (0)                │
  │  GPU 利用率: 100%! (相比 FlashAttention 的 30%)       │
  └─────────────────────────────────────────────────────┘
```

#### 10.4.5 FlashDecoding 的性能分析

FlashDecoding 的 benchmark 数据（A100, FP16, batch=1, GQA 16 heads / 2 KV heads, headdim=128）：

| 序列长度 | FlashAttention v2 | FlashDecoding | 加速比 |
|---------|-------------------|---------------|--------|
| 512 | ~相同 | ~相同 | ~1x |
| 2K | 较慢 | 快 | ~3-5x |
| 8K | 很慢 | 快 | ~10-20x |
| 32K | 极慢 | 快 | ~30-50x |
| 64K | -- | 最快 | ~50x |

序列长度增加时 FlashDecoding 的运行时间**几乎保持不变**（直到 split 数量足以饱和 GPU），而 FlashAttention 线性增长。

#### 10.4.6 FlashDecoding 的代价与权衡

| 方面 | 好处 | 代价 |
|------|------|------|
| GPU 利用率 | 从 <30% 提升到 ~100% | — |
| Kernel 数量 | — | compute + reduce 两个 kernel |
| 临时缓冲区 | — | 需要 `num_splits × head_num × head_size` 的 global memory |
| Launch 开销 | — | 两次 kernel launch 的 CPU 开销 |
| 短序列 | 不适用（开销大于收益）| splits=1 时退化为普通 FlashAttention |
| 实现复杂度 | — | 需要额外的 lse 传递和 reduce kernel |

### 10.5 FlashAttention 源码结构 (Dao-AILab/flash-attention)

```
flash-attention/
├── flash_attn/                   # Python 接口层
│   ├── flash_attn_interface.py   # 核心 API: flash_attn_func, flash_attn_with_kvcache
│   └── modules/mha.py            # Multi-Head Attention 封装
│
├── csrc/                          # CUDA/C++ 实现
│   └── flash_attn/                # SM80 (Ampere) CUDA kernels
│       ├── flash_fwd_kernel.h     # Forward kernel 模板
│       ├── flash_bwd_kernel.h     # Backward kernel 模板
│       ├── flash_fwd_launch_template.h
│       ├── flash_api.cpp          # PyTorch C++ 扩展接口
│       ├── softmax.h              # Online softmax 实现
│       └── utils.h                # Tensor Core MMA helpers
│
├── hopper/                        # SM90 (Hopper) H100 专用实现 (FlashAttention-3)
│   ├── flash_fwd_kernel.h
│   ├── named_barrier.hpp          # Hopper 命名屏障
│   └── ...                        # WGMMA, TMA 异步拷贝等
│
├── AI/                            # SM90/SM100 CuTeDSL 实现 (FlashAttention-4)
│
├── benchmarks/                    # 性能测试
├── tests/                         # 正确性测试
└── examples/inference/            # LLM 推理示例
```

**Kernel 技术栈**：

| 版本 | 核心库 | Tensor Core 指令 | Memory | 目标 GPU |
|------|--------|-----------------|--------|----------|
| FA v2 (csrc/) | CUTLASS 3.x | `mma.m16n8k16` (SM80) | Global → SMEM → Register | A100/A800 |
| FA v3 (hopper/) | CUTLASS 3.x | WGMMA (`wgmma.mma_async`) | TMA 异步 | H100/H800 |
| FA v4 (AI/) | CuTeDSL | 同 v3 + SM100 | 同 v3 + SM100 | H100/B200 |

### 10.6 本实现与开源方案的架构对比总览

```
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                     三种 Decode Attention 方案对比                        │
  ├──────────────────────────────────────────────────────────────────────────┤
  │                                                                        │
  │  ① FlashAttention v2 (Decode)                                          │
  │  ─────────────────────────────                                         │
  │  Grid: (batch × heads, 1)    ← 仅跨 batch 和 head 并行                  │
  │  每个 block: 顺序遍历所有 KV                                              │
  │  优点: 单 kernel, 无 reduce 开销                                         │
  │  缺点: GPU 利用率低 (batch=1 时 ~30% on A100)                            │
  │  适用: 大 batch 训练/长 prefill                                          │
  │                                                                        │
  │  ② FlashDecoding (v2.2+)                                               │
  │  ────────────────────────                                              │
  │  Grid: (batch × heads × splits)  ← 额外跨 KV split 并行                 │
  │  Kernel 1: 各 split 独立计算 partial attention + lse                     │
  │  Kernel 2: reduce 合并各 split                                          │
  │  优点: 长序列 GPU 利用率 ~100%                                           │
  │  缺点: 2 个 kernel launch, 需要临时 buffer                               │
  │  适用: 大 GPU + 长序列 + 小 batch                                        │
  │                                                                        │
  │  ③ 本实现 (OrinMLLM)                                                    │
  │  ─────────────────────                                                 │
  │  Grid: (heads)               ← 仅跨 head 并行                           │
  │  每个 block: tiled online softmax, 256 线程, 固定 shared memory          │
  │  优点: 单 kernel, CUDA Graph 兼容, Orin 硬件适配                         │
  │  缺点: 不适合超大 batch 或超长序列                                         │
  │  适用: 边端 Orin + 中短序列 + batch=1                                     │
  │                                                                        │
  └──────────────────────────────────────────────────────────────────────────┘
```

**为什么 OrinMLLM 不使用 FlashDecoding 的 Split-K 策略？**

Orin 的硬件限制使得 FlashDecoding 的收益有限：

| 因素 | A100 | Orin AGX | 影响 |
|------|------|----------|------|
| SM 数量 | 108 | 16 | Orin 上 32 blocks 已可填满, split-K 意义不大 |
| CPU 性能 | x86 高性能 | ARM A78AE | 多 kernel launch 在 Orin 上更昂贵 |
| 内存带宽 | 2TB/s HBM2e | 102.4 GB/s LPDDR5 | Orin 带宽低, reduce kernel 的额外 global memory 读写代价更高 |
| 典型序列长度 | 32K-128K | ≤8K | Orin 上序列较短, 单 block tiling 完全足够 |
| CUDA Graph | 可选优化 | 关键优化 | 两个 kernel 的 CUDA Graph capture 更复杂 |

---

## 11. 相对于开源实现的优化点

基于上节对 [FlashAttention](https://github.com/Dao-AILab/flash-attention) 和 [FlashDecoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) 的分析，本实现针对 **Orin 边端推理** 做了以下独特优化：

### 11.1 优化对比总表

| 优化维度 | 开源 Flash Attention | 本实现 | 优化收益 |
|---------|---------------------|--------|---------|
| **目标场景** | 通用 GPU（A100/H100）训练+推理 | Orin 边端推理 Decode | 针对性更强 |
| **Softmax 策略** | 标准 online softmax | online softmax + FTZ | 减少无效 exp 计算 |
| **Tile 大小** | 由 SM shared memory 大小决定 | 固定 512（CUDA Graph 兼容）| 支持 Graph 重放 |
| **线程数** | 取决于 tile 维度 | 256（8 warp），超配输出维度 | Q·K 阶段 2x 并行度 |
| **Shared Memory** | 固定+动态分配 | 固定 2336 bytes | CUDA Graph + 高 occupancy |
| **Position 传递** | kernel 参数 | GPU memory 指针 | CUDA Graph 兼容 |
| **V 读取** | Tensor Core MMA | 标量 fmaf with unroll-4 | Orin 无 FP16 HMMA，用 CUDA Core |
| **Q·K 点积** | Tensor Core (MMA) | half2 → float2 + fmaf 手动向量化 | Orin CUDA Core 友好 |
| **向量化宽度** | 最高 128-bit | float4 (128-bit) + half2 解包 | 最大化带宽利用 |
| **GQA 支持** | 需要额外处理 | 原生 kv_mul 映射 | 无额外开销 |

### 11.2 各优化点详细说明

#### 优化 1: CUDA Graph 兼容设计

**开源方案的问题**：
- 开源 FlashAttention 的 shared memory 大小取决于 kv_len
- Kernel 参数中包含 pos/kv_len 等动态值
- 这使得每一步 decode 都需要重新 launch kernel，无法使用 CUDA Graph

**本实现的方案**：
- Shared memory 大小**固定为编译期常量**（2336 bytes），不随 kv_len 变化
- Position 通过 GPU memory 指针传入（`pos_ptr`），kernel 内部通过 volatile 读取
- Grid/Block 维度固定
- 这使得整个 kernel launch 参数完全固定，可以被 CUDA Graph capture 并重放

**收益**：CUDA Graph 消除了每步 decode 的 CPU-GPU launch 延迟（约 5-15μs/launch），对 Orin 这种 CPU 较弱的平台效果显著。

#### 优化 2: 非对称线程分工 (256 线程 vs 128 输出维度)

**开源方案**：通常 block size = head_size，线程与输出维度 1:1 映射

**本实现**：
- Block size = 256，是 head_size = 128 的两倍
- Q·K 打分和 softmax 阶段：全部 256 线程并行，每个线程处理 `⌈tile_len / 256⌉` 个 KV 位置
- V 累加阶段：只有 tid < 128 的线程参与（每线程负责一个输出维度）

**原理**：
```
Q·K 打分是 compute-bound：
  - 256 线程同时计算不同 KV 位置的点积
  - 计算量 = tile_len × head_size × 2 FLOPs
  - 256 线程可以 2x 提升吞吐

V 累加是 memory-bound：
  - 每个输出维度需读取 tile_len 个 V 值
  - 128 线程已经可以 coalesce 覆盖 128 维
  - 额外线程无法加速（受带宽限制）
```

**收益**：在不增加 shared memory 的情况下，Q·K 阶段获得 2x 并行度提升。

#### 优化 3: Flush-to-Zero (FTZ) Softmax

**开源方案**：对所有 score 都计算 `expf(score - max)`

**本实现**：
```cpp
float exp_val = (val > SOFTMAX_FTZ) ? expf(val) : 0.0f;  // SOFTMAX_FTZ = -20.0
```

**原理**：
- $e^{-20} \approx 2.06 \times 10^{-9}$
- FP16 的最小正规数为 $\sim 6 \times 10^{-8}$，$e^{-20}$ 已远小于 FP16 精度
- 对于长序列，大部分 score 与 max 之差远超 -20，可以直接跳过 exp 计算
- `expf()` 是昂贵的特殊函数单元（SFU）操作，约 8-20 cycles

**收益**：长序列时可跳过大量 exp 计算。例如 kv_len=2000 时，若只有 top-50 左右的 score 显著，则约 97.5% 的 exp 可被跳过。

#### 优化 4: float4 向量化 Q·K 点积

**开源方案（GPU 端）**：使用 Tensor Core MMA 指令（mma.m16n8k16）

**本实现**：
```cpp
float4 q_packed = q_ptr_f4[d];           // 128-bit shared mem load
float4 k_packed = __ldg(k_ptr_f4 + d);   // 128-bit global mem load via read-only cache
// 解包为 4 个 half2 → 转 float2 → fmaf
```

**Orin 适配原理**：
- Orin（SM 8.7）的 Tensor Core 支持 INT8/INT4 MMA，但 FP16 HMMA 性能有限
- CUDA Core 上的 FP32 fmaf 在 Orin 上有更好的吞吐
- float4 加载一次读取 128 bits = 8 个 half，最大化内存事务效率
- 手动向量化完全控制了寄存器使用和指令调度

**收益**：在 Orin 上，手动 float4+fmaf 比调用 Tensor Core HMMA 更快（因为 Tensor Core 在 Orin 上主要优化 INT8）。

#### 优化 5: V 累加 Unroll-4 with fmaf

```cpp
for (; k + 3 < tile_len; k += 4) {
    float s0 = s_scores[k];     // shared mem read
    float s1 = s_scores[k + 1];
    float s2 = s_scores[k + 2];
    float s3 = s_scores[k + 3];
    float v0 = __half2float(__ldg(v_base + (int64_t)(base_pos) * kv_dim));
    float v1 = __half2float(__ldg(v_base + (int64_t)(base_pos + 1) * kv_dim));
    float v2 = __half2float(__ldg(v_base + (int64_t)(base_pos + 2) * kv_dim));
    float v3 = __half2float(__ldg(v_base + (int64_t)(base_pos + 3) * kv_dim));
    acc_o = fmaf(s0, v0, acc_o);
    acc_o = fmaf(s1, v1, acc_o);
    acc_o = fmaf(s2, v2, acc_o);
    acc_o = fmaf(s3, v3, acc_o);
}
```

**原理**：
- 4 路展开让编译器可以交错安排 memory load 和 compute 指令
- 当 v0 的加载还在等待时，v1/v2/v3 的加载请求已经发出
- fmaf 是单精度 fused multiply-add，比分开的 multiply + add 更精确且更快（1 op vs 2 ops）
- 4 路展开是 Orin memory 延迟隐藏的最佳平衡点（更多展开会增加寄存器压力）

#### 优化 6: Warp Shuffle 取代 Shared Memory 规约

**开源方案**：部分实现使用 atomicMax 或多次 shared memory 读写做规约

**本实现**：
```cpp
// Warp 内规约：纯寄存器操作，零延迟
for (int offset = 16; offset > 0; offset >>= 1)
    tile_max_local = fmaxf(tile_max_local, __shfl_xor_sync(0xffffffff, tile_max_local, offset));
```

**原理**：
- `__shfl_xor_sync` 在 warp 内直接交换寄存器值，无需经过共享内存
- 延迟：1 cycle vs shared memory 的 ~20-30 cycles
- 5 次 shuffle 即可完成 32 线程的规约
- 跨 warp 只需 8 个 shared memory 位置（8 warp × 1 float）

#### 优化 7: 序列长度自适应分支

```cpp
if (kv_len > 256) {
    // Tiled online softmax kernel
} else {
    // Non-tiled optimized kernel (lower overhead)
}
```

**原理**：
- 短序列（≤256）：所有 score 可直接放入 shared memory，无需 tiling，避免 online softmax 的 correction 计算开销
- 长序列（>256）：tiling + online softmax 避免 shared memory 溢出，且固定大小支持 CUDA Graph

### 11.3 与 FlashDecoding (Tri Dao) 的详细对比

基于第 10.4 节的 FlashDecoding 算法分析，下面总结本实现与之的核心差异：

| 方面 | FlashDecoding | 本实现 |
|------|--------------|--------|
| 并行粒度 | 多 block per head (split-K) | 单 block per head (tiled) |
| Kernel 数量 | 2（compute + reduce）| 1 |
| 适用场景 | 超长序列 + 大 GPU (A100/H100) | 中短序列 + 边端 (Orin) |
| Reduce 开销 | 需要 global memory 临时缓冲 + reduce kernel | 无（单 block 内 online softmax）|
| CUDA Graph 兼容 | 需要 capture 两个 kernel | 天然兼容（单 kernel + 固定参数）|
| GPU 利用率策略 | 增加 block 数填满 SM | 32 blocks 已填满 Orin 的 8-16 SM |
| 实现复杂度 | 高（需要跨 block 同步、临时 buffer）| 低 |

本实现采用单 block/head 的设计因为：
1. Orin 的 SM 数量有限（8/16 SM），32 个 block 已足够填充
2. 单 block 内完成避免了 reduce kernel 的额外 launch 开销
3. 序列长度通常 ≤8K，单 block tiling 完全足够
4. 单 kernel 设计使 CUDA Graph capture 更简单，减少 Orin ARM CPU 的调度开销

---

## 12. 优化原理详解

### 12.1 Online Softmax 原理

标准 softmax 需要两次遍历:
```
Pass 1: max_val = max(scores)
Pass 2: sum = Σ exp(score_i - max_val)
Pass 3: output_i = exp(score_i - max_val) / sum
```

Online softmax（Milakov & Gimelshein, 2018）将其合并为单次遍历，通过维护运行状态实现：

**数学推导**：

设处理到第 $j$ 个 tile 后的状态为 $(m_j, l_j, O_j)$，其中：
- $m_j$：前 $j$ 个 tile 的全局最大值
- $l_j$：归一化分母
- $O_j$：未归一化的输出

当处理第 $j+1$ 个 tile 时：

$$
m_{j+1} = \max(m_j, \max_{k \in \text{tile}_{j+1}} s_k)
$$

$$
\alpha = e^{m_j - m_{j+1}} \quad (\text{correction factor})
$$

$$
l_{j+1} = \alpha \cdot l_j + \sum_{k \in \text{tile}_{j+1}} e^{s_k - m_{j+1}}
$$

$$
O_{j+1} = \alpha \cdot O_j + \sum_{k \in \text{tile}_{j+1}} e^{s_k - m_{j+1}} \cdot V_k
$$

最终归一化：
$$
O_{\text{final}} = O_N / l_N
$$

**证明等价性**：

考虑最终的 $O_{\text{final}}$，展开递推式可以证明：

$$
O_{\text{final}} = \frac{\sum_{k=0}^{n-1} e^{s_k - m_N} \cdot V_k}{\sum_{k=0}^{n-1} e^{s_k - m_N}} = \text{softmax}(s) \cdot V
$$

因为每次 correction 都精确补偿了 max 变化带来的偏差。

### 12.2 IO 复杂度分析

| 操作 | 数据量 (bytes) | 访问次数 |
|------|---------------|---------|
| 读 Q（shared memory） | head_size × 2 = 256 | 1 次 global → 多次 shared |
| 读 K（global memory） | kv_len × head_size × 2 | 1 次（分 tile） |
| 读 V（global memory） | kv_len × head_size × 2 | 1 次（分 tile） |
| 写 O（global memory） | head_size × 2 = 256 | 1 次 |
| 读写 scores（shared memory） | ONLINE_TILE_K × 4 = 2048 | 每 tile 2 次 |

总 global memory 访问量 ≈ $2 \times n \times d \times 2$ bytes（K + V 各读一次）

这已是理论最优——不需要额外存储完整的 attention score 矩阵。

### 12.3 Orin 平台特定优化

Jetson Orin 的硬件特点：
- **GPU**: Ampere 架构, 1024/2048 CUDA Cores, 32/64 Tensor Cores
- **SM 数量**: 8 (Orin NX) / 16 (Orin AGX)
- **Shared Memory**: 48KB per SM (可配置)
- **L2 Cache**: 4MB (Orin AGX) / 2MB (Orin NX)
- **内存带宽**: 102.4 GB/s (LPDDR5)

针对这些特点的优化：

1. **低 shared memory 使用**（2336 bytes）→ 高 occupancy，每 SM 可跑多个 block
2. **float4 向量化**而非 Tensor Core → 在 Orin 上 CUDA Core FP32 性能更稳定
3. **TILE_K=512** 使得每 tile 的 K 数据 $= 512 \times 128 \times 2 = 128$KB，适配 L2 Cache
4. **单 kernel 设计**避免多 kernel launch 的 CPU 开销（Orin 的 ARM CPU 较慢）

---

## 13. Nsight Compute 性能分析

本节使用 NVIDIA Nsight Compute (ncu) 对三个 FP16 Flash Attention 核函数进行了详细的性能 profiling 分析。分析在 Jetson Orin AGX（SM 8.7, 16 SMs）上进行，使用 `--set full` 采集所有可用的硬件计数器数据。

> **Profiling 工具**：NVIDIA Nsight Compute 2024.3.1.0  
> **Benchmark 程序**：`cuda_kernel_optimized/flash_attention_kernel/bench_flash_attn`  
> **测试参数**：head_size=128, head_num=32, kv_head_num=8, kv_dim=1024, decode_pos=290 (kv_len=291), prefill_seq_len=8  
> **NCU 报告文件**：  
> - `docs/ncu_flash_attn_decode_fp16_online_softmax.ncu-rep`  
> - `docs/ncu_flash_attn_decode_fp16_optimized.ncu-rep`  
> - `docs/ncu_flash_attn_prefill_fp16.ncu-rep`

### 13.1 三核函数 Profiling 概览对比

| 指标 | `decode_fp16_online_softmax` | `decode_fp16_optimized` | `prefill_fp16` |
|------|:---:|:---:|:---:|
| **Grid** | (32,1,1) | (32,1,1) | (32,8,1) |
| **Block** | (128,1,1) | (256,1,1) | (128,1,1) |
| **总线程数** | 4,096 | 8,192 | 32,768 |
| **寄存器/线程** | 40 | 40 | 42 |
| **动态 Shared Memory** | 1.31 KB | 2.37 KB | 4.35 KB |
| **Waves Per SM** | 0.17 | 0.33 | 1.60 |
| **Duration (us)** | 58.98 | 75.42 | 22.53 |
| **SM Frequency (GHz)** | 1.30 | 1.30 | 1.30 |
| **Elapsed Cycles** | 76,498 | 97,701 | 29,203 |
| **Compute (SM) Throughput** | 10.95% | 8.97% | 25.69% |
| **Memory Throughput** | 23.26% | 17.29% | 15.12% |
| **L1/TEX Cache Throughput** | 24.05% | 17.77% | 16.78% |
| **L2 Cache Throughput** | 23.26% | 17.29% | 5.65% |

### 13.2 `flash_attention_decode_kernel_fp16_online_softmax` 详细分析

#### 13.2.1 GPU Speed Of Light

```
SM Frequency:               1.30 GHz
Elapsed Cycles:             76,498
Duration:                   58.98 us
Compute (SM) Throughput:    10.95%
Memory Throughput:          23.26%
L1/TEX Cache Throughput:    24.05%
L2 Cache Throughput:        23.26%
SM Active Cycles:           62,799.31
```

**分析**：该 kernel 是典型的 **memory-bound** 操作（Memory Throughput 23.26% > Compute Throughput 10.95%），但整体带宽利用率仍然较低。主要瓶颈在于 grid 太小（仅 32 blocks），只能填充约 0.17 个 wave（Orin AGX 有 16 SMs），大部分硬件资源闲置。这是 Decode 阶段的固有限制——每次只处理一个 token 的 32 个 attention head。

#### 13.2.2 Compute Workload Analysis

```
Executed IPC Active:        0.46 inst/cycle
Executed IPC Elapsed:       0.43 inst/cycle
Issue Slots Busy:           11.49%
SM Busy:                    11.49%
```

**分析**：IPC 仅 0.46，远低于峰值，表明计算单元严重欠利用。这是因为 warp 大部分时间在等待内存操作完成（L1TEX scoreboard stall），符合 memory-bound kernel 的特征。

#### 13.2.3 Memory Workload Analysis

```
Mem Busy:                   22.90%
Max Bandwidth:              23.26%
L1/TEX Hit Rate:            33.30%
L2 Hit Rate:                74.82%
Mem Pipes Busy:             8.72%
```

**关键发现**：
1. **L1 命中率 33.30%**：Q 向量常驻 shared memory 避免了 L1 压力，但 K/V 的 stride 访问模式（`kv_dim=1024` 字节偏移）导致 L1 命中率不高
2. **L2 命中率 74.82%**：TILE_K=512 的 tiling 策略有效利用了 L2 cache，同一 tile 的 K/V 数据被多个 block（GQA 共享同一 kv_head 的 4 个 block）复用
3. **Global Memory 非合并访问**：平均每个 sector 只有 21.3/32 字节被利用（66.6%），产生 74,240 个多余 sector（占总 224,128 sector 的 33%），来自 V 累加阶段的跨步访问模式

#### 13.2.4 Scheduler & Warp State Statistics

```
One or More Eligible:       11.38%
Active Warps Per Scheduler: 2.00
Eligible Warps Per Scheduler: 0.12
Warp Cycles Per Issued:     17.58
Avg. Active Threads/Warp:   30.90
```

**Stall 分析**：每个 warp 平均每 17.6 个 cycle 才发射一条指令。
- **69.7% L1TEX scoreboard stall**：主要瓶颈，warp 等待 K/V 的 global memory 读取完成
- 每个 scheduler 仅 2.00 个 active warp（最大 12），只有 0.12 个 eligible，说明所有 warp 都在等待内存

#### 13.2.5 Occupancy

```
Theoretical Occupancy:      100%
Achieved Occupancy:         16.51%
Achieved Active Warps/SM:   7.93
Block Limit Registers:      12 blocks
Block Limit Shared Mem:     13 blocks
Block Limit Warps:          12 blocks
```

**分析**：理论 occupancy 达到 100%（寄存器 40 个/线程，shared memory 仅 1.31KB），但实际只有 16.51%，完全受限于 grid 大小（32 blocks / 16 SMs = 2 blocks/SM）。该 kernel 的资源占用已经最优化，瓶颈在于 Decode 阶段本身的工作量太小。

#### 13.2.6 优化建议

Nsight Compute 给出的优化建议及对应分析：

| NCU 建议 | 估计加速 | 分析 |
|----------|:--------:|------|
| Grid 太小无法填满 SM | — | Decode 阶段固有限制，可考虑 Flash Decoding（split-KV 并行）增大 grid |
| L1TEX scoreboard stall | 69.7% | V 累加阶段的 stride 访问是主因，可考虑 V 转置存储或 shared memory 预加载 |
| 全局内存非合并访问 | 26.14% | V 按列访问 `V[k, dim]` 产生 stride，可通过 V 的 head 维度连续存储优化 |
| 低 achieved occupancy | 76.74% | grid 大小限制，仅限 Decode 场景 |

### 13.3 `flash_attention_decode_kernel_fp16_optimized`（非 Tiled）详细分析

#### 13.3.1 GPU Speed Of Light

```
SM Frequency:               1.30 GHz
Elapsed Cycles:             97,701
Duration:                   75.42 us
Compute (SM) Throughput:    8.97%
Memory Throughput:          17.29%
L1/TEX Cache Throughput:    17.77%
L2 Cache Throughput:        17.29%
SM Active Cycles:           85,011.50
```

**与 online_softmax 对比**：该 kernel 耗时 **75.42us**，比 online_softmax 的 **58.98us** 慢 **27.9%**。虽然使用了 256 线程（vs online_softmax 的 128 线程），但 Compute 和 Memory Throughput 均更低，说明额外的线程并未有效转化为吞吐量。

#### 13.3.2 Compute & Scheduler Analysis

```
Executed IPC Active:        0.37 inst/cycle
Issue Slots Busy:           9.30%
Active Warps/Scheduler:     2.51
Eligible Warps/Scheduler:   0.10
Warp Cycles Per Issued:     27.15
```

**关键差异**：Warp Cycles Per Issued 高达 27.15（vs online_softmax 的 17.58），每条指令的平均等待周期增加了 54.4%。原因是非 tiled kernel 需要将全部 `kv_len` 个 score 存储在 shared memory 中，增大了 shared memory 用量（2.37KB vs 1.31KB），且一次性处理所有 K 位置缺少 tiling 带来的 L2 cache 局部性。

#### 13.3.3 Memory Workload

```
Mem Busy:                   17.14%
L1/TEX Hit Rate:            33.30%
L2 Hit Rate:                74.81%
Mem Pipes Busy:             6.55%
```

**分析**：L1/L2 命中率与 online_softmax 几乎相同（L1: 33.30%, L2: 74.81%），说明两者的内存访问模式本质相同。但 Mem Busy 从 22.90% 降至 17.14%，表明非 tiled kernel 的内存管线利用效率更低。

#### 13.3.4 Warp Stall 分析

- **70.54% L1TEX scoreboard stall**：与 online_softmax 相近（69.7%），都受限于 V 的 stride 访问
- **Shared Memory Bank Conflict**：检测到 1.2-way bank conflict（200 次冲突 / 1485 wavefront = 13.47%），这是 online_softmax 中未出现的额外开销，来自 `s_scores` 的写入模式

#### 13.3.5 Occupancy

```
Theoretical Occupancy:      100%
Achieved Occupancy:         20.94%
Achieved Active Warps/SM:   10.05
Waves Per SM:               0.33
```

256 线程 = 8 warps/block，32 blocks 分配到 16 SMs = 2 blocks/SM = 16 warps/SM。理论上 active warps 更多，但实际 achieved occupancy 仅 20.94%（vs online_softmax 的 16.51%），改善有限。

### 13.4 `flash_attention_prefill_kernel_fp16` 详细分析

#### 13.4.1 GPU Speed Of Light

```
SM Frequency:               1.30 GHz
Elapsed Cycles:             29,203
Duration:                   22.53 us
Compute (SM) Throughput:    25.69%
Memory Throughput:          15.12%
L1/TEX Cache Throughput:    16.78%
L2 Cache Throughput:        5.65%
SM Active Cycles:           23,939.44
```

**分析**：Prefill kernel 的 Compute Throughput（25.69%）是三个 kernel 中最高的，因为 Grid 为 (32, 8, 1) = 256 blocks，能填充 1.60 个 wave，SM 利用率显著提升。L2 Throughput 仅 5.65%，说明大部分数据命中了 L1 cache（63.13% L1 命中率）。

#### 13.4.2 Compute & Scheduler Analysis

```
Executed IPC Active:        1.13 inst/cycle
Issue Slots Busy:           28.51%
Active Warps/Scheduler:     8.00
Eligible Warps/Scheduler:   0.58
Warp Cycles Per Issued:     28.64
```

**关键观察**：IPC 达到 1.13，是三个 kernel 中最高的。Active Warps/Scheduler = 8.00 也大幅领先（decode 仅 2.00-2.51）。但 Warp Cycles Per Issued 高达 28.64，主要受限于 CTA barrier stall（44.53%）——这是因为 prefill kernel 中存在大量 `__syncthreads()` 同步点（每个 tile 内的 max reduction、exp 计算、V 累加等步骤之间都需要同步）。

#### 13.4.3 Memory Workload

```
Mem Busy:                   15.12%
L1/TEX Hit Rate:            63.13%
L2 Hit Rate:                73.10%
Mem Pipes Busy:             10.18%
```

**分析**：L1 命中率高达 63.13%（decode 仅 33.30%），得益于 prefill kernel 中每个线程负责固定的输出维度，V 的访问有更好的空间局部性。L2 命中率 73.10% 与 decode 相当。

#### 13.4.4 Warp Stall 分析

- **44.53% Barrier stall**：prefill kernel 的主要瓶颈！大量 `__syncthreads()` 导致 warp 在 barrier 处等待（decode kernel 中该值几乎为 0）
- **Avg. Active Threads/Warp: 23.69**（decode 为 30.90），说明存在显著的线程分化（divergence），来自 tile 尾部不满 128 线程时的条件分支
- **8 Divergent Branches**：远高于 decode 的 1，确认了分支分化问题

#### 13.4.5 Occupancy

```
Theoretical Occupancy:      83.33%
Achieved Occupancy:         69.00%
Achieved Active Warps/SM:   33.12
Registers/Thread:           42
Waves Per SM:               1.60
```

**分析**：受寄存器限制（42 regs/thread），理论 occupancy 降至 83.33%（Block Limit Registers = 10 blocks/SM）。但实际 achieved 69.00% 仍远高于 decode kernel（16-21%），主要得益于充足的 grid 大小（256 blocks）。

#### 13.4.6 Prefill 特有优化建议

| NCU 建议 | 估计加速 | 分析 |
|----------|:--------:|------|
| CTA Barrier Stall | 44.53% | 减少 `__syncthreads()` 次数或合并同步点 |
| 线程分化 | 8.52% | Tile 边界和条件分支导致，可尝试 padding 到 tile 对齐 |
| 非合并全局访问 | 17.28% | V 的跨步访问，可通过预加载到 shared memory 优化 |
| 非融合 FP32 指令 | 1.38% | 部分 FP32 运算可替换为 fmaf 以提升吞吐 |

### 13.5 三核函数性能瓶颈对比总结

```
                          decode_online_softmax    decode_optimized    prefill_fp16
                          ────────────────────    ────────────────    ────────────
 Duration (us)                    58.98                75.42              22.53
 IPC Active                       0.46                 0.37               1.13
 SM Busy (%)                     11.49                 9.30              28.51
 Memory Throughput (%)           23.26                17.29              15.12
 L1 Hit Rate (%)                 33.30                33.30              63.13
 L2 Hit Rate (%)                 74.82                74.81              73.10
 Achieved Occupancy (%)          16.51                20.94              69.00
 Waves Per SM                     0.17                 0.33               1.60
 Top Stall (%)            L1TEX 69.7%          L1TEX 70.5%       Barrier 44.5%
 Uncoalesced Sectors (%)         33.0%                33.0%              26.0%
```

**关键结论**：

1. **`decode_online_softmax` vs `decode_optimized`**：online_softmax 版本快 27.9%（58.98us vs 75.42us），尽管使用更少的线程（128 vs 256），但 tiling 策略和更低的 shared memory 用量带来了更好的内存局部性和更高的 Memory Throughput（23.26% vs 17.29%）

2. **Decode 核函数的共同瓶颈**：
   - Grid 太小（32 blocks / 16 SMs），SM 利用率极低（achieved occupancy 16-21%）
   - 69-70% 的 warp 周期花在等待 L1TEX scoreboard，受限于 V 的 stride 访问模式
   - 33% 的 global memory sector 是多余的（非合并访问），来自 V[k, dim] 的列访问

3. **Prefill 的差异化瓶颈**：
   - 更高的 SM 利用率（occupancy 69%，IPC 1.13），得益于充足的并行度
   - 但瓶颈从 L1TEX stall 转为 CTA barrier stall（44.5%），大量 `__syncthreads()` 是主因
   - 线程分化更严重（23.7 active threads/warp vs decode 的 30.9），来自 tile 边界条件

4. **潜在的优化方向**：
   - **Flash Decoding（Split-KV）**：将 kv_len 切分到多个 block 并行处理，增大 grid 以提升 SM 利用率
   - **V Cache 转置存储**：将 V 按 `[head_size, kv_len]` 而非 `[kv_len, head_size]` 存储，消除 stride 访问
   - **异步内存拷贝 (cp.async)**：在 V 累加阶段用异步拷贝预取下一个 tile 的数据，隐藏内存延迟
   - **Prefill Barrier 优化**：合并 tile 内的多个同步点，减少 `__syncthreads()` 频次

---

## 14. 总结

`flash_attention_decode_kernel_fp16_online_softmax` 是一个精心为 Orin 边端推理设计的 Flash Attention Decode kernel，其核心设计要点：

1. **Grid/Block 设计**：32 blocks（每个 head 一个）× 256 threads（8 warps），充分利用 Orin 的 SM 资源

2. **分块策略**：固定 TILE_K=512, 配合 online softmax 实现单 pass 处理任意长度序列，shared memory 固定 2336 bytes

3. **非对称线程分工**：256 线程在 Q·K 阶段全部参与（2x 并行度），V 阶段只有 128 线程工作（memory-bound 无需更多线程）

4. **CUDA Graph 兼容**：通过 GPU memory 传递 pos、固定 shared memory 和 grid/block 尺寸，实现完全兼容

5. **针对 Orin 的硬件适配**：放弃 Tensor Core 转用 CUDA Core float4+fmaf 向量化，适配 Orin 的 FP16 Tensor Core 弱势

6. **数值稳定性**：online softmax + FTZ threshold 保证精度的同时减少无效计算

这些优化使得该 kernel 在 Orin 平台上相比直接移植开源 FlashAttention 有显著的性能优势，同时保持了 CUDA Graph 兼容性以进一步降低推理延迟。
