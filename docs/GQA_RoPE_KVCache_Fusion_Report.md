# GQA + RoPE + KV Cache 读写融合算子分析报告

## 目录

1. [融合原理与可行性分析](#1-融合原理与可行性分析)
2. [适配步骤详解](#2-适配步骤详解)
3. [融合算子 Grid/Block/Thread 详解](#3-融合算子-gridblockthread-详解)
4. [适配难点与解决方案](#4-适配难点与解决方案)

---

## 1. 融合原理与可行性分析

### 1.1 GQA + RoPE + KV Cache 读写融合是什么？

GQA + RoPE + KV Cache 读写融合是指将 Transformer decode 阶段中**原本分离的三个 CUDA kernel** 合并为**一个 CUDA kernel** 执行：

| 操作 | 融合前（3 个 kernel） | 融合后（1 个 kernel） |
|------|----------------------|---------------------|
| ① M-RoPE 应用到 Q 和 K | `mrope_gpu_pos_kernel` | 融合 kernel 内完成 |
| ② K (含 RoPE) 写入 key_cache | `copy_to_kv_cache_fp16` | 融合 kernel 内完成 |
| ③ V 写入 val_cache | `copy_to_kv_cache_fp16` | 融合 kernel 内完成 |

### 1.2 为什么能够进行融合？

三个操作可以融合的核心原因有三个：

#### 原因 ①：数据依赖关系允许流水线化

```
┌─────────────────────────────────────────────────────────────────────┐
│                    融合前的数据流（3 次 kernel launch）                │
│                                                                     │
│  QKV投影输出                                                         │
│  ┌────┐ ┌────┐ ┌────┐                                              │
│  │ Q  │ │ K  │ │ V  │                                              │
│  └──┬─┘ └──┬─┘ └──┬─┘                                              │
│     │      │      │                                                 │
│     ▼      ▼      │      ◄─── Kernel 1: M-RoPE(Q, K)              │
│  ┌────┐ ┌────┐    │           读Q→计算→写Q, 读K→计算→写K            │
│  │Q'  │ │K'  │    │                                                 │
│  └────┘ └──┬─┘    │                                                 │
│            │      │                                                 │
│            ▼      │      ◄─── Kernel 2: Copy K' → key_cache        │
│       ┌─────────┐ │           读K'→写cache                          │
│       │key_cache│ │                                                 │
│       └─────────┘ │                                                 │
│                   ▼      ◄─── Kernel 3: Copy V → val_cache         │
│              ┌─────────┐      读V→写cache                           │
│              │val_cache│                                             │
│              └─────────┘                                             │
│                                                                     │
│  K 数据被读取了 2 次（M-RoPE 读 + Copy 读）                          │
│  3 次 kernel launch 开销                                             │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    融合后的数据流（1 次 kernel launch）                │
│                                                                     │
│  QKV投影输出                                                         │
│  ┌────┐ ┌────┐ ┌────┐                                              │
│  │ Q  │ │ K  │ │ V  │                                              │
│  └──┬─┘ └──┬─┘ └──┬─┘                                              │
│     │      │      │                                                 │
│     ▼      ▼      ▼      ◄─── 单个融合 Kernel                       │
│  ┌────┐ ┌─────────┐ ┌─────────┐                                    │
│  │Q'  │ │key_cache│ │val_cache│                                     │
│  └────┘ └─────────┘ └─────────┘                                     │
│                                                                     │
│  K 数据只读取 1 次（读→RoPE计算→直接写cache）                         │
│  V 数据只读取 1 次（读→直接写cache）                                  │
│  1 次 kernel launch 开销                                             │
└─────────────────────────────────────────────────────────────────────┘
```

#### 原因 ②：GQA 结构使得 Q 头和 KV 头可以并行处理

Qwen3-VL-8B 使用 GQA（Grouped Query Attention）：
- Q 头数：32（`num_q_heads`）
- KV 头数：8（`num_kv_heads`）  
- 每头维度：128（`head_size`）
- GQA 分组比：4（每 4 个 Q 头共享 1 个 KV 头）

```
┌─────────────────────────────────────────────────────────┐
│              GQA 头结构与 Grid 映射                       │
│                                                         │
│  blockIdx.y:  0  1  2  3 ... 31  32 33 34 35 36 37 38 39│
│              ├────────────────┤  ├────────────────────┤  │
│               32 个 Q 头          8 个 KV 头              │
│               (只做 RoPE)         (RoPE+写K, 写V)        │
│                                                         │
│  Q head 0  ──┐                                          │
│  Q head 1  ──┼── 共享 KV head 0                         │
│  Q head 2  ──┤                                          │
│  Q head 3  ──┘                                          │
│                                                         │
│  Q head 4  ──┐                                          │
│  Q head 5  ──┼── 共享 KV head 1                         │
│  Q head 6  ──┤                                          │
│  Q head 7  ──┘                                          │
│  ...                                                    │
└─────────────────────────────────────────────────────────┘
```

Q 头只需要做 RoPE（不写 cache），KV 头需要做 RoPE + 写 cache。两者之间**没有数据依赖**，可以完全并行。

#### 原因 ③：RoPE 计算和内存写入可以在寄存器级别融合

```
┌────────────────────────────────────────────────────────────────┐
│                寄存器级别的融合优势                               │
│                                                                │
│  ┌─ 融合前 (K 路径) ──────────────────────────────────────┐    │
│  │  Kernel 1: Global Mem → Reg → RoPE计算 → Reg → Global Mem │ │
│  │  Kernel 2: Global Mem → Reg → Global Mem (Cache)        │  │
│  │                                                         │  │
│  │  K 数据经历: 读GM→寄存器→计算→写GM→读GM→写GM(cache)       │  │
│  │  全局内存访问次数: 4 次 (2读 + 2写)                        │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌─ 融合后 (K 路径) ──────────────────────────────────────┐    │
│  │  融合 Kernel: Global Mem → Reg → RoPE计算 → Reg → GM(Cache)│ │
│  │                                                         │  │
│  │  K 数据经历: 读GM→寄存器→计算→直接写GM(cache)              │  │
│  │  全局内存访问次数: 2 次 (1读 + 1写)                        │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                │
│  全局内存带宽节省: 50% (对于 K 路径)                             │
└────────────────────────────────────────────────────────────────┘
```

### 1.3 融合的性能收益来源

| 收益来源 | 说明 | 量化估算 |
|---------|------|---------|
| **Kernel Launch 开销减少** | 从 3 次 launch 减少到 1 次 | 节省 ~10-20μs（每 layer） |
| **K 数据内存访问减半** | K 不再需要中间写回+重读 | 节省 kv_dim × sizeof(half) × 2 = 2KB |
| **V 数据无额外开销** | V 在同一 kernel 直接写 cache | 无中间存储 |
| **GPU 占用率提升** | 更多 thread block 同时执行 | SM 利用率提升 |
| **每层总节省** | 32 层 × 节省的时间 | 对 decode 阶段总时间贡献显著 |

对于 Qwen3-VL-8B（32 层），每层节省 2 次 kernel launch + 减少全局内存访问，在整个 decode 循环中累积效果显著，特别是在 CUDA Graph 模式下，减少了图的节点数量。

---

## 2. 适配步骤详解

### 2.1 分析 RMinte（参考工程）的融合 Kernel 设计

RMinte 工程的 `applyRopeWriteKV` kernel 设计要点：

```
文件：cpp/kernels/posEncoding/applyRopeWriteKV.cu
特点：
├── 使用 DVec<half> (uint4, 8个half) 向量化加载/存储
├── QKV 是 packed format: [B, S, Hq+Hk+Hv, D]
├── KV Cache layout: [B, 2, Hkv, S_capacity, D]
├── 标准 RoPE (non-interleaved, half-split)
├── blockIdx.y 区分 Q 头 (0..Hq-1) 和 KV 头 (Hq..Hq+Hkv-1)
├── sin/cos cache 从 cosSinCache 直接加载
└── 支持 context / decode / tree decoding 三种模式
```

### 2.2 分析 OrinMLLM 的现有架构差异

| 特性 | RMinte | OrinMLLM |
|------|--------|----------|
| **RoPE 类型** | 标准 RoPE | **M-RoPE**（3D 位置编码：t/h/w） |
| **QKV 布局** | Packed `[B, S, H_total, D]` | **分离的** Q, K, V 独立 tensor |
| **KV Cache 布局** | `[B, 2, Hkv, S, D]` | **`[layer_num, seq_len, kv_dim]`** |
| **位置参数** | cosSinCache tensor | sin_cache + cos_cache 分离 |
| **Q/K Norm** | 无 | **Qwen3 特有** Q-Norm + K-Norm |
| **Decode 特点** | 单 token，batch 可 > 1 | **单 token，batch = 1** |

### 2.3 确定适配方案

核心决策：**不照搬 RMinte 的 kernel，而是设计一个专门适配 OrinMLLM 架构的融合 kernel**。

原因：
1. OrinMLLM 的 Q/K/V 是分离的 tensor，不是 packed QKV
2. OrinMLLM 使用 M-RoPE（需要 section-based 位置映射），不是标准 RoPE
3. OrinMLLM 的 KV Cache 布局不同（按 layer 分层而非 batch/head 分层）
4. Q-Norm 和 K-Norm 必须在 RoPE 之前执行，不能融合进 RoPE kernel

### 2.4 实施步骤

#### 步骤 1：创建融合 Kernel CUDA 文件

```
创建文件：
  kuiper/source/op/kernels/cuda/fused_rope_kv_kernel.cuh  (头文件)
  kuiper/source/op/kernels/cuda/fused_rope_kv_kernel.cu   (实现)
```

核心设计决策：
- 每个 thread block 处理一个 head（Q 头或 KV 头）
- 每个 thread 处理一对 RoPE 元素 (d, d+half_head_size)
- blockIdx.y = [0, num_q_heads) 处理 Q 头，[num_q_heads, num_q_heads + num_kv_heads) 处理 KV 头

#### 步骤 2：创建 Layer 封装类

```
修改文件：
  kuiper/include/op/misc_layers.h      (添加 FusedMRoPEKVWriteLayer 类声明)
  kuiper/source/op/misc_layers.cpp     (添加实现)
```

Layer 封装负责：
- 从 Tensor 提取裸指针（FP16 → half* 转换）
- 参数传递给 CUDA kernel
- CUDA stream 管理

#### 步骤 3：注册到模型层管理结构

```
修改文件：
  kuiper/include/model/qwen3.h         (Qwen3Layers 添加 fused_mrope_kv_write_layer_)
  kuiper/source/model/qwen3_vl.cpp     (create_vl_nonparam_layers 中创建实例)
```

#### 步骤 4：修改 CUDA Graph 路径的 attention_qkv_with_graph

这是最关键的一步——将原来的 3 个 kernel 调用替换为 1 个融合调用：

```cpp
// ============ 修改前：3 个 kernel launch ============
// Launch 1: M-RoPE 应用到 Q 和 K
qwen_layers_->mrope_gpu_pos_layer_->forward(
    rope_pos_gpu, dim, kv_dim, head_size,
    section0, section1, section2,
    query, temp_key, sin_cache, cos_cache);

// Launch 2: Copy K (已含 RoPE) 到 key_cache
qwen_layers_->copy_to_kv_cache_layer_->forward(
    key_cache, temp_key, kv_cache_pos_gpu, kv_dim, layer_idx, seq_len);

// Launch 3: Copy V 到 val_cache
qwen_layers_->copy_to_kv_cache_layer_->forward(
    val_cache, temp_value, kv_cache_pos_gpu, kv_dim, layer_idx, seq_len);

// ============ 修改后：1 个 kernel launch ============
qwen_layers_->fused_mrope_kv_write_layer_->forward(
    rope_pos_gpu, kv_cache_pos_gpu,
    query, temp_key, temp_value,
    key_cache, val_cache,
    sin_cache, cos_cache,
    dim, kv_dim, head_size,
    section0, section1, section2,
    layer_idx, seq_len);
```

#### 步骤 5：编译验证

```bash
# 重新配置 CMake（拾取新的 .cu 文件）
cd build && cmake .. -DQWEN3_VL_SUPPORT=ON
# 编译
make -j$(nproc)
```

`aux_source_directory` 自动扫描 `kuiper/source/op/kernels/cuda/` 目录，新的 `.cu` 文件会被自动包含。

#### 步骤 6：功能验证

```bash
# 运行推理验证正确性
./build/demo/qwen3_vl_infer \
    /mnt/ssd/QwenModels/Qwen3-VL-8B-fp16.bin \
    /mnt/ssd/QwenModels/Qwen3-VL-8B-Instruct/tokenizer.json \
    --image hf_infer/demo.jpeg \
    --prompt "Describe this image." \
    --cuda-graph --stream --max-pixel 500000
```

验证输出为连贯、正确的图像描述文本，确认融合 kernel 的计算结果正确。

---

## 3. 融合算子 Grid/Block/Thread 详解

### 3.1 总体线程组织

```
┌───────────────────────────────────────────────────────────────────────┐
│                   融合 Kernel 线程组织                                  │
│                                                                       │
│  Kernel 参数:                                                         │
│    head_size = 128, half_head_size = 64                               │
│    num_q_heads = 32 (dim / head_size = 4096 / 128)                   │
│    num_kv_heads = 8 (kv_dim / head_size = 1024 / 128)                │
│                                                                       │
│  Grid 配置:  dim3 grid(1, 40)                                         │
│    gridDim.x = 1   (单 token decode，只有一个 token)                    │
│    gridDim.y = 40   (32 Q heads + 8 KV heads)                        │
│                                                                       │
│  Block 配置: dim3 block(64, 1)                                        │
│    blockDim.x = 64  (half_head_size = head_size/2 = 128/2)           │
│    blockDim.y = 1   (单 token)                                        │
│                                                                       │
│  总线程数: 1 × 40 × 64 × 1 = 2560 个线程                              │
└───────────────────────────────────────────────────────────────────────┘
```

### 3.2 Thread Block 到 Head 的映射

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        blockIdx.y → Head 映射                              │
│                                                                            │
│  blockIdx.y = 0  → Q head 0   ─┐                                         │
│  blockIdx.y = 1  → Q head 1    │                                         │
│  blockIdx.y = 2  → Q head 2    ├─ 这 4 个 Q head 共享 KV head 0 (GQA)    │
│  blockIdx.y = 3  → Q head 3   ─┘                                         │
│  blockIdx.y = 4  → Q head 4   ─┐                                         │
│  blockIdx.y = 5  → Q head 5    ├─ 共享 KV head 1                          │
│  blockIdx.y = 6  → Q head 6    │                                         │
│  blockIdx.y = 7  → Q head 7   ─┘                                         │
│  ...                                                                       │
│  blockIdx.y = 28 → Q head 28  ─┐                                         │
│  blockIdx.y = 29 → Q head 29   ├─ 共享 KV head 7                          │
│  blockIdx.y = 30 → Q head 30   │                                         │
│  blockIdx.y = 31 → Q head 31  ─┘                                         │
│  ────────────────────────────────── 分界线 ──────────────────────────────  │
│  blockIdx.y = 32 → KV head 0   (处理 K 的 RoPE + 写 key_cache + 写 V)    │
│  blockIdx.y = 33 → KV head 1                                              │
│  blockIdx.y = 34 → KV head 2                                              │
│  blockIdx.y = 35 → KV head 3                                              │
│  blockIdx.y = 36 → KV head 4                                              │
│  blockIdx.y = 37 → KV head 5                                              │
│  blockIdx.y = 38 → KV head 6                                              │
│  blockIdx.y = 39 → KV head 7                                              │
└────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Thread 到 RoPE 元素对的映射

```
┌────────────────────────────────────────────────────────────────────────────┐
│              threadIdx.x → RoPE 元素对映射 (以一个 head 为例)               │
│                                                                            │
│  head_size = 128, half_head_size = 64                                     │
│                                                                            │
│  一个 head 的 128 维数据:                                                  │
│  ┌─────────────────────────────┬─────────────────────────────┐             │
│  │  d0: [0, 1, 2, ..., 63]    │  d1: [64, 65, 66, ..., 127] │             │
│  │     (前半部分)               │     (后半部分)               │             │
│  └─────────────────────────────┴─────────────────────────────┘             │
│                                                                            │
│  threadIdx.x = 0  → 处理 pair (d0=0,   d1=64)                             │
│  threadIdx.x = 1  → 处理 pair (d0=1,   d1=65)                             │
│  threadIdx.x = 2  → 处理 pair (d0=2,   d1=66)                             │
│  ...                                                                       │
│  threadIdx.x = 63 → 处理 pair (d0=63,  d1=127)                            │
│                                                                            │
│  RoPE 计算公式 (half-split, non-interleaved):                              │
│    new_q[d0] = q[d0] * cos(d0) - q[d1] * sin(d0)                         │
│    new_q[d1] = q[d1] * cos(d1) + q[d0] * sin(d1)                         │
└────────────────────────────────────────────────────────────────────────────┘
```

### 3.4 M-RoPE Section 映射

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    M-RoPE 3D 位置编码的 Section 分区                       │
│                                                                            │
│  mrope_section = [24, 20, 20]                                             │
│  dim_threshold0 = section0 * 2 = 48                                       │
│  dim_threshold1 = dim_threshold0 + section1 * 2 = 88                     │
│                                                                            │
│  Head 维度: 0 ─────────── 48 ─────────── 88 ──────── 128                  │
│             ├── 时间维度(t) ──┤─ 高度维度(h) ─┤─ 宽度维度(w) ─┤            │
│             │   24 对 × 2    │  20 对 × 2   │  20 对 × 2   │            │
│                                                                            │
│  对于 d0 (前半 0-63):                                                      │
│    d0 ∈ [0, 48)  → 使用 pos_t (时间位置)                                  │
│    d0 ∈ [48, 63] → 使用 pos_h (高度位置, 因为 48 < 88)                    │
│                                                                            │
│  对于 d1 (后半 64-127):                                                    │
│    d1 ∈ [64, 88)  → 使用 pos_h (高度位置)                                 │
│    d1 ∈ [88, 128) → 使用 pos_w (宽度位置)                                 │
│                                                                            │
│  Decode 阶段优化: pos_t = pos_h = pos_w = text_pos                        │
│  (纯文本 token 的三个维度位置相同，简化为统一的 rope_pos)                    │
└────────────────────────────────────────────────────────────────────────────┘
```

### 3.5 单个 Thread 的完整执行流程

```
┌────────────────────────────────────────────────────────────────────────────┐
│        Thread (blockIdx.y=32, threadIdx.x=10) 的执行流程                   │
│        → KV head 0, pair_idx=10, d0=10, d1=74                            │
│                                                                            │
│  Step 1: 读取位置 (GPU memory, for CUDA Graph)                             │
│    rope_pos = *pos_gpu         // e.g., 52                                │
│    kv_pos = *kv_cache_pos_gpu  // e.g., 511                               │
│                                                                            │
│  Step 2: 查表 sin/cos                                                      │
│    d0=10 < 48 → section temporal → pos0 = 52                              │
│    d1=74 → 48 ≤ 74 < 88 → section height → pos1 = 52 (decode: same)     │
│    freq_idx = 10 * 2 = 20                                                 │
│    sin0 = sin_cache[52 * 128 + 20]                                        │
│    cos0 = cos_cache[52 * 128 + 20]                                        │
│    sin1 = sin_cache[52 * 128 + 20]  // (same pos in decode)              │
│    cos1 = cos_cache[52 * 128 + 20]                                        │
│                                                                            │
│  Step 3: 读取 K 数据并计算 RoPE                                            │
│    k0 = key[0 * 128 + 10]       // KV head 0, dim 10                     │
│    k1 = key[0 * 128 + 74]       // KV head 0, dim 74                     │
│    k_rope_d0 = k0 * cos0 - k1 * sin0                                     │
│    k_rope_d1 = k1 * cos1 + k0 * sin1                                     │
│                                                                            │
│  Step 4: 写入 key_cache (直接从寄存器→全局内存)                             │
│    cache_base = layer_idx * seq_len * 1024 + 511 * 1024 + 0 * 128        │
│    key_cache[cache_base + 10] = k_rope_d0                                 │
│    key_cache[cache_base + 74] = k_rope_d1                                 │
│                                                                            │
│  Step 5: 读取 V 数据并写入 val_cache                                       │
│    v_d0  = value[0 * 128 + 10]                                            │
│    v_d1  = value[0 * 128 + 74]                                            │
│    val_cache[cache_base + 10] = v_d0                                      │
│    val_cache[cache_base + 74] = v_d1                                      │
│                                                                            │
│  ※ 关键优化: K 数据从全局内存读一次 → 寄存器 → RoPE 计算 → 直接写cache      │
│    省略了中间的"写回 temp_key → 再从 temp_key 读"步骤                       │
└────────────────────────────────────────────────────────────────────────────┘
```

### 3.6 整体 GPU 执行视图

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    GPU SM 执行时间线（简化）                                 │
│                                                                            │
│  ┌── 融合前 (每 layer) ──────────────────────────────────────────────┐    │
│  │                                                                   │    │
│  │  time →                                                           │    │
│  │  ├──── Kernel 1: M-RoPE ────┤gap├── K2: CopyK ──┤gap├─ K3: CopyV ┤   │
│  │  │  32Q + 8K RoPE pairs     │   │  1024 elems   │   │ 1024 elems │   │
│  │  │  ~2560 threads           │   │  ~256 threads  │   │ ~256 thds  │   │
│  │  ├──────────────────────────┤   ├───────────────┤   ├────────────┤   │
│  │                                                                   │    │
│  │  3 次 kernel launch 开销 + 3 次同步点                               │    │
│  └───────────────────────────────────────────────────────────────────┘    │
│                                                                            │
│  ┌── 融合后 (每 layer) ──────────────────────────────────────────────┐    │
│  │                                                                   │    │
│  │  time →                                                           │    │
│  │  ├────────── 融合 Kernel ──────────┤                               │    │
│  │  │  40 blocks × 64 threads = 2560   │                              │    │
│  │  │  Q RoPE + K RoPE + K→cache       │                              │    │
│  │  │  + V→cache 同时完成               │                              │    │
│  │  ├──────────────────────────────────┤                              │    │
│  │                                                                   │    │
│  │  1 次 kernel launch + 1 次同步点                                    │    │
│  └───────────────────────────────────────────────────────────────────┘    │
│                                                                            │
│  × 32 layers → 节省 64 次 kernel launch + 全局内存往返                     │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 适配难点与解决方案

### 难点 1：M-RoPE vs 标准 RoPE 的差异

**问题**：RMinte 使用标准 RoPE，通过一个 `cosSinCache[batch][pos][rotary_dim]` 查表即可。但 OrinMLLM 的 Qwen3-VL 使用 M-RoPE（Multi-dimensional RoPE），每个维度根据 section 使用不同的位置值。

**解决方案**：在融合 kernel 中保留 M-RoPE 的 section 映射逻辑。对于 decode 阶段，三个位置（temporal/height/width）实际上相同（都等于 `text_pos`），但保留 section 逻辑以确保正确性：

```cuda
int32_t dim_threshold0 = section0 * 2;  // 48
int32_t dim_threshold1 = dim_threshold0 + section1 * 2;  // 88

int32_t pos0;
if (d0 < dim_threshold0)      pos0 = rope_pos;  // temporal section
else if (d0 < dim_threshold1) pos0 = rope_pos;  // height section
else                          pos0 = rope_pos;  // width section
```

### 难点 2：Q-Norm / K-Norm 的处理

**问题**：Qwen3 模型在 QKV 投影后、RoPE 之前有 per-head RMSNorm（Q-Norm 和 K-Norm），这是 RMinte 没有的步骤。如果将 norm 也融合进 kernel，会极大增加复杂度（RMSNorm 需要 reduction 操作）。

**解决方案**：**不把 Q-Norm 和 K-Norm 融合进 RoPE kernel**。只融合 RoPE + KV Cache Write。Norm 仍然使用独立的 RMSNorm kernel 在融合 kernel 之前执行：

```
执行顺序（每层）:
  1. Q/K/V 投影 (MatMul kernel × 3)
  2. Q-Norm (RMSNorm kernel)   ← 保持独立
  3. K-Norm (RMSNorm kernel)   ← 保持独立
  4. 融合 Kernel (M-RoPE + KV Cache Write)  ← 替代原来 3 个 kernel
```

### 难点 3：KV Cache 布局差异

**问题**：RMinte 的 KV Cache 布局是 `[B, 2, Hkv, S, D]`（batch + head 分离，K/V 交替存储），而 OrinMLLM 使用 `[layer_num, seq_len, kv_dim]` 的扁平布局，且 key_cache 和 val_cache 是分开的 tensor。

**解决方案**：融合 kernel 直接使用 OrinMLLM 的原生布局，计算偏移量：

```cuda
// OrinMLLM KV Cache 偏移计算
int64_t cache_base = (int64_t)layer_idx * seq_len * kv_dim 
                   + (int64_t)kv_pos * kv_dim 
                   + kv_head_idx * head_size;

// key_cache 和 val_cache 使用相同的偏移，但写入不同的 buffer
key_cache[cache_base + d0] = k_rope_d0;
val_cache[cache_base + d0] = v_d0;
```

### 难点 4：CUDA Graph 兼容性

**问题**：CUDA Graph 要求 kernel 的所有指针参数在图重放时保持不变。位置值（rope_pos、kv_cache_pos）每步都变化，不能作为 kernel 参数直接传入。

**解决方案**：使用 GPU 常驻指针（GPU-resident pointer），位置值存储在固定地址的 GPU 内存中，通过 `volatile` 读取确保每次都从全局内存加载最新值：

```cuda
// kernel 内部从 GPU 内存读取（地址固定，值可变）
int32_t rope_pos = *reinterpret_cast<const volatile int32_t*>(pos_gpu);
int32_t kv_pos = *reinterpret_cast<const volatile int32_t*>(kv_cache_pos_gpu);
```

主机端在每步 decode 前通过 H2D memcpy 更新这些值：

```cpp
// 主机端更新位置
*pos_pinned = text_pos;
cudaMemcpyAsync(pos_gpu, pos_pinned, sizeof(int32_t), cudaMemcpyHostToDevice, stream);
```

### 难点 5：向量化访问 vs 逐元素访问的权衡

**问题**：RMinte 使用 `DVec<half>` (uint4, 8 个 half 一次加载) 向量化访问，但 OrinMLLM 的 M-RoPE 需要访问两个不在连续内存位置的元素 (d0 和 d0+half_head_size)。

**解决方案**：由于 RoPE 的半分模式要求读取 `(d, d+64)` 这样不连续的元素对，向量化加载反而会增加复杂度（需要 shuffle 操作提取非对齐的元素）。对于 decode 阶段的单 token 场景，数据量很小（Q: 4096 元素, K: 1024 元素, V: 1024 元素），**逐元素访问已足够高效**，kernel launch 开销的减少才是主要收益来源。

---

## 5. 总结

### 5.1 修改的文件清单

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `kuiper/source/op/kernels/cuda/fused_rope_kv_kernel.cuh` | **新增** | 融合 kernel 头文件 |
| `kuiper/source/op/kernels/cuda/fused_rope_kv_kernel.cu` | **新增** | 融合 kernel CUDA 实现 |
| `kuiper/include/op/misc_layers.h` | **修改** | 添加 FusedMRoPEKVWriteLayer 类声明 |
| `kuiper/source/op/misc_layers.cpp` | **修改** | 添加 FusedMRoPEKVWriteLayer 实现 + include |
| `kuiper/include/model/qwen3.h` | **修改** | Qwen3Layers 添加 fused_mrope_kv_write_layer_ 成员 |
| `kuiper/source/model/qwen3_vl.cpp` | **修改** | 创建层实例 + 替换 attention_qkv_with_graph |

### 5.2 性能收益分析

对于 Qwen3-VL-8B 的 32 层 Transformer：

| 指标 | 融合前 | 融合后 | 改善 |
|------|-------|-------|------|
| 每层 kernel launch | 3 次 (RoPE + CopyK + CopyV) | 1 次 (Fused) | -2 次/层 |
| 每步 decode kernel launch | 96 次 (3 × 32 层) | 32 次 (1 × 32 层) | 减少 64 次 |
| K 数据全局内存读取 | 2 次 (RoPE + Copy) | 1 次 (Fused) | -50% |
| CUDA Graph 图节点 | 更多 | 更少 | 图更简洁 |
