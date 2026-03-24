# GQA + MRoPE + KV Cache 读写融合算子分析报告

## 目录
- [a. 融合原理及为什么能够融合](#a-融合原理及为什么能够融合)
- [b. 适配过程详解](#b-适配过程详解)
- [c. Grid/Block/Thread 层面详解](#c-gridblockthread-层面详解)
- [d. 适配过程中的困难点与解决方案](#d-适配过程中的困难点与解决方案)

---

## a. 融合原理及为什么能够融合

### 1. 融合前的 Decode 数据流

在 Qwen3-VL-8B 模型中，每个 Transformer 层的 decode 阶段（单 token 推理）的注意力计算包含以下步骤：

```
┌─────────────────────────────────────────────────────────────────────┐
│                     融合前：3 个独立 Kernel                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Kernel 1: Fused MRoPE + KV Write                                  │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │ Q [4096]     │───>│ Apply MRoPE  │───>│ Q' [4096] (写回全局)  │   │
│  │ (post-norm)  │    │              │    │                      │   │
│  ├─────────────┤    │              │    ├──────────────────────┤   │
│  │ K [1024]     │───>│ Apply MRoPE  │───>│ K' → Key Cache      │   │
│  │ (post-norm)  │    │              │    │       [写入位置 pos]   │   │
│  ├─────────────┤    │              │    ├──────────────────────┤   │
│  │ V [1024]     │───>│    直接拷贝   │───>│ V  → Val Cache      │   │
│  │ (post-norm)  │    │              │    │       [写入位置 pos]   │   │
│  └─────────────┘    └──────────────┘    └──────────────────────┘   │
│                                                                     │
│         ↓ Q' 写入全局内存后，被下一个 Kernel 读取                      │
│                                                                     │
│  Kernel 2: Flash Attention Decode                                   │
│  ┌──────────────────┐    ┌──────────────┐    ┌─────────────────┐   │
│  │ Q' [4096]        │    │ K Cache      │    │ V Cache         │   │
│  │ (从全局内存读回)   │───>│ [0..pos]     │───>│ [0..pos]        │   │
│  │                  │    │ (全局内存读取) │    │ (全局内存读取)    │   │
│  └──────────────────┘    └──────────────┘    └─────────────────┘   │
│           │                    │                     │              │
│           └────────┬───────────┘                     │              │
│                    ▼                                 │              │
│            Q'·K 评分 + Softmax                       │              │
│                    │                                 │              │
│                    └─────────┬───────────────────────┘              │
│                              ▼                                      │
│                     V 加权累加 → Attention Output [4096]             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**问题所在：**
- Q' 在 Kernel 1 中计算完 MRoPE 后**写入全局内存** (8 KB per layer)
- Q' 在 Kernel 2 中又**从全局内存读回** (8 KB per layer)
- 36 层 × 16 KB = **576 KB 冗余全局内存流量/token**
- 2 个 Kernel Launch = CUDA Graph 中额外的节点开销

### 2. 融合后的数据流

```
┌─────────────────────────────────────────────────────────────────────┐
│                     融合后：1 个统一 Kernel                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Kernel: Fused GQA + MRoPE + KV Cache Read/Write                   │
│                                                                     │
│  Phase 0: MRoPE (在 Shared Memory 中完成)                           │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │ Q [4096]     │───>│ Apply MRoPE  │───>│ Q' → Shared Memory  │   │
│  │ (全局读一次)  │    │              │    │    (永不写回全局!)     │   │
│  ├─────────────┤    │              │    ├──────────────────────┤   │
│  │ K [1024]     │───>│ Apply MRoPE  │───>│ K' → Shared Memory  │   │
│  │ (全局读一次)  │    │              │    │  + 写入 Key Cache    │   │
│  ├─────────────┤    │              │    ├──────────────────────┤   │
│  │ V [1024]     │───>│              │───>│ V  → 写入 Val Cache  │   │
│  │ (全局读一次)  │    │              │    │  + 保留用于累加       │   │
│  └─────────────┘    └──────────────┘    └──────────────────────┘   │
│                                                                     │
│  Phase 1-3: Flash Attention (直接消费 Shared Memory 中的 Q')        │
│  ┌──────────────────┐    ┌──────────────┐    ┌─────────────────┐   │
│  │ Q' (Shared Mem)  │    │ K Cache      │    │ V Cache         │   │
│  │ (零全局内存开销!) │───>│ [0..pos-1]   │───>│ [0..pos-1]      │   │
│  │                  │    │ (过去的token)  │    │ (过去的token)    │   │
│  └──────────────────┘    └──────────────┘    └─────────────────┘   │
│           │                    │                     │              │
│           └────────┬───────────┘                     │              │
│                    ▼                                 │              │
│            Q'·K 评分 + Online Softmax                │              │
│                    │                                 │              │
│                    └─────────┬───────────────────────┘              │
│                              ▼                                      │
│  Phase 4: 融入当前 Token      ┌──────────────────────────────┐      │
│                              │ K'(Shared Mem) 参与最后评分     │      │
│                              │ V(Input Tensor) 参与最后累加   │      │
│                              └──────────────────────────────┘      │
│                              ▼                                      │
│                     Normalize → Attention Output [4096]             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3. 为什么能够融合

**数据依赖分析：**

```
        Q GEMV ──> Q-norm ──> MRoPE(Q) ──> Q'·K scoring ──> Softmax ──> V acc ──> Output
        K GEMV ──> K-norm ──> MRoPE(K) ──> KV Cache Write ──┐
        V GEMV ──────────────────────────> KV Cache Write ──┤
                                                             │
                                 KV Cache ←──────────────────┘
                                    │
        Q'·K scoring <──────────────┘ (读取历史 K/V)
```

关键洞察：
1. **Q 的生命周期短**：Q 经过 MRoPE 后只被 Flash Attention 读取一次，之后不再使用。因此 MRoPE'd Q 无需写回全局内存，可以停留在 Shared Memory 中。
2. **当前 Token 的 K/V 不需要从 Cache 读回**：当前 token 的 MRoPE'd K 和 V 在 kernel 内部已经可用（Shared Memory / 寄存器），无需写入 Cache 后再读回。Cache 写入只为后续 decode 步骤服务。
3. **GQA 天然适合 blockIdx.y 并行**：GQA 有 32 个 Q heads 但只有 8 个 KV heads (kv_mul=4)。每个 block 处理一个 Q head，通过 `kv_head = q_head / kv_mul` 映射到对应 KV head。KV Cache 写入只需第一个负责该 KV head 的 block 执行即可。
4. **Shared Memory 足够**：MRoPE'd Q (256 bytes) + MRoPE'd K (256 bytes) + scores tile (2048 bytes) + reduction (32 bytes) = **2592 bytes**，远小于 Orin 的 48 KB Shared Memory。

### 4. 节省的资源量化

每个 token 每层节省：
- **全局内存带宽**：Q 写回 (8 KB) + Q 读回 (8 KB) = 16 KB
- **36 层合计**：36 × 16 KB = **576 KB/token**
- **Kernel Launch 减少**：从 2 个 kernel（MRoPE+KV write, FA decode）合并为 1 个
- **CUDA Graph 节点减少**：36 层 × 1 个节点 = 36 个节点

---

## b. 适配过程详解

### Step 1: 分析参考实现

分析了 RMinte-Orin-TensorRT-EDGE-LLM 工程中的 `applyRopeWriteKV.cu` 融合算子：
- 该算子融合 RoPE + KV Cache 写入（但**不含 Attention 计算**）
- Grid 设计：`(ceil(tokens/tokenPerCTA), numQHeads + numKVHeads)` — Q heads 和 KV heads 通过 `blockIdx.y` 分离
- 使用 `DVec<half>` (uint4) 实现 8 个 half 的向量化加载

### Step 2: 分析现有 OrinMLLM 实现

现有架构中 decode 阶段每层的 kernel 调用：

| 步骤 | Kernel | 输入 | 输出 |
|------|--------|------|------|
| 1 | RMSNorm | input | rmsnorm_output |
| 2 | Q/K/V GEMV (3个) | rmsnorm_output | query, temp_key, temp_value |
| 3 | Q-norm, K-norm (2个) | query, temp_key | query, temp_key (in-place) |
| 4 | Fused MRoPE + KV Write (1个) | query, temp_key, temp_value | query(in-place), KV cache |
| 5 | Flash Attention Decode (1个) | query, KV cache | mha_output |
| 6 | WO GEMV (1个) | mha_output | attn_output |
| 7 | Residual Add + FFN (多个) | input, attn_output, ... | output |

**融合目标**：将步骤 4 和 5 合并为单个 kernel：

| 步骤 | Kernel | 输入 | 输出 |
|------|--------|------|------|
| 4+5 | **Fused GQA + MRoPE + KV Decode** | query, temp_key, temp_value, KV cache, sin/cos cache | mha_output + KV cache updated |

### Step 3: 设计融合 Kernel

核心设计决策：

1. **Grid/Block**：复用 Flash Attention decode 的 `Grid(num_q_heads), Block(256)` 配置
2. **MRoPE 计算**：threads 0..63 (head_size/2) 处理 RoPE 元素对
3. **当前 Token 处理策略**：不从 Cache 读回当前 token 的 K/V，而是从 Shared Memory/寄存器直接使用
4. **KV Cache 写入**：只有 `head % kv_mul == 0` 的 block 执行写入
5. **在线 Softmax**：当前 token 作为最后一个"tile"融入在线 Softmax

### Step 4: 实现 CUDA Kernel

在 `fused_rope_kv_kernel.cu` 中添加 `fused_gqa_mrope_kv_decode_fp16_kernel`：
- Phase 0: 64 threads 计算 MRoPE(Q) → s_query, MRoPE(K) → s_k_current
- Phase 0b: 128 threads 写入 K/V 到 Cache（仅第一个 Q head per KV group）
- Phase 0c: Thread 0 计算 Q·K_current 评分
- Phase 1-3: 256 threads 执行 Tiled Online Softmax Flash Attention（过去 token）
- Phase 4: 融入当前 token 的评分和 V 累加
- Phase 5: 归一化并写出 Attention Output

### Step 5: 创建 Layer 封装

1. 在 `fused_rope_kv_kernel.cuh` 中添加 `fused_gqa_mrope_kv_decode_fp16()` 声明
2. 在 `misc_layers.h` 中创建 `FusedGQAMRoPEKVDecodeLayer` 类
3. 在 `misc_layers.cpp` 中实现 layer 的 `forward()` 方法

### Step 6: Model 集成

修改 `qwen3_vl.cpp`：

1. **Layer 创建** (`create_vl_nonparam_layers`)：创建 `fused_gqa_mrope_kv_decode_layer_`

2. **`attention_qkv_with_graph` 修改**：
   ```cpp
   if (use_fused_gqa_) {
     // 跳过 MRoPE 和 KV 写入（将在 attention_mha_with_graph 中完成）
   } else if (use_fused_rope_kv_) {
     fused_mrope_kv_write_layer_->forward(...);
   } else {
     mrope_gpu_pos_layer_->forward(...);
     copy_to_kv_cache_layer_->forward(...);  // K
     copy_to_kv_cache_layer_->forward(...);  // V
   }
   ```

3. **`attention_mha_with_graph` 修改**：
   - 签名扩展为接收 `rope_pos_gpu` 和 `kv_cache_pos_gpu` 两个位置张量
   - 添加 fused GQA 路径分支：
   ```cpp
   if (use_fused_gqa_ && fp16 mode) {
     fused_gqa_mrope_kv_decode_layer_->forward(
         rope_pos_gpu, kv_cache_pos_gpu,
         query, temp_key, temp_value,
         key_cache, val_cache, mha_output,
         sin_cache, cos_cache, ...config...);
   } else {
     flash_attention_decode_gpu_pos_layer_->forward(...);
   }
   ```

4. **调用点更新**：所有 `attention_mha_with_graph(layer_idx, kv_cache_pos_gpu)` 更新为 `attention_mha_with_graph(layer_idx, pos_tensor_gpu, kv_cache_pos_gpu)`

### Step 7: CLI 集成

`--fused-rope-kv` 标志现在同时启用 `use_fused_gqa_` 和 `use_fused_rope_kv_`。

---

## c. Grid/Block/Thread 层面详解

### 整体 Kernel 配置

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Kernel Launch 配置                                    │
│                                                                         │
│  Grid:  (num_q_heads,) = (32,)                                         │
│  Block: (FUSED_GQA_BLOCK_SIZE,) = (256,)                               │
│  Shared Memory: 2592 bytes                                              │
│                                                                         │
│  每个 Block 负责 1 个 Q Head 的完整注意力计算                             │
│  共 32 个 Block 并行执行（对应 32 个 Q Head）                             │
└─────────────────────────────────────────────────────────────────────────┘
```

### Grid 维度（blockIdx.x = Q Head 索引）

```
Grid (32 blocks)
┌────┬────┬────┬────┬────┬────┬────┬────┬─...─┬────┐
│B0  │B1  │B2  │B3  │B4  │B5  │B6  │B7  │     │B31 │
│Q0  │Q1  │Q2  │Q3  │Q4  │Q5  │Q6  │Q7  │     │Q31 │
│KV0 │KV0 │KV0 │KV0 │KV1 │KV1 │KV1 │KV1 │     │KV7 │
│    │    │    │    │    │    │    │    │     │    │
│写KV│    │    │    │写KV│    │    │    │     │    │
└────┴────┴────┴────┴────┴────┴────┴────┴─...─┴────┘
  ↑                   ↑
  kv_mul=4           kv_mul=4
  Q0-Q3 共享 KV0     Q4-Q7 共享 KV1
  只有 Q0 写 Cache    只有 Q4 写 Cache
```

- **32 个 CTA**，每个处理 1 个 Q head
- **GQA 映射**：`kv_head = q_head / kv_mul = q_head / 4`
- **KV Cache 写入**：仅 `q_head % kv_mul == 0` 的 block 执行（Block 0, 4, 8, ..., 28）
- **KV 数据访问**：所有 block 独立计算 MRoPE'd K（相同 KV group 内冗余但避免同步）

### Block 维度（256 Threads = 8 Warps）

```
Block (256 threads, 8 warps)
┌──────────────────────────────────────────────────────────────────┐
│ Warp 0  │ Warp 1  │ Warp 2  │ Warp 3  │ ... │ Warp 6  │ Warp 7 │
│ T0-T31  │ T32-T63 │ T64-T95 │ T96-T127│     │T192-T223│T224-T255│
└──────────────────────────────────────────────────────────────────┘
```

### Thread 角色映射（各 Phase）

```
Phase 0: MRoPE 计算（64 threads active）
┌────────────────────────────────────────────────────────────────┐
│ Thread 0-63:  每个 thread 处理 1 个 RoPE 对                    │
│   tid=0: (d0=0, d1=64)   → sin/cos lookup → Q'[0], Q'[64]    │
│   tid=1: (d0=1, d1=65)   → sin/cos lookup → Q'[1], Q'[65]    │
│   ...                                                          │
│   tid=63: (d0=63, d1=127) → sin/cos lookup → Q'[63], Q'[127]  │
│                                                                │
│   同时对 K 做相同的 MRoPE 计算 → s_k_current[]                  │
│                                                                │
│ Thread 64-255: 空闲（等待 __syncthreads）                       │
└────────────────────────────────────────────────────────────────┘

Phase 0b: KV Cache 写入（128 threads active）
┌────────────────────────────────────────────────────────────────┐
│ 仅当 head % kv_mul == 0 时执行：                                │
│ Thread 0-127: 写 K'[0..127] → Key Cache[kv_pos]               │
│ Thread 0-127: 写 V[0..127]  → Val Cache[kv_pos]               │
│                                                                │
│ Thread 128-255: 空闲                                           │
└────────────────────────────────────────────────────────────────┘

Phase 0c: Q·K_current 点积（1 thread active）
┌────────────────────────────────────────────────────────────────┐
│ Thread 0:                                                      │
│   dot = Σ s_query[d] × s_k_current[d], d ∈ [0, head_size)     │
│   score_current = dot × scale                                  │
│   → 写入 s_reduce[0] 供所有 thread 读取                        │
│                                                                │
│   使用 float4 向量化：16 次 float4 load (128 half / 8 per f4)   │
│   128 次 FMA 操作                                              │
│                                                                │
│ Thread 1-255: 空闲（等待 __syncthreads）                       │
└────────────────────────────────────────────────────────────────┘

Phase 1: Q·K 评分（256 threads fully active）
┌────────────────────────────────────────────────────────────────┐
│ 对 KV Cache 中的过去 token (0..kv_pos-1) 按 Tile 处理          │
│ TILE_K = 512                                                   │
│                                                                │
│ 每个 thread 处理 tile 中不同的 K 位置:                          │
│   Thread tid 处理 k_idx = tid, tid+256, tid+512, ...           │
│                                                                │
│ 每个 K 位置的 Q·K 点积 (head_size=128):                        │
│   16 次 float4 加载 (Q from smem, K from global via __ldg)      │
│   128 次 FMA → 1 个 score                                      │
│                                                                │
│ Warp-level max reduction:                                      │
│   __shfl_xor_sync 5 轮 → warp max                              │
│   s_reduce[warp_id] = warp_max                                 │
│   Thread 0 汇总 8 个 warp max → tile max                       │
└────────────────────────────────────────────────────────────────┘

Phase 2: Exp + Sum（256 threads）
┌────────────────────────────────────────────────────────────────┐
│ 每个 thread 对其负责的 K 位置:                                  │
│   exp_val = expf(score - m_new)                                │
│   累加 tile_sum                                                │
│                                                                │
│ Warp-level sum reduction → s_reduce → Thread 0 汇总            │
└────────────────────────────────────────────────────────────────┘

Phase 3: V 加权累加（128 threads active for output）
┌────────────────────────────────────────────────────────────────┐
│ my_dim = tid % head_size (128)                                 │
│                                                                │
│ Thread 0-127: 每个负责 output 的 1 个维度                       │
│   acc_o += Σ s_scores[k] × V_cache[k][my_dim]                 │
│                                                                │
│   4x unroll for ILP:                                           │
│   for k in range(0, tile_len, 4):                              │
│     acc_o = fmaf(s0, V[k][dim], acc_o)                         │
│     acc_o = fmaf(s1, V[k+1][dim], acc_o)                       │
│     acc_o = fmaf(s2, V[k+2][dim], acc_o)                       │
│     acc_o = fmaf(s3, V[k+3][dim], acc_o)                       │
│                                                                │
│ Thread 128-255: 冗余计算（不写 output）                         │
│ Rescale: acc_o *= exp(old_max - new_max)                       │
└────────────────────────────────────────────────────────────────┘

Phase 4: 当前 Token 融入（all threads）
┌────────────────────────────────────────────────────────────────┐
│ 所有 threads 统一更新 Online Softmax 状态:                      │
│   m_new = max(row_max, score_current)                          │
│   correction = exp(row_max - m_new)                            │
│   exp_current = exp(score_current - m_new)                     │
│   acc_o *= correction                                          │
│                                                                │
│ Thread 0-127: V 累加 (当前 token 的 V 从 input tensor 直接读) │
│   acc_o += exp_current × V_in[kv_head * head_size + my_dim]   │
│                                                                │
│ 更新: row_sum = correction * row_sum + exp_current             │
└────────────────────────────────────────────────────────────────┘

Phase 5: 输出（128 threads）
┌────────────────────────────────────────────────────────────────┐
│ Thread 0-127:                                                  │
│   O[head * head_size + my_dim] = half(acc_o / row_sum)         │
└────────────────────────────────────────────────────────────────┘
```

### Shared Memory 布局

```
Shared Memory (2592 bytes)
┌──────────────────────────────────────────────────────────────┐
│ Offset  │ Name         │ Size        │ Type    │ 用途         │
├─────────┼──────────────┼─────────────┼─────────┼──────────────┤
│ 0       │ s_query      │ 128 half    │ 256B    │ MRoPE'd Q    │
│ 256     │ s_k_current  │ 128 half    │ 256B    │ MRoPE'd K    │
│ 512     │ s_scores     │ 512 float   │ 2048B   │ tile 评分    │
│ 2560    │ s_reduce     │ 8 float     │ 32B     │ warp 归约    │
├─────────┼──────────────┼─────────────┼─────────┼──────────────┤
│ Total   │              │             │ 2592B   │              │
└─────────┴──────────────┴─────────────┴─────────┴──────────────┘
```

### 执行时间线

```
Timeline (per block, 1 SM)
Phase:       0     0b    0c  │  1───2───3  │  1───2───3  │  4    5
Action:    MRoPE KVWr QK.cur │  Tile 0     │  Tile 1     │  Cur  Out
Threads:   64    128   1     │  256        │  256        │  256  128
           ├──────────────────┤             │             │
           ~5 µs              │ ~5-50µs/tile│             │ ~1µs
                              │ (取决于序列长度)           │
```

---

## d. 适配过程中的困难点与解决方案

### 困难 1: 跨 Block 的 KV Cache 一致性

**问题描述：**

在 GQA 中，多个 Q heads 共享同一个 KV head（kv_mul=4）。如果 Block 0 写入 KV Cache 后、Block 1 需要从同一 Cache 位置读取，但 CUDA 不保证不同 block 之间的执行顺序，存在**读写竞争**的风险。

```
Block 0 (Q0, KV0):  Write K'[pos] to Cache → Read Cache[pos] for attention
Block 1 (Q1, KV0):  No write to Cache      → Read Cache[pos] for attention ???
                                              ↑ 可能读到旧数据！
```

**解决方案：本地数据策略**

当前 token 的 K 和 V **不从 Cache 读取**，而是从本地 Shared Memory 和 Input Tensor 直接访问：

```cpp
// Phase 0: MRoPE'd K 存在 s_k_current (shared memory)
// Phase 0c: Q·K_current 使用 s_query 和 s_k_current 计算
// Phase 1-3: 只读 Cache[0..kv_pos-1] (过去 token，已确认写入)
// Phase 4: 当前 token V 从 V_in (input tensor) 读取
```

Attention 只遍历过去的 token（`kv_past_len = kv_pos`），当前 token 作为独立的最后一步融入 Online Softmax。这完全避免了跨 block 的 Cache 一致性问题。

### 困难 2: 当前 Token 融入 Online Softmax

**问题描述：**

Online Softmax 的 tile-based 算法维护运行状态 `(row_max, row_sum, acc_o)`。在 tile 循环结束后，需要将当前 token 的评分融入这些状态，而当前 token 只有 1 个 position，不值得单独开一个 tile。

**解决方案：后处理步骤**

在 tile 循环结束后，添加 Phase 4 来处理当前 token：

```cpp
{
    float m_new = fmaxf(row_max, score_current);
    float correction = expf(row_max - m_new);
    float exp_current = expf(score_current - m_new);

    acc_o *= correction;  // 重新缩放过去的累加

    // 从 input tensor 直接读 V_current (不读 cache)
    if (my_dim < head_size) {
        float v_current = __half2float(V_in[kv_head * head_size + my_dim]);
        acc_o = fmaf(exp_current, v_current, acc_o);
    }

    row_max = m_new;
    row_sum = fmaf(correction, row_sum, exp_current);
}
```

数学保证：这与将当前 token 包含在最后一个 tile 中计算结果完全一致，因为 Online Softmax 的 rescale 操作是可结合的。

### 困难 3: Q·K_current 的高效计算

**问题描述：**

在 Flash Attention 的评分阶段，每个 thread 独立计算一个 K 位置的完整 Q·K 点积（128 FMA）。但对于当前 token 的 K，只有一个位置需要计算，分配 256 个 thread 浪费算力，而用 cooperative reduction 又引入同步开销。

**解决方案：单线程向量化点积**

只用 Thread 0 计算 Q·K_current，使用 float4 向量化加载：

```cpp
if (tid == 0) {
    const float4* q_f4 = reinterpret_cast<const float4*>(s_query);
    const float4* k_f4 = reinterpret_cast<const float4*>(s_k_current);
    float2 dot = make_float2(0.0f, 0.0f);
    for (int d = 0; d < head_size / 8; d++) {  // 16 iterations
        // 8 half per float4, 128 total → 16 float4 loads
        // 128 FMA operations
    }
    s_reduce[0] = (dot.x + dot.y) * scale;
}
__syncthreads();
score_current = s_reduce[0];  // 所有 thread 广播读取
```

单 thread 完成 128 次 FMA + 32 次 load 只需 ~200 cycles（~0.2 µs @ 1 GHz）。与 cooperative dot product 的额外同步开销（__syncthreads × 2）相比，单线程方案更高效。

### 困难 4: CUDA Graph 兼容性

**问题描述：**

CUDA Graph 要求 kernel 参数在 capture 时固定。位置信息（rope_pos, kv_cache_pos）每步变化，不能硬编码为 kernel 参数。

**解决方案：GPU-Resident Position Pointers**

使用 volatile GPU 内存读取（与现有 FA decode kernel 一致）：

```cpp
const int rope_pos = *reinterpret_cast<const volatile int32_t*>(pos_gpu);
const int kv_pos = *reinterpret_cast<const volatile int32_t*>(kv_pos_gpu);
```

- 固定 GPU 分配地址 (`kInputPosGPU`, `kKVCachePosGPU`)
- 每步通过 pinned memory H2D copy 更新值
- Kernel 用 `volatile` 防止编译器缓存优化
- 固定 Shared Memory 大小（2592 bytes），不依赖运行时变量

### 困难 5: `attention_mha_with_graph` 函数签名变更

**问题描述：**

原 `attention_mha_with_graph` 只接收 `kv_cache_pos_gpu`（用于 Flash Attention 查询长度），但融合 kernel 还需要 `rope_pos_gpu`（用于 MRoPE 计算）。

**解决方案：扩展函数签名**

```cpp
// 原签名
void attention_mha_with_graph(int32_t layer_idx, 
                              const tensor::Tensor& kv_cache_pos_gpu) const;

// 新签名
void attention_mha_with_graph(int32_t layer_idx,
                              const tensor::Tensor& rope_pos_gpu,
                              const tensor::Tensor& kv_cache_pos_gpu) const;
```

在非融合路径中，`rope_pos_gpu` 被忽略（FA decode 只需 `kv_cache_pos_gpu`）。所有调用点统一更新。

---

## 性能测试结果

### 测试环境
- 硬件：NVIDIA Jetson Orin (LPDDR5 ~170 GB/s)
- 模型：Qwen3-VL-8B-fp16
- 配置：CUDA Graph + Flash Attention v1 + max_pixel=500000

### 对比数据

| 配置 | 吞吐量 (tok/s) | 延迟 (ms/tok) | 相比 Baseline |
|------|----------------|---------------|---------------|
| Baseline (无融合) | 9.87 | 101.35 | — |
| Fused GQA+MRoPE+KV | 9.89 | 101.16 | -0.19 ms/tok |

### 分析

1. **提速幅度有限（~0.2%）的原因**：Qwen3-VL-8B-fp16 的 decode 阶段是**内存带宽受限**的，14.44 GB FP16 权重/token 占据了 ~84% 的理论带宽（170 GB/s LPDDR5）。注意力 kernel 本身只占每层计算时间的很小比例。

2. **实际节省的量化**：
   - Q 全局内存带宽节省：576 KB/token
   - CUDA Graph 节点减少：36 个/token
   - 预计时间节省：~0.3 ms/token（实测 ~0.19 ms/token）

3. **在以下场景中收益更显著**：
   - 量化模型（AWQ INT4/INT8）：权重带宽压力降低，注意力占比提升
   - 长序列：KV Cache 读取量增加，注意力计算占比提升
   - 较小模型：权重更少，注意力瓶颈相对突出

### 正确性验证

融合 kernel 的输出与非融合路径完全一致——模型对同一张图片生成了相同的、语义连贯的描述文本，确认了 MRoPE 计算、KV Cache 管理和 GQA 注意力计算的正确性。
