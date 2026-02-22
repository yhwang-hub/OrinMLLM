# FlashAttention & FlashAttention2 深度分析报告

> **工程**: OrinMLLM — 基于 NVIDIA Orin (SM87) 的大模型推理引擎  
> **日期**: 2026-02-22  
> **涉及源码**:
> - `kuiper/source/op/kernels/cuda/flash_attention_kernel.cu` (FlashAttention v1)
> - `kuiper/source/op/kernels/cuda/flash_attention2_kernel.cu` (FlashAttention v2)
> - `kuiper/source/op/flash_attention.cpp` (层级分发逻辑)
> - `kuiper/source/model/qwen2.cpp`, `qwen3.cpp`, `qwen3_vl.cpp` (模型集成)

---

## 一、FlashAttention 与 FlashAttention2 的数学原理及伪代码

### 1.1 标准 Attention 回顾

标准自注意力机制的数学定义为：

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

其中 $Q \in \mathbb{R}^{N \times d}$，$K \in \mathbb{R}^{M \times d}$，$V \in \mathbb{R}^{M \times d}$。

传统实现的步骤为：

1. 计算注意力分数矩阵 $S = QK^T / \sqrt{d_k}$，大小为 $N \times M$
2. 对 $S$ 的每一行做 softmax：$P = \text{softmax}(S)$
3. 计算输出 $O = PV$

**核心问题**：$S$ 和 $P$ 均为 $N \times M$ 的矩阵，当序列长度 $M$ 很长时，需要 $O(NM)$ 的显存来存储，而且需要多次遍历 HBM（高带宽显存），产生大量 I/O 开销。

### 1.2 FlashAttention v1 的数学原理

FlashAttention 的核心思想是 **Online Softmax + 分块 (Tiling)** —— 将 KV 序列分成若干 tile，逐 tile 计算注意力，始终维护在线统计量而无需存储完整的 $N \times M$ 注意力矩阵。

#### 1.2.1 Online Softmax 理论基础

标准 softmax 对向量 $x = [x_1, x_2, \ldots, x_n]$ 的定义为：

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
$$

为了数值稳定性，使用 Safe Softmax：

$$
\text{softmax}(x_i) = \frac{e^{x_i - m}}{\sum_{j=1}^{n} e^{x_j - m}}, \quad m = \max_j x_j
$$

**Online Softmax** 的关键洞察是：我们可以**增量地**维护 $m$（全局最大值）和 $l$（归一化因子），逐块处理数据，每处理一个新块时，通过修正因子来更新之前的累加结果。

设已经处理了前 $t-1$ 个 tile，维护了：
- $m^{(t-1)}$：前 $t-1$ 个 tile 的最大分数
- $l^{(t-1)}$：前 $t-1$ 个 tile 的归一化因子（指数和）
- $O^{(t-1)}$：前 $t-1$ 个 tile 的**未归一化**输出累积

处理第 $t$ 个 tile（包含 K/V 的位置 $[b_t, e_t)$）时：

**Step 1**：计算当前 tile 的注意力分数：

$$
S_j^{(t)} = \frac{q \cdot k_j}{\sqrt{d_k}}, \quad j \in [b_t, e_t)
$$

**Step 2**：计算当前 tile 的局部最大值：

$$
m_j^{(t)} = \max_{j \in [b_t, e_t)} S_j^{(t)}
$$

**Step 3**：更新全局最大值：

$$
m^{(t)} = \max(m^{(t-1)}, m_j^{(t)})
$$

**Step 4**：计算修正因子（rescaling correction）：

$$
\alpha = e^{m^{(t-1)} - m^{(t)}}
$$

这是因为之前的指数值 $e^{S_j - m^{(t-1)}}$ 需要调整为 $e^{S_j - m^{(t)}}$：

$$
e^{S_j - m^{(t)}} = e^{S_j - m^{(t-1)}} \cdot e^{m^{(t-1)} - m^{(t)}} = e^{S_j - m^{(t-1)}} \cdot \alpha
$$

**Step 5**：计算当前 tile 的 softmax 权重和局部归一化因子：

$$
p_j^{(t)} = e^{S_j^{(t)} - m^{(t)}}, \quad l_j^{(t)} = \sum_{j \in [b_t, e_t)} p_j^{(t)}
$$

**Step 6**：更新全局归一化因子：

$$
l^{(t)} = \alpha \cdot l^{(t-1)} + l_j^{(t)}
$$

**Step 7**：更新输出累积（先修正旧值，再加上新贡献）：

$$
O^{(t)} = \alpha \cdot O^{(t-1)} + \sum_{j \in [b_t, e_t)} p_j^{(t)} \cdot v_j
$$

**Step 8**：处理完所有 tile 后，最终归一化：

$$
O_{\text{final}} = \frac{O^{(T)}}{l^{(T)}}
$$

#### 1.2.2 FlashAttention v1 伪代码

```
Algorithm: FlashAttention v1 Forward Pass
Input: Q[N, d], K[M, d], V[M, d], tile_size B_c
Output: O[N, d]

for each query position i in [0, N):
    # Initialize online softmax state
    m ← -∞          # running max
    l ← 0            # running sum of exp
    O[i] ← 0         # output accumulator (d-dimensional)
    
    # Process K/V in tiles
    for tile_start = 0 to M step B_c:
        tile_end ← min(tile_start + B_c, M)
        
        # Step 1: Compute attention scores for this tile
        for j in [tile_start, tile_end):
            S[j] ← Q[i] · K[j]^T / √d_k
        
        # Step 2: Find tile maximum
        m_j ← max(S[tile_start:tile_end])
        
        # Step 3: Update global maximum
        m_new ← max(m, m_j)
        
        # Step 4: Correction factor for previous accumulators
        α ← exp(m - m_new)                         ← KEY: rescale old values
        
        # Step 5: Compute softmax weights
        for j in [tile_start, tile_end):
            P[j] ← exp(S[j] - m_new)
        l_j ← sum(P[tile_start:tile_end])
        
        # Step 6: Update running sum
        l_new ← α * l + l_j
        
        # Step 7: Rescale previous output + accumulate new contribution
        O[i] ← α * O[i] + Σ_j P[j] * V[j]        ← KEY: rescale O every tile
        
        # Step 8: Update state
        m ← m_new
        l ← l_new
    
    # Step 9: Final normalization
    O[i] ← O[i] / l
```

#### 1.2.3 正确性证明

FlashAttention 的正确性基于以下等价变换。设完整的 softmax 权重为：

$$
p_j = \frac{e^{S_j - m_{\text{global}}}}{\sum_k e^{S_k - m_{\text{global}}}}
$$

FlashAttention 的输出为：

$$
O = \frac{\sum_j e^{S_j - m^{(T)}} v_j}{l^{(T)}}
$$

由于 $m^{(T)} = m_{\text{global}}$（处理完所有 tile 后的最大值就是全局最大值），且 $l^{(T)} = \sum_j e^{S_j - m_{\text{global}}}$，因此：

$$
O = \frac{\sum_j e^{S_j - m_{\text{global}}} v_j}{\sum_k e^{S_k - m_{\text{global}}}} = \sum_j p_j v_j
$$

这与标准 attention 完全等价，FlashAttention 是 **精确** 的，不是近似。

### 1.3 FlashAttention v2 的数学原理

FlashAttention v2 (Tri Dao, 2023) 在 v1 的数学基础上做了以下**算法层面**的优化：

#### 1.3.1 延迟归一化 (Delayed Rescaling)

FA1 在每个 tile 处理完后都会对输出累积器 $O$ 做一次 rescaling：

$$
O^{(t)} = e^{m^{(t-1)} - m^{(t)}} \cdot O^{(t-1)} + \sum_j p_j^{(t)} v_j
$$

这意味着每个 tile 都需要对 $O$ 的每个维度（共 $d$ 个）做一次乘法修正，产生 $O(\text{num\_tiles} \times d)$ 次非 GEMM 浮点运算。

FA2 的核心改进是：**仍然维护 $m$ 和 $l$**，但将修正因子的应用方式进行了重组。具体而言：

- 修正因子 $\alpha = e^{m^{(t-1)} - m^{(t)}}$ 仍然在每个 tile 计算
- 但 FA2 通过更小的 tile 和更好的 warp 并行来均摊修正开销
- 不同的 warp 独立处理不同的 K/V tile 子集，最终合并结果

数学上，FA2 的最终结果仍然是：

$$
O = \frac{\sum_j e^{S_j - m_{\text{global}}} v_j}{\sum_k e^{S_k - m_{\text{global}}}}
$$

与 FA1 完全相同，正确性完全等价。

#### 1.3.2 更好的工作分配

FA1 中一个 thread block 内的所有 warp 处理**相同的** K tile 的不同输出维度。FA2 将不同 warp 分配到**不同的** K tile 上，减少了 shared memory 读取和同步次数。

具体改进：
- **FA1 (warp 间冗余)**：所有 warp 读取同一个 K tile → 更多 shared memory 带宽争用
- **FA2 (warp 间分工)**：不同 warp 处理不同 K tile 范围 → 减少共享内存读取次数

#### 1.3.3 减少同步

FA2 通过 warp-level 的独立 online softmax 减少了 `__syncthreads()` 的数量：
- FA1 在 tile max 求解、softmax 计算、V 累加之间都需要 block 级同步
- FA2 允许 warp 在各自的 tile 范围内独立计算，仅在最后合并时需要同步

#### 1.3.4 Flush-to-Zero (FTZ) 策略

FA2 引入了一个数值优化：当 $S_j - m$ 低于某个阈值（如 $-20.0$）时，直接将 $e^{S_j - m}$ 设为 0，避免计算过小的指数值：

$$
p_j = \begin{cases} e^{S_j - m} & \text{if } S_j - m > \text{FTZ\_THRESHOLD} \\ 0 & \text{otherwise} \end{cases}
$$

由于 $e^{-20} \approx 2 \times 10^{-9}$，这些值对最终结果的贡献微乎其微。

#### 1.3.5 FlashAttention v2 伪代码

```
Algorithm: FlashAttention v2 Forward Pass
Input: Q[N, d], K[M, d], V[M, d], tile_size B_c (smaller than FA1)
Output: O[N, d]

FTZ_THRESHOLD ← -20.0

for each query position i in [0, N):
    m ← -∞
    l ← 0
    O[i] ← 0
    
    # FA2: Use smaller tiles for better warp-level parallelism
    for tile_start = 0 to M step B_c:          # B_c = 64 (vs FA1's 1024)
        tile_end ← min(tile_start + B_c, M)
        
        # Step 1: Each warp independently computes Q·K for subset of tile
        # (rather than all warps computing the full tile redundantly)
        for j in [tile_start, tile_end) (warp-parallel):
            S[j] ← Q[i] · K[j]^T / √d_k
        
        # Step 2: Warp-level max → block-level max (fewer __syncthreads)
        m_j ← warp_reduce_max(block_reduce(S[tile_start:tile_end]))
        
        # Step 3: Update global max
        m_new ← max(m, m_j)
        
        # Step 4: Correction + softmax with FTZ
        α ← exp(m - m_new)
        O[i] ← α * O[i]                       ← FA2: same rescaling, but fewer dims per tile
        
        for j in [tile_start, tile_end):
            val ← S[j] - m_new
            P[j] ← val > FTZ_THRESHOLD ? exp(val) : 0    ← FA2: flush-to-zero
        
        l_j ← sum(P[tile_start:tile_end])
        
        # Step 5: Accumulate V (better ILP with fmaf unrolling)
        O[i] += Σ_j P[j] * V[j]               ← FA2: fmaf for better pipelining
        
        # Step 6: Update state  
        m ← m_new
        l ← α * l + l_j
    
    # Final normalization
    O[i] ← O[i] / l
```

### 1.4 Prefill vs Decode 场景的差异

在 LLM 推理中，attention 有两种使用模式：

| 特征 | Prefill（预填充） | Decode（解码） |
|------|-------------------|----------------|
| Query 数量 | 多个 token 的 query | 单个 token 的 query |
| Grid 维度 | `[head_num, seq_len]` | `[head_num]` |
| 计算特征 | Compute-bound | Memory-bound |
| Online Softmax | 必须（因为 KV 可能很长） | 对短序列可用全量 softmax，长序列用 online |
| 并行度 | 高（多个 query 并行） | 低（只有 head_num 个 block） |

---

## 二、适配过程中的关键点与困难点

### 2.1 关键点一：FP16/FP32 数据类型的双路径支持

**问题描述**：工程中的模型有 FP16（Qwen3-8B-fp16、Qwen2.5-7B-fp16、Qwen3-VL-8B）和 FP32（Qwen2.5-7B）两种精度。标准 MHA 仅支持 FP32 数据，当用户对 FP16 模型指定 `--attention mha` 时，会因为数据类型不匹配而崩溃（"The tensor has a wrong data type"）。

**解决方案**：在每个模型的 `attention_mha()`、`attention_mha_with_graph()`、`batched_attention_mha()` 函数中实现**三路分发逻辑**：

```cpp
// 1. FP16 数据 → 始终使用 Flash Attention（FA1 或 FA2，由 attention_type_ 决定）
if (query.data_type() == base::DataType::kDataTypeFp16 &&
    key_cache.data_type() == base::DataType::kDataTypeFp16) {
    // 调用 flash_attention_decode_layer_->forward()
}
// 2. FP32 数据 + 用户选择 MHA → 使用标准 MHA
else if (attention_type_ == base::AttentionType::kAttentionMHA) {
    // 调用 mha_layer_->forward()
}
// 3. FP32 数据 + 用户选择 FA1/FA2 → 使用 Flash Attention FP32 路径
else {
    // 调用 flash_attention_decode_layer_->forward()  (FP32 路径)
}
```

### 2.2 关键点二：CUDA Graph 兼容性

**问题描述**：CUDA Graph 要求 kernel launch 时 shared memory 大小**固定不变**——不能依赖于运行时变量（如当前 `pos` 位置）。原始的非 Online Softmax decode kernel 需要 `pos+1` 大小的 score buffer，这会随每个 decode step 变化，导致无法进行 CUDA Graph capture。

**解决方案**：使用 **Online Softmax Tiling** 来实现 CUDA Graph 兼容：

- 将 KV 序列分成固定大小的 tile（256 个位置为一组）
- Shared memory 仅分配固定的 tile 大小（而非整个 KV 长度）
- Position 通过 GPU 内存指针传递（`pos_ptr`），kernel 通过 `volatile` 读取

```cpp
// 固定 shared memory 大小，不依赖 pos
const int smem_size = head_size * sizeof(half) + 
                      TILE_K * sizeof(float) +           // 固定 tile 大小
                      2 * N_WARPS * sizeof(float);       // max/sum reduction buffer

// Position 存储在 GPU 显存中
const int pos = *reinterpret_cast<const volatile int32_t*>(pos_ptr);
```

### 2.3 关键点三：FP32 路径没有 GPU-pos Flash Attention Kernel

**问题描述**：FP16 模型有一对完整的 kernel：
- `flash_attention_decode_kernel_fp16_online_softmax`（FA1 GPU-pos）
- `flash_attention2_decode_kernel_fp16_gpu_pos`（FA2 GPU-pos）

但 FP32 路径的 flash attention decode kernel 没有对应的 GPU-pos 变体。当 FP32 模型（如 Qwen2.5-7B.bin）选择 `--attention flash1/flash2` 时，在 CUDA Graph 的 `attention_mha_with_graph` 路径中会尝试调用不存在的 FP32 GPU-pos flash attention kernel，导致崩溃。

**解决方案**：对于 FP32 模型，在 CUDA Graph 路径 (`attention_mha_with_graph`) 中，FA1/FA2 也回退到标准的 `mha_gpu_pos_layer_`（它本身就是 CUDA Graph 兼容的）：

```cpp
// qwen2.cpp / qwen3.cpp 中的 attention_mha_with_graph:
} else {
    // FP32 + FA1/FA2: no GPU-pos flash attention kernel for FP32,
    // fall back to MHA with GPU pos
    tensor::Tensor score_storage = get_buffer(ModelBufferType::kScoreStorage);
    STATUS_CHECK(qwen_layers_->mha_gpu_pos_layer_->forward(...));
}
```

### 2.4 关键点四：Warp 间数据广播 Bug

**问题描述**：FP32 decode kernel 中的 global max/sum reduction 存在严重 bug。代码使用 `__shfl_sync(0xffffffff, global_max, 0)` 试图将结果广播到所有线程，但 `__shfl_sync` 仅在**同一个 warp 内**有效。256 线程的 block 有 8 个 warp，因此只有 warp 0 的线程能读到正确的 `global_max`，其他 warp 的线程使用了**未初始化的值**。

这导致后续 softmax 计算使用不一致的 max 值，产生非法内存访问。

**解决方案**：通过 shared memory 广播结果到所有线程：

```cuda
// 错误写法（仅 warp 0 内广播）：
global_max = __shfl_sync(0xffffffff, global_max, 0);  // ← BUG

// 正确写法（通过 shared memory 广播到所有 warp）：
if (tid == 0) {
    s_max[0] = global_max;  // 写入 shared memory
}
__syncthreads();
global_max = s_max[0];      // 所有线程从 shared memory 读取
```

### 2.5 关键点五：FlashAttentionDecodeLayer 的 FP32 分发缺陷

**问题描述**：`flash_attention.cpp` 中 `FlashAttentionDecodeLayer::forward()` 的 FP32 路径只有两种分发：FA2 和"其他"。"其他"路径错误地调用了标准 MHA kernel：

```cpp
// 原始错误代码
} else {
    kernel::get_mha_kernel(device_type_)(...);  // ← FA1 也走到这里
}
```

当 `attention_type_ == kAttentionFlash1` 时，应该调用 `flash_attention_decode_cu` 而不是 MHA kernel。

**解决方案**：改为三路分发：

```cpp
if (attention_type_ == kAttentionFlash2) {
    kernel::flash_attention2_decode_cu(...);     // FA2
} else if (attention_type_ == kAttentionFlash1) {
    kernel::flash_attention_decode_cu(...);      // FA1
} else {
    kernel::get_mha_kernel(device_type_)(...);   // MHA fallback
}
```

### 2.6 关键点六：运行时 CLI 参数到模型层的传递链

**问题描述**：用户通过 `--attention mha|flash1|flash2` 指定的注意力类型需要从 CLI 穿透到最底层的 CUDA kernel。传递链路为：

```
CLI (inference_common.h) → Model::set_attention_type() → 
  各模型(qwen2/qwen3/qwen3_vl)::set_attention_type() →
    flash_attention_decode_layer_->set_attention_type(type)
    flash_attention_prefill_layer_->set_attention_type(type)
    flash_attention_decode_gpu_pos_layer_->set_attention_type(type) (仅VL)
```

需要确保每个层级都正确转发，且 `AttentionType` 枚举定义一致。

### 2.7 关键点七：缺失的 `#include` 导致编译失败

**问题描述**：FA2 kernel 使用了 CUB 库的 `cub::BlockReduce` 模板类，但文件头部没有 `#include <cub/cub.cuh>`，导致编译错误。

**解决方案**：在 `flash_attention2_kernel.cu` 头部添加 `#include <cub/cub.cuh>`。

### 2.8 关键点八：Batched MHA 路径的 const 正确性

**问题描述**：`batched_attention_mha()` 方法中 `batched_mha_layer_->forward()` 接受 `tensor::Tensor&` 非 const 引用，但传入的 `mha_out` 是 `const tensor::Tensor`，需要 `const_cast` 转换。

---

## 三、CUDA 核函数源码逐步解读

### 3.1 FlashAttention v1 — FP16 Prefill Kernel

**文件**: `flash_attention_kernel.cu` → `flash_attention_prefill_kernel_fp16`

```
Grid: [head_num, seq_len]    // 每个 block 处理一个 (head, query_position) 对
Block: 128 threads            // BLOCK_SIZE=128，恰好等于 head_size=128
```

#### 第一步：Query 加载到 Shared Memory

```cuda
const half* q_ptr = Q + seq_idx * dim + head * head_size;
for (int d = tid; d < head_size; d += BLOCK_SIZE) {
    s_query[d] = q_ptr[d];
}
__syncthreads();
```

**对应原理**：从 HBM 加载当前 query 向量 $q_i$ 到 shared memory，后续复用（避免反复访问 HBM）。由于 `BLOCK_SIZE = head_size = 128`，每个线程恰好加载一个元素。

#### 第二步：初始化 Online Softmax 状态

```cuda
float acc_o = 0.0f;       // 输出累积器（每线程负责一个维度）
float row_max = -FLT_MAX;  // m^(0) = -∞
float row_sum = 0.0f;      // l^(0) = 0
```

**对应原理**：初始化 $m \leftarrow -\infty$，$l \leftarrow 0$，$O \leftarrow 0$。

#### 第三步：分 Tile 循环处理 KV

```cuda
for (int tile_start = 0; tile_start < kv_len; tile_start += TILE_K) {
    // TILE_K = 1024，对应 B_c
```

**对应原理**：外层循环，将 KV 序列分成大小为 $B_c = 1024$ 的 tile。

#### 第四步（Tile 内）：计算 Q·K^T 分数

```cuda
for (int k_idx = tid; k_idx < tile_len; k_idx += BLOCK_SIZE) {
    const float4* k_ptr_f4 = reinterpret_cast<const float4*>(K_cache + kv_pos * kv_dim + head_offset);
    const float4* q_ptr_f4 = reinterpret_cast<const float4*>(s_query);
    float2 acc = make_float2(0.0f, 0.0f);
    for (int d = 0; d < head_size / 8; d++) {
        float4 q_packed = q_ptr_f4[d];
        float4 k_packed = __ldg(k_ptr_f4 + d);
        const half2* q_h2 = reinterpret_cast<const half2*>(&q_packed);
        const half2* k_h2 = reinterpret_cast<const half2*>(&k_packed);
        for (int i = 0; i < 4; i++) {
            float2 q_f = __half22float2(q_h2[i]);
            float2 k_f = __half22float2(k_h2[i]);
            acc.x = fmaf(q_f.x, k_f.x, acc.x);
            acc.y = fmaf(q_f.y, k_f.y, acc.y);
        }
    }
    float score = (acc.x + acc.y) * scale;   // scale = 1/√d_k
    s_scores[k_idx] = score;
    tile_max_local = fmaxf(tile_max_local, score);
}
```

**对应原理**：计算 $S_j = q \cdot k_j / \sqrt{d_k}$。

**优化细节**：
- 使用 `float4` (128-bit) 一次加载 4 个 `half` 对（共 8 个半精度值），减少 4 倍内存事务
- 将 `float4` 重解释为 4 个 `half2`，利用 `__half22float2` 转换后用 `fmaf` 融合乘加
- `__ldg()` 使用只读缓存（texture cache）加速全局内存读取
- 累加器使用 `float2`（两路并行累加），最终合并 `acc.x + acc.y`

#### 第五步（Tile 内）：Warp 级 Max Reduction

```cuda
// Warp reduce max
for (int offset = 16; offset > 0; offset >>= 1) {
    tile_max_local = fmaxf(tile_max_local, __shfl_xor_sync(0xffffffff, tile_max_local, offset));
}
// Store warp max to shared memory
if (lane_id == 0) s_warp_max[warp_id] = tile_max_local;
__syncthreads();
// Thread 0 reduces across warps
if (tid == 0) {
    m_j = fmaxf(fmaxf(s_warp_max[0], s_warp_max[1]), fmaxf(s_warp_max[2], s_warp_max[3]));
    s_warp_max[0] = m_j;
}
__syncthreads();
m_j = s_warp_max[0];
```

**对应原理**：计算 $m_j^{(t)} = \max_{j \in \text{tile}} S_j$。

**优化细节**：
- 第一层：warp 内使用 `__shfl_xor_sync` 做 butterfly reduction（5 次 shuffle 把 32 个值规约为 1 个）
- 第二层：4 个 warp 的结果写入 shared memory，thread 0 做最终规约（4 次比较）
- 通过 shared memory 广播结果给所有线程

#### 第六步（Tile 内）：更新全局最大值 + 修正

```cuda
float m_new = fmaxf(row_max, m_j);
```

**对应原理**：$m^{(t)} = \max(m^{(t-1)}, m_j^{(t)})$。

#### 第七步（Tile 内）：计算 Softmax 权重 + Tile Sum

```cuda
float tile_sum_local = 0.0f;
for (int k_idx = tid; k_idx < tile_len; k_idx += BLOCK_SIZE) {
    float val = s_scores[k_idx] - m_new;
    float exp_score = (val > SOFTMAX_FTZ) ? expf(val) : 0.0f;
    s_scores[k_idx] = exp_score;
    tile_sum_local += exp_score;
}
```

**对应原理**：$p_j = e^{S_j - m^{(t)}}$，同时累加 $l_j^{(t)} = \sum p_j$。

**注意**：这里使用了 FTZ 阈值（$-20.0$），当 $S_j - m$ 过小时跳过 `expf` 计算。

#### 第八步（Tile 内）：Tile Sum 的 Block Reduction

```cuda
// 类似 max reduction 的两层 warp→block 规约
```

**对应原理**：将 128 个线程的局部 sum 规约为全局 tile sum $l_j$。

#### 第九步（Tile 内）：修正旧累积 + 累加 V

```cuda
float correction = expf(row_max - m_new);
acc_o *= correction;

if (tid < head_size) {
    const half* v_ptr = v_thread_base + tile_start * kv_dim;
    for (int k = 0; k < tile_len; k++) {
        acc_o = fmaf(s_scores[k], __half2float(__ldg(v_ptr)), acc_o);
        v_ptr += kv_dim;
    }
}
```

**对应原理**：
$$
O^{(t)} = e^{m^{(t-1)} - m^{(t)}} \cdot O^{(t-1)} + \sum_j p_j^{(t)} \cdot v_j
$$

**核心**：`acc_o *= correction` 就是将之前的累积值乘以修正因子 $\alpha$，然后累加新 tile 的贡献。

**优化细节**：
- 每个线程负责一个输出维度（因为 `BLOCK_SIZE = head_size = 128`）
- 使用 `fmaf(s, v, acc)` 融合乘加指令
- 使用 8 路展开（`k += 8`）提高指令级并行度（ILP）

#### 第十步（Tile 内）：更新状态

```cuda
row_max = m_new;
row_sum = correction * row_sum + l_j;
```

**对应原理**：$m \leftarrow m^{(t)}$，$l \leftarrow \alpha \cdot l^{(t-1)} + l_j$。

#### 第十一步：最终归一化输出

```cuda
if (tid < head_size) {
    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    half* o_ptr = O + seq_idx * dim + head * head_size;
    o_ptr[tid] = __float2half(acc_o * inv_sum);
}
```

**对应原理**：$O_{\text{final}} = O^{(T)} / l^{(T)}$。将 float 结果转回 half 并写入输出。

---

### 3.2 FlashAttention v1 — FP16 Decode Kernel

**文件**: `flash_attention_kernel.cu` → `flash_attention_decode_kernel_fp16_optimized`

```
Grid: [head_num]       // 每个 block 处理一个 head
Block: 256 threads     // DECODE_BLOCK_SIZE=256, 8 warps
```

Decode 阶段只有单个 query token，所以去掉了 tile 循环（对于短序列可以一次性计算所有 KV），采用**全量 softmax** 方案：

#### Phase 1：计算所有 Q·K 分数

```cuda
for (int k = tid; k < kv_len; k += DECODE_BLOCK_SIZE) {
    // float4 vectorized Q·K with half2
    // ... (same vectorized dot product as prefill)
    s_scores[k] = score;
    local_max = fmaxf(local_max, score);
}
```

**与 Prefill 区别**：不分 tile，一次性计算所有 $S_j$。256 线程协作处理 kv_len 个位置。

#### Phase 2：两层 Warp→Block Max Reduction

```cuda
// Warp-level: __shfl_xor_sync butterfly reduction
// Block-level: 8 warps → shared memory → thread 0 reduction → shared memory broadcast
if (tid == 0) { s_max[0] = local_max; }
__syncthreads();
global_max = s_max[0];
```

**对应原理**：$m = \max_j S_j$——全局最大值。

#### Phase 3：Softmax + FTZ

```cuda
for (int k = tid; k < kv_len; k += DECODE_BLOCK_SIZE) {
    float val = s_scores[k] - global_max;
    float exp_val = (val > SOFTMAX_FTZ) ? expf(val) : 0.0f;  // FTZ
    s_scores[k] = exp_val;
    local_sum += exp_val;
}
```

**对应原理**：$p_j = e^{S_j - m}$，然后 $l = \sum p_j$。

#### Phase 4：加权 V 累加

```cuda
for (int d = tid; d < head_size; d += DECODE_BLOCK_SIZE) {
    float acc = 0.0f;
    // Unroll by 4
    for (; k + 3 < kv_len; k += 4) {
        acc += s_scores[k+0] * __half2float(__ldg(v0 + d));
        acc += s_scores[k+1] * __half2float(__ldg(v1 + d));
        // ...
    }
    o_ptr[d] = __float2half(acc * inv_sum);
}
```

**对应原理**：$O_d = \frac{1}{l} \sum_j p_j \cdot v_{j,d}$。每个线程负责一个或多个输出维度。

---

### 3.3 FlashAttention v1 — FP16 Decode with GPU-pos (CUDA Graph 兼容)

**文件**: `flash_attention_kernel.cu` → `flash_attention_decode_kernel_fp16_online_softmax`

```
Grid: [head_num]
Block: 128 threads (ONLINE_BLOCK_SIZE=128, = head_size)
```

这个 kernel 是专为 CUDA Graph 设计的：

#### 关键差异 1：Position 从 GPU 内存读取

```cuda
const int pos = *reinterpret_cast<const volatile int32_t*>(pos_ptr);
```

`volatile` 确保 CUDA Graph 回放时不使用缓存的旧值。

#### 关键差异 2：固定 Shared Memory + Online Softmax Tiling

```cuda
// 固定 tile 大小，smem 大小不随 pos 变化
constexpr int ONLINE_TILE_K = 256;
float* s_scores = ...;  // 仅 ONLINE_TILE_K 大小

for (int tile_start = 0; tile_start < kv_len; tile_start += ONLINE_TILE_K) {
    // ← 使用完整的 online softmax（和 prefill 相同逻辑）
}
```

这里 decode 退化为和 prefill 类似的 online softmax 结构，只是处理的是单个 query。

---

### 3.4 FlashAttention v1 — FP32 Prefill Kernel

**文件**: `flash_attention_kernel.cu` → `flash_attention_prefill_kernel`

```
Grid: [head_num, seq_len]
Block: 256 threads (BLOCK_SIZE_FP32)
```

与 FP16 prefill 的主要区别：

| 方面 | FP16 | FP32 |
|------|------|------|
| Block size | 128 | 256 |
| 数据类型 | half, half2 | float, float4 |
| Q·K 向量化 | float4 → half2 × 4 | float4 直接使用 |
| Max reduction | warp shuffle + smem | CUB BlockReduce |
| 每线程输出维度 | 1 (tid < head_size) | 多个 acc_o[4] |

FP32 kernel 使用 `cub::BlockReduce` 模板做 max/sum reduction，代码更简洁但依赖 CUB 库：

```cuda
typedef cub::BlockReduce<float, BLOCK_SIZE_FP32> BlockReduce;
float block_max = BlockReduce(temp_storage).Reduce(tile_max, cub::Max());
```

输出累积器使用数组 `acc_o[4]`，因为 256 个线程处理 128 个维度时不能 1:1 映射。

---

### 3.5 FlashAttention v1 — FP32 Decode Kernel

**文件**: `flash_attention_kernel.cu` → `flash_attention_decode_kernel_optimized`

```
Grid: [head_num]
Block: 256 threads
```

使用全量 softmax（非 tiling），但通过 shared memory 广播修复了 warp 间通信 bug：

```cuda
// Broadcast via shared memory (not warp shuffle)
if (tid == 0) { s_max[0] = global_max; }
__syncthreads();
global_max = s_max[0];
```

---

### 3.6 FlashAttention v2 — FP16 Prefill Kernel

**文件**: `flash_attention2_kernel.cu` → `flash_attention2_prefill_kernel_fp16`

```
Grid: [head_num, seq_len]
Block: 128 threads (FA2_BLOCK_SIZE)
```

#### 与 FA1 的关键代码差异：

**差异 1：更小的 Tile**

```cuda
constexpr int FA2_TILE_K = 64;    // FA2: 64 (vs FA1's 1024)
```

更小的 tile 意味着更少的 shared memory 需求和更好的寄存器压力管理。

**差异 2：Warp Shuffle 代替 CUB BlockReduce**

FA2 全程使用 warp-level `__shfl_xor_sync` 做 reduction，避免 CUB 模板实例化的额外 shared memory 开销：

```cuda
// FA2: 手写 warp shuffle reduction
#pragma unroll
for (int offset = 16; offset > 0; offset >>= 1) {
    tile_max_local = fmaxf(tile_max_local, __shfl_xor_sync(0xffffffff, tile_max_local, offset));
}
__shared__ float s_warp_max[FA2_NUM_WARPS];
if (lane_id == 0) s_warp_max[warp_id] = tile_max_local;
```

**差异 3：Flush-to-Zero 阈值**

```cuda
float exp_score = (val > FA2_SOFTMAX_FTZ) ? expf(val) : 0.0f;
// FA2_SOFTMAX_FTZ = -20.0
```

FA1 的 prefill FP16 kernel 也使用了 FTZ，但 FA2 将其作为标志性特征显式定义为常量。

**差异 4：V 累加的 4 路展开**

```cuda
// FA2: 固定 4 路展开
for (; k + 3 < tile_len; k += 4) {
    acc_o = fmaf(s0, v0, acc_o);
    acc_o = fmaf(s1, v1, acc_o);
    acc_o = fmaf(s2, v2, acc_o);
    acc_o = fmaf(s3, v3, acc_o);
}
```

FA1 的 FP16 prefill 使用 8 路展开。FA2 选择 4 路是因为更小的 tile（64 vs 1024）使得展开收益有差异。

#### 逐步对应原理：

| 代码步骤 | 原理对应 |
|----------|---------|
| `s_query[d] = q_ptr[d]` | 加载 $q_i$ 到 SRAM |
| `acc_qk += q·k` with half2 | 计算 $S_j = q \cdot k_j / \sqrt{d_k}$ |
| Warp shuffle max | $m_j = \max S_j$ |
| `m_new = max(row_max, m_j)` | $m^{(t)} = \max(m^{(t-1)}, m_j)$ |
| `correction = exp(row_max - m_new)` | $\alpha = e^{m^{(t-1)} - m^{(t)}}$ |
| `acc_o *= correction` | $O^{(t)} \leftarrow \alpha \cdot O^{(t-1)}$ (FA2 delayed rescaling) |
| `exp_score = exp(val)` with FTZ | $p_j = e^{S_j - m^{(t)}}$ |
| `acc_o += s * v` with fmaf | $O^{(t)} \leftarrow O^{(t)} + \sum p_j v_j$ |
| `row_sum = correction * row_sum + l_j` | $l^{(t)} = \alpha \cdot l^{(t-1)} + l_j$ |
| `acc_o * inv_sum` | $O = O^{(T)} / l^{(T)}$ |

---

### 3.7 FlashAttention v2 — FP16 Decode Kernel

**文件**: `flash_attention2_kernel.cu` → `flash_attention2_decode_kernel_fp16`

```
Grid: [head_num]
Block: 256 threads (FA2_DECODE_BLOCK)
```

FA2 decode kernel 的核心差异在于：

**使用 `__shfl_xor_sync` 而非 `__shfl_down_sync`**：

```cuda
// FA2 decode: butterfly reduction (all-to-all)
for (int offset = 16; offset > 0; offset /= 2) {
    local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
}
```

`__shfl_xor_sync` 做 butterfly 模式的 all-reduce，结束后 warp 内每个线程都持有正确的结果（无需额外广播）。而 `__shfl_down_sync` 只有 lane 0 持有结果。

**更快的 warp→block reduction**：

```cuda
// FA2: 用 shfl_xor 做跨 warp reduction
if (tid < FA2_DECODE_WARPS) local_max = s_max[tid];
for (int offset = FA2_DECODE_WARPS / 2; offset > 0; offset /= 2) {
    local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
}
if (tid == 0) s_max[0] = local_max;
__syncthreads();
global_max = s_max[0];
```

**V 累加使用 fmaf**：

```cuda
acc = fmaf(s_scores[k + 0], __half2float(__ldg(v0 + d)), acc);
```

FA1 decode 用 `acc += s * v`，FA2 用 `fmaf(s, v, acc)` 融合乘加，减少 1 条指令。

---

### 3.8 FlashAttention v2 — FP16 Decode with GPU-pos

**文件**: `flash_attention2_kernel.cu` → `flash_attention2_decode_kernel_fp16_gpu_pos`

```
Grid: [head_num]
Block: 128 threads
```

与 FA1 的 `flash_attention_decode_kernel_fp16_online_softmax` 结构基本相同（都是 Online Softmax + Tiling），主要差异是 FA2 的 FTZ 阈值和 tile 参数。

**关键实现细节**：

```cuda
// CUDA Graph 兼容：pos 从 GPU memory 读取
const int pos = *reinterpret_cast<const volatile int32_t*>(pos_ptr);

// 固定 tile + online softmax（与 FA1 GPU-pos 相同设计）
constexpr int TILE_K = 256;
for (int tile_start = 0; tile_start < kv_len; tile_start += TILE_K) {
    // online softmax with correction
    acc_o *= correction;
    // ... accumulate V
    row_sum = correction * row_sum + l_j;
}
```

---

### 3.9 FP32 Decode Kernels (FA1 vs FA2)

两者在 FP32 路径上结构非常相似，因为 FP32 没有 half2 向量化优势。主要差异在于：

| 方面 | FA1 FP32 Decode | FA2 FP32 Decode |
|------|-----------------|-----------------|
| Block size | 256 (BLOCK_SIZE_FP32) | 256 (FA2_BLOCK_SIZE_FP32) |
| Q·K 计算 | `score += q.x * k.x` | `score = fmaf(q.x, kv.x, score)` |
| Reduction | `__shfl_down_sync` + smem broadcast | `__shfl_down_sync` + smem broadcast |
| V 累加 | `acc += s * v` | `acc += s * v` |
| 广播 | smem (`s_max[0]`) | smem (`s_max[0]`) |

FA2 的 FP32 路径使用 `fmaf` 替代分立乘加，可以在支持的硬件上少一条指令。

---

## 四、FlashAttention v1 vs v2 的详细对比

### 4.1 算法层面对比

| 特征 | FlashAttention v1 | FlashAttention v2 |
|------|-------------------|-------------------|
| **Tile 大小** | 大 (TILE_K=1024) | 小 (FA2_TILE_K=64) |
| **Rescaling** | 每 tile 修正一次 O | 同样每 tile 修正，但 tile 更小 |
| **FTZ 阈值** | SOFTMAX_FTZ = -20.0 | FA2_SOFTMAX_FTZ = -20.0 |
| **Warp 并行** | 同一 tile 内冗余工作 | 不同 warp 处理不同 K 范围 |
| **同步次数** | 较多 __syncthreads | 较少（warp 独立计算更多） |
| **Max Reduction** | CUB BlockReduce (FP32) / warp shuffle (FP16) | 全程使用 warp shuffle |
| **sum Reduction** | 同上 | 同上 |

### 4.2 实现层面对比

#### 4.2.1 常量定义

```cuda
// FA1
constexpr int BLOCK_SIZE = 128;       // FP16
constexpr int BLOCK_SIZE_FP32 = 256;  // FP32
constexpr int TILE_K = 1024;

// FA2
constexpr int FA2_BLOCK_SIZE = 128;        // FP16
constexpr int FA2_BLOCK_SIZE_FP32 = 256;   // FP32
constexpr int FA2_TILE_K = 64;             // ← 16x smaller tiles!
constexpr int FA2_DECODE_BLOCK = 256;      // Decode 专用
constexpr int FA2_DECODE_TILE_K = 128;
```

FA2 的 tile 从 1024 缩小到 64，带来以下效果：
- **正面**：shared memory 需求降低 16x（`64 * 4B = 256B` vs `1024 * 4B = 4KB`）
- **正面**：更低的寄存器压力，更多寄存器用于数据预取
- **负面**：更多的 tile 迭代次数（但每次迭代更轻量）

#### 4.2.2 Q·K 点积实现

**FA1 FP16**:
```cuda
// 使用 float2 累加器
float2 acc = make_float2(0.0f, 0.0f);
for (int d = 0; d < head_size / 8; d++) {
    // ... half2 dot product
    acc.x += q_f.x * k_f.x;   // plain multiply-add
    acc.y += q_f.y * k_f.y;
}
```

**FA2 FP16**:
```cuda
// 使用 float2 累加器 + fmaf
float2 acc_qk = make_float2(0.0f, 0.0f);
for (int d = 0; d < head_size / 8; d++) {
    // ... half2 dot product
    acc_qk.x = fmaf(q_f.x, k_f.x, acc_qk.x);  // fused multiply-add
    acc_qk.y = fmaf(q_f.y, k_f.y, acc_qk.y);
}
```

`fmaf` 相比分立 `a * b + c` 的优势：
1. 只产生一次舍入（更高精度）
2. 在 SM87 上只需一条指令（vs 两条）

#### 4.2.3 Reduction 策略

**FA1 FP32 Prefill**: 使用 CUB BlockReduce

```cuda
typedef cub::BlockReduce<float, BLOCK_SIZE_FP32> BlockReduce;
float block_max = BlockReduce(temp_storage).Reduce(tile_max, cub::Max());
```

优点：代码简洁。缺点：CUB 需要额外的 shared memory（`TempStorage`），增加 smem 使用。

**FA2 FP32 Prefill**: 同样使用 CUB BlockReduce（这里与 FA1 相同）

```cuda
typedef cub::BlockReduce<float, FA2_BLOCK_SIZE_FP32> BlockReduce;
float block_max = BlockReduce(temp_storage).Reduce(tile_max, cub::Max());
```

**FA1/FA2 FP16**: 全程使用手写 warp shuffle

```cuda
// Step 1: Intra-warp reduction via butterfly shuffle
for (int offset = 16; offset > 0; offset >>= 1)
    val = op(val, __shfl_xor_sync(0xffffffff, val, offset));
// Step 2: Cross-warp via shared memory
if (lane_id == 0) s_warp[warp_id] = val;
__syncthreads();
// Step 3: Thread 0 final reduction + broadcast
```

#### 4.2.4 V 累加展开度

| Kernel | 展开级别 |
|--------|---------|
| FA1 FP16 Prefill | **8** (k += 8, 加载 8 个 V 行) |
| FA2 FP16 Prefill | **4** (k += 4, 加载 4 个 V 行) |
| FA1 FP16 Decode | **4** (k += 4) |
| FA2 FP16 Decode | **4** (k += 4, fmaf) |
| FA1 FP32 Decode | **1** (无展开) |
| FA2 FP32 Decode | **1** (无展开) |

FA1 的 8 路展开在大 tile（1024）场景下有优势，因为内循环迭代次数多。FA2 使用小 tile（64），展开到 4 是更好的平衡点。

#### 4.2.5 Decode Kernel 的 Reduction 风格

**FA1 Decode FP16** (`flash_attention_decode_kernel_fp16_optimized`):

```cuda
// 使用 __shfl_xor_sync → 每个线程都有结果
for (int offset = 16; offset > 0; offset /= 2) {
    local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
}
```

**FA2 Decode FP16** (`flash_attention2_decode_kernel_fp16`):

```cuda
// 完全相同的策略
for (int offset = 16; offset > 0; offset /= 2) {
    local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
}
```

在 decode kernel 中，FA1 和 FA2 的 reduction 策略完全一致。

### 4.3 Shared Memory 使用对比

| Kernel | Shared Memory 组成 | 典型大小 |
|--------|-------------------|---------|
| FA1 FP16 Prefill | query(256B) + scores(4KB) | ~4.3KB |
| FA2 FP16 Prefill | query(256B) + scores(256B) | ~0.5KB |
| FA1 FP16 Decode | query(256B) + scores(kv_len×4B) + max/sum(64B) | 变长 |
| FA2 FP16 Decode | query(256B) + scores(kv_len×4B) + max/sum(64B) | 变长（相同） |
| FA1 FP16 GPU-pos | query(256B) + scores(1KB) + max/sum(32B) | ~1.3KB 固定 |
| FA2 FP16 GPU-pos | query(256B) + scores(1KB) + max/sum(32B) | ~1.3KB 固定 |
| FA1 FP32 Prefill | query(512B) + scores(4KB) + CUB | ~5KB |
| FA2 FP32 Prefill | query(512B) + scores(256B) + CUB | ~1KB |

FA2 prefill 的 shared memory 使用量显著降低（tile 从 1024→64），使得更多 block 可以同时驻留在 SM 上（更高 occupancy）。

### 4.4 性能特征对比（基于实测数据）

| 模型 | FA1 Prefill | FA2 Prefill | FA1 Decode | FA2 Decode |
|------|------------|------------|------------|------------|
| Qwen3-8B FP16 | 144.3 tok/s | 145.0 tok/s | 10.26 tok/s | 10.29 tok/s |
| Qwen3-8B AWQ | 134.5 tok/s | 132.5 tok/s | 9.73 tok/s | 9.71 tok/s |
| Qwen2.5-7B FP32 | 6.09 tok/s | 6.11 tok/s | 5.70 tok/s | 5.68 tok/s |
| Qwen2.5-7B FP16 | 132.3 tok/s | 154.4 tok/s | 10.86 tok/s | 10.92 tok/s |
| Qwen3-VL FP16 | 662.6 tok/s | 658.6 tok/s | 10.24 tok/s | 10.24 tok/s |

**分析**：

1. **Decode 速度几乎相同**：Decode 阶段是 memory-bound（只有一个 query token），attention 计算不是瓶颈，因此 FA1/FA2 差异不明显。

2. **Prefill 差幅有限**：在当前测试的短序列（34 token）下，tiling 优势不明显。FA2 在 Qwen2.5-7B FP16 上的 prefill 提升 16.7%（132.3→154.4 tok/s），可能是因为更小的 tile 带来更好的 L2 cache 利用率。

3. **长序列场景下 FA2 的理论优势更大**：tile 越小，对 shared memory 和 L2 cache 的压力越小，当 KV 长度增大到数千 token 时，FA2 的 tile 策略将更有优势。

### 4.5 总结

| 维度 | FlashAttention v1 | FlashAttention v2 |
|------|-------------------|-------------------|
| 核心数学 | Online Softmax + Tiling | 相同（数学等价） |
| Tile 大小 | 大 (1024) | 小 (64) |
| Non-GEMM FLOPs | 较多（每 tile 修正 d 个维度） | 较少（tile 小但迭代多，总量类似） |
| Shared Memory | ~4KB | ~0.5KB |
| Warp 并行 | 冗余计算 | 分工计算 |
| 同步量 | 较多 | 较少 |
| FP16 Q·K | `a * b + c` | `fmaf(a, b, c)` |
| V 累加展开 | 8x | 4x |
| FTZ | ✓ | ✓ (显式常量) |
| CUDA Graph | ✓ (online softmax kernel) | ✓ (online softmax kernel) |
| 数值精度 | 完全精确 | 完全精确 |

---

## 附录 A：工程中的 Kernel 索引

| Kernel 函数名 | 类型 | 精度 | 场景 | 文件 |
|---------------|------|------|------|------|
| `flash_attention_prefill_kernel` | FA1 | FP32 | Prefill | flash_attention_kernel.cu |
| `flash_attention_prefill_kernel_fp16` | FA1 | FP16 | Prefill | flash_attention_kernel.cu |
| `flash_attention_decode_kernel_optimized` | FA1 | FP32 | Decode | flash_attention_kernel.cu |
| `flash_attention_decode_kernel_fp16_optimized` | FA1 | FP16 | Decode | flash_attention_kernel.cu |
| `flash_attention_decode_kernel_fp16_online_softmax` | FA1 | FP16 | Decode+CUDA Graph | flash_attention_kernel.cu |
| `flash_attention2_prefill_kernel_fp16` | FA2 | FP16 | Prefill | flash_attention2_kernel.cu |
| `flash_attention2_prefill_kernel` | FA2 | FP32 | Prefill | flash_attention2_kernel.cu |
| `flash_attention2_decode_kernel_fp16` | FA2 | FP16 | Decode | flash_attention2_kernel.cu |
| `flash_attention2_decode_kernel` | FA2 | FP32 | Decode | flash_attention2_kernel.cu |
| `flash_attention2_decode_kernel_fp16_gpu_pos` | FA2 | FP16 | Decode+CUDA Graph | flash_attention2_kernel.cu |

## 附录 B：层级分发流程

```
用户 CLI: --attention flash2
    │
    ▼
inference_common.h: parse_args() → config.attention_type = kAttentionFlash2
    │
    ▼
run_model_inference(): model.set_attention_type(config.attention_type)
    │
    ▼
QwenXModel::set_attention_type(type)
    ├── flash_attention_decode_layer_->set_attention_type(type)
    ├── flash_attention_prefill_layer_->set_attention_type(type)
    └── flash_attention_decode_gpu_pos_layer_->set_attention_type(type) (VL only)
    │
    ▼
QwenXModel::attention_mha() / attention_mha_with_graph()
    ├── FP16 data → flash_attention_decode_layer_->forward()
    │                 ├── kAttentionFlash2 → flash_attention2_decode_fp16_cu()
    │                 └── kAttentionFlash1 → flash_attention_decode_fp16_cu()
    ├── FP32 + MHA → mha_layer_->forward()
    └── FP32 + FA  → flash_attention_decode_layer_->forward()
                       ├── kAttentionFlash2 → flash_attention2_decode_cu()
                       ├── kAttentionFlash1 → flash_attention_decode_cu()
                       └── kAttentionMHA → get_mha_kernel()
```
