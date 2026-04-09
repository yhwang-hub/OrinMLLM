# Flash Attention Prefill FP16 Kernel — Grid/Block/Thread 详细分析报告

## 1. 问题背景

本报告针对 `flash_attention_prefill_kernel_fp16` CUDA kernel，从 **Grid → Block → Thread** 三个层面详细剖析其网格划分策略和 Attention 计算过程。

### 1.1 关键常量

| 常量 | 值 | 含义 |
|------|-----|------|
| `BLOCK_SIZE` | 128 | 每个 Block 中的线程数（针对 Orin 优化） |
| `TILE_K` | 1024 | KV 序列分块大小（每次处理 1024 个 KV 位置） |
| `head_size` | 128 | 每个注意力头的维度（Qwen3-8B） |
| `SOFTMAX_FTZ` | -20.0f | Softmax 下溢截断阈值，`exp(-20) ≈ 2e-9`，视为零 |

### 1.2 核心设计原则

**BLOCK_SIZE = head_size = 128**，这意味着：
- 每个线程恰好负责输出向量的**一个维度**
- 无需循环处理多个输出维度，最大化线程利用率

---

## 2. Grid 层面：任务的全局划分

### 2.1 Grid 配置

```cpp
dim3 grid(head_num, seq_len);   // 2D Grid
dim3 block(BLOCK_SIZE);          // 1D Block, 128 threads
```

Grid 是一个 **二维网格**：
- **X 维度** = `head_num`（注意力头数量，Qwen3-8B 为 32）
- **Y 维度** = `seq_len`（当前 prefill 序列长度）

### 2.2 Grid 全局视图

以 `head_num=32, seq_len=5` 为例：

```
                              Grid (head_num × seq_len)
        ┌─────────────────────────────────────────────────────────────┐
        │                     head_num = 32 (X轴)                     │
        │   head 0    head 1    head 2   ...   head 30   head 31     │
        ├─────────┬─────────┬─────────┬─────┬──────────┬──────────┤
seq 0   │ Block   │ Block   │ Block   │ ... │ Block    │ Block    │  ← Q[0]对所有head
        │ (0,0)   │ (1,0)   │ (2,0)   │     │ (30,0)  │ (31,0)  │
        ├─────────┼─────────┼─────────┼─────┼──────────┼──────────┤
seq 1   │ Block   │ Block   │ Block   │ ... │ Block    │ Block    │  ← Q[1]对所有head
        │ (0,1)   │ (1,1)   │ (2,1)   │     │ (30,1)  │ (31,1)  │
        ├─────────┼─────────┼─────────┼─────┼──────────┼──────────┤
seq 2   │ Block   │ Block   │ Block   │ ... │ Block    │ Block    │  ← Q[2]对所有head
        │ (0,2)   │ (1,2)   │ (2,2)   │     │ (30,2)  │ (31,2)  │
        ├─────────┼─────────┼─────────┼─────┼──────────┼──────────┤
seq 3   │ Block   │ Block   │ Block   │ ... │ Block    │ Block    │  ← Q[3]对所有head
        │ (0,3)   │ (1,3)   │ (2,3)   │     │ (30,3)  │ (31,3)  │
        ├─────────┼─────────┼─────────┼─────┼──────────┼──────────┤
seq 4   │ Block   │ Block   │ Block   │ ... │ Block    │ Block    │  ← Q[4]对所有head
        │ (0,4)   │ (1,4)   │ (2,4)   │     │ (30,4)  │ (31,4)  │
        └─────────┴─────────┴─────────┴─────┴──────────┴──────────┘

总 Block 数 = head_num × seq_len = 32 × 5 = 160 个 Block
每个 Block = 128 个线程
总线程数 = 160 × 128 = 20,480
```

### 2.3 每个 Block 的职责

```
Block(head, seq_idx) 的任务：
  计算第 seq_idx 个 query token 在第 head 个注意力头上的 attention 输出

  输入: Q[seq_idx, head, :] — 一个 head_size=128 维的 query 向量
  输出: O[seq_idx, head, :] — 一个 head_size=128 维的 output 向量

  需要访问: K_cache[0..kv_len-1, kv_head, :]  — kv_len 个 key 向量
            V_cache[0..kv_len-1, kv_head, :]  — kv_len 个 value 向量
            其中 kv_head = head / kv_mul (GQA分组)
```

### 2.4 因果掩码（Causal Mask）via `kv_len`

Prefill 使用**因果注意力**（每个 token 只能看到自己及之前的 token）。因果掩码通过 `kv_len` 隐式实现：

```cpp
const int cur_pos = start_pos + seq_idx;   // 当前 token 在全局序列中的位置
const int kv_len = cur_pos + 1;            // 只看 [0, cur_pos] 范围的 KV
```

```
                        KV Cache 中的位置
              0     1     2     3     4     5     6
            ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┐
            │     │     │     │     │     │     │     │  ← 历史token  ← 新token
            └─────┴─────┴─────┴─────┴─────┴─────┴─────┘
              ↑ start_pos=3        ↑ start_pos+seq_len-1=6

seq_idx=0:  kv_len = 3+0+1 = 4   → 看 [0,1,2,3]
seq_idx=1:  kv_len = 3+1+1 = 5   → 看 [0,1,2,3,4]
seq_idx=2:  kv_len = 3+2+1 = 6   → 看 [0,1,2,3,4,5]
seq_idx=3:  kv_len = 3+3+1 = 7   → 看 [0,1,2,3,4,5,6]
                                       │← 前缀-→│← prefill→│
```

**不同的 Block（不同的 `seq_idx`）处理的 KV 长度不同**，这是因果掩码的自然实现——靠前的 query 看到更短的 KV 序列。

---

## 3. Block 层面：128 线程的协作

### 3.1 Block 内部结构

每个 Block 包含 128 个线程，组成 **4 个 Warp**（每 Warp 32 线程）：

```
Block (head=h, seq_idx=s)
┌──────────────────────────────────────────────────────────┐
│                    128 threads (4 warps)                  │
│                                                          │
│  Warp 0: tid [0..31]      Warp 1: tid [32..63]          │
│  Warp 2: tid [64..95]     Warp 3: tid [96..127]         │
│                                                          │
│  每个线程 tid 负责输出向量 O 的第 tid 个维度             │
│  tid=0 → O[s,h,0]                                       │
│  tid=1 → O[s,h,1]                                       │
│  ...                                                      │
│  tid=127 → O[s,h,127]                                   │
└──────────────────────────────────────────────────────────┘
```

### 3.2 共享内存布局

```cpp
const int smem_size = head_size * sizeof(half) + TILE_K * sizeof(float);
//                    256 bytes                  4096 bytes
//                    = 4,352 bytes 总共
```

```
Shared Memory (4,352 bytes)
┌──────────────────────────┬─────────────────────────────────────────┐
│   s_query[128] (half)    │         s_scores[1024] (float)          │
│      256 bytes            │           4,096 bytes                   │
│                          │                                         │
│  Q向量的128个FP16分量     │  当前 tile 中每个 KV 位置的注意力分数    │
│  被所有线程共享读取       │  线程协作计算，用于 softmax + V 加权     │
└──────────────────────────┴─────────────────────────────────────────┘
```

### 3.3 每个线程的私有状态

每个线程维护 3 个 float 寄存器，用于 **Online Softmax** 的增量更新：

```
Thread tid 的私有寄存器:
┌─────────────────────────────────────────────────┐
│  acc_o   : float  — O[tid] 维度的加权累加值      │
│  row_max : float  — 截至目前所有 tile 的最大 score │
│  row_sum : float  — 截至目前 exp(score) 的总和    │
└─────────────────────────────────────────────────┘
```

---

## 4. Thread 层面：每个线程的计算过程

### 4.1 整体计算流程图

```
Block(head=h, seq_idx=s) 的执行流程

    ┌─────────────────────────────┐
    │ 步骤 0: 加载 Q → 共享内存    │  ← 所有128线程协作
    │ s_query[tid] = Q[s,h,tid]   │     一次 coalesced load
    └────────────┬────────────────┘
                 │
                 ▼
    ┌─────────────────────────────┐
    │ 初始化线程私有状态:          │
    │   acc_o = 0.0               │
    │   row_max = -FLT_MAX        │
    │   row_sum = 0.0             │
    └────────────┬────────────────┘
                 │
      ╔══════════╧═══════════╗
      ║  for tile_start = 0  ║ ← 外层循环：每次处理 TILE_K=1024 个 KV 位置
      ║  to kv_len           ║
      ║  step TILE_K         ║
      ╠══════════╤═══════════╣
                 │
                 ▼
    ┌─────────────────────────────┐
    │ 步骤 1: 计算 Q·K scores     │  ← 128 线程协作
    │         (half2 向量化)       │     每线程处理 tile中的部分KV位置
    │ 结果 → s_scores[0..tile-1]  │
    └────────────┬────────────────┘
                 │
                 ▼
    ┌─────────────────────────────┐
    │ 步骤 2: Block Reduce Max    │  ← Warp shuffle + 共享内存
    │ 求 s_scores 的最大值 m_j    │
    └────────────┬────────────────┘
                 │
                 ▼
    ┌─────────────────────────────┐
    │ 步骤 3: 更新全局最大值       │
    │ m_new = max(row_max, m_j)   │
    └────────────┬────────────────┘
                 │
                 ▼
    ┌─────────────────────────────┐
    │ 步骤 4: 计算 exp(score-m_new)│  ← 128 线程协作
    │ 刷新 s_scores 为 softmax 权重│
    │ + 累加 tile_sum              │
    └────────────┬────────────────┘
                 │
                 ▼
    ┌─────────────────────────────┐
    │ 步骤 5: Block Reduce Sum    │  ← Warp shuffle + 共享内存
    │ 求 tile_sum → l_j           │
    └────────────┬────────────────┘
                 │
                 ▼
    ┌─────────────────────────────┐
    │ 步骤 6: 修正历史累加值       │  ← 每个线程独立
    │ correction = exp(row_max -   │
    │              m_new)          │
    │ acc_o *= correction          │
    └────────────┬────────────────┘
                 │
                 ▼
    ┌─────────────────────────────┐
    │ 步骤 7: 累加 softmax(Q·K)×V │  ← 每个线程处理自己的 O[tid]
    │ V 通过 stride 访问全局内存   │     展开循环 ×8 提高 ILP
    └────────────┬────────────────┘
                 │
                 ▼
    ┌─────────────────────────────┐
    │ 更新 row_max, row_sum       │
    │ row_max = m_new             │
    │ row_sum = correction *      │
    │           row_sum + l_j     │
    └────────────┬────────────────┘
                 │
      ╚══════════╧═══════════╝  (下一个 tile)
                 │
                 ▼
    ┌─────────────────────────────┐
    │ 最终归一化 + 写回            │
    │ O[s,h,tid] = acc_o / row_sum│  ← 每个线程写一个 half
    └─────────────────────────────┘
```

### 4.2 步骤 0：加载 Query 到共享内存

```cpp
const half* q_ptr = Q + seq_idx * dim + head * head_size;
for (int d = tid; d < head_size; d += BLOCK_SIZE) {
    s_query[d] = q_ptr[d];
}
__syncthreads();
```

由于 `head_size = BLOCK_SIZE = 128`，每个线程**恰好加载一个元素**：

```
全局内存 Q:
  Q[seq_idx=s] = [ head 0 (128 half) | head 1 (128 half) | ... | head 31 (128 half) ]
                                        ↑
                                  head * head_size

共享内存 s_query:
  ┌────┬────┬────┬────┬─────┬──────┬──────┐
  │ d0 │ d1 │ d2 │ d3 │ ... │ d126 │ d127 │   ← 128 个 half
  └────┴────┴────┴────┴─────┴──────┴──────┘
    ↑    ↑    ↑    ↑           ↑      ↑
   t0   t1   t2   t3         t126   t127      ← 每个线程加载1个
```

**访存模式**: 128 线程连续读取 128 个连续 half（256 bytes），完美 **coalesced**，一次事务完成。

### 4.3 步骤 1：计算 Q·K Scores（float4 向量化）

这一步 128 线程**协作**计算 Q 与当前 tile 中所有 K 的点积：

```cpp
for (int k_idx = tid; k_idx < tile_len; k_idx += BLOCK_SIZE) {
    // 每个线程处理一个 KV 位置的 Q·K 点积
    const float4* k_ptr_f4 = reinterpret_cast<const float4*>(K_cache + kv_pos * kv_dim + head_offset);
    const float4* q_ptr_f4 = reinterpret_cast<const float4*>(s_query);
    
    float2 acc = make_float2(0.0f, 0.0f);
    for (int d = 0; d < head_size / 8; d++) {  // 128/8 = 16 次迭代
        float4 q_packed = q_ptr_f4[d];           // 从 smem 读 4 个 half2
        float4 k_packed = __ldg(k_ptr_f4 + d);   // 从 global 读 4 个 half2
        // 每个 float4 = 4 个 half2 = 8 个 half
        // 展开 4 次 half2→float2 乘加
    }
    float score = (acc.x + acc.y) * scale;
    s_scores[k_idx] = score;
}
```

**线程-KV位置映射**（假设 tile_len=1024, BLOCK_SIZE=128）：

```
tile 中有 1024 个 KV 位置需要计算 Q·K：

线程 tid=0:   计算 k_idx = 0,  128, 256, 384, 512, 640, 768, 896  → 8 个 Q·K
线程 tid=1:   计算 k_idx = 1,  129, 257, 385, 513, 641, 769, 897  → 8 个 Q·K
线程 tid=2:   计算 k_idx = 2,  130, 258, 386, 514, 642, 770, 898  → 8 个 Q·K
...
线程 tid=127: 计算 k_idx = 127,255, 383, 511, 639, 767, 895,1023  → 8 个 Q·K

每个线程计算一个 Q·K 需要: 128 维点积
  = 16 次 float4 load（128-bit） × 2（Q 和 K）
  = 16 次 fma 操作（4 个 half2 → 8 个乘加）
  最终累加为 1 个 float score
```

**float4 向量化读取详解**：

```
一个 float4 = 128 bit = 4 × float = 8 × half

K_cache 中一个 key 向量: [d0, d1, d2, d3, d4, d5, d6, d7, ..., d127]  (128 half)
                          ├─── float4[0] ────┤├── float4[1] ──┤  ...  ├ float4[15] ┤
                          8 half = 16 bytes     8 half            ...   8 half

s_query (共享内存):      [d0, d1, d2, d3, d4, d5, d6, d7, ..., d127]
                          ├─── float4[0] ────┤├── float4[1] ──┤  ...  ├ float4[15] ┤

Q·K = Σ(d=0..127) q[d]*k[d]
    = Σ(d=0..15) dot(q_float4[d], k_float4[d])
    每个 float4 内部:
      half2[0..3] → float2 → fmaf × 4
```

### 4.4 步骤 2-3：Block Reduce Max（求 tile 内最大 score）

```
Warp 级 reduce:
  每个 warp 内 32 线程用 __shfl_xor_sync 做蝶形归约
  5 轮（offset=16,8,4,2,1），每轮交换并取 max

  Warp 0 (tid 0-31):   max → s_warp_max[0]
  Warp 1 (tid 32-63):  max → s_warp_max[1]
  Warp 2 (tid 64-95):  max → s_warp_max[2]
  Warp 3 (tid 96-127): max → s_warp_max[3]

Thread 0 跨 warp reduce:
  m_j = max(s_warp_max[0..3])
  s_warp_max[0] = m_j  → 广播给所有线程
```

```
┌─────────────────────────────────────────────────────────────────────┐
│               Block Reduce Max 示意图 (128 threads)                 │
│                                                                     │
│  Warp 0                    Warp 1                                   │
│  t0  t1  t2 ... t31       t32 t33 ... t63                          │
│  ↓   ↓   ↓      ↓         ↓   ↓       ↓                           │
│  ├──shfl_xor──┤            ├──shfl_xor──┤                          │
│  │ offset=16  │            │ offset=16  │                          │
│  ├──shfl_xor──┤            ├──shfl_xor──┤                          │
│  │ offset=8   │            │ offset=8   │                          │
│  ├──shfl_xor──┤            ├──shfl_xor──┤                          │
│  │ offset=4   │            │ offset=4   │                          │
│  ├──shfl_xor──┤            ├──shfl_xor──┤                          │
│  │ offset=2   │            │ offset=2   │                          │
│  ├──shfl_xor──┤            ├──shfl_xor──┤                          │
│  │ offset=1   │            │ offset=1   │                          │
│  ↓                         ↓                                        │
│  s_warp_max[0]             s_warp_max[1]                           │
│       ↘                    ↙                                        │
│            Thread 0: max(4 warp maxes)                              │
│                   → m_j (广播)                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.5 步骤 4-5：Softmax 权重计算 + Block Reduce Sum

```cpp
for (int k_idx = tid; k_idx < tile_len; k_idx += BLOCK_SIZE) {
    float val = s_scores[k_idx] - m_new;
    float exp_score = (val > SOFTMAX_FTZ) ? expf(val) : 0.0f;  // flush-to-zero
    s_scores[k_idx] = exp_score;          // 覆写为 softmax 权重
    tile_sum_local += exp_score;           // 局部累加
}
// 然后 Block Reduce Sum（与 Max 结构相同，替换 fmaxf 为 +）
```

线程分工与步骤 1 相同——每线程处理 `tile_len / BLOCK_SIZE` 个 KV 位置。

### 4.6 步骤 6-7：Online Softmax 修正 + V 累加（核心！）

这是每个线程**独立**处理自己负责的输出维度 `O[tid]` 的阶段。

#### 4.6.1 Online Softmax 修正

```cpp
float correction = expf(row_max - m_new);
acc_o *= correction;
```

当新 tile 的最大值 `m_j` 大于历史最大值 `row_max` 时，`m_new > row_max`，所以 `correction < 1`，历史累加值需要缩小。这保证了数值稳定性，等价于在所有 tile 处理完后用全局最大值做 softmax。

#### 4.6.2 V 累加

```
Thread tid 的任务: 计算 O[s, h, tid] = Σ_k softmax_weight[k] × V[k, kv_head, tid]

V_cache 内存布局 (线性存储):
  V_cache[pos, kv_head, d] = V_cache + pos * kv_dim + kv_head * head_size + d

  对于 thread tid:
    v_thread_base = V_cache + kv_head * head_size + tid
    V[k, tid] = *(v_thread_base + k * kv_dim)     ← stride = kv_dim
```

```
Thread tid 读取 V 的内存访问模式:

V_cache 全局内存（按 KV 位置排列）:
┌──────────────────────────────────────────────────────────────┐
│ pos=0: [kv_head0: d0 d1 ... d127 | kv_head1: ... | ...]     │
│ pos=1: [kv_head0: d0 d1 ... d127 | kv_head1: ... | ...]     │
│ pos=2: [kv_head0: d0 d1 ... d127 | kv_head1: ... | ...]     │
│ ...                                                          │
│ pos=k: [kv_head0: d0 d1 ... d127 | kv_head1: ... | ...]     │
└──────────────────────────────────────────────────────────────┘

Thread tid=3 (处理 O[s,h,3]) 读取的位置（↓ 表示读取）:
  pos=0: [..., d3↓, ...]
  pos=1: [..., d3↓, ...]     每次跳 kv_dim 个 half
  pos=2: [..., d3↓, ...]     = stride 访问
  ...

128 个线程同时读取同一个 pos 的 128 个连续维度:
  pos=k: [d0↓ d1↓ d2↓ d3↓ ... d127↓]  ← coalesced! 128 × 2B = 256B
```

**循环展开优化**（×8 展开）：

```cpp
for (; k + 7 < tile_len; k += 8) {
    // 一次迭代处理 8 个 KV 位置
    // 读 8 个 s_scores[k..k+7] (共享内存 → 广播给所有线程)
    // 读 8 个 V[k..k+7, tid]   (全局内存 → 每线程读自己的维度)
    // 8 次 fmaf 累加到 acc_o
}
```

```
展开 ×8 的数据流 (Thread tid):

                 s_scores[]          V_cache (通过 __ldg 缓存读取)
                 (共享内存)           (全局内存, stride=kv_dim)
                    │                       │
  k=0:  score[0] ──┤    V[tile+0, tid] ────┤
  k=1:  score[1] ──┤    V[tile+1, tid] ────┤
  k=2:  score[2] ──┤    V[tile+2, tid] ────┤
  k=3:  score[3] ──┤    V[tile+3, tid] ────┤  → 8 × fmaf → acc_o
  k=4:  score[4] ──┤    V[tile+4, tid] ────┤
  k=5:  score[5] ──┤    V[tile+5, tid] ────┤
  k=6:  score[6] ──┤    V[tile+6, tid] ────┤
  k=7:  score[7] ──┤    V[tile+7, tid] ────┤
                    │                       │
```

---

## 5. Tiled Online Softmax + V 累加的完整数据流

以 `kv_len = 2500, TILE_K = 1024` 为例：

```
Tile 0: [0, 1024)     tile_len = 1024
Tile 1: [1024, 2048)  tile_len = 1024
Tile 2: [2048, 2500)  tile_len = 452

═══════════════════════════════════════════════════════════════
Tile 0 处理:
  ① Q·K → s_scores[0..1023]           (128 线程协作)
  ② tile_max = max(s_scores)           (block reduce)
  ③ m_new = max(-∞, tile_max) = tile_max
  ④ s_scores[i] = exp(s_scores[i] - m_new)  (128 线程协作)
  ⑤ l_j = Σ s_scores[i]               (block reduce)
  ⑥ correction = exp(-∞ - m_new) = 0, acc_o *= 0 → acc_o = 0
  ⑦ acc_o += Σ_k s_scores[k] × V[k, tid]  (每线程独立)
  更新: row_max = m_new, row_sum = 0 × 0 + l_j = l_j

═══════════════════════════════════════════════════════════════
Tile 1 处理:
  ① Q·K → s_scores[0..1023]
  ② tile_max = max(s_scores)
  ③ m_new = max(row_max, tile_max)     ← 可能更新全局最大值!
  ④ s_scores[i] = exp(s_scores[i] - m_new)
  ⑤ l_j = Σ s_scores[i]
  ⑥ correction = exp(row_max_old - m_new)  ← 修正历史值!
     acc_o *= correction                    ← 缩放历史累加
  ⑦ acc_o += Σ_k s_scores[k] × V[k+1024, tid]
  更新: row_max = m_new, row_sum = correction × row_sum_old + l_j

═══════════════════════════════════════════════════════════════
Tile 2 处理:
  ① Q·K → s_scores[0..451]           (只有452个有效, 128线程各处理≈4个)
  ② - ⑦ 同上
  更新: row_max = m_final, row_sum = l_final

═══════════════════════════════════════════════════════════════
最终输出:
  O[s, h, tid] = acc_o / row_sum      (就是标准的 softmax(QK^T) × V)
```

---

## 6. GQA（Grouped Query Attention）的处理

Qwen3-8B 使用 GQA：`head_num=32, kv_head_num=8, kv_mul=4`。

```cpp
const int kv_head = head / kv_mul;  // 4 个 Q head 共享 1 个 KV head
```

```
Q heads:  0  1  2  3 │ 4  5  6  7 │ 8  9 10 11 │ ... │ 28 29 30 31
          ↓  ↓  ↓  ↓ │ ↓  ↓  ↓  ↓ │ ↓  ↓  ↓  ↓ │     │  ↓  ↓  ↓  ↓
KV heads:    0        │     1       │     2       │ ... │     7

Block(head=0,s), Block(head=1,s), Block(head=2,s), Block(head=3,s)
  都读取 KV head 0 的 K 和 V
  → 各自有不同的 Q，但共享相同的 K/V
  → 4 个 Block 独立计算，通过 L1/L2 cache 自然复用 KV 数据
```

---

## 7. 内存访问模式总结

### 7.1 全局内存访问

| 数据 | 访问方式 | 合并度 |
|------|----------|--------|
| Q (加载到 smem) | 128 线程连续读 128 half = 256B | 完美 coalesced |
| K (Q·K 点积) | 每线程独立读一行 K，float4 向量化 | 每线程内连续，线程间无合并 |
| V (加权累加) | 128 线程同时读 V[k] 的 128 维 | 完美 coalesced (256B) |
| O (写回) | 128 线程连续写 128 half = 256B | 完美 coalesced |

### 7.2 共享内存访问

| 数据 | 读/写 | 冲突分析 |
|------|-------|----------|
| s_query (Q·K 中读取) | 所有线程读相同地址（广播） | float4 读取，无 bank conflict |
| s_scores (写入 score) | 每线程写不同位置 | 无冲突 (stride = BLOCK_SIZE) |
| s_scores (读 softmax 权重) | V 累加时所有线程读同一 s_scores[k] | 广播读，无冲突 |

---

## 8. 计算量与性能分析

### 8.1 每个 Block 的计算量

以 `head_size=128, kv_len=2048` 为例：

| 阶段 | 运算量 |
|------|--------|
| Q·K 点积 | 2048 × 128 × 2 = 524,288 FLOPs |
| Softmax (exp + reduce) | 2048 × ~5 ≈ 10,240 FLOPs |
| score × V | 2048 × 128 × 2 = 524,288 FLOPs |
| **总计** | **~1,058,816 FLOPs / Block** |

### 8.2 全局计算量

```
总 Blocks = head_num × seq_len = 32 × 2048 = 65,536
总 FLOPs ≈ 65,536 × 1.06M ≈ 69.2 GFLOPs (单层)
```

### 8.3 Orin 上的 SM 占用

```
Orin GPU: 16 SMs (Jetson AGX Orin)
每个 Block: 128 threads, 4352B smem, ~40 寄存器/线程
每个 SM 可容纳: ~8 Blocks (受寄存器限制)
活跃 Blocks: 16 × 8 = 128 Blocks
总 Blocks: 65,536
每个 SM 处理: ~4,096 Blocks (轮转调度)
```

---

## 9. 完整的网格划分示意图

```
                    ┌─────────────────────────────────────────────────┐
                    │              CUDA Kernel Launch                  │
                    │ grid(head_num=32, seq_len=N), block(128)        │
                    └─────────────────────┬───────────────────────────┘
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              │                           │                           │
              ▼                           ▼                           ▼
     ┌────────────────┐         ┌────────────────┐         ┌────────────────┐
     │ Block(0, 0)    │         │ Block(h, s)    │         │ Block(31,N-1)  │
     │ head=0, seq=0  │         │ head=h, seq=s  │         │ head=31,seq=N-1│
     │                │         │                │         │                │
     │ Q[0,0,:128]    │         │ Q[s,h,:128]    │         │ Q[N-1,31,:128] │
     │ × K[0..0]      │         │ × K[0..s+sp]   │         │ × K[0..N-1+sp] │
     │ → O[0,0,:128]  │         │ → O[s,h,:128]  │         │ → O[N-1,31,128]│
     └───────┬────────┘         └───────┬────────┘         └───────┬────────┘
             │                          │                          │
             ▼                          ▼                          ▼
     ┌────────────────┐         ┌────────────────┐         ┌────────────────┐
     │  128 Threads   │         │  128 Threads   │         │  128 Threads   │
     │                │         │                │         │                │
     │ t0→O[dim 0]    │         │ t0→O[dim 0]    │         │ t0→O[dim 0]    │
     │ t1→O[dim 1]    │         │ t1→O[dim 1]    │         │ t1→O[dim 1]    │
     │ t2→O[dim 2]    │         │ t2→O[dim 2]    │         │ t2→O[dim 2]    │
     │ ...            │         │ ...            │         │ ...            │
     │ t127→O[dim 127]│         │ t127→O[dim 127]│         │ t127→O[dim 127]│
     └───────┬────────┘         └───────┬────────┘         └───────┬────────┘
             │                          │                          │
             ▼                          ▼                          ▼
     ┌────────────────────────────────────────────────────────────────────┐
     │                    Tiled KV Processing (每个Block内部)             │
     │                                                                    │
     │   KV 序列: [0 ─────────── kv_len ─────────────────]               │
     │            │← Tile 0 →│← Tile 1 →│← Tile 2 →│←T3→│              │
     │             TILE_K=1024  TILE_K     TILE_K    余数                 │
     │                                                                    │
     │   每个 Tile 内:                                                    │
     │   ┌─────────────────────────────────────────────┐                 │
     │   │ 128线程协作 → Q·K scores (float4 向量化)     │                 │
     │   │ 128线程协作 → Block Reduce Max               │                 │
     │   │ 128线程协作 → exp(score-max), Reduce Sum     │                 │
     │   │ 每线程独立  → acc_o += score[k] × V[k, tid] │                 │
     │   └─────────────────────────────────────────────┘                 │
     │                                                                    │
     │   Online Softmax: 跨 tile 修正，无需两遍扫描                       │
     └────────────────────────────────────────────────────────────────────┘
```

---

## 10. 与标准 Attention 的对比

| 特性 | 标准 Attention | 本 Kernel (Tiled Flash Attention) |
|------|---------------|-----------------------------------|
| Score 存储 | 分配 `[seq_len, kv_len]` 矩阵 | 仅需 `[TILE_K=1024]` 共享内存 |
| Softmax | 两遍扫描（求 max → 求 sum → normalize） | Online Softmax（一遍扫描 + 修正） |
| 内存复杂度 | $O(N^2)$（N = 序列长度） | $O(N)$（仅 tile 大小的共享内存） |
| V 访问 | 通常用 GEMM 库 | 手写 fmaf 展开，stride 访问 |
| 并行粒度 | Block 处理矩阵 tile | **Block 处理单个 (head, query) 对** |

---

## 11. 总结

本 kernel 的网格划分策略可以用一句话概括：

> **每个 Block 负责一个 (head, query_token) 对的完整 attention 计算，Block 内 128 个线程各自负责输出向量的一个维度，通过 tiled online softmax 沿 KV 序列方向分块处理。**

核心设计亮点：
1. **BLOCK_SIZE = head_size = 128**：线程与输出维度的一一映射，零冗余
2. **2D Grid (head, seq)**：完美的任务划分，Block 间完全独立
3. **TILE_K = 1024**：共享内存固定大小（4.25KB），适配 Orin 的 SM 限制
4. **Online Softmax**：避免两遍扫描，减少 KV 重复读取
5. **float4 向量化**：Q·K 计算吞吐量提升 4×（128-bit 全带宽利用）
6. **V 读取 coalesced**：128 线程同时读 V 的 128 维，一次 256B 事务
7. **循环展开 ×8**：隐藏指令延迟，提高 ILP（指令级并行度）

---

## 附录 A：Flash Attention 结合 Online Softmax 的计算过程详解

### A.1 标准 Attention 的问题

标准 Self-Attention 的计算公式为：

$$O = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d}}\right) V$$

其中 $\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$。为保证数值稳定性，通常使用 **safe softmax**：

$$\text{softmax}(x_i) = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}, \quad m = \max_j(x_j)$$

标准实现需要 **三遍扫描**（Three-Pass）：

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    标准 Attention 的三遍扫描                                  │
│                                                                              │
│   S = Q·K^T / √d        (score 矩阵, 大小 seq_len × kv_len)                │
│       ↓                                                                      │
│   Pass 1 (求max): m = max(S[i,:])         需要读一遍 S，大小 O(N²)          │
│       ↓                                                                      │
│   Pass 2 (求sum): l = Σ exp(S[i,j] - m)  需要再读一遍 S，大小 O(N²)        │
│       ↓                                                                      │
│   Pass 3 (加权):  O[i] = Σ (exp(S[i,j] - m) / l) × V[j]                   │
│                                           需要第三遍读 S + 读 V             │
│                                                                              │
│   ❌ 问题：需要 O(N²) 的显存存储完整 S 矩阵                                  │
│   ❌ 问题：S 矩阵需要被反复读取三次，HBM 带宽成为瓶颈                          │
└──────────────────────────────────────────────────────────────────────────────┘
```

### A.2 Online Softmax 的核心思想

Online Softmax（Milakov & Gimelshein, 2018）将三遍扫描合并为 **单遍扫描 + 增量修正**，其核心公式如下：

处理到第 $j$ 个元素时，维护三个状态变量：
- $m^{(j)}$：前 $j$ 个元素的最大值
- $l^{(j)}$：以 $m^{(j)}$ 为基准的 exp 求和
- $O^{(j)}$：以 $m^{(j)}$ 为基准的加权 V 累加

**递推关系**：

$$m^{(j)} = \max\!\left(m^{(j-1)},\; \tilde{m}^{(j)}\right)$$

$$l^{(j)} = e^{m^{(j-1)} - m^{(j)}} \cdot l^{(j-1)} + e^{\tilde{m}^{(j)} - m^{(j)}} \cdot \tilde{l}^{(j)}$$

$$O^{(j)} = e^{m^{(j-1)} - m^{(j)}} \cdot O^{(j-1)} + e^{\tilde{m}^{(j)} - m^{(j)}} \cdot \tilde{O}^{(j)}$$

其中带 $\tilde{}$ 的量表示当前 tile 内的局部统计量。

**直觉理解**：当新 tile 中出现更大的 score 值时，之前所有累加结果都需要乘以一个 **修正因子** $e^{m_{\text{old}} - m_{\text{new}}} < 1$ 来"缩小"，保持与新基准一致。

### A.3 Flash Attention + Online Softmax 完整计算过程图

以单个 query 向量 $q$ 对长度为 $N$ 的 KV 序列计算 Attention 为例，KV 被分成 $T$ 个 tile：

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║          Flash Attention + Online Softmax 单 Query 计算全流程                    ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  q: [1 × d]  (一个 query 向量, d=128)                                           ║
║  K: [N × d]  (N 个 key 向量)  → 分成 T 个 tile, 每个 TILE_K=1024                ║
║  V: [N × d]  (N 个 value 向量) → 与 K 相同的 tile 划分                           ║
║                                                                                  ║
║  初始化: m⁽⁰⁾ = -∞,  l⁽⁰⁾ = 0,  O⁽⁰⁾ = 0⃗                                     ║
║                                                                                  ║
║  ┌─────────────── Tile 1: K₁[TILE_K × d], V₁[TILE_K × d] ──────────────────┐  ║
║  │                                                                            │  ║
║  │  ① 计算局部 score:  s₁ = q · K₁ᵀ / √d          → [1 × TILE_K]           │  ║
║  │                                                                            │  ║
║  │  ② 求局部最大值:    m̃₁ = max(s₁)                                          │  ║
║  │                                                                            │  ║
║  │  ③ 更新全局最大值:  m⁽¹⁾ = max(m⁽⁰⁾, m̃₁)                                  │  ║
║  │                     = max(-∞, m̃₁) = m̃₁                                    │  ║
║  │                                                                            │  ║
║  │  ④ 计算 softmax 权重: p₁[k] = exp(s₁[k] - m⁽¹⁾)   (flush-to-zero)       │  ║
║  │                                                                            │  ║
║  │  ⑤ 局部求和:        l̃₁ = Σ p₁[k]                                          │  ║
║  │                                                                            │  ║
║  │  ⑥ 修正历史: correction = exp(m⁽⁰⁾ - m⁽¹⁾) = exp(-∞) = 0                 │  ║
║  │              O⁽¹⁾ = 0 × O⁽⁰⁾ + p₁ · V₁     (历史为零，直接累加)           │  ║
║  │              l⁽¹⁾ = 0 × l⁽⁰⁾ + l̃₁            = l̃₁                         │  ║
║  │                                                                            │  ║
║  └────────────────────────────────────────────────────────────────────────────┘  ║
║                              │                                                   ║
║                              ▼                                                   ║
║  ┌─────────────── Tile 2: K₂[TILE_K × d], V₂[TILE_K × d] ──────────────────┐  ║
║  │                                                                            │  ║
║  │  ① 计算局部 score:  s₂ = q · K₂ᵀ / √d          → [1 × TILE_K]           │  ║
║  │                                                                            │  ║
║  │  ② 求局部最大值:    m̃₂ = max(s₂)                                          │  ║
║  │                                                                            │  ║
║  │  ③ 更新全局最大值:  m⁽²⁾ = max(m⁽¹⁾, m̃₂)                                  │  ║
║  │                     假设 m̃₂ > m⁽¹⁾ → m⁽²⁾ = m̃₂   (最大值更新了!!)         │  ║
║  │                                                                            │  ║
║  │  ④ 计算 softmax 权重: p₂[k] = exp(s₂[k] - m⁽²⁾)                          │  ║
║  │                                                                            │  ║
║  │  ⑤ 局部求和:        l̃₂ = Σ p₂[k]                                          │  ║
║  │                                                                            │  ║
║  │  ⑥ 修正历史:                                                               │  ║
║  │     correction = exp(m⁽¹⁾ - m⁽²⁾) < 1    ← 关键！缩小之前的累加值         │  ║
║  │     ┌──────────────────────────────────────────────────────────────────┐   │  ║
║  │     │             Online Softmax 修正示意                              │   │  ║
║  │     │                                                                  │   │  ║
║  │     │  O⁽¹⁾ 的每个分量都是基于旧基准 m⁽¹⁾ 计算的:                     │   │  ║
║  │     │    O⁽¹⁾ = Σₖ exp(s₁[k] - m⁽¹⁾) × V₁[k]                       │   │  ║
║  │     │                                                                  │   │  ║
║  │     │  现在基准变成了 m⁽²⁾ > m⁽¹⁾，需要统一：                         │   │  ║
║  │     │    exp(s₁[k] - m⁽¹⁾) → exp(s₁[k] - m⁽²⁾)                      │   │  ║
║  │     │                       = exp(s₁[k] - m⁽¹⁾) × exp(m⁽¹⁾ - m⁽²⁾)  │   │  ║
║  │     │                       = old_weight × correction                  │   │  ║
║  │     │                                                                  │   │  ║
║  │     │  所以整个 O⁽¹⁾ 乘以 correction 即可完成基准统一                  │   │  ║
║  │     └──────────────────────────────────────────────────────────────────┘   │  ║
║  │                                                                            │  ║
║  │     O⁽²⁾ = correction × O⁽¹⁾ + p₂ · V₂                                  │  ║
║  │     l⁽²⁾ = correction × l⁽¹⁾ + l̃₂                                        │  ║
║  │                                                                            │  ║
║  └────────────────────────────────────────────────────────────────────────────┘  ║
║                              │                                                   ║
║                              ▼                                                   ║
║                           ......                                                 ║
║                              │                                                   ║
║                              ▼                                                   ║
║  ┌─────────────── Tile T (最后一个 tile) ────────────────────────────────────┐  ║
║  │                                                                            │  ║
║  │  ① ~ ⑥ 同上                                                               │  ║
║  │                                                                            │  ║
║  │  得到: m⁽ᵀ⁾ = 全局最大值, l⁽ᵀ⁾ = 全局 exp sum, O⁽ᵀ⁾ = 未归一化累加值    │  ║
║  │                                                                            │  ║
║  └────────────────────────────────────────────────────────────────────────────┘  ║
║                              │                                                   ║
║                              ▼                                                   ║
║  ┌────────────────────── 最终归一化 ─────────────────────────────────────────┐  ║
║  │                                                                            │  ║
║  │  O_final = O⁽ᵀ⁾ / l⁽ᵀ⁾                                                   │  ║
║  │                                                                            │  ║
║  │  此时 O_final 与标准 Attention 结果完全等价 ✅                              │  ║
║  │                                                                            │  ║
║  └────────────────────────────────────────────────────────────────────────────┘  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

### A.4 Online Softmax 修正因子的数值稳定性保证

下图展示了 3 个 tile 处理过程中状态变量的演变：

```
                   Tile 1              Tile 2              Tile 3
              ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
scores:       │ max=5.0     │    │ max=8.0     │    │ max=6.0     │
              └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
                     │                  │                  │
                     ▼                  ▼                  ▼
m (全局max):    m⁽¹⁾=5.0          m⁽²⁾=8.0          m⁽³⁾=8.0
                                  (更新! 8>5)        (不变, 6<8)
                     │                  │                  │
                     ▼                  ▼                  ▼
correction:     exp(-∞-5)=0       exp(5-8)=0.050     exp(8-8)=1.0
                                  ↑ 缩小历史值!       ↑ 无需修正
                     │                  │                  │
                     ▼                  ▼                  ▼
O 累加值:       O⁽¹⁾(基准5)     O⁽²⁾(基准8)        O⁽³⁾(基准8)
               = p₁·V₁         = 0.050×O⁽¹⁾        = 1.0×O⁽²⁾
                                  + p₂·V₂             + p₃·V₃
                     │                  │                  │
                     ▼                  ▼                  ▼
l 累加值:       l⁽¹⁾(基准5)     l⁽²⁾(基准8)        l⁽³⁾(基准8)
               = Σexp(s-5)     = 0.050×l⁽¹⁾        = 1.0×l⁽²⁾
                                  + Σexp(s-8)          + Σexp(s-8)

                                                         │
                                                         ▼
                                                  O_final = O⁽³⁾/l⁽³⁾

关键观察：
  • 所有 exp 参数都是 (score - m_global) ≤ 0，保证 exp 不溢出
  • correction factor < 1 时缩小历史值，等价于将所有 exp 统一到全局 max 基准
  • 最终结果与标准三遍 softmax 在数学上完全等价
```

### A.5 对照代码的变量映射

| 数学符号 | 代码变量 | 含义 |
|----------|----------|------|
| $m^{(j-1)}$ | `row_max` | 截至前一个 tile 的全局最大值 |
| $\tilde{m}^{(j)}$ | `m_j` | 当前 tile 内的最大 score |
| $m^{(j)}$ | `m_new` | 更新后的全局最大值 `fmaxf(row_max, m_j)` |
| $e^{m^{(j-1)} - m^{(j)}}$ | `correction` | `expf(row_max - m_new)` |
| $\tilde{l}^{(j)}$ | `l_j` | 当前 tile 内 `exp(score - m_new)` 的总和 |
| $l^{(j)}$ | `row_sum` | `correction * row_sum + l_j` |
| $O^{(j)}[tid]$ | `acc_o` | `correction * acc_o + Σ s_scores[k] * V[k,tid]` |

---

## 附录 B：Q、K、V 的分块划分与逐步计算过程详解

### B.1 全局视角：Attention 矩阵的分块结构

Flash Attention 的核心思想是**不物化完整的 Score 矩阵** $S = QK^T$，而是分块计算、流式累加。在本 kernel 中：

- **Q 不分块**：每个 CUDA Block 只处理**一个** query token（一行 Q）
- **K/V 沿序列维度分块**：每次加载 TILE_K=1024 个 KV 位置
- **Score 矩阵按行分块**：每次只计算 $S$ 的一行中连续 TILE_K 个元素

```
      完整 Attention Score 矩阵 S = Q·K^T/√d  (概念上存在，但从不物化)
      ─────────────────────────────────────────────────────────────────
                                  kv_len (例: 4096)
                    ┌─────────┬─────────┬─────────┬─────────┐
                    │         │         │         │         │
         seq_len    │  Tile 0 │  Tile 1 │  Tile 2 │  Tile 3 │
         (例:512)   │ K₀..₁₀₂₃│K₁₀₂₄..₂₀₄₇│K₂₀₄₈..₃₀₇₁│K₃₀₇₂..₄₀₉₅│
                    │         │         │         │         │
                    │         │         │         │         │
    q₀ ─────────── │─ s₀,₀₋₁₀₂₃─│─ s₀,₁₀₂₄₋₂₀₄₇─│─ s₀,₂₀₄₈₋₃₀₇₁─│─...─│
    q₁ ─────────── │─ s₁,₀₋₁₀₂₃─│─ s₁,₁₀₂₄₋₂₀₄₇─│─...─│    │
    q₂ ─────────── │─ s₂,₀₋₁₀₂₃─│─...─│    │    │
    ...             │         │         │         │         │
    q₅₁₁ ───────── │─ s₅₁₁,₀₋₁₀₂₃─│─...─│─...─│─...─│
                    └─────────┴─────────┴─────────┴─────────┘

    ⬆ 因果掩码：q_i 只看到 kv_pos ≤ start_pos + i 的位置
      所以靠上方的行有效列更少（三角形区域）

    📌 每个 CUDA Block 只负责上图中的 **一行**
       在该行内，按 Tile 0 → Tile 1 → ... 顺序从左到右扫描
```

### B.2 单个 Block 内 Q、K、V 的分块方式

以 Block(head=h, seq_idx=s) 为例，`kv_len = 3000`，`head_size = 128`：

```
                    Q、K、V 的分块与形状
══════════════════════════════════════════════════════════════════

  Q (Query): 不分块，一个 Block 只有一个 query 向量
  ┌──────────────────────────────────────────────┐
  │  q = Q[s, h, :]    形状: [1 × 128]          │  ← 加载到 Shared Memory
  │  [q₀, q₁, q₂, ..., q₁₂₇]                   │     所有线程共享
  └──────────────────────────────────────────────┘

  K (Key): 沿序列维度分成 T 个 tile      V (Value): 与 K 完全相同的分块方式
  ┌─────────────────┐                      ┌─────────────────┐
  │ K_tile0          │ kv_pos [0, 1024)    │ V_tile0          │
  │ [1024 × 128]     │                      │ [1024 × 128]     │
  ├─────────────────┤                      ├─────────────────┤
  │ K_tile1          │ kv_pos [1024, 2048) │ V_tile1          │
  │ [1024 × 128]     │                      │ [1024 × 128]     │
  ├─────────────────┤                      ├─────────────────┤
  │ K_tile2          │ kv_pos [2048, 3000) │ V_tile2          │
  │ [952 × 128]      │ (尾部不足 1024)      │ [952 × 128]      │
  └─────────────────┘                      └─────────────────┘
       ↑ 从 Global Memory (KV Cache) 直接读取，不加载到 Shared Memory
```

### B.3 逐 Tile 计算详图

下图展示**每一步**中 Q、K、V 各自的角色和数据流：

```
═══════════════════════════════════════════════════════════════════════════
            Tile 0:  kv_pos = [0, 1024)
═══════════════════════════════════════════════════════════════════════════

  ┌─ Step 1: Q·K → Scores ──────────────────────────────────────────────┐
  │                                                                      │
  │    s_query (Shared Mem)          K_tile0 (Global Mem, KV Cache)      │
  │    [1 × 128] (half)             [1024 × 128] (half)                 │
  │                                                                      │
  │    ┌─────────────┐              ┌─────────────┐                      │
  │    │q₀ q₁...q₁₂₇│       ×      │k₀₀ k₀₁...k₀,₁₂₇│ ← kv_pos=0   │
  │    └─────────────┘              │k₁₀ k₁₁...k₁,₁₂₇│ ← kv_pos=1   │
  │          ↑                       │      ...        │                │
  │    128 线程各处理                │k₁₀₂₃,₀...k₁₀₂₃,₁₂₇│ ← kv_pos=1023│
  │    部分 KV 位置                  └─────────────┘                      │
  │    (tid=i 处理 k_idx=i, i+128, i+256, ...)                          │
  │                                                                      │
  │    每次点积：线程读 s_query 全部 128 维 (shared mem broadcast)        │
  │              + 读 K[kv_pos] 全部 128 维 (global mem, float4 向量化)   │
  │              → 1 个 float score                                      │
  │                     ↓                                                │
  │    s_scores[0..1023] (Shared Mem, float)                             │
  │    ┌────────────────────────────────────────┐                        │
  │    │ s[0]  s[1]  s[2] ... s[1023]           │                        │
  │    └────────────────────────────────────────┘                        │
  └──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌─ Step 2-5: Online Softmax (Block 协作) ─────────────────────────────┐
  │                                                                      │
  │    s_scores[0..1023]                                                 │
  │    ┌────────────────────────────────────────┐                        │
  │    │ 5.2  3.1  7.8  4.5 ... 6.3             │  原始 scores           │
  │    └───────────────┬────────────────────────┘                        │
  │                    │                                                  │
  │     Block Reduce Max → m̃₁ = 7.8                                     │
  │     m_new = max(row_max=-∞, 7.8) = 7.8                              │
  │                    │                                                  │
  │                    ▼                                                  │
  │    ┌────────────────────────────────────────┐                        │
  │    │exp(5.2-7.8) exp(3.1-7.8) ... exp(...)  │                        │
  │    │ = 0.074      0.0093      ...           │  softmax weights       │
  │    └───────────────┬────────────────────────┘                        │
  │                    │                                                  │
  │     Block Reduce Sum → l̃₁ = Σ weights                               │
  │                    │                                                  │
  │     correction = exp(-∞ - 7.8) = 0   ← 第一个 tile，无历史           │
  │     acc_o *= 0   → acc_o 归零                                        │
  │     row_sum = 0 × 0 + l̃₁ = l̃₁                                      │
  └──────────────────┬───────────────────────────────────────────────────┘
                     │
                     ▼
  ┌─ Step 7: Score × V → 累加输出 ──────────────────────────────────────┐
  │                                                                      │
  │    s_scores (softmax weights)      V_tile0 (Global Mem, KV Cache)   │
  │    [1024] (float, Shared Mem)      [1024 × 128] (half)              │
  │                                                                      │
  │    ┌──────────────────┐            ┌─────────────────────────┐       │
  │    │ w₀ w₁ w₂...w₁₀₂₃│            │v₀₀  v₀₁  v₀₂ ... v₀,₁₂₇│     │
  │    └──────────────────┘            │v₁₀  v₁₁  v₁₂ ... v₁,₁₂₇│     │
  │           ↑                         │         ...              │     │
  │    所有线程广播读取同一个 w_k       │v₁₀₂₃,₀ ... v₁₀₂₃,₁₂₇   │     │
  │                                    └─────────────────────────┘       │
  │                                           ↑                          │
  │                                    128 线程同时读同一行的 128 维      │
  │                                    (coalesced, 256 bytes/行)          │
  │                                                                      │
  │    Thread tid 的计算:                                                │
  │      for k = 0 to 1023:                                              │
  │        acc_o += s_scores[k] × V[tile_start+k, kv_head, tid]         │
  │                 ↑广播读取       ↑ 每线程只读自己负责的第 tid 维       │
  │                                                                      │
  │    循环展开×8 (每次处理 8 个 k):                                      │
  │    ┌──────────────────────────────────────────────┐                  │
  │    │ acc_o += w₀×V[0,tid] + w₁×V[1,tid] + ...    │                  │
  │    │        + w₆×V[6,tid] + w₇×V[7,tid]          │  ← 8× fmaf      │
  │    │ acc_o += w₈×V[8,tid] + ...                   │                  │
  │    │ ...                                          │                  │
  │    └──────────────────────────────────────────────┘                  │
  └──────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════
            Tile 1:  kv_pos = [1024, 2048)
═══════════════════════════════════════════════════════════════════════════

  ┌─ Step 1: Q·K → Scores ──────────────────────────────────────────────┐
  │                                                                      │
  │    s_query (Shared Mem, 不变)    K_tile1 (Global Mem, KV Cache)      │
  │    [1 × 128]                     [1024 × 128]                        │
  │    ┌─────────────┐              ┌─────────────┐                      │
  │    │q₀ q₁...q₁₂₇│       ×      │K[1024, :]   │                      │
  │    └─────────────┘              │K[1025, :]   │                      │
  │    (同一个 q，重复使用)          │    ...       │                      │
  │                                  │K[2047, :]   │                      │
  │                                  └─────────────┘                      │
  │                     ↓                                                │
  │    s_scores[0..1023] ← 覆写为新 tile 的 scores                      │
  └──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌─ Step 2-5: Online Softmax ──────────────────────────────────────────┐
  │                                                                      │
  │    m̃₂ = max(s_scores)                                                │
  │    m_new = max(row_max=m̃₁, m̃₂)    ← 可能更新全局 max!               │
  │                                                                      │
  │    s_scores[k] = exp(s_scores[k] - m_new)                           │
  │    l̃₂ = Σ s_scores[k]                                                │
  │                                                                      │
  │    ┌──────────────────────────────────────────────┐                  │
  │    │  ⚡ 关键修正步骤:                             │                  │
  │    │  correction = exp(m̃₁ - m_new)                │                  │
  │    │  acc_o = correction × acc_o   ← 缩放 Tile 0 的累加值  │         │
  │    │  row_sum = correction × row_sum + l̃₂                  │         │
  │    └──────────────────────────────────────────────┘                  │
  └──────────────────────┬───────────────────────────────────────────────┘
                         │
                         ▼
  ┌─ Step 7: Score × V → 累加输出 ──────────────────────────────────────┐
  │                                                                      │
  │    acc_o += Σ_k s_scores[k] × V[1024+k, kv_head, tid]               │
  │              ↑ 注意：acc_o 已经被 correction 缩放过                   │
  │              现在基准统一为 m_new                                      │
  └──────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════
            Tile 2 (最后):  kv_pos = [2048, 3000)    tile_len = 952
═══════════════════════════════════════════════════════════════════════════

  ┌─ 同上流程，但 tile_len=952 < TILE_K=1024 ──────────────────────────┐
  │                                                                      │
  │  128 线程中每线程处理 ⌈952/128⌉ = 8 个 KV 位置 (最后一轮部分线程空闲)│
  │  s_scores[0..951] 有效，[952..1023] 不使用                           │
  │  其余流程完全相同                                                     │
  └──────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════
            最终输出
═══════════════════════════════════════════════════════════════════════════

  O[s, h, tid] = acc_o / row_sum    ← 每个线程写 1 个 half 到 Global Memory
                                      128 线程写 128 维 = 256 bytes, coalesced
```

### B.4 Q、K、V 分块方式综合对比

```
┌────────┬──────────────┬─────────────┬──────────────────────────────────────┐
│ 张量   │ 形状         │ 分块方式    │ 说明                                 │
├────────┼──────────────┼─────────────┼──────────────────────────────────────┤
│ Q      │ [1 × 128]    │ ❌ 不分块    │ 一个 Block 只处理 1 个 query token   │
│        │              │             │ 加载到 Shared Memory，所有线程共享    │
│        │              │             │ 每个 tile 重复使用同一个 q             │
├────────┼──────────────┼─────────────┼──────────────────────────────────────┤
│ K      │ [N × 128]    │ ✅ 按行分块  │ 沿 KV 序列维度切分                   │
│        │              │ (TILE_K=1024)│ 每次从 KV Cache 读 1024 行           │
│        │              │             │ 用于计算 Q·K score → s_scores[]      │
│        │              │             │ 每行通过 float4 向量化读取(16B/行)     │
├────────┼──────────────┼─────────────┼──────────────────────────────────────┤
│ V      │ [N × 128]    │ ✅ 按行分块  │ 与 K 完全相同的 tile 划分            │
│        │              │ (TILE_K=1024)│ 但读取方式不同：                      │
│        │              │             │ 128 线程同时读同一行的 128 维          │
│        │              │             │ (coalesced, 每行 256B)                │
│        │              │             │ 按列拆分给各线程 (tid→第tid维)        │
├────────┼──────────────┼─────────────┼──────────────────────────────────────┤
│ S      │ [1 × N]      │ ✅ 分段存储  │ 每次只存 s_scores[TILE_K] 在 smem   │
│ (score)│              │ (TILE_K=1024)│ 先写入 Q·K score，再覆写为           │
│        │              │             │ exp(score-max) 的 softmax 权重        │
├────────┼──────────────┼─────────────┼──────────────────────────────────────┤
│ O      │ [1 × 128]    │ ❌ 不分块    │ 每线程 1 个 float 寄存器 (acc_o)     │
│ (输出) │              │             │ 跨 tile 增量累加 + online softmax 修正│
│        │              │             │ 最后归一化写入 Global Memory           │
└────────┴──────────────┴─────────────┴──────────────────────────────────────┘
```

### B.5 数据在存储层级中的位置

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          存储层级与数据放置                                   │
│                                                                             │
│  ┌──────────────────── 寄存器 (Register File) ────────────────────────┐    │
│  │  • acc_o    : Thread tid 负责的输出维度的累加值 (float)             │    │
│  │  • row_max  : 截至当前 tile 的全局最大 score (float)               │    │
│  │  • row_sum  : 截至当前 tile 的 exp sum (float)                     │    │
│  │  • tile_max_local, tile_sum_local : tile 内局部统计量              │    │
│  │  每线程 ~40 个寄存器                                                │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                              ↕ 极快 (~0 cycle latency)                      │
│  ┌──────────────────── 共享内存 (Shared Memory, 4.25KB) ─────────────┐    │
│  │  • s_query[128]  : Q 向量 (half, 256B)  — 加载一次，所有 tile 复用│    │
│  │  • s_scores[1024] : 当前 tile 的 scores/weights (float, 4KB)      │    │
│  │    - Q·K 计算后存 score                                            │    │
│  │    - softmax 计算后覆写为 exp weights                              │    │
│  │    - V 累加时读取 weights                                          │    │
│  │    → 每个 tile 被覆写一次                                          │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                              ↕ ~20-30 cycle latency                         │
│  ┌──────────────────── 全局内存 / L2 Cache (KV Cache) ───────────────┐    │
│  │  • K_cache[N × kv_dim] : 所有 key 向量，每 tile 读 1024 行        │    │
│  │  • V_cache[N × kv_dim] : 所有 value 向量，每 tile 读 1024 行      │    │
│  │  • Q[seq_len × dim]    : Query，只在初始化时读 1 次               │    │
│  │  • O[seq_len × dim]    : Output，只在最后写 1 次                  │    │
│  │  GQA: 同 kv_head 的 4 个 Block 通过 L2 cache 自然复用 K/V          │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                              ↕ ~200-400 cycle latency                       │
│  ┌──────────────────── HBM (显存) ───────────────────────────────────┐    │
│  │  物理存储 KV Cache，通过 L2 cache line 按需加载                    │    │
│  │  Flash Attention 的核心收益：避免物化 O(N²) 的 S 矩阵到此层级       │    │
│  └────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 附录 C：Qwen3-8B Prefill 阶段 Q/K/V 在 Grid、Block、Thread 中的划分详解

### C.1 Qwen3-8B 的 Attention 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `head_num` | 32 | Query 注意力头数 |
| `kv_head_num` | 8 | KV 注意力头数 (GQA) |
| `kv_mul` | 4 | `head_num / kv_head_num`，每 4 个 Q head 共享 1 个 KV head |
| `head_size` | 128 | 每个头的维度 |
| `dim` | 4096 | `head_num × head_size = 32 × 128` |
| `kv_dim` | 1024 | `kv_head_num × head_size = 8 × 128` |
| `BLOCK_SIZE` | 128 | 每个 CUDA Block 的线程数 |
| `TILE_K` | 1024 | KV 序列分块大小 |

**输入张量的逻辑形状与物理布局**：

```
Q:  逻辑形状 [seq_len, 32, 128]  →  物理布局 [seq_len, 4096]  (32 个头连续存储)
K:  逻辑形状 [N, 8, 128]         →  物理布局 [N, 1024]        (KV Cache, N=总序列长度)
V:  逻辑形状 [N, 8, 128]         →  物理布局 [N, 1024]        (KV Cache, N=总序列长度)
O:  逻辑形状 [seq_len, 32, 128]  →  物理布局 [seq_len, 4096]
```

### C.2 Grid 层面：Q/K/V 的全局任务划分

```cpp
dim3 grid(head_num, seq_len);   // grid(32, seq_len)
dim3 block(BLOCK_SIZE);          // block(128)
```

**Grid 的二维结构将 Q 张量的前两维 `[seq_len, 32]` 完全映射到 Block 网格**：

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    Grid 层面的 Q / K / V 划分                                ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  Q: [seq_len, 32, 128]                                                       ║
║  ─────────────────────                                                        ║
║  Grid 的 X 轴 = head (0..31)       → 选择 Q 的第 2 维 (哪个注意力头)         ║
║  Grid 的 Y 轴 = seq_idx (0..seq_len-1) → 选择 Q 的第 1 维 (哪个 token)       ║
║                                                                               ║
║  即: Block(head=h, seq_idx=s) 负责 Q[s, h, :] — 一个 [128] 维的向量          ║
║                                                                               ║
║                        head 维度 (32 个 Q head)                               ║
║        h=0    h=1    h=2    h=3   ...  h=28   h=29   h=30   h=31             ║
║       ┌──────┬──────┬──────┬──────┬───┬──────┬──────┬──────┬──────┐          ║
║  s=0  │Q[0,0]│Q[0,1]│Q[0,2]│Q[0,3]│...│Q[0,28]│Q[0,29]│Q[0,30]│Q[0,31]│   ║
║       ├──────┼──────┼──────┼──────┼───┼──────┼──────┼──────┼──────┤          ║
║  s=1  │Q[1,0]│Q[1,1]│Q[1,2]│Q[1,3]│...│      │      │      │      │         ║
║       ├──────┼──────┼──────┼──────┼───┼──────┼──────┼──────┼──────┤          ║
║  s=2  │Q[2,0]│Q[2,1]│      │      │   │      │      │      │      │         ║
║       ├──────┼──────┼──────┼──────┼───┼──────┼──────┼──────┼──────┤          ║
║  ...  │ ...  │ ...  │      │      │   │      │      │      │      │         ║
║       └──────┴──────┴──────┴──────┴───┴──────┴──────┴──────┴──────┘          ║
║                                                                               ║
║       每个格子 = 1 个 CUDA Block (128 线程)                                   ║
║       每个格子处理 Q 的一个 [128] 维向量                                       ║
║       总 Block 数 = 32 × seq_len                                              ║
║                                                                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  K: [N, 8, 128]   V: [N, 8, 128]                                             ║
║  ────────────────────────────────                                             ║
║  K/V 不在 Grid 维度上划分!                                                     ║
║  每个 Block 通过 GQA 映射确定自己读取哪个 KV head:                              ║
║                                                                               ║
║     kv_head = head / kv_mul = head / 4                                        ║
║                                                                               ║
║  Q head 0,1,2,3   → KV head 0    (4个Block共享同一份K/V)                     ║
║  Q head 4,5,6,7   → KV head 1                                                ║
║  Q head 8,9,10,11 → KV head 2                                                ║
║  ...                                                                          ║
║  Q head 28,29,30,31 → KV head 7                                              ║
║                                                                               ║
║  K/V 沿序列维度 (N) 由每个 Block 内部通过 TILE_K=1024 分块遍历                  ║
║  (不在 Grid 层面划分，而是在 Block 内部的 for 循环中逐 tile 处理)                ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

**GQA 映射的 Grid 层面视图**：

```
                    32 个 Q head 被分成 8 组，每组共享一个 KV head
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                                                                         │
  │  ┌── KV head 0 ──┐ ┌── KV head 1 ──┐         ┌── KV head 7 ──┐       │
  │  │ Q head 0      │ │ Q head 4      │         │ Q head 28     │       │
  │  │ Q head 1      │ │ Q head 5      │  . . .  │ Q head 29     │       │
  │  │ Q head 2      │ │ Q head 6      │         │ Q head 30     │       │
  │  │ Q head 3      │ │ Q head 7      │         │ Q head 31     │       │
  │  └───────────────┘ └───────────────┘         └───────────────┘       │
  │         ↓                  ↓                          ↓                │
  │  K[*, 0, :128]      K[*, 1, :128]            K[*, 7, :128]           │
  │  V[*, 0, :128]      V[*, 1, :128]            V[*, 7, :128]           │
  │                                                                         │
  │  同组内 4 个 Block 各自有不同的 Q 向量，但读取完全相同的 K/V 数据          │
  │  → 通过 GPU L2 Cache 自然实现数据复用，无需额外同步                       │
  └─────────────────────────────────────────────────────────────────────────┘
```

### C.3 Block 层面：单个 Block 内 Q/K/V 的处理方式

以 `Block(head=5, seq_idx=3)` 为例，`start_pos=100`，因此 `kv_len = 100 + 3 + 1 = 104`:

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║  Block(head=5, seq_idx=3)    kv_head=5/4=1   kv_len=104   128 threads       ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ┌─── Q 在 Block 内的处理 ─────────────────────────────────────────────────┐ ║
║  │                                                                          │ ║
║  │  Q[3, 5, :] = Q[seq_idx=3, head=5, 0..127]                             │ ║
║  │                                                                          │ ║
║  │  物理地址: Q + 3 × 4096 + 5 × 128 = Q + 12288 + 640 = Q + 12928       │ ║
║  │            ↑ seq_idx×dim  ↑ head×head_size                               │ ║
║  │                                                                          │ ║
║  │  加载方式: 128 线程 1:1 映射，tid=i 加载 Q[3,5,i]                       │ ║
║  │                                                                          │ ║
║  │  Global Memory          Shared Memory                                    │ ║
║  │  Q[3,5,0..127]    →    s_query[0..127]                                  │ ║
║  │  [128 half = 256B]      [128 half = 256B]                                │ ║
║  │                                                                          │ ║
║  │  ⚡ 加载一次后在整个 kernel 生命周期内反复使用（所有 tile 共享）           │ ║
║  └──────────────────────────────────────────────────────────────────────────┘ ║
║                                                                               ║
║  ┌─── K 在 Block 内的处理 ─────────────────────────────────────────────────┐ ║
║  │                                                                          │ ║
║  │  需要访问: K[0..103, kv_head=1, 0..127]   (kv_len=104 个 key 向量)      │ ║
║  │                                                                          │ ║
║  │  物理地址: K_cache + kv_pos × 1024 + 1 × 128                            │ ║
║  │                     ↑ pos×kv_dim    ↑ kv_head×head_size                   │ ║
║  │                                                                          │ ║
║  │  分块方式: 沿 kv_pos 维度按 TILE_K=1024 分块                             │ ║
║  │  本例 kv_len=104 < 1024，只有 1 个 tile:                                 │ ║
║  │    Tile 0: K[0..103, 1, :] — 104 个 key 向量，每个 128 维                │ ║
║  │                                                                          │ ║
║  │  ❌ K 不加载到 Shared Memory — 直接从 Global Memory 通过 __ldg 读取       │ ║
║  │     (利用 L1/L2 只读缓存)                                                │ ║
║  └──────────────────────────────────────────────────────────────────────────┘ ║
║                                                                               ║
║  ┌─── V 在 Block 内的处理 ─────────────────────────────────────────────────┐ ║
║  │                                                                          │ ║
║  │  需要访问: V[0..103, kv_head=1, 0..127]   (kv_len=104 个 value 向量)    │ ║
║  │                                                                          │ ║
║  │  物理地址: V_cache + kv_pos × 1024 + 1 × 128 + tid                      │ ║
║  │                                                                          │ ║
║  │  分块方式: 与 K 完全相同的 tile 划分                                      │ ║
║  │  但访问模式不同: 每个线程只读自己负责的第 tid 维                           │ ║
║  │                                                                          │ ║
║  │  ❌ V 不加载到 Shared Memory — 直接从 Global Memory 逐元素读取            │ ║
║  └──────────────────────────────────────────────────────────────────────────┘ ║
║                                                                               ║
║  ┌─── Scores 在 Block 内的处理 ────────────────────────────────────────────┐ ║
║  │                                                                          │ ║
║  │  s_scores[0..TILE_K-1] — Shared Memory, float                           │ ║
║  │  每个 tile 内存放 Q·K 的点积结果，后覆写为 softmax 权重                   │ ║
║  │  大小固定 = TILE_K × sizeof(float) = 1024 × 4 = 4096 bytes              │ ║
║  └──────────────────────────────────────────────────────────────────────────┘ ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

**Block 内 kv_len 较大时的 K/V 分块示例** (kv_len=3000):

```
  K/V 沿序列维度的分块 (Block 内部 for 循环):

  KV Cache 序列维度 [0 ────────────────────────────────── 2999]
       │← ─ ─ ─ Tile 0 ─ ─ ─ →│← ─ ─ Tile 1 ─ ─ →│← Tile 2 →│
       │   kv_pos [0, 1024)    │  [1024, 2048)      │[2048,3000)│
       │   1024 个 K/V 向量     │  1024 个 K/V 向量   │ 952 个    │
       │  s_scores[0..1023]    │  s_scores[0..1023] │s_scores[0..951]│

  每个 tile 的处理流程:
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │ 128线程协作   │→│ 128线程协作   │→│ 每线程独立    │→│ 更新状态     │
  │ Q·K→scores   │  │ online softmax│  │ score×V累加  │  │ row_max,sum  │
  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
         ↑                  ↑                 ↑
    读K from Global    读写 s_scores      读V from Global
    + s_query(smem)    (Shared Mem)       + s_scores(smem)
```

### C.4 Thread 层面：128 个线程各自处理 Q/K/V 的哪些元素

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║           Thread 层面的 Q / K / V 元素划分 (BLOCK_SIZE=128, head_size=128)   ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ◆ 阶段一: Q 加载 (每线程加载 1 个元素)                                       ║
║  ─────────────────────────────────────────                                     ║
║                                                                               ║
║    Thread tid=0  → s_query[0]   = Q[s, h, 0]                                 ║
║    Thread tid=1  → s_query[1]   = Q[s, h, 1]                                 ║
║    Thread tid=2  → s_query[2]   = Q[s, h, 2]                                 ║
║    ...                                                                        ║
║    Thread tid=127 → s_query[127] = Q[s, h, 127]                              ║
║                                                                               ║
║    128 线程 → 128 个 half → 256 bytes → 1 次 coalesced 事务                   ║
║                                                                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ◆ 阶段二: Q·K 点积 (每线程计算 ⌈tile_len/128⌉ 个完整点积)                   ║
║  ──────────────────────────────────────────────────────────                     ║
║                                                                               ║
║  以 tile_len=1024 为例，每线程计算 1024/128 = 8 个 Q·K 点积:                  ║
║                                                                               ║
║  Thread tid=0:                                                                ║
║    k_idx=0:   score = Σ(d=0..127) s_query[d] × K[tile+0, kv_head, d]         ║
║    k_idx=128: score = Σ(d=0..127) s_query[d] × K[tile+128, kv_head, d]       ║
║    k_idx=256: score = Σ(d=0..127) s_query[d] × K[tile+256, kv_head, d]       ║
║    ... (共8个)                                                                ║
║                                                                               ║
║  Thread tid=1:                                                                ║
║    k_idx=1:   score = Σ(d=0..127) s_query[d] × K[tile+1, kv_head, d]         ║
║    k_idx=129: score = Σ(d=0..127) s_query[d] × K[tile+129, kv_head, d]       ║
║    ... (共8个)                                                                ║
║                                                                               ║
║  每个点积的计算过程 (float4 向量化, 128维=16次迭代):                           ║
║                                                                               ║
║    s_query:  [q₀q₁q₂q₃q₄q₅q₆q₇|q₈...q₁₅|...|q₁₂₀...q₁₂₇]                ║
║              ├── float4[0] ──┤├─float4[1]┤    ├─float4[15]─┤                 ║
║                                                                               ║
║    K[pos]:   [k₀k₁k₂k₃k₄k₅k₆k₇|k₈...k₁₅|...|k₁₂₀...k₁₂₇]               ║
║              ├── float4[0] ──┤├─float4[1]┤    ├─float4[15]─┤                 ║
║                                                                               ║
║    每个 float4 = 8 个 half: 解释为 4 个 half2，用 __half22float2 转换后       ║
║    fmaf 累加到 float2(acc.x, acc.y)，最终 score = (acc.x + acc.y) × scale    ║
║                                                                               ║
║  ⚠ 注意: 这里每线程是读取 s_query 的全部 128 维 (广播读共享内存)               ║
║          + 读取 K 向量的全部 128 维 (全局内存 float4 向量化)                    ║
║          → 1 个线程独立完成 1 个完整的 128 维点积                              ║
║                                                                               ║
║  结果存入: s_scores[k_idx] = score                                            ║
║                                                                               ║
║  ┌────────────────────────────────────────────────────────────────────┐       ║
║  │  s_scores[] 的线程写入映射 (tile_len=1024):                        │       ║
║  │                                                                    │       ║
║  │  index:  0   1   2  ... 127 128 129 ... 255 256 ... 1023          │       ║
║  │  writer: t0  t1  t2 ... t127 t0  t1 ... t127 t0 ... t127         │       ║
║  │          ├─── 第1轮 ────┤├─── 第2轮 ────┤├── ... ──┤├─ 第8轮 ─┤  │       ║
║  └────────────────────────────────────────────────────────────────────┘       ║
║                                                                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ◆ 阶段三: Softmax 计算 (每线程处理 ⌈tile_len/128⌉ 个 score)                 ║
║  ──────────────────────────────────────────────────────────                     ║
║                                                                               ║
║  与 Q·K 完全相同的线程分工:                                                    ║
║  Thread tid=i 处理 s_scores[i], s_scores[i+128], s_scores[i+256], ...         ║
║                                                                               ║
║  - Block Reduce Max: 4 个 Warp 各自 shuffle reduce → Thread 0 汇总            ║
║  - 每线程计算 exp(score - m_new)，覆写 s_scores[]                              ║
║  - Block Reduce Sum: 同上结构                                                  ║
║                                                                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ◆ 阶段四: Score × V 累加 (每线程负责输出的 1 个维度)                          ║
║  ──────────────────────────────────────────────────────                         ║
║                                                                               ║
║  ⚡ 这里线程的角色发生了根本性切换:                                              ║
║     阶段二/三: tid → 负责不同的 KV 位置 (k_idx = tid, tid+128, ...)            ║
║     阶段四:   tid → 负责输出向量的第 tid 个维度 (固定)                          ║
║                                                                               ║
║  Thread tid=0:  acc_o += Σ_k s_scores[k] × V[tile+k, kv_head, 0]             ║
║  Thread tid=1:  acc_o += Σ_k s_scores[k] × V[tile+k, kv_head, 1]             ║
║  Thread tid=2:  acc_o += Σ_k s_scores[k] × V[tile+k, kv_head, 2]             ║
║  ...                                                                          ║
║  Thread tid=127: acc_o += Σ_k s_scores[k] × V[tile+k, kv_head, 127]          ║
║                                                                               ║
║  V 的访问模式 (128 线程同时读 V 的同一行):                                     ║
║                                                                               ║
║    V[pos=k, kv_head=1]:                                                       ║
║    ┌─────────────────────────────────────────────────────┐                    ║
║    │ v₀  v₁  v₂  v₃  v₄ ... v₁₂₆ v₁₂₇                 │ ← 128 half        ║
║    │ ↑   ↑   ↑   ↑   ↑      ↑    ↑                      │                    ║
║    │ t0  t1  t2  t3  t4     t126  t127                   │ ← 128 线程同时读   ║
║    └─────────────────────────────────────────────────────┘                    ║
║    128 × 2B = 256 bytes → 完美 coalesced，2 个 128B 事务                      ║
║                                                                               ║
║    每线程的内循环 (展开×8):                                                    ║
║    ┌─────────────────────────────────────────────────────────┐                ║
║    │ Thread tid 读取:                                        │                ║
║    │   V[tile+0, 1, tid] ← 全局内存 stride=kv_dim=1024      │                ║
║    │   V[tile+1, 1, tid] ← v_ptr + kv_dim                   │                ║
║    │   V[tile+2, 1, tid] ← v_ptr + 2×kv_dim                 │                ║
║    │   ...                                                   │                ║
║    │   V[tile+7, 1, tid] ← v_ptr + 7×kv_dim                 │                ║
║    │                                                         │                ║
║    │ 同时从 s_scores 广播读取:                               │                ║
║    │   s_scores[0], s_scores[1], ..., s_scores[7]           │                ║
║    │   (所有 128 线程读同一个 s_scores[k] — 广播无冲突)      │                ║
║    │                                                         │                ║
║    │ 计算: acc_o = fmaf(s0,v0, fmaf(s1,v1, ... ))           │                ║
║    └─────────────────────────────────────────────────────────┘                ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

### C.5 三层划分的全景总结图

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  Q: [seq_len, 32, 128]    K: [N, 8, 128]    V: [N, 8, 128]                 │
│                                                                              │
│  ═══════════ Grid 层 (任务分配) ═══════════                                  │
│                                                                              │
│  grid(32, seq_len):                                                          │
│    blockIdx.x = head ∈ [0,31]   → 选择 Q 的哪个 head                        │
│    blockIdx.y = seq_idx ∈ [0,seq_len-1] → 选择 Q 的哪个 token               │
│    kv_head = head / 4            → 确定对应的 K/V head                       │
│                                                                              │
│  Q 划分: seq_len × 32 = 每个(token,head)对分配 1 个 Block ✅                 │
│  K 划分: Grid 层不划分 K 序列维度 (Block 内部 tiling) ❌                       │
│  V 划分: 与 K 相同 ❌                                                         │
│                                                                              │
│  ═══════════ Block 层 (数据管理) ═══════════                                 │
│                                                                              │
│  128 threads, Shared Memory 4.25KB:                                          │
│    s_query[128 half]  ← Q[seq_idx, head, :] 加载一次                         │
│    s_scores[1024 float] ← 每个 tile 覆写                                     │
│                                                                              │
│  Q: 加载到 smem 后所有 tile 复用 (只读) ✅                                    │
│  K: 按 TILE_K=1024 分块，每 tile 从 Global Memory 流式读取 ✅                 │
│  V: 与 K 相同 tile 边界，从 Global Memory 流式读取 ✅                          │
│                                                                              │
│  ═══════════ Thread 层 (计算执行) ═══════════                                │
│                                                                              │
│  ┌─────────────────────┬──────────────────────────────────────┐              │
│  │ 阶段                │ Thread tid 的职责                     │              │
│  ├─────────────────────┼──────────────────────────────────────┤              │
│  │ Q 加载              │ 加载 Q[s,h,tid] → s_query[tid]       │              │
│  │                     │ (1个元素, 2 bytes)                    │              │
│  ├─────────────────────┼──────────────────────────────────────┤              │
│  │ Q·K 点积            │ 计算 k_idx=tid,tid+128,...的完整点积  │              │
│  │                     │ 每次读 s_query 全部128维(广播)        │              │
│  │                     │ + K[kv_pos]全部128维(float4向量化)    │              │
│  │                     │ → 每线程 ⌈tile_len/128⌉ 个 score     │              │
│  ├─────────────────────┼──────────────────────────────────────┤              │
│  │ Softmax (max/sum)   │ 处理 s_scores[tid,tid+128,...] + reduce│             │
│  │                     │ 与 Q·K 相同的线程-数据映射            │              │
│  ├─────────────────────┼──────────────────────────────────────┤              │
│  │ Score×V 累加   ⚡   │ 只负责输出的第 tid 维:                │              │
│  │ (角色切换!)         │ acc_o += Σ_k s_scores[k] × V[k,tid]  │              │
│  │                     │ 读 s_scores 广播 + 读 V[:,tid] stride │              │
│  ├─────────────────────┼──────────────────────────────────────┤              │
│  │ 最终输出            │ O[s,h,tid] = acc_o / row_sum          │              │
│  │                     │ (1个元素, 2 bytes)                    │              │
│  └─────────────────────┴──────────────────────────────────────┘              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### C.6 具体数值示例：seq_len=512, start_pos=0

```
  模型参数: head_num=32, kv_head_num=8, head_size=128
  输入形状: Q[512, 32, 128], K[512, 8, 128], V[512, 8, 128]

  Grid 配置: grid(32, 512) = 16,384 个 Block
  Block 配置: 128 threads/block
  总线程数: 16,384 × 128 = 2,097,152

  ┌────────────────────────────────────────────────────────────────────────┐
  │                     完整的 Block 分配表                                │
  │                                                                        │
  │  Block(h=0, s=0):  Q[0,0,:] × K[*,0,:] → O[0,0,:]   kv_len=1        │
  │  Block(h=0, s=1):  Q[1,0,:] × K[*,0,:] → O[1,0,:]   kv_len=2        │
  │  Block(h=0, s=2):  Q[2,0,:] × K[*,0,:] → O[2,0,:]   kv_len=3        │
  │  ...                                                                   │
  │  Block(h=0, s=511): Q[511,0,:] × K[*,0,:] → O[511,0,:] kv_len=512   │
  │                                                                        │
  │  Block(h=1, s=0):  Q[0,1,:] × K[*,0,:] → O[0,1,:]   kv_len=1        │
  │  ... (h=0..3 都使用 KV head 0)                                        │
  │                                                                        │
  │  Block(h=4, s=0):  Q[0,4,:] × K[*,1,:] → O[0,4,:]   kv_len=1        │
  │  ... (h=4..7 使用 KV head 1)                                          │
  │  ...                                                                   │
  │  Block(h=31,s=511): Q[511,31,:] × K[*,7,:] → O[511,31,:] kv_len=512 │
  │                                                                        │
  │  最大 kv_len = 512 → ⌈512/1024⌉ = 1 个 tile/Block                    │
  │  最小 kv_len = 1   → 1 个 tile (极小)                                 │
  │  因果掩码: 靠前的 Block (小 seq_idx) 处理的 kv_len 更短                │
  └────────────────────────────────────────────────────────────────────────┘

  Orin GPU 上的调度 (16 SMs):
    每 SM 最多容纳 ~8 个 Block (受寄存器/smem 限制)
    活跃 Block: 16 × 8 = 128 个
    总 Block: 16,384 个
    每 SM 需处理: ~1,024 个 Block (轮转调度)
    靠前的 Block (kv_len 小) 执行很快，SM 可快速切换到后续 Block
```

---

## 附录 D：TILE_K = 1024 的设计决策分析

### D.1 问题陈述

在 `flash_attention_prefill_kernel_fp16` 中，K 和 V 沿序列维度按 `TILE_K = 1024` 分块处理。为什么选择 1024？这个值是如何在**共享内存容量、线程利用率、计算/访存比、L2 Cache 利用率**之间取得平衡的？

### D.2 TILE_K 的约束因素分析

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                    TILE_K 大小的决定因素                                  ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  约束 1: Shared Memory 容量                                               ║
║  ──────────────────────────                                               ║
║  Orin GPU (SM87): 每个 SM 共享内存 = 48KB (可配置最高 48KB)               ║
║                                                                           ║
║  本 kernel 的 smem 使用:                                                  ║
║    s_query[128 half]   = 128 × 2B = 256 bytes                           ║
║    s_scores[TILE_K float] = TILE_K × 4B                                  ║
║    s_warp_max[4 float] = 16 bytes  (静态分配)                            ║
║    s_warp_sum[4 float] = 16 bytes  (静态分配)                            ║
║    ─────────────────────────────────                                      ║
║    总动态 smem ≈ 256 + TILE_K × 4 bytes                                  ║
║                                                                           ║
║  TILE_K=1024 → smem = 256 + 4096 = 4,352 bytes (4.25 KB) ✅             ║
║  TILE_K=2048 → smem = 256 + 8192 = 8,448 bytes (8.25 KB) ✅ 仍可行     ║
║  TILE_K=4096 → smem = 256 + 16384 = 16,640 bytes (16.25 KB) ⚠️         ║
║  TILE_K=8192 → smem = 256 + 32768 = 33,024 bytes (32.25 KB) ❌ 占满    ║
║                                                                           ║
║  smem 越小 → 每 SM 可驻留更多 Block → SM 占用率越高                       ║
║                                                                           ║
║  ┌──────────┬──────────┬────────────────┬────────────┐                   ║
║  │ TILE_K   │ smem/Block│ Blocks/SM (48KB)│ SM 占用率  │                   ║
║  ├──────────┼──────────┼────────────────┼────────────┤                   ║
║  │ 256      │ 1.25 KB  │ ≤16 (受寄存器) │ 高         │                   ║
║  │ 512      │ 2.25 KB  │ ≤16            │ 高         │                   ║
║  │ 1024     │ 4.25 KB  │ ~8-11          │ 中高 ← ✅  │                   ║
║  │ 2048     │ 8.25 KB  │ ~5             │ 中         │                   ║
║  │ 4096     │ 16.25 KB │ ~2             │ 低 ❌      │                   ║
║  └──────────┴──────────┴────────────────┴────────────┘                   ║
║                                                                           ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  约束 2: 每线程工作量 (线程利用率)                                         ║
║  ──────────────────────────────                                           ║
║  Q·K 阶段: 128 线程协作计算 tile_len 个点积                               ║
║  每线程计算 ⌈TILE_K / BLOCK_SIZE⌉ 个 Q·K 点积                            ║
║                                                                           ║
║  ┌──────────┬────────────┬──────────────────────────────────────┐        ║
║  │ TILE_K   │ 点积数/线程 │ 效果                                 │        ║
║  ├──────────┼────────────┼──────────────────────────────────────┤        ║
║  │ 128      │ 1          │ 太少，tile 开销占比高(sync+reduce)   │        ║
║  │ 256      │ 2          │ 偏少，reduce 开销仍然显著             │        ║
║  │ 512      │ 4          │ 较好                                 │        ║
║  │ 1024     │ 8          │ 良好，计算充分掩盖 tile 固定开销 ← ✅│        ║
║  │ 2048     │ 16         │ 计算更充分，但 smem 翻倍，占用率下降  │        ║
║  └──────────┴────────────┴──────────────────────────────────────┘        ║
║                                                                           ║
║  Score×V 阶段: 128 线程各自遍历 tile_len 个 V 值                          ║
║  每线程执行 TILE_K 次 fmaf (展开×8 → 128 次 8-wide fmaf 迭代)            ║
║                                                                           ║
║  TILE_K=1024 时:                                                          ║
║    ×8 展开主循环: 1024/8 = 128 次迭代                                     ║
║    每次迭代: 8 次 global load + 8 次 smem load + 8 次 fmaf               ║
║    → 足够的指令流水线深度来隐藏全局内存延迟                                ║
║                                                                           ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  约束 3: Tile 固定开销的摊销                                               ║
║  ────────────────────────                                                 ║
║  每个 tile 有固定开销:                                                     ║
║    • 2 × Block Reduce (Max + Sum): ~10 μs                                ║
║    • 3 × __syncthreads(): ~1.5 μs                                        ║
║    • Online Softmax 修正 (correction × acc_o): ~0.5 μs                   ║
║    • 合计: ~12 μs / tile                                                  ║
║                                                                           ║
║  有效计算时间 (Q·K + Score×V):                                            ║
║    TILE_K=1024, head_size=128:                                            ║
║    Q·K: 1024 × 128 × 2 / 128线程 ≈ 2048 FLOPs/线程                     ║
║    S×V: 1024 × 2 / 1 = 2048 FLOPs/线程                                  ║
║    → ~4096 FLOPs/线程 → ~50 μs (Orin CUDA core ~80 GFLOPS/SM)           ║
║                                                                           ║
║  tile 开销占比: 12 / (50+12) ≈ 19% — 可接受                              ║
║                                                                           ║
║  如果 TILE_K=256:                                                         ║
║    有效计算 ~12.5 μs → 开销占比: 12 / (12.5+12) ≈ 49% — 太高! ❌         ║
║                                                                           ║
║  如果 TILE_K=2048:                                                        ║
║    有效计算 ~100 μs → 开销占比: 12 / (100+12) ≈ 11% — 更好               ║
║    但 smem=8.25KB → Block/SM 减少 → 总体吞吐可能反而下降                   ║
║                                                                           ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  约束 4: L2 Cache 友好度                                                   ║
║  ──────────────────────                                                   ║
║  Orin L2 Cache = 4MB (Jetson AGX Orin)                                    ║
║                                                                           ║
║  一个 tile 读取的 K/V 数据量:                                              ║
║    K_tile: TILE_K × head_size × sizeof(half) = 1024 × 128 × 2 = 256 KB  ║
║    V_tile: TILE_K × head_size × sizeof(half) = 1024 × 128 × 2 = 256 KB  ║
║    合计: 512 KB / tile                                                    ║
║                                                                           ║
║  GQA 场景: 4 个 Q head 共享 1 个 KV head                                  ║
║  → 同一 seq_idx 的 4 个 Block 读取相同的 K/V tile                          ║
║  → 首个 Block 将 tile 加载到 L2，后续 3 个 Block 命中 L2                   ║
║                                                                           ║
║  512 KB × 8 个 KV head = 4 MB → 恰好填满 L2                              ║
║  TILE_K=2048 → 1024 KB/tile → 单 tile 就占 L2 的 25% → 可能驱逐           ║
║  TILE_K=1024 → 512 KB/tile → 可在 L2 中并行驻留多个 tile ← ✅             ║
║                                                                           ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  约束 5: 2 的幂次对齐                                                      ║
║  ────────────────────                                                     ║
║  TILE_K = 1024 = 2¹⁰                                                     ║
║  • BLOCK_SIZE = 128 能整除 1024 → 无尾部处理                                ║
║  • 访存地址自然对齐 (128-bit, 256-bit 边界)                                 ║
║  • 循环展开×8: 1024/8 = 128 → 整除，无余数处理                             ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

### D.3 TILE_K 与线程工作量的映射图

```
═══════════════════════════════════════════════════════════════════════════════
  TILE_K=1024, BLOCK_SIZE=128 时每个线程在不同阶段的工作量
═══════════════════════════════════════════════════════════════════════════════

  ◆ Q·K 阶段: 128 线程协作 → 每线程 8 个点积 (stride=128)

  s_scores[0..1023] 的生产者映射:
  ┌───────────────────────────────────────────────────────────────────────┐
  │ index: 0    1    2   ...  127 │ 128  129  130  ... 255 │ ...        │
  │ 生产:  t0   t1   t2      t127│ t0   t1   t2      t127 │ ...        │
  │        ├── 第1轮 (k=tid) ────┤├── 第2轮 (k=tid+128) ─┤            │
  │                                                                      │
  │ index: 256  ... 383 │ 384  ... 511 │ 512 ... 639│ 640 ... 767│     │
  │ 生产:  t0  ... t127 │ t0  ... t127 │ t0 ... t127│ t0 ... t127│     │
  │        ├─ 第3轮 ────┤├── 第4轮 ────┤├─ 第5轮 ──┤├── 第6轮 ──┤     │
  │                                                                      │
  │ index: 768  ... 895 │ 896  ... 1023│                                │
  │ 生产:  t0  ... t127 │ t0  ... t127 │                                │
  │        ├── 第7轮 ──┤├── 第8轮 ────┤                                │
  └───────────────────────────────────────────────────────────────────────┘

  每个点积的计算: 128 维 → 16 次 float4 迭代 → 每次 4 个 half2 → 8 fmaf
  单线程每 tile: 8 个点积 × 128 fmaf = 1024 fmaf

  ◆ Softmax 阶段: 与 Q·K 相同的 stride 映射
  Thread tid 处理: s_scores[tid], s_scores[tid+128], ..., s_scores[tid+896]
  每线程 8 个 exp() 计算 + 局部累加

  ◆ Score×V 阶段: 线程角色切换! 每线程固定负责 1 个输出维度
  Thread tid 遍历 s_scores[0..1023]，每个乘以 V[k, tid]

  ┌───────────────────────────────────────────────────────────────────────┐
  │ Thread tid=0 的 V 累加 (展开×8):                                     │
  │                                                                       │
  │ 迭代 1:  acc += s[0]×V[0,0] + s[1]×V[1,0] + ... + s[7]×V[7,0]     │
  │ 迭代 2:  acc += s[8]×V[8,0] + s[9]×V[9,0] + ... + s[15]×V[15,0]   │
  │ ...                                                                   │
  │ 迭代 128: acc += s[1016]×V[1016,0] + ... + s[1023]×V[1023,0]        │
  │                                                                       │
  │ 共 128 次迭代 × 8 fmaf = 1024 fmaf/线程                             │
  │                                                                       │
  │ Thread tid=1 同时执行完全相同的模式，但读 V[*,1] 而非 V[*,0]          │
  │ → 128 线程同时读 V 的同一行不同列 → coalesced                        │
  └───────────────────────────────────────────────────────────────────────┘
```

### D.4 不同 TILE_K 取值的综合对比

```
┌──────────┬──────────┬───────────┬───────────┬───────────┬──────────────┐
│ TILE_K   │smem(KB)  │Block/SM   │点积/线程  │展开效率   │tile开销占比  │
├──────────┼──────────┼───────────┼───────────┼───────────┼──────────────┤
│ 128      │ 0.75     │ ~16(高)   │ 1         │ 很差      │ ~50% ❌      │
│ 256      │ 1.25     │ ~16(高)   │ 2         │ 差        │ ~49% ❌      │
│ 512      │ 2.25     │ ~12       │ 4         │ 尚可      │ ~32%         │
│ 1024 ✅  │ 4.25     │ ~8-11     │ 8         │ 好(×8整除)│ ~19% ✅      │
│ 2048     │ 8.25     │ ~5        │ 16        │ 好        │ ~11%         │
│ 4096     │ 16.25    │ ~2(低)    │ 32        │ 好        │ ~6%          │
├──────────┼──────────┼───────────┼───────────┼───────────┼──────────────┤
│ 最优     │ ← 越小   │ ← 越多   │ → 越多   │ → 越好   │ → 越低      │
│ 方向     │ 越好     │ 越好     │ 越好     │ 越好     │ 越好        │
└──────────┴──────────┴───────────┴───────────┴───────────┴──────────────┘

  TILE_K=1024 是 Orin SM87 架构上的 **帕累托最优点**:
  • smem 足够小 → 每 SM 驻留 8+ Block → 高占用率
  • 每线程 8 个点积 → 计算量充足，掩盖固定开销
  • ×8 展开完美整除 → 零余数处理
  • 512KB/tile → 适配 4MB L2 的 GQA 复用
  • 总 tile 数适中 → 不会因过多 sync 和 reduce 拖慢
```

---

## 附录 E：flash_attention_prefill_kernel_fp16 核函数优化手段详解

本节对 [flash_attention_kernel.cu](../kuiper/source/op/kernels/cuda/flash_attention_kernel.cu) 中 `flash_attention_prefill_kernel_fp16`（第 554 行）所采用的全部优化技术进行逐一剖析。

### E.1 优化手段总览

```
┌────┬─────────────────────────────────┬──────────────┬────────────────────────┐
│ #  │ 优化手段                         │ 优化类别      │ 受益阶段               │
├────┼─────────────────────────────────┼──────────────┼────────────────────────┤
│ 1  │ Online Softmax (单遍扫描)        │ 算法         │ 全局: 避免物化 S 矩阵   │
│ 2  │ Tiled KV 处理 (TILE_K=1024)     │ 算法/访存     │ 全局: O(1) smem 使用   │
│ 3  │ BLOCK_SIZE = head_size = 128    │ 线程映射      │ 全局: 零冗余线程       │
│ 4  │ float4 向量化 Q·K 点积           │ 指令级优化    │ Step 1: Q·K 计算       │
│ 5  │ half2 → float2 转换 + fmaf      │ 指令级优化    │ Step 1: Q·K 计算       │
│ 6  │ Q 加载到 Shared Memory          │ 访存优化      │ Step 1: 消除重复读取   │
│ 7  │ Warp Shuffle Block Reduce       │ 并行归约      │ Step 2,5: max/sum      │
│ 8  │ Softmax Flush-to-Zero (FTZ)     │ 数值优化      │ Step 4: exp 计算       │
│ 9  │ V 读取 Coalesced 访存           │ 访存优化      │ Step 7: V 累加         │
│ 10 │ V 累加循环展开 ×8               │ 指令级优化    │ Step 7: V 累加         │
│ 11 │ __ldg() 只读 Cache 提示         │ 访存优化      │ Step 1,7: K/V 读取     │
│ 12 │ __restrict__ 指针限定           │ 编译器优化    │ 全局: 消除别名分析     │
│ 13 │ GQA KV head 共享 + L2 复用      │ 架构级优化    │ 全局: 减少 HBM 带宽    │
│ 14 │ 因果掩码通过 kv_len 隐式实现     │ 算法          │ 全局: 零额外计算       │
│ 15 │ #pragma unroll 编译器指令       │ 编译器优化    │ Step 1,2,5: 内循环     │
└────┴─────────────────────────────────┴──────────────┴────────────────────────┘
```

### E.2 优化 1: Online Softmax — 单遍扫描避免物化 Score 矩阵

**问题**：标准 Attention 需要先计算完整的 $S = QK^T / \sqrt{d}$ 矩阵（$O(N^2)$ 显存），然后三遍扫描做 softmax。

**解决方案**：Online Softmax 将 max、sum、加权累加合并为单遍扫描 + 增量修正。

```cpp
// 对应代码 (每个 tile 处理后):
float correction = expf(row_max - m_new);   // 修正因子
acc_o *= correction;                         // 缩放历史累加值
// ... 累加新 tile 的 score × V ...
row_max = m_new;
row_sum = correction * row_sum + l_j;        // 更新全局和
```

```
  标准方法:                              Online Softmax:
  ┌────────────────────┐                ┌────────────────────┐
  │ 分配 S[N × N] 矩阵 │                │ 只需 s_scores[1024] │
  │ (N=2048 → 16MB!!)  │                │ (固定 4KB)          │
  │                    │                │                    │
  │ Pass 1: 求 max     │                │ 每 tile:           │
  │ Pass 2: exp + sum  │   合并为 →     │   Q·K + max        │
  │ Pass 3: normalize  │                │   correction + exp │
  │ Pass 4: × V        │                │   sum + × V        │
  └────────────────────┘                │   (单遍完成)       │
  3× 读取 S 矩阵                        └────────────────────┘
  → 3N² 次 HBM 访问                     0 次额外 HBM 访问 ✅
```

**收益**：内存 $O(N^2) \rightarrow O(1)$（仅 tile 大小），HBM 带宽大幅降低。

### E.3 优化 2: Tiled KV 处理 — 固定大小共享内存

**问题**：KV 序列长度可变（1 到数千），不能为 `s_scores` 分配可变大小 smem。

**解决方案**：以 `TILE_K=1024` 固定分块，循环处理所有 tile。

```cpp
const int smem_size = head_size * sizeof(half) + TILE_K * sizeof(float);
// = 128×2 + 1024×4 = 4352 bytes (固定!)

for (int tile_start = 0; tile_start < kv_len; tile_start += TILE_K) {
    const int tile_len = min(TILE_K, kv_len - tile_start);
    // ... 处理当前 tile ...
}
```

**收益**：smem 大小固定 → CUDA Graph 兼容、SM 占用率可预测、编译器优化空间更大。

### E.4 优化 3: BLOCK_SIZE = head_size = 128 — 线程-维度一一映射

**问题**：如果 `BLOCK_SIZE ≠ head_size`，线程要么闲置（BLOCK_SIZE > head_size），要么需要循环处理多维（BLOCK_SIZE < head_size）。

**解决方案**：令 `BLOCK_SIZE = head_size = 128`，每线程恰好负责输出的一个维度。

```cpp
float acc_o = 0.0f;    // 每线程仅 1 个累加器 (寄存器)
// 而非 float acc_o[N]; 的数组

// 最终写回:
o_ptr[tid] = __float2half(acc_o * inv_sum);  // tid 直接作为维度索引
```

```
  BLOCK_SIZE=128, head_size=128 → 映射关系:

  Thread 0   → O[*, *, 0]     1 个寄存器 acc_o
  Thread 1   → O[*, *, 1]     1 个寄存器 acc_o
  ...
  Thread 127 → O[*, *, 127]   1 个寄存器 acc_o

  ✅ 零冗余: 没有 idle 线程，没有循环处理多维
  ✅ 寄存器压力最小: 每线程仅 ~40 个寄存器 (含临时变量)
  ✅ 输出写回完美 coalesced: 128 线程写 128 个连续 half = 256B
```

**收益**：最大化线程利用率，最小化寄存器消耗，消除条件分支。

### E.5 优化 4: float4 向量化 Q·K 点积 — 128-bit 宽度读取

**问题**：逐元素读取 K 向量需要 128 次 16-bit 读取 = 128 次全局内存事务。

**解决方案**：将 K 和 Q 的读取重新解释为 `float4`（128-bit），每次读取 8 个 half。

```cpp
const float4* k_ptr_f4 = reinterpret_cast<const float4*>(K_cache + kv_pos * kv_dim + head_offset);
const float4* q_ptr_f4 = reinterpret_cast<const float4*>(s_query);

float2 acc = make_float2(0.0f, 0.0f);
#pragma unroll
for (int d = 0; d < head_size / 8; d++) {   // 128/8 = 16 次迭代
    float4 q_packed = q_ptr_f4[d];           // 1 次 128-bit 读 (smem)
    float4 k_packed = __ldg(k_ptr_f4 + d);   // 1 次 128-bit 读 (global)
    // ... 拆解为 4 个 half2，执行 8 次乘加 ...
}
```

```
  逐元素读取 (naive):                     float4 向量化:
  ┌────────────────────────┐              ┌────────────────────────┐
  │ 128 次 16-bit load     │              │ 16 次 128-bit load     │
  │ = 128 次内存事务       │              │ = 16 次内存事务        │
  │                        │              │                        │
  │ K[0], K[1], ..., K[127]│              │ float4[0], ..., [15]  │
  │ 每次 2 bytes           │              │ 每次 16 bytes          │
  └────────────────────────┘              └────────────────────────┘
  128 次事务                               16 次事务 → 8× 减少 ✅
```

**收益**：全局内存事务数减少 8×，充分利用 128-bit 总线宽度。

### E.6 优化 5: half2 → float2 转换 + fmaf 融合乘加

**问题**：直接在 half 精度下计算会导致精度丢失和溢出。

**解决方案**：将 `half2` 转为 `float2` 后用 `fmaf`（fused multiply-add）在 FP32 下累加。

```cpp
const half2* q_h2 = reinterpret_cast<const half2*>(&q_packed);
const half2* k_h2 = reinterpret_cast<const half2*>(&k_packed);
#pragma unroll
for (int i = 0; i < 4; i++) {
    float2 q_f = __half22float2(q_h2[i]);    // half2 → float2 (1 条指令)
    float2 k_f = __half22float2(k_h2[i]);
    acc.x = fmaf(q_f.x, k_f.x, acc.x);      // fused multiply-add (1 条指令)
    acc.y = fmaf(q_f.y, k_f.y, acc.y);
}
```

```
  每个 float4 (128-bit) 内部的数据解析:

  float4 = 16 bytes = 8 × half = 4 × half2
  ┌──────────┬──────────┬──────────┬──────────┐
  │ half2[0] │ half2[1] │ half2[2] │ half2[3] │
  │ (q0,q1)  │ (q2,q3)  │ (q4,q5)  │ (q6,q7)  │
  └──────────┴──────────┴──────────┴──────────┘
       ↓           ↓           ↓           ↓
  __half22float2  → (f0,f1)  (f2,f3)  (f4,f5)  (f6,f7)
       ↓           ↓           ↓           ↓
  fmaf(q,k,acc) × 2 for each half2 → 8 fmaf per float4 pair
```

**收益**：
- `__half22float2` 在硬件级别是单条指令
- `fmaf` 比分开的 `*` 和 `+` 更快且更精确（只有一次舍入）
- 在 FP32 下累加避免了 FP16 累加的精度损失

### E.7 优化 6: Q 加载到 Shared Memory 并重复利用

**问题**：每线程在 Q·K 阶段要计算 8 个点积，每次都要读 Q 的全部 128 维。如果每次从全局内存读 Q，就是 8 × 128 = 1024 次冗余读取。

**解决方案**：启动时将 Q 加载到 Shared Memory，后续所有 tile 的所有点积都从 smem 读取。

```cpp
// 加载 Q (一次性):
const half* q_ptr = Q + seq_idx * dim + head * head_size;
for (int d = tid; d < head_size; d += BLOCK_SIZE) {
    s_query[d] = q_ptr[d];
}
__syncthreads();

// 后续所有 tile 中:
float4 q_packed = q_ptr_f4[d];   // ← 从 smem 读取 (~20 cycle)
                                  //    而非 global mem (~400 cycle)
```

**收益**：
- Q 从全局内存只读 1 次（256 bytes），后续全部命中 smem
- tiles=3 时节省: 2 × 8 × 16 × 16 = 4096 bytes 的重复全局读取
- smem 延迟 ~20 cycle vs 全局 ~400 cycle → 20× 加速

### E.8 优化 7: Warp Shuffle Block Reduce — 零共享内存的 Warp 内归约

**问题**：Block 级 reduce (max/sum) 传统上需要大量 smem 和多轮 `__syncthreads()`。

**解决方案**：使用 `__shfl_xor_sync` 在 Warp 内做蝶形归约（5 轮），仅在跨 Warp 时用少量 smem。

```cpp
// Warp 内归约 (零 smem, 零 sync):
#pragma unroll
for (int offset = 16; offset > 0; offset >>= 1) {
    tile_max_local = fmaxf(tile_max_local,
        __shfl_xor_sync(0xffffffff, tile_max_local, offset));
}

// 跨 Warp 归约 (仅需 4 个 float smem + 1 次 sync):
__shared__ float s_warp_max[4];
if (lane_id == 0) s_warp_max[warp_id] = tile_max_local;
__syncthreads();
if (tid == 0) {
    m_j = fmaxf(fmaxf(s_warp_max[0], s_warp_max[1]),
                 fmaxf(s_warp_max[2], s_warp_max[3]));
    s_warp_max[0] = m_j;
}
__syncthreads();
m_j = s_warp_max[0];
```

```
  传统 Block Reduce:                     Warp Shuffle Reduce:
  ┌────────────────────────┐              ┌────────────────────────┐
  │ smem[128] + 7 轮 sync  │              │ 5 轮 __shfl_xor (Warp内) │
  │ O(log₂128 = 7) 轮     │              │ + smem[4] + 2 轮 sync  │
  │ 每轮: sync + smem 读写 │              │ 总: ~7 轮，但其中 5 轮 │
  │ ~7 × 300 cycle         │              │ 是零延迟 Warp 指令     │
  │ = ~2100 cycle           │              │ ~5×1 + 2×300 = ~605 cycle │
  └────────────────────────┘              └────────────────────────┘
                                           3.5× 加速 ✅
```

**收益**：归约延迟减少 ~3.5×，smem 占用从 128 float 降至 4 float。

### E.9 优化 8: Softmax Flush-to-Zero (FTZ) — 避免 exp() 下溢

**问题**：`expf(x)` 在 x < -87 时下溢为极小的非规格化数（denormalized），处理 denorm 数在 GPU 上可能很慢。

**解决方案**：当 `score - m_new < -20` 时直接置零。$e^{-20} \approx 2 \times 10^{-9}$，对 softmax 结果的影响可忽略不计。

```cpp
float val = s_scores[k_idx] - m_new;
float exp_score = (val > SOFTMAX_FTZ) ? expf(val) : 0.0f;  // SOFTMAX_FTZ = -20.0f
```

```
  score - max 的分布:
  ────────────────────────────────────────────────────→ 0
  -80   -60   -40   -20    -10    -5    0
  │      │      │      │      │      │    │
  └──── 全部 flush to 0 ────┘      └──正常 exp──┘
        (对 softmax 贡献 < 1e-9)

  不做 FTZ: expf(-50) = 1.93e-22 → denormalized number → 硬件惩罚
  做 FTZ:   直接设为 0.0f → 消除 expf 调用 + 消除 denorm 惩罚
```

**收益**：
- 避免 `expf()` 在极小值上的调用开销
- 消除 denormalized float 的硬件处理惩罚
- 对精度几乎零影响（误差 < $10^{-9}$）

### E.10 优化 9: V 读取 Coalesced 访存 — 128 线程同读一行

**问题**：V 累加时每线程需遍历 `tile_len` 个 V 向量，如何高效读取？

**解决方案**：利用 `BLOCK_SIZE = head_size = 128` 的一一映射，128 个线程同时读取 V 同一行的 128 个连续维度。

```cpp
// 预计算每线程的 V 基地址 (只依赖 tid):
const half* v_thread_base = V_cache + head_offset + tid;

// 每个 tile 内:
const half* v_ptr = v_thread_base + tile_start * kv_dim;
// v_ptr 指向 V[tile_start, kv_head, tid] — 第 tid 个维度

// 遍历 k:
float v0 = __half2float(__ldg(v_ptr));              // V[tile+0, kv_head, tid]
float v1 = __half2float(__ldg(v_ptr + kv_dim));     // V[tile+1, kv_head, tid]
// ...
v_ptr += 8 * kv_dim;                                // 跳到下一批 8 个 V
```

```
  128 线程同时读取 V[k] 的访存模式:

  V_cache 内存布局 (行主序):
  ┌─────────────────────────────────────────────────────────┐
  │ V[k, kv_head, 0] V[k, kv_head, 1] ... V[k, kv_head, 127] │
  │      ↑                  ↑                     ↑            │
  │    Thread 0          Thread 1             Thread 127       │
  └─────────────────────────────────────────────────────────┘
  128 × 2 bytes = 256 bytes → 2 个 128B 内存事务 → 完美 coalesced ✅

  如果是"每线程遍历不同 K 位置的同一维度":
  Thread 0: V[0,*,0], V[128,*,0], V[256,*,0] ...  ← stride=kv_dim=1024
  Thread 1: V[0,*,1], V[128,*,1], V[256,*,1] ...
  → 非 coalesced，但同一 k 的读取是 coalesced ← 本 kernel 的设计
```

**收益**：V 读取的合并度为 100%（128 线程读 128 个连续 half），内存带宽利用率最大化。

### E.11 优化 10: V 累加循环展开 ×8 — 指令级并行 (ILP)

**问题**：逐个处理 V 值时，每次 `__ldg` 的全局内存延迟 (~400 cycle) 无法被掩盖。

**解决方案**：一次加载 8 个 score 和 8 个 V 值，发出 8 条独立的 `__ldg` 和 `fmaf` 指令，让 GPU 流水线/乱序执行机制掩盖延迟。

```cpp
for (; k + 7 < tile_len; k += 8) {
    // 批量读 8 个 score (smem, broadcast):
    float s0 = s_scores[k];
    float s1 = s_scores[k+1];
    // ... s2..s7 ...

    // 批量读 8 个 V 值 (global, stride=kv_dim):
    float v0 = __half2float(__ldg(v_ptr));
    float v1 = __half2float(__ldg(v_ptr + kv_dim));
    // ... v2..v7 ...

    // 批量 fmaf:
    acc_o = fmaf(s0, v0, acc_o);
    acc_o = fmaf(s1, v1, acc_o);
    // ... 共 8 次 ...

    v_ptr += 8 * kv_dim;
}
```

```
  无展开 (逐个):                          ×8 展开:
  ┌──────────────────────┐                ┌──────────────────────────────┐
  │ load V[k]   (400 cy) │                │ load V[k]                    │
  │ fmaf        (4 cy)   │                │ load V[k+1]   ← 在 V[k] 返回│
  │ load V[k+1] (400 cy) │                │ load V[k+2]     之前就发出   │
  │ fmaf        (4 cy)   │                │ ...                          │
  │ ...                  │                │ load V[k+7]   ← 8 条并行飞行 │
  │                      │                │ fmaf(s0,v0)   ← V[k]已就绪  │
  │ 串行: ~404 cy/iter   │                │ fmaf(s1,v1)   ← V[k+1]已就绪│
  └──────────────────────┘                │ ...                          │
                                          │ fmaf(s7,v7)                  │
                                          │                              │
                                          │ ~400+8×4 ≈ 432 cy / 8 iters │
                                          │ = 54 cy/iter → 7.5× 加速  │
                                          └──────────────────────────────┘
```

**收益**：通过发射多条独立 load 指令，利用 GPU 的内存级并行 (MLP) 掩盖延迟，实际吞吐提升 ~7.5×。

### E.12 优化 11: __ldg() 只读缓存提示

**问题**：默认的全局内存 load 走 L1 data cache，可能与 store 冲突。

**解决方案**：对 K/V 的读取使用 `__ldg()`（load via texture/read-only cache），指示编译器生成 `ld.global.nc`（non-coherent load）指令。

```cpp
float4 k_packed = __ldg(k_ptr_f4 + d);        // K 读取
float v0 = __half2float(__ldg(v_ptr));          // V 读取
```

**收益**：
- 使用独立的只读缓存通路，不与 store 操作争抢 L1
- 在 Orin (SM87) 架构上 `ld.global.nc` 有独立的 48KB 纹理缓存
- K/V 是只读数据，非常适合这种访问模式

### E.13 优化 12: __restrict__ 指针限定 — 消除别名分析

**问题**：如果编译器不确定 Q/K/V/O 指针是否重叠（aliasing），会插入额外的 load/store fence。

**解决方案**：所有输入输出指针都标注 `__restrict__`。

```cpp
__global__ void flash_attention_prefill_kernel_fp16(
    const half* __restrict__ Q,
    const half* __restrict__ K_cache,
    const half* __restrict__ V_cache,
    half* __restrict__ O,
    // ...
```

**收益**：编译器可以自由重排 load/store 指令，更积极地进行寄存器分配和指令调度。

### E.14 优化 13: GQA KV Head 共享与 L2 Cache 复用

**问题**：Qwen3-8B 使用 GQA（32 个 Q head，8 个 KV head），如果每个 Block 独立读取 KV，同一 KV head 的数据被重复读取 4 次。

**解决方案**：通过 `kv_head = head / kv_mul` 映射，同组的 4 个 Block 读取相同的 KV 地址。GPU 硬件的 L2 cache 自动实现数据复用。

```cpp
const int kv_head = head / kv_mul;   // head 0,1,2,3 → kv_head 0
const int head_offset = kv_head * head_size;  // 4 个 Block 计算出相同的 offset
```

```
  Block(h=0,s) ──┐
  Block(h=1,s) ──┤── 都读 K[*, kv_head=0, :] 和 V[*, kv_head=0, :]
  Block(h=2,s) ──┤
  Block(h=3,s) ──┘
       ↓
  第一个 Block 的读取: HBM → L2 Cache (冷读)
  后续三个 Block: L2 Cache 命中 (热读) → 延迟从 ~400 cycle → ~100 cycle
  带宽节省: 4× → 75% 的 KV 读取命中 L2
```

**收益**：对于 `kv_mul=4`，KV 的有效 HBM 带宽消耗降低 ~4×。

### E.15 优化 14: 因果掩码通过 kv_len 隐式实现

**问题**：因果 Attention 要求 `q[i]` 只能看到位置 `≤ i` 的 token。传统做法需要显式的 mask 矩阵。

**解决方案**：通过 `kv_len = start_pos + seq_idx + 1` 限制每个 Block 遍历的 KV 范围。

```cpp
const int cur_pos = start_pos + seq_idx;
const int kv_len = cur_pos + 1;   // 只看 [0, cur_pos]

for (int tile_start = 0; tile_start < kv_len; tile_start += TILE_K) {
    // kv_len 自动截断了未来位置
}
```

**收益**：
- 零额外计算（无 mask 矩阵乘法或条件判断）
- 零额外内存（无 mask 矩阵分配）
- 靠前的 Block（小 `seq_idx`）自然处理更少的 KV → 更快完成 → SM 尽早释放

### E.16 优化 15: #pragma unroll 编译器展开指令

**问题**：循环次数已知的 tight loop 如果不展开，会有循环控制指令（比较、跳转）的开销。

**解决方案**：在关键内循环标注 `#pragma unroll`，提示 `nvcc` 完全展开。

```cpp
#pragma unroll
for (int d = 0; d < head_size / 8; d++) {   // 128/8=16，编译期已知 → 完全展开
    // ...
}

#pragma unroll
for (int i = 0; i < 4; i++) {              // 4 次，完全展开
    // ...
}

#pragma unroll
for (int offset = 16; offset > 0; offset >>= 1) {  // 5 次，完全展开
    // ...
}
```

**收益**：
- 消除循环控制指令（`cmp`, `bra`）的开销
- 编译器可以跨迭代优化寄存器分配
- `head_size/8 = 16` 次迭代展开为 16 段直线代码 → 无分支流水线

### E.17 优化效果汇总与对应代码位置

```
┌────┬──────────────────────────┬──────────────────┬─────────────────────────┐
│ #  │ 优化手段                  │ 代码行 (大约)    │ 性能影响                │
├────┼──────────────────────────┼──────────────────┼─────────────────────────┤
│ 1  │ Online Softmax           │ L668-672, L764-765│ 内存 O(N²)→O(1)        │
│ 2  │ Tiled KV (TILE_K=1024)   │ L608-611         │ smem 固定 4.25KB       │
│ 3  │ BLOCK_SIZE=head_size=128 │ L571, launch     │ 100% 线程利用率        │
│ 4  │ float4 向量化 Q·K         │ L618-622         │ 全局读事务 8× 减少     │
│ 5  │ half2→float2 + fmaf      │ L626-632         │ 指令数减半+精度提升     │
│ 6  │ Q→Shared Memory          │ L596-599         │ Q 读取延迟 20× 降低    │
│ 7  │ Warp Shuffle Reduce      │ L641-658         │ 归约延迟 3.5× 降低     │
│ 8  │ Softmax FTZ (-20.0f)     │ L664-667         │ 消除 denorm + 减少 exp │
│ 9  │ V Coalesced 读取          │ L604, L676-724   │ 100% 内存合并度        │
│ 10 │ V 累加展开 ×8             │ L682-715         │ ILP 提升 ~7.5×        │
│ 11 │ __ldg() 只读缓存          │ L622, L692-699   │ 独立读缓存通路         │
│ 12 │ __restrict__             │ L554-558         │ 编译器指令重排          │
│ 13 │ GQA L2 Cache 复用         │ L576-577         │ KV 带宽 4× 降低       │
│ 14 │ 因果掩码→kv_len           │ L579-580         │ 零额外计算/内存        │
│ 15 │ #pragma unroll           │ L623, L627, L641 │ 消除循环控制开销        │
└────┴──────────────────────────┴──────────────────┴─────────────────────────┘
```
