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
