# OrinMLLM PagedAttention 详细分析报告

> 生成日期：2026-02-23  
> 工程路径：/mnt/ssd/workspace/OrinMLLM  
> 目标平台：NVIDIA Jetson Orin (SM87, 统一内存架构)

---

## 目录

- [第一章：PagedAttention 的架构和原理](#第一章pagedattention-的架构和原理)
- [第二章：PagedAttention 的适配过程、关键点与难点解决](#第二章pagedattention-的适配过程关键点与难点解决)
- [第三章：结合源码详解 PagedAttention 的使用流程](#第三章结合源码详解-pagedattention-的使用流程)
- [第四章：PagedAttention CUDA Kernel 详细解读与优化分析](#第四章pagedattention-cuda-kernel-详细解读与优化分析)
- [第五章：PagedAttention 与连续 KV Cache 的内存管理对比](#第五章pagedattention-与连续-kv-cache-的内存管理对比)
- [第六章：Block Size 选择与 Copy-on-Write 实现](#第六章block-size-选择与-copy-on-write-实现)
- [第七章：Block Table 在 CUDA Kernel 中的传递与数据结构设计](#第七章block-table-在-cuda-kernel-中的传递与数据结构设计)
- [第八章：Orin 内存架构分析与针对性优化](#第八章orin-内存架构分析与针对性优化)
- [第九章：Orin 上的低延迟调度器设计与 vLLM 连续批处理对比](#第九章orin-上的低延迟调度器设计与-vllm-连续批处理对比)
- [第十章：Orin 上的内存碎片处理与 Block Pool 耗尽策略](#第十章orin-上的内存碎片处理与-block-pool-耗尽策略)
- [第十一章：PagedAttention 正确性验证与 Orin CUDA 调试难点](#第十一章pagedattention-正确性验证与-orin-cuda-调试难点)
- [第十二章：Orin 上 PagedAttention 的最大改进点](#第十二章orin-上-pagedattention-的最大改进点)
- [第十三章：PagedAttention 与 Prefix Cache、CUDA Graph 的关系](#第十三章pagedattention-与-prefix-cachecuda-graph-的关系)
- [第十四章：各模型带 PagedAttention 的运行指令](#第十四章各模型带-pagedattention-的运行指令)
- [第十五章：CUDA Graph + PagedAttention 乱码问题的分析与修复](#第十五章cuda-graph--pagedattention-乱码问题的分析与修复)
- [第十六章：PagedAttention 与连续 KV Cache 的显存使用对比](#第十六章pagedattention-与连续-kv-cache-的显存使用对比)
- [第十七章：如何证明 PagedAttention 已生效](#第十七章如何证明-pagedattention-已生效)

---

## 第一章：PagedAttention 的架构和原理

### 1.1 背景：传统 KV Cache 的问题

在标准 Transformer 推理中，KV Cache 按照连续内存方式分配：

```
传统布局：key_cache / value_cache = [layer_num, max_seq_len, kv_dim]
```

对于 Qwen3-8B (36 层, kv_dim=1024, max_seq_len=8192, FP16) 而言：

```
单个缓存大小 = 36 × 8192 × 1024 × 2 bytes = 576 MB
Key + Value 总计 = 1,152 MB ≈ 1.1 GB
```

这种方式存在三个核心问题：

| 问题 | 描述 |
|------|------|
| **内存浪费** | 即使推理只用了几十个 token，也需要预分配整个 max_seq_len 的内存 |
| **内存碎片化** | 多序列并发时，连续大块难以分配，容易导致 OOM |
| **扩展性差** | 序列长度动态变化时，无法灵活调整内存用量 |

在 Orin 平台上，总 GPU 内存仅 8-16GB（与 CPU 共享），内存效率尤为关键。

### 1.2 PagedAttention 核心思想

PagedAttention 借鉴操作系统的**虚拟内存分页**机制，将 KV Cache 的管理从"连续大块"改为"固定大小的页（Block）"：

```
核心类比：
  操作系统虚拟内存              PagedAttention
  ─────────────────            ──────────────────
  虚拟页 (Virtual Page)    →   逻辑块 (Logical Block)
  物理页帧 (Physical Frame) →   物理块 (Physical Block)
  页表 (Page Table)        →   块表 (Block Table)
  页大小 (4KB)            →   Block大小 (16 tokens)
  MMU (地址翻译)           →   paged_off() (GPU地址翻译)
```

### 1.3 内存布局设计

PagedAttention 的内存由三部分组成：

#### (1) Key Pool 和 Value Pool

```
Key Pool:   [num_blocks, page_size, kv_dim]   — 所有层共享的物理块池
Value Pool: [num_blocks, page_size, kv_dim]   — 所有层共享的物理块池
```

在本工程中（参见 `kuiper/source/base/paged_kv_cache.cpp` 构造函数）：

```cpp
// paged_kv_cache.cpp 第27行
max_blocks_per_seq_ = (max_seq_len + page_size - 1) / page_size;
num_blocks_ = num_layers * max_blocks_per_seq_;

// 第47-50行：分配 GPU 内存
size_t pool_bytes = (size_t)num_blocks_ * page_size * kv_dim * dtype_size_;
cudaMalloc(&key_pool_gpu_, pool_bytes);
cudaMalloc(&value_pool_gpu_, pool_bytes);
```

对于 Qwen3-8B（page_size=16, max_seq_len=8192）：
- `max_blocks_per_seq = ceil(8192/16) = 512`
- `num_blocks = 36 × 512 = 18,432`
- `pool_bytes = 18432 × 16 × 1024 × 2 = 576 MB`（单个 pool）

#### (2) Block Table（块表）

Block Table 是一个二维映射表：

```
Block Table: [num_layers, max_blocks_per_seq]   — 逻辑块号 → 物理块号
```

每个 layer 有独立的块表行。块表同时维护 CPU 版本和 GPU 版本：

```cpp
// paged_kv_cache.h 第128-132行
std::vector<int32_t> block_table_cpu_;   // CPU 端，用于分配管理
int32_t* block_table_gpu_ = nullptr;     // GPU 端，用于 kernel 访问
```

#### (3) Free List（空闲块栈）

```cpp
// paged_kv_cache.cpp 第40行
free_list_.resize(num_blocks_);
std::iota(free_list_.rbegin(), free_list_.rend(), 0);
// 结果：[N-1, N-2, ..., 1, 0]，pop_back 按 0, 1, 2... 顺序分配
```

Free List 使用栈式分配器（`push_back`/`pop_back`），O(1) 时间复杂度。

### 1.4 地址翻译机制

这是 PagedAttention 最核心的操作。给定一个 token 位置 `pos`，如何找到它在 GPU 内存中的物理地址？

```
地址翻译公式：
  logical_block   = pos / page_size        =  pos >> PAGE_SHIFT
  block_offset    = pos % page_size        =  pos & PAGE_MASK
  physical_block  = block_table[layer_idx * max_blocks_per_seq + logical_block]
  element_offset  = (physical_block * page_size + block_offset) * kv_dim
  final_address   = pool + element_offset
```

在 CUDA Kernel 中，这个翻译由 `paged_off()` 函数实现（`paged_attention_kernel.cu` 第43-45行）：

```cuda
__device__ __forceinline__ size_t paged_off(const int32_t* bt, int32_t pos, int32_t kv_dim) {
  return ((size_t)bt[pos >> PAGE_SHIFT] * PAGE_SIZE + (pos & PAGE_MASK)) * kv_dim;
}
```

使用位运算（`>>` 和 `&`）替代除法和取模，这是选择 page_size 为 2 的幂次的原因。

**地址翻译示意图**（以 page_size=16 为例）：

```
Token Position: pos = 35
  logical_block  = 35 >> 4 = 2      (第 3 个逻辑块)
  block_offset   = 35 & 15 = 3      (块内第 4 个位置)
  physical_block = block_table[layer_idx * 512 + 2] = 7  (假设映射到物理块 7)
  element_offset = (7 * 16 + 3) * 1024 = 115,712

即：pool[115712 ... 115712+1023] 存储 layer_idx 层、position 35 的 KV 向量
```

### 1.5 页分配流程

页分配在 CPU 端完成，通过 `ensure_allocated_to()` 方法（`paged_kv_cache.cpp` 第80-97行）：

```cpp
void PagedKVCacheManager::ensure_allocated_to(int32_t pos) {
  if (pos <= allocated_pos_) return;  // 已分配到此位置，跳过

  for (int32_t p = allocated_pos_ + 1; p <= pos; ++p) {
    int32_t logical_block = p / page_size_;

    // 对所有层检查：该逻辑块是否已分配
    for (int32_t layer = 0; layer < num_layers_; ++layer) {
      int32_t table_idx = layer * max_blocks_per_seq_ + logical_block;
      if (block_table_cpu_[table_idx] == -1) {
        // 未分配 → 从 free list 取一个物理块
        block_table_cpu_[table_idx] = allocate_block();
      }
    }
  }
  allocated_pos_ = pos;
}
```

关键点：
- **惰性分配**：只在需要时才分配新页，不会预分配 max_seq_len 的所有页
- **跨层同步**：同一个逻辑块在所有层同时分配（因为每个 token 需要在所有层写入 KV）
- **去重检测**：同一逻辑块只分配一次（通过 `-1` 标记检测）

### 1.6 block table 同步

CPU 端完成分配后，需要将 block table 复制到 GPU：

```cpp
// paged_kv_cache.cpp 第99-107行
void PagedKVCacheManager::sync_block_table() {
  size_t bytes = num_layers_ * max_blocks_per_seq_ * sizeof(int32_t);
  if (stream_) {
    cudaMemcpyAsync(block_table_gpu_, block_table_cpu_.data(), bytes,
                    cudaMemcpyHostToDevice, stream_);
  } else {
    cudaMemcpy(block_table_gpu_, block_table_cpu_.data(), bytes,
               cudaMemcpyHostToDevice);
  }
}
```

### 1.7 PagedAttention 的总体架构图

```
┌───────────────────────────────────────────────────────────────────┐
│                   PagedKVCacheManager (CPU)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐    │
│  │  Free List   │  │ Block Table  │  │  allocated_pos_      │    │
│  │ (栈式分配器) │  │    (CPU)     │  │  跟踪最高已分配位置  │    │
│  │ [N-1...0]    │  │ [layer*512+  │  │                      │    │
│  │ pop_back→0   │  │  logical] =  │  │                      │    │
│  │ pop_back→1   │  │  physical    │  │                      │    │
│  └──────────────┘  └──────┬───────┘  └──────────────────────┘    │
│          ↑                │                                      │
│     allocate_block()      │ sync_block_table()                   │
│     free_block()          │ cudaMemcpy CPU→GPU                   │
│                           ↓                                      │
├───────────────────────────────────────────────────────────────────┤
│                      GPU Memory                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │ Block Table (GPU)  [num_layers, max_blocks_per_seq]      │    │
│  │ int32_t, 用于 kernel 中的 paged_off() 查表              │    │
│  └──────────────────────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │ Key Pool    [num_blocks, page_size, kv_dim]              │    │
│  │ 所有层共享的 Key 物理块池                                 │    │
│  └──────────────────────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │ Value Pool  [num_blocks, page_size, kv_dim]              │    │
│  │ 所有层共享的 Value 物理块池                               │    │
│  └──────────────────────────────────────────────────────────┘    │
│                           ↓                                      │
│            Paged Attention CUDA Kernels                           │
│  ┌─────────────────────┐ ┌──────────────────────────────────┐    │
│  │ Paged KV Write      │ │ Paged Flash Attention            │    │
│  │ paged_copy_kv_*()   │ │ paged_prefill_*/paged_decode_*() │    │
│  │ 写入单个 KV token   │ │ 读取 KV 进行注意力计算           │    │
│  └─────────────────────┘ └──────────────────────────────────┘    │
└───────────────────────────────────────────────────────────────────┘
```

---

## 第二章：PagedAttention 的适配过程、关键点与难点解决

### 2.1 适配的整体思路

PagedAttention 的适配遵循自底向上的分层设计：

```
第 1 层：基础设施层   — PagedKVCacheManager（内存管理器）
第 2 层：CUDA Kernel  — 7 个 paged kernel（地址翻译 + 注意力计算）
第 3 层：Layer 适配   — FlashAttentionDecodeLayer / PrefillLayer / GpuPosLayer 增加 paged 分支
第 4 层：Model 适配   — qwen3.cpp / qwen2.cpp / qwen3_vl.cpp 的 9 个函数修改
第 5 层：CLI 适配     — --paged-attention 命令行参数
```

### 2.2 第一步：设计 PagedKVCacheManager

**关键设计决策**：

1. **Page Size = 16**：选择 16 作为页大小，因为：
   - 2 的幂次，允许使用位运算 `>>4` 和 `&15` 替代除法/取模
   - 16 × 128 (kv_dim per head) × 2 (FP16) = 4KB，恰好对齐 GPU L2 cache line
   - 不会太大（浪费内存）也不会太小（块表过多）

2. **所有层共享物理块池**：一个大的 key_pool 和 value_pool，通过 block table 实现逻辑→物理映射。这使得物理块可以被任意层使用。

3. **CPU 管理 + GPU 查表**：block table 的分配逻辑在 CPU 端执行（简单、确定性强），GPU 端只做只读查表。

**代码实现**（新建文件）：

- `kuiper/include/base/paged_kv_cache.h`：类声明，约 150 行
- `kuiper/source/base/paged_kv_cache.cpp`：实现，约 146 行

### 2.3 第二步：实现 Paged Attention CUDA Kernel

**需要实现的 7 个 kernel**：

| 编号 | Kernel 名称 | 用途 | 线程配置 |
|------|-------------|------|----------|
| 1 | `paged_prefill_fp32_kernel` | FP32 prefill (CUB reduction) | grid(head_num, seq_len), block(256) |
| 2 | `paged_decode_fp32_kernel` | FP32 decode (warp shuffle) | grid(head_num), block(256) |
| 3 | `paged_decode_fp16_kernel` | FP16 decode (full softmax) | grid(head_num), block(256) |
| 4 | `paged_prefill_fp16_kernel` | FP16 prefill (online softmax) | grid(head_num, seq_len), block(128) |
| 5 | `paged_decode_fp16_gpu_pos_kernel` | FP16 decode + GPU pos (CUDA Graph) | grid(head_num), block(128) |
| 6 | `paged_copy_kv_fp32_cu` | FP32 KV 写入 | grid(ceil(kv_dim/256)), block(256) |
| 7 | `paged_copy_kv_fp16_cu` | FP16 KV 写入 | grid(ceil(kv_dim/256)), block(256) |

**关键难点 1：Kernel 内部的地址翻译开销**

与连续 KV Cache 不同，Paged Kernel 中每次访问 K 或 V 都需要经过 `paged_off()` 函数查表。这意味着：

```cuda
// 连续版本：一次计算偏移，顺序访问
const float* k_ptr = key_cache + layer_idx * seq_len * kv_dim + k * kv_dim;

// Paged 版本：每次都要查表
const float* k_ptr = key_pool + paged_off(block_table, k, kv_dim) + head_offset;
```

**解决方案**：
- `paged_off()` 标记为 `__forceinline__`，消除函数调用开销
- 使用 `__ldg()` (read-only cache) 加载 KV 数据，利用 L2 cache 减少重复查表代价
- 对 FP16 decode 使用 4 路展开（k+0, k+1, k+2, k+3），减少地址翻译次数：

```cuda
// paged_attention_kernel.cu 第284-293行
for (; k + 3 < kv_len; k += 4) {
    const half* v0 = value_pool + paged_off(block_table, k+0, kv_dim) + head_offset;
    const half* v1 = value_pool + paged_off(block_table, k+1, kv_dim) + head_offset;
    const half* v2 = value_pool + paged_off(block_table, k+2, kv_dim) + head_offset;
    const half* v3 = value_pool + paged_off(block_table, k+3, kv_dim) + head_offset;
    acc += s_scores[k+0] * __half2float(__ldg(v0 + d));
    acc += s_scores[k+1] * __half2float(__ldg(v1 + d));
    acc += s_scores[k+2] * __half2float(__ldg(v2 + d));
    acc += s_scores[k+3] * __half2float(__ldg(v3 + d));
}
```

**关键难点 2：CUDA Graph 兼容的 GPU Position 读取**

CUDA Graph 要求 kernel 的所有参数在 capture 时固定。但 decode 阶段 pos 每一步都在变化。

**解决方案**：设计 `paged_decode_fp16_gpu_pos_kernel`，从 GPU 内存指针读取 pos：

```cuda
// paged_attention_kernel.cu 第447行
const int pos = *reinterpret_cast<const volatile int32_t*>(pos_ptr);
```

`volatile` 关键字确保每次都从内存读取，不会被编译器缓存到寄存器。同样，KV 写入也需要 GPU pos：

```cuda
// paged_copy_kv_fp16_cu kernel
int32_t position = *reinterpret_cast<const volatile int32_t*>(pos);
const int32_t* bt = block_table + layer_idx * max_blocks_per_seq;
size_t off = paged_off(bt, position, kv_dim);
kv_pool[off + idx] = src[idx];
```

**关键难点 3：online softmax 下的精度保证**

Prefill 和 GPU-pos Decode kernel 使用 online softmax（FlashAttention 风格），分 tile 处理：

```cuda
// paged_prefill_fp16_kernel 中的 online softmax
float correction = expf(row_max - m_new);  // 修正因子
acc_o *= correction;                        // 修正已有累加值
// ... 累加新 tile 的 V 贡献 ...
row_max = m_new;
row_sum = correction * row_sum + l_j;       // 更新全局 sum
```

为避免 `expf` 中的下溢，使用 Flush-to-Zero 阈值：

```cuda
constexpr float PG_SOFTMAX_FTZ = -20.0f;
float e = (v > PG_SOFTMAX_FTZ) ? expf(v) : 0.f;
```

### 2.4 第三步：Layer 层适配

需要修改三个 Layer 类以支持 paged 模式：

**FlashAttentionDecodeLayer**（`flash_attention.h/cpp`）：

增加 paged 模式成员变量和 `set_paged_mode()` 方法：

```cpp
// flash_attention.h 第52-63行
void set_paged_mode(bool paged, int32_t page_size, int32_t max_blocks_per_seq,
                    const void* key_pool, const void* value_pool,
                    const int32_t* block_table) {
    paged_mode_ = paged;
    page_size_ = page_size;
    max_blocks_per_seq_ = max_blocks_per_seq;
    key_pool_ = key_pool;
    value_pool_ = value_pool;
    block_table_ = block_table;
}
```

在 `forward()` 方法中，paged 分支放在最前面：

```cpp
// flash_attention.cpp 第51-78行
if (paged_mode_) {
    if (use_fp16_) {
        if (use_gpu_pos_) {
            // CUDA Graph 路径：GPU pos + paged
            kernel::paged_flash_attention_decode_fp16_gpu_pos_cu(...);
        } else {
            // 普通路径：CPU pos + paged
            kernel::paged_flash_attention_decode_fp16_cu(...);
        }
    } else {
        kernel::paged_flash_attention_decode_cu(...);
    }
    return base::error::Success();
}
// ... 后续是原来的连续 KV cache 路径 ...
```

同样修改了：
- `FlashAttentionPrefillLayer`（prefill 阶段）
- `FlashAttentionDecodeGpuPosLayer`（misc_layers.h/cpp，VL 模型专用）

**关键设计决策：先 paged 后 contiguous**

所有 forward() 函数中，paged 分支放在开头并提前 return，这确保了：
1. paged 模式不会影响原有连续路径
2. 清晰的 if-return 结构避免嵌套
3. 原有代码几乎不需要修改

### 2.5 第四步：Model 层适配

这是工作量最大的一步，需要修改三个模型文件的 9 个关键函数。以 Qwen3 为例：

#### (1) `init_mem()` — 内存分配

```cpp
// qwen3.cpp 第868-892行
if (use_paged_attention_) {
    // 创建 PagedKVCacheManager
    paged_kv_cache_manager_ = std::make_unique<base::PagedKVCacheManager>(
        config_->layer_num_, base::PagedKVCacheManager::kDefaultPageSize,
        config_->kv_dim_, config_->seq_len_, activation_dtype, stream);

    // 创建 1 元素占位 buffer 防止 get_buffer() 崩溃
    tensor::Tensor key_cache(activation_dtype, 1, true, alloc);
    tensor::Tensor value_cache(activation_dtype, 1, true, alloc);
    CHECK(insert_buffer(ModelBufferType::kKeyCache, key_cache));
    CHECK(insert_buffer(ModelBufferType::kValueCache, value_cache));
} else {
    // 原有连续分配
    tensor::Tensor key_cache(activation_dtype, config_->layer_num_, config_->seq_len_,
                             config_->kv_dim_, true, alloc);
    ...
}
```

**难点：placeholder buffer**

原有代码中大量地方使用 `get_buffer(ModelBufferType::kKeyCache)`。在 paged 模式下，这个 buffer 不再有意义（实际数据在 key_pool 中），但如果不注册一个 buffer，调用会崩溃。

**解决方案**：注册一个 1 元素的 placeholder tensor，使得 `get_buffer()` 不会 crash，但 paged 路径不会使用它的内容。

#### (2) `attention_qkv()` — Decode 阶段的 KV 写入

```cpp
// qwen3.cpp 第1059-1062行
if (use_paged_attention_ && paged_kv_cache_manager_) {
    paged_kv_cache_manager_->ensure_allocated_to(pos);
    paged_kv_cache_manager_->sync_block_table();
}
auto [key, val] = slice_kv_cache(layer_idx, pos);
```

`ensure_allocated_to(pos)` 确保物理页已分配，`sync_block_table()` 同步到 GPU。然后 `slice_kv_cache()` 在 paged 模式下通过 `get_kv_byte_offset()` 计算物理地址。

#### (3) `attention_mha()` — Decode 阶段的注意力计算

使用 `configure_paged` lambda 在每次调用前配置 flash attention 层：

```cpp
// qwen3.cpp 第1126-1134行
auto configure_paged = [&](std::shared_ptr<op::FlashAttentionDecodeLayer>& layer) {
    if (use_paged_attention_ && paged_kv_cache_manager_) {
        auto* mgr = paged_kv_cache_manager_.get();
        layer->set_paged_mode(true, mgr->page_size(), mgr->max_blocks_per_seq(),
                              mgr->key_pool_gpu(), mgr->value_pool_gpu(), mgr->block_table_gpu());
    } else {
        layer->set_paged_mode(false, 16, 0, nullptr, nullptr, nullptr);
    }
};
```

#### (4) `attention_qkv_with_graph()` — CUDA Graph 路径的 KV 写入

这里是最大的难点之一。CUDA Graph 要求参数地址固定，但 KV cache 的写入位置每步都变。

**解决方案**：使用 paged copy kernel 从 GPU 内存读取 pos：

```cpp
// qwen3.cpp 第1333-1346行
if (use_paged_attention_ && paged_kv_cache_manager_) {
    auto* mgr = paged_kv_cache_manager_.get();
    cudaStream_t stream = cuda_config_->stream;
    kernel::paged_copy_to_kv_cache_kernel_fp16(
        static_cast<half*>(mgr->key_pool_gpu()), temp_key.ptr<half>(),
        pos_tensor.ptr<int32_t>(),    // GPU 内存上的 pos（固定地址）
        mgr->block_table_gpu(),       // GPU 内存上的 block table（固定地址）
        config_->kv_dim_, layer_idx,
        mgr->max_blocks_per_seq(), mgr->page_size(), stream);
    // ... 同样处理 value ...
}
```

所有参数的 GPU 指针在整个推理期间不变，满足 CUDA Graph 的要求。

#### (5) `batched_attention_qkv()` — Prefill 阶段的批量 KV 写入

Prefill 需要一次写入多个 token 的 KV：

```cpp
// qwen3.cpp 第1663-1679行
if (use_paged_attention_ && paged_kv_cache_manager_) {
    auto* mgr = paged_kv_cache_manager_.get();
    mgr->ensure_allocated_to(start_pos + seq_len - 1);  // 一次分配所有页
    mgr->sync_block_table();

    for (int i = 0; i < seq_len; ++i) {
        int32_t pos = start_pos + i;
        size_t byte_offset = mgr->get_kv_byte_offset(layer_idx, pos);

        void* v_dst = static_cast<char*>(mgr->value_pool_gpu()) + byte_offset;
        const void* v_src = value_out.get_buffer()->ptr() + i * config_->kv_dim_ * elem_size;
        cudaMemcpyAsync(v_dst, v_src, config_->kv_dim_ * elem_size,
                        cudaMemcpyDeviceToDevice, cuda_config_->stream);
        // ... 同样处理 key ...
    }
}
```

**难点**：每个 token 的物理位置不一定连续（跨页边界时地址不连续），因此必须逐 token 拷贝。

#### (6) `batched_attention_mha()` — Prefill 阶段的注意力计算

```cpp
// qwen3.cpp 第1722-1726行
if (use_paged_attention_ && paged_kv_cache_manager_) {
    auto* mgr = paged_kv_cache_manager_.get();
    prefill_layer->set_paged_mode(true, mgr->page_size(), mgr->max_blocks_per_seq(),
                                  mgr->key_pool_gpu(), mgr->value_pool_gpu(), mgr->block_table_gpu());
}
```

#### (7) `clear_kv_cache()` — 会话清理

```cpp
// qwen3.cpp 第2108-2112行
if (use_paged_attention_ && paged_kv_cache_manager_) {
    paged_kv_cache_manager_->clear();  // 重置所有页到 free list
    invalidate_cuda_graph();            // CUDA Graph 需要重新捕获
    return;
}
```

### 2.6 第五步：Qwen2 和 Qwen3-VL 模型适配

Qwen2 和 Qwen3 的区别：
- Qwen2 没有 Q/K norm，有 bias_add
- 其余 paged 逻辑完全相同

Qwen3-VL 的特殊性：
- 使用 M-RoPE（多维旋转位置编码），位置信息在 3D 空间中
- 有两套位置追踪：`rope_pos_gpu`（用于 RoPE）和 `kv_cache_pos_gpu`（用于 KV cache 位置）
- 使用 `FlashAttentionDecodeGpuPosLayer`（在 misc_layers.h 中）专门处理 GPU pos decode

### 2.7 适配过程中遇到的关键困难总结

| 困难 | 描述 | 解决方案 |
|------|------|----------|
| **Placeholder Buffer 崩溃** | paged 模式下 `get_buffer(kKeyCache)` 返回无效 buffer | 注册 1 元素占位 tensor |
| **CUDA Graph 地址固定** | KV write 的目标地址每步变化 | 用 GPU pos + block table 在 kernel 内部计算 |
| **Prefill 跨页拷贝** | 批量写入时物理地址不连续 | 逐 token 计算 byte_offset 后 cudaMemcpyAsync |
| **三种 forward 重载** | Layer 有 set_input/forward() 和直接参数 forward() 两种模式 | 两种都加 paged 分支 |
| **VL 双位置系统** | VL 模型的 RoPE pos 与 KV cache pos 不同 | KV write 用 kv_cache_pos_gpu |
| **FP32 + CUDA Graph** | 没有 FP32 GPU-pos paged kernel | Fall back 到 MHA with GPU pos |
| **数值一致性** | paged 和 contiguous 必须产生完全相同的输出 | 使用完全相同的 softmax 算法，仅改变数据访问模式 |

### 2.8 验证结果

所有 5 个模型配置均通过一致性验证：

| 模型 | 精度 | 基线输出 | Paged 输出 | 结果 |
|------|------|----------|-----------|------|
| Qwen3-8B | FP16 | `好的，用户让我介绍一下自己。首先，我需要按照之前的指示` | 完全相同 | ✅ |
| Qwen3-8B | AWQ INT4 | `好的，用户让我介绍一下自己。首先，我需要保持友好和` | 完全相同 | ✅ |
| Qwen2.5-7B | FP32 | `你好！我叫Qwen，是由阿里云开发的大型语言模型。` | 完全相同 | ✅ |
| Qwen2.5-7B | FP16 | `你好！我叫Qwen，是由阿里云开发的大型语言模型。` | 完全相同 | ✅ |
| Qwen3-VL-8B | FP16 | `这是一张充满温馨与宁静氛围的海滩照片。画面中，` | 完全相同 | ✅ |

---

## 第三章：结合源码详解 PagedAttention 的使用流程

### 3.1 端到端流程概览

以 `qwen3_infer --paged-attention --attention flash1` 为例，完整流程如下：

```
CLI 解析 → Model 初始化 → Prefill（批量处理prompt） → Decode（逐token生成）→ 清理
   ①            ②                   ③                         ④              ⑤
```

下面逐步骤详细解读。

### 3.2 步骤 ①：CLI 解析

**文件**：`demo/inference_common.h`

用户通过 `--paged-attention` 标志启用 PagedAttention：

```cpp
// inference_common.h 第201-202行
} else if (arg == "--paged-attention") {
    config.use_paged_attention = true;
}
```

配置打印时会显示：

```cpp
// inference_common.h 第225行
LOG(INFO) << "Paged Attention: " << (use_paged_attention ? "enabled" : "disabled");
```

### 3.3 步骤 ②：Model 初始化

**文件**：`demo/inference_common.h` → `kuiper/source/model/qwen3.cpp`

#### 3.3.1 启用 paged 模式（必须在 init 之前）

```cpp
// inference_common.h 第1708-1710行
if (config.use_paged_attention) {
    model.enable_paged_attention(true);  // 设置 use_paged_attention_ = true
}
auto init_status = model.init(base::DeviceType::kDeviceCUDA);
```

`enable_paged_attention()` 定义在 `model.h` 第68行：

```cpp
void enable_paged_attention(bool enable) { use_paged_attention_ = enable; }
```

这必须在 `init()` 之前调用，因为 `init()` → `init_mem()` 时需要根据此标志决定分配方式。

#### 3.3.2 内存分配（init_mem）

`init()` 最终调用 `init_mem()`。在 KV cache 分配阶段：

```cpp
// qwen3.cpp 第868-891行
if (use_paged_attention_) {
    cudaStream_t stream = (cuda_config_ ? cuda_config_->stream : nullptr);
    paged_kv_cache_manager_ = std::make_unique<base::PagedKVCacheManager>(
        config_->layer_num_,                          // 36（Qwen3-8B）
        base::PagedKVCacheManager::kDefaultPageSize,  // 16
        config_->kv_dim_,                             // 1024
        config_->seq_len_,                            // 8192
        activation_dtype,                              // FP16 或 FP32
        stream);
```

PagedKVCacheManager 构造函数的内部动作（`paged_kv_cache.cpp`）：

```
1. 验证参数(page_size 是 2 的幂, kv_dim>0 等)
2. 计算 max_blocks_per_seq = ceil(8192/16) = 512
3. 计算 num_blocks = 36 × 512 = 18,432
4. 初始化 block_table_cpu_[36*512] 全部填 -1（未分配）
5. 初始化 free_list_ = [18431, 18430, ..., 1, 0]
6. cudaMalloc block_table_gpu_（36 × 512 × 4 = 72 KB）
7. cudaMalloc key_pool_gpu_（18432 × 16 × 1024 × 2 = 576 MB）
8. cudaMalloc value_pool_gpu_（576 MB）
9. cudaMemset 全部清零
```

总 GPU 内存使用：576 + 576 + 0.07 ≈ **1,152 MB**（与连续方式相同总量，但可以按需分配页）

然后创建 placeholder：

```cpp
    tensor::Tensor key_cache(activation_dtype, 1, true, alloc);   // 1 个元素
    tensor::Tensor value_cache(activation_dtype, 1, true, alloc); // 1 个元素
    CHECK(insert_buffer(ModelBufferType::kKeyCache, key_cache));
    CHECK(insert_buffer(ModelBufferType::kValueCache, value_cache));
```

### 3.4 步骤 ③：Prefill 阶段

假设用户输入 "你好，请介绍一下你自己！"，经 tokenizer 编码后得到 34 个 tokens。

#### 3.4.1 Prefill 入口

推理框架对这 34 个 token 执行批量 prefill，遍历所有 36 层：

```
对每一层 layer_idx = 0, 1, ..., 35:
    batched_attention_rms(layer_idx, input, rms_out, seq_len)
    batched_attention_qkv(layer_idx, rms_out, query_out, key_out, value_out, seq_len=34, start_pos=0)
    batched_attention_mha(layer_idx, query_out, mha_out, seq_len=34, start_pos=0)
    batched_feed_forward(layer_idx, input, seq_len)
```

#### 3.4.2 Prefill KV 写入（batched_attention_qkv）

**文件**：`qwen3.cpp` 第1558行

首先执行 Q/K/V 矩阵乘法和 RoPE，然后写入 KV cache：

```cpp
// qwen3.cpp 第1660-1679行
if (use_paged_attention_ && paged_kv_cache_manager_) {
    auto* mgr = paged_kv_cache_manager_.get();
    // 步骤 A：分配页（pos=0 到 pos=33 的所有页）
    mgr->ensure_allocated_to(start_pos + seq_len - 1);  // ensure_allocated_to(33)
    // 步骤 B：同步 block table 到 GPU
    mgr->sync_block_table();
```

`ensure_allocated_to(33)` 的内部过程：

```
pos=0:  logical_block=0, 对 36 层各分配一个物理块（36 个块：0~35）
pos=1:  logical_block=0, 已分配，跳过
...
pos=15: logical_block=0, 已分配，跳过
pos=16: logical_block=1, 对 36 层各分配一个物理块（36 个块：36~71）
pos=17-31: logical_block=1, 已分配，跳过
pos=32: logical_block=2, 对 36 层各分配一个物理块（36 个块：72~107）
pos=33: logical_block=2, 已分配，跳过

总计分配：3 × 36 = 108 个物理块（而非 18,432 全部分配）
```

然后逐 token 拷贝到 paged KV pool：

```cpp
    for (int i = 0; i < seq_len; ++i) {   // i = 0..33
        int32_t pos = start_pos + i;       // pos = 0..33
        size_t byte_offset = mgr->get_kv_byte_offset(layer_idx, pos);

        // 写 Value
        void* v_dst = static_cast<char*>(mgr->value_pool_gpu()) + byte_offset;
        const void* v_src = value_out.get_buffer()->ptr() + i * config_->kv_dim_ * elem_size;
        cudaMemcpyAsync(v_dst, v_src, config_->kv_dim_ * elem_size,
                        cudaMemcpyDeviceToDevice, cuda_config_->stream);

        // 写 Key
        void* k_dst = static_cast<char*>(mgr->key_pool_gpu()) + byte_offset;
        const void* k_src = key_out.get_buffer()->ptr() + i * config_->kv_dim_ * elem_size;
        cudaMemcpyAsync(k_dst, k_src, config_->kv_dim_ * elem_size,
                        cudaMemcpyDeviceToDevice, cuda_config_->stream);
    }
}
```

`get_kv_byte_offset()` 的内部过程（以 layer_idx=0, pos=35 为例）：

```cpp
// paged_kv_cache.cpp 第134-141行
size_t PagedKVCacheManager::get_kv_byte_offset(int32_t layer_idx, int32_t pos) const {
    int32_t logical_block = pos / page_size_;     // 35/16 = 2
    int32_t block_offset = pos % page_size_;      // 35%16 = 3
    int32_t table_idx = layer_idx * max_blocks_per_seq_ + logical_block;  // 0*512+2 = 2
    int32_t physical_block = block_table_cpu_[table_idx];  // 例如 72
    size_t offset = ((size_t)physical_block * page_size_ + block_offset) * kv_dim_ * dtype_size_;
    // = (72 * 16 + 3) * 1024 * 2 = 2,363,392 bytes
    return offset;
}
```

#### 3.4.3 Prefill 注意力计算（batched_attention_mha）

**文件**：`qwen3.cpp` 第1696行

```cpp
// qwen3.cpp 第1720-1726行
auto prefill_layer = qwen_layers_->flash_attention_prefill_layer_;
if (use_paged_attention_ && paged_kv_cache_manager_) {
    auto* mgr = paged_kv_cache_manager_.get();
    prefill_layer->set_paged_mode(true, mgr->page_size(), mgr->max_blocks_per_seq(),
                                  mgr->key_pool_gpu(), mgr->value_pool_gpu(), mgr->block_table_gpu());
} else {
    prefill_layer->set_paged_mode(false, 16, 0, nullptr, nullptr, nullptr);
}
prefill_layer->set_cur_seq_len(seq_len);   // 34
prefill_layer->set_start_pos(start_pos);    // 0
prefill_layer->set_layer_index(layer_idx);
```

然后 `prefill_layer->forward()` 内部（`flash_attention.cpp` 第220-237行）：

```cpp
if (paged_mode_) {
    if (use_fp16_) {
        kernel::paged_flash_attention_prefill_fp16_cu(
            start_pos_, cur_seq_len_,     // 0, 34
            head_num_, kv_head_num_,      // 32, 8
            head_size_, kv_mul_,          // 128, 4
            layer_idx_, kv_dim_,          // 当前层, 1024
            page_size_, max_blocks_per_seq_,  // 16, 512
            query, output,
            key_pool_, value_pool_, block_table_,  // paged 池和块表
            cuda_config_.get());
    }
}
```

最终启动的 CUDA kernel：

```
grid  = (head_num=32, seq_len=34) = 1,088 个 block
block = (PG_BLOCK_FP16=128) threads
smem  = 128*2 + 1024*4 = 4,352 bytes
```

kernel 内部每个 query token 的计算：

```cuda
// paged_prefill_fp16_kernel
const int cur_pos = start_pos + seq_idx;  // 当前 query 位置
const int kv_len = cur_pos + 1;           // 需要 attend 的 KV 长度
// 对 kv_len 个 KV token 分 tile 计算：
for (tile_start = 0; tile_start < kv_len; tile_start += PG_TILE_K) {
    // 通过 paged_off() 查表获取每个 KV token 的物理地址
    const float4* kf4 = reinterpret_cast<const float4*>(
        key_pool + paged_off(block_table, kv_pos, kv_dim) + head_offset);
    // ... 计算 Q·K, softmax, 累加 V ...
}
```

### 3.5 步骤 ④：Decode 阶段

Prefill 完成后进入逐 token 生成。每步生成一个 token，直到达到 max_tokens 或遇到 EOS。

#### 3.5.1 非 CUDA Graph 路径

对于每个新 token（假设当前 pos=34）：

**attention_qkv() — KV 写入**：

```cpp
// qwen3.cpp 第1059-1062行
if (use_paged_attention_ && paged_kv_cache_manager_) {
    paged_kv_cache_manager_->ensure_allocated_to(pos);  // pos=34
    paged_kv_cache_manager_->sync_block_table();
}
auto [key, val] = slice_kv_cache(layer_idx, pos);
```

`ensure_allocated_to(34)`：pos=34 对应 logical_block=2，该块在 prefill 时已分配（pos=32 时分配），所以直接跳过。

`slice_kv_cache()` 返回指向 paged pool 中正确物理位置的 tensor view（`model.cpp` 第351-376行）：

```cpp
if (use_paged_attention_ && paged_kv_cache_manager_) {
    auto* mgr = paged_kv_cache_manager_.get();
    size_t byte_offset = mgr->get_kv_byte_offset(layer_idx, token_pos);
    // byte_offset 指向 physical_block 中的正确偏移

    uint16_t* key_ptr = reinterpret_cast<uint16_t*>(
        static_cast<char*>(mgr->key_pool_gpu()) + byte_offset);
    uint16_t* val_ptr = reinterpret_cast<uint16_t*>(
        static_cast<char*>(mgr->value_pool_gpu()) + byte_offset);

    // 返回 tensor view（不拥有内存，只是指向 pool 中的一段）
    tensor::Tensor key(base::DataType::kDataTypeFp16, config_->kv_dim_, false, nullptr, key_ptr);
    tensor::Tensor val(base::DataType::kDataTypeFp16, config_->kv_dim_, false, nullptr, val_ptr);
    return {key, val};
}
```

然后 Wk/Wv 矩阵乘法的结果直接写入这些 tensor view，等效于写入 paged pool 的正确位置。

**attention_mha() — 注意力计算**：

```cpp
// qwen3.cpp 第1126-1134行
auto configure_paged = [&](auto& layer) {
    if (use_paged_attention_ && paged_kv_cache_manager_) {
        auto* mgr = paged_kv_cache_manager_.get();
        layer->set_paged_mode(true, mgr->page_size(), mgr->max_blocks_per_seq(),
                              mgr->key_pool_gpu(), mgr->value_pool_gpu(), mgr->block_table_gpu());
    }
};

auto flash_attn = qwen_layers_->flash_attention_decode_layer_;
configure_paged(flash_attn);
flash_attn->set_layer_index(layer_idx);
flash_attn->set_pos(pos);               // pos=34
flash_attn->set_use_gpu_pos(false);     // CPU pos 路径
// ...
STATUS_CHECK(flash_attn->forward());
```

`forward()` 最终调用：

```cpp
kernel::paged_flash_attention_decode_fp16_cu(
    pos_,                           // 34
    head_num_, kv_head_num_,        // 32, 8
    head_size_, kv_mul_,            // 128, 4
    layer_index_, kv_dim_,          // 当前层, 1024
    page_size_, max_blocks_per_seq_, // 16, 512
    query, mha_output,
    key_pool_, value_pool_, block_table_,
    cuda_config_.get());
```

kernel 配置：
```
grid  = (head_num=32)
block = (PG_DECODE_BLOCK=256)
smem  = 128*2 + ceil(35/256)*256*4 + 2*8*4 = 1,344 bytes
```

kernel 遍历 kv_len=35 个 KV token，对每个计算 Q·K 和加权 V。

#### 3.5.2 CUDA Graph 路径

当启用 `--cuda-graph` 时，decode 路径使用 `attention_qkv_with_graph()` 和 `attention_mha_with_graph()`。

**KV 写入（attention_qkv_with_graph）**：

```cpp
// qwen3.cpp 第1333-1343行
if (use_paged_attention_ && paged_kv_cache_manager_) {
    auto* mgr = paged_kv_cache_manager_.get();
    cudaStream_t stream = cuda_config_->stream;
    // 使用 paged copy kernel（从 GPU 内存读 pos，CUDA Graph 安全）
    kernel::paged_copy_to_kv_cache_kernel_fp16(
        static_cast<half*>(mgr->key_pool_gpu()),   // 目标：key pool（固定地址）
        temp_key.ptr<half>(),                        // 源：临时 key buffer（固定地址）
        pos_tensor.ptr<int32_t>(),                   // pos：GPU 内存（固定地址，值每步更新）
        mgr->block_table_gpu(),                      // block table（固定地址，值预同步）
        config_->kv_dim_, layer_idx,
        mgr->max_blocks_per_seq(), mgr->page_size(), stream);
```

`paged_copy_kv_fp16_cu` kernel 的工作方式：

```cuda
__global__ void paged_copy_kv_fp16_cu(
    half* kv_pool, const half* src, const int32_t* pos,
    const int32_t* block_table, int32_t kv_dim,
    int32_t layer_idx, int32_t max_blocks_per_seq)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < kv_dim) {
        // 从 GPU 内存读取当前 position（每步不同，但地址固定）
        int32_t position = *reinterpret_cast<const volatile int32_t*>(pos);
        // 查 block table 找到物理块
        const int32_t* bt = block_table + layer_idx * max_blocks_per_seq;
        size_t off = paged_off(bt, position, kv_dim);
        // 写入 KV pool
        kv_pool[off + idx] = src[idx];
    }
}
```

所有指针参数（kv_pool, src, pos, block_table）的 GPU 地址在 CUDA Graph capture 时固定，但其指向的值可以变化，这使得同一段 captured graph 可以复用。

**注意力计算（attention_mha_with_graph）**：

```cpp
// qwen3.cpp 第1405-1416行
auto flash_attn = qwen_layers_->flash_attention_decode_layer_;
configure_paged(flash_attn);
flash_attn->set_layer_index(layer_idx);
flash_attn->set_use_gpu_pos(true);                // 标记使用 GPU pos
flash_attn->set_input(4, pos_tensor_gpu);          // GPU 上的 pos tensor
STATUS_CHECK(flash_attn->forward());
```

`forward()` 中 paged + gpu_pos 分支：

```cpp
// flash_attention.cpp 第55-62行
if (paged_mode_) {
    if (use_fp16_) {
        if (use_gpu_pos_) {
            const tensor::Tensor& pos_tensor = get_input(4);
            kernel::paged_flash_attention_decode_fp16_gpu_pos_cu(
                pos_tensor.ptr<int32_t>(),   // GPU pos 指针
                head_num_, kv_head_num_, head_size_, kv_mul_,
                layer_index_, kv_dim_, page_size_, max_blocks_per_seq_,
                query, mha_output, key_pool_, value_pool_, block_table_,
                cuda_config_.get());
```

`paged_decode_fp16_gpu_pos_kernel` 的关键差异是使用 online softmax（固定 shared memory 大小），使得 Graph 可以捕获：

```cuda
// paged_attention_kernel.cu 第448行
const int pos = *reinterpret_cast<const volatile int32_t*>(pos_ptr);
const int kv_len = pos + 1;
// 使用 online softmax，shared memory 大小固定为 PG_ONLINE_TILE_K
// 不依赖于 kv_len，满足 CUDA Graph 要求
```

### 3.6 步骤 ⑤：会话清理

当交互模式中开始新一轮对话时：

```cpp
// qwen3.cpp 第2107-2112行
void Qwen3Model::clear_kv_cache() {
    if (use_paged_attention_ && paged_kv_cache_manager_) {
        paged_kv_cache_manager_->clear();
        invalidate_cuda_graph();
        return;
    }
    // ... 原有连续清理逻辑 ...
}
```

`clear()` 的内部动作（`paged_kv_cache.cpp` 第109-132行）：

```
1. std::fill(block_table_cpu_.begin(), end, -1)  — 重置所有映射
2. free_list_ 重新填满 [N-1, ..., 0]             — 所有块回到空闲
3. allocated_pos_ = -1                             — 重置分配状态
4. cudaMemsetAsync(key_pool, 0)                    — 清零 GPU 内存
5. cudaMemsetAsync(value_pool, 0)                  — 清零 GPU 内存
6. cudaMemsetAsync(block_table_gpu, 0xFF)          — GPU 端也重置为 -1
7. cudaStreamSynchronize(stream)                    — 等待完成
```

### 3.7 slice_kv_cache() 的双路径详解

`slice_kv_cache()` 是 Decode 阶段 KV 写入的核心桥梁函数（`model.cpp` 第348行）。它在 paged 和 contiguous 两种模式下返回不同的 tensor view：

```
┌─────────────────────────────┐
│     slice_kv_cache()         │
│     (layer_idx, pos)         │
├─────────────┬───────────────┤
│  Paged 路径  │ Contiguous    │
│              │ 路径          │
│ get_kv_byte_ │ layer_offset  │
│ offset()     │ = layer_idx * │
│ → byte_offset│   seq_len *   │
│              │   kv_dim      │
│ key_pool +   │ key_cache +   │
│ byte_offset  │ cache_offset  │
│              │               │
│ 返回 tensor  │ 返回 tensor   │
│ view 指向    │ view 指向     │
│ paged pool   │ contiguous    │
│ 物理位置     │ buffer        │
└─────────────┴───────────────┘
```

Paged 路径中 `get_kv_byte_offset()` 通过 CPU 端的 block table 查表完成地址翻译，返回的 tensor view 直接指向 pool 中的物理位置。Wk/Wv 的输出直接写入此处。

### 3.8 运行指令

以下是启用 PagedAttention 的各模型运行指令：

```bash
# Qwen3-8B FP16
./qwen3_infer /mnt/ssd/QwenModels/Qwen3-8B-fp16.bin \
    /mnt/ssd/QwenModels/Qwen3-8B/tokenizer.json \
    --attention flash1 --paged-attention --prefix-cache --interactive

# Qwen3-8B AWQ
./qwen3_infer /mnt/ssd/QwenModels/Qwen3-8B-awq.bin \
    /mnt/ssd/QwenModels/Qwen3-8B-awq/tokenizer.json \
    --attention flash1 --paged-attention --prefix-cache --interactive

# Qwen2.5-7B FP32
./qwen_infer /mnt/ssd/QwenModels/Qwen2.5-7B.bin \
    /mnt/ssd/QwenModels/Qwen2.5-7B-Instruct/tokenizer.json \
    --attention flash1 --paged-attention --prefix-cache --interactive

# Qwen2.5-7B FP16
./qwen_infer /mnt/ssd/QwenModels/Qwen2.5-7B-fp16.bin \
    /mnt/ssd/QwenModels/Qwen2.5-7B-Instruct/tokenizer.json \
    --attention flash1 --paged-attention --prefix-cache --interactive

# Qwen3-VL-8B FP16
./qwen3_vl_infer /mnt/ssd/QwenModels/Qwen3-VL-8B-fp16.bin \
    /mnt/ssd/QwenModels/Qwen3-VL-8B-Instruct/tokenizer.json \
    --image /path/to/image.jpg --attention flash1 --paged-attention --cuda-graph
```

### 3.9 文件清单与修改摘要

| 文件 | 状态 | 关键内容 |
|------|------|----------|
| `kuiper/include/base/paged_kv_cache.h` | **新建** | PagedKVCacheManager 类声明 |
| `kuiper/source/base/paged_kv_cache.cpp` | **新建** | 构造/析构/分配/同步/清理实现 |
| `kuiper/source/op/kernels/cuda/paged_attention_kernel.cuh` | **新建** | 7 个 kernel 接口声明 |
| `kuiper/source/op/kernels/cuda/paged_attention_kernel.cu` | **新建** | 5 个注意力 + 2 个 KV 写入 kernel |
| `kuiper/include/op/flash_attention.h` | **修改** | Decode/Prefill Layer 增加 paged 成员 |
| `kuiper/source/op/flash_attention.cpp` | **修改** | 两种 forward() 重载增加 paged 分支 |
| `kuiper/include/op/misc_layers.h` | **修改** | GpuPosLayer 增加 paged 成员 |
| `kuiper/source/op/misc_layers.cpp` | **修改** | GpuPosLayer forward 增加 paged 分支 |
| `kuiper/include/model/model.h` | **修改** | 基类增加 paged 管理接口 |
| `kuiper/source/model/model.cpp` | **修改** | slice_kv_cache() 增加 paged 路径 |
| `kuiper/source/model/qwen3.cpp` | **修改** | 9 个函数的 paged 适配 |
| `kuiper/source/model/qwen2.cpp` | **修改** | 9 个函数的 paged 适配 |
| `kuiper/source/model/qwen3_vl.cpp` | **修改** | 9 个函数的 paged 适配 |
| `demo/inference_common.h` | **修改** | --paged-attention CLI 参数 |
| `demo/main_qwen3_vl.cpp` | **修改** | VL 模型 --paged-attention 参数 |

---

## 第四章：PagedAttention CUDA Kernel 详细解读与优化分析

### 4.1 Kernel 全景

本工程在 `kuiper/source/op/kernels/cuda/paged_attention_kernel.cu`（742行）中实现了 7 个 CUDA Kernel，按功能分为三类：

| 类别 | Kernel | 数据类型 | 线程配置 | Grid 维度 | 算法 |
|------|--------|----------|----------|----------|------|
| **Prefill Attention** | `paged_prefill_fp32_kernel` | FP32 | 256 threads | (head_num, seq_len) | CUB BlockReduce + online softmax |
| **Prefill Attention** | `paged_prefill_fp16_kernel` | FP16 | 128 threads | (head_num, seq_len) | warp shuffle + online softmax |
| **Decode Attention** | `paged_decode_fp32_kernel` | FP32 | 256 threads | (head_num) | warp shuffle + full softmax |
| **Decode Attention** | `paged_decode_fp16_kernel` | FP16 | 256 threads | (head_num) | warp shuffle + full softmax |
| **Decode Attention** | `paged_decode_fp16_gpu_pos_kernel` | FP16 | 128 threads | (head_num) | online softmax（CUDA Graph 兼容） |
| **KV Write** | `paged_copy_kv_fp32_cu` | FP32 | 256 threads | (ceil(kv_dim/256)) | 逐元素写入 |
| **KV Write** | `paged_copy_kv_fp16_cu` | FP16 | 256 threads | (ceil(kv_dim/256)) | 逐元素写入 |

### 4.2 核心基础：`paged_off()` 地址翻译函数

所有 Paged Kernel 的基石是 `paged_off()` 设备函数：

```cuda
// paged_attention_kernel.cu 第43-45行
__device__ __forceinline__ size_t paged_off(const int32_t* bt, int32_t pos, int32_t kv_dim) {
  return ((size_t)bt[pos >> PAGE_SHIFT] * PAGE_SIZE + (pos & PAGE_MASK)) * kv_dim;
}
```

**逐步解析**：

```
输入：bt = block_table (当前层的块表指针), pos = token位置, kv_dim = KV向量维度
  1. pos >> PAGE_SHIFT  → pos / 16 = 逻辑块号
  2. bt[逻辑块号]       → 物理块号（从GPU block table读取）
  3. 物理块号 * PAGE_SIZE + (pos & PAGE_MASK)  → 物理块内偏移 + 块基地址
  4. × kv_dim           → 最终元素偏移（非字节偏移）
```

**优化要点**：
- `__forceinline__`：消除函数调用开销，编译后直接内联到调用点
- 位运算 `>>4` 和 `&15`：替代除法和取模，单周期完成
- `(size_t)` 类型转换：防止 32 位溢出（大模型场景下偏移可能超过 4GB）

与连续 KV cache 的地址计算对比：

```cuda
// 连续版本（flash_attention_kernel.cu）：
const float* k_ptr = K_cache + kv_pos * kv_dim + head_offset;
// 1次乘法 + 1次加法

// Paged 版本：
const float* k_ptr = key_pool + paged_off(block_table, kv_pos, kv_dim) + head_offset;
// 1次移位 + 1次查表 + 1次乘法 + 1次加法 + 1次掩码
```

额外开销约 2-3 条指令（移位 + 查表 + 掩码），但由于 block_table 在 L2 cache 中命中率极高（整个 block_table 仅 512 × 4 = 2KB），实际延迟可忽略。

### 4.3 Kernel 1：FP32 Prefill Kernel 详解

```cuda
__global__ void paged_prefill_fp32_kernel(
    const float* Q, const float* key_pool, const float* value_pool,
    float* O, const int32_t* block_table,
    const int seq_len, const int start_pos,
    const int head_num, const int kv_head_num,
    const int head_size, const int kv_mul,
    const int dim, const int kv_dim, const float scale)
```

**Grid/Block 配置**：
- Grid: `(head_num, seq_len)` — 每个 block 处理一个 (head, query_token) 对
- Block: 256 threads
- Shared Memory: `(head_size + PG_TILE_K) × sizeof(float)`

**算法流程**：

```
Phase 1: 加载 Query → s_query[head_size]
Phase 2: 分 tile 处理 K（每 tile 1024 个 KV token）
  ├── 2a: Q·K 点积 → s_scores[]（float4 向量化）
  ├── 2b: CUB BlockReduce 求 tile_max
  ├── 2c: online softmax correction: acc_o *= exp(old_max - new_max)
  ├── 2d: exp(score - new_max) 并求 tile_sum
  └── 2e: 累加 V 权重: acc_o += score * V[k]
Phase 3: 最终归一化 o = acc_o / row_sum
```

**关键代码（Q·K 点积，使用 float4 向量化）**：

```cuda
// paged_attention_kernel.cu 第94-101行
const float* k_ptr = key_pool + paged_off(block_table, kv_pos, kv_dim) + head_offset;
float score = 0.f;
const float4* sq4 = reinterpret_cast<const float4*>(s_query);
const float4* kk4 = reinterpret_cast<const float4*>(k_ptr);
#pragma unroll
for (int d = 0; d < head_size / 4; d++) {
    float4 q = sq4[d]; float4 kk = __ldg(kk4 + d);
    score += q.x*kk.x + q.y*kk.y + q.z*kk.z + q.w*kk.w;
}
```

**CUB BlockReduce 求 tile_max**：

```cuda
// paged_attention_kernel.cu 第107-112行
typedef cub::BlockReduce<float, PG_BLOCK_FP32> BR;
__shared__ typename BR::TempStorage ts;
float bm = BR(ts).Reduce(tile_max, cub::Max());
__shared__ float s_tm;
if (tid == 0) s_tm = bm;
__syncthreads();
```

CUB 是 NVIDIA 官方库，`BlockReduce` 的 `Reduce(val, Max())` 在 block 内做高效归约，利用 warp shuffle 避免 shared memory bank conflicts。

**Online Softmax 修正（跨 tile 累积）**：

```cuda
// paged_attention_kernel.cu 第120-121行
float correction = expf(row_max - m_new);   // 修正因子
// ...
acc_o[i] *= correction;  // 把之前 tile 的 V 累积值按新 max 修正
for (int k = 0; k < tile_len; k++) {
    const float* v_ptr = value_pool + paged_off(block_table, tile_start + k, kv_dim) + head_offset;
    acc_o[i] += s_scores[k] * __ldg(v_ptr + d);
}
row_max = m_new; row_sum = l_new;
```

这就是 FlashAttention 的核心算法：通过修正因子 `exp(old_max - new_max)`，无需存储完整 attention score 矩阵，实现 O(1) 辅助空间的 attention 计算。

### 4.4 Kernel 2：FP32 Decode Kernel 详解

Decode 阶段每步只有 1 个 query token，因此 Grid 降为 1D：

```cuda
// Grid: (head_num), Block: (256)
```

与 Prefill 不同，Decode kernel 使用**全量 softmax**（非 online），因为 decode 阶段序列长度已知且 score 可以存入 shared memory：

```cuda
// Phase 1: 计算所有 Q·K scores → s_scores[kv_len]
// Phase 2: warp shuffle 求 global_max
// Phase 3: exp + warp shuffle 求 global_sum
// Phase 4: 加权求和 V
```

**Warp Shuffle 归约（替代 CUB）**：

```cuda
// paged_attention_kernel.cu 第178-181行
#pragma unroll
for (int o = 16; o > 0; o /= 2)
    lmax = fmaxf(lmax, __shfl_down_sync(0xffffffff, lmax, o));
if (lane == 0) s_max[warp] = lmax;
```

这里采用两级归约：
1. **Warp 内**：`__shfl_down_sync` 在 32 线程内做 log₂(32)=5 步蝶形归约
2. **Warp 间**：8 个 warp 结果写入 shared memory，再由单个 warp 归约

### 4.5 Kernel 3：FP16 Decode Kernel 详解（最常用）

这是生产推理中最核心的 kernel（FP16 Decode 占推理计算量的 90%+）。

**关键特性**：
- 256 线程 / 8 warps
- half2 向量化 Q·K 点积
- `__shfl_xor_sync` 替代 `__shfl_down_sync`
- 4 路循环展开 V 累积
- Flush-to-Zero 精度保护

**Q·K 点积（half2 向量化）**：

```cuda
// paged_attention_kernel.cu 第250-267行
const float4* kf4 = reinterpret_cast<const float4*>(
    key_pool + paged_off(block_table, k, kv_dim) + head_offset);
const float4* qf4 = reinterpret_cast<const float4*>(s_query);
float2 acc = make_float2(0.f, 0.f);
#pragma unroll
for (int d = 0; d < head_size / 8; d++) {
    float4 qp = qf4[d]; float4 kp = __ldg(kf4 + d);
    const half2* qh = reinterpret_cast<const half2*>(&qp);
    const half2* kh = reinterpret_cast<const half2*>(&kp);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float2 qf = __half22float2(qh[i]), kf = __half22float2(kh[i]);
        acc.x += qf.x * kf.x; acc.y += qf.y * kf.y;
    }
}
float score = (acc.x + acc.y) * scale;
```

**数据访问模式**：将 `float4`（128-bit）加载视为 4 个 `half2` 对，每次处理 8 个 half 元素。对于 `head_size=128`，循环仅需 16 次迭代。

**4 路 V 累积展开**：

```cuda
// paged_attention_kernel.cu 第284-298行
int k = 0;
for (; k + 3 < kv_len; k += 4) {
    const half* v0 = value_pool + paged_off(block_table, k+0, kv_dim) + head_offset;
    const half* v1 = value_pool + paged_off(block_table, k+1, kv_dim) + head_offset;
    const half* v2 = value_pool + paged_off(block_table, k+2, kv_dim) + head_offset;
    const half* v3 = value_pool + paged_off(block_table, k+3, kv_dim) + head_offset;
    acc += s_scores[k+0] * __half2float(__ldg(v0 + d));
    acc += s_scores[k+1] * __half2float(__ldg(v1 + d));
    acc += s_scores[k+2] * __half2float(__ldg(v2 + d));
    acc += s_scores[k+3] * __half2float(__ldg(v3 + d));
}
for (; k < kv_len; k++) {  // 处理余数
    const half* vp = value_pool + paged_off(block_table, k, kv_dim) + head_offset;
    acc += s_scores[k] * __half2float(__ldg(vp + d));
}
```

展开的好处：
1. **隐藏内存延迟**：4 个 `__ldg` 同时发射，GPU 可以 pipeline 这些内存请求
2. **减少循环开销**：循环迭代次数减为 1/4
3. **指令级并行**：编译器可以交叉调度 4 个独立的乘加操作

### 4.6 Kernel 4：FP16 Prefill Kernel 详解

```cuda
// Grid: (head_num, seq_len), Block: 128 threads (4 warps)
```

**与 FP32 Prefill 的差异**：
- 使用 `fmaf()` 替代分步乘加（Fused Multiply-Add，单周期完成）
- 线程数减少到 128（因 FP16 指令吞吐量更高，不需要那么多线程）
- V 累积不使用多维寄存器数组，而是单变量 `acc_o`（`if (tid < head_size)` 保护）

**FMA 指令优化**：

```cuda
// paged_attention_kernel.cu 第351-354行
#pragma unroll
for (int i = 0; i < 4; i++) {
    float2 qf = __half22float2(qh[i]), kf = __half22float2(kh[i]);
    acc.x = fmaf(qf.x, kf.x, acc.x); acc.y = fmaf(qf.y, kf.y, acc.y);
}
```

`fmaf(a, b, c) = a*b + c` 在 GPU 上是单条指令（FMA），比 `a*b + c` 分两步更快且精度更高（减少一次浮点舍入）。

**V 累积方式**：

```cuda
// paged_attention_kernel.cu 第394-401行
if (tid < head_size) {
    for (int k = 0; k < tile_len; k++) {
        int32_t kv_pos = tile_start + k;
        const half* v_ptr = value_pool + paged_off(block_table, kv_pos, kv_dim) + head_offset;
        acc_o = fmaf(s_scores[k], __half2float(__ldg(v_ptr + tid)), acc_o);
    }
}
```

由于 `head_size=128 = PG_BLOCK_FP16=128`，每个线程恰好处理一个输出维度，代码更简洁。

### 4.7 Kernel 5：FP16 Decode with GPU Position（CUDA Graph 专用）

这是最特殊的 kernel，专为 CUDA Graph 设计：

```cuda
__global__ void paged_decode_fp16_gpu_pos_kernel(
    ..., const int32_t* __restrict__ pos_ptr, ...)
```

**CUDA Graph 约束**：
- 所有 kernel 参数的**GPU 指针**在 capture 时固定
- 不允许按值传递每步变化的标量（如 `pos`）
- shared memory 大小不能依赖运行时变量

**解决方法 1：GPU 内存读 pos**

```cuda
// paged_attention_kernel.cu 第447行
const int pos = *reinterpret_cast<const volatile int32_t*>(pos_ptr);
const int kv_len = pos + 1;
```

`pos_ptr` 指向一个 GPU int32_t，每步 decode 前由 host 代码用 `cudaMemcpyAsync` 更新其值。`volatile` 防止编译器优化掉读取。

**解决方法 2：固定 shared memory 大小**

普通 decode kernel 的 shared memory 大小依赖 `kv_len`：
```cuda
// paged_decode_fp16_kernel host launch:
int score_buf = ((kv_len + PG_DECODE_BLOCK - 1) / PG_DECODE_BLOCK) * PG_DECODE_BLOCK;
int smem = head_size * sizeof(half) + score_buf * sizeof(float) + ...;
// smem 随 kv_len 变化！不兼容 CUDA Graph
```

GPU-pos kernel 改用 online softmax + 固定 tile：
```cuda
// paged_decode_fp16_gpu_pos_kernel host launch:
int smem = head_size * sizeof(half) + PG_ONLINE_TILE_K * sizeof(float)
         + 2 * PG_ONLINE_WARPS * sizeof(float);
// = 128×2 + 256×4 + 2×4×4 = 1312 字节，与 kv_len 无关
```

**Online softmax 的 tile 处理**：

```cuda
// paged_attention_kernel.cu 第468-530行
for (int tile_start = 0; tile_start < kv_len; tile_start += PG_ONLINE_TILE_K) {
    const int tile_len = min(tile_start + PG_ONLINE_TILE_K, kv_len) - tile_start;
    // 1. 计算当前 tile 的 Q·K scores
    // 2. warp shuffle 求 tile_max → m_j
    // 3. m_new = max(row_max, m_j)
    // 4. correction = exp(row_max - m_new)
    // 5. acc_o *= correction  (修正之前 tile 的 V 累积)
    // 6. 累加当前 tile 的 V
    // 7. 更新 row_max, row_sum
}
```

这样 shared memory 只需要 `PG_ONLINE_TILE_K=256` 个 float 存 score，而非 `kv_len` 个。

### 4.8 Kernel 6-7：Paged KV Write Kernel

最简单但不可或缺的 kernel，用于 CUDA Graph 路径下将 KV 写入 paged pool：

```cuda
// paged_attention_kernel.cu 第554-565行
__global__ void paged_copy_kv_fp16_cu(
    half* kv_pool, const half* src, const int32_t* pos,
    const int32_t* block_table, int32_t kv_dim,
    int32_t layer_idx, int32_t max_blocks_per_seq)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < kv_dim) {
    int32_t position = *reinterpret_cast<const volatile int32_t*>(pos);
    const int32_t* bt = block_table + layer_idx * max_blocks_per_seq;
    size_t off = paged_off(bt, position, kv_dim);
    kv_pool[off + idx] = src[idx];
  }
}
```

Grid = `ceil(kv_dim / 256)` = `ceil(1024/256)` = 4 个 block，共 1024 线程，每个线程拷贝 1 个元素。

与 `cudaMemcpy` 相比，用 kernel 写 KV 的优势：
1. **CUDA Graph 兼容**：所有参数地址固定
2. **自动地址翻译**：kernel 内部调用 `paged_off()` 算出目标地址
3. **流内执行**：无需同步，自然串入 kernel 执行流

### 4.9 Host Launch 函数

每个 kernel 都有对应的 host launch 函数，负责：
1. 计算 `layer_bt = block_table + layer_index * max_blocks_per_seq`（偏移到当前层的块表）
2. 设置 grid/block/shared memory 大小
3. 启动 kernel

以 FP16 Decode 为例：

```cuda
// paged_attention_kernel.cu 第653-668行
void paged_flash_attention_decode_fp16_cu(...) {
  const int32_t* layer_bt = block_table + layer_index * max_blocks_per_seq;
  const int kv_len = pos + 1;
  float scale = 1.0f / sqrtf((float)head_size);

  dim3 grid(head_num);
  dim3 block(PG_DECODE_BLOCK);  // 256
  int score_buf = ((kv_len + PG_DECODE_BLOCK - 1) / PG_DECODE_BLOCK) * PG_DECODE_BLOCK;
  int smem = head_size * sizeof(half)
           + score_buf * sizeof(float)
           + 2 * PG_DECODE_WARPS * sizeof(float);

  paged_decode_fp16_kernel<<<grid, block, smem, stream>>>(...);
}
```

### 4.10 对 Paged Attention Kernel 做的优化总结

下面对比本工程 Paged Kernel 与连续版 FA1 Kernel（`flash_attention_kernel.cu`）的差异和优化：

#### 优化 1：`__forceinline__` 地址翻译

```cuda
__device__ __forceinline__ size_t paged_off(const int32_t* bt, int32_t pos, int32_t kv_dim)
```

强制内联确保 `paged_off` 永远不会产生函数调用开销。编译器将位运算 + 查表 + 乘法直接内嵌到每个调用点。

#### 优化 2：位运算替代除法/取模

```cuda
constexpr int PAGE_SIZE = 16;
constexpr int PAGE_SHIFT = 4;   // log2(16)
constexpr int PAGE_MASK = 15;   // 16-1

pos >> PAGE_SHIFT   // 替代 pos / 16
pos & PAGE_MASK     // 替代 pos % 16
```

整数除法在 GPU 上需要 20+ 周期（没有硬件除法单元），位运算只需 1 周期。

#### 优化 3：`__ldg()` Read-Only Cache 加载

```cuda
float4 kk = __ldg(kk4 + d);     // 通过 texture/read-only cache 加载
acc += s_scores[k] * __half2float(__ldg(vp + d));
```

`__ldg()` 走 read-only data cache（与 L1/texture cache 共享），减轻 L1 data cache 压力。对于 KV pool 这种只读数据特别有效。

#### 优化 4：float4 / half2 向量化内存访问

```cuda
// FP32: float4 = 128-bit load, 减少 4x 内存事务
const float4* sq4 = reinterpret_cast<const float4*>(s_query);
const float4* kk4 = reinterpret_cast<const float4*>(k_ptr);

// FP16: float4 包含 4 个 half2 = 8 个 half = 128-bit load
const float4* kf4 = reinterpret_cast<const float4*>(...);
```

Orin 的内存总线以 128-bit 为粒度传输，向量化确保每次传输都是满带宽。

#### 优化 5：4 路循环展开（Value 累积）

```cuda
for (; k + 3 < kv_len; k += 4) {
    acc += s_scores[k+0] * __half2float(__ldg(v0 + d));
    acc += s_scores[k+1] * __half2float(__ldg(v1 + d));
    acc += s_scores[k+2] * __half2float(__ldg(v2 + d));
    acc += s_scores[k+3] * __half2float(__ldg(v3 + d));
}
```

4 个 `__ldg()` 同时发射，利用 GPU 的内存级并行（Memory Level Parallelism），可前后重叠内存请求和计算。

#### 优化 6：`__shfl_xor_sync` 替代 `__shfl_down_sync`

```cuda
// FP16 kernel 使用 xor：
for (int o = 16; o > 0; o /= 2)
    lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffff, lmax, o));

// FP32 kernel 使用 down：
for (int o = 16; o > 0; o /= 2)
    lmax = fmaxf(lmax, __shfl_down_sync(0xffffffff, lmax, o));
```

`__shfl_xor_sync` 实现蝶形归约（butterfly reduction），所有线程在每轮都参与且都获得归约值，避免后续需要 broadcast。`__shfl_down_sync` 只有 lane 0 获得最终结果，需要额外通信。

#### 优化 7：Flush-to-Zero (FTZ) 精度保护

```cuda
constexpr float PG_SOFTMAX_FTZ = -20.0f;
float e = (v > PG_SOFTMAX_FTZ) ? expf(v) : 0.f;
```

当 score 差距超过 20 时（`exp(-20) ≈ 2e-9`），直接置零，避免：
1. `expf` 对极小值的计算开销
2. 浮点下溢产生的 denormal 数（denormal 在 GPU 上处理极慢）

#### 优化 8：`fmaf()` 融合乘加指令

```cuda
acc.x = fmaf(qf.x, kf.x, acc.x);  // a*b+c 单条指令
```

FMA 相比独立 multiply + add：
- 少一次浮点舍入（更高精度）
- GPU 上 FMA 与普通乘法同吞吐量

#### 优化 9：Online Softmax 实现 O(1) 辅助空间

Prefill 和 GPU-pos Decode kernel 不需要为完整序列分配 score buffer，只需 tile 大小的 buffer：

```
普通 Decode: shared memory = O(kv_len) — 随序列增长
Online Decode: shared memory = O(tile_k) = O(256) — 固定常量
```

这使得 GPU-pos kernel 可以在任意长度序列上运行，且 shared memory 大小固定——CUDA Graph 的关键约束。

#### 优化 10：`volatile` GPU 内存读取

```cuda
const int pos = *reinterpret_cast<const volatile int32_t*>(pos_ptr);
```

`volatile` 告诉编译器"该值可能被外部修改"，阻止：
- 寄存器缓存（register caching）
- 指令重排（instruction reordering）
- 循环不变量外提（loop-invariant code motion）

确保 CUDA Graph replay 时每次都读到 host 更新的最新 pos 值。

### 4.11 Paged Kernel 与 Contiguous Kernel 的性能对比

理论分析（以 FP16 Decode, head_size=128, kv_len=1000 为例）：

| 指标 | Contiguous Kernel | Paged Kernel | 开销 |
|------|----------|-------|------|
| Q·K 每个 KV token 额外指令 | 0 | ~3（shift + mask + table lookup） | +3 指令/token |
| V 累积每个 KV token 额外指令 | 0 | ~3（同上） | +3 指令/token |
| block_table L2 cache miss | 0 | ~0（2KB 表常驻L2） | ≈0 |
| KV data locality | 完美连续 | 页内连续，跨页跳转 | 每 16 token 跳转 1 次 |

实际测试显示 Paged 与 Contiguous 生成完全相同的输出，性能损失 < 3%（地址翻译开销被内存延迟掩盖）。

---

## 第五章：PagedAttention 与连续 KV Cache 的内存管理对比

### 5.1 两种管理方式的本质区别

| 维度 | 连续 KV Cache | Paged KV Cache |
|------|-------------|---------------|
| **内存布局** | `[layer_num, max_seq_len, kv_dim]` 一整块 | `[num_blocks, page_size, kv_dim]` 分页块池 |
| **分配时机** | 模型初始化时一次性分配 | 推理过程中按需分配 |
| **地址计算** | `layer_offset + pos * kv_dim` 直接偏移 | `paged_off(block_table, pos, kv_dim)` 查表翻译 |
| **内存利用率** | 与 max_seq_len 成正比（可能浪费） | 与实际 token 数成正比（精确使用） |
| **多序列支持** | 需要预分配 batch_size × max_seq_len | 共享物理块池，按需分配 |
| **碎片化** | 无碎片（连续）但可能 OOM | 有页级碎片（最多浪费 page_size-1 个slot/layer） |

### 5.2 连续 KV Cache 的源码实现

#### 5.2.1 内存分配（init_mem 中的连续分配路径）

```cpp
// qwen3.cpp 第890-898行 — 连续 KV Cache 分配
} else {
    // Contiguous KV cache
    tensor::Tensor key_cache(activation_dtype, config_->layer_num_, config_->seq_len_,
                             config_->kv_dim_, true, alloc);
    tensor::Tensor value_cache(activation_dtype, config_->layer_num_, config_->seq_len_,
                               config_->kv_dim_, true, alloc);
    CHECK(insert_buffer(ModelBufferType::kKeyCache, key_cache));
    CHECK(insert_buffer(ModelBufferType::kValueCache, value_cache));
}
```

这里 `tensor::Tensor(dtype, layer_num, seq_len, kv_dim, true, alloc)` 创建一个三维张量并立即分配 GPU 内存：

```
内存大小 = layer_num × seq_len × kv_dim × dtype_size
         = 36 × 8192 × 1024 × 2 (FP16)
         = 576 MB（单个 cache）
总计 = 1152 MB
```

**关键点**：这 1152 MB 在 `init_mem()` 时就已完全分配，无论实际推理用了多少 token。

#### 5.2.2 地址计算（slice_kv_cache 的连续路径）

```cpp
// model.cpp 第388-413行 — 连续 KV Cache 地址计算
int32_t layer_offset = layer_idx * config_->seq_len_ * config_->kv_dim_;
int32_t cache_offset = layer_offset + token_pos * config_->kv_dim_;

const auto& key_cache_buffer = get_buffer(ModelBufferType::kKeyCache);
const auto& val_cache_buffer = get_buffer(ModelBufferType::kValueCache);

uint16_t* key_cache_ptr = const_cast<uint16_t*>(key_cache_buffer.ptr<uint16_t>(cache_offset));
uint16_t* val_cache_ptr = const_cast<uint16_t*>(val_cache_buffer.ptr<uint16_t>(cache_offset));
```

地址计算仅需一次乘法和一次加法，内存连续，GPU 可以完美预取。

#### 5.2.3 Attention Kernel 中的连续访问模式

```cuda
// flash_attention_kernel.cu 第257行 — 连续访问
const float* k_ptr = K_cache + k * kv_dim + head_offset;
```

`K_cache` 是一维连续数组，`k * kv_dim` 的步进是固定的，GPU 的 L2 cache 可以高效预取相邻内存行。

### 5.3 Paged KV Cache 的源码实现

#### 5.3.1 内存分配

```cpp
// qwen3.cpp 第872-877行 — Paged KV Cache 分配
paged_kv_cache_manager_ = std::make_unique<base::PagedKVCacheManager>(
    config_->layer_num_, base::PagedKVCacheManager::kDefaultPageSize,
    config_->kv_dim_, config_->seq_len_, activation_dtype, stream);
```

PagedKVCacheManager 构造函数（`paged_kv_cache.cpp`）：

```cpp
// paged_kv_cache.cpp 第47-52行
size_t pool_bytes = (size_t)num_blocks_ * page_size * kv_dim * dtype_size_;
cudaMalloc(&key_pool_gpu_, pool_bytes);     // 分配 key 块池
cudaMalloc(&value_pool_gpu_, pool_bytes);   // 分配 value 块池
cudaMemset(key_pool_gpu_, 0, pool_bytes);
cudaMemset(value_pool_gpu_, 0, pool_bytes);
```

**注意**：当前实现中 `num_blocks_ = num_layers * max_blocks_per_seq`，总内存与连续方式相同。这是因为当前为单序列推理优化。在多序列（batch serving）场景中，paged 方式的优势才充分体现：

```
单序列：paged 总内存 ≈ 连续总内存（1152 MB）
多序列：
  连续方式：batch_size × max_seq_len × 全量分配  → 可能 OOM
  Paged 方式：共享块池 × 按需分配              → 高效利用
```

#### 5.3.2 按需分配（ensure_allocated_to）

```cpp
// paged_kv_cache.cpp 第80-97行
void PagedKVCacheManager::ensure_allocated_to(int32_t pos) {
  if (pos <= allocated_pos_) return;  // 快速路径：已分配

  for (int32_t p = allocated_pos_ + 1; p <= pos; ++p) {
    int32_t logical_block = p / page_size_;
    for (int32_t layer = 0; layer < num_layers_; ++layer) {
      int32_t table_idx = layer * max_blocks_per_seq_ + logical_block;
      if (block_table_cpu_[table_idx] == -1) {
        block_table_cpu_[table_idx] = allocate_block();
      }
    }
  }
  allocated_pos_ = pos;
}
```

**与连续方式的核心差异**：
- 连续方式：**分配一次，永不回收，永不扩展**
- Paged 方式：**惰性分配 + 可以回收 + 可以扩展**

实际推理中的分配时间线：

```
连续方式：                    Paged 方式：
init_mem() → 分配 1152 MB   init_mem() → 分配 1152 MB 块池
                              （但 block table 全为 -1，未映射）

Prefill(34 tokens):           Prefill(34 tokens):
  无需额外操作                  ensure_allocated_to(33)
                                → 分配 3×36=108 个块（3 页 × 36 层）
                                → 仅映射 108/18432 = 0.6% 的块池

Decode(pos=34..100):          Decode(pos=34..100):
  无需额外操作                  ensure_allocated_to(34..100)
                                → pos 34-47 不需新块（block 2 已分配）
                                → pos 48 分配 block 3（36 个块）
                                → pos 64 分配 block 4（36 个块）
                                → ...
```

#### 5.3.3 地址计算对比

```cpp
// 连续方式（model.cpp 第388行）：
int32_t cache_offset = layer_idx * config_->seq_len_ * config_->kv_dim_
                     + token_pos * config_->kv_dim_;
// CPU: 2 次乘法 + 1 次加法

// Paged 方式（model.cpp 第356行）：
size_t byte_offset = mgr->get_kv_byte_offset(layer_idx, pos);
// CPU: 查表 block_table[layer*512 + pos/16] + 乘法
// GPU Kernel 中: paged_off(bt, pos, kv_dim) = 位运算 + 查表 + 乘法
```

### 5.4 KV 写入方式对比

#### 连续方式的 KV 写入（Decode）

```
slice_kv_cache(layer, pos)
  → 返回 tensor view 指向 key_cache[layer_offset + pos*kv_dim]
  → Wk/Wv matmul 输出直接写入此位置
```

矩阵乘法的输出直接写入 KV cache 的正确位置，零拷贝。

#### Paged 方式的 KV 写入（Decode）

```
ensure_allocated_to(pos) → sync_block_table()
slice_kv_cache(layer, pos)
  → get_kv_byte_offset(layer, pos) → 查 CPU block table
  → 返回 tensor view 指向 key_pool[physical_offset]
  → Wk/Wv matmul 输出直接写入此位置
```

同样是零拷贝，但多了页分配和 block table 同步开销。

#### CUDA Graph 路径的差异

```
// 连续方式（qwen3.cpp attention_qkv_with_graph）：
使用 KVCacheLayer kernel 写入，直接 offset = pos * kv_dim

// Paged 方式：
使用 paged_copy_kv kernel 写入，kernel 内部查表计算 offset
```

### 5.5 Attention Kernel 中的内存访问模式对比

#### 连续方式的访问模式

```
K_cache 物理地址：
  pos=0: base + 0 * kv_dim
  pos=1: base + 1 * kv_dim
  pos=2: base + 2 * kv_dim
  ...
  pos=15: base + 15 * kv_dim
  pos=16: base + 16 * kv_dim  (紧邻 pos=15)
```

完全顺序访问，GPU prefetcher 可以完美预测下一次访问地址。L2 cache 命中率极高。

#### Paged 方式的访问模式

```
Key Pool 物理地址（假设 block 0→物理块7, block 1→物理块3）：
  pos=0:  pool + (7*16 + 0) * kv_dim     = pool + 112 * kv_dim
  pos=1:  pool + (7*16 + 1) * kv_dim     = pool + 113 * kv_dim
  ...
  pos=15: pool + (7*16 + 15) * kv_dim    = pool + 127 * kv_dim
  pos=16: pool + (3*16 + 0) * kv_dim     = pool + 48 * kv_dim   ← 跳转！
  pos=17: pool + (3*16 + 1) * kv_dim     = pool + 49 * kv_dim
  ...
```

**页内连续，跨页跳转**。每 16 个 token 发生一次地址跳转。对于 kv_dim=1024, FP16 的场景：
- 每个 KV token = 1024 × 2 = 2KB
- 每页 = 16 × 2KB = 32KB
- L2 cache line = 128 bytes

页内的 16 个 token（32KB）一旦加载到 L2，后续 15 次访问都是 cache hit。跨页跳转只影响 L2 的一个 cache line miss，开销很小。

### 5.6 内存回收对比

#### 连续方式（clear_kv_cache）

```cpp
// qwen3.cpp 第2114行（原有连续路径）
key_cache.memset_zero();
val_cache.memset_zero();
```

对整个 1152 MB 执行 `cudaMemset`。内存不会释放，只是清零。

#### Paged 方式（clear_kv_cache）

```cpp
// paged_kv_cache.cpp 第109-132行
void PagedKVCacheManager::clear() {
  // 1. 重置 block table（CPU）
  std::fill(block_table_cpu_.begin(), block_table_cpu_.end(), -1);

  // 2. 恢复 free list（所有块回到空闲池）
  free_list_.resize(num_blocks_);
  std::iota(free_list_.rbegin(), free_list_.rend(), 0);

  // 3. 重置分配状态
  allocated_pos_ = -1;

  // 4. 清零 GPU 内存
  cudaMemsetAsync(key_pool_gpu_, 0, pool_bytes, stream_);
  cudaMemsetAsync(value_pool_gpu_, 0, pool_bytes, stream_);
  cudaMemsetAsync(block_table_gpu_, 0xFF, bt_bytes, stream_);
}
```

当前实现也是清零全部 pool（与连续相同），但架构上已具备选择性释放的能力——只需将特定块的 block_table 条目设为 -1 并 push 回 free_list，而不需要清零实际数据。

### 5.7 对比总结图

```
                  连续 KV Cache                          Paged KV Cache
            ┌──────────────────┐                ┌──────────────────────────┐
 分配       │ init_mem():      │                │ init_mem():              │
            │ cudaMalloc       │                │ cudaMalloc (pool)        │
            │ 1152 MB 一次到位  │                │ 1152 MB pool + 72KB table│
            └──────────────────┘                │ 实际映射：按需分配        │
                                                └──────────────────────────┘
            ┌──────────────────┐                ┌──────────────────────────┐
 写入       │ slice_kv_cache:  │                │ ensure_allocated_to()    │
 (Decode)   │ offset = layer×  │                │ sync_block_table()       │
            │  seq×kv + pos×kv │                │ slice_kv_cache:          │
            │ → 直接指针       │                │ get_kv_byte_offset()     │
            └──────────────────┘                │ → 查表获取物理地址        │
                                                └──────────────────────────┘
            ┌──────────────────┐                ┌──────────────────────────┐
 Attention  │ K_cache + k*kv   │                │ key_pool + paged_off()   │
 (Kernel)   │ 顺序内存访问     │                │ 页内连续 + 跨页跳转      │
            └──────────────────┘                └──────────────────────────┘
            ┌──────────────────┐                ┌──────────────────────────┐
 清理       │ memset_zero()    │                │ block_table→[-1]         │
            │ (清零全部 1152MB) │                │ free_list→全部回收       │
            │ 内存不释放        │                │ pool 清零（可选）         │
            └──────────────────┘                └──────────────────────────┘
```

### 5.8 什么时候该用哪种方式？

| 场景 | 推荐方式 | 原因 |
|------|---------|------|
| 单序列、固定长度推理 | 连续 | 更简单，无查表开销 |
| 单序列、长度变化大 | Paged | 短对话不浪费长缓存内存 |
| 多序列并发 (batch serving) | Paged | 共享块池，提高利用率 |
| CUDA Graph 优化 | 两者都可 | Paged 需要专用 GPU-pos kernel |
| 内存极度紧张（Orin 8GB） | Paged | 按需分配，同一内存支持更多序列 |
| Prefix Cache / Beam Search | Paged | 块级共享和 Copy-on-Write 的基础 |

---

## 第六章：Block Size 选择与 Copy-on-Write 实现

### 6.1 Block Size（Page Size）的选择

#### 6.1.1 本工程中的选择

```cpp
// paged_kv_cache.h 第33行
static constexpr int32_t kDefaultPageSize = 16;

// paged_attention_kernel.cu 第25-27行
constexpr int PAGE_SIZE = 16;
constexpr int PAGE_SHIFT = 4;
constexpr int PAGE_MASK = 15;
```

本工程选择 **Block Size = 16 tokens**。下面详细分析这个选择的原因。

#### 6.1.2 影响 Block Size 选择的因素

**因素 1：GPU 内存对齐**

Orin (SM87) 的全局内存以 **128 字节（32 个 float 或 64 个 half）**为最小事务粒度。

```
对于 FP16 + head_size=128:
  一个 KV token = 128 × 2 bytes = 256 bytes
  一个 Block(16 tokens) = 16 × 256 = 4,096 bytes = 4KB

4KB = 32 × 128 bytes，恰好是 32 个内存事务。
这与 GPU L2 cache line 大小完美对齐。
```

**因素 2：Block Table 大小**

```
block_table_size = num_layers × max_blocks_per_seq × 4 bytes

Block Size = 16:  max_blocks = 8192/16 = 512  → table = 36×512×4 = 72 KB
Block Size = 8:   max_blocks = 8192/8 = 1024  → table = 36×1024×4 = 144 KB
Block Size = 32:  max_blocks = 8192/32 = 256  → table = 36×256×4 = 36 KB
Block Size = 1:   max_blocks = 8192           → table = 36×8192×4 = 1152 KB
```

Block Size 越小，table 越大。72KB 的 block table 可以完全放入 Orin 的 L2 cache（2MB），确保 `paged_off()` 查表时几乎总是 cache hit。如果 block_size=1（token 级分页），table 高达 1.1MB，占用大量 L2 空间。

**因素 3：内存碎片（内部碎片）**

每个序列在最后一个 block 中平均浪费 `page_size/2` 个 slot（每层）：

```
Block Size = 16: 平均浪费 = 8 tokens × 36 layers × 1024 × 2 = 576 KB
Block Size = 32: 平均浪费 = 16 tokens × 36 layers × 1024 × 2 = 1,152 KB
Block Size = 64: 平均浪费 = 32 tokens × 36 layers × 1024 × 2 = 2,304 KB
Block Size = 8:  平均浪费 = 4 tokens × 36 layers × 1024 × 2 = 288 KB
```

Block Size 越大，碎片越多。16 是较好的平衡点。

**因素 4：位运算效率**

Block Size 必须是 **2 的幂次**，以使用位运算替代除法/取模：

```cuda
pos >> PAGE_SHIFT    // 替代 pos / page_size (1 周期 vs 20+ 周期)
pos & PAGE_MASK      // 替代 pos % page_size (1 周期 vs 20+ 周期)
```

候选值：2, 4, 8, 16, 32, 64, 128, 256...

**因素 5：Kernel 中的空间局部性**

Kernel 中 KV 的访问模式是 `k = 0, 1, 2, ...` 的顺序遍历。页内 token 物理连续，跨页时发生跳转。Block Size 越大，跳转越少：

```
Block Size = 16: seq_len=1000 时跳转 62 次 (1000/16 - 1)
Block Size = 32: seq_len=1000 时跳转 31 次
Block Size = 64: seq_len=1000 时跳转 15 次
Block Size = 8:  seq_len=1000 时跳转 124 次
```

但由于每次跳转只影响一个 L2 cache miss（~100 cycles），而 kernel 本身每个 token 需要 ~1000 cycles（点积 + 累加），跳转开销占比 < 1%。

#### 6.1.3 综合分析：最优 Block Size 范围

| Block Size | L2 对齐 | Table 大小 | 碎片 | 跳转次数 | 位运算 | 综合评分 |
|:----------:|:------:|:----------:|:----:|:-------:|:-----:|:-------:|
| **4** | 1KB/block | 288KB | 144KB | 249次 | ✅ | ★★★ |
| **8** | 2KB/block | 144KB | 288KB | 124次 | ✅ | ★★★★ |
| **16** | **4KB/block** | **72KB** | **576KB** | **62次** | ✅ | **★★★★★** |
| **32** | 8KB/block | 36KB | 1152KB | 31次 | ✅ | ★★★★ |
| **64** | 16KB/block | 18KB | 2304KB | 15次 | ✅ | ★★★ |
| **128** | 32KB/block | 9KB | 4608KB | 7次 | ✅ | ★★ |

**结论**：
- **16 是最优选择**：4KB block 恰好对齐 GPU 内存页，block table 足够小（72KB ⊂ L2 2MB），碎片可控
- **8 和 32 也可接受**：8 碎片更小但 table 更大；32 跳转更少但碎片翻倍
- **< 8 或 > 64 不推荐**：要么 table 过大，要么碎片过多

#### 6.1.4 业界参考

| 系统 | Block Size | 备注 |
|------|-----------|------|
| vLLM (原始论文) | 16 | 标准选择 |
| TensorRT-LLM | 64 / 128 | 强调连续性，减少查表 |
| SGLang | 1 (token 级) | RadixAttention 需要 token 粒度共享 |
| 本工程 (OrinMLLM) | 16 | 平衡内存效率和性能 |

### 6.2 Copy-on-Write (CoW) 的原理与实现

#### 6.2.1 什么是 Copy-on-Write？

Copy-on-Write 是一种延迟复制优化：多个序列共享相同的 KV cache 页，只有当某个序列需要**修改**共享页时，才将该页复制一份后修改副本。

**典型应用场景**：

```
场景 1：Beam Search (K=4)
  所有 4 个 beam 共享 prompt 的 KV cache 页
  当 beam 分叉（选择不同 token）时 → 需要 CoW

场景 2：Prefix Caching
  多个请求共享相同 system prompt 的 KV cache
  请求 A: "You are a helpful assistant. What is 2+2?"
  请求 B: "You are a helpful assistant. Translate hello."
  "You are a helpful assistant." 的 KV 页可共享
```

#### 6.2.2 CoW 的数据结构支持

CoW 需要对 PagedKVCacheManager 增加**引用计数**：

```cpp
// 当前实现（paged_kv_cache.h）：
std::vector<int32_t> block_table_cpu_;     // 逻辑块 → 物理块 映射
std::vector<int32_t> free_list_;           // 空闲物理块栈

// CoW 扩展需要增加：
std::vector<int32_t> ref_count_;           // ref_count_[physical_block] = 引用计数
// 当 ref_count[block] > 1 时，该块被多个逻辑块共享
```

#### 6.2.3 CoW 的完整实现方案

以下是基于本工程架构的 CoW 实现方案（伪代码 + 实际代码扩展）：

**Step 1：扩展 PagedKVCacheManager**

```cpp
class PagedKVCacheManager {
public:
  // ... 现有接口 ...

  // CoW 新增接口：
  void share_blocks(int32_t src_seq, int32_t dst_seq);  // 共享页
  void cow_on_write(int32_t seq_id, int32_t pos);       // 写时复制

private:
  // CoW 新增成员：
  std::vector<int32_t> ref_count_;  // 物理块引用计数
  // 多序列支持：每个序列有独立的 block table
  // block_tables_[seq_id][layer * max_blocks_per_seq + logical_block] = physical_block
  std::vector<std::vector<int32_t>> block_tables_;
};
```

**Step 2：分配时初始化引用计数**

```cpp
int32_t PagedKVCacheManager::allocate_block() {
  CHECK(!free_list_.empty());
  int32_t block_idx = free_list_.back();
  free_list_.pop_back();
  ref_count_[block_idx] = 1;  // 新块引用计数 = 1
  return block_idx;
}
```

**Step 3：共享页（Beam Search 分叉时）**

```cpp
void PagedKVCacheManager::share_blocks(int32_t src_seq, int32_t dst_seq) {
  // 将 src 序列的所有已分配页与 dst 共享
  auto& src_table = block_tables_[src_seq];
  auto& dst_table = block_tables_[dst_seq];

  for (int layer = 0; layer < num_layers_; ++layer) {
    for (int lb = 0; lb < max_blocks_per_seq_; ++lb) {
      int32_t idx = layer * max_blocks_per_seq_ + lb;
      int32_t physical = src_table[idx];
      if (physical >= 0) {
        dst_table[idx] = physical;   // 共享同一个物理块
        ref_count_[physical] += 1;   // 引用计数 +1
      }
    }
  }
}
```

**Step 4：写时复制（在需要修改共享页时触发）**

```cpp
void PagedKVCacheManager::cow_on_write(int32_t seq_id, int32_t layer, int32_t pos) {
  int32_t logical_block = pos / page_size_;
  int32_t table_idx = layer * max_blocks_per_seq_ + logical_block;
  int32_t old_physical = block_tables_[seq_id][table_idx];

  if (old_physical < 0) {
    // 未分配 → 直接分配新块
    block_tables_[seq_id][table_idx] = allocate_block();
    return;
  }

  if (ref_count_[old_physical] == 1) {
    // 独占 → 直接写入，无需复制
    return;
  }

  // ref_count > 1 → 需要 Copy-on-Write
  int32_t new_physical = allocate_block();

  // 复制旧块数据到新块
  size_t block_bytes = page_size_ * kv_dim_ * dtype_size_;
  size_t old_offset = (size_t)old_physical * page_size_ * kv_dim_ * dtype_size_;
  size_t new_offset = (size_t)new_physical * page_size_ * kv_dim_ * dtype_size_;

  // Key pool 拷贝
  cudaMemcpyAsync(
      static_cast<char*>(key_pool_gpu_) + new_offset,
      static_cast<char*>(key_pool_gpu_) + old_offset,
      block_bytes, cudaMemcpyDeviceToDevice, stream_);

  // Value pool 拷贝
  cudaMemcpyAsync(
      static_cast<char*>(value_pool_gpu_) + new_offset,
      static_cast<char*>(value_pool_gpu_) + old_offset,
      block_bytes, cudaMemcpyDeviceToDevice, stream_);

  // 更新映射：该序列指向新块
  block_tables_[seq_id][table_idx] = new_physical;

  // 旧块引用计数 -1
  ref_count_[old_physical] -= 1;
  if (ref_count_[old_physical] == 0) {
    free_block(old_physical);  // 引用归零 → 回收
  }
}
```

**Step 5：释放页时的引用计数管理**

```cpp
void PagedKVCacheManager::free_block(int32_t block_idx) {
  ref_count_[block_idx] = 0;
  free_list_.push_back(block_idx);
}

void PagedKVCacheManager::release_sequence(int32_t seq_id) {
  auto& table = block_tables_[seq_id];
  for (int idx = 0; idx < num_layers_ * max_blocks_per_seq_; ++idx) {
    int32_t physical = table[idx];
    if (physical >= 0) {
      ref_count_[physical] -= 1;
      if (ref_count_[physical] == 0) {
        free_block(physical);
      }
      table[idx] = -1;
    }
  }
}
```

#### 6.2.4 CoW 的时序示例（Beam Search, width=2）

```
时间线：
                                            Block Table         ref_count
t=0: Prefill "Hello world" (2 tokens)
     Seq 0: [block_table[0]=phys_0]       phys_0: [Hello, world]  ref=1

t=1: Beam split → Seq 0 产生 Seq 1
     share_blocks(src=0, dst=1)
     Seq 0: [block_table[0]=phys_0]       phys_0: [Hello, world]  ref=2 ←共享
     Seq 1: [block_table[0]=phys_0]

t=2: Seq 0 生成 "How"
     cow_on_write(seq=0, pos=2)
     ref_count[phys_0]=2 > 1 → 触发 CoW！
     1. 分配 phys_1, cudaMemcpy(phys_0 → phys_1)
     2. block_tables[0][0] = phys_1
     3. ref_count[phys_0]-- = 1
     然后写入 "How" 到 phys_1[2]
     Seq 0: [block_table[0]=phys_1]       phys_0: [Hello, world]  ref=1
     Seq 1: [block_table[0]=phys_0]       phys_1: [Hello, world, How] ref=1

t=3: Seq 1 生成 "Why"
     cow_on_write(seq=1, pos=2)
     ref_count[phys_0]=1 → 独占，无需复制
     直接写入 "Why" 到 phys_0[2]
     Seq 0: [block_table[0]=phys_1]       phys_0: [Hello, world, Why] ref=1
     Seq 1: [block_table[0]=phys_0]       phys_1: [Hello, world, How] ref=1
```

#### 6.2.5 CoW 在本工程中的当前状态

当前 OrinMLLM 的 PagedKVCacheManager **尚未实现 CoW**，原因是：

1. **单序列推理**：当前所有模型（Qwen3、Qwen2、Qwen3-VL）只支持单序列推理，没有 beam search 或多请求并发
2. **Prefix Cache 已有独立实现**：工程中使用 Radix Tree 实现了 prefix cache（见 `docs/prefix_cache_and_radix_tree_analysis_report.md`），在当前架构下无需 CoW

但 PagedAttention 的架构已经为 CoW 做好了准备：
- `free_list_` 已经实现了块的分配和回收
- block table 的间接映射层使得 CoW 只需修改映射，无需移动数据
- pool 的固定大小物理块使得 `cudaMemcpy` 可以按块复制

#### 6.2.6 CoW 的性能分析

```
CoW 开销分析（Qwen3-8B, FP16, block_size=16）：
  一个块的数据量 = 16 × 1024 × 2 = 32 KB
  cudaMemcpy D2D 32 KB ≈ 5 μs
  CoW 需要拷贝 K + V = 2 × 32 KB ≈ 10 μs

对比一步 decode attention（pos=1000, 32 heads）：
  计算量 ≈ 32 × 1000 × 128 × 3 FLOPs ≈ 12.3 MFLOPs
  Orin FP16 TFLOPS ≈ 100 TFLOPS → 理论 0.1 μs，但受带宽限制 ≈ 200 μs

CoW 开销占比 = 10 / 200 ≈ 5%，是可接受的。
```

且 CoW 只在 beam 分叉时触发（每步最多 1 次），不影响正常的 sequential decode。

### 6.3 Block Size 对 CoW 的影响

Block Size 还影响 CoW 的粒度和效率：

| Block Size | CoW 拷贝量 | CoW 触发频率 | 内存共享粒度 |
|:----------:|:---------:|:----------:|:-----------:|
| **8** | 16 KB | 每 8 token 可能触发 | 细粒度，共享更多 |
| **16** | 32 KB | 每 16 token 可能触发 | 中等 |
| **32** | 64 KB | 每 32 token 可能触发 | 粗粒度，共享较少 |
| **64** | 128 KB | 每 64 token 可能触发 | 很粗，可能浪费 |

对于 Beam Search（通常 beam width=4-8），Block Size = 16 的 CoW 粒度是合理的：
- 拷贝 32KB 只需 ~5μs
- 大多数 beam 在同一个 block 内不会分叉，触发频率低

对于 Prefix Caching（共享长 system prompt），Block Size = 16 也是合适的：
- 256 token 的 system prompt = 16 个 block
- 共享时只需复制 block table 指针，不拷贝数据
- 当用户请求写入新 token 时，只对最后一个 block 触发 CoW

---

## 第七章：Block Table 在 CUDA Kernel 中的传递与数据结构设计

### 7.1 Block Table 的数据结构

Block Table 是 PagedAttention 的核心映射表，连接逻辑地址与物理地址。

**数据类型**：`int32_t`（4 字节有符号整数）

**逻辑布局**：二维数组 `[num_layers, max_blocks_per_seq]`

**物理存储**：行主序一维数组，大小 = `num_layers × max_blocks_per_seq`

```cpp
// paged_kv_cache.h 第128-130行
// CPU 端：std::vector，用于分配管理
std::vector<int32_t> block_table_cpu_;
// block_table_cpu_[layer * max_blocks_per_seq + logical_block] = physical_block_idx
// -1 表示未分配

// GPU 端：cudaMalloc 分配的裸指针，用于 Kernel 查表
int32_t* block_table_gpu_ = nullptr;
```

**内存占用**（Qwen3-8B, max_seq_len=8192, page_size=16）：

```
max_blocks_per_seq = ceil(8192 / 16) = 512
num_layers = 36
block_table_size = 36 × 512 × 4 bytes = 73,728 bytes ≈ 72 KB
```

72 KB 完全可以放入 Orin 的 L2 Cache（2 MB），因此 Kernel 中的查表几乎总是 L2 hit。

### 7.2 CPU → GPU 同步机制

Block Table 在 CPU 上维护逻辑→物理映射，在 GPU Kernel 访问前必须同步：

```cpp
// paged_kv_cache.cpp 第99-107行
void PagedKVCacheManager::sync_block_table() {
  size_t bytes = num_layers_ * max_blocks_per_seq_ * sizeof(int32_t);
  if (stream_) {
    cudaMemcpyAsync(block_table_gpu_, block_table_cpu_.data(), bytes,
                    cudaMemcpyHostToDevice, stream_);
  } else {
    cudaMemcpy(block_table_gpu_, block_table_cpu_.data(), bytes,
               cudaMemcpyHostToDevice);
  }
}
```

**设计决策**：每次同步拷贝**整个** block table（72 KB），而非增量更新。原因：
1. 72 KB 的 `cudaMemcpyAsync` 只需 ~2μs（Orin PCIe/统一内存）
2. 增量更新需要追踪脏页，逻辑复杂且可能多次小拷贝反而更慢
3. 整表拷贝保证一致性，不会有部分更新的数据竞争问题

### 7.3 从 Host 到 Kernel 的传递链路

Block Table 从 Manager 到 Kernel 的完整传递路径如下：

```
Host 代码 (qwen3.cpp)
  │
  ├── attention_mha() / attention_mha_with_graph()
  │     │
  │     ▼
  │   configure_paged lambda:
  │     layer->set_paged_mode(true, page_size, max_blocks_per_seq,
  │                           key_pool_gpu, value_pool_gpu, block_table_gpu)
  │     │
  │     ▼
  │   FlashAttentionDecodeLayer 成员变量:
  │     block_table_ = block_table_gpu  (const int32_t*)
  │     │
  │     ▼
  │   forward() → 调用 kernel launch function:
  │     kernel::paged_flash_attention_decode_fp16_cu(
  │         ..., block_table_, ...)
  │
  ▼
Host Launch Function (paged_attention_kernel.cu 第653行)
  │
  │   // 偏移到当前层的 block table 行
  │   const int32_t* layer_bt = block_table + layer_index * max_blocks_per_seq;
  │
  │   paged_decode_fp16_kernel<<<grid, block, smem, stream>>>(
  │       ..., layer_bt, ...);
  │
  ▼
Device Kernel (paged_decode_fp16_kernel)
  │
  │   // 参数中的 block_table 已经是当前层的起始指针
  │   // 直接通过 paged_off() 查表
  │   size_t off = paged_off(block_table, k, kv_dim);
  │   //            ↓
  │   // bt[pos >> PAGE_SHIFT] * PAGE_SIZE + (pos & PAGE_MASK)
  │
  ▼
Global Memory 访问：
  block_table[logical_block]  →  physical_block_id  →  pool[element_offset]
```

### 7.4 Kernel 中的 Block Table 访问模式

**关键设计：host launch 函数做层偏移**

```cuda
// paged_attention_kernel.cu host launch function
const int32_t* layer_bt = block_table + layer_index * max_blocks_per_seq;
```

这意味着 Kernel 内部的 `block_table` 指针**已经指向当前层的块表行**，`paged_off()` 中的 `bt[pos >> PAGE_SHIFT]` 直接就是当前层的物理块号，无需额外加 `layer * stride`。

这种设计的优势：
1. Kernel 内部无需知道 `layer_index`（减少一个参数传递和一次乘法）
2. 所有 5 个 attention kernel 共用同一个 `paged_off()` 函数
3. `bt[pos >> PAGE_SHIFT]` 访问模式对 L2 cache 友好（连续 512 个 int32）

**Kernel 中的实际访问**：

```cuda
// Decode Kernel：每个线程遍历 k = tid, tid+256, tid+512, ...
for (int k = tid; k < kv_len; k += PG_DECODE_BLOCK) {
    // 对每个 KV position 查一次表
    const half* k_ptr = key_pool + paged_off(block_table, k, kv_dim) + head_offset;
    //                              ↑ 读 block_table[k/16]
}
```

由于 `k` 每次步进 256，而 page_size=16，理论上每 16 次循环访问同一个 block_table 条目。GPU 的 L2 cache 会缓存这些条目，实际查表开销接近零。

### 7.5 KV Write Kernel 中的 Block Table 使用

KV Write Kernel（CUDA Graph 路径）的 block table 传递略有不同：

```cuda
// paged_copy_kv_fp16_cu kernel
__global__ void paged_copy_kv_fp16_cu(
    half* kv_pool, const half* src, const int32_t* pos,
    const int32_t* block_table,  // 完整 block table（所有层）
    int32_t kv_dim, int32_t layer_idx, int32_t max_blocks_per_seq)
{
    int32_t position = *reinterpret_cast<const volatile int32_t*>(pos);
    const int32_t* bt = block_table + layer_idx * max_blocks_per_seq;  // Kernel 内部做偏移
    size_t off = paged_off(bt, position, kv_dim);
    kv_pool[off + idx] = src[idx];
}
```

这里 **Kernel 自己做层偏移**（而非 host launch 做），因为：
- CUDA Graph 要求所有 kernel 参数地址固定
- `block_table` 指针必须是固定的全局 block_table 起始地址
- `layer_idx` 作为标量参数传入，在 Graph replay 时不需要变（每层有独立的 captured kernel）

### 7.6 数据结构设计总结

```
┌─────────────────────────────────────────────────────────────┐
│                Block Table 数据结构                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  CPU 端 (std::vector<int32_t>):                             │
│  ┌───┬───┬───┬───┬───┬─────┬───┬───┬───┬───┬─────┐        │
│  │ 0 │ 1 │ 2 │-1 │...│ -1  │36 │37 │38 │-1 │...  │        │
│  └───┴───┴───┴───┴───┴─────┴───┴───┴───┴───┴─────┘        │
│  ← layer 0 (512 entries) →  ← layer 1 (512 entries) → ... │
│                                                             │
│  sync_block_table()  ↕ cudaMemcpy (72 KB)                  │
│                                                             │
│  GPU 端 (int32_t*):                                         │
│  ┌───┬───┬───┬───┬───┬─────┬───┬───┬───┬───┬─────┐        │
│  │ 0 │ 1 │ 2 │-1 │...│ -1  │36 │37 │38 │-1 │...  │        │
│  └───┴───┴───┴───┴───┴─────┴───┴───┴───┴───┴─────┘        │
│                                                             │
│  Kernel 访问:                                               │
│    Attention: layer_bt = block_table + layer * 512 (host偏移)│
│    KV Write:  bt = block_table + layer_idx * 512 (kernel偏移)│
│    查表:      bt[pos >> 4] → physical_block_id              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 第八章：Orin 内存架构分析与针对性优化

### 8.1 Orin 与服务器 GPU 的内存架构对比

| 特性 | NVIDIA Orin (SM87) | 服务器 GPU (A100/H100) |
|------|-------------------|----------------------|
| **内存类型** | LPDDR5 统一内存 | HBM2e / HBM3 显存 |
| **内存容量** | 8-64 GB（CPU/GPU 共享） | 40-80 GB（GPU 独占） |
| **内存带宽** | 102-204 GB/s | 2,000-3,350 GB/s |
| **CPU-GPU 传输** | 零拷贝（统一地址空间） | PCIe 4.0/5.0 (~32-64 GB/s) |
| **L2 Cache** | 2 MB | 40-50 MB |
| **SM 数量** | 12-16 | 108-144 |
| **FP16 算力** | 100 TOPS | 312-989 TFLOPS |
| **功耗** | 15-60W | 300-700W |

### 8.2 Orin 统一内存架构的核心差异

Orin 使用统一内存（Unified Memory）架构，CPU 和 GPU 共享同一块物理 LPDDR5 内存：

```
服务器 GPU:                        Orin:
┌──────────┐    PCIe     ┌─────┐   ┌──────────────────────────┐
│ CPU RAM  │◄──32GB/s──►│ GPU │   │     LPDDR5 统一内存       │
│ (DDR5)   │             │(HBM)│   │  ┌─────┐    ┌─────┐     │
│  64 GB   │             │80GB │   │  │ CPU  │    │ GPU │     │
└──────────┘             └─────┘   │  │cores │    │ SM  │     │
                                    │  └──┬──┘    └──┬──┘     │
                                    │     │  共享    │         │
                                    │     └───204GB/s─┘        │
                                    │      8-64 GB             │
                                    └──────────────────────────┘
```

### 8.3 针对 Orin 内存架构的优化

#### 优化 1：利用统一内存省去 CPU→GPU 数据拷贝

在服务器 GPU 上，block table 的 CPU→GPU 同步需要经过 PCIe 总线。在 Orin 上，由于统一内存，`cudaMemcpy(HostToDevice)` 实际上只是修改页表映射或触发缓存一致性操作，延迟远低于 PCIe 传输：

```cpp
// paged_kv_cache.cpp 第100-106行
// 在 Orin 上这个拷贝的延迟远低于服务器 GPU
cudaMemcpyAsync(block_table_gpu_, block_table_cpu_.data(), bytes,
                cudaMemcpyHostToDevice, stream_);
```

Orin 上 72 KB 的 block table 同步延迟 < 1μs，而服务器 GPU 上相同操作可能需要 5-10μs（PCIe 延迟 + 传输）。

#### 优化 2：最小化 GPU 内存占用

Orin 的 8-16 GB 内存需要 CPU 和 GPU 共享。模型权重 + KV cache + 系统开销不能超过总内存。

```
Qwen3-8B FP16 内存预算：
  模型权重: ~16 GB（FP16, 8B 参数）
  KV Cache Pool: ~1.15 GB（Key + Value pools）
  Block Table: ~72 KB
  中间 buffer: ~50 MB（query, RoPE, FFN 等）
  系统开销: ~1 GB（Linux kernel, CUDA runtime）
  ─────────────────────
  总计: ~18.3 GB → 需要 32 GB Orin
```

**PagedAttention 的内存节约**：虽然总池大小相同，但按需分配意味着短对话不会浪费长缓存空间。这对多序列场景至关重要。

#### 优化 3：FP16 全面采用

本工程中 KV Cache、MatMul、Attention 全部支持 FP16：

```cpp
// qwen3.cpp init_mem()
base::DataType activation_dtype = is_fp16_model_ ? base::DataType::kDataTypeFp16
                                                  : base::DataType::kDataTypeFp32;
```

FP16 vs FP32 在 Orin 上的影响：
- 内存占用减半（KV pool: 576 MB vs 1152 MB）
- 带宽需求减半（Orin 带宽是关键瓶颈）
- FP16 张量核心（Tensor Core）吞吐量是 FP32 的 4x
- 对 8 GB Orin，FP16 是唯一可行方案

#### 优化 4：Shared Memory 精细控制

Orin SM87 每个 SM 有 128 KB shared memory。我们的 kernel 精确控制 shared memory 用量：

```cuda
// FP16 decode kernel:
int smem = head_size * sizeof(half)                    // 256 bytes (query)
         + score_buf * sizeof(float)                    // 按需（kv_len）
         + 2 * PG_DECODE_WARPS * sizeof(float);        // 64 bytes (reduction)
// 典型值: 256 + 4000 + 64 ≈ 4.3 KB

// GPU-pos decode kernel (固定):
int smem = 128*2 + 256*4 + 2*4*4 = 1,312 bytes
```

两个 kernel 的 shared memory 都远小于 48 KB（默认 L1/shared 分配），确保每个 SM 可以同时运行多个 block。

#### 优化 5：减少 Kernel Launch 开销

Orin 的 CPU 主频较低（2.2 GHz vs 服务器 4+ GHz），kernel launch 的 host 端开销更显著。因此：

1. **CUDA Graph**：将多个 kernel launch 合并为单次 graph launch
2. **Fused FFN**：将 W1 + W3 + SwiGLU 合并为单个 kernel
3. **Paged Copy Kernel**：用 kernel 替代多次 `cudaMemcpy` 调用

#### 优化 6：`__ldg()` 利用 Read-Only Cache

Orin 的 L2 cache 只有 2 MB（A100 有 40 MB）。`__ldg()` 走 texture/read-only cache 通道，与 L1 data cache 分离，减轻 L2 压力：

```cuda
float4 kk = __ldg(kk4 + d);  // 走 read-only cache
```

### 8.4 统一内存在 PagedAttention 场景中是优势还是劣势？

#### 优势

| 方面 | 分析 |
|------|------|
| **Block Table 同步零开销** | CPU→GPU 的 block_table sync 是内存内拷贝（~1μs），不走 PCIe |
| **CPU 端页管理零延迟** | `ensure_allocated_to()` 操作的 `block_table_cpu_` 与 GPU 在同一内存控制器上 |
| **Pinned Memory 免费** | 所有内存天然 pinned（不需要 `cudaMallocHost`），`cudaMemcpyAsync` 始终有效 |
| **内存碎片可统一管理** | 一个 allocator 管理全部内存，不存在 CPU/GPU 内存碎片独立问题 |
| **调试友好** | CPU 可以直接读 GPU 内存（通过统一指针），方便验证 KV cache 内容 |

#### 劣势

| 方面 | 分析 |
|------|------|
| **绝对带宽低** | 204 GB/s vs HBM 2000+ GB/s，Attention 是带宽密集型操作 |
| **CPU/GPU 争抢带宽** | block_table CPU 操作和 GPU kernel 共享内存带宽 |
| **总容量受限** | 8-64 GB 需要 CPU+GPU 共享，留给 KV pool 的空间有限 |
| **Cache 一致性开销** | 统一内存的 coherency 协议有额外延迟 |

#### 总结判断

**PagedAttention 场景中，Orin 的统一内存总体是优势**，原因：

1. PagedAttention 的核心开销是 block table 的 CPU↔GPU 同步，统一内存将此开销降到接近零
2. 真正的瓶颈是 KV pool 的带宽（attention 计算），而非 block table 管理
3. Orin 的 PagedAttention 可以省去服务器 GPU 上 `cudaMallocHost` + `cudaMemcpyAsync` 的复杂双缓冲机制
4. `ensure_allocated_to()` 的 CPU 逻辑可以与 GPU kernel 真正并行（无 PCIe 串行化）

但需要通过 FP16 + 向量化 + 减少内存访问来缓解带宽劣势。

---

## 第九章：Orin 上的低延迟调度器设计与 vLLM 连续批处理对比

### 9.1 当前 OrinMLLM 的调度模型

OrinMLLM 当前为**单序列同步推理**模型：

```
调度流程：
  1. 接收用户输入
  2. Tokenize → Prefill（逐层处理所有 prompt tokens）
  3. Decode loop：生成 1 token → 检查 EOS → 生成下一个
  4. 输出结果
  5. 等待下一个请求
```

关键代码路径（`inference_common.h`）：

```cpp
// Prefill
for (int32_t layer_idx = 0; layer_idx < layer_num; ++layer_idx) {
    batched_attention_rms(layer_idx, input, rms_out, seq_len);
    batched_attention_qkv(layer_idx, rms_out, query, key, value, seq_len, start_pos);
    batched_attention_mha(layer_idx, query, mha_out, seq_len, start_pos);
    batched_feed_forward(layer_idx, input, seq_len);
}

// Decode (可能启用 CUDA Graph)
while (token_count < max_tokens && !is_eos) {
    model.forward(input, pos_tensor, next_token);  // 整个模型一步
    // ... output token ...
}
```

### 9.2 vLLM 连续批处理 (Continuous Batching) 简介

vLLM 的调度器远比单序列推理复杂：

```
vLLM Scheduler:
  while True:
    1. 从等待队列中选择可调度的序列
    2. 确定每个序列的状态（prefill / decode / swap / preempt）
    3. 按 PagedAttention 分配/回收物理块
    4. 组装 batch（可能混合 prefill 和 decode 序列）
    5. 一次 forward 处理整个 batch
    6. 更新每个序列状态
    7. 抢占低优先级序列（如需要）归还块给高优先级序列
```

### 9.3 核心差异对比

| 维度 | OrinMLLM (当前) | vLLM |
|------|----------------|------|
| **批处理方式** | 单序列完整处理 | 连续批处理（迭代级调度） |
| **调度粒度** | 请求级（整个推理过程不可中断） | 迭代级（每步 decode 可调度） |
| **序列管理** | 1 个活跃序列 | N 个并发序列 |
| **内存管理** | 单序列 block table | 多序列共享 block pool |
| **抢占策略** | 无 | Swap-out / Recompute |
| **目标场景** | 边缘设备低延迟单用户 | 服务器高吞吐多用户 |

### 9.4 如何在 Orin 上实现低延迟调度器

针对 Orin 的特性，低延迟调度器应该与 vLLM 不同：

```
Orin 低延迟调度器设计原则：
  1. 优先延迟，而非吞吐量
  2. 最小化调度开销（CPU 主频低）
  3. 利用统一内存避免数据搬运
  4. 限制并发序列数（内存有限）
```

**方案：轻量级双序列 Ping-Pong 调度器**

```
┌───────────────────────────────────────────────────────┐
│  Orin Low-Latency Scheduler                           │
│                                                       │
│  Slot A: [sequence_0]  →  active, decoding            │
│  Slot B: [sequence_1]  →  prefilling (async)          │
│                                                       │
│  当 Slot A 完成:                                       │
│    - Slot B 已 prefill 完成 → 立即切换到 decode        │
│    - Slot A 释放 blocks → 回收到 free list            │
│    - 新请求进入 Slot A → 开始 prefill                  │
│                                                       │
│  优势:                                                 │
│    - Prefill 和 Decode 在 GPU 流中重叠                 │
│    - 最多 2 个序列共享 block pool                      │
│    - 调度决策 < 1μs（简单 if-else）                    │
│    - 无需复杂的抢占/swap 逻辑                          │
└───────────────────────────────────────────────────────┘
```

本工程的 PagedAttention 实现已经为此做好了基础——只需扩展 block table 从 `[num_layers, max_blocks_per_seq]` 到 `[num_seqs, num_layers, max_blocks_per_seq]`。

### 9.5 为什么 Orin 不适合 vLLM 式连续批处理

1. **内存不足**：8 GB Orin 上 1 个 Qwen3-8B 的 KV pool 就占 1.15 GB，无法支持大 batch
2. **CPU 开销**：vLLM 调度器每步需要复杂的序列选择、抢占判断，Orin 的 ARM CPU 不够快
3. **带宽瓶颈**：batch size 增大后内存带宽成为瓶颈（204 GB/s 已经很紧张）
4. **延迟需求**：边缘场景通常需要最小延迟（单用户交互），而非最大吞吐量

---

## 第十章：Orin 上的内存碎片处理与 Block Pool 耗尽策略

### 10.1 PagedAttention 中的碎片类型

#### 内部碎片（Internal Fragmentation）

每个序列最后一个 block 中可能有未使用的 slot：

```
序列长度 = 35 tokens, page_size = 16
  Block 0: [token 0-15]   — 100% 使用
  Block 1: [token 16-31]  — 100% 使用
  Block 2: [token 32-34, _, _, _, _, _, _, _, _, _, _, _, _]
                           — 3/16 = 18.75% 使用，浪费 13 个 slot

平均内部碎片 = page_size / 2 = 8 slots × 36 layers × 1024 × 2 bytes = 576 KB / 序列
```

#### 外部碎片（External Fragmentation）

PagedAttention **不存在传统意义上的外部碎片**——这是它相对于连续分配的核心优势。因为所有 block 大小相同，任何空闲 block 都可以分配给任何序列的任何层。

### 10.2 当前实现的碎片最小化设计

**顺序分配策略**：

```cpp
// paged_kv_cache.cpp 第40行
free_list_.resize(num_blocks_);
std::iota(free_list_.rbegin(), free_list_.rend(), 0);
// free_list_ = [N-1, N-2, ..., 1, 0]
// pop_back 顺序: 0, 1, 2, 3, ...
```

通过 `std::iota(rbegin, rend, 0)` 使得 `pop_back()` 按 0, 1, 2... 顺序分配。连续分配的物理块在内存中也连续，提高 GPU 缓存局部性。

### 10.3 Block Pool 耗尽时的处理

当前实现通过 `CHECK` 宏直接终止：

```cpp
// paged_kv_cache.cpp 第73行
int32_t PagedKVCacheManager::allocate_block() {
  CHECK(!free_list_.empty()) << "PagedKVCache: out of free blocks!";
  int32_t block_idx = free_list_.back();
  free_list_.pop_back();
  return block_idx;
}
```

这在单序列推理中不会发生（块池容量 = 单序列最大所需），但多序列场景需要额外策略。

### 10.4 Block Pool 耗尽的应对策略

#### 策略 1：截断生成（当前可行方案）

```cpp
// 改进版 allocate_block
int32_t PagedKVCacheManager::try_allocate_block() {
  if (free_list_.empty()) {
    return -1;  // 表示分配失败
  }
  int32_t block_idx = free_list_.back();
  free_list_.pop_back();
  return block_idx;
}

// 上层调用
bool ok = ensure_allocated_to(pos);
if (!ok) {
    // 停止生成，返回已有结果
    LOG(WARNING) << "KV cache exhausted at pos=" << pos << ", truncating output";
    return false;
}
```

#### 策略 2：序列驱逐（多序列场景）

```
当 free_list 为空时：
  1. 选择最长/最老的序列
  2. 释放该序列的所有 blocks（所有层的所有块表条目）
  3. 将释放的 blocks 归还 free_list
  4. 将被驱逐的序列状态保存到 CPU RAM（统一内存中无需实际拷贝）
  5. 重新分配需要的 blocks
```

#### 策略 3：分层回收

```
对于长序列，可以选择性释放较早的 KV blocks：
  序列长度 = 2048 tokens = 128 blocks/layer
  需求: 释放 32 blocks

  方案: 释放 pos [0, 511] 对应的 blocks（最前面 32 个 logical blocks）
  代价: 被释放的 token 需要重新 prefill（recompute）

  这就是 vLLM 中的 "prompt recomputation" 策略
```

#### 策略 4：动态 Pool 大小

```cpp
// 构造函数中不一次性分配全部 pool
// 而是分批 cudaMalloc，按需扩展
void PagedKVCacheManager::grow_pool(int32_t additional_blocks) {
    // 分配新的内存块
    // 将新块加入 free_list
    // 更新 num_blocks_
}
```

在 Orin 统一内存架构下，`cudaMalloc` 实际上只是预留虚拟地址空间，真正的物理页在首次访问时才分配（overcommit），因此动态扩展的开销比服务器 GPU 低。

### 10.5 Orin 特有的内存压力缓解

| 策略 | 描述 | Orin 适用性 |
|------|------|------------|
| **FP16 量化** | KV cache 使用 FP16 | ✅ 已实现，内存减半 |
| **AWQ/GPTQ 权重量化** | 模型权重 INT4 | ✅ 已支持（Qwen3-8B-awq） |
| **KV Cache Quantization** | KV cache 用 INT8/FP8 | 可扩展，进一步减半 |
| **Sliding Window Attention** | 只保留最近 N 个 token 的 KV | 可与 paged 结合 |
| **统一内存 overcommit** | 先分配虚拟地址，物理页按需映射 | Orin 自然支持 |

---

## 第十一章：PagedAttention 正确性验证与 Orin CUDA 调试难点

### 11.1 验证方法：位级一致性对比

采用**基线对比法**：对相同输入，比较连续 KV Cache 和 Paged KV Cache 的输出是否完全相同。

**验证脚本思路**：

```bash
# 基线：连续 KV Cache
echo "Hello" | ./qwen3_infer model.bin tokenizer.json \
    --attention flash1 --max-tokens 15 --no-cuda-graph > baseline.txt

# 被测：Paged KV Cache
echo "Hello" | ./qwen3_infer model.bin tokenizer.json \
    --attention flash1 --max-tokens 15 --no-cuda-graph --paged-attention > paged.txt

# 对比
diff baseline.txt paged.txt && echo "PASS" || echo "FAIL"
```

### 11.2 验证结果

所有 5 个模型配置通过：

| 模型 | 精度 | 输出对比 | 结果 |
|------|------|---------|------|
| Qwen3-8B | FP16 | 完全相同 | ✅ PASS |
| Qwen3-8B | AWQ INT4 | 完全相同 | ✅ PASS |
| Qwen2.5-7B | FP32 | 完全相同 | ✅ PASS |
| Qwen2.5-7B | FP16 | 完全相同 | ✅ PASS |
| Qwen3-VL-8B | FP16 | 完全相同 | ✅ PASS |

**位级一致性的保证来源**：
1. Paged kernel 和 Contiguous kernel 使用完全相同的数学算法（点积、softmax、加权求和）
2. 相同的浮点运算顺序保证相同的舍入误差
3. 地址翻译只改变数据存放位置，不改变数据值

### 11.3 验证的关键测试点

| 测试项 | 验证内容 | 验证方法 |
|--------|---------|---------|
| **Prefill 正确性** | 批量 KV 写入→分页 KV 读取 | 对比 prefill 后第一个 decode token |
| **Decode 正确性** | 逐 token KV 写入→分页注意力计算 | 对比 15 个 decode token 序列 |
| **跨页边界** | pos=15→16 的地址翻译正确性 | 包含在 15 token 测试中 |
| **CUDA Graph 兼容** | block_table GPU 同步 + GPU pos | 启用 --cuda-graph 后验证 |
| **多轮清理** | clear() 后重新分配 | 交互模式多轮对话 |
| **FP32/FP16 混合** | 不同精度的地址计算 | 分别测试 FP32 和 FP16 模型 |
| **AWQ 量化** | 量化权重 + Paged KV | AWQ 模型验证 |
| **VL 多模态** | 图像特征 + 文本 KV 分页 | VL 模型验证 |

### 11.4 Orin 上调试 CUDA 的难点

#### 难点 1：cuda-gdb 不稳定

Orin 上的 `cuda-gdb` 经常遇到：
- Kernel 断点设置失败（SM87 支持问题）
- Warp 级调试时冻结整个系统
- 统一内存导致 CPU/GPU 断点冲突

**应对方案**：使用 `printf` 调试 + 结果对比，避免依赖 gdb。

#### 难点 2：NSight Compute 版本兼容

Orin JetPack 的 NSight Compute 版本可能与最新版不兼容：
- 某些 profiling metric 在 SM87 上不可用
- `ncu --set full` 可能导致系统 hang

**应对方案**：使用 `--set basic` 或选择性 metric。

#### 难点 3：统一内存的 Memory Checker 误报

CUDA `memcheck` / `compute-sanitizer` 在统一内存上可能误报 "invalid write"，因为 CPU 和 GPU 可以同时访问同一地址（这在分离显存架构上是非法的）。

**应对方案**：
```bash
# 使用 --check-api-memory-access no 忽略统一内存误报
compute-sanitizer --check-api-memory-access no ./qwen3_infer ...
```

#### 难点 4：Shared Memory Bank Conflict 定位

Bank conflict 在 Orin 上的表现不同于数据中心 GPU：
- SM87 的 shared memory bank 宽度为 4 bytes
- Warp size = 32，但 Orin 的 SM 可能不满载

**应对方案**：在 paged kernel 中确保 shared memory 布局与 bank 对齐：
```cuda
// s_scores[] 是 float (4 bytes)，每个 bank 一个 float
// 32 个线程访问 32 个连续 float → 32 个不同 bank → 零冲突
```

#### 难点 5：热功耗限制导致的非确定性行为

Orin 在持续高负载下会触发热节流（thermal throttling），导致：
- 核心频率降低
- 相同计算的耗时波动
- 极端情况下 GPU kernel 超时

**应对方案**：
```bash
# 设置最大功耗模式
sudo nvpmodel -m 0
# 固定 GPU 频率
sudo jetson_clocks
```

### 11.5 调试 PagedAttention 的具体技巧

1. **分层验证**：先验证 `paged_off()` 的地址翻译正确性，再验证 kernel 输出
2. **单层测试**：只跑 1 层，对比连续和分页的 KV cache 内容
3. **已知输入**：使用固定 prompt（如 "Hello"）确保可重复
4. **中间值输出**：在 kernel 中用条件 `printf` 输出关键中间值

```cuda
#ifdef DEBUG_PAGED
if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("paged_off(bt, pos=%d, kv=%d) = %zu, phys_block=%d\n",
           pos, kv_dim, paged_off(bt, pos, kv_dim), bt[pos >> 4]);
}
#endif
```

---

## 第十二章：Orin 上 PagedAttention 的最大改进点

### 12.1 当前实现的瓶颈分析

| 瓶颈 | 严重程度 | 描述 |
|------|---------|------|
| **单序列限制** | ★★★★★ | 无法并发处理多个请求 |
| **Prefill 逐 token 拷贝** | ★★★★ | batched_attention_qkv 中逐 token cudaMemcpy |
| **Block Table 全量同步** | ★★★ | 每次 sync 拷贝 72 KB（即使只改了 1 个条目） |
| **无 KV Cache 量化** | ★★★ | KV 使用 FP16，FP8/INT8 可进一步减半 |
| **pool 大小固定** | ★★ | 构造时分配全部内存，无法动态扩缩 |

### 12.2 最大改进点 1：Prefill 用 Kernel 替代逐 token cudaMemcpy

当前 prefill KV 写入：

```cpp
// qwen3.cpp batched_attention_qkv — 当前实现
for (int i = 0; i < seq_len; ++i) {
    size_t byte_offset = mgr->get_kv_byte_offset(layer_idx, start_pos + i);
    cudaMemcpyAsync(dst + byte_offset, src + i * kv_dim * elem, ...);
}
```

seq_len=256 时产生 256 次 `cudaMemcpyAsync`。改进方案：

```cuda
// 优化：一个 kernel 完成批量 paged KV 写入
__global__ void paged_batch_copy_kv(
    half* kv_pool, const half* src, const int32_t* block_table,
    int32_t kv_dim, int32_t start_pos, int32_t seq_len,
    int32_t layer_idx, int32_t max_blocks_per_seq)
{
    int token_idx = blockIdx.y;  // 哪个 token
    int dim_idx = threadIdx.x + blockIdx.x * blockDim.x;  // 哪个维度
    if (token_idx < seq_len && dim_idx < kv_dim) {
        int32_t pos = start_pos + token_idx;
        const int32_t* bt = block_table + layer_idx * max_blocks_per_seq;
        size_t off = paged_off(bt, pos, kv_dim);
        kv_pool[off + dim_idx] = src[token_idx * kv_dim + dim_idx];
    }
}
// 一次 launch 替代 seq_len 次 cudaMemcpy
```

预计改进：Prefill 阶段 KV 写入延迟减少 10-50x。

### 12.3 最大改进点 2：多序列支持

扩展 PagedKVCacheManager 支持多个独立 block table：

```cpp
class PagedKVCacheManager {
    // 从单序列扩展为多序列
    std::vector<std::vector<int32_t>> block_tables_cpu_;  // [seq_id][table]
    int32_t* block_tables_gpu_ = nullptr;   // [max_seqs, num_layers, max_blocks]
    std::vector<int32_t> allocated_pos_;    // 每个序列独立跟踪
};
```

这使得 Orin 可以同时处理 2-4 个序列（内存限制），大幅提高吞吐量。

### 12.4 最大改进点 3：KV Cache FP8/INT8 量化

将 KV cache 从 FP16 量化到 FP8：
- 内存占用：576 MB → 288 MB（单池）
- 带宽需求减半（Orin 带宽是主要瓶颈）
- 精度损失 < 0.1% perplexity（实验表明 KV 对精度不敏感）

### 12.5 最大改进点 4：增量 Block Table 同步

```cpp
// 改进：只同步变化的层和块
void sync_block_table_incremental() {
    for (auto& [table_idx, physical_block] : dirty_entries_) {
        size_t offset = table_idx * sizeof(int32_t);
        cudaMemcpyAsync(block_table_gpu_ + table_idx, &physical_block,
                        sizeof(int32_t), cudaMemcpyHostToDevice, stream_);
    }
    dirty_entries_.clear();
}
```

对于 decode（每步只新增 1 个逻辑块最多 36 层），从拷贝 72 KB 减少到拷贝 144 bytes。

### 12.6 优先级排序

```
改进优先级（投入产出比排序）：
  1. ★★★★★ Prefill batch copy kernel（1 天工作量，10-50x prefill 改进）
  2. ★★★★  增量 block table 同步（2 小时工作量，decode 延迟减少）
  3. ★★★★  多序列支持（1 周工作量，吞吐量 2-4x）
  4. ★★★   KV FP8 量化（3 天工作量，内存减半 + 带宽减半）
  5. ★★    动态 pool 扩缩（1 天工作量，灵活性提升）
```

---

## 第十三章：PagedAttention 与 Prefix Cache、CUDA Graph 的关系

### 13.1 三者的关系概览

```
┌─────────────────────────────────────────────┐
│              OrinMLLM 推理优化栈             │
│                                             │
│  ┌────────────┐  ┌──────────┐  ┌─────────┐ │
│  │ Prefix     │  │ Paged    │  │ CUDA    │ │
│  │ Cache      │  │ Attention│  │ Graph   │ │
│  │(KV 复用)   │  │(KV 分页) │  │(Launch  │ │
│  │            │  │          │  │ 优化)   │ │
│  └──────┬─────┘  └────┬─────┘  └────┬────┘ │
│         │             │             │       │
│         │    互相独立但可组合        │       │
│         │             │             │       │
│  ┌──────▼─────────────▼─────────────▼────┐  │
│  │        Model Layer (qwen3.cpp)         │  │
│  │  根据启用的优化组合选择执行路径         │  │
│  └────────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

三者是**独立的正交优化**，可以任意组合：

| 组合 | CLI 参数 | 含义 |
|------|---------|------|
| 无优化 | `--attention flash1 --no-cuda-graph` | 连续 KV + 基础 launch |
| 仅 CUDA Graph | `--attention flash1 --cuda-graph` | 连续 KV + Graph |
| 仅 Paged | `--attention flash1 --paged-attention --no-cuda-graph` | 分页 KV + 基础 launch |
| Paged + Graph | `--attention flash1 --paged-attention --cuda-graph` | 分页 KV + Graph |
| Paged + Prefix | `--attention flash1 --paged-attention --prefix-cache` | 分页 KV + KV 复用 |
| 三者全开 | `--attention flash1 --paged-attention --prefix-cache --cuda-graph` | 全部优化 |

### 13.2 初始化阶段的组合处理

**文件**：`demo/inference_common.h` 第1707-1720行

```cpp
// 1. 先启用 paged attention（影响 KV cache 分配方式）
if (config.use_paged_attention) {
    model.enable_paged_attention(true);  // 必须在 init() 之前
}

// 2. 初始化模型（init 中的 init_mem 根据 paged 标志分配内存）
auto init_status = model.init(base::DeviceType::kDeviceCUDA);

// 3. 启用其他优化（在 init 之后）
model.enable_fused_ffn(config.use_fused_ffn);
model.set_attention_type(config.attention_type);
if (config.use_cuda_graph) {
    model.enable_cuda_graph(true);
}
```

**关键顺序**：`enable_paged_attention()` → `init()` → `enable_cuda_graph()`

因为 `init()` 中的 `init_mem()` 根据 paged 标志决定是否创建 `PagedKVCacheManager`：

```cpp
// qwen3.cpp init_mem() 第870行
if (use_paged_attention_) {
    paged_kv_cache_manager_ = std::make_unique<base::PagedKVCacheManager>(...);
    // placeholder buffers
} else {
    // 连续 KV cache
}
```

### 13.3 PagedAttention + CUDA Graph 的协作

#### 问题：CUDA Graph 要求参数地址固定

CUDA Graph 在 capture 时记录所有 kernel 的参数。如果参数地址每步变化，graph 需要重新 capture。

#### 解决方案：所有 GPU 指针不变，值通过 GPU 内存间接传递

**KV Write 路径**（`qwen3.cpp` `attention_qkv_with_graph`）：

```cpp
// qwen3.cpp 第1336-1346行
if (use_paged_attention_ && paged_kv_cache_manager_) {
    auto* mgr = paged_kv_cache_manager_.get();
    cudaStream_t stream = cuda_config_->stream;
    kernel::paged_copy_to_kv_cache_kernel_fp16(
        static_cast<half*>(mgr->key_pool_gpu()),    // 固定地址（pool 起始）
        temp_key.ptr<half>(),                        // 固定地址（temp buffer）
        pos_tensor.ptr<int32_t>(),                   // 固定地址（GPU pos scalar）
        mgr->block_table_gpu(),                      // 固定地址（GPU block table）
        config_->kv_dim_, layer_idx,
        mgr->max_blocks_per_seq(), mgr->page_size(), stream);
}
```

每步 decode 前，host 代码更新：
1. `pos_tensor_gpu` 的**值**（通过 `cudaMemcpyAsync` 写入新 pos）
2. `block_table_gpu_` 的**内容**（通过 `sync_block_table()`）

但这些操作的 GPU **指针不变**，因此 CUDA Graph 可以 replay。

**Attention 计算路径**（`qwen3.cpp` `attention_mha_with_graph`）：

```cpp
// qwen3.cpp 第1405-1416行
auto flash_attn = qwen_layers_->flash_attention_decode_layer_;
configure_paged(flash_attn);  // set_paged_mode(true, ...)
flash_attn->set_use_gpu_pos(true);
flash_attn->set_input(4, pos_tensor_gpu);  // GPU pos tensor（固定地址）
STATUS_CHECK(flash_attn->forward());
```

进入 `forward()` 后：

```cpp
// flash_attention.cpp 第56-62行
if (paged_mode_) {
    if (use_fp16_ && use_gpu_pos_) {
        // 使用 online softmax kernel（shared memory 固定大小）
        kernel::paged_flash_attention_decode_fp16_gpu_pos_cu(
            pos_tensor.ptr<int32_t>(),  // GPU pos（固定地址）
            ..., key_pool_, value_pool_, block_table_, ...);
    }
}
```

`paged_decode_fp16_gpu_pos_kernel` 的 shared memory 是固定的（不依赖 kv_len）：

```cuda
// host launch
int smem = head_size * sizeof(half)     // 256 bytes
         + PG_ONLINE_TILE_K * sizeof(float)  // 1024 bytes
         + 2 * PG_ONLINE_WARPS * sizeof(float);  // 32 bytes
// = 1312 bytes，完全固定
```

这使得 CUDA Graph capture 的 smem 参数不需要每步更新。

#### CUDA Graph 预同步

在 CUDA Graph 执行路径中，block table 和 pos 的更新发生在 graph launch 之前：

```cpp
// qwen3.cpp forward_with_cuda_graph()（概念流程）：
// Step 1：更新 GPU pos（graph 外部）
cudaMemcpyAsync(pos_gpu, &current_pos, sizeof(int32_t), H2D, stream);

// Step 2：确保 block table 已同步（graph 外部）
if (use_paged_attention_) {
    paged_kv_cache_manager_->ensure_allocated_to(current_pos);
    paged_kv_cache_manager_->sync_block_table();
}

// Step 3：launch captured graph（内部使用更新后的 pos 和 block table）
cudaGraphLaunch(graph_exec, stream);
```

### 13.4 PagedAttention + Prefix Cache 的协作

#### Prefix Cache 的工作原理

Prefix Cache（本工程使用 RadixTree 实现）缓存之前对话轮次的 KV cache，避免重复计算。

#### 当前实现：两者独立

当前 Prefix Cache 与 PagedAttention 是独立的：
- Prefix Cache 在 `inference_common.h` 中管理 token 序列级别的缓存
- PagedAttention 在 `PagedKVCacheManager` 中管理物理内存级别的分页
- 两者通过 Model 层接口组合使用

```cpp
// inference_common.h — prefix cache 初始化
std::unique_ptr<PrefixCacheManager> cache_manager;
if (config.use_prefix_cache) {
    cache_manager = std::make_unique<PrefixCacheManager>(config.prefix_cache_size);
}
```

#### Prefix Cache hit 时的 paged 路径

当 prefix cache 命中时，之前的 KV 已经在 cache 中：
1. Prefix Cache 提供已有的 KV 数据
2. PagedAttention 的 `ensure_allocated_to()` 分配新 page
3. KV 数据写入 paged pool 的对应物理位置

这种组合中 Prefix Cache 减少了计算（避免 re-prefill），而 PagedAttention 提高了内存效率。

### 13.5 清理时的三者协调

```cpp
// qwen3.cpp clear_kv_cache() 第2108-2112行
void Qwen3Model::clear_kv_cache() {
  if (use_paged_attention_ && paged_kv_cache_manager_) {
    paged_kv_cache_manager_->clear();    // 1. 清理 paged 块
    invalidate_cuda_graph();              // 2. CUDA Graph 必须重新 capture
    return;
  }
  // 连续路径...
  invalidate_cuda_graph();
}
```

**关键**：`clear()` 之后必须 `invalidate_cuda_graph()`，因为：
- block table 已重置为全 -1
- 已 captured 的 graph 中的 block_table 内容已过时
- 下一轮对话需要重新 capture graph

### 13.6 组合使用的完整时序

以 "Paged + Graph + Prefix Cache" 全开为例：

```
对话轮次 1："你好，请介绍一下你自己"
  1. Prefix Cache miss → 需要完整 prefill
  2. ensure_allocated_to(33) → 分配 3 个 logical block × 36 层
  3. sync_block_table() → 72KB CPU→GPU
  4. Prefill：batched_attention_qkv + batched_attention_mha （分页路径）
  5. Decode with CUDA Graph:
     a. 第一步：capture graph（paged kernels 被录入）
     b. 后续步：replay graph（pos 和 block_table 值已更新）
  6. Prefix Cache 保存 KV cache 状态

对话轮次 2："你能写代码吗？"（多轮对话）
  1. Prefix Cache 查找 → 部分命中（system prompt 部分）
  2. clear_kv_cache() → paged pool 清零，graph invalidated
  3. 从 prefix cache 恢复命中部分的 KV
  4. 只 prefill 新增的 token
  5. 重新 capture CUDA Graph（因为 block_table 状态改变）
  6. Decode 继续...
```

---

## 第十四章：各模型带 PagedAttention 的运行指令

### 14.1 Qwen2.5-7B FP32

```bash
cd /mnt/ssd/workspace/OrinMLLM/build/demo

echo "Hello" | ./qwen_infer \
    /mnt/ssd/QwenModels/Qwen2.5-7B.bin \
    /mnt/ssd/QwenModels/Qwen2.5-7B-Instruct/tokenizer.json \
    --attention flash1 \
    --paged-attention \
    --no-cuda-graph \
    --max-tokens 256
```

交互模式：

```bash
./qwen_infer \
    /mnt/ssd/QwenModels/Qwen2.5-7B.bin \
    /mnt/ssd/QwenModels/Qwen2.5-7B-Instruct/tokenizer.json \
    --attention flash1 \
    --paged-attention \
    --prefix-cache \
    --interactive
```

### 14.2 Qwen2.5-7B FP16

```bash
cd /mnt/ssd/workspace/OrinMLLM/build/demo

echo "Hello" | ./qwen_infer \
    /mnt/ssd/QwenModels/Qwen2.5-7B-fp16.bin \
    /mnt/ssd/QwenModels/Qwen2.5-7B-Instruct/tokenizer.json \
    --attention flash1 \
    --paged-attention \
    --no-cuda-graph \
    --max-tokens 256
```

交互模式（推荐，带 CUDA Graph + Prefix Cache）：

```bash
./qwen_infer \
    /mnt/ssd/QwenModels/Qwen2.5-7B-fp16.bin \
    /mnt/ssd/QwenModels/Qwen2.5-7B-Instruct/tokenizer.json \
    --attention flash1 \
    --paged-attention \
    --prefix-cache \
    --cuda-graph \
    --interactive
```

### 14.3 Qwen3-8B FP16

```bash
cd /mnt/ssd/workspace/OrinMLLM/build/demo

echo "Hello" | ./qwen3_infer \
    /mnt/ssd/QwenModels/Qwen3-8B-fp16.bin \
    /mnt/ssd/QwenModels/Qwen3-8B/tokenizer.json \
    --attention flash1 \
    --paged-attention \
    --no-cuda-graph \
    --max-tokens 256
```

交互模式（推荐）：

```bash
./qwen3_infer \
    /mnt/ssd/QwenModels/Qwen3-8B-fp16.bin \
    /mnt/ssd/QwenModels/Qwen3-8B/tokenizer.json \
    --attention flash1 \
    --paged-attention \
    --prefix-cache \
    --cuda-graph \
    --interactive
```

### 14.4 Qwen3-8B AWQ (INT4 量化)

```bash
cd /mnt/ssd/workspace/OrinMLLM/build/demo

echo "Hello" | ./qwen3_infer \
    /mnt/ssd/QwenModels/Qwen3-8B-awq.bin \
    /mnt/ssd/QwenModels/Qwen3-8B-awq/tokenizer.json \
    --attention flash1 \
    --paged-attention \
    --no-cuda-graph \
    --max-tokens 256
```

交互模式：

```bash
./qwen3_infer \
    /mnt/ssd/QwenModels/Qwen3-8B-awq.bin \
    /mnt/ssd/QwenModels/Qwen3-8B-awq/tokenizer.json \
    --attention flash1 \
    --paged-attention \
    --prefix-cache \
    --cuda-graph \
    --interactive
```

### 14.5 Qwen3-VL-8B FP16（视觉语言模型）

```bash
cd /mnt/ssd/workspace/OrinMLLM/build/demo

# 单张图片推理
./qwen3_vl_infer \
    /mnt/ssd/QwenModels/Qwen3-VL-8B-fp16.bin \
    /mnt/ssd/QwenModels/Qwen3-VL-8B-Instruct/tokenizer.json \
    --image /path/to/image.jpg \
    --attention flash1 \
    --paged-attention \
    --cuda-graph \
    --max-tokens 256
```

交互模式：

```bash
./qwen3_vl_infer \
    /mnt/ssd/QwenModels/Qwen3-VL-8B-fp16.bin \
    /mnt/ssd/QwenModels/Qwen3-VL-8B-Instruct/tokenizer.json \
    --image /path/to/image.jpg \
    --attention flash1 \
    --paged-attention \
    --prefix-cache \
    --cuda-graph \
    --interactive
```

### 14.6 指令速查表

| 模型 | 可执行文件 | 模型文件 | Tokenizer |
|------|-----------|---------|-----------|
| Qwen2.5-7B FP32 | `qwen_infer` | `Qwen2.5-7B.bin` | `Qwen2.5-7B-Instruct/tokenizer.json` |
| Qwen2.5-7B FP16 | `qwen_infer` | `Qwen2.5-7B-fp16.bin` | `Qwen2.5-7B-Instruct/tokenizer.json` |
| Qwen3-8B FP16 | `qwen3_infer` | `Qwen3-8B-fp16.bin` | `Qwen3-8B/tokenizer.json` |
| Qwen3-8B AWQ | `qwen3_infer` | `Qwen3-8B-awq.bin` | `Qwen3-8B-awq/tokenizer.json` |
| Qwen3-VL-8B FP16 | `qwen3_vl_infer` | `Qwen3-VL-8B-fp16.bin` | `Qwen3-VL-8B-Instruct/tokenizer.json` |

**通用 PagedAttention 参数**：

```
--paged-attention     # 启用分页 KV Cache
--attention flash1    # 使用 FlashAttention1（与 paged 兼容）
--prefix-cache        # [可选] 启用 RadixTree 前缀缓存
--cuda-graph          # [可选] 启用 CUDA Graph 优化 decode
--interactive / -i    # [可选] 交互式多轮对话模式
--max-tokens N        # [可选] 最大生成 token 数（默认 256）
```

---

## 第十五章：CUDA Graph + PagedAttention 乱码问题的分析与修复

### 15.1 问题现象

在启用 `--paged-attention --cuda-graph` 运行 Qwen2.5-7B-fp16 时，前几个 token 输出正确，之后迅速退化为乱码：

```
>>> 你好，请简单介绍一下你自己！
你好！我叫Qwen，是由阿里云开发的大型语言模型|\\\\\\\\\\\\== = =
** ** ** *  agle ermög ermög ApiController ApiController啻啻 ApiController
ApiControllerovel ApiController ApiController ApiController...
```

关键现象：
- **前 ~10 个 token 正确**（"你好！我叫Qwen，是由阿里云开发的大型语言模型"）
- **第 ~11 个 token 开始乱码**
- **不带 `--paged-attention` 时输出完全正常**
- **不带 `--cuda-graph` 但带 `--paged-attention` 时也正常**（即 `--no-cuda-graph --paged-attention`）

→ 问题只在 **CUDA Graph + Paged Attention 同时启用** 时出现。

### 15.2 分析思路

page_size=16，前 ~10 个 token 正确意味着第一个 page（token 0-15）工作正常。乱码从 token ~11 开始（加上 prompt 的 ~35 token，实际 KV pos ≈ 46），恰好跨过第 3 个 page 边界（pos 48 = page 3）。

这立即指向一个假设：**跨 page 边界时 block 没有被正确分配**。

### 15.3 根因定位

对比两条 decode 路径的代码：

**正常路径**（`attention_qkv`，无 CUDA Graph）：

```cpp
// qwen2.cpp 第968-972行
void Qwen2Model::attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  int32_t pos = pos_tensor.index<int32_t>(0);
  // ✅ 每层每步都会调用 ensure_allocated_to + sync
  if (use_paged_attention_ && paged_kv_cache_manager_) {
    paged_kv_cache_manager_->ensure_allocated_to(pos);
    paged_kv_cache_manager_->sync_block_table();
  }
  // ... QKV 计算 ...
}
```

**CUDA Graph 路径**（`decode()` 函数中）：

```cpp
// qwen2.cpp decode() — 修复前
bool use_graph = cuda_config_ && cuda_config_->should_use_graph();
if (use_graph) {
    // ① 拷贝 input 到固定地址 buffer
    // ② 拷贝 pos 到 GPU
    // ❌ 缺失：没有调用 ensure_allocated_to / sync_block_table
    // ③ 直接进入 graph capture 或 graph launch
    if (need_capture) {
        graph->begin_capture(stream);
        for (layer_idx ...) {
            attention_rms(...);
            attention_qkv_with_graph(...);  // ← 内部也没有 ensure/sync
            attention_mha_with_graph(...);
            feed_forward(...);
        }
        graph->end_capture(stream);
    }
    graph->launch(stream);  // ← block table 是旧的！
}
```

**`attention_qkv_with_graph()` 内部**也没有调用 `ensure_allocated_to()` —— 因为在 CUDA Graph capture 期间不能做 `cudaMemcpy`（会被录入 graph 导致重复拷贝），所以当初设计时跳过了。但这导致了一个遗漏：**在 graph capture/launch 之前也没有执行这些操作**。

### 15.4 根因分析：为什么前几个 token 正确？

```
Prefill 阶段（35 token prompt）：
  batched_attention_qkv() 中调用 ensure_allocated_to(34)
  → 分配了 logical blocks 0, 1, 2（覆盖 pos 0-47）
  → sync_block_table() 将映射写入 GPU

Decode 开始（pos = 35, 36, 37, ...）：
  pos 35-47 仍在 logical block 2 内（已在 prefill 时分配）
  → block table 中有对应的 physical block
  → 输出正确

Decode 到 pos = 48（跨入 logical block 3）：
  CUDA Graph 路径中 ❌ 没有调用 ensure_allocated_to(48)
  → logical block 3 在 block_table 中仍为 -1
  → GPU 上的 block_table 仍是旧值
  → paged_off() 查表得到 -1，计算出的物理偏移 = (-1 * 16 + 0) * kv_dim
  → 读取未初始化/越界内存 → 乱码
```

这解释了为什么正好是前 13 个 decode token（pos 35-47）正确，第 14 个开始乱码。

### 15.5 修复方案

在 CUDA Graph decode 路径中，**pos 更新之后、graph capture/launch 之前**，插入 block 分配和 table 同步：

```cpp
// qwen2.cpp decode() — 修复后
if (use_graph) {
    // ① 拷贝 input 到固定地址 buffer
    cudaMemcpyAsync(decode_input, input, ...);

    // ② 拷贝 pos 到 GPU
    *pos_pinned = pos;
    cudaMemcpyAsync(pos_tensor_gpu, pos_pinned, ...);

    // ③ ✅ 新增：分配 block 并同步 block table
    if (use_paged_attention_ && paged_kv_cache_manager_) {
      paged_kv_cache_manager_->ensure_allocated_to(pos);
      paged_kv_cache_manager_->sync_block_table();
    }

    // ④ Graph capture 或 launch
    if (need_capture) { ... }
    graph->launch(stream);
}
```

**关键设计要点**：

1. `ensure_allocated_to()` 和 `sync_block_table()` 必须在 **graph capture 之外**执行
   - `ensure_allocated_to()` 涉及 CPU 端的 free_list 操作和 block_table_cpu 写入
   - `sync_block_table()` 执行 `cudaMemcpyAsync(H2D)`，如果录入 graph 会每次 replay 都重复拷贝旧数据

2. 它们必须在 **pos 更新之后**执行，因为需要知道当前 pos 来决定是否需要新 block

3. 它们必须在 **graph launch 之前**执行，因为 graph 内的 kernel 需要读取最新的 block table

### 15.6 第二个问题：FP32 + Paged + CUDA Graph

修复上述问题后，FP16 模型全部正常。但 FP32 模型（Qwen2.5-7B.bin）仍然乱码。

**原因**：CUDA Graph 路径的 `attention_mha_with_graph()` 中：

```cpp
// FP16 路径：使用 paged flash attention kernel（支持 paged pool + GPU pos）
if (query.data_type() == kDataTypeFp16) {
    flash_attn->set_paged_mode(true, ...);  // ✅ 从 paged pool 读 KV
    flash_attn->forward();
}
// FP32 路径：使用 mha_gpu_pos_layer_（只支持连续 KV cache）
else {
    mha_gpu_pos_layer_->forward(key_cache, val_cache);  // ❌ 从连续 buffer 读 KV
    // 但 paged 模式下 KV 写入了 paged pool，contiguous buffer 是空的！
}
```

没有 FP32 paged GPU-pos kernel 可用。**修复方案**：当 FP32 + Paged Attention 时自动禁用 CUDA Graph，回退到正常 decode 路径：

```cpp
bool use_graph = cuda_config_ && cuda_config_->should_use_graph();
// FP32 + Paged Attention 无兼容的 GPU-pos paged kernel，禁用 Graph
if (use_graph && !is_fp16_model_ && use_paged_attention_) {
    use_graph = false;
}
```

### 15.7 修复涉及的文件

| 文件 | 修改内容 |
|------|-------|
| `qwen2.cpp` `decode()` | 添加 ensure_allocated_to + sync_block_table；FP32+Paged 禁用 Graph |
| `qwen3.cpp` `decode()` | 同上 |
| `qwen3_vl.cpp` `decode_step()` | 添加 ensure_allocated_to + sync_block_table |
| `qwen3_vl.cpp` `decode_step_optimized()` | 同上 |

### 15.8 修复后验证

| 模型 | 配置 | 修复前 | 修复后 |
|------|------|--------|--------|
| Qwen2.5-7B FP32 | paged + graph | 乱码 | ✅ 正确（Graph 自动禁用） |
| Qwen2.5-7B FP16 | paged + graph | 乱码 | ✅ 正确，与 baseline 位级一致 |
| Qwen3-8B FP16 | paged + graph | 乱码 | ✅ 正确 |
| Qwen3-8B AWQ | paged + graph | 乱码 | ✅ 正确 |
| Qwen3-VL-8B FP16 | paged + graph | 乱码 | ✅ 正确 |

### 15.9 经验总结

1. **CUDA Graph 与动态内存管理是天然矛盾**
   - Graph 要求所有参数地址和值在 capture 时固定
   - PagedAttention 的 block table 内容每次 decode 都可能变化
   - 解决方案：将动态部分（block 分配 + table 同步）放在 **graph 外面**执行

2. **跨页边界是 PagedAttention 最容易出 bug 的地方**
   - 在同一个 page 内，block table 不变，容易通过测试
   - 跨 page 时才需要新 block，是 boundary condition
   - 测试用例必须包含跨页的场景（token 数 > page_size）

3. **"前 N 个 token 正确，之后乱码" 的模式是地址翻译错误的标志**
   - 计算逻辑没有 bug（正确的 token 证明了这一点）
   - 数据访问地址出错（指向了未初始化或错误的内存）
   - 错误位置通常与 page_size 的倍数相关

---

## 第十六章：PagedAttention 与连续 KV Cache 的显存使用对比

### 16.1 GPU 内存总分配量对比

在当前**单序列**设计中，Paged KV Cache 的总池大小与连续 KV Cache **完全相同**：

#### Qwen2.5-7B FP16 (28 层, kv_dim=512, max_seq_len=8192)

| 方案 | Key Cache | Value Cache | Block Table | 总计 |
|------|-----------|-------------|-------------|------|
| **连续 KV** | 28×8192×512×2 = 224 MB | 224 MB | — | **448 MB** |
| **Paged KV** | 14336块×16×512×2 = 224 MB | 224 MB | 56 KB | **448 MB + 56 KB** |

#### Qwen3-8B FP16 (36 层, kv_dim=1024, max_seq_len=8192)

| 方案 | Key Cache | Value Cache | Block Table | 总计 |
|------|-----------|-------------|-------------|------|
| **连续 KV** | 36×8192×1024×2 = 576 MB | 576 MB | — | **1,152 MB** |
| **Paged KV** | 18432块×16×1024×2 = 576 MB | 576 MB | 72 KB | **1,152 MB + 72 KB** |

> **结论**：总分配量基本一致，Paged 模式多了 ~60-70 KB 的 block table 开销（可以忽略）。

### 16.2 为什么总分配量相同？

因为当前设计中：

```cpp
// paged_kv_cache.cpp 第27行
num_blocks_ = num_layers * max_blocks_per_seq_;
max_blocks_per_seq_ = (max_seq_len + page_size - 1) / page_size;
```

total_blocks = num_layers × ⌈max_seq_len / page_size⌉，乘上 page_size × kv_dim × dtype_size，恰好等于连续分配的 num_layers × max_seq_len × kv_dim × dtype_size。

这是因为 block pool 预先为**单个序列的最大长度**分配了所有可能需要的块。

### 16.3 实际 KV 数据占用对比

总分配量相同，但**实际被使用的内存量**有显著差异：

#### 短对话场景（50 tokens）

| 方案 | 分配量 | 实际使用量 | 利用率 |
|------|--------|-----------|--------|
| **连续 KV (Qwen2.5-7B)** | 448 MB | 448 MB（整块分配无法分离） | 0.61% 有效数据 |
| **Paged KV (Qwen2.5-7B)** | 448 MB | 3.5 MB（112 blocks） | 0.78% pool 利用率 |
| **连续 KV (Qwen3-8B)** | 1,152 MB | 1,152 MB | 0.61% 有效数据 |
| **Paged KV (Qwen3-8B)** | 1,152 MB | 9.0 MB（144 blocks） | 0.78% pool 利用率 |

```
50 tokens → 4 logical blocks per layer (⌈50/16⌉ = 4)

Qwen2.5-7B: 4 blocks × 28 layers = 112 blocks
  实际内存: 112 × 16 × 512 × 2 bytes = 3.5 MB (占 pool 的 0.78%)

Qwen3-8B:   4 blocks × 36 layers = 144 blocks
  实际内存: 144 × 16 × 1024 × 2 bytes = 9.0 MB (占 pool 的 0.78%)
```

#### 不同对话长度的使用量对比

| 对话 token 数 | Page Blocks/层 | Qwen3-8B 实际使用 | 占 Pool 比例 |
|:---:|:---:|:---:|:---:|
| 50 | 4 | 9 MB | 0.78% |
| 200 | 13 | 29.3 MB | 2.54% |
| 500 | 32 | 72 MB | 6.25% |
| 1000 | 63 | 141.8 MB | 12.30% |
| 2000 | 125 | 281.3 MB | 24.41% |
| 4096 | 256 | 576 MB | 50.00% |
| 8192 | 512 | 1,152 MB | 100.00% |

### 16.4 连续 vs Paged 的内存分配方式对比

```
连续 KV Cache (不使用 PagedAttention):
┌─────────────────────────────────────────────────────────────────────┐
│ key_cache[layer_0]  ← pos 0 ... pos 8191  (全部预分配)              │
├─────────────────────────────────────────────────────────────────────┤
│ key_cache[layer_1]  ← pos 0 ... pos 8191                           │
├─────────────────────────────────────────────────────────────────────┤
│ ...                                                                 │
├─────────────────────────────────────────────────────────────────────┤
│ key_cache[layer_35] ← pos 0 ... pos 8191                           │
└─────────────────────────────────────────────────────────────────────┘
 total: 576 MB 连续内存, 一次性 cudaMalloc

Paged KV Cache (使用 PagedAttention):
┌──────┬──────┬──────┬──────┬──────┬──────┬─── ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┐
│blk 0 │blk 1 │blk 2 │blk 3 │ ...  │blk143│ blk 144 ... blk 18431 │
│(L0B0)│(L0B1)│(L0B2)│(L0B3)│      │(L3B3)│    (未分配, 空闲)       │
│ 32KB │ 32KB │ 32KB │ 32KB │      │ 32KB │                         │
└──────┴──────┴──────┴──────┴──────┴──────┴─── ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┘
 pool: 576 MB 预分配, 但只有前 144 个 block 被映射使用 (50 tokens)
 实际写入: 9 MB
```

### 16.5 单序列 vs 多序列场景的内存优势

| 场景 | 连续 KV | Paged KV | Paged 优势 |
|------|---------|----------|------------|
| **单序列, 长对话** | 使用 100% | 使用 100% | 无优势 |
| **单序列, 短对话** | 分配 100%, 用 ~1% | 分配 100%, 实际只写入 ~1% | 未使用的 block 可供其他用途 |
| **2 序列并发** | 需要 2× 连续块（可能 OOM） | 两序列共享 block pool | **节省 50% 以上** |
| **4 序列并发** | 需要 4× 连续块（几乎必定 OOM） | 四序列共享 block pool | **节省 75% 以上** |

### 16.6 Orin 统一内存架构下的实际影响

在 Orin 上，GPU 和 CPU 共享同一块 LPDDR5 内存。`cudaMalloc` 在统一内存下的行为是：

1. **虚拟地址空间预留**：`cudaMalloc(576 MB)` 立即预留 576 MB 虚拟地址
2. **物理页按需映射**：只有在首次访问时才真正分配物理页（Linux 的 overcommit 行为）
3. **实际物理内存占用**：取决于被访问的页数

因此在 Orin 上：
- **连续 KV Cache**：虽然分配 576 MB，但 prefill 阶段会**顺序遍历整个 cache**（初始化为 0），触发所有物理页分配 → 实际占用约 576 MB
- **Paged KV Pool**：pool 也被 `cudaMemset(0)` 初始化，同样会触发全部物理页 → 实际占用也约 576 MB

> **结论**：在当前实现下（pool 初始化时 memset 0），Orin 上两种方式的实际物理内存占用基本相同。如果将来改为**惰性初始化**（只 memset 被分配的 block），Paged 模式可以真正节省物理内存。

### 16.7 额外内存开销对比

| 开销项 | 连续 KV | Paged KV | 差异 |
|--------|---------|----------|------|
| Block Table (CPU) | — | 56-72 KB | +72 KB |
| Block Table (GPU) | — | 56-72 KB | +72 KB |
| Free List (CPU) | — | ~74 KB (18432×4 bytes) | +74 KB |
| Placeholder Buffers | — | 2×(1 element) | ~4 bytes |
| **额外开销总计** | **0** | **~218 KB** | **可忽略** |

### 16.8 性能影响

| 指标 | 连续 KV | Paged KV | 说明 |
|------|---------|----------|------|
| Decode 速度 (Qwen2.5-7B FP16) | 10.85 tok/s | 10.97 tok/s | 基本一致 |
| Prefill 速度 (Qwen2.5-7B FP16) | 155 tok/s | 153 tok/s | Paged 有微小开销 |
| 地址计算 | 直接偏移（1 次乘法） | 查表+偏移（1 次读+2 次运算） | 多 1 次内存读（L2 命中） |
| Block Table 同步 | 无 | ~1μs/step | 统一内存下接近零 |

---

## 第十七章：如何证明 PagedAttention 已生效

### 17.1 证明方法 1：观察启动日志

**最直接的方法**：检查模型启动时的日志输出。

```bash
# 启用 Paged Attention
$ echo "Hello" | ./qwen3_infer model.bin tok.json --attention flash1 --paged-attention 2>&1 | grep -i paged

# 预期输出：
I... paged_kv_cache.cpp:28] PagedKVCache: page_size=16, max_blocks_per_seq=512, total_blocks=18432, dtype=FP16
I... paged_kv_cache.cpp:53] PagedKVCache: allocated 1152 MB GPU memory (pools: 2x576 MB, block_table: 72 KB)
I... qwen3.cpp:874] PagedAttention: Created paged KV cache manager
I... inference_common.h:225] Paged Attention: enabled
```

**对比：未启用时**

```bash
$ echo "Hello" | ./qwen3_infer model.bin tok.json --attention flash1 2>&1 | grep -i paged

# 预期输出：
I... inference_common.h:225] Paged Attention: disabled
# 没有 PagedKVCache 相关日志
```

**关键日志含义**：

| 日志 | 含义 |
|------|------|
| `PagedKVCache: page_size=16` | Block pool 按 16 token 分页 |
| `total_blocks=18432` | 总计 18432 个物理块（36层×512块/层） |
| `allocated 1152 MB GPU memory` | Key Pool + Value Pool 分配成功 |
| `PagedAttention: Created paged KV cache manager` | Manager 实例创建成功 |
| `Paged Attention: enabled` | 配置确认已启用 |

### 17.2 证明方法 2：输出一致性对比

PagedAttention 不应改变计算结果。如果输出**完全相同**，说明 paged 地址翻译正确工作。

```bash
# Baseline（连续 KV）
echo "Hello" | ./qwen3_infer \
    /mnt/ssd/QwenModels/Qwen3-8B-fp16.bin \
    /mnt/ssd/QwenModels/Qwen3-8B/tokenizer.json \
    --attention flash1 --no-cuda-graph --max-tokens 15 2>&1 | \
    grep "Assistant:" > /tmp/baseline.txt

# Paged Attention
echo "Hello" | ./qwen3_infer \
    /mnt/ssd/QwenModels/Qwen3-8B-fp16.bin \
    /mnt/ssd/QwenModels/Qwen3-8B/tokenizer.json \
    --attention flash1 --no-cuda-graph --paged-attention --max-tokens 15 2>&1 | \
    grep "Assistant:" > /tmp/paged.txt

# 对比
diff /tmp/baseline.txt /tmp/paged.txt && echo "✅ MATCH: PagedAttention 输出正确" || echo "❌ DIFFER"
```

**实际验证结果**（5 个模型全部通过）：

```
Qwen2.5-7B FP32:  ✅ MATCH
Qwen2.5-7B FP16:  ✅ MATCH
Qwen3-8B FP16:    ✅ MATCH
Qwen3-8B AWQ:     ✅ MATCH
Qwen3-VL-8B FP16: ✅ MATCH
```

### 17.3 证明方法 3：添加运行时日志追踪 Block 分配

在 `ensure_allocated_to()` 中添加条件日志，可以直接观察 block 分配过程：

```cpp
// paged_kv_cache.cpp ensure_allocated_to() — 添加追踪日志
void PagedKVCacheManager::ensure_allocated_to(int32_t pos) {
  if (pos <= allocated_pos_) return;

  for (int32_t p = allocated_pos_ + 1; p <= pos; ++p) {
    int32_t logical_block = p / page_size_;
    for (int32_t layer = 0; layer < num_layers_; ++layer) {
      int32_t table_idx = layer * max_blocks_per_seq_ + logical_block;
      if (block_table_cpu_[table_idx] == -1) {
        int32_t phys = allocate_block();
        block_table_cpu_[table_idx] = phys;
        // 追踪日志
        LOG_EVERY_N(INFO, 100) << "PagedKV: allocated block"
            << " layer=" << layer << " logical=" << logical_block
            << " -> physical=" << phys
            << " (free_list remaining: " << free_list_.size() << ")";
      }
    }
  }
  allocated_pos_ = pos;
}
```

运行时可观察到类似输出：

```
PagedKV: allocated block layer=0 logical=0 -> physical=0 (free_list remaining: 18431)
PagedKV: allocated block layer=0 logical=1 -> physical=36 (free_list remaining: 18395)
PagedKV: allocated block layer=0 logical=2 -> physical=72 (free_list remaining: 18359)
...
```

### 17.4 证明方法 4：检查 Block Table 内容

在 sync_block_table 后 dump block table 内容，可以看到逻辑→物理映射：

```cpp
// 调试代码：dump block table
void PagedKVCacheManager::dump_block_table(int32_t max_pos) const {
  int32_t max_logical = (max_pos + page_size_ - 1) / page_size_;
  LOG(INFO) << "Block Table (pos 0-" << max_pos << ", " << max_logical << " logical blocks):";
  for (int32_t layer = 0; layer < std::min(num_layers_, 3); ++layer) {
    std::string row = "  L" + std::to_string(layer) + ": [";
    for (int32_t b = 0; b < max_logical && b < 10; ++b) {
      int32_t phys = block_table_cpu_[layer * max_blocks_per_seq_ + b];
      row += std::to_string(phys) + ", ";
    }
    row += "...]";
    LOG(INFO) << row;
  }
  LOG(INFO) << "  Free blocks remaining: " << free_list_.size()
            << " / " << num_blocks_;
}
```

对于一个 50 token 的推理（Qwen3-8B, 36 层），预期输出：

```
Block Table (pos 0-49, 4 logical blocks):
  L0:  [0, 36, 72, 108, ...]
  L1:  [1, 37, 73, 109, ...]
  L2:  [2, 38, 74, 110, ...]
  Free blocks remaining: 18288 / 18432

已分配: 4 blocks/layer × 36 layers = 144 blocks
已使用: 144 / 18432 = 0.78%
```

### 17.5 证明方法 5：利用 free_list 大小验证按需分配

**核心论证**：如果 PagedAttention 真正在按需分配，则 `free_list_.size()` 应该随推理进程逐步减少，且减少量与 token 数成正比。

```
推理前:  free_list_.size() = 18432 (全部空闲)

Prefill (35 tokens → 3 logical blocks):
  新分配: 3 blocks × 36 layers = 108 blocks
  free_list_.size() = 18432 - 108 = 18324

Decode 到 pos=48 (跨入第 4 个 logical block):
  新分配: 1 block × 36 layers = 36 blocks
  free_list_.size() = 18324 - 36 = 18288

Decode 到 pos=63 (仍在第 4 个 logical block):
  无新分配
  free_list_.size() = 18288

Decode 到 pos=64 (跨入第 5 个 logical block):
  新分配: 1 block × 36 layers = 36 blocks
  free_list_.size() = 18288 - 36 = 18252

clear() 后:
  free_list_.size() = 18432 (全部归还)
```

**对比连续 KV**：连续 KV cache 在 init 时就分配了全部内存，没有按需分配的过程。

### 17.6 证明方法 6：CUDA Kernel 调度路径验证

通过日志确认执行了 paged 版本的 kernel 而非连续版本：

```cpp
// flash_attention.cpp forward() — paged 路径
if (paged_mode_) {
  if (use_fp16_ && use_gpu_pos_) {
    // ✅ 调用 paged_flash_attention_decode_fp16_gpu_pos_cu
    kernel::paged_flash_attention_decode_fp16_gpu_pos_cu(...);
  } else if (use_fp16_) {
    // ✅ 调用 paged_flash_attention_decode_fp16_cu
    kernel::paged_flash_attention_decode_fp16_cu(...);
  } else {
    // ✅ 调用 paged_flash_attention_decode_cu
    kernel::paged_flash_attention_decode_cu(...);
  }
  return base::error::Success();
}
// 以下是连续路径（paged_mode_=false 时才会执行到这里）
```

可以通过 `nsys profile` 或在 kernel launch 前加 `printf` 来确认执行了哪条路径。

### 17.7 完整验证示例

以 Qwen2.5-7B-fp16 为例，一次完整的端到端验证流程：

```bash
#!/bin/bash
MODEL=/mnt/ssd/QwenModels/Qwen2.5-7B-fp16.bin
TOK=/mnt/ssd/QwenModels/Qwen2.5-7B-Instruct/tokenizer.json
EXE=/mnt/ssd/workspace/OrinMLLM/build/demo/qwen_infer
PROMPT="Hello"

echo "=== Step 1: 验证 Paged 模式启动日志 ==="
echo "$PROMPT" | $EXE $MODEL $TOK --attention flash1 --paged-attention \
    --no-cuda-graph --max-tokens 5 2>&1 | grep -E "PagedKV|Paged Attention"

echo ""
echo "=== Step 2: 验证输出一致性 ==="
BASE=$(echo "$PROMPT" | $EXE $MODEL $TOK --attention flash1 \
    --no-cuda-graph --max-tokens 15 2>&1 | grep "Assistant:")
PAGED=$(echo "$PROMPT" | $EXE $MODEL $TOK --attention flash1 --paged-attention \
    --no-cuda-graph --max-tokens 15 2>&1 | grep "Assistant:")

if [ "$BASE" = "$PAGED" ]; then
    echo "✅ 输出完全一致 — PagedAttention 地址翻译正确"
else
    echo "❌ 输出不一致"
    echo "Baseline: $BASE"
    echo "Paged:    $PAGED"
fi

echo ""
echo "=== Step 3: 验证 CUDA Graph + Paged 兼容性 ==="
GRAPH=$(echo "$PROMPT" | $EXE $MODEL $TOK --attention flash1 --paged-attention \
    --cuda-graph --max-tokens 15 2>&1 | grep "Assistant:")

if [ "$BASE" = "$GRAPH" ]; then
    echo "✅ Graph+Paged 输出一致 — CUDA Graph 兼容性正确"
else
    echo "❌ Graph+Paged 输出不一致"
fi
```

实际运行输出：

```
=== Step 1: 验证 Paged 模式启动日志 ===
PagedKVCache: page_size=16, max_blocks_per_seq=512, total_blocks=14336, dtype=FP16
PagedKVCache: allocated 448 MB GPU memory (pools: 2x224 MB, block_table: 56 KB)
PagedAttention: Created paged KV cache manager
Paged Attention: enabled

=== Step 2: 验证输出一致性 ===
✅ 输出完全一致 — PagedAttention 地址翻译正确

=== Step 3: 验证 CUDA Graph + Paged 兼容性 ===
✅ Graph+Paged 输出一致 — CUDA Graph 兼容性正确
```

以上 3 个步骤分别证明了：
1. PagedAttention 模块确实被初始化和使用
2. Paged 地址翻译产生的计算结果与连续 KV 完全相同
3. CUDA Graph 与 Paged Attention 的组合也工作正常

三者共同证明 **PagedAttention 已正确生效**。

---

*报告完毕*
