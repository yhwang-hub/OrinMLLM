# OrinMLLM 工程 Prefix Cache 与 Radix Tree 深度分析报告

> 源码文件：  
> `kuiper/include/base/radix_tree.h` — RadixTree 核心数据结构  
> `kuiper/include/base/prefix_cache.h` — PrefixCache 上层缓存  
> `demo/inference_common.h` — 推理框架中的实际使用  
> `tools/test_radix_tree.cpp` / `tools/test_prefix_cache_full.cpp` — 单元测试

---

## 目录

1. [Prefix Cache 的实现原理与使用流程](#1-prefix-cache-的实现原理与使用流程)
2. [Radix Tree 的实现与详细解读](#2-radix-tree-的实现与详细解读)
3. [Prefix Cache 命中率的计算方式](#3-prefix-cache-命中率的计算方式)
4. [节点引用计数机制详解](#4-节点引用计数机制详解)
5. [Prefix Cache 在 Prefill 阶段的逐步使用详解（源码级）](#5-prefix-cache-在-prefill-阶段的逐步使用详解源码级)

---

## 1. Prefix Cache 的实现原理与使用流程

### 1.1 核心思想

Prefix Cache 基于 **SGLang 风格的 RadixAttention** 算法实现，核心思想是：

- 在 LLM 推理中，多个请求（尤其是多轮对话）通常共享大量相同的前缀 token（如系统提示 system prompt、历史对话内容等）
- 这些前缀 token 对应的 KV Cache 是相同的，如果每次都重新计算，浪费了大量算力
- Prefix Cache 使用 **RadixTree（基数树/压缩前缀树）** 存储 token 序列到 KV Cache 的映射关系
- 新请求到来时，先在 RadixTree 中查找最长公共前缀，复用已有的 KV Cache，只对新增部分做 prefill 计算

### 1.2 核心类结构概览

```
PrefixCache（上层封装）
  ├── PrefixCacheConfig    — 配置（最大缓存 token 数、最小前缀长度等）
  ├── PrefixCacheStats     — 统计信息（命中率、复用率、淘汰次数等）
  ├── PrefixMatchResult    — 匹配结果
  └── RadixTree（底层数据结构）
        ├── RadixNode          — 树节点
        ├── RadixMatchResult   — 底层匹配结果
        └── Stats              — 树统计信息
```

### 1.3 各结构体/类详细说明

#### `PrefixCacheConfig` — 配置

| 成员 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `max_cached_tokens` | `int64_t` | 65536 | 最大缓存 token 数（超过后触发 LRU 淘汰） |
| `min_prefix_length` | `int32_t` | 4 | 最小前缀匹配长度（太短的前缀不值得复用） |
| `enable_auto_eviction` | `bool` | true | 是否启用自动淘汰 |
| `eviction_threshold` | `float` | 0.9 | 淘汰阈值（使用率超过此值时触发淘汰） |
| `enable_stats` | `bool` | true | 是否启用统计 |

#### `PrefixMatchResult` — 前缀匹配结果

| 成员 | 类型 | 说明 |
|---|---|---|
| `matched_tokens` | `int32_t` | 匹配到的 token 数量 |
| `cache_hit` | `bool` | 是否命中缓存 |
| `prefill_start_pos` | `int32_t` | 从哪个位置开始 prefill |
| `prefill_count` | `int32_t` | 需要 prefill 的 token 数量 |
| `reuse_ratio` | `float` | KV Cache 复用比例 (0.0 - 1.0) |
| `matched_prefix` | `vector<int32_t>` | 匹配到的缓存 key（用于引用计数管理） |

#### `PrefixCacheStats` — 统计信息

| 成员 | 类型 | 说明 |
|---|---|---|
| `total_requests` | `atomic<int64_t>` | 总请求数 |
| `cache_hits` | `atomic<int64_t>` | 缓存命中次数 |
| `cache_misses` | `atomic<int64_t>` | 缓存未命中次数 |
| `total_tokens_processed` | `atomic<int64_t>` | 处理的总 token 数 |
| `tokens_reused` | `atomic<int64_t>` | 复用的 token 数 |
| `tokens_computed` | `atomic<int64_t>` | 需要重新计算的 token 数 |
| `eviction_count` | `atomic<int64_t>` | 淘汰触发次数 |
| `tokens_evicted` | `atomic<int64_t>` | 被淘汰的 token 数 |

**关键方法**：
- `hit_rate()` — 返回 `cache_hits / total_requests`
- `reuse_rate()` — 返回 `tokens_reused / total_tokens_processed`

#### `PrefixCache` 主类 — 成员函数

| 方法 | 参数 | 返回值 | 说明 |
|---|---|---|---|
| 构造函数 | `PrefixCacheConfig` | - | 初始化 RadixTree 和配置 |
| `match()` | `vector<int32_t> tokens` | `PrefixMatchResult` | **核心方法**：查找最长前缀匹配 |
| `insert()` | `tokens, kv_start_pos, kv_length` | `void` | 注册前缀到缓存 |
| `release()` | `vector<int32_t> tokens` | `void` | 释放前缀引用（请求完成时调用） |
| `remove()` | `vector<int32_t> tokens` | `bool` | 删除指定前缀 |
| `clear()` | - | `void` | 清空所有缓存 |
| `get_cached_tokens()` | - | `int64_t` | 获取当前缓存的 token 数 |
| `get_usage_ratio()` | - | `float` | 获取缓存使用率 |
| `evict()` | `int64_t target_tokens` | `int64_t` | 手动触发 LRU 淘汰 |
| `get_stats()` | - | `PrefixCacheStats&` | 获取统计信息 |
| `dump_tree()` | - | `string` | 打印 RadixTree 结构 |

### 1.4 使用流程详解

Prefix Cache 的完整使用流程可以分为 **5 个阶段**：

#### 阶段 1：初始化

在 `demo/inference_common.h` 的 `run_model_inference()` 函数中：

```cpp
// 根据配置决定是否启用 PrefixCache
std::unique_ptr<PrefixCacheManager> cache_manager;
if (config.use_prefix_cache) {
    cache_manager = std::make_unique<PrefixCacheManager>(config.prefix_cache_size);
}
```

`PrefixCacheManager` 是对 `PrefixCache` 的封装，内部创建：
```cpp
PrefixCacheManager(int64_t max_tokens = 65536) {
    base::PrefixCacheConfig config;
    config.max_cached_tokens = max_tokens;
    config.min_prefix_length = 4;        // 至少4个token才值得复用
    config.enable_auto_eviction = true;  // 自动淘汰
    config.eviction_threshold = 0.9f;    // 90%使用率时触发淘汰
    prefix_cache_ = std::make_unique<base::PrefixCache>(config);
}
```

#### 阶段 2：前缀匹配（在 prefill 之前）

在 `generate_response()` 函数中，收到新请求后，先查找缓存匹配：

```cpp
if (config.use_prefix_cache && cache_manager) {
    auto match_result = cache_manager->match(tokens_i32);
    if (match_result.cache_hit && match_result.matched_tokens > 0) {
        start_pos = match_result.matched_tokens;    // 从匹配位置开始prefill
        matched_prefix = match_result.matched_prefix; // 保存用于后续释放
        // 例如：总共100个token，匹配了80个，则只需prefill 20个
    } else {
        model.clear_kv_cache();   // 未命中，清空KV cache从头计算
        conv.reset_kv_state();
    }
}
```

`match()` 内部流程：
1. 调用 `radix_tree_->find_longest_prefix(tokens)` 在树中查找最长前缀
2. 如果匹配长度 ≥ `min_prefix_length`（默认4），认定为缓存命中
3. 命中后自动调用 `add_reference()` 增加引用计数
4. 更新统计信息（hits/misses/tokens_reused 等）
5. 返回 `PrefixMatchResult`

#### 阶段 3：增量 Prefill

只计算新增部分的 token embedding 和 attention：

```cpp
// 如果 start_pos > 0，说明前面的KV cache可以复用
if (start_pos > 0) {
    // 只取 tokens[start_pos:] 的 embedding 做 prefill
    tensor::Tensor new_embeddings(dtype, prefill_tokens, dim, true, alloc);
    // 只拷贝新增部分的embedding
    cudaMemcpyAsync(new_embeddings, src_ptr + start_pos * dim, ...);
    model.prefill(new_embeddings, prefill_tokens, start_pos);
} else {
    // 没有可复用的，全量prefill
    model.prefill(embedding_out, total_len, 0);
}
```

#### 阶段 4：注册前缀（在生成完成后）

生成完所有 token 后，将完整序列注册到 RadixTree：

```cpp
if (config.use_prefix_cache && cache_manager) {
    // 获取完整的token序列（包含新生成的token）
    std::vector<int32_t> final_tokens = conv.get_cached_tokens();
    cache_manager->register_prefix(final_tokens, final_tokens.size());
    
    // 释放之前匹配时增加的引用
    if (!matched_prefix.empty()) {
        cache_manager->release(matched_prefix);
    }
}
```

#### 阶段 5：自动淘汰

在 `insert()` 时自动检查，如果使用率超过阈值则触发 LRU 淘汰：

```cpp
void maybe_evict() {
    float usage = get_usage_ratio();  // 当前缓存量 / 最大容量
    if (usage > config_.eviction_threshold) {  // 默认0.9
        int64_t target = max_cached_tokens * 0.8;  // 淘汰到80%
        evict(target);
    }
}
```

### 1.5 实例详解：多轮对话场景

以下结合 `test_radix_tree.cpp` 中的 `test_multi_turn_simulation()` 进行说明：

```
场景：模拟 Qwen 模型的多轮对话

=== Turn 1 ===
输入 tokens（19个）:
  system: [151644, 8948, 198, 100, 101, 102, 151645, 198]  (8个)
  user1:  [151644, 872, 198, 1, 2, 3, 151645, 198]         (8个)
  asst:   [151644, 77091, 198]                               (3个)

操作: cache.insert(turn1_prompt, 0, 19)
树状态: ROOT → Edge[151644,8948,198,100,101...] *KV(pos=0,len=19)
此时缓存了19个token的KV Cache

=== Turn 1 生成回复后 ===
生成了10个token的回复: [200, 201, ..., 209]
完整序列（31个token）:
  turn1_prompt(19) + response(10) + [151645, 198](2)

操作: cache.insert(turn1_full, 0, 31)
树状态: 更新为31个token的完整序列

=== Turn 2 ===
输入 tokens（42个）:
  turn1_full(31) + user2: [151644, 872, 198, 4, 5, 6, 151645, 198] + asst

操作: match = cache.match(turn2_prompt)
结果:
  ✓ cache_hit = true
  ✓ matched_tokens = 31     （复用了Turn1的全部31个token的KV Cache）
  ✓ prefill_count = 11      （只需计算新增的11个token）
  ✓ reuse_ratio = 31/42 = 74%

效益: 42个token只需计算11个，节省了74%的prefill计算量！
```

### 1.6 与简单 KV Cache 复用的对比

| 特性 | 简单复用（MultiTurnConversation） | RadixTree PrefixCache |
|---|---|---|
| 复用范围 | 单个对话内 | 跨多个对话 |
| 匹配方式 | 线性逐token比较 | 树结构高效匹配 |
| 多请求共享 | 不支持 | 支持 |
| 内存管理 | 无 | LRU 淘汰 |
| 引用计数 | 无 | 有 |
| 时间复杂度 | O(n) | O(n)，但树深度更浅 |

---

## 2. Radix Tree 的实现与详细解读

### 2.1 RadixTree vs 普通前缀树（Trie）

**普通前缀树（Trie）**：
- 每个节点只存储一个字符/token
- 树的深度等于最长序列的长度
- 只有一个子节点的节点也占据一个层级

**RadixTree（基数树/压缩前缀树）**：
- 每个节点可以存储一个 **token 序列段**（多个连续 token）
- 将只有一个子节点的节点与其子节点**合并**，压缩路径
- 大大减少了树的深度和节点数量

**图示对比**：

```
普通 Trie（存储 [1,2,3,4,5] 和 [1,2,3,6,7]）:
ROOT → [1] → [2] → [3] → [4] → [5]*
                        → [6] → [7]*

RadixTree（压缩后）:
ROOT → Edge[1,2,3] → Edge[4,5]*
                   → Edge[6,7]*

普通Trie需要7个节点，RadixTree只需3个节点
```

RadixTree 的优势：
- **空间效率**：节点数量大幅减少（O(m) → O(k)，k为唯一序列数）
- **查询效率**：树深度更浅，减少指针跳转
- **缓存友好**：边上的 token 序列连续存储，利于CPU缓存

### 2.2 `RadixNode` 结构体详解

```cpp
struct RadixNode {
    std::vector<int32_t> edge_tokens;           // 边上的 token 序列
    std::unordered_map<int32_t, std::shared_ptr<RadixNode>> children;  // 子节点映射
    bool is_terminal = false;                    // 是否是完整序列的终点
    struct KVCacheInfo {
        int32_t kv_start_pos = 0;               // KV cache 起始位置
        int32_t kv_length = 0;                  // KV cache 长度
        int64_t last_access_time = 0;           // 最后访问时间（LRU）
        int32_t ref_count = 0;                  // 引用计数
    } kv_info;
    int32_t prefix_length = 0;                  // 从根到此节点的总 token 长度
    std::weak_ptr<RadixNode> parent;            // 父节点（弱引用避免循环引用）

    RadixNode() = default;
    explicit RadixNode(const std::vector<int32_t>& tokens);
    bool is_leaf() const;                       // 是否为叶子节点
};
```

#### 各成员详细说明

| 成员 | 说明 | 作用 |
|---|---|---|
| `edge_tokens` | 从父节点到当前节点的边上存储的 token 序列 | RadixTree 的核心——路径压缩的实体。普通 Trie 每条边只有1个token，而 RadixTree 的边可以是一个 token 子序列 |
| `children` | 子节点映射表，key 为子节点 `edge_tokens[0]`（第一个token） | 用于快速定位子节点。使用 `unordered_map` 保证 O(1) 查找 |
| `is_terminal` | 标记该节点是否代表一个完整的已插入序列的终点 | 区分"路径中间节点"和"有效数据节点"。只有 `is_terminal=true` 的节点才关联有效的 KV Cache |
| `kv_info` | KV Cache 元信息（位置、长度、访问时间、引用计数） | 将树节点与实际的 KV Cache 存储位置关联起来 |
| `kv_info.kv_start_pos` | KV Cache 在存储中的起始位置 | 告诉推理引擎从哪里读取缓存的 KV 数据 |
| `kv_info.kv_length` | KV Cache 的长度（token 数） | 记录该节点关联的 KV Cache 覆盖多少个 token 位置 |
| `kv_info.last_access_time` | 最后一次被访问的时间戳（毫秒级） | 用于 LRU 淘汰策略——越久未访问的越优先被淘汰 |
| `kv_info.ref_count` | 引用计数——多少个正在进行的请求正在使用此节点 | 保护正在使用的缓存不被淘汰 |
| `prefix_length` | 从根节点到当前节点经过的总 token 数 | 用于快速计算匹配长度和淘汰时释放的 token 数 |
| `parent` | 指向父节点的弱引用 | 支持从子节点向上遍历。使用 `weak_ptr` 避免与 `shared_ptr children` 形成循环引用 |

#### 成员函数说明

- **`RadixNode()`**：默认构造函数
- **`RadixNode(const vector<int32_t>& tokens)`**：用 token 序列初始化 `edge_tokens`
- **`is_leaf()`**：返回 `children.empty()`，判断是否为叶子节点。叶子节点是 LRU 淘汰的优先目标

### 2.3 `RadixMatchResult` 结构体详解

```cpp
struct RadixMatchResult {
    int32_t matched_length = 0;                // 匹配的 token 数量
    std::shared_ptr<RadixNode> matched_node;   // 匹配到的节点
    int32_t edge_offset = 0;                   // 在 edge_tokens 中匹配的偏移
    int32_t kv_cache_pos = -1;                 // KV cache 起始位置（-1表示无效）
    bool has_kv_cache() const;                 // 是否有有效的 KV cache
};
```

| 成员 | 说明 |
|---|---|
| `matched_length` | 从输入 token 序列开头开始，成功匹配的 token 总数。例如查询 `[1,2,3,4,5,6,7]` 匹配到 `[1,2,3,4,5]`，则 `matched_length=5` |
| `matched_node` | 指向匹配到的最后一个有效节点。如果是完全匹配某条边但节点非终端，记录当前节点；如果匹配到终端节点，记录最后的终端节点 |
| `edge_offset` | 当发生"部分边匹配"时，记录在当前边的 `edge_tokens` 中匹配到的位置。例如边 `[1,2,3,4,5]`，查询 `[1,2,3]`，则 `edge_offset=3` |
| `kv_cache_pos` | 匹配到的终端节点对应的 KV Cache 起始位置。`-1` 表示没有有效的 KV Cache 关联 |
| `has_kv_cache()` | 便捷方法：`kv_cache_pos >= 0 && matched_length > 0`，判断是否找到了可复用的 KV Cache |

### 2.4 `RadixTree` 类详解

#### 2.4.1 私有成员

```cpp
std::shared_ptr<RadixNode> root_;         // 根节点（空节点，prefix_length=0）
mutable std::mutex mutex_;                 // 互斥锁——保证线程安全
std::atomic<int64_t> total_cached_tokens_; // 缓存的总 token 数（原子量）
```

- **`root_`**：空根节点，不存储任何 token，作为所有序列的公共起点
- **`mutex_`**：用 `mutable` 修饰，使得 `const` 方法（如 `find_longest_prefix`）也能加锁。所有公开方法都通过 `lock_guard` 加锁，保证线程安全
- **`total_cached_tokens_`**：使用 `atomic` 保证对这个统计量的原子读取（虽然修改都在锁内）

#### 2.4.2 构造函数

```cpp
RadixTree() {
    root_ = std::make_shared<RadixNode>();
    root_->prefix_length = 0;
}
```

创建空根节点，`prefix_length=0` 表示根节点不携带任何 token 前缀。

#### 2.4.3 公开方法详解

##### `insert(tokens, kv_start_pos, kv_length)` — 插入

**功能**：将 token 序列插入树中，并关联 KV Cache 信息。

**内部实现 `insert_impl()` 流程**：

```
1. 如果 tokens 为空 → 标记根节点为终端并设置KV信息
2. 从根节点开始遍历：
   a. 取当前未处理token的第一个 first_token
   b. 在当前节点的 children 中查找 first_token
   c. 如果没找到 → 创建新节点，edge_tokens 为剩余所有token —— 直接创建
   d. 如果找到了 → 计算与 child 的 edge_tokens 的公共前缀长度
      i.  完全匹配边（common_len == edge.size()）→ 移到子节点继续
      ii. 部分匹配 → 需要"分裂节点" —— 这是 RadixTree 的核心操作
3. 如果遍历完所有token到达已有节点 → 标记该节点为终端
```

**实例：插入操作的节点分裂**

```
初始状态：树中已有 [1,2,3,4,5]
    ROOT → Edge[1,2,3,4,5]*  （* 表示终端节点）

插入 [1,2,3,6,7]：
Step 1: first_token=1，找到子节点 Edge[1,2,3,4,5]
Step 2: 比较 edge=[1,2,3,4,5] 与剩余tokens=[1,2,3,6,7]
        公共前缀=[1,2,3]，common_len=3，不等于 edge.size()=5
Step 3: 需要分裂！

分裂过程：
  a. 创建 split_node，edge_tokens=[1,2,3]，prefix_length=3
  b. 原节点 edge_tokens 修改为 [4,5]（去掉公共前缀）
  c. split_node 的 children 添加原节点（key=4）
  d. 创建新节点 new_node，edge_tokens=[6,7]，is_terminal=true
  e. split_node 的 children 添加 new_node（key=6）
  f. ROOT 的 children 更新：key=1 指向 split_node

最终状态：
    ROOT → Edge[1,2,3] → Edge[4,5]*
                       → Edge[6,7]*
```

```
继续插入 [1,2,8,9,10]：
Step 1: first_token=1，找到 Edge[1,2,3]
Step 2: 比较 edge=[1,2,3] 与 [1,2,8,9,10]
        公共前缀=[1,2]，common_len=2，不等于 edge.size()=3
Step 3: 分裂 Edge[1,2,3]!

分裂后：
    ROOT → Edge[1,2] → Edge[3] → Edge[4,5]*
                                → Edge[6,7]*
                     → Edge[8,9,10]*

三条序列共享前缀 [1,2]，其中 [1,2,3,4,5] 和 [1,2,3,6,7] 进一步共享 [1,2,3]
```

##### `find_longest_prefix(tokens)` — 查找最长前缀匹配

**功能**：在树中查找与输入 token 序列的最长公共前缀，关键是找最后一个 **终端节点**。

**内部实现 `find_longest_prefix_impl()` 流程**：

```
1. 维护两个关键变量：
   - last_terminal: 遍历过程中遇到的最后一个终端节点
   - last_terminal_length: 到该终端节点的匹配长度

2. 从根节点开始：
   a. 取 first_token，在 children 中查找
   b. 未找到 → 结束搜索
   c. 找到子节点 child → 逐token比较 child.edge_tokens
   d. 完全匹配边 → 如果 child 是终端，更新 last_terminal
   e. 部分匹配边 → 记录 edge_offset，结束搜索

3. 返回 last_terminal 的信息（而非最深到达的节点）
   因为只有终端节点才关联有效的 KV Cache
```

**实例**：

```
树状态（来自上面的例子）：
    ROOT → Edge[1,2] → Edge[3] → Edge[4,5]* (KV pos=0)
                                → Edge[6,7]* (KV pos=10)
                     → Edge[8,9,10]* (KV pos=20)

查找 [1,2,3,4,5,6,7]：
  Step 1: token[0]=1, 匹配 Edge[1,2], common_len=2, 完全匹配边
          Edge[1,2] 非终端, 不更新 last_terminal
  Step 2: token[2]=3, 匹配 Edge[3], common_len=1, 完全匹配边
          Edge[3] 非终端, 不更新 last_terminal（它是分裂产生的中间节点）
  Step 3: token[3]=4, 匹配 Edge[4,5], common_len=2, 完全匹配边
          Edge[4,5] 是终端！last_terminal=此节点, last_terminal_length=5
  Step 4: token[5]=6, 在 Edge[4,5] 的 children 中找不到 → 结束
  结果: matched_length=5, kv_cache_pos=0

查找 [1,2,3]：
  Step 1: token[0]=1, 匹配 Edge[1,2], common_len=2, 完全匹配边
          Edge[1,2] 非终端
  Step 2: token[2]=3, 匹配 Edge[3], common_len=1, 完全匹配边
          Edge[3] 非终端
  Step 3: tokens 已用完
  结果: matched_length=3, 但 last_terminal=nullptr（无终端节点）
         所以 matched_node=Edge[3], kv_cache_pos=-1
         has_kv_cache()=false → 不能复用！

查找 [1,2,8,9,10,100]：
  Step 1: token[0]=1, 匹配 Edge[1,2], common_len=2
  Step 2: token[2]=8, 匹配 Edge[8,9,10], common_len=3, 完全匹配边
          Edge[8,9,10] 是终端！last_terminal_length=5
  Step 3: token[5]=100, 无匹配子节点 → 结束
  结果: matched_length=5, kv_cache_pos=20
```

##### `contains(tokens)` — 检查序列是否存在

```cpp
bool contains(const std::vector<int32_t>& tokens) const {
    auto result = find_longest_prefix_impl(tokens);
    return result.matched_length == tokens.size() && 
           result.matched_node && result.matched_node->is_terminal;
}
```

要求**完全匹配**（matched_length 等于 tokens 长度）且匹配到的节点是**终端节点**。

##### `remove(tokens)` — 删除

**内部实现 `remove_impl()` 流程**：

```
1. 调用 find_longest_prefix_impl 查找
2. 验证：matched_length == tokens.size() 且 is_terminal → 否则返回 false
3. 将节点标记为非终端（is_terminal = false）
4. 清空 kv_info
5. 减少 total_cached_tokens_
注意：不删除节点本身——保留树结构（避免复杂的合并操作）
```

##### `add_reference(tokens)` / `release_reference(tokens)` — 引用计数

```cpp
void add_reference(const std::vector<int32_t>& tokens) {
    auto result = find_longest_prefix_impl(tokens);
    if (result.matched_node && result.matched_length == tokens.size()) {
        result.matched_node->kv_info.ref_count++;          // 增加引用
        result.matched_node->kv_info.last_access_time = get_current_time();  // 更新时间
    }
}

void release_reference(const std::vector<int32_t>& tokens) {
    auto result = find_longest_prefix_impl(tokens);
    if (...) {
        if (result.matched_node->kv_info.ref_count > 0) {
            result.matched_node->kv_info.ref_count--;      // 减少引用
        }
    }
}
```

##### `evict_lru(max_tokens)` — LRU 淘汰

**流程**：

```
1. 如果 total_cached_tokens_ <= max_tokens → 无需淘汰
2. collect_evictable_nodes(): 递归收集所有 is_terminal && is_leaf() 的节点
   （只淘汰叶子终端节点——它们没有子节点依赖）
3. 按 last_access_time 升序排序（最早被访问的在前面）
4. 从最早的开始淘汰：
   - 跳过 ref_count > 0 的节点（正在使用，不能淘汰！）
   - 将节点标记为非终端，清空 KV 信息
   - 减少 total_cached_tokens_
5. 直到 total_cached_tokens_ <= max_tokens 或无节点可淘汰
```

##### `get_stats()` — 获取统计信息

```cpp
struct Stats {
    int64_t total_nodes = 0;          // 总节点数
    int64_t terminal_nodes = 0;       // 终端节点数
    int64_t total_cached_tokens = 0;  // 总缓存 token 数
    int32_t max_depth = 0;            // 最大树深度
};
```

通过 `collect_stats()` 递归遍历所有节点统计。

##### `dump()` — 打印树结构

生成可读的树结构字符串，用于调试。输出格式：

```
ROOT
  Edge[1,2,3] 
    Edge[4,5] *KV(pos=0,len=5,ref=0)
    Edge[6,7] *KV(pos=10,len=5,ref=0)
  Edge[8,9,10] *KV(pos=20,len=5,ref=0)
```

##### `clear()` — 清空树

重建空根节点，重置 `total_cached_tokens_`。

##### `get_current_time()` — 获取当前时间

```cpp
static int64_t get_current_time() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
```

使用 `steady_clock`（单调递增时钟）获取毫秒级时间戳，用于 LRU 排序。

### 2.5 线程安全设计

所有公开方法都使用 `std::lock_guard<std::mutex> lock(mutex_)` 加锁：

```cpp
void insert(...) {
    std::lock_guard<std::mutex> lock(mutex_);  // 自动加锁，函数结束自动释放
    insert_impl(...);
}
```

- `mutex_` 声明为 `mutable`，允许在 `const` 方法中加锁
- `total_cached_tokens_` 使用 `atomic`，支持无锁读取
- 所有实际操作在 `_impl` 后缀的私有方法中完成

---

## 3. Prefix Cache 命中率的计算方式

### 3.1 统计数据收集

命中率的计算基于 `PrefixCacheStats` 中维护的原子计数器，数据在 `PrefixCache::match()` 方法中收集：

```cpp
PrefixMatchResult match(const std::vector<int32_t>& tokens) {
    // 每次调用 match 都记录一次请求
    if (config_.enable_stats) {
        stats_.total_requests++;                          // 总请求数 +1
        stats_.total_tokens_processed += tokens.size();   // 总处理 token 数
    }

    auto radix_result = radix_tree_->find_longest_prefix(tokens);

    if (radix_result.has_kv_cache() && 
        radix_result.matched_length >= config_.min_prefix_length) {
        // 缓存命中
        stats_.cache_hits++;                                   // 命中次数 +1
        stats_.tokens_reused += radix_result.matched_length;   // 复用的 token 数
        stats_.tokens_computed += result.prefill_count;        // 需要计算的 token 数
    } else {
        // 缓存未命中
        stats_.cache_misses++;                                 // 未命中次数 +1
        stats_.tokens_computed += tokens.size();               // 全部需要计算
    }
}
```

### 3.2 命中率计算公式

`PrefixCacheStats` 提供两个维度的命中率：

#### 请求级命中率（hit_rate）

$$\text{hit\_rate} = \frac{\text{cache\_hits}}{\text{total\_requests}}$$

```cpp
float hit_rate() const {
    int64_t total = total_requests.load();
    if (total == 0) return 0.0f;
    return static_cast<float>(cache_hits.load()) / total;
}
```

含义：所有请求中，有多少比例的请求命中了缓存（找到了 ≥ `min_prefix_length` 的匹配）。

#### Token 级复用率（reuse_rate）

$$\text{reuse\_rate} = \frac{\text{tokens\_reused}}{\text{total\_tokens\_processed}}$$

```cpp
float reuse_rate() const {
    int64_t total = total_tokens_processed.load();
    if (total == 0) return 0.0f;
    return static_cast<float>(tokens_reused.load()) / total;
}
```

含义：处理的所有 token 中，有多少比例是从缓存复用的，不需要重新计算 KV。

### 3.3 实例说明

```
总共 3 个请求：

请求1: [1,2,3,4,5,10,11,12] (8 tokens) → 首次，cache miss
  total_requests=1, cache_misses=1, tokens_computed=8

请求2: [1,2,3,4,5,10,11,12,20,21,22,30,31,32] (14 tokens) → 命中前缀8个token
  total_requests=2, cache_hits=1, tokens_reused=8, tokens_computed=6

请求3: [1,2,3,4,5,10,11,12,20,21,22,30,31,32] (14 tokens) → 命中14个token
  total_requests=3, cache_hits=2, tokens_reused=22, tokens_computed=6

计算结果：
  hit_rate = 2/3 = 66.7%
  reuse_rate = 22/36 = 61.1%
  total_tokens_processed = 8+14+14 = 36
```

### 3.4 命中条件

命中需要同时满足两个条件：

1. **`radix_result.has_kv_cache()`**：匹配到的节点必须关联有效的 KV Cache（`kv_cache_pos >= 0` 且 `matched_length > 0`），即必须匹配到一个终端节点
2. **`radix_result.matched_length >= config_.min_prefix_length`**：匹配长度必须达到最小要求（默认4 tokens），太短的前缀复用效益太低，不值得

### 3.5 统计信息输出

```cpp
std::string to_string() const {
    // 输出示例：
    // PrefixCache Stats:
    //   Total requests: 100
    //   Cache hits: 85 (85%)
    //   Cache misses: 15
    //   Total tokens: 50000
    //   Tokens reused: 35000 (70%)
    //   Tokens computed: 15000
    //   Evictions: 3
    //   Tokens evicted: 10000
}
```

在 `PrefixCacheManager::print_stats()` 中还会额外输出 RadixTree 的统计：
```
RadixTree Stats:
  Total nodes: 42
  Terminal nodes: 15
  Max depth: 8
  Cached tokens: 45000
```

---

## 4. 节点引用计数机制详解

### 4.1 引用计数的作用

引用计数（`ref_count`）是 PrefixCache 中一个关键的**安全机制**，它的核心作用是：

> **防止正在被推理请求使用的 KV Cache 被 LRU 淘汰策略意外删除。**

在 LLM 推理服务中，可能同时有多个请求在进行推理。如果请求 A 正在使用某段 KV Cache，而此时内存不足触发了 LRU 淘汰，如果没有引用计数保护，这段 KV Cache 可能被淘汰，导致请求 A 读取到无效数据，产生错误的推理结果甚至程序崩溃。

### 4.2 引用计数的生命周期

```
请求到来 → match() → 命中缓存 → ref_count++ → 使用KV Cache做推理
     ↓                                              ↓
     ↓         （此期间即使触发LRU淘汰，                ↓
     ↓          ref_count>0 的节点不会被淘汰）          ↓
     ↓                                              ↓
请求完成 → release() → ref_count-- → ref_count=0 → 可以被淘汰
```

### 4.3 源码级分析

#### 增加引用（在 `PrefixCache::match()` 中自动调用）

```cpp
PrefixMatchResult match(const std::vector<int32_t>& tokens) {
    auto radix_result = radix_tree_->find_longest_prefix(tokens);
    
    if (radix_result.has_kv_cache() && ...) {
        // 命中缓存，自动增加引用计数
        radix_tree_->add_reference(result.matched_prefix);
        // ...
    }
}
```

`add_reference()` 的实现：
```cpp
void add_reference(const std::vector<int32_t>& tokens) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto result = find_longest_prefix_impl(tokens);
    if (result.matched_node && result.matched_length == tokens.size()) {
        result.matched_node->kv_info.ref_count++;            // 引用计数+1
        result.matched_node->kv_info.last_access_time = get_current_time();  // 更新访问时间
    }
}
```

**注意**：`add_reference` 同时更新 `last_access_time`，这意味着被引用的节点即使后来 `ref_count` 降为0，也因为访问时间较新而不容易被 LRU 淘汰。

#### 释放引用（在请求完成后手动调用）

在 `generate_response()` 中：
```cpp
// 推理完成后，注册新前缀并释放旧的引用
if (config.use_prefix_cache && cache_manager) {
    cache_manager->register_prefix(final_tokens, final_tokens.size());
    
    if (!matched_prefix.empty()) {
        cache_manager->release(matched_prefix);  // 释放引用
    }
}
```

`release_reference()` 的实现：
```cpp
void release_reference(const std::vector<int32_t>& tokens) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto result = find_longest_prefix_impl(tokens);
    if (result.matched_node && result.matched_length == tokens.size()) {
        if (result.matched_node->kv_info.ref_count > 0) {
            result.matched_node->kv_info.ref_count--;  // 引用计数-1，不会减到负数
        }
    }
}
```

#### 引用计数在 LRU 淘汰中的保护作用

在 `evict_lru()` 中：
```cpp
for (const auto& [time, node] : evictable) {
    if (total_cached_tokens_ <= max_tokens) break;
    
    // 关键：跳过正在使用的节点！
    if (node->kv_info.ref_count > 0) {
        continue;  // ref_count > 0 → 有请求正在使用 → 不淘汰
    }
    
    // ref_count == 0 → 没有请求使用 → 可以安全淘汰
    node->is_terminal = false;
    node->kv_info = RadixNode::KVCacheInfo{};
    total_cached_tokens_ -= tokens_freed;
}
```

### 4.4 实例详解

以 `test_prefix_cache_full.cpp` 中的 `test_reference_counting()` 为例：

```
=== 场景：小缓存（max=30 tokens）+ 引用计数保护 ===

Step 1: 插入 seq1=[1..10]（10 tokens），总缓存=10
Step 2: match(seq1) → 命中，ref_count=1 ← 模拟"正在使用"
Step 3: 插入 seq2=[20..29]（10 tokens），总缓存=20
Step 4: 插入 seq3=[30..39]（10 tokens），总缓存=30
        触发淘汰（30 > 30*0.7=21）
        
淘汰检查：
  - seq1: ref_count=1 → 跳过！不淘汰
  - seq2: ref_count=0, last_access_time 较早 → 淘汰候选
  - seq3: ref_count=0, last_access_time 较晚 → 淘汰候选

结果：seq1 因为 ref_count=1 被保护，不会被淘汰
      seq2（或seq3）被淘汰以释放空间

Step 5: 验证 seq1 仍然可用 → match(seq1) → cache_hit=true ✓
Step 6: release(seq1) → ref_count=0
        现在 seq1 不再被保护，可以被后续淘汰
```

### 4.5 多请求并发场景

```
时间线:
────────────────────────────────────────────────
  请求A: match([1..100]) → ref_count=1
  请求B: match([1..100]) → ref_count=2 (同一前缀被两个请求使用)
  
  ... 触发LRU淘汰 → ref_count=2 > 0 → 不淘汰 [1..100] ...
  
  请求A完成: release() → ref_count=1 (还有请求B在用)
  
  ... 再次触发淘汰 → ref_count=1 > 0 → 仍然不淘汰 ...
  
  请求B完成: release() → ref_count=0 (没有请求在用了)
  
  ... 下次淘汰时 → ref_count=0 → 可以被淘汰了 ...
────────────────────────────────────────────────
```

### 4.6 设计意义总结

| 方面 | 说明 |
|---|---|
| **数据一致性** | 保证正在推理的请求不会因为缓存淘汰读取到无效数据 |
| **并发安全** | 多个请求可以同时引用同一个前缀，只有所有请求都释放后才允许淘汰 |
| **访问时间更新** | `add_reference` 同时更新 `last_access_time`，使被引用的节点在 LRU 排序中更靠后 |
| **防御性编程** | `release_reference` 检查 `ref_count > 0` 后才递减，避免计数器变为负数 |
| **灵活性** | 引用是可选的——如果不调用 `match()`，节点 `ref_count` 默认为 0，LRU 正常工作 |

---

## 5. Prefix Cache 在 Prefill 阶段的逐步使用详解（源码级）

本章以两轮对话为完整实例，结合 `generate_response()`、`PrefixCache::match()`、`RadixTree::find_longest_prefix_impl()`、`Qwen3Model::prefill()` 等核心源码，**逐行、逐步**解析 Prefix Cache 在 prefill 阶段到底做了什么。

### 5.1 场景设定

假设用户使用 Qwen3-8B 进行两轮对话：

```
Round 1 用户输入: "什么是大语言模型？"
Round 2 用户输入: "它有哪些应用场景？"
```

系统使用 Qwen3 ChatML 格式，system prompt 为 `"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."`

#### 对应的 token 序列

```
Round 1 完整 prompt tokenize 后（示例，假设42个token）:
tokens_r1 = [
  // <|im_start|>system\n
  151644, 8948, 198,
  // "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
  2610, 525, 1207, 16948, ..., 13,        
  // <|im_end|>\n
  151645, 198,
  // <|im_start|>user\n
  151644, 872, 198,
  // "什么是大语言模型？"
  104136, 101067, 99489, 101912, 11319,
  // <|im_end|>\n<|im_start|>assistant\n
  151645, 198, 151644, 77091, 198
]
// 共 42 个 token

Round 1 生成回复后，完整对话序列（假设模型生成了25个token回复）:
tokens_r1_full = tokens_r1 + [回复token * 25] + [151645, 198]
// 共 42 + 25 + 2 = 69 个 token

Round 2 完整 prompt tokenize 后（在 Round 1 完整历史基础上追加）:
tokens_r2 = tokens_r1_full + [
  // <|im_start|>user\n
  151644, 872, 198,
  // "它有哪些应用场景？"
  100244, 103110, 104429, 99622, 11319,
  // <|im_end|>\n<|im_start|>assistant\n
  151645, 198, 151644, 77091, 198
]
// 共 69 + 11 = 80 个 token
```

### 5.2 整体流程概览

```
                    generate_response() 入口
                           │
                ┌──────────┴──────────┐
                │ Step 1: Tokenize     │   将对话格式化并编码为 tokens
                └──────────┬──────────┘
                           │
                ┌──────────┴──────────┐
                │ Step 2: Prefix Cache │   在 RadixTree 中查找最长缓存前缀
                │         Match        │   确定可复用的 KV Cache 长度
                └──────────┬──────────┘
                           │
            ┌──────────────┼──────────────┐
          命中                           未命中
            │                              │
   start_pos = matched_tokens     start_pos = 0
   prefill_count = 新增部分       prefill_count = 全部
   KV Cache [0, start_pos) 保留   clear_kv_cache()
            │                              │
            └──────────────┬──────────────┘
                           │
                ┌──────────┴──────────┐
                │ Step 3: Embedding    │   计算所有 token 的 embedding
                └──────────┬──────────┘
                           │
                ┌──────────┴──────────┐
                │ Step 4: Prefill      │   只对 [start_pos, total) 做 prefill
                │   (增量 or 全量)     │   KV Cache 写入从 start_pos 开始
                └──────────┬──────────┘
                           │
                ┌──────────┴──────────┐
                │ Step 5: Decode Loop  │   逐 token 自回归生成
                └──────────┬──────────┘
                           │
                ┌──────────┴──────────┐
                │ Step 6: 注册前缀 +   │   将完整序列注册到 RadixTree
                │         释放引用     │   释放 match 时增加的引用计数
                └─────────────────────┘
```

### 5.3 Round 1：首次对话（Cache Miss 场景）

#### Step 1：Tokenize

```cpp
// inference_common.h L1048-1051
std::string full_prompt = conv.get_full_prompt(user_input);  
auto tokens = model.encode(full_prompt);                      
std::vector<int32_t> tokens_i32(tokens.begin(), tokens.end()); // tokens_r1, 42个token
int32_t total_len = 42;
```

#### Step 2：Prefix Cache 匹配

```cpp
// inference_common.h L1070-1086
int32_t start_pos = 0;
bool used_radix_cache = false;
std::vector<int32_t> matched_prefix;

if (config.use_prefix_cache && cache_manager) {
    auto match_result = cache_manager->match(tokens_i32);
    // ...
}
```

**进入 `PrefixCacheManager::match()` → `PrefixCache::match()`**：

```cpp
// prefix_cache.h L165-170
PrefixMatchResult match(const std::vector<int32_t>& tokens) {
    PrefixMatchResult result;
    result.prefill_count = 42;       // 默认需要全量 prefill
    result.prefill_start_pos = 0;   

    // 更新统计
    stats_.total_requests++;                // total_requests = 1
    stats_.total_tokens_processed += 42;    // total_tokens_processed = 42

    // 在 RadixTree 中查找最长前缀
    auto radix_result = radix_tree_->find_longest_prefix(tokens);
```

**进入 `RadixTree::find_longest_prefix()` → `find_longest_prefix_impl()`**：

```cpp
// radix_tree.h L347-411
RadixMatchResult find_longest_prefix_impl(tokens) {
    // 此时树是空的，只有空根节点 ROOT，ROOT 非终端
    // root_->children 为空
    
    current = root_;
    last_terminal = nullptr;    // 没有终端节点记录
    token_idx = 0;
    
    // root_ 非终端节点，不设 last_terminal
    
    // while 循环第一次迭代：
    //   first_token = tokens[0] = 151644
    //   current->children.find(151644) → end()  ← 找不到！
    //   break;
    
    // last_terminal == nullptr → 无有效匹配
    result.matched_length = 0;  // token_idx = 0
    result.kv_cache_pos = -1;   // 无 KV Cache
    return result;              // has_kv_cache() = false
}
```

**回到 `PrefixCache::match()`**：

```cpp
    // radix_result.has_kv_cache() = false → 进入 else 分支
    } else {
        result.cache_hit = false;       // 未命中
        stats_.cache_misses++;          // cache_misses = 1
        stats_.tokens_computed += 42;   // tokens_computed = 42
    }
    return result;
    // result = { matched_tokens=0, cache_hit=false, prefill_start_pos=0, prefill_count=42 }
```

**回到 `generate_response()`**：

```cpp
    // match_result.cache_hit = false → 进入 else
    } else {
        model.clear_kv_cache();    // 清空 KV Cache，确保干净状态
        conv.reset_kv_state();     // 重置对话的缓存 token 记录
        stats.kv_reuse_len = 0;
    }
    stats.prefill_tokens = 42 - 0 = 42;  // 需要全量 prefill 42 个 token
```

#### Step 3：Embedding 计算

```cpp
// inference_common.h L1112-1113
const auto& embedding_out = model.embedding(tokens);
// 对全部 42 个 token 计算 embedding → [42, dim] 的 tensor
// dim = 4096（Qwen3-8B）
```

#### Step 4：全量 Prefill

```cpp
// inference_common.h L1119-1143
if (stats.prefill_tokens > 0) {          // 42 > 0 ✓
    if (start_pos > 0) {                  // 0 > 0 ✗ → 走 else
        // ... 增量 prefill（本次不执行）
    } else {
        // 全量 prefill：从头开始
        base::Status status = model.prefill(embedding_out.input_embeddings, total_len=42, start_pos=0);
    }
}
```

**进入 `Qwen3Model::prefill(input, seq_len=42, start_pos=0)`**：

```cpp
// qwen3.cpp prefill 核心流程
// 分配批量计算缓冲区
hidden_buf0(fp16, 42, 4096);      // 双缓冲 hidden state
query_out(fp16, 42, 4096);        // Q 矩阵 [42, dim]
key_out(fp16, 42, 1024);          // K 矩阵 [42, kv_dim]
value_out(fp16, 42, 1024);        // V 矩阵 [42, kv_dim]

for (layer_idx = 0; layer_idx < 36; layer_idx++) {  // Qwen3-8B 有 36 层
    // 1. RMSNorm
    batched_attention_rms(layer_idx, input, rms_out, seq_len=42);
    
    // 2. Q/K/V 投影 + RoPE + KV Cache 写入
    batched_attention_qkv(layer_idx, rms_out, query_out, key_out, value_out, 
                          seq_len=42, start_pos=0);
    //   ┌─────────────────────────────────────────────────────────────────┐
    //   │ 内部关键操作 (qwen3.cpp batched_attention_qkv):                 │
    //   │                                                                 │
    //   │ // RoPE 位置编码：位置从 start_pos=0 开始，编码 [0,1,2,...,41]  │
    //   │ batched_rope->set_start_pos(0);                                 │
    //   │ batched_rope->set_seq_len(42);                                  │
    //   │ // 即 token[i] 的位置编码 = rope(0 + i)                         │
    //   │                                                                 │
    //   │ // KV Cache 写入：写入位置 [0, 42)                              │
    //   │ for (i = 0; i < 42; i++) {                                      │
    //   │     cache_offset = layer_offset + (0 + i) * kv_dim;             │
    //   │     cudaMemcpyAsync(kv_cache[cache_offset], key[i], ...);       │
    //   │ }                                                               │
    //   └─────────────────────────────────────────────────────────────────┘
    
    // 3. Multi-Head Attention
    batched_attention_mha(layer_idx, query_out, mha_out, seq_len=42, start_pos=0);
    //   ┌─────────────────────────────────────────────────────────────────┐
    //   │ FlashAttention prefill:                                         │
    //   │   Q: [42, dim]                                                  │
    //   │   K: kv_cache[0:42]    ← 刚写入的                              │
    //   │   V: kv_cache[0:42]    ← 刚写入的                              │
    //   │   attention_mask: causal mask [42, 42]                          │
    //   │   有效 KV 范围: [0, 0+42) = [0, 42)                            │
    //   └─────────────────────────────────────────────────────────────────┘
    
    // 4. 残差连接 + FFN + 残差
    // ...
}

// 取最后一个 token (token[41]) 的 hidden state → cls_logits → 输出logits
```

**此时 KV Cache 状态**：

```
KV Cache 内存布局（每层）:
  ┌─────────────────────────────────────────────────────────────┐
  │ pos:  0  1  2  ... 41 │ 42  43  ...  max_seq_len-1         │
  │       ═══════════════  │  空  空  ...  空                   │
  │       Round 1 的 42 个   │                                   │
  │       token 的 K/V      │                                   │
  └─────────────────────────────────────────────────────────────┘
```

#### Step 5：Decode 循环

```cpp
// inference_common.h L1160-1245
// 从 logits 中采样第一个 token，然后进入 decode loop
int32_t next = sample_argmax(logits_cpu);     // 采样第一个输出 token
int32_t pos = total_len;                       // pos = 42

while (decode_steps < config.max_tokens) {
    if (model.is_sentence_ending(next)) break;
    
    // 单 token embedding → decode → 写入 KV Cache[pos] → 采样下一个
    model.decode(input, pos, next);
    pos++;                                     // pos: 42 → 43 → ... → 66
    decode_steps++;
}
// 假设生成了25个token，pos最终 = 67 (42+25)
// 加上 <|im_end|>\n = 2 个 token，conv中完整序列 = 69 个 token
```

#### Step 6：注册前缀到 RadixTree

```cpp
// inference_common.h L1252-1259
if (config.use_prefix_cache && cache_manager) {
    std::vector<int32_t> final_tokens = conv.get_cached_tokens(); 
    // final_tokens = tokens_r1_full, 共 69 个 token
    
    cache_manager->register_prefix(final_tokens, 69);
    // → prefix_cache_->insert(final_tokens, kv_start_pos=0, kv_length=69)
```

**进入 `PrefixCache::insert()` → `RadixTree::insert_impl()`**：

```cpp
// radix_tree.h insert_impl
// 树是空的，token_idx=0, first_token=151644
// current(ROOT)->children.find(151644) → end() → 没找到
// 创建新叶子节点:
//   new_node->edge_tokens = [151644, 8948, 198, 2610, ..., 198]  // 全部69个token
//   new_node->prefix_length = 69
//   new_node->is_terminal = true
//   new_node->kv_info = { kv_start_pos=0, kv_length=69, last_access_time=T1, ref_count=0 }
//   ROOT->children[151644] = new_node

// 树状态:
//   ROOT → Edge[151644,8948,198,...+64] *KV(pos=0,len=69,ref=0)
```

```cpp
    // 释放引用（Round 1 没有匹配，matched_prefix 为空，不需要释放）
    if (!matched_prefix.empty()) {
        cache_manager->release(matched_prefix);  // 不执行
    }
}
```

**Round 1 结束后的完整状态**：

```
RadixTree:
  ROOT ─── Edge[151644,8948,198,...共69个token] *KV(pos=0,len=69,ref=0)

PrefixCacheStats:
  total_requests = 1
  cache_hits = 0
  cache_misses = 1
  total_tokens_processed = 42
  tokens_reused = 0
  tokens_computed = 42
  hit_rate = 0/1 = 0%
  reuse_rate = 0/42 = 0%

KV Cache（GPU内存中）:
  位置 [0, 68] 存有完整的 Round 1 对话 KV 数据
```

---

### 5.4 Round 2：第二轮对话（Cache Hit 场景）

#### Step 1：Tokenize

```cpp
// inference_common.h L1048-1051
std::string full_prompt = conv.get_full_prompt("它有哪些应用场景？");
// full_prompt 包含 Round 1 的完整历史 + Round 2 的新问题

auto tokens = model.encode(full_prompt);
std::vector<int32_t> tokens_i32(tokens.begin(), tokens.end());  
// tokens_r2 = tokens_r1_full(前69个) + 新问题tokens(11个) = 共80个token
int32_t total_len = 80;
```

#### Step 2：Prefix Cache 匹配（核心！）

```cpp
// inference_common.h L1070-1086
int32_t start_pos = 0;
bool used_radix_cache = false;
std::vector<int32_t> matched_prefix;

auto match_result = cache_manager->match(tokens_i32);
```

**进入 `PrefixCache::match(tokens_r2)`** (prefix_cache.h L165)：

```cpp
PrefixMatchResult match(tokens_r2) {
    PrefixMatchResult result;
    result.prefill_count = 80;          // 默认全量
    result.prefill_start_pos = 0;

    // 统计更新
    stats_.total_requests++;             // total_requests = 2
    stats_.total_tokens_processed += 80; // total_tokens_processed = 42 + 80 = 122

    // 查找最长前缀
    auto radix_result = radix_tree_->find_longest_prefix(tokens_r2);
```

**进入 `find_longest_prefix_impl(tokens_r2)`** (radix_tree.h L355)：

```cpp
RadixMatchResult find_longest_prefix_impl(tokens_r2) {
    current = root_;
    last_terminal = nullptr;
    last_terminal_length = 0;
    token_idx = 0;
    
    // root_ 非终端，不设 last_terminal
    
    // ========== while 循环 ==========
    
    // --- 迭代 1 ---
    // first_token = tokens_r2[0] = 151644
    // current(ROOT)->children.find(151644) → 找到！
    // child = Edge[151644,8948,198,...共69个token]
    // edge = child->edge_tokens (69个token)
    
    // 逐 token 比较:
    //   remaining = 80 - 0 = 80
    //   max_common = min(69, 80) = 69
    //
    //   edge[0]=151644 vs tokens_r2[0]=151644  ✓ common_len=1
    //   edge[1]=8948   vs tokens_r2[1]=8948    ✓ common_len=2
    //   edge[2]=198    vs tokens_r2[2]=198     ✓ common_len=3
    //   ...
    //   edge[68]=198   vs tokens_r2[68]=198    ✓ common_len=69
    //
    //   ★ tokens_r2 的前69个 token 与 Round 1 注册的完全一致！
    
    token_idx += 69;                        // token_idx = 69
    
    // common_len(=69) == edge.size()(=69) → 完全匹配边！
    current = child;                        // 移到该子节点
    
    // child->is_terminal = true → 更新 last_terminal！
    last_terminal = child;
    last_terminal_length = 69;              // 匹配了69个token
    
    // --- 迭代 2 ---
    // first_token = tokens_r2[69] = 151644（Round 2 的 <|im_start|>）
    // current(child)->children 为空（Round 1 的节点是叶子节点）
    // current->children.find(151644) → end()
    // break!
    
    // ========== 构造返回结果 ==========
    // last_terminal != nullptr → 有效匹配
    result.matched_node = last_terminal;          // Round 1 的终端节点
    result.matched_length = 69;                    // 匹配了69个token
    result.kv_cache_pos = last_terminal->kv_info.kv_start_pos;  // = 0
    
    return result;
    // result = { matched_length=69, kv_cache_pos=0, has_kv_cache()=true }
}
```

**回到 `PrefixCache::match()`**：

```cpp
    // radix_result.has_kv_cache() = true ✓
    // radix_result.matched_length = 69 >= config_.min_prefix_length = 4 ✓
    // → 进入命中分支！
    
    result.matched_tokens = 69;
    result.cache_hit = true;
    result.prefill_start_pos = 69;              // 从第69个位置开始 prefill
    result.prefill_count = 80 - 69 = 11;        // 只需 prefill 11 个新 token！
    result.reuse_ratio = 69.0 / 80 = 0.8625;    // 复用了 86.25%
    result.matched_prefix = tokens_r2[0:69];     // 前69个token作为引用 key
    
    // 增加引用计数（防止在 prefill 过程中被淘汰）
    radix_tree_->add_reference(result.matched_prefix);
    //   → find_longest_prefix_impl(前69个token) 找到终端节点
    //   → 该节点 ref_count: 0 → 1
    //   → 更新 last_access_time = 当前时间
    
    // 更新统计
    stats_.cache_hits++;                         // cache_hits = 1
    stats_.tokens_reused += 69;                  // tokens_reused = 69
    stats_.tokens_computed += 11;                // tokens_computed = 42 + 11 = 53
    
    return result;
```

**回到 `generate_response()`**：

```cpp
    // match_result.cache_hit = true && match_result.matched_tokens = 69 > 0
    start_pos = 69;                              // ★ 关键：从第69个位置开始
    stats.kv_reuse_len = 69;
    used_radix_cache = true;
    matched_prefix = tokens_r2[0:69];            // 保存，用于后续释放引用
    
    // 不清空 KV Cache！前69个位置的数据保留
    
    LOG(INFO) << "[RadixTree] Cache hit! Reusing 69/80 tokens (86%)";
    
    stats.prefill_tokens = 80 - 69 = 11;         // 只需 prefill 11 个 token
```

#### Step 3：Embedding 计算

```cpp
// inference_common.h L1112-1113
const auto& embedding_out = model.embedding(tokens);
// 对全部80个 token 计算 embedding → [80, 4096] 的 tensor
// 注意：embedding 对所有 token 都做了，但 prefill 时只用后 11 个
```

#### Step 4：增量 Prefill（核心！）

```cpp
// inference_common.h L1119-1143
if (stats.prefill_tokens > 0) {          // 11 > 0 ✓
    if (start_pos > 0) {                  // 69 > 0 ✓ → 增量 prefill！
        
        // 分配仅包含新增 token embedding 的 tensor
        auto alloc = base::CUDADeviceAllocatorFactory::get_instance();
        base::DataType dtype = fp16;
        size_t dim = 4096;
        size_t elem_size = sizeof(uint16_t);  // FP16
        
        // 创建 [11, 4096] 的新 tensor
        tensor::Tensor new_embeddings(fp16, 11, 4096, true, alloc);
        
        // ★ 关键：从完整 embedding 的第 69 个位置开始切片
        const void* src_ptr = embedding_out.input_embeddings.ptr<uint16_t>(69 * 4096);
        //                                                                  ^^^
        //                                           start_pos * dim = 69 * 4096 = 第69个token的embedding起始地址
        
        // GPU → GPU 拷贝：只拷贝后 11 个 token 的 embedding
        cudaMemcpyAsync(
            new_embeddings.get_buffer()->ptr(),     // 目标：new_embeddings 的起始
            src_ptr,                                 // 源：embedding[69*4096]
            11 * 4096 * sizeof(uint16_t),            // 大小：11 * 4096 * 2 bytes = 90112 bytes
            cudaMemcpyDeviceToDevice,
            stream
        );
        
        // ★★★ 核心调用：增量 prefill ★★★
        base::Status status = model.prefill(new_embeddings, seq_len=11, start_pos=69);
    }
}
```

**进入 `Qwen3Model::prefill(input=[11,4096], seq_len=11, start_pos=69)`**：

```cpp
// qwen3.cpp prefill
// 分配批量计算缓冲区（大小为 11 而非 80！）
hidden_buf0(fp16, 11, 4096);      // 只需处理 11 个 token
query_out(fp16, 11, 4096);
key_out(fp16, 11, 1024);
value_out(fp16, 11, 1024);

for (layer_idx = 0; layer_idx < 36; layer_idx++) {
    // 1. RMSNorm（只处理 11 个 token）
    batched_attention_rms(layer_idx, input, rms_out, seq_len=11);
    
    // 2. Q/K/V 投影 + RoPE + KV Cache 写入
    batched_attention_qkv(layer_idx, rms_out, query_out, key_out, value_out,
                          seq_len=11, start_pos=69);
```

**`batched_attention_qkv` 中 `start_pos=69` 的三重作用**：

```
作用 1：RoPE 位置编码偏移
─────────────────────────

batched_rope->set_start_pos(69);    // 位置从69开始
batched_rope->set_seq_len(11);      // 处理 11 个 token

// 效果：这 11 个 token 的位置编码分别为：
//   token[0] → rope(69)    而非 rope(0)
//   token[1] → rope(70)    而非 rope(1)
//   ...
//   token[10] → rope(79)   而非 rope(10)
//
// 这保证了位置编码的连续性！
// KV Cache 中已有的 [0,68] 位置用的是 rope(0)~rope(68)
// 新计算的 [69,79] 位置用的是 rope(69)~rope(79)
// ✓ 与全量 prefill 80 个 token 时的位置编码完全一致

作用 2：KV Cache 写入位置偏移
─────────────────────────────

for (i = 0; i < 11; i++) {
    cache_offset = layer_offset + (69 + i) * kv_dim;
    //                             ^^
    //                       start_pos = 69
    cudaMemcpyAsync(key_cache[cache_offset], key_out[i], kv_dim * elem_size, D2D, stream);
    cudaMemcpyAsync(val_cache[cache_offset], val_out[i], kv_dim * elem_size, D2D, stream);
}

// Key/Value 写入 KV Cache 的位置 [69, 79]
// 而 [0, 68] 保持不变——那是 Round 1 prefill 时写入的！

作用 3：Attention 计算范围
──────────────────────────

// Flash Attention:
prefill_layer->set_start_pos(69);
prefill_layer->set_cur_seq_len(11);

// 效果：
//   Q: [11, dim]  — 只有 11 个 query（新 token）
//   K: kv_cache[0:80]  — 包含全部 80 个位置的 key！
//       ├── [0:69]   ← Round 1 的缓存（prefix cache 保留的）
//       └── [69:80]  ← 本次新写入的
//   V: kv_cache[0:80]  — 同理
//
// 每个 query 可以 attend 到 [0, 69+i] 范围内的所有 key/value
// 这与全量 prefill 80 个 token 时的 causal attention 结果完全等价！
```

```cpp
    // 3. Multi-Head Attention（关键）
    batched_attention_mha(layer_idx, query_out, mha_out, seq_len=11, start_pos=69);
    //   Q: [11, 4096] — 11 个新 token 的 query
    //   K/V Cache: 有效范围 [0, 80) — 69 个旧 + 11 个新
    //   输出: mha_out [11, 4096]
    
    // 4. 残差 + FFN + 残差（只处理 11 个 token）
    // ...
}

// 取最后一个 token (token[10]，即全局第79个) 的 hidden → logits
```

**此时 KV Cache 状态 — 无缝拼接**：

```
KV Cache 内存布局（每层）:
  ┌──────────────────────────────────────────────────────────┐
  │ pos:  0  1  2  ... 68 │ 69  70  ... 79 │ 80  ...  空    │
  │       ═══════════════  │  ═════════════  │                │
  │       Round 1 旧数据    │  Round 2 新数据 │                │
  │       (prefix cache   │  (增量 prefill  │                │
  │        保留，未重算)    │   新计算写入)   │                │
  └──────────────────────────────────────────────────────────┘
                           ↑
                     start_pos = 69
                     新数据从这里开始写入

★ 效果等价于对全部 80 个 token 做全量 prefill，但计算量只有 11/80 = 13.75%！
```

#### Step 5：Decode 循环

```cpp
// 与 Round 1 相同，从 pos=80 开始逐 token 生成
int32_t pos = total_len;  // pos = 80
while (decode_steps < max_tokens) {
    model.decode(input, pos, next);  // KV Cache 写入 [80], [81], ...
    pos++;
    decode_steps++;
}
```

#### Step 6：注册前缀 + 释放引用

```cpp
// inference_common.h L1252-1259
if (config.use_prefix_cache && cache_manager) {
    // 获取包含生成回复的完整 token 序列
    std::vector<int32_t> final_tokens = conv.get_cached_tokens();
    // final_tokens = [80个prompt + 生成的回复token + im_end] ≈ 108 个 token
    
    cache_manager->register_prefix(final_tokens, 108);
    // → prefix_cache_->insert(final_tokens, 0, 108)
```

**进入 `RadixTree::insert_impl(final_tokens, 0, 108)`**：

```cpp
// 当前树状态: ROOT → Edge[共69个token] *KV(pos=0,len=69,ref=1)
// 要插入: [共108个token]，其前69个与现有边完全一致

// Step 1: first_token = 151644
// current(ROOT)->children.find(151644) → 找到 Edge[69个token]

// Step 2: 逐 token 比较
//   edge = [69个token], remaining = 108, max_common = min(69, 108) = 69
//   比较 edge[0..68] vs tokens[0..68] → 全部相等, common_len = 69

// Step 3: common_len(69) == edge.size()(69) → 完全匹配！
//   token_idx += 69;  // token_idx = 69
//   current = child;  // 移到该节点

// Step 4: 继续循环
//   first_token = tokens[69] = 151644
//   current->children.find(151644) → end()  ← 找不到

// Step 5: 创建新叶子节点
//   new_node->edge_tokens = tokens[69:108]  // 后39个token
//   new_node->prefix_length = 108
//   new_node->is_terminal = true
//   new_node->kv_info = { pos=0, len=108, last_access=T2, ref=0 }
//   current->children[151644] = new_node

// 新的树状态:
//   ROOT → Edge[151644,8948,...共69个token] *KV(pos=0,len=69,ref=1)
//            └── Edge[151644,872,...共39个token] *KV(pos=0,len=108,ref=0)
```

```cpp
    // 释放 Round 2 匹配时增加的引用
    if (!matched_prefix.empty()) {     // matched_prefix = tokens_r2[0:69]
        cache_manager->release(matched_prefix);
        // → radix_tree_->release_reference(前69个token)
        // → 找到 Edge[69个token] 节点
        // → ref_count: 1 → 0
    }
}
```

**Round 2 结束后的完整状态**：

```
RadixTree:
  ROOT
    └── Edge[151644,8948,198,...共69个token] *KV(pos=0,len=69,ref=0)  ← Round 1 的完整对话
          └── Edge[151644,872,198,...共39个token] *KV(pos=0,len=108,ref=0) ← Round 2 的完整对话

PrefixCacheStats:
  total_requests = 2
  cache_hits = 1
  cache_misses = 1
  total_tokens_processed = 42 + 80 = 122
  tokens_reused = 69
  tokens_computed = 42 + 11 = 53
  hit_rate = 1/2 = 50%
  reuse_rate = 69/122 = 56.6%

实际计算节省:
  无 PrefixCache: 需计算 42 + 80 = 122 个 token 的 prefill
  有 PrefixCache: 只计算  42 + 11 = 53  个 token 的 prefill
  节省率 = (122-53)/122 = 56.6%
```

### 5.5 Round 3 假设：又一个新对话（前缀共享场景）

假设此时来了一个新用户的请求，使用相同的 system prompt 但问了不同的问题 `"介绍一下 CUDA 编程"`。Tokenize 后：

```
tokens_r3 = [
  151644, 8948, 198,           // <|im_start|>system\n（与 Round 1/2 相同）
  2610, 525, 1207, 16948, ..., // system prompt（与 Round 1/2 相同）
  151645, 198,                  // <|im_end|>\n（与 Round 1/2 相同）
  151644, 872, 198,            // <|im_start|>user\n（与 Round 1/2 相同）
  ← 从这里开始不同 →
  12345, 67890, ...            // "介绍一下 CUDA 编程"
  151645, 198, 151644, 77091, 198
]
// 假设共35个token，其中前25个与 Round 1 的 tokens_r1 的前25个相同
```

**`find_longest_prefix_impl(tokens_r3)` 的执行过程**：

```cpp
// 当前树:
//   ROOT → Edge[151644,8948,198,...共69个token] *KV(pos=0,len=69)
//            └── Edge[151644,872,...共39个token] *KV(pos=0,len=108)

// 比较 edge[69个token] vs tokens_r3[35个token]:
//   max_common = min(69, 35) = 35
//   edge[0]=151644 vs tokens_r3[0]=151644  ✓
//   edge[1]=8948   vs tokens_r3[1]=8948    ✓
//   ...
//   edge[24]=...   vs tokens_r3[24]=...    ✓  common_len=25（前25个相同）
//   edge[25]=...   vs tokens_r3[25]=12345  ✗  不匹配！
//
//   common_len = 25 < edge.size() = 69 → 部分匹配边！

// result.edge_offset = 25
// 关键: Edge[69个token] 是终端节点，但实际上匹配只到了边的第25个位置
//       此时 last_terminal 尚未被设置（因为边没有完全匹配）
//       所以 last_terminal仍然是nullptr（或root如果root是terminal的话）

// 结果: matched_length = 25, 但 last_terminal = nullptr
//       kv_cache_pos = -1, has_kv_cache() = false!

// ★ 重要: 即使tokens前25个与缓存匹配，但因为25个token在同一条边的中间，
//   没有到达任何终端节点，所以没有有效的 KV Cache 可以复用。
//   这是 RadixTree 的一个特性——KV Cache 只关联到终端节点。
```

**如果想要复用这25个token的前缀**，需要在 `insert` 时将69个token的序列用25个token作为前缀进行分裂。实际上，如果系统在 Round 1 生成之前就先注册了25个token的system+user前缀，那么树的结构就会更细粒度，匹配率更高。但当前实现是在整个对话生成完成后才整体注册，所以共享粒度取决于注册时序列的长度。

### 5.6 关键源码映射总结

下表梳理 prefill 阶段 prefix cache 涉及的所有关键源码位置和调用关系：

| 步骤 | 源码位置 | 函数 | 作用 |
|---|---|---|---|
| 1 | inference_common.h L1070-1086 | `generate_response()` | 调用 `cache_manager->match()` 获取匹配结果 |
| 2 | prefix_cache.h L165-210 | `PrefixCache::match()` | 查找前缀、更新统计、增加引用计数 |
| 3 | radix_tree.h L355-411 | `find_longest_prefix_impl()` | 在树中遍历，找最后一个终端节点 |
| 4 | radix_tree.h L167-175 | `add_reference()` | 对匹配节点 `ref_count++`，更新 `last_access_time` |
| 5 | inference_common.h L1119-1143 | `generate_response()` 增量prefill | 切片 embedding，调用 `model.prefill(new_emb, prefill_count, start_pos)` |
| 6 | qwen3.cpp L1773-1862 | `Qwen3Model::prefill()` | 只处理 `seq_len` 个 token 的 Transformer 前向 |
| 7 | qwen3.cpp L1456-1574 | `batched_attention_qkv()` | RoPE 从 `start_pos` 偏移，KV Cache 写入 `[start_pos, start_pos+seq_len)` |
| 8 | qwen3.cpp L1575-1640 | `batched_attention_mha()` | Attention 范围 `[0, start_pos+seq_len)`，新 query attend 到所有历史 |
| 9 | inference_common.h L1252-1259 | `generate_response()` 注册 | 调用 `cache_manager->register_prefix()` 注册完整序列 |
| 10 | radix_tree.h L277-338 | `insert_impl()` | 插入 RadixTree，可能触发节点分裂 |
| 11 | inference_common.h L1256-1258 | `generate_response()` 释放 | 调用 `cache_manager->release()` 减少引用计数 |

### 5.7 为什么增量 Prefill 的结果等价于全量 Prefill？

这是理解 Prefix Cache 正确性的关键。下面以 Round 2 的实例（80个 token 总序列，前 69 个命中缓存，增量 prefill 后 11 个）为贯穿案例，从 Transformer 的每一个计算步骤出发，结合源码逐步证明等价性。

#### 5.7.1 前置概念：两种 Prefill 路径对比

```
全量 Prefill (start_pos=0):
  input: embedding[0:80]，共 80 个 token 的 embedding
  输出: 将 80 个 token 的 KV 全部写入 cache[0:80]
  计算: 每层做 80 次 matmul、80 次 RoPE、80 query 的 attention

增量 Prefill (start_pos=69):
  input: embedding[69:80]，只有 11 个 token 的 embedding
  输出: 将 11 个 token 的 KV 写入 cache[69:80]（cache[0:69] 已有数据）
  计算: 每层做 11 次 matmul、11 次 RoPE、11 query 的 attention
```

**目标**：证明两种路径下，最终 logits 输出完全相同。

#### 5.7.2 步骤 0：Embedding —— 逐 token 独立查表

**源码** (qwen3.cpp `embedding()`):
```cpp
// 对每个 token ID 独立查找 embedding 权重表
for (int32_t i = 0; i < tokens.size(); ++i) {
    input_tokens.index<int32_t>(i) = tokens.at(i);
}
STATUS_CHECK(qwen_layers_->embedding_layer_->forward(input_tokens, input_token_num, input_embeddings));
```

Embedding 层本质上是一个**查表操作**：每个 token 的 embedding 仅取决于该 token 的 ID，与序列中其他 token 完全无关。

**数学表达**：

$$\text{Embed}(x_i) = W_{\text{embed}}[x_i, :]$$

其中 $W_{\text{embed}}$ 是固定的权重矩阵，$x_i$ 是第 $i$ 个 token 的 ID。

**源码** (inference_common.h 增量 prefill 切片):
```cpp
// 全量 embedding: 对全部 80 个 token 做 embedding
const auto& embedding_out = model.embedding(tokens);
// embedding_out.input_embeddings: [80, 4096]

// 增量切片: 只取后 11 个 token 的 embedding
const void* src_ptr = embedding_out.input_embeddings.template ptr<uint16_t>(start_pos * dim);
//                                                                          ^^^ = 69 * 4096
cudaMemcpyAsync(new_embeddings.get_buffer()->ptr(), src_ptr,
                stats.prefill_tokens * dim * elem_size, cudaMemcpyDeviceToDevice, stream);
// new_embeddings: [11, 4096]
```

**等价性证明**：

```
全量 embedding[80, 4096] 的第 k 行:  Embed(token[k]) = W_embed[token[k], :]
增量 new_embeddings[11, 4096] 的第 j 行: 是 embedding[69+j] 的拷贝

因此: new_embeddings[j] = embedding[69+j] = Embed(token[69+j])

结论: 增量路径中每个 token 的 embedding 向量与全量路径中对应位置的
      embedding 向量在数值上完全相同（同一段 GPU 显存的拷贝）。
```

#### 5.7.3 步骤 1：RMSNorm —— 逐 token 独立归一化

**源码** (qwen3.cpp `batched_attention_rms()`):
```cpp
void Qwen3Model::batched_attention_rms(int32_t layer_idx, const tensor::Tensor& input,
                                       const tensor::Tensor& output, int32_t seq_len) const {
    std::shared_ptr<op::Layer> rmsnorm_layer = qwen_layers_->rmsnorm_layers_.at(layer_idx);
    STATUS_CHECK(rmsnorm_layer->forward(input, output));
}
```

RMSNorm 的计算公式（批量模式下对每个 token 独立计算）：

$$\text{RMSNorm}(x_i) = \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d}x_{i,j}^2 + \epsilon}} \odot \gamma$$

关键点：分母中的均方根只在 $d$ 维（隐藏维度）上求和，**不涉及序列中其他 token**。

**等价性分析**：

```
全量路径:
  input  = [x_0, x_1, ..., x_68, x_69, ..., x_79]  shape=[80, 4096]
  output = [RMSNorm(x_0), ..., RMSNorm(x_68), RMSNorm(x_69), ..., RMSNorm(x_79)]

增量路径:
  input  = [x_69, x_70, ..., x_79]  shape=[11, 4096]
  output = [RMSNorm(x_69), RMSNorm(x_70), ..., RMSNorm(x_79)]

因为 RMSNorm 对每个 token 独立计算，所以:
  增量 output[j] = RMSNorm(x_{69+j}) = 全量 output[69+j]  ✓ 完全一致
```

#### 5.7.4 步骤 2：Q/K/V 线性投影 —— 逐 token 独立矩阵乘法

**源码** (qwen3.cpp `batched_attention_qkv()`):
```cpp
// AWQ 量化路径
STATUS_CHECK(query_awq->forward(rms_out, query_out));   // [seq_len, dim] × W_q → [seq_len, dim]
STATUS_CHECK(key_awq->forward(rms_out, key_out));       // [seq_len, dim] × W_k → [seq_len, kv_dim]
STATUS_CHECK(value_awq->forward(rms_out, value_out));   // [seq_len, dim] × W_v → [seq_len, kv_dim]

// 或标准 matmul 路径
STATUS_CHECK(qwen_layers_->batched_matmul_helper_layer_->forward(
    rms_out, query_matmul->get_weight(0), query_out, seq_len, 1.f));
```

线性投影的数学公式：

$$Q_i = \text{RMSNorm}(x_i) \cdot W_Q, \quad K_i = \text{RMSNorm}(x_i) \cdot W_K, \quad V_i = \text{RMSNorm}(x_i) \cdot W_V$$

**每个 token 的 Q/K/V 仅取决于该 token 自身的 hidden state，与其他 token 无关。**

```
全量: Q[80, 4096], K[80, 1024], V[80, 1024]
增量: Q[11, 4096], K[11, 1024], V[11, 1024]

增量 Q[j] = rms_out[j] · W_Q = rms_out_全量[69+j] · W_Q = 全量 Q[69+j]  ✓
增量 K[j] = rms_out[j] · W_K = 全量 K[69+j]  ✓
增量 V[j] = rms_out[j] · W_V = 全量 V[69+j]  ✓
```

#### 5.7.5 步骤 2.5：Qwen3 Per-Head Q/K RMSNorm —— 逐 token 独立

**源码** (qwen3.cpp `batched_attention_qkv()` 中间部分):
```cpp
// Qwen3 特有: 对 Q 和 K 做 per-head RMSNorm
auto q_norm = qwen_layers_->rmsnorm_layers_.at(layer_idx + 2 * config_->layer_num_ + 1);
auto k_norm = qwen_layers_->rmsnorm_layers_.at(layer_idx + 3 * config_->layer_num_ + 1);

// reshape [seq_len, dim] → [seq_len * head_num, head_size]
tensor::Tensor q_reshaped(activation_dtype, seq_len * config_->head_num_, config_->head_size_, ...);
tensor::Tensor k_reshaped(activation_dtype, seq_len * config_->kv_head_num_, config_->head_size_, ...);

STATUS_CHECK(q_norm->forward(q_reshaped, q_reshaped));   // in-place per-head normalize
STATUS_CHECK(k_norm->forward(k_reshaped, k_reshaped));
```

Per-head RMSNorm 是将 `[seq_len, dim]` reshape 为 `[seq_len * head_num, head_size]`，然后对每个 `head_size` 维做 RMSNorm。这仍然是**逐行独立**的操作——每个 token 的每个 head 的 norm 只取决于该 token 该 head 的值。

```
增量 q_norm[j * head_num + h] = Per-Head-RMSNorm(Q[j, h])
                               = Per-Head-RMSNorm(Q_全量[69+j, h])
                               = 全量 q_norm[(69+j) * head_num + h]  ✓
```

#### 5.7.6 步骤 3：RoPE 位置编码 —— 绝对位置，通过 start_pos 偏移

这是等价性证明中最关键的一步。RoPE 的正确偏移是增量 prefill 成立的核心条件。

**源码** (qwen3.cpp `batched_attention_qkv()` RoPE 部分):
```cpp
auto batched_rope = qwen_layers_->batched_rope_layer_;
batched_rope->set_seq_len(seq_len);           // 全量: 80, 增量: 11
batched_rope->set_start_pos(start_pos);       // 全量: 0,  增量: 69
batched_rope->set_input(0, query_out);
batched_rope->set_input(1, key_out);
batched_rope->set_input(2, get_buffer(ModelBufferType::kSinCache));
batched_rope->set_input(3, get_buffer(ModelBufferType::kCosCache));
STATUS_CHECK(batched_rope->forward());
```

**CUDA Kernel 源码** (rope_kernel.cu `batched_rope_kernel_cu_fp16_impl`):
```cuda
__global__ void batched_rope_kernel_cu_fp16_impl(
    int start_pos, int seq_len, int dim, int kv_dim, int head_size,
    half* input_q, half* input_k,
    const float* __restrict__ sin_cache, const float* __restrict__ cos_cache)
{
    int seq_idx = blockIdx.x;   // 每个 block 处理一个序列位置
    if (seq_idx >= seq_len) return;
    
    int pos = start_pos + seq_idx;  // ★★★ 核心: 绝对位置 = start_pos + 序列内偏移
    
    int idx = threadIdx.x + blockDim.x * blockIdx.y;
    // ... 计算 head_idx, head_dim ...
    
    // 从 sin/cos 缓存表中读取该绝对位置的旋转角
    float fci = sin_cache[pos * head_size + head_dim * 2];
    float fcr = cos_cache[pos * head_size + head_dim * 2];
    
    // 计算序列内偏移（确定读写 Q/K 的位置）
    int q_offset = seq_idx * dim;
    int k_offset = seq_idx * kv_dim;
    
    // 对 Q 和 K 应用旋转
    for (int v = 0; v < rotn; v++) {
        half* vec = (v == 0) ? (input_q + q_offset) : (input_k + k_offset);
        float v0 = __half2float(vec[actual_v0_idx]);
        float v1 = __half2float(vec[actual_v1_idx]);
        // RoPE 旋转公式
        vec[actual_v0_idx] = __float2half(fcr * v0 - fci * v1);
        vec[actual_v1_idx] = __float2half(fcr * v1 + fci * v0);
    }
}
```

**RoPE 数学公式**（以 2D 旋转为例）：

$$\text{RoPE}(x, \text{pos}) = \begin{pmatrix} x_0 \cos\theta_{\text{pos}} - x_1 \sin\theta_{\text{pos}} \\ x_0 \sin\theta_{\text{pos}} + x_1 \cos\theta_{\text{pos}} \end{pmatrix}$$

其中 $\theta_{\text{pos}}$ 仅由绝对位置 `pos` 决定，预计算在 `sin_cache` 和 `cos_cache` 中。

**等价性逐步验证**：

```
全量路径 (start_pos=0, seq_len=80):
  seq_idx=0:   pos = 0 + 0  = 0   → sin_cache[0],  cos_cache[0]   → RoPE(Q[0], 0)
  seq_idx=1:   pos = 0 + 1  = 1   → sin_cache[1],  cos_cache[1]   → RoPE(Q[1], 1)
  ...
  seq_idx=68:  pos = 0 + 68 = 68  → sin_cache[68], cos_cache[68]  → RoPE(Q[68], 68)
  seq_idx=69:  pos = 0 + 69 = 69  → sin_cache[69], cos_cache[69]  → RoPE(Q[69], 69)
  ...
  seq_idx=79:  pos = 0 + 79 = 79  → sin_cache[79], cos_cache[79]  → RoPE(Q[79], 79)

增量路径 (start_pos=69, seq_len=11):
  seq_idx=0:   pos = 69 + 0  = 69  → sin_cache[69], cos_cache[69]  → RoPE(Q'[0], 69)
  seq_idx=1:   pos = 69 + 1  = 70  → sin_cache[70], cos_cache[70]  → RoPE(Q'[1], 70)
  ...
  seq_idx=10:  pos = 69 + 10 = 79  → sin_cache[79], cos_cache[79]  → RoPE(Q'[10], 79)

关键验证:
  因为 Q'[j] = Q[69+j]（步骤 2 已证明），
  所以 RoPE(Q'[j], 69+j) = RoPE(Q[69+j], 69+j) = 全量路径中 seq_idx=69+j 的结果  ✓
  同理 K 也完全一致  ✓

★ 如果 start_pos 设错（比如设为0），那么:
  增量 RoPE(Q'[0], 0) ≠ 全量 RoPE(Q[69], 69)  ← 位置编码错误！
  这会导致 attention 分布完全不同，产生错误结果。
  所以 start_pos 的正确传递是增量 prefill 的核心保障。
```

#### 5.7.7 步骤 4：KV Cache 写入 —— 通过 start_pos 偏移写入正确位置

**源码** (qwen3.cpp `batched_attention_qkv()` KV Cache 写入):
```cpp
tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);

// layer_offset: 每层有独立的 KV Cache 区域
int32_t layer_offset = layer_idx * config_->seq_len_ * config_->kv_dim_;

// 写入 Value Cache
for (int i = 0; i < seq_len; ++i) {
    // ★ cache 写入位置 = layer_offset + (start_pos + i) * kv_dim
    int32_t cache_offset = layer_offset + (start_pos + i) * config_->kv_dim_;
    void* dst = val_cache.ptr() + cache_offset * elem_size;
    const void* src = value_out.ptr() + i * config_->kv_dim_ * elem_size;
    cudaMemcpyAsync(dst, src, config_->kv_dim_ * elem_size, cudaMemcpyDeviceToDevice, stream);
}

// 写入 Key Cache（RoPE 后的）
for (int i = 0; i < seq_len; ++i) {
    int32_t cache_offset = layer_offset + (start_pos + i) * config_->kv_dim_;
    void* dst = key_cache.ptr() + cache_offset * elem_size;
    const void* src = key_out.ptr() + i * config_->kv_dim_ * elem_size;
    cudaMemcpyAsync(dst, src, config_->kv_dim_ * elem_size, cudaMemcpyDeviceToDevice, stream);
}
```

**KV Cache 写入位置计算**：

```
cache_offset = layer_offset + (start_pos + i) * kv_dim

全量路径 (start_pos=0, seq_len=80):
  i=0:  cache_offset = layer_offset + 0 * kv_dim     → 写入位置 0
  i=1:  cache_offset = layer_offset + 1 * kv_dim     → 写入位置 1
  ...
  i=68: cache_offset = layer_offset + 68 * kv_dim    → 写入位置 68
  i=69: cache_offset = layer_offset + 69 * kv_dim    → 写入位置 69
  ...
  i=79: cache_offset = layer_offset + 79 * kv_dim    → 写入位置 79

增量路径 (start_pos=69, seq_len=11):
  i=0:  cache_offset = layer_offset + 69 * kv_dim    → 写入位置 69
  i=1:  cache_offset = layer_offset + 70 * kv_dim    → 写入位置 70
  ...
  i=10: cache_offset = layer_offset + 79 * kv_dim    → 写入位置 79
```

**KV Cache 最终内存状态对比**：

```
全量路径写入后:
  ┌────────────────────────────────────────────────────┐
  │ pos: 0   1   2  ... 68 │ 69  70  ... 79           │
  │      K₀  K₁  K₂ ... K₆₈│ K₆₉ K₇₀ ... K₇₉         │
  │  (全部由本次 prefill 写入)                           │
  └────────────────────────────────────────────────────┘

增量路径写入后:
  ┌────────────────────────────────────────────────────┐
  │ pos: 0   1   2  ... 68 │ 69  70  ... 79           │
  │      K₀  K₁  K₂ ... K₆₈│ K₆₉ K₇₀ ... K₇₉         │
  │  (Round 1 已写入,保留)   │ (本次 prefill 写入)       │
  └────────────────────────────────────────────────────┘

比较:
  位置 0-68:
    全量: K_i = RoPE(rms_out[i] · W_K, i)
    增量: 来自 Round 1 prefill 时写入的 K_i = RoPE(rms_out_r1[i] · W_K, i)
    
    因为 Round 2 的 tokens[0:69] 与 Round 1 完全一致（prefix cache 命中的前提条件），
    所以 rms_out_r1[i] == rms_out[i]，两者的 KV 数值完全相同  ✓
    
  位置 69-79:
    全量: K_{69+j} = RoPE(rms_out[69+j] · W_K, 69+j)
    增量: K'_j = RoPE(rms_out'[j] · W_K, 69+j)
    
    由步骤 2-3 已证明: rms_out'[j] = rms_out[69+j], 且 RoPE 位置相同
    所以 K'_j = K_{69+j}  ✓

结论: KV Cache 在位置 [0, 80) 中的每一个数值，两种路径完全相同  ✓
```

#### 5.7.8 步骤 5：Multi-Head Attention —— causal mask 通过 start_pos 正确设定

这是证明中最核心也最微妙的步骤。Attention 涉及跨 token 的交互，需要仔细验证。

**源码** (qwen3.cpp `batched_attention_mha()`):
```cpp
void Qwen3Model::batched_attention_mha(int32_t layer_idx, const tensor::Tensor& query,
                                       const tensor::Tensor& mha_out, 
                                       int32_t seq_len, int32_t start_pos) const {
    tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
    tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);

#if USE_FLASH_ATTENTION
    auto prefill_layer = qwen_layers_->flash_attention_prefill_layer_;
    prefill_layer->set_cur_seq_len(seq_len);    // 全量: 80, 增量: 11
    prefill_layer->set_start_pos(start_pos);    // 全量: 0,  增量: 69
    prefill_layer->set_layer_index(layer_idx);
    prefill_layer->set_input(0, query);          // Q: [seq_len, dim]
    prefill_layer->set_input(1, mha_out);        // Output buffer
    prefill_layer->set_input(2, key_cache);      // 整个 Key Cache
    prefill_layer->set_input(3, val_cache);      // 整个 Value Cache
    STATUS_CHECK(prefill_layer->forward());
#else
    // 非 Flash Attention: 显式分配 score tensor
    tensor::Tensor attn_scores(fp32, seq_len, head_num, max_seq_len, true, alloc);
    STATUS_CHECK(qwen_layers_->batched_mha_layer_->forward(
        start_pos, seq_len, head_num, layer_idx, max_seq_len,
        dim, kv_dim, kv_mul, head_size,
        mha_out, query, attn_scores, key_cache, val_cache));
#endif
    // 后续 Wo 投影 ...
}
```

**Flash Attention Prefill Kernel 源码** (flash_attention_kernel.cu):
```cuda
__global__ void flash_attention_prefill_kernel_fp16(
    const half* Q, const half* K_cache, const half* V_cache, half* O,
    const int seq_len, const int start_pos,
    const int head_num, const int kv_head_num, const int head_size,
    const int kv_mul, const int dim, const int kv_dim, const float scale)
{
    const int head = blockIdx.x;     // 当前处理的 attention head
    const int seq_idx = blockIdx.y;  // 当前处理的序列内位置 (0 ~ seq_len-1)
    
    if (head >= head_num || seq_idx >= seq_len) return;
    
    // ★★★ 核心: 计算绝对位置和有效 KV 长度 ★★★
    const int cur_pos = start_pos + seq_idx;  // 当前 query 的绝对位置
    const int kv_len = cur_pos + 1;           // causal mask: attend to [0, cur_pos]
    
    // 从 Q 中加载当前 query (基于序列内偏移)
    const half* q_ptr = Q + seq_idx * dim + head * head_size;
    
    // 输出写入位置 (基于序列内偏移)
    half* o_ptr = O + seq_idx * dim + head * head_size;
    
    // Online Softmax + Attention 主循环
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    float acc_o = 0.0f;
    
    // 分 tile 处理 KV Cache 中 [0, kv_len) 范围的 key/value
    for (int tile_start = 0; tile_start < kv_len; tile_start += TILE_K) {
        const int tile_len = min(TILE_K, kv_len - tile_start);
        
        // Step 1: Q · K^T (score 计算)
        for (int k_idx = tid; k_idx < tile_len; k_idx += BLOCK_SIZE) {
            const int kv_pos = tile_start + k_idx;
            const half* k_ptr = K_cache + kv_pos * kv_dim + head_offset;
            // 向量化计算 Q · K
            float score = dot(q_ptr, k_ptr) * scale;
            s_scores[k_idx] = score;
        }
        __syncthreads();
        
        // Step 2: Online Softmax 更新
        // ... (更新 row_max, row_sum, 重新缩放 acc_o)
        
        // Step 3: V 累加
        for (int k = 0; k < tile_len; k++) {
            float p = expf(s_scores[k] - new_max);
            acc_o += p * V_cache[(tile_start + k) * kv_dim + head_offset + tid];
        }
    }
    
    // 最终输出
    o_ptr[tid] = __float2half(acc_o / row_sum);
}
```

**Host Launcher 源码** (flash_attention_kernel.cu):
```cpp
void flash_attention_prefill_fp16_cu(int32_t start_pos, int32_t seq_len, ...) {
    // ★ 关键: K/V 指针指向该层的 KV Cache 起始位置（pos=0 处）
    const int layer_offset = layer_index * max_seq_len * kv_dim;
    half* K = key_cache.ptr<half>() + layer_offset;     // 指向 pos=0
    half* V = value_cache.ptr<half>() + layer_offset;   // 指向 pos=0
    
    // grid = (head_num, seq_len): 每个 block 处理一个 (head, query) 对
    dim3 grid(head_num, seq_len);   // 全量: (32, 80), 增量: (32, 11)
    
    flash_attention_prefill_kernel_fp16<<<grid, block, smem_size, stream>>>(
        Q, K, V, O, seq_len, start_pos, ...);
}
```

**等价性逐步验证 —— 以位置 72（全局第 72 个 token）为例**：

```
=== 全量路径: start_pos=0, seq_len=80, seq_idx=72 ===
  cur_pos = 0 + 72 = 72
  kv_len = 72 + 1 = 73
  q_ptr = Q + 72 * dim + head * head_size        ← Q 矩阵第 72 行
  K_cache 起始 = layer_offset                     ← 从 pos=0 开始
  
  Attention 循环:
    tile [0, 64):   Q[72] · K[0], Q[72] · K[1], ..., Q[72] · K[63]
    tile [64, 73):  Q[72] · K[64], Q[72] · K[65], ..., Q[72] · K[72]
    
  输出: O[72] = softmax(Q[72] · K[0:73]^T / √d_k) · V[0:73]

=== 增量路径: start_pos=69, seq_len=11, seq_idx=3 ===
  cur_pos = 69 + 3 = 72                         ← 同一绝对位置！
  kv_len = 72 + 1 = 73                           ← 同样 attend 到 73 个位置！
  q_ptr = Q' + 3 * dim + head * head_size        ← Q' 矩阵第 3 行
  K_cache 起始 = layer_offset                     ← 同样从 pos=0 开始！
  
  Attention 循环:
    tile [0, 64):   Q'[3] · K[0], Q'[3] · K[1], ..., Q'[3] · K[63]
    tile [64, 73):  Q'[3] · K[64], Q'[3] · K[65], ..., Q'[3] · K[72]
    
  输出: O'[3] = softmax(Q'[3] · K[0:73]^T / √d_k) · V[0:73]

验证:
  1. Q'[3] = Q[72]  (步骤 2-3 已证明，包含 RoPE)  ✓
  2. K[0:69] 来自 Round 1 缓存 = 全量路径的 K[0:69]  ✓
  3. K[69:73] 来自本次增量 prefill = 全量路径的 K[69:73]  ✓
  4. V[0:73] 同理  ✓
  5. kv_len = 73，attend 范围完全相同  ✓
  
  因此: O'[3] = O[72]  ✓
```

**完整对应关系**：

```
增量路径 seq_idx | 全局绝对位置 | kv_len | 全量路径 seq_idx
─────────────────┼─────────────┼────────┼─────────────────
      0          |    69       |   70   |      69
      1          |    70       |   71   |      70
      2          |    71       |   72   |      71
      3          |    72       |   73   |      72
      ...        |    ...      |  ...   |      ...
     10          |    79       |   80   |      79

每一行: kv_len、Q 的数值 、K[0:kv_len] 的数值、V[0:kv_len] 的数值 全部一致
→ Attention 输出 全部一致  ✓
```

**关键的等价条件总结**：

| Kernel 参数 | 全量 (seq_idx=72) | 增量 (seq_idx=3) | 是否一致 |
|---|---|---|---|
| `cur_pos = start_pos + seq_idx` | `0 + 72 = 72` | `69 + 3 = 72` | ✓ 一致 |
| `kv_len = cur_pos + 1` | `73` | `73` | ✓ 一致 |
| `Q` 读取位置 | `Q + 72 * dim` | `Q' + 3 * dim` | ✓ 数值一致 |
| `K_cache` 起始 | `layer_offset` | `layer_offset` | ✓ 同一起点 |
| `K[0:69]` 数据来源 | 本次全量写入 | Round 1 缓存 | ✓ 数值一致 |
| `K[69:73]` 数据来源 | 本次全量写入 | 本次增量写入 | ✓ 数值一致 |
| `V[0:73]` 同 K | — | — | ✓ 一致 |

#### 5.7.9 步骤 6：Wo 投影 —— 逐 token 独立

**源码** (qwen3.cpp `batched_attention_mha()` 后半部分):
```cpp
// Wo 投影: mha_out[seq_len, dim] × W_o → wo_out[seq_len, dim]
const auto& wo_layer = qwen_layers_->wo_layers_.at(layer_idx);
tensor::Tensor wo_out(activation_dtype, seq_len, config_->dim_, true, alloc);

auto wo_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(wo_layer);
if (wo_awq) {
    STATUS_CHECK(wo_awq->forward(mha_out, wo_out));
} else {
    STATUS_CHECK(qwen_layers_->batched_matmul_helper_layer_->forward(
        mha_out, wo_matmul->get_weight(0), wo_out, seq_len, 1.f));
}
cudaMemcpyAsync(mha_out.ptr(), wo_out.ptr(), seq_len * dim * elem_size, D2D, stream);
```

Wo 是逐 token 的线性投影：$\text{Attn}_i = \text{MHA}_i \cdot W_O$。已证明 `MHA` 结果一致，Wo 只取决于 MHA 输出，所以 Wo 输出一致。

#### 5.7.10 步骤 7：残差连接 —— 逐元素加法

**源码** (qwen3.cpp `prefill()` 主循环中):
```cpp
// 4. Residual add: layer_output = layer_input + mha_out
STATUS_CHECK(qwen_layers_->batched_add_layer_->forward(*layer_input, mha_out, *layer_output));
```

**BatchedAddLayer 源码** (batched_add.cpp):
```cpp
base::Status BatchedAddLayer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                      const tensor::Tensor& output1) {
    // 逐元素加法: output[i] = input1[i] + input2[i]
    kernel::get_add_kernel(device_type_)(input1, input2, output1, stream);
    return base::error::Success();
}
```

残差 $\text{output}_i = x_i + \text{Attn}_i$ 是**逐元素**运算，每个 token 的残差只取决于该 token 自身的 hidden state 和 attention 输出。

```
增量 output[j] = input'[j] + attn'[j]
               = input[69+j] + attn[69+j]     (已证明两者分别一致)
               = 全量 output[69+j]  ✓
```

#### 5.7.11 步骤 8：FFN（SwiGLU） —— 逐 token 独立

**源码** (qwen3.cpp `batched_feed_forward_optimized()`):
```cpp
void Qwen3Model::batched_feed_forward_optimized(int32_t layer_idx, const tensor::Tensor& input,
    tensor::Tensor& ffn_norm_out, tensor::Tensor& w1_out, tensor::Tensor& w3_out,
    tensor::Tensor& w2_out, int32_t seq_len) const {
    
    // FFN RMSNorm (逐 token 独立)
    STATUS_CHECK(ffn_rmsnorm->forward(input, ffn_norm_out));
    
    // Gate 和 Up 投影 (逐 token 独立的 matmul)
    STATUS_CHECK(w1_awq->forward(ffn_norm_out, w1_out));    // [seq_len, dim] → [seq_len, hidden_dim]
    STATUS_CHECK(w3_awq->forward(ffn_norm_out, w3_out));    // [seq_len, dim] → [seq_len, hidden_dim]
    
    // SwiGLU 激活 (逐元素)
    STATUS_CHECK(qwen_layers_->batched_swiglu_layer_->forward(w1_out, w3_out, w1_out));
    // w1_out[i] = SiLU(w1_out[i]) * w3_out[i]
    
    // Down 投影 (逐 token 独立的 matmul)
    STATUS_CHECK(w2_awq->forward(w1_out, w2_out));          // [seq_len, hidden_dim] → [seq_len, dim]
    
    // 残差连接 (逐元素)
    STATUS_CHECK(qwen_layers_->batched_add_layer_->forward(input, w2_out, input));
    // input[i] = input[i] + w2_out[i]
}
```

FFN 的数学公式：

$$\text{FFN}(x_i) = W_2 \cdot [\text{SiLU}(x_i \cdot W_1) \odot (x_i \cdot W_3)]$$

每一步操作——RMSNorm、W1/W3 matmul、SwiGLU、W2 matmul、残差加——都是**逐 token 独立计算**的。

```
增量 FFN_output'[j] = FFN(hidden'[j])
                    = FFN(hidden[69+j])     (输入 hidden state 已证一致)
                    = 全量 FFN_output[69+j]  ✓
```

#### 5.7.12 步骤 9：逐层累积 —— 数学归纳法

**源码** (qwen3.cpp `prefill()` 主循环):
```cpp
for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
    // layer_input: 上一层的输出（第0层用 embedding 输入）
    if (layer_idx == 0) {
        layer_input = &input;                              // embedding 切片
    } else {
        layer_input = hidden_buffers[(layer_idx - 1) % 2]; // 上一层输出
    }
    layer_output = hidden_buffers[layer_idx % 2];

    batched_attention_rms(layer_idx, *layer_input, rms_out, seq_len);
    batched_attention_qkv(layer_idx, rms_out, query_out, key_out, value_out, seq_len, start_pos);
    batched_attention_mha(layer_idx, query_out, mha_out, seq_len, start_pos);
    batched_add(*layer_input, mha_out, *layer_output);        // 残差
    batched_feed_forward_optimized(layer_idx, *layer_output, ...);   // FFN + 残差
}
```

使用数学归纳法：

**基础情况 (Layer 0)**：
- 输入 = embedding 切片，已证明等价（步骤 0）
- 该层的 RMSNorm → QKV → RoPE → KV Cache → Attention → Wo → 残差 → FFN → 残差
- 每一步都已证明等价 → Layer 0 输出等价 ✓

**归纳步骤 (Layer $l$ → Layer $l+1$)**：
- 假设 Layer $l$ 的输出等价
- 则 Layer $l+1$ 的输入（= Layer $l$ 的输出）等价
- 由步骤 1-8 的分析，Layer $l+1$ 的每一步计算都等价
- 因此 Layer $l+1$ 的输出也等价 ✓

**结论**：所有 36 层（Qwen3-8B）的输出都等价。

#### 5.7.13 步骤 10：Final Logits —— 只取最后一个 token

**源码** (qwen3.cpp `prefill()` 末尾):
```cpp
// 取最后一个 token 的 hidden state
tensor::Tensor* final_hidden = hidden_buffers[(config_->layer_num_ - 1) % 2];

void* last_token_ptr = static_cast<char*>(final_hidden->get_buffer()->ptr()) + 
                       (seq_len - 1) * dim * elem_size;
//                      ^^^^^^^^^^
//                      全量: seq_len-1 = 79, 取 final_hidden[79]
//                      增量: seq_len-1 = 10, 取 final_hidden'[10]

tensor::Tensor last_hidden(activation_dtype, dim, false, nullptr, last_token_ptr);
cls_logits(last_hidden);  // Final RMSNorm + LM Head → logits
```

```
全量: last_hidden = final_hidden[79] = 第79个token(全局最后一个)的hidden state
增量: last_hidden = final_hidden'[10] = 第10个token(序列内最后一个)

由归纳法已证明: final_hidden'[10] = final_hidden_全量[69+10] = final_hidden_全量[79]

因此: cls_logits 的输入相同 → 输出 logits 相同  ✓
```

#### 5.7.14 完整等价性总结

下表总结了 Transformer 每一步计算中，增量 prefill 与全量 prefill 等价的原因：

| 步骤 | 操作 | 为什么等价 | 依赖 start_pos |
|---|---|---|---|
| 0 | Embedding | 查表操作，逐 token 独立 | 否 |
| 1 | RMSNorm | 逐 token 归一化，不涉及其他 token | 否 |
| 2 | Q/K/V Matmul | 逐 token 线性投影 | 否 |
| 2.5 | Per-Head Q/K Norm | 逐 head 独立归一化 | 否 |
| 3 | **RoPE** | **`pos = start_pos + seq_idx` 保证绝对位置正确** | **是** |
| 4 | **KV Cache 写入** | **`cache_offset = (start_pos + i) * kv_dim` 写入正确位置** | **是** |
| 5 | **Attention** | **`cur_pos = start_pos + seq_idx` 和 `kv_len = cur_pos + 1` 保证 causal mask 正确** | **是** |
| 6 | Wo 投影 | 逐 token 线性投影 | 否 |
| 7 | 残差连接 | 逐元素加法 | 否 |
| 8 | FFN (SwiGLU) | 所有子操作（RMSNorm/Matmul/SiLU/Add）均逐 token 独立 | 否 |
| 9 | 逐层累积 | 数学归纳法，每层输入等价 → 输出等价 | 间接 |
| 10 | Final Logits | 取最后一个 token，由归纳法保证等价 | 否 |

**结论**：在 Transformer 的所有计算步骤中，只有 **RoPE、KV Cache 写入、Attention** 三个步骤涉及跨 token 的位置信息或历史数据。这三个步骤通过 `start_pos` 参数正确偏移后，增量 prefill 与全量 prefill 产生**完全相同的数值结果**。其余所有步骤（Embedding、RMSNorm、Matmul、SwiGLU、残差加法）均为逐 token 独立计算，天然不受序列拆分影响。

#### 5.7.15 一个反例：如果不传 start_pos 会怎样？

为加深理解，假设增量 prefill 时错误地传入 `start_pos=0`：

```
错误路径 (start_pos=0, seq_len=11):

1. RoPE 位置错误:
   seq_idx=0: pos = 0 + 0 = 0  → 应该是 69！
   token[69] 被编码为位置 0，而不是位置 69
   → Q/K 的旋转角完全错误

2. KV Cache 覆盖:
   i=0: cache_offset = 0 * kv_dim → 写入位置 0
   这会覆盖 Round 1 缓存的 K/V[0]！
   → 破坏了已有的前缀缓存数据

3. Attention 范围错误:
   seq_idx=0: cur_pos = 0, kv_len = 1
   → 只 attend 到 1 个 KV，应该 attend 到 70 个！
   → 完全丢失了前 69 个 token 的上下文信息

结果: 生成的 logits 完全错误，模型输出毫无意义的内容。
```

这个反例清楚地说明了 **`start_pos` 是增量 prefill 正确性的唯一关键桥梁**。

---

## 6. 总结

OrinMLLM 工程中的 Prefix Cache 系统是一个精心设计的多层架构：

1. **底层 RadixTree**：高效的压缩前缀树结构，支持 O(n) 的插入和查找，通过节点分裂实现路径共享
2. **中层 PrefixCache**：在 RadixTree 之上封装了配置管理、统计信息、LRU 淘汰策略和引用计数保护
3. **上层 PrefixCacheManager**：面向推理框架的接口，提供 match/register/release/extend 等简洁API
4. **应用层 generate_response()**：在实际推理流程中实现增量 prefill，只计算未缓存的 token

这套机制在多轮对话等场景下可以显著减少 prefill 计算量，提升推理吞吐量。引用计数机制确保了并发安全，LRU 淘汰策略则有效控制了内存使用。
