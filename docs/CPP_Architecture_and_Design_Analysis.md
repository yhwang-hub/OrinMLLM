# OrinMLLM 框架 C++ 架构与设计深度分析

本文档详细分析 OrinMLLM 推理框架中运用的 C++ 语言特性、STL 容器、设计模式，以及各核心模块之间的衔接机制。

---

## 目录

- [第一部分：C++ 语言特性与容器使用](#第一部分c-语言特性与容器使用)
  - [1.1 STL 容器使用总览](#11-stl-容器使用总览)
  - [1.2 智能指针体系](#12-智能指针体系)
  - [1.3 模板与泛型编程](#13-模板与泛型编程)
  - [1.4 枚举与类型安全](#14-枚举与类型安全)
  - [1.5 移动语义与右值引用](#15-移动语义与右值引用)
  - [1.6 RAII 资源管理](#16-raii-资源管理)
  - [1.7 运算符重载](#17-运算符重载)
  - [1.8 Lambda 表达式](#18-lambda-表达式)
  - [1.9 预处理器与条件编译](#19-预处理器与条件编译)
  - [1.10 C++11/14/17 其他特性](#110-c111417-其他特性)
- [第二部分：设计模式分析](#第二部分设计模式分析)
  - [2.1 工厂模式（Factory Pattern）](#21-工厂模式factory-pattern)
  - [2.2 单例模式（Singleton Pattern）](#22-单例模式singleton-pattern)
  - [2.3 策略模式（Strategy Pattern）](#23-策略模式strategy-pattern)
  - [2.4 模板方法模式（Template Method Pattern）](#24-模板方法模式template-method-pattern)
  - [2.5 组合模式（Composite Pattern）](#25-组合模式composite-pattern)
  - [2.6 观察者/回调模式](#26-观察者回调模式)
  - [2.7 对象池模式（Object Pool Pattern）](#27-对象池模式object-pool-pattern)
  - [2.8 不可拷贝习语（Non-Copyable Idiom）](#28-不可拷贝习语non-copyable-idiom)
- [第三部分：模块间衔接机制](#第三部分模块间衔接机制)
  - [3.1 Allocator → Buffer 衔接](#31-allocator--buffer-衔接)
  - [3.2 Buffer → Tensor 衔接](#32-buffer--tensor-衔接)
  - [3.3 Tensor → Op（Layer）衔接](#33-tensor--oplayer-衔接)
  - [3.4 Op（Layer）→ Model 衔接](#34-oplayer-model-衔接)
  - [3.5 Model → Buffer 缓冲区管理](#35-model--buffer-缓冲区管理)
  - [3.6 CudaConfig → Layer/Model 衔接](#36-cudaconfig--layermodel-衔接)
  - [3.7 RawModelData → Layer 权重加载](#37-rawmodeldata--layer-权重加载)
  - [3.8 完整数据流示意](#38-完整数据流示意)

---

## 第一部分：C++ 语言特性与容器使用

### 1.1 STL 容器使用总览

#### 1.1.1 `std::vector` — 框架中最核心的容器

`std::vector` 在框架中无处不在，承担着从张量维度描述到层管理的各种核心职责。

> **🔍 OrinMLLM 实战示例：Token 生成与对话历史**
>
> 以下是 `std::vector` 在推理流水线中的三个典型使用场景：
>
> **场景 A：Decode 阶段逐 token 累积生成结果**（`demo/chat_qwen.cpp`）
> ```cpp
> std::vector<int32_t> words;
> words.push_back(next);               // 首个 token
> 
> while (pos < max_length) {
>   // ... decode 步骤生成 next token ...
>   words.push_back(next);             // 逐步追加
>   if (pos >= 3) {
>     // 取最后 4 个 token 检测结束标记
>     auto decoded = model_->decode(std::vector<int32_t>(words.end() - 4, words.end()));
>     if (decoded.find("<|im_end|>") != std::string::npos) break;
>   }
>   pos += 1;
> }
> // 提取生成的回复部分
> std::vector<int32_t> response_tokens(words.begin() + prompt_len, words.end());
> ```
> 这里展示了 `push_back` 逐步追加、子范围构造 `vector(begin, end)` 两种常用操作。
>
> **场景 B：多轮对话历史管理**（`demo/chat_qwen.cpp`）
> ```cpp
> std::vector<ChatMessage> chat_history;
> chat_history.push_back({"system", "You are Qwen..."});  // 系统提示
> chat_history.push_back({"user", user_input});            // 用户输入
> auto response = assistant.chat(chat_history, gen_config);
> chat_history.push_back(response);                        // 助手回复
>
> // 格式化为 ChatML prompt 时遍历
> for (const auto& message : messages) {
>   prompt += "<|im_start|>" + message.role + "\n" + message.content + "\n<|im_end|>\n";
> }
> ```
>
> **场景 C：Embedding 查表时 vector → Tensor 的逐元素填充**（`kuiper/source/model/qwen_base.cpp`）
> ```cpp
> for (int32_t i = 0; i < tokens.size(); ++i) {
>   input_tokens.index<int32_t>(i) = tokens.at(i);   // vector 元素逐个写入 Tensor
> }
> ```
> `.at()` 提供边界检查，适合在调试阶段捕获越界错误。

**（1）Tensor 维度描述**

```cpp
// kuiper/include/tensor/tensor.h
class Tensor {
 private:
  std::vector<int32_t> dims_;   // 存储张量各维度大小，如 {batch, seq_len, dim}
  // ...
};
```

`dims_` 使用 `vector<int32_t>` 来动态表示张量的维度信息。Tensor 支持 1D~4D 的构造函数，统一通过 `dims_` 来管理：

```cpp
// kuiper/source/tensor/tensor.cpp
Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, ...)
    : data_type_(data_type) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  // ...
  size_ = dim0 * dim1 * ...;
}
```

也支持直接用 `vector` 构造并利用 `std::move` 语义避免拷贝：

```cpp
Tensor::Tensor(base::DataType data_type, std::vector<int32_t> dims, ...)
    : dims_(std::move(dims)), data_type_(data_type) {
  size_ = reduce_dimension(dims_.begin(), dims_.end(), 1);
}
```

**（2）Layer 中的输入/输出/权重管理**

```cpp
// kuiper/include/op/layer.h
class Layer : public BaseLayer {
 protected:
  std::vector<tensor::Tensor> inputs_;   // 算子输入张量列表
  std::vector<tensor::Tensor> outputs_;  // 算子输出张量列表
};

class LayerParam : public Layer {
 protected:
  std::vector<tensor::Tensor> weights_;  // 带权重的算子的参数列表
};
```

通过 `vector` 容器，每个算子统一管理其输入、输出以及权重。使用 `resize()` 预分配空间，通过索引访问：

```cpp
// kuiper/source/op/layer.cpp
void Layer::reset_input_size(size_t size) { inputs_.resize(size); }
void Layer::reset_output_size(size_t size) { outputs_.resize(size); }
void LayerParam::reset_weight_size(size_t size) { weights_.resize(size); }
```

**（3）Model 中每层权重算子的管理**

```cpp
// kuiper/include/model/qwen_base.h
struct QwenBaseLayers {
  std::vector<std::shared_ptr<op::Layer>> wq_layers_;       // 每层的 Q 投影算子
  std::vector<std::shared_ptr<op::Layer>> wk_layers_;       // 每层的 K 投影算子
  std::vector<std::shared_ptr<op::Layer>> wv_layers_;       // 每层的 V 投影算子
  std::vector<std::shared_ptr<op::Layer>> wo_layers_;       // 每层的 O 投影算子
  std::vector<std::shared_ptr<op::Layer>> w1_layers_;       // 每层的 FFN gate 算子
  std::vector<std::shared_ptr<op::Layer>> w2_layers_;       // 每层的 FFN down 算子
  std::vector<std::shared_ptr<op::Layer>> w3_layers_;       // 每层的 FFN up 算子
  std::vector<std::shared_ptr<op::Layer>> rmsnorm_layers_;  // 所有 RMSNorm 层
};
```

每种类型的权重算子使用一个 `vector<shared_ptr<Layer>>` 来管理，通过 `layer_idx` 索引：

```cpp
// kuiper/source/model/qwen_base.cpp
const auto& wo_layer = layers->wo_layers_.at(layer_idx);
STATUS_CHECK(wo_layer->forward(mha_output, attn_output));
```

**（4）CUDA 内存池管理**

```cpp
// kuiper/include/base/alloc.h
class CUDADeviceAllocator : public DeviceAllocator {
 private:
  mutable std::map<int, std::vector<CudaMemoryBuffer>> big_buffers_map_;
  mutable std::map<int, std::vector<CudaMemoryBuffer>> cuda_buffers_map_;
};
```

每个 GPU 设备 ID 对应一个 `vector<CudaMemoryBuffer>` 作为内存池，通过遍历 vector 查找空闲的可复用缓冲区。

**（5）RadixTree 中的 Token 序列存储**

```cpp
// kuiper/include/base/radix_tree.h
struct RadixNode {
  std::vector<int32_t> edge_tokens;  // 从父节点到此节点的 token 序列
  // ...
};
```

#### 1.1.2 `std::map` — 有序键值映射

> **🔍 OrinMLLM 实战示例：推理中的 Buffer 查找链路**
>
> 在 Transformer 每一层的推理中，所有中间张量都通过 `buffers_` 这个 `std::map<ModelBufferType, Tensor>` 查找获取：
>
> ```cpp
> // kuiper/source/model/qwen_base.cpp - attention_mha() 中
> tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);     // map 查找 ①
> tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);   // map 查找 ②
> tensor::Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);   // map 查找 ③
> tensor::Tensor query = get_buffer(ModelBufferType::kQuery);            // map 查找 ④
> ```
>
> 每次 `get_buffer` 内部调用 `buffers_.count(idx)` 做存在性检查 + `buffers_.at(idx)` 做安全访问。推理一次 forward 会执行数十次 map 查找，`std::map` 的 $O(\log n)$ 查找在 28+ 个 buffer 的规模下性能完全足够。

**（1）Model 的缓冲区注册表**

```cpp
// kuiper/include/model/model.h
class Model {
 protected:
  std::map<ModelBufferType, tensor::Tensor> buffers_;  // 枚举->张量的映射表
};
```

所有推理过程中的中间缓冲区（输入/输出/KV Cache/临时缓存等）通过枚举类型做 key 存入 `std::map`：

```cpp
// kuiper/source/model/model.cpp
base::Status Model::insert_buffer(ModelBufferType buffer_idx, const tensor::Tensor& tensor) {
  if (buffers_.count(buffer_idx) > 0) {
    return base::error::KeyHasExits(...);
  }
  buffers_.insert({buffer_idx, tensor});
  return base::error::Success();
}

tensor::Tensor& Model::get_buffer(ModelBufferType buffer_idx) {
  CHECK_GT(buffers_.count(buffer_idx), 0);
  return buffers_.at(buffer_idx);
}
```

**（2）CUDA 内存池的设备映射**

```cpp
// kuiper/include/base/alloc.h
mutable std::map<int, size_t> no_busy_cnt_;                       // 每设备空闲内存计数
mutable std::map<int, std::vector<CudaMemoryBuffer>> cuda_buffers_map_; // 每设备的缓冲区列表
```

使用 `std::map<int, ...>` 以 GPU 设备 ID 为 key，支持多 GPU 场景下的内存隔离管理。

#### 1.1.3 `std::unordered_map` — 哈希映射

> **🔍 OrinMLLM 实战示例：RadixTree 中的子节点查找与插入**
>
> PrefixCache 的 RadixTree 在插入/查找 token 序列时，每一步都通过 `children.find()` 进行 $O(1)$ 的子节点匹配：
>
> ```cpp
> // kuiper/include/base/radix_tree.h - insert_impl() 中
> while (token_idx < tokens.size()) {
>     int32_t first_token = tokens[token_idx];
>     
>     auto it = current->children.find(first_token);   // O(1) 哈希查找
>     if (it == current->children.end()) {
>         // 未找到匹配子节点 → 创建新分支
>         auto new_node = std::make_shared<RadixNode>();
>         new_node->edge_tokens.assign(tokens.begin() + token_idx, tokens.end());
>         new_node->parent = current;
>         current->children[first_token] = new_node;   // 插入新子节点
>         return;
>     }
>     auto child = it->second;  // 找到匹配 → 沿树继续深入
>     // ... 处理 edge_tokens 的前缀匹配 ...
> }
> ```
> 选择 `unordered_map` 而非 `map` 的原因：token ID 是整数（天然好的哈希），且树的每个节点通常只有少量子节点，哈希表的 $O(1)$ 查找比红黑树的 $O(\log n)$ 更高效。

```cpp
// kuiper/include/base/radix_tree.h
struct RadixNode {
  std::unordered_map<int32_t, std::shared_ptr<RadixNode>> children;  // token -> 子节点
};
```

RadixTree 的子节点映射使用 `unordered_map`，以 O(1) 的均摊时间复杂度查找子节点。

#### 1.1.4 其他 STL 容器

| 容器 | 使用场景 | 代码位置 |
|------|---------|---------|
| `std::string` | 层名称、文件路径、错误消息 | `layer_name_`、`model_path_` |
| `std::pair` | KV Cache 切片返回 | `Model::slice_kv_cache()` 返回 `std::pair<Tensor, Tensor>` |
| `std::set` | 去重操作（如停止 token 集合） | `qwen3.cpp` 中 |
| `std::atomic` | 线程安全的统计计数器 | `PrefixCacheStats` 中的 `std::atomic<int64_t>` |
| `std::mutex` | PrefixCache 的线程安全保护 | `prefix_cache.h` |
| `std::optional` | 可选返回值 | `prefix_cache.h` 中的前缀匹配结果 |
| `std::weak_ptr` | RadixTree 父节点引用（避免循环引用） | `RadixNode::parent` |
| `std::chrono` | 性能计时 | `tick.h` 中的 `TICK`/`TOCK` 宏 |

> **🔍 OrinMLLM 实战示例：表中关键容器的使用场景**
>
> **`std::pair` — KV Cache 切片的多返回值**
> ```cpp
> // kuiper/source/model/model.cpp
> std::pair<tensor::Tensor, tensor::Tensor> Model::slice_kv_cache(
>     int32_t layer_idx, int32_t token_pos) const {
>   // ... 计算偏移 ...
>   return {key, val};  // 返回 pair，调用端用 C++17 结构化绑定接收
> }
> 
> // kuiper/source/model/qwen3.cpp - 调用端
> auto [key, val] = slice_kv_cache(layer_idx, pos);   // 结构化绑定解构 pair
> ```
>
> **`std::mutex` + `std::lock_guard` — RadixTree 的线程安全保护**
> ```cpp
> // kuiper/include/base/radix_tree.h
> void insert(const std::vector<int32_t>& tokens, int32_t kv_start_pos, int32_t kv_length) {
>     std::lock_guard<std::mutex> lock(mutex_);  // 获取互斥锁
>     insert_impl(tokens, kv_start_pos, kv_length);
>     // 作用域结束 → lock_guard 析构 → 自动释放锁（即使内部抛异常也安全）
> }
> ```
>
> **`std::optional` — BPE 分词中的可选结果**
> ```cpp
> // kuiper/include/base/tiktoken.h - byte_pair_encode() 中
> auto get_rank = [&piece, &ranks](
>     const std::vector<std::pair<int, int>>& parts, int start_idx, int skip
> ) -> std::optional<int> {
>     if (start_idx + skip + 2 < parts.size()) {
>         auto iter = ranks.find(key);
>         if (iter != ranks.end()) return iter->second;  // 找到 → 返回 rank
>     }
>     return std::nullopt;  // 未找到 → 返回空
> };
>
> auto rank = get_rank(parts, i, 0);
> if (rank) {                   // optional 的 bool 转换
>     parts[i].second = *rank;  // 解引用获取值
> }
> ```

---

### 1.2 智能指针体系

OrinMLLM 广泛使用 C++11 智能指针进行资源管理，几乎不使用裸 `new`/`delete`。

#### 1.2.1 `std::shared_ptr` — 共享所有权

**（1）Allocator 的共享**

```cpp
// kuiper/include/base/buffer.h
class Buffer {
 private:
  std::shared_ptr<DeviceAllocator> allocator_;  // 多个 Buffer 可共享同一个 Allocator
};
```

**（2）Buffer 的共享**

```cpp
// kuiper/include/tensor/tensor.h
class Tensor {
 private:
  std::shared_ptr<base::Buffer> buffer_;  // 多个 Tensor 可共享同一块内存
};
```

这是**零拷贝**的关键：多个 Tensor 可以通过 `shared_ptr<Buffer>` 指向同一块内存。项目中通过两种等价机制实现零拷贝：① **Tensor 值拷贝**（`shared_ptr` 引用计数 +1，两个 Tensor 共享同一块物理内存）；② **外部指针 Buffer**（`use_external=true`，新 Buffer 指向已有内存的子区域，不负责释放）。以下是项目中的典型实例：

**实例 ①：多 Buffer 槽复用同一 Tensor — 中间激活 scratch buffer 共享**

```cpp
// kuiper/source/model/qwen3.cpp  init_mem() 中
tensor::Tensor rms_output(activation_dtype, model_dim, true, alloc);
// 同一个 rms_output 被插入到三个不同的 buffer 槽：
CHECK(insert_buffer(ModelBufferType::kOutputRMSNorm, rms_output));   // ← 三者共享
CHECK(insert_buffer(ModelBufferType::kW2Output, rms_output));        //    同一块
CHECK(insert_buffer(ModelBufferType::kFFNRMSNorm, rms_output));      //    GPU 内存
```

`insert_buffer` 按值拷贝 Tensor（见 `model.cpp`：`buffers_.insert({buffer_idx, tensor})`），拷贝后三个槽持有的 `shared_ptr<Buffer>` 指向同一块 GPU 内存。这之所以安全，是因为在推理流水线中 `kOutputRMSNorm`、`kW2Output`、`kFFNRMSNorm` 的**生命期互不重叠**——它们在 Transformer 一层的不同阶段被写入，不会同时被使用，因此可以复用同一块内存，避免为每个中间结果分配独立显存。同样的模式在 `qwen2.cpp`、`llama3.cpp`、`qwen3_vl.cpp` 中都有使用。

**实例 ②：`slice_kv_cache` — 从 KV Cache 大块内存中取子区域视图**

```cpp
// kuiper/source/model/model.cpp  slice_kv_cache()
std::pair<tensor::Tensor, tensor::Tensor> Model::slice_kv_cache(
    int32_t layer_idx, int32_t token_pos) const {
  int32_t layer_offset = layer_idx * config_->seq_len_ * config_->kv_dim_;
  int32_t cache_offset = layer_offset + token_pos * config_->kv_dim_;

  const auto& key_cache_buffer = get_buffer(ModelBufferType::kKeyCache);
  float* key_cache_ptr = const_cast<float*>(key_cache_buffer.ptr<float>(cache_offset));
  float* val_cache_ptr = const_cast<float*>(val_cache_buffer.ptr<float>(cache_offset));

  // 构造外部指针 Tensor，直接指向 KV Cache 内部偏移，不拷贝数据
  tensor::Tensor key(base::DataType::kDataTypeFp32, config_->kv_dim_,
                     false, nullptr, key_cache_ptr);
  tensor::Tensor val(base::DataType::kDataTypeFp32, config_->kv_dim_,
                     false, nullptr, val_cache_ptr);
  // key/val 的 Buffer 标记为 use_external=true，不拥有底层内存
```

`kKeyCache` 是形状为 `[layer_num × seq_len × kv_dim]` 的一整块显存，`slice_kv_cache` 计算出某一层某一个 token 位置的偏移量，直接用指针构建新 Tensor。新 Tensor 的 Buffer 设置 `use_external=true`，表示它**不拥有**这块内存、析构时不释放。这个函数被每层 attention 的 `attention_qkv()` 调用，用于将当前步的 K/V 写入 Cache 的正确位置。

**实例 ③：`fill_input` — 从 Embedding 输出中取单 token 视图**

```cpp
// kuiper/source/model/model.cpp  fill_input()
// 从 embedding lookup 结果中取出第 index 个 token 的嵌入向量
std::shared_ptr<base::Buffer> input_emb_buffer = std::make_shared<base::Buffer>(
    dim * sizeof(uint16_t), nullptr,
    input_embeddings.ptr<uint16_t>(index * dim), true);  // use_external=true
input.assign(input_emb_buffer);
// input 现在是 input_embeddings 中第 index 个 token 的零拷贝视图
```

在 decode 阶段，每步只需要处理一个 token 的嵌入，但 embedding 查表的结果可能包含多个 token。通过 `ptr(offset)` + 外部 Buffer，零拷贝地取出单个 token 的嵌入，无需 `memcpy`。

**实例 ④：reshape + assign — Q/K 的 per-head 重组视图**

```cpp
// kuiper/source/model/qwen3.cpp  attention_qkv() 中
// query_out 形状为 [seq_len, dim]，需要 reshape 为 [seq_len*head_num, head_size] 做 per-head norm
auto q_buffer = std::make_shared<base::Buffer>(
    seq_len * config_->dim_ * elem_size, nullptr,
    const_cast<void*>(query_out.get_buffer()->ptr()), true);  // 指向同一块内存
tensor::Tensor q_reshaped(activation_dtype,
                          seq_len * config_->head_num_, config_->head_size_,
                          false, nullptr, nullptr);
q_reshaped.assign(q_buffer);
// q_reshaped 和 query_out 共享同一块 GPU 内存，只是维度解释不同
```

QKV 投影后需要将输出 reshape 为多头格式来做 per-head RMSNorm（Qwen3 的 Q-Norm/K-Norm）。这里没有拷贝数据，而是创建一个新的外部 Buffer 包裹原 Tensor 的底层指针，然后 `assign` 给不同维度的新 Tensor——**数据完全相同，仅 dims 不同**。

**实例 ⑤：权重 Tensor 共享 mmap 模型文件内存**

```cpp
// kuiper/source/op/layer.cpp  set_weight() 中
std::shared_ptr<base::Buffer> buffer =
    std::make_shared<base::Buffer>(size, nullptr, const_cast<void*>(weight_ptr), true);
tensor::Tensor weight(base::DataType::kDataTypeFp32, dims);
CHECK(weight.assign(buffer));
weights_.at(idx) = weight;
// 所有权重 Tensor 的 Buffer 都直接指向 mmap 映射的模型文件数据，零拷贝
```

模型加载时，权重文件通过 `mmap`（或 `fread`）映射到内存。每个权重 Tensor 的 Buffer 直接指向该内存区域中的对应偏移，`use_external=true` 表示不拥有该内存。整个模型的数百个权重 Tensor 共享同一段 mmap 映射，**没有任何权重数据拷贝**。

> **设计总结**：项目没有实现传统框架中的 `Tensor::slice()` / `Tensor::view()` 方法，而是通过 `shared_ptr<Buffer>` 引用计数共享 + 外部指针 Buffer 两种机制，在需要零拷贝视图的地方手动构建。这种设计更底层、更灵活，适合推理引擎这种内存生命周期高度可控的场景。

**（3）Layer 的共享**

```cpp
// kuiper/include/model/qwen_base.h
struct QwenBaseLayers {
  std::shared_ptr<op::Layer> add_layer_;       // 所有层共享同一个 Add 算子实例
  std::shared_ptr<op::Layer> swiglu_layer_;    // 所有层共享同一个 SwiGLU 算子实例
  std::shared_ptr<op::Layer> mha_layer_;       // 所有层共享同一个 MHA 算子实例
  // ...per-layer lists...
  std::vector<std::shared_ptr<op::Layer>> wq_layers_; // 每层各自的权重算子
};
```

无参数的算子（如 Add、SwiGLU）在所有 Transformer 层之间**共享单个实例**，而带权重的算子（如 MatMul）则**每层独立实例**。

**（4）CudaConfig 的共享**

```cpp
// kuiper/include/op/layer.h
class Layer {
 protected:
  std::shared_ptr<kernel::CudaConfig> cuda_config_;  // 所有 Layer 共享同一个 CudaConfig
};
```

所有算子共享同一个 `CudaConfig`（包含 CUDA Stream、cuBLAS handle 等），确保在同一个 CUDA 流上有序执行。

#### 1.2.2 `std::unique_ptr` — 独占所有权

```cpp
// kuiper/include/model/model.h
class Model {
 protected:
  std::unique_ptr<TransformerConfig> config_;         // 模型配置（独占）
  std::unique_ptr<op::EncodeLayerBase> encode_layer_; // Tokenizer（独占）
  std::unique_ptr<sampler::Sampler> sampler_;          // 采样器（独占）
};

// kuiper/include/model/qwen3.h
class Qwen3Model : public QwenBaseModel {
 protected:
  std::unique_ptr<Qwen3Layers> qwen_layers_;  // 层管理结构（独占）
};
```

模型独占的资源使用 `unique_ptr` 管理，明确表达了所有权语义：一个模型拥有且仅拥有一份配置/tokenizer/采样器。

#### 1.2.3 `std::weak_ptr` — 弱引用（避免循环引用）

```cpp
// kuiper/include/base/radix_tree.h
struct RadixNode {
  std::weak_ptr<RadixNode> parent;  // 父节点弱引用，避免 parent↔children 循环引用
};
```

#### 1.2.4 `std::enable_shared_from_this`

```cpp
// kuiper/include/base/buffer.h
class Buffer : public NoCopyable, std::enable_shared_from_this<Buffer> {
 public:
  std::shared_ptr<Buffer> get_shared_from_this() {
    return shared_from_this();  // 安全地从 this 获取 shared_ptr
  }
};
```

当 `Buffer` 对象需要将自身以 `shared_ptr` 形式传出时，继承 `enable_shared_from_this` 可以安全地获取 `shared_ptr<Buffer>`。

---

### 1.3 模板与泛型编程

> **🔍 OrinMLLM 实战示例：模板在推理流水线中的作用**
>
> 模板的核心价值在推理引擎中体现为"一套代码支持多种数据精度"。以下展示了模板如何贯穿从 KV Cache 访问到 Embedding 填充的完整链路：
>
> **多精度 KV Cache 访问**（`kuiper/source/model/model.cpp`）
> ```cpp
> // FP16 模型 → 使用 uint16_t 指针访问 KV Cache
> uint16_t* key_cache_ptr = const_cast<uint16_t*>(
>     key_cache_buffer.ptr<uint16_t>(cache_offset));   // ptr<T> 模板实例化
> 
> // FP32 模型 → 使用 float 指针访问同一结构的 KV Cache
> float* key_cache_ptr = const_cast<float*>(
>     key_cache_buffer.ptr<float>(cache_offset));       // 同一模板，不同类型
> ```
>
> **Token ID 写入 Tensor**（`kuiper/source/model/qwen_base.cpp`）
> ```cpp
> for (int32_t i = 0; i < tokens.size(); ++i) {
>   input_tokens.index<int32_t>(i) = tokens.at(i);  // index<T> 模板：int32 访问
> }
> int pos = pos_tensor.index<int32_t>(0);             // 读取位置信息
> ```
>
> 无论底层数据是 FP32、FP16 还是 INT32，调用方只需更换模板参数 `<T>`，编译器自动生成对应的指针偏移计算代码。

#### 1.3.1 Tensor 的类型安全模板访问

```cpp
// kuiper/include/tensor/tensor.h
class Tensor {
 public:
  template <typename T>
  T* ptr();                    // 获取指定类型的数据指针

  template <typename T>
  T* ptr(int64_t index);       // 获取偏移位置的指针

  template <typename T>
  T& index(int64_t offset);    // 按偏移下标访问元素
};
```

实现中通过 `reinterpret_cast` 将 `void*` 转换为具体类型：

```cpp
template <typename T>
T* Tensor::ptr() {
  if (!buffer_) return nullptr;
  return reinterpret_cast<T*>(buffer_->ptr());
}

template <typename T>
T& Tensor::index(int64_t offset) {
  CHECK_GE(offset, 0);
  CHECK_LT(offset, this->size());
  T& val = *(reinterpret_cast<T*>(buffer_->ptr()) + offset);
  return val;
}
```

框架中根据不同精度使用不同模板参数：
- `tensor.ptr<float>()` — FP32 数据
- `tensor.ptr<uint16_t>()` — FP16 数据（`__half`）
- `tensor.ptr<int8_t>()` — INT8 量化数据
- `tensor.ptr<int32_t>()` — INT32 数据（如 AWQ packed weights、token ID）

#### 1.3.2 维度计算的泛型函数

```cpp
// kuiper/source/tensor/tensor.cpp
template <typename T, typename Tp>
static size_t reduce_dimension(T begin, T end, Tp init) {
  if (begin >= end) return 0;
  size_t size = std::accumulate(begin, end, init, std::multiplies<>());
  return size;
}
```

这是一个双模板参数的函数，使用 `std::accumulate` 配合 `std::multiplies<>()` 实现通用的维度乘积计算。

---

### 1.4 枚举与类型安全

框架大量使用 **强类型枚举** (`enum class`) 代替传统枚举，避免命名空间污染和隐式转换。

> **🔍 OrinMLLM 实战示例：枚举驱动的类型分发**
>
> 在推理流水线中，枚举值被用于分发到不同的计算路径——这是框架实现多精度/多设备支持的核心机制：
>
> **示例 ①：DataType + AttentionType 联合分发 — 选择注意力实现**（`kuiper/source/model/qwen_base.cpp`）
> ```cpp
> void QwenBaseModel::attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
>   tensor::Tensor query = get_buffer(ModelBufferType::kQuery);
>   tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
>   int pos = pos_tensor.index<int32_t>(0);
>
>   if (query.data_type() == base::DataType::kDataTypeFp16 &&
>       key_cache.data_type() == base::DataType::kDataTypeFp16) {
>     // ① FP16 路径 → 强制使用 Flash Attention（MHA 不支持 FP16）
>     auto flash_attn = layers->flash_attention_decode_layer_;
>     flash_attn->set_layer_index(layer_idx);
>     STATUS_CHECK(flash_attn->forward());
>   } else if (attention_type_ == base::AttentionType::kAttentionMHA) {
>     // ② FP32 标准 MHA 路径
>     auto mha_layer = std::dynamic_pointer_cast<op::MultiHeadAttention>(layers->mha_layer_);
>     mha_layer->set_pos(pos);
>     STATUS_CHECK(mha_layer->forward(query, score_storage, key_cache, val_cache, mha_output));
>   } else {
>     // ③ FP32 Flash Attention 路径
>     auto flash_attn = layers->flash_attention_decode_layer_;
>     STATUS_CHECK(flash_attn->forward());
>   }
> }
> ```
> 三种路径由 `DataType` 和 `AttentionType` 两个枚举联合决定：数据类型决定是否能走标准 MHA，注意力策略枚举决定走哪个 FA 实现。
>
> **示例 ②：DeviceType 枚举控制设备分支**（`kuiper/source/model/qwen_base.cpp`）
> ```cpp
> if (device_type_ != base::DeviceType::kDeviceCUDA) {
>   return base::error::InternalError("Batched prefill only supports CUDA device");
> }
> ```
>
> **示例 ③：DataType 枚举驱动字节大小计算**（`kuiper/include/base/base.h`）
> ```cpp
> inline size_t DataTypeSize(DataType data_type) {
>   if (data_type == DataType::kDataTypeFp32) return sizeof(float);    // 4
>   else if (data_type == DataType::kDataTypeInt8) return sizeof(int8_t);  // 1
>   else if (data_type == DataType::kDataTypeFp16) return 2;           // __half
>   // ...
> }
> 
> // 在 prefill 中使用：
> size_t elem_size = (activation_dtype == base::DataType::kDataTypeFp16)
>     ? sizeof(uint16_t) : sizeof(float);
> ```
> 枚举值在内存分配、张量构造、CUDA kernel 选择等各处被频繁使用，确保类型安全的同时支持多精度推理。

#### 1.4.1 设备类型

```cpp
// kuiper/include/base/base.h
enum class DeviceType : uint8_t {
  kDeviceUnknown = 0,
  kDeviceCPU = 1,
  kDeviceCUDA = 2,
};
```

#### 1.4.2 数据类型

```cpp
enum class DataType : uint8_t {
  kDataTypeUnknown = 0,
  kDataTypeFp32 = 1,
  kDataTypeInt8 = 2,
  kDataTypeInt32 = 3,
  kDataTypeFp16 = 4,
};
```

#### 1.4.3 算子类型

```cpp
// kuiper/include/op/layer.h
enum class LayerType : uint8_t {
  kLayerUnknown = 0,
  kLayerLinear = 1,
  kLayerEncode = 2,
  kLayerEmbedding = 3,
  kLayerRMSNorm = 4,
  kLayerMatmul = 5,
  kLayerRoPe = 6,
  kLayerMHA = 7,
  // ...
};
```

#### 1.4.4 注意力策略

```cpp
enum class AttentionType : uint8_t {
  kAttentionMHA = 0,       // 标准多头注意力
  kAttentionFlash1 = 1,    // FlashAttention v1
  kAttentionFlash2 = 2,    // FlashAttention v2
};
```

#### 1.4.5 内存拷贝方向

```cpp
enum class MemcpyKind {
  kMemcpyCPU2CPU = 0,
  kMemcpyCPU2CUDA = 1,
  kMemcpyCUDA2CPU = 2,
  kMemcpyCUDA2CUDA = 3,
};
```

#### 1.4.6 模型缓冲区类型（传统枚举 + 整型 key，用于 map 查找）

```cpp
// kuiper/include/base/base.h
enum class ModelBufferType {
  kInputTokens = 0,
  kInputEmbeddings = 1,
  kOutputRMSNorm = 2,
  kKeyCache = 3,
  kValueCache = 4,
  kQuery = 5,
  // ... 共计 28 种缓冲区类型
};
```

---

### 1.5 移动语义与右值引用

> **🔍 OrinMLLM 实战示例：移动语义在推理引擎中的性能作用**
>
> 移动语义在推理引擎中的核心价值是**避免不必要的深拷贝**。以下示例展示了移动语义在不同层次的应用：
>
> **场景 A：模型初始化时层名称/路径的零拷贝传递**
>
> 模型加载时，文件路径字符串从 `main()` 到 `Model` 基类要经过多层传递。如果没有 `std::move`，每一层都会触发一次字符串深拷贝（包括堆内存分配）。使用移动语义后，字符串的内部指针直接转移，开销从 $O(n)$ 降为 $O(1)$。
>
> **场景 B：`deepstack_features_` 的整体移动**（`kuiper/source/model/qwen3_vl.cpp`）
> ```cpp
> deepstack_features_.clear();
> deepstack_features_ = std::move(deepstack_features);  
> // 移动整个 vector<Tensor>，避免逐元素拷贝
> // 移动后 deepstack_features 变为空（moved-from 状态）
> ```
> Vision Encoder 输出的深层特征列表可能包含数十个 Tensor（每个持有 `shared_ptr<Buffer>`），移动语义将整个 vector 的内部数组指针一次性转移，避免了逐个 Tensor 的引用计数操作。
>
> **场景 C：CudaConfig 的移动保证资源唯一性**
>
> `CudaConfig` 持有 CUDA Stream、cuBLAS Handle 等不可共享的原生资源，通过**禁止拷贝 + 允许移动**实现资源的唯一拥有者语义。移动后源对象的指针被清零，确保析构时不会 double-free。

#### 1.5.1 构造函数中的 `std::move`

```cpp
// kuiper/source/op/layer.cpp
BaseLayer::BaseLayer(base::DeviceType device_type, LayerType layer_type,
                     base::DataType data_type, std::string layer_name)
    : device_type_(device_type),
      layer_type_(layer_type),
      data_type_(data_type),
      layer_name_(std::move(layer_name)) {}  // 移动字符串避免拷贝
```

```cpp
// kuiper/source/model/model.cpp
Model::Model(base::TokenizerType tokenizer_type, base::ModelType model_type,
             std::string token_path, std::string model_path, bool is_quant_model)
    : token_path_(std::move(token_path)),
      model_path_(std::move(model_path)) {}  // 移动文件路径
```

#### 1.5.2 Tensor 构造中的维度移动

```cpp
Tensor::Tensor(base::DataType data_type, std::vector<int32_t> dims, ...)
    : dims_(std::move(dims)), data_type_(data_type) {  // 移动 vector 避免深拷贝
  size_ = reduce_dimension(dims_.begin(), dims_.end(), 1);
}
```

#### 1.5.3 CudaConfig 的移动语义

```cpp
// kuiper/include/base/cuda_config.h
struct CudaConfig {
  CudaConfig(const CudaConfig&) = delete;             // 禁止拷贝
  CudaConfig& operator=(const CudaConfig&) = delete;  // 禁止拷贝赋值
  
  CudaConfig(CudaConfig&& other) noexcept             // 支持移动
    : stream(other.stream),
      cublas_handle(other.cublas_handle),
      graph_context(std::move(other.graph_context)),
      // ...
  {
    other.stream = nullptr;        // 清空源对象
    other.cublas_handle = nullptr;
  }
};
```

`CudaConfig` 持有 CUDA 原生资源（stream、cuBLAS handle、GPU 内存），只允许移动不允许拷贝，确保资源不被重复释放。

#### 1.5.4 EmbeddingOutput 的移动语义

```cpp
// kuiper/include/op/embedding.h
struct EmbeddingOutput {
  explicit EmbeddingOutput(tensor::Tensor input_tokens, tensor::Tensor input_embeddings,
                           tensor::Tensor input_token_num)
      : input_tokens(std::move(input_tokens)),
        input_embeddings(std::move(input_embeddings)),
        input_token_num(std::move(input_token_num)) {}
};
```

---

### 1.6 RAII 资源管理

> **🔍 OrinMLLM 实战示例：RAII 如何保证 GPU 资源不泄漏**
>
> 推理引擎管理着大量 GPU 资源（显存、CUDA Stream、cuBLAS Handle、CUDA Graph），RAII 确保即使在异常路径下这些资源也能正确释放。以下是三层 RAII 嵌套的典型场景：
>
> ```
> Model 对象析构
>   └→ unique_ptr<CudaConfig> 自动析构 CudaConfig
>        ├→ cudaStreamDestroy(stream)
>        ├→ cublasDestroy(cublas_handle)
>        ├→ cudaFree(cublas_workspace)
>        └→ cudaFree(fp16_input_workspace)
>   └→ unique_ptr<Qwen3Layers> 析构层管理结构
>        └→ 每个 shared_ptr<Layer> 引用计数归零时析构
>             └→ vector<Tensor> weights_ 析构
>                  └→ shared_ptr<Buffer> 引用计数归零时调用 allocator_->release()
> ```
>
> **RadixTree 的 lock_guard RAII** —— 即使 `insert_impl` 内部抛异常，锁也能被正确释放：
> ```cpp
> // kuiper/include/base/radix_tree.h
> void insert(const std::vector<int32_t>& tokens, int32_t kv_start_pos, int32_t kv_length) {
>     std::lock_guard<std::mutex> lock(mutex_);  // 构造时加锁
>     insert_impl(tokens, kv_start_pos, kv_length);
>     // 无论正常返回还是异常退出，lock_guard 析构时自动解锁
> }
> ```
> 对比手动 `mutex_.lock()` / `mutex_.unlock()` 的方式，RAII 的 `lock_guard` 消除了忘记解锁或异常路径未解锁的风险。

#### 1.6.1 Buffer 的自动释放

```cpp
// kuiper/source/base/buffer.cpp
Buffer::~Buffer() {
  if (!use_external_) {
    if (ptr_ && allocator_) {
      allocator_->release(ptr_);  // 析构时自动释放内存
      ptr_ = nullptr;
    }
  }
}
```

`Buffer` 在构造时分配内存，析构时通过 `allocator_->release()` 自动释放，外部用户无需手动管理内存生命周期。

#### 1.6.2 CudaConfig 的资源清理

```cpp
// kuiper/include/base/cuda_config.h
~CudaConfig() {
  if (cublas_workspace) cudaFree(cublas_workspace);
  if (fp16_input_workspace) cudaFree(fp16_input_workspace);
  if (fp16_output_workspace) cudaFree(fp16_output_workspace);
  if (cublas_handle) cublasDestroy(cublas_handle);
  if (stream) cudaStreamDestroy(stream);
}
```

#### 1.6.3 CudaGraph 的生命周期管理

```cpp
// kuiper/include/base/cuda_graph.h
class CudaGraph {
 public:
  ~CudaGraph() { destroy(); }
  
  CudaGraph(const CudaGraph&) = delete;   // 禁止拷贝
  CudaGraph(CudaGraph&& other) noexcept;  // 支持移动
  
  void destroy() {
    if (instance_) { cudaGraphExecDestroy(instance_); instance_ = nullptr; }
    if (graph_) { cudaGraphDestroy(graph_); graph_ = nullptr; }
  }
};
```

---

### 1.7 运算符重载

> **🔍 OrinMLLM 实战示例：Status 运算符重载如何简化错误处理链路**
>
> `Status` 类的 `operator bool()` 让错误处理代码极为简洁。以下是推理主循环中的实际使用：
>
> **predict() → forward() 的错误传播链**（`kuiper/source/model/qwen_base.cpp`）
> ```cpp
> base::Status QwenBaseModel::predict(const tensor::Tensor& input,
>                                     const tensor::Tensor& pos_tensor,
>                                     bool is_prompt, int& next) const {
>   auto status = forward(input, pos_tensor, next);
>   if (!status) {           // ← operator bool() 隐式调用，等价于 status.code_ != kSuccess
>     return status;         // 失败时直接返回错误 Status
>   }
>   next = post_processing(pos_tensor, is_prompt);
>   return base::error::Success();
> }
> ```
>
> **ChatAssistant 中 Status 与 bool 的联合使用**（`demo/chat_qwen.cpp`）
> ```cpp
> auto init_status = model_->init(base::DeviceType::kDeviceCUDA);
> if (!init_status) {       // ← operator bool()
>   LOG(ERROR) << "模型初始化失败: " << init_status.get_err_msg();  // ← operator<< 输出错误
>   return false;
> }
> ```
> 通过 `operator bool()` + `operator<<()`，Status 对象可以像原始类型一样参与条件判断和日志输出，无需调用额外的成员函数。

#### 1.7.1 Status 类的布尔转换运算符

```cpp
// kuiper/include/base/base.h
class Status {
 public:
  operator bool() const;   // 允许 if(!status) 形式的错误检查
  operator int() const;    // 允许与 StatusCode 比较
  bool operator==(int code) const;
  bool operator!=(int code) const;
};
```

使用示例：

```cpp
auto status = forward(input, pos_tensor, next);
if (!status) {
  return status;  // status 可以隐式转 bool，失败时直接 return
}
```

#### 1.7.2 流输出运算符

```cpp
std::ostream& operator<<(std::ostream& os, const Status& x);
```

支持将 `Status` 对象直接打印到日志流中。

---

### 1.8 Lambda 表达式

> **🔍 OrinMLLM 实战示例：Lambda 的三种典型使用模式**
>
> **模式 ①：Lambda 作为批处理操作的封装**（`kuiper/source/model/qwen3_vl.cpp`）
>
> Vision Encoder 需要将数十个权重 Tensor 逐一拷贝到 GPU，用 Lambda 封装重复逻辑：
> ```cpp
> void Qwen3VLVisionLayers::to_cuda(cudaStream_t stream) {
>   auto copy_to_cuda = [stream](tensor::Tensor& tensor) {       // 值捕获 stream
>     if (!tensor.is_empty() && tensor.device_type() != base::DeviceType::kDeviceCUDA) {
>       tensor.to_cuda(stream);
>     }
>   };
>   copy_to_cuda(patch_embed_weight);      // 每个调用只需一行
>   copy_to_cuda(patch_embed_bias);
>   for (auto& block : blocks) {
>     copy_to_cuda(block.norm1_weight);    // 对所有 Transformer block 批量操作
>     copy_to_cuda(block.norm1_bias);
>     copy_to_cuda(block.qkv_weight);
>     // ... 共 10+ 个权重 ...
>   }
> }
> ```
>
> **模式 ②：Lambda 作为 `std::sort` 的比较函数**（`kuiper/include/base/radix_tree.h`）
> ```cpp
> // LRU 淘汰：按最后访问时间排序可淘汰节点
> std::sort(evictable.begin(), evictable.end(),
>           [](const auto& a, const auto& b) {      // auto 参数（C++14 泛型 Lambda）
>               return a.first < b.first;            // pair.first = last_access_time
>           });
> ```
>
> **模式 ③：Lambda 捕获引用 + 格式化 UI**（`demo/chat_qwen.cpp`）
> ```cpp
> auto print_chatml_prompt = [&assistant](const std::vector<ChatMessage>& history) {
>   std::string prompt = assistant.format_messages(history);   // 引用捕获 assistant
>   std::cout << prompt;
> };
> ```
> 此处 Lambda 捕获了外部的 `ChatAssistant` 对象引用，避免在全局作用域定义辅助函数，保持代码的局部性。

```cpp
// kuiper/source/model/qwen3.cpp
void Qwen3Layers::to_cuda(std::shared_ptr<kernel::CudaConfig> config, bool keep_fp16_weights) {
  // Lambda 用于批量设置 FP16 权重标志
  auto set_fp16_flag = [keep_fp16_weights](const std::shared_ptr<op::Layer>& layer) {
    if (auto layer_param = std::dynamic_pointer_cast<op::LayerParam>(layer)) {
      layer_param->set_keep_fp16_weights(keep_fp16_weights);
    }
  };
  
  // 在多个权重层上统一调用
  for (auto& weight_layer : wq_layers_) {
    if (weight_layer) {
      set_fp16_flag(weight_layer);  // 调用 lambda
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }
}
```

这里 Lambda 捕获了 `keep_fp16_weights` 值，并在内部使用 `std::dynamic_pointer_cast` 进行安全的多态向下转型。

---

### 1.9 预处理器与条件编译

> **🔍 OrinMLLM 实战示例：预处理器宏在推理代码中的实际使用**
>
> **`STATUS_CHECK` 宏的密集使用 — Transformer 一层的完整前向传播**（`kuiper/source/model/qwen_base.cpp`）
> ```cpp
> // attention_rms → QKV 投影 → Attention → FFN 的完整链路
> STATUS_CHECK(rmsnorm_layer->forward(input, rmsnorm_output));      // 1. RMSNorm
> STATUS_CHECK(query_layer->forward(rmsnorm_output, query));        // 2. Q 投影
> STATUS_CHECK(key_layer->forward(rmsnorm_output, key));            // 3. K 投影
> STATUS_CHECK(value_layer->forward(rmsnorm_output, val));          // 4. V 投影
> STATUS_CHECK(flash_attn->forward());                              // 5. Attention
> STATUS_CHECK(wo_layer->forward(mha_output, attn_output));         // 6. O 投影
> STATUS_CHECK(layers->add_layer_->forward(input, attn_output, input));  // 7. 残差连接
> STATUS_CHECK(ffn_rmsnorm->forward(input, ffn_norm_output));       // 8. FFN RMSNorm
> STATUS_CHECK(w1_layer->forward(ffn_norm_output, w1_output));      // 9. FFN Gate
> STATUS_CHECK(w3_layer->forward(ffn_norm_output, w3_output));      // 10. FFN Up
> STATUS_CHECK(layers->swiglu_layer_->forward(w1_output, w3_output, w1_output)); // 11. SwiGLU
> STATUS_CHECK(w2_layer->forward(w1_output, w2_output));            // 12. FFN Down
> STATUS_CHECK(layers->add_layer_->forward(input, w2_output, input));  // 13. 残差连接
> ```
> 每一步都通过 `STATUS_CHECK` 宏检查返回状态。如果任何一步失败，宏会打印包含 `__FILE__` 和 `__LINE__` 的精确错误位置，然后通过 `LOG(FATAL)` 终止程序。这种模式让 12+ 步的推理流水线中的错误可以被精确定位到具体的算子调用。

```cpp
// 模型架构的条件编译
#ifdef QWEN3_SUPPORT
  config.immediate_dim_ = config.hidden_dim;
#endif

// 平台特定内存分配
#if (defined(_POSIX_ADVISORY_INFO) && (_POSIX_ADVISORY_INFO >= 200112L))
#define KUIPER_HAVE_POSIX_MEMALIGN
#endif

// CPU 分配器实现
void* CPUDeviceAllocator::allocate(size_t byte_size) const {
#ifdef KUIPER_HAVE_POSIX_MEMALIGN
  void* data = nullptr;
  const size_t alignment = (byte_size >= 1024) ? 32 : 16;
  posix_memalign(&data, alignment, byte_size);  // 对齐分配
  return data;
#else
  return malloc(byte_size);
#endif
}
```

#### 性能计时宏

```cpp
// kuiper/include/base/tick.h
#define TICK(x) auto bench_##x = std::chrono::steady_clock::now();
#define TOCK(x) printf("%s: %lfs\n", #x, \
    std::chrono::duration_cast<std::chrono::duration<double>>( \
        std::chrono::steady_clock::now() - bench_##x).count());
```

使用 `##` 令牌粘贴和 `#` 字符串化操作符实现通用的计时工具。

#### STATUS_CHECK 宏

```cpp
// kuiper/include/base/base.h
#define STATUS_CHECK(call)                                \
  do {                                                    \
    const base::Status& status = call;                    \
    if (!status) {                                        \
      snprintf(buf, buf_size - 1,                         \
               "Infer error\n File:%s Line:%d\n"          \
               "Error code:%d\n Error msg:%s\n",          \
               __FILE__, __LINE__, int(status),            \
               status.get_err_msg().c_str());              \
      LOG(FATAL) << buf;                                   \
    }                                                      \
  } while (0)
```

利用 `__FILE__` 和 `__LINE__` 预定义宏实现精确的错误定位。

---

### 1.10 C++11/14/17 其他特性

> **🔍 OrinMLLM 实战示例：现代 C++ 特性的综合运用**
>
> **C++17 结构化绑定 `auto [a, b] = ...`** —— 在项目中被广泛使用：
> ```cpp
> // KV Cache 切片（model.cpp / qwen3.cpp）
> auto [key, val] = slice_kv_cache(layer_idx, pos);
> 
> // Embedding 输出解构（model.cpp）
> auto [input_tokens, input_embeddings, input_token_num] = embedding_output;
> 
> // Vision 图像预处理（qwen3_vl.cpp）
> auto [resized_pixels, new_width, new_height] = image_utils::smart_resize(...);
>
> // Vision 旋转位置编码（qwen3_vl.cpp）
> auto [cos_cache, sin_cache] = compute_vision_rotary_emb(...);
> ```
> 结构化绑定让返回多个值的函数调用代码更简洁、语义更清晰。
>
> **`override` 关键字** —— 保证虚函数正确覆盖（`kuiper/include/model/qwen3.h`）：
> ```cpp
> class Qwen3Model : public QwenBaseModel {
>   base::Status init(base::DeviceType device_type) override;       // 初始化
>   void attention_qkv(int32_t layer_idx, ...) const override;      // QKV 投影
>   void init_mem() override;                                        // 内存分配
>   base::Status create_layers() override;                           // 层创建
>   QwenBaseLayers* get_base_layers() const override { return qwen_layers_.get(); }
> };
> ```
> 如果拼写错误或参数不匹配，编译器会立即报错，避免无声地创建了新虚函数而非覆盖。
>
> **`dynamic_pointer_cast` 安全多态转型** —— Prefill 时获取权重（`kuiper/source/model/qwen3.cpp`）：
> ```cpp
> auto query_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(query_layer);
> CHECK_NE(query_matmul, nullptr) << "Query layer is not a MatmulLayer";
> STATUS_CHECK(batched_matmul_helper->forward(
>     rms_out, query_matmul->get_weight(0), query_out, seq_len, 1.f));
> ```
> `query_layer` 存储为 `shared_ptr<Layer>` 基类指针，但 `get_weight()` 是子类 `MatmulLayer` 的方法。`dynamic_pointer_cast` 同时完成类型检查和引用计数的正确维护。
>
> **`constexpr` 编译期常量**（`kuiper/include/base/cuda_graph.h`）：
> ```cpp
> static constexpr int kMaxConsecutiveFailures = 3;  // CUDA Graph 连续失败阈值
> ```
> **`explicit` 防止隐式转换**（`kuiper/include/model/model.h`）：
> ```cpp
> explicit Model(base::TokenizerType tokenizer_type, base::ModelType model_type,
>                std::string token_path, std::string model_path, bool is_quant_model);
> // 防止 Model m = {type, type, "path", "path", true}; 这样的隐式构造
> ```

| 特性 | 使用示例 | 代码位置 |
|------|---------|---------|
| `explicit` 构造 | `explicit Tensor()` 防止隐式转换 | `tensor.h` |
| `= default` / `= delete` | `NoCopyable(const NoCopyable&) = delete` | `base.h` |
| `auto` 类型推导 | `auto status = forward(...)` | 全局使用 |
| `override` 关键字 | `base::Status forward() override` | 所有 Layer 子类 |
| `noexcept` | `CudaConfig(CudaConfig&& other) noexcept` | `cuda_config.h` |
| `constexpr` / `inline` | `inline size_t DataTypeSize(DataType)` | `base.h` |
| 基于范围的 for | `for (auto& input : inputs_)` | `layer.cpp` |
| 列表初始化 | `buffers_.insert({buffer_idx, tensor})` | `model.cpp` |
| `std::accumulate` | 维度乘积计算 | `tensor.cpp`, `layer.cpp` |
| `std::multiplies<>` | C++14 透明函数对象 | `tensor.cpp` |
| 可变参数 `va_list` | `check_tensor_with_dim(..., ...)` | `layer.cpp` |
| `reinterpret_cast` | 类型擦除与恢复 `void* -> T*` | `tensor.h` |
| `dynamic_pointer_cast` | 安全多态向下转型 | `qwen3.cpp`, `qwen_base.cpp` |
| `static_cast` | 数值类型转换 | 全局使用 |

---

## 第二部分：设计模式分析

### 2.1 工厂模式（Factory Pattern）

#### 2.1.1 Allocator 工厂

```cpp
// kuiper/include/base/alloc.h
class CPUDeviceAllocatorFactory {
 public:
  static std::shared_ptr<CPUDeviceAllocator> get_instance() {
    if (instance == nullptr) {
      instance = std::make_shared<CPUDeviceAllocator>();
    }
    return instance;
  }
 private:
  static std::shared_ptr<CPUDeviceAllocator> instance;
};

class CUDADeviceAllocatorFactory {
 public:
  static std::shared_ptr<CUDADeviceAllocator> get_instance() {
    if (instance == nullptr) {
      instance = std::make_shared<CUDADeviceAllocator>();
    }
    return instance;
  }
 private:
  static std::shared_ptr<CUDADeviceAllocator> instance;
};

class CPUPinnedAllocatorFactory {
 public:
  static std::shared_ptr<CPUPinnedAllocator> get_instance() { ... }
};
```

三个工厂类（CPU、CUDA、Pinned）提供统一的 `get_instance()` 接口，对使用者屏蔽了不同设备内存分配的细节。

使用示例：

```cpp
// 根据设备类型选择分配器
std::shared_ptr<base::DeviceAllocator> alloc;
if (device_type_ == base::DeviceType::kDeviceCPU) {
  alloc = base::CPUDeviceAllocatorFactory::get_instance();
} else {
  alloc = base::CUDADeviceAllocatorFactory::get_instance();
}
```

#### 2.1.2 RawModelData 多态工厂

```cpp
// kuiper/include/model/raw_model_data.h
struct RawModelData {
  virtual const void* weight(size_t offset) const = 0;  // 纯虚函数
};

struct RawModelDataFp32 : RawModelData {
  const void* weight(size_t offset) const override;  // FP32: 按 float 偏移
};

struct RawModelDataFp16 : RawModelData {
  const void* weight(size_t offset) const override;  // FP16: 按 half 偏移
};

struct RawModelDataInt8 : RawModelData {
  const void* weight(size_t offset) const override;  // INT8: 按 byte 偏移
};
```

根据模型文件格式选择对应的 `RawModelData` 子类，通过虚函数 `weight()` 统一提供权重指针。

---

### 2.2 单例模式（Singleton Pattern）

每个 Allocator 工厂内部都实现了**懒汉式单例**：

```cpp
class CUDADeviceAllocatorFactory {
 public:
  static std::shared_ptr<CUDADeviceAllocator> get_instance() {
    if (instance == nullptr) {
      instance = std::make_shared<CUDADeviceAllocator>();
    }
    return instance;
  }
 private:
  static std::shared_ptr<CUDADeviceAllocator> instance;  // 全局唯一实例
};
```

全局只存在一个 CUDA Allocator 实例，所有 Buffer/Tensor 共用同一个内存池，避免了内存碎片化。

---

### 2.3 策略模式（Strategy Pattern）

#### 2.3.1 Allocator 策略

```cpp
// kuiper/include/base/alloc.h
class DeviceAllocator {
 public:
  virtual void release(void* ptr) const = 0;
  virtual void* allocate(size_t byte_size) const = 0;
  virtual void memcpy(const void* src, void* dest, size_t size, MemcpyKind kind, ...) const;
};

class CPUDeviceAllocator : public DeviceAllocator { ... };
class CPUPinnedAllocator : public DeviceAllocator { ... };
class CUDADeviceAllocator : public DeviceAllocator { ... };
```

`Buffer` 持有 `shared_ptr<DeviceAllocator>`，运行时可以替换为不同的分配策略：
- **CPU 普通内存**：`posix_memalign` / `malloc`
- **CPU 锁页内存**：`cudaMallocHost`（用于异步 DMA 传输）
- **CUDA 设备内存**：`cudaMalloc` + 内存池复用

#### 2.3.2 注意力策略

```cpp
// 运行时切换注意力实现
if (attention_type_ == base::AttentionType::kAttentionMHA) {
  // 标准 MHA：物化 score 矩阵
  mha_layer->forward(query, score_storage, key_cache, val_cache, mha_output);
} else {
  // Flash Attention：在线 softmax，无需物化 score
  flash_attn->set_attention_type(attention_type_);
  flash_attn->forward();
}
```

#### 2.3.3 采样策略

```cpp
// kuiper/include/sampler/sampler.h
class Sampler {
 public:
  virtual size_t sample(const float* logits, size_t size, void* stream = nullptr) = 0;
};

// kuiper/include/sampler/argmax_sampler.h
class ArgmaxSampler : public Sampler {
 public:
  size_t sample(const float* logits, size_t size, void* stream) override;
};
```

采样器通过多态实现，可扩展为 Top-K、Top-P 等采样策略。

#### 2.3.4 Tokenizer 策略

```cpp
// kuiper/include/op/encode.h
class EncodeLayerBase : public Layer {
 public:
  virtual std::vector<int32_t> encode(const std::string& sentence) const = 0;
  virtual std::string decode(int32_t token_id) const = 0;
};

class SpeEncodeLayer : public EncodeLayerBase { ... };   // SentencePiece
class BpeEncodeLayer : public EncodeLayerBase { ... };   // TikToken BPE
class QwenEncodeLayer : public BpeEncodeLayer { ... };   // Qwen 定制 BPE
```

---

### 2.4 模板方法模式（Template Method Pattern）

这是框架中**最核心**的设计模式，`QwenBaseModel` 定义了 Transformer 推理的**骨架算法**，子类只需实现差异化的步骤。

#### 骨架算法定义

```cpp
// kuiper/source/model/qwen_base.cpp — 固定的推理流程
base::Status QwenBaseModel::forward(const tensor::Tensor& input,
                                    const tensor::Tensor& pos_tensor, int& next) const {
  for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
    attention_rms(layer_idx, input);         // 步骤1：RMSNorm（通用）
    attention_qkv(layer_idx, pos_tensor);    // 步骤2：QKV 投影（子类实现）
    attention_mha(layer_idx, pos_tensor);    // 步骤3：多头注意力（通用）
    feed_forward(layer_idx, input);          // 步骤4：前馈网络（通用）
  }
  cls_logits(input);                          // 步骤5：LM Head（通用）
  return base::error::Success();
}
```

#### 子类实现差异化步骤

```cpp
// kuiper/include/model/qwen_base.h — 纯虚函数声明
class QwenBaseModel : public Model {
 protected:
  virtual void attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const = 0;
  virtual void attention_qkv_with_graph(int32_t layer_idx, const tensor::Tensor& pos_tensor) const = 0;
  virtual void batched_attention_qkv(int32_t layer_idx, ...) const = 0;
  virtual QwenBaseLayers* get_base_layers() const = 0;
  
  // 虚函数（有默认实现，子类可覆盖）
  virtual void batched_matmul_forward(...) const;
  virtual void gate_up_swiglu(...) const;
};
```

不同模型子类实现各自的差异化逻辑：

| 子类 | `attention_qkv` 差异 | `gate_up_swiglu` 差异 |
|------|---------------------|-----------------------|
| `Qwen3Model` | Q/K Norm + 标准 RoPE | FP16 fused FFN kernel |
| `Qwen3AWQModel` | AWQ INT4 fused QKV GEMV | AWQ 分步 W1/W3 + SwiGLU |
| `Qwen3SQModel` | Shared quantize + 3x GEMV | SQ fused FFN |
| `Qwen3VLModel` | M-RoPE（3D 位置编码）| 继承 Qwen3Model |

```
Model (抽象基类)
  └── QwenBaseModel (模板方法骨架)
        ├── Qwen3Model (FP16/FP32)
        │     ├── Qwen3AWQModel (AWQ INT4)
        │     ├── Qwen3SQModel (SmoothQuant INT8)
        │     └── Qwen3VLModel (视觉语言模型)
        └── Qwen2Model (Qwen2.5 架构)
```

---

### 2.5 组合模式（Composite Pattern）

模型由**多种层（Layer）组合**构成，每种层可以独立存在也可以组合使用：

```cpp
struct QwenBaseLayers {
  // 无参数层（共享实例）
  std::shared_ptr<op::Layer> add_layer_;
  std::shared_ptr<op::Layer> swiglu_layer_;
  std::shared_ptr<op::Layer> mha_layer_;
  
  // 有参数层（每层独立实例）
  std::vector<std::shared_ptr<op::Layer>> wq_layers_;
  std::vector<std::shared_ptr<op::Layer>> rmsnorm_layers_;
  
  // 特化层
  std::shared_ptr<op::FlashAttentionDecodeLayer> flash_attention_decode_layer_;
  std::shared_ptr<op::KVCacheLayer> kv_cache_key_layer_;
  std::shared_ptr<op::FusedFFNLayer> fused_ffn_layer_;
};
```

推理过程通过组合调用这些层来完成：

```cpp
void QwenBaseModel::feed_forward(int32_t layer_idx, const tensor::Tensor& input) const {
  // Add + RMSNorm + W1 + W3 + SwiGLU + W2 + Add — 7 个 Layer 的组合
  layers->add_layer_->forward(input, attn_output, input);
  ffn_rmsnorm->forward(input, ffn_norm_output);
  w1_layer->forward(ffn_norm_output, w1_output);
  w3_layer->forward(ffn_norm_output, w3_output);
  layers->swiglu_layer_->forward(w1_output, w3_output, w1_output);
  w2_layer->forward(w1_output, w2_output);
  layers->add_layer_->forward(input, w2_output, input);
}
```

---

### 2.6 观察者/回调模式

`forward()` 方法的**多重载**机制实际上是一种变体的回调模式。`Layer::forward()` 接受不同数量的 Tensor 参数，内部将它们设置到 `inputs_`/`outputs_` 后调用无参 `forward()`：

```cpp
// kuiper/source/op/layer.cpp
base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);
  this->set_output(0, output1);
  return this->forward();  // 调用子类具体实现
}
```

这使得每个具体算子只需实现无参的 `forward()`，参数传递由基类统一处理。

---

### 2.7 对象池模式（Object Pool Pattern）

CUDA 内存分配器实现了**内存池**，避免频繁的 `cudaMalloc`/`cudaFree` 系统调用：

```cpp
// kuiper/source/base/alloc_cu.cpp
void* CUDADeviceAllocator::allocate(size_t byte_size) const {
  auto& cuda_buffers = cuda_buffers_map_[id];
  // 1. 优先查找池中空闲的可复用缓冲区
  for (int i = 0; i < cuda_buffers.size(); i++) {
    if (cuda_buffers[i].byte_size >= byte_size && !cuda_buffers[i].busy) {
      cuda_buffers[i].busy = true;
      return cuda_buffers[i].data;  // 复用已有内存
    }
  }
  // 2. 池中无可用缓冲区，才调用 cudaMalloc
  void* ptr = nullptr;
  cudaMalloc(&ptr, byte_size);
  cuda_buffers.emplace_back(ptr, byte_size, true);  // 放入池中
  return ptr;
}

void CUDADeviceAllocator::release(void* ptr) const {
  // 不真正释放，只标记为空闲
  cuda_buffers[i].busy = false;
  // 当空闲内存超过 1GB 阈值时才真正 cudaFree
}
```

大缓冲区（>1MB）和小缓冲区分开管理，大缓冲区匹配时还要求 size 差异不超过 1MB 以避免浪费。

---

### 2.8 不可拷贝习语（Non-Copyable Idiom）

```cpp
// kuiper/include/base/base.h
class NoCopyable {
 protected:
  NoCopyable() = default;
  ~NoCopyable() = default;
  NoCopyable(const NoCopyable&) = delete;
  NoCopyable& operator=(const NoCopyable&) = delete;
};
```

`Buffer` 继承 `NoCopyable`，防止缓冲区被意外拷贝导致 double-free：

```cpp
class Buffer : public NoCopyable, std::enable_shared_from_this<Buffer> { ... };
```

---

## 第三部分：模块间衔接机制

### 3.1 Allocator → Buffer 衔接

**核心关系：Buffer 持有 Allocator 的 shared_ptr，通过 Allocator 分配和释放内存。**

```
DeviceAllocator (接口)
  ├── CPUDeviceAllocator   (malloc / posix_memalign)
  ├── CPUPinnedAllocator   (cudaMallocHost)
  └── CUDADeviceAllocator  (cudaMalloc + 内存池)
         │
         │  shared_ptr<DeviceAllocator>
         ▼
       Buffer (持有 allocator_ 和 ptr_)
```

**衔接代码路径：**

```cpp
// 步骤1: Buffer 构造时，通过 Allocator 分配内存
// kuiper/source/base/buffer.cpp
Buffer::Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator,
               void* ptr, bool use_external)
    : byte_size_(byte_size), allocator_(allocator), ptr_(ptr), use_external_(use_external) {
  if (!ptr_ && allocator_) {
    device_type_ = allocator_->device_type();  // 从 Allocator 获取设备类型
    use_external_ = false;
    ptr_ = allocator_->allocate(byte_size);    // 调用 Allocator 分配内存
  }
}

// 步骤2: Buffer 析构时，通过 Allocator 释放内存
Buffer::~Buffer() {
  if (!use_external_) {
    if (ptr_ && allocator_) {
      allocator_->release(ptr_);  // 调用 Allocator 释放内存
    }
  }
}
```

**外部指针模式**（`use_external = true`）：当权重数据来自 mmap 映射的文件时，Buffer 只持有指针不管理内存：

```cpp
// kuiper/source/op/layer.cpp — 权重加载
std::shared_ptr<base::Buffer> buffer =
    std::make_shared<base::Buffer>(size, nullptr, const_cast<void*>(weight_ptr), true);
    //                                  ^ allocator=null  ^ 外部指针    ^ use_external=true
```

**内存拷贝衔接**：Buffer 的 `copy_from()` 方法根据源和目标的设备类型自动选择正确的 `MemcpyKind`：

```cpp
// kuiper/source/base/buffer.cpp
void Buffer::copy_from(const Buffer& buffer) const {
  const DeviceType& src = buffer.device_type();
  const DeviceType& dst = this->device_type();
  
  if (src == DeviceType::kDeviceCPU && dst == DeviceType::kDeviceCPU) {
    allocator_->memcpy(buffer.ptr(), this->ptr_, byte_size);
  } else if (src == DeviceType::kDeviceCUDA && dst == DeviceType::kDeviceCPU) {
    allocator_->memcpy(buffer.ptr(), this->ptr_, byte_size, MemcpyKind::kMemcpyCUDA2CPU);
  } else if (src == DeviceType::kDeviceCPU && dst == DeviceType::kDeviceCUDA) {
    allocator_->memcpy(buffer.ptr(), this->ptr_, byte_size, MemcpyKind::kMemcpyCPU2CUDA);
  } else {
    allocator_->memcpy(buffer.ptr(), this->ptr_, byte_size, MemcpyKind::kMemcpyCUDA2CUDA);
  }
}
```

---

### 3.2 Buffer → Tensor 衔接

**核心关系：Tensor 持有 Buffer 的 shared_ptr，通过 Buffer 访问底层内存。**

```
Buffer  ─── shared_ptr<Buffer> ──→  Tensor
  │                                    │
  ├─ ptr_ (void*)                      ├─ dims_ (vector<int32_t>)
  ├─ byte_size_                        ├─ size_ (元素数量)
  ├─ device_type_                      ├─ data_type_
  └─ allocator_ (shared_ptr)           └─ buffer_ (shared_ptr<Buffer>)
```

**创建时衔接**（内部分配）：

```cpp
// kuiper/source/tensor/tensor.cpp
bool Tensor::allocate(std::shared_ptr<base::DeviceAllocator> allocator, bool need_realloc) {
  size_t byte_size = this->byte_size();  // size_ * DataTypeSize(data_type_)
  buffer_ = std::make_shared<base::Buffer>(byte_size, allocator, nullptr);
  //                                       ^ 分配大小  ^ 分配器   ^ 自动分配
  return buffer_->ptr() != nullptr;
}
```

**创建时衔接**（外部内存绑定）：

```cpp
void Tensor::init_buffer(std::shared_ptr<base::DeviceAllocator> alloc,
                         base::DataType data_type, bool need_alloc, void* ptr) {
  if (ptr != nullptr) {
    // 绑定到外部内存（不管理生命周期）
    buffer_ = std::make_shared<base::Buffer>(byte_size(), alloc, ptr, true);
  }
}
```

**assign 衔接**（共享 Buffer）：

```cpp
bool Tensor::assign(std::shared_ptr<base::Buffer> buffer) {
  if (byte_size > buffer->byte_size()) {
    LOG(ERROR) << "The size of buffer is too small!";
    return false;
  }
  buffer_ = buffer;  // 直接共享 Buffer（零拷贝）
  return true;
}
```

**设备迁移衔接**（CPU → CUDA）：

```cpp
// kuiper/source/tensor/tensor.cpp
void Tensor::to_cuda(cudaStream_t stream) {
  size_t byte_size = this->byte_size();
  auto cu_alloc = base::CUDADeviceAllocatorFactory::get_instance();
  // 创建新的 CUDA Buffer
  auto cu_buffer = std::make_shared<base::Buffer>(byte_size, cu_alloc);
  // 通过 Allocator 拷贝数据
  cu_alloc->memcpy(buffer_->ptr(), cu_buffer->ptr(), byte_size,
                   base::MemcpyKind::kMemcpyCPU2CUDA, stream);
  // 替换 Buffer（旧 CPU Buffer 引用计数减 1）
  this->buffer_ = cu_buffer;
}
```

**数据访问衔接**（Tensor → Buffer → 原始指针）：

```cpp
template <typename T>
T* Tensor::ptr() {
  return reinterpret_cast<T*>(buffer_->ptr());
  //                          ^^^^^^^^^^^^^^^^ Buffer::ptr() 返回 void*
  //     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ reinterpret_cast 为具体类型
}

template <typename T>
T* Tensor::ptr(int64_t index) {
  return reinterpret_cast<const T*>(buffer_->ptr()) + index;
  //                                                 ^ 指针算术偏移
}
```

---

### 3.3 Tensor → Op（Layer）衔接

**核心关系：Layer 通过 `inputs_`/`outputs_` vector 持有 Tensor，每次 forward 调用时设置并使用。**

```
Tensor ──── inputs_[0..N] ──→ Layer ──── outputs_[0..M] ──→ Tensor
                                │
                            weights_[0..K]  (LayerParam 子类)
```

#### 3.3.1 设置输入输出（参数绑定）

```cpp
// kuiper/source/op/layer.cpp
void Layer::set_input(int32_t idx, const tensor::Tensor& input) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  this->inputs_.at(idx) = input;    // 拷贝 Tensor（shared_ptr<Buffer> 共享）
}

void Layer::set_output(int32_t idx, const tensor::Tensor& output) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  this->outputs_.at(idx) = output;
}
```

> **关键点**：这里拷贝的是 `Tensor` 对象本身（包含 dims_、shared_ptr<Buffer>），但底层 Buffer 通过 shared_ptr 共享，并不产生内存拷贝。

#### 3.3.2 便捷 forward 接口（多参数版本）

基类提供了 1~5 个输入参数的 forward 重载，自动完成参数绑定：

```cpp
// kuiper/source/op/layer.cpp
base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& output1) {
  this->set_input(0, input1);   // 绑定第 1 个输入
  this->set_input(1, input2);   // 绑定第 2 个输入
  this->set_output(0, output1); // 绑定第 1 个输出
  return this->forward();       // 调用子类无参 forward 实现
}
```

#### 3.3.3 具体算子中的 Tensor 访问

以 `MatmulLayer` 为例：

```cpp
// kuiper/source/op/matmul.cpp（简化）
base::Status MatmulLayer::forward() {
  auto status = check();  // 验证输入输出 Tensor 合法性
  
  tensor::Tensor input = this->get_input(0);     // 获取输入 Tensor
  tensor::Tensor output = this->get_output(0);   // 获取输出 Tensor
  tensor::Tensor weight = this->get_weight(0);   // 获取权重 Tensor
  
  // 根据数据类型分派不同的 CUDA kernel
  if (weight.data_type() == base::DataType::kDataTypeFp16) {
    matmul_kernel_cu_pure_fp16(input.ptr<half>(), weight.ptr<half>(),
                                output.ptr<half>(), M, K, N, stream);
  } else {
    matmul_kernel_cu_fp16_weight(input.ptr<float>(), weight.ptr<float>(),
                                 output.ptr<float>(), M, K, N, stream);
  }
  return base::error::Success();
}
```

数据流：`Tensor.ptr<T>()` → `Buffer.ptr()` → `void*` → `reinterpret_cast<T*>` → CUDA kernel 参数。

#### 3.3.4 权重加载流程（RawModelData → Layer）

```cpp
// kuiper/source/model/qwen3.cpp
auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, false);
wq->set_weight_fp16(0, {dim, dim}, raw_model_data_->weight(pos), cpu_device_type);
//                  ^idx  ^shape     ^指向 mmap 文件的指针          ^设备类型
qwen_layers_->wq_layers_.push_back(wq);
```

`set_weight_fp16` 内部创建一个 external Buffer（不管理内存生命周期），并绑定到权重 Tensor：

```cpp
// kuiper/source/op/layer.cpp
base::Status LayerParam::set_weight_fp16(int32_t idx, const std::vector<int32_t>& dims,
                                         const void* weight_ptr, base::DeviceType device_type) {
  size_t size = std::accumulate(dims.begin(), dims.end(), sizeof(uint16_t), std::multiplies<>());
  auto buffer = std::make_shared<base::Buffer>(size, nullptr, const_cast<void*>(weight_ptr), true);
  //                                                 ^ no alloc  ^ external ptr     ^ no manage
  buffer->set_device_type(device_type);
  
  tensor::Tensor weight(base::DataType::kDataTypeFp16, dims);
  weight.set_device_type(device_type);
  weight.assign(buffer);       // Tensor 绑定到 Buffer
  weights_.at(idx) = weight;   // 存入 weights_ vector
  return base::error::Success();
}
```

#### 3.3.5 权重迁移到 GPU（to_cuda 流程）

```cpp
// kuiper/source/op/layer.cpp — LayerParam::to_cuda()
void LayerParam::to_cuda() {
  Layer::to_cuda();   // 先迁移 inputs_/outputs_
  
  for (auto& weight : weights_) {
    if (weight.data_type() == base::DataType::kDataTypeFp16) {
      if (keep_fp16_weights_) {
        // 直接 FP16 拷贝到 GPU
        auto cu_alloc = base::CUDADeviceAllocatorFactory::get_instance();
        auto fp16_buffer = std::make_shared<base::Buffer>(fp16_byte_size, cu_alloc);
        cudaMemcpyAsync(fp16_gpu_ptr, fp16_cpu_ptr, fp16_byte_size,
                        cudaMemcpyHostToDevice, stream);
        weight = tensor::Tensor(base::DataType::kDataTypeFp16, dims);
        weight.assign(fp16_buffer);
      } else {
        // FP16 → FP32 转换并拷贝到 GPU
        auto fp32_buffer = std::make_shared<base::Buffer>(fp32_byte_size, cu_alloc);
        kernel::fp16_cpu_to_fp32_gpu(fp16_ptr, fp32_gpu_ptr, num_elements, stream);
        weight = tensor::Tensor(base::DataType::kDataTypeFp32, dims);
        weight.assign(fp32_buffer);
      }
    } else {
      weight.to_cuda(stream);  // 普通 FP32 迁移
    }
  }
}
```

---

### 3.4 Op（Layer）→ Model 衔接

**核心关系：Model 持有 Layer 的集合（通过 QwenBaseLayers 结构体），在推理流程中按顺序调用各 Layer。**

```
Model
  ├── config_ (unique_ptr<TransformerConfig>)
  ├── buffers_ (map<ModelBufferType, Tensor>)
  ├── cuda_config_ (shared_ptr<CudaConfig>)  ──→ 注入到每个 Layer
  └── qwen_layers_ (unique_ptr<QwenBaseLayers>)
        ├── shared layers: add_layer_, swiglu_layer_, mha_layer_
        └── per-layer vectors: wq_layers_[i], rmsnorm_layers_[i], ...
```

#### 3.4.1 Layer 创建与注册

```cpp
// kuiper/source/model/qwen3.cpp
base::Status Qwen3Model::create_layers() {
  qwen_layers_ = std::make_unique<Qwen3Layers>();  // 创建层容器
  create_nonparam_layers();    // 创建无参数层（Add, SwiGLU, MHA, RoPE, ...）
  create_param_layers();       // 创建有参数层（MatMul, RMSNorm, Embedding, ...）
  create_param_quant_layers(); // 创建量化层（AWQ/SQ 子类覆盖）
  return base::error::Success();
}
```

#### 3.4.2 CudaConfig 注入与 GPU 迁移

```cpp
// kuiper/source/model/qwen3.cpp — 模型初始化
base::Status Qwen3Model::init(base::DeviceType device_type) {
  // 创建 CudaConfig
  cuda_config_ = std::make_shared<kernel::CudaConfig>();
  cudaStreamCreate(&cuda_config_->stream);
  cublasCreate(&cuda_config_->cublas_handle);
  
  // 读取模型并创建层
  gen_model_from_file();
  init_mem();
  
  // 所有层迁移到 GPU 并注入 CudaConfig
  qwen_layers_->to_cuda(cuda_config_, /*keep_fp16_weights=*/true);
}
```

`to_cuda()` 遍历所有层，给每个层注入 CudaConfig 并迁移权重：

```cpp
void Qwen3Layers::to_cuda(std::shared_ptr<kernel::CudaConfig> config, ...) {
  for (auto& weight_layer : wq_layers_) {
    weight_layer->set_cuda_config(config);  // 注入 CudaConfig
    weight_layer->to_cuda();                // 迁移权重到 GPU
  }
  // ... 所有其他层同理
}
```

#### 3.4.3 推理时 Model 与 Layer 的交互

```cpp
// kuiper/source/model/qwen_base.cpp
void QwenBaseModel::attention_rms(int32_t layer_idx, const tensor::Tensor& input) const {
  auto* layers = get_base_layers();                                         // 获取层容器
  tensor::Tensor rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm); // 获取缓冲区
  auto rmsnorm_layer = layers->rmsnorm_layers_.at(layer_idx);               // 获取第 i 层算子
  STATUS_CHECK(rmsnorm_layer->forward(input, rmsnorm_output));              // 调用 forward
}
```

交互路径：`Model.get_buffer()` → `Tensor` → `Layer.forward(input, output)` → CUDA kernel → 结果写入 output Tensor 的 Buffer。

---

### 3.5 Model → Buffer 缓冲区管理

Model 使用 `std::map<ModelBufferType, Tensor>` 管理所有推理缓冲区，在 `init_mem()` 中预分配：

```cpp
// kuiper/source/model/qwen3.cpp — init_mem() 简化
void Qwen3Model::init_mem() {
  auto alloc = base::CUDADeviceAllocatorFactory::get_instance();
  DataType act_type = is_fp16_model_ ? DataType::kDataTypeFp16 : DataType::kDataTypeFp32;
  
  // 预分配推理缓冲区
  tensor::Tensor input_tokens(DataType::kDataTypeInt32, config_->seq_len_, true, alloc);
  tensor::Tensor input_embeddings(act_type, config_->seq_len_, config_->dim_, true, alloc);
  tensor::Tensor rms_output(act_type, config_->dim_, true, alloc);
  tensor::Tensor query(act_type, config_->dim_, true, alloc);
  
  // KV Cache: [layer_num, seq_len, kv_dim]
  tensor::Tensor key_cache(act_type, config_->layer_num_, config_->seq_len_,
                           config_->kv_dim_, true, alloc);
  tensor::Tensor val_cache(act_type, config_->layer_num_, config_->seq_len_,
                           config_->kv_dim_, true, alloc);
  
  // 注册到 Model 的 buffers_ map
  insert_buffer(ModelBufferType::kInputTokens, input_tokens);
  insert_buffer(ModelBufferType::kInputEmbeddings, input_embeddings);
  insert_buffer(ModelBufferType::kOutputRMSNorm, rms_output);
  insert_buffer(ModelBufferType::kQuery, query);
  insert_buffer(ModelBufferType::kKeyCache, key_cache);
  insert_buffer(ModelBufferType::kValueCache, val_cache);
  // ... 共 ~20 个缓冲区
}
```

所有中间计算结果都复用这些预分配的缓冲区，避免推理过程中的动态内存分配。

---

### 3.6 CudaConfig → Layer/Model 衔接

`CudaConfig` 是所有 CUDA 资源的**集中管理点**：

```cpp
struct CudaConfig {
  cudaStream_t stream;              // CUDA 流（所有 kernel 在此流上执行）
  cublasHandle_t cublas_handle;     // cuBLAS 句柄（用于 GEMM）
  std::shared_ptr<CudaGraphContext> graph_context;  // CUDA Graph 上下文
  bool use_cuda_graph;              // 是否启用 CUDA Graph
  __half* fp16_input_workspace;     // FP16 工作区
  __half* fp16_output_workspace;
  void* cublas_workspace;           // cuBLAS 工作区
};
```

衔接方式：

```
CudaConfig (一个实例)
    │
    ├──→ Model.cuda_config_ (shared_ptr)
    │
    ├──→ Layer[0].cuda_config_ (shared_ptr)  → kernel 中访问 stream
    ├──→ Layer[1].cuda_config_ (shared_ptr)
    ├──→ Layer[2].cuda_config_ (shared_ptr)
    └──→ ... (所有 Layer 共享同一个 CudaConfig)
```

算子中获取 stream 的路径：

```cpp
// 在各 kernel 调用中
cudaStream_t stream = cuda_config_->stream;
kernel<<<grid, block, 0, stream>>>(...);        // 所有 kernel 在同一流上有序执行
cublasSetStream(cuda_config_->cublas_handle, stream);  // cuBLAS 也绑定此流
```

---

### 3.7 RawModelData → Layer 权重加载

**完整的权重加载链路：**

```
磁盘文件 (.bin)
    │  mmap / fread
    ▼
RawModelData (虚基类)
  ├── RawModelDataFp32::weight(offset) → (float*)(weight_data) + offset
  ├── RawModelDataFp16::weight(offset) → (uint16_t*)(weight_data) + offset
  └── RawModelDataInt8::weight(offset) → (int8_t*)(weight_data) + offset
    │
    │  raw_model_data_->weight(pos)
    ▼
const void* weight_ptr  (指向 mmap 内存)
    │
    │  set_weight / set_weight_fp16
    ▼
Buffer (external mode, use_external_=true)
    │
    │  Tensor::assign(buffer)
    ▼
Tensor (weights_ vector 中)
    │
    │  to_cuda()
    ▼
Buffer (CUDA mode，新分配的 GPU 内存)
    │
    │  Tensor.ptr<T>()
    ▼
CUDA kernel 消费
```

---

### 3.8 完整数据流示意

以**单次 decode forward**为例，展示完整的模块衔接链路：

```
用户输入 token
      │
      ▼
[EmbeddingLayer]
  input: buffers_[kInputTokens]         (Tensor → Buffer → GPU int32)
  weight: embedding_layer_->weights_[0] (Tensor → Buffer → GPU fp16)
  output: buffers_[kInputEmbeddings]    (Tensor → Buffer → GPU fp16)
      │
      ▼  ×36 层循环
[RMSNormLayer]  rmsnorm_layers_[layer_idx]
  input: 上一层输出 / input_embeddings
  weight: rmsnorm weights (GPU fp16→fp32)
  output: buffers_[kOutputRMSNorm]
      │
      ▼
[MatmulLayer × 3]  wq_layers_[i], wk_layers_[i], wv_layers_[i]
  input: buffers_[kOutputRMSNorm]
  weight: 各自的权重 (GPU fp16)
  output: buffers_[kQuery], kTempKey, kTempValue
      │
      ▼
[RoPEGpuPosLayer]  rope_gpu_pos_layer_
  input: query, key, sin/cos cache, pos_gpu
  output: 原地修改 query, key
      │
      ▼
[KVCacheLayer × 2]  kv_cache_key_layer_, kv_cache_value_layer_
  input: key/value, KV cache, pos_gpu
  output: 写入 KV cache 对应位置
      │
      ▼
[FlashAttentionDecodeLayer]  flash_attention_decode_layer_
  input: query, mha_output, key_cache, val_cache, pos_gpu
  output: buffers_[kOutputMHA]
      │
      ▼
[MatmulLayer]  wo_layers_[i]
  input: buffers_[kOutputMHA]
  output: buffers_[kAttnOutput]
      │
      ▼
[VecAddLayer]  add_layer_  (残差连接)
  input: input + attn_output
  output: input (原地)
      │
      ▼
[FFN: RMSNorm → FusedFFN(W1+W3+SwiGLU) → W2 → Add]
      │
      ▼  ×36 层循环结束
[RMSNormLayer]  final norm
[MatmulLayer]  cls_layer_ (LM Head)
  output: buffers_[kForwardOutput]  → logits [vocab_size]
      │
      ▼
[ArgmaxSampler]  sampler_
  input: logits 指针
  output: next token ID
```

所有步骤中，**数据始终通过 Tensor → Buffer → 原始指针** 在 Layer 之间传递，所有 Tensor 共享预分配的 GPU Buffer，实现了**零动态分配**的高效推理流水线。

---

### 3.9 Encode（Tokenizer）→ Model 衔接

**核心关系：Model 持有 `unique_ptr<EncodeLayerBase>` 作为 Tokenizer，通过多态将文本编解码能力注入模型。**

```
EncodeLayerBase (抽象基类, 继承自 Layer)
  ├── SpeEncodeLayer    (SentencePiece, 用于 LLaMA)
  ├── BpeEncodeLayer    (TikToken BPE, 用于 LLaMA3)
  └── QwenEncodeLayer   (Qwen 定制 BPE, 继承 BpeEncodeLayer)
         │
         │  unique_ptr<EncodeLayerBase>
         ▼
       Model::encode_layer_
```

#### 3.9.1 Tokenizer 的创建与注册

**根据 `tokenizer_type_` 枚举值选择具体的 Tokenizer 实现：**

```cpp
// kuiper/source/model/model.cpp
base::Status Model::create_encode_layer() {
  if (tokenizer_type_ == TokenizerType::kEncodeSpe) {
    encode_layer_ = std::make_unique<op::SpeEncodeLayer>(this->token_path_, true, false);
  } else {
#if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
    encode_layer_ = std::make_unique<op::QwenEncodeLayer>(this->token_path_, false, false);
#endif
  }
  config_->vocab_size_ = encode_layer_->vocab_size();  // 从 tokenizer 获取词表大小
  return error::Success();
}
```

关键衔接点：
1. `create_encode_layer()` 在 `gen_model_from_file()` 中**最先调用**，早于模型文件读取
2. Tokenizer 确定的 `vocab_size_` 被写入 `config_`，后续创建 embedding 层和 lm_head 层时依赖此值
3. 使用条件编译（`#ifdef QWEN3_SUPPORT`）控制可用的 Tokenizer 类型

#### 3.9.2 Model 通过 Tokenizer 进行文本处理

```cpp
// kuiper/source/model/model.cpp
std::vector<int32_t> Model::encode(const std::string& sentence) const {
  CHECK(encode_layer_ != nullptr);
  return encode_layer_->encode(sentence);  // 虚函数调用 → QwenEncodeLayer::encode()
}

std::string Model::decode(int32_t token_idx) const {
  CHECK(encode_layer_ != nullptr);
  return encode_layer_->decode(token_idx);  // 虚函数调用 → QwenEncodeLayer::decode()
}

bool Model::is_sentence_ending(int32_t token_idx) const {
  return encode_layer_->is_sentence_ending(token_idx);  // 判断是否为结束 token
}
```

#### 3.9.3 Tokenizer 编码层的继承链

```cpp
// kuiper/include/op/encode.h
class EncodeLayerBase : public Layer {  // 继承自 Layer（统一的算子接口）
 public:
  virtual std::vector<int32_t> encode(const std::string& sentence) const = 0;
  virtual std::string decode(int32_t token_id) const = 0;
  virtual bool is_sentence_ending(int32_t token_id) const = 0;
  virtual int32_t vocab_size() const = 0;
 protected:
  bool has_bos_;
  bool has_eos_;
  std::string token_model_path_;  // tiktoken.json 或 .model 文件路径
};

class BpeEncodeLayer : public EncodeLayerBase {
 protected:
  std::unique_ptr<tiktoken::tiktoken> tiktoken_;  // TikToken 分词器（独占）
};

class QwenEncodeLayer : public BpeEncodeLayer {
  // 覆盖 encode/decode 以适配 Qwen tokenizer 的特殊行为
  std::vector<int32_t> encode(const std::string& sentence) const override;
  std::string decode(const std::vector<int32_t>& token_ids) const override;
};
```

---

### 3.10 Sampler → Model 衔接

**核心关系：Model 持有 `unique_ptr<Sampler>` 作为采样器，在 `post_processing` 中将 logits 转换为 token。**

```
Sampler (抽象基类)
  └── ArgmaxSampler (Argmax 采样)
         │
         │  unique_ptr<Sampler>
         ▼
       Model::sampler_
```

#### 3.10.1 Sampler 的创建

```cpp
// kuiper/source/model/qwen3.cpp — 在 init() 中创建
sampler_ = std::make_unique<sampler::ArgmaxSampler>(device_type_);
```

#### 3.10.2 Sampler 与推理流程的衔接

```cpp
// kuiper/source/model/qwen_base.cpp
int32_t QwenBaseModel::post_processing(const tensor::Tensor& pos, bool is_prompt) const {
  tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);  // logits
  const float* forward_logits = forward_output.ptr<float>();  // Tensor→Buffer→float*
  
  int32_t next = 0;
  if (is_prompt) {
    next = -1;  // prefill 不输出 token
  } else {
    // Sampler 消费 logits 指针，输出 token ID
    next = static_cast<int32_t>(
        sampler_->sample(forward_logits, forward_output.size(),
                        cuda_config_ ? cuda_config_->stream : nullptr));
  }
  return next;
}
```

**数据流衔接**：cls_layer（LM Head）→ `buffers_[kForwardOutput]`（Tensor）→ `.ptr<float>()`（原始指针）→ `sampler_->sample()`。

#### 3.10.3 ArgmaxSampler 的优化接口

```cpp
// kuiper/include/sampler/argmax_sampler.h
class ArgmaxSampler : public Sampler {
 public:
  size_t sample(const float* logits, size_t size, void* stream) override;
  
  // 使用预分配 GPU + Pinned 缓冲区的优化变体（避免每次 decode 动态分配）
  void sample_prealloc(const float* logits, size_t size,
                       size_t* output_gpu, size_t* output_pinned, void* stream);
};
```

优化的 `sample_prealloc` 与 Model 的预分配缓冲区衔接：

```
buffers_[kForwardOutput].ptr<float>()  →  logits (GPU)
buffers_[kArgmaxOutput].ptr<size_t>()  →  output_gpu
buffers_[kArgmaxOutputPinned].ptr<size_t>()  →  output_pinned (锁页内存, 异步 D2H)
```

---

### 3.11 Model 继承链与虚函数派发衔接

**核心关系：通过 C++ 虚函数机制，父类定义推理骨架，子类按需覆盖差异化步骤。**

```
Model (model.h) — 最顶层抽象基类
  │  纯虚函数: init(), predict(), forward(), embedding(), ...
  │  持有: config_, buffers_, encode_layer_, sampler_, raw_model_data_
  │
  └── QwenBaseModel (qwen_base.h) — 模板方法骨架
        │  定义: forward() 骨架算法 (attention_rms → attention_qkv → attention_mha → feed_forward)
        │  纯虚函数: attention_qkv(), get_base_layers(), batched_attention_qkv()
        │  虚函数(有默认实现): batched_matmul_forward(), gate_up_swiglu()
        │  持有: cuda_config_, use_fused_ffn_
        │
        ├── Qwen3Model (qwen3.h) — FP16/FP32 实现
        │     │  实现: attention_qkv() — Q/K Norm + RoPE
        │     │  实现: get_base_layers() → return qwen_layers_.get()
        │     │  虚函数: batched_qkv_projection() (子类可覆盖 QKV 投影方式)
        │     │  持有: unique_ptr<Qwen3Layers> qwen_layers_
        │     │
        │     ├── Qwen3AWQModel (qwen3_awq.h) — AWQ INT4 量化
        │     │     覆盖: create_param_layers() → 加载 AWQ 权重到 AWQMatmulLayer
        │     │     覆盖: batched_qkv_projection() → AWQ fused QKV kernel
        │     │     覆盖: batched_matmul_forward() → AWQMatmulLayer::forward()
        │     │     覆盖: gate_up_swiglu() → AWQ fused gate+up+SwiGLU kernel
        │     │
        │     ├── Qwen3SQModel (qwen3_sq.h) — SmoothQuant INT8 量化
        │     │     覆盖: create_param_layers() → 加载 SQ 权重到 SQMatmulLayer
        │     │     覆盖: attention_qkv() → 共享量化 + 3 次 GEMV
        │     │     覆盖: gate_up_swiglu() → SQ fused FFN
        │     │
        │     └── (Qwen3VLModel 单独继承 Model，不走 QwenBaseModel)
        │
        └── Qwen2Model (qwen2.h) — Qwen2.5 架构
              实现: attention_qkv() — 带 bias 的 QKV 投影
```

#### 3.11.1 虚函数派发的关键衔接点

`QwenBaseModel::forward()` 中的一个调用如 `attention_qkv(layer_idx, pos)` 在运行时根据实际对象类型派发到不同实现：

```cpp
// 父类骨架
base::Status QwenBaseModel::forward(...) const {
  for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
    attention_qkv(layer_idx, pos_tensor);  // 虚函数调用
    //  ↓ 根据 this 的实际类型派发：
    //  Qwen3Model      → Q/K Norm + 标准 MatMul + RoPE
    //  Qwen3AWQModel   → (继承 Qwen3Model 的默认实现，但 wq_layers_ 中存的是 AWQMatmulLayer)
    //  Qwen3SQModel    → 先 quantize_input()，再 3x forward_preq()
  }
}
```

#### 3.11.2 `get_base_layers()` 的多态桥接

```cpp
// kuiper/include/model/qwen3.h
class Qwen3Model : public QwenBaseModel {
 protected:
  // 将 unique_ptr<Qwen3Layers> 以基类指针形式暴露给父类
  QwenBaseLayers* get_base_layers() const override { return qwen_layers_.get(); }
  std::unique_ptr<Qwen3Layers> qwen_layers_;
};
```

父类 `QwenBaseModel` 通过 `get_base_layers()` 获取层容器的**基类视图**，不需要知道具体是 `Qwen3Layers` 还是其他子类。这使得所有共享的推理逻辑（如 `attention_mha`、`feed_forward`）可以统一操作层容器。

#### 3.11.3 `dynamic_pointer_cast` 实现量化层的多态派发

量化模型子类将量化算子（`AWQMatmulLayer`/`SQMatmulLayer`）存入父类的 `vector<shared_ptr<Layer>>` 中，推理时通过向下转型获取量化特有接口：

```cpp
// kuiper/source/model/qwen3_awq.cpp — batched_qkv_projection()
void Qwen3AWQModel::batched_qkv_projection(int32_t layer_idx, ...) const {
  const auto& query_layer = qwen_layers_->wq_layers_.at(layer_idx);  // shared_ptr<Layer>
  
  // 向下转型为 AWQMatmulLayer 以访问量化特有的成员
  auto query_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(query_layer);
  CHECK_NE(query_awq, nullptr);
  
  // 使用 AWQ 特有接口
  query_awq->get_qweight_t();   // 获取转置后的量化权重
  query_awq->get_qzeros();      // 获取零点
  query_awq->get_scales();      // 获取缩放因子
  query_awq->in_features();     // 获取输入维度
  query_awq->group_size();      // 获取量化组大小
}
```

这种设计允许在**不修改父类 `QwenBaseLayers` 结构体**的情况下，支持 FP16/AWQ/SQ 三种不同的量化格式——通过 C++ 的运行时多态（`shared_ptr<Layer>` 存储 + `dynamic_pointer_cast` 恢复）。

---

### 3.12 量化层（AWQ/SQ）→ 标准层的替换衔接

**核心关系：AWQ/SQ 量化层与标准 MatmulLayer 共享相同的 `shared_ptr<Layer>` 容器槽位，通过子类覆盖 `create_param_layers()` 填充不同的算子类型。**

```
标准路径 (Qwen3Model):
  wq_layers_[i] = make_shared<MatmulLayer>(...)        → MatmulLayer::forward()

AWQ 路径 (Qwen3AWQModel):
  wq_layers_[i] = make_shared<AWQMatmulLayer>(...)     → AWQMatmulLayer::forward()

SQ 路径 (Qwen3SQModel):
  wq_layers_[i] = make_shared<SQMatmulLayer>(...)      → SQMatmulLayer::forward()
```

#### 3.12.1 标准 FP16 权重加载

```cpp
// kuiper/source/model/qwen3.cpp — Qwen3Model::create_param_layers_fp16()
auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, false);
wq->set_weight_fp16(0, {dim, dim}, raw_model_data_->weight(pos), cpu_device_type);
qwen_layers_->wq_layers_.push_back(wq);  // 存入 vector<shared_ptr<Layer>>
```

#### 3.12.2 AWQ INT4 权重加载

```cpp
// kuiper/source/model/qwen3_awq.cpp — Qwen3AWQModel::create_param_layers_awq()
auto load_awq_layer = [&](int32_t in_features, int32_t out_features,
                          std::vector<std::shared_ptr<op::Layer>>& layer_list, ...) {
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto awq_layer = std::make_shared<op::AWQMatmulLayer>(
        device_type_, in_features, out_features, group_size_);
    
    // AWQ 三元组权重设置
    awq_layer->set_awq_weights(qweight_ptr, qzeros_ptr, scales_ptr, cpu_device_type);
    layer_list.push_back(awq_layer);  // 存入同一个 vector<shared_ptr<Layer>>
  }
};

load_awq_layer(dim, dim, qwen_layers_->wq_layers_, "wq");   // 替换标准层
load_awq_layer(dim, kv_dim, qwen_layers_->wk_layers_, "wk");
```

#### 3.12.3 SQ INT8 权重加载

```cpp
// kuiper/source/model/qwen3_sq.cpp — Qwen3SQModel::create_param_layers_sq()
auto load_sq_layer = [&](int32_t in_features, int32_t out_features,
                         std::vector<std::shared_ptr<op::Layer>>& layer_list, ...) {
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto sq_layer = std::make_shared<op::SQMatmulLayer>(
        device_type_, in_features, out_features);
    
    // SQ 权重 = INT8 量化权重 + 权重缩放因子 + 输入缩放因子
    sq_layer->set_sq_weights(qweight_ptr, weight_scale_ptr, input_scale_ptr, cpu_device_type);
    layer_list.push_back(sq_layer);  // 存入同一个 vector<shared_ptr<Layer>>
  }
};

load_sq_layer(dim, dim, qwen_layers_->wq_layers_, "wq");   // 替换标准层
```

#### 3.12.4 `batched_matmul_forward` 的虚拟派发

父类提供默认实现（使用 `batched_matmul_helper_layer_`），量化子类覆盖为各自的实现：

```cpp
// QwenBaseModel 默认实现（FP16 路径）
void QwenBaseModel::batched_matmul_forward(const shared_ptr<Layer>& layer,
                                           const Tensor& input, const Tensor& output,
                                           int32_t seq_len) const {
  auto* layers = get_base_layers();
  auto layer_param = std::dynamic_pointer_cast<op::LayerParam>(layer);
  layers->batched_matmul_helper_layer_->forward(input, layer_param->get_weight(0),
                                                 output, seq_len, 1.0f);
}

// AWQ 覆盖：向下转型为 AWQMatmulLayer
void Qwen3AWQModel::batched_matmul_forward(const shared_ptr<Layer>& layer,
                                            const Tensor& input, const Tensor& output,
                                            int32_t seq_len) const {
  auto awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(layer);
  if (awq) {
    STATUS_CHECK(awq->forward(input, output));  // AWQ 特有的 batched forward
  } else {
    Qwen3Model::batched_matmul_forward(layer, input, output, seq_len);  // fallback
  }
}
```

---

### 3.13 Vision Encoder → LLM 的多模态衔接（Qwen3-VL）

**核心关系：VL 模型包含独立的 Vision Encoder 子系统，通过 Tensor 作为桥梁将视觉特征注入 LLM 的 Embedding 空间。**

```
[图像文件]
    │  preprocess_image()
    ▼
ImageData { pixel_values: Tensor, grid_h, grid_w, ... }
    │  encode_image()
    ▼
tensor::Tensor visual_embeddings  [num_vision_tokens, out_hidden_size]
    │  prepare_multimodal_embeddings()
    ▼
tensor::Tensor combined_embeddings  [seq_len, dim]
（<image> token 位置被替换为 visual_embeddings）
    │  prefill() → 36 层 Transformer
    ▼
logits
```

#### 3.13.1 Vision Encoder 的层结构

Vision Encoder 使用**独立的权重结构体**而非复用 LLM 的 Layer 体系：

```cpp
// kuiper/include/model/qwen3_vl.h
struct Qwen3VLVisionLayers {
  tensor::Tensor patch_embed_weight;    // Conv3d 权重 [hidden, 3, t, h, w]
  tensor::Tensor patch_embed_bias;      // Conv3d 偏置
  tensor::Tensor pos_embed_weight;      // 位置嵌入

  struct Block {                        // 27 个 Transformer Block
    tensor::Tensor norm1_weight;
    tensor::Tensor norm1_bias;
    tensor::Tensor qkv_weight;          // 融合的 QKV 投影 [3*hidden, hidden]
    tensor::Tensor qkv_bias;
    tensor::Tensor proj_weight;         // 输出投影
    tensor::Tensor mlp_fc1_weight;      // MLP 第一层
    tensor::Tensor mlp_fc2_weight;      // MLP 第二层
    // ...
  };
  std::vector<Block> blocks;            // 27 个 Block

  struct Merger {                       // Vision→LLM 投影
    tensor::Tensor fc1_weight;          // [merged_hidden, merged_hidden]
    tensor::Tensor fc2_weight;          // [out_hidden_size, merged_hidden]
    // ...
  };
  Merger merger;
  std::vector<Merger> deepstack_mergers;  // 3 个深层特征 merger
};
```

#### 3.13.2 Vision 特有的算子层

Vision 模块拥有独立的 op 层集合，封装了 ViT 需要的特殊运算：

```cpp
// kuiper/include/model/qwen3_vl.h
struct VisionVLLayers {
  shared_ptr<op::ExtractPatchesLayer> extract_patches_layer_;
  shared_ptr<op::BiasAddResidualLayer> bias_add_residual_layer_;
  shared_ptr<op::PosEmbedInterpolateLayer> pos_embed_interpolate_layer_;
  shared_ptr<op::LayerNormWithBiasLayer> layernorm_with_bias_layer_;
  shared_ptr<op::FusedSplitRopeTransposeLayer> fused_split_rope_transpose_layer_;
  shared_ptr<op::VisionAttentionLayer> vision_attention_layer_;
  shared_ptr<op::VisionMLPLayer> vision_mlp_layer_;
  shared_ptr<op::SpatialMergeLayer> spatial_merge_layer_;
  shared_ptr<op::VisionMergerMLPLayer> vision_merger_mlp_layer_;
  shared_ptr<op::FusedMultimodalEmbedLayer> fused_multimodal_embed_layer_;
};
```

#### 3.13.3 VisionWorkspace — 预分配的中间缓冲区

ViT 推理也采用**预分配缓冲区**策略，与 LLM 的 `buffers_` map 设计理念一致：

```cpp
struct VisionWorkspace {
  int max_patches = 0;
  tensor::Tensor normed1;           // [max_patches, hidden_size]
  tensor::Tensor qkv;               // [max_patches, 3*hidden_size]
  tensor::Tensor query, key, value; // 分离后的 QKV
  tensor::Tensor attn_out;          // 注意力输出
  tensor::Tensor mlp_intermediate;  // MLP 中间层
  tensor::Tensor output, output2;   // 双缓冲（ping-pong buffer）
  // 注意力工作区（避免动态分配）
  tensor::Tensor q_transposed, k_transposed, v_transposed;
  tensor::Tensor attn_scores;
};
```

#### 3.13.4 Qwen3VLModel 的独立继承路径

注意 `Qwen3VLModel` **直接继承 `Model`** 而非 `QwenBaseModel`，因为 VL 模型的推理流程与纯文本模型差异较大（需要处理图像预处理、多模态嵌入融合、M-RoPE 等）：

```cpp
class Qwen3VLModel : public Model {   // 直接继承 Model，不走 QwenBaseModel
 public:
  // VL 特有接口
  ImageData preprocess_image(const std::string& image_path, int max_pixels) const;
  tensor::Tensor encode_image(const ImageData& image_data) const;
  tensor::Tensor prepare_multimodal_embeddings(const std::vector<int>& tokens,
                                                const ImageData* image_data) const;
  base::Status multimodal_prefill(const std::vector<int>& tokens,
                                   const std::string& image_path) const;
 private:
  Qwen3VLConfig vl_config_;                       // VL 专属配置
  std::unique_ptr<Qwen3VLVisionLayers> vision_layers_;  // 视觉编码器权重
  std::unique_ptr<VisionVLLayers> vision_op_layers_;    // 视觉算子层
  std::unique_ptr<VisionWorkspace> vision_workspace_;   // 视觉推理缓冲区
  std::unique_ptr<Qwen3Layers> qwen_layers_;            // LLM 权重（复用 Qwen3 结构）
};
```

---

### 3.14 Config → Model/Layer 的配置驱动衔接

**核心关系：`TransformerConfig` 驱动 Model 的内存分配、Layer 创建和推理参数。**

```
ModelConfig (从模型文件 header 读取)
    │  generate_model_infos()
    ▼
TransformerConfig (unique_ptr<TransformerConfig> config_)
    │
    ├──→ init_mem()：根据 dim_, seq_len_, layer_num_ 等预分配缓冲区
    ├──→ create_nonparam_layers()：用 head_num_, kv_dim_, head_size_ 创建 MHA/RoPE/FA 层
    ├──→ create_param_layers()：用 dim_, kv_dim_, immediate_dim_ 创建各 MatMul 层
    └──→ forward()：用 layer_num_ 控制循环次数
```

#### 3.14.1 文件头解析 → Config 生成

```cpp
// kuiper/source/model/model.cpp
base::Status Model::read_model_file() {
  uint32_t magic = 0;
  fread(&magic, sizeof(uint32_t), 1, file);
  
  // 根据 magic number 识别模型格式
  if (magic == 0x616b3437) {  // "ak47" = Qwen3
    is_fp16_model_ = true;
    fread(&config, ...);      // 读取 dim, hidden_dim, layer_num, ...
  }
  
  // 根据模型精度创建对应的 RawModelData
  if (is_fp16_model_) {
    raw_model_data_ = std::make_shared<RawModelDataFp16>();   // FP16 偏移计算
  } else {
    raw_model_data_ = std::make_shared<RawModelDataFp32>();   // FP32 偏移计算
  }
  
  // mmap 映射权重文件
  raw_model_data_->data = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  raw_model_data_->weight_data = (int8_t*)raw_model_data_->data + 256;  // 跳过 header
}
```

#### 3.14.2 Config 驱动 Layer 参数化

```cpp
// kuiper/source/model/qwen3.cpp — create_nonparam_layers()
qwen_layers_->flash_attention_decode_layer_ = std::make_shared<op::FlashAttentionDecodeLayer>(
    device_type_,
    config_->head_num_,      // ← 从 Config 获取注意力头数
    config_->kv_head_num_,   // ← 从 Config 获取 KV 头数（GQA）
    config_->head_size_,     // ← 从 Config 获取每头维度
    config_->kv_mul_,        // ← 从 Config 获取 KV 倍数
    config_->seq_len_,       // ← 从 Config 获取最大序列长度
    config_->kv_dim_,        // ← 从 Config 获取 KV 维度
    is_fp16_model_);         // ← 从 Model 标志位获取精度
```

#### 3.14.3 完整的初始化链路

```
Model::gen_model_from_file()
  │
  ├── create_encode_layer()     → Tokenizer 创建
  │     └── vocab_size_ → config_
  │
  ├── read_model_file()         → 读取 header + mmap 权重
  │     ├── magic → is_fp16_model_, is_awq_model_, is_sq_model_
  │     ├── header fields → config_->dim_, head_num_, ...
  │     └── raw_model_data_ = make_shared<RawModelDataFp16>()
  │
  └── create_layers()           → 创建所有算子
        ├── create_nonparam_layers()   → 用 config_ 参数化无权重层
        ├── create_param_layers()      → 用 raw_model_data_->weight(pos) 加载权重到层
        └── create_param_quant_layers()
```

---

### 3.15 模块衔接总览图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             OrinMLLM 模块衔接全景                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐     mmap      ┌────────────────┐    weight(pos)          │
│  │  磁盘 .bin   │───────────────▶│  RawModelData  │──────────┐             │
│  │  模型文件     │               │ (Fp16/Fp32/Int8)│          │             │
│  └──────────────┘               └────────────────┘          │             │
│                                                               ▼             │
│  ┌──────────────────────────────────────────────────────────────┐          │
│  │                 Layer 层 (Op 算子)                            │          │
│  │                                                              │          │
│  │  ┌──────────────┐  ┌───────────────┐  ┌─────────────────┐  │          │
│  │  │  LayerParam   │  │ AWQMatmulLayer│  │  SQMatmulLayer  │  │          │
│  │  │  (MatMul/     │  │ (INT4 量化)   │  │  (INT8 量化)    │  │          │
│  │  │   RMSNorm/    │  └───────┬───────┘  └───────┬─────────┘  │          │
│  │  │   Embedding)  │          │                   │            │          │
│  │  └──────┬───────┘          │                   │            │          │
│  │         │        ┌─────────┴───────────────────┘            │          │
│  │         │        │  存入同一个 vector<shared_ptr<Layer>>      │          │
│  │         │        ▼                                           │          │
│  │  ┌──────────────────────────────────────┐                   │          │
│  │  │      QwenBaseLayers 结构体            │                   │          │
│  │  │  wq_layers_[i], wk_layers_[i], ...  │◀── shared_ptr ──┤          │
│  │  │  add_layer_, swiglu_layer_ (共享)     │                   │          │
│  │  └──────────────────────────────────────┘                   │          │
│  └──────────────────────────────────────────────────────────────┘          │
│         │                                                                   │
│         │ unique_ptr<Qwen3Layers>                                          │
│         ▼                                                                   │
│  ┌──────────────────────────────────────────────────────────────┐          │
│  │                    Model 层                                   │          │
│  │                                                              │          │
│  │  ┌──────────┐   ┌──────────────────────┐   ┌────────────┐   │          │
│  │  │ config_  │   │ buffers_ (map)       │   │cuda_config_│   │          │
│  │  │(unique)  │   │ kInputTokens→Tensor  │   │ (shared)   │──┼──▶ 所有  │
│  │  └──────────┘   │ kKeyCache→Tensor     │   └────────────┘   │   Layer  │
│  │                  │ kForwardOutput→Tensor│                    │          │
│  │  ┌────────────┐ └──────────────────────┘  ┌──────────────┐ │          │
│  │  │encode_layer│                            │   sampler_   │ │          │
│  │  │ (unique)   │                            │  (unique)    │ │          │
│  │  └──────┬─────┘                            └──────┬───────┘ │          │
│  └─────────┼─────────────────────────────────────────┼─────────┘          │
│            │                                          │                     │
│     encode(text)                          sample(logits)→token             │
│            │                                          │                     │
│  ┌─────────┴─────────────────────────────────────────┴────────────┐       │
│  │                     外部调用接口                                 │       │
│  │  tokens = model.encode("Hello")                                │       │
│  │  model.predict(input, pos, is_prompt, next_token)              │       │
│  │  text = model.decode(next_token)                               │       │
│  └────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────┐          │
│  │  底层基础设施                                                 │          │
│  │                                                              │          │
│  │  DeviceAllocator ──shared_ptr──▶ Buffer ──shared_ptr──▶ Tensor        │
│  │   (CPU/CUDA/Pinned)     alloc/release   ptr()/byte_size()   dims_     │
│  │                                                              │          │
│  │  AllocatorFactory (单例) ──▶ 全局唯一 Allocator 实例          │          │
│  │  CudaConfig (共享) ──▶ stream + cublas_handle + graph_context│          │
│  └─────────────────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 总结

| 维度 | 关键技术点 |
|------|-----------|
| **容器** | `vector`（维度/层/池）、`map`（缓冲区注册表）、`unordered_map`（RadixTree） |
| **智能指针** | `shared_ptr`（Buffer/Allocator/Layer/CudaConfig 共享）、`unique_ptr`（Model 独占）、`weak_ptr`（树结构回避循环引用）|
| **类型安全** | `enum class` 设备/数据/算子类型、模板化 Tensor 访问 |
| **RAII** | Buffer 自动释放、CudaConfig 析构清理资源 |
| **移动语义** | 构造参数移动、CudaConfig 仅可移动 |
| **设计模式** | 工厂（Allocator）、单例（AllocatorFactory）、策略（Allocator/Attention/Sampler）、模板方法（QwenBaseModel）、组合（Layer 组合推理）、对象池（CUDA 内存池）|
| **模块衔接** | Allocator→Buffer（分配/释放）、Buffer→Tensor（持有/共享）、Tensor→Layer（输入/输出/权重）、Layer→Model（创建/管理/调用）、CudaConfig→全局（流/句柄共享）|
