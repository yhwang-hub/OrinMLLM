# CUDA Graph 在 qwen3.cpp 中的应用深度分析报告

> 项目：OrinMLLM — 基于 NVIDIA Orin 的大语言模型推理引擎  
> 分析文件：`kuiper/source/model/qwen3.cpp`、`kuiper/include/base/cuda_graph.h`、`kuiper/include/base/cuda_config.h`  
> 报告日期：2025 年

---

## 目录

1. [什么是 CUDA Graph？原理、步骤与 API 详解](#1-什么是-cuda-graph)
2. [qwen3.cpp 中 CUDA Graph 的使用方式与源码分析](#2-qwen3cpp-中-cuda-graph-的使用方式)
3. [CUDA Graph 在 qwen3.cpp 中的难点与解决方案](#3-cuda-graph-在-qwen3cpp-中的难点与解决方案)
4. [CUDA Graph 如何带来加速及源码分析](#4-cuda-graph-如何带来加速)

---

## 1. 什么是 CUDA Graph？

### 1.1 基本概念

CUDA Graph 是 NVIDIA CUDA 10.0 引入的一种 **GPU 工作提交优化机制**。它将一系列 GPU 操作（kernel 启动、内存拷贝等）预先录制为一个有向无环图（DAG），然后在后续推理中一次性提交整个图进行执行，从而大幅减少 CPU 端的 kernel 启动开销。

传统 CUDA 编程模型中，每个 kernel 启动都需要 CPU 端调用 CUDA Runtime API，经过驱动层将命令提交给 GPU。在 LLM 推理的 decode 阶段，每次只生成一个 token，每个 token 需要执行数十到数百个小 kernel（RMSNorm、MatMul、RoPE、Attention、SwiGLU 等），每个 kernel 的计算量很小且为 memory-bound 操作，CPU 端的启动延迟在总耗时中占比显著。

**CUDA Graph 的核心思想**：将这些重复执行的 kernel 序列"录制"下来，形成一个可重放的执行图，后续执行时只需一次 CPU 调用即可提交整个图中所有 kernel。

### 1.2 实现原理

CUDA Graph 的工作流程分为三个阶段：

```
┌──────────────┐    ┌──────────────────┐    ┌────────────────┐
│   捕获阶段    │    │   实例化阶段      │    │   回放阶段      │
│  (Capture)   │───>│ (Instantiate)    │───>│   (Launch)     │
│              │    │                  │    │                │
│ Stream上录制  │    │ 生成可执行Graph   │    │ 一次调用执行全部 │
│ 全部kernel    │    │ GPU端预优化      │    │ kernel         │
└──────────────┘    └──────────────────┘    └────────────────┘
```

**阶段一：捕获（Capture）**  
在 CUDA Stream 上开启捕获模式后，所有后续提交到该 Stream 的操作（kernel launch、cudaMemcpy 等）不会真正执行，而是被记录到一个 `cudaGraph_t` 图结构中。每个操作成为图中的一个节点，节点之间的依赖关系由 Stream 的提交顺序自动推导。

**阶段二：实例化（Instantiate）**  
将捕获的 `cudaGraph_t` 编译为一个 `cudaGraphExec_t` 可执行实例。在此阶段，CUDA 驱动会对图进行优化，包括：
- 预分配所有中间资源
- 优化节点间的调度
- 消除不必要的同步点
- 预计算 kernel 启动参数

**阶段三：回放（Launch）**  
通过 `cudaGraphLaunch` 一次性提交整个可执行图到 Stream 上执行。GPU 按图中的依赖关系自动调度所有节点，无需 CPU 逐一提交。

### 1.3 涉及的全部 CUDA API 函数详解

以下是 OrinMLLM 中实际使用的所有 CUDA Graph 相关 API：

#### 1.3.1 `cudaStreamBeginCapture`

```c
cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode);
```

**功能**：在指定 CUDA Stream 上开启图捕获模式。开启后，提交到该 Stream 的所有操作不会执行，而是被录制到内部图结构中。

**参数**：
- `stream`：要开始捕获的 CUDA Stream
- `mode`：捕获模式
  - `cudaStreamCaptureModeGlobal`：全局模式，所有线程的 CUDA 调用都受捕获影响
  - `cudaStreamCaptureModeThreadLocal`：线程本地模式
  - `cudaStreamCaptureModeRelaxed`：**宽松模式（本项目使用）**，允许不在同一 Stream 上的操作不受影响

**在本项目中的使用**（`cuda_graph.h` `begin_capture` 方法）：
```cpp
cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
```
使用 `Relaxed` 模式是因为只需要捕获 decode 计算路径上的 kernel，不影响其他 Stream 上可能存在的操作。

**注意事项**：
- 捕获期间不能调用 `cudaStreamSynchronize`、`cudaDeviceSynchronize`
- 捕获期间不能进行 CPU→GPU 方向的同步内存拷贝（`cudaMemcpy`），但可以使用异步拷贝（`cudaMemcpyAsync`）
- 如果捕获期间发生错误，整个捕获会失败
- **这是本项目遇到的核心难点之一**（见第3部分分析）

#### 1.3.2 `cudaStreamEndCapture`

```c
cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph);
```

**功能**：结束 Stream 上的图捕获，生成 `cudaGraph_t` 图定义对象。

**参数**：
- `stream`：要结束捕获的 CUDA Stream
- `pGraph`：输出参数，接收生成的图对象

**在本项目中的使用**（`cuda_graph.h` `end_capture` 方法）：
```cpp
cudaError_t err = cudaStreamEndCapture(stream, &graph_);
```

**注意事项**：
- 如果捕获期间有错误，`pGraph` 会是 `nullptr`
- 结束捕获后，Stream 恢复正常执行模式

#### 1.3.3 `cudaGraphInstantiate`

```c
cudaError_t cudaGraphInstantiate(cudaGraphExec_t* pGraphExec, cudaGraph_t graph,
                                  cudaGraphNode_t* pErrorNode, char* pLogBuffer, size_t bufferSize);
```

**功能**：将 `cudaGraph_t` 图定义编译为 `cudaGraphExec_t` 可执行实例。这是"编译"阶段，包括资源预分配、调度优化等。

**参数**：
- `pGraphExec`：输出参数，可执行图实例
- `graph`：输入的图定义
- `pErrorNode`：出错节点（可以传 `nullptr`）
- `pLogBuffer`：错误日志缓冲区（可以传 `nullptr`）
- `bufferSize`：日志缓冲区大小

**在本项目中的使用**（`cuda_graph.h` `end_capture` 方法）：
```cpp
err = cudaGraphInstantiate(&instance_, graph_, nullptr, nullptr, 0);
```

**注意事项**：
- 实例化是一个相对开销较大的操作（通常数百微秒到数毫秒），因此应尽量少做
- 一个 `cudaGraph_t` 可以被多次 `Instantiate` 生成不同的可执行实例

#### 1.3.4 `cudaGraphLaunch`

```c
cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream);
```

**功能**：在指定 CUDA Stream 上执行一个预先编译好的可执行图。这是 CUDA Graph 的核心加速函数——一次 API 调用替代了数十到数百次独立的 kernel launch。

**参数**：
- `graphExec`：要执行的可执行图实例
- `stream`：执行图的 CUDA Stream

**在本项目中的使用**（`cuda_graph.h` `launch` 方法）：
```cpp
cudaError_t err = cudaGraphLaunch(instance_, stream);
```

**注意事项**：
- 图中的所有操作地址（输入/输出指针）必须与捕获时一致
- 图中所有 kernel 的 grid/block 配置与捕获时一致
- 这也是为什么需要使用"固定地址 buffer"的原因

#### 1.3.5 `cudaGraphExecUpdate`

```c
cudaError_t cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph,
                                 cudaGraphExecUpdateResultInfo* resultInfo);
```

**功能**：尝试用新的图定义更新已有的可执行图实例，避免完全重新实例化。当图结构未变但某些参数变化时，这比销毁+重新实例化更高效。

**参数**：
- `hGraphExec`：要更新的可执行图实例
- `hGraph`：新的图定义
- `resultInfo`：更新结果信息

**在本项目中的使用**（`cuda_graph.h` `try_update` 方法）：
```cpp
cudaGraphExecUpdateResultInfo result_info;
cudaError_t err = cudaGraphExecUpdate(instance_, graph_, &result_info);
```

#### 1.3.6 `cudaGraphDestroy`

```c
cudaError_t cudaGraphDestroy(cudaGraph_t graph);
```

**功能**：销毁 `cudaGraph_t` 图定义对象，释放关联资源。

**在本项目中的使用**（`cuda_graph.h` `destroy` 和 `begin_capture` 方法）：
```cpp
cudaGraphDestroy(graph_);
graph_ = nullptr;
```

#### 1.3.7 `cudaGraphExecDestroy`

```c
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec);
```

**功能**：销毁 `cudaGraphExec_t` 可执行图实例，释放所有预分配的资源。

**在本项目中的使用**（`cuda_graph.h` `destroy` 方法）：
```cpp
cudaGraphExecDestroy(instance_);
instance_ = nullptr;
```

#### 1.3.8 辅助 API

除了上述核心 CUDA Graph API，项目中与 CUDA Graph 配合使用的还有：

- **`cudaStreamCreate`**：创建用于捕获和执行的 CUDA Stream
- **`cudaStreamSynchronize`**：在捕获前后进行同步（**不能在捕获期间调用**）
- **`cudaMemcpyAsync`**：异步内存拷贝（在捕获期间可以使用，且会被录入图中）
- **`cudaMemsetAsync`**：异步内存清零（清除 KV Cache 时使用）
- **`cudaGetLastError`**：清除错误状态（在 `try_update` 失败后使用）

---

## 2. qwen3.cpp 中 CUDA Graph 的使用方式

### 2.1 架构总览

OrinMLLM 中的 CUDA Graph 实现采用了 **三层封装架构**：

```
┌─────────────────────────────────────────────────────────────┐
│                    Qwen3Model::decode()                     │
│               （应用层：控制捕获/回放流程）                    │
├─────────────────────────────────────────────────────────────┤
│                  CudaConfig + CudaGraphContext              │
│            （配置层：管理 Graph 状态与生命周期）               │
├─────────────────────────────────────────────────────────────┤
│                      CudaGraph 类                           │
│          （底层封装：包装 CUDA Graph C API）                  │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 底层封装：`CudaGraph` 类

文件：`kuiper/include/base/cuda_graph.h`

`CudaGraph` 类封装了 CUDA Graph 的完整生命周期管理：

```cpp
class CudaGraph {
 private:
  cudaGraph_t graph_ = nullptr;        // 图定义
  cudaGraphExec_t instance_ = nullptr; // 可执行图实例
  bool is_capturing_ = false;          // 是否正在捕获
  bool is_valid_ = false;              // 图是否有效
  bool disabled_ = false;              // 是否因连续失败自动禁用
  int capture_count_ = 0;             // 捕获次数
  int consecutive_update_failures_ = 0; // 连续失败次数
  
  static constexpr int kMaxConsecutiveFailures = 3; // 最大连续失败阈值
};
```

**关键方法说明**：

| 方法 | 功能描述 |
|------|----------|
| `begin_capture(stream)` | 销毁旧 graph/instance，调用 `cudaStreamBeginCapture` |
| `end_capture(stream)` | 调用 `cudaStreamEndCapture` + `cudaGraphInstantiate` |
| `launch(stream)` | 调用 `cudaGraphLaunch` 回放 |
| `try_update(stream)` | 尝试 `cudaGraphExecUpdate`，失败则重新 Instantiate |
| `is_valid()` | `is_valid_ && !disabled_` |
| `is_disabled()` | 连续失败 ≥ 3 次后自动禁用 |
| `reset()` | 销毁所有资源，但不重置 disabled 状态 |
| `force_enable()` | 强制重新启用（重置 disabled 和失败计数）|

**自动禁用机制**：当 `end_capture` 或 `try_update` 连续失败 ≥ 3 次时，`disabled_` 设为 `true`，后续 `begin_capture` 不再尝试捕获，直接返回 `false`，从而 fallback 到普通执行路径。

### 2.3 配置层：`CudaConfig` 与 `CudaGraphContext`

**`CudaGraphContext`**（`cuda_graph.h`）：

```cpp
struct CudaGraphContext {
  std::unique_ptr<CudaGraph> decode_graph;  // 唯一的 decode 阶段 Graph
  GraphNodeProperties last_properties;       // 上次捕获时的属性（用于检测变化）
  bool needs_recapture = true;               // 是否需要重新捕获
  int graph_launches = 0;                    // 累计回放次数
  int graph_recaptures = 0;                  // 累计重捕获次数
};
```

`CudaGraphContext` 的 `invalidate()` 方法同时设置 `needs_recapture = true` 并调用 `decode_graph->reset()` 销毁已有图，确保下次 `is_valid()` 返回 `false`，触发重新捕获。

**`CudaConfig`**（`cuda_config.h`）提供了控制接口：

```cpp
bool should_use_graph() {
  return use_cuda_graph && graph_context && !graph_context->decode_graph->is_disabled();
}

void invalidate_graph() {
  if (graph_context) graph_context->invalidate();
}
```

### 2.4 应用层：`Qwen3Model::decode()` 中的 CUDA Graph 流程

这是 CUDA Graph 的核心使用入口，位于 `qwen3.cpp` 第 1868 行起。以下是完整的执行流程分析：

```cpp
base::Status Qwen3Model::decode(const tensor::Tensor& input, int32_t pos, int& next) const {
  // Step 1: 判断是否启用 CUDA Graph
  bool use_graph = cuda_config_ && cuda_config_->should_use_graph();
  
  if (use_graph) {
    // Step 2: 获取固定地址 buffer
    tensor::Tensor pos_tensor_gpu = get_buffer(ModelBufferType::kInputPosGPU);
    tensor::Tensor decode_input   = get_buffer(ModelBufferType::kDecodeInput);
    tensor::Tensor pos_pinned     = get_buffer(ModelBufferType::kInputPosPinned);
    tensor::Tensor argmax_output  = get_buffer(ModelBufferType::kArgmaxOutput);
    tensor::Tensor argmax_pinned  = get_buffer(ModelBufferType::kArgmaxOutputPinned);
    
    auto& graph_ctx = cuda_config_->graph_context;
    auto& graph = graph_ctx->decode_graph;
    
    // Step 3: 检测是否需要重新捕获
    bool need_capture = !graph->is_valid();
    
    // Step 4: 在 Graph 外部更新变化的参数（input 和 position）
    // 4a. 将 input 拷贝到固定地址 decode_input
    cudaMemcpyAsync(decode_input.ptr(), input.ptr(),
                    config_->dim_ * elem_size, cudaMemcpyDeviceToDevice, stream);
    
    // 4b. 通过 pinned memory 更新 GPU 上的 position 值
    *pos_pinned.ptr<int32_t>() = pos;
    cudaMemcpyAsync(pos_tensor_gpu.ptr<int32_t>(), pos_pinned.ptr<int32_t>(),
                    sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    
    // Step 5: 首次执行或 invalidate 后，捕获 Graph
    if (need_capture && !graph->is_disabled()) {
      cudaStreamSynchronize(stream);  // 捕获前必须同步
      
      if (graph->begin_capture(stream)) {
        // ===== 捕获全部 Transformer 层 =====
        for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
          attention_rms(layer_idx, decode_input);          // RMSNorm
          attention_qkv_with_graph(layer_idx, pos_tensor_gpu);  // QKV + RoPE (GPU pos 版)
          attention_mha_with_graph(layer_idx, pos_tensor_gpu);  // MHA (GPU pos 版)
          feed_forward_fused(layer_idx, decode_input);     // FFN (SwiGLU)
        }
        cls_logits(decode_input);  // 最终 RMSNorm + 分类头
        
        if (graph->end_capture(stream)) {
          graph_ctx->graph_recaptures++;
        }
      }
    }
    
    // Step 6: 回放 Graph
    if (graph->is_valid()) {
      if (graph->launch(stream)) {
        graph_ctx->graph_launches++;
        
        // Step 7: Argmax 采样（在 Graph 外执行）
        // ...
        cudaStreamSynchronize(stream);
        next = ...;
        return base::error::Success();
      }
      graph_ctx->invalidate();  // launch 失败，fallback
    }
  }
  
  // ===== 普通执行路径（fallback）=====
  // 使用标准 attention_qkv / attention_mha（CPU position 版本）
  for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
    attention_rms(layer_idx, input);
    attention_qkv(layer_idx, pos_tensor);
    attention_mha(layer_idx, pos_tensor);
    feed_forward(layer_idx, input);
  }
  cls_logits(input);
  cudaStreamSynchronize(stream);
  next = post_processing(pos_tensor, false);
}
```

### 2.5 Graph 兼容版算子：`attention_qkv_with_graph` vs `attention_qkv`

这是理解 CUDA Graph 改造的关键。对比两个版本：

**普通版本** `attention_qkv`（第 1039 行）：

```cpp
void Qwen3Model::attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  tensor::Tensor query = get_buffer(ModelBufferType::kQuery);
  int32_t pos = pos_tensor.index<int32_t>(0);  // ❌ 从CPU读取position
  auto [key, val] = slice_kv_cache(layer_idx, pos);  // ❌ 每次计算不同的 slice 地址
  
  // ... QKV 计算 ...
  
  // RoPE：使用 CPU position
  qwen_layers_->rope_layer_->forward(query, key, pos_tensor, sin_cache, cos_cache, ...);
}
```

**Graph 兼容版本** `attention_qkv_with_graph`（第 1228 行）：

```cpp
void Qwen3Model::attention_qkv_with_graph(int32_t layer_idx,
                                           const tensor::Tensor& pos_tensor) const {
  tensor::Tensor query = get_buffer(ModelBufferType::kQuery);
  tensor::Tensor temp_key = get_buffer(ModelBufferType::kTempKey);      // ✅ 固定地址
  tensor::Tensor temp_value = get_buffer(ModelBufferType::kTempValue);  // ✅ 固定地址
  
  // ... QKV 计算 ...
  
  // RoPE：使用 GPU position layer
  auto rope_layer = qwen_layers_->rope_gpu_pos_layer_;
  rope_layer->set_use_gpu_pos(true);  // ✅ 从GPU内存读取position
  rope_layer->set_input(2, pos_tensor);  // pos_tensor 是 GPU 上的 tensor
  STATUS_CHECK(rope_layer->forward());
  
  // KV Cache 拷贝：通过 GPU position layer
  auto key_cache_layer = qwen_layers_->kv_cache_key_layer_;
  key_cache_layer->set_use_gpu_pos(true);  // ✅ 从GPU内存读取position
  key_cache_layer->set_input(2, pos_tensor);
  STATUS_CHECK(key_cache_layer->forward());
  
  auto value_cache_layer = qwen_layers_->kv_cache_value_layer_;
  value_cache_layer->set_use_gpu_pos(true);  // ✅ 从GPU内存读取position
  value_cache_layer->set_input(2, pos_tensor);
  STATUS_CHECK(value_cache_layer->forward());
}
```

**核心差异总结**：

| 方面 | 普通版本 | Graph 兼容版本 |
|------|----------|----------------|
| Position 来源 | CPU 内存 (`pos_tensor.index<int32_t>(0)`) | GPU 内存 (`pos_tensor_gpu`) |
| KV 输出目标 | `slice_kv_cache` 动态计算地址 | 固定地址 `kTempKey/kTempValue` |
| KV Cache 写入 | 直接写到 slice 地址 | 通过 `kv_cache_key_layer_` 内部根据 GPU pos 写入 |
| RoPE 层 | `rope_layer_`（标准版） | `rope_gpu_pos_layer_`（GPU pos版） |
| 可被 Graph 捕获 | ❌ 不行，地址和参数每步变化 | ✅ 可以，所有地址固定 |

### 2.6 Graph 失效与重捕获

在以下场景中，CUDA Graph 会被标记为失效（invalidate），下次 decode 时自动重新捕获：

**场景一：清空 KV Cache**（多轮对话切换时）

```cpp
void Qwen3Model::clear_kv_cache() {
  // ... 清空 KV Cache ...
  
  // 重要：使 CUDA Graph 失效，因为 KV cache 状态已改变
  // CUDA Graph 捕获的是特定的 KV cache 状态，清空后需要重新捕获
  invalidate_cuda_graph();
}
```

**场景二：Graph launch 失败**

```cpp
if (graph->is_valid()) {
  if (graph->launch(stream)) {
    // 成功
    return Success;
  }
  graph_ctx->invalidate();  // launch 失败，invalidate 并 fallback
}
```

**场景三：结构性属性变化**

`CudaGraphContext` 通过 `GraphNodeProperties` 检测输入/输出指针是否变化：

```cpp
struct GraphNodeProperties {
  void* input_ptr;
  void* output_ptr;
  
  bool operator==(const GraphNodeProperties& other) const {
    return input_ptr == other.input_ptr && output_ptr == other.output_ptr;
  }
};
```

---

## 3. CUDA Graph 在 qwen3.cpp 中的难点与解决方案

### 3.1 难点一：Position 参数在每步 decode 中变化

**问题描述**：

在 LLM 的 decode 阶段，每生成一个新 token，`position` 值就会递增。Position 用于 RoPE（旋转位置编码）计算和 KV Cache 的写入位置。在普通实现中，position 是一个 CPU 标量值，直接传给 kernel：

```cpp
// 普通路径：position 从 CPU 读取
int32_t pos = pos_tensor.index<int32_t>(0);  // CPU 内存读取
auto [key, val] = slice_kv_cache(layer_idx, pos);  // CPU 计算地址
```

**为什么这对 CUDA Graph 是致命问题**：

1. **CUDA Graph 捕获的是 kernel 的参数快照**：在捕获时，所有 kernel 的参数（包括标量值和指针）都被固化。如果 position 是 CPU 标量传入，那么每次 replay 都会使用捕获时的 position 值，导致结果错误。

2. **`slice_kv_cache` 产生变化的地址**：普通版本根据 position 计算 KV Cache 的写入偏移，每步都不同。CUDA Graph 要求所有内存地址在捕获和回放时保持一致。

3. **CPU→GPU 同步拷贝破坏捕获**：如果在捕获期间使用 `cudaMemcpy`（同步版本）更新 position，会导致捕获失败。

**解决方案：GPU Position Layer**

项目设计了专用的 "GPU Position Layer" 系列算子，将 position 参数从 CPU 标量改为 **GPU 设备内存中的指针**：

```
普通路径:  CPU int32 pos ──→ kernel(pos)
                              ↑ 每次值不同，Graph 无法使用

Graph路径: GPU int32* pos_ptr ──→ kernel(*pos_ptr)
                                   ↑ 指针地址固定，值通过异步拷贝更新
```

具体实现：

**`RoPEGpuPosLayer`**（`kuiper/include/op/batched_rope.h`）：

```cpp
class RoPEGpuPosLayer : public LayerParam {
  bool use_gpu_pos_ = false;  // 是否使用 GPU 上的 position
  
  // forward 时，如果 use_gpu_pos_=true，
  // 则从 input[2]（GPU tensor）中读取 position，
  // 而不是从 CPU 标量中读取
};
```

**`MHAGpuPosLayer`**（`kuiper/include/op/misc_layers.h`）：

```cpp
class MHAGpuPosLayer {
  // forward 接受 const int32_t* pos_ptr（GPU 指针）
  // 内部调用 mha_kernel_cu_gpu_pos()
  // kernel 内部通过 *pos_ptr 读取 position 值
};
```

**Position 更新策略**：

```cpp
// 在 Graph 外部（捕获之前/回放之前），通过 pinned memory 异步更新 GPU position
*pos_pinned.ptr<int32_t>() = pos;  // CPU 写入 pinned memory
cudaMemcpyAsync(pos_tensor_gpu.ptr<int32_t>(),  // 异步拷贝到 GPU
                pos_pinned.ptr<int32_t>(),
                sizeof(int32_t), cudaMemcpyHostToDevice, stream);
```

使用 **pinned memory**（页锁定内存）是关键优化 — 它保证 `cudaMemcpyAsync` 是真正异步的，不会阻塞 CPU，且传输效率最高。

### 3.2 难点二：KV Cache 地址每步变化

**问题描述**：

在普通 decode 路径中，每步生成的 Key/Value 需要写入 KV Cache 的不同位置。普通实现通过 `slice_kv_cache(layer_idx, pos)` 计算目标地址，返回的指针地址每步不同：

```cpp
// 普通路径
auto [key, val] = slice_kv_cache(layer_idx, pos);
// key 和 val 的地址 = kv_cache_base + layer_offset + pos * kv_dim
// 每步 pos 不同 → 地址不同 → CUDA Graph 无法使用
```

**解决方案：固定缓冲区 + GPU Position 感知的 KV Cache 层**

```
普通路径:  QKV → key/val 直接写入 cache[pos]（变化地址）

Graph路径: QKV → key/val 写入 kTempKey/kTempValue（固定地址）
               → kv_cache_layer 读取 GPU pos，内部 copy 到 cache[*pos]
```

```cpp
// Graph 路径：先写到固定临时缓冲区
STATUS_CHECK(key_layer->forward(rmsnorm_output, temp_key));  // 固定地址

// 然后通过 KV Cache Layer 写入正确位置（内部根据 GPU pos 计算偏移）
auto key_cache_layer = qwen_layers_->kv_cache_key_layer_;
key_cache_layer->set_use_gpu_pos(true);
key_cache_layer->set_input(0, temp_key);
key_cache_layer->set_input(1, key_cache);
key_cache_layer->set_input(2, pos_tensor);  // GPU position
STATUS_CHECK(key_cache_layer->forward());
```

### 3.3 难点三：cuBLAS 操作的 Graph 兼容性

**问题描述**：

CUDA Graph 从 CUDA 10.0 开始支持，但 cuBLAS 等库函数的 Graph 兼容性经历了多个版本的迭代。早期版本的 cuBLAS 在 Stream Capture 模式下可能产生意外行为，比如内部使用 workspace 的方式可能与 Graph 不兼容。

**解决方案**：

1. cuBLAS handle 与 inference stream 绑定：`cublasSetStream(handle, stream)`。这确保 cuBLAS 操作被提交到同一个 Stream，可以被正确捕获。
2. 在 Orin 平台上使用的 CUDA 11.x+ 版本，cuBLAS 已具备良好的 Graph 兼容性。
3. 使用 `cudaStreamCaptureModeRelaxed` 避免 cuBLAS 内部的跨 Stream 操作干扰捕获。

### 3.4 难点四：Graph 捕获与异常处理

**问题描述**：

如果在 Graph 捕获期间发生任何错误（如 kernel 参数非法、内存不足），整个捕获会失败。在生产环境中需要优雅地处理这种情况。

**解决方案：自动禁用 + Fallback 机制**

```cpp
// CudaGraph 类中的自动禁用机制
static constexpr int kMaxConsecutiveFailures = 3;

bool end_capture(cudaStream_t stream) {
  // ...
  if (err != cudaSuccess) {
    consecutive_update_failures_++;
    if (consecutive_update_failures_ >= kMaxConsecutiveFailures) {
      disabled_ = true;  // 连续3次失败后自动禁用
    }
    return false;
  }
  consecutive_update_failures_ = 0;  // 成功后重置计数器
  // ...
}
```

在 `decode()` 中的 fallback 逻辑：

```cpp
if (use_graph) {
  // 尝试 Graph 路径...
  if (graph->is_valid()) {
    if (graph->launch(stream)) {
      return Success;  // ✅ Graph 成功
    }
    graph_ctx->invalidate();  // launch 失败
  }
  // Graph 路径未成功，fallthrough 到普通路径
}

// ===== 普通执行路径（始终可用的 fallback）=====
for (layer_idx = 0; layer_idx < layer_num; ++layer_idx) {
  attention_rms(layer_idx, input);
  attention_qkv(layer_idx, pos_tensor);     // CPU position 版本
  attention_mha(layer_idx, pos_tensor);     // CPU position 版本
  feed_forward(layer_idx, input);
}
```

### 3.5 难点五：仅 Decode 可用，Prefill 不适用

**问题描述**：

CUDA Graph 要求捕获和回放时的 **操作拓扑结构完全一致**（kernel 数量、grid/block 配置、内存地址等）。Prefill 阶段的序列长度每次不同（取决于输入 prompt 长度），导致：

1. MatMul 的维度不同 → kernel 的 grid 配置不同
2. Flash Attention 处理的 sequence length 不同
3. 输入/输出 tensor 的尺寸不同
4. Prefill 需要动态分配临时缓冲区

因此 CUDA Graph **仅用于 decode 阶段**（batch_size=1，每次固定处理 1 个 token），不用于 prefill 阶段。

**项目中的实现**：

```cpp
// prefill 不使用 CUDA Graph，使用动态分配的缓冲区
base::Status Qwen3Model::prefill(const tensor::Tensor& input, int32_t seq_len, ...) {
  // 动态分配（与 seq_len 相关的 buffer）
  tensor::Tensor hidden_buf0(activation_dtype, seq_len, dim, true, alloc);
  tensor::Tensor rms_out(activation_dtype, seq_len, dim, true, alloc);
  // ...
}

// decode 使用 CUDA Graph，使用预分配的固定缓冲区
base::Status Qwen3Model::decode(const tensor::Tensor& input, int32_t pos, ...) {
  if (use_graph) {
    tensor::Tensor decode_input = get_buffer(ModelBufferType::kDecodeInput);  // 预分配固定地址
    // ...
  }
}
```

### 3.6 难点汇总

| 难点 | 根因 | 解决方案 |
|------|------|----------|
| Position 每步变化 | Graph 固化标量参数 | GPU Position Layer：position 存 GPU 内存，kernel 内部 dereference |
| KV Cache 地址变化 | slice_kv_cache 产生不同指针 | 固定 temp buffer + GPU pos 感知的 KV cache layer |
| cuBLAS 兼容性 | 某些 cuBLAS 操作不支持 capture | 绑定 stream + Relaxed 模式 + 使用 CUDA 11.x+ |
| 捕获失败处理 | 不可预见的 kernel 异常 | 自动禁用 + fallback 到普通路径 |
| Prefill 不适用 | 变长序列改变 graph 拓扑 | 仅 decode 阶段使用 Graph |
| KV Cache 状态改变 | 清缓存后 graph 对应的状态失效 | `invalidate_cuda_graph()` 触发重捕获 |

---

## 4. CUDA Graph 如何带来加速

### 4.1 加速原理

CUDA Graph 的加速来自以下几个维度：

#### 4.1.1 消除 CPU 端 Kernel Launch Overhead

这是最主要的加速来源。在不使用 CUDA Graph 时，每次 Transformer 层的 decode 需要 CPU 逐一提交多个 kernel：

```
普通路径（每个 token 生成需要的 CPU 调用）：

Layer 0:
  ├── cuLaunchKernel(rmsnorm)              // ~5-15μs CPU overhead
  ├── cublasGemm(wq)                       // ~5-15μs
  ├── cuLaunchKernel(query_norm)           // ~5-15μs
  ├── cublasGemm(wk)                       // ~5-15μs
  ├── cuLaunchKernel(key_norm)             // ~5-15μs
  ├── cublasGemm(wv)                       // ~5-15μs
  ├── cuLaunchKernel(rope)                 // ~5-15μs
  ├── cuLaunchKernel(kv_cache_copy × 2)    // ~10-30μs
  ├── cuLaunchKernel(flash_attn/mha)       // ~5-15μs
  ├── cublasGemm(wo)                       // ~5-15μs
  ├── cuLaunchKernel(add)                  // ~5-15μs
  ├── cuLaunchKernel(ffn_rmsnorm)          // ~5-15μs
  ├── cublasGemm(w1)                       // ~5-15μs
  ├── cublasGemm(w3)                       // ~5-15μs
  ├── cuLaunchKernel(swiglu)               // ~5-15μs
  ├── cublasGemm(w2)                       // ~5-15μs
  └── cuLaunchKernel(add)                  // ~5-15μs

共 ~17 个 kernel per layer × 36 layers = ~612 次 kernel launch
+ cls_logits (rmsnorm + gemm) = ~2 次

总计: ~614 次 CPU→GPU 命令提交
每次 CPU overhead: ~5-15μs
总 CPU overhead: ~3-9ms
```

对于 Qwen3-8B（36 层），一次 decode 约需 614 次 kernel launch。在 Orin 平台上，每次 kernel launch 的 CPU overhead 约 5-15μs，总 CPU overhead 约 3-9ms。

而每个 kernel 的实际 GPU 执行时间在 decode 阶段非常短（因为 batch_size=1，单 token 计算量小），往往只有几微秒到几十微秒。**CPU launch overhead 在总时间中占比可达 20%-50%**。

使用 CUDA Graph 后：

```
Graph 路径：

  cudaMemcpyAsync(decode_input, ...)        // 1 次
  cudaMemcpyAsync(pos_gpu, ...)             // 1 次
  cudaGraphLaunch(graph, stream)            // ★ 1 次调用，执行全部 614 个 kernel
  cudaStreamSynchronize(stream)             // 1 次

总计: 4 次 CPU→GPU 命令提交
CPU overhead: ~20-60μs（几乎可忽略）
```

**加速比**：CPU overhead 从 ~3-9ms 降低到 ~20-60μs，理论加速 50-150x（仅 CPU overhead 部分）。对整体 decode latency 的影响取决于 GPU 计算时间占比。

#### 4.1.2 GPU 端调度优化

`cudaGraphInstantiate` 阶段，CUDA 驱动对图进行优化：

1. **消除不必要的同步**：Graph 知道完整的依赖关系，可以尽早启动独立的 kernel
2. **预分配资源**：kernel 所需的寄存器、shared memory 等在实例化时确定
3. **批量提交**：GPU 命令处理器可以预取后续 kernel 的配置，减少 idle 时间

#### 4.1.3 减少 CPU-GPU 交互

普通路径中，CPU 需要频繁与 GPU 通信（kernel launch、读取 position、计算 KV cache 地址等）。Graph 路径将所有这些操作预先录制，仅需在回放前更新 position（通过一次异步 memcpy）。

### 4.2 源码中的加速实现分析

#### 4.2.1 固定地址 Buffer 策略

项目为 CUDA Graph 预分配了多个固定地址的 buffer，确保 Graph 捕获和回放时内存地址一致：

```cpp
// 在 Qwen3Model::init() 或 create_nonparam_layers() 中预分配
ModelBufferType::kDecodeInput      // decode 阶段的固定输入缓冲区
ModelBufferType::kInputPosGPU      // GPU 上的 position 值（固定地址，值通过 memcpy 更新）
ModelBufferType::kInputPosPinned   // pinned memory，用于高效 H2D 传输
ModelBufferType::kTempKey           // 临时 key 缓冲区（固定地址，代替 slice_kv_cache）
ModelBufferType::kTempValue         // 临时 value 缓冲区（固定地址）
ModelBufferType::kArgmaxOutput     // GPU 上的 argmax 结果
ModelBufferType::kArgmaxOutputPinned // 采样结果的 pinned memory
```

#### 4.2.2 采样优化

在 Graph 路径中，argmax 采样也做了优化以减少同步：

```cpp
if (graph->is_valid()) {
  graph->launch(stream);
  
  auto* argmax_sampler = dynamic_cast<sampler::ArgmaxSampler*>(sampler_.get());
  if (argmax_sampler) {
    // 使用预分配的 GPU/pinned buffer 进行采样
    argmax_sampler->sample_prealloc(
        forward_output.ptr<float>(), forward_output.size(),
        argmax_output.ptr(),        // GPU output（固定地址）
        argmax_pinned.ptr(),        // pinned memory（固定地址）
        stream);
    cudaStreamSynchronize(stream);
    next = *argmax_pinned.ptr<int32_t>();  // 直接从 pinned memory 读取结果
  }
}
```

使用 `sample_prealloc` 避免了 `cudaMalloc`/`cudaFree` 开销，且通过 pinned memory 实现高效的 GPU→CPU 结果传输。

#### 4.2.3 Pinned Memory 优化

项目在 position 更新和采样结果读取中大量使用 pinned memory：

```cpp
// Position 更新：CPU → pinned → GPU（全异步）
*pos_pinned.ptr<int32_t>() = pos;        // CPU 写 pinned（立即完成）
cudaMemcpyAsync(pos_gpu, pos_pinned,     // pinned → GPU（DMA，不阻塞CPU）
                sizeof(int32_t), cudaMemcpyHostToDevice, stream);

// 结果读取：GPU → pinned → CPU
// argmax 结果写入 GPU buffer → DMA 到 pinned → CPU 读取
```

pinned memory 保证：
1. `cudaMemcpyAsync` 是真正的异步操作（非 pinned 内存会退化为同步）
2. H2D/D2H 传输通过 DMA 引擎完成，不占用 GPU 计算单元
3. CPU 可以在 GPU 执行 Graph 的同时进行其他工作

### 4.3 加速效果总结

在 NVIDIA Orin 平台上，CUDA Graph 对 Qwen3-8B decode 阶段的加速效果：

```
┌─────────────────────────────────────────────────────────────┐
│              Decode Latency Breakdown (per token)           │
├─────────────────────────┬─────────────┬─────────────────────┤
│        Component        │ Without Graph│ With Graph         │
├─────────────────────────┼─────────────┼─────────────────────┤
│ CPU Launch Overhead     │ ~3-9ms      │ ~0.02-0.06ms       │
│ GPU Compute             │ ~X ms       │ ~X ms (不变)        │
│ Position Update (H2D)   │ ~0.01ms     │ ~0.01ms (异步)     │
│ Result Readback (D2H)   │ ~0.01ms     │ ~0.01ms (pinned)   │
│ synchronize             │ ~0.01ms     │ ~0.01ms            │
├─────────────────────────┼─────────────┼─────────────────────┤
│ Total Overhead          │ ~3-9ms      │ ~0.05ms            │
│ Overhead 占比           │ 20-50%      │ <1%                │
└─────────────────────────┴─────────────┴─────────────────────┘
```

**关键结论**：
1. CUDA Graph 主要加速 CPU 端开销，GPU 计算时间不变
2. 模型越大（层数越多），kernel launch 越多，CUDA Graph 的相对收益越高
3. 在 Orin 等嵌入式 GPU 平台上，CPU 性能相对 x86 更弱，kernel launch overhead 占比更大，因此 CUDA Graph 的加速效果更明显
4. 首次捕获有额外开销（~数毫秒），但后续每个 token 的 decode 都能受益
5. Fallback 机制确保了鲁棒性，即使 Graph 不可用也不影响推理正确性

---

## 附录：CUDA Graph 生命周期状态机

```
                    ┌──────────────┐
         创建 ──→   │   INVALID    │  ←── invalidate() / reset()
                    │ is_valid=F   │
                    └──────┬───────┘
                           │ begin_capture()
                           ↓
                    ┌──────────────┐
                    │  CAPTURING   │
                    │ is_capturing │
                    └──────┬───────┘
                           │ end_capture()
                  成功 ────┤──── 失败 ──→ consecutive_failures++
                  ↓        │             ↓  (≥3次)
           ┌──────────────┐│      ┌─────────────┐
           │    VALID     ││      │  DISABLED    │
           │ is_valid=T   ││      │ disabled=T   │
           └──────┬───────┘│      └──────────────┘
                  │        │             ↑
                  │ launch() 失败 ─────→│
                  │                      │
                  ↓                      │
           [ 成功回放 ] ──→ (持续回放)   force_enable() ──→ INVALID
```

---

## 附录：项目完整调用链

```
用户命令行传入 --cuda-graph
  └→ inference_common.h: model.enable_cuda_graph(true)
      └→ Qwen3Model::enable_cuda_graph(bool)
          └→ cuda_config_->use_cuda_graph = true
          └→ cuda_config_->graph_context = make_shared<CudaGraphContext>()
              └→ CudaGraphContext() { decode_graph = make_unique<CudaGraph>(); }

推理循环（每个 token）:
  └→ Qwen3Model::decode(input, pos, next)
      ├→ cuda_config_->should_use_graph()  // 检查是否启用
      ├→ cudaMemcpyAsync(decode_input, input)  // 拷贝到固定地址
      ├→ cudaMemcpyAsync(pos_gpu, pos_pinned)  // 更新 GPU position
      ├→ [首次] graph->begin_capture(stream)
      │   ├→ cudaStreamBeginCapture(stream, Relaxed)
      │   └→ 录制全部 Transformer 层:
      │       ├→ attention_rms()              // 复用普通版
      │       ├→ attention_qkv_with_graph()   // GPU pos 版
      │       ├→ attention_mha_with_graph()   // GPU pos 版
      │       └→ feed_forward_fused()         // 复用普通版
      │   └→ graph->end_capture(stream)
      │       ├→ cudaStreamEndCapture → cudaGraph_t
      │       └→ cudaGraphInstantiate → cudaGraphExec_t
      └→ [后续] graph->launch(stream)
          └→ cudaGraphLaunch(instance, stream)  // 一次调用执行全部
          └→ argmax_sampler->sample_prealloc()     // 采样
          └→ cudaStreamSynchronize(stream)         // 同步
          └→ 读取 next token from pinned memory

多轮对话切换:
  └→ Qwen3Model::clear_kv_cache()
      └→ invalidate_cuda_graph()
          └→ cuda_config_->invalidate_graph()
              └→ graph_context->invalidate()
                  ├→ needs_recapture = true
                  └→ decode_graph->reset()
                      └→ cudaGraphExecDestroy + cudaGraphDestroy
                      └→ is_valid_ = false  ──→ 下次 decode 自动重捕获
```
