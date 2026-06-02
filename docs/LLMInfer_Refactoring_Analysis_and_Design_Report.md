# OrinMLLM `kuiper` → `LLMInfer` 重构分析与设计报告
# Refactoring Analysis & Design Report: `kuiper` → `LLMInfer`

> 双语报告 / Bilingual report. 中文在前，English follows each section.
>
> 适用范围 / Scope：本报告覆盖用户提出的 8 个重构步骤，记录对原 `kuiper` 工程
> `allocator / buffer / tensor / op` 等模块的分析、不足点，以及 `LLMInfer` 中
> 各模块（memory / tensor / task / scheduler / stream / ops / kernel / models /
> inference_optimization / pipeline）的重构设计与落地状态。
>
> 目标硬件 / Target：NVIDIA Jetson **Orin (SM87)** 为主，JIT 兼容
> **SM86 / SM87 / SM89 / SM90 / SM120**。

---

## 0. 总体架构 / Overall Architecture

```
                ┌───────────────────────────────────────────────┐
   service  ──▶ │                  pipeline                      │  launcher: 按需选择模型
                │   (qwen2 / qwen3 / qwen3_vl / qwen3_5 ...)      │  load → run → stream out
                └───────────────────────┬───────────────────────┘
                                        │ 调用
                ┌───────────────────────▼───────────────────────┐
                │                    models                      │  算子串联 = DAG 构图
                │   build Graph(embed→layers→norm→lm_head→samp)  │
                └───────┬───────────────────────────────┬───────┘
            构图依赖     │                               │  系统级优化
        ┌───────────────▼──────────┐        ┌───────────▼─────────────────┐
        │  task + scheduler (DAG)  │        │   inference_optimization     │
        │  Task/Edge/Graph/Sched   │        │ kv_cache/prefix_cache/radix  │
        │  Sequential/Parallel/Pipe│        │ cuda_graph                   │
        └───────────────┬──────────┘        └───────────┬─────────────────┘
                        │ run() 调用算子                  │ 复用 KV / 固定地址
        ┌───────────────▼───────────────────────────────▼─────────┐
        │                          ops                             │  统一算子接口
        │   cpu/  +  cuda/(多架构核函数)                            │  Dispatcher
        └───────────────┬───────────────────────────────┬─────────┘
            AOT / JIT    │                               │ 句柄
        ┌───────────────▼──────────┐        ┌───────────▼─────────────┐
        │         kernel           │        │         stream          │  屏蔽平台
        │  JIT(多架构 -gencode)     │        │  Stream/Event/Manager   │  CUDA stream
        └───────────────┬──────────┘        └───────────┬─────────────┘
        ┌───────────────▼───────────────────────────────▼─────────┐
        │                    tensor  +  memory                     │  数据 + 内存
        │  Tensor / Buffer / Allocator(CPU/Pinned/CUDA/ZeroCopy)   │
        └──────────────────────────────────────────────────────────┘
                            common (Status / 基础类型枚举 / NoCopyable)
```

**设计原则 / Principles**
- 分层解耦：`common → memory → tensor → stream/kernel → ops → task/scheduler →
  models → inference_optimization → pipeline → service`，下层不依赖上层。
- 接口统一：算子层、推理引擎层（TensorRT/ONNXRuntime/MNN）对外接口一致。
- 端侧友好：内存池化、Pinned/统一内存、CUDA Graph、KV/Prefix Cache、多流重叠。

> **EN —** Strict layering bottom-up; unified operator & inference-engine
> interfaces; edge-oriented optimizations (memory pooling, pinned/unified memory,
> CUDA Graph, KV/Prefix cache, multi-stream overlap).

---

## Step 1 — `memory` 模块（allocator / buffer）/ Memory module

### 1.1 原 kuiper 实现分析 / Analysis of original kuiper

源文件：`kuiper/include/base/{alloc.h,buffer.h}`、`source/base/{alloc.cpp,
alloc_cpu.cpp,alloc_cu.cpp,buffer.cpp}`。

**优点 / Strengths**
- `CUDADeviceAllocator` 实现了**大小分级的内存池**（`>1MB` 大块、`<=1MB` 小块），
  通过 `busy` 标记复用，避免频繁 `cudaMalloc/cudaFree`——这是端侧推理（反复
  分配 KV/激活 buffer）的关键优化，予以保留。
- `CPUPinnedAllocator` 使用 `cudaMallocHost`，支持真正异步 H2D/D2H。
- 工厂单例保证内存池全局唯一。

**不足 / Weaknesses**（重构针对性改进）
1. **线程不安全**：`big_buffers_map_` / `cuda_buffers_map_` 在多流/多线程
   调度（本工程引入 DAG 并行）下存在数据竞争。
2. **缺少内存类型维度**：`Buffer` 只记录 `DeviceType`，无法区分
   普通 CUDA / Pinned / 统一(零拷贝)内存，上层难以为 Orin 统一内存做零拷贝决策。
3. **无统一内存 / 零拷贝**：Orin 上 host/device 共享物理 DRAM，缺 `cudaMallocManaged`
   路径会产生不必要的显式拷贝。
4. **`memcpy` 方向冗余推断**：`buffer.cpp` 中四分支重复，易出错。

### 1.2 LLMInfer 重构设计 / Refactored design

文件：`LLMInfer/memory/{alloc.h,alloc.cpp,alloc_cpu.cpp,alloc_cu.cpp,buffer.h,buffer.cpp}`

- **`DeviceAllocator` 增加 `MemoryType`**：`{kMemoryCPU, kMemoryCPUPinned,
  kMemoryCUDA, kMemoryCUDAZeroCopy}`，与 `DeviceType` 正交。
- **`CUDADeviceAllocator` 加 `std::mutex`**：`allocate()/release()` 全程加锁，
  池在 DAG 并行调度下安全；保留原大小分级池策略。
- **新增 `CUDAZeroCopyAllocator`**：基于 `cudaMallocManaged`，面向 Orin 统一内存，
  消除 H2D/D2H 拷贝。
- **`Buffer` 记录并透传 `MemoryType`**：`set/get_memory_type()`，并用单一
  `PickMemcpyKind()` 推断方向，消除重复分支。
- **工厂统一为 `get_instance()`**：含 CPU / Pinned / CUDA / ZeroCopy 四个工厂。

> **EN —** Kept the size-tiered CUDA memory pool (key for edge inference), but
> added thread-safety (mutex), an orthogonal `MemoryType` axis, a new
> unified/zero-copy allocator (`cudaMallocManaged`, ideal for Jetson Orin shared
> DRAM), and consolidated the redundant memcpy-direction logic.

**状态 / Status：✅ 已落地并自洽（headers/.cpp 一致）。**

---

## Step 2 — `tensor` 模块 / Tensor module

### 2.1 原 kuiper 不足 / Weaknesses

- **强依赖 `armadillo` 与 `driver_types.h`**：`tensor.h` `#include <armadillo>`，
  对端侧无价值且拖慢编译、增大体积。
- **设备迁移耦合**：`to_cpu/to_cuda` 内联直接拿工厂，难以走零拷贝路径。
- **缺 `strides()` 的清晰语义**，不利于 view/切片（KV cache slice）。

### 2.2 重构设计 / Design

文件：`LLMInfer/tensor/{tensor.h,tensor.cpp}`

- **去除 armadillo 依赖**，仅保留 `dims/dtype/Buffer` 的逻辑视图。
- **零拷贝友好**：`need_alloc=false + 外部 ptr` 支持权重零拷贝与 CUDA Graph
  固定地址；`assign()` 共享底层 `Buffer`（view，用于 KV cache 切片）。
- **`clone()` 深拷贝、`strides()` 行主序**，便于上层做 reshape/slice。
- 同步修正：原 skeleton 误用 `Factory::create()` → 统一 `get_instance()`。

> **EN —** Removed the heavyweight `armadillo` dependency; the tensor is now a
> pure logical view over `memory::Buffer`. Supports external-pointer zero-copy
> (weights / CUDA-Graph fixed addresses) and buffer-sharing `assign()` for KV
> cache slicing. Fixed stale factory calls to `get_instance()`.

**状态 / Status：✅ 已落地（修正工厂调用）。**

---

## Step 3 — `task` + `scheduler` 模块（DAG）/ Task & Scheduler

参考 / Reference：`CGraph`（`GNode/GElement/UThreadPool`）、`nndeploy`（`Node/Edge`）。

### 3.1 设计 / Design

文件：`LLMInfer/task/{task.h,task.cpp}`、`LLMInfer/scheduler/{scheduler.h,scheduler.cpp}`

- **`Task`（≈ CGraph GNode / nndeploy Node）**：生命周期 `init()→run()×N→deinit()`；
  携带 `inputs/outputs/depends`、`device/stream`、`workspace_bytes()`（供静态
  内存预规划，借鉴 nndeploy `getMemorySize`）。`LambdaTask` 便于算子级小任务。
- **`Edge`**：数据流通道，区分 `kDataFlow`（逐 step 覆盖）与 `kStateful`
  （KV cache，跨 step 持有，不引入执行顺序依赖）。
- **`Graph`**：`add_task/add_edge/connect`，Kahn 拓扑排序（检测环），
  `total_workspace_bytes()` 峰值估计。
- **`Scheduler`** 三种调度策略：
  - `kSequential`：拓扑序串行——prefill / 确定性 decode 最稳。
  - `kParallel`：依赖驱动并行——入度归零即提交线程池，配合多流算子级并行。
  - `kPipeline`：跨 step 重叠——decode 吞吐优化。
  - `run_loop(max_steps, step_cb)`：decode 循环，回调取 logits/采样/判 EOS。
- **`ThreadPool`**（≈ CGraph UThreadPool）：端侧默认 `min(hw, 12)` 线程。

> **EN —** A CGraph/nndeploy-style DAG engine: `Task` (node) + `Edge`
> (data/stateful) + `Graph` (Kahn topo-sort) + `Scheduler` with sequential /
> dependency-driven parallel / pipeline modes and a decode `run_loop`. Stateful
> edges model KV cache without forcing ordering. Static `workspace_bytes`
> enables memory pre-planning.

**状态 / Status：✅ 已落地（headers + .cpp 完整，含并行执行引擎）。**

---

## Step 3.5 — `stream` 模块 / Stream module

文件：`LLMInfer/stream/{stream.h,stream.cpp}`（本次新建）

- **`Event`**：`cudaEvent` 封装，默认关闭计时位降低同步开销；支持跨流
  `wait`/计时。
- **`Stream`**：平台无关异步队列（CUDA 后端），支持高优先级、包装外部 stream、
  `record/wait` 跨流依赖，`raw()` 直接喂给 ops 的 `void* stream`。
- **`StreamManager`（单例）**：`compute`（高优先级）/ `h2d` / `d2h` / `workers[]`
  四类流，实现 **copy/compute 重叠** 与算子级并行；`next_worker()` 轮询派发。

> **EN —** Unified async-queue abstraction over CUDA streams: a high-priority
> compute stream plus dedicated H2D/D2H copy streams (true copy/compute overlap)
> and round-robin worker streams for operator-level parallelism. Events provide
> cross-stream dependencies compatible with CUDA Graph capture.

**状态 / Status：✅ 已落地。**

---

## Step 4 — `ops` 模块 / Operators

### 4.1 原 `kuiper/source/op/kernels` 不足 / Weaknesses

- **`kernels_interface` 用裸函数指针 + 全局 `get_xxx_kernel(device)`**：
  按 `DeviceType` 二选一，无法按 **GPU 架构 / dtype / shape** 选择最优核，
  也无法接入 JIT。
- **`CudaConfig` 聚合 stream/cublas/graph/workspace**：职责过载，且与
  本工程的 `stream` 模块重复。
- **核函数与调度紧耦合**：算子直接 launch，难以纳入 DAG 与 CUDA Graph 捕获。
- **多架构缺失**：手写 kernel 未按 SM86/87/89/90/120 分别 tune。

### 4.2 重构设计 / Design

- **`ops` 拆为 `cpu/` 与 `cuda/`**：`cuda/` 下已含 flash-attention、fused FFN、
  fused RoPE+KV、AWQ/SQ/FP8 GEMM、GDN、vision encoder 等高性能核（保留并接入新架构）。
- **统一 Dispatcher**：算子对外仅暴露 `tensor::Tensor` 接口 + `stream` 句柄，
  内部经 `kernel::KernelFactory` 选择 **AOT 或 JIT** 实现（见 Step 5）。
- **`CudaConfig` 瘦身**：stream 归 `stream` 模块；cublas handle / workspace 作为
  算子级资源由 Dispatcher 持有；CUDA Graph 归 `inference_optimization`。
- **dtype 优先 FP16**：`common::DataType` 以 FP16 为端侧主精度，量化路径
  （AWQ/SQ/FP8）通过 `KernelTraits` 选择。

> **EN —** Replace the device-only `get_xxx_kernel` raw-pointer dispatch with a
> `KernelTraits`-keyed dispatcher (arch + dtype + shape aware) that routes to AOT
> or JIT kernels. Operators expose only `Tensor` + stream handles; the bloated
> `CudaConfig` is decomposed (stream → stream module, graph → optimization
> module). High-performance fused kernels are retained and re-targeted per arch.

**状态 / Status：⚙️ CUDA 核函数已大量落地（flash-attn / fused-ffn / fused-rope-kv /
AWQ/SQ/FP8 等）；Dispatcher 接口对齐 `common::` 命名空间为后续工作。**

---

## Step 5 — `kernel` 模块（JIT 内核系统编译）/ JIT kernel compilation

参考 / Reference：`flashinfer`（`JitSpec / build_and_load / generate_ninja_build`）、
`mlc-llm`（多 target 编译）。

### 5.1 设计 / Design

文件：`LLMInfer/kernel/{base_kernel_jit_builder.h,kernel_traits.h,kernel_factory.h/.cpp,
cuda_kernel_jit_builder.cpp}`

- **`KernelTraits`**：算子模板参数签名（dtype、tile、head_dim、是否 fused…），
  作为缓存键的一部分。
- **`JitSpec`（借鉴 flashinfer）**：`name / sources / cuda_flags(-gencode 多架构) /
  include_dirs / ld_flags`，计算 `cache_dir / library_path / build.ninja / .lock`。
- **`IKernelJitBuilder`**：`emit_source()`（模板渲染源码）+ `make_spec()`
  （组装含 `-gencode` 的编译参数）。`CudaKernelJitBuilder` 为 CUDA 实现。
- **多架构 `-gencode`**：针对 `SM86/87/89/90/120` 生成
  `-gencode arch=compute_XX,code=sm_XX`，并对 SM90/120 启用相应特性宏。
- **`KernelJitCache`（进程级单例，借鉴 flashinfer `build_and_load`）**：
  1) 命中内存缓存→直接返回；2) AOT 预编译 `.so` 存在→`dlopen`；
  3) 否则生成 `build.ninja`→调 `nvcc` 编译→`dlopen`→入缓存。
  多进程/多线程经 `.lock` 文件锁避免重复编译；`WriteIfDifferent` 避免无谓重编。
- **`KernelFactory`**：`(op_name, device, traits.signature)` 为键，AOT 优先、
  JIT 兜底，二次调用零开销返回函数指针。

> **EN —** A flashinfer-style JIT system: `KernelTraits`-parameterized source
> templates → `JitSpec` (multi-arch `-gencode` for SM86/87/89/90/120) →
> `KernelJitCache.build_and_load` (memory cache → AOT `.so` → ninja+nvcc compile
> → `dlopen`), guarded by file locks and content-diff writes. `KernelFactory`
> caches resolved function pointers keyed by op + device + traits signature,
> preferring AOT and falling back to JIT.

**状态 / Status：⚙️ 接口与 CUDA builder/缓存框架已落地；各算子的源码模板与
AOT 注册为后续逐算子补齐。**

---

## Step 6 — `models` 模块（DAG + JIT 推理流程）/ Model inference graphs

参考算子串联顺序 / Reference (sequence only)：`kuiper/source/model/*`；
**构图方式用 DAG + JIT，不照搬原命令式实现。**

### 6.1 设计 / Design

文件：`LLMInfer/models/{base_model, qwen2, qwen3, qwen3_vl, qwen3_5}/...`

- **`BaseModel`**：负责权重加载（零拷贝映射）、`build_graph()` 构建
  `scheduler::Graph`，对外暴露 `prefill()/decode()`。
- **构图（以 Qwen3 为例）**：
  ```
  EmbeddingTask
      └▶ for each layer L:
            RMSNormTask(attn) ─▶ FusedQKV+RoPE+KVWriteTask ─▶ AttentionTask(flash/MHA)
                                                                  └▶ O-ProjTask ─▶ AddResidualTask
            RMSNormTask(ffn)  ─▶ FusedFFNTask(SwiGLU)        ─▶ AddResidualTask
      └▶ FinalRMSNormTask ─▶ LMHeadTask ─▶ SamplerTask
  ```
  KV cache 以 `kStateful` Edge 在 layer 间共享；算子节点 `run()` 内经
  `KernelFactory` 取（按架构 JIT 的）核函数。
- **模型差异以子图/节点替换表达**：
  - `qwen2/qwen3`：标准 GQA + RoPE；
  - `qwen3_vl`：前置 `VisionEncoder` 子图 + M-RoPE 位置；
  - `qwen3_5`：GDN（gated delta-net）层节点；
  - `qwen3_awq / qwen3_sq / qwen3_fp8`：替换 GEMM 节点为量化核（经 `KernelTraits`）。
- **prefill / decode 双子图**：decode 子图配合 CUDA Graph 固定地址（见 Step 7）。

> **EN —** Each model builds a `scheduler::Graph` instead of imperative op calls.
> Per-layer tasks (RMSNorm → fused QKV+RoPE+KV-write → attention → O-proj →
> residual → FFN) are wired with a stateful KV-cache edge; nodes fetch
> arch-specialized kernels via `KernelFactory`. Model variants are expressed by
> node/subgraph substitution (vision encoder for VL, GDN for 3.5, quantized GEMM
> nodes for AWQ/SQ/FP8). Separate prefill/decode subgraphs enable CUDA-Graph
> capture on the decode path.

**状态 / Status：📐 设计已定义；模型构图代码为后续按模型落地。**

---

## Step 7 — `inference_optimization` 模块 / System-level optimizations

文件：`LLMInfer/inference_optimization/{kv_cache, prefix_cache, cuda_graph}.*`
参考 / Reference：`kuiper/include/base/{prefix_cache.h, radix_tree.h, cuda_graph.h}`。

### 7.1 设计 / Design

- **`kv_cache`**：分页式 KV cache（paged KV，借鉴 vLLM/flashinfer），以
  `kStateful` Edge 暴露给 DAG；支持 GQA（`kv_mul`）与 M-RoPE 位置；FP16 存储。
- **`prefix_cache` + `radix_tree`**：SGLang 风格 RadixAttention——`RadixTree`
  存 token 序列→KV 位置映射，新请求匹配最长公共前缀，复用已算 KV，仅 prefill
  增量；带引用计数与 LRU 淘汰（`PrefixCacheConfig.max_cached_tokens` 等）。
- **`cuda_graph`**：捕获 decode 子图（固定输入/输出地址，`pos` 经参数更新而非
  重捕获），消除 kernel launch 开销——对 batch=1、访存受限的端侧 decode 收益显著
  （借鉴 llama.cpp）。`GraphNodeProperties` 做变更检测，必要时重捕获。

> **EN —** Paged KV cache (vLLM/flashinfer-style) exposed as stateful DAG edges;
> SGLang-style RadixAttention prefix cache (RadixTree longest-prefix reuse +
> refcount + LRU eviction); CUDA-Graph capture of the decode subgraph (fixed
> addresses, parameter-only `pos` update) to remove launch overhead for the
> memory-bound batch-1 decode path.

**状态 / Status：📐 设计与参考已定义；实现为后续落地（原 kuiper 头文件可作算法蓝本）。**

---

## Step 8 — `pipeline` 模块（launcher）/ Pipeline launcher

文件：`LLMInfer/pipeline/{qwen2, qwen3, qwen3_vl, qwen3_5}/..._pipeline.*`、`service/`

### 8.1 设计 / Design

- **`Launcher` 模式**：统一入口 `Pipeline::create(model_type, config)` 工厂，
  按用户请求选择具体模型 pipeline；隐藏 `models` + `scheduler` + `optimization`
  的装配细节。
- **流程**：`load(weights)` →（可选）`warmup`/CUDA-Graph 捕获 →
  `prefill(prompt)` → `decode loop`（经 `Scheduler::run_loop` + `step_cb` 采样/
  流式输出）→ `prefix_cache` 复用。
- **统一对外接口**：`generate(prompt, sampling_params, stream_callback)`，
  与 `inference/`（TensorRT/ONNXRuntime/MNN）后端共享同一抽象，做到
  “一套接口、多后端”。
- **VL pipeline**：额外接收图像输入，先跑 vision encoder 子图再进入文本 decode。

> **EN —** A launcher-style `Pipeline` factory selects the requested model
> pipeline and hides assembly of models + scheduler + optimizations. The flow is
> load → (warmup/CUDA-Graph capture) → prefill → `run_loop` decode with
> sampling/streaming callback and prefix-cache reuse. A single `generate(...)`
> interface is shared with the `inference/` engine backends (TensorRT /
> ONNXRuntime / MNN) for "one API, many backends". VL pipelines prepend a vision
> encoder subgraph.

**状态 / Status：📐 设计已定义；pipeline/service 代码为后续落地。**

---

## 9. 落地状态汇总 / Implementation status

| 模块 / Module | 状态 / Status | 说明 / Note |
|---|---|---|
| `common` | ✅ | Status/枚举/`NoCopyable`/`MemoryType` |
| `memory` | ✅ | 池化+线程安全+零拷贝+`MemoryType` |
| `tensor` | ✅ | 去 armadillo，view/clone/zero-copy |
| `task` | ✅ | Task/Edge/LambdaTask |
| `scheduler` | ✅ | Graph + 3 调度策略 + 线程池 |
| `stream` | ✅ | Stream/Event/Manager(本次新建) |
| `ops` | ⚙️ | CUDA 核大量就绪；Dispatcher 对齐 `common::` 待办 |
| `kernel` | ⚙️ | JIT 框架/缓存/CUDA builder 就绪；源码模板逐算子补齐 |
| `models` | 📐 | DAG 构图设计已定义，代码待落地 |
| `inference_optimization` | 📐 | KV/Prefix/RadixTree/CUDA-Graph 设计已定义 |
| `pipeline` / `service` | 📐 | launcher 设计已定义 |
| `inference` | 📐 | TensorRT/ONNXRuntime/MNN 统一封装设计 |

图例 / Legend：✅ 已落地且自洽 · ⚙️ 框架/核就绪、接口收口中 · 📐 设计已定义、待编码。

---

## 10. 后续工作建议 / Next steps

1. **`ops` 接口收口**：将 `kernels_interface.h` 的 `base::`/`CudaConfig` 迁到
   `common::` + `stream::`，并接入 `KernelFactory`。
2. **`kernel` 源码模板**：逐算子补 `emit_source`，先覆盖 GEMM / RMSNorm / RoPE /
   FlashAttention，并完成 SM86/87/89/90/120 的 AOT 预编译注册。
3. **`models` 构图**：先实现 `qwen3_fp16` 完整 DAG（prefill+decode 子图），
   打通 `tensor→ops→kernel→scheduler` 全链路。
4. **`inference_optimization`**：移植 RadixTree/PrefixCache 算法到 `common::`
   类型，KV cache 改分页式，decode 子图接 CUDA Graph。
5. **`pipeline`**：实现 `Pipeline::create` 工厂与 `generate` 流式接口，打通端到端。
6. **测试**：为 memory/tensor/scheduler/stream 增加单元测试，逐模块回归。

> **EN —** Finalize the ops dispatcher onto `common::`/`stream::`; fill per-op
> JIT source templates + multi-arch AOT registration (GEMM/RMSNorm/RoPE/Flash
> first); implement the `qwen3_fp16` end-to-end DAG; port RadixTree/PrefixCache
> and wire paged-KV + CUDA-Graph; build the `Pipeline::create` launcher with a
> streaming `generate`; add unit tests per module.
