# Fused FFN 在 qwen3.cpp 中的应用深度分析报告

> 项目：OrinMLLM — 基于 NVIDIA Orin 的大语言模型推理引擎  
> 核心文件：`kuiper/source/model/qwen3.cpp`、`kuiper/source/op/kernels/cuda/fused_ffn_kernel.cu`  
> 报告日期：2025 年

---

## 目录

1. [Fused FFN 的原理：数学层面详解](#1-fused-ffn-的原理)
2. [qwen3.cpp 中 Fused FFN 的使用方式与源码分析](#2-qwen3cpp-中-fused-ffn-的使用方式)
3. [Fused FFN 加速原理：数学与源码双重分析](#3-fused-ffn-加速原理)
4. [算子融合的底层限制与不可融合场景](#4-算子融合的底层限制)

---

## 1. Fused FFN 的原理

### 1.1 Transformer FFN 层的数学定义

在 Qwen3（以及 LLaMA 系列）模型中，每个 Transformer 层的 Feed-Forward Network（FFN）采用的是 **SwiGLU** 结构，其数学表达为：

$$\text{FFN}(x) = W_2 \cdot \left[ \text{SiLU}(W_1 x) \odot (W_3 x) \right]$$

其中：
- $x \in \mathbb{R}^{d}$：FFN 的输入向量（经过 RMSNorm 的隐藏状态），$d$ 为模型隐藏维度
- $W_1 \in \mathbb{R}^{h \times d}$：Gate Projection 权重矩阵（门控投影）
- $W_3 \in \mathbb{R}^{h \times d}$：Up Projection 权重矩阵（上升投影）
- $W_2 \in \mathbb{R}^{d \times h}$：Down Projection 权重矩阵（下降投影）
- $h$ 为 FFN 的中间维度（intermediate_size），通常 $h \approx 2.7d$
- $\odot$ 表示逐元素相乘（Hadamard 积）
- $\text{SiLU}(\cdot)$ 是 Sigmoid Linear Unit 激活函数

**SiLU 激活函数**定义为：

$$\text{SiLU}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$$

其中 $\sigma(z) = \frac{1}{1 + e^{-z}}$ 是 Sigmoid 函数。

### 1.2 标准实现：四步串行计算

在非融合的标准实现中，FFN 计算分为以下四个独立步骤：

$$\begin{aligned}
\textbf{Step 1:} \quad & g = W_1 x \quad & \text{(Gate Projection: GEMV)} \\
\textbf{Step 2:} \quad & u = W_3 x \quad & \text{(Up Projection: GEMV)} \\
\textbf{Step 3:} \quad & s = \text{SiLU}(g) \odot u \quad & \text{(SwiGLU 激活)} \\
\textbf{Step 4:} \quad & y = W_2 s \quad & \text{(Down Projection: GEMV)}
\end{aligned}$$

每一步都是一个独立的 CUDA kernel：
- Step 1, 2：通过 cuBLAS GEMV（`cublasGemm`）实现
- Step 3：SwiGLU 逐元素 kernel
- Step 4：通过 cuBLAS GEMV 实现

### 1.3 为什么 Step 1 + Step 2 + Step 3 可以融合？

**关键观察**：在 decode 阶段（batch_size=1），$x$ 是一个一维向量（$x \in \mathbb{R}^d$），Step 1 和 Step 2 本质是两个独立的 **矩阵-向量乘法**（GEMV），它们：

1. **共享相同的输入向量 $x$**
2. **输出维度相同**（都是 $\mathbb{R}^h$）
3. **彼此完全独立**，没有数据依赖

将 $g = W_1 x$ 和 $u = W_3 x$ 展开到逐元素层面，对于输出向量的第 $k$ 行（$k = 0, 1, \ldots, h-1$）：

$$g_k = \sum_{j=0}^{d-1} W_{1}[k, j] \cdot x[j]$$

$$u_k = \sum_{j=0}^{d-1} W_{3}[k, j] \cdot x[j]$$

$$y_k = \text{SiLU}(g_k) \cdot u_k = \frac{g_k}{1 + e^{-g_k}} \cdot u_k$$

**融合的数学本质**：对于每一行 $k$，$g_k$ 和 $u_k$ 都需要遍历相同的输入向量 $x[0:d]$。我们可以在同一次遍历中同时计算 $g_k$ 和 $u_k$，然后立即应用 SiLU 激活并相乘，完全不需要将中间结果 $g$ 和 $u$ 写入全局内存：

$$y_k = \text{SiLU}\left(\sum_{j=0}^{d-1} W_1[k,j] \cdot x[j]\right) \cdot \left(\sum_{j=0}^{d-1} W_3[k,j] \cdot x[j]\right)$$

这三步运算在数学上可以表示为一个单一的复合函数：

$$y_k = f\left(\{W_1[k,:]\}, \{W_3[k,:]\}, x\right) \quad \text{where} \quad f(w_g, w_u, x) = \text{SiLU}\left(\langle w_g, x \rangle\right) \cdot \langle w_u, x \rangle$$

### 1.4 为什么 Step 4（$W_2$）不能融合进去？

$W_2$ 的 Down Projection 不适合与前三步融合，原因如下：

**数据依赖分析**：

$$y = W_2 s, \quad \text{where} \quad s_k = \text{SiLU}(g_k) \cdot u_k$$

$W_2$ 的每一行输出需要用到 **所有** $s_k$（$k=0,\ldots,h-1$），即：

$$y_i = \sum_{k=0}^{h-1} W_2[i,k] \cdot s_k$$

但 fused kernel 中每个 CUDA block 仅计算 **一个** $s_k$。要计算 $y_i$，需要等全部 $h$ 个 $s_k$ 计算完毕。这意味着：

1. 需要跨 block 的全局同步（CUDA 编程模型不支持 kernel 内部的全局 barrier）
2. 或者需要在一个 block 中计算所有 $h$ 行，这远超单个 block 的计算能力

因此，$W_2$ 必须作为一个独立的 kernel 执行。

---

## 2. qwen3.cpp 中 Fused FFN 的使用方式

### 2.1 架构总览

OrinMLLM 中 Fused FFN 的实现采用 **三层架构**：

```
┌──────────────────────────────────────────────────────────────────┐
│                 Qwen3Model::feed_forward_fused()                │
│              （应用层：选择融合/非融合路径，管理 buffer）           │
├──────────────────────────────────────────────────────────────────┤
│                      FusedFFNLayer                              │
│         （算子层：封装 forward 调用，管理数据类型分发）             │
├──────────────────────────────────────────────────────────────────┤
│           fused_gate_up_swiglu_kernel_cu[_fp16/_mixed]          │
│                （CUDA Kernel：底层融合计算）                      │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 Fused FFN Layer 的创建与初始化

**创建**（`qwen3.cpp` 第 295 行，`create_nonparam_layers` 方法）：

```cpp
qwen_layers_->fused_ffn_layer_ = std::make_shared<op::FusedFFNLayer>(
    device_type_, config_->dim_, config_->hidden_dim_, is_fp16_model_, false);
```

参数说明：
- `device_type_`：`kDeviceCUDA`
- `config_->dim_`：模型隐藏维度（如 Qwen3-8B 中 $d = 4096$）
- `config_->hidden_dim_`：FFN 中间维度（如 $h = 11008$）
- `is_fp16_model_`：是否为纯 FP16 模型
- `false`：初始不使用混合精度（运行时动态决定）

**to_cuda 初始化**（`qwen3.cpp` 第 156-158 行，`Qwen3Layers::to_cuda` 方法）：

```cpp
if (fused_ffn_layer_) {
    fused_ffn_layer_->set_cuda_config(config);
    fused_ffn_layer_->to_cuda();
}
```

**运行时开关**（`qwen3.h` 第 129 行）：

```cpp
void enable_fused_ffn(bool enable) { use_fused_ffn_ = enable; }
```

默认值 `use_fused_ffn_ = true`；可通过命令行 `--no-fused-ffn` 禁用。

### 2.3 标准 FFN 路径 vs 融合 FFN 路径：源码逐行对比

#### 标准路径 `feed_forward`（`qwen3.cpp` 第 1132-1179 行）

```cpp
void Qwen3Model::feed_forward(int32_t layer_idx, const tensor::Tensor& input) const {
  // ① Residual Add: input = input + attn_output
  qwen_layers_->add_layer_->forward(input, get_buffer(ModelBufferType::kAttnOutput), input);

  // ② FFN RMSNorm
  tensor::Tensor ffn_norm_output = get_buffer(ModelBufferType::kFFNRMSNorm);
  ffn_rmsnorm->forward(input, ffn_norm_output);

  // ③ W1 GEMV（Gate Projection）→ w1_output       ← Kernel Launch #1
  tensor::Tensor w1_output = get_buffer(ModelBufferType::kW1Output);
  w1_layer->forward(ffn_norm_output, w1_output);

  // ④ W3 GEMV（Up Projection）→ w3_output          ← Kernel Launch #2
  tensor::Tensor w3_output = get_buffer(ModelBufferType::kW3Output);
  w3_layer->forward(ffn_norm_output, w3_output);

  // ⑤ SwiGLU: w1_output = SiLU(w1_output) * w3_output  ← Kernel Launch #3
  qwen_layers_->swiglu_layer_->forward(w1_output, w3_output, w1_output);

  // ⑥ W2 GEMV（Down Projection）→ w2_output        ← Kernel Launch #4
  tensor::Tensor w2_output = get_buffer(ModelBufferType::kW2Output);
  w2_layer->forward(w1_output, w2_output);

  // ⑦ Residual Add: input = input + w2_output
  qwen_layers_->add_layer_->forward(input, w2_output, input);
}
```

**总计**：W1、W3、SwiGLU、W2 = **4 次 kernel launch**（不含 RMSNorm 和 Add）。

#### 融合路径 `feed_forward_fused`（`qwen3.cpp` 第 1354-1430 行）

```cpp
void Qwen3Model::feed_forward_fused(int32_t layer_idx, const tensor::Tensor& input) const {
  // ① Residual Add（与标准路径相同）
  qwen_layers_->add_layer_->forward(input, get_buffer(ModelBufferType::kAttnOutput), input);

  // ② FFN RMSNorm（与标准路径相同）
  tensor::Tensor ffn_norm_output = get_buffer(ModelBufferType::kFFNRMSNorm);
  ffn_rmsnorm->forward(input, ffn_norm_output);

  tensor::Tensor w1_output = get_buffer(ModelBufferType::kW1Output);
  const auto& w1_layer = qwen_layers_->w1_layers_.at(layer_idx);
  const auto& w3_layer = qwen_layers_->w3_layers_.at(layer_idx);

  // ======= AWQ 路径检测 =======
  auto w1_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(w1_layer);
  auto w3_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(w3_layer);

  if (w1_awq || w3_awq) {
    // AWQ 量化模型：不支持融合，fallback 到标准 3-kernel 路径
    tensor::Tensor w3_output = get_buffer(ModelBufferType::kW3Output);
    w1_layer->forward(ffn_norm_output, w1_output);           // Kernel #1
    w3_layer->forward(ffn_norm_output, w3_output);           // Kernel #2
    qwen_layers_->swiglu_layer_->forward(w1_output, w3_output, w1_output);  // Kernel #3
  } else {
    // ======= 标准精度路径：使用 Fused FFN Kernel =======
    auto w1_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(w1_layer);
    auto w3_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(w3_layer);

    const auto& w1_weight = w1_matmul->get_weight(0);  // 直接获取权重 tensor
    const auto& w3_weight = w3_matmul->get_weight(0);

    // 运行时检测数据类型
    bool is_fp16 = input.data_type() == kDataTypeFp16 &&
                   w1_weight.data_type() == kDataTypeFp16;
    bool is_mixed = input.data_type() == kDataTypeFp32 &&
                    w1_weight.data_type() == kDataTypeFp16;

    auto fused_ffn = qwen_layers_->fused_ffn_layer_;
    fused_ffn->set_use_fp16(is_fp16);
    fused_ffn->set_use_mixed(is_mixed);
    fused_ffn->set_input(0, ffn_norm_output);   // 输入 x
    fused_ffn->set_input(1, w1_weight);          // W1 权重
    fused_ffn->set_input(2, w3_weight);          // W3 权重
    fused_ffn->set_output(0, w1_output);         // 融合输出
    fused_ffn->set_cuda_config(cuda_config_);
    STATUS_CHECK(fused_ffn->forward());          // ★ 单 Kernel Launch!
  }

  // ③ W2 GEMV（Down Projection）→ w2_output
  tensor::Tensor w2_output = get_buffer(ModelBufferType::kW2Output);
  w2_layer->forward(w1_output, w2_output);

  // ④ Residual Add
  qwen_layers_->add_layer_->forward(input, w2_output, input);
}
```

**融合路径总计**：FusedFFN、W2 = **2 次 kernel launch**（减少 2 次）。

### 2.4 Decode 循环中的调度

在 `decode()` 函数中（第 1908 行和第 1960 行），通过运行时标志选择路径：

```cpp
for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
    attention_rms(layer_idx, decode_input);
    attention_qkv_with_graph(layer_idx, pos_tensor_gpu);
    attention_mha_with_graph(layer_idx, pos_tensor_gpu);
    
    if (use_fused_ffn_) {
        feed_forward_fused(layer_idx, decode_input);   // ★ 融合路径
    } else {
        feed_forward(layer_idx, decode_input);          // 标准路径
    }
}
```

### 2.5 数据类型分发

`FusedFFNLayer::forward()`（`fused_ffn.cpp`）根据 `use_fp16_` 和 `use_mixed_` 标志分发到三种 CUDA kernel 实现：

```cpp
base::Status FusedFFNLayer::forward() {
  const tensor::Tensor& input = get_input(0);  // x
  const tensor::Tensor& w1 = get_input(1);     // W1 权重
  const tensor::Tensor& w3 = get_input(2);     // W3 权重
  tensor::Tensor& output = get_output(0);       // 融合输出

  if (use_fp16_) {
    kernel::fused_gate_up_swiglu_kernel_cu_fp16(input, w1, w3, output, cuda_config_.get());
  } else if (use_mixed_) {
    kernel::fused_gate_up_swiglu_kernel_cu_mixed(input, w1, w3, output, cuda_config_.get());
  } else {
    kernel::fused_gate_up_swiglu_kernel_cu(input, w1, w3, output, cuda_config_.get());
  }
  return base::error::Success();
}
```

| 路径 | 输入精度 | 权重精度 | 计算精度 | 输出精度 |
|------|----------|----------|----------|----------|
| FP32 | FP32 | FP32 | FP32 | FP32 |
| FP16 | FP16 | FP16 | FP32（内部） | FP16 |
| Mixed | FP32 | FP16 | FP32 | FP32 |

> 注意：即使纯 FP16 路径，内部累加也使用 FP32 以保证数值精度。

---

## 3. Fused FFN 加速原理

### 3.1 数学层面：内存访问量分析

Fused FFN 的加速本质是 **减少全局内存访问**。在 decode 阶段，所有 GEMV 操作均为 memory-bound（内存带宽受限），因此减少内存读写量直接转化为性能提升。

设 $d$ 为隐藏维度，$h$ 为 FFN 中间维度。

#### 标准路径的内存访问量

| 步骤 | 操作 | 读取（元素数） | 写入（元素数） |
|------|------|----------------|----------------|
| W1 GEMV | $g = W_1 x$ | $h \times d + d$ （权重 + 输入）| $h$ |
| W3 GEMV | $u = W_3 x$ | $h \times d + d$ （权重 + 输入）| $h$ |
| SwiGLU | $s = \text{SiLU}(g) \odot u$ | $2h$（读取 $g$ 和 $u$）| $h$ |
| **总计** | | $2hd + 2d + 2h$ | $3h$ |

总内存传输量：$2hd + 2d + 5h$ 个元素。

**关键浪费**：
- 输入向量 $x$（$d$ 个元素）被读取了 **2 次**（W1 和 W3 各读 1 次）
- 中间结果 $g$（$h$ 个元素）被写入、再读出（写 1 次 + 读 1 次）
- 中间结果 $u$（$h$ 个元素）被写入、再读出（写 1 次 + 读 1 次）

#### 融合路径的内存访问量

| 步骤 | 操作 | 读取（元素数） | 写入（元素数） |
|------|------|----------------|----------------|
| Fused Kernel | $y_k = \text{SiLU}(\sum W_1[k,:] \cdot x) \cdot (\sum W_3[k,:] \cdot x)$ | $h \times d + h \times d + d$ | $h$ |
| **总计** | | $2hd + d$ | $h$ |

总内存传输量：$2hd + d + h$ 个元素。

#### 对比

$$\text{内存节省} = (2hd + 2d + 5h) - (2hd + d + h) = d + 4h$$

对于 Qwen3-8B（$d = 4096, h = 11008$），以 FP16（2 字节/元素）计算：

$$\text{节省} = (4096 + 4 \times 11008) \times 2 \text{ bytes} = 48128 \times 2 = 96256 \text{ bytes} \approx 94 \text{ KB/layer}$$

$$\text{节省比例} = \frac{d + 4h}{2hd + 2d + 5h} = \frac{4096 + 44032}{2 \times 11008 \times 4096 + 2 \times 4096 + 5 \times 11008} \approx \frac{48128}{90226688} \approx 0.05\%$$

可以看到，权重读取 $2hd$ 在总量中占绝对主导，而融合节省的 $d + 4h$ 只是很小的比例。但在实践中，**kernel launch overhead 的消除而非内存节省才是主要加速来源**（见下文 3.2）。

### 3.2 源码层面：Kernel Launch Overhead 消除

这是 Fused FFN 带来加速的 **最主要原因**。

#### 标准路径的 Kernel Launch 时序

```
时间线 ──────────────────────────────────────────────────────────►

CPU:  [Launch W1] ─ idle ─ [Launch W3] ─ idle ─ [Launch SwiGLU] ─ idle ─
        │                    │                    │
GPU:    └──[W1 GEMV]─┘      └──[W3 GEMV]─┘      └──[SwiGLU]─┘

每次 Launch 的 CPU overhead: ~5-15μs
总 Launch overhead: 3 × 5-15μs = 15-45μs
```

#### 融合路径的 Kernel Launch 时序

```
时间线 ──────────────────────────────────────────────────────────►

CPU:  [Launch FusedFFN] ─ idle ───────────────────────────────────
        │
GPU:    └──────────[Fused W1+W3+SwiGLU]──────────┘

Launch overhead: 1 × 5-15μs = 5-15μs
```

**节省 2 次 kernel launch**，在 Orin 平台上每次 launch 约 5-15μs，节省 10-30μs/layer。对于 Qwen3-8B 的 36 层：

$$\text{总节省} = 36 \times 10\text{-}30\mu s = 360\text{-}1080\mu s \approx 0.36\text{-}1.08ms$$

在 decode 阶段，单 token 延迟约 5-15ms，这代表 **3%-20% 的性能提升**。

### 3.3 源码层面：输入向量单次加载

在 CUDA kernel 层面，融合的核心优势在于 **输入向量 $x$ 仅从全局内存加载一次**。

**标准路径**中，W1 GEMV 和 W3 GEMV 各自独立地从 DRAM 加载输入向量 $x$：

```
W1 Kernel: for each row k:  load x[0..d-1] from DRAM → compute g_k
W3 Kernel: for each row k:  load x[0..d-1] from DRAM → compute u_k  ← 重复加载!
```

**融合 Kernel**（`fused_ffn_kernel.cu` 第 60-68 行）：

```cuda
// 在同一次循环中同时累加 gate 和 up
for (int i = tid; i < num_vecs; i += BLOCK_SIZE) {
    float4 x = __ldg(input_vec + i);     // ★ 输入只加载一次
    float4 g = __ldg(w1_vec + i);        // W1 权重行
    float4 u = __ldg(w3_vec + i);        // W3 权重行
    
    // 同时计算两个点积
    sum_gate = fmaf(g.x, x.x, fmaf(g.y, x.y, fmaf(g.z, x.z, fmaf(g.w, x.w, sum_gate))));
    sum_up   = fmaf(u.x, x.x, fmaf(u.y, x.y, fmaf(u.z, x.z, fmaf(u.w, x.w, sum_up))));
}
```

加载 `x` 一次后，同时用于 `sum_gate` 和 `sum_up` 的计算。虽然 $x$ 可能被 L2 Cache 缓存而使得第二次加载命中缓存，但融合版本完全消除了这个不确定性。

### 3.4 源码层面：消除中间结果的全局内存写入

**标准路径**需要将 $g$、$u$ 写入全局内存，再在 SwiGLU kernel 中读回：

```
W1 Kernel: compute g_k → write g[k] to DRAM        (h 次写入)
W3 Kernel: compute u_k → write u[k] to DRAM        (h 次写入)
SwiGLU:    read g[k], u[k] from DRAM → compute s_k  (2h 次读取)
```

**融合 Kernel**（第 88-91 行）：

```cuda
// Thread 0 中：g_k 和 u_k 都在寄存器中，直接应用 SiLU
if (tid == 0) {
    float gate_activated = sum_gate / (1.0f + expf(-sum_gate));  // SiLU(g_k)
    output[row] = gate_activated * sum_up;                       // SiLU(g_k) * u_k
}
```

$g_k$ 和 $u_k$ 经过 BlockReduce 后保留在 thread 0 的寄存器中，直接计算 SiLU 并输出，**中间结果 $g$ 和 $u$ 完全不触及全局内存**。

### 3.5 源码层面：FP16 版本的额外优化

FP16 融合 kernel（第 260-348 行）包含多项针对性优化：

**（1）Warp-level 并行：每个 Warp 处理一行**

```cuda
template <int WARP_SIZE = 32, int WARPS_PER_BLOCK = 4>
__global__ void fused_gate_up_swiglu_kernel_fp16_v2(...) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;  // 每 warp 一行
```

FP32 版本每个 block（256 线程）处理一行，而 FP16 版本每个 block 包含多个 warp（`WARPS_PER_BLOCK=8`），每个 warp（32 线程）处理一行。这提高了 SM 的占用率，且 warp shuffle reduction 比 shared memory reduction 更快。

**（2）4 路 ILP（指令级并行）**

```cuda
// 4 组独立累加器，利用 GPU 流水线
float sum_gate0 = 0.0f, sum_gate1 = 0.0f, sum_gate2 = 0.0f, sum_gate3 = 0.0f;
float sum_up0 = 0.0f, sum_up1 = 0.0f, sum_up2 = 0.0f, sum_up3 = 0.0f;

// 每次加载 8 个 half（float4），分 4 组累加
float2 xf0 = __half22float2(x_h2[0]);
float2 gf0 = __half22float2(g_h2[0]);
sum_gate0 = fmaf(gf0.x, xf0.x, fmaf(gf0.y, xf0.y, sum_gate0));

float2 xf1 = __half22float2(x_h2[1]);
float2 gf1 = __half22float2(g_h2[1]);
sum_gate1 = fmaf(gf1.x, xf1.x, fmaf(gf1.y, xf1.y, sum_gate1));
// ... sum_gate2, sum_gate3 类似
```

使用多个独立累加器，消除 FMA（fused multiply-add）指令之间的数据依赖（Read-After-Write Hazard），让 GPU 的 FMA 流水线保持满载。

**（3）Warp Shuffle Reduction（比 Shared Memory 更快）**

```cuda
// Warp shuffle 不需要 shared memory，也不需要 __syncthreads
for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    sum_gate += __shfl_down_sync(0xffffffff, sum_gate, offset);
    sum_up += __shfl_down_sync(0xffffffff, sum_up, offset);
}
```

Warp shuffle 在寄存器之间直接交换数据，延迟约 1 个时钟周期，比 shared memory（~20-30 个时钟周期）快一个数量级。

### 3.6 加速效果总结

| 加速来源 | 机制 | 预估收益 |
|----------|------|----------|
| Kernel Launch 减少 | 3 kernel → 1 kernel | 10-30μs/layer × 36 层 ≈ 0.4-1.1ms |
| 输入向量单次加载 | $x$ 仅从 DRAM 读取 1 次 | 小幅带宽节省（可能被 L2 Cache 抵消） |
| 消除中间结果写入 | $g$, $u$ 不写入 DRAM | 节省 $4h$ 个元素的 DRAM 读写 |
| FP16 ILP + Warp Shuffle | 流水线满载 + 低延迟规约 | 单 kernel 内部吞吐提升 ~20-30% |
| **总体** | | **FFN 层加速 ~20-30%，端到端 decode 加速 ~5-15%** |

---

## 4. 算子融合的底层限制

### 4.1 限制一：数据依赖导致的跨 Kernel 不可融合

**原理**：CUDA 编程模型中，不同 thread block 之间没有全局同步机制（没有 global barrier）。如果操作 B 依赖于操作 A 的 **全部输出**，而 A 的输出分布在多个 block 中计算，则 A 和 B 不能融合到一个 kernel 中。

**在 FFN 中的体现**：

如第 1.4 节所述，$W_2$ 的 Down Projection 不能与前面的 FusedFFN 融合，因为：

$$y_i = \sum_{k=0}^{h-1} W_2[i,k] \cdot s_k$$

需要先完成 $s_0, s_1, \ldots, s_{h-1}$ 的计算（分布在不同 block），再才能开始 $W_2$ 计算。这是一个 **全局数据依赖**。

**源码验证**：在 `feed_forward_fused` 中，W2 始终作为独立 kernel 调用：

```cpp
// FusedFFN 输出到 w1_output（s 向量）
fused_ffn->forward();

// W2 必须等 FusedFFN 全部完成后才能开始
w2_layer->forward(w1_output, w2_output);  // 独立 kernel
```

### 4.2 限制二：AWQ 量化权重不兼容

**原理**：AWQ（Activation-aware Weight Quantization）使用 INT4/INT8 量化权重 + FP16 scale/zero-point 进行反量化计算。AWQ 的 dequantize-and-multiply 逻辑与标准 FP16/FP32 GEMV 完全不同，需要在 kernel 中执行 scale + zero-point 反量化操作。

**在源码中的体现**（`qwen3.cpp` 第 1377-1386 行）：

```cpp
auto w1_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(w1_layer);
auto w3_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(w3_layer);

if (w1_awq || w3_awq) {
    // AWQ path: fall back to standard forward since fused kernel doesn't support AWQ
    w1_layer->forward(ffn_norm_output, w1_output);    // 3 个独立 kernel
    w3_layer->forward(ffn_norm_output, w3_output);
    qwen_layers_->swiglu_layer_->forward(w1_output, w3_output, w1_output);
}
```

**为什么不能融合 AWQ**：

1. AWQ 权重的内存布局（packed INT4 + group scale/zero-point）与标准 FP16 layout 完全不同
2. fused kernel 中的 `__ldg(w1_vec + i)` 假设权重是连续的 float/half 数组，不适用于 AWQ 的 packed format
3. 要支持 AWQ 融合，需要编写全新的 fused kernel 将 dequantize + GEMV + SiLU 三者融合，这是一个独立的优化工作

### 4.3 限制三：Batch Size > 1 时的 GEMV → GEMM 转换

**原理**：Fused FFN kernel 是基于 **GEMV**（矩阵-向量乘法）设计的：每个 CUDA block/warp 处理权重矩阵的一行与输入向量的点积。当 batch_size 大于 1 时：

$$Y = W \cdot X, \quad X \in \mathbb{R}^{d \times B}, Y \in \mathbb{R}^{h \times B}$$

这变成了 **GEMM**（矩阵-矩阵乘法），需要完全不同的 tiling 和并行策略（如 2D tile 分块、shared memory 协作加载等），手写 GEMM kernel 很难超越 cuBLAS 的高度优化实现。

**在源码中的体现**：fused kernel 的启动配置为：

```cuda
// FP32 版本：每 block 一行，GEMV 模式
fused_gate_up_swiglu_kernel<BLOCK_SIZE><<<K, BLOCK_SIZE, 0, stream>>>(...)
//                                          ↑
//                                    K 个 block，每个处理一行

// FP16 版本：每 warp 一行
const int num_blocks = (K + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
fused_gate_up_swiglu_kernel_fp16_v2<32, WARPS_PER_BLOCK><<<num_blocks, ...>>>(...)
```

这些配置明确假设输入是一维向量（$M$ 个元素），不支持 $B > 1$ 的 batch 输入。因此 **Fused FFN 仅在 decode 阶段使用**，Prefill 阶段（$B = \text{seq\_len} > 1$）使用标准的 cuBLAS GEMM + SwiGLU 路径。

### 4.4 限制四：Shared Memory 和寄存器资源限制

**原理**：每个 SM（Streaming Multiprocessor）的 shared memory 和寄存器文件大小有限。当一个 kernel 需要大量 shared memory 或寄存器时，SM 上能同时调度的 block 数减少（occupancy 降低），可能反而降低性能。

**在源码中的体现**：

FP32 版本使用 CUB BlockReduce 需要两块 shared memory：

```cuda
using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
__shared__ typename BlockReduce::TempStorage temp_storage_gate;  // ~1KB
__shared__ typename BlockReduce::TempStorage temp_storage_up;    // ~1KB
```

FP16 版本通过改用 warp shuffle 完全消除 shared memory 使用（每个 warp 32 线程内部通过 `__shfl_down_sync` 规约），使得 occupancy 更高。这也是 FP16 版本设计与 FP32 版本不同的原因之一。

融合更多操作（如试图将 RMSNorm 也融入），会进一步增加 shared memory/寄存器需求，可能导致 occupancy 过低而得不偿失。

### 4.5 限制五：计算图拓扑结构约束

**原理**：只有在计算图中构成 **线性链** 或 **fork-join** 模式的操作才适合融合。如果操作的输出被多个后续操作使用（fan-out > 1），或者操作需要多个不同阶段的输入（fan-in > 1 来自不同 kernel），融合会变得复杂甚至不可行。

**在 FFN 中的体现**：

```
         ┌── W1 @ x ──→ SiLU(·) ──┐
x ──────┤                          ├──→ Element-wise * ──→ W2 @ · ──→ output
         └── W3 @ x ──────────────┘
```

这里 $x$ 分叉（fork）到 W1 和 W3，然后 SiLU 和 W3 的结果汇合（join）。这种 **fork-join** 结构适合融合，因为 fork 的两个分支可以在同一个 kernel 的同一次遍历中完成。

但如果拓扑更复杂（如某些 MoE 结构中，gating 结果影响 expert 选择），融合就不再直接适用。

### 4.6 限制汇总

| 限制类型 | 具体表现 | 在项目中的处理方式 |
|----------|----------|-------------------|
| 全局数据依赖 | $W_2$ 需要完整的 $s$ 向量 | $W_2$ 单独作为独立 kernel 执行 |
| 量化权重不兼容 | AWQ INT4 layout 与 fused kernel 不匹配 | `dynamic_pointer_cast` 检测 AWQ 并 fallback |
| Batch size > 1 | GEMV kernel 不支持 GEMM | 仅 decode（B=1）使用，prefill 用 cuBLAS |
| 硬件资源限制 | shared memory / 寄存器不足 | FP16 改用 warp shuffle 消除 shared memory |
| 复杂计算图 | 非线性/非 fork-join 拓扑 | 仅融合可行的 fork-join 子图（W1+W3+SiLU）|
| 精度约束 | 融合可能改变计算顺序 | FP16 内部使用 FP32 累加器保证精度 |

---

## 附录：完整调用链路图

```
用户命令行：--fused-ffn (默认) / --no-fused-ffn
  └→ inference_common.h: model.enable_fused_ffn(config.use_fused_ffn)
      └→ Qwen3Model::enable_fused_ffn(true)  →  use_fused_ffn_ = true

模型初始化：
  └→ Qwen3Model::create_nonparam_layers()
      └→ fused_ffn_layer_ = make_shared<FusedFFNLayer>(device_type_, dim_, hidden_dim_, ...)
  └→ Qwen3Layers::to_cuda(cuda_config)
      └→ fused_ffn_layer_->set_cuda_config(config) + to_cuda()

Decode 循环（每个 token，每层）：
  └→ Qwen3Model::decode()
      └→ for layer_idx in 0..layer_num:
          └→ [if use_fused_ffn_] feed_forward_fused(layer_idx, input)
              ├→ add_layer->forward(input, attn_output, input)        // Residual
              ├→ ffn_rmsnorm->forward(input, ffn_norm_output)         // RMSNorm
              ├→ [if AWQ]
              │   ├→ w1_awq->forward(ffn_norm_output, w1_output)     // Kernel #1
              │   ├→ w3_awq->forward(ffn_norm_output, w3_output)     // Kernel #2
              │   └→ swiglu->forward(w1_output, w3_output, w1_output) // Kernel #3
              ├→ [else: 标准/FP16/Mixed]
              │   └→ fused_ffn->forward()                            // ★ 单 Kernel!
              │       └→ [if fp16]  fused_gate_up_swiglu_kernel_cu_fp16(...)
              │       └→ [if mixed] fused_gate_up_swiglu_kernel_cu_mixed(...)
              │       └→ [else]     fused_gate_up_swiglu_kernel_cu(...)
              ├→ w2_layer->forward(w1_output, w2_output)              // Down Proj
              └→ add_layer->forward(input, w2_output, input)          // Residual
```

---

## 5. 适配 Fused FFN 过程中的难点、关键点与解决方案

### 5.1 难点一：输入向量共享与重复加载消除

**问题描述**：

标准 FFN 路径中，W1 GEMV 和 W3 GEMV 是两个完全独立的 kernel 调用，各自从 Global Memory 加载输入向量 $x$：

```cpp
// 标准路径（feed_forward）—— qwen3.cpp 第 1153-1161 行
w1_layer->forward(ffn_norm_output, w1_output);  // Kernel #1: 从 DRAM 读取 x
w3_layer->forward(ffn_norm_output, w3_output);  // Kernel #2: 再次从 DRAM 读取 x
```

每次调用的底层 FP16 GEMV kernel 都会独立加载输入向量。对于 Qwen3-8B（$d = 4096$, FP16），$x$ 占 8KB：

```cuda
// fp16_gemv_kernel_optimized —— 每次独立加载
for (int i = lane_id; i < num_h2; i += WARP_SIZE) {
    half2 x = input_h2[i];   // ← 每个 GEMV kernel 都要从 DRAM 加载 x
    half2 w = weight_h2[i];
    // ...
}
```

虽然第二次加载可能命中 L2 Cache（Orin L2 = 4MB），但这取决于 Cache 替换策略且不确定，特别是在多 layer 连续执行时 x 可能被其他数据驱逐。

**解决方案**：

在 fused kernel 中，每个 block/warp 只加载一次 $x$ 的对应段，同时用于 gate 和 up 的点积计算：

```cuda
// fused_gate_up_swiglu_kernel_fp16_v2 —— 输入只加载一次
for (int i = lane_id; i < num_float4; i += WARP_SIZE) {
    float4 x_f4 = __ldg(input_f4 + i);   // ★ x 只加载一次
    float4 g_f4 = __ldg(w1_f4 + i);      // W1 权重
    float4 u_f4 = __ldg(w3_f4 + i);      // W3 权重
    // 同时累加到 sum_gate 和 sum_up
}
```

**关键设计**：循环体内先加载 `x_f4`，再分别加载 `g_f4`（W1）和 `u_f4`（W3），利用 `x_f4` 同时做两个点积。`x_f4` 只占一个寄存器组（4 个 float 寄存器），不增加寄存器压力。

### 5.2 难点二：中间结果消除 — SiLU 激活的就地计算

**问题描述**：

标准路径中，W1 和 W3 的 GEMV 结果（$g$ 和 $u$，各 $h$ 个元素）先写入 Global Memory，然后 SwiGLU kernel 再从 Global Memory 读取这两个中间向量：

```cpp
// 标准路径 —— 3 次 kernel launch，2 个中间 buffer
w1_layer->forward(ffn_norm_output, w1_output);    // 写 g[0..h-1] 到 DRAM
w3_layer->forward(ffn_norm_output, w3_output);    // 写 u[0..h-1] 到 DRAM
swiglu_layer_->forward(w1_output, w3_output, w1_output);  // 读 g, u 再写回
```

对于 Qwen3-8B（$h = 11008$, FP16），中间结果的 DRAM 读写量：
- 写入 $g$：$11008 \times 2 = 22016$ 字节
- 写入 $u$：$11008 \times 2 = 22016$ 字节
- SwiGLU 读取 $g + u$：$22016 \times 2 = 44032$ 字节
- SwiGLU 写入 $s$：$22016$ 字节
- **总计冗余访问**：$44032 + 22016 + 22016 + 22016 = 110080$ 字节 ≈ 107 KB/layer

**解决方案**：

在 fused kernel 中，$g_k$ 和 $u_k$ 经过 BlockReduce/Warp Reduction 后直接保留在 thread 0 的寄存器中，立即应用 SiLU 并输出结果：

```cuda
// fused_gate_up_swiglu_kernel_fp16_v2 —— 第 338-341 行
// sum_gate 和 sum_up 经过 warp shuffle 汇聚到 lane 0
if (lane_id == 0) {
    float gate_activated = sum_gate / (1.0f + expf(-sum_gate));  // SiLU
    output[row] = __float2half(gate_activated * sum_up);          // 直接输出
}
```

**$g_k$ 和 $u_k$ 完全不需要写入 Global Memory**——它们从寄存器计算完成后直接被消费。

### 5.3 难点三：权重获取方式的适配

**问题描述**：

标准路径中，W1 和 W3 作为 `MatmulLayer`（或 `AWQMatmulLayer`）的成员权重，通过 `layer->forward(input, output)` 隐式使用权重。而 fused kernel 需要同时获取 **两个层的权重 tensor**，作为输入参数传入单个 kernel。

这带来了两个技术挑战：
1. 需要从 `MatmulLayer` 中提取内部权重 tensor
2. AWQ 量化层的权重格式（packed INT4 + scale/zero）与标准 FP16 权重完全不同

**解决方案**（`qwen3.cpp` 第 1389-1414 行）：

```cpp
// 关键：通过 dynamic_pointer_cast 区分权重类型
auto w1_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(w1_layer);
auto w3_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(w3_layer);

if (w1_awq || w3_awq) {
    // ★ AWQ 权重不兼容 fused kernel → 回退到标准 3-kernel 路径
    w1_layer->forward(ffn_norm_output, w1_output);
    w3_layer->forward(ffn_norm_output, w3_output);
    swiglu_layer_->forward(w1_output, w3_output, w1_output);
} else {
    // ★ 标准 MatmulLayer：提取内部权重 tensor
    auto w1_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(w1_layer);
    auto w3_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(w3_layer);
    
    const auto& w1_weight = w1_matmul->get_weight(0);  // 获取权重 tensor
    const auto& w3_weight = w3_matmul->get_weight(0);
    
    // 传入 fused kernel
    fused_ffn->set_input(1, w1_weight);
    fused_ffn->set_input(2, w3_weight);
    fused_ffn->forward();
}
```

**关键设计决策**：
1. 使用 `dynamic_pointer_cast` 运行时类型检测实现 **零成本分发**——AWQ 和标准模型共用同一个 `feed_forward_fused` 函数入口
2. 通过 `get_weight(0)` 获取 `MatmulLayer` 内部持有的权重 tensor 引用（零拷贝），避免额外内存分配

### 5.4 难点四：混合精度的处理

**问题描述**：

OrinMLLM 支持三种精度配置：纯 FP32、纯 FP16、FP16 权重 + FP32 激活（混合精度）。fused kernel 需要根据运行时的实际精度配置选择对应的 kernel 实现。

同时，即使是纯 FP16 路径，中间的点积累加也必须使用 FP32 精度。如果直接用 `__hfma2` 原生 FP16 FMA 累加，在 $d = 4096$ 的长向量点积中会产生严重的精度损失（FP16 最大精确表示约 2048 个整数，累加 4096 个元素时溢出风险极高）。

**解决方案**（`qwen3.cpp` 第 1403-1408 行）：

```cpp
// 运行时精度检测
bool is_fp16 = input.data_type() == kDataTypeFp16 && w1_weight.data_type() == kDataTypeFp16;
bool is_mixed = input.data_type() == kDataTypeFp32 && w1_weight.data_type() == kDataTypeFp16;

fused_ffn->set_use_fp16(is_fp16);    // 纯 FP16
fused_ffn->set_use_mixed(is_mixed);  // 混合精度
```

在 FP16 fused kernel 内部，使用 FP32 累加器保证精度：

```cuda
// 全部累加器都是 float（FP32），不是 half
float sum_gate0 = 0.0f, sum_gate1 = 0.0f, sum_gate2 = 0.0f, sum_gate3 = 0.0f;

// half → float 转换后累加
float2 xf0 = __half22float2(x_h2[0]);   // half2 → float2
float2 gf0 = __half22float2(g_h2[0]);
sum_gate0 = fmaf(gf0.x, xf0.x, fmaf(gf0.y, xf0.y, sum_gate0));  // FP32 FMA
```

### 5.5 难点五：CUDA Graph 兼容性

**问题描述**：

Fused FFN 需要与 CUDA Graph 机制兼容。在 CUDA Graph 捕获期间，所有 kernel 的参数（包括输入输出指针）被固化。如果 fused FFN 的输入/输出 buffer 地址在每次 decode 时变化，Graph 回放就会失败。

**解决方案**：

fused FFN 使用预分配的固定地址 buffer（与标准 FFN 路径共享），完全兼容 CUDA Graph：

```cpp
// 所有 buffer 都是预分配、固定地址的
tensor::Tensor ffn_norm_output = get_buffer(ModelBufferType::kFFNRMSNorm);  // 固定地址
tensor::Tensor w1_output = get_buffer(ModelBufferType::kW1Output);          // 固定地址

// 权重 tensor 在模型加载后地址不变
const auto& w1_weight = w1_matmul->get_weight(0);  // 权重地址固定
const auto& w3_weight = w3_matmul->get_weight(0);   // 权重地址固定
```

在 decode 的 CUDA Graph 路径中，fused FFN 通过 `use_fused_ffn_` 开关无缝集成：

```cpp
// qwen3.cpp 第 1908 行 —— CUDA Graph 捕获路径
if (graph->begin_capture(stream)) {
    for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
        // ...
        if (use_fused_ffn_) {
            feed_forward_fused(layer_idx, decode_input);  // ★ 在 Graph 捕获中也有效
        } else {
            feed_forward(layer_idx, decode_input);
        }
    }
    graph->end_capture(stream);
}
```

### 5.6 难点六：FP16 Kernel 的数值稳定性

**问题描述**：

SiLU 激活函数 $\text{SiLU}(z) = z / (1 + e^{-z})$ 在 $z$ 的绝对值较大时数值敏感。FP16 的动态范围有限（约 $6.5 \times 10^{4}$），直接在 FP16 下计算 `expf(-z)` 可能产生 INF/NaN。

**解决方案**：

在 FP16 fused kernel 中，SiLU 计算始终在 FP32 精度下完成，最终结果才转回 FP16：

```cuda
if (lane_id == 0) {
    // sum_gate 已经是 float（FP32），SiLU 在 FP32 下计算
    float gate_activated = sum_gate / (1.0f + expf(-sum_gate));  // FP32 expf
    output[row] = __float2half(gate_activated * sum_up);          // 最终转回 FP16
}
```

### 5.7 难点汇总表

| 难点 | 根因 | 解决方案 | 源码位置 |
|------|------|----------|----------|
| 输入重复加载 | 两个 GEMV 各自加载 $x$ | 单 kernel 内共享 `x_f4` | `fused_ffn_kernel.cu` L289-291 |
| 中间结果写入 DRAM | $g$, $u$ 作为独立 buffer 存在 | 归约后寄存器内直接 SiLU+乘 | `fused_ffn_kernel.cu` L338-341 |
| AWQ 权重不兼容 | INT4 packed format 不匹配 | `dynamic_pointer_cast` 检测 + fallback | `qwen3.cpp` L1377-1386 |
| 混合精度分发 | 3 种精度配置的 kernel 不同 | 运行时 `set_use_fp16/set_use_mixed` 分发 | `qwen3.cpp` L1403-1408 |
| CUDA Graph 兼容 | Graph 要求固定地址 | 使用预分配固定 buffer + 固定权重地址 | `qwen3.cpp` L1909 |
| FP16 数值溢出 | `expf(-z)` 在 FP16 下溢 | SiLU 计算全程 FP32 | `fused_ffn_kernel.cu` L339 |

---

## 6. `fused_gate_up_swiglu_kernel_fp16_v2` 详解

### 6.1 Kernel 全貌

该 kernel 位于 `kuiper/source/op/kernels/cuda/fused_ffn_kernel.cu` 第 256-342 行，是 Fused FFN 在纯 FP16 模式下的核心实现。它将 W1 GEMV + W3 GEMV + SwiGLU 三个操作融合在单个 CUDA kernel 中执行。

```
输入：
  input[M]    —— FFN RMSNorm 的输出（FP16）
  w1[K, M]    —— Gate Projection 权重矩阵（FP16）
  w3[K, M]    —— Up Projection 权重矩阵（FP16）
输出：
  output[K]   —— SiLU(W1·x) ⊙ (W3·x) 的结果（FP16）
  
其中 M = dim = 4096, K = hidden_dim = 11008（以 Qwen3-8B 为例）
```

### 6.2 逐步实现详解

#### Step 1：线程映射与行分配

```cuda
template <int WARP_SIZE = 32, int WARPS_PER_BLOCK = 4>
__global__ void fused_gate_up_swiglu_kernel_fp16_v2(
    const half* __restrict__ input,    // [M]
    const half* __restrict__ w1,       // [K, M]
    const half* __restrict__ w3,       // [K, M]
    half* __restrict__ output,         // [K]
    const int M, const int K
) {
    const int warp_id = threadIdx.x / WARP_SIZE;   // 当前线程在 block 内的 warp 编号
    const int lane_id = threadIdx.x % WARP_SIZE;   // 当前线程在 warp 内的 lane 编号
    const int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;  // 当前 warp 负责的输出行
    
    if (row >= K) return;
```

**设计解读**：

- 实际启动配置为 `WARPS_PER_BLOCK = 8`，即每个 block 包含 $8 \times 32 = 256$ 个线程
- 每个 **warp**（32 个线程）负责计算输出向量的 **一行**（即 $y_k$ 的一个值）
- 总共需要 $K = 11008$ 行，因此需要 $\lceil 11008 / 8 \rceil = 1376$ 个 block

与 FP32 版本的对比：FP32 版本使用 1 block = 1 行，即 $K = 11008$ 个 block。FP16 版本使用 1 warp = 1 行，8 个 warp/block，因此只需 1376 个 block。**减少 block 数量有助于减少 block 调度开销。**

#### Step 2：权重行指针与累加器初始化

```cuda
    const half* w1_row = w1 + static_cast<int64_t>(row) * M;
    const half* w3_row = w3 + static_cast<int64_t>(row) * M;
    
    // Multiple accumulators for ILP
    float sum_gate0 = 0.0f, sum_gate1 = 0.0f, sum_gate2 = 0.0f, sum_gate3 = 0.0f;
    float sum_up0 = 0.0f, sum_up1 = 0.0f, sum_up2 = 0.0f, sum_up3 = 0.0f;
```

**设计解读**：

- `static_cast<int64_t>(row) * M` 使用 64 位乘法避免 32 位溢出（当 $K \times M > 2^{31}$ 时）
- **8 个 FP32 累加器**（4 个 gate + 4 个 up），这是 **4 路指令级并行（ILP）** 的关键

为什么使用 FP32 累加器：FP16 的有效尾数只有 10 位（约 3 位十进制精度）。如果用 FP16 累加，在 $M = 4096$ 个元素的求和中，后面的小值元素会被先前的大和"吞掉"（catastrophic cancellation）。使用 FP32（23 位尾数）累加器可以确保精度。

#### Step 3：float4 向量化加载与双点积计算

```cuda
    const int num_float4 = M / 8;  // 每个 float4 包含 8 个 half
    const float4* input_f4 = reinterpret_cast<const float4*>(input);
    const float4* w1_f4 = reinterpret_cast<const float4*>(w1_row);
    const float4* w3_f4 = reinterpret_cast<const float4*>(w3_row);
    
    #pragma unroll 4
    for (int i = lane_id; i < num_float4; i += WARP_SIZE) {
        float4 x_f4 = __ldg(input_f4 + i);   // 128-bit 加载 = 8 个 half
        float4 g_f4 = __ldg(w1_f4 + i);      // W1 权重 8 个 half
        float4 u_f4 = __ldg(w3_f4 + i);      // W3 权重 8 个 half
```

**设计解读**：

**为什么用 `float4` 而不是 `half2`?**

| 加载方式 | 每次读取宽度 | 每次读取的 half 数 | 内存事务效率 |
|----------|-------------|-------------------|-------------|
| `half` | 16-bit (2B) | 1 | 极差 |
| `half2` | 32-bit (4B) | 2 | 较差 |
| `float4` | 128-bit (16B) | 8 | **最优** |

GPU 的内存控制器以 **128 字节（1024-bit）为粒度** 从 DRAM 读取数据。使用 `float4` 每次读 128-bit，极大提高带宽利用率。一个 warp 的 32 个线程同时发出 `float4` 读取 = $32 \times 16B = 512B$，接近一个完整的 cache line 事务。

**为什么使用 `__ldg`?**

`__ldg()` 指令通过 **只读数据缓存（texture cache / L1 read-only cache）** 加载数据，而非 L1 cache 的常规路径。对于 GEMV 中的只读输入和权重数据，`__ldg` 避免与读写数据竞争 L1 cache 容量，提高 cache 命中率。

#### Step 4：half2 → float2 转换与 4 路 ILP FMA

```cuda
        // 将 float4 解释为 4 个 half2
        const half2* x_h2 = reinterpret_cast<const half2*>(&x_f4);
        const half2* g_h2 = reinterpret_cast<const half2*>(&g_f4);
        const half2* u_h2 = reinterpret_cast<const half2*>(&u_f4);
        
        // 第 0 组 (half2[0])
        float2 xf0 = __half22float2(x_h2[0]);
        float2 gf0 = __half22float2(g_h2[0]);
        float2 uf0 = __half22float2(u_h2[0]);
        sum_gate0 = fmaf(gf0.x, xf0.x, fmaf(gf0.y, xf0.y, sum_gate0));
        sum_up0 = fmaf(uf0.x, xf0.x, fmaf(uf0.y, xf0.y, sum_up0));
        
        // 第 1 组 (half2[1])
        float2 xf1 = __half22float2(x_h2[1]);
        float2 gf1 = __half22float2(g_h2[1]);
        float2 uf1 = __half22float2(u_h2[1]);
        sum_gate1 = fmaf(gf1.x, xf1.x, fmaf(gf1.y, xf1.y, sum_gate1));
        sum_up1 = fmaf(uf1.x, xf1.x, fmaf(uf1.y, xf1.y, sum_up1));
        
        // 第 2 组 (half2[2])
        float2 xf2 = __half22float2(x_h2[2]);
        float2 gf2 = __half22float2(g_h2[2]);
        float2 uf2 = __half22float2(u_h2[2]);
        sum_gate2 = fmaf(gf2.x, xf2.x, fmaf(gf2.y, xf2.y, sum_gate2));
        sum_up2 = fmaf(uf2.x, xf2.x, fmaf(uf2.y, xf2.y, sum_up2));
        
        // 第 3 组 (half2[3])
        float2 xf3 = __half22float2(x_h2[3]);
        float2 gf3 = __half22float2(g_h2[3]);
        float2 uf3 = __half22float2(u_h2[3]);
        sum_gate3 = fmaf(gf3.x, xf3.x, fmaf(gf3.y, xf3.y, sum_gate3));
        sum_up3 = fmaf(uf3.x, xf3.x, fmaf(uf3.y, xf3.y, sum_up3));
    }
```

**设计解读**：

**数据拆解过程**（这是整个 kernel 最精妙的部分）：

```
float4 x_f4 (128-bit) 在寄存器中被重新解释为 4 个 half2：
┌─────────┬─────────┬─────────┬─────────┐
│ x_h2[0] │ x_h2[1] │ x_h2[2] │ x_h2[3] │
│ (h0,h1) │ (h2,h3) │ (h4,h5) │ (h6,h7) │
└─────────┴─────────┴─────────┴─────────┘
    ↓ __half22float2
┌──────────┬──────────┬──────────┬──────────┐
│ xf0      │ xf1      │ xf2      │ xf3      │
│ (f0,f1)  │ (f2,f3)  │ (f4,f5)  │ (f6,f7)  │
└──────────┴──────────┴──────────┴──────────┘
```

每个 `half2 → float2` 转换由 `__half22float2` 内建函数完成，编译为 1 条 `H2F` PTX 指令（0 cycle throughput on Orin SM 8.7）。

**4 路 ILP（指令级并行）的原理**：

GPU 的 FMA 流水线有固定的延迟（通常 4-6 个时钟周期）。如果只用一个累加器 `sum`，那么下一条 `fmaf(a, b, sum)` 必须等上一条的结果写回 `sum` 才能开始——产生 **RAW（Read-After-Write）数据冒险**。

使用 4 个独立累加器消除了这个依赖链：

```
时钟周期  操作
  1       fmaf(gf0.x, xf0.x, sum_gate0)  ← 启动
  2       fmaf(gf1.x, xf1.x, sum_gate1)  ← 不等 sum_gate0，直接启动
  3       fmaf(gf2.x, xf2.x, sum_gate2)  ← 不等 sum_gate1，直接启动
  4       fmaf(gf3.x, xf3.x, sum_gate3)  ← 不等 sum_gate2，直接启动
  5       fmaf(gf0.y, xf0.y, sum_gate0)  ← sum_gate0 此时已就绪（延迟 = 4 cycles）
  ...     （流水线持续满载）
```

这样 FMA 单元每个周期都能发射一条指令，吞吐量提升至接近理论峰值。

**为什么同时做 gate 和 up 的双点积？**

注意循环体内实际上做了 **8 组 FMA**（gate 4 组 + up 4 组）。由于 gate 和 up 使用同一个 `xf0/xf1/xf2/xf3`（已在寄存器中），对 `sum_up0/1/2/3` 的 FMA 与 `sum_gate0/1/2/3` 的 FMA 也是独立的，GPU 的双发射调度器可以交替调度这些指令，进一步填满流水线。

#### Step 5：累加器合并

```cuda
    // Merge 4 accumulators into 1
    float sum_gate = sum_gate0 + sum_gate1 + sum_gate2 + sum_gate3;
    float sum_up = sum_up0 + sum_up1 + sum_up2 + sum_up3;
```

这一步只执行 3 次加法（$a_0 + a_1 + a_2 + a_3$），开销可忽略。合并后每个线程持有该行的部分和。

#### Step 6：余数处理

```cuda
    const int base = num_float4 * 8;
    for (int i = base + lane_id; i < M; i += WARP_SIZE) {
        float x_val = __half2float(__ldg(input + i));
        sum_gate = fmaf(__half2float(__ldg(w1_row + i)), x_val, sum_gate);
        sum_up = fmaf(__half2float(__ldg(w3_row + i)), x_val, sum_up);
    }
```

当 $M$ 不是 8 的整数倍时，剩余的 $M \mod 8$ 个元素逐个处理。对于 Qwen3-8B（$M = 4096 = 512 \times 8$），不会进入此分支。

#### Step 7：Warp Shuffle 归约

```cuda
    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum_gate += __shfl_down_sync(0xffffffff, sum_gate, offset);
        sum_up += __shfl_down_sync(0xffffffff, sum_up, offset);
    }
```

**设计解读**：

这是 warp 内 32 个线程的部分和汇聚为总和的过程：

```
初始：lane 0..31 各持有部分和

第 1 步 (offset=16): lane i += lane (i+16) 的值
  lane 0 = lane0 + lane16
  lane 1 = lane1 + lane17
  ...

第 2 步 (offset=8): lane i += lane (i+8) 的值
  lane 0 = (lane0+lane16) + (lane8+lane24)
  ...

第 3 步 (offset=4):
第 4 步 (offset=2):
第 5 步 (offset=1): 
  lane 0 = 全部 32 个线程的总和
```

5 步 butterfly reduction 后，`lane_id == 0` 的线程持有完整的行点积结果。

**为什么 Warp Shuffle 比 Shared Memory 快？**

| 归约方式 | 延迟 | 需要内存 | 同步 |
|----------|------|----------|------|
| Shared Memory + `__syncthreads` | ~20-30 cycles/step | 需要 shared memory | 需要显式 barrier |
| CUB BlockReduce | ~20 cycles/step | ~1KB shared | 内部有 barrier |
| **Warp Shuffle** | **~1 cycle/step** | **0** | **隐式（warp 锁步）** |

Warp shuffle 通过寄存器文件的跨 lane 交换实现数据传递，不经过任何内存层级，延迟极低。且 warp 内的 32 个线程天然同步（SIMT 锁步执行），不需要显式 barrier。

#### Step 8：SiLU 激活与最终输出

```cuda
    if (lane_id == 0) {
        float gate_activated = sum_gate / (1.0f + expf(-sum_gate));
        output[row] = __float2half(gate_activated * sum_up);
    }
```

**设计解读**：

这是 $y_k = \text{SiLU}(g_k) \cdot u_k$ 的最终计算：

$$y_k = \frac{g_k}{1 + e^{-g_k}} \cdot u_k$$

- `expf` 是 FP32 精度的指数函数（精度更高）
- `sum_gate` 和 `sum_up` 都是 FP32，SiLU 全程 FP32 计算
- 最终通过 `__float2half` 转回 FP16 写入输出
- **Only thread 0 (lane_id == 0) writes**：32 个线程中只有一个写出结果，无 bank conflict

### 6.3 启动配置

```cuda
constexpr int WARPS_PER_BLOCK = 8;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;  // = 256
const int num_blocks = (K + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
// K=11008: num_blocks = 1376

fused_gate_up_swiglu_kernel_fp16_v2<32, 8><<<1376, 256, 0, stream>>>(...);
```

| 参数 | 值 | 说明 |
|------|-----|------|
| Grid | 1376 blocks | $\lceil 11008 / 8 \rceil$ |
| Block | 256 threads | 8 warps × 32 threads/warp |
| Shared Memory | 0 bytes | Warp shuffle 不需要 shared memory |
| 寄存器/线程 | ~32 个 | 8 个 FP32 累加器 + 临时变量 |

### 6.4 从硬件层次理解 Kernel 的计算过程

本节从 **Global Memory、Shared Memory、Block、Thread（Warp/Lane）** 四个硬件抽象层次，自顶向下剖析 `fused_gate_up_swiglu_kernel_fp16_v2` 如何完成 FFN 算子的计算。

#### 6.4.1 Global Memory 层面

Global Memory（全局显存 / DRAM）是 GPU 上容量最大、延迟最高的存储层级（Orin 上带宽约 102.4 GB/s，延迟 ~400-600 cycles）。本 kernel 的全局内存访问模式如下：

**输入数据（只读）**：

| 数据 | 形状 | 大小（Qwen3-8B, FP16） | 访问模式 |
|------|------|------------------------|----------|
| `input[M]` | $[4096]$ | 8 KB | 所有 warp 共享读取同一份 input |
| `w1[K, M]` | $[11008, 4096]$ | ~86 MB | 每个 warp 读取不同的行 |
| `w3[K, M]` | $[11008, 4096]$ | ~86 MB | 每个 warp 读取不同的行 |

**输出数据（只写）**：

| 数据 | 形状 | 大小 | 访问模式 |
|------|------|------|----------|
| `output[K]` | $[11008]$ | ~22 KB | 每个 warp 中仅 lane 0 写一个元素 |

**全局内存访问总量**（整个 kernel 执行期间）：

$$\text{读取} = \underbrace{K \times M \times 2B}_{W1\ 权重} + \underbrace{K \times M \times 2B}_{W3\ 权重} + \underbrace{K \times M \times 2B}_{input\ 被\ K\ 个\ warp\ 各读一次} = 3 \times 11008 \times 4096 \times 2 \approx 258\ \text{MB}$$

> **注**：虽然 input 只有 8KB，但每个 warp 独立加载一遍。由于 input 较小（8KB < L2 Cache 4MB），第一个 warp 读取后会被缓存到 L2/L1，后续 warp 的读取均命中缓存，实际 DRAM 读取约 86 MB + 86 MB + 8 KB ≈ 172 MB。

$$\text{写入} = K \times 2B = 11008 \times 2 = 22\ \text{KB}$$

**关键设计：`__ldg` 只读缓存路径**

所有全局内存读取均通过 `__ldg()` 内建函数执行。`__ldg` 走 GPU 的 **只读纹理缓存路径（L1 Read-Only Cache / Texture Cache）**，而非默认的 L1 Data Cache。这意味着：

1. 只读数据不会被写操作（`output[row] = ...`）的 cache 一致性协议干扰
2. 只读缓存有独立的 tag 和替换逻辑，对广播式访问模式（如 input 被所有 warp 共享）更友好
3. 编译器不需要在 `__ldg` 加载前后插入 fence 指令，减少指令开销

**关键设计：float4 向量化合并访问（Coalesced Access）**

每个线程使用 `float4`（128-bit = 16 Bytes）一次加载 8 个 half 元素。当一个 warp 的 32 个线程同时执行 `__ldg(input_f4 + i)` 时（`i = lane_id, lane_id + 32, ...`），它们访问的地址是连续的：

```
Warp 中 32 个线程的 float4 加载地址：
  lane 0:  input_f4[0]   → 地址 0x0000 ~ 0x000F  (16B)
  lane 1:  input_f4[1]   → 地址 0x0010 ~ 0x001F  (16B)
  lane 2:  input_f4[2]   → 地址 0x0020 ~ 0x002F  (16B)
  ...
  lane 31: input_f4[31]  → 地址 0x01F0 ~ 0x01FF  (16B)
  
  总计：32 × 16B = 512B 连续访问 → 4 个 128B cache line 事务
  → 100% 带宽利用率（无浪费字节）
```

如果使用标量 `half` 加载（2B/次），同样 32 线程只覆盖 64B，仍需 1 个 128B cache line 事务，但有一半带宽被浪费。`float4` 将带宽利用率提升至最优。

#### 6.4.2 Shared Memory 层面

**本 kernel 没有使用任何 Shared Memory**。这是一个重要的设计决策。

Shared Memory 是片上 SRAM（Orin SM 8.7 上每个 SM 最多 164 KB，与 L1 Cache 共享），位于 Block 内所有线程可见的地址空间中，延迟约 20-30 cycles。

**为什么不使用 Shared Memory？**

在 FP32 版本的 `fused_gate_up_swiglu_kernel` 中，使用了 CUB 的 `BlockReduce`，需要约 2KB shared memory 作为 `TempStorage`。而 FP16 版本选择了完全不同的策略——**以 Warp（32 线程）为计算粒度**，用 Warp Shuffle 代替 Shared Memory Reduction：

| 策略 | 归约范围 | 需要 Shared Memory？ | 需要 `__syncthreads`？ | 延迟 |
|------|---------|---------------------|---------------------|------|
| CUB BlockReduce (FP32 版) | Block 内 256 线程 | 是（~1KB × 2） | 是 | ~20-30 cycles/step |
| Warp Shuffle (FP16 版) | Warp 内 32 线程 | **否（0 字节）** | **否（warp 锁步）** | **~1 cycle/step** |

**不使用 Shared Memory 带来的好处**：

1. **Occupancy 最大化**：Shared Memory = 0 意味着 SM 上 block 数量的唯一限制因素变为寄存器数量和线程数，而非 Shared Memory 容量。Orin 每个 SM 有 164 KB Shared Memory，如果每个 block 用 2KB，最多同时驻留 82 个 block；不用 Shared Memory 则没有这个限制
2. **消除 `__syncthreads` 开销**：`BlockReduce` 需要多次 `__syncthreads()` 调用，每次需要等待 block 内所有 256 个线程到达 barrier，这在 warp 执行进度不一致时会造成等待。Warp Shuffle 利用 warp 内线程天然同步（SIMT 锁步执行），完全无需显式同步
3. **避免 Bank Conflict**：Shared Memory 被组织为 32 个 bank，如果多个线程同时访问同一 bank 的不同地址，会产生 bank conflict（串行化访问）。Warp Shuffle 在寄存器文件之间直接传输数据，不存在 bank conflict 问题

**代价**：每个 warp 只能归约 32 个线程的部分和，因此每个 warp 只能处理一行输出。如果需要更多线程协作处理一行（比如 M 特别大），则需要跨 warp 通信，那时就无法避免使用 Shared Memory。在本 kernel 中 $M = 4096$，32 个线程每个处理 $4096 / 8 / 32 = 16$ 次 `float4` 加载，工作量适中。

#### 6.4.3 Block 层面

**启动配置**：

```cuda
constexpr int WARPS_PER_BLOCK = 8;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;  // = 256
const int num_blocks = (K + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
// 对于 K = 11008: num_blocks = ceil(11008 / 8) = 1376
```

**每个 Block 的职责**：

- 每个 Block 包含 **8 个 Warp**（256 个线程）
- 每个 Block **负责计算 8 行输出**（$y_{8b}, y_{8b+1}, \ldots, y_{8b+7}$，其中 $b$ = `blockIdx.x`）
- Block 内的 8 个 Warp 完全独立工作，**没有** warp 间的数据交换或同步

```
Block b（256 个线程）：
┌─────────────────────────────────────────────────────────────────┐
│  Warp 0 (tid 0-31)   → 计算 output[8b + 0] = SiLU(W1[8b+0]·x) × W3[8b+0]·x   │
│  Warp 1 (tid 32-63)  → 计算 output[8b + 1] = SiLU(W1[8b+1]·x) × W3[8b+1]·x   │
│  Warp 2 (tid 64-95)  → 计算 output[8b + 2] = SiLU(W1[8b+2]·x) × W3[8b+2]·x   │
│  Warp 3 (tid 96-127) → 计算 output[8b + 3] = SiLU(W1[8b+3]·x) × W3[8b+3]·x   │
│  Warp 4 (tid 128-159)→ 计算 output[8b + 4] = SiLU(W1[8b+4]·x) × W3[8b+4]·x   │
│  Warp 5 (tid 160-191)→ 计算 output[8b + 5] = SiLU(W1[8b+5]·x) × W3[8b+5]·x   │
│  Warp 6 (tid 192-223)→ 计算 output[8b + 6] = SiLU(W1[8b+6]·x) × W3[8b+6]·x   │
│  Warp 7 (tid 224-255)→ 计算 output[8b + 7] = SiLU(W1[8b+7]·x) × W3[8b+7]·x   │
└─────────────────────────────────────────────────────────────────┘
```

**为什么选 8 个 Warp/Block？**

这是在 occupancy 和资源利用之间的平衡：

- **8 Warp = 256 线程/Block**：这是 CUDA 编程中常用的 block size，通常能达到较高的 SM occupancy
- **太少的 Warp/Block**（如 1 或 2）：block 数量暴增（11008 或 5504），GPU block scheduler 调度开销增大，且每个 SM 需要更多 block 驻留才能隐藏延迟
- **太多的 Warp/Block**（如 16 或 32）：每个 block 占用大量寄存器文件，SM 上能驻留的 block 数减少，灵活性降低

**Block 间的 input 共享问题**：

所有 1376 个 block 中的所有 warp 都需要读取同一个 `input[M]` 向量。由于 CUDA 不支持跨 block 的显式数据共享（除了 Global Memory），input 被每个 warp 独立从 Global Memory 读取。但由于 input 只有 8KB，远小于 L2 Cache（4MB），实际上第一个 block 读取后 input 就被缓存在 L2 中，后续所有 block 的读取都命中 L2 Cache（延迟 ~200 cycles，远低于 DRAM 的 ~500 cycles）。

#### 6.4.4 Thread（Warp / Lane）层面

每个 Warp 由 32 个线程（Lane 0 - Lane 31）组成，它们以 **SIMT（单指令多线程）** 方式锁步执行相同的指令序列。以下是每个 lane 的完整计算流程：

**Phase 1：初始化**

每个 lane 确定自身身份和工作范围：

```cuda
const int warp_id = threadIdx.x / 32;     // 所在 warp 编号（0-7）
const int lane_id = threadIdx.x % 32;     // 在 warp 内的编号（0-31）
const int row = blockIdx.x * 8 + warp_id; // 负责的输出行号
```

每个 lane 初始化 8 个 FP32 累加器（4 个 gate + 4 个 up），全部为 0.0f，存储在 **寄存器** 中。

**Phase 2：向量化加载与 FMA 累加（主循环）**

以 $M = 4096$ 为例，`num_float4 = 4096 / 8 = 512`。每个 lane 以步长 32 遍历 512 个 float4 元素：

```
Lane 0:  处理 float4 索引 = 0, 32, 64, ..., 480  → 共 16 次迭代
Lane 1:  处理 float4 索引 = 1, 33, 65, ..., 481  → 共 16 次迭代
...
Lane 31: 处理 float4 索引 = 31, 63, 95, ..., 511 → 共 16 次迭代
```

每次迭代中，每个 lane 的操作：

```
1. 从 Global Memory 加载 3 个 float4（共 48 字节 = 24 个 half）：
   x_f4 = __ldg(input_f4 + i)    // 8 个 half from input
   g_f4 = __ldg(w1_f4 + i)       // 8 个 half from W1[row]
   u_f4 = __ldg(w3_f4 + i)       // 8 个 half from W3[row]

2. 将 3 个 float4 各拆为 4 个 half2，再转为 4 个 float2：
   xf0, xf1, xf2, xf3 = __half22float2(x_h2[0..3])
   gf0, gf1, gf2, gf3 = __half22float2(g_h2[0..3])
   uf0, uf1, uf2, uf3 = __half22float2(u_h2[0..3])

3. 执行 16 次 FMA 操作（8 次 gate + 8 次 up），分布在 4 路累加器上：
   sum_gate0 += gf0.x * xf0.x + gf0.y * xf0.y  （2 次 fmaf）
   sum_gate1 += gf1.x * xf1.x + gf1.y * xf1.y  （2 次 fmaf）
   sum_gate2 += gf2.x * xf2.x + gf2.y * xf2.y  （2 次 fmaf）
   sum_gate3 += gf3.x * xf3.x + gf3.y * xf3.y  （2 次 fmaf）
   sum_up0   += uf0.x * xf0.x + uf0.y * xf0.y   （2 次 fmaf）
   sum_up1   += uf1.x * xf1.x + uf1.y * xf1.y   （2 次 fmaf）
   sum_up2   += uf2.x * xf2.x + uf2.y * xf2.y   （2 次 fmaf）
   sum_up3   += uf3.x * xf3.x + uf3.y * xf3.y   （2 次 fmaf）
```

每个 lane 在整个主循环中执行：
- 加载次数：$16 \times 3 = 48$ 次 `float4` 加载
- FMA 次数：$16 \times 16 = 256$ 次 `fmaf` 调用
- 处理的元素数：$16 \times 8 = 128$ 个 half 元素（每个权重矩阵）

**Phase 3：累加器合并**

每个 lane 将 4 路累加器合并为 1 个值：

```cuda
float sum_gate = sum_gate0 + sum_gate1 + sum_gate2 + sum_gate3;  // 3 次加法
float sum_up   = sum_up0   + sum_up1   + sum_up2   + sum_up3;    // 3 次加法
```

此时 warp 内每个 lane 持有 input 向量 **不同段** 的部分点积和。

**Phase 4：Warp Shuffle 树形归约**

通过 5 步 butterfly reduction 将 32 个 lane 的部分和汇聚到 Lane 0：

```
步骤 1 (offset=16): 每个 lane i 加上 lane i+16 的值
  Lane 0  = Lane0  + Lane16     Lane 16 = (不再使用)
  Lane 1  = Lane1  + Lane17     Lane 17 = (不再使用)
  ...                            
  Lane 15 = Lane15 + Lane31     Lane 31 = (不再使用)

步骤 2 (offset=8):  Lane 0-7 各加上 Lane 8-15 的值
步骤 3 (offset=4):  Lane 0-3 各加上 Lane 4-7 的值
步骤 4 (offset=2):  Lane 0-1 各加上 Lane 2-3 的值
步骤 5 (offset=1):  Lane 0 加上 Lane 1 的值

→ Lane 0 持有：sum_gate = Σ_{j=0}^{M-1} W1[row,j] × input[j]
               sum_up   = Σ_{j=0}^{M-1} W3[row,j] × input[j]
```

每步执行 2 条 `__shfl_down_sync` + 2 条 `fadd`，5 步共 20 条指令。整个归约仅需 ~5-10 个时钟周期（warp shuffle 延迟约 1 cycle）。

**Phase 5：SiLU 激活与输出（仅 Lane 0）**

```cuda
if (lane_id == 0) {
    float gate_activated = sum_gate / (1.0f + expf(-sum_gate));
    output[row] = __float2half(gate_activated * sum_up);
}
```

只有 Lane 0 执行最终计算并写出结果。其他 31 个 lane 此时空闲（但由于 SIMT 锁步，它们实际上也执行了指令，只是写操作被 predicate mask 屏蔽）。

**每个 Lane 的完整工作量汇总**：

| 阶段 | 操作 | 次数 | 存储层级 |
|------|------|------|----------|
| 加载 input float4 | `__ldg` 128-bit 读取 | 16 次 | Global → L1 Read-Only Cache |
| 加载 W1 float4 | `__ldg` 128-bit 读取 | 16 次 | Global → L1 Read-Only Cache |
| 加载 W3 float4 | `__ldg` 128-bit 读取 | 16 次 | Global → L1 Read-Only Cache |
| half2→float2 转换 | `__half22float2` | 48 次 | 寄存器 |
| FMA 累加 | `fmaf` | 256 次 | 寄存器 |
| 累加器合并 | `fadd` | 6 次 | 寄存器 |
| Warp Shuffle | `__shfl_down_sync` | 10 次 | 寄存器 |
| SiLU + 输出 | `fdiv/fexp/fmul/st` | 4 次（仅 Lane 0） | 寄存器 → Global Memory |

#### 6.4.5 四层硬件抽象总览图

```
┌────────────────────────── GPU Grid ──────────────────────────────┐
│                       1376 个 Block                              │
│  Block 0     Block 1     Block 2     ...     Block 1375          │
│  ┌────────┐  ┌────────┐  ┌────────┐          ┌────────┐          │
│  │ 8 Warps│  │ 8 Warps│  │ 8 Warps│          │ 8 Warps│          │
│  │ row 0-7│  │ row 8-15│ │row16-23│          │row11000│          │
│  │        │  │        │  │        │          │-11007  │          │
│  └────────┘  └────────┘  └────────┘          └────────┘          │
└──────────────────────────────────────────────────────────────────┘

┌────────────────────── Block b (256 threads) ─────────────────────┐
│  Warp 0          Warp 1          ...          Warp 7             │
│  (32 threads)    (32 threads)                 (32 threads)       │
│  row = 8b+0      row = 8b+1                   row = 8b+7        │
│                                                                  │
│  Shared Memory: 0 bytes（完全不使用）                              │
│  Block 内无任何跨 Warp 通信或同步                                   │
└──────────────────────────────────────────────────────────────────┘

┌────────────────────── Warp w (32 lanes) ─────────────────────────┐
│  每个 Lane 的数据流：                                              │
│                                                                  │
│  Global Mem ──ldg──→ 寄存器(float4) ──reinterpret──→ half2[4]    │
│                                       ──half22float2──→ float2[4]│
│                                       ──fmaf──→ 8个FP32累加器     │
│                                                                  │
│  主循环结束后：                                                    │
│  寄存器(8路)──fadd──→ 寄存器(2路)──shfl_down──→ Lane0(2个标量和)    │
│                                  ──SiLU──→ ──float2half──→ Global│
│                                                                  │
│  内存层级使用: Global Memory → L1 Read-Only Cache → 寄存器          │
│  未使用: Shared Memory, L1 Data Cache (写路径)                     │
└──────────────────────────────────────────────────────────────────┘
```

### 6.5 FMA（Fused Multiply-Add）详解

#### 6.5.1 什么是 FMA？

**FMA（Fused Multiply-Add，融合乘加）** 是一种将乘法和加法合并为单条硬件指令执行的运算。其数学定义为：

$$\text{FMA}(a, b, c) = a \times b + c$$

在 CUDA 中，FP32 精度的 FMA 通过内建函数 `fmaf(a, b, c)` 调用，编译为单条 PTX 指令 `fma.rn.f32`。

#### 6.5.2 FMA vs 分离的 MUL + ADD

传统的乘加操作需要两条独立指令：

```
// 分离指令序列
temp = a * b;     // MUL 指令：计算乘积，结果舍入到 FP32
result = temp + c; // ADD 指令：计算和，结果再次舍入到 FP32
```

FMA 将两步合并：

```
// FMA 单条指令
result = fmaf(a, b, c);  // 计算 a*b+c，中间乘积保持全精度，最终结果只舍入一次
```

**FMA 的三大优势**：

**1. 性能优势（2 合 1）**

FMA 只占用 **1 条指令的发射槽和流水线周期**，却完成了乘法 + 加法两个操作。对于 GPU 的 FP32 ALU 流水线来说：

| 方式 | 指令数 | 流水线延迟 | 吞吐率 |
|------|--------|-----------|--------|
| MUL + ADD | 2 条 | ~8 cycles（串行） | 1 个乘加 / 2 个发射周期 |
| FMA | 1 条 | ~4 cycles | 1 个乘加 / 1 个发射周期 |

理论上 FMA 让乘加运算的 **吞吐量翻倍**。

**2. 精度优势（单次舍入）**

这是 FMA 最重要的数学特性。IEEE 754-2008 标准规定 FMA 操作对中间乘积 $a \times b$ **不进行舍入**，而是保留其完整精度（对于 FP32，中间乘积保持约 48 位有效尾数），直到加上 $c$ 之后才执行一次舍入：

$$\text{FMA}(a, b, c) = \text{round}(a \times b + c)$$

而分离的 MUL + ADD 需要两次舍入：

$$\text{MUL\_ADD}(a, b, c) = \text{round}(\text{round}(a \times b) + c)$$

两次舍入会累积舍入误差。特别是在执行长向量点积（如 $d = 4096$ 个元素的求和）时，FMA 的单次舍入可以显著减少误差累积：

$$\text{dot}(w, x) = \sum_{j=0}^{d-1} w_j \cdot x_j$$

使用 FMA 链式计算：

$$\text{sum} = \text{fma}(w_{d-1}, x_{d-1}, \text{fma}(w_{d-2}, x_{d-2}, \ldots \text{fma}(w_1, x_1, \text{fma}(w_0, x_0, 0))\ldots))$$

每一步的 $w_j \times x_j$ 都以全精度参与累加，最终只在最外层舍入一次，误差远小于分离指令的逐步舍入。

**3. 能效优势（减少寄存器写回）**

FMA 不需要将中间乘积 `temp` 写回寄存器文件然后再读出给 ADD 指令。中间结果在 ALU 内部的全精度线路上直接流入加法器，减少了一次寄存器文件的写-读往返，节省寄存器端口带宽和能耗。

#### 6.5.3 FMA 在本 Kernel 中的使用

在 `fused_gate_up_swiglu_kernel_fp16_v2` 的主循环中，`fmaf` 被 **嵌套调用** 以实现连续的乘加链：

```cuda
sum_gate0 = fmaf(gf0.x, xf0.x, fmaf(gf0.y, xf0.y, sum_gate0));
//          │                   │
//          │                   └─ 内层 FMA: temp = gf0.y * xf0.y + sum_gate0
//          └─ 外层 FMA: result = gf0.x * xf0.x + temp
```

展开后等价于：

$$\text{sum\_gate0} = gf0.x \times xf0.x + (gf0.y \times xf0.y + \text{sum\_gate0})$$

这构成一条 **FMA 依赖链**：外层 FMA 依赖内层 FMA 的结果。这条链的延迟为 $2 \times 4 = 8$ 个时钟周期。但由于使用了 **4 路独立累加器**（见 6.6 节 ILP 详解），GPU 调度器可以在等待某一路 FMA 结果的同时，发射其他路的 FMA 指令，从而保持流水线满载。

**每次循环迭代的 FMA 统计**：

| 类别 | FMA 次数 | 说明 |
|------|---------|------|
| Gate 点积 (sum_gate0) | 2 | `fmaf(gf0.x, xf0.x, fmaf(gf0.y, xf0.y, sum_gate0))` |
| Gate 点积 (sum_gate1) | 2 | `fmaf(gf1.x, xf1.x, fmaf(gf1.y, xf1.y, sum_gate1))` |
| Gate 点积 (sum_gate2) | 2 | 同上模式 |
| Gate 点积 (sum_gate3) | 2 | 同上模式 |
| Up 点积 (sum_up0) | 2 | `fmaf(uf0.x, xf0.x, fmaf(uf0.y, xf0.y, sum_up0))` |
| Up 点积 (sum_up1) | 2 | 同上模式 |
| Up 点积 (sum_up2) | 2 | 同上模式 |
| Up 点积 (sum_up3) | 2 | 同上模式 |
| **合计** | **16** | 每次迭代处理 8 对 half 元素 |

每个 lane 在整个主循环中总共执行 $16 \times 16 = 256$ 次 FMA 操作。

#### 6.5.4 GPU 硬件中的 FMA 单元

NVIDIA GPU 的每个 SM（Streaming Multiprocessor）包含多个 **FP32 CUDA Core**，每个 CUDA Core 就是一个 **FMA 流水线**。在 Orin（SM 8.7 / Ampere 架构）上：

- 每个 SM 有 128 个 FP32 CUDA Core
- 每个 CUDA Core 每周期可执行 1 次 FMA 操作
- FMA 流水线深度约 4 个周期（发射到结果就绪）
- 每个 SM 的 FP32 FMA 峰值吞吐：128 FMA/cycle = 256 FLOP/cycle（每次 FMA 算 2 个 FLOP）

FMA 是 GPU 计算吞吐的基石——GPU 的理论算力（TFLOPS）就是以 FMA 吞吐量来衡量的。

### 6.6 ILP（指令级并行）与 4 路 ILP 详解

#### 6.6.1 什么是 ILP？

**ILP（Instruction-Level Parallelism，指令级并行）** 是指在单个线程的指令流中，让多条互不依赖的指令同时在处理器流水线中执行的技术。ILP 的核心思想是：当一条指令正在等待结果就绪时（处于流水线的后续阶段），可以发射另一条不依赖于该结果的指令进入流水线。

**流水线（Pipeline）** 是理解 ILP 的前提。GPU 的 FMA 单元是一个多级流水线：

```
     ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
─in→ │ Stage 1│→ │ Stage 2│→ │ Stage 3│→ │ Stage 4│ →out
     │(取操作 │  │(乘法   │  │(加法   │  │(写回   │
     │ 数)    │  │ 计算)  │  │ 计算)  │  │ 结果)  │
     └────────┘  └────────┘  └────────┘  └────────┘
```

一条 FMA 指令从发射到结果就绪需要经过 4 个流水线阶段（约 4 个时钟周期的延迟）。但流水线的每个阶段在每个周期都可以处理一条不同的指令——前提是这些指令之间没有数据依赖。

#### 6.6.2 无 ILP 时的流水线气泡

考虑最简单的点积累加——使用单个累加器 `sum`：

```cuda
float sum = 0.0f;
for (int i = 0; i < N; i++) {
    sum = fmaf(w[i], x[i], sum);  // 每次都依赖上一次的 sum
}
```

每条 `fmaf(w[i], x[i], sum)` 都读取上一条的 `sum` 结果作为输入。这产生了 **RAW（Read-After-Write）数据冒险**：第 $i+1$ 条 FMA 的第三个操作数 `sum` 需要等第 $i$ 条 FMA 写回结果后才能读取。

流水线执行时序（延迟 = 4 cycles）：

```
周期:  1    2    3    4    5    6    7    8    9   10   11   12
      ┌────────────────┐
FMA₀: │ S1 │ S2 │ S3 │ S4 │                          → sum 在周期 4 就绪
      └────────────────┘
                   ↓↓↓ 等待 sum 就绪（3 个气泡）↓↓↓
                             ┌────────────────┐
FMA₁:                        │ S1 │ S2 │ S3 │ S4 │   → sum 在周期 8 就绪
                             └────────────────┘
                                          ↓↓↓ 再等 3 个气泡 ↓↓↓
                                                    ┌────────────────┐
FMA₂:                                               │ S1 │ S2 │ S3 │ S4 │
                                                    └────────────────┘

→ 实际吞吐：1 FMA / 4 cycles = 25% 流水线利用率
→ 流水线 75% 的时间在空转（气泡）
```

#### 6.6.3 4 路 ILP 如何消除气泡

**4 路 ILP** 是指使用 **4 个独立的累加器**，消除 FMA 指令之间的数据依赖链，使得流水线在等待某一路结果时可以发射其他路的指令。

本 kernel 中 4 路 gate 累加器和 4 路 up 累加器的展开：

```cuda
// 4 路独立累加器——彼此之间无数据依赖
float sum_gate0 = 0.0f, sum_gate1 = 0.0f, sum_gate2 = 0.0f, sum_gate3 = 0.0f;
float sum_up0 = 0.0f, sum_up1 = 0.0f, sum_up2 = 0.0f, sum_up3 = 0.0f;

// 在主循环中，每组 half2 使用不同的累加器：
sum_gate0 = fmaf(gf0.x, xf0.x, fmaf(gf0.y, xf0.y, sum_gate0)); // 依赖 sum_gate0
sum_gate1 = fmaf(gf1.x, xf1.x, fmaf(gf1.y, xf1.y, sum_gate1)); // 依赖 sum_gate1（独立！）
sum_gate2 = fmaf(gf2.x, xf2.x, fmaf(gf2.y, xf2.y, sum_gate2)); // 依赖 sum_gate2（独立！）
sum_gate3 = fmaf(gf3.x, xf3.x, fmaf(gf3.y, xf3.y, sum_gate3)); // 依赖 sum_gate3（独立！）
```

4 路 ILP 下的流水线执行时序：

```
周期:  1    2    3    4    5    6    7    8    9
      ┌────────────────┐
FMA₀: │ S1 │ S2 │ S3 │ S4 │                    → sum_gate0 在周期 4 就绪
      └────────────────┘
           ┌────────────────┐
FMA₁:      │ S1 │ S2 │ S3 │ S4 │               → sum_gate1 在周期 5 就绪
           └────────────────┘
                ┌────────────────┐
FMA₂:           │ S1 │ S2 │ S3 │ S4 │          → sum_gate2 在周期 6 就绪
                └────────────────┘
                     ┌────────────────┐
FMA₃:                │ S1 │ S2 │ S3 │ S4 │     → sum_gate3 在周期 7 就绪
                     └────────────────┘
                          ┌────────────────┐
FMA₄:                     │ S1 │ S2 │ S3 │ S4 │  → sum_gate0 的下一次（周期 5 发射，
                          └────────────────┘       周期 4 的结果已就绪 ✓）

→ 实际吞吐：1 FMA / 1 cycle = 100% 流水线利用率
→ 零气泡！
```

**关键观察**：FMA₄ 操作的是 `sum_gate0`，它需要等待 FMA₀ 的结果（周期 4 就绪）。FMA₄ 在周期 5 发射，此时 FMA₀ 的结果已经就绪，所以不产生等待。4 路 ILP 正好匹配 FMA 流水线的 4 级延迟。

#### 6.6.4 为什么是 "4 路" 而不是 2 路或 8 路？

流水线利用率与 ILP 路数的关系：

| ILP 路数 | 流水线利用率 | 寄存器需求 | 说明 |
|----------|------------|-----------|------|
| 1 路 | 25% (1/4) | 1 个累加器 | 每发射 1 条 FMA 需等 3 cycles |
| 2 路 | 50% (2/4) | 2 个累加器 | 每发射 2 条 FMA 等 2 cycles |
| **4 路** | **100% (4/4)** | **4 个累加器** | **刚好填满 4 级流水线** |
| 8 路 | 100% (受限) | 8 个累加器 | 流水线已满，多余的路只占用寄存器 |

**4 路是最优选择的原因**：

1. **刚好匹配流水线深度**：FMA 流水线延迟约 4 cycles，4 路独立累加器正好在每个 cycle 都有一条可发射的 FMA 指令，流水线 100% 利用
2. **寄存器压力可控**：每多一路需要 2 个额外的 FP32 寄存器（gate + up 各 1 个）。4 路需要 8 个 FP32 累加器（`sum_gate0-3` + `sum_up0-3`），占 8 × 4B = 32B 寄存器，是可接受的开销
3. **超过 4 路无额外收益**：流水线已经 100% 利用，第 5 路及以上的累加器只会浪费寄存器而不提升吞吐
4. **代码结构契合**：每次 `float4` 加载 8 个 half 元素 = 4 个 `half2`，自然分为 4 组，每组使用一路累加器

#### 6.6.5 ILP 与 TLP 的协同

在 GPU 编程中，ILP（指令级并行）与 **TLP（Thread-Level Parallelism，线程级并行）** 是两种互补的延迟隐藏机制：

| 机制 | 并行粒度 | 延迟隐藏方式 | 代价 |
|------|---------|-------------|------|
| TLP | 线程间 | warp scheduler 在多个 warp 间切换 | 需要更多活跃 warp（更高 occupancy） |
| ILP | 线程内 | 单线程内多条独立指令同时在流水线中 | 需要更多寄存器（更多独立累加器） |

**TLP 的工作方式**：当 Warp A 的指令因为等待内存加载（~500 cycles）或 FMA 结果（~4 cycles）而停滞时，warp scheduler 切换到 Warp B 继续执行。这需要 SM 上同时驻留足够多的 warp。

**ILP 的优势**：即使 occupancy 不高（活跃 warp 较少），ILP 也能让单个 warp 的流水线保持高利用率。在本 kernel 中，两者协同工作：

```
SM 上的执行时序：

时钟周期   1    2    3    4    5    6    7    8    9   10
          ╔════╗
Warp A:   ║FMA₀║FMA₁ FMA₂ FMA₃ FMA₄ FMA₅ FMA₆ FMA₇     ← ILP: 4 路独立指令填满流水线
          ╚════╝
           ╔════╗
Warp B:    ║FMA₀║FMA₁ FMA₂ FMA₃ FMA₄ FMA₅ FMA₆ FMA₇    ← TLP: 另一个 warp 交替执行
           ╚════╝
```

在本 kernel 中，每个 Block 有 8 个 Warp，每个 SM 可以同时驻留多个 Block。TLP（跨 warp 切换）用于隐藏 **内存延迟**（~500 cycles），而 ILP（4 路累加器）用于隐藏 **FMA 计算延迟**（~4 cycles）。两者协同最大化了 SM 的利用率。

#### 6.6.6 4 路 ILP 在本 Kernel 中的完整展开

以一次主循环迭代为例，将所有 FMA 指令及其依赖关系展开：

```
数据加载：
  float4 x_f4 = __ldg(input_f4 + i)     // 加载 8 个 half: x[0..7]
  float4 g_f4 = __ldg(w1_f4 + i)        // 加载 8 个 half: W1[row, 0..7]
  float4 u_f4 = __ldg(w3_f4 + i)        // 加载 8 个 half: W3[row, 0..7]

half2→float2 转换（12 次）：
  xf0 = __half22float2(x_h2[0])  →  (x0, x1)
  gf0 = __half22float2(g_h2[0])  →  (g0, g1)
  uf0 = __half22float2(u_h2[0])  →  (u0, u1)
  xf1 = __half22float2(x_h2[1])  →  (x2, x3)
  gf1 = __half22float2(g_h2[1])  →  (g2, g3)
  uf1 = __half22float2(u_h2[1])  →  (u2, u3)
  xf2 = __half22float2(x_h2[2])  →  (x4, x5)
  gf2 = __half22float2(g_h2[2])  →  (g4, g5)
  uf2 = __half22float2(u_h2[2])  →  (u4, u5)
  xf3 = __half22float2(x_h2[3])  →  (x6, x7)
  gf3 = __half22float2(g_h2[3])  →  (g6, g7)
  uf3 = __half22float2(u_h2[3])  →  (u6, u7)

FMA 指令依赖图（16 条 FMA，4 路独立链）：

  路 0:  fmaf(g1, x1, sum_gate0) → fmaf(g0, x0, ·)  → sum_gate0'  [依赖链长度: 2]
  路 1:  fmaf(g3, x3, sum_gate1) → fmaf(g2, x2, ·)  → sum_gate1'  [依赖链长度: 2]
  路 2:  fmaf(g5, x5, sum_gate2) → fmaf(g4, x4, ·)  → sum_gate2'  [依赖链长度: 2]
  路 3:  fmaf(g7, x7, sum_gate3) → fmaf(g6, x6, ·)  → sum_gate3'  [依赖链长度: 2]
  路 4:  fmaf(u1, x1, sum_up0)   → fmaf(u0, x0, ·)  → sum_up0'    [依赖链长度: 2]
  路 5:  fmaf(u3, x3, sum_up1)   → fmaf(u2, x2, ·)  → sum_up1'    [依赖链长度: 2]
  路 6:  fmaf(u5, x5, sum_up2)   → fmaf(u4, x4, ·)  → sum_up2'    [依赖链长度: 2]
  路 7:  fmaf(u7, x7, sum_up3)   → fmaf(u6, x6, ·)  → sum_up3'    [依赖链长度: 2]

  路 0-3 和路 4-7 之间完全独立（gate vs up）
  路 0,1,2,3 之间完全独立（不同累加器）
  → GPU 调度器可在 8 条独立链之间自由调度发射
```

实际上不仅有 4 路 gate ILP，还有 4 路 up ILP，总共 **8 条独立的 FMA 依赖链**。这远超 FMA 流水线深度（4 cycles），确保流水线在任何情况下都不会因数据依赖而停顿。

### 6.7 与标准 FP16 GEMV Kernel 的逐项优化对比

下面将 fused kernel 与标准的 `fp16_gemv_kernel_optimized`（用于非融合路径的 W1、W3 单独 GEMV）进行详细对比：

#### 对比一：向量化宽度

**标准 GEMV**：
```cuda
// half2 加载 = 32-bit = 4 字节/次
const half2* input_h2 = reinterpret_cast<const half2*>(input);
const half2* weight_h2 = reinterpret_cast<const half2*>(row_ptr);
for (int i = lane_id; i < num_h2; i += WARP_SIZE) {
    half2 w = weight_h2[i];  // 32-bit 加载
    half2 x = input_h2[i];   // 32-bit 加载
}
```

**Fused Kernel**：
```cuda
// float4 加载 = 128-bit = 16 字节/次
const float4* input_f4 = reinterpret_cast<const float4*>(input);
const float4* w1_f4 = reinterpret_cast<const float4*>(w1_row);
const float4* w3_f4 = reinterpret_cast<const float4*>(w3_row);
for (int i = lane_id; i < num_float4; i += WARP_SIZE) {
    float4 x_f4 = __ldg(input_f4 + i);   // 128-bit 加载
    float4 g_f4 = __ldg(w1_f4 + i);      // 128-bit 加载
    float4 u_f4 = __ldg(w3_f4 + i);      // 128-bit 加载
}
```

**加速原理**：`float4`（128-bit）每次读取的数据宽度是 `half2`（32-bit）的 **4 倍**。DRAM 访问的固定开销（地址计算、行激活延迟）被分摊到更多有效数据上。此外，128-bit 加载一次完整对齐一个 GPU cache line 的 1/8（cache line = 128B），coalescing 效果更好。

**量化提升**：在 $M = 4096$ 时：
- 标准：$4096/2 = 2048$ 次 `half2` 加载 / warp（每 lane 约 64 次）
- 融合：$4096/8 = 512$ 次 `float4` 加载 / warp（每 lane 约 16 次）
- 循环迭代减少 4 倍，循环控制开销（branch, increment, compare）减少 75%

#### 对比二：缓存提示

**标准 GEMV**：
```cuda
half2 w = weight_h2[i];   // 普通 L1 cache 加载
half2 x = input_h2[i];    // 普通 L1 cache 加载
```

**Fused Kernel**：
```cuda
float4 x_f4 = __ldg(input_f4 + i);   // 通过只读缓存加载
float4 g_f4 = __ldg(w1_f4 + i);      // 通过只读缓存加载
```

**加速原理**：`__ldg()` 使用 GPU 的 **只读数据缓存路径（Texture/L1 Read-Only Cache）**。在 Orin（SM 8.7）上，L1 cache 被分为可读写部分和只读部分。权重和输入在 GEMV 中是纯只读数据，使用只读缓存路径：
1. 避免与可写数据竞争 L1 容量
2. 只读缓存有独立的 tag 比较逻辑，命中率更高
3. 避免 L1 cache 的写回/一致性开销

#### 对比三：ILP（指令级并行）

**标准 GEMV**（单累加器）：
```cuda
float sum = 0.0f;
for (...) {
    sum += wf.x * xf.x + wf.y * xf.y;
    //    ↑ 每次 FMA 都依赖上一次的 sum → RAW 冒险
}
```

**Fused Kernel**（4 路累加器）：
```cuda
float sum_gate0=0, sum_gate1=0, sum_gate2=0, sum_gate3=0;
// 4 组独立 FMA，无数据依赖
sum_gate0 = fmaf(gf0.x, xf0.x, fmaf(gf0.y, xf0.y, sum_gate0));
sum_gate1 = fmaf(gf1.x, xf1.x, fmaf(gf1.y, xf1.y, sum_gate1));
sum_gate2 = fmaf(gf2.x, xf2.x, fmaf(gf2.y, xf2.y, sum_gate2));
sum_gate3 = fmaf(gf3.x, xf3.x, fmaf(gf3.y, xf3.y, sum_gate3));
```

**加速原理**：

GPU 的 FMA 单元流水线深度约 4-6 个周期。单累加器情况下：

```
周期 1: FMA(a0, b0, sum) → 等待 sum 就绪（4 周期）
周期 2: stall
周期 3: stall
周期 4: stall
周期 5: FMA(a1, b1, sum) → 等待 sum 就绪
  → 吞吐率: 1 FMA / 4-5 周期
```

4 路累加器：
```
周期 1: FMA(a0, b0, sum0)
周期 2: FMA(a1, b1, sum1)    ← 不等 sum0
周期 3: FMA(a2, b2, sum2)    ← 不等 sum1
周期 4: FMA(a3, b3, sum3)    ← 不等 sum2
周期 5: FMA(a4, b4, sum0)    ← sum0 此时已就绪！
  → 吞吐率: 1 FMA / 1 周期（理论峰值）
```

**理论加速**：FMA 吞吐从 ~25% 提升到 ~100%，即 **4 倍 ALU 利用率提升**（但实际受限于内存带宽，GEMV 是 memory-bound 操作）。

#### 对比四：kernel launch 次数

**标准路径（非融合）**：
```
Kernel #1: fp16_gemv_kernel_cu(x, W1, out1)   → gate = W1 @ x
Kernel #2: fp16_gemv_kernel_cu(x, W3, out3)   → up = W3 @ x
Kernel #3: swiglu_kernel_cu_fp16_vec(out1, out3, out1)  → SiLU(gate) * up
= 3 次 kernel launch
```

**融合路径**：
```
Kernel #1: fused_gate_up_swiglu_kernel_fp16_v2(x, W1, W3, output)
= 1 次 kernel launch
```

**加速原理**：每次 kernel launch 在 CPU 端经过 CUDA Driver→Kernel Dispatch→GPU Scheduler 链路，耗时约 5-15μs。在 Orin 上 CPU 核心性能相对较弱，launch overhead 更显著。节省 2 次 launch = 节省 10-30μs。

#### 对比五：内存访问模式

**标准路径总内存访问**（FP16，per row）：

| 操作 | 读取 | 写入 |
|------|------|------|
| W1 GEMV: 加载 x[0..M-1] | $M \times 2B$ | - |
| W1 GEMV: 加载 W1[row, 0..M-1] | $M \times 2B$ | - |
| W1 GEMV: 写出 g[row] | - | $2B$ |
| W3 GEMV: 加载 x[0..M-1] **（第二次！）** | $M \times 2B$ | - |
| W3 GEMV: 加载 W3[row, 0..M-1] | $M \times 2B$ | - |
| W3 GEMV: 写出 u[row] | - | $2B$ |
| SwiGLU: 读取 g[row], u[row] | $4B$ | - |
| SwiGLU: 写出 s[row] | - | $2B$ |
| **总计/行** | $4M \times 2B + 4B$ | $6B$ |

**融合路径总内存访问**（FP16，per row）：

| 操作 | 读取 | 写入 |
|------|------|------|
| 加载 x[0..M-1] **（仅一次）** | $M \times 2B$ | - |
| 加载 W1[row, 0..M-1] | $M \times 2B$ | - |
| 加载 W3[row, 0..M-1] | $M \times 2B$ | - |
| 写出 output[row] | - | $2B$ |
| **总计/行** | $3M \times 2B$ | $2B$ |

**节省**：$M \times 2B + 4B + 4B = M \times 2B + 8B$ / 行

对于 $M = 4096$：每行节省 $4096 \times 2 + 8 = 8200B \approx 8KB$
全部 $K = 11008$ 行：$11008 \times 8200 \approx 86MB$

### 6.8 优化手段汇总

| 优化手段 | 标准 FP16 GEMV | Fused FP16 Kernel | 加速原理 |
|----------|---------------|-------------------|----------|
| **向量化宽度** | `half2` (32-bit) | `float4` (128-bit) | 4 倍带宽利用，循环次数减少 75% |
| **缓存提示** | 普通 L1 加载 | `__ldg` 只读缓存 | 避免读写竞争，提升 cache 命中率 |
| **ILP** | 单累加器 | 4 路独立累加器 | FMA 流水线满载，ALU 利用率 ~4 倍提升 |
| **输入共享** | 每个 GEMV 各自加载 x | 单次加载共享 | 消除 $M \times 2B$ 冗余 DRAM 读取 |
| **中间结果** | 写入 DRAM + 读回 | 寄存器内直接计算 | 消除 $4h$ 字节中间 DRAM 读写 |
| **Kernel Launch** | 3 次 (W1 + W3 + SwiGLU) | 1 次 | 节省 ~10-30μs CPU 开销 |
| **归约方式** | Warp Shuffle | Warp Shuffle | 相同，两者均无 shared memory 开销 |
| **Block 利用率** | 8 warp/block, 1 行/warp | 8 warp/block, 1 行/warp | 相同的 SM occupancy |
| **循环展开** | `#pragma unroll 4` | `#pragma unroll 4` | 相同，减少循环控制开销 |
| **精度策略** | FP32 累加 | FP32 累加 | 相同，保证数值精度 |
| **SiLU 融合** | 独立 kernel 计算 | 归约后立即在寄存器中计算 | 零额外内存访问 |

### 6.9 Fused Kernel 完整数据流

```
                               ┌─── float4 加载 ──→ half2 拆解 ──→ float2 转换
                               │    (128-bit)       (4×half2)      (__half22float2)
                               │
input[M]  ──→ input_f4[M/8] ──┤                          ┌→ fmaf → sum_gate0
                               │                          │  fmaf → sum_gate1
w1[row]   ──→ w1_f4[M/8]   ──→ g_f4 → g_h2[0..3] → gf ─┤  fmaf → sum_gate2
                               │                          │  fmaf → sum_gate3
                               │                          │
w3[row]   ──→ w3_f4[M/8]   ──→ u_f4 → u_h2[0..3] → uf ─┤→ fmaf → sum_up0
                                                          │  fmaf → sum_up1
                                                          │  fmaf → sum_up2
                                                          └  fmaf → sum_up3
                                                          
                ↓ 循环 M/8 次后
                
sum_gate = sum_gate0 + sum_gate1 + sum_gate2 + sum_gate3  (合并)
sum_up   = sum_up0   + sum_up1   + sum_up2   + sum_up3

                ↓ Warp Shuffle Reduction (5 步)
                
lane0: sum_gate = Σ(全 warp gate)
lane0: sum_up   = Σ(全 warp up)

                ↓ SiLU + 乘法 (仅 lane0)
                
output[row] = __float2half( SiLU(sum_gate) × sum_up )
```

---

## 附录：三种数据类型 Fused Kernel 的优化策略对比

| 特性 | FP32 版本 | Mixed 版本 | FP16 版本 |
|------|-----------|------------|-----------|
| 向量化方式 | `float4`×1（4 floats） | `float4`×2 (input) + `float4`×1 (weight → 8 halfs) | `float4`×1（8 halfs）|
| 每次加载宽度 | 128-bit | 128-bit (权重) + 256-bit (输入) | 128-bit |
| 并行粒度 | 1 block → 1 行 | 1 block → 1 行 | 1 warp → 1 行 |
| Reduction | CUB BlockReduce (shared mem) | CUB BlockReduce (shared mem) | Warp Shuffle (寄存器) |
| ILP | 无（单累加器） | 无（单累加器） | 4 路累加器 |
| 精度保证 | 原生 FP32 | FP16→FP32 转换后 FP32 累加 | `__half22float2` + FP32 累加 |
| Block Size | 256 threads | 256 threads | 8 warps × 32 = 256 threads |
| 每 Block 行数 | 1 | 1 | 8 (WARPS_PER_BLOCK) |
| Shared Memory | ~2KB (CUB) | ~2KB (CUB) | 0 |
