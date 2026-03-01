# OrinMLLM 工程 AWQ 算子与推理流程深度分析报告

> 生成日期：2026-03-01  
> 工程路径：`/mnt/ssd/workspace/OrinMLLM`  
> 分析范围：AWQ INT4 量化算子、LOP3 指令、反量化机制、Tensor Core MMA 原理

---

## 目录

- [1. AWQ 算子使用与完整推理流程详解](#1-awq-算子使用与完整推理流程详解)
  - [1.1 AWQ 量化基础原理](#11-awq-量化基础原理)
  - [1.2 AWQMatmulLayer 算子定义与实现](#12-awqmatmullayer-算子定义与实现)
  - [1.3 AWQ 权重加载流程](#13-awq-权重加载流程)
  - [1.4 完整推理流程](#14-完整推理流程)
  - [1.5 AWQ 在推理管线中的集成方式](#15-awq-在推理管线中的集成方式)
  - [1.6 CUDA Kernel 分发策略](#16-cuda-kernel-分发策略)
- [2. LOP3 指令的细节原理](#2-lop3-指令的细节原理)
  - [2.1 LOP3 指令概述](#21-lop3-指令概述)
  - [2.2 LOP3 真值表编码原理](#22-lop3-真值表编码原理)
  - [2.3 为什么选择 LOP3 进行 INT4 解包](#23-为什么选择-lop3-进行-int4-解包)
  - [2.4 工程中 LOP3 的具体使用](#24-工程中-lop3-的具体使用)
- [3. 反量化机制与 LOP3 解包的融合](#3-反量化机制与-lop3-解包的融合)
  - [3.1 AWQ 反量化公式](#31-awq-反量化公式)
  - [3.2 AWQ 位序（Bit Order）问题](#32-awq-位序bit-order问题)
  - [3.3 标量反量化路径（awq_gemm_fast.cu）](#33-标量反量化路径awq_gemm_fastcu)
  - [3.4 向量化 LOP3 反量化路径（awq_gemm_vllm.cu）](#34-向量化-lop3-反量化路径awq_gemm_vllmcu)
  - [3.5 LOP3 反量化的完整流水线](#35-lop3-反量化的完整流水线)
- [4. Tensor Core MMA 原理与指令详解](#4-tensor-core-mma-原理与指令详解)
  - [4.1 Tensor Core 硬件原理](#41-tensor-core-硬件原理)
  - [4.2 MMA 指令格式与语义](#42-mma-指令格式与语义)
  - [4.3 ldmatrix 指令详解](#43-ldmatrix-指令详解)
  - [4.4 工程中 Tensor Core MMA 的完整使用流程](#44-工程中-tensor-core-mma-的完整使用流程)
  - [4.5 MMA 指令中的数据布局与线程映射](#45-mma-指令中的数据布局与线程映射)

---

## 1. AWQ 算子使用与完整推理流程详解

### 1.1 AWQ 量化基础原理

AWQ（Activation-aware Weight Quantization）是一种 INT4 权重量化方法，其核心思想是：**不是所有权重通道同等重要，少数与显著激活值对应的通道应保持更高精度**。

在 OrinMLLM 工程中，AWQ 采用 **W4A16** 方案：
- **权重（W）**：INT4 量化（4-bit），8 个 INT4 值打包在 1 个 INT32 中
- **激活值（A）**：保持 FP16（16-bit）不量化
- **反量化公式**：`dequant(w) = scale × (w_int4 - zero_int4)`

存储结构（以一个线性层 `[in_features, out_features]` 为例）：

| 张量名 | 形状 | 数据类型 | 说明 |
|--------|------|----------|------|
| `qweight` | `[in_features, out_features/8]` | INT32 | 量化权重，每个 INT32 存 8 个 INT4 |
| `qzeros` | `[in_features/group_size, out_features/8]` | INT32 | 量化零点 |
| `scales` | `[in_features/group_size, out_features]` | FP16 | 缩放因子 |

其中 `group_size` 通常为 128，表示每 128 个输入通道共享一组 scale/zero 参数。

### 1.2 AWQMatmulLayer 算子定义与实现

#### 头文件定义

AWQ 算子在 `kuiper/include/op/awq_matmul.h` 中定义：

```cpp
class AWQMatmulLayer : public Layer {
 public:
  explicit AWQMatmulLayer(base::DeviceType device_type, 
                          int32_t in_features, int32_t out_features,
                          int32_t group_size = 128);

  base::Status check() const override;
  base::Status forward() override;
  base::Status forward(const tensor::Tensor& input, const tensor::Tensor& output);
  
  void set_awq_weights(const void* qweight_ptr, const void* qzeros_ptr,
                       const void* scales_ptr, base::DeviceType src_device);
  void to_cuda() override;

 private:
  int32_t in_features_ = 0;      // 输入特征维度
  int32_t out_features_ = 0;     // 输出特征维度
  int32_t group_size_ = 128;     // 量化分组大小
  
  tensor::Tensor qweight_;       // [in_features, out_features/8] INT32
  tensor::Tensor qzeros_;        // [in_features/group_size, out_features/8] INT32
  tensor::Tensor scales_;        // [in_features/group_size, out_features] FP16
};
```

关键设计：`AWQMatmulLayer` **直接继承 `Layer` 基类**，与 `MatmulLayer`（FP16/FP32 权重）并列，通过多态实现在推理管线中的无缝替换。它独立管理 `qweight_/qzeros_/scales_` 三组量化参数。

#### 权重设置（`set_awq_weights`）

源码位于 `kuiper/source/op/awq_matmul.cpp` 第 42-63 行：

```cpp
void AWQMatmulLayer::set_awq_weights(const void* qweight_ptr, 
                                      const void* qzeros_ptr,
                                      const void* scales_ptr,
                                      base::DeviceType src_device) {
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  
  // 计算打包后的维度
  int32_t packed_out = out_features_ / 8;  // 8 个 INT4 装入 1 个 INT32
  int32_t num_groups = in_features_ / group_size_;
  
  // qweight: [in_features, out_features/8] INT32
  int32_t qweight_size = in_features_ * packed_out;
  qweight_ = tensor::Tensor(base::DataType::kDataTypeInt32, qweight_size, true, alloc);
  std::memcpy(qweight_.ptr<void>(), qweight_ptr, qweight_size * sizeof(int32_t));
  
  // qzeros: [num_groups, out_features/8] INT32
  int32_t qzeros_size = num_groups * packed_out;
  qzeros_ = tensor::Tensor(base::DataType::kDataTypeInt32, qzeros_size, true, alloc);
  std::memcpy(qzeros_.ptr<void>(), qzeros_ptr, qzeros_size * sizeof(int32_t));
  
  // scales: [num_groups, out_features] FP16
  int32_t scales_size = num_groups * out_features_;
  scales_ = tensor::Tensor(base::DataType::kDataTypeFp16, scales_size, true, alloc);
  std::memcpy(scales_.ptr<void>(), scales_ptr, scales_size * sizeof(uint16_t));
}
```

此函数从 mmap 映射的模型文件内存中拷贝量化参数到 CPU 张量。

#### 权重迁移至 GPU（`to_cuda`）

源码位于 `awq_matmul.cpp` 第 65-101 行，将三组张量从 CPU 搬运到 CUDA：

```cpp
void AWQMatmulLayer::to_cuda() {
  auto cuda_alloc = base::CUDADeviceAllocatorFactory::get_instance();
  
  // Move qweight to CUDA
  tensor::Tensor cuda_qweight(base::DataType::kDataTypeInt32, 
                               qweight_.size(), true, cuda_alloc);
  cudaMemcpy(cuda_qweight.ptr<void>(), qweight_.ptr<void>(),
             qweight_.byte_size(), cudaMemcpyHostToDevice);
  qweight_ = std::move(cuda_qweight);
  
  // ... qzeros 和 scales 同理 ...
}
```

#### 前向推理（forward）

源码位于 `awq_matmul.cpp` 第 103-137 行：

```cpp
base::Status AWQMatmulLayer::forward(const tensor::Tensor& input, 
                                     const tensor::Tensor& output) {
  int batch_size = input.size() / in_features_;
  
  cudaStream_t stream = nullptr;
  if (cuda_config_) {
    stream = cuda_config_->stream;
  }
  
  int split_k_iters = (batch_size == 1) ? 4 : 1;
  
  // 调用底层 CUDA kernel
  kernel::awq_gemm_tensorcore_cu(
      reinterpret_cast<const half*>(input.ptr<uint16_t>()),
      qweight_.ptr<int32_t>(),
      qzeros_.ptr<int32_t>(),
      reinterpret_cast<const half*>(scales_.ptr<uint16_t>()),
      reinterpret_cast<half*>(const_cast<uint16_t*>(output.ptr<uint16_t>())),
      batch_size, in_features_, out_features_,
      group_size_, split_k_iters, stream
  );
  return base::error::Success();
}
```

关键点：`batch_size` 通过 `input.size() / in_features_` 动态推导；`split_k_iters` 根据是否为 decode（M=1）自适应调整。

### 1.3 AWQ 权重加载流程

AWQ 模型文件通过 magic header `0x616b3438`（"ak48"）标识，版本号为 v5。文件头为 256 字节，包含模型配置信息及 `group_size` 参数。

权重加载在 `kuiper/source/model/qwen3.cpp` 的 `create_param_layers_awq()` 函数中实现（第 601-750 行）。文件中的权重按以下**严格顺序**排列：

```
== FP16 权重 ==
1. attention_norm (input_layernorm) × layer_num     — FP16, dim 个半精度元素
2. ffn_norm (post_attention_layernorm) × layer_num  — FP16, dim 个半精度元素
3. final_norm                                       — FP16, dim 个半精度元素
4. token_embeddings                                 — FP16, vocab_size × dim 个半精度元素

== AWQ INT4 量化权重（逐层） ==
5. wq (q_proj)     × layer_num  — qweight(INT32) + qzeros(INT32) + scales(FP16)
6. wk (k_proj)     × layer_num
7. wv (v_proj)     × layer_num  
8. wo (o_proj)     × layer_num
9. w1 (gate_proj)  × layer_num
10. w2 (down_proj) × layer_num
11. w3 (up_proj)   × layer_num

== FP16 权重 ==
12. lm_head                            — FP16, vocab_size × dim
13. q_norm × layer_num                — FP16, head_size
14. k_norm × layer_num                — FP16, head_size
```

核心加载逻辑通过 lambda 函数 `load_awq_layer` 实现：

```cpp
auto load_awq_layer = [&](int32_t in_features, int32_t out_features, 
                          std::vector<std::shared_ptr<op::Layer>>& layer_list,
                          const std::string& name) {
    int32_t packed_out = out_features / 8;
    int32_t num_groups = in_features / group_size_;
    
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
      // 创建 AWQMatmulLayer 实例
      auto awq_layer = std::make_shared<op::AWQMatmulLayer>(
          device_type_, in_features, out_features, group_size_);
      
      // 按顺序读取 qweight → qzeros → scales
      const void* qweight_ptr = base_ptr + pos;
      pos += in_features * packed_out * sizeof(int32_t);      // qweight 字节数
      
      const void* qzeros_ptr = base_ptr + pos;
      pos += num_groups * packed_out * sizeof(int32_t);       // qzeros 字节数
      
      const void* scales_ptr = base_ptr + pos;
      pos += num_groups * out_features * sizeof(uint16_t);    // scales 字节数
      
      awq_layer->set_awq_weights(qweight_ptr, qzeros_ptr, scales_ptr, cpu_device_type);
      layer_list.push_back(awq_layer);
    }
};

// 逐层类型加载
load_awq_layer(dim, dim, qwen_layers_->wq_layers_, "wq");         // q_proj
load_awq_layer(dim, kv_dim, qwen_layers_->wk_layers_, "wk");      // k_proj
load_awq_layer(dim, kv_dim, qwen_layers_->wv_layers_, "wv");      // v_proj
load_awq_layer(dim, dim, qwen_layers_->wo_layers_, "wo");          // o_proj
load_awq_layer(dim, immediate_dim, qwen_layers_->w1_layers_, "w1"); // gate_proj
load_awq_layer(immediate_dim, dim, qwen_layers_->w2_layers_, "w2"); // down_proj
load_awq_layer(dim, immediate_dim, qwen_layers_->w3_layers_, "w3"); // up_proj
```

以 Qwen3-8B 为例（`dim=4096, kv_dim=512, immediate_dim=14336`），每层有 7 个 AWQ 线性层，共 36 层 × 7 = 252 个 AWQ 算子实例。

### 1.4 完整推理流程

OrinMLLM 的推理流程分为 **Prefill（首次预填充）** 和 **Decode（逐 Token 解码）** 两个阶段：

```
用户输入文本
    │
    ▼
Tokenizer 分词
    │
    ▼
┌─────────────────── Prefill 阶段（M = seq_len > 1）───────────────────┐
│  Embedding(tokens) → FP16 激活值                                      │
│                                                                       │
│  for layer_idx in 0..layer_num:                                      │
│    ┌─ Attention Block ──────────────────────────────────────────┐    │
│    │  RMSNorm(input)                                             │    │
│    │  query = AWQ_wq(rms_out)   ← AWQ INT4 矩阵乘法             │    │
│    │  key   = AWQ_wk(rms_out)   ← AWQ INT4 矩阵乘法             │    │
│    │  value = AWQ_wv(rms_out)   ← AWQ INT4 矩阵乘法             │    │
│    │  Q/K Norm → RoPE → KV Cache 写入                           │    │
│    │  FlashAttention(Q, K, V)                                    │    │
│    │  attn_out = AWQ_wo(mha_out)  ← AWQ INT4 矩阵乘法           │    │
│    │  input += attn_out  （残差连接）                             │    │
│    └──────────────────────────────────────────────────────────────┘    │
│    ┌─ FFN Block ────────────────────────────────────────────────┐    │
│    │  RMSNorm(input)                                             │    │
│    │  w1_out = AWQ_w1(ffn_norm)   ← gate_proj                   │    │
│    │  w3_out = AWQ_w3(ffn_norm)   ← up_proj                     │    │
│    │  swiglu_out = SwiGLU(w1_out, w3_out)                       │    │
│    │  w2_out = AWQ_w2(swiglu_out)  ← down_proj                  │    │
│    │  input += w2_out  （残差连接）                               │    │
│    └──────────────────────────────────────────────────────────────┘    │
│                                                                       │
│  Final RMSNorm → LM Head(FP16) → Argmax → First Token                │
└───────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────── Decode 阶段（M = 1，可用 CUDA Graph）──────────────┐
│  循环直到生成 EOS：                                                    │
│    Embedding(last_token) → 单 Token 激活                              │
│    for layer_idx in 0..layer_num:                                    │
│      同上结构，但 M=1 时使用 GEMV 而非 GEMM                           │
│    Argmax → Next Token → 输出                                        │
└───────────────────────────────────────────────────────────────────────┘
```

### 1.5 AWQ 在推理管线中的集成方式

AWQ 算子通过 C++ **动态分派（`dynamic_pointer_cast`）** 无缝集成到推理管线中。在 `kuiper/source/model/qwen_base.cpp` 中，关键位置都通过类型判断来区分 AWQ 和 FP16 路径：

**Attention QKV 投影**（`batched_attention_qkv`）：

```cpp
auto query_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(query_layer);
if (query_awq && key_awq && value_awq) {
    // AWQ 路径：直接调用 forward
    STATUS_CHECK(query_awq->forward(rms_out, query_out));
    STATUS_CHECK(key_awq->forward(rms_out, key_out));
    STATUS_CHECK(value_awq->forward(rms_out, value_out));
} else {
    // FP16 路径：使用 batched_matmul_helper
    ...
}
```

**Attention WO 投影**（`batched_attention_mha`，第 466-468 行）：

```cpp
auto wo_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(wo_layer);
if (wo_awq) {
    STATUS_CHECK(wo_awq->forward(mha_out, wo_out));
} else {
    // FP16 路径
}
```

**FFN 前馈模块**（`feed_forward_fused`，第 264-275 行）：

```cpp
auto w1_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(w1_layer);
auto w3_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(w3_layer);

if (w1_awq || w3_awq) {
    // AWQ fallback：不支持 fused FFN kernel，改为分开执行
    STATUS_CHECK(w1_layer->forward(ffn_norm_output, w1_output));
    STATUS_CHECK(w3_layer->forward(ffn_norm_output, w3_output));
    STATUS_CHECK(layers->swiglu_layer_->forward(w1_output, w3_output, w1_output));
} else {
    // FP16：使用融合 FFN kernel
}
```

**设计要点**：AWQ 不支持 fused FFN kernel（因为权重是 INT4 打包格式无法直接融合），所以在 FFN 中回退到标准的 3 步分离执行（w1 → w3 → SwiGLU）。除了 Embedding 和 LM Head 使用 FP16 外，Transformer 7 个线性层全部由 AWQ 替换。

### 1.6 CUDA Kernel 分发策略

`awq_gemm_tensorcore_cu` 是顶层分发入口（`awq_gemm_tensorcore.cu` 第 48-76 行）：

```cpp
void awq_gemm_tensorcore_cu(...) {
    ensure_initialized();
    
    if (M == 1) {
        // Decode 阶段：内存带宽优化的 GEMV
        awq_gemm_fast_cu(...);
    } else {
        // Prefill 阶段：Tensor Core MMA + LOP3 反量化
        awq_gemm_vllm_cu(...);
    }
}
```

`awq_gemm_fast_cu` 内部再根据 M 值细分（`awq_gemm_fast.cu` 第 582-617 行）：

| 条件 | 使用的 Kernel | 策略 |
|------|--------------|------|
| M = 1 | `awq_gemv_fast_kernel` | GEMV，每 warp 处理 8 个输出通道，warp shuffle 归约 |
| 1 < M ≤ 8 | `awq_gemm_small_batch_kernel` | 小批量优化，每 warp 处理多行输出 |
| M > 8（fast 路径） | `awq_gemm_fast_kernel` | 双缓冲共享内存 + 软件流水线 GEMM |
| M > 1（vllm 路径） | `awq_gemm_vllm_kernel` | Tensor Core MMA + LOP3 反量化 |

---

## 2. LOP3 指令的细节原理

### 2.1 LOP3 指令概述

`lop3.b32` 是 NVIDIA GPU PTX ISA 中的一条强大指令，全称 **Logical Operation on 3 inputs**。它能在**单条指令**中对三个 32-bit 操作数的每一对应 bit 执行**任意三输入布尔函数**。

PTX 语法：

```
lop3.b32 d, a, b, c, immLut;
```

其中 `immLut` 是一个 8-bit 立即数（0x00~0xFF），编码了一个三输入真值表。对于 `a`, `b`, `c` 中每一位 $i$：

$$d[i] = F(a[i], b[i], c[i])$$

其中 $F$ 由 `immLut` 定义的真值表确定。

### 2.2 LOP3 真值表编码原理

三个输入有 8 种可能组合（$2^3 = 8$），`immLut` 的每一位对应一种组合的输出：

| immLut bit | $a[i]$ | $b[i]$ | $c[i]$ | 输出 $d[i]$ = immLut[该bit位] |
|:----------:|:------:|:------:|:------:|:---:|
| bit 0 | 0 | 0 | 0 | immLut[0] |
| bit 1 | 0 | 0 | 1 | immLut[1] |
| bit 2 | 0 | 1 | 0 | immLut[2] |
| bit 3 | 0 | 1 | 1 | immLut[3] |
| bit 4 | 1 | 0 | 0 | immLut[4] |
| bit 5 | 1 | 0 | 1 | immLut[5] |
| bit 6 | 1 | 1 | 0 | immLut[6] |
| bit 7 | 1 | 1 | 1 | immLut[7] |

**约定**：NVIDIA 规定 $a = \texttt{0xF0}$（高4位为1），$b = \texttt{0xCC}$（交替），$c = \texttt{0xAA}$（间隔2）。所以：
- $a$ 的真值表 = `11110000` = 0xF0
- $b$ 的真值表 = `11001100` = 0xCC
- $c$ 的真值表 = `10101010` = 0xAA

任意布尔表达式可以通过对这三个常量进行位运算来计算 `immLut`。

**工程中的例子**：

在 `awq_gemm_fast.cu` 第 92 行：

```cpp
static constexpr uint32_t IMM_LUT = (0xf0 & 0xcc) | 0xaa;
// 计算过程: 0xf0 & 0xcc = 0xc0, 0xc0 | 0xaa = 0xea
// 即 immLut = 0xea
```

`0xea` = `11101010` 二进制，对应的布尔函数为 $(a \wedge b) \vee c$。

在 `awq_gemm_vllm.cu` 中直接使用了立即数 `0xea`：

```cpp
asm volatile("lop3.b32 %0, %1, %2, %3, 0xea;" 
    : "=r"(w_tmp1) 
    : "r"(packed_w), "n"(BOTTOM_MASK), "n"(I4s_TO_FP16_MAGIC));
```

### 2.3 为什么选择 LOP3 进行 INT4 解包

INT4 解包需要**同时完成两个操作**：
1. **位掩码**（AND）：从 INT32 中提取 4-bit 字段
2. **位组合**（OR）：将提取的 INT4 嵌入 FP16 的尾数位中

传统方式需要至少 2 条指令：

```
AND result, packed_w, MASK      // 提取 4-bit
OR  result, result, MAGIC       // 嵌入 FP16 格式
```

而 **LOP3 的 `(a & b) | c` 模式（immLut=0xea）** 在单条指令中完成等效操作：

```
lop3.b32 result, packed_w, MASK, FP16_MAGIC, 0xea;
// 等价于: result = (packed_w & MASK) | FP16_MAGIC
```

**选择 LOP3 的优势总结**：

| 方面 | 传统方式（AND+OR） | LOP3 方式 |
|------|-------------------|-----------|
| 指令数 | 2 条 | **1 条** |
| 吞吐量 | 2 周期 | **1 周期** |
| 寄存器压力 | 需要中间寄存器 | 无中间结果 |
| 功耗 | 两条指令发射 | 一次发射 |
| 适用性 | 仅限 2 输入 | **支持任意三输入布尔** |

在 AWQ 内核中，每个 INT32 需要解包 8 个 INT4 值，要执行 4 次 LOP3 操作（每次处理 2 个 INT4）。使用 LOP3 将指令数从 8 条减少到 4 条，**吞吐量提升一倍**。

### 2.4 工程中 LOP3 的具体使用

#### 在 `dequant_s4_to_fp16x2` 中（`awq_gemm_fast.cu` 第 83-120 行）

此函数将一个 INT32（包含 8 个 INT4）转换为 8 个 FP16 值：

```cpp
// Magic 常量
static constexpr uint32_t BOTTOM_MASK = 0x000f000f;     // 掩码: 提取低4位
static constexpr uint32_t TOP_MASK = 0x00f000f0;         // 掩码: 提取高4位
static constexpr uint32_t I4S_TO_F16S_MAGIC = 0x64006400; // FP16的1024.0

// 将 packed 右移8位得到高16bit的数据
const uint32_t top_i4s = packed_w >> 8;

// 用 LOP3 同时执行 AND + OR：
// result = (packed_w & 0x000f000f) | 0x64006400
// 提取 bits[0:3] 和 bits[16:19]，同时嵌入 FP16 指数位
asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
             : "=r"(h[0])
             : "r"(packed_w), "n"(BOTTOM_MASK), "n"(I4S_TO_F16S_MAGIC), "n"(IMM_LUT));

// 提取 bits[4:7] 和 bits[20:23]
asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
             : "=r"(h[1])
             : "r"(packed_w), "n"(TOP_MASK), "n"(I4S_TO_F16S_MAGIC), "n"(IMM_LUT));

// 提取 bits[8:11] 和 bits[24:27]
asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
             : "=r"(h[2])
             : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4S_TO_F16S_MAGIC), "n"(IMM_LUT));

// 提取 bits[12:15] 和 bits[28:31]
asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
             : "=r"(h[3])
             : "r"(top_i4s), "n"(TOP_MASK), "n"(I4S_TO_F16S_MAGIC), "n"(IMM_LUT));
```

#### 在 `dequant_vllm_lop3` 中（`awq_gemm_vllm.cu` 第 37-96 行）

此版本直接同时处理权重和零点，并利用 `half2` 向量化操作：

```cpp
// 同时对 weight 和 zeros 执行 LOP3 解包
asm volatile("lop3.b32 %0, %1, %2, %3, 0xea;" 
    : "=r"(w_tmp1) : "r"(packed_w), "n"(BOTTOM_MASK), "n"(I4s_TO_FP16_MAGIC));
asm volatile("lop3.b32 %0, %1, %2, %3, 0xea;" 
    : "=r"(z_tmp1) : "r"(packed_z), "n"(BOTTOM_MASK), "n"(I4s_TO_FP16_MAGIC));
```

---

## 3. 反量化机制与 LOP3 解包的融合

### 3.1 AWQ 反量化公式

AWQ 的反量化公式为：

$$W_{fp16}[i] = \text{scale}[g][i] \times (W_{int4}[i] - Z_{int4}[g][i])$$

其中 $g = \lfloor k / \text{group\_size} \rfloor$ 是当前输入通道 $k$ 所属的量化组。

### 3.2 AWQ 位序（Bit Order）问题

AWQ 使用**非连续的特殊位序**来打包 8 个 INT4 值到 INT32 中。在工程中记录的映射关系为：

```
输出索引:    0  1  2  3  4  5  6  7
AWQ 位位置:  0  16  4  20  8  24  12  28   (即 awq_order[i] * 4)
awq_order:   0  4  1  5  2  6  3  7
```

这意味着输出的第 0 个元素在 INT32 的 bits[0:3]，第 1 个元素在 bits[16:19]，第 2 个元素在 bits[4:7]...

这种交错排列不是随意的，而是为了能让 vllm 风格的重排后恰好匹配 LOP3 的 half2 并行处理模式。vllm 重排后的格式为：

```
bits 0:15  (低半部分): 偶数索引元素 0, 2, 4, 6
bits 16:31 (高半部分): 奇数索引元素 1, 3, 5, 7
```

这样 `BOTTOM_MASK = 0x000f000f` 恰好能同时从低半部和高半部各取一个 4-bit 值，得到一对 `(even, odd)` 元素，形成 `half2` 用于后续向量化运算。

### 3.3 标量反量化路径（awq_gemm_fast.cu）

在 decode（M=1）路径的 `awq_gemv_fast_kernel` 中（第 170-226 行），反量化采用标量方式并融合在计算循环中：

```cpp
// 1. 按组预计算 scale 和 -scale * zero（避免重复计算）
float s[8], neg_sz[8];
for (int i = 0; i < 8; i++) {
    s[i] = __half2float(scale_half[i]);
    int z = (qz >> (awq_order[i] * 4)) & 0xF;  // 按 AWQ 位序提取零点
    neg_sz[i] = -s[i] * (float)z;               // 预计算 -scale * zero
}

// 2. 内层循环：解包 + 反量化 + 累加 融合在一起
for (int k = lane_id; k < group_size; k += 32) {
    float x = __half2float(__ldg(&X[k_idx]));
    const int32_t w_packed = __ldg(&qweight[k_idx * packed_N + packed_out_idx]);
    
    for (int i = 0; i < 8; i++) {
        int w = (w_packed >> (awq_order[i] * 4)) & 0xF;  // 位移 + 掩码提取 INT4
        // FMA 优化: acc += x * s * w + x * (-s * z)
        //         = x * scale * (w - zero)
        acc[i] = fmaf(x * s[i], (float)w, acc[i] + x * neg_sz[i]);
    }
}
```

**融合策略**：
1. 零点的 `scale * zero` 在组级别预计算（amortized over `group_size = 128` 次迭代）
2. 反量化公式 `scale * (w - z)` 被拆解为 `scale * w - scale * z`，利用 FMA 单指令完成
3. 整个过程在寄存器中完成，无需写回 shared memory 或 global memory

### 3.4 向量化 LOP3 反量化路径（awq_gemm_vllm.cu）

在 prefill 路径的 `dequant_vllm_lop3` 函数中（第 37-96 行），INT4 解包和反量化通过 LOP3 + half2 向量运算融合实现。完整流程分为 3 步：

#### 第 1 步：LOP3 位提取 + FP16 格式嵌入

```cpp
constexpr uint32_t FP16_TOP_MAGIC = 0x64006400;  // half2(1024.0, 1024.0)
constexpr uint32_t BOTTOM_MASK = 0x000f000f;
constexpr uint32_t I4s_TO_FP16_MAGIC = 0x64006400;

// LOP3: result = (packed_w & BOTTOM_MASK) | I4s_TO_FP16_MAGIC
asm volatile("lop3.b32 %0, %1, %2, %3, 0xea;" 
    : "=r"(w_tmp1) : "r"(packed_w), "n"(BOTTOM_MASK), "n"(I4s_TO_FP16_MAGIC));
```

**原理详解**：

`BOTTOM_MASK = 0x000f000f` 作为一个 `half2` 来看，低 16 位的 `0x000f` 和高 16 位的 `0x000f` 各保留一个 half 的低 4 bit。

`I4s_TO_FP16_MAGIC = 0x64006400` 对应 FP16 值 `1024.0`（指数部分 = 0x6400），其格式为：

```
FP16 格式: s eeeee mmmmmmmmmm
0x6400:    0 11001 0000000000  → 值 = 2^(25-15) × 1.0 = 1024.0
```

LOP3 执行 `(packed_w & 0x000f000f) | 0x64006400` 后：
- 低 16 位 = `0x6400 | (w[3:0])`，即 FP16 的 `1024.0 + w_int4_value`
- 高 16 位 = `0x6400 | (w[19:16])`，即 FP16 的 `1024.0 + w_int4_value`

这样 INT4 值（0~15）被编码为 FP16 的 `1024.0 ~ 1039.0`。

#### 第 2 步：减去 Magic Number 恢复真实值

```cpp
half2 w01 = __hsub2(*reinterpret_cast<half2*>(&w_tmp1), 
                    *reinterpret_cast<const half2*>(&FP16_TOP_MAGIC));
// w01 = (1024.0 + w_int4) - 1024.0 = (float)w_int4
```

对于 `TOP_MASK = 0x00f000f0` 提取的 bits[4:7,20:23]，由于 4-bit 值位于半精度尾数的高 4 位（相当于乘了 16），需要额外除以 16：

```cpp
w23 = __hmul2(w23, __float2half2_rn(0.0625f));  // ÷16 校正位偏移
```

#### 第 3 步：反量化公式应用

```cpp
// output = scale * (w - z)
output[0] = __hmul2(scales_h2[0], __hsub2(w01, z01));  // 元素对 (o0, o1)
output[1] = __hmul2(scales_h2[1], __hsub2(w23, z23));  // 元素对 (o2, o3)
output[2] = __hmul2(scales_h2[2], __hsub2(w45, z45));  // 元素对 (o4, o5)
output[3] = __hmul2(scales_h2[3], __hsub2(w67, z67));  // 元素对 (o6, o7)
```

使用 `half2` 运算，每次操作同时处理 2 个元素，8 个元素只需 4 次减法 + 4 次乘法 = 8 条 FP16 指令。

### 3.5 LOP3 反量化的完整流水线

将 LOP3 解包与反量化结合后，处理一个 INT32（8 个 INT4 权重）的完整流水线如下：

```
输入: packed_w (INT32, 8×INT4), packed_z (INT32, 8×INT4 zeros), scales (4×half2)
                                   
Step 1: 右移 8 位准备高半部分
  packed_w_hi = packed_w >> 8
  packed_z_hi = packed_z >> 8
                        ↓
Step 2: 4 × LOP3 提取权重 (每次提取2个INT4并嵌入FP16格式)
  lop3 w_tmp1 = (packed_w   & 0x000f000f) | 0x64006400  → bits[0:3,16:19]
  lop3 w_tmp2 = (packed_w   & 0x00f000f0) | 0x64006400  → bits[4:7,20:23]
  lop3 w_tmp3 = (packed_w_hi & 0x000f000f) | 0x64006400  → bits[8:11,24:27]
  lop3 w_tmp4 = (packed_w_hi & 0x00f000f0) | 0x64006400  → bits[12:15,28:31]
                        ↓
Step 3: 4 × LOP3 提取零点 (同上，对 packed_z)
  lop3 z_tmp1~4 = ...
                        ↓
Step 4: half2 减去 magic (8 × hsub2)
  w01 = w_tmp1_h2 - 1024.0h2      z01 = z_tmp1_h2 - 1024.0h2
  w23 = w_tmp2_h2 - 1024.0h2      z23 = z_tmp2_h2 - 1024.0h2
  ...
                        ↓
Step 5: 位偏移校正 (4 × hmul2，仅对 TOP_MASK 提取的)
  w23 *= 0.0625h2      z23 *= 0.0625h2
  w67 *= 0.0625h2      z67 *= 0.0625h2
                        ↓
Step 6: 反量化 (4 × hsub2 + 4 × hmul2)
  out0 = scale01 * (w01 - z01)
  out1 = scale23 * (w23 - z23)
  out2 = scale45 * (w45 - z45)
  out3 = scale67 * (w67 - z67)
                        ↓
输出: 4 × half2 = 8 个 FP16 反量化权重
```

**总指令数**：2 次移位 + 8 次 LOP3 + 8 次 hsub2 + 4 次 hmul2 + 4 次 hsub2 + 4 次 hmul2 = **约 30 条指令**处理 8 个权重。而传统标量方式需要约 48+ 条指令（每个 INT4：移位、掩码、减法、乘法、INT→FP 转换 × 8）。

---

## 4. Tensor Core MMA 原理与指令详解

### 4.1 Tensor Core 硬件原理

Tensor Core 是 NVIDIA GPU 从 Volta 架构（SM70）引入的专用矩阵运算单元。其核心能力是在**单个时钟周期**内完成一个小矩阵的**乘加运算**：

$$D = A \times B + C$$

其中 A, B, C, D 都是小尺寸矩阵片段（fragment）。

**硬件层面**：
- 每个 Tensor Core 在一个时钟周期完成 4×4×4 的 FMA（Fuse Multiply-Add）
- 一个 SM（SM80/SM86/SM89）通常包含 4 个 Tensor Core
- 一个 warp（32 个线程）协作驱动 Tensor Core 完成更大尺寸的矩阵块运算

**支持的数据类型**（以 Ampere 架构为例）：

| 输入类型 | 累加类型 | 矩阵尺寸 | 指令 |
|---------|---------|----------|------|
| FP16 | FP16/FP32 | m16n8k8, m16n8k16 | `mma.sync.aligned` |
| BF16 | FP32 | m16n8k8, m16n8k16 | `mma.sync.aligned` |
| TF32 | FP32 | m16n8k4, m16n8k8 | `mma.sync.aligned` |
| INT8 | INT32 | m16n8k16, m16n8k32 | `mma.sync.aligned` |
| INT4 | INT32 | m16n8k32, m16n8k64 | `mma.sync.aligned` |

### 4.2 MMA 指令格式与语义

本工程使用的是 **`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`** 指令。下面逐字段解析：

```
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
│   │     │      │        │   │   │   │   │   │
│   │     │      │        │   │   │   │   │   └─ D 累加类型: FP32
│   │     │      │        │   │   │   │   └─── B 输入类型: FP16
│   │     │      │        │   │   │   └─────── A 输入类型: FP16  
│   │     │      │        │   │   └─────────── C 累加类型: FP32
│   │     │      │        │   └─────────────── B 矩阵列主序 (column-major)
│   │     │      │        └─────────────────── A 矩阵行主序 (row-major)
│   │     │      └──────────────────────────── 矩阵尺寸: M=16, N=8, K=16
│   │     └─────────────────────────────────── 对齐要求
│   └───────────────────────────────────────── 线程同步 (warp 级别)
└───────────────────────────────────────────── Matrix Multiply-Accumulate
```

**语义**：一个 warp（32线程）协作计算：

$$D_{16 \times 8} = A_{16 \times 16} \times B_{16 \times 8} + C_{16 \times 8}$$

其中 A 为 FP16 行主序，B 为 FP16 列主序，C/D 为 FP32。

**PTX 语法**：

```ptx
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 
    {%0,%1,%2,%3},           // D: 4 个 FP32 输出寄存器
    {%4,%5,%6,%7},           // A: 4 个 32-bit 寄存器（存放 8 个 FP16）
    {%8,%9},                 // B: 2 个 32-bit 寄存器（存放 4 个 FP16）
    {%10,%11,%12,%13};       // C: 4 个 FP32 累加输入寄存器
```

**寄存器数量说明**：
- **A 片段**：每个线程持有 4 个 32-bit 寄存器 = 8 个 FP16 元素（16×16 矩阵的一部分）
- **B 片段**：每个线程持有 2 个 32-bit 寄存器 = 4 个 FP16 元素（16×8 矩阵的一部分）
- **C/D 片段**：每个线程持有 4 个 FP32 寄存器（16×8 结果矩阵的一部分）

### 4.3 ldmatrix 指令详解

`ldmatrix` 是 Tensor Core 生态中的关键辅助指令，用于**高效地从 shared memory 加载矩阵片段到寄存器**，其布局恰好匹配 MMA 指令的输入要求。

**指令格式**：

```ptx
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {r0, r1, r2, r3}, [addr];
```

- `.m8n8`：每个矩阵片段为 8×8
- `.x4`：一次加载 4 个 m8n8 片段
- `.shared`：从 shared memory 读取
- `.b16`：每个元素 16-bit（FP16）

变体 `.trans` 用于加载的同时做转置：

```ptx
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {r0, r1, r2, r3}, [addr];
```

**工程中的使用**（`awq_gemm_vllm.cu` 第 172-188 行）：

```cpp
// 加载 A 矩阵片段（不转置）
asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
    : "=r"(((unsigned*)(A_shared_warp))[0]), "=r"(((unsigned*)(A_shared_warp))[1]),
      "=r"(((unsigned*)(A_shared_warp))[2]), "=r"(((unsigned*)(A_shared_warp))[3])
    : "r"(addr));

// 加载 B 矩阵片段（带转置）
asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
    : "=r"(((unsigned*)(B_shared_warp + ax1_0 * 8))[0]), 
      "=r"(((unsigned*)(B_shared_warp + ax1_0 * 8))[1]),
      "=r"(((unsigned*)(B_shared_warp + ax1_0 * 8))[2]), 
      "=r"(((unsigned*)(B_shared_warp + ax1_0 * 8))[3])
    : "r"(addr));
```

A 矩阵（激活值）使用不转置的 `ldmatrix`，因为 A 是行主序；B 矩阵（权重）使用带 `.trans` 的 `ldmatrix`，将行主序的 shared memory 数据转置为列主序以匹配 MMA 的 `.col` 要求。

### 4.4 工程中 Tensor Core MMA 的完整使用流程

`awq_gemm_vllm_kernel` 模板函数（`awq_gemm_vllm.cu` 第 119-240 行）是工程中 Tensor Core 的主要使用场景。以 N=128 为例，完整流程如下：

#### 阶段 1：共享内存声明与初始化

```cpp
float C_warp[32];                          // 每个线程的累加寄存器
__shared__ half A_shared[16 * (32 + 8)];   // A 共享内存 (16×40, padded)
__shared__ half B_shared[32 * (128 + 8)];  // B 共享内存 (32×136, padded)

half A_shared_warp[8];        // A 的 warp 级片段寄存器
half B_shared_warp[128 / 4];  // B 的 warp 级片段寄存器 = 32 个 half
```

共享内存的 padding（+8）是为了避免 bank conflict。

#### 阶段 2：K 维度分块迭代

```cpp
int k_bound = IC / 32;  // K 方向分块，每块 32

for (int k_0_0 = 0; k_0_0 < k_bound; ++k_0_0) {
    __syncthreads();
```

#### 阶段 3：加载 A 矩阵到共享内存

```cpp
// 从 global 加载 16×32 的 FP16 激活值块到共享内存
if (ld_A_flag) 
    *(uint4*)(A_shared_ptr) = *(uint4*)(A_ptr + k_0_0 * 32);
else 
    *(uint4*)(A_shared_ptr) = make_uint4(0, 0, 0, 0);
```

使用 `uint4`（128-bit）向量化加载，每次读取 8 个 FP16 元素。

#### 阶段 4：加载 B 权重并反量化到共享内存

```cpp
// 加载量化权重、零点、缩放因子
uint32_t zeros_loaded = *(uint32_t*)(zeros_ptr + k_0_0 * 32 / G * (OC / 8));
half* scales_loaded = sf_ptr + k_0_0 * 32 / G * OC;
int* B_ptr_local = B_ptr + k_0_0 * 32 * (OC / 8);

for (int ax0 = 0; ax0 < N / 16; ++ax0) {
    uint32_t B_loaded = *(uint32_t*)(B_ptr_local + ax0 * row_stride * (OC / 8));
    
    // 使用 LOP3 快速反量化
    half2 B_dequant_h2[4];
    half2* scales_h2 = reinterpret_cast<half2*>(scales_loaded);
    dequant_vllm_lop3(B_loaded, zeros_loaded, scales_h2, B_dequant_h2);
    
    // 写入共享内存（128-bit 向量化写入）
    *(uint4*)(B_shared_ptr + ax0 * row_stride * (N + 8)) = *(uint4*)B_dequant_h2;
}
```

**关键：LOP3 反量化在写入共享内存之前完成**，这意味着共享内存中存储的是已经反量化的 FP16 权重，后续 MMA 可以直接使用。

#### 阶段 5：ldmatrix 加载到 warp 级寄存器

```cpp
__syncthreads();

for (int k_0_1 = 0; k_0_1 < 2; ++k_0_1) {
    // 计算共享内存地址（需要转为 generic → shared 地址空间）
    unsigned int addr;
    asm volatile(
        "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
        : "=r"(addr)
        : "l"((void*)((&(A_shared[(k_0_1 * 16)])) + 
              ((threadIdx.x & 15) * 40) + ((threadIdx.x >> 4) * 8))));
    
    // ldmatrix 加载 A 片段（4 个 m8n8 块，不转置）
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(((unsigned*)(A_shared_warp))[0]), "=r"(((unsigned*)(A_shared_warp))[1]),
          "=r"(((unsigned*)(A_shared_warp))[2]), "=r"(((unsigned*)(A_shared_warp))[3])
        : "r"(addr));
    
    // ldmatrix 加载 B 片段（4 个 m8n8 块，带转置）
    for (int ax1_0 = 0; ax1_0 < N / 32; ++ax1_0) {
        // ... 计算地址 ...
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(((unsigned*)(B_shared_warp + ax1_0 * 8))[0]), ...
            : "r"(addr));
    }
```

**地址计算细节**：

```cpp
// A 矩阵的共享内存寻址
// threadIdx.x & 15: warp 内线程在 16 行中的位置
// threadIdx.x >> 4: 0 或 1，选择 K 方向的偏移
// * 40: 每行 32+8 = 40 个 half 元素（含 padding）
((threadIdx.x & 15) * 40) + ((threadIdx.x >> 4) * 8)
```

`cvta.to.shared.u64` 指令将通用地址空间转换为 shared memory 地址空间，这是 `ldmatrix` 指令所要求的。

#### 阶段 6：MMA 矩阵乘加

```cpp
for (int j_0_4 = 0; j_0_4 < N / 32; ++j_0_4) {
    // 第一次 MMA: m16n8k16
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(((float*)(C_warp + j_0_4 * 8))[0]),      // D[0]
          "=f"(((float*)(C_warp + j_0_4 * 8))[1]),      // D[1]
          "=f"(((float*)(C_warp + j_0_4 * 8))[2]),      // D[2]
          "=f"(((float*)(C_warp + j_0_4 * 8))[3])       // D[3]
        : "r"(((unsigned*)(A_shared_warp))[0]),    // A 片段寄存器 [0]
          "r"(((unsigned*)(A_shared_warp))[1]),    // A 片段寄存器 [1]
          "r"(((unsigned*)(A_shared_warp))[2]),    // A 片段寄存器 [2]
          "r"(((unsigned*)(A_shared_warp))[3]),    // A 片段寄存器 [3]
          "r"(((unsigned*)(B_shared_warp + j_0_4 * 8))[0]),  // B 片段 [0]
          "r"(((unsigned*)(B_shared_warp + j_0_4 * 8))[1]),  // B 片段 [1]
          "f"(((float*)(C_warp + j_0_4 * 8))[0]),  // C 累加 [0]
          "f"(((float*)(C_warp + j_0_4 * 8))[1]),  // C 累加 [1]
          "f"(((float*)(C_warp + j_0_4 * 8))[2]),  // C 累加 [2]
          "f"(((float*)(C_warp + j_0_4 * 8))[3])   // C 累加 [3]
    );
    
    // 第二次 MMA: 处理 N 方向的下一个 8 列
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(((float*)(C_warp + j_0_4 * 8 + 4))[0]), ...
        : "r"(((unsigned*)(A_shared_warp))[0]), ...  // 共用同一份 A
          "r"(((unsigned*)(B_shared_warp + j_0_4 * 8 + 4))[0]), ...  // B 的下一个片段
          "f"(((float*)(C_warp + j_0_4 * 8 + 4))[0]), ...
    );
}
```

**关键观察**：
1. 两次 MMA 共用同一份 A 片段（A 是 16×16），但使用不同的 B 片段，这样合并计算 16×16 的 N 方向输出
2. `j_0_4` 循环处理 N/32 个 16 列的输出块，对于 N=128 有 4 次迭代
3. 每次 MMA 计算 16×8 的子矩阵，两次 MMA 合并为 16×16

#### 阶段 7：结果写回 Global Memory

```cpp
for (int ax1 = 0; ax1 < N / 32; ++ax1) {
    for (int local_id = 0; local_id < 8; ++local_id) {
        int row_offset = (blockIdx_y / j_factors1) * 16 + 
                         (threadIdx.x / 4) + (local_id % 4) / 2 * 8;
        if (row_offset < M) {
            *(C_ptr + ax1 * 16 + row_offset * OC + 
              (local_id / 4) * 8 + local_id % 2) =
                __float2half(C_warp[ax1 * 8 + local_id]);
        }
    }
}
```

每个线程将自己的 FP32 累加结果转换为 FP16 写回全局内存。

### 4.5 MMA 指令中的数据布局与线程映射

`m16n8k16` MMA 指令中，32 个线程如何分布在 16×8 的输出矩阵上：

```
          ┌── N=8 列 ──┐
     col:  0  1  2  3  4  5  6  7
row 0-1:  T0 T0 T1 T1 T0 T0 T1 T1     ← 线程 0,1
row 2-3:  T2 T2 T3 T3 T2 T2 T3 T3     ← 线程 2,3
row 4-5:  T4 T4 T5 T5 T4 T4 T5 T5     ← 线程 4,5
row 6-7:  T6 T6 T7 T7 T6 T6 T7 T7     ← 线程 6,7
row 8-9:  T0 T0 T1 T1 T0 T0 T1 T1     ← 线程 0,1 (重复)
row10-11: T2 T2 T3 T3 T2 T2 T3 T3
row12-13: T4 T4 T5 T5 T4 T4 T5 T5
row14-15: T6 T6 T7 T7 T6 T6 T7 T7
```

（上图中的 T0-T7 代表 threadIdx.x % 8，完整的 32 线程由 threadIdx.x / 4 确定行偏移）

每个线程持有 4 个 FP32 输出寄存器，对应 16×8 矩阵中的 4 个分散位置。写回时需要根据 `threadIdx.x / 4` 和 `local_id` 计算全局坐标。

### 常用 Tensor Core 指令总结

| 指令 | 功能 | 本工程使用 |
|------|------|-----------|
| `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` | 16×8×16 FP16 矩阵乘加 | ✅ 主计算指令 |
| `ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 从共享内存加载 4 个 8×8 矩阵片段 | ✅ 加载 A 矩阵 |
| `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16` | 加载并转置 4 个 8×8 矩阵片段 | ✅ 加载 B 矩阵 |
| `cvta.to.shared.u64` | 通用地址→共享内存地址转换 | ✅ ldmatrix 前的地址转换 |
| `lop3.b32` | 三输入位逻辑操作 | ✅ INT4→FP16 解包 |
| `sub.f16x2` | FP16×2 向量减法 | ✅ 去除 magic number |
| `fma.rn.f16x2` | FP16×2 融合乘加 | ✅ 位偏移校正 |

---

## 附录：关键源文件路径

| 文件 | 说明 |
|------|------|
| `kuiper/include/op/awq_matmul.h` | AWQ 算子头文件定义 |
| `kuiper/source/op/awq_matmul.cpp` | AWQ 算子实现（权重加载、forward） |
| `kuiper/source/op/kernels/cuda/awq_gemm_tensorcore.cu` | 顶层分发器 |
| `kuiper/source/op/kernels/cuda/awq_gemm_fast.cu` | Decode 路径 GEMV/GEMM 内核 |
| `kuiper/source/op/kernels/cuda/awq_gemm_vllm.cu` | Prefill 路径 Tensor Core 内核 |
| `kuiper/source/model/qwen3.cpp` | 模型定义与 AWQ 权重加载 |
| `kuiper/source/model/qwen_base.cpp` | 推理管线（forward/attention/feedforward） |
| `demo/main_qwen3.cpp` | 推理入口 |

---

## 5. 工程中 Tensor Core MMA 的具体使用（源码级详解）

### 5.1 Tensor Core MMA 的调用位置

在本工程中，**Tensor Core MMA 仅在 Prefill 路径的 `awq_gemm_vllm_kernel` 内核中使用**。调用链为：

```
AWQMatmulLayer::forward()                    // awq_matmul.cpp
  → kernel::awq_gemm_tensorcore_cu()         // awq_gemm_tensorcore.cu
    → [M > 1] awq_gemm_vllm_cu()            // awq_gemm_vllm.cu
      → awq_gemm_vllm_kernel<64/128>()       // ★ 此处使用 Tensor Core MMA
```

Decode 路径（M=1）使用的 `awq_gemm_fast_cu()` 系列内核**不使用 Tensor Core**，而是通过标量/向量化操作和 warp shuffle 实现 GEMV。

### 5.2 内核整体架构

`awq_gemm_vllm_kernel<N>` 是一个模板内核，N 取 64 或 128，表示每个 thread block 处理的输出列数：

```cpp
template <int N>  // N = 64 or 128
__global__ void __launch_bounds__(64)
awq_gemm_vllm_kernel(
    int G, half* A, int* B,
    half* scaling_factors, int* zeros,
    int M, int IC, int OC, half* C
);
```

- **线程配置**：`dim3(32, 2)` = 64 个线程 = 2 个 warp
- **Tile 尺寸**：每个 block 处理 M 方向 16 行 × N 方向 64/128 列
- **K 维度循环**：每次迭代处理 32 个 K 元素

关键数据结构：

```cpp
float C_warp[32];                           // 每个线程的累加器（FP32）
__shared__ half A_shared[16 * (32 + 8)];    // A 矩阵共享内存（padding 防 bank conflict）
__shared__ half B_shared[32 * (N + 8)];     // B 矩阵共享内存（已解量化）
half A_shared_warp[8];                      // A 矩阵寄存器fragment
half B_shared_warp[N / 4];                  // B 矩阵寄存器fragment
```

### 5.3 PTX 指令详解：`cvta.to.shared.u64`

在使用 `ldmatrix` 指令前，必须将通用地址空间的指针转换为共享内存地址空间。源码中的实现（`awq_gemm_vllm.cu` 第 170-173 行）：

```cpp
unsigned int addr;
asm volatile(
    "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
    : "=r"(addr)
    : "l"((void*)((&(A_shared[(k_0_1 * 16)])) +
           ((threadIdx.x & 15) * 40) + ((threadIdx.x >> 4) * 8))));
```

**指令分解**：

| 步骤 | PTX 指令 | 含义 |
|------|----------|------|
| 1 | `cvta.to.shared.u64 addr, %1` | 将 64 位通用地址 `%1` 转换为共享内存地址空间 |
| 2 | `cvt.u32.u64 %0, addr` | 将 64 位地址截断为 32 位（`ldmatrix` 要求 32 位地址） |

**地址计算逻辑**：

对于 A 矩阵，共享内存布局为 `A_shared[16][40]`（含 padding），warp 中 32 个线程的访问模式为：

- `threadIdx.x & 15`：线程在 16 行中的行号（0-15）
- `threadIdx.x >> 4`：thread group 内偏移（0 或 1），每组偏移 8 个 half 元素
- `k_0_1 * 16`：K 维度内层循环的起始偏移（0 或 16）

这确保了 warp 内 32 个线程覆盖 16 行 × 2 列组的数据，匹配 `ldmatrix` 的 warp 级合作加载模式。

### 5.4 PTX 指令详解：`ldmatrix.sync.aligned`

`ldmatrix` 是 Tensor Core 专用的 warp 级矩阵片段加载指令，一次加载多个 8×8 矩阵片段到寄存器。

#### 5.4.1 加载 A 矩阵（非转置）

源码位置：`awq_gemm_vllm.cu` 第 174-178 行：

```cpp
asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
    : "=r"(((unsigned*)(A_shared_warp))[0]), "=r"(((unsigned*)(A_shared_warp))[1]),
      "=r"(((unsigned*)(A_shared_warp))[2]), "=r"(((unsigned*)(A_shared_warp))[3])
    : "r"(addr));
```

**指令语义**：

```
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {r0, r1, r2, r3}, [addr];
```

| 修饰符 | 含义 |
|--------|------|
| `.sync` | warp 内所有线程同步参与 |
| `.aligned` | 地址对齐（16 字节） |
| `.m8n8` | 每个片段为 8×8 矩阵 |
| `.x4` | 一次加载 4 个 8×8 片段 |
| `.shared` | 从共享内存加载 |
| `.b16` | 数据类型为 16 位（FP16） |

**寄存器分配**：4 个 32 位寄存器 `{r0, r1, r2, r3}`，每个包含 2 个 FP16 值（`half2`），共 8 个 FP16 → 对应 `A_shared_warp[8]`。

**warp 内数据分布**：对于 `m16n8k16` MMA 所需的 A 矩阵片段 (16×16)：
- 4 个 8×8 片段组成 16×16 矩阵
- warp 中每个线程持有不同行的元素
- 线程 `t` 持有 A 矩阵中第 `t%16` 行和第 `t/16 * 2 + {0,1}` 列的两个元素

#### 5.4.2 加载 B 矩阵（转置）

源码位置：`awq_gemm_vllm.cu` 第 180-187 行：

```cpp
asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
    : "=r"(((unsigned*)(B_shared_warp + ax1_0 * 8))[0]),
      "=r"(((unsigned*)(B_shared_warp + ax1_0 * 8))[1]),
      "=r"(((unsigned*)(B_shared_warp + ax1_0 * 8))[2]),
      "=r"(((unsigned*)(B_shared_warp + ax1_0 * 8))[3])
    : "r"(addr));
```

**关键区别**：此处多了 `.trans` 修饰符。

| 对比 | A 矩阵 | B 矩阵 |
|------|--------|--------|
| 修饰符 | `.m8n8.x4.shared.b16` | `.m8n8.x4.trans.shared.b16` |
| 转置 | 否 | 是 |
| MMA 中的角色 | row-major（行主序） | col-major（列主序） |
| 共享内存布局 | 按行存储 | 按行存储 → 加载时自动转置 |

`.trans` 指令在加载时自动完成矩阵转置，避免了在共享内存中手动转置 B 矩阵的开销。这是因为 MMA 指令要求 B 矩阵以列主序提供（`.col` 操作数），而 B 在共享内存中是按行存储的（解量化后直接写入）。

**B 矩阵循环**：当 N=128 时，`ax1_0` 从 0 到 3 循环（`N/32 = 4`），每次加载一个 16×8 的 B 片段，覆盖 128 列输出中的 32 列。

### 5.5 PTX 指令详解：`mma.sync.aligned.m16n8k16`

这是核心计算指令，执行 16×8×16 的矩阵乘累加运算。

#### 5.5.1 第一条 MMA 指令（下半部分 N）

源码位置：`awq_gemm_vllm.cu` 第 190-198 行：

```cpp
asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=f"(((float*)(C_warp + j_0_4 * 8))[0]),    // D[0]
      "=f"(((float*)(C_warp + j_0_4 * 8))[1]),    // D[1]
      "=f"(((float*)(C_warp + j_0_4 * 8))[2]),    // D[2]
      "=f"(((float*)(C_warp + j_0_4 * 8))[3])     // D[3]
    : "r"(((unsigned*)(A_shared_warp))[0]),          // A[0] (half2)
      "r"(((unsigned*)(A_shared_warp))[1]),          // A[1] (half2)
      "r"(((unsigned*)(A_shared_warp))[2]),          // A[2] (half2)
      "r"(((unsigned*)(A_shared_warp))[3]),          // A[3] (half2)
      "r"(((unsigned*)(B_shared_warp + j_0_4*8))[0]),  // B[0] (half2)
      "r"(((unsigned*)(B_shared_warp + j_0_4*8))[1]),  // B[1] (half2)
      "f"(((float*)(C_warp + j_0_4 * 8))[0]),       // C[0] (累加器)
      "f"(((float*)(C_warp + j_0_4 * 8))[1]),       // C[1]
      "f"(((float*)(C_warp + j_0_4 * 8))[2]),       // C[2]
      "f"(((float*)(C_warp + j_0_4 * 8))[3])        // C[3]
);
```

#### 5.5.2 第二条 MMA 指令（上半部分 N）

源码位置：`awq_gemm_vllm.cu` 第 199-210 行：

```cpp
asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=f"(((float*)(C_warp + j_0_4 * 8 + 4))[0]),    // D[4]
      "=f"(((float*)(C_warp + j_0_4 * 8 + 4))[1]),    // D[5]
      "=f"(((float*)(C_warp + j_0_4 * 8 + 4))[2]),    // D[6]
      "=f"(((float*)(C_warp + j_0_4 * 8 + 4))[3])     // D[7]
    : "r"(((unsigned*)(A_shared_warp))[0]),              // A[0] （同一A片段）
      "r"(((unsigned*)(A_shared_warp))[1]),              // A[1]
      "r"(((unsigned*)(A_shared_warp))[2]),              // A[2]
      "r"(((unsigned*)(A_shared_warp))[3]),              // A[3]
      "r"(((unsigned*)(B_shared_warp + j_0_4*8 + 4))[0]), // B[2] (下一组列)
      "r"(((unsigned*)(B_shared_warp + j_0_4*8 + 4))[1]), // B[3]
      "f"(((float*)(C_warp + j_0_4 * 8 + 4))[0]),       // C[4]
      "f"(((float*)(C_warp + j_0_4 * 8 + 4))[1]),       // C[5]
      "f"(((float*)(C_warp + j_0_4 * 8 + 4))[2]),       // C[6]
      "f"(((float*)(C_warp + j_0_4 * 8 + 4))[3])        // C[7]
);
```

**指令修饰符详解**：

```
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
       │     │       │      │   │   │   │   │   │
       │     │       │      │   │   │   │   │   └─ D 类型：FP32（累加结果）
       │     │       │      │   │   │   │   └──── B 类型：FP16
       │     │       │      │   │   │   └─────── A 类型：FP16
       │     │       │      │   │   └──────────── C 类型：FP32（累加输入）
       │     │       │      │   └──────────────── B 布局：col-major
       │     │       │      └──────────────────── A 布局：row-major
       │     │       └─────────────────────────── 矩阵尺寸：M=16, N=8, K=16
       │     └─────────────────────────────────── 数据对齐
       └───────────────────────────────────────── warp 同步
```

**寄存器映射**：

| 操作数 | 寄存器数 | 类型 | 含义 |
|--------|----------|------|------|
| A | 4 × `u32` | 每个含 2 × FP16 | 16×16 矩阵的 A 片段（每线程 8 个 FP16） |
| B | 2 × `u32` | 每个含 2 × FP16 | 16×8 矩阵的 B 片段（每线程 4 个 FP16） |
| C/D | 4 × `f32` | FP32 | 16×8 结果矩阵的一部分（每线程 4 个 FP32） |

**两条 MMA 的组合效果**：

每次 `j_0_4` 迭代执行两条 MMA 指令，A 矩阵片段相同，B 矩阵使用不同列组：
- 第 1 条 MMA：计算 16×8 结果块 → `C_warp[j_0_4*8 + 0..3]`
- 第 2 条 MMA：计算另一个 16×8 结果块 → `C_warp[j_0_4*8 + 4..7]`

合计每次迭代产出 **16×16** 输出元素（对应 N 维度的 16 列）。

### 5.6 完整计算流程

以 N=128 为例，完整的内层循环结构：

```
外层循环: k_0_0 = 0 .. IC/32-1  (每次处理 K=32)
│
├─ 1. 加载 A[16×32] → A_shared (uint4 向量化)
├─ 2. 加载/解量化 B[32×128] → B_shared (LOP3 dequant)
├─ __syncthreads()
│
└─ 内层循环: k_0_1 = 0, 1  (每次处理 K=16)
   │
   ├─ 3. cvta + ldmatrix: A_shared → A_shared_warp[8] (非转置)
   │
   ├─ 4. 循环 ax1_0 = 0..3: (N/32 = 128/32 = 4)
   │     cvta + ldmatrix.trans: B_shared → B_shared_warp[32] (转置)
   │
   └─ 5. 循环 j_0_4 = 0..3: (N/32 = 4)
         ├─ mma.sync m16n8k16 (列 0-7)   → C_warp[j*8 + 0..3]
         └─ mma.sync m16n8k16 (列 8-15)  → C_warp[j*8 + 4..7]

每次外层迭代: 2 × 4 × 2 = 16 条 MMA 指令
每条 MMA: 16×8×16 = 2048 FMA
总计: 16 × 2048 = 32768 FMA / 外层迭代
```

### 5.7 MMA 计算的数据流图

```
Global Memory (INT4 packed)
     │
     ▼ uint32 加载 + LOP3 dequant_vllm_lop3()
     │
Shared Memory (FP16)
  ┌──────────────────────────────────────┐
  │ A_shared[16][40]  B_shared[32][136]  │  ← 含 padding
  └──────────────────────────────────────┘
     │                    │
     ▼ ldmatrix(非转置)    ▼ ldmatrix.trans(转置)
     │                    │
Registers (FP16)         Registers (FP16)
  A_shared_warp[8]       B_shared_warp[32]
     │                    │
     └────────┬───────────┘
              ▼
     mma.sync.aligned.m16n8k16
              │
              ▼
     C_warp[32] (FP32 累加器)
              │
              ▼ __float2half() 转换
              │
     Global Memory C (FP16 输出)
```

### 5.8 线程-数据映射关系

在 `m16n8k16` MMA 中，warp 内 32 个线程的数据映射规则：

**A 矩阵 (16×16, row-major)**：
```
线程 t (0 ≤ t ≤ 31):
  组号 = t / 4          → 决定持有的行区间
  组内偏移 = t % 4      → 决定持有的列区间
  
  A[0]: (row[组号*2],   col[组内偏移*2]),   (row[组号*2],   col[组内偏移*2+1])
  A[1]: (row[组号*2+1], col[组内偏移*2]),   (row[组号*2+1], col[组内偏移*2+1])
  A[2]: (row[组号*2+8], col[组内偏移*2]),   (row[组号*2+8], col[组内偏移*2+1])
  A[3]: (row[组号*2+9], col[组内偏移*2]),   (row[组号*2+9], col[组内偏移*2+1])
```

**B 矩阵 (16×8, col-major)**：
```
线程 t:
  B[0]: (row[t%4*2],   col[t/4])
  B[1]: (row[t%4*2+1], col[t/4])
```

**C/D 矩阵 (16×8, row-major)**：
```
线程 t:
  C[0]: (row[t/4],     col[(t%4)*2])     → 即 (row[t/4], col[t%4*2])
  C[1]: (row[t/4],     col[(t%4)*2+1])
  C[2]: (row[t/4 + 8], col[(t%4)*2])
  C[3]: (row[t/4 + 8], col[(t%4)*2+1])
```

### 5.9 输出回写

MMA 计算完成后，每个线程将其 `C_warp` 中的 FP32 结果转换为 FP16 并写回全局内存：

```cpp
for (int ax1 = 0; ax1 < N / 32; ++ax1) {
    for (int local_id = 0; local_id < 8; ++local_id) {
        int row_offset = (blockIdx_y / j_factors1) * 16 +
                         (threadIdx.x / 4) + (local_id % 4) / 2 * 8;
        if (row_offset < M) {
            *(C_ptr + ax1 * 16 + row_offset * OC + (local_id / 4) * 8 + local_id % 2) =
                __float2half(C_warp[ax1 * 8 + local_id]);
        }
    }
}
```

`local_id` 的映射关系：
- `local_id % 4 / 2 * 8`：在 M 维度上区分上半部分（行 0-7）和下半部分（行 8-15）
- `local_id / 4 * 8`：在 N 维度上区分前 8 列和后 8 列
- `local_id % 2`：N 维度内的精确列偏移

---

## 6. `awq_matmul.cpp` 中使用的 CUDA Kernel 详解

### 6.1 整体调用链与分发逻辑

`awq_matmul.cpp` 中的 `AWQMatmulLayer::forward()` 方法是所有 AWQ 矩阵乘法的入口。其完整的 kernel 调用链如下：

```
AWQMatmulLayer::forward()          [awq_matmul.cpp:105-133]
  │
  └→ kernel::awq_gemm_tensorcore_cu()  [awq_gemm_tensorcore.cu:51-73]
       │
       ├─ [M == 1] → awq_gemm_fast_cu()      [awq_gemm_fast.cu:556-589]
       │    ├─ [M == 1]  → awq_gemv_fast_kernel      ← Kernel ①
       │    ├─ [M ≤ 8]   → awq_gemm_small_batch_kernel  ← Kernel ②
       │    └─ [M > 8]   → awq_gemm_fast_kernel      ← Kernel ③
       │
       └─ [M > 1]  → awq_gemm_vllm_cu()      [awq_gemm_vllm.cu:237-265]
            ├─ [N%128==0] → awq_gemm_vllm_kernel<128>  ← Kernel ④a
            └─ [N%64==0]  → awq_gemm_vllm_kernel<64>   ← Kernel ④b
```

**顶层分发 (`awq_gemm_tensorcore.cu`)**：

```cpp
void awq_gemm_tensorcore_cu(..., int M, ...) {
    ensure_initialized();     // 初始化 cuBLAS handle
    if (M == 1) {
        awq_gemm_fast_cu(...);   // Decode: 标量/SIMT 路径
    } else {
        awq_gemm_vllm_cu(...);   // Prefill: Tensor Core 路径
    }
}
```

分发策略简洁明了：**M=1 走带宽优化的 GEMV/小矩阵路径，M>1 走 Tensor Core 路径**。

### 6.2 AWQ 权重格式与反量化公式

在深入内核之前，先明确 AWQ INT4 的打包格式：

**一个 INT32 打包 8 个 INT4 权重**，AWQ 使用特殊的位排列顺序：

```
输出索引:    0    1    2    3    4    5    6    7
位置(bit):  0:3  16:19 4:7  20:23 8:11 24:27 12:15 28:31
AWQ order:   0    4    1    5    2    6    3    7
```

**反量化公式**：
$$\text{output}[i] = \text{scale}[i] \times (w[i] - z[i])$$

其中 $w[i]$ 和 $z[i]$ 分别是权重和零点的 INT4 值（0-15），scale 为 FP16 缩放因子，按 group 共享（每 `group_size=128` 个 K 元素共用一组 scale/zero）。

### 6.3 Kernel ①：`awq_gemv_fast_kernel` — Decode 单行向量乘

#### 6.3.1 适用场景

自回归 Decode 阶段 (M=1)，输入为单行向量 X[K]，输出为 Y[N]。这是推理中最频繁的路径。

#### 6.3.2 启动配置

```cpp
// awq_gemm_fast.cu 第 559-563 行
const int num_blocks = (N + 63) / 64;
awq_gemv_fast_kernel<<<num_blocks, 256, 0, stream>>>(...);
```

- **Block 大小**：256 线程 = 8 个 warp
- **Grid 大小**：`ceil(N / 64)` 个 block
- **每个 warp**：处理 8 个输出通道（1 个 packed INT32）
- **每个 block**：处理 8 × 8 = 64 个输出通道

#### 6.3.3 源码逐行注释

```cpp
__global__ __launch_bounds__(256, 4)       // 256 threads, 至少4个block常驻
void awq_gemv_fast_kernel(
    const half* X, const int32_t* qweight, const int32_t* qzeros,
    const half* scales, half* Y, int K, int N, int group_size
) {
    // ---- 1. 线程定位 ----
    const int warp_id = threadIdx.x / 32;       // warp 编号 (0-7)
    const int lane_id = threadIdx.x % 32;       // warp 内线程号 (0-31)

    // 每个 warp 负责 1 个 packed INT32 = 8 个输出通道
    const int packed_out_idx = blockIdx.x * 8 + warp_id;
    const int out_base = packed_out_idx * 8;    // 输出起始列号
    if (out_base >= N) return;

    // ---- 2. 累加器初始化 ----
    float acc[8] = {0};     // 8 个 FP32 累加器，对应 8 个输出

    // ---- 3. 按 group 遍历 K 维度 ----
    for (int g = 0; g < n_groups; g++) {
        // 3a. 加载本 group 的零点（1 个 INT32 = 8 个 INT4 零点）
        const int32_t qz = __ldg(&qzeros[g * packed_N + packed_out_idx]);

        // 3b. 加载 8 个 FP16 scale（uint4 向量化读取 = 16 字节）
        uint4 scale_vec = *(const uint4*)(&scales[g * N + out_base]);
        half* scale_half = (half*)&scale_vec;

        // 3c. 预计算 s[i] 和 -s[i]*z[i]，用于后续 FMA 优化
        float s[8], neg_sz[8];
        for (int i = 0; i < 8; i++) {
            s[i] = __half2float(scale_half[i]);
            int z = (qz >> (awq_order[i] * 4)) & 0xF;
            neg_sz[i] = -s[i] * (float)z;  // 预计算，避免内层循环重复计算
        }

        // 3d. warp 内 32 个线程并行处理 K 维度
        for (int k = lane_id; k < group_size; k += 32) {
            float x = __half2float(__ldg(&X[group_start + k]));
            const int32_t w_packed = __ldg(&qweight[(group_start + k) * packed_N + packed_out_idx]);

            for (int i = 0; i < 8; i++) {
                int w = (w_packed >> (awq_order[i] * 4)) & 0xF;
                // FMA 优化: x * s * w + (acc + x * neg_sz)
                // 等价于: acc += x * s * (w - z)
                acc[i] = fmaf(x * s[i], (float)w, acc[i] + x * neg_sz[i]);
            }
        }
    }

    // ---- 4. Warp Shuffle 归约 ----
    for (int offset = 16; offset > 0; offset /= 2) {
        for (int i = 0; i < 8; i++) {
            acc[i] += __shfl_down_sync(0xffffffff, acc[i], offset);
        }
    }
    // 5 轮 shuffle: 32→16→8→4→2→1, 最终 lane_id=0 持有完整结果

    // ---- 5. 写回输出 ----
    if (lane_id == 0) {
        half out_half[8];
        for (int i = 0; i < 8; i++) out_half[i] = __float2half(acc[i]);
        *(uint4*)(&Y[out_base]) = *(uint4*)out_half;  // 向量化写入 16 字节
    }
}
```

#### 6.3.4 实际例子：Qwen3-8B q_proj Decode

**场景**：Qwen3-8B 的 `q_proj` 线性层，输入维度 K=4096，输出维度 N=4096，group_size=128。

```
参数计算：
  packed_N = N / 8 = 4096 / 8 = 512
  n_groups = K / group_size = 4096 / 128 = 32
  num_blocks = (4096 + 63) / 64 = 64
  threads_per_block = 256

Grid: <<<64, 256>>>
  → 64 blocks × 8 warps = 512 warps
  → 每个 warp 处理 8 outputs → 512 × 8 = 4096 outputs ✓

每个 warp 的工作量：
  外层循环: 32 groups
  内层循环: group_size / 32 = 128 / 32 = 4 次迭代
  每次迭代: 1 次 INT32 weight 读取 + 8 次 FMA
  总计: 32 × 4 × 8 = 1024 FMA / warp

内存访问：
  权重: 32 × 4 × 4B(INT32) = 512B / warp
  输入: 32 × 4 × 2B(FP16)  = 256B / warp (实际从 L2 cache hit)
  Scale: 32 × 16B(uint4)   = 512B / warp (每组加载一次)
  写出: 16B / warp
```

### 6.4 Kernel ②：`awq_gemm_small_batch_kernel` — 小批量 GEMM

#### 6.4.1 适用场景

小批量推理 (2 ≤ M ≤ 8)，如 speculative decoding 的验证阶段或 batch decode。

#### 6.4.2 启动配置

```cpp
// awq_gemm_fast.cu 第 567-571 行
const int num_blocks = (N + 63) / 64;
awq_gemm_small_batch_kernel<<<num_blocks, 256, 0, stream>>>(...);
```

配置与 GEMV 相同：256 线程，每 block 64 输出列。

#### 6.4.3 核心设计思路

与 GEMV kernel 的关键区别在于**每个 warp 同时处理 M 行**：

```cpp
// 累加器: [最多 8 行][8 个输出通道]
float acc[8][8] = {{0}};

// 内层循环: 对每个 K 位置的权重，乘以所有 M 行的输入
for (int m = 0; m < 8 && m < M; m++) {
    float x = __half2float(__ldg(&X[m * K + k_idx]));
    for (int i = 0; i < 8; i++) {
        acc[m][i] += x * dw[i];     // dw[i] = scale * (w - zero)
    }
}
```

**优化关键**：权重 `dw[i]` 只解量化一次，被 M 行输入复用。当 M=4 时，权重读取的开销被 4 行计算分摊。

#### 6.4.4 实际例子：M=4 的 Batch Decode

```
Qwen3-8B q_proj: M=4, K=4096, N=4096

Grid: <<<64, 256>>>

每个 warp 的工作量：
  累加器: 4 × 8 = 32 个 FP32 寄存器
  每次内层迭代:
    - 1 次 INT32 weight 读取
    - 8 次标量解量化
    - 4 × 8 = 32 次 FMA (4 行 × 8 输出)
  总 FMA: 32 × 4 × 32 = 4096 FMA / warp

相比 GEMV:
  GEMV 总 FMA: 1024 / warp (M=1)
  Small batch: 4096 / warp (M=4)
  权重读取量相同 → 计算访存比提高 4 倍
```

### 6.5 Kernel ③：`awq_gemm_fast_kernel` — 大批量流水线 GEMM

#### 6.5.1 适用场景

当 M > 8 时使用，但在实际推理中，`awq_gemm_tensorcore_cu` 会将 M > 1 的情况路由到 Tensor Core kernel（Kernel ④）。因此该 kernel 主要在 `awq_gemm_fast_cu` 被直接调用时使用。

#### 6.5.2 启动配置

```cpp
// awq_gemm_fast.cu 第 574-579 行
dim3 grid((N + FAST_TILE_N - 1) / FAST_TILE_N,    // N 方向 tile 数
          (M + FAST_TILE_M - 1) / FAST_TILE_M);    // M 方向 tile 数
awq_gemm_fast_kernel<<<grid, 256, 0, stream>>>(...);
```

- **Tile 大小**：FAST_TILE_M=32 × FAST_TILE_N=128 × FAST_TILE_K=32
- **线程映射**：256 线程，每个线程计算 4×4 输出元素
  - `thread_row = (tid / 32) * 4`：行方向步进 4（8 组 × 4 = 32 行 ✓）
  - `thread_col = (tid % 32) * 4`：列方向步进 4（32 × 4 = 128 列 ✓）

#### 6.5.3 双缓冲软件流水线

这是该 kernel 最重要的优化——在计算当前 tile 的同时预取下一个 tile：

```cpp
// 双缓冲共享内存
__shared__ half smem_A[2][32][36];    // +4 padding
__shared__ half smem_B[2][32][136];   // +8 padding

int buf = 0;
// 预加载第一个 tile → buffer 0
load_and_dequant(smem_A[0], smem_B[0], k_tile=0);
__syncthreads();

for (int k_tile = 0; k_tile < n_k_tiles; k_tile++) {
    int next_buf = 1 - buf;

    // ① 预取: 加载下一个 K tile 到 next_buf（与计算并行）
    if (k_tile + 1 < n_k_tiles) {
        load_and_dequant(smem_A[next_buf], smem_B[next_buf], k_tile+1);
    }

    // ② 计算: 使用当前 buf 的数据
    for (int k = 0; k < 32; k++) {
        // 从 smem_A[buf] 和 smem_B[buf] 读取
        // 计算 4×4 外积并累加到 acc[4][4]
    }

    buf = next_buf;
    __syncthreads();
}
```

**流水线时序图**：

```
K tile:   0        1        2        3       ...
         ┌────┐  ┌────┐  ┌────┐  ┌────┐
Load:    │ L0  │  │ L1  │  │ L2  │  │ L3  │
         └──┬─┘  └──┬─┘  └──┬─┘  └──┬─┘
            │       │       │       │
         ┌──▼─┐  ┌──▼─┐  ┌──▼─┐  ┌──▼─┐
Compute: │    │  │ C0  │  │ C1  │  │ C2  │  ...
         └────┘  └────┘  └────┘  └────┘
Buffer:   [0]     [1]     [0]     [1]
```

#### 6.5.4 逐元素标量解量化

与 Kernel ④ 的 LOP3 批量解量化不同，该 kernel 使用逐元素标量方式：

```cpp
const int packed_n_idx = n_global / 8;     // 划入哪个 packed INT32
const int n_in_pack = n_global % 8;        // 在 packed INT32 中的位置
const int g_cur = k_global / group_size;   // 当前 group

const int32_t w_packed = __ldg(&qweight[k_global * packed_N + packed_n_idx]);
const int32_t z_packed = __ldg(&qzeros[g_cur * packed_N + packed_n_idx]);
const half scale = __ldg(&scales[g_cur * N + n_global]);

int w = (w_packed >> (awq_order[n_in_pack] * 4)) & 0xF;
int z = (z_packed >> (awq_order[n_in_pack] * 4)) & 0xF;
smem_B[buf][k_local][n_local] = __hmul(__float2half((float)(w - z)), scale);
```

每个线程只解量化所需的单个权重，然后存入共享内存供后续计算使用。虽然效率低于 LOP3，但对于非 Tensor Core 路径仍可接受。

#### 6.5.5 实际例子

```
Qwen3-8B q_proj: M=32, K=4096, N=4096
（注：实际不走此 kernel，因为 M>1 被路由到 vllm kernel）

假设直接调用:
Grid: <<<(32, 1), 256>>>
  grid.x = (4096 + 127) / 128 = 32  (N 方向)
  grid.y = (32 + 31) / 32 = 1       (M 方向)
  
n_k_tiles = (4096 + 31) / 32 = 128

Shared memory per block:
  smem_A: 2 × 32 × 36 × 2B = 4608B
  smem_B: 2 × 32 × 136 × 2B = 17408B
  Total: ~22KB (远小于 48KB 限制)

计算量:
  128 个 K tile × 32 × 32 × 128 = 16,777,216 FMA / block
  32 blocks → 总计 536,870,912 FMA

权重读取:
  每 block 每 tile: 32 × 128 × 4B (INT4打包) = 2KB (解量化前)
  实际: 32 × 128 / 8 × 4 = 2KB
  总计: 128 × 32 × 2KB = 8MB (全部权重)
```

### 6.6 Kernel ④：`awq_gemm_vllm_kernel<N>` — Tensor Core Prefill 内核

#### 6.6.1 适用场景

Prefill 阶段 (M > 1)，这是推理 prompt 处理时的主要计算内核。使用 Tensor Core MMA 指令实现高吞吐 FP16 GEMM。

#### 6.6.2 启动配置

```cpp
// awq_gemm_vllm.cu 第 241-253 行
// N ≥ 128 且 N % 128 == 0 时:
int j_factors1 = N / 128;
dim3 num_blocks((M + 15) / 16 * j_factors1);
dim3 threads_per_block(32, 2);  // 64 threads = 2 warps
awq_gemm_vllm_kernel<128><<<num_blocks, threads_per_block, 0, stream>>>(...);

// 否则 N % 64 == 0:
int j_factors1 = N / 64;
dim3 num_blocks((M + 15) / 16 * j_factors1);
dim3 threads_per_block(32, 2);
awq_gemm_vllm_kernel<64><<<num_blocks, threads_per_block, 0, stream>>>(...);
```

Grid 是一维的，`blockIdx.x` 同时编码了 M 和 N 方向的 tile 信息：
- `blockIdx_y = blockIdx.x % ((M+15)/16 * j_factors1)`
- M 方向 tile: `blockIdx_y / j_factors1`
- N 方向 tile: `blockIdx_y % j_factors1`

#### 6.6.3 源码逐行注释（核心部分）

```cpp
template <int N>  // N = 64 or 128
__global__ void __launch_bounds__(64)
awq_gemm_vllm_kernel(int G, half* A, int* B,
    half* scaling_factors, int* zeros,
    int M, int IC, int OC, half* C)
{
    // ---- 1. 数据结构分配 ----
    float C_warp[32];                        // 每线程的MMA累加器
    __shared__ half A_shared[16 * 40];       // 16行 × (32+8) FP16 (padding)
    __shared__ half B_shared[32 * (N+8)];    // 32行 × (N+8) FP16 (padding)

    half A_shared_warp[8];                   // ldmatrix 加载的 A fragment
    half B_shared_warp[N / 4];               // ldmatrix 加载的 B fragment

    // 初始化累加器为零
    for (int j = 0; j < N/32; ++j)
        for (int i = 0; i < 8; ++i) C_warp[j*8+i] = 0.0f;

    // ---- 2. 指针计算 ----
    // A_ptr: 指向当前 block 负责的 A 矩阵区域
    half* A_ptr = A + (block_m * 16 + threadIdx.y * 8 + threadIdx.x / 4) * IC
                    + (threadIdx.x % 4) * 8;
    // 每线程加载 8 个连续的 half (uint4 = 16B)

    // B_ptr: 指向当前 block 负责的 B 权重区域 (INT4 packed)
    int* B_ptr = B + threadIdx.y * (OC/8) * (256/N)
                   + (threadIdx.x / (N/8)) * (OC/8)
                   + block_n * (N/8)
                   + (threadIdx.x % (N/8));

    // ---- 3. K 维度主循环 ----
    for (int k_0_0 = 0; k_0_0 < IC / 32; ++k_0_0) {
        __syncthreads();

        // 3a. 加载 A[16×32] 到共享内存 (向量化 uint4)
        if (ld_A_flag)
            *(uint4*)(A_shared_ptr) = *(uint4*)(A_ptr + k_0_0 * 32);
        else
            *(uint4*)(A_shared_ptr) = make_uint4(0,0,0,0);

        // 3b. 加载零点和 scale（按 group 索引）
        uint32_t zeros_loaded = *(uint32_t*)(zeros_ptr + k_0_0*32/G * (OC/8));
        half* scales_loaded = sf_ptr + k_0_0*32/G * OC;

        // 3c. 加载 + LOP3 解量化 B[32×N] 到共享内存
        for (int ax0 = 0; ax0 < N/16; ++ax0) {
            uint32_t B_loaded = *(uint32_t*)(B_ptr_local + ax0 * row_stride * (OC/8));

            // ★ LOP3 快速解量化：1 个 INT32 → 4 个 half2 = 8 个 FP16
            half2 B_dequant_h2[4];
            dequant_vllm_lop3(B_loaded, zeros_loaded,
                              (half2*)scales_loaded, B_dequant_h2);

            // 写入共享内存 (uint4 = 16B = 8 个 half)
            *(uint4*)(B_shared_ptr + ax0 * row_stride * (N+8)) = *(uint4*)B_dequant_h2;
        }

        __syncthreads();

        // 3d. K 维度内层循环 (每次 16 个 K 元素)
        for (int k_0_1 = 0; k_0_1 < 2; ++k_0_1) {

            // ★ ldmatrix 加载 A fragment（非转置）
            // cvta 地址转换 + ldmatrix.sync.aligned.m8n8.x4.shared.b16
            // → A_shared_warp[8] (4 个 half2)

            // ★ ldmatrix 加载 B fragment（转置）
            for (int ax1_0 = 0; ax1_0 < N/32; ++ax1_0) {
                // cvta + ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16
                // → B_shared_warp[ax1_0*8 .. ax1_0*8+8]
            }

            // ★ MMA 计算
            for (int j_0_4 = 0; j_0_4 < N/32; ++j_0_4) {
                // 第 1 条: mma.sync m16n8k16 → C_warp[j*8+0..3]
                // 第 2 条: mma.sync m16n8k16 → C_warp[j*8+4..7]
                // 合计: 每次产出 16×16 结果
            }
        }
    }

    // ---- 4. 写回结果 ----
    // FP32 → FP16 转换后写回全局内存
}
```

#### 6.6.4 LOP3 解量化详解

`dequant_vllm_lop3()` 函数将 1 个 INT32 权重解包为 8 个 FP16 值，核心是 4 次 `lop3.b32` 指令：

```cpp
// 工作原理示例：
// packed_w = 0xABCDEF01 (包含 8 个 INT4: A,B,C,D,E,F,0,1)
// BOTTOM_MASK = 0x000f000f

// lop3.b32 结果 = (packed_w & BOTTOM_MASK) | I4s_TO_FP16_MAGIC
//               = (0x000D0001) | (0x64006400)
//               = 0x640D6401
// 解释为 half2: (1024.0 + 1, 1024.0 + 13) = (1025.0h, 1037.0h)
// 减去 1024.0: (1.0h, 13.0h) = 正确的 INT4 值 (1, 13)

// 对于 TOP_MASK 提取的高 4 位，还需要除以 16 (乘 0.0625)
// 因为位移了 4 位，FP16 值是实际值的 16 倍
```

整个 dequant 流程：

```
1 个 INT32 → lop3 ×4  → 4 个 uint32 (含 FP16 magic)
           → sub ×4   → 4 个 half2 (减去 1024.0 偏移)
           → hmul ×2  → 修正高位的位移 (÷16)
           → 右移 8 位 → 重复上述步骤处理高 8 位
           → hsub ×4  → w - z
           → hmul ×4  → scale * (w - z)
合计: 8 次 lop3 + 8 次 hsub/hmul + 2 次 hmul + 4 次 hmul
    = ~22 条指令 / 8 个权重 = ~2.75 条指令/权重
```

#### 6.6.5 实际例子：Qwen3-8B q_proj Prefill

**场景**：Prefill 512 个 token，q_proj 层 (K=4096, N=4096, group_size=128)。

```
参数:
  M = 512, K = 4096, N = 4096, G = 128
  使用 N=128 模板 (4096 % 128 == 0)
  j_factors1 = 4096 / 128 = 32

Grid 计算:
  num_blocks.x = (512 + 15) / 16 * 32 = 32 * 32 = 1024
  threads_per_block = (32, 2) = 64 threads

Block → Tile 映射:
  blockIdx.x = 0..1023
  block_m = (blockIdx.x / 32) → 0..31 (对应 M 维度 0..511, 每组 16 行)
  block_n = (blockIdx.x % 32) → 0..31 (对应 N 维度 0..4095, 每组 128 列)

K 维度循环:
  k_bound = 4096 / 32 = 128 次外层迭代
  每次外层迭代:
    - 2 次内层迭代 (k_0_1)
    - 每次内层: 4 次 ldmatrix(B) + 4 组 × 2 条 = 8 条 MMA
    - 每条 MMA: 16×8×16 = 2048 FMA
    - 小计: 2 × 8 × 2048 = 32,768 FMA

总计算量:
  128 × 32,768 = 4,194,304 FMA / block
  1024 blocks × 4,194,304 = 4,294,967,296 FMA ≈ 4.3 GFMA

理论 FLOPS (2 FLOP per FMA):
  512 × 4096 × 4096 × 2 = 17,179,869,184 FLOP = 17.2 GFLOP

Shared Memory per Block:
  A_shared: 16 × 40 × 2B = 1,280B
  B_shared: 32 × 136 × 2B = 8,704B
  Total: ~10KB (非常小，有利于高占用率)

MMA 吞吐 (Orin, SM8.7):
  4 Tensor Cores per SM
  每个 TC: 1 m16n8k16 / cycle
  峰值: 4 × 2048 FMA / cycle / SM
  32 SM × 4 × 2048 × 1.3GHz ≈ 340 TFLOPS (FP16)
```

### 6.7 各 Kernel 对比总结

| 特性 | Kernel ① GEMV | Kernel ② Small Batch | Kernel ③ Pipelined GEMM | Kernel ④ Tensor Core |
|------|-------------|-------------------|----------------------|---------------------|
| **适用条件** | M=1 | 2≤M≤8 | M>8 (直接调用) | M>1 (默认路由) |
| **线程配置** | 256 (8 warps) | 256 (8 warps) | 256 | 64 (2 warps) |
| **计算单元** | CUDA Core | CUDA Core | CUDA Core | **Tensor Core** |
| **解量化方式** | 标量 + FMA 优化 | 标量 AWQ order | 标量逐元素 | **LOP3 批量** |
| **归约方式** | Warp Shuffle | Warp Shuffle | 共享内存外积 | MMA 内置 |
| **内存优化** | 向量化读取 | 向量化读取 | 双缓冲流水线 | 单缓冲 + ldmatrix |
| **共享内存** | 无 | 无 | ~22KB (双缓冲) | ~10KB |
| **每 Tile 产出** | 64 outputs | 64×M outputs | 32×128 outputs | 16×N outputs |
| **典型场景** | 逐 token 生成 | 小 batch 验证 | 通用大矩阵 | Prompt 处理 |

### 6.8 从推理角度看 Kernel 选择

以 Qwen3-8B 推理为例，一次完整的推理过程中 kernel 的选择：

```
用户输入 "请解释 CUDA 编程" (假设 tokenize 后 10 个 token)

1. Prefill 阶段 (M=10):
   q_proj, k_proj, v_proj, o_proj 各调用 1 次
   gate_proj, up_proj, down_proj 各调用 1 次
   每层 7 次矩阵乘, 共 32 层 → 224 次 kernel 调用
   全部走 Kernel ④ (awq_gemm_vllm_kernel<128>)
   
2. Decode 阶段 (M=1, 逐 token 生成):
   每个新 token:
     同样 7 次矩阵乘 × 32 层 = 224 次 kernel 调用
     全部走 Kernel ① (awq_gemv_fast_kernel)
   
   假设生成 50 个 token → 50 × 224 = 11,200 次 Kernel ① 调用

总计:
  Kernel ④: 224 次 (Prefill, 使用 Tensor Core)
  Kernel ①: 11,200 次 (Decode, 标量 GEMV)
  Kernel ②③: 0 次 (标准推理不触发)
```
