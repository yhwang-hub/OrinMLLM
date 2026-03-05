# Qwen3-8B SmoothQuant INT8 模型适配与反量化技术报告

> 平台：NVIDIA Orin (SM87) | CUDA 12.6 | CUTLASS INT8 Tensor Core  
> 模型：Qwen3-8B-sq.bin（SmoothQuant 逐张量 INT8 量化）  
> 日期：2026-03-04

---

## 目录

- [第一部分：Qwen3-8B-sq.bin 模型适配全流程](#第一部分qwen3-8b-sqbin-模型适配全流程)
  - [1.1 整体架构概览](#11-整体架构概览)
  - [1.2 步骤一：模型权重导出（Python 端）](#12-步骤一模型权重导出python-端)
  - [1.3 步骤二：二进制文件格式设计](#13-步骤二二进制文件格式设计)
  - [1.4 步骤三：模型文件检测与加载入口（C++ 端）](#14-步骤三模型文件检测与加载入口c-端)
  - [1.5 步骤四：权重加载——create_param_layers_sq()](#15-步骤四权重加载create_param_layers_sq)
  - [1.6 步骤五：SQMatmulLayer 层的实现](#16-步骤五sqmatmullayer-层的实现)
  - [1.7 步骤六：运行时多态分发——虚函数覆盖](#17-步骤六运行时多态分发虚函数覆盖)
  - [1.8 步骤七：CUDA Kernel 层——双路径调度](#18-步骤七cuda-kernel-层双路径调度)
  - [1.9 适配流程总结](#19-适配流程总结)
- [第二部分：INT8 数据解包与反量化原理详解](#第二部分int8-数据解包与反量化原理详解)
  - [2.1 SmoothQuant 量化方案概述](#21-smoothquant-量化方案概述)
  - [2.2 量化数学原理](#22-量化数学原理)
  - [2.3 权重 INT8 数据的存储与解包](#23-权重-int8-数据的存储与解包)
  - [2.4 激活值的动态量化（FP16 → INT8）](#24-激活值的动态量化fp16--int8)
  - [2.5 INT8 矩阵乘法与 `__dp4a` 硬件指令](#25-int8-矩阵乘法与-__dp4a-硬件指令)
  - [2.6 反量化：INT32 累加结果 → FP16 输出](#26-反量化int32-累加结果--fp16-输出)
  - [2.7 CUTLASS Prefill 路径的反量化](#27-cutlass-prefill-路径的反量化)
  - [2.8 共享量化优化（Shared Quantization）](#28-共享量化优化shared-quantization)
  - [2.9 融合 FFN 中的反量化](#29-融合-ffn-中的反量化)
  - [2.10 端到端数据流总结](#210-端到端数据流总结)
- [第三部分：sq_gemm_kernel.cu CUDA Kernel 逐一详解](#第三部分sq_gemm_kernelcu-cuda-kernel-逐一详解)
  - [3.1 文件总览与架构](#31-文件总览与架构)
  - [3.2 CUTLASS INT8 GEMM 类型定义](#32-cutlass-int8-gemm-类型定义)
  - [3.3 Kernel 1：sq_absmax_kernel — 逐张量AbsMax归约](#33-kernel-1sq_absmax_kernel--逐张量absmax归约)
  - [3.4 Kernel 2：sq_quantize_and_alpha_kernel — 量化+Alpha计算](#34-kernel-2sq_quantize_and_alpha_kernel--量化alpha计算)
  - [3.5 Kernel 3：sq_gemv_int8_kernel — INT8 GEMV（dp4a + 128-bit加载）](#35-kernel-3sq_gemv_int8_kernel--int8-gemvdp4a--128-bit加载)
  - [3.6 Kernel 4：sq_gemv_preq_kernel — 预量化GEMV](#36-kernel-4sq_gemv_preq_kernel--预量化gemv)
  - [3.7 Kernel 5：sq_fused_ffn_gemv_kernel — 融合FFN GEMV](#37-kernel-5sq_fused_ffn_gemv_kernel--融合ffn-gemv)
  - [3.8 SQWorkspace — GPU 工作区管理](#38-sqworkspace--gpu-工作区管理)
  - [3.9 调度函数 1：sq_gemv_m1 — Decode路径调度](#39-调度函数-1sq_gemv_m1--decode路径调度)
  - [3.10 调度函数 2：sq_gemm_cutlass — Prefill路径调度](#310-调度函数-2sq_gemm_cutlass--prefill路径调度)
  - [3.11 公共入口 1：sq_gemm_cu — 主入口](#311-公共入口-1sq_gemm_cu--主入口)
  - [3.12 公共入口 2：sq_fused_ffn_cu — 融合FFN入口](#312-公共入口-2sq_fused_ffn_cu--融合ffn入口)
  - [3.13 公共入口 3：sq_quantize_input_cu — 共享量化入口](#313-公共入口-3sq_quantize_input_cu--共享量化入口)
  - [3.14 公共入口 4：sq_gemv_preq_cu — 预量化GEMV入口](#314-公共入口-4sq_gemv_preq_cu--预量化gemv入口)
  - [3.15 Kernel 参数与启动配置速查表](#315-kernel-参数与启动配置速查表)
- [第四部分：Decode 阶段性能优化深度分析（10.6 → 17.66 tokens/s）](#第四部分decode-阶段性能优化深度分析106--1766-tokenss)
  - [4.1 优化全景概览](#41-优化全景概览)
  - [4.2 优化 A：修复竞态条件 Bug（正确性修复）](#42-优化-a修复竞态条件-bug正确性修复)
    - [4.2.1 旧方案的致命缺陷：fused absmax+quantize 的竞态条件](#421-旧方案的致命缺陷fused-absmaxquantize-的竞态条件)
    - [4.2.2 新方案：2-Kernel 分离架构](#422-新方案2-kernel-分离架构)
    - [4.2.3 正确性保证的数学证明](#423-正确性保证的数学证明)
    - [4.2.4 性能影响分析](#424-性能影响分析)
  - [4.3 优化 B：__dp4a 硬件指令（计算效率 ~3x）](#43-优化-b__dp4a-硬件指令计算效率-3x)
    - [4.3.1 旧方案：手动 INT8 拆包与标量乘法](#431-旧方案手动-int8-拆包与标量乘法)
    - [4.3.2 新方案：__dp4a 单指令 4-MAC](#432-新方案__dp4a-单指令-4-mac)
    - [4.3.3 指令级性能对比](#433-指令级性能对比)
    - [4.3.4 源码中的 __dp4a 应用](#434-源码中的-__dp4a-应用)
  - [4.4 优化 C：128-bit 向量化加载（带宽利用率 4x）](#44-优化-c128-bit-向量化加载带宽利用率-4x)
    - [4.4.1 旧方案：32-bit 标量加载](#441-旧方案32-bit-标量加载)
    - [4.4.2 新方案：int4 128-bit 向量化加载](#442-新方案int4-128-bit-向量化加载)
    - [4.4.3 内存事务效率对比](#443-内存事务效率对比)
    - [4.4.4 与 __dp4a 的完美配合](#444-与-__dp4a-的完美配合)
  - [4.5 优化 D：QKV 共享量化（减少 216 个 Kernel Launch/步）](#45-优化-dqkv-共享量化减少-216-个-kernel-launch步)
    - [4.5.1 旧方案：独立量化的 Kernel Launch 风暴](#451-旧方案独立量化的-kernel-launch-风暴)
    - [4.5.2 新方案：共享量化 + 预量化 GEMV](#452-新方案共享量化--预量化-gemv)
    - [4.5.3 源码实现详解](#453-源码实现详解)
    - [4.5.4 Kernel Launch 数量化分析](#454-kernel-launch-数量化分析)
  - [4.6 四项优化的协同效应](#46-四项优化的协同效应)
  - [4.7 性能提升总结与分析](#47-性能提升总结与分析)

---

## 第一部分：Qwen3-8B-sq.bin 模型适配全流程

### 1.1 整体架构概览

模型适配的核心目标是：将 HuggingFace SmoothQuant INT8 量化模型转换为自定义二进制格式，并在 C++/CUDA 推理引擎中高效加载和执行。整体流程涉及以下文件：

| 文件 | 作用 |
|------|------|
| `tools/export_qwen3-8B-sq.py` | Python 导出脚本：HF 模型 → .bin 二进制文件 |
| `kuiper/include/model/qwen3_sq.h` | C++ 模型类声明（继承 Qwen3Model） |
| `kuiper/source/model/qwen3_sq.cpp` | C++ 模型实现：权重加载 + 运行时分发 |
| `kuiper/include/op/sq_matmul.h` | SQ INT8 矩阵乘法层声明 |
| `kuiper/source/op/sq_matmul.cpp` | SQ INT8 矩阵乘法层实现 |
| `kuiper/source/op/kernels/cuda/sq_gemm_kernel.cuh` | CUDA Kernel 接口声明 |
| `kuiper/source/op/kernels/cuda/sq_gemm_kernel.cu` | CUDA Kernel 实现（659 行，核心计算） |

调用链路：

```
[Python 导出] export_qwen3-8B-sq.py
       ↓  生成 .bin 文件
[C++ 加载] Qwen3SQModel::create_param_layers_sq()
       ↓  创建 SQMatmulLayer, 设置权重
[C++ 推理] SQMatmulLayer::forward() / forward_preq() / fused_ffn_forward()
       ↓  调用 CUDA Kernel
[CUDA 计算] sq_gemm_cu() / sq_fused_ffn_cu() / sq_gemv_preq_cu()
       ↓  双路径调度
[Decode M=1]  absmax → quantize → __dp4a GEMV
[Prefill M>1] absmax → quantize → CUTLASS INT8 Tensor Core GEMM
```

---

### 1.2 步骤一：模型权重导出（Python 端）

**文件**：`tools/export_qwen3-8B-sq.py`（402 行）

#### 1.2.1 加载 HuggingFace 模型

```python
def load_hf_weights(model_path):
    hf_config = AutoConfig.from_pretrained(model_path)
    # 从 safetensors 文件加载所有权重张量
    safetensor_files = sorted(list(model_path.glob("*.safetensors")))
    hf_dict = {}
    for sf_file in safetensor_files:
        with safe_open(sf_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                hf_dict[key] = f.get_tensor(key)
```

关键验证步骤——确认模型确实是 SmoothQuant 格式：

```python
# 验证 SQ 关键 key 存在
if 'model.layers.0.self_attn.q_proj.qweight' in hf_dict:
    print("  ✅ SQ qweight found")
if 'model.layers.0.self_attn.q_proj.weight_scale' in hf_dict:
    print("  ✅ SQ weight_scale found")
if 'model.layers.0.self_attn.q_proj.input_scale' in hf_dict:
    print("  ✅ SQ input_scale found")
```

每个 SmoothQuant 量化的线性层包含三个参数：
- `qweight`：`[out_features, in_features]` INT8 量化权重矩阵
- `weight_scale`：BF16 标量，权重量化的缩放因子
- `input_scale`：FP32 标量，校准数据集上计算的输入激活缩放因子

#### 1.2.2 序列化辅助函数

```python
def serialize_fp16(file, tensor):
    """将张量转为 FP16 后写入文件"""
    d = tensor.detach().cpu().view(-1).to(torch.float16).numpy()
    file.write(d.tobytes())

def serialize_int8(file, tensor):
    """将 INT8 张量直接写入文件"""
    d = tensor.detach().cpu().view(-1).to(torch.int8).numpy()
    file.write(d.tobytes())

def serialize_fp16_scalar(file, value):
    """将标量以 FP16 格式写入（2 字节）"""
    value = torch.tensor(value, dtype=torch.float16)
    file.write(value.numpy().tobytes())

def serialize_fp32_scalar(file, value):
    """将标量以 FP32 格式写入（4 字节）"""
    file.write(struct.pack('f', value))
```

#### 1.2.3 SQ 量化权重写入

```python
def write_sq_weights(layer_name, prefix=""):
    qweight = hf_dict[f'{layer_name}.qweight']
    weight_scale = hf_dict[f'{layer_name}.weight_scale']
    input_scale = hf_dict[f'{layer_name}.input_scale']

    # 1) INT8 权重矩阵 [out_features, in_features]
    serialize_int8(out_file, qweight)

    # 2) weight_scale: FP16 标量（2 字节）
    serialize_fp16_scalar(out_file, weight_scale)

    # 3) input_scale: FP32 标量（4 字节）
    serialize_fp32_scalar(out_file, input_scale)
```

每个 SQ 线性层在文件中的布局为：

```
┌──────────────────────────────────────────────┐
│ qweight: [out_features × in_features] INT8   │  ← out×in 字节
│ weight_scale: FP16 scalar                    │  ← 2 字节
│ input_scale: FP32 scalar                     │  ← 4 字节
└──────────────────────────────────────────────┘
```

---

### 1.3 步骤二：二进制文件格式设计

**文件头**（256 字节）：

```python
# 1) magic: "sq48" (0x73713438) — 标识 Qwen3 SmoothQuant 格式
out_file.write(struct.pack('I', 0x73713438))

# 2) version: 6 — SQ INT8 版本号
out_file.write(struct.pack('i', version))

# 3) 模型参数（7 个 int32）
header = struct.pack('iiiiiii', dim, hidden_dim, n_layers, n_heads,
                     n_kv_heads, vocab_size, max_seq_len)
out_file.write(header)

# 4) shared_classifier 标志（1 字节）
out_file.write(struct.pack('B', int(shared_classifier)))

# 5) head_dim（Qwen3 特有）
out_file.write(struct.pack('i', head_dim))

# 6) 填充到 256 字节
pad = 256 - out_file.tell()
out_file.write(b'\0' * pad)
```

对于 Qwen3-8B 模型，具体参数为：

| 参数 | 值 | 说明 |
|------|----|------|
| magic | 0x73713438 ("sq48") | SmoothQuant Qwen3 标识 |
| version | 6 | SQ INT8 版本 |
| dim | 4096 | 隐藏维度 |
| hidden_dim | 12288 | FFN 中间维度 |
| n_layers | 36 | Transformer 层数 |
| n_heads | 32 | 注意力头数 |
| n_kv_heads | 8 | KV 头数（GQA） |
| vocab_size | 151936 | 词表大小 |
| head_size | 128 | 每头维度 |

**权重序列化顺序**：

```
== FP16 权重（非量化） ==
 1. attention_norm (input_layernorm)    × 36 层   [4096] FP16
 2. ffn_norm (post_attention_layernorm) × 36 层   [4096] FP16
 3. final_norm                          × 1       [4096] FP16
 4. token_embeddings                    × 1       [151936, 4096] FP16

== SQ INT8 量化权重 ==
 5. wq  (q_proj)   × 36 层  每层: [4096, 4096] INT8 + ws(FP16) + is(FP32)
 6. wk  (k_proj)   × 36 层  每层: [1024, 4096] INT8 + ws(FP16) + is(FP32)
 7. wv  (v_proj)   × 36 层  每层: [1024, 4096] INT8 + ws(FP16) + is(FP32)
 8. wo  (o_proj)   × 36 层  每层: [4096, 4096] INT8 + ws(FP16) + is(FP32)
 9. w1  (gate_proj) × 36 层 每层: [12288, 4096] INT8 + ws(FP16) + is(FP32)
10. w2  (down_proj) × 36 层 每层: [4096, 12288] INT8 + ws(FP16) + is(FP32)
11. w3  (up_proj)  × 36 层  每层: [12288, 4096] INT8 + ws(FP16) + is(FP32)

== FP16 权重（非量化） ==
12. lm_head                            × 1       [151936, 4096] FP16
13. q_norm                              × 36 层   [128] FP16
14. k_norm                              × 36 层   [128] FP16
```

---

### 1.4 步骤三：模型文件检测与加载入口（C++ 端）

**文件**：`kuiper/source/model/qwen3_sq.cpp`

首先，通过 magic number 和 version 检测文件是否为 SQ 格式：

```cpp
bool is_sq_model_file(const std::string& model_path) {
  FILE* file = fopen(model_path.c_str(), "rb");
  if (!file) return false;

  uint32_t magic = 0;
  int32_t version = 0;
  bool is_sq = false;

  if (fread(&magic, sizeof(uint32_t), 1, file) == 1 &&
      fread(&version, sizeof(int32_t), 1, file) == 1) {
    // SQ 格式: magic=0x73713438 ("sq48"), version=6
    is_sq = (magic == 0x73713438 && version == 6);
  }

  fclose(file);
  return is_sq;
}
```

检测到 SQ 格式后，推理框架自动创建 `Qwen3SQModel` 实例（而非普通的 `Qwen3Model`），该类继承体系如下：

```
Qwen3Model (FP16 基类)
    └── Qwen3SQModel (SQ INT8 子类)
            - 覆盖 create_param_layers()      → 加载 SQ 权重
            - 覆盖 batched_qkv_projection()    → 共享量化 QKV
            - 覆盖 batched_matmul_forward()     → SQ GEMM 分发
            - 覆盖 gate_up_swiglu()             → 融合 FFN
```

---

### 1.5 步骤四：权重加载——create_param_layers_sq()

**文件**：`kuiper/source/model/qwen3_sq.cpp` 中 `Qwen3SQModel::create_param_layers_sq()`

这是适配过程中最关键的函数，它从二进制文件中按照与 Python 导出脚本**完全一致**的顺序逐一读取所有权重。

#### 1.5.1 使用指针偏移遍历 mmap 数据

```cpp
void Qwen3SQModel::create_param_layers_sq() {
  const uint8_t* base_ptr = static_cast<const uint8_t*>(raw_model_data_->weight_data);
  size_t pos = 0;  // 从 header 之后开始

  int32_t dim = config_->dim_;           // 4096
  int32_t kv_dim = config_->kv_dim_;     // 1024
  int32_t hidden_dim = config_->hidden_dim_;  // 未使用（用 immediate_dim）
  int32_t immediate_dim = config_->immediate_dim_;  // 12288
```

#### 1.5.2 加载 FP16 非量化权重

```cpp
  // 1. attention_norm (input_layernorm) - FP16
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto rms_norm_layer = std::make_shared<op::RmsNormLayer>(device_type_, dim);
    rms_norm_layer->set_weight_fp16(0, {dim}, base_ptr + pos, cpu_device_type);
    qwen_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    pos += dim * sizeof(uint16_t);  // 4096 × 2 = 8192 字节
  }

  // 2. ffn_norm - FP16（同上结构）
  // 3. final_norm - FP16
  // 4. token_embeddings - FP16 [vocab_size, dim]
```

#### 1.5.3 加载 SQ INT8 量化权重——lambda 封装

```cpp
  // 通用 SQ 层加载函数
  auto load_sq_layer = [&](int32_t in_features, int32_t out_features,
                           std::vector<std::shared_ptr<op::Layer>>& layer_list,
                           const std::string& name) {
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
      // 创建 SQMatmulLayer 实例
      auto sq_layer = std::make_shared<op::SQMatmulLayer>(
          device_type_, in_features, out_features);

      // 读取 qweight [out_features, in_features] INT8
      const void* qweight_ptr = base_ptr + pos;
      size_t qweight_size = static_cast<size_t>(out_features) * in_features * sizeof(int8_t);
      pos += qweight_size;

      // 读取 weight_scale FP16 标量（2 字节）
      const void* weight_scale_ptr = base_ptr + pos;
      pos += sizeof(uint16_t);

      // 读取 input_scale FP32 标量（4 字节）
      const void* input_scale_ptr = base_ptr + pos;
      pos += sizeof(float);

      // 设置权重到 SQMatmulLayer
      sq_layer->set_sq_weights(qweight_ptr, weight_scale_ptr, input_scale_ptr, cpu_device_type);
      layer_list.push_back(sq_layer);
    }
  };

  // 按顺序加载 7 类 SQ 层，每类 36 层
  load_sq_layer(dim, dim,           qwen_layers_->wq_layers_, "wq");  // 5. q_proj
  load_sq_layer(dim, kv_dim,        qwen_layers_->wk_layers_, "wk");  // 6. k_proj
  load_sq_layer(dim, kv_dim,        qwen_layers_->wv_layers_, "wv");  // 7. v_proj
  load_sq_layer(dim, dim,           qwen_layers_->wo_layers_, "wo");  // 8. o_proj
  load_sq_layer(dim, immediate_dim, qwen_layers_->w1_layers_, "w1");  // 9. gate_proj
  load_sq_layer(immediate_dim, dim, qwen_layers_->w2_layers_, "w2");  // 10. down_proj
  load_sq_layer(dim, immediate_dim, qwen_layers_->w3_layers_, "w3");  // 11. up_proj
```

这段代码的关键设计：
- **pos 指针严格按字节偏移**，与 Python 导出脚本的写入顺序一一对应
- 每个层读取 `out_features × in_features` 字节的 INT8 数据 + 2 字节 weight_scale + 4 字节 input_scale
- 通过 lambda 避免 7 × 36 = 252 个层的重复代码

#### 1.5.4 加载 lm_head 和 QK Norm

```cpp
  // 12. lm_head - FP16（不量化，保持精度）
  if (!config_->is_shared_weight_) {
    auto lm_head = std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_, dim, false);
    lm_head->set_weight_fp16(0, {config_->vocab_size_, dim}, base_ptr + pos, cpu_device_type);
    qwen_layers_->cls_layer_ = lm_head;
    pos += config_->vocab_size_ * dim * sizeof(uint16_t);
  }

  // 13. q_norm - FP16 [head_size=128]
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto rms_norm_layer = std::make_shared<op::RmsNormLayer>(device_type_, config_->head_size_);
    rms_norm_layer->set_weight_fp16(0, {config_->head_size_}, base_ptr + pos, cpu_device_type);
    qwen_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    pos += config_->head_size_ * sizeof(uint16_t);
  }

  // 14. k_norm - FP16 [head_size=128]（同上结构）
```

---

### 1.6 步骤五：SQMatmulLayer 层的实现

**文件**：`kuiper/include/op/sq_matmul.h` + `kuiper/source/op/sq_matmul.cpp`

#### 1.6.1 类设计

```cpp
class SQMatmulLayer : public Layer {
 private:
  int32_t in_features_ = 0;
  int32_t out_features_ = 0;
  tensor::Tensor qweight_;      // INT8 量化权重 [out_features, in_features]
  float weight_scale_ = 0.0f;   // 权重量化缩放因子
  float input_scale_ = 0.0f;    // 校准输入缩放因子（仅存储参考，运行时不使用）
};
```

关键设计决策：`input_scale_` 在运行时**不使用**，因为我们采用**动态逐张量量化**——每次推理时根据实际输入的 absmax 值计算量化比例，而非使用离线校准的固定比例。这提高了精度。

#### 1.6.2 权重设置——FP16 到 FP32 的主机端转换

```cpp
// 主机端 FP16→FP32 转换（不依赖 CUDA intrinsics）
static float fp16_bits_to_float(uint16_t h) {
  uint32_t sign = (h >> 15) & 1;
  uint32_t exponent = (h >> 10) & 0x1F;
  uint32_t mantissa = h & 0x3FF;
  uint32_t fp32_bits;

  if (exponent == 0) {
    if (mantissa == 0) {
      fp32_bits = sign << 31;  // ±0
    } else {
      // 次正规数处理
      exponent = 1;
      while (!(mantissa & 0x400)) { mantissa <<= 1; exponent--; }
      mantissa &= 0x3FF;
      fp32_bits = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
  } else if (exponent == 0x1F) {
    fp32_bits = (sign << 31) | (0xFF << 23) | (mantissa << 13);  // Inf/NaN
  } else {
    fp32_bits = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);  // 正规数
  }

  float result;
  std::memcpy(&result, &fp32_bits, sizeof(float));
  return result;
}
```

这个函数的意义：weight_scale 在二进制文件中以 FP16（2 字节）存储以节省空间，但运行时需要 FP32 精度。该函数按 IEEE 754 标准在 CPU 端完成位级转换，不依赖任何 GPU 函数。

```cpp
void SQMatmulLayer::set_sq_weights(const void* qweight_ptr,
                                    const void* weight_scale_ptr,
                                    const void* input_scale_ptr,
                                    base::DeviceType src_device) {
  // 1) INT8 权重：直接 memcpy（无需转换）
  int32_t qweight_size = out_features_ * in_features_;
  qweight_ = tensor::Tensor(base::DataType::kDataTypeInt8, qweight_size, true, alloc);
  std::memcpy(qweight_.ptr<void>(), qweight_ptr, qweight_size * sizeof(int8_t));

  // 2) weight_scale: FP16 → FP32
  uint16_t ws_fp16;
  std::memcpy(&ws_fp16, weight_scale_ptr, sizeof(uint16_t));
  weight_scale_ = fp16_bits_to_float(ws_fp16);

  // 3) input_scale: FP32 直接读取
  std::memcpy(&input_scale_, input_scale_ptr, sizeof(float));
}
```

#### 1.6.3 GPU 上传——零反量化加载

```cpp
void SQMatmulLayer::to_cuda() {
  if (!qweight_.is_empty()) {
    // 直接将 INT8 数据上传到 GPU——无需 CPU 端反量化！
    // 这是模型加载速度的关键优化
    auto cuda_alloc = base::CUDADeviceAllocatorFactory::get_instance();
    int32_t total = out_features_ * in_features_;

    tensor::Tensor gpu_qweight(base::DataType::kDataTypeInt8, total, true, cuda_alloc);
    cudaMemcpy(gpu_qweight.ptr<void>(), qweight_.ptr<void>(),
               total * sizeof(int8_t), cudaMemcpyHostToDevice);

    qweight_ = std::move(gpu_qweight);
  }
}
```

对比传统方案（先在 CPU 反量化为 FP16 再上传），INT8 直传方案有两大优势：
1. **传输量减半**：INT8 每元素 1 字节 vs FP16 每元素 2 字节
2. **零 CPU 计算**：没有 CPU 端的乘法/类型转换开销

---

### 1.7 步骤六：运行时多态分发——虚函数覆盖

**文件**：`kuiper/include/model/qwen3_sq.h` + `kuiper/source/model/qwen3_sq.cpp`

`Qwen3SQModel` 通过覆盖父类虚函数实现 SQ INT8 推理路径的无缝切入：

```cpp
class Qwen3SQModel : public Qwen3Model {
 protected:
  void create_param_layers() override;            // 加载 SQ 权重
  void create_param_quant_layers() override;       // 空实现（SQ 层已在上面加载）

  // 核心运行时分发
  void batched_qkv_projection(...) const override; // 共享量化 QKV
  void batched_matmul_forward(...) const override;  // SQ GEMM
  void gate_up_swiglu(...) const override;          // 融合 FFN
};
```

#### 1.7.1 共享量化 QKV 投影

```cpp
void Qwen3SQModel::batched_qkv_projection(int32_t layer_idx, ...) const {
  auto query_sq = std::dynamic_pointer_cast<op::SQMatmulLayer>(query_layer);
  auto key_sq   = std::dynamic_pointer_cast<op::SQMatmulLayer>(key_layer);
  auto value_sq = std::dynamic_pointer_cast<op::SQMatmulLayer>(value_layer);

  int batch_size = rms_out.size() / in_features;

  if (batch_size == 1) {
    // Decode 路径：共享量化——量化一次，复用三次
    // 节省 6 个 kernel 启动（每层）→ 36 层共节省 216 次
    op::SQMatmulLayer::quantize_input(rms_out, stream);          // 量化（3 kernels）
    op::SQMatmulLayer::forward_preq(query_out, *query_sq, stream);  // Q GEMV（1 kernel）
    op::SQMatmulLayer::forward_preq(key_out, *key_sq, stream);     // K GEMV（1 kernel）
    op::SQMatmulLayer::forward_preq(value_out, *value_sq, stream); // V GEMV（1 kernel）
    return;
  }

  // Prefill 路径：各自独立执行完整 SQ GEMM
  query_sq->forward(rms_out, query_out);
  key_sq->forward(rms_out, key_out);
  value_sq->forward(rms_out, value_out);
}
```

#### 1.7.2 融合 FFN（Gate + Up + SwiGLU）

```cpp
void Qwen3SQModel::gate_up_swiglu(int32_t layer_idx, ...) const {
  auto w1_sq = std::dynamic_pointer_cast<op::SQMatmulLayer>(w1_layer);
  auto w3_sq = std::dynamic_pointer_cast<op::SQMatmulLayer>(w3_layer);

  if (w1_sq && w3_sq) {
    int batch_size = input.size() / in_features;
    if (batch_size == 1) {
      // 融合路径：1 次量化 + 1 个融合 kernel（W1·x + W3·x + SwiGLU）
      op::SQMatmulLayer::fused_ffn_forward(input, output, *w1_sq, *w3_sq, stream);
      return;
    }
  }

  // Prefill 回退：分别执行 W1、W3 GEMM + 独立 SwiGLU
  w1_layer->forward(input, output);
  w3_layer->forward(input, w3_output);
  layers->swiglu_layer_->forward(output, w3_output, output);
}
```

---

### 1.8 步骤七：CUDA Kernel 层——双路径调度

**文件**：`kuiper/source/op/kernels/cuda/sq_gemm_kernel.cu`

主入口函数 `sq_gemm_cu()` 根据 batch_size 分发：

```cpp
void sq_gemm_cu(const half* input_fp16, const int8_t* qweight,
                half* output_fp16, float weight_scale,
                int batch_size, int in_features, int out_features,
                cudaStream_t stream)
{
    const int M = batch_size;
    if (M == 1) {
        // 路径 1: Decode — 带宽优化的 INT8 GEMV
        sq_gemv_m1(input_fp16, qweight, output_fp16, weight_scale, K, N, stream);
    } else {
        // 路径 2: Prefill — CUTLASS INT8 Tensor Core GEMM
        sq_gemm_cutlass(input_fp16, qweight, output_fp16, weight_scale, M, K, N, stream);
    }
}
```

两条路径共享同一个 Workspace：

```cpp
struct SQWorkspace {
    int8_t* input_int8 = nullptr;  // 量化后的 INT8 输入
    int*    max_int    = nullptr;  // absmax 累加器（int 位模式表示 float）
    float*  alpha      = nullptr;  // 反量化系数 = input_scale × weight_scale
    size_t  input_cap  = 0;        // 当前分配容量

    void ensure(size_t need) {
        if (need > input_cap) {
            if (input_int8) cudaFree(input_int8);
            input_cap = need * 2;  // 2× 增长策略
            cudaMalloc(&input_int8, input_cap);
        }
        if (!max_int) {
            cudaMalloc(&max_int, sizeof(int));
            cudaMalloc(&alpha, sizeof(float));
        }
    }
};
static SQWorkspace g_workspace;
```

---

### 1.9 适配流程总结

```
┌─────────────────────────────────────────────────────────┐
│ 步骤 1: Python 导出                                      │
│   HuggingFace SQ 模型 → .bin 二进制文件                    │
│   (qweight INT8 + weight_scale FP16 + input_scale FP32)  │
└───────────────────────┬─────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 步骤 2: 文件格式设计                                      │
│   256 字节 header (magic/version/模型参数)                 │
│   + 严格有序的权重数据流                                    │
└───────────────────────┬─────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 步骤 3: C++ 检测 (is_sq_model_file)                       │
│   读取 magic=0x73713438 + version=6 → 创建 Qwen3SQModel  │
└───────────────────────┬─────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 步骤 4: 权重加载 (create_param_layers_sq)                  │
│   mmap + pos 偏移精确匹配 Python 写入顺序                   │
│   FP16 norm/emb + INT8 SQ 线性层 + FP16 lm_head/qk_norm  │
└───────────────────────┬─────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 步骤 5: SQMatmulLayer                                     │
│   set_sq_weights() → INT8 memcpy + FP16→FP32 scale 转换   │
│   to_cuda() → INT8 直传 GPU（零反量化）                     │
└───────────────────────┬─────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 步骤 6: 虚函数覆盖                                        │
│   batched_qkv_projection → 共享量化 (M=1) / 独立 GEMM     │
│   gate_up_swiglu → 融合 FFN (M=1) / 分离 GEMM+SwiGLU      │
└───────────────────────┬─────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 步骤 7: CUDA Kernel 双路径                                 │
│   M=1: absmax → quantize → __dp4a GEMV (128-bit loads)   │
│   M>1: absmax → quantize → CUTLASS INT8 Tensor Core GEMM │
└─────────────────────────────────────────────────────────┘
```

---

## 第二部分：INT8 数据解包与反量化原理详解

### 2.1 SmoothQuant 量化方案概述

SmoothQuant 的核心思想是：通过 **"平滑"变换** 将激活值中的量化难度（异常值）转移到权重上，使得 **权重和激活都可以用 INT8 表示** 而不显著损失精度。

在 SmoothQuant 逐张量（per-tensor）方案中，每个线性层 $Y = XW^T$ 的量化形式为：

$$Y \approx \alpha \cdot (X_{\text{int8}} \cdot W_{\text{int8}}^T)$$

其中：
- $W_{\text{int8}}$ 是离线量化的 INT8 权重，$\text{weight\_scale} = \frac{\max|W|}{127}$
- $X_{\text{int8}}$ 是运行时动态量化的 INT8 激活，$\text{input\_scale} = \frac{\max|X|}{127}$
- $\alpha = \text{input\_scale} \times \text{weight\_scale}$ 是反量化系数

---

### 2.2 量化数学原理

#### 2.2.1 权重量化（离线，Python 端完成）

给定 FP32/BF16 权重张量 $W$，逐张量对称量化为：

$$W_{\text{int8}}[i,j] = \text{clamp}\left(\text{round}\left(\frac{W[i,j]}{\text{weight\_scale}}\right), -128, 127\right)$$

其中 $\text{weight\_scale} = \frac{\max_{i,j}|W[i,j]|}{127}$。

反量化近似：$W[i,j] \approx W_{\text{int8}}[i,j] \times \text{weight\_scale}$

#### 2.2.2 激活量化（在线，GPU 运行时动态完成）

给定 FP16 输入 $X$（形状 $[M, K]$），逐张量对称量化为：

$$X_{\text{int8}}[i,j] = \text{clamp}\left(\text{round}\left(\frac{X[i,j]}{\text{input\_scale}}\right), -128, 127\right)$$

其中 $\text{input\_scale} = \frac{\max_{i,j}|X[i,j]|}{127}$，**每次推理动态计算**。

#### 2.2.3 INT8 矩阵乘法

$$Y_{\text{int32}}[i,j] = \sum_{k=0}^{K-1} X_{\text{int8}}[i,k] \times W_{\text{int8}}^T[k,j]$$

INT8 × INT8 的结果需要 INT32 累加器（避免溢出）。

#### 2.2.4 反量化

$$Y_{\text{fp16}}[i,j] = \alpha \times Y_{\text{int32}}[i,j]$$

其中 $\alpha = \text{input\_scale} \times \text{weight\_scale}$。

整个过程的正确性推导：

$$
\begin{aligned}
Y &= X \cdot W^T \\
  &\approx (X_{\text{int8}} \times \text{input\_scale}) \cdot (W_{\text{int8}} \times \text{weight\_scale})^T \\
  &= (\text{input\_scale} \times \text{weight\_scale}) \cdot (X_{\text{int8}} \cdot W_{\text{int8}}^T) \\
  &= \alpha \cdot Y_{\text{int32}}
\end{aligned}
$$

---

### 2.3 权重 INT8 数据的存储与解包

#### 2.3.1 磁盘格式

权重在 .bin 文件中以**原生 INT8 字节序列**存储，矩阵按行主序（row-major）排列：

```
地址偏移:  [0]  [1]  [2]  [3]  ...  [out_features × in_features - 1]
数据:     w00  w01  w02  w03  ...  w_{N-1,K-1}
类型:     int8 int8 int8 int8       int8
```

每个 `int8` 值的范围是 `[-128, 127]`，表示量化后的权重值。

#### 2.3.2 加载到 CPU

`set_sq_weights()` 直接 memcpy，无需任何格式转换：

```cpp
// INT8 数据完全无需解包——1 字节 = 1 个权重值
std::memcpy(qweight_.ptr<void>(), qweight_ptr, qweight_size * sizeof(int8_t));
```

与 AWQ 等分组量化方案（需要位操作拆包 4-bit 数据）不同，SmoothQuant 的 INT8 权重是**自描述的单字节数据**，无需任何解包操作。

#### 2.3.3 上传到 GPU

```cpp
void SQMatmulLayer::to_cuda() {
    // INT8 数据直传 GPU — 比 FP16 传输量减半
    cudaMemcpy(gpu_qweight.ptr<void>(), qweight_.ptr<void>(),
               total * sizeof(int8_t), cudaMemcpyHostToDevice);
}
```

上传后，INT8 权重**常驻 GPU 显存**，在整个推理过程中不再改变。

---

### 2.4 激活值的动态量化（FP16 → INT8）

激活值的量化在 GPU 上实时完成，分为两个 kernel 顺序执行：

#### 2.4.1 阶段 1：AbsMax 归约（sq_absmax_kernel）

**目的**：计算输入张量所有元素的绝对值最大值 $\text{absmax} = \max_i |X_i|$

```cuda
__global__ void sq_absmax_kernel(
    const half* __restrict__ input,
    int* __restrict__ d_max_as_int,   // atomicMax 目标（float 以 int 位模式存储）
    int total_elements)
{
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    const int gid = (blockIdx.x * blockDim.x + tid) * 4;

    // 每线程处理 4 个元素，使用 half2 向量化加载
    float local_max = 0.0f;
    if (gid + 3 < total_elements) {
        const half2* h2 = reinterpret_cast<const half2*>(input + gid);
        half2 v0 = __ldg(h2);      // 加载 2 个 FP16
        half2 v1 = __ldg(h2 + 1);  // 再加载 2 个 FP16
        float2 f0 = __half22float2(v0);
        float2 f1 = __half22float2(v1);
        local_max = fmaxf(fmaxf(fabsf(f0.x), fabsf(f0.y)),
                          fmaxf(fabsf(f1.x), fabsf(f1.y)));
    }

    // 共享内存树形归约
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    // block 级结果通过 atomicMax 汇总到全局
    if (tid == 0) {
        atomicMax(d_max_as_int, __float_as_int(sdata[0]));
    }
}
```

**关键技术细节**：

1. **atomicMax 的 float 技巧**：CUDA 没有 `atomicMax(float*)`，但 IEEE 754 正浮点数的位模式作为整数比较时保持大小关系（即 `__float_as_int(a) > __float_as_int(b)` 当且仅当 `a > b`（对于正数））。因此，将 float 的位模式转为 int 后使用 `atomicMax(int*)` 即可正确工作。调用前必须将 `d_max_as_int` 清零（`cudaMemsetAsync`），确保初始值为 0.0f 的位模式。

2. **共享内存树形归约**：每个 block（256 线程）先在共享内存中做 log2(256)=8 轮归约，得到 block 内最大值；再通过 `atomicMax` 跨 block 汇总。这比全局 atomicMax（每线程一次原子操作）高效得多。

3. **half2 向量化加载**：每次 `__ldg` 加载 4 字节（= 2 个 FP16），每线程处理 4 个元素，提高显存带宽利用率。

#### 2.4.2 阶段 2：量化 + Alpha 计算（sq_quantize_and_alpha_kernel）

此 kernel 作为**独立 kernel** 在 absmax kernel 之后启动（同一 stream 内隐式同步），读取已确定的 absmax 值：

```cuda
__global__ void sq_quantize_and_alpha_kernel(
    const half* __restrict__ input_fp16,
    int8_t* __restrict__ output_int8,
    const int* __restrict__ d_max_as_int,
    float weight_scale,
    float* __restrict__ d_alpha,
    int total_elements)
{
    // 从设备内存读取最终的 absmax（所有 block 的 atomicMax 结果）
    const float absmax = __int_as_float(*d_max_as_int);

    // 计算量化比例和反量化系数
    const float inv_scale = (absmax > 1e-6f) ? 127.0f / absmax : 0.0f;

    // thread(0,0) 计算并写入 alpha（只需写一次）
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const float input_scale = (absmax > 1e-6f) ? absmax / 127.0f : 0.0f;
        *d_alpha = input_scale * weight_scale;
    }

    // 每线程量化 4 个元素（向量化写入）
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (idx + 3 < total_elements) {
        const half2* h2 = reinterpret_cast<const half2*>(input_fp16 + idx);
        half2 v0 = __ldg(h2);
        half2 v1 = __ldg(h2 + 1);
        float2 f0 = __half22float2(v0);
        float2 f1 = __half22float2(v1);

        // 量化：round(x * 127/absmax)，并 clamp 到 [-128, 127]
        int i0 = max(-128, min(127, __float2int_rn(f0.x * inv_scale)));
        int i1 = max(-128, min(127, __float2int_rn(f0.y * inv_scale)));
        int i2 = max(-128, min(127, __float2int_rn(f1.x * inv_scale)));
        int i3 = max(-128, min(127, __float2int_rn(f1.y * inv_scale)));

        // 4 个 INT8 打包为 1 个 INT32，单次 32-bit 写入
        int32_t packed = (i0 & 0xFF) | ((i1 & 0xFF) << 8) |
                         ((i2 & 0xFF) << 16) | ((i3 & 0xFF) << 24);
        *reinterpret_cast<int32_t*>(output_int8 + idx) = packed;
    }
}
```

**关键技术细节**：

1. **两个 kernel 分离的原因**：v1 版本尝试在单个 kernel 中融合 absmax 和量化，但存在**inter-block 竞态条件**——当某些 block 还未完成 absmax 的 atomicMax 时，其他 block 已经开始读取 absmax 并量化。分离为两个 kernel 后，CUDA 流内的隐式同步（kernel 串行执行）保证阶段 1 的所有 block 在阶段 2 启动前必定完成。

2. **INT8 打包写入**：将 4 个 INT8 值打包进一个 `int32_t`，每个占 8 位：
   ```
   int32_t packed:
   ┌────────┬────────┬────────┬────────┐
   │ i3[7:0]│ i2[7:0]│ i1[7:0]│ i0[7:0]│
   │ bit 31 │ bit 23 │ bit 15 │ bit 7  │
   └────────┴────────┴────────┴────────┘
   ```
   这种打包布局恰好与 `__dp4a` 指令的输入格式一致，后续无需再拆包。

3. **Alpha 在 GPU 上计算**：$\alpha = \text{input\_scale} \times \text{weight\_scale}$ 直接在设备端完成，避免 GPU→CPU→GPU 的往返，保证 **CUDA Graph 兼容性**。

---

### 2.5 INT8 矩阵乘法与 `__dp4a` 硬件指令

#### 2.5.1 `__dp4a` 指令原理

`__dp4a(int a, int b, int c)` 是 NVIDIA 在 SM61+ 架构上提供的硬件 INT8 点积指令，它将两个 `int32` 解释为各 4 个 `int8`，做 4 元素点积并累加到 `int32`：

$$
\text{result} = c + \sum_{i=0}^{3} a_i \times b_i
$$

其中 $a_i, b_i$ 是 `a, b` 中的第 $i$ 个 `int8` 字节。

**一条指令完成 4 次乘加（4 MACs）**，相比标量代码效率提升 ~3×。

#### 2.5.2 128-bit 向量化加载（int4）

CUDA 的 `int4` 类型是 128-bit 结构体（4 × int32 = 16 字节 = 16 个 INT8），每次内存加载获取 16 个 INT8 元素：

```cuda
// int4 = { int x, int y, int z, int w } = 128 bits = 16 bytes
const int4* input_i16 = reinterpret_cast<const int4*>(input_int8);
const int4* weight_i16 = reinterpret_cast<const int4*>(w_row);

for (int i = lane_id; i < num_vec16; i += 32) {
    int4 x = __ldg(input_i16 + i);   // 加载 16 个 INT8 输入元素
    int4 w = __ldg(weight_i16 + i);  // 加载 16 个 INT8 权重元素

    // 4 次 __dp4a，每次处理 4 个 INT8 → 16 个元素全部完成
    acc = __dp4a(x.x, w.x, acc);  // 元素 [0..3]
    acc = __dp4a(x.y, w.y, acc);  // 元素 [4..7]
    acc = __dp4a(x.z, w.z, acc);  // 元素 [8..11]
    acc = __dp4a(x.w, w.w, acc);  // 元素 [12..15]
}
```

所以**每次循环迭代**：
- 加载 16 + 16 = 32 字节
- 执行 4 次 `__dp4a` = 16 次 MAC
- 全部在寄存器中完成（无共享内存开销）

#### 2.5.3 Warp Shuffle 归约

32 个 lane 的部分和通过 warp shuffle 汇总：

```cuda
// 5 轮 shuffle：32 → 16 → 8 → 4 → 2 → 1
for (int offset = 16; offset > 0; offset /= 2) {
    acc += __shfl_down_sync(0xffffffff, acc, offset);
}
```

最终 `lane_id == 0` 持有完整的点积结果。

---

### 2.6 反量化：INT32 累加结果 → FP16 输出

GEMV kernel（`sq_gemv_int8_kernel`）的反量化过程非常直接：

```cuda
if (lane_id == 0) {
    // alpha = input_scale × weight_scale（从设备内存读取）
    // acc = INT32 点积结果
    // 反量化 + 类型转换：FP32 → FP16
    output_fp16[out_idx] = __float2half(alpha * static_cast<float>(acc));
}
```

数学推导：

$$
\begin{aligned}
Y[n] &= \sum_k X[k] \cdot W[n,k] \\
     &\approx \sum_k (X_{\text{int8}}[k] \cdot \text{input\_scale}) \cdot (W_{\text{int8}}[n,k] \cdot \text{weight\_scale}) \\
     &= \underbrace{(\text{input\_scale} \times \text{weight\_scale})}_{\alpha} \cdot \underbrace{\sum_k X_{\text{int8}}[k] \cdot W_{\text{int8}}[n,k]}_{\text{acc (INT32)}} \\
     &= \alpha \cdot \text{acc}
\end{aligned}
$$

**反量化就是一次标量乘法**。由于 $\alpha$ 是标量（per-tensor 量化），所有输出通道共享同一个 $\alpha$，计算开销极低。

---

### 2.7 CUTLASS Prefill 路径的反量化

当 $M > 1$（Prefill 阶段），使用 CUTLASS INT8 Tensor Core GEMM，反量化通过 **CUTLASS Epilogue** 自动融合：

```cpp
// CUTLASS GEMM 类型定义
using CutlassInt8Gemm = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,      // A: INT8 输入 [M,K]
    int8_t, cutlass::layout::ColumnMajor,    // B: INT8 权重 [K,N] 列主序
    cutlass::half_t, cutlass::layout::RowMajor,  // C/D: FP16 输出 [M,N]
    int32_t,                                  // 累加器类型: INT32
    cutlass::arch::OpClassTensorOp,           // 使用 Tensor Core
    cutlass::arch::Sm80,                      // SM80/SM87 架构
    cutlass::gemm::GemmShape<256, 128, 64>,   // Threadblock tile [M,N,K]
    cutlass::gemm::GemmShape<64, 64, 64>,     // Warp tile
    cutlass::gemm::GemmShape<16, 8, 32>,      // MMA 指令 (16×8×32)
    cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t, 8, int32_t, float>,  // Epilogue: D = alpha * C_int32
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3>;                                        // Pipeline stages
```

**Epilogue 反量化机制**：

```cpp
// alpha 指针指向 GPU 内存（由 sq_quantize_and_alpha_kernel 写入）
typename CutlassInt8Gemm::EpilogueOutputOp::Params epilogue_params(
    g_workspace.alpha,  // device-side alpha pointer
    nullptr);           // beta = nullptr (不累加旧的 C)

// 在 CUTLASS 内部，epilogue 自动执行:
// output_fp16[m,n] = alpha * accumulator_int32[m,n]
```

CUTLASS 的 `LinearCombination` epilogue 在 Tensor Core 矩阵乘法完成后（INT32 累加结果在寄存器中），立即在同一个 kernel 内将 INT32 乘以 alpha 并转换为 FP16 输出。**这是融合的——没有额外的 kernel 启动或中间 buffer**。

**自适应 tile 选择**：

```cpp
if (M <= 32) {
    // 小 tile (128×128×64) — 减少 SM 浪费
    CutlassInt8GemmSmall gemm_op;
    gemm_op(stream);
} else {
    // 大 tile (256×128×64) — 最大化 Tensor Core 利用率
    CutlassInt8Gemm gemm_op;
    gemm_op(stream);
}
```

---

### 2.8 共享量化优化（Shared Quantization）

#### 2.8.1 问题分析

在 Transformer 的 Attention 层中，Q/K/V 三个投影的输入相同（都是 `rms_out`）。若各自独立执行 SQ GEMM，每次都要：

```
memset → absmax → quantize → GEMV   (4 kernels × 3 = 12 kernels/layer)
```

但三次量化的结果完全相同！

#### 2.8.2 优化方案

量化一次，复用三次：

```
quantize_input():  memset → absmax → quantize   (3 kernels, workspace 存储)
forward_preq(Q):   preq_GEMV                      (1 kernel)
forward_preq(K):   preq_GEMV                      (1 kernel)
forward_preq(V):   preq_GEMV                      (1 kernel)
                                         共计 6 kernels/layer（减少 50%）
```

#### 2.8.3 量化共享的实现

```cuda
// sq_quantize_input_cu: 量化输入并存储到全局 workspace
void sq_quantize_input_cu(const half* input_fp16, int K, cudaStream_t stream)
{
    g_workspace.ensure(static_cast<size_t>(K));
    constexpr int kThreads = 256;
    int blocks = (K + kThreads * 4 - 1) / (kThreads * 4);

    cudaMemsetAsync(g_workspace.max_int, 0, sizeof(int), stream);

    sq_absmax_kernel<<<blocks, kThreads, kThreads * sizeof(float), stream>>>(
        input_fp16, g_workspace.max_int, K);

    // 注意 weight_scale=1.0，所以 alpha = input_scale × 1.0 = input_scale
    sq_quantize_and_alpha_kernel<<<blocks, kThreads, 0, stream>>>(
        input_fp16, g_workspace.input_int8, g_workspace.max_int,
        1.0f, g_workspace.alpha, K);
    // 此时 g_workspace.alpha = input_scale = absmax/127
    // 此时 g_workspace.input_int8 = 量化后的 INT8 输入
}
```

#### 2.8.4 预量化 GEMV

```cuda
__global__ void sq_gemv_preq_kernel(
    const int8_t* __restrict__ input_int8,
    const int8_t* __restrict__ weight_int8,
    half* __restrict__ output_fp16,
    const float* __restrict__ d_input_scale,  // 从 workspace 读取
    float weight_scale,
    int K, int N)
{
    // alpha = input_scale(设备内存) × weight_scale(每层不同)
    const float alpha = (*d_input_scale) * weight_scale;

    // dp4a GEMV（与 sq_gemv_int8_kernel 相同的计算逻辑）
    // ...

    if (lane_id == 0) {
        output_fp16[out_idx] = __float2half(alpha * static_cast<float>(acc));
    }
}
```

这里 `d_input_scale` 读取的是 `sq_quantize_input_cu` 写入 `g_workspace.alpha` 的值，而 `weight_scale` 是每个层各自不同的常量——在 Q、K、V 三个 GEMV 中分别传入对应层的 `weight_scale`。

**每层节省**：12 kernels → 6 kernels = 减少 6 次 kernel 启动  
**全模型节省**：36 层 × 6 = **216 次 kernel 启动**（显著降低 kernel launch overhead）

---

### 2.9 融合 FFN 中的反量化

#### 2.9.1 FFN 计算流程

Qwen3 的 FFN 结构为 SwiGLU：

$$\text{FFN}(x) = \text{SiLU}(W_1 \cdot x) \odot (W_3 \cdot x)$$

其中 $\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$

#### 2.9.2 融合 kernel 的反量化

`sq_fused_ffn_gemv_kernel` 在单个 kernel 中完成 W1 点积、W3 点积、反量化和 SwiGLU 激活：

```cuda
if (lane_id == 0) {
    // 反量化：各自使用自己的 alpha
    float gate = alpha_w1 * static_cast<float>(acc_gate);  // W1·x 的反量化
    float up   = alpha_w3 * static_cast<float>(acc_up);    // W3·x 的反量化

    // SiLU(gate) × up
    float gate_activated = gate / (1.0f + __expf(-gate));
    output_fp16[row] = __float2half(gate_activated * up);
}
```

注意这里有**两个不同的 alpha**：
- $\alpha_{W1} = \text{input\_scale} \times \text{w1\_weight\_scale}$
- $\alpha_{W3} = \text{input\_scale} \times \text{w3\_weight\_scale}$

`input_scale` 相同（共享量化），但 $\text{w1\_weight\_scale}$ 和 $\text{w3\_weight\_scale}$ 不同（各层各自的权重缩放因子）。

#### 2.9.3 融合的收益

| 方案 | Kernel 启动次数 | 中间 Buffer |
|------|----------------|-------------|
| 分离 W1 + 分离 W3 + SwiGLU | 4+4+1 = 9 | 2 × [hidden_dim] FP16 |
| 融合 FFN kernel | 3+1 = 4 | 0（直接输出） |

融合方案节省 5 次 kernel 启动和 2 个中间缓冲区的显存。

---

### 2.10 端到端数据流总结

以 Decode 阶段（M=1）的一个 Attention 层为例，完整的数据流：

```
输入: rms_out [4096] FP16（RMSNorm 输出）
  │
  ├── [共享量化] sq_quantize_input_cu()
  │     ├── cudaMemsetAsync(max_int = 0)
  │     ├── sq_absmax_kernel:
  │     │     FP16 输入 → half2 向量化加载 → 块内 shmem 归约 → atomicMax
  │     │     → max_int = __float_as_int(absmax)
  │     └── sq_quantize_and_alpha_kernel:
  │           读取 absmax → inv_scale = 127/absmax
  │           FP16 × inv_scale → round → clamp[-128,127] → 4×INT8 打包为 INT32 写入
  │           alpha = (absmax/127) × 1.0 = input_scale
  │
  │     workspace 中: input_int8[4096], alpha = input_scale
  │
  ├── [Q 投影] sq_gemv_preq_kernel(weight=wq[4096,4096], ws=wq_scale)
  │     alpha = input_scale × wq_weight_scale
  │     int4 加载 → __dp4a 点积 → warp shuffle 归约
  │     → output = __float2half(alpha × acc)
  │     → query_out [4096] FP16
  │
  ├── [K 投影] sq_gemv_preq_kernel(weight=wk[1024,4096], ws=wk_scale)
  │     → key_out [1024] FP16
  │
  └── [V 投影] sq_gemv_preq_kernel(weight=wv[1024,4096], ws=wv_scale)
        → value_out [1024] FP16

... (Attention 计算 + O 投影) ...

输入: FFN 输入 [4096] FP16
  │
  └── [融合 FFN] sq_fused_ffn_cu()
        ├── cudaMemsetAsync(max_int = 0)
        ├── sq_absmax_kernel → absmax
        ├── sq_quantize_and_alpha_kernel → input_int8, input_scale
        └── sq_fused_ffn_gemv_kernel:
              对每个输出行 n:
                acc_gate = Σ_k input_int8[k] × w1[n,k]  (dp4a)
                acc_up   = Σ_k input_int8[k] × w3[n,k]  (dp4a)
                gate = input_scale × w1_ws × acc_gate    (反量化)
                up   = input_scale × w3_ws × acc_up      (反量化)
                output[n] = SiLU(gate) × up              (SwiGLU 激活)
              → ffn_out [12288] FP16
```

**性能对比**：

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| Decode 吞吐量 | 10.6 tokens/s | 17.66 tokens/s |
| 提升 | — | **+66.6%** |
| 每层 Kernel 启动 (QKV) | 12 | 6 |
| 每层 Kernel 启动 (FFN) | 9 | 4 |
| 全模型 Kernel 启动节省 | — | 396 次/step |

---

## 第三部分：sq_gemm_kernel.cu CUDA Kernel 逐一详解

> 源码文件：`kuiper/source/op/kernels/cuda/sq_gemm_kernel.cu`（659 行）

### 3.1 文件总览与架构

该文件是 SmoothQuant INT8 推理的计算核心，包含 **5 个 `__global__` CUDA Kernel**、**1 个 Workspace 结构体**、**2 个内部调度函数**和 **4 个公共 API 函数**。整体架构如下：

```
sq_gemm_kernel.cu (659 行)
├── CUTLASS 类型定义 (第 40-73 行)
│   ├── CutlassInt8Gemm         — 大 tile (256×128×64)
│   └── CutlassInt8GemmSmall    — 小 tile (128×128×64)
│
├── __global__ Kernel 函数 (第 82-399 行)
│   ├── sq_absmax_kernel              — AbsMax 归约
│   ├── sq_quantize_and_alpha_kernel  — 量化 + Alpha 计算
│   ├── sq_gemv_int8_kernel           — INT8 GEMV (dp4a)
│   ├── sq_gemv_preq_kernel           — 预量化 GEMV
│   └── sq_fused_ffn_gemv_kernel      — 融合 FFN GEMV + SwiGLU
│
├── SQWorkspace 结构体 (第 401-419 行)
│   └── g_workspace (static 全局实例)
│
├── 内部调度函数 (第 428-530 行)
│   ├── sq_gemv_m1()       — M=1 Decode 路径
│   └── sq_gemm_cutlass()  — M>1 Prefill 路径
│
└── 公共 API (第 535-659 行)
    ├── sq_gemm_cu()            — 主入口 (M=1/M>1 分发)
    ├── sq_fused_ffn_cu()       — 融合 FFN 入口
    ├── sq_quantize_input_cu()  — 共享量化入口
    └── sq_gemv_preq_cu()       — 预量化 GEMV 入口
```

---

### 3.2 CUTLASS INT8 GEMM 类型定义

**源码位置**：第 40-73 行

```cuda
// 大 tile — 用于长序列 Prefill (M>32)
using CutlassInt8Gemm = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,           // A: INT8 输入 [M,K] 行主序
    int8_t, cutlass::layout::ColumnMajor,         // B: INT8 权重 [K,N] 列主序
    cutlass::half_t, cutlass::layout::RowMajor,   // C/D: FP16 输出 [M,N]
    int32_t,                                       // 累加器: INT32
    cutlass::arch::OpClassTensorOp,                // 使用 Tensor Core
    cutlass::arch::Sm80,                           // SM80/SM87 (Orin 兼容)
    cutlass::gemm::GemmShape<256, 128, 64>,        // Threadblock tile
    cutlass::gemm::GemmShape<64, 64, 64>,          // Warp tile
    cutlass::gemm::GemmShape<16, 8, 32>,           // MMA 指令形状
    cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t, 8, int32_t, float>,       // Epilogue: D = alpha * acc
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3>;                                            // Pipeline stages

// 小 tile — 用于短序列 Prefill (M≤32)
using CutlassInt8GemmSmall = cutlass::gemm::device::Gemm<
    /* 同上，仅 ThreadblockShape 改为 128×128×64 */
    ...
    cutlass::gemm::GemmShape<128, 128, 64>,        // 更小的 tile
    ...>;
```

**原理讲解**：

| 模板参数 | 大 tile | 小 tile | 说明 |
|----------|---------|---------|------|
| ThreadblockShape | 256×128×64 | 128×128×64 | 每个 threadblock 计算的输出块大小 |
| WarpShape | 64×64×64 | 64×64×64 | 每个 warp 负责的分片 |
| InstructionShape | 16×8×32 | 16×8×32 | SM80 的 INT8 MMA 指令 |
| Pipeline Stages | 3 | 3 | 双缓冲+预取深度 |
| Epilogue align | 8 | 8 | FP16 向量化写回宽度 |

- **为何需要两套 tile**：当 $M \leq 32$ 时，大 tile（256×128）的 M 维度会产生大量空闲线程，利用率低；小 tile（128×128）更匹配小 batch。
- **ColumnMajor 权重**：权重存储为 `[N, K]` 行主序（row-major），等价于 `[K, N]` 列主序（column-major），CUTLASS 按列主序访问权重实现转置乘法。
- **LinearCombination Epilogue**：自动将 INT32 累加结果乘以 `alpha`（设备指针）并转为 FP16，无需额外 kernel。

---

### 3.3 Kernel 1：sq_absmax_kernel — 逐张量 AbsMax 归约

**源码位置**：第 82-116 行  
**功能**：计算输入张量所有元素的绝对值最大值 $\text{absmax} = \max_i |x_i|$

#### 带注释的完整源码

```cuda
__global__ void sq_absmax_kernel(
    const half* __restrict__ input,       // [total_elements] FP16 输入
    int* __restrict__ d_max_as_int,       // [1] 全局 absmax（float 以 int 位模式存储）
    int total_elements)                   // 元素总数
{
    // ① 动态共享内存：每个线程一个 float 槽位
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    // ② 全局索引：每线程处理 4 个元素（向量化访问）
    const int gid = (blockIdx.x * blockDim.x + tid) * 4;

    // ③ 每线程局部最大值初始化
    float local_max = 0.0f;

    // ④ 主路径：对齐的 half2×2 向量化加载（4 个 FP16 = 8 字节）
    if (gid + 3 < total_elements) {
        const half2* h2 = reinterpret_cast<const half2*>(input + gid);
        half2 v0 = __ldg(h2);      // 加载 input[gid+0..1]
        half2 v1 = __ldg(h2 + 1);  // 加载 input[gid+2..3]
        float2 f0 = __half22float2(v0);  // 转 FP32
        float2 f1 = __half22float2(v1);
        // 4 个绝对值取最大
        local_max = fmaxf(fmaxf(fabsf(f0.x), fabsf(f0.y)),
                          fmaxf(fabsf(f1.x), fabsf(f1.y)));
    } else {
        // ⑤ 尾部处理：标量逐个加载
        for (int i = gid; i < total_elements && i < gid + 4; ++i) {
            local_max = fmaxf(local_max, fabsf(__half2float(input[i])));
        }
    }

    // ⑥ 写入共享内存
    sdata[tid] = local_max;
    __syncthreads();

    // ⑦ 块内树形归约：log2(blockDim.x) 轮
    //    256 线程 → 8 轮：128→64→32→16→8→4→2→1
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // ⑧ 块内最大值通过 atomicMax 汇总到全局
    if (tid == 0) {
        atomicMax(d_max_as_int, __float_as_int(sdata[0]));
    }
}
```

#### 原理详解

**1. 为什么用 `int` 存储 `float` 的 absmax？**

CUDA 不提供 `atomicMax(float*)` 原子操作。但 IEEE 754 正浮点数有一个关键性质：**对于正数 $a > b \geq 0$，有 `__float_as_int(a) > __float_as_int(b)`**。因此可以将 float 的位模式作为 int 使用整数 `atomicMax`，对于非负数（absmax 一定非负）结果正确。

初始化要求：调用前必须 `cudaMemsetAsync(d_max_as_int, 0, sizeof(int))`，这将 d_max_as_int 设为全零，等价于 `__float_as_int(0.0f) = 0`。

**2. 分层归约策略**

```
层次 1: 每线程 → 4 个元素取 max   (向量化减少全局访存次数)
层次 2: 块内共享内存 → 树形归约   (256 线程 → 1 个值)
层次 3: 块间 → atomicMax          (多块 → 全局 1 个值)
```

对于 K=4096，block 数 = $\lceil 4096 / (256 \times 4) \rceil = 4$，仅需 4 次 atomicMax。

**3. 启动配置**

```
Grid:  (K + 256*4 - 1) / (256*4) 个 block
Block: 256 threads
Shared Memory: 256 × sizeof(float) = 1024 bytes
```

---

### 3.4 Kernel 2：sq_quantize_and_alpha_kernel — 量化 + Alpha 计算

**源码位置**：第 124-166 行  
**功能**：(1) 读取已确定的 absmax，(2) 将 FP16 输入量化为 INT8，(3) 计算反量化系数 $\alpha$

#### 带注释的完整源码

```cuda
__global__ void sq_quantize_and_alpha_kernel(
    const half* __restrict__ input_fp16,  // [N] FP16 输入
    int8_t* __restrict__ output_int8,     // [N] INT8 输出
    const int* __restrict__ d_max_as_int, // [1] absmax (来自 kernel 1)
    float weight_scale,                   // 主机端传入的权重缩放因子
    float* __restrict__ d_alpha,          // [1] 输出：反量化系数 alpha
    int total_elements)                   // 元素个数
{
    // ① 从设备端读取最终 absmax（所有 block 的 atomicMax 结果）
    const float absmax = __int_as_float(*d_max_as_int);
    // ② 量化比例：inv_scale = 127 / absmax
    const float inv_scale = (absmax > 1e-6f) ? 127.0f / absmax : 0.0f;

    // ③ 只有 block0 的 thread0 计算并写入 alpha（全局只需写一次）
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const float input_scale = (absmax > 1e-6f) ? absmax / 127.0f : 0.0f;
        // alpha = input_scale × weight_scale
        //       = (absmax/127) × weight_scale
        *d_alpha = input_scale * weight_scale;
    }

    // ④ 每线程处理 4 个元素
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (idx + 3 < total_elements) {
        // ⑤ 向量化加载 4 个 FP16
        const half2* h2 = reinterpret_cast<const half2*>(input_fp16 + idx);
        half2 v0 = __ldg(h2);
        half2 v1 = __ldg(h2 + 1);
        float2 f0 = __half22float2(v0);
        float2 f1 = __half22float2(v1);

        // ⑥ 量化：round(x × 127/absmax)，clamp 到 [-128, 127]
        int i0 = max(-128, min(127, __float2int_rn(f0.x * inv_scale)));
        int i1 = max(-128, min(127, __float2int_rn(f0.y * inv_scale)));
        int i2 = max(-128, min(127, __float2int_rn(f1.x * inv_scale)));
        int i3 = max(-128, min(127, __float2int_rn(f1.y * inv_scale)));

        // ⑦ 4 个 INT8 打包为 1 个 INT32，单次 32-bit 写入
        //    内存布局：[i0][i1][i2][i3]，每个占 8 bit
        //    这种布局恰好与 __dp4a 输入格式一致
        int32_t packed = (i0 & 0xFF) | ((i1 & 0xFF) << 8) |
                         ((i2 & 0xFF) << 16) | ((i3 & 0xFF) << 24);
        *reinterpret_cast<int32_t*>(output_int8 + idx) = packed;
    } else {
        // ⑧ 尾部标量处理
        for (int i = idx; i < total_elements && i < idx + 4; ++i) {
            float val = __half2float(input_fp16[i]) * inv_scale;
            output_int8[i] = static_cast<int8_t>(max(-128, min(127, __float2int_rn(val))));
        }
    }
}
```

#### 原理详解

**1. 为什么是独立 kernel 而非与 absmax 融合？**

v1 版本尝试在**单个 kernel** 中同时做 absmax 和量化（使用 `atomicAdd` 做 grid-level 计数器模拟 barrier）。但这存在**竞态条件**：当某些 block 还未完成 atomicMax 写入时，其他 block 已经读取了不完整的 absmax 并开始量化，导致量化结果不确定。

分成 2 个 kernel 后，CUDA 同一 stream 内 kernel 是串行的——kernel 1 的所有 block 必定在 kernel 2 启动前全部完成，保证 kernel 2 读到的 absmax 是最终值。

**2. Alpha 的含义与计算位置**

$$\alpha = \frac{\text{absmax}}{127} \times \text{weight\_scale} = \text{input\_scale} \times \text{weight\_scale}$$

$\alpha$ 在**设备端计算并存储到 GPU 内存**（`d_alpha`），后续的 GEMV/GEMM kernel 直接从设备内存读取。这种设计保证了 **CUDA Graph 兼容性**——不需要 GPU→CPU→GPU 的往返。

**3. 4×INT8 打包写入**

将 4 个量化后的 INT8 值打包为一个 `int32_t`，一次 32-bit 对齐写入：

```
int32_t packed 内存布局:
  Byte 0: i0  (bit [7:0])
  Byte 1: i1  (bit [15:8])
  Byte 2: i2  (bit [23:16])
  Byte 3: i3  (bit [31:24])
```

这种布局与 `__dp4a` 输入格式天然对齐，后续 GEMV kernel 直接将 `int32_t` 传给 `__dp4a` 无需拆包。

---

### 3.5 Kernel 3：sq_gemv_int8_kernel — INT8 GEMV（dp4a + 128-bit 加载）

**源码位置**：第 175-231 行  
**功能**：Decode 阶段（M=1）的核心计算——INT8 矩阵-向量乘法

$$\text{output}[n] = \alpha \times \sum_{k=0}^{K-1} \text{input\_int8}[k] \times \text{weight\_int8}[n, k]$$

#### 带注释的完整源码

```cuda
// __launch_bounds__(256, 4)：
//   256 = 每 block 最多 256 线程
//   4   = 每 SM 最多 4 个活跃 block（控制寄存器使用量）
__global__ __launch_bounds__(256, 4)
void sq_gemv_int8_kernel(
    const int8_t* __restrict__ input_int8,   // [K] 量化后的激活向量
    const int8_t* __restrict__ weight_int8,  // [N, K] 权重矩阵（行主序）
    half* __restrict__ output_fp16,          // [N] FP16 输出
    const float* __restrict__ d_alpha,       // [1] 反量化系数（设备内存）
    int K,                                    // 输入维度
    int N)                                    // 输出维度
{
    // ① 线程组织：256 线程 = 8 个 warp
    const int warp_id = threadIdx.x / 32;    // 0..7
    const int lane_id = threadIdx.x % 32;    // 0..31

    // ② 每个 warp 处理一个输出通道
    //    blockIdx.x * 8 + warp_id → 第几个输出元素
    const int out_idx = blockIdx.x * 8 + warp_id;
    if (out_idx >= N) return;

    // ③ 从设备内存读取 alpha（所有线程共享同一个值）
    const float alpha = *d_alpha;

    // ④ 定位该输出通道对应的权重行
    const int8_t* w_row = weight_int8 + static_cast<int64_t>(out_idx) * K;

    // ⑤ INT32 累加器（防止 INT8 × INT8 溢出）
    int32_t acc = 0;

    // ⑥ 主循环：128-bit 向量化加载 + __dp4a
    //    int4 = 128 bit = 4 × int32 = 16 × int8
    //    每次迭代处理 16 个 INT8 元素
    const int num_vec16 = K / 16;
    const int4* input_i16 = reinterpret_cast<const int4*>(input_int8);
    const int4* weight_i16 = reinterpret_cast<const int4*>(w_row);

    // ⑦ warp 内 32 个 lane 协作遍历 K 维度
    //    lane_id=0 处理 chunk 0,32,64,...
    //    lane_id=1 处理 chunk 1,33,65,...
    //    每个 chunk = 16 个 INT8 元素
    #pragma unroll 4
    for (int i = lane_id; i < num_vec16; i += 32) {
        // 一次 __ldg 加载 128 bit = 16 个 INT8
        int4 x = __ldg(input_i16 + i);   // 输入 [i*16 .. i*16+15]
        int4 w = __ldg(weight_i16 + i);  // 权重 [i*16 .. i*16+15]

        // 4 次 __dp4a，每次处理 4 个 INT8 乘累加
        // __dp4a(a, b, c) = c + Σ(a_byte_i × b_byte_i), i=0..3
        acc = __dp4a(x.x, w.x, acc);   // 元素 [0..3]
        acc = __dp4a(x.y, w.y, acc);   // 元素 [4..7]
        acc = __dp4a(x.z, w.z, acc);   // 元素 [8..11]
        acc = __dp4a(x.w, w.w, acc);   // 元素 [12..15]
    }

    // ⑧ 处理 K 不能被 16 整除的余数部分
    const int base = num_vec16 * 16;
    for (int i = base + lane_id; i < K; i += 32) {
        acc += static_cast<int32_t>(input_int8[i]) * static_cast<int32_t>(w_row[i]);
    }

    // ⑨ Warp 内 shuffle 归约：32 个 lane 的部分和 → lane 0
    //    5 轮：offset=16,8,4,2,1
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    // ⑩ lane 0 写出反量化后的 FP16 结果
    if (lane_id == 0) {
        // 反量化：FP16 = alpha × INT32 累加值
        output_fp16[out_idx] = __float2half(alpha * static_cast<float>(acc));
    }
}
```

#### 原理详解

**1. 线程组织模型**

```
Block (256 threads)
├── Warp 0 (lane 0-31)  → output[blockIdx.x * 8 + 0]
├── Warp 1 (lane 0-31)  → output[blockIdx.x * 8 + 1]
├── ...
└── Warp 7 (lane 0-31)  → output[blockIdx.x * 8 + 7]
```

- 每个 **warp**（32 线程）计算一个输出元素
- 每个 **block**（8 warps）计算 8 个输出元素
- Grid 大小 = $\lceil N / 8 \rceil$

**2. `__dp4a` 指令详解**

```
__dp4a(int a, int b, int c):
  a = [a3|a2|a1|a0]  (4 packed INT8)
  b = [b3|b2|b1|b0]  (4 packed INT8)
  return c + a0*b0 + a1*b1 + a2*b2 + a3*b3
```

一条指令执行 **4 次 INT8 乘加**。对比 v1 的标量实现（手动拆字节 + 4 次乘法 + 3 次加法 = 7 条指令），`__dp4a` 将同样工作压缩到 **1 条**指令。

**3. 128-bit 向量化加载**

```
int4 = { int x, int y, int z, int w } = 4 × 32bit = 128 bit = 16 个 INT8
```

每次 `__ldg(int4*)` 从全局内存加载 16 字节，配合 4 次 `__dp4a` 消费全部 16 个 INT8 元素。相比 v1 的 `int32_t` 加载（4 字节 / 次），带宽利用率提升 **4×**。

**4. 对于 K=4096 的工作量分析**

```
num_vec16 = 4096 / 16 = 256 个 int4 chunk
每 lane 处理: 256 / 32 = 8 个 chunk
每 lane 迭代: 8 次循环 × 4 dp4a = 32 条 dp4a 指令
```

**5. `__launch_bounds__` 的意义**

`__launch_bounds__(256, 4)` 告诉编译器每 block 最多 256 线程、每 SM 最多 4 个 block。这让编译器可以更激进地分配寄存器（每线程最多 $32768 / (256 \times 4) = 32$ 个寄存器），避免 register spill。

---

### 3.6 Kernel 4：sq_gemv_preq_kernel — 预量化 GEMV

**源码位置**：第 240-290 行  
**功能**：使用已预量化的 INT8 输入执行 GEMV，适用于 QKV 共享量化场景

#### 带注释的完整源码

```cuda
__global__ __launch_bounds__(256, 4)
void sq_gemv_preq_kernel(
    const int8_t* __restrict__ input_int8,   // [K] 预量化的 INT8 输入（来自 workspace）
    const int8_t* __restrict__ weight_int8,  // [N, K] 当前层的 INT8 权重
    half* __restrict__ output_fp16,          // [N] FP16 输出
    const float* __restrict__ d_input_scale, // [1] 设备端指针：input_scale = absmax/127
    float weight_scale,                      // 主机端常量：当前层的 weight_scale
    int K,
    int N)
{
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    const int out_idx = blockIdx.x * 8 + warp_id;
    if (out_idx >= N) return;

    // ① 关键区别：alpha = input_scale(设备内存) × weight_scale(主机常量)
    //    input_scale 来自共享量化步骤（所有 QKV 层相同）
    //    weight_scale 是每层独立的（Q/K/V 各不同）
    const float alpha = (*d_input_scale) * weight_scale;

    const int8_t* w_row = weight_int8 + static_cast<int64_t>(out_idx) * K;
    int32_t acc = 0;

    // ② 主循环：与 sq_gemv_int8_kernel 完全相同的 dp4a + int4 计算
    const int num_vec16 = K / 16;
    const int4* input_i16 = reinterpret_cast<const int4*>(input_int8);
    const int4* weight_i16 = reinterpret_cast<const int4*>(w_row);

    #pragma unroll 4
    for (int i = lane_id; i < num_vec16; i += 32) {
        int4 x = __ldg(input_i16 + i);
        int4 w = __ldg(weight_i16 + i);
        acc = __dp4a(x.x, w.x, acc);
        acc = __dp4a(x.y, w.y, acc);
        acc = __dp4a(x.z, w.z, acc);
        acc = __dp4a(x.w, w.w, acc);
    }

    // ③ 余数处理 + warp shuffle 归约（同 kernel 3）
    const int base = num_vec16 * 16;
    for (int i = base + lane_id; i < K; i += 32) {
        acc += static_cast<int32_t>(input_int8[i]) * static_cast<int32_t>(w_row[i]);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    if (lane_id == 0) {
        output_fp16[out_idx] = __float2half(alpha * static_cast<float>(acc));
    }
}
```

#### 与 sq_gemv_int8_kernel 的关键差异

| 对比项 | sq_gemv_int8_kernel | sq_gemv_preq_kernel |
|--------|---------------------|---------------------|
| 输入来源 | kernel 2 刚量化的 INT8 | workspace 中预存的 INT8 |
| alpha 来源 | `*d_alpha`（预计算好） | `(*d_input_scale) * weight_scale`（运行时乘） |
| 使用场景 | 独立 SQ GEMM | QKV 共享量化后的 3 次 GEMV |
| 前置 kernel | 要求 absmax + quantize | 仅需 workspace 中已有数据 |

**设计意图**：QKV 三个投影共用同一个输入（`rms_out`），量化结果（INT8 数据和 `input_scale`）完全相同。预量化后，Q/K/V 的 GEMV 只需各调用一次此 kernel，每次传入各自不同的 `weight_int8` 和 `weight_scale`。

---

### 3.7 Kernel 5：sq_fused_ffn_gemv_kernel — 融合 FFN GEMV + SwiGLU

**源码位置**：第 299-394 行  
**功能**：在单个 kernel 中完成 FFN 的全部计算——W1 点积 + W3 点积 + SwiGLU 激活

$$\text{output}[n] = \text{SiLU}(\alpha_{W1} \cdot \sum_k x_k \cdot W1_{n,k}) \times (\alpha_{W3} \cdot \sum_k x_k \cdot W3_{n,k})$$

#### 带注释的完整源码

```cuda
__global__ __launch_bounds__(256, 4)
void sq_fused_ffn_gemv_kernel(
    const int8_t* __restrict__ input_int8,   // [K] 量化后的输入
    const int8_t* __restrict__ w1_int8,      // [hidden_dim, K] gate 权重
    const int8_t* __restrict__ w3_int8,      // [hidden_dim, K] up 权重
    half* __restrict__ output_fp16,          // [hidden_dim] 输出
    const float* __restrict__ d_input_scale, // [1] input_scale（设备内存）
    float w1_weight_scale,                   // W1 的 weight_scale（主机常量）
    float w3_weight_scale,                   // W3 的 weight_scale（主机常量）
    int K,                                    // 输入维度 (4096)
    int hidden_dim)                          // FFN 中间维度 (12288)
{
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    // ① 每个 warp 处理一个输出行
    const int row = blockIdx.x * 8 + warp_id;
    if (row >= hidden_dim) return;

    // ② 读取共享的 input_scale，分别乘以 W1/W3 各自的 weight_scale
    const float input_scale = *d_input_scale;
    const float alpha_w1 = input_scale * w1_weight_scale;
    const float alpha_w3 = input_scale * w3_weight_scale;

    // ③ 定位 W1 和 W3 对应行
    const int8_t* w1_row = w1_int8 + static_cast<int64_t>(row) * K;
    const int8_t* w3_row = w3_int8 + static_cast<int64_t>(row) * K;

    // ④ 两个独立的 INT32 累加器
    int32_t acc_gate = 0;  // W1 · x 的累加
    int32_t acc_up = 0;    // W3 · x 的累加

    const int num_vec16 = K / 16;
    const int4* input_i16 = reinterpret_cast<const int4*>(input_int8);
    const int4* w1_i16 = reinterpret_cast<const int4*>(w1_row);
    const int4* w3_i16 = reinterpret_cast<const int4*>(w3_row);

    // ⑤ 主循环：同时计算 W1·x 和 W3·x
    //    每次迭代加载 3 个 int4（input + w1 + w3），共 48 字节
    //    执行 8 次 dp4a（gate 4 次 + up 4 次）
    #pragma unroll 4
    for (int i = lane_id; i < num_vec16; i += 32) {
        int4 x = __ldg(input_i16 + i);   // 输入（W1 和 W3 共享）
        int4 g = __ldg(w1_i16 + i);      // gate 权重
        int4 u = __ldg(w3_i16 + i);      // up 权重

        // gate 累加
        acc_gate = __dp4a(x.x, g.x, acc_gate);
        acc_gate = __dp4a(x.y, g.y, acc_gate);
        acc_gate = __dp4a(x.z, g.z, acc_gate);
        acc_gate = __dp4a(x.w, g.w, acc_gate);

        // up 累加（复用同一个 x）
        acc_up = __dp4a(x.x, u.x, acc_up);
        acc_up = __dp4a(x.y, u.y, acc_up);
        acc_up = __dp4a(x.z, u.z, acc_up);
        acc_up = __dp4a(x.w, u.w, acc_up);
    }

    // ⑥ 余数处理
    const int base = num_vec16 * 16;
    for (int i = base + lane_id; i < K; i += 32) {
        int8_t x = input_int8[i];
        acc_gate += static_cast<int32_t>(x) * static_cast<int32_t>(w1_row[i]);
        acc_up   += static_cast<int32_t>(x) * static_cast<int32_t>(w3_row[i]);
    }

    // ⑦ Warp 归约：gate 和 up 同时归约
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        acc_gate += __shfl_down_sync(0xffffffff, acc_gate, offset);
        acc_up   += __shfl_down_sync(0xffffffff, acc_up, offset);
    }

    // ⑧ lane 0：反量化 + SwiGLU 激活 + 写出
    if (lane_id == 0) {
        float gate = alpha_w1 * static_cast<float>(acc_gate);  // 反量化 gate
        float up   = alpha_w3 * static_cast<float>(acc_up);    // 反量化 up
        // SiLU(gate) = gate × σ(gate) = gate / (1 + e^{-gate})
        float gate_activated = gate / (1.0f + __expf(-gate));
        // SwiGLU 输出 = SiLU(gate) × up
        output_fp16[row] = __float2half(gate_activated * up);
    }
}
```

#### 原理详解

**1. 融合的收益分析**

不融合时，FFN 需要：
```
sq_gemm_cu(input, w1, gate_buf)    →  4 kernels (memset+absmax+quantize+GEMV)
sq_gemm_cu(input, w3, up_buf)      →  4 kernels (memset+absmax+quantize+GEMV)
swiglu_kernel(gate_buf, up_buf, out) →  1 kernel
共计: 9 kernels + 2 个中间 buffer [12288] FP16
```

融合后：
```
sq_fused_ffn_cu(input, w1, w3, out) →  4 kernels (memset+absmax+quantize+fused_ffn)
共计: 4 kernels + 0 个中间 buffer
```

**2. 输入复用**

每次循环迭代从全局内存加载 1 份 `input_int8`（16 字节），同时用于 W1 和 W3 的点积。相比分别执行两次 GEMV（各加载一次 input），**input 内存访问量减半**。

**3. SwiGLU 激活的就地计算**

反量化后立即执行 SiLU 和乘法，避免写入 FP16 中间结果再读取。SiLU 使用快速数学函数 `__expf()`（单精度，足够精度）。

---

### 3.8 SQWorkspace — GPU 工作区管理

**源码位置**：第 401-419 行

```cuda
struct SQWorkspace {
    int8_t* input_int8 = nullptr;  // 量化后的 INT8 输入缓冲区
    int*    max_int    = nullptr;  // absmax 累加器（1 个 int）
    float*  alpha      = nullptr;  // alpha 或 input_scale（1 个 float）
    size_t  input_cap  = 0;        // 当前 input_int8 的分配容量

    void ensure(size_t need) {
        // 按需增长策略：当需求超过容量时，分配 2× 空间
        if (need > input_cap) {
            if (input_int8) cudaFree(input_int8);
            input_cap = need * 2;  // 2× 增长，避免频繁 realloc
            cudaMalloc(&input_int8, input_cap);
        }
        // max_int 和 alpha 只分配一次（各 4 字节）
        if (!max_int) {
            cudaMalloc(&max_int, sizeof(int));
            cudaMalloc(&alpha, sizeof(float));
        }
    }
};

// 全局唯一实例
static SQWorkspace g_workspace;
```

**设计要点**：
- **单例模式**：整个推理过程共享一个 workspace，避免每次 GEMM 都分配/释放显存
- **单调增长**：`input_cap` 只增不减，2× 策略减少 `cudaMalloc` 调用次数
- **CUDA Graph 安全**：buffer 地址在图录制后不变（单调策略保证不会 realloc）
- **极小额外显存**：`max_int`（4 字节）+ `alpha`（4 字节）+ `input_int8`（最大等于 $M \times K$ 字节）

---

### 3.9 调度函数 1：sq_gemv_m1 — Decode 路径调度

**源码位置**：第 428-456 行  
**功能**：编排 M=1 的 3-kernel 流水线

```cuda
static void sq_gemv_m1(
    const half* input_fp16,    // [K] FP16 输入
    const int8_t* qweight,     // [N, K] INT8 权重
    half* output_fp16,         // [N] FP16 输出
    float weight_scale,        // 权重缩放因子
    int K, int N,
    cudaStream_t stream)
{
    // ① 确保 workspace 有 K 字节容量
    g_workspace.ensure(static_cast<size_t>(K));

    constexpr int kThreads = 256;
    int quant_blocks = (K + kThreads * 4 - 1) / (kThreads * 4);

    // ② Phase 0: 重置 absmax 累加器为 0
    //    cudaMemsetAsync 是异步的，在 stream 中排队
    cudaMemsetAsync(g_workspace.max_int, 0, sizeof(int), stream);

    // ③ Phase 1: AbsMax 归约
    //    Grid = quant_blocks, Block = 256, SharedMem = 256×4 = 1024 bytes
    sq_absmax_kernel<<<quant_blocks, kThreads, kThreads * sizeof(float), stream>>>(
        input_fp16, g_workspace.max_int, K);

    // ④ Phase 2: 量化 + Alpha（同 stream 内顺序执行，保证读到完整 absmax）
    sq_quantize_and_alpha_kernel<<<quant_blocks, kThreads, 0, stream>>>(
        input_fp16, g_workspace.input_int8, g_workspace.max_int,
        weight_scale, g_workspace.alpha, K);

    // ⑤ Phase 3: INT8 GEMV
    //    Grid = ceil(N/8), Block = 256 (8 warps, 每 warp 1 个输出)
    int gemv_blocks = (N + 7) / 8;
    sq_gemv_int8_kernel<<<gemv_blocks, 256, 0, stream>>>(
        g_workspace.input_int8, qweight, output_fp16,
        g_workspace.alpha, K, N);
}
```

**流水线时序**（同一 stream 内串行）：

```
Stream: ──[memset]──[absmax_kernel]──[quantize_kernel]──[gemv_kernel]──
                  ↑                 ↑                   ↑
              隐式 barrier       隐式 barrier        隐式 barrier
```

---

### 3.10 调度函数 2：sq_gemm_cutlass — Prefill 路径调度

**源码位置**：第 461-530 行  
**功能**：编排 M>1 的 CUTLASS Tensor Core GEMM 流水线

```cuda
static void sq_gemm_cutlass(
    const half* input_fp16, const int8_t* qweight,
    half* output_fp16, float weight_scale,
    int M, int K, int N, cudaStream_t stream)
{
    const int input_elements = M * K;
    g_workspace.ensure(static_cast<size_t>(input_elements));

    // Phase 0-2: 与 sq_gemv_m1 相同（memset + absmax + quantize）
    // 但处理 M×K 个元素而非 K 个
    cudaMemsetAsync(g_workspace.max_int, 0, sizeof(int), stream);
    sq_absmax_kernel<<<blocks, kThreads, ...>>>(input_fp16, g_workspace.max_int, input_elements);
    sq_quantize_and_alpha_kernel<<<blocks, kThreads, ...>>>(
        input_fp16, g_workspace.input_int8, g_workspace.max_int,
        weight_scale, g_workspace.alpha, input_elements);

    // Phase 3: CUTLASS GEMM
    // 构建 TensorRef（告诉 CUTLASS 数据在哪、什么布局）
    cutlass::TensorRef<int8_t, cutlass::layout::RowMajor> input_ref(...);
    cutlass::TensorRef<int8_t, cutlass::layout::ColumnMajor> weight_ref(...);
    cutlass::TensorRef<cutlass::half_t, cutlass::layout::RowMajor> output_ref(...);

    // 自适应 tile 选择
    if (M <= 32) {
        // 小 tile (128×128×64) — SM 利用率更高
        CutlassInt8GemmSmall gemm_op;
        // epilogue_params 的 alpha 指向设备内存 g_workspace.alpha
        typename CutlassInt8GemmSmall::EpilogueOutputOp::Params epilogue_params(
            g_workspace.alpha, nullptr);  // nullptr = beta（不累加旧值）
        gemm_op.initialize(arguments, nullptr, stream);
        gemm_op(stream);  // 启动 CUTLASS kernel
    } else {
        // 大 tile (256×128×64) — 最大化吞吐
        CutlassInt8Gemm gemm_op;
        ...
        gemm_op(stream);
    }
}
```

**CUTLASS Epilogue 的反量化**：

`LinearCombination<half_t, 8, int32_t, float>` 在 CUTLASS kernel 内部自动执行：

$$D_{fp16}[m,n] = \alpha \times C_{int32}[m,n]$$

- `alpha` 是设备端指针（`g_workspace.alpha`），CUTLASS 内部解引用读取
- `8` 表示 epilogue 向量化宽度，一次写回 8 个 FP16
- 反量化完全融合在 GEMM kernel 中，无额外 kernel 开销

---

### 3.11 公共入口 1：sq_gemm_cu — 主入口

**源码位置**：第 535-555 行

```cuda
void sq_gemm_cu(
    const half* input_fp16, const int8_t* qweight,
    half* output_fp16, float weight_scale,
    int batch_size, int in_features, int out_features,
    cudaStream_t stream)
{
    const int M = batch_size;
    const int K = in_features;
    const int N = out_features;

    if (M == 1) {
        // Decode: 带宽优化的 INT8 GEMV（dp4a + 128-bit loads）
        sq_gemv_m1(input_fp16, qweight, output_fp16, weight_scale, K, N, stream);
    } else {
        // Prefill: CUTLASS INT8 Tensor Core GEMM（MMA 16×8×32）
        sq_gemm_cutlass(input_fp16, qweight, output_fp16, weight_scale, M, K, N, stream);
    }
}
```

**分发逻辑**：M=1 时 GEMV（带宽受限→用 dp4a 手写 kernel），M>1 时 GEMM（计算受限→用 Tensor Core CUTLASS）。

---

### 3.12 公共入口 2：sq_fused_ffn_cu — 融合 FFN 入口

**源码位置**：第 563-603 行

```cuda
void sq_fused_ffn_cu(
    const half* input_fp16,
    const int8_t* w1_int8, const int8_t* w3_int8,
    half* output_fp16,
    float w1_weight_scale, float w3_weight_scale,
    int in_features, int hidden_dim,
    cudaStream_t stream)
{
    const int K = in_features;
    g_workspace.ensure(static_cast<size_t>(K));

    constexpr int kThreads = 256;
    int quant_blocks = (K + kThreads * 4 - 1) / (kThreads * 4);

    // Phase 0: 重置 absmax
    cudaMemsetAsync(g_workspace.max_int, 0, sizeof(int), stream);

    // Phase 1: AbsMax 归约
    sq_absmax_kernel<<<quant_blocks, kThreads, kThreads * sizeof(float), stream>>>(
        input_fp16, g_workspace.max_int, K);

    // Phase 2: 量化 + input_scale
    // 注意 weight_scale=1.0 → alpha = input_scale × 1.0 = input_scale
    // 因为 W1 和 W3 有各自不同的 weight_scale，不能在此处合并
    sq_quantize_and_alpha_kernel<<<quant_blocks, kThreads, 0, stream>>>(
        input_fp16, g_workspace.input_int8, g_workspace.max_int,
        1.0f, g_workspace.alpha, K);

    // Phase 3: 融合 GEMV + SwiGLU
    int ffn_blocks = (hidden_dim + 7) / 8;
    sq_fused_ffn_gemv_kernel<<<ffn_blocks, 256, 0, stream>>>(
        g_workspace.input_int8, w1_int8, w3_int8, output_fp16,
        g_workspace.alpha,       // d_input_scale
        w1_weight_scale,         // W1 专属 weight_scale
        w3_weight_scale,         // W3 专属 weight_scale
        K, hidden_dim);
}
```

**为什么 `weight_scale=1.0`？**

因为 `sq_quantize_and_alpha_kernel` 计算 `alpha = input_scale × weight_scale`。传入 1.0 使得 `alpha = input_scale`。后续 `sq_fused_ffn_gemv_kernel` 内部再分别乘以 `w1_weight_scale` 和 `w3_weight_scale`（因为两者不同，无法在量化阶段合并）。

---

### 3.13 公共入口 3：sq_quantize_input_cu — 共享量化入口

**源码位置**：第 618-641 行

```cuda
void sq_quantize_input_cu(const half* input_fp16, int K, cudaStream_t stream)
{
    g_workspace.ensure(static_cast<size_t>(K));
    constexpr int kThreads = 256;
    int blocks = (K + kThreads * 4 - 1) / (kThreads * 4);

    // 标准 2-kernel 量化流水线
    cudaMemsetAsync(g_workspace.max_int, 0, sizeof(int), stream);

    sq_absmax_kernel<<<blocks, kThreads, kThreads * sizeof(float), stream>>>(
        input_fp16, g_workspace.max_int, K);

    // weight_scale=1.0 → alpha = input_scale
    sq_quantize_and_alpha_kernel<<<blocks, kThreads, 0, stream>>>(
        input_fp16, g_workspace.input_int8, g_workspace.max_int,
        1.0f, g_workspace.alpha, K);
    // 执行后 workspace 中保存：
    //   g_workspace.input_int8 = INT8 量化输入
    //   g_workspace.alpha      = input_scale = absmax/127
}
```

**调用方式**（在 `qwen3_sq.cpp` 中）：

```cpp
// 量化一次
op::SQMatmulLayer::quantize_input(rms_out, stream);
// 复用三次
op::SQMatmulLayer::forward_preq(query_out, *query_sq, stream);  // Q
op::SQMatmulLayer::forward_preq(key_out,   *key_sq,   stream);  // K
op::SQMatmulLayer::forward_preq(value_out, *value_sq, stream);  // V
```

---

### 3.14 公共入口 4：sq_gemv_preq_cu — 预量化 GEMV 入口

**源码位置**：第 644-656 行

```cuda
void sq_gemv_preq_cu(
    const int8_t* qweight, half* output_fp16,
    float weight_scale, int K, int N,
    cudaStream_t stream)
{
    int blocks = (N + 7) / 8;
    // 直接调用预量化 GEMV kernel
    // input 和 input_scale 均从 g_workspace 读取
    sq_gemv_preq_kernel<<<blocks, 256, 0, stream>>>(
        g_workspace.input_int8,  // 来自 sq_quantize_input_cu()
        qweight,
        output_fp16,
        g_workspace.alpha,       // input_scale
        weight_scale,            // 当前层的 weight_scale
        K, N);
}
```

此函数**不执行任何量化操作**，直接使用 workspace 中已有的量化数据。所以调用前必须确保已调用 `sq_quantize_input_cu()`。

---

### 3.15 Kernel 参数与启动配置速查表

| Kernel | Grid | Block | Shared Mem | 每线程处理量 | 关键指令 |
|--------|------|-------|------------|-------------|---------|
| `sq_absmax_kernel` | $\lceil\frac{K}{256 \times 4}\rceil$ | 256 | 1024 B | 4 元素 | `atomicMax`, `__ldg(half2*)` |
| `sq_quantize_and_alpha_kernel` | $\lceil\frac{K}{256 \times 4}\rceil$ | 256 | 0 | 4 元素 | `__float2int_rn`, 位打包 |
| `sq_gemv_int8_kernel` | $\lceil\frac{N}{8}\rceil$ | 256 (8 warps) | 0 | 1 输出/warp | `__dp4a`, `__ldg(int4*)`, `__shfl_down_sync` |
| `sq_gemv_preq_kernel` | $\lceil\frac{N}{8}\rceil$ | 256 (8 warps) | 0 | 1 输出/warp | `__dp4a`, `__ldg(int4*)`, `__shfl_down_sync` |
| `sq_fused_ffn_gemv_kernel` | $\lceil\frac{H}{8}\rceil$ | 256 (8 warps) | 0 | 1 输出/warp | `__dp4a` ×8, `__expf`, SwiGLU |
| CUTLASS GEMM (大 tile) | 自动 | 自动 | 自动 | MMA 16×8×32 | Tensor Core INT8 MMA |
| CUTLASS GEMM (小 tile) | 自动 | 自动 | 自动 | MMA 16×8×32 | Tensor Core INT8 MMA |

**Qwen3-8B 典型参数下的具体配置**：

| 调用场景 | K | N | Kernel Grid | 总 Kernel 数 |
|----------|---|---|-------------|-------------|
| Q 投影 (独立) | 4096 | 4096 | absmax: 4, quant: 4, gemv: 512 | 4 |
| K 投影 (独立) | 4096 | 1024 | absmax: 4, quant: 4, gemv: 128 | 4 |
| QKV 共享量化 | 4096 | - | absmax: 4, quant: 4 | 3 (memset+abs+quant) |
| Q preq GEMV | 4096 | 4096 | gemv: 512 | 1 |
| K preq GEMV | 4096 | 1024 | gemv: 128 | 1 |
| V preq GEMV | 4096 | 1024 | gemv: 128 | 1 |
| FFN 融合 | 4096 | 12288 | absmax: 4, quant: 4, fused: 1536 | 4 |
| W2 独立 | 12288 | 4096 | absmax: 12, quant: 12, gemv: 512 | 4 |

---

## 第四部分：Decode 阶段性能优化深度分析（10.6 → 17.66 tokens/s）

本部分详细分析将 Qwen3-8B SmoothQuant INT8 模型 decode 阶段推理速度从 **10.6 tokens/s 提升到 17.66 tokens/s**（提升 **66.6%**）的四项关键优化。这些优化覆盖了**正确性修复**、**计算指令优化**、**内存访问优化**和**系统级 Kernel Launch 削减**四个层面，展现了从底层 ISA 到上层算法的全栈优化思路。

### 4.1 优化全景概览

| 优化项 | 分类 | 核心改动 | 影响维度 |
|--------|------|----------|----------|
| A. 修复竞态条件 Bug | 正确性修复 | fused 1-kernel → 分离 2-kernel | 消除量化结果不确定性 |
| B. `__dp4a` 硬件指令 | 计算优化 | 7 条指令 → 1 条指令 | ALU 吞吐量 ~3x |
| C. 128-bit 向量化加载 | 内存优化 | 32-bit → 128-bit 加载 | 内存带宽利用率 4x |
| D. QKV 共享量化 | 系统优化 | 每层 12 → 6 个 Kernel | Launch 开销减少 216/步 |

Decode 阶段（M=1）是典型的 **memory-bound** 场景：每个 token 的推理只涉及矩阵-向量乘法（GEMV），计算量小但权重访问量大。因此优化策略的核心是：

1. **确保结果正确**（竞态条件修复是一切的基础）
2. **最大化单次内存访问的计算回报**（`__dp4a` + 128-bit 加载）
3. **最小化 GPU 空闲时间**（减少 Kernel Launch 开销）

```
优化前 Decode Pipeline (每层):
┌────────────────────────────────────────────────────────────────┐
│ Q投影: memset→absmax→quantize→GEMV (4 kernels, 有竞态bug)      │
│ K投影: memset→absmax→quantize→GEMV (4 kernels, 有竞态bug)      │
│ V投影: memset→absmax→quantize→GEMV (4 kernels, 有竞态bug)      │
│ Attention + RoPE + ...                                         │
│ O投影: memset→absmax→quantize→GEMV (4 kernels, 有竞态bug)      │
│ W1: memset→absmax→quantize→GEMV  (4 kernels)                  │
│ W3: memset→absmax→quantize→GEMV  (4 kernels)                  │
│ SwiGLU                            (1 kernel)                   │
│ W2: memset→absmax→quantize→GEMV  (4 kernels)                  │
├────────────────────────────────────────────────────────────────┤
│ 总计: 29 kernels/层 × 36层 = 1044 kernels/步                   │
│ GEMV: 标量加载 + 手动拆包乘法 (低效)                             │
└────────────────────────────────────────────────────────────────┘

优化后 Decode Pipeline (每层):
┌────────────────────────────────────────────────────────────────┐
│ QKV共享: memset→absmax→quantize  (3 kernels, 正确2-kernel方案) │
│ Q GEMV: preq_gemv               (1 kernel, dp4a+128bit)       │
│ K GEMV: preq_gemv               (1 kernel, dp4a+128bit)       │
│ V GEMV: preq_gemv               (1 kernel, dp4a+128bit)       │
│ Attention + RoPE + ...                                         │
│ O投影: memset→absmax→quantize→GEMV (4 kernels, dp4a+128bit)   │
│ FFN融合: memset→absmax→quantize→fused_ffn (4 kernels)         │
│ W2投影: memset→absmax→quantize→GEMV (4 kernels, dp4a+128bit)  │
├────────────────────────────────────────────────────────────────┤
│ 总计: 18 kernels/层 × 36层 = 648 kernels/步                    │
│ GEMV: int4向量化加载 + __dp4a硬件指令 (高效)                     │
└────────────────────────────────────────────────────────────────┘
```

---

### 4.2 优化 A：修复竞态条件 Bug（正确性修复）

这是所有优化中最关键的一项——**不是性能优化，而是正确性修复**。旧的实现存在致命的 inter-block 同步 bug，导致量化结果不确定，进而产生错误的推理输出。

#### 4.2.1 旧方案的致命缺陷：fused absmax+quantize 的竞态条件

旧实现试图将 absmax 归约和量化融合到一个 kernel 中，使用 `atomicAdd` 做 grid-level 计数器来实现 inter-block 同步：

```cuda
// ❌ 旧方案（有竞态条件 bug）
__global__ void sq_fused_absmax_quantize_kernel(
    const half* input_fp16,
    int8_t* output_int8,
    int* d_max_as_int,
    float weight_scale,
    float* d_alpha,
    int total_elements)
{
    // Step 1: 每个 block 做局部 absmax 归约
    extern __shared__ float sdata[];
    float local_max = ...;  // 读取输入，求局部最大值
    // shared memory tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { ... }
    
    // Step 2: 原子操作更新全局 absmax
    if (threadIdx.x == 0) {
        atomicMax(d_max_as_int, __float_as_int(sdata[0]));
    }
    
    // ⚠️ 致命问题：试图用 atomicAdd 做 grid barrier
    __shared__ bool is_last;
    if (threadIdx.x == 0) {
        // 计数器：每个块完成 absmax 后加 1
        int old = atomicAdd(&g_block_counter, 1);  
        is_last = (old == gridDim.x - 1);  // 最后一个块？
    }
    __syncthreads();
    
    if (!is_last) {
        // ❌ 非最后的 block 在 absmax 未全部完成时就读取并量化
        // 此时 d_max_as_int 可能还不是最终值！
        float absmax = __int_as_float(*d_max_as_int);  // 读到的是中间值
        // ... 使用错误的 absmax 进行量化 ...
    }
    
    // 只有 "最后一个" block 才能看到正确的 absmax
    // 但前面的 block 已经用错误的值量化了！
}
```

**竞态条件的根本原因**：

```
时间线：
  Block 0: absmax=3.5 → atomicMax → atomicAdd(counter)=0 → 不是最后 → 读absmax=3.5 → 量化 ❌
  Block 1: absmax=7.2 → atomicMax → atomicAdd(counter)=1 → 不是最后 → 读absmax=7.2 → 量化 ❌
  Block 2: absmax=5.1 → atomicMax → atomicAdd(counter)=2 → 不是最后 → 读absmax=7.2 → 量化 ❌
  Block 3: absmax=9.8 → atomicMax → ...（还没执行完）
  
  ↑ Block 0-2 读到的 absmax 不同！应该都是 9.8，但：
  - Block 0 在 Block 3 之前运行，读到的值可能是 3.5（只有自己的）
  - Block 1 可能读到 7.2（Block 0 和 1 的最大值）
  - 每个 block 用不同的 scale 量化 → 量化结果不一致 → 推理输出错误
```

**核心问题**：`atomicAdd` 计数器只能告诉"最后一个"block 所有 block 都完成了 absmax，但无法让先完成的 block 等待。CUDA **不支持 inter-block barrier**（不同 block 可能不在 SM 上同时运行），因此在单个 kernel 内部无法正确实现全局同步。

#### 4.2.2 新方案：2-Kernel 分离架构

正确的方案是将 absmax 归约和量化拆分为两个独立的 kernel，利用 **kernel launch 之间的隐式同步**（同一 stream 上的 kernel 按序执行）来保证数据一致性：

```cuda
// ✅ 新方案：Kernel 1 — 只做 absmax 归约
__global__ void sq_absmax_kernel(
    const half* __restrict__ input,
    int* __restrict__ d_max_as_int,    // 输出：全局 absmax (as int for atomicMax)
    int total_elements)
{
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    const int gid = (blockIdx.x * blockDim.x + tid) * 4;

    // 每个线程处理 4 个元素，用 half2 向量化读取
    float local_max = 0.0f;
    if (gid + 3 < total_elements) {
        const half2* h2 = reinterpret_cast<const half2*>(input + gid);
        half2 v0 = __ldg(h2);
        half2 v1 = __ldg(h2 + 1);
        float2 f0 = __half22float2(v0);
        float2 f1 = __half22float2(v1);
        local_max = fmaxf(fmaxf(fabsf(f0.x), fabsf(f0.y)),
                          fmaxf(fabsf(f1.x), fabsf(f1.y)));
    } else {
        for (int i = gid; i < total_elements && i < gid + 4; ++i)
            local_max = fmaxf(local_max, fabsf(__half2float(input[i])));
    }

    // Block 内 shared memory 树归约
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    // Block 间用 atomicMax 汇总（安全：只写不读）
    if (tid == 0) {
        atomicMax(d_max_as_int, __float_as_int(sdata[0]));
    }
    // ✅ 这里不读 d_max_as_int，不做量化，kernel 结束
}
```

```cuda
// ✅ 新方案：Kernel 2 — 读取最终 absmax，量化 + 计算 alpha
__global__ void sq_quantize_and_alpha_kernel(
    const half* __restrict__ input_fp16,
    int8_t* __restrict__ output_int8,
    const int* __restrict__ d_max_as_int,  // 来自 Kernel 1 的最终结果
    float weight_scale,
    float* __restrict__ d_alpha,
    int total_elements)
{
    // ✅ 此时 d_max_as_int 已经是所有 block 归约后的最终值
    // （因为 Kernel 1 已经完全执行完毕）
    const float absmax = __int_as_float(*d_max_as_int);
    const float inv_scale = (absmax > 1e-6f) ? 127.0f / absmax : 0.0f;

    // Block 0, Thread 0 计算并存储 alpha
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const float input_scale = (absmax > 1e-6f) ? absmax / 127.0f : 0.0f;
        *d_alpha = input_scale * weight_scale;
    }

    // 每个线程将 4 个 FP16 量化为 INT8 并打包为 int32
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < total_elements) {
        const half2* h2 = reinterpret_cast<const half2*>(input_fp16 + idx);
        half2 v0 = __ldg(h2);
        half2 v1 = __ldg(h2 + 1);
        float2 f0 = __half22float2(v0);
        float2 f1 = __half22float2(v1);

        int i0 = max(-128, min(127, __float2int_rn(f0.x * inv_scale)));
        int i1 = max(-128, min(127, __float2int_rn(f0.y * inv_scale)));
        int i2 = max(-128, min(127, __float2int_rn(f1.x * inv_scale)));
        int i3 = max(-128, min(127, __float2int_rn(f1.y * inv_scale)));

        // 4 个 INT8 打包为 1 个 int32（合并写入，高效）
        int32_t packed = (i0 & 0xFF) | ((i1 & 0xFF) << 8) |
                         ((i2 & 0xFF) << 16) | ((i3 & 0xFF) << 24);
        *reinterpret_cast<int32_t*>(output_int8 + idx) = packed;
    }
}
```

**调度代码**（来源：[sq_gemm_kernel.cu](kuiper/source/op/kernels/cuda/sq_gemm_kernel.cu#L434-L460)）：

```cpp
static void sq_gemv_m1(..., cudaStream_t stream) {
    // 1. 重置 absmax 累加器
    cudaMemsetAsync(g_workspace.max_int, 0, sizeof(int), stream);
    
    // 2. Kernel 1: absmax 归约（所有 block 的 atomicMax）
    sq_absmax_kernel<<<quant_blocks, 256, 256*sizeof(float), stream>>>(
        input_fp16, g_workspace.max_int, K);
    
    // ✅ 隐式同步点：同一 stream 上 Kernel 1 完成后才启动 Kernel 2
    
    // 3. Kernel 2: 量化 + alpha 计算（读取最终 absmax）
    sq_quantize_and_alpha_kernel<<<quant_blocks, 256, 0, stream>>>(
        input_fp16, g_workspace.input_int8, g_workspace.max_int,
        weight_scale, g_workspace.alpha, K);
    
    // 4. Kernel 3: INT8 GEMV
    sq_gemv_int8_kernel<<<gemv_blocks, 256, 0, stream>>>(...);
}
```

#### 4.2.3 正确性保证的数学证明

**定理**：2-kernel 方案中，所有线程在 `sq_quantize_and_alpha_kernel` 中读到的 absmax 值是相同的且正确的。

**证明**：

1. `sq_absmax_kernel` 的所有 block 都只做 `atomicMax` 写入 `d_max_as_int`，不读取最终结果
2. CUDA stream 保证：同一 stream 上，Kernel 1 的**所有 block 的所有线程**执行完毕后，Kernel 2 才会被调度
3. 因此 `sq_quantize_and_alpha_kernel` 启动时，`d_max_as_int` 已经包含了全局最大值
4. 所有线程读到的 `absmax` 值相同 → 使用相同的 `inv_scale` → 量化结果一致 ∎

**与 CUDA Graph 的兼容性**：2-kernel 方案完全兼容 CUDA Graph：
- 两个 kernel launch 的依赖关系被 Graph 捕获
- `cudaMemsetAsync` 也被捕获为 Graph Node
- `g_workspace.alpha` 是 device-side 指针，不需要 host-device 同步

#### 4.2.4 性能影响分析

| 维度 | 旧方案（fused） | 新方案（2-kernel） |
|------|----------------|-------------------|
| 正确性 | ❌ 竞态条件，结果不确定 | ✅ 保证正确 |
| Kernel 数量 | 1 (但结果错误) | 2 |
| 额外开销 | — | ~2μs Kernel Launch 开销 |
| CUDA Graph | ❌ 全局计数器不可复用 | ✅ 完全兼容 |

> **启示**：正确性是性能的前提。fused kernel 看似减少了 launch 开销，但由于 CUDA 缺乏 inter-block barrier 机制，在需要全局同步的场景下必须拆分为多个 kernel。

---

### 4.3 优化 B：`__dp4a` 硬件指令（计算效率 ~3x）

`__dp4a`（Dot Product of 4 Accumulate）是 NVIDIA GPU 从 Pascal (SM61) 开始提供的硬件内联函数，能在单条指令内完成 4 个 INT8 乘累加操作。

#### 4.3.1 旧方案：手动 INT8 拆包与标量乘法

旧实现从 `int32_t` 中手动按字节提取 4 个 INT8 值，再逐一做乘法和加法：

```cuda
// ❌ 旧方案：手动拆包 + 标量乘法
for (int i = lane_id; i < K / 4; i += 32) {
    int32_t x_packed = *reinterpret_cast<const int32_t*>(input_int8 + i * 4);
    int32_t w_packed = *reinterpret_cast<const int32_t*>(w_row + i * 4);
    
    // 手动拆包 4 个 INT8
    int8_t x0 = (x_packed >>  0) & 0xFF;  // 指令 1: 位移
    int8_t x1 = (x_packed >>  8) & 0xFF;  // 指令 2: 位移
    int8_t x2 = (x_packed >> 16) & 0xFF;  // 指令 3: 位移
    int8_t x3 = (x_packed >> 24) & 0xFF;  // 指令 4: 位移（隐含在符号扩展中）
    int8_t w0 = (w_packed >>  0) & 0xFF;
    int8_t w1 = (w_packed >>  8) & 0xFF;
    int8_t w2 = (w_packed >> 16) & 0xFF;
    int8_t w3 = (w_packed >> 24) & 0xFF;
    
    // 4 次乘法 + 3 次加法 = 7 条 ALU 指令
    acc += (int)x0 * (int)w0;   // 乘法 1
    acc += (int)x1 * (int)w1;   // 乘法 2 + 加法 1
    acc += (int)x2 * (int)w2;   // 乘法 3 + 加法 2
    acc += (int)x3 * (int)w3;   // 乘法 4 + 加法 3
}
// 每 4 个元素: ~8 条位移/掩码 + 4 条 IMAD + 3 条 IADD ≈ 15 条指令
```

#### 4.3.2 新方案：`__dp4a` 单指令 4-MAC

```cuda
// ✅ 新方案：__dp4a 硬件指令
for (int i = lane_id; i < num_vec16; i += 32) {
    int4 x = __ldg(input_i16 + i);    // 128-bit load: 16 个 INT8
    int4 w = __ldg(weight_i16 + i);   // 128-bit load: 16 个 INT8
    
    // 4 条 __dp4a = 16 个 INT8 乘累加
    acc = __dp4a(x.x, w.x, acc);  // x.x 和 w.x 各含 4 个 INT8，1 条指令
    acc = __dp4a(x.y, w.y, acc);  // 1 条指令
    acc = __dp4a(x.z, w.z, acc);  // 1 条指令
    acc = __dp4a(x.w, w.w, acc);  // 1 条指令
}
// 每 16 个元素: 2 条 LDG.128 + 4 条 DP4A = 6 条指令
// 对比旧方案每 16 个元素: 4 条 LDG.32 + 60 条 ALU ≈ 64 条指令
```

#### 4.3.3 指令级性能对比

`__dp4a(int a, int b, int c)` 的硬件语义：

$$
\text{result} = c + \sum_{i=0}^{3} \text{a.byte}[i] \times \text{b.byte}[i]
$$

即把两个 32-bit 整数各视为 4 个 signed/unsigned INT8 的打包，做点积后累加到 32-bit 累加器。

| 维度 | 旧方案（手动拆包） | 新方案（`__dp4a`） | 提升倍数 |
|------|-------------------|-------------------|----------|
| 每 4 个元素的指令数 | ~15 条（移位+掩码+乘+加） | 1 条 | **15x** |
| 每 16 个元素的 ALU 指令 | ~60 条 | 4 条 | **15x** |
| 每 16 个元素的总指令（含 Load） | ~64 条 | 6 条 | **~10x** |
| INT8 乘累加吞吐量（单 SM/clock） | 受限于 IMAD 流水线 | 专用 INT8 单元 | **~3x 实测** |

> 注：理论指令数减少 10-15x，但实测加速约 3x，因为 decode 阶段 GEMV 是 **memory-bound**，计算指令的减少主要释放了流水线资源，减少了指令调度压力。

**Orin (SM87) 上的 `__dp4a` 规格**：
- 每个 SM 每 clock 可执行多条 `__dp4a`（INT8 DP 单元）
- 延迟：~4 cycles
- 吞吐量：远高于等效的标量 IMAD 序列

#### 4.3.4 源码中的 `__dp4a` 应用

项目中有 **三个 kernel** 使用 `__dp4a`：

**1. `sq_gemv_int8_kernel`**（[sq_gemm_kernel.cu](kuiper/source/op/kernels/cuda/sq_gemm_kernel.cu#L174-L232)）— 标准 GEMV：
```cuda
acc = __dp4a(x.x, w.x, acc);  // x,w 来自 int4 加载
acc = __dp4a(x.y, w.y, acc);
acc = __dp4a(x.z, w.z, acc);
acc = __dp4a(x.w, w.w, acc);  // 4 条指令处理 16 个元素
```

**2. `sq_gemv_preq_kernel`**（[sq_gemm_kernel.cu](kuiper/source/op/kernels/cuda/sq_gemm_kernel.cu#L243-L296)）— 预量化 GEMV（共享量化路径）：
```cuda
// 与 sq_gemv_int8_kernel 完全相同的 dp4a 内循环
acc = __dp4a(x.x, w.x, acc);
acc = __dp4a(x.y, w.y, acc);
acc = __dp4a(x.z, w.z, acc);
acc = __dp4a(x.w, w.w, acc);
```

**3. `sq_fused_ffn_gemv_kernel`**（[sq_gemm_kernel.cu](kuiper/source/op/kernels/cuda/sq_gemm_kernel.cu#L316-L393)）— 融合 FFN GEMV：
```cuda
// 每次迭代计算 gate 和 up 两个点积，共 8 条 __dp4a
acc_gate = __dp4a(x.x, g.x, acc_gate);
acc_gate = __dp4a(x.y, g.y, acc_gate);
acc_gate = __dp4a(x.z, g.z, acc_gate);
acc_gate = __dp4a(x.w, g.w, acc_gate);
acc_up = __dp4a(x.x, u.x, acc_up);
acc_up = __dp4a(x.y, u.y, acc_up);
acc_up = __dp4a(x.z, u.z, acc_up);
acc_up = __dp4a(x.w, u.w, acc_up);
```

---

### 4.4 优化 C：128-bit 向量化加载（带宽利用率 4x）

在 memory-bound 的 GEMV 场景下，**内存访问效率是决定性能的第一因素**。128-bit 向量化加载将每次内存事务从 4 个 INT8 提升到 16 个 INT8，极大提高了带宽利用率。

#### 4.4.1 旧方案：32-bit 标量加载

```cuda
// ❌ 旧方案：每次加载 4 个 INT8（32-bit）
for (int i = lane_id; i < K / 4; i += 32) {
    int32_t x = *reinterpret_cast<const int32_t*>(input_int8 + i * 4);  // LDG.32
    int32_t w = *reinterpret_cast<const int32_t*>(w_row + i * 4);      // LDG.32
    // ... 手动拆包 + 标量乘法 ...
}
```

**问题分析**：
- 每条 `LDG.32` 指令只加载 4 字节
- GPU 内存系统的最小事务粒度是 32 字节（L2 cache line = 128 字节）
- 32-bit 加载只利用了事务的 $4/32 = 12.5\%$
- Warp 中 32 个线程的 32-bit 加载如果地址连续，可合并为 4 个 128-byte 事务（$ 32 \times 4 = 128$ 字节），合并效率尚可
- 但每个线程需要更多次循环迭代来处理所有数据

#### 4.4.2 新方案：`int4` 128-bit 向量化加载

```cuda
// ✅ 新方案：每次加载 16 个 INT8（128-bit = int4）
const int num_vec16 = K / 16;                                    // K=4096 → 256 次迭代/warp
const int4* input_i16 = reinterpret_cast<const int4*>(input_int8);
const int4* weight_i16 = reinterpret_cast<const int4*>(w_row);

#pragma unroll 4
for (int i = lane_id; i < num_vec16; i += 32) {
    int4 x = __ldg(input_i16 + i);    // LDG.128: 16 字节 = 16 个 INT8
    int4 w = __ldg(weight_i16 + i);   // LDG.128: 16 字节 = 16 个 INT8
    acc = __dp4a(x.x, w.x, acc);
    acc = __dp4a(x.y, w.y, acc);
    acc = __dp4a(x.z, w.z, acc);
    acc = __dp4a(x.w, w.w, acc);
}
```

**`int4` 数据类型解析**：

```
int4 的内存布局（128-bit = 16 字节）:
┌─────────┬─────────┬─────────┬─────────┐
│  x (32b) │  y (32b) │  z (32b) │  w (32b) │
├──┬──┬──┬──┼──┬──┬──┬──┼──┬──┬──┬──┼──┬──┬──┬──┤
│b0│b1│b2│b3│b4│b5│b6│b7│b8│b9│bA│bB│bC│bD│bE│bF│  ← 16 个 INT8
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘

每个 .x/.y/.z/.w 分量 = 4 个 INT8 打包在 int32 中
 → 正好是 __dp4a 的输入格式，零拆包开销
```

#### 4.4.3 内存事务效率对比

以 Qwen3-8B Q 投影（K=4096, N=4096）为例：

| 维度 | 旧方案（LDG.32） | 新方案（LDG.128） | 提升 |
|------|-----------------|------------------|------|
| 每线程每次加载 | 4 字节 | 16 字节 | 4x |
| 每 warp 每次迭代加载 | 128 字节 | 512 字节 | 4x |
| K=4096 的循环次数/线程 | $4096/4/32 = 32$ | $4096/16/32 = 8$ | 4x 减少 |
| 总 Load 指令数/warp | $32 \times 2 = 64$ | $8 \times 2 = 16$ | 4x 减少 |
| 有效带宽利用率 | 中等 | 接近理论峰值 | — |

**Orin 内存带宽**：
- LPDDR5 理论峰值：~102.4 GB/s（Orin NX）或 ~204.8 GB/s（Orin AGX）
- 128-bit 加载能更好地触发 memory coalescing，减少 L2 cache line 浪费

#### 4.4.4 与 `__dp4a` 的完美配合

`int4` 加载和 `__dp4a` 的组合是精心设计的：

```
int4 x = __ldg(input_i16 + i);
     ↓
┌──────────────────────────────┐
│ x.x = [b0,b1,b2,b3] (4×INT8) │ → __dp4a(x.x, w.x, acc)  → acc += Σ(x.x[j]*w.x[j])
│ x.y = [b4,b5,b6,b7] (4×INT8) │ → __dp4a(x.y, w.y, acc)  → acc += Σ(x.y[j]*w.y[j])
│ x.z = [b8,b9,bA,bB] (4×INT8) │ → __dp4a(x.z, w.z, acc)  → acc += Σ(x.z[j]*w.z[j])
│ x.w = [bC,bD,bE,bF] (4×INT8) │ → __dp4a(x.w, w.w, acc)  → acc += Σ(x.w[j]*w.w[j])
└──────────────────────────────┘

1 次 LDG.128 + 4 次 __dp4a = 16 个 INT8 元素的完整处理
无格式转换开销：int4 的 .x/.y/.z/.w 直接是 __dp4a 需要的 packed INT8×4 格式
```

这种一对一对应关系意味着：
- **零拆包开销**：不需要位移、掩码操作
- **最大化指令级并行**：Load 和 Compute 可以流水线重叠
- **最小化寄存器压力**：`int4` 只占 4 个 32-bit 寄存器

---

### 4.5 优化 D：QKV 共享量化（减少 216 个 Kernel Launch/步）

这是一项**系统级优化**，利用 Transformer 架构中 Q/K/V 投影共享相同输入的特点，消除了大量冗余的量化 kernel launch。

#### 4.5.1 旧方案：独立量化的 Kernel Launch 风暴

在旧方案中，Q、K、V 三个投影各自独立执行完整的 SQ GEMM 流程：

```
旧方案 — QKV 投影（每层）：

  Q 投影 (sq_gemm_cu):
    1. cudaMemsetAsync(max_int, 0)        ← Kernel Launch #1
    2. sq_absmax_kernel(rms_out)           ← Kernel Launch #2  ← 与 K,V 重复！
    3. sq_quantize_and_alpha_kernel(...)   ← Kernel Launch #3  ← 与 K,V 重复！
    4. sq_gemv_int8_kernel(wq)            ← Kernel Launch #4

  K 投影 (sq_gemm_cu):                                         
    5. cudaMemsetAsync(max_int, 0)        ← Kernel Launch #5  ← 完全冗余
    6. sq_absmax_kernel(rms_out)           ← Kernel Launch #6  ← 完全冗余
    7. sq_quantize_and_alpha_kernel(...)   ← Kernel Launch #7  ← 完全冗余
    8. sq_gemv_int8_kernel(wk)            ← Kernel Launch #8

  V 投影 (sq_gemm_cu):
    9.  cudaMemsetAsync(max_int, 0)       ← Kernel Launch #9   ← 完全冗余
    10. sq_absmax_kernel(rms_out)          ← Kernel Launch #10  ← 完全冗余
    11. sq_quantize_and_alpha_kernel(...)  ← Kernel Launch #11  ← 完全冗余
    12. sq_gemv_int8_kernel(wv)           ← Kernel Launch #12

  总计: 12 Kernel Launches
  其中 6 个对 rms_out 的重复量化是完全冗余的
```

**冗余的根本原因**：Q、K、V 三个线性投影的输入都是同一个 `rms_out`（RMSNorm 归一化后的隐状态向量），因此对 `rms_out` 的量化（absmax 归约 + FP16→INT8 转换）只需做一次。

#### 4.5.2 新方案：共享量化 + 预量化 GEMV

```
新方案 — QKV 投影（每层）：

  共享量化 (sq_quantize_input_cu):
    1. cudaMemsetAsync(max_int, 0)        ← Kernel Launch #1
    2. sq_absmax_kernel(rms_out)           ← Kernel Launch #2  ← 只做一次
    3. sq_quantize_and_alpha_kernel(...)   ← Kernel Launch #3  ← 只做一次
                                                                  (weight_scale=1.0)
  Q 预量化 GEMV (sq_gemv_preq_cu):                              
    4. sq_gemv_preq_kernel(wq)            ← Kernel Launch #4

  K 预量化 GEMV (sq_gemv_preq_cu):
    5. sq_gemv_preq_kernel(wk)            ← Kernel Launch #5

  V 预量化 GEMV (sq_gemv_preq_cu):
    6. sq_gemv_preq_kernel(wv)            ← Kernel Launch #6

  总计: 6 Kernel Launches (减少 6 个/层)
```

#### 4.5.3 源码实现详解

**调用层**（来源：[qwen3_sq.cpp](kuiper/source/model/qwen3_sq.cpp#L188-L219)）：

```cpp
void Qwen3SQModel::batched_qkv_projection(
    int32_t layer_idx,
    const tensor::Tensor& rms_out,     // Q, K, V 的共同输入
    const tensor::Tensor& query_out,
    const tensor::Tensor& key_out,
    const tensor::Tensor& value_out,
    int32_t seq_len) const
{
    auto query_sq = std::dynamic_pointer_cast<op::SQMatmulLayer>(...);
    auto key_sq   = std::dynamic_pointer_cast<op::SQMatmulLayer>(...);
    auto value_sq = std::dynamic_pointer_cast<op::SQMatmulLayer>(...);

    int batch_size = rms_out.size() / query_sq->in_features();

    if (batch_size == 1) {  // Decode 路径
        // Step 1: 量化一次 rms_out → workspace.input_int8 + workspace.alpha
        op::SQMatmulLayer::quantize_input(rms_out, stream);  // 3 kernels
        
        // Step 2: 三次预量化 GEMV（各 1 kernel，共用已量化的输入）
        op::SQMatmulLayer::forward_preq(query_out, *query_sq, stream);  // 1 kernel
        op::SQMatmulLayer::forward_preq(key_out, *key_sq, stream);      // 1 kernel
        op::SQMatmulLayer::forward_preq(value_out, *value_sq, stream);  // 1 kernel
        return;
    }
    // Prefill 路径不变：各自独立 SQ GEMM
    ...
}
```

**共享量化 Kernel 层**（来源：[sq_gemm_kernel.cu](kuiper/source/op/kernels/cuda/sq_gemm_kernel.cu#L594-L620)）：

```cpp
void sq_quantize_input_cu(const half* input_fp16, int K, cudaStream_t stream) {
    g_workspace.ensure(static_cast<size_t>(K));
    constexpr int kThreads = 256;
    int blocks = (K + kThreads * 4 - 1) / (kThreads * 4);

    // 1. 重置 absmax
    cudaMemsetAsync(g_workspace.max_int, 0, sizeof(int), stream);
    
    // 2. AbsMax 归约
    sq_absmax_kernel<<<blocks, kThreads, kThreads * sizeof(float), stream>>>(
        input_fp16, g_workspace.max_int, K);
    
    // 3. 量化 + 存储 input_scale
    // 注意 weight_scale=1.0 → alpha = input_scale × 1.0 = input_scale
    sq_quantize_and_alpha_kernel<<<blocks, kThreads, 0, stream>>>(
        input_fp16, g_workspace.input_int8, g_workspace.max_int,
        1.0f,                     // ← weight_scale = 1.0：只存 input_scale
        g_workspace.alpha, K);    // ← alpha 此时 = input_scale = absmax/127
}
```

**预量化 GEMV Kernel 的 alpha 计算**（来源：[sq_gemm_kernel.cu](kuiper/source/op/kernels/cuda/sq_gemm_kernel.cu#L253)）：

```cuda
// sq_gemv_preq_kernel 内部
const float alpha = (*d_input_scale) * weight_scale;
//                   ↑ 来自 workspace     ↑ 每层不同
//                   = absmax/127          = 该层的 weight_scale
```

这里巧妙利用了反量化公式的可分解性：

$$
\text{output} = \underbrace{\frac{\text{absmax}}{127}}_{\text{input\_scale}} \times \underbrace{s_w}_{\text{weight\_scale}} \times \sum_k x_{\text{int8}}[k] \cdot w_{\text{int8}}[k]
$$

`input_scale` 对 Q/K/V 三层相同，`weight_scale` 则各不相同。因此：
- 共享量化阶段：存储 `input_scale = absmax/127`（weight_scale=1.0）
- 每层 GEMV 阶段：`alpha = input_scale * weight_scale`（device-side 乘法）

#### 4.5.4 Kernel Launch 数量化分析

| 投影操作 | 旧方案 Kernel 数 | 新方案 Kernel 数 | 节省 |
|----------|-----------------|-----------------|------|
| Q 投影 | 4 (memset+abs+quant+gemv) | 1 (preq_gemv) | 3 |
| K 投影 | 4 (memset+abs+quant+gemv) | 1 (preq_gemv) | 3 |
| V 投影 | 4 (memset+abs+quant+gemv) | 1 (preq_gemv) | 3 |
| 共享量化 | 0 | 3 (memset+abs+quant) | -3 |
| **QKV 小计** | **12** | **6** | **6** |

**每步（全模型）的节省**：

$$
\text{节省} = 6 \text{ kernels/层} \times 36 \text{ 层} = 216 \text{ kernel launches/步}
$$

**Launch 开销估算**：
- 典型 CUDA Kernel Launch 开销：~3-5 μs（不含 CUDA Graph）
- 216 个 Launch 的额外时间：$216 \times 4\mu s \approx 864 \mu s \approx 0.86$ ms/步
- Decode 步时间约 56.6 ms（17.66 tok/s），节省 $0.86/56.6 \approx 1.5\%$
- 但更重要的是减少了 GPU 空闲间隙，提高了 SM 利用率

> **注意**：在 CUDA Graph 模式下，Kernel Launch 开销被大幅降低（Graph replay 每个 kernel 只有 ~0.5μs 开销），但共享量化仍然节省了冗余计算（重复的 absmax + quantize）的执行时间本身。

---

### 4.6 四项优化的协同效应

四项优化并非独立发挥作用，它们之间存在深层的协同关系：

```
    ┌─────────────────────────────────────────────────────┐
    │          Decode 全栈优化协同图                        │
    ├─────────────────────────────────────────────────────┤
    │                                                      │
    │  优化A: 2-Kernel 分离                                │
    │    ↓ 保证正确性                                      │
    │    ↓ 确保量化结果一致                                 │
    │                                                      │
    │  优化D: 共享量化 ──────────────────────┐              │
    │    ↓ 量化只做一次                       │              │
    │    ↓ 结果存入 workspace                │              │
    │                                        │              │
    │  优化C: 128-bit 加载 ←─── 优化B: __dp4a │              │
    │    ↓ int4 读 16 个 INT8      ↑ 4个MAC/指令           │
    │    ↓                         ↑                        │
    │    └── int4.x/y/z/w ────────→┘                       │
    │         直接喂给 __dp4a                               │
    │         零转换开销                                     │
    │                                                      │
    │  最终效果:                                             │
    │    Pipeline: quant(3k) → preq_gemv(1k) × 3           │
    │    GEMV 内循环: LDG.128 + 4×DP4A                      │
    │    18 kernels/层 vs 旧方案 29 kernels/层              │
    └─────────────────────────────────────────────────────┘
```

**协同效应的具体体现**：

1. **A + D**：正确的 2-kernel 量化方案使得共享量化成为可能——如果 absmax 不正确，共享量化的前提（所有 GEMV 共用同一份正确的 INT8 输入）就不成立

2. **B + C**：`int4` 的 `.x/.y/.z/.w` 分量恰好是 `__dp4a` 的输入格式——这不是巧合，而是刻意选择的 128-bit 加载类型（`int4` 而非 `float4` 或其他），使得 Load 和 Compute 无缝衔接

3. **D + (B+C)**：共享量化减少了量化 kernel 的次数，而优化后的 GEMV kernel（dp4a + int4）使得每个 GEMV kernel 本身执行更快——两者叠加，总执行时间大幅降低

4. **A + CUDA Graph**：2-kernel 方案中 `d_alpha` 是 device-side 指针，`g_workspace` 使用 monotonic growth（只增不减），完全兼容 CUDA Graph capture/replay，避免了每步重新分配显存

---

### 4.7 性能提升总结与分析

| 维度 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| Decode 速度 | 10.6 tok/s | 17.66 tok/s | **+66.6%** |
| 每步延迟 | ~94.3 ms | ~56.6 ms | -40.0% |
| 每层 Kernel 数 | ~29 | ~18 | -37.9% |
| 每步 Kernel 数 | ~1044 | ~648 | -37.9% |
| GEMV 内循环指令/16元素 | ~64 | ~6 | -90.6% |
| 量化正确性 | ❌ 竞态条件 | ✅ 保证正确 | — |
| CUDA Graph 兼容性 | ❌ 部分 | ✅ 完全 | — |

**性能提升的分解估算**：

$$
\text{总提升} = \frac{17.66 - 10.6}{10.6} = 66.6\%
$$

各优化项的贡献（定性估算，实际相互耦合）：

| 优化项 | 估算贡献 | 主要作用机制 |
|--------|---------|-------------|
| A. 竞态修复 | 基础 | 消除错误输出，使其他优化有意义 |
| B. `__dp4a` | ~25-30% | 减少 ALU 指令，释放流水线 |
| C. 128-bit 加载 | ~20-25% | 提高有效带宽，减少 Load 指令 |
| D. 共享量化 | ~10-15% | 减少 Launch 开销和冗余计算 |

> 注：由于 GEMV 是 memory-bound，B 和 C 的效果高度耦合——`__dp4a` 减少计算指令使得 GPU 更快地达到同步点来消费 Load 结果，128-bit 加载提供了更高效的数据供给。两者同时作用时效果大于各自单独的贡献之和。

**Qwen3-8B 每步 Kernel Launch 详细对比**：

| 操作 | 旧方案 (Kernels/层) | 新方案 (Kernels/层) | 变化 |
|------|---------------------|---------------------|------|
| QKV 投影 | 12 (3×4) | 6 (3+3×1) | -6 |
| O 投影 | 4 | 4 | 0 |
| FFN (W1+W3+SwiGLU) | 9 (2×4+1) | 4 (fused) | -5 |
| W2 投影 | 4 | 4 | 0 |
| **层合计** | **29** | **18** | **-11** |
| **全模型 (×36)** | **1044** | **648** | **-396** |

---

## 附录：源码文件索引

| 文件 | 行数 | 主要内容 |
|------|------|----------|
| `tools/export_qwen3-8B-sq.py` | 402 | Python 导出脚本 |
| `kuiper/include/model/qwen3_sq.h` | 55 | SQ 模型类声明 |
| `kuiper/source/model/qwen3_sq.cpp` | 287 | SQ 模型实现 |
| `kuiper/include/op/sq_matmul.h` | 103 | SQ MatMul 层声明 |
| `kuiper/source/op/sq_matmul.cpp` | 207 | SQ MatMul 层实现 |
| `kuiper/source/op/kernels/cuda/sq_gemm_kernel.cuh` | 68 | CUDA Kernel 接口 |
| `kuiper/source/op/kernels/cuda/sq_gemm_kernel.cu` | 659 | CUDA Kernel 实现 |
