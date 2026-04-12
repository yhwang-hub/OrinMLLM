# OrinMLLM Qwen3-4B FP8 推理框架适配技术报告

> **项目**: OrinMLLM C++ 推理框架  
> **模型**: Qwen3-4B-FP8 (E4M3 Block-Quantized)  
> **硬件**: NVIDIA RTX 5070 Laptop GPU (SM 12.0, Blackwell架构, 8GB GDDR7)  
> **日期**: 2025年7月  

---

## 目录

1. [推理流程分析](#1-推理流程分析)
2. [FP8算子开发方案](#2-fp8算子开发方案)
3. [遇到的困难与解决方案](#3-遇到的困难与解决方案)
4. [性能优化技术](#4-性能优化技术)

---

## 1. 推理流程分析

### 1.1 整体推理流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Qwen3-4B FP8 推理流程                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────────────┐  │
│  │ Token ID  │───▶│  Embedding   │───▶│     x ∈ R^{2560} (FP16)     │  │
│  └──────────┘    │   (FP16)     │    └──────────┬───────────────────┘  │
│                  └──────────────┘               │                       │
│                                                 ▼                       │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                  Transformer Layer × 36                           │  │
│  │                                                                  │  │
│  │  ┌────────────────┐                                              │  │
│  │  │ RMSNorm (FP16) │◄── attention_norm                            │  │
│  │  └───────┬────────┘                                              │  │
│  │          │                                                       │  │
│  │          ▼                                                       │  │
│  │  ┌────────────────────────────────────────────────────────────┐  │  │
│  │  │            ★ FP8 注意力投影 (batched_qkv_projection)        │  │  │
│  │  │                                                            │  │  │
│  │  │  Q = FP8_GEMM(x, Wq)  →  R^{4096}  ← [n_heads×head_dim]  │  │  │
│  │  │  K = FP8_GEMM(x, Wk)  →  R^{1024}  ← [kv_heads×head_dim] │  │  │
│  │  │  V = FP8_GEMM(x, Wv)  →  R^{1024}  ← [kv_heads×head_dim] │  │  │
│  │  │                                                            │  │  │
│  │  │  FP8MatmulLayer.forward() → fp8_gemm_cu()                 │  │  │
│  │  │    ├─ M=1 (Decode): fp8_gemv_multirow CUDA kernel          │  │  │
│  │  │    └─ M>1 (Prefill): fp8_dequant + cuBLAS cublasHgemm     │  │  │
│  │  └────────────────────────────────────────────────────────────┘  │  │
│  │          │                                                       │  │
│  │          ▼                                                       │  │
│  │  ┌────────────────────────────────────────────────────────────┐  │  │
│  │  │  Q_norm, K_norm (RMSNorm per-head, dim=128, FP16)          │  │  │
│  │  │  RoPE 旋转位置编码 (sin/cos cache, head_dim=128)           │  │  │
│  │  │  KV Cache 写入/读取 (FP16, seq_len × kv_dim)               │  │  │
│  │  └────────────────────────────────────────────────────────────┘  │  │
│  │          │                                                       │  │
│  │          ▼                                                       │  │
│  │  ┌────────────────────────────────────────────────────────────┐  │  │
│  │  │  Multi-Head Attention (GQA: 32 heads, 8 KV heads)          │  │  │
│  │  │  FlashAttention2 / Standard Attention                       │  │  │
│  │  │  → attn_output ∈ R^{4096}                                  │  │  │
│  │  └────────────────────────────────────────────────────────────┘  │  │
│  │          │                                                       │  │
│  │          ▼                                                       │  │
│  │  ┌────────────────────────────────────────────────────────────┐  │  │
│  │  │  ★ FP8 输出投影 (batched_matmul_forward)                    │  │  │
│  │  │  o = FP8_GEMM(attn, Wo)  →  R^{2560}                      │  │  │
│  │  └────────────────────────────────────────────────────────────┘  │  │
│  │          │                                                       │  │
│  │          ▼                                                       │  │
│  │      x = x + o  (残差连接)                                      │  │
│  │          │                                                       │  │
│  │          ▼                                                       │  │
│  │  ┌────────────────┐                                              │  │
│  │  │ RMSNorm (FP16) │◄── ffn_norm                                 │  │
│  │  └───────┬────────┘                                              │  │
│  │          │                                                       │  │
│  │          ▼                                                       │  │
│  │  ┌────────────────────────────────────────────────────────────┐  │  │
│  │  │  ★ FP8 FFN (gate_up_swiglu)                                 │  │  │
│  │  │                                                            │  │  │
│  │  │  gate = FP8_GEMM(x, W1)  →  R^{9728}  (gate_proj)        │  │  │
│  │  │  up   = FP8_GEMM(x, W3)  →  R^{9728}  (up_proj)          │  │  │
│  │  │  h    = SwiGLU(gate, up) = silu(gate) ⊙ up                │  │  │
│  │  │  out  = FP8_GEMM(h, W2)  →  R^{2560}  (down_proj)        │  │  │
│  │  └────────────────────────────────────────────────────────────┘  │  │
│  │          │                                                       │  │
│  │          ▼                                                       │  │
│  │      x = x + out  (残差连接)                                    │  │
│  │                                                                  │  │
│  └────────────────────────────── × 36 层 ───────────────────────────┘  │
│                                                 │                       │
│                                                 ▼                       │
│  ┌──────────────┐    ┌───────────────┐    ┌──────────┐                 │
│  │ RMSNorm Final │───▶│ LM Head (FP16)│───▶│ Argmax   │───▶ Token ID   │
│  │   (FP16)      │    │ [vocab, dim]  │    │ / Sample │                 │
│  └──────────────┘    └───────────────┘    └──────────┘                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

图例: ★ = FP8 量化算子 (本次开发重点)
```

### 1.2 模型参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `dim` (hidden_size) | 2560 | 模型隐藏维度 |
| `head_dim` | 128 | 每个注意力头的维度 |
| `n_heads` | 32 | 注意力头数 |
| `n_kv_heads` | 8 | KV 注意力头数 (GQA) |
| `attn_dim` | 4096 | n_heads × head_dim (≠ dim) |
| `kv_dim` | 1024 | n_kv_heads × head_dim |
| `hidden_dim` | 9728 | FFN 中间维度 |
| `n_layers` | 36 | Transformer 层数 |
| `vocab_size` | 151936 | 词表大小 |
| `block_size` | 128 | FP8 量化块大小 |

**关键特征**: Qwen3-4B 使用 GQA (Grouped Query Attention)，且 `attn_dim = n_heads × head_dim = 32 × 128 = 4096 ≠ dim = 2560`。这意味着 Q 投影的输出维度（4096）大于模型隐藏维度（2560），这是 Qwen3 系列独有的非标准设计。

### 1.3 各算子数据流详解

#### 1.3.1 FP8 QKV 投影 → `qwen3_fp8.cpp: batched_qkv_projection()`

```
输入: rms_out [M, 2560] FP16
  ├── Wq: [4096, 2560] FP8 + scale_inv [32, 20] FP16
  │     └── Q = rms_out × Wq^T → [M, 4096] FP16
  ├── Wk: [1024, 2560] FP8 + scale_inv [8, 20] FP16
  │     └── K = rms_out × Wk^T → [M, 1024] FP16
  └── Wv: [1024, 2560] FP8 + scale_inv [8, 20] FP16
        └── V = rms_out × Wv^T → [M, 1024] FP16
```

源码参考 (`kuiper/source/model/qwen3_fp8.cpp`, L204-L222):
```cpp
void Qwen3FP8Model::batched_qkv_projection(...) const {
  auto query_fp8 = std::dynamic_pointer_cast<op::FP8MatmulLayer>(query_layer);
  auto key_fp8   = std::dynamic_pointer_cast<op::FP8MatmulLayer>(key_layer);
  auto value_fp8 = std::dynamic_pointer_cast<op::FP8MatmulLayer>(value_layer);
  // FP8 GEMM/GEMV dispatch handles M=1 and M>1 internally
  STATUS_CHECK(query_fp8->forward(rms_out, query_out));
  STATUS_CHECK(key_fp8->forward(rms_out, key_out));
  STATUS_CHECK(value_fp8->forward(rms_out, value_out));
}
```

#### 1.3.2 FP8 FFN → `qwen3_fp8.cpp: gate_up_swiglu()`

```
输入: x [M, 2560] FP16
  ├── W1 (gate_proj): [9728, 2560] FP8 → gate [M, 9728] FP16
  ├── W3 (up_proj):   [9728, 2560] FP8 → up   [M, 9728] FP16
  │
  ├── SwiGLU: out = silu(gate) ⊙ up → [M, 9728] FP16
  │
  └── W2 (down_proj): [2560, 9728] FP8 → out  [M, 2560] FP16
```

源码参考 (`kuiper/source/model/qwen3_fp8.cpp`, L234-L250):
```cpp
void Qwen3FP8Model::gate_up_swiglu(...) const {
  // FP8 path: separate forward calls + SwiGLU
  // Cannot use fused FFN kernel (needs FP16 weight tensors)
  tensor::Tensor w3_output = get_buffer(ModelBufferType::kW3Output);
  STATUS_CHECK(w1_layer->forward(input, output));   // gate = W1 * x
  STATUS_CHECK(w3_layer->forward(input, w3_output)); // up = W3 * x
  STATUS_CHECK(layers->swiglu_layer_->forward(output, w3_output, output));
}
```

### 1.4 Decode vs Prefill 双路径架构

整个推理流程在 FP8 算子层面区分两种计算路径：

| 阶段 | 场景 | M 值 | 计算路径 | 核心算子 |
|------|------|------|----------|----------|
| **Decode** | 自回归生成 | M = 1 | GEMV | `fp8_gemv_multirow` CUDA kernel |
| **Prefill** | prompt 填充 | M > 1 | GEMM | `fp8_dequant_kernel_v2` + cuBLAS `cublasHgemm` |

路径分发逻辑 (`kuiper/source/op/kernels/cuda/fp8_gemm_kernel.cu`, L234-L270):
```cpp
void fp8_gemm_cu(...) {
    if (M == 1) {
        // GEMV path: choose kernel based on N
        if (N >= 8192)      // w1/w3：4行/block，每行1个warp
            fp8_gemv_multirow<128, 4><<<...>>>();
        else if (N >= 2560)  // wq/wo/w2：2行/block，每行2个warp
            fp8_gemv_multirow<128, 2><<<...>>>();
        else                 // wk/wv (N=1024)：单行，128线程
            fp8_block_gemv_kernel<128><<<N, 128>>>();
    } else {
        // GEMM: dequant → cuBLAS Tensor Core
        fp8_dequant_kernel_v2<<<...>>>();    // FP8 → FP16
        cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, ...);
    }
}
```

---

## 2. FP8算子开发方案

### 2.1 FP8 量化格式说明

本项目采用 **FP8 E4M3FN Block-Quantized** 格式，具体规格如下：

| 属性 | 说明 |
|------|------|
| 数据类型 | FP8 E4M3FN (4位指数, 3位尾数) |
| 量化粒度 | Block-wise, block_size = 128 |
| 权重存储 | `weight[out_features, in_features]`，1字节/元素 |
| 缩放因子 | `scale_inv[⌈out/128⌉, ⌈in/128⌉]`，FP16 存储 |
| 反量化公式 | `fp16_val = fp8_val × scale_inv[row/128, col/128]` |

相比 FP16，FP8 权重体积减半（1字节 vs 2字节），同时通过分块缩放因子保持合理精度。

### 2.2 整体架构设计

FP8 算子在框架中的层次结构：

```
demo/main_qwen3.cpp           ← 入口：自动检测 FP8 格式
    │
    ▼
model/qwen3_fp8.h/cpp         ← Qwen3FP8Model 模型类
    │  (继承 Qwen3Model，覆写 QKV/FFN 方法)
    ▼
op/fp8_matmul.h/cpp            ← FP8MatmulLayer 算子封装
    │  (管理 FP8 权重/缩放因子，分发到 CUDA)
    ▼
op/kernels/cuda/fp8_gemm_kernel.cu/cuh   ← CUDA 核心实现
    │  (GEMV/GEMM 双路径, E4M3→FP32 转换)
    ▼
tools/export_qwen3-8B-fp8.py  ← 模型导出脚本
    (HuggingFace → 自定义二进制格式)
```

### 2.3 模型导出 (export_qwen3-8B-fp8.py)

**目标**: 将 HuggingFace 格式的 Qwen3-4B-FP8 模型转换为 OrinMLLM 自定义二进制格式。

**二进制文件格式**:

```
偏移         内容                         字节数
────────────────────────────────────────────────
0x00         magic = 0x66703838 ("fp88")    4
0x04         version = 7                     4
0x08         dim                             4
0x0C         hidden_dim                      4
0x10         n_layers                        4
0x14         n_heads                         4
0x18         n_kv_heads                      4
0x1C         vocab_size                      4
0x20         max_seq_len                     4
0x24         shared_classifier               1
0x25         head_dim                        4
0x29         block_size                      4
0x2D ~ 0xFF  padding                        211
────────────────────────────────────────────────
0x100        权重数据开始 (256字节对齐)
```

**权重存储顺序**:
1. FP16 非量化权重: attention_norm (×36层), ffn_norm (×36层), final_norm, token_embeddings
2. FP8 量化权重: 按 wq, wk, wv, wo, w1, w2, w3 依次存储，每层写入 FP8 权重 + FP16 scale_inv
3. FP16 非量化权重: lm_head, q_norm (×36层), k_norm (×36层)

源码参考 (`tools/export_qwen3-8B-fp8.py`, write_fp8_weights 辅助函数):
```python
def write_fp8_weights(layer_name, prefix=""):
    weight = hf_dict[f'{layer_name}.weight']           # FP8 E4M3
    scale_inv = hf_dict[f'{layer_name}.weight_scale_inv']  # BF16
    serialize_fp8_raw(out_file, weight)   # 写入 1 字节/元素
    serialize_fp16(out_file, scale_inv)   # BF16→FP16 写入 2 字节/元素
```

### 2.4 FP8MatmulLayer 算子封装

**文件**: `kuiper/source/op/fp8_matmul.cpp`, `kuiper/include/op/fp8_matmul.h`

`FP8MatmulLayer` 继承自 `Layer`，负责：
- 存储 FP8 权重张量和 FP16 缩放因子
- CPU → GPU 内存传输 (`to_cuda()`)
- 推理分发：根据输入 batch_size 调度不同的 CUDA kernel

核心 forward 逻辑 (`kuiper/source/op/fp8_matmul.cpp`, L89-L115):
```cpp
base::Status FP8MatmulLayer::forward(const tensor::Tensor& input,
                                     const tensor::Tensor& output) {
    int batch_size = input.size() / in_features_;  // M = 总元素数 / K

    kernel::fp8_gemm_cu(
        fp8_weight_.ptr<uint8_t>(),   // [N, K] FP8
        scale_inv_.ptr<half>(),        // [scale_rows, scale_cols] FP16
        input.ptr<half>(),             // [M, K] FP16
        output.ptr<half>(),            // [M, N] FP16
        batch_size, out_features_, in_features_,
        block_size_, scale_cols_,
        cublas_handle, stream);
}
```

**全局 Dequant Buffer 管理**: Prefill 路径需要将 FP8 权重反量化为 FP16 后传给 cuBLAS。为避免每层重复分配显存，使用全局共享缓冲区：

```cpp
// fp8_gemm_kernel.cu
static half* g_dequant_buffer = nullptr;
static size_t g_dequant_buffer_size = 0;

void fp8_init_dequant_buffer(size_t max_weight_elements) {
    if (g_dequant_buffer && g_dequant_buffer_size >= max_weight_elements) return;
    if (g_dequant_buffer) cudaFree(g_dequant_buffer);
    cudaMalloc(&g_dequant_buffer, max_weight_elements * sizeof(half));
    g_dequant_buffer_size = max_weight_elements;
}
```

### 2.5 CUDA Kernel 实现

#### 2.5.1 FP8 E4M3 → FP32 转换

源码参考 (`fp8_gemm_kernel.cu`, L10-L22):
```cpp
__device__ __forceinline__ float fp8e4m3_to_float(uint8_t val) {
    uint32_t s = (val >> 7);          // 符号位
    uint32_t e = (val >> 3) & 0xF;    // 4位指数
    uint32_t m = val & 0x7;           // 3位尾数
    if (e == 0 && m == 0) return s ? -0.0f : 0.0f;  // 零
    if (e == 0) {                      // 非规格化数
        float f = (float)m * 1.953125e-3f;
        return s ? -f : f;
    }
    // 规格化数: 指数偏移 E4M3→FP32: bias_fp8=7, bias_fp32=127, 差=120
    uint32_t fp32 = (s << 31) | ((e + 120) << 23) | (m << 20);
    return __uint_as_float(fp32);
}
```

**原理**: FP8 E4M3 使用 4 位指数（bias=7）和 3 位尾数。转换到 FP32 时，指数加上偏移差 120（= 127 - 7），尾数左移 20 位对齐到 FP32 的 23 位尾数字段。该函数使用位操作实现，避免浮点运算，确保零开销。

#### 2.5.2 Decode GEMV Kernel (fp8_gemv_multirow)

Decode 阶段 M=1，计算本质是矩阵-向量乘法 (GEMV)，瓶颈为权重带宽。

**多行 GEMV 设计**: 每个 CUDA block（128线程）同时处理 `ROWS_PER_BLOCK` 行输出，通过三级分发策略优化不同尺寸的线性层：

| 线性层 | 维度 [N, K] | 策略 | ROWS_PER_BLOCK | 线程/行 |
|--------|-------------|------|:---:|:---:|
| w1, w3 (gate/up_proj) | [9728, 2560] | 4行/block | 4 | 32 (1 warp) |
| wq, wo, w2 | [4096/2560, *] | 2行/block | 2 | 64 (2 warps) |
| wk, wv | [1024, 2560] | 单行 | 1 | 128 (4 warps) |

源码参考 (`fp8_gemm_kernel.cu`, L60-L136):
```cpp
template<int BLOCK_DIM = 128, int ROWS_PER_BLOCK = 4>
__global__ void fp8_gemv_multirow(
    const uint8_t* weight, const half* scale_inv,
    const half* input, half* output,
    int N, int K, int block_size, int scale_cols)
{
    constexpr int THREADS_PER_ROW = BLOCK_DIM / ROWS_PER_BLOCK;
    const int row_in_block = tid / THREADS_PER_ROW;
    const int tid_in_row = tid % THREADS_PER_ROW;

    // 向量化加载: 每次加载 16 字节 (uint4 = 16 × FP8)
    const uint4* w_row_v = reinterpret_cast<const uint4*>(weight + row * K);
    for (int kv = tid_in_row; kv < k_vec16; kv += THREADS_PER_ROW) {
        uint4 w128 = w_row_v[kv];  // 128-bit load = 16 FP8 elements
        // 块内 scale 优化: 判断 16 元素是否在同一 block 内
        if (sc_start == sc_end) {
            const float s = __half2float(__ldg(scale_row_base + sc_start));
            for (int i = 0; i < 16; i++)
                sum += fp8e4m3_to_float(wb[i]) * s * __half2float(input[k_base + i]);
        }
    }

    // 跨warp归约 (当 THREADS_PER_ROW > 32 时需要)
    if constexpr (THREADS_PER_ROW <= 32) {
        // 单warp: 直接 warp shuffle 归约
        if (lane_id == 0) output[row] = __float2half(sum);
    } else {
        // 多warp: shared memory 中继
        __shared__ float smem[ROWS_PER_BLOCK * WARPS_PER_ROW];
        // warp内 shuffle → smem → warp0 最终归约
    }
}
```

**关键优化点**:
1. **128位向量化加载** (`uint4`): 每次从全局内存加载 16 个 FP8 权重，充分利用内存带宽
2. **块内 scale 快速路径**: 当 16 个元素落在同一个 128-元素块时，只需读取一次 scale_inv
3. **Generalized 跨warp归约**: 通过 `constexpr if` 在编译期确定归约路径，避免运行时判断

#### 2.5.3 Prefill GEMM 路径

Prefill 阶段 M > 1，数据足以利用 Tensor Core 加速。采用两步策略：

**步骤 1 — FP8 反量化** (`fp8_dequant_kernel_v2`):
```cpp
__global__ void fp8_dequant_kernel_v2(...) {
    const int base = idx * 8;  // 每线程处理 8 个元素
    // 64-bit 向量化加载 (uint2 = 8 字节 = 8 FP8)
    uint2 w8 = *reinterpret_cast<const uint2*>(fp8_weight + base);
    half results[8];
    for (int i = 0; i < 8; i++) {
        const float s = __half2float(__ldg(scale_inv + ...));
        results[i] = __float2half(fp8e4m3_to_float(wb[i]) * s);
    }
    // 128-bit 向量化写入 (uint4 = 16 字节 = 8 FP16)
    *reinterpret_cast<uint4*>(fp16_weight + base) =
        *reinterpret_cast<uint4*>(results);
}
```

**步骤 2 — cuBLAS FP16 GEMM**:
```cpp
cublasHgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K, &alpha_h,
            g_dequant_buffer, K,  // 反量化后的 FP16 权重
            input_fp16, K,
            &beta_h, output_fp16, N);
```

### 2.6 Qwen3FP8Model 模型类设计

**文件**: `kuiper/source/model/qwen3_fp8.cpp`, `kuiper/include/model/qwen3_fp8.h`

`Qwen3FP8Model` 继承自 `Qwen3Model`，通过覆写关键方法实现 FP8 推理路径：

| 覆写方法 | 功能 | 原因 |
|----------|------|------|
| `create_param_layers()` | 权重加载 | FP8 权重需要专门的加载逻辑（1字节权重+FP16 scale） |
| `batched_qkv_projection()` | QKV 投影 | 需 `dynamic_pointer_cast` 到 `FP8MatmulLayer` |
| `batched_matmul_forward()` | 通用矩阵乘 | 自动区分 FP8/FP16 层 |
| `gate_up_swiglu()` | FFN 前向 | 基类 Fused FFN 需要 FP16 权重指针，FP8 不兼容 |

**自动格式检测**: 通过读取文件头的 magic number 自动判断模型格式：

```cpp
// main_qwen3.cpp
if (model::is_fp8_model_file(argv[1])) {
    return inference::run_model_inference<model::Qwen3FP8Model>(...);
}
// 否则走 FP16/AWQ/SQ 路径
```

---

## 3. 遇到的困难与解决方案

### 3.1 CUDA SM 架构不匹配

**问题**: RTX 5070 Laptop GPU 是 NVIDIA Blackwell 架构，对应 SM 12.0 (`sm_120`)。CMake 自动检测返回错误的架构编号，导致 CUDA kernel 编译失败。

**表现**: 编译报错，CUDA 无法识别目标架构。

**解决方案**: 手动在 `cmake/cuda.cmake` 中指定正确的 SM 架构：
```cmake
set(CMAKE_CUDA_ARCHITECTURES "120")
```

并确保使用 CUDA 12.8+ 工具链（Blackwell 支持从 CUDA 12.6 开始）。

**根因**: Blackwell 是截至开发时最新的 GPU 架构，CMake 的 `FindCUDA` 模块尚未完全支持 SM 12.0 的自动检测。

### 3.2 head_dim ≠ dim / n_heads 导致维度错误

**问题**: Qwen3-4B 的 `head_dim=128` 但 `dim/n_heads = 2560/32 = 80`。框架基类中 `head_size_` 通过 `dim/head_num` 计算为 80，导致全部基于 `head_size_` 的下游计算（KV Cache 大小、Attention buffer、RoPE 缓存等）全部错误。

**表现**: 输出全为 NaN 或垃圾值；KV Cache 越界访问导致 segfault。

**影响范围**:
- `kv_dim = kv_heads × head_size` = 8 × 80 = 640（正确值应为 1024）
- `attn_dim = n_heads × head_size` = 32 × 80 = 2560（正确值应为 4096）
- RoPE sin/cos cache 大小错误
- Q/K/V 投影输出 buffer 大小不匹配

**解决方案**: 在 `model.cpp` 的模型加载流程中，读取 header 中的 `head_dim` 字段并覆写 `config_->head_size_`：

```cpp
// model.cpp L193-L197
if (is_qwen3_format && head_dim > 0 && head_dim != config_->head_size_) {
    LOG(INFO) << "Overriding head_size_ from " << config_->head_size_
              << " to " << head_dim;
    config_->head_size_ = head_dim;           // 80 → 128
    config_->kv_dim_ = config_->kv_head_num_ * head_dim;  // 640 → 1024
}
```

同时在 `init_mem()` 中确保所有 buffer 使用 `attn_dim = n_heads × head_size` 而非 `dim`：

```cpp
int32_t attn_dim = config_->head_num_ * config_->head_size_;  // 4096
tensor::Tensor out_mha(activation_dtype, attn_dim, true, alloc);
tensor::Tensor query(activation_dtype, attn_dim, true, alloc);
```

**教训**: 当模型的 `head_dim` 不等于 `dim/n_heads` 时（Qwen3-4B, Qwen3-8B 均如此），不能假设传统的维度关系。必须从模型 header 中显式读取 `head_dim`。

### 3.3 Fused FFN 与 FP8 权重不兼容

**问题**: 基类 `Qwen3Model` 的 FFN 使用 Fused FFN kernel，该 kernel 内部需要直接访问 `MatmulLayer` 的 FP16 权重指针。而 `FP8MatmulLayer` 存储的是 FP8 权重，无法转换为 `MatmulLayer` 的 FP16 权重格式。

**表现**: `dynamic_pointer_cast<MatmulLayer>` 返回 nullptr，触发 CHECK 失败。

**解决方案**: 在 `Qwen3FP8Model` 中覆写 `gate_up_swiglu()`，使用三次独立的 FP8 forward 调用替代 Fused FFN：

```cpp
void Qwen3FP8Model::gate_up_swiglu(int32_t layer_idx,
                                   const tensor::Tensor& input,
                                   const tensor::Tensor& output) const {
  // 分步计算：gate → up → SwiGLU → down
  STATUS_CHECK(w1_layer->forward(input, output));     // gate = W1*x
  STATUS_CHECK(w3_layer->forward(input, w3_output));  // up   = W3*x
  STATUS_CHECK(swiglu_layer->forward(output, w3_output, output)); // silu(gate)⊙up
}
```

虽然分步计算增加了 kernel launch 开销，但 FP8 带来的带宽减半收益远大于该开销。

### 3.4 HuggingFace 参考推理的 Triton 不兼容

**问题**: 验证 FP8 推理正确性时，需要 HuggingFace `transformers` 的参考输出。但 HuggingFace 使用 Triton 实现 FP8 矩阵乘法，而 Triton 不支持 SM 12.0 (Blackwell)，导致参考推理无法运行。

**表现**: `RuntimeError: Triton does not support SM 12.0`

**解决方案**: 编写 Monkey-patch 脚本，替换 HuggingFace 内部的 Triton FP8 matmul 为手动反量化 + PyTorch matmul：

```python
def patched_fp8_linear(input, weight, weight_scale_inv, ...):
    # 手动反量化: FP8 → FP16
    w_fp16 = weight.to(torch.float16)  # 位模式保持
    # 逐块缩放
    for block_row in range(0, out, 128):
        for block_col in range(0, inp, 128):
            w_fp16[block_row:block_row+128, block_col:block_col+128] *= \
                weight_scale_inv[block_row//128, block_col//128]
    return torch.matmul(input.half(), w_fp16.T)
```

通过此 patch，成功在 RTX 5070 上运行 HuggingFace 参考推理，验证了我们 FP8 框架的输出与 HF 逐 token 完全一致。

### 3.5 SSH 大文件传输截断

**问题**: 通过 SSH heredoc 传输超过 ~8KB 的 Python 脚本时，内容会被静默截断。

**解决方案**: 改用 `scp` 传输文件到远程机器：
```bash
scp local_script.py wangyh@192.168.5.102:/path/to/remote/
```

---

## 4. 性能优化技术

### 4.1 优化概览

在基线 FP8 实现的基础上，通过三项主要优化获得了显著的性能提升：

| 优化项 | Decode (tok/s) | Prefill @256tok (tok/s) |
|--------|:---------:|:-----------:|
| **基线** | 36.3 | 1791 |
| + Multi-row GEMV | 38.9 (+7.2%) | — |
| + 向量化 Dequant | — | 2062 (+15.1%) |
| + FlashAttention2 @4096tok | 27.9 (+15.8%) | 662 (+37.3%) |

### 4.2 Multi-row GEMV 优化

**问题分析**: 原始 GEMV kernel 每个 CUDA block 只处理权重矩阵的一行。对于大 N（如 w1/w3 有 9728 行），有足够多的 block 来填充 GPU；但对于中等 N（如 wq 的 4096 行），block 数量不足以充分利用所有 SM。更重要的是，单行处理时每个 block 内 128 线程处理 K=2560 维向量，每线程仅处理 20 个元素，计算量太低无法掩盖 launch 延迟。

**优化方案**: 引入 `ROWS_PER_BLOCK` 模板参数，在每个 block 内并行处理多行输出：

```
原始: N 个 blocks × 128 threads/block → 每 block 1 行
优化: N/R 个 blocks × 128 threads/block → 每 block R 行

R=4 (N≥8192): 128线程 / 4行 = 32线程(1warp)/行
  → warp-level shuffle 归约, 无 shared memory
  
R=2 (N≥2560): 128线程 / 2行 = 64线程(2warps)/行
  → 需要 shared memory 跨warp归约
```

**三级分发策略的设计依据**:

- **N ≥ 8192 (w1/w3)**: 行数多，4行/block 可将 block 数减到 2432，仍远超 SM 数量（36），同时每行仅需 32 线程（1 warp），避免跨 warp 归约开销
- **N ≥ 2560 (wq/wo/w2)**: 行数中等，2行/block 平衡 block 数和计算密度。每行 64 线程（2 warps）需要 shared memory 归约，但额外的线程使 K 维处理更快
- **N < 2560 (wk/wv, N=1024)**: 行数较少，维持单行模式，128 线程充分利用 K=2560 向量

**跨 warp 归约的泛化实现**:

```cpp
if constexpr (THREADS_PER_ROW <= 32) {
    // 单 warp: lane 0 直接写结果
    if (lane_id == 0) output[row] = __float2half(sum);
} else {
    // 多 warp: smem 中继 → warp 0 最终归约
    constexpr int WARPS_PER_ROW = THREADS_PER_ROW / 32;
    __shared__ float smem[ROWS_PER_BLOCK * WARPS_PER_ROW];
    // 各 warp lane 0 写入 smem
    // warp 0 从 smem 读取并做最终 shuffle 归约
}
```

使用 `constexpr if` 确保编译器在 ROWS_PER_BLOCK=4 (单 warp/行)时完全消除 shared memory 归约代码。

### 4.3 向量化 Dequant Kernel 优化

**问题分析**: 原始 dequant kernel 每线程处理 1 个元素，全局内存 load/store 严重低效（1 字节 FP8 读 + 2 字节 FP16 写 = 3 字节/线程）。

**优化方案**: 使用 64-bit load + 128-bit store 向量化，每线程处理 8 个元素：

```cpp
// V2: 8 elements per thread, vectorized I/O
uint2 w8 = *reinterpret_cast<const uint2*>(fp8_weight + base);   // 64-bit load (8 FP8)
half results[8];
for (int i = 0; i < 8; i++)
    results[i] = __float2half(fp8e4m3_to_float(wb[i]) * s);
*reinterpret_cast<uint4*>(fp16_weight + base) = ...;  // 128-bit store (8 FP16)
```

**效果**: 全局内存事务减少 ~8×，显著降低内存带宽压力，Prefill 吞吐提升 15.1%（256 token 输入）。

### 4.4 FlashAttention2 集成

**背景**: 在长序列场景中，标准 Attention 的 O(n²) 显存和计算开销成为瓶颈。FlashAttention2 通过分块计算和 online softmax 将显存复杂度降至 O(n)。

**集成方式**: OrinMLLM 框架已支持 FlashAttention1/2 切换。对于 FP8 模型，Attention 的输入/输出均为 FP16（在 FP8 权重矩阵乘之后），因此与浮点 FlashAttention2 完全兼容。

**性能对比** (seq_len=4096):

| 指标 | Standard Attn | FlashAttention2 | 提升 |
|------|:---:|:---:|:---:|
| Decode (tok/s) | 24.1 | 27.9 | +15.8% |
| Prefill (tok/s) | 482 | 662 | +37.3% |

### 4.5 综合性能分析

**RTX 5070 硬件约束**:
- 内存带宽: 192 GB/s (GDDR7, 128-bit × 12001 MHz)
- 计算峰值: ~15 TFLOPS FP16 (36 SMs × ~420 GFLOPS/SM)
- 显存: 8 GB

**Decode 阶段分析** (M=1):

Decode 为 bandwidth-bound。单次 forward 需加载全部 FP8 权重一次。Qwen3-4B-FP8 总权重约 4.2 GB，36层需全部遍历。

理论最大 tok/s（忽略非线性层和 attention）:
$$\text{max\_tok/s} = \frac{\text{BW}}{\text{weight\_size}} = \frac{192 \text{GB/s}}{4.2 \text{GB}} \approx 45.7 \text{ tok/s}$$

实测 38.9 tok/s，达到理论峰值的 85%——说明 FP8 GEMV kernel 已接近带宽上限，剩余 15% 开销来自 Attention、RMSNorm、Embedding、Argmax 等非线性操作。

**Prefill 阶段分析** (M>1):

Prefill 为 compute-bound（当 M 足够大时）。dequant 开销在总时间中占比随 M 增大而降低（dequant 是一次性的，cuBLAS GEMM 随 M 线性增长）。

---

## 附录: 文件清单

| 文件路径 | 功能 | 行数 |
|----------|------|------|
| `tools/export_qwen3-8B-fp8.py` | HuggingFace → 自定义二进制格式导出 | ~370 |
| `kuiper/source/model/qwen3_fp8.cpp` | Qwen3FP8Model 模型类实现 | ~276 |
| `kuiper/include/model/qwen3_fp8.h` | Qwen3FP8Model 头文件 | ~30 |
| `kuiper/source/op/fp8_matmul.cpp` | FP8MatmulLayer 算子实现 | ~130 |
| `kuiper/include/op/fp8_matmul.h` | FP8MatmulLayer 头文件 | ~80 |
| `kuiper/source/op/kernels/cuda/fp8_gemm_kernel.cu` | CUDA kernel 实现 | ~330 |
| `kuiper/source/op/kernels/cuda/fp8_gemm_kernel.cuh` | CUDA kernel 头文件 | ~20 |
| `kuiper/source/model/model.cpp` | 模型加载（FP8 格式检测+head_dim覆写） | 修改 |
| `demo/main_qwen3.cpp` | 推理入口（FP8 自动检测） | 修改 |
| `cmake/cuda.cmake` | CUDA 编译配置（SM 12.0） | 修改 |
