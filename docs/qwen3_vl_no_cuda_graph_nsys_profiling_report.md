# Qwen3-VL 无 CUDA Graph 模式 — Nsight Systems 深度分析报告

> **Profile 文件**: `qwen3_vl_fused_rope_kv_profile_no_cuda_graph.nsys-rep`
> **Nsight Systems 版本**: 2024.5.4
> **硬件平台**: NVIDIA Jetson Orin (Ampere GPU, LPDDR5 ~170 GB/s)
> **模型**: Qwen3-VL-8B FP16, 36 层 Transformer, GQA (32 Q heads / 8 KV heads)
> **运行配置**: `--fused-rope-kv --stream --max-pixels 500000`（无 CUDA Graph）
> **分析日期**: 2026-03-23

---

## 目录

- [1. 整体执行概览](#1-整体执行概览)
- [2. 推理阶段划分与时间线](#2-推理阶段划分与时间线)
- [3. CUDA Kernel 统计分析](#3-cuda-kernel-统计分析)
- [4. CUDA API 调用分析](#4-cuda-api-调用分析)
- [5. 内存传输 (H2D / D2H / D2D) 分析](#5-内存传输-h2d--d2h--d2d-分析)
- [6. Stream 使用与 Overlap 分析](#6-stream-使用与-overlap-分析)
- [7. 单步 Decode 详细 Kernel 执行序列](#7-单步-decode-详细-kernel-执行序列)
- [8. 与 CUDA Graph 模式的对比](#8-与-cuda-graph-模式的对比)
- [9. 当前推理流程存在的问题与优化建议](#9-当前推理流程存在的问题与优化建议)

---

## 1. 整体执行概览

### 关键性能指标

| 指标 | 数值 |
|------|------|
| 总 Profile 时长 | ~36.7 s |
| 生成 Token 数 | 249 |
| Decode 吞吐量 | **9.50 tok/s** |
| 平均 Decode 延迟 | **105.25 ms/tok** |
| 最小 Decode 延迟 | 102.50 ms/tok |
| 最大 Decode 延迟 | 116.20 ms/tok |
| StdDev | 2.95 ms |
| GPU 总 Kernel 实例数 | ~138,000 |
| CUDA Stream 数 | 2 (Stream 7, Stream 13) |
| CUDA Graph | **禁用** |
| `cudaLaunchKernel` 调用总数 | 127,515 |

### GPU Kernel 总时间分布（Top 4 占 95%）

| 类别 | 总时间 (ms) | 占比 |
|------|----------:-:|:----:|
| Fused FFN (W1+W3+SwiGLU) | 11,136.3 | **41.9%** |
| GEMV (Q/K/V/WO/W2) | 10,489.7 | **39.4%** |
| LM Head GEMV | 1,896.9 | **7.1%** |
| Flash Attention Decode | 1,730.0 | **6.5%** |
| cuBLAS GEMM (仅 Prefill/ViT) | 639.6 | 2.4% |
| 其他 (Norm, RoPE, Add, etc.) | 720.3 | 2.7% |
| **合计** | **26,612.8** | **100%** |

---

## 2. 推理阶段划分与时间线

```
时间线 (秒)
0.0     0.4     0.6   2.0                     9.3  9.5  9.9  9.9  10.5                        36.7
 │       │       │     │                       │    │    │    │    │                              │
 │ 初始化 │权重   │     │  权重加载 (续)          │Sin │ VE │ ME │ PF │       Decode ×249            │
 │       │load  │     │  (stream 7&13)        │Cos │    │    │    │       (直接 Kernel Launch)    │
 │       │(S7)  │     │                       │    │    │    │    │                              │
 ├───────┼──────┤     ├──────────────────────┤    │    │    │    ├──────────────────────────────┤
                                                   │    │    │    │
                                                 0.95ms 467ms 0.06ms 550ms
```

| 阶段 | 开始时间 | 结束时间 | 耗时 | 说明 |
|------|---------|---------|------|------|
| 模型权重加载 | ~0.4 s | ~9.3 s | ~8.9 s | H2D 传输 ~15.7 GB FP16 权重 |
| Sin/Cos 缓存 | ~9.34 s | ~9.34 s | 0.955 ms | 1 个 kernel |
| Vision Encoder | ~9.46 s | ~9.93 s | **~467 ms** | 27 层 ViT + 图像归一化 + Merger |
| Multimodal Embed | ~9.93 s | ~9.93 s | 0.065 ms | 嵌入融合 |
| LLM Prefill (511 tokens) | ~9.93 s | ~10.48 s | **~550 ms** | 36 层 Transformer（cuBLAS GEMM） |
| Decode ×249 | ~10.48 s | ~36.68 s | **~26,206 ms** | 直接 `cudaLaunchKernel` ×510/步 |

---

## 3. CUDA Kernel 统计分析

### 3.1 GPU Kernel 汇总（按总时间排序）

| 排名 | Kernel 名称 | 实例数 | 总时间 (ms) | 平均 (ms) | 占比 | 功能说明 |
|:---:|------------|:-----:|----------:|--------:|:----:|---------|
| 1 | `fused_gate_up_swiglu_kernel_fp16_v2` | 8,964 | 11,136.3 | 1.24 | 41.9% | FFN W1+W3+SwiGLU 融合 |
| 2 | `gemv_pure_fp16_kernel_v2` | 44,820 | 10,489.7 | 0.23 | 39.4% | Q/K/V/WO/W2 GEMV |
| 3 | `gemv_fp16_input_fp16_weight_fp32_output` | 250 | 1,896.9 | 7.59 | 7.1% | LM Head GEMV |
| 4 | `flash_attention_decode_kernel_fp16_online_softmax` | 8,964 | 1,730.0 | 0.19 | 6.5% | Flash Attention (Decode) |
| 5 | `ampere_h16816gemm_128x128 (32x5)` | 306 | 339.4 | 1.11 | 1.3% | cuBLAS GEMM (Prefill) |
| 6 | `row_rmsnorm_pure_fp16_dim<128>` | 18,072 | 149.6 | 0.008 | 0.6% | QK-Norm |
| 7 | `row_rmsnorm_pure_fp16<128>` | 18,178 | 147.8 | 0.008 | 0.6% | Input/FFN RMSNorm |
| 8 | `ampere_h16816gemm_128x128 (64x3)` | 36 | 113.9 | 3.16 | 0.4% | Prefill W2 GEMM |
| 9 | `add_kernel_cu_fp16_impl` | 18,000 | 107.4 | 0.006 | 0.4% | 残差连接 |
| 10 | `vision_softmax_fp16_kernel` | 27 | 89.0 | 3.30 | 0.3% | Vision Softmax |
| 11 | `argmax_kernel_fp32` | 250 | 63.0 | 0.25 | 0.2% | Token 采样 |
| 12 | `mrope_kernel_cu_fp16_impl` | 8,964 | 53.4 | 0.006 | 0.2% | MRoPE 编码 |
| 13 | `cutlass::Kernel2 (tn)` | 27 | 47.3 | 1.75 | 0.2% | ViT Attention GEMM |
| 14 | `ampere_h16816gemm_128x64_ldg8_tn` | 27 | 47.3 | 1.75 | 0.2% | ViT GEMM |
| 15 | `ampere_h16816gemm_128x64_ldg8_nn` | 27 | 43.5 | 1.61 | 0.2% | ViT S·V GEMM |
| 16 | `cutlass::Kernel2 (64x64 tn)` | 36 | 33.9 | 0.94 | 0.1% | Prefill K·Q GEMM |
| 17 | `causal_softmax_fp16_kernel` | 36 | 30.4 | 0.84 | 0.1% | Prefill Softmax |
| 18 | `cutlass::Kernel2 (64x64 nn)` | 36 | 21.7 | 0.60 | 0.1% | Prefill S·V GEMM |
| 19 | `bias_add_residual_fp16_kernel` | 86 | 12.7 | 0.15 | <0.1% | ViT 偏置残差 |
| 20 | `swiglu_kernel_cu_fp16_vec` | 36 | 10.1 | 0.28 | <0.1% | Prefill SwiGLU |

### 3.2 关键观察：Decode 使用自定义 GEMV 而非 cuBLAS

无 CUDA Graph 模式下，Decode 阶段的所有矩阵运算均使用**自定义 GEMV kernel**，而非 cuBLAS GEMM：

| 投影 | Grid | Block | 耗时 (ms) | 输出维度 |
|------|------|-------|--------:|---------|
| Q GEMV | (512, 1, 1) | (256, 1, 1) | 0.210 | 4096 |
| K GEMV | (128, 1, 1) | (256, 1, 1) | 0.059 | 1024 |
| V GEMV | (128, 1, 1) | (256, 1, 1) | 0.058 | 1024 |
| WO GEMV | (512, 1, 1) | (256, 1, 1) | 0.211 | 4096 |
| Fused W1+W3+SwiGLU | (1536, 1, 1) | (256, 1, 1) | 1.224 | 12288 → SwiGLU → 12288 |
| W2 GEMV | (512, 1, 1) | (256, 1, 1) | 0.622 | 4096 |
| LM Head GEMV | (18992, 1, 1) | (256, 1, 1) | 7.477 | 151936 |

**对比 CUDA Graph 模式**：CUDA Graph 模式下使用 cuBLAS `ampere_h16816gemm` (GEMM)，而非 GEMV。GEMV 是为 batch_size=1 优化的 kernel，理论上更适合 decode 单 token 场景。

---

## 4. CUDA API 调用分析

### 4.1 API 调用汇总（按总时间排序）

| API 函数 | 调用次数 | 总时间 (ms) | 平均 (ms) | 占比 | 说明 |
|---------|:------:|----------:|--------:|:----:|------|
| `cudaStreamSynchronize` | 504 | 23,590.2 | 46.81 | **69.0%** | GPU 等待 |
| `cudaMemcpyAsync` | 9,767 | 4,259.2 | 0.44 | 12.5% | 异步内存拷贝 |
| `cudaMalloc` | 1,059 | 3,510.5 | 3.31 | 10.3% | GPU 内存分配 |
| `cudaLaunchKernel` | 127,515 | 1,919.3 | **0.015** | **5.6%** | Kernel 发射 |
| `cudaFree` | 9 | 503.7 | 55.96 | 1.5% | GPU 内存释放 |
| `cudaMemcpy` | 602 | 355.4 | 0.59 | 1.0% | 同步内存拷贝 |
| `cuKernelGetFunction` | 99 | 26.3 | 0.27 | <0.1% | Kernel 函数查找 |
| `cuLibraryLoadData` | 4 | 5.3 | 1.32 | <0.1% | Kernel 库加载 |
| `cudaDeviceSynchronize` | 9 | 1.4 | 0.15 | <0.1% | 设备同步 |

### 4.2 关键发现

**`cudaLaunchKernel` 高频调用**：
- 127,515 次调用，平均 15.05 µs/次
- 每个 decode 步骤 ~510 次 kernel launch
- 每步 CPU 侧 launch 开销：~7.68 ms
- 对比 CUDA Graph 模式的 0.25 ms/步（单次 `cudaGraphLaunch`），**launch 开销增大 30.7×**

**`cudaStreamSynchronize` 模式变化**：
- 504 次调用（CUDA Graph 模式为 256 次 — 因为 decode 阶段每步只需 1 次 sync）
- 251 次长等待 (>1 ms)：总计 23,585.9 ms，平均 93.97 ms
- 253 次短等待 (<1 ms)：总计 4.35 ms
- 平均长等待时间 93.97 ms < CUDA Graph 模式的 99.3 ms，因为 CPU 侧 launch 延迟使 GPU 出现气泡

**`cudaMemcpyAsync` 调用量大幅增加**：
- 9,767 次（CUDA Graph 模式为 1,301 次）
- 原因：无 CUDA Graph 时，每层每步都需要显式 H2D 传位置数据（37 次/步 × 249 步 ≈ 9,213）

---

## 5. 内存传输 (H2D / D2H / D2D) 分析

### 5.1 内存操作汇总

| 操作类型 | 传输次数 | 总数据量 (MB) | 总时间 (ms) | 说明 |
|---------|:------:|----------:|----------:|------|
| Host-to-Device | 10,113 | 17,536.4 | 4,224.8 | 权重 + 位置数据 |
| Device-to-Host | 250 | 0.002 | 0.41 | argmax 结果 |
| Device-to-Device | 6 | 17.9 | 0.40 | ViT 内部 |
| Memset | 73 | 0.009 | 0.056 | GEMM 输出清零 |

### 5.2 Decode 阶段 H2D 分析

| 传输类型 | 次数 | 总时间 (ms) | 每步次数 | 说明 |
|---------|:---:|----------:|:------:|------|
| 位置数据 (Stream 13) | 9,213 | 5.87 | 37 | 每层 1 次 rope_pos H2D |
| 位置更新 (Stream 7) | 249 | 0.25 | 1 | 每步 1 次 step_pos H2D |
| **合计** | 9,462 | 6.12 | 38 | |

对比 CUDA Graph 模式（每步 2-3 次 H2D），无 CUDA Graph 模式的 H2D 次数增加 **~13×**，虽然每次都极小（几字节），但增加了 `cudaMemcpyAsync` 的 CPU 侧调用开销。

---

## 6. Stream 使用与 Overlap 分析

### 6.1 Stream 操作分布

| Stream | GPU 操作数 | 用途 |
|:------:|:--------:|------|
| Stream 13 | 137,454 | **全部 Kernel 计算** + 大部分 H2D/memset |
| Stream 7 | 602 | 权重加载 + Decode 阶段位置更新 |

### 6.2 Decode 阶段 Stream Overlap

```
Decode 单步时间线 (~105 ms)
                                                                    
Stream 13: ┌─emb─┬─L0:norm─QKV─norm─rope─H2D─FA─WO─add─norm─FFN─W2─add─┬─...─┬─L35─┬─norm─┬─LM Head──┬─argmax─┐
           │0.009│                 ~2.59 ms                              │×35  │     │     │ 7.48 ms  │0.25 ms │
           │     │                                                       │     │     │     │          │        │
Stream 7:  │     │ [H2D 0.001ms]                                         │     │     │     │          │        │
           │     │  (position)                                            │     │     │     │          │        │
           └─────┴───────────────────────────────────────────────────────┴─────┴─────┴─────┴──────────┴────────┘
```

**Overlap 分析**：
- **Stream 7 仅在每步开始**执行 1 次微小 H2D (位置更新)，与 Stream 13 上的 embedding kernel 并行
- **无有效 Compute-Transfer Overlap**：所有 137,454 个 GPU 操作在 Stream 13 上**串行执行**
- 与 CUDA Graph 模式相同的 stream 模式：双 stream 设计在 decode 阶段几乎无并行收益

### 6.3 Stream 13 内 Kernel 串行化

| 度量 | 数值 |
|------|------|
| 每步 GPU Kernel 执行时间（纯计算） | 100.45 ms |
| 每步 wall-clock 延迟 | 105.25 ms |
| **CPU 调度 Gap（GPU 气泡）** | **4.80 ms/步** |
| GPU 利用率（Kernel 占总时间） | 95.4% |

4.80 ms 的 GPU 气泡来源：
- CPU 发射 510 个 kernel 的调度延迟（~7.68 ms CPU 时间，但与 GPU 执行有重叠）
- `cudaMemcpyAsync` 设置开销（37 次 H2D + 1 次 D2H）
- `cudaStreamSynchronize` 返回后 CPU 侧处理（tokenizer decode, printf, etc.）

---

## 7. 单步 Decode 详细 Kernel 执行序列

### 7.1 完整单步 Decode 结构 (546 个 GPU 操作)

```
D2H (argmax结果) → H2D (step_pos, Stream 7)
→ emb_kernel (0.009 ms)
→ [Layer 0 ~ Layer 35] × 36
→ row_rmsnorm_pure_fp16 (final norm, 0.008 ms)
→ gemv_fp16 (LM Head, 7.477 ms)
→ argmax_kernel (0.254 ms)
```

### 7.2 单层 Attention Sub-layer

| 步骤 | Kernel | Grid | Block | 耗时 (ms) | 功能 |
|:---:|--------|------|-------|--------:|------|
| 1 | `row_rmsnorm_pure_fp16` | (1,1,1) | (128,1,1) | 0.010 | Input RMSNorm |
| 2 | `gemv_pure_fp16_kernel_v2` | (512,1,1) | (256,1,1) | 0.210 | Q 投影 (4096→4096) |
| 3 | `gemv_pure_fp16_kernel_v2` | (128,1,1) | (256,1,1) | 0.059 | K 投影 (4096→1024) |
| 4 | `gemv_pure_fp16_kernel_v2` | (128,1,1) | (256,1,1) | 0.058 | V 投影 (4096→1024) |
| 5 | `row_rmsnorm_pure_fp16_dim` | (32,1,1) | (128,1,1) | 0.008 | Q-head 归一化 (32 heads) |
| 6 | `row_rmsnorm_pure_fp16_dim` | (8,1,1) | (128,1,1) | 0.007 | K-head 归一化 (8 heads) |
| 7 | `mrope_kernel_cu_fp16_impl` | (16,1,1) | (128,1,1) | 0.007 | MRoPE 位置编码 |
| 8 | `[H2D]` | — | — | 0.001 | rope_pos 传递 |
| 9 | `flash_attention_decode` | (32,1,1) | (256,1,1) | 0.150 | Flash Attention (Online Softmax) |
| 10 | `gemv_pure_fp16_kernel_v2` | (512,1,1) | (256,1,1) | 0.211 | WO 投影 (4096→4096) |
| 11 | `add_kernel_cu_fp16_impl` | (2,1,1) | (256,1,1) | 0.006 | 残差连接 |
| | **Attention 小计** | | | **0.727** | |

### 7.3 单层 FFN Sub-layer

| 步骤 | Kernel | Grid | Block | 耗时 (ms) | 功能 |
|:---:|--------|------|-------|--------:|------|
| 12 | `row_rmsnorm_pure_fp16` | (1,1,1) | (128,1,1) | 0.009 | FFN 前 RMSNorm |
| 13 | `fused_gate_up_swiglu_fp16_v2` | (1536,1,1) | (256,1,1) | 1.224 | W1+W3+SwiGLU **融合** |
| 14 | `gemv_pure_fp16_kernel_v2` | (512,1,1) | (256,1,1) | 0.622 | W2 down_proj (12288→4096) |
| 15 | `add_kernel_cu_fp16_impl` | (2,1,1) | (256,1,1) | 0.006 | 残差连接 |
| | **FFN 小计** | | | **1.861** | |

### 7.4 单层与完整 Decode 步骤时间分解

| 组件 | 耗时 (ms) | 占比 |
|------|--------:|:----:|
| **Attention Sub-layer** (每层) | 0.727 | 28.1% |
| **FFN Sub-layer** (每层) | 1.861 | 71.9% |
| **单层合计** | **2.588** | 100% |

| 组件 | 耗时 (ms) | 占 Decode 步骤占比 |
|------|--------:|:---------:|
| 36 层 Transformer | ~93.2 | 92.8% |
| ↳ GEMV (Q/K/V/WO/W2) × 36 | 41.6 | 41.4% |
| ↳ Fused FFN (W1+W3+SwiGLU) × 36 | 44.0 | 43.8% |
| ↳ Flash Attention × 36 | 5.4 | 5.4% |
| ↳ Norm + MRoPE + Add × 36 | 2.2 | 2.2% |
| LM Head GEMV | 7.48 | 7.4% |
| Argmax + D2H + Emb + H2D | 0.29 | 0.3% |
| **GPU Kernel 合计** | **100.45** | 100% |
| CPU 调度 Gap | ~4.80 | — |
| **Wall-clock 合计** | **~105.25** | — |

---

## 8. 与 CUDA Graph 模式的对比

### 8.1 性能对比总表

| 指标 | 无 CUDA Graph | 有 CUDA Graph | 差异 |
|------|:-----------:|:-----------:|:----:|
| Decode 吞吐量 | 9.50 tok/s | 9.87 tok/s | **-3.7%** |
| 平均 Decode 延迟 | 105.25 ms | 101.52 ms | **+3.73 ms** |
| 最小 Decode 延迟 | 102.50 ms | 100.36 ms | +2.14 ms |
| 最大 Decode 延迟 | 116.20 ms | 110.81 ms | +5.39 ms |
| StdDev | 2.95 ms | — | — |
| GPU Kernel 时间/步 | 100.45 ms | ~100.5 ms | ≈ 持平 |
| CPU Gap/步 | 4.80 ms | ~1.0 ms | **+3.80 ms** |
| `cudaLaunchKernel` / 步 | 510 次 | 1 次 | **510× ↑** |
| Launch 时间/步 | 7.68 ms | 0.25 ms | **30.7× ↑** |
| H2D 次数/步 | 38 | 2-3 | **~13× ↑** |
| GPU 总 Kernel 实例 | ~138,000 | ~3,600 | **38× ↑** |

### 8.2 Kernel 执行策略差异

| 操作 | 无 CUDA Graph | 有 CUDA Graph | 说明 |
|------|:----------:|:----------:|------|
| Q/K/V/WO GEMV | `gemv_pure_fp16_v2` | `ampere_h16816gemm` (cuBLAS) | 自定义 vs cuBLAS |
| FFN W1+W3 | `fused_gate_up_swiglu` (融合) | `ampere_h16816gemm` × 2 + `swiglu` | **融合 vs 分离** |
| FFN W2 | `gemv_pure_fp16_v2` | `ampere_h16816gemm` (cuBLAS) | 自定义 vs cuBLAS |
| Attention | `flash_attention_decode` | `cutlass::Kernel2` + `causal_softmax` | **Flash Attn vs GEMM** |
| KV Cache Write | MRoPE kernel 内部处理 | `fused_kv_cache_update` | 不同实现 |
| memset | 无 | 72 次/步 (K/V GEMM) | cuBLAS 的 GEMM 需要 memset |

### 8.3 关键差异分析

**1. 自定义 GEMV vs cuBLAS GEMM**

cuBLAS 的 `ampere_h16816gemm` 是通用矩阵乘法，即使 batch_size=1，也会使用 tile-based GEMM 策略，包含 shared memory 搬运、tile 分割等开销。而自定义 `gemv_pure_fp16_kernel_v2` 针对向量-矩阵乘做了特化优化，直接读取权重的一行并点积。

以 Q 投影为例：
- cuBLAS GEMM (CUDA Graph): 0.839 ms（包含 tile 设置开销）
- 自定义 GEMV (无 Graph): 0.210 ms
- **GEMV 快 4.0×**

但 GEMV 的这一优势被 CUDA Graph 省去的 launch 开销所**部分抵消**。

**2. Fused FFN Kernel**

无 CUDA Graph 模式使用了 `fused_gate_up_swiglu_kernel_fp16_v2`，将 W1 GEMV、W3 GEMV 和 SwiGLU 激活融合为一个 kernel（1.224 ms），避免了中间结果的全局内存往返。CUDA Graph 模式下使用 3 个独立 kernel（W1: 0.83ms + W3: 0.83ms + SwiGLU: 0.22ms = 1.88ms）。

- 融合 FFN: 1.224 ms
- 分离 FFN: 1.88 ms
- **融合方案快 35%**

**3. Flash Attention vs GEMM-based Attention**

无 CUDA Graph 模式在 decode 阶段使用真正的 Flash Attention（`flash_attention_decode_kernel_fp16_online_softmax`，0.150 ms），而 CUDA Graph 模式使用 GEMM-based 注意力（K·Q: 0.93ms + Softmax: 0.85ms + S·V: 0.56ms = 2.34ms）。

- Flash Attention: 0.150 ms
- GEMM Attention: 2.34 ms
- **Flash Attention 快 15.6×**

**4. 综合影响**

| 因素 | 无 CUDA Graph 更优 | 有 CUDA Graph 更优 |
|------|:----------------:|:----------------:|
| Kernel 效率 (GEMV+FlashAttn+FusedFFN) | ✓ 快 ~10ms/层 | |
| Launch 开销 (510 vs 1 per step) | | ✓ 省 4.8ms/步 |
| H2D 次数 (38 vs 3 per step) | | ✓ 省调用开销 |
| **净效果** | | **有 CUDA Graph 总体快 3.7%** |

---

## 9. 当前推理流程存在的问题与优化建议

### 问题 1: 每步 510 次 `cudaLaunchKernel` 造成 CPU 瓶颈

**现象**：
- 127,515 次 `cudaLaunchKernel`，总耗时 1,919.3 ms（占 API 时间 5.6%）
- 每步 510 次 launch，CPU 侧耗时 ~7.68 ms
- GPU 气泡 4.80 ms/步，占 wall-clock 的 4.6%
- 对比 CUDA Graph 模式（1 次 launch，0.25 ms），**launch 开销增大 30.7×**

**分析**：CPU 提交 kernel 的速率无法完全掩盖 GPU 执行间隙。在 Orin 的 ARM Cortex-A78AE CPU 上，每次 `cudaLaunchKernel` 平均 15 µs，510 次累计 7.68 ms。虽然 CPU 和 GPU 可以流水线并行，但 GPU kernel 平均只有 ~0.2 ms，这意味着每 ~13 个 kernel，CPU 的 launch 延迟就会形成一次 GPU 空闲。

**优化建议**：
- **启用 CUDA Graph**：将 510 次 launch 合并为 1 次 Graph Launch，消除 4.80 ms GPU 气泡
- **Kernel 融合**：减少 kernel 数量（如融合 QKV 投影、融合 QK-Norm + MRoPE）
- 预期收益：消除 ~4.80 ms/步，等效 ~4.6% 加速

---

### 问题 2: Fused FFN 和 GEMV 无法在 CUDA Graph 中使用

**现象**：
- 无 CUDA Graph 模式使用了更高效的 `fused_gate_up_swiglu_kernel_fp16_v2` 和 `gemv_pure_fp16_kernel_v2`
- CUDA Graph 模式回退使用 cuBLAS `ampere_h16816gemm`，效率较低
- Fused FFN 每层省 0.65 ms，Flash Attention 每层省 2.19 ms，GEMV vs GEMM 总体也有提升

**根因分析**：CUDA Graph 捕获时不支持某些动态行为（如自定义 GEMV 用的 host-side 参数传递、或者 H2D memcpy 位于计算流中间），导致代码回退使用 cuBLAS 以确保 Graph 录制兼容性。

**优化建议**：
- **修改 CUDA Graph 捕获逻辑**，使其支持 GEMV + Fused FFN + Flash Attention：
  - 将 rope_pos 的 H2D 改为 GPU 端 update（通过一个小 kernel 更新 device 端的 position buffer），避免 Graph 中插入 H2D
  - 确保所有 kernel 参数在 Graph capture 时可确定（device pointer 固定、config 不变）
- **预期性能提升**：
  - 当前 CUDA Graph decode: 101.52 ms
  - 理想 (CUDA Graph + GEMV + Fused FFN + FlashAttn): ~95-96 ms
  - 预计额外提速 **~5-6%**

---

### 问题 3: 内存带宽仍是根本瓶颈

**现象**：
- FP16 模型权重 14.44 GB
- 理论带宽限速最优 decode: 14.44 GB / 170 GB/s = 84.9 ms
- 实际 GPU kernel 时间: 100.45 ms
- **带宽利用率: 143.8 GB/s / 170 GB/s = 84.6%**
- GEMV + Fused FFN 总计 85.6 ms 是纯带宽受限操作

**分析**：自定义 GEMV kernel 的带宽利用率 (84.6%) 低于 CUDA Graph 中 cuBLAS GEMM 的 88.2%。虽然 GEMV 减少了计算冗余，但其对 Orin 内存系统的访问模式可能不如 cuBLAS 的高度优化实现那样充分利用 LPDDR5 的 burst 模式和 bank interleaving。

**优化建议**：
- **模型量化 (AWQ INT4/INT8)**：将权重压缩到 3.6-7.2 GB，可将 decode 延迟降至 25-50 ms
- GEMV kernel 优化：增加向量化宽度、对齐内存访问、利用 warp shuffle 减少 shared memory 开销
- 这是 **P0 优先级** 优化

---

### 问题 4: LM Head GEMV 占 Decode 7.4%

**现象**：
- `gemv_fp16_input_fp16_weight_fp32_output`: 7.48 ms，Grid (18992, 1, 1)
- 权重矩阵 151936 × 4096 = 1.19 GB FP16
- 是单步 decode 中**最慢的单个 kernel**
- 理论带宽限速: 1.19 GB / 170 GB/s = 7.0 ms → **利用率 93.6%**

**分析**：LM Head GEMV 的带宽利用率已相当高（93.6%），接近理论最优。进一步优化空间有限。

**优化建议**：
- 使用词表剪枝（Top-K 采样初筛后精确计算）
- 量化后 LM Head 权重减半，可将该 kernel 降至 ~3.7 ms
- 优先级：中

---

### 问题 5: Vision Encoder Softmax 仍然耗时过长

**现象**：
- `vision_softmax_fp16_kernel`: 27 实例，总耗时 89.0 ms，平均 3.30 ms/次
- Grid (31104, 1, 1)，Block (256, 1, 1)
- 与 CUDA Graph 模式中相同（ViT 部分不受 CUDA Graph 影响）

**优化建议**：
- 融合 Vision Attention 为 Flash Attention (Q·K + Softmax + S·V → 单 kernel)
- 预期可节省 ~60-70 ms 的 Vision Encoder 时间

---

### 问题 6: `cudaMemcpyAsync` 调用过多

**现象**：
- 9,767 次 `cudaMemcpyAsync`，总 CPU 时间 4,259.2 ms
- 其中 ~9,213 次用于 decode 阶段的 rope 位置数据传递（每层 1 次 H2D）
- 每次 H2D 仅传几字节，但 CPU 侧 API 调用开销为 ~0.44 ms/次

**分析**：每次 `cudaMemcpyAsync` 调用涉及 driver 层参数校验、stream 排队、DMA 启动。对于几字节数据，API 开销远大于实际传输时间。

**优化建议**：
- 将 rope 位置数据预计算并存储在 GPU 端，用一个小 kernel 更新而非 H2D
- 或者合并多层的位置数据为一次 H2D（36 层 → 1 次 H2D）
- 预期可减少 ~9,000 次 cudaMemcpyAsync 调用

---

### 问题 7: `cudaMalloc` 初始化耗时仍然偏高

**现象**：
- 1,059 次 `cudaMalloc`，总耗时 3,510.5 ms
- 占初始化阶段显著份额

**优化建议**：
- 内存池化（一次分配，arena 分配器管理）
- 使用 `cudaMallocAsync`

---

### 问题 8: Prefill 注意力使用 GEMM 而非 Flash Attention

**现象**：
- Prefill 阶段 (511 tokens) 使用 `cutlass::Kernel2` + `causal_softmax` 的分离注意力
- 36 层注意力共耗时 ~86 ms（K·Q: 33.9 + Softmax: 30.4 + S·V: 21.7）
- 需要物化 32×511×511 的中间矩阵

**优化建议**：
- 在 Prefill 阶段也使用 Flash Attention
- 消除中间矩阵内存开销并减少 kernel 数量

---

### 问题优先级排序

| 优先级 | 问题 | 预期提升 | 难度 |
|:-----:|------|---------|:----:|
| **P0** | 模型量化 (AWQ INT4) | Decode 2-4× 加速 | 中 |
| **P1** | CUDA Graph 兼容 GEMV + Fused FFN + FlashAttn | Decode 5-6% 加速 | 中 |
| **P2** | 减少 H2D 调用（位置数据 GPU 端化） | 减少 CPU 开销 | 低 |
| **P3** | Prefill Flash Attention | Prefill 30-40% 加速 | 中 |
| **P4** | Vision Encoder 注意力融合 | Vision ~50% 加速 | 高 |
| **P5** | 内存池替代频繁 cudaMalloc | 初始化 2-3s 加速 | 低 |

---

## 附录：关键指标速查

| 指标 | 值 |
|------|---|
| Profile 文件 | `qwen3_vl_fused_rope_kv_profile_no_cuda_graph.nsys-rep` |
| GPU | Orin Ampere (LPDDR5 ~170 GB/s) |
| 模型 | Qwen3-VL-8B FP16, 36L, GQA 32/8, dim=4096 |
| Decode | 249 tok, 105.25 ms/tok, 9.50 tok/s |
| 自定义 Kernel | GEMV, Fused FFN, Flash Attention Decode |
| Stream | 2 (无有效 overlap) |
| CUDA Graph | 禁用 |
| Kernel Launch | 510/步, 15 µs/次, 7.68 ms/步 CPU 开销 |
| GPU 计算时间/步 | 100.45 ms |
| CPU Gap/步 | 4.80 ms |
| 带宽利用率 | 84.6% |
