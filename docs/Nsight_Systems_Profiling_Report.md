# Nsight Systems 性能分析报告

## 概述

本报告基于 NVIDIA Nsight Systems 对 Qwen3-VL-8B-fp16 模型推理过程的性能剖析，分析模型在 NVIDIA Orin 平台上的算子分布和数据拷贝情况。

**测试配置**：
- 模型：Qwen3-VL-8B-fp16（36 层 Transformer + 27 层 ViT）
- 平台：NVIDIA Orin（SM 8.7, ARM aarch64）
- 优化选项：CUDA Graph + Fused M-RoPE+KV Cache Write
- 输入：1 张图片（864×576, 1944 patches → 486 vision tokens）+ "Describe this image."
- 输出：249 tokens

**推理性能摘要**：

| 阶段 | 耗时 | 吞吐量 |
|------|------|--------|
| 图像预处理 | 109.88 ms | - |
| ViT 视觉编码 | 480.57 ms | - |
| Prefill（511 tokens） | 553.25 ms | 923.63 tok/s |
| Decode（249 tokens） | 25277.88 ms | 9.85 tok/s（101.52 ms/token） |
| **总计** | **26421.58 ms** | - |

---

## 1. Nsight Systems 采集指令

```bash
nsys profile \
  --output /mnt/ssd/workspace/OrinMLLM/docs/qwen3_vl_fused_rope_kv_profile \
  --force-overwrite true \
  --trace cuda,cudnn,cublas,osrt,nvtx \
  --cuda-memory-usage true \
  --stats true \
  ./build/demo/qwen3_vl_infer \
    /mnt/ssd/QwenModels/Qwen3-VL-8B-fp16.bin \
    /mnt/ssd/QwenModels/Qwen3-VL-8B-Instruct/tokenizer.json \
    --image /mnt/ssd/workspace/OrinMLLM/hf_infer/demo.jpeg \
    --prompt "Describe this image." \
    --cuda-graph --fused-rope-kv --stream --max-pixel 500000
```

**参数说明**：

| 参数 | 说明 |
|------|------|
| `--trace cuda,cudnn,cublas,osrt,nvtx` | 追踪 CUDA Runtime/Driver、cuDNN、cuBLAS、OS Runtime、NVTX 标记 |
| `--cuda-memory-usage true` | 追踪 CUDA 内存分配和释放 |
| `--stats true` | 在采集结束后自动输出统计摘要 |
| `--force-overwrite true` | 覆盖已存在的输出文件 |

**输出文件**：

| 文件 | 大小 | 说明 |
|------|------|------|
| `qwen3_vl_fused_rope_kv_profile.nsys-rep` | 557 KB | Nsight Systems 可视化文件（可在 Nsight Systems GUI 中打开） |
| `qwen3_vl_fused_rope_kv_profile.sqlite` | 2.3 MB | SQLite 数据库（可用于离线查询分析） |

---

## 2. CUDA 算子运行情况

### 2.1 全局算子分布（GPU 耗时 Top 20）

| 占比 | 总耗时(ms) | 调用次数 | 平均(us) | 算子名称 | 所属阶段 |
|-----:|----------:|--------:|---------:|----------|---------|
| 36.1% | 336.75 | 306 | 1100.49 | `ampere_h16816gemm_128x128_ldg8_stages_32x5_tn` | Prefill MatMul |
| 12.3% | 114.88 | 36 | 3191.02 | `ampere_h16816gemm_128x128_ldg8_stages_64x3_tn` | Prefill MatMul |
| 9.9% | 92.61 | 27 | 3430.07 | `vision_softmax_fp16_kernel` | ViT Softmax |
| 6.8% | 63.38 | 250 | 253.52 | `argmax_kernel_fp32` | Decode Argmax |
| 5.2% | 48.26 | 27 | 1787.25 | `ampere_h16816gemm_128x64_ldg8_tn` | ViT MatMul |
| 5.1% | 47.66 | 27 | 1765.36 | `cutlass_80_tensorop_h16816gemm_256x64_32x4_tn` | ViT MatMul |
| 4.9% | 46.09 | 27 | 1707.09 | `ampere_h16816gemm_128x64_ldg8_nn` | ViT MatMul |
| 3.7% | 34.23 | 36 | 950.93 | `cutlass_75_tensorop_h1688gemm_64x64_tn` | Prefill FlashAttn |
| 3.3% | 30.67 | 36 | 851.82 | `causal_softmax_fp16_kernel` | Prefill Softmax |
| 2.2% | 20.31 | 36 | 564.07 | `cutlass_75_tensorop_h1688gemm_64x64_nn` | Prefill FlashAttn |
| 2.1% | 19.98 | 144 | 138.77 | `row_rmsnorm_pure_fp16_dim<128>` | Prefill RMSNorm |
| 1.4% | 12.77 | 86 | 148.44 | `bias_add_residual_fp16_kernel` | ViT Residual |
| 1.2% | 11.46 | 36 | 318.40 | `swiglu_kernel_cu_fp16_vec` | Prefill SwiGLU |
| 0.9% | 8.44 | 27 | 312.60 | `bias_gelu_fp16_kernel` | ViT GELU |
| 0.9% | 8.41 | 58 | 144.98 | `layernorm_with_bias_fp16_kernel` | ViT LayerNorm |
| 0.8% | 7.91 | 27 | 292.85 | `fused_split_rope_transpose_kernel` | ViT RoPE |
| 0.8% | 7.48 | 1 | 7479.97 | `gemv_fp16_input_fp16_weight_fp32_output` | Prefill GEMV (lm_head) |
| 0.7% | 6.57 | 72 | 91.31 | `add_kernel_cu_fp16_impl` | Prefill Add |
| 0.7% | 6.23 | 36 | 173.08 | `batched_mrope_kernel_cu_fp16_impl` | Prefill M-RoPE |
| 0.2% | 2.08 | 250 | 8.30 | `emb_kernel_cu_pure_fp16_impl` | Decode Embedding |

### 2.2 按推理阶段拆分

#### ViT + Prefill 阶段

- **Kernel 调用次数**：1121 次
- **GPU 总耗时**：867.08 ms

主要算子类别分布：

```
MatMul (GEMM)        ███████████████████████████████████████████████  ~65%
  - LLM Prefill: ampere_h16816gemm 128x128 (306+36 次)
  - ViT: ampere_h16816gemm 128x64, cutlass gemm (27×3 种)
  
Softmax              ██████████████                                  ~14%
  - ViT: vision_softmax (27 次, 92.61 ms)
  - Prefill: causal_softmax (36 次, 30.67 ms)
  
FlashAttention       ████████                                        ~6%
  - cutlass_75 GEMM tn + nn (36 次 × 2)
  
RMSNorm/LayerNorm    ████                                            ~3%
  - row_rmsnorm_fp16 (144 次)
  - layernorm_with_bias (58 次)
  
Activation           ███                                             ~3%  
  - SwiGLU (36 次), BiasGELU (27 次)
  
RoPE/Position        ██                                              ~2%
  - fused_split_rope_transpose (ViT, 27 次)
  - batched_mrope (Prefill, 36 次)
  
Residual/Add         █                                               ~2%
  - bias_add_residual (86 次), add_kernel (72 次)
  
Other                █                                               ~1%
  - KV cache update, transpose, embedding, etc.
```

#### Decode 阶段（CUDA Graph 模式）

- **可见 Kernel 调用次数**：499 次
- **可见 GPU 耗时**：65.43 ms
- **CUDA Graph Launch**：249 次，62.32 ms（每次平均 250.28 us）

CUDA Graph 模式下，每步 decode 的 36 层 Transformer 计算被封装为一个 Graph 并通过 `cudaGraphLaunch` 统一提交。具体 Transformer 内部各算子**不会**作为独立 kernel 出现在 nsys trace 中，而是以 Graph Replay 形式执行。

decode 阶段可见的 kernel：

| 算子 | 调用次数 | 总耗时(ms) | 平均(us) | Grid | Block | 说明 |
|------|--------:|----------:|---------:|------|-------|------|
| `argmax_kernel_fp32` | 250 | 63.38 | 253.52 | 1×1×1 | 512×1×1 | Token 采样（含 prefill 首 token） |
| `emb_kernel_cu_pure_fp16_impl` | 249 | 2.06 | 8.25 | 1×1×1 | 256×1×1 | Token Embedding 查表 |

**decode 单步耗时分解（估算）**：

| 组件 | 耗时(us) | 说明 |
|------|--------:|------|
| H2D memcpy (位置更新) | ~1.2 | 2 次 × 4B（rope_pos + kv_cache_pos） |
| cudaGraphLaunch (提交) | ~250 | 36 层 Transformer + cls_logits |
| GPU 执行 (Graph Replay) | ~100,800 | 36 层 × {GEMM, RoPE, Attn, FFN} |
| cudaStreamSynchronize | ~100 | 等待 GPU 完成 |
| argmax_kernel | ~254 | Token 采样 |
| emb_kernel | ~8 | 下一 token embedding |
| **单步总计** | **~101,400** | ≈ 101.52 ms/token |

**CUDA Graph 内部算子组成（每步、每层）**：

每步 decode 在 CUDA Graph 内执行 36 层 Transformer，每层包含以下算子：

```
每层 (×36):
  ├── RMSNorm (attention pre-norm)        ×1
  ├── MatMul (Q projection)               ×1
  ├── MatMul (K projection)               ×1
  ├── MatMul (V projection)               ×1
  ├── RMSNorm (Q-norm)                    ×1
  ├── RMSNorm (K-norm)                    ×1
  ├── FusedMRoPEKVWrite                   ×1  ← 融合算子（替代原来 3 个 kernel）
  ├── Flash Attention Decode              ×1
  ├── MatMul (WO projection)              ×1
  ├── Add (attention residual)            ×1
  ├── RMSNorm (FFN pre-norm)              ×1
  ├── Fused Gate+Up+SwiGLU                ×1
  ├── MatMul (W2/down projection)         ×1
  └── Add (FFN residual)                  ×1
Final:
  ├── RMSNorm (final norm)                ×1
  └── MatMul (lm_head → logits)           ×1

每步 decode 总计: 36 × 14 + 2 = 506 个 kernel（CUDA Graph 内）
```

### 2.3 关键 CUDA API 调用统计

| 占比 | 总耗时(ms) | 调用次数 | 平均耗时 | API |
|-----:|----------:|--------:|---------:|-----|
| 73.7% | 25,422.67 | 256 | 99.31 ms | `cudaStreamSynchronize` |
| 11.8% | 4,053.96 | 813 | 4.99 ms | `cudaMalloc` |
| 11.6% | 4,016.24 | 1,301 | 3.09 ms | `cudaMemcpyAsync` |
| 1.4% | 496.85 | 9 | 55.21 ms | `cudaFree` |
| 1.0% | 330.99 | 602 | 549.82 us | `cudaMemcpy` |
| 0.2% | 62.32 | 249 | 250.28 us | `cudaGraphLaunch` |
| 0.2% | 53.94 | 2,027 | 26.61 us | `cudaLaunchKernel` |
| - | 4.70 | 1 | 4.70 ms | `cudaGraphInstantiate` |
| - | 1.00 | 1 | 1.00 ms | `cudaGraphExecDestroy` |
| - | 0.27 | 1 | 0.27 ms | `cudaGraphDestroy` |

**关键观察**：
- `cudaStreamSynchronize` 占 CPU 端 73.7% 时间 —— CPU 等待 GPU 执行完成
- `cudaMalloc` 占 11.8%（4.05 秒）—— 模型初始化阶段的内存分配
- `cudaMemcpyAsync` 占 11.6%（4.02 秒）—— 主要是模型权重 H2D 传输
- `cudaGraphLaunch` 平均仅 250 us —— CUDA Graph 极大降低了 kernel launch 开销

---

## 3. 数据拷贝情况

### 3.1 全局数据传输统计

| 方向 | 次数 | 总耗时(ms) | 平均(us) | 总大小(MB) | 平均大小(KB) |
|------|-----:|----------:|---------:|----------:|------------:|
| Host → Device | 1,647 | 4,213.58 | 2,558.34 | 17,536.36 | 10,647.46 |
| Device → Device | 6 | 0.40 | 66.78 | 17.93 | 2,988.71 |
| Device → Host | 250 | 0.40 | 1.60 | 0.002 | 0.01 |
| Memset | 73 | 0.06 | 0.79 | 0.01 | - |

### 3.2 H2D 传输分阶段分析

#### 模型加载阶段（ViT + Prefill 前）

| 统计项 | 值 |
|--------|-----|
| 传输次数 | 900 |
| 总耗时 | 4,213.14 ms |
| 总数据量 | 17,536.36 MB (≈17.1 GB) |
| 平均每次 | 19.48 MB |

这部分对应 Qwen3-VL-8B-fp16 模型权重从 Host 内存到 GPU 显存的传输：
- LLM 权重：~16.3 GB（36 层 × {Q,K,V,O,W1,W2,W3} + embedding + lm_head）
- ViT 权重：~0.8 GB（27 层 Vision Transformer + Merger）

大块传输（>1 MB）占绝对主导：373 次，17.53 GB，4,211.88 ms。

#### Decode 阶段

| 统计项 | 值 |
|--------|-----|
| H2D 传输次数 | 747 |
| 总耗时 | 0.44 ms |
| 总数据量 | ~3 KB |
| 平均每次 | 4 Bytes |

Decode 阶段的 H2D 传输极小 —— 全部是 **GPU 端位置更新**：
- 每步 decode 传输 3 次 × 4 Bytes = 12 Bytes：
  - `rope_pos`（M-RoPE 位置，int32）
  - `kv_cache_pos`（KV Cache 写入位置，int32）
  - `argmax_output`（采样结果回传用的地址初始化）
- 总计 747 次 × 4B = ~3 KB，耗时仅 0.44 ms（全程可忽略）

#### D2H 传输

| 统计项 | 值 |
|--------|-----|
| D2H 传输次数 | 250 |
| 总耗时 | 0.40 ms |
| 平均每次 | 1.60 us |
| 总数据量 | ~2 KB |

D2H 传输对应 argmax 采样结果从 GPU 回传到 CPU（每步 decode 1 次，每次 8 bytes / `size_t`）。

#### D2D 传输

6 次 D2D 传输，共 17.93 MB，对应 ViT 阶段的 Deepstack 特征拷贝（3 层 × 2 次 = 6 次，每次 ~3 MB）。

### 3.3 数据传输时间线

```
                模型加载          ViT  Prefill     Decode (×249)
时间:        ──────────────── ──── ──── ─────────────────────────────
H2D:         ████████████████  ·      ·    ·  ·  ·  ·  ·  ·  ·  ·  ·
             17.1 GB 权重传输        每步 12B 位置更新（可忽略）

D2D:                          ██            
                              17.9 MB Deepstack

D2H:                                        ·  ·  ·  ·  ·  ·  ·  ·  ·
                                           每步 8B argmax 结果
```

---

## 4. CUDA Graph 分析

### 4.1 Graph 生命周期

| 操作 | 耗时 | 说明 |
|------|------|------|
| Graph Capture | （包含在首次 decode step 中） | 录制 36 层 Transformer 计算图 |
| `cudaGraphInstantiate` | 4.70 ms | 将计算图编译为可执行图 |
| `cudaGraphLaunch` × 249 | 62.32 ms (avg 250 us) | 重放计算图 |
| `cudaGraphExecDestroy` | 1.00 ms | 销毁可执行图 |
| `cudaGraphDestroy` | 0.27 ms | 销毁计算图 |

### 4.2 CUDA Graph 性能优势

不使用 CUDA Graph 时，每步 decode 需要 506 次 `cudaLaunchKernel`：
- 常规 launch 开销：506 × ~26 us ≈ **13.2 ms/step**

使用 CUDA Graph 后，仅需 1 次 `cudaGraphLaunch`：
- Graph launch 开销：1 × ~250 us ≈ **0.25 ms/step**

**Kernel Launch 开销节省：13.2 → 0.25 ms/step（降低 98%）**

---

## 5. 融合算子观察

本次 profiling 使用了 `--fused-rope-kv` 选项，启用了 Fused M-RoPE + KV Cache Write 融合算子。

Prefill 阶段可以观察到：
- `fused_kv_cache_update_fp16_kernel`：36 次调用，总耗时 1.46 ms，平均 40.62 us
- `batched_mrope_kernel_cu_fp16_impl`：36 次调用，总耗时 6.23 ms，平均 173.08 us

Decode 阶段（CUDA Graph 内部）：
- 融合前每层需要 3 个 kernel：M-RoPE + CopyK + CopyV
- 融合后每层只需要 1 个 kernel：`fused_mrope_kv_write_kernel`
- 每步 decode 节省 36 × 2 = **72 个 kernel launch**
- 该优化已计入 CUDA Graph，进一步减少了 Graph 节点数

---

## 6. 性能瓶颈总结

### Decode 阶段瓶颈

1. **GEMM 计算**（~95% GPU 时间）：每层 7 次 MatMul（Q/K/V/O/W1/W2/W3），36 层 × 7 = 252 次 GEMM
2. **Flash Attention**（~3%）：每层 1 次，与序列长度相关
3. **Argmax 采样**：253 us/step 是一个相对固定的开销

### ViT 阶段瓶颈

1. **Vision Softmax**：92.61 ms（27 层 × ~3.4 ms），是 ViT 中最慢的单个算子
2. **ViT GEMM**：4 种 GEMM 共计 ~190 ms，占 ViT 计算的主体

### 优化建议

| 优化方向 | 当前状态 | 潜在收益 |
|---------|---------|---------|
| CUDA Graph | ✅ 已启用 | 节省 ~13 ms/step launch 开销 |
| Fused RoPE+KV | ✅ 已启用 | 每层减少 2 个 kernel，共减少 72 个 |
| INT8/AWQ 量化 | 未启用 | GEMM 可提速 2-4× |
| ViT Softmax 优化 | 可优化 | 92 ms → 可能降至 40-50 ms |
| Speculative Decoding | 未启用 | decode 吞吐可能提升 2-3× |
