# Qwen3 系列模型推理算子调用分析报告

> **工程路径**: `/mnt/ssd/workspace/OrinMLLM`  
> **目标平台**: NVIDIA Jetson Orin (SM 8.7, Ampere)  
> **CUDA Kernel 路径**: `kuiper/source/op/kernels/cuda/`  
> **分析日期**: 2026-03-28

---

## 目录

1. [Qwen3-8B FP16 模型推理流程（qwen3.cpp）](#1-qwen3-8b-fp16-模型推理流程)
2. [Qwen3-8B AWQ INT4 模型推理流程（qwen3_awq.cpp）](#2-qwen3-8b-awq-int4-模型推理流程)
3. [Qwen3-8B SmoothQuant INT8 模型推理流程（qwen3_sq.cpp）](#3-qwen3-8b-smoothquant-int8-模型推理流程)
4. [Qwen3-VL-8B 多模态模型推理流程（qwen3_vl.cpp）](#4-qwen3-vl-8b-多模态模型推理流程)
5. [四模型算子对比总结](#5-四模型算子对比总结)

---

## 模型通用参数（Qwen3-8B）

| 参数 | 值 | 说明 |
|------|-----|------|
| `dim` | 4096 | 模型隐藏维度 |
| `kv_dim` | 1024 | KV Cache 维度 (8 heads × 128) |
| `head_num` | 32 | Query 注意力头数 |
| `kv_head_num` | 8 | KV 头数 (GQA) |
| `head_size` | 128 | 每头维度 |
| `layer_num` | 36 | Transformer 层数 |
| `immediate_dim` | 12288 | FFN 中间维度 |
| `vocab_size` | 151936 | 词表大小 |
| `seq_len` | 8192 | 最大序列长度 |

---

## 1. Qwen3-8B FP16 模型推理流程

> **源文件**: `kuiper/source/model/qwen3.cpp` + `kuiper/source/model/qwen_base.cpp`  
> **运行命令**:  
> ```bash
> ./build/demo/qwen3_infer /mnt/ssd/QwenModels/Qwen3-8B-fp16.bin \
>   /mnt/ssd/QwenModels/Qwen3-8B/tokenizer.json \
>   --stream --max-tokens 1024 --prefix-cache --interactive
> ```

### 1.1 推理两阶段概述

Qwen3-8B FP16 推理分为 **Prefill（首次处理整个 prompt）** 和 **Decode（逐 token 自回归生成）** 两个阶段。

### 1.2 Prefill 阶段算子调用流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PREFILL 阶段（处理完整 prompt）                    │
│                  seq_len = prompt_length, start_pos = 0              │
└─────────────────────────────────────────────────────────────────────┘

      ┌─────────────────────────────────────────┐
      │  1. Token Embedding (CPU → GPU 数据搬运)  │
      │  emb_kernel.cuh: emb_kernel_cu()         │
      │  输入: token_ids [seq_len] (CPU)          │
      │  输出: embeddings [seq_len, 4096] (GPU)   │
      │  📦 H2D: token_ids via cudaMemcpy         │
      └──────────────────┬──────────────────────┘
                         │
    ╔════════════════════╧═══════════════════════════╗
    ║  循环 layer_idx = 0 .. 35 (36 个 Transformer)   ║
    ╚════════════════════╤═══════════════════════════╝
                         │
      ┌──────────────────▼──────────────────────┐
      │  2. Attention RMSNorm                    │
      │  rmsnorm_kernel.cuh: rmsnorm_kernel_cu() │
      │  输入: hidden [seq_len, 4096]             │
      │  输出: rms_out [seq_len, 4096]            │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  3. Batched Q/K/V Projection (cuBLAS)    │
      │  matmul_kernel.cuh:                      │
      │    batched_matmul_kernel_cu_pure_fp16()  │
      │  实际调用 cublasHgemm() × 3 次:           │
      │   Q: [seq_len, 4096] × W_q → [seq_len, 4096]  │
      │   K: [seq_len, 4096] × W_k → [seq_len, 1024]  │
      │   V: [seq_len, 4096] × W_v → [seq_len, 1024]  │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  4. Qwen3 Per-Head Q/K Norm              │
      │  rmsnorm_kernel.cuh: rmsnorm_kernel_cu() │
      │  Q: reshape [seq_len×32, 128] → RMSNorm  │
      │  K: reshape [seq_len×8, 128]  → RMSNorm  │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  5. Batched RoPE                         │
      │  rope_kernel.cuh:                        │
      │    batched_rope_kernel_cu()              │
      │  对 [seq_len] 个 Q/K 向量施加位置编码      │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  6. KV Cache 写入                         │
      │  📦 D2D: cudaMemcpyAsync (DeviceToDevice)│
      │  K → key_cache[layer, start:start+seq, :] │
      │  V → val_cache[layer, start:start+seq, :] │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  7. Batched Multi-Head Attention          │
      │  (cuBLAS + Flash Attention)               │
      │  flash_attention_kernel.cuh:              │
      │    flash_attention_prefill_fp16_cu()      │
      │  或 cuBLAS batched GEMM:                  │
      │    S = Q × K^T (cublasHgemmBatched)       │
      │    causal_softmax_fp16_cu()               │
      │    O = S × V (cublasHgemmBatched)          │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  8. Output Projection (Wo)               │
      │  matmul_kernel.cuh:                      │
      │    batched_matmul_kernel_cu_pure_fp16()  │
      │  mha_out × W_o → [seq_len, 4096]         │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  9. Residual Add                          │
      │  add_kernel.cuh: add_kernel_cu()          │
      │  hidden += wo_out                         │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  10. FFN RMSNorm                          │
      │  rmsnorm_kernel.cuh: rmsnorm_kernel_cu()  │
      │  输入: hidden [seq_len, 4096]              │
      │  输出: ffn_norm [seq_len, 4096]            │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  11. Batched FFN: Gate + Up + SwiGLU      │
      │  matmul_kernel.cuh:                       │
      │    batched_matmul_kernel_cu_pure_fp16()   │
      │  W1 (gate): ffn_norm × W1 → [seq_len, 12288]  │
      │  W3 (up):   ffn_norm × W3 → [seq_len, 12288]  │
      │  swiglu_kernel.cuh: swiglu_kernel_cu()    │
      │  SwiGLU(W1_out, W3_out) → [seq_len, 12288]    │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  12. Down Projection (W2)                 │
      │  matmul_kernel.cuh:                       │
      │    batched_matmul_kernel_cu_pure_fp16()   │
      │  swiglu_out × W2 → [seq_len, 4096]        │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  13. Residual Add                          │
      │  add_kernel.cuh: add_kernel_cu()           │
      │  hidden += w2_out                          │
      └──────────────────┬──────────────────────┘
                         │
                    (循环回到 step 2)
                         │
      ┌──────────────────▼──────────────────────┐
      │  14. Final RMSNorm                        │
      │  rmsnorm_kernel.cuh: rmsnorm_kernel_cu()  │
      │  last_token hidden → normalized            │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  15. LM Head (cls_logits)                 │
      │  matmul_kernel.cuh:                       │
      │    matmul_kernel_cu_pure_fp16()           │
      │  hidden [4096] × W_lm → logits [151936]   │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  16. Argmax Sampling                      │
      │  argmax_kernel.cuh:                       │
      │    argmax_kernel_cu_prealloc()            │
      │  📦 D2H: next_token_id → CPU              │
      └─────────────────────────────────────────┘
```

### 1.3 Decode 阶段算子调用流程图（CUDA Graph 路径）

```
┌─────────────────────────────────────────────────────────────────────┐
│              DECODE 阶段（逐 token 自回归生成）                       │
│            每步处理 1 个 token, pos = current_position               │
│            📦 = 数据搬运   🔁 = CUDA Graph 重放                      │
└─────────────────────────────────────────────────────────────────────┘

      ┌─────────────────────────────────────────┐
      │  0. Embedding + 位置更新                  │
      │  emb_kernel.cuh: emb_kernel_cu()         │
      │  📦 H2D: pos → d_pos (pinned → GPU)      │
      │  📦 D2D: input → decode_input (固定地址)   │
      └──────────────────┬──────────────────────┘
                         │
    ╔════════════════════╧════════════════════════╗
    ║  🔁 CUDA Graph 捕获/重放 (36 层 Transformer) ║
    ╚════════════════════╤════════════════════════╝
                         │
      ┌──────────────────▼──────────────────────┐
      │  1. Attention RMSNorm                    │
      │  rmsnorm_kernel.cuh: rmsnorm_kernel_cu() │
      │  输入: decode_input [4096]                │
      │  输出: rms_out [4096]                     │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  2. Q/K/V Projection (GEMV)              │
      │  matmul_kernel.cuh:                      │
      │    matmul_kernel_cu_pure_fp16()          │
      │  (M=1 → cuBLAS GEMV)                    │
      │   Q: rms_out × W_q → [4096]              │
      │   K: rms_out × W_k → [1024] → temp_key   │
      │   V: rms_out × W_v → [1024] → temp_value  │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  3. Qwen3 Per-Head Q/K Norm              │
      │  rmsnorm_kernel.cuh: rmsnorm_kernel_cu() │
      │  Q: reshape [32, 128] → RMSNorm          │
      │  K: reshape [8, 128]  → RMSNorm          │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  4. RoPE (GPU pos 版本)                   │
      │  rope_kernel.cuh:                        │
      │    rope_kernel_cu_fp16_gpu_pos()         │
      │  从 GPU 内存读取 pos (CUDA Graph 兼容)     │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  5. KV Cache 写入 (GPU pos 版本)          │
      │  kv_cache_kernel.cuh:                    │
      │    copy_to_kv_cache_kernel_fp16()        │
      │  📦 D2D: temp_key/value → cache[layer, pos]│
      │  从 GPU 内存读取 pos (CUDA Graph 兼容)     │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  6. Flash Attention Decode (GPU pos)      │
      │  flash_attention_kernel.cuh:              │
      │    flash_attention_decode_fp16_gpu_pos_cu()│
      │  内部 kernel:                              │
      │  flash_attention_decode_kernel_fp16_       │
      │    online_softmax()                       │
      │  (kv_len > 256 时使用 online softmax tiled) │
      │  从 GPU 内存读取 pos (CUDA Graph 兼容)     │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  7. Output Projection (Wo)               │
      │  matmul_kernel.cuh:                      │
      │    matmul_kernel_cu_pure_fp16()          │
      │  mha_out × W_o → attn_out [4096]         │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  8. Residual Add                          │
      │  add_kernel.cuh: add_kernel_cu()          │
      │  decode_input += attn_out                 │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  9. FFN RMSNorm                           │
      │  rmsnorm_kernel.cuh: rmsnorm_kernel_cu()  │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  10. Fused Gate+Up+SwiGLU                 │
      │  fused_ffn_kernel.cuh:                    │
      │    fused_gate_up_swiglu_kernel_cu_fp16()  │
      │  单个 kernel 完成:                         │
      │  W1·x (gate) + W3·x (up) + SwiGLU        │
      │  → [12288]                                │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  11. Down Projection (W2)                 │
      │  matmul_kernel.cuh:                       │
      │    matmul_kernel_cu_pure_fp16()           │
      │  swiglu_out × W2 → w2_out [4096]          │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  12. Residual Add                          │
      │  add_kernel.cuh: add_kernel_cu()           │
      │  decode_input += w2_out                    │
      └──────────────────┬──────────────────────┘
                         │
                    (循环回到 step 1)
                         │
      ┌──────────────────▼──────────────────────┐
      │  13. Final RMSNorm + LM Head              │
      │  rmsnorm_kernel.cuh: rmsnorm_kernel_cu()  │
      │  matmul_kernel.cuh:                       │
      │    matmul_kernel_cu_pure_fp16()           │
      │  → logits [151936]                        │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  14. Argmax Sampling                      │
      │  argmax_kernel.cuh:                       │
      │    argmax_kernel_cu_prealloc()            │
      │  📦 D2H: next_token (GPU → pinned → CPU)  │
      └─────────────────────────────────────────┘
```

### 1.4 Decode 阶段每层 CUDA Kernel 启动汇总

| 步骤 | CUDA Kernel 文件 | 具体函数 | 次数/层 |
|------|-----------------|---------|:------:|
| RMSNorm (注意力) | `rmsnorm_kernel.cuh` | `rmsnorm_kernel_cu()` | 1 |
| Q 投影 | `matmul_kernel.cuh` | `matmul_kernel_cu_pure_fp16()` | 1 |
| K 投影 | `matmul_kernel.cuh` | `matmul_kernel_cu_pure_fp16()` | 1 |
| V 投影 | `matmul_kernel.cuh` | `matmul_kernel_cu_pure_fp16()` | 1 |
| Q Norm | `rmsnorm_kernel.cuh` | `rmsnorm_kernel_cu()` | 1 |
| K Norm | `rmsnorm_kernel.cuh` | `rmsnorm_kernel_cu()` | 1 |
| RoPE | `rope_kernel.cuh` | `rope_kernel_cu_fp16_gpu_pos()` | 1 |
| KV Cache Key | `kv_cache_kernel.cuh` | `copy_to_kv_cache_kernel_fp16()` | 1 |
| KV Cache Value | `kv_cache_kernel.cuh` | `copy_to_kv_cache_kernel_fp16()` | 1 |
| Flash Attention | `flash_attention_kernel.cuh` | `flash_attention_decode_fp16_gpu_pos_cu()` | 1 |
| Wo 投影 | `matmul_kernel.cuh` | `matmul_kernel_cu_pure_fp16()` | 1 |
| Residual Add1 | `add_kernel.cuh` | `add_kernel_cu()` | 1 |
| RMSNorm (FFN) | `rmsnorm_kernel.cuh` | `rmsnorm_kernel_cu()` | 1 |
| Fused FFN | `fused_ffn_kernel.cuh` | `fused_gate_up_swiglu_kernel_cu_fp16()` | 1 |
| W2 投影 | `matmul_kernel.cuh` | `matmul_kernel_cu_pure_fp16()` | 1 |
| Residual Add2 | `add_kernel.cuh` | `add_kernel_cu()` | 1 |
| **每层合计** | | | **16** |
| **36 层总计** | | | **576** |
| Final RMSNorm | `rmsnorm_kernel.cuh` | `rmsnorm_kernel_cu()` | 1 |
| LM Head | `matmul_kernel.cuh` | `matmul_kernel_cu_pure_fp16()` | 1 |
| Argmax | `argmax_kernel.cuh` | `argmax_kernel_cu_prealloc()` | 1 |
| **每步 Decode 总计** | | | **579** |

---

## 2. Qwen3-8B AWQ INT4 模型推理流程

> **源文件**: `kuiper/source/model/qwen3_awq.cpp`（继承 `qwen3.cpp`）  
> **运行命令**:  
> ```bash
> ./build/demo/qwen3_infer /mnt/ssd/QwenModels/Qwen3-8B-awq.bin \
>   /mnt/ssd/QwenModels/Qwen3-8B/tokenizer.json \
>   --stream --max-tokens 1024 --prefix-cache --interactive
> ```

### 2.1 AWQ 与 FP16 的关键差异

AWQ (Activation-aware Weight Quantization) 将线性层的权重量化为 INT4，推理时通过反量化 + GEMV 加速。AWQ 模型覆盖了以下父类方法：
- `batched_qkv_projection()` → 使用 `awq_fused_qkv_cu()`（Decode M=1 时融合 Q/K/V 为单次 launch）
- `batched_matmul_forward()` → 使用 AWQ GEMM/GEMV
- `gate_up_swiglu()` → 使用 `awq_fused_gate_up_swiglu_cu()`（融合 Gate+Up+SwiGLU）

### 2.2 Decode 阶段算子调用流程图（AWQ 特化部分标 🟡）

```
┌─────────────────────────────────────────────────────────────────────┐
│        DECODE 阶段 — Qwen3-8B AWQ INT4 (逐 token 生成)               │
│        🟡 = AWQ 特化的算子, 其余与 FP16 相同                         │
└─────────────────────────────────────────────────────────────────────┘

      ┌─────────────────────────────────────────┐
      │  0. Embedding + 位置更新 (同 FP16)        │
      │  emb_kernel.cuh: emb_kernel_cu()         │
      │  📦 H2D: pos → GPU                       │
      └──────────────────┬──────────────────────┘
                         │
    ╔════════════════════╧═══════════════════════╗
    ║  循环 layer_idx = 0 .. 35 (CUDA Graph)      ║
    ╚════════════════════╤═══════════════════════╝
                         │
      ┌──────────────────▼──────────────────────┐
      │  1. Attention RMSNorm (同 FP16)           │
      │  rmsnorm_kernel.cuh: rmsnorm_kernel_cu() │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  🟡 2. Fused AWQ QKV Projection           │
      │  awq_gemm_fast.cuh:                      │
      │    awq_fused_qkv_cu()                    │
      │  单次 kernel launch 完成 Q+K+V 投影:       │
      │  内部调 awq_gemv_coalesced_cu() × 3      │
      │  INT4 反量化 → FP16 GEMV                  │
      │   Q: rms_out × W_q(INT4) → [4096]        │
      │   K: rms_out × W_k(INT4) → [1024]        │
      │   V: rms_out × W_v(INT4) → [1024]        │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  3-6. Q/K Norm + RoPE + KV Cache (同 FP16) │
      │  rmsnorm_kernel_cu() × 2                   │
      │  rope_kernel_cu_fp16_gpu_pos()              │
      │  copy_to_kv_cache_kernel_fp16() × 2         │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  7. Flash Attention Decode (同 FP16)       │
      │  flash_attention_decode_fp16_gpu_pos_cu() │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  🟡 8. Wo Projection (AWQ)                │
      │  awq_gemm_fast.cuh:                      │
      │    awq_gemv_coalesced_cu()               │
      │  mha_out × W_o(INT4) → [4096]            │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  9. Residual Add (同 FP16)                │
      │  add_kernel.cuh: add_kernel_cu()          │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  10. FFN RMSNorm (同 FP16)                │
      │  rmsnorm_kernel.cuh: rmsnorm_kernel_cu() │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  🟡 11. Fused AWQ Gate+Up+SwiGLU          │
      │  awq_gemm_fast.cuh:                      │
      │    awq_fused_gate_up_swiglu_cu()         │
      │  单次 kernel 完成:                         │
      │  W1·x(INT4) + W3·x(INT4) + SwiGLU        │
      │  → [12288]                                │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  🟡 12. W2 Projection (AWQ)               │
      │  awq_gemm_fast.cuh:                      │
      │    awq_gemv_coalesced_cu()               │
      │  swiglu_out × W2(INT4) → [4096]           │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  13. Residual Add (同 FP16)                │
      │  add_kernel.cuh: add_kernel_cu()           │
      └──────────────────┬──────────────────────┘
                         │
                    (循环回到 step 1)
                         │
      ┌──────────────────▼──────────────────────┐
      │  14-15. Final RMSNorm + LM Head (FP16)    │
      │  注意: LM Head 不量化, 仍为 FP16           │
      │  matmul_kernel_cu_pure_fp16()             │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  16. Argmax Sampling (同 FP16)             │
      │  argmax_kernel_cu_prealloc()              │
      │  📦 D2H: next_token → CPU                 │
      └─────────────────────────────────────────┘
```

### 2.3 AWQ Decode 每层 Kernel 启动对比

| 步骤 | FP16 Kernel | AWQ Kernel | 变化 |
|------|-------------|------------|------|
| Q/K/V 投影 | `matmul_kernel_cu_pure_fp16()` × 3 | `awq_fused_qkv_cu()` × 1 | 3→1 ✅ |
| Wo 投影 | `matmul_kernel_cu_pure_fp16()` | `awq_gemv_coalesced_cu()` | 1→1 |
| Gate+Up+SwiGLU | `fused_gate_up_swiglu_kernel_cu_fp16()` | `awq_fused_gate_up_swiglu_cu()` | 1→1 |
| W2 投影 | `matmul_kernel_cu_pure_fp16()` | `awq_gemv_coalesced_cu()` | 1→1 |
| **每层 kernel 数** | **16** | **14** | **-2** |
| **36层总计** | **576** | **504** | **-72** |

### 2.4 Prefill 阶段差异

Prefill 阶段 AWQ 使用单独的 Q/K/V 投影（fallback 到 `awq_gemm_fast_cu()` 做 batched GEMM），其余流程与 FP16 相同。

---

## 3. Qwen3-8B SmoothQuant INT8 模型推理流程

> **源文件**: `kuiper/source/model/qwen3_sq.cpp`（继承 `qwen3.cpp`）  
> **运行命令**:  
> ```bash
> ./build/demo/qwen3_infer /mnt/ssd/QwenModels/Qwen3-8B-sq.bin \
>   /mnt/ssd/QwenModels/Qwen3-8B-sq/tokenizer.json \
>   --stream --max-tokens 1024 --prefix-cache --interactive
> ```

### 3.1 SmoothQuant 与 FP16 的关键差异

SmoothQuant 将权重量化为 INT8，推理时先量化输入（FP16→INT8）再做 INT8 GEMM。SQ 模型覆盖了以下方法：
- `attention_qkv()` → 共享量化：量化一次 `rms_out`，复用于 Q/K/V 三次 GEMV
- `attention_qkv_with_graph()` → 同上，CUDA Graph 兼容版本
- `gate_up_swiglu()` → 使用 `sq_fused_ffn_cu()` 融合 FFN
- `batched_matmul_forward()` → 使用 SQ GEMM

### 3.2 Decode 阶段算子调用流程图（SQ 特化部分标 🟢）

```
┌─────────────────────────────────────────────────────────────────────┐
│     DECODE 阶段 — Qwen3-8B SmoothQuant INT8 (逐 token 生成)          │
│     🟢 = SQ 特化的算子, 其余与 FP16 相同                              │
└─────────────────────────────────────────────────────────────────────┘

      ┌─────────────────────────────────────────┐
      │  0. Embedding + 位置更新 (同 FP16)        │
      └──────────────────┬──────────────────────┘
                         │
    ╔════════════════════╧═══════════════════════╗
    ║  循环 layer_idx = 0 .. 35 (CUDA Graph)      ║
    ╚════════════════════╤═══════════════════════╝
                         │
      ┌──────────────────▼──────────────────────┐
      │  1. Attention RMSNorm (同 FP16)           │
      │  rmsnorm_kernel.cuh: rmsnorm_kernel_cu() │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  🟢 2. 共享量化 (Quantize Once)            │
      │  sq_gemm_kernel.cuh:                     │
      │    sq_quantize_input_cu()                │
      │  FP16 rms_out → INT8 量化输入              │
      │  步骤: memset + absmax + quantize         │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  🟢 3. Pre-quantized Q/K/V GEMV × 3       │
      │  sq_gemm_kernel.cuh:                     │
      │    sq_gemv_preq_cu() × 3                 │
      │  复用已量化的输入 → INT8 GEMV:             │
      │   Q: quantized_input × W_q(INT8) → [4096] │
      │   K: quantized_input × W_k(INT8) → [1024] │
      │   V: quantized_input × W_v(INT8) → [1024] │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  4-6. Q/K Norm + RoPE + KV Cache (同 FP16) │
      │  rmsnorm_kernel_cu() × 2                   │
      │  rope_kernel_cu_fp16_gpu_pos()              │
      │  copy_to_kv_cache_kernel_fp16() × 2         │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  7. Flash Attention Decode (同 FP16)       │
      │  flash_attention_decode_fp16_gpu_pos_cu() │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  🟢 8. Wo Projection (SQ)                 │
      │  sq_gemm_kernel.cuh: sq_gemm_cu()        │
      │  mha_out → quantize → INT8 GEMM → FP16    │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  9-10. Residual Add + FFN RMSNorm (同 FP16)│
      │  add_kernel_cu() + rmsnorm_kernel_cu()   │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  🟢 11. Fused SQ FFN                       │
      │  sq_gemm_kernel.cuh:                     │
      │    sq_fused_ffn_cu()                     │
      │  单次调用完成:                              │
      │  quantize → W1·x(INT8) + W3·x(INT8)      │
      │  + SwiGLU → [12288]                       │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  🟢 12. W2 Projection (SQ)                │
      │  sq_gemm_kernel.cuh: sq_gemm_cu()        │
      │  swiglu_out → quantize → INT8 GEMM → FP16 │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  13. Residual Add (同 FP16)                │
      │  add_kernel.cuh: add_kernel_cu()           │
      └──────────────────┬──────────────────────┘
                         │
                    (循环回到 step 1)
                         │
      ┌──────────────────▼──────────────────────┐
      │  14-16. Final RMSNorm + LM Head + Argmax   │
      │  (同 FP16, LM Head 不量化)                 │
      └─────────────────────────────────────────┘
```

### 3.3 SQ 关键优化：共享量化节省 Kernel Launch

传统 SQ 每次 GEMV 需要 `quantize + gemv` 两个 kernel。共享量化方案：

```
  传统 (9 kernels):               共享量化 (4 kernels):
  ┌────────────────┐              ┌────────────────┐
  │ quantize(input)│              │ quantize(input)│ ← 只做一次！
  │ gemv Q         │              │ gemv_preq Q    │ ← 复用量化结果
  │ quantize(input)│ ← 冗余!      │ gemv_preq K    │
  │ gemv K         │              │ gemv_preq V    │
  │ quantize(input)│ ← 冗余!      └────────────────┘
  │ gemv V         │
  └────────────────┘
  节省: 36 层 × 2 次冗余 quantize = 72 次 kernel launch/step
```

### 3.4 SQ Decode 每层 Kernel 启动对比

| 步骤 | FP16 Kernel | SQ Kernel | 变化 |
|------|-------------|-----------|------|
| Q/K/V (共享量化) | `matmul_*_fp16()` × 3 | `sq_quantize_input_cu()` + `sq_gemv_preq_cu()` × 3 | 3→4 |
| Wo | `matmul_*_fp16()` | `sq_gemm_cu()` (含内部量化) | 1→1 |
| FFN (融合) | `fused_gate_up_swiglu_*_fp16()` | `sq_fused_ffn_cu()` | 1→1 |
| W2 | `matmul_*_fp16()` | `sq_gemm_cu()` (含内部量化) | 1→1 |

---

## 4. Qwen3-VL-8B 多模态模型推理流程

> **源文件**: `kuiper/source/model/qwen3_vl.cpp`  
> **运行命令**:  
> ```bash
> ./build/demo/qwen3_vl_infer /mnt/ssd/QwenModels/Qwen3-VL-8B-fp16.bin \
>   /mnt/ssd/QwenModels/Qwen3-VL-8B-Instruct/tokenizer.json \
>   --image /mnt/ssd/workspace/OrinMLLM/hf_infer/demo.jpeg \
>   --prompt "Describe this image." \
>   --cuda-graph --fused-rope-kv --stream --max-pixel 500000
> ```

### 4.1 多模态推理三阶段

Qwen3-VL 推理分为三个阶段：
1. **Vision Encode**: 图像预处理 + ViT 编码 + Merger + Deepstack
2. **Multimodal Prefill**: 文本+视觉 token 融合 + LLM Prefill (含 M-RoPE 和 Deepstack 注入)
3. **Decode**: 逐 token 生成（使用 M-RoPE 替代标准 RoPE）

### 4.2 Vision Encode 阶段算子调用流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│              VISION ENCODE 阶段（图像 → 视觉 Token）                  │
│              ViT: 27 blocks, hidden=1152, heads=16, head_dim=72     │
└─────────────────────────────────────────────────────────────────────┘

      ┌─────────────────────────────────────────┐
      │  1. 图像加载与预处理                      │
      │  CPU: STB image load + smart_resize      │
      │  📦 H2D: pixels (CPU → GPU)              │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  2. GPU 端归一化 + Patch 提取 (融合)       │
      │  fused_kernels.cuh:                      │
      │    fused_resize_normalize_patches_cu()   │
      │  或分步:                                   │
      │    fused_normalize_patches_cu()          │
      │    extract_patches_cu()                  │
      │  输入: pixels [H, W, 3] (uint8, GPU)      │
      │  输出: patches [num_patches, 1152] (FP16)  │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  3. Patch Embedding (Conv3D)              │
      │  cuBLAS cublasHgemm()                    │
      │  patches × patch_embed_weight             │
      │  + bias (broadcast_add_bias_fp16_cu)      │
      │  add_kernel.cuh:                         │
      │    broadcast_add_bias_fp16_cu()          │
      │  → [num_patches, 1152]                    │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  4. Position Embedding (插值)              │
      │  vision_encoder_kernel.cuh:              │
      │    pos_embed_interpolate_cu()            │
      │  双线性插值 → [num_patches, 1152]          │
      │  + hidden_states (残差相加)                │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  5. Vision Rotary Embedding (CPU 计算)     │
      │  vision_encoder_kernel.cuh:              │
      │    vision_rotary_emb_cu()                │
      │  📦 H2D: cos/sin cache → GPU              │
      └──────────────────┬──────────────────────┘
                         │
    ╔════════════════════╧═══════════════════════╗
    ║  循环 block_idx = 0 .. 26 (27 个 ViT blocks) ║
    ║  使用 double-buffering (hidden↔output 交替)  ║
    ╚════════════════════╤═══════════════════════╝
                         │
      ┌──────────────────▼──────────────────────┐
      │  5a. LayerNorm (norm1, 含 bias)           │
      │  vision_encoder_kernel.cuh:              │
      │    layernorm_with_bias_cu()              │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  5b. QKV Projection                       │
      │  cuBLAS cublasHgemm()                    │
      │  [num_patches, 1152] × [1152, 3456]      │
      │  + QKV bias:                              │
      │  vision_encoder_kernel.cuh:              │
      │    bias_add_residual_cu()                │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  5c. Fused Split+RoPE+Transpose           │
      │  vision_encoder_kernel.cuh:              │
      │    fused_split_rope_transpose_cu()       │
      │  QKV → Q, K, V (各含 RoPE 旋转 + 转置)   │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  5d. Self-Attention (cuBLAS)              │
      │  vision_encoder_kernel.cuh:              │
      │    vision_attention_pretransposed_cu()   │
      │  S = Q × K^T → softmax → S × V            │
      │  (内部调 cublasHgemmStridedBatched × 2)   │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  5e. Output Projection + Residual         │
      │  cuBLAS cublasHgemm()                    │
      │  vision_encoder_kernel.cuh:              │
      │    bias_add_residual_cu()                │
      │  output = proj(attn_out) + bias + input   │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  5f. LayerNorm (norm2, 含 bias)           │
      │  vision_encoder_kernel.cuh:              │
      │    layernorm_with_bias_cu()              │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  5g. MLP: fc1+bias+GELU → fc2+residual    │
      │  vision_encoder_kernel.cuh:              │
      │    vision_mlp_cu()                       │
      │  (内部: cublasHgemm + gelu_cu +           │
      │   cublasHgemm + bias_add_residual_cu)    │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────┤ (在 block 8,16,24 保存 │
      │  Deepstack:       │  中间特征用于后续注入)  │
      │  保存 hidden 快照  ◄─────────────────────┘
      └──────────────────┬──────────────────────┘
                         │
                    (循环回到 step 5a)
                         │
      ┌──────────────────▼──────────────────────┐
      │  6. Main Merger (最终 ViT 输出)            │
      │  6a. LayerNorm:                           │
      │    layernorm_with_bias_cu()              │
      │  6b. Spatial Merge (4→1):                 │
      │    vision_encoder_kernel.cuh:            │
      │      spatial_merge_cu()                  │
      │    [num_patches, 1152] → [num_tokens, 4608] │
      │  6c. Merger MLP:                          │
      │    vision_encoder_kernel.cuh:            │
      │      vision_merger_mlp_cu()              │
      │    [num_tokens, 4608] → [num_tokens, 4096]  │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  7. Deepstack Mergers × 3                  │
      │  对 block 8, 16, 24 的中间特征分别执行:     │
      │    layernorm_with_bias_cu()              │
      │    spatial_merge_cu()                    │
      │    vision_merger_mlp_cu()                │
      │  → 3 × [num_tokens, 4096]                │
      └─────────────────────────────────────────┘
```

### 4.3 Multimodal Prefill 阶段算子调用流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│          MULTIMODAL PREFILL 阶段（文本+视觉融合 → LLM 处理）          │
│          📦 = 数据搬运   🔵 = VL 特化算子                            │
└─────────────────────────────────────────────────────────────────────┘

      ┌─────────────────────────────────────────┐
      │  1. Text Embedding                        │
      │  emb_kernel.cuh: emb_kernel_cu()         │
      │  token_ids → [text_len, 4096]             │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  🔵 2. Multimodal Embedding Assembly       │
      │  fused_kernels.cuh:                      │
      │    fused_multimodal_embed_cu()           │
      │  text_embeds + visual_embeds → merged     │
      │  替换 <image_pad> 位置为视觉 token          │
      │  [text_tokens-1+vision_tokens, 4096]      │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  🔵 3. M-RoPE 位置生成 (CPU)               │
      │  生成 3D 位置: (temporal, height, width)   │
      │  📦 H2D: mrope_pos_t/h/w → GPU            │
      │  (pinned memory + async transfer)         │
      └──────────────────┬──────────────────────┘
                         │
    ╔════════════════════╧═══════════════════════╗
    ║  循环 layer_idx = 0 .. 35 (36 层 Transformer) ║
    ║  使用 double-buffering (hidden_buf0↔buf1)    ║
    ╚════════════════════╤═══════════════════════╝
                         │
      ┌──────────────────▼──────────────────────┐
      │  4. Batched Attention RMSNorm             │
      │  rmsnorm_kernel.cuh: rmsnorm_kernel_cu() │
      │  [seq_len, 4096] → [seq_len, 4096]        │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  5. Batched Q/K/V Projections (cuBLAS)    │
      │  cublasHgemm() × 3 (batched GEMM)        │
      │  Q: [seq_len, 4096], K/V: [seq_len, 1024] │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  6. Qwen3 Per-Head Q/K Norm               │
      │  rmsnorm_kernel.cuh:                     │
      │    rmsnorm_kernel_cu_dim()               │
      │  按 head_size=128 维度做 RMSNorm           │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  🔵 7. Batched M-RoPE                      │
      │  rope_kernel.cuh:                        │
      │    batched_mrope_kernel_cu_fp16()         │
      │  3D 位置编码: temporal/height/width        │
      │  section: [24, 20, 20] pairs              │
      │  读取 GPU 上的 mrope_pos_t/h/w 数组        │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  🔵 8. Fused KV Cache Update               │
      │  fused_kernels.cuh:                      │
      │    fused_kv_cache_update_cu()            │
      │  📦 D2D: K+V → cache (单次 kernel)        │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  9. Batched MHA (cuBLAS + causal softmax) │
      │  cublasHgemmBatched() × 2 (S=QK^T, O=SV)│
      │  flash_attention_kernel.cuh:              │
      │    causal_softmax_fp16_cu()              │
      │  + Output Projection: cublasHgemm(Wo)     │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  10. Residual Add                          │
      │  add_kernel.cuh: add_kernel_cu()           │
      │  layer_output = layer_input + wo_out       │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  11. Batched FFN                           │
      │  rmsnorm_kernel.cuh: rmsnorm_kernel_cu() │
      │  cublasHgemm() × 2 (W1, W3)              │
      │  swiglu_kernel.cuh: swiglu_kernel_cu()   │
      │  cublasHgemm() × 1 (W2)                  │
      │  add_kernel.cuh: add_kernel_cu()          │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  🔵 12. Deepstack Feature Injection         │
      │  (仅 layer_idx < 3 且有视觉 token 时)      │
      │  add_kernel.cuh: add_kernel_cu()           │
      │  hidden[vis_start:vis_end] +=              │
      │    deepstack_feat[layer_idx]              │
      └──────────────────┬──────────────────────┘
                         │
                    (循环回到 step 4)
                         │
      ┌──────────────────▼──────────────────────┐
      │  13. Extract last token → decode_input     │
      │  📦 D2D: cudaMemcpyAsync (单 token)       │
      └─────────────────────────────────────────┘
```

### 4.4 VL Decode 阶段算子调用流程图 (CUDA Graph)

```
┌─────────────────────────────────────────────────────────────────────┐
│     DECODE 阶段 — Qwen3-VL (逐 token 生成, CUDA Graph)               │
│     🔵 = VL 特化 (M-RoPE), 其余与 FP16 类似                         │
└─────────────────────────────────────────────────────────────────────┘

      ┌─────────────────────────────────────────┐
      │  0. Embedding + 位置更新                   │
      │  emb_kernel.cuh: emb_kernel_cu()          │
      │  📦 H2D: pos → GPU (pinned async)         │
      └──────────────────┬──────────────────────┘
                         │
    ╔════════════════════╧═══════════════════════╗
    ║  🔁 CUDA Graph 捕获/重放 (36 层)             ║
    ╚════════════════════╤═══════════════════════╝
                         │
      ┌──────────────────▼──────────────────────┐
      │  1. Attention RMSNorm                     │
      │  rmsnorm_kernel.cuh: rmsnorm_kernel_cu()  │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  2. Q/K/V Projection (GEMV)               │
      │  matmul_kernel.cuh:                       │
      │    matmul_kernel_cu_pure_fp16() × 3       │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  3. Q/K Norm                               │
      │  rmsnorm_kernel.cuh: rmsnorm_kernel_cu()×2│
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  🔵 4. M-RoPE (decode 使用 gpu_pos)        │
      │  fused_rope_kv_kernel.cuh:               │
      │  路径 A (--fused-rope-kv):                │
      │    fused_gqa_mrope_kv_decode_fp16()      │
      │    (融合 M-RoPE + KV Cache Write +        │
      │     GQA Attention 到单个 kernel)           │
      │  路径 B (非融合):                          │
      │    rope_kernel.cuh:                      │
      │      mrope_kernel_cu_fp16_gpu_pos()      │
      │    + KV Cache Write + Flash Attention      │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  5. (如路径 B) KV Cache + Flash Attention  │
      │  kv_cache_kernel.cuh:                    │
      │    copy_to_kv_cache_kernel_fp16() × 2    │
      │  flash_attention_kernel.cuh:              │
      │    flash_attention_decode_fp16_gpu_pos_cu()│
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  6. Output Projection (Wo)                │
      │  matmul_kernel.cuh:                       │
      │    matmul_kernel_cu_pure_fp16()           │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  7-8. Residual Add + FFN RMSNorm           │
      │  add_kernel_cu() + rmsnorm_kernel_cu()    │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  9. Fused Gate+Up+SwiGLU                   │
      │  fused_ffn_kernel.cuh:                    │
      │    fused_gate_up_swiglu_kernel_cu_fp16()  │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  10. W2 Projection                         │
      │  matmul_kernel.cuh:                        │
      │    matmul_kernel_cu_pure_fp16()            │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  11. Residual Add                          │
      │  add_kernel.cuh: add_kernel_cu()           │
      └──────────────────┬──────────────────────┘
                         │
                    (循环回到 step 1)
                         │
      ┌──────────────────▼──────────────────────┐
      │  12. Final RMSNorm + LM Head               │
      │  rmsnorm_kernel_cu()                      │
      │  matmul_kernel_cu_pure_fp16()             │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼──────────────────────┐
      │  13. Argmax + Output                       │
      │  argmax_kernel_cu_prealloc()              │
      │  📦 D2H: next_token → CPU (pinned async)  │
      └─────────────────────────────────────────┘
```

### 4.5 VL 模型 Vision Encoder 每 Block Kernel 汇总

| 步骤 | CUDA Kernel 文件 | 具体函数 | 说明 |
|------|-----------------|---------|------|
| LayerNorm (norm1) | `vision_encoder_kernel.cuh` | `layernorm_with_bias_cu()` | 含 bias 的 LayerNorm |
| QKV Projection | cuBLAS | `cublasHgemm()` | [N, 1152]→[N, 3456] |
| QKV Bias | `vision_encoder_kernel.cuh` | `bias_add_residual_cu()` | 加 bias |
| Split+RoPE+Transpose | `vision_encoder_kernel.cuh` | `fused_split_rope_transpose_cu()` | 融合 3 操作 |
| Self-Attention | `vision_encoder_kernel.cuh` | `vision_attention_pretransposed_cu()` | 含 cuBLAS batched GEMM |
| Proj+Bias+Residual | cuBLAS + `vision_encoder_kernel.cuh` | `cublasHgemm()` + `bias_add_residual_cu()` | |
| LayerNorm (norm2) | `vision_encoder_kernel.cuh` | `layernorm_with_bias_cu()` | |
| MLP | `vision_encoder_kernel.cuh` | `vision_mlp_cu()` | fc1+GELU+fc2+残差 |

---

## 5. 四模型算子对比总结

### 5.1 Decode 阶段算子差异对比表

| 算子 | FP16 | AWQ INT4 | SQ INT8 | VL FP16 |
|------|------|----------|---------|---------|
| **Embedding** | `emb_kernel_cu` | 同左 | 同左 | 同左 |
| **Attn RMSNorm** | `rmsnorm_kernel_cu` | 同左 | 同左 | 同左 |
| **Q/K/V 投影** | `matmul_*_fp16` ×3 | `awq_fused_qkv_cu` ×1 | `sq_quantize_input_cu` + `sq_gemv_preq_cu` ×3 | `matmul_*_fp16` ×3 |
| **Q/K Norm** | `rmsnorm_kernel_cu` ×2 | 同左 | 同左 | 同左 |
| **位置编码** | `rope_*_fp16_gpu_pos` | 同左 | 同左 | `mrope_*_fp16_gpu_pos` 或 `fused_gqa_mrope_kv_decode_fp16` |
| **KV Cache** | `copy_to_kv_cache_fp16` ×2 | 同左 | 同左 | 含在融合 kernel 或 `copy_to_kv_cache_fp16` ×2 |
| **Attention** | `flash_attn_decode_fp16_gpu_pos` | 同左 | 同左 | 含在融合 kernel 或 `flash_attn_decode_fp16_gpu_pos` |
| **Wo 投影** | `matmul_*_fp16` | `awq_gemv_coalesced_cu` | `sq_gemm_cu` | `matmul_*_fp16` |
| **FFN (Gate+Up+SwiGLU)** | `fused_gate_up_swiglu_*_fp16` | `awq_fused_gate_up_swiglu_cu` | `sq_fused_ffn_cu` | `fused_gate_up_swiglu_*_fp16` |
| **W2 投影** | `matmul_*_fp16` | `awq_gemv_coalesced_cu` | `sq_gemm_cu` | `matmul_*_fp16` |
| **Add** | `add_kernel_cu` ×2 | 同左 | 同左 | 同左 |
| **LM Head** | `matmul_*_fp16` | `matmul_*_fp16` (不量化) | `matmul_*_fp16` (不量化) | `matmul_*_fp16` |
| **Argmax** | `argmax_kernel_cu_prealloc` | 同左 | 同左 | 同左 |

### 5.2 CUDA Kernel 文件路径索引

| Kernel 文件 | 路径 | 主要算子 |
|------------|------|---------|
| `add_kernel.cuh` | `kuiper/source/op/kernels/cuda/` | 向量加法、残差连接 |
| `argmax_kernel.cuh` | 同上 | GPU Argmax 采样 |
| `awq_gemm_fast.cuh` | 同上 | AWQ INT4 GEMM/GEMV、融合 QKV、融合 FFN |
| `emb_kernel.cuh` | 同上 | Token Embedding 查表 |
| `flash_attention_kernel.cuh` | 同上 | Flash Attention Prefill/Decode、Causal Softmax |
| `fused_ffn_kernel.cuh` | 同上 | 融合 Gate+Up+SwiGLU |
| `fused_kernels.cuh` | 同上 | 多模态 embedding 组装、KV cache 更新、图像 patch 提取 |
| `fused_rope_kv_kernel.cuh` | 同上 | 融合 M-RoPE+KV+Attention (VL decode) |
| `kv_cache_kernel.cuh` | 同上 | KV Cache 写入 |
| `matmul_kernel.cuh` | 同上 | FP16/FP32 矩阵乘法 (cuBLAS 封装) |
| `mha_kernel.cuh` | 同上 | 标准多头注意力 |
| `rmsnorm_kernel.cuh` | 同上 | RMSNorm (全维度/指定维度) |
| `rope_kernel.cuh` | 同上 | RoPE/M-RoPE 位置编码、Sin/Cos 缓存 |
| `sq_gemm_kernel.cuh` | 同上 | SmoothQuant INT8 GEMM、融合 FFN、共享量化 |
| `swiglu_kernel.cuh` | 同上 | SwiGLU 激活函数 |
| `vision_encoder_kernel.cuh` | 同上 | ViT LayerNorm、GELU、Self-Attention、Spatial Merge、MLP |

### 5.3 数据搬运汇总

| 搬运类型 | 时机 | 数据内容 | 大小 |
|---------|------|---------|------|
| **H2D (CPU→GPU)** | Decode 每步 | `pos` (位置索引) | 4 bytes |
| **H2D (CPU→GPU)** | Decode 每步 | `token_id` → embedding 查表 | 4 bytes |
| **D2D (GPU→GPU)** | Decode 每步 | `input → decode_input` (固定地址) | dim × 2 bytes = 8 KB |
| **D2D (GPU→GPU)** | Decode 每层 | `temp_key/value → KV cache` | kv_dim × 2 bytes = 2 KB × 2 |
| **D2H (GPU→CPU)** | Decode 每步 | `next_token_id` (argmax 结果) | 8 bytes |
| **H2D (CPU→GPU)** | Prefill 一次 | M-RoPE 位置数组 (VL) | seq_len × 3 × 4 bytes |
| **H2D (CPU→GPU)** | Vision 一次 | 图像像素数据 | H × W × 3 bytes |
| **D2D (GPU→GPU)** | Prefill 每层 | KV cache bulk write | seq_len × kv_dim × 2 bytes × 2 |
| **D2D (GPU→GPU)** | Prefill 结束 | last_token → decode_input | dim × 2 bytes = 8 KB |
