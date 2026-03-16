# Qwen3-8B 在 NVIDIA Orin 上的推理性能预估分析报告

> 日期：2026-03-16
> 目标平台：NVIDIA Jetson AGX Orin / Orin NX / Orin Nano
> 目标模型：Qwen3-8B（Dense, 8.2B 参数）

---

## 目录

1. [概述](#1-概述)
2. [NVIDIA Orin 硬件规格](#2-nvidia-orin-硬件规格)
3. [Qwen3-8B 模型架构参数](#3-qwen3-8b-模型架构参数)
4. [性能预估理论框架：Roofline 模型](#4-性能预估理论框架roofline-模型)
5. [推理阶段分析](#5-推理阶段分析)
6. [FP16 全精度推理性能预估](#6-fp16-全精度推理性能预估)
7. [量化推理性能预估](#7-量化推理性能预估)
8. [显存占用分析](#8-显存占用分析)
9. [关键瓶颈分析](#9-关键瓶颈分析)
10. [优化建议](#10-优化建议)
11. [附录：计算公式推导](#11-附录计算公式推导)

---

## 1. 概述

本报告基于 **Roofline 模型** 和 **算子级别的计算/访存分析**，对 Qwen3-8B 模型在 NVIDIA Orin 系列平台上的推理性能进行理论预估。报告涵盖 Prefill（首次推理）和 Decode（自回归生成）两个阶段，并对 FP16、INT8 (SmoothQuant)、INT4 (AWQ)、FP8 等多种精度方案进行对比分析。

### 1.1 预估方法论

推理性能预估的核心思路：

```
推理延迟 = max(计算时间, 访存时间)
         = max(计算量 / 算力, 数据搬运量 / 带宽)

推理吞吐 = 1 / 推理延迟
```

其中，**Operational Intensity（算术强度）** 决定了推理是计算瓶颈还是带宽瓶颈：

$$
\text{Operational Intensity} = \frac{\text{FLOPs}}{\text{Bytes}} \quad (\text{FLOP/Byte})
$$

当 OI < 硬件的 **Compute-to-Memory Ratio** 时，推理受带宽限制；反之受算力限制。

---

## 2. NVIDIA Orin 硬件规格

### 2.1 Orin 系列对比

| 参数 | AGX Orin 64GB | AGX Orin 32GB | Orin NX 16GB | Orin NX 8GB | Orin Nano 8GB | Orin Nano 4GB |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| **GPU 架构** | Ampere | Ampere | Ampere | Ampere | Ampere | Ampere |
| **SM 数量** | 16 | 16 | 8 | 8 | 6 | 6 |
| **CUDA Cores** | 2048 | 2048 | 1024 | 1024 | 768 | 768 |
| **Tensor Cores** | 64 | 64 | 32 | 32 | 24 | 24 |
| **FP16 算力 (TOPS)** | 137※ | 100 | 100 | 70 | 40 | 20 |
| **INT8 算力 (TOPS)** | 275※ | 200 | 200 | 140 | 80 | 40 |
| **内存类型** | LPDDR5 | LPDDR5 | LPDDR5 | LPDDR5 | LPDDR5 | LPDDR5 |
| **内存容量 (GB)** | 64 | 32 | 16 | 8 | 8 | 4 |
| **内存带宽 (GB/s)** | 204.8 | 204.8 | 102.4 | 102.4 | 68 | 68 |
| **功耗 (W)** | 15-60 | 15-40 | 10-25 | 10-25 | 7-15 | 7-15 |

> ※ AGX Orin 64GB 在 MAXN 功耗模式下的峰值性能。

### 2.2 关键硬件指标

以 **AGX Orin 64GB (MAXN)** 为主分析平台：

| 指标 | 数值 |
|------|------|
| FP16 Tensor Core 算力 | 137 TFLOPS |
| INT8 Tensor Core 算力 | 275 TOPS |
| INT4 Tensor Core 算力 | 275 TOPS（取决于指令支持） |
| 内存带宽 | 204.8 GB/s |
| **FP16 Compute-to-Memory Ratio** | **137 TFLOPS / 204.8 GB/s ≈ 669 FLOP/Byte** |
| **INT8 Compute-to-Memory Ratio** | **275 TOPS / 204.8 GB/s ≈ 1343 OP/Byte** |
| L2 Cache | 4 MB |
| 共享内存 (per SM) | 128 KB |

> **注意**：Orin 的 GPU 是 **统一内存（Unified Memory）** 架构，CPU 和 GPU 共享同一块 LPDDR5 内存，没有独立显存。这意味着所有内存带宽由 CPU 和 GPU 共享，实际可用 GPU 带宽可能略低于峰值。

---

## 3. Qwen3-8B 模型架构参数

### 3.1 核心架构参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 模型参数量 | ~8.2B | 包含 embedding 和 lm_head |
| 隐藏维度 (d_model) | 4096 | |
| 中间维度 (d_ff) | 12288 | MLP FFN 中间层 (3×d_model, SwiGLU) |
| 注意力头数 (n_heads) | 32 | |
| KV 头数 (n_kv_heads) | 8 | Grouped Query Attention (GQA) |
| 每头维度 (d_head) | 128 | d_model / n_heads |
| Transformer 层数 (L) | 36 | |
| 词表大小 (V) | 151936 | |
| 最大序列长度 | 32768 | |
| 激活函数 | SwiGLU | Gate + Up 投影后 SiLU 激活 |
| 归一化 | RMSNorm | 含 q_norm 和 k_norm |
| 位置编码 | RoPE | Rotary Position Embedding |

### 3.2 每层参数量详细分解

每个 Transformer 层包含以下线性层：

| 组件 | 权重形状 | 参数量 | 说明 |
|------|---------|--------|------|
| Q 投影 (q_proj) | [4096, 4096] | 16.78M | 全注意力头 |
| K 投影 (k_proj) | [4096, 1024] | 4.19M | GQA: 8 个 KV 头 |
| V 投影 (v_proj) | [4096, 1024] | 4.19M | GQA: 8 个 KV 头 |
| 输出投影 (o_proj) | [4096, 4096] | 16.78M | |
| Gate 投影 (gate_proj) | [4096, 12288] | 50.33M | SwiGLU |
| Up 投影 (up_proj) | [4096, 12288] | 50.33M | SwiGLU |
| Down 投影 (down_proj) | [12288, 4096] | 50.33M | |
| q_norm / k_norm | 128 + 128 | 256 | 极小，可忽略 |
| RMSNorm ×2 | 4096 + 4096 | 8192 | |
| **每层合计** | | **~192.94M** | |

非 Transformer 层参数：

| 组件 | 权重形状 | 参数量 |
|------|---------|--------|
| Embedding (embed_tokens) | [151936, 4096] | 622.33M |
| LM Head (lm_head) | [4096, 151936] | 622.33M |
| 最终 RMSNorm | [4096] | 4096 |

**总参数量验证**：

$$
P_{total} = 36 \times 192.94M + 622.33M \times 2 + 4096 \approx 8.19B
$$

---

## 4. 性能预估理论框架：Roofline 模型

### 4.1 Roofline 模型原理

对于 GPU 上的任意算子，性能受两个因素约束：
- **算力上限 (π)**：GPU 的峰值计算吞吐量，单位 FLOP/s
- **带宽上限 (β)**：GPU 的峰值内存带宽，单位 Byte/s

给定一个算子的 **计算量 W** (FLOPs) 和 **数据搬运量 Q** (Bytes)，其 **算术强度** 为：

$$
I = \frac{W}{Q} \quad (\text{FLOP/Byte})
$$

算子的 **理论吞吐** 为：

$$
\text{Attainable Performance} = \min(\pi, \; I \times \beta) \quad (\text{FLOP/s})
$$

**Ridge Point**（脊点）：

$$
I^* = \frac{\pi}{\beta}
$$

- 当 $I < I^*$：**内存受限**（Memory-bound），性能由带宽决定
- 当 $I \geq I^*$：**计算受限**（Compute-bound），性能由算力决定

### 4.2 Orin AGX 64GB 的 Ridge Point

| 精度 | 算力 π | 带宽 β | Ridge Point $I^*$ |
|------|--------|--------|--------|
| FP16 | 137 TFLOPS | 204.8 GB/s | **669 FLOP/Byte** |
| INT8 | 275 TOPS | 204.8 GB/s | **1343 OP/Byte** |

---

## 5. 推理阶段分析

### 5.1 Prefill 阶段（首 Token 生成）

**特点**：输入序列的所有 token 一次性并行处理，计算密集。

设输入序列长度为 $s$，则每个 Transformer 层：

#### 5.1.1 Attention 部分

| 操作 | 计算量 (FLOPs) | 权重访存 (Bytes, FP16) |
|------|---------------|----------------------|
| Q 投影 | $2 \times s \times d_{model}^2 = 2s \times 4096^2$ | $d_{model}^2 \times 2 = 32\text{MB}$ |
| K 投影 | $2 \times s \times d_{model} \times d_{kv} = 2s \times 4096 \times 1024$ | $d_{model} \times d_{kv} \times 2 = 8\text{MB}$ |
| V 投影 | $2s \times 4096 \times 1024$ | $8\text{MB}$ |
| O 投影 | $2s \times 4096^2$ | $32\text{MB}$ |
| QK^T | $2 \times s^2 \times d_{model}$ | 无权重（来自 Q, K 激活） |
| Softmax(QK^T)×V | $2 \times s^2 \times d_{model}$ | 无权重 |

#### 5.1.2 MLP 部分

| 操作 | 计算量 (FLOPs) | 权重访存 (Bytes, FP16) |
|------|---------------|----------------------|
| Gate 投影 | $2s \times 4096 \times 12288$ | $96\text{MB}$ |
| Up 投影 | $2s \times 4096 \times 12288$ | $96\text{MB}$ |
| Down 投影 | $2s \times 12288 \times 4096$ | $96\text{MB}$ |

#### 5.1.3 单层合计（忽略 Attention Score 计算）

$$
W_{layer,prefill} = 2s \times (4096^2 + 4096 \times 1024 \times 2 + 4096^2 + 4096 \times 12288 \times 3)
$$
$$
= 2s \times (16M + 4M + 4M + 16M + 50M + 50M + 50M)
$$
$$
= 2s \times 190M \approx 380s \text{ MFLOPs}
$$

加上 Attention Score：$+4 \times s^2 \times d_{model} = 4s^2 \times 4096$

**全模型 36 层**：

$$
W_{total,prefill} = 36 \times (380s \times 10^6 + 4s^2 \times 4096)
$$

#### 5.1.4 Prefill 算术强度

由于 Prefill 时每个权重被 $s$ 个 token 复用，权重只需加载一次：

$$
I_{prefill} = \frac{W}{Q} \approx \frac{2 \times s \times P_{linear}}{P_{linear} \times \text{bytes\_per\_param} + s \times d_{model} \times \text{bytes\_per\_act}} \approx s \times \frac{2}{\text{bytes\_per\_param}} = s
$$

对于 FP16 (2 bytes/param)，$I_{prefill} ≈ s$。

| 序列长度 s | 算术强度 I | 瓶颈类型 (FP16) |
|-----------|-----------|-----------------|
| 1 | ~1 | 内存受限 |
| 64 | ~64 | 内存受限 |
| 128 | ~128 | 内存受限 |
| 512 | ~512 | 内存受限 |
| 1024 | ~1024 | **计算受限** |
| 2048 | ~2048 | 计算受限 |

> Orin FP16 Ridge Point ≈ 669，所以 s ≥ ~669 时 Prefill 变为计算受限。

### 5.2 Decode 阶段（逐 Token 生成）

**特点**：每次处理 1 个 token（s=1），典型的 **内存受限** 场景。

每步 Decode 的计算量与单 token 的 Prefill 相同，但 KV Cache 中还需要读取历史 KV 值。

#### 5.2.1 每层计算量和访存量

| 操作 | 计算量 (FLOPs) | 访存量 (Bytes, FP16) |
|------|---------------|---------------------|
| QKV + O 投影 | $2 \times (4096^2 \times 2 + 4096 \times 1024 \times 2) \approx 84M$ | 权重: ~80 MB |
| MLP (Gate+Up+Down) | $2 \times 4096 \times 12288 \times 3 \approx 302M$ | 权重: ~288 MB |
| KV Cache 读取 | $4 \times t \times d_{model}$ (Attention Score) | KV: $2 \times t \times d_{kv\_total} \times 2$ |
| **合计 (不含 KV Cache)** | **~386 MFLOPs** | **~368 MB** |

其中 $t$ 为当前已生成的序列长度，$d_{kv\_total} = n_{kv\_heads} \times d_{head} = 8 \times 128 = 1024$。

KV Cache 每层大小：$2 \times t \times 1024 \times 2 = 4096t$ Bytes

#### 5.2.2 Decode 算术强度

忽略 KV Cache 时：

$$
I_{decode} = \frac{386 \times 10^6}{368 \times 10^6} \approx 1.05 \text{ FLOP/Byte}
$$

**远低于 Ridge Point (669)**，Decode 阶段完全由内存带宽决定性能。

---

## 6. FP16 全精度推理性能预估

### 6.1 模型权重大小

$$
\text{FP16 模型大小} = 8.19B \times 2 \text{ Bytes} = 16.38 \text{ GB}
$$

### 6.2 Decode 阶段性能（目标平台：AGX Orin 64GB）

Decode 阶段为内存受限，每生成一个 token 需要读取全部模型权重（加上输入/输出激活和 KV Cache）。

**简化估算**（忽略 KV Cache，假设权重访存主导）：

$$
T_{decode} = \frac{\text{模型权重大小}}{\text{内存带宽}} = \frac{16.38 \text{ GB}}{204.8 \text{ GB/s}} \approx 80 \text{ ms/token}
$$

$$
\text{Decode 吞吐} = \frac{1}{T_{decode}} \approx 12.5 \text{ tokens/s}
$$

考虑 KV Cache 和实际带宽利用率（通常为峰值的 70%-85%），实际性能：

| 带宽利用率 | Decode 延迟 | Decode 吞吐 |
|-----------|------------|------------|
| 100% | 80 ms | 12.5 tok/s |
| 85% | 94 ms | 10.6 tok/s |
| 70% | 114 ms | 8.7 tok/s |

> **结论**：FP16 全精度下，AGX Orin 64GB 的 Decode 吞吐预计 **8-12 tokens/s**。

### 6.3 Prefill 阶段性能

Prefill 阶段，当序列长度 $s \geq 669$ 时变为计算受限：

| 序列长度 s | 总计算量 (TFLOP) | 瓶颈 | 预估延迟 | 预估吞吐 |
|-----------|-----------------|------|---------|---------|
| 64 | $36 \times 380 \times 64 \times 10^6 = 0.88T$ | 带宽 | ~80 ms※ | 800 tok/s |
| 128 | 1.75T | 带宽 | ~80 ms※ | 1600 tok/s |
| 512 | 7.0T | 带宽 | ~80 ms※ | 6400 tok/s |
| 1024 | 14.0T | 计算 | ~102 ms | 10000 tok/s |
| 2048 | 28.2T | 计算 | ~206 ms | 9942 tok/s |

> ※ 带宽受限时，延迟≈加载一次模型权重时间，不随 s 线性增长（因为权重只加载一次）。实际因 Attention Score 自注意力产生额外的 $O(s^2)$ 开销，延迟会高于此值。

### 6.4 各 Orin 平台 FP16 Decode 性能对比

| 平台 | 内存 | 能否加载 FP16 模型 | 带宽 | 预估 Decode 吞吐 |
|------|------|------------------|------|-----------------|
| AGX Orin 64GB | 64 GB | ✅ | 204.8 GB/s | ~10 tok/s |
| AGX Orin 32GB | 32 GB | ✅（但 RAM 紧张） | 204.8 GB/s | ~10 tok/s |
| Orin NX 16GB | 16 GB | ❌（模型 16.4GB > 16GB） | 102.4 GB/s | N/A |
| Orin NX 8GB | 8 GB | ❌ | 102.4 GB/s | N/A |
| Orin Nano 8GB | 8 GB | ❌ | 68 GB/s | N/A |
| Orin Nano 4GB | 4 GB | ❌ | 68 GB/s | N/A |

---

## 7. 量化推理性能预估

### 7.1 各量化方案的权重大小

| 量化方案 | 每参数 Bits | 有效字节/参数 | 模型权重大小 | 适用 Orin 平台 |
|---------|-----------|-------------|------------|--------------|
| FP16 | 16 | 2.0 | 16.38 GB | AGX 64/32 GB |
| FP8 | 8 | 1.0 | 8.19 GB | AGX 64/32 GB, NX 16 GB |
| INT8 (SmoothQuant) | 8 | 1.0 | 8.19 GB | AGX 64/32 GB, NX 16 GB |
| INT4 (AWQ, W4A16) | 4 + meta | ~0.56※ | ~4.59 GB | AGX 全系列, NX 全系列, Nano 8GB |

> ※ AWQ W4A16 中，每 128 个权重附带一个 FP16 scale 和 FP16 zero_point (共 4 Bytes)，等效 `4 + 32/128 = 4.25 bits/param`，约 0.53 Bytes/param。加上一部分无法量化的 Embedding/LM Head 层，实际约 0.56 Bytes/param（此处为估计值）。

### 7.2 AWQ INT4 (W4A16) Decode 性能预估

AWQ 是 **权重量化** 方案，Decode 阶段瓶颈是读取权重。量化为 INT4 后权重大小降低约 3.6×。

**AGX Orin 64GB**：

$$
T_{decode,awq} = \frac{4.59 \text{ GB}}{204.8 \text{ GB/s}} \approx 22.4 \text{ ms/token}
$$

$$
\text{Decode 吞吐}_{awq} \approx 44.6 \text{ tokens/s (理论峰值)}
$$

考虑反量化开销、实际带宽利用率（75%-85%）以及其他 overhead：

| 假设带宽利用率 | Decode 延迟 | Decode 吞吐 |
|-------------|------------|------------|
| 100% | 22.4 ms | 44.6 tok/s |
| 85% | 26.4 ms | 37.9 tok/s |
| 75% | 29.9 ms | 33.4 tok/s |
| 65%（保守） | 34.5 ms | 29.0 tok/s |

> **参考数据**：在 LLMQRT 项目的 GPU 实测中，Qwen3-8B AWQ 在桌面 GPU 上的 Decode 吞吐为 ~23.8 tok/s（含 PyTorch 调度等额外开销）。Orin 上由于统一内存和较低带宽，实际性能将受影响。

**各 Orin 平台 AWQ Decode 性能对比**：

| 平台 | 带宽 | 理论 Decode 延迟 | 预估实际吞吐 (75% 利用率) |
|------|------|-----------------|------------------------|
| AGX Orin 64GB | 204.8 GB/s | 22.4 ms | **~33 tok/s** |
| AGX Orin 32GB | 204.8 GB/s | 22.4 ms | **~33 tok/s** |
| Orin NX 16GB | 102.4 GB/s | 44.8 ms | **~17 tok/s** |
| Orin NX 8GB | 102.4 GB/s | 44.8 ms | **~17 tok/s** |
| Orin Nano 8GB | 68 GB/s | 67.5 ms | **~11 tok/s** |

### 7.3 INT8 (SmoothQuant / FP8) Decode 性能预估

**AGX Orin 64GB**：

$$
T_{decode,int8} = \frac{8.19 \text{ GB}}{204.8 \text{ GB/s}} \approx 40 \text{ ms/token}
$$

$$
\text{Decode 吞吐}_{int8} \approx 25.0 \text{ tokens/s (理论峰值)}
$$

实际预估（75% 带宽利用率）：

| 平台 | 带宽 | 理论 Decode 延迟 | 预估实际吞吐 |
|------|------|-----------------|------------|
| AGX Orin 64GB | 204.8 GB/s | 40 ms | **~19 tok/s** |
| Orin NX 16GB | 102.4 GB/s | 80 ms | **~9 tok/s** |
| Orin Nano 8GB | 68 GB/s | 120 ms | **~6 tok/s** |

### 7.4 各方案 Prefill 性能对比 (AGX Orin 64GB)

Prefill 阶段当输入足够长时变为计算受限。量化方案的优势体现在两方面：
1. **降低权重加载量**：带宽受限阶段更快
2. **提供更高算力**：INT8 算力是 FP16 的 2×

| 量化方案 | s=128 Prefill 延迟 | s=512 Prefill 延迟 | s=1024 Prefill 延迟 | s=2048 Prefill 延迟 |
|---------|-------------------|-------------------|--------------------|--------------------|
| FP16 | ~80 ms | ~80 ms | ~102 ms | ~206 ms |
| INT8/FP8 | ~40 ms | ~51 ms | ~51 ms | ~103 ms |
| AWQ W4A16※ | ~22 ms | ~56 ms | ~102 ms | ~206 ms |

> ※ AWQ 中权重为 INT4 但计算仍在 FP16 进行（反量化后 GEMM），因此计算受限阶段性能与 FP16 相同，优势仅在带宽受限阶段。

---

## 8. 显存占用分析

### 8.1 各组件内存占用

推理时的总内存占用 = 模型权重 + KV Cache + 激活值 + 框架 Overhead

#### 8.1.1 KV Cache 大小

每层 KV Cache 大小（FP16）：

$$
\text{KV\_per\_layer} = 2 \times n_{kv\_heads} \times d_{head} \times \text{seq\_len} \times 2 \text{ bytes}
$$
$$
= 2 \times 8 \times 128 \times s \times 2 = 4096s \text{ bytes/layer}
$$

全模型 36 层：

$$
\text{KV\_total} = 36 \times 4096s = 147456s \text{ bytes} = 0.14s \text{ MB}
$$

| 序列长度 s | FP16 KV Cache | INT8 KV Cache | FP8 KV Cache |
|-----------|--------------|--------------|-------------|
| 256 | 36 MB | 18 MB | 18 MB |
| 512 | 72 MB | 36 MB | 36 MB |
| 1024 | 144 MB | 72 MB | 72 MB |
| 2048 | 288 MB | 144 MB | 144 MB |
| 4096 | 576 MB | 288 MB | 288 MB |
| 8192 | 1.15 GB | 576 MB | 576 MB |
| 32768 | 4.6 GB | 2.3 GB | 2.3 GB |

#### 8.1.2 激活值内存

峰值激活内存（每层，FP16）：

$$
\text{Act\_peak} \approx s \times d_{model} \times 2 \times (\text{num\_intermediates}) \approx s \times 4096 \times 2 \times 6 \approx 49152s \text{ bytes}
$$

| 序列长度 s | 峰值激活内存 |
|-----------|------------|
| 128 | ~ 6 MB |
| 512 | ~ 24 MB |
| 2048 | ~ 96 MB |

#### 8.1.3 综合内存占用

| 场景 | 权重 | KV Cache (s=2048) | 激活 | 框架 | **总计** |
|------|------|------------------|------|------|---------|
| FP16 | 16.38 GB | 288 MB | ~96 MB | ~2 GB | **~18.8 GB** |
| INT8/FP8 | 8.19 GB | 144 MB | ~96 MB | ~2 GB | **~10.4 GB** |
| AWQ W4A16 | 4.59 GB | 288 MB | ~96 MB | ~2 GB | **~7.0 GB** |
| AWQ + FP8 KV | 4.59 GB | 144 MB | ~96 MB | ~2 GB | **~6.8 GB** |

### 8.2 各平台可行性总结

| 平台 | 内存 | FP16 | INT8/FP8 | AWQ W4A16 |
|------|------|------|---------|-----------|
| AGX Orin 64GB | 64 GB | ✅ 充裕 | ✅ 充裕 | ✅ 充裕 |
| AGX Orin 32GB | 32 GB | ✅ 紧张 | ✅ 充裕 | ✅ 充裕 |
| Orin NX 16GB | 16 GB | ❌ 不足 | ✅ 可行 | ✅ 充裕 |
| Orin NX 8GB | 8 GB | ❌ 不足 | ❌ 紧张 | ✅ 可行 |
| Orin Nano 8GB | 8 GB | ❌ 不足 | ❌ 紧张 | ✅ 可行 |
| Orin Nano 4GB | 4 GB | ❌ 不足 | ❌ 不足 | ❌ 不足 |

---

## 9. 关键瓶颈分析

### 9.1 Decode 阶段瓶颈

```
┌────────────────────────────────────────────────────────────┐
│                    Decode 性能瓶颈分析                       │
│                                                            │
│  算术强度 I ≈ 1 FLOP/Byte  << Ridge Point ≈ 669           │
│                                                            │
│  ──────────────────────────────────────────────────         │
│  |  内 存 带 宽 受 限 区 域  |计算受限|                       │
│  |◄─────────────────────────►|                             │
│  |          ▲ Decode (I≈1)   |        |                    │
│  |          │                |        |                    │
│  ──────────────────────────────────────────────────         │
│  1       10      100     669   1000                        │
│                           ↑                                │
│                      Ridge Point                           │
│                                                            │
│  结论：Decode 完全受内存带宽约束                               │
│  优化方向 → 降低权重大小（量化）、提高带宽利用率                  │
└────────────────────────────────────────────────────────────┘
```

### 9.2 Prefill 阶段瓶颈转换

| 输入长度 | 主要瓶颈 | 优化方向 |
|---------|---------|---------|
| s < 669 | 内存带宽 | 量化降低权重大小 |
| s ≥ 669 | 计算 | 使用更高算力精度（INT8 2× FP16） |
| s >> 2000 | 计算 + Attention $O(s^2)$ | Flash Attention、稀疏注意力 |

### 9.3 Orin 特有挑战

1. **统一内存竞争**：CPU 和 GPU 共享 LPDDR5 带宽，OS、数据预处理等会竞争带宽
2. **功耗模式影响**：不同功耗模式下算力差异巨大（MAXN vs 15W 差距 3-4×）
3. **Tensor Core 利用率**：Orin 的 Ampere Tensor Core 需要特定矩阵尺寸对齐，小矩阵（Decode 阶段 M=1）利用率低
4. **L2 Cache 有限**：仅 4MB，对大权重矩阵命中率低
5. **INT4 指令支持**：Orin (SM 8.7) 对 INT4 Tensor Core 指令支持有限，AWQ 的 INT4 反量化在 CUDA Core 上进行，实际吞吐受反量化 overhead 影响

---

## 10. 优化建议

### 10.1 量化策略选择

| 场景 | 推荐方案 | 原因 |
|------|---------|------|
| AGX Orin 64GB，精度敏感 | FP8 动态量化 | 精度损失极小，带宽减半 |
| AGX Orin 64GB，追求速度 | AWQ W4A16 | 带宽降低 3.6×，精度损失 <0.5% |
| Orin NX 16GB | AWQ W4A16 | 仅 AWQ 能舒适放入 16GB |
| Orin NX/Nano 8GB | AWQ W4A16 + FP8 KV Cache | 极限压缩，勉强可用 |

### 10.2 性能优化方向

#### 10.2.1 带宽优化（Decode 阶段核心）

1. **权重量化**：AWQ W4A16 是降低带宽需求最有效的手段
2. **KV Cache 量化**：使用 FP8 KV Cache 降低长序列时的额外带宽消耗
3. **权重预取与流水线**：利用 CUDA Stream 重叠权重加载与计算
4. **算子融合**：将 RMSNorm + Linear、Activation + Linear 等融合，减少中间数据读写
5. **GEMV 优化**：Decode 阶段的 Matrix-Vector 乘法需要专门优化（如 LLMQRT 中的 `gemv_kernel_g128`）

#### 10.2.2 计算优化（Prefill 阶段）

1. **Flash Attention**：降低 Attention 的 $O(s^2)$ 内存开销和访存量
2. **INT8 计算**：利用 Orin 275 TOPS INT8 算力，Prefill 吞吐翻倍
3. **Tensor Core 对齐**：确保矩阵维度对齐 m16n8k16（Ampere mma 指令要求）

#### 10.2.3 系统级优化

1. **功耗模式**：设置 MAXN 模式以获取最大性能
2. **CPU 侧优化**：最小化 tokenizer、采样等 CPU 开销
3. **DLA 协同**：探索将部分算子卸载到 Orin 的 DLA（深度学习加速器）
4. **TensorRT 集成**：利用 NVIDIA TensorRT 的图优化和 kernel auto-tuning
5. **内存管理**：使用 CUDA Graph 减少 kernel launch overhead

### 10.3 推荐部署配置

#### AGX Orin 64GB 最优配置

```yaml
模型: Qwen3-8B
量化方案: AWQ W4A16
KV Cache: FP16 (内存充裕, 保留精度)
功耗模式: MAXN (60W)
最大序列长度: 8192
批大小: 1
预期 Decode 吞吐: 28-35 tokens/s
预期 Prefill 吞吐 (s=512): ~5000 tokens/s
```

#### Orin NX 16GB 最优配置

```yaml
模型: Qwen3-8B
量化方案: AWQ W4A16
KV Cache: FP8 量化
功耗模式: 25W
最大序列长度: 4096
批大小: 1
预期 Decode 吞吐: 13-17 tokens/s
预期 Prefill 吞吐 (s=512): ~2500 tokens/s
```

#### Orin Nano 8GB 最低可行配置

```yaml
模型: Qwen3-8B
量化方案: AWQ W4A16
KV Cache: FP8 量化
功耗模式: 15W
最大序列长度: 2048
批大小: 1
预期 Decode 吞吐: 8-11 tokens/s
预期 Prefill 吞吐 (s=512): ~1500 tokens/s
内存余量: ~1 GB (紧张)
```

---

## 11. 附录：计算公式推导

### A.1 通用 Decode Token 延迟公式

$$
T_{token} = \frac{P_{model} \times B_{param}}{BW_{mem} \times \eta_{bw}} + \frac{2 \times L \times 2 \times n_{kv} \times d_h \times t \times B_{kv}}{BW_{mem} \times \eta_{bw}} + T_{overhead}
$$

其中：
- $P_{model}$: 模型参数量（不含 Embedding/LM Head 时约 6.95B）
- $B_{param}$: 每参数字节数（FP16=2, INT8=1, INT4≈0.56）
- $BW_{mem}$: 内存带宽
- $\eta_{bw}$: 带宽利用率（通常 0.7-0.85）
- $L$: Transformer 层数 (36)
- $n_{kv}$: KV 头数 (8)
- $d_h$: 每头维度 (128)
- $t$: 当前序列位置
- $B_{kv}$: KV Cache 每元素字节数
- $T_{overhead}$: CPU 调度、采样等开销

### A.2 Prefill 延迟公式

**带宽受限阶段** ($s < I^*$)：

$$
T_{prefill} \approx \frac{P_{model} \times B_{param}}{BW_{mem} \times \eta_{bw}}
$$

**计算受限阶段** ($s \geq I^*$)：

$$
T_{prefill} \approx \frac{2 \times s \times P_{linear}}{FLOPS_{peak} \times \eta_{compute}} + \frac{4 \times s^2 \times d_{model} \times L}{FLOPS_{peak} \times \eta_{compute}}
$$

其中：
- $P_{linear}$: 线性层参数量（每层约 190M × 36 层）
- $FLOPS_{peak}$: 峰值算力
- $\eta_{compute}$: 计算利用率（通常 0.5-0.7）

### A.3 端到端推理时间

$$
T_{total} = T_{prefill}(s_{input}) + \sum_{i=1}^{N_{output}} T_{decode}(s_{input} + i)
$$

**示例**：AGX Orin 64GB，AWQ W4A16，输入 512 tokens，输出 256 tokens：

$$
T_{total} \approx 22ms + 256 \times 30ms \approx 7.7s
$$

等效吞吐：$(512 + 256) / 7.7 \approx 100 \text{ tokens/s}$

### A.4 性能预估汇总表（AGX Orin 64GB, MAXN模式）

| 指标 | FP16 | FP8 | INT8 (SQ) | AWQ W4A16 |
|------|------|-----|-----------|-----------|
| 模型大小 | 16.38 GB | 8.19 GB | 8.19 GB | ~4.59 GB |
| Decode 延迟 (理论) | 80 ms | 40 ms | 40 ms | 22 ms |
| Decode 吞吐 (理论) | 12.5 tok/s | 25 tok/s | 25 tok/s | 45 tok/s |
| **Decode 吞吐 (预估)** | **8-10 tok/s** | **17-20 tok/s** | **17-20 tok/s** | **28-35 tok/s** |
| Prefill 吞吐 s=512 | ~6400 tok/s | ~12800 tok/s | ~12800 tok/s | ~6400 tok/s |
| KV Cache (s=2048) | 288 MB | 144 MB | 288 MB | 288 MB |
| 总内存占用 (s=2048) | ~18.8 GB | ~10.4 GB | ~10.4 GB | ~7.0 GB |
| 精度损失 (PPL) | 基准 | <0.1% | ~1-3% | ~0.45% |
| NX 16GB | ❌ | ✅ | ✅ | ✅ |
| Nano 8GB | ❌ | ❌ | ❌ | ✅ |

---

## 参考资料

1. NVIDIA Jetson AGX Orin Technical Specifications
2. Qwen3 Technical Report (通义千问团队)
3. Williams, S., Waterman, A., & Patterson, D. (2009). "Roofline: An Insightful Visual Performance Model for Multicore Architectures"
4. Lin, J., et al. (2024). "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"
5. Xiao, G., et al. (2023). "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models"
6. LLMQRT 项目性能测试数据（Qwen3-8B AWQ profiling results）
7. NVIDIA Jetson Orin Series Module Data Sheet (DS-10653-001)

---

> 本报告基于理论分析和公开硬件规格进行预估，实际性能受功耗模式、系统负载、软件栈优化程度、CUDA kernel 实现质量等因素影响，建议以实际测试数据为准。
