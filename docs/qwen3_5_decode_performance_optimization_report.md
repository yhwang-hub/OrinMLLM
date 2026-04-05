# Qwen3.5-9B Decode 性能优化分析报告

## 1. 摘要

本报告分析了 Qwen3.5-9B 混合架构模型在 NVIDIA Jetson Orin AGX 上进行 FP16 decode 推理时的性能瓶颈，并实现了针对性优化。

### 关键结论

| 指标 | 优化前 | 优化后 | 变化 |
|------|--------|--------|------|
| Decode 吞吐 (无 Graph) | 8.79 tok/s | 8.97 tok/s | +2.0% |
| Decode 吞吐 (CUDA Graph) | 9.02 tok/s | 9.10 tok/s | +0.9% |
| 每 token 延迟 | 110.83 ms | 109.93 ms | -0.8% |
| 理论带宽利用率 | ~80% | ~81% | +1% |

**核心发现**：该模型在 FP16 decode 场景下已接近 LPDDR5 带宽极限（170 GB/s 峰值），当前实际利用率约 81%，FFN 层单独达到 94.1% 利用率。进一步的显著提升需要量化（INT4/INT8）或架构级优化（投机解码）。

## 2. 硬件与模型概述

### 2.1 硬件平台
- **GPU**: NVIDIA Jetson Orin AGX (sm_87)
- **内存**: LPDDR5，峰值带宽 170 GB/s（GPU 与 CPU 共享）
- **CUDA Cores**: 2048
- **SMs**: 16
- **L2 Cache**: 4 MB

### 2.2 模型架构
- **Qwen3.5-9B**: 混合 Vision-Language 模型
- **总参数量**: ~9.3B（FP16 权重 ~17.5 GB）
- **32 层 LLM**:
  - 8 层 Full Attention（层索引 3,7,11,15,19,23,27,31），含 output gate
  - 24 层 GDN（Gated Delta Net）线性注意力
- **维度**: hidden_size=4096, intermediate=12288, head_dim=256
- **全注意力**: 16 Q heads, 4 KV heads, Q+gate 维度=8192
- **线性注意力**: 16 K heads (dim=128), 32 V heads (dim=128), conv_dim=8192

## 3. 性能分析

### 3.1 Decode 步骤时间分解

通过 CUDA Event 计时得到单步 decode 的实际分解（seq_len≈980）：

| 操作类别 | 时间 (ms) | 占比 | GEMV 权重读取量 | 理论带宽时间 | 带宽利用率 |
|----------|-----------|------|-----------------|-------------|-----------|
| FFN (32 layers) | 60.40 | 54.1% | 9.67 GB | 56.9 ms | **94.1%** |
| GDN Linear Attn (24 layers) | 30.30 | 27.1% | 3.23 GB + non-GEMV | ~19 ms (GEMV) | — |
| LM Head | 12.24 | 11.0% | 2.03 GB | 11.9 ms | **97.2%** |
| Full Attention (8 layers) | 8.81 | 7.9% | 0.94 GB + flash attn | ~5.5 ms (GEMV) | — |
| **总计** | **111.74** | **100%** | **~15.9 GB** | **~93.3 ms** | **83.5%** |

### 3.2 分析

#### FFN 层 (54.1%)
- 为最大开销来源，每层 3 个 GEMV（gate: [12288,4096], up: [12288,4096], down: [4096,12288]）
- 已使用 **Fused Gate+Up+SwiGLU** 内核，将 gate 和 up 投影合并为单次内核启动
- 带宽利用率 94.1%，接近理论极限，几乎无优化空间

#### GDN 线性注意力层 (27.1%)
- GEMV 部分：QKV (8192×4096) + Z (4096×4096) + tiny A,B = ~134.8 MB/层
- Non-GEMV 部分：Conv1D + SiLU, L2 Norm, Gate 计算, Delta Net 状态更新, Gated RMSNorm
- 非 GEMV 操作约贡献 30.3 - 19 = **11.3 ms**（GDN 状态更新涉及 24×2MB FP32 读写改）

#### LM Head (11.0%)
- 单次大型 GEMV [248320, 4096]，读取 2.03 GB 权重
- 实测 12.24ms vs 理论 11.9ms = 97.2% 利用率，已近最优

#### Full Attention (7.9%)
- GEMV：Q+gate (8192×4096) + K (1024×4096) + V (1024×4096) + O (4096×4096) = ~117.4 MB/层
- Flash Attention Decode：扫描 ~1000 KV pairs，额外 ~4 MB/层
- 占比最小，优化空间有限

### 3.3 理论性能极限

FP16 decode 的理论极限由内存带宽决定：
- 总权重读取量：~15.9 GB/token
- 理论最小时间：15.9 GB ÷ 170 GB/s = **93.5 ms/token**
- 理论最大吞吐：**10.7 tok/s**
- 当前实际：109.93 ms/token → **81% 带宽利用率**

差距来源：
- GDN 非 GEMV 操作：~11.3 ms
- Flash Attention 扫描：~3.3 ms
- 内核调度/同步开销：~2.0 ms

## 4. 已实现的优化

### 4.1 Fused QKV GEMV 内核（Full Attention）

**问题**：Full Attention 层的 Q+gate、K、V 投影分别使用 3 次独立 GEMV 内核启动。

**方案**：实现 block-dispatch 融合内核 `fused_fp16_qkv_gemv_kernel`，在单次 kernel launch 中计算所有三个投影。

```cuda
// Block dispatch: Q blocks → K blocks → V blocks
if (blockIdx.x < q_blocks) {
    // compute Q projection row
} else if (blockIdx.x < q_blocks + k_blocks) {
    // compute K projection row
} else {
    // compute V projection row
}
```

**效果**：
- 减少 8 层 × 2 次 = 16 次 kernel launch
- 输入向量 L2 cache 复用（8KB 仅读取一次）
- 无 Graph 时节省 ~16 × 5µs ≈ 0.08 ms

### 4.2 Fused GDN Projection GEMV（Linear Attention）

**问题**：GDN 层的 QKV 和 Z 投影分别使用 2 次独立 GEMV。

**方案**：复用 block-dispatch 融合框架，将 QKV (8192 行) 和 Z (4096 行) 合并为单次 kernel launch（2-way dispatch，无冗余 V blocks）。

```cpp
// 2-way: Q-blocks handle QKV, K-blocks handle Z
int total = (qkv_dim + WPB - 1) / WPB + (z_dim + WPB - 1) / WPB;
```

**效果**：
- 减少 24 层 × 1 次 = 24 次 kernel launch
- 无 Graph 时节省 ~24 × 5µs ≈ 0.12 ms

### 4.3 Layer Wrapper 模式（架构改进）

所有新的 fused kernel 都遵循 Layer→forward() 封装模式：
- `FusedQKVGemvLayer`：封装 fused_fp16_qkv_gemv_cu
- `FusedGDNProjGemvLayer`：封装 fused_fp16_gdn_proj_gemv_cu

### 4.4 Per-Category Profiling

添加了可通过环境变量 `Q35_PROFILE_DECODE` 启用的 CUDA Event 计时功能，可精确分析 decode 步骤中各类操作的耗时分布。

```bash
Q35_PROFILE_DECODE=1 ./qwen3_5_infer ...
```

## 5. 优化效果有限的原因

### 5.1 CUDA Graph 已消除 Launch 开销

当 CUDA Graph 开启时，所有 kernel 的启动开销已被消除（graph replay 无逐 kernel API 调用）。因此 kernel 融合的 launch 优化在 graph 模式下基本无效，仅保留 input vector cache reuse 的微小收益。

### 5.2 带宽利用率已接近极限

FFN 层 94.1%、LM Head 97.2% 的带宽利用率表明 GEMV kernel 本身效率极高。瓶颈不在计算kernel，而在内存带宽。

### 5.3 Non-GEMV 操作不可忽略

GDN 层的非 GEMV 操作（Conv1D, L2 Norm, GDN 状态更新等）约占 10% 总时间。这些操作计算密度低，由多个小 kernel 组成，优化空间有限。

## 6. 进一步优化方向

### 6.1 高收益优化（推荐）

| 方案 | 预估提升 | 复杂度 | 说明 |
|------|---------|--------|------|
| **INT4 AWQ 量化** | 3-4× | 高 | 权重量化为 4-bit，GEMV 读取量降低 4×，理论可达 35+ tok/s |
| **INT8 SmoothQuant** | 1.8-2× | 中 | 8-bit 权重+激活量化，理论可达 18+ tok/s |
| **投机解码 (EAGLE3)** | 2-3× | 高 | 小模型草案 + 大模型验证，batch 验证提高带宽利用 |

### 6.2 中等收益优化

| 方案 | 预估提升 | 说明 |
|------|---------|------|
| GDN Conv1D + L2 Norm + Gate 融合 | +2-3% | 将 GDN 的多个小 kernel 合并 |
| Flash Attention 优化 | +1-2% | 针对长序列优化 KV 扫描 |
| AsyncCopy + Double Buffering | +2-5% | GEMV 权重预取与计算重叠 |

### 6.3 低收益优化（已接近极限）

| 方案 | 说明 |
|------|------|
| 进一步 kernel 融合 | 已在 Graph 模式下无效 |
| LM Head 词表裁剪 | 仅对 top-k 采样有效，需修改采样逻辑 |
| CPU-GPU 异步重叠 | embedding lookup 已异步 |

## 7. 文件变更清单

### 新增文件
- `kuiper/include/op/gdn_layers.h`: 16 个 GDN Layer wrapper 类（含 2 个 Fused 层）
- `kuiper/source/op/gdn_layers.cpp`: 实现

### 修改文件
- `kuiper/source/op/kernels/cuda/gdn_kernel.cu`: 添加 `fused_fp16_qkv_gemv_kernel`, `fused_fp16_qkv_gemv_cu`, `fused_fp16_gdn_proj_gemv_cu`
- `kuiper/source/op/kernels/cuda/gdn_kernel.cuh`: 添加 fused kernel 声明
- `kuiper/include/model/qwen3_5.h`: 添加 fused layer 成员指针
- `kuiper/source/model/qwen3_5.cpp`:
  - full_attn_decode: 3 GEMV → 1 fused GEMV
  - full_attn_decode_graph: 3 GEMV → 1 fused GEMV
  - linear_attn_decode: QKV+Z 独立 → fused 2-way dispatch
  - create_q35_nonparam_layers: 实例化 fused layers
  - decode_step_optimized: 添加 per-category profiling

## 8. 结论

Qwen3.5-9B 在 Orin AGX 上 FP16 decode 已达到理论带宽极限的 ~81%，其中 FFN（最大开销来源，54%）已达 94% 利用率。本次 fused GEMV 优化带来约 2% 的非 Graph 模式提升，但在 CUDA Graph 模式下提升有限（<1%），因为 Graph 已消除了 kernel launch 开销。

**要突破当前 9.1 tok/s 的瓶颈，最有效的路径是 INT4/INT8 量化**，可将 decode 吞吐提升至 20-35 tok/s。
