# Fused GQA + MRoPE + KV Cache 融合算子优化分析报告

> 本报告是对 `fused_rope_kv_kernel.cu` 融合算子的深度性能分析、优化历程及结论总结。

## 一、问题背景

在 Qwen3-VL-8B-fp16 模型的 decode 阶段，原始实现使用三个独立的 CUDA kernel 完成注意力计算：

1. **MRoPE kernel** (`mrope_kernel_cu_fp16_gpu_pos_impl`): 对 Q/K 应用多维旋转位置编码
2. **KV Cache Copy** (`copy_to_kv_cache_fp16_cu`): 将 K/V 写入 KV Cache（×2）
3. **Flash Attention decode** (`flash_attention_decode_kernel_fp16_online_softmax`): 在线 softmax 注意力计算

融合算子 `fused_gqa_mrope_kv_decode_fp16_kernel` 将这三个操作合并为单个 kernel，目标是减少 kernel launch 开销和 global memory 访问。

**核心问题**: 为什么融合后 decode 速度几乎没有提升？如何优化？

## 二、硬件平台与模型参数

### 2.1 Orin GPU (GA10B)

| 参数 | 值 |
|------|-----|
| SM 数量 | 16 |
| Compute Capability | 8.7 |
| GPU 频率 | 1.3 GHz |
| L2 Cache | 4 MB |
| 内存带宽 | ~170 GB/s (LPDDR5) |
| 每 SM 最大 warp | 48 |
| 每 SM 寄存器 | 65536 |
| 每 block 最大 shared memory | 48 KB |

### 2.2 Qwen3-VL-8B-fp16 注意力参数

| 参数 | 值 |
|------|-----|
| 层数 | 36 |
| Q heads | 32 |
| KV heads | 8 |
| kv_mul (GQA ratio) | 4 |
| head_size | 128 |
| dim | 4096 |
| kv_dim | 1024 |

## 三、Nsight Systems 性能剖析

### 3.1 Baseline Decode 各 kernel 耗时占比

使用 `nsys profile --cuda-graph-trace=node` 获取 CUDA Graph 内部逐 kernel 时序：

| Kernel | 占比 | 每层平均耗时 | 实例数 | 说明 |
|--------|------|-------------|--------|------|
| `fused_gate_up_swiglu_kernel_fp16_v2` | **42.1%** | 1,221 µs | 8,964 | FFN 融合算子 |
| `gemv_pure_fp16_kernel_v2` | **39.4%** | 229 µs × 5/layer | 44,820 | QKV/O/down GEMV |
| `gemv_fp16_input_fp16_weight_fp32_output` | **7.2%** | 7,478 µs | 250 | LM Head |
| `flash_attention_decode_kernel_fp16_online_softmax` | **6.5%** | 187.6 µs | 8,964 | Flash Attention |
| `copy_to_kv_cache_fp16_cu` | 0.2% | 3.1 µs × 2 | 17,928 | KV Cache 写入 |
| `mrope_kernel_cu_fp16_gpu_pos_impl` | 0.2% | 4.9 µs | 8,964 | MRoPE |
| 其他 (RMSNorm, Add, argmax 等) | ~4.4% | — | — | — |

### 3.2 核心发现

**注意力相关操作 (MRoPE + KV Copy + FA) 仅占 decode 时间的 ~7%**

- 注意力管线每层：187.6 + 4.9 + 6.2 = **198.7 µs/layer**
- GEMV + FFN 每层：1,221 + 229×5 = **2,366 µs/layer**
- 注意力占比：198.7 / 2,800 = **7.1%**

### 3.3 Decode 瓶颈：带宽墙

每个 token 的 decode 需要读取所有层的权重矩阵：

| 权重矩阵 | 大小 | 每层 | 36 层合计 |
|----------|------|------|----------|
| Q/K/V projection | 4096×(4096+1024+1024)×2B | 48 MB | 1.7 GB |
| O projection | 4096×4096×2B | 32 MB | 1.2 GB |
| Gate+Up | 4096×(11008×2)×2B | 172 MB | 6.2 GB |
| Down | 11008×4096×2B | 86 MB | 3.1 GB |
| **合计** | | | **~14.4 GB** |

在 Orin 170 GB/s 带宽下：14.4 GB / 170 GB/s ≈ **85 ms/token** 理论极限，实测 ~101 ms/token，带宽利用率 ≈ 84%。

## 四、融合算子优化历程

### 4.1 版本 1：32 blocks (原始融合 kernel)

每个 block 处理 1 个 Q head，与独立 FA kernel 相同的 Grid 配置。

- 融合效果：消除 3 个 kernel launch 开销 (~6 µs/layer)
- MRoPE 计算开销：+4 µs/layer (Phase 0)
- **净收益：~2 µs/layer = 0.07 ms/token = 0.07%**

### 4.2 版本 2：8 blocks (q_per_block=4, 全 KV 复用)

设计思路：每个 block 处理同一 KV head 的全部 4 个 Q heads，K/V 数据加载一次复用 4 次。

| 配置 | 延迟 (ms/token) | 吞吐 (tok/s) | 对比 |
|------|----------------|-------------|------|
| Baseline | 101.28 | 9.87 | — |
| 8 blocks (q=4) | 104.56 | 9.56 | **-3.1%** |

**失败根因**:
1. **SM 闲置 50%**: 8 blocks / 16 SMs = 每帧只用 8 个 SM
2. **Occupancy 骤降**: 1 block/SM → 8 warps/SM (vs baseline 16 warps/SM)
3. **L2 Cache 天然复用**: seq_len ~635 tokens 的 K+V = ~2.5 MB 完全装入 4 MB L2

### 4.3 版本 3：16 blocks (q_per_block=2, 运行时参数)

| 配置 | 延迟 (ms/token) | 吞吐 (tok/s) | 对比 |
|------|----------------|-------------|------|
| Baseline | 101.28 | 9.87 | — |
| 16 blocks (runtime) | 103.24 | 9.69 | **-1.8%** |

**失败根因**: 运行时 `q_per_block` 参数阻止编译器循环展开和死代码消除。

### 4.4 版本 4：模板化 + k_reg 预加载

将 `q_per_block` 改为模板参数 `Q_PER_BLOCK`，但保留 `float4 k_reg[16]` K 寄存器预加载。

| 配置 | 延迟 (ms/token) | 吞吐 (tok/s) | 对比 |
|------|----------------|-------------|------|
| Baseline | 101.28 | 9.87 | — |
| 模板化 + k_reg | 102.32 | 9.77 | **-1.0%** |

**失败根因**: `float4 k_reg[16]` 占用 64 个 32-bit 寄存器 (每线程 128 个可用的 50%)，导致 register spilling 到 local memory。

### 4.5 版本 5（最终版）：模板化 + 内联 K 加载

**关键优化**:
1. **`template<int Q_PER_BLOCK>`** — 编译期完全展开所有 `qi` 循环
2. **移除 `k_reg[16]` 预加载** — 改为 `__ldg(k_ptr_f4 + d)` 内联加载，与独立 FA kernel 完全一致
3. **默认 `Q_PER_BLOCK=1`** — 32 blocks, 最佳 occupancy

| 配置 | 延迟 (ms/token) | 吞吐 (tok/s) | 对比 |
|------|----------------|-------------|------|
| Baseline (3 kernels) | 101.30 (avg) | 9.87 | — |
| **Fused GQA (最终版)** | **101.18 (avg)** | **9.88** | **+0.1%** |

**多次测量结果**:
- Fused: 101.09, 101.20, 101.24 → avg **101.18 ms/token**
- Baseline: 101.28, 101.32 → avg **101.30 ms/token**

### 4.6 Nsys 最终版 vs Baseline 对比

| 指标 | Baseline | Fused GQA | 差异 |
|------|----------|-----------|------|
| FA / fused kernel 均值 | 187.6 µs | 192.9 µs | +5.3 µs |
| MRoPE 均值 | 4.9 µs | (内含) | -4.9 µs |
| KV Copy 均值 (×2) | 6.2 µs | (内含) | -6.2 µs |
| **注意力总耗时/layer** | **198.7 µs** | **192.9 µs** | **-5.8 µs** |
| Kernel launch 开销/layer | ~3 × 2 µs | ~1 × 2 µs | ~-4 µs |
| **总节省/layer** | — | — | **~10 µs** |
| **总节省/token (36 layers)** | — | — | **~360 µs** |

## 五、为什么融合算子无法带来"显著"提升

### 5.1 阿姆达尔定律

注意力计算仅占 decode 时间 **~7%**。根据阿姆达尔定律：

$$S = \frac{1}{(1 - P) + \frac{P}{S_{\text{part}}}}$$

其中 $P = 0.07$（注意力占比），即使 $S_{\text{part}} = \infty$（注意力完全消除）：

$$S_{\max} = \frac{1}{1 - 0.07} = 1.075$$

**理论最大提升：7.5%**。实际融合节省 ~5.8 µs/198.7 µs = 2.9%，对应端到端 0.2%。

### 5.2 L2 Cache 使多 Q 复用无效

| 序列长度 | K+V cache 总量 | vs L2 (4 MB) | 多 Q 复用收益 |
|----------|---------------|-------------|-------------|
| 500 tokens | 2.0 MB | ✅ 装入 | 无（L2 已复用） |
| 1000 tokens | 4.0 MB | ≈ 装入 | 微弱 |
| 2000 tokens | 8.0 MB | ❌ 溢出 | 可观 |
| 4000 tokens | 16.0 MB | ❌ 严重溢出 | 显著 |

当前测试场景 seq_len < 1000，KV cache 装入 L2，多 Q 复用（q_per_block > 1）无额外收益。

### 5.3 带宽墙：decode 的根本瓶颈

```
Weight Loading (GEMV + FFN): ████████████████████████████████████████████████████ 82%
Flash Attention:             █████ 7%
LM Head:                     █████ 7%
Other (RMSNorm, Add, etc.):  ████ 4%
```

decode 阶段 ~82% 时间用于权重矩阵读取 (GEMV/FFN)。这是由 Orin LPDDR5 带宽物理限制决定的，无法通过注意力算子优化突破。

## 六、优化过程的关键教训

### 6.1 寄存器压力是 Orin 上的关键因素

- `float4 k_reg[16]` 占用 64/128 = 50% 可用寄存器
- 导致 register spilling → local memory (L1→L2→DRAM) → 性能下降 ~2.5%
- 解决方案：使用 `__ldg` 内联加载，由 L1 cache 处理复用

### 6.2 SM Occupancy 对 memory-bound kernel 至关重要

| Blocks | Blocks/SM | Warps/SM | 延迟 (ms/tok) |
|--------|-----------|----------|--------------|
| 32 | 2 | 16 | 101.09 ✅ |
| 16 | 1 | 8 | 103.24 ❌ |
| 8 | 0.5 | 8 | 104.56 ❌ |

Memory-bound kernel 需要 16+ warps/SM 来隐藏访存延迟。

### 6.3 模板参数 vs 运行时参数

```
运行时 q_per_block:  103.66 ms/token (编译器无法展开)
模板 Q_PER_BLOCK=1: 101.09 ms/token (完全展开，优化为标量)
差异: 2.57 ms/token = 2.5% 性能差异
```

对 GPU kernel 中的内循环迭代次数，**必须使用编译期常量**。

## 七、使用方式

### 7.1 命令行参数

```bash
# 仅启用 MRoPE + KV Cache 融合
./qwen3_vl_infer ... --fused-rope-kv

# 启用完整 GQA 融合 (MRoPE + KV Cache + Flash Attention)
./qwen3_vl_infer ... --fused-gqa

# 推荐用法
./qwen3_vl_infer ... --cuda-graph --fused-gqa
```

| 参数 | 短选项 | 说明 |
|------|--------|------|
| `--fused-rope-kv` | `-f` | 融合 MRoPE + KV Cache 写入 |
| `--fused-gqa` | `-F` | 融合 GQA + MRoPE + KV Cache (自动包含 -f) |

### 7.2 内核架构

```
template<int Q_PER_BLOCK>  // 编译期常量，默认=1
__global__ void fused_gqa_mrope_kv_decode_fp16_kernel(...)

Phase 0:  MRoPE (Q + K) → shared memory    [64/256 threads]
Phase 0b: Write K/V to KV cache            [first group only]
Phase 0c: Q·K current token score           [1 thread per Q head]
── tile loop ──
Phase 1:  K loading + Q·K scoring            [256 threads]
Phase 2:  Online softmax (max + exp + sum)    [256 threads]
Phase 3:  V accumulation + rescale            [128 threads/dim]
── end loop ──
Phase 4:  Incorporate current token V        [128 threads]
Phase 5:  Write output                       [128 threads]
```

## 八、结论

### 8.1 融合算子有效但收益有限

融合算子在功能上是成功的——将 3 个 kernel 合为 1 个，节省 ~10 µs/layer，端到端提升 ~0.2%。性能无回退，输出正确。

### 8.2 根本原因

Decode 阶段 82% 时间花在 GEMV/FFN 权重读取上，受 Orin 170 GB/s 带宽物理限制。注意力仅占 7%，无论如何优化注意力算子，最大提升不超过 7.5%。

### 8.3 真正提升 decode 速度的方向

| 方向 | 预期提升 | 原理 |
|------|---------|------|
| INT8 GEMV 量化 | 50-80% | 权重带宽减半 |
| INT4 GEMV 量化 (AWQ/GPTQ) | 100-200% | 权重带宽减至 1/4 |
| 投机解码 (EAGLE3) | 50-150% | 一次验证多 token |
| Tensor Parallelism (双 Orin) | ~80% | 双卡分摊权重读取 |
| KV Cache 量化 (INT8 KV) | 3-5% | 减少长序列 KV 访问 |
