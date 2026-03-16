# AWQ INT4 Decode 阶段性能优化报告

## 平台信息

| 项目 | 规格 |
|------|------|
| 硬件 | NVIDIA Jetson AGX Orin 64GB |
| GPU | 2048 CUDA Cores, 64 Tensor Cores, 16 SMs, SM 8.7 |
| 内存 | 64 GB LPDDR5, 峰值 204.8 GB/s |
| 模型 | Qwen3-8B-AWQ INT4 (dim=4096, hidden_dim=12288, layers=36, heads=32, kv_heads=8, vocab=151936) |
| 量化 | AWQ INT4, group_size=128 |

## 性能总结

| 阶段 | Decode (tok/s) | 相对基线提升 | 累计提升 |
|------|:--------------:|:----------:|:-------:|
| 基线 (优化前) | 10.48 | — | — |
| P0: GEMV 内存合并访问 | 17.93 | +71.1% | +71.1% |
| P1: 向量化 uint4 加载 + ILP | 18.86 | +5.2% | +80.0% |
| P1: Fused FFN (W1+W3+SwiGLU) | 18.86 | ~0% | +80.0% |
| P2: Fused QKV (Q+K+V) | 18.87 | ~0% | +80.1% |

**最终结果：10.48 → 18.87 tok/s (+80.1%)，推理输出完全一致。**

Prefill 性能保持稳定：~150-155 tok/s (不受 decode 优化影响)。

---

## 优化详情

### 1. P0: AWQ GEMV 内存合并访问 (qweight 转置)

**问题分析**

原始 AWQ GEMV kernel 的 qweight 存储布局为 `[K, N/8]`（行优先）。内核中 32 个 warp lane 沿 K 维度并行，每个 lane 访问：

```
qweight[k_idx * packed_N + packed_out_idx]
```

由于 `packed_out_idx` 对同一 warp 的所有 lane 相同，而 `k_idx` 因 lane 而异（相邻 lane 的 k_idx 相差 1），因此相邻 lane 的地址跨度为：

```
stride = packed_N × sizeof(int32) = packed_N × 4 字节
```

- Q/K/V/O 投影 (N=4096): stride = 512 × 4 = 2048 字节
- W1/W3 (N=12288): stride = 1536 × 4 = 6144 字节

**一个 warp 的 32 次读取分散在 32 条不同的 cache line 上**，完全无法合并，浪费了 97% 的 cache line 带宽。

**解决方案**

在模型加载时（`AWQMatmulLayer::to_cuda()`），创建转置副本 `qweight_t_[N/8, K]`。新 kernel 访问：

```
qweight_t[packed_out_idx * K + k_idx]
```

相邻 lane 地址差 4 字节（1 个 int32），32 lane × 4 字节 = 128 字节 = **恰好 1 条 cache line**，实现完美合并。

**修改文件**
- `kuiper/include/op/awq_matmul.h` — 添加 `qweight_t_` 成员和 getter
- `kuiper/source/op/awq_matmul.cpp` — `to_cuda()` 中调用转置 kernel
- `kuiper/source/op/kernels/cuda/awq_gemm_fast.cu` — 新增 `awq_gemv_coalesced_kernel` 和 `transpose_qweight_kernel`
- `kuiper/source/op/kernels/cuda/awq_gemm_fast.cuh` — 新增函数声明
- `kuiper/source/op/kernels/cuda/awq_gemm_tensorcore.cu/cuh` — 分发逻辑添加 `qweight_t` 参数

**结果：10.48 → 17.93 tok/s (+71.1%)**

---

### 2. P1: 向量化 uint4 加载 + ILP

**问题分析**

合并访问后，每个 lane 每次迭代从 `qweight_t` 加载 1 个 int32（4 字节），32 lane 总计 128 字节/迭代。内循环对 group_size=128 执行 `128/32 = 4` 次迭代。

**解决方案**

利用转置后 K 维度连续的特性，每个 lane 使用 `uint4` 一次加载 4 个连续 int32（16 字节）。同时加载 4 个对应的输入值（`uint2`，8 字节）。这使得：

- 每次迭代处理 4 个 K 位置（32 lane × 4 = 128 = 整个 group）
- 内循环从 4 次迭代变为 **1 次迭代**，完全消除循环开销
- 每个 warp 每次迭代加载 32 × 16 = 512 字节（4 条 cache line），访问效率翻倍

```cuda
// 向量化加载：一次读取 4 个 packed weight
const uint4 w4 = *reinterpret_cast<const uint4*>(&warp_qweight[k_idx]);
const uint2 x2 = *reinterpret_cast<const uint2*>(&X[k_idx]);  // 4 个 half 输入
```

**修改文件**
- `kuiper/source/op/kernels/cuda/awq_gemm_fast.cu` — 更新 `awq_gemv_coalesced_kernel` 使用 uint4/uint2 向量化加载

**结果：17.93 → 18.86 tok/s (+5.2%)**

---

### 3. P1: AWQ Fused FFN (Gate+Up+SwiGLU)

**实现**

将原来的 3 个独立 kernel（W1 GEMV + W3 GEMV + SwiGLU）合并为单个 kernel `awq_fused_gate_up_swiglu_kernel`。每个 warp 依次计算 gate (W1*x) 和 up (W3*x)，然后就地计算 `SiLU(gate) * up`。

```
Phase 1: gate_acc = W1 * x  (复用 coalesced GEMV 逻辑)
Phase 2: up_acc = W3 * x    (x 从 L2 cache 读取)
Phase 3: output = SiLU(gate_acc) * up_acc  (就地融合)
```

**修改文件**
- `kuiper/source/op/kernels/cuda/awq_gemm_fast.cu/cuh` — 新增 `awq_fused_gate_up_swiglu_kernel`
- `kuiper/source/model/qwen3_awq.cpp` — `gate_up_swiglu()` 对 M=1 调用 fused kernel

**结果：18.86 → 18.86 tok/s (~0%)**

**分析**：增益不可测量，原因如下：
- 权重读取 (48 MB/层) 远大于中间激活 (~100 KB/层)，减少激活读写的贡献微乎其微
- CUDA Graph 已将 kernel launch 开销优化到接近零
- 输入 X (8 KB) 在 L2 cache 中，第二次读取几乎零开销

尽管 decode 性能无明显提升，该优化对非 CUDA Graph 场景（如 profiling、动态 batch）有结构性收益，且避免了中间 buffer 分配。

---

### 4. P2: AWQ Fused QKV (Q+K+V 投影合并)

**实现**

将 Q/K/V 三个独立 AWQ GEMV 合并为单个 kernel `awq_fused_qkv_kernel`。通过 block 索引分配确定当前处理的投影：

```
blocks [0, q_blocks)                         → Q 投影 (N=4096, 64 blocks)
blocks [q_blocks, q_blocks + k_blocks)       → K 投影 (N=1024, 16 blocks)
blocks [q_blocks + k_blocks, total_blocks)   → V 投影 (N=1024, 16 blocks)
```

所有 block 共享相同输入 X，使用相同的 coalesced+vectorized GEMV 内核体。

**修改文件**
- `kuiper/source/op/kernels/cuda/awq_gemm_fast.cu/cuh` — 新增 `awq_fused_qkv_kernel`
- `kuiper/source/model/qwen3_awq.cpp` — `batched_qkv_projection()` 对 M=1 调用 fused kernel

**结果：18.86 → 18.87 tok/s (~0%)**

**分析**：与 Fused FFN 相同的原因——权重读取主导 decode 时间，input 共享带来的带宽节省可忽略。

---

### 5. P0: LM Head 量化（未实施）

**分析**

LM Head 使用 FP16 权重 `[151936, 4096]`，每 token 读取 151936 × 4096 × 2 = 1186 MB，占总 decode 内存流量的 ~26%。

若运行时量化为 AWQ INT4，读取量降为 ~323 MB，节省 ~863 MB，预计带来 ~10% 的 decode 加速。

**未实施原因**
1. **质量风险**：LM Head 输出直接用于 token 采样，简单 min-max INT4 量化（不同于 AWQ 的 activation-aware 优化）可能显著降低生成质量
2. **输出类型不兼容**：当前 LM Head 输出 FP32 logits（`matmul_kernel_cu_fp16_input_fp16_weight` → FP32），AWQMatmulLayer 输出 FP16，需要修改采样代码和 buffer 类型
3. **实现复杂度**：需要 GPU 量化 kernel、AWQ bit packing、输出类型转换等
4. **推荐方案**：在导出脚本 (`export_qwen3-8B-awq.py`) 中使用完整 AWQ 算法量化 LM Head，保证量化质量

---

## 内存流量分析

### Decode 阶段每 token 内存读取

| 组件 | 读取量 (MB) | 占比 |
|------|----------:|:----:|
| AWQ qweight (36层 × 7投影) | 3,312 | 71% |
| AWQ zeros + scales | ~90 | 2% |
| LM Head FP16 | 1,186 | 26% |
| KV Cache (ctx=255) | ~38 | 1% |
| Other (RMSNorm, activations) | ~14 | <1% |
| **Total** | **~4,640** | 100% |

### 理论极限 vs 实际

| 指标 | 值 |
|------|:--:|
| 总读取量 | 4,640 MB/token |
| LPDDR5 峰值带宽 | 204.8 GB/s |
| 理论最小时间 | 22.7 ms/token |
| 理论最大速度 | 44.1 tok/s |
| 实际速度 | 18.87 tok/s |
| 带宽利用率 | 42.8% |

**剩余 gap 主要来自**：
1. LPDDR5 持续带宽约 170 GB/s（非峰值 204.8 GB/s）→ 27.3 ms → 36.6 tok/s
2. LOP3 解量化和 FMA 计算开销
3. CUDA Graph replay 和 kernel 调度开销
4. 注意力机制 (Flash Attention + KV cache) 的计算开销
5. 非 AWQ kernel (RMSNorm, RoPE, softmax, SwiGLU) 的执行时间

---

## 修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `kuiper/include/op/awq_matmul.h` | 添加 `qweight_t_` 成员和 `get_qweight_t()` getter |
| `kuiper/source/op/awq_matmul.cpp` | `to_cuda()` 添加转置；`forward()` 传递 `qweight_t_` |
| `kuiper/source/op/kernels/cuda/awq_gemm_fast.cu` | 新增 coalesced GEMV、transpose、fused FFN、fused QKV kernel |
| `kuiper/source/op/kernels/cuda/awq_gemm_fast.cuh` | 新增函数声明 |
| `kuiper/source/op/kernels/cuda/awq_gemm_tensorcore.cu` | 分发逻辑支持 `qweight_t` |
| `kuiper/source/op/kernels/cuda/awq_gemm_tensorcore.cuh` | 函数签名添加 `qweight_t` 参数 |
| `kuiper/source/model/qwen3_awq.cpp` | `gate_up_swiglu()` 和 `batched_qkv_projection()` 使用 fused kernel |

---

## 进一步优化方向

1. **LM Head AWQ 量化**（导出时）：在导出脚本中对 LM Head 应用完整 AWQ 量化，预计 +10% decode 速度
2. **FlashAttention v2/v3**：优化注意力 kernel 减少读写放大
3. **两阶段 GEMV**：对大 N 投影（W1/W3 N=12288）使用 shared memory 分块，提高 L2 命中率
4. **INT8 KV Cache**：将 KV cache 从 FP16 量化为 INT8，减少 attention 阶段内存流量
5. **Speculative decoding**：利用小模型草稿+大模型验证，提高有效 throughput
