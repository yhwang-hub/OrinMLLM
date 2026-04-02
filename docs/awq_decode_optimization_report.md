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

#### 1.1 P0 性能提升原理：源码级深度分析

##### 1.1.1 核心瓶颈：GPU 内存合并访问机制与 GEMV 内存布局的冲突

**GPU 内存系统基础**

NVIDIA GPU 的全局内存（Global Memory）以 **128 字节 cache line** 为最小传输单元。当一个 warp（32 个线程）同时发起内存访问时，硬件会将这些请求合并（coalesce）为尽可能少的 cache line 事务：

- **完美合并**：32 个线程访问连续的 128 字节 → 1 次内存事务
- **完全不合并**：32 个线程访问分散的地址 → 最多 32 次内存事务

内存事务数量直接决定了有效内存带宽利用率。Orin 的 LPDDR5 峰值带宽为 204.8 GB/s，而 decode 阶段（M=1 GEMV）完全是**内存带宽受限**的（计算强度仅 ~4 FLOPs/byte，远低于 Orin ~49 FLOPs/byte 的 roofline 拐点），因此内存访问效率几乎等价于 kernel 性能。

**原始内核的致命访问模式**

查看原始 `awq_gemv_fast_kernel` 的qweight 访问模式（`awq_gemm_fast.cu` 中优化前的版本）：

```cuda
// 原始内核（优化前）——qweight 布局 [K, N/8]
// 每个 warp 处理 8 个输出通道（1 个 packed INT32 列）
// 32 lanes 沿 K 维度步进 32

for (int k = lane_id; k < group_size; k += 32) {
    int k_idx = group_start + k;
    // ★ 关键非合并访问：
    const int32_t w_packed = __ldg(&qweight[k_idx * packed_N + packed_out_idx]);
    //                              ^^^^^^^^                    ^^^^^^^^^^^^^^^
    //                              lane_id 不同 → k_idx 不同   同一 warp 内完全相同
}
```

展开 warp 内 32 个 lane 实际访问的地址：

```
Lane 0:  qweight[(group_start + 0)  * packed_N + packed_out_idx]  →  base + 0         * packed_N * 4
Lane 1:  qweight[(group_start + 1)  * packed_N + packed_out_idx]  →  base + 1         * packed_N * 4
Lane 2:  qweight[(group_start + 2)  * packed_N + packed_out_idx]  →  base + 2         * packed_N * 4
...
Lane 31: qweight[(group_start + 31) * packed_N + packed_out_idx]  →  base + 31        * packed_N * 4
```

相邻 lane 的地址间距 = `packed_N × 4` 字节。以 W1/W3 投影 (N=12288) 为例：

```
stride = (12288/8) × 4 = 6144 字节 = 48 条 cache line
```

**一个 warp 的 32 次访问命中 32 条不同的 cache line**，每条 cache line 128 字节中仅使用了 4 字节（一个 int32），**有效带宽利用率仅 4/128 = 3.125%**。

这意味着每读取 1 字节有效权重数据，实际产生了 32 字节的内存事务。对于整个模型 3,312 MB 的 AWQ 权重，**等效于需要传输 ~106 GB 的 DRAM 流量**。

##### 1.1.2 优化方案：转置 + 合并访问

**Step 1: 模型加载时一次性转置**

在 `AWQMatmulLayer::to_cuda()`（`kuiper/source/op/awq_matmul.cpp`）中，将 qweight 从 `[K, N/8]` 转置为 `[N/8, K]`：

```cpp
// awq_matmul.cpp — to_cuda()
// 创建转置权重用于合并 decode 访问: [K, N/8] → [N/8, K]
if (!qweight_.is_empty()) {
    int32_t packed_out = out_features_ / 8;
    int32_t total = in_features_ * packed_out;
    qweight_t_ = tensor::Tensor(base::DataType::kDataTypeInt32, total, true, cuda_alloc);
    kernel::awq_transpose_qweight_cu(
        qweight_.ptr<int32_t>(),     // src: [K, N/8]
        qweight_t_.ptr<int32_t>(),   // dst: [N/8, K]
        in_features_,                 // K
        packed_out,                   // N/8
        nullptr
    );
    cudaDeviceSynchronize();
}
```

转置 kernel 本身非常简单（`awq_gemm_fast.cu`），只在初始化时执行一次：

```cuda
__global__ void transpose_qweight_kernel(
    const int32_t* __restrict__ src,  // [K, packed_N]
    int32_t* __restrict__ dst,        // [packed_N, K]
    int K, int packed_N
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (k < K && n < packed_N) {
        dst[n * K + k] = src[k * packed_N + n];
    }
}
```

**Step 2: 合并访问的 GEMV 内核**

优化后的 `awq_gemv_coalesced_kernel`（`awq_gemm_fast.cu`）使用转置后的 `qweight_t[N/8, K]`：

```cuda
// awq_gemv_coalesced_kernel — 合并访问版本
// 每个 warp 负责 1 个 packed_out_idx（8 个输出通道）
const int32_t* warp_qweight = qweight_t + packed_out_idx * K;
//                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                            该 warp 的权重行是 K 维度连续存储的

for (int k = lane_id * 4; k < group_size; k += 128) {
    int k_idx = group_start + k;
    // ★ 合并访问：warp_qweight 基址 + k_idx，相邻 lane 地址连续
    const uint4 w4 = *reinterpret_cast<const uint4*>(&warp_qweight[k_idx]);
}
```

展开 warp 内 32 个 lane 实际访问的地址（向量化版本，每 lane 读 4 个 int32 = 16 字节）：

```
Lane 0:  warp_qweight[0×4]  → base + 0   字节  ┐
Lane 1:  warp_qweight[1×4]  → base + 16  字节  │ 32 × 16 = 512 字节
Lane 2:  warp_qweight[2×4]  → base + 32  字节  │ = 4 条 cache line
...                                              │ 完美合并！
Lane 31: warp_qweight[31×4] → base + 496 字节  ┘
```

**32 个 lane 的 512 字节访问恰好覆盖 4 条连续 cache line，每字节均为有效数据，合并率 100%**。

##### 1.1.3 分发逻辑：如何在 M=1 时使用合并版本

`awq_gemm_tensorcore.cu` 中的分发逻辑根据 batch_size 选择不同路径：

```cuda
void awq_gemm_tensorcore_cu(
    /* ... */
    const int32_t* qweight_t   // ← P0 新增参数：转置后权重
) {
    if (M == 1) {
        // M=1 decode: 使用合并访问的 GEMV（带转置权重）
        awq_gemv_coalesced_cu(
            input, qweight_t, qzeros, scales, output,
            in_features, out_features, group_size, stream
        );
    } else {
        // M>1 prefill: 使用 Tensor Core MMA（原始权重布局）
        awq_gemm_vllm_cu(
            input, qweight, qzeros, scales, output,
            M, in_features, out_features, group_size, stream
        );
    }
}
```

`AWQMatmulLayer::forward()`（`awq_matmul.cpp`）在调用时传入转置权重指针：

```cpp
kernel::awq_gemm_tensorcore_cu(
    /* ... 常规参数 ... */
    stream,
    qweight_t_.is_empty() ? nullptr : qweight_t_.ptr<int32_t>()  // 传入转置权重
);
```

##### 1.1.4 定量分析：为什么带来 71.1% 的性能提升

| 指标 | 优化前（非合并） | 优化后（合并） | 改善倍数 |
|:---:|:---:|:---:|:---:|
| 每 warp 每次迭代 cache line 事务 | 32 条 | 1 条（标量）/ 4 条（uint4） | 8× / 32× |
| 每 cache line 有效数据利用率 | 4/128 = 3.125% | 128/128 = 100% | 32× |
| AWQ 权重等效 DRAM 流量 (3,312 MB) | ~106 GB | ~3.3 GB | 32× |
| 实际 DRAM 事务减少（考虑 L2 缓存） | — | — | ~8-10× |

理论上合并访问可将权重读取带宽提升 32 倍，但实际提升为 71.1% 而非 32 倍的原因：

1. **L2 cache 部分缓解非合并访问**：Orin 有 4MB L2 cache，部分 cache line 可被复用
2. **权重之外的其他开销**：LM Head (FP16, 1,186 MB) 不受此优化影响，占总流量 26%
3. **其他 kernel 开销**：RMSNorm、RoPE、Flash Attention、SwiGLU 等不受影响
4. **合并后接近理论带宽极限**：优化后 AWQ GEMV 已接近 LPDDR5 持续带宽瓶颈

##### 1.1.5 空间代价

转置需要额外存储 `qweight_t_`，与原始 `qweight_` 大小相同。对于 Qwen3-8B-AWQ：

```
每层 7 个投影 × 36 层:
  qweight_t 额外内存 = 3,312 MB（与 qweight 相同）
  总 GPU 内存增加 ≈ 原模型 AWQ 权重的 2 倍
```

考虑到 Orin 64 GB 内存充足，用 ~3.3 GB 额外内存换取 71.1% 的 decode 提速是完全值得的。

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

#### 2.1 P1 性能提升原理：源码级深度分析

##### 2.1.1 P0 后遗留的三个效率瓶颈

P0 完成转置后，如果仅做标量合并（每 lane 每次读 1 个 int32），内循环结构为：

```cuda
// P0 合并但非向量化的假设版本
for (int k = lane_id; k < group_size; k += 32) {   // group_size=128 → 4 次迭代
    int k_idx = group_start + k;
    int32_t w_packed = warp_qweight[k_idx];        // 4 字节/lane, 合并
    half x_val = X[k_idx];                         // 2 字节/lane
    // ... 解量化和 FMA ...
}
```

存在三个效率问题：

1. **循环开销**：4 次迭代意味着 4 次循环判断、4 次地址计算、4 次 index 递增
2. **加载指令数量多**：每次迭代 2 条 load 指令（weight + input），4 次 = 8 条 load 指令
3. **指令级并行度（ILP）不足**：GPU 的 load-store unit 在每次迭代中只被发射 1-2 条指令，无法充分利用内存管线的深度来隐藏 DRAM 延迟

##### 2.1.2 优化方案：uint4 向量化加载 + 循环展开 ILP

P1 优化在 `awq_gemv_coalesced_kernel` 中将标量加载替换为向量化加载（`awq_gemm_fast.cu`）：

```cuda
// P1 优化后：每 lane 一次读 4 个 int32（128 位 = 16 字节 uint4）
for (int k = lane_id * 4; k < group_size; k += 128) {
    //        ^^^^^^^^^^                      ^^^
    //        每 lane 偏移 4 个 int32        32 lanes × 4 = 128 = group_size

    const int k_idx = group_start + k;

    // ★ 关键优化 1: uint4 向量化加载权重（16 字节 = 4 个 packed INT32）
    const uint4 w4 = *reinterpret_cast<const uint4*>(&warp_qweight[k_idx]);

    // ★ 关键优化 2: uint2 向量化加载输入（8 字节 = 4 个 half）
    const uint2 x2 = *reinterpret_cast<const uint2*>(&X[k_idx]);
    const half* x_ptr = reinterpret_cast<const half*>(&x2);

    // ★ 关键优化 3: ILP — 将 4 个 packed weight 放入寄存器数组
    const uint32_t w_arr[4] = {w4.x, w4.y, w4.z, w4.w};

    // 4 个 K 位置的解量化和 FMA 完全展开
    #pragma unroll
    for (int v = 0; v < 4; v++) {
        const half2 x_h2 = __half2half2(x_ptr[v]);  // broadcast 输入值
        uint32_t w_h[4];
        lop3_extract_int4_to_fp16x2(w_arr[v], w_h); // 解量化 8 个 INT4 → 4 个 half2

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            half2 w_h2 = *reinterpret_cast<const half2*>(&w_h[j]);
            half2 dq_h2 = __hfma2(s_h2[j], w_h2, neg_sz_h2[j]);  // scale * w - scale * zero
            half2 prod = __hmul2(x_h2, dq_h2);                    // x * dequant(w)
            acc[j * 2]     += __low2float(prod);
            acc[j * 2 + 1] += __high2float(prod);
        }
    }
}
```

##### 2.1.3 三个性能提升机制详解

**机制 1: 减少内存事务指令数量（Load Coalescing Amplification）**

GPU 的 Load-Store Unit (LSU) 对 `uint4`（128 位）加载只发射 **1 条 LDG.128 指令**，而 4 次标量 int32 加载需要 **4 条 LDG.32 指令**。

```
优化前（标量合并×4迭代）:
  每 lane 每 group: 4 × LDG.32(weight) + 4 × LDG.16(input) = 8 条 load 指令
  每 warp 内存事务: 4 × 1(weight) + 4 × 1(input) = 8 次

优化后（向量化×1迭代）:
  每 lane 每 group: 1 × LDG.128(weight) + 1 × LDG.64(input) = 2 条 load 指令
  每 warp 内存事务: 4(weight cache lines) + 2(input cache lines) = 6 次
```

load 指令减少 4 倍，LSU 的指令队列压力显著降低，使得 LSU 完成每次加载的有效吞吐更高。

**机制 2: 循环开销消除**

对于 group_size=128，32 lanes × 4 位置/lane = 128，意味着一个 group 只需 **1 次循环迭代**：

```
优化前: for 循环 4 次 → 4 次分支判断 + 4 次 k 递增 + 4 次 k_idx 计算
优化后: for 循环 1 次 → 编译器可完全消除循环（loop body 直接内联）
```

在纯内存带宽受限的 kernel 中，这些"小"开销看似微不足道，但它们占据了有限的指令发射槽位。GPU SM 每周期能发射的指令数有限（Orin SM 8.7: 每 sub-partition 1 条 warp 指令/周期），减少非内存指令意味着更多发射槽位可以被 load/store 占用。

**机制 3: 指令级并行（ILP）与内存延迟隐藏**

这是 P1 的核心价值。DRAM 访问延迟约 400-600 个时钟周期，GPU 通过两种机制隐藏延迟：

- **线程级并行 (TLP)**：多个 warp 轮流执行，一个 warp 等待数据时切换到另一个
- **指令级并行 (ILP)**：单个 warp 内多条独立指令可以同时在流水线中飞行

向量化加载将 4 个 K 位置的数据一次性预取到寄存器：

```cuda
// 1 条 LDG.128 加载 4 个 packed weight → 分别存入 w4.x, w4.y, w4.z, w4.w
const uint4 w4 = *reinterpret_cast<const uint4*>(&warp_qweight[k_idx]);

// 数据到达后，4 个解量化+FMA 操作链完全独立，可以并行发射
const uint32_t w_arr[4] = {w4.x, w4.y, w4.z, w4.w};
// w_arr[0] 的 lop3+fma 与 w_arr[1] 的 lop3+fma 之间无数据依赖
```

对比标量迭代版本：

```cuda
// 迭代 1: load w[k+0] → 等待 → 解量化 → FMA → 循环跳转
// 迭代 2: load w[k+32] → 等待 → 解量化 → FMA → 循环跳转  ← 必须等迭代1完成
// ...串行化的 load-compute 链
```

在标量版本中，每次 `for` 循环中的 `k += 32` 改变了 `k_idx`，形成了 load-use 依赖链。而向量化版本中，4 个 weight 通过一条 `uint4` 加载**同时**到达寄存器，4 组 lop3 解量化和 FMA 计算可以**同时**在 ALU 流水线中排队，大幅提高了 compute 和 memory 的并行重叠度。

##### 2.1.4 定量带宽分析

```
每 warp 每 group 的数据加载量:
  Weight: 32 lanes × 4 int32 × 4 bytes = 512 bytes (4 cache lines)
  Input:  32 lanes × 4 half  × 2 bytes = 256 bytes (2 cache lines)
  Zeros:  1 int32                       = 4 bytes   (from __ldg, L2 cached)
  Scales: 1 uint4 (8 half)             = 16 bytes  (from L2)
  Total DRAM traffic ≈ 768 bytes/group

  每 warp 每 group 计算: 4(K位置) × 8(输出) × 2(乘加) = 64 FLOPs
  计算强度: 64/768 ≈ 0.083 FLOPs/byte → 极度内存带宽受限
```

在这种极端带宽受限的场景下，P1 的 5.2% 提升完全来自**减少无效指令开销 + 提升内存管线利用率**。提升幅度相对 P0 较小，正因为 P0 已经消除了最大的瓶颈（cache line 浪费），P1 的优化是在已经接近理论带宽极限的基础上"挤压最后的性能"。

##### 2.1.5 P0+P1 在 qwen3_awq.cpp 模型代码中的使用路径

在 `qwen3_awq.cpp` 中，所有 AWQ 线性层在 decode 阶段均经过以下调用路径使用 P0+P1 优化的 GEMV：

1. **常规投影（wo, w2）** → `batched_matmul_forward()` → `AWQMatmulLayer::forward()` → `awq_gemm_tensorcore_cu(M=1)` → `awq_gemv_coalesced_cu()` → `awq_gemv_coalesced_kernel`

2. **QKV 投影** → `batched_qkv_projection()` 对 `batch_size==1` 直接调用 `kernel::awq_fused_qkv_cu()`，该 fused kernel 内部使用与 `awq_gemv_coalesced_kernel` 完全相同的向量化合并 GEMV 计算体（相同的 `uint4` 权重加载 + `uint2` 输入加载 + LOP3 解量化 + half2 FMA）

3. **FFN Gate+Up** → `gate_up_swiglu()` 对 `batch_size==1` 直接调用 `kernel::awq_fused_gate_up_swiglu_cu()`，该 fused kernel 内部做两次完全相同的向量化合并 GEMV（分别用于 W1 和 W3），然后就地融合 SiLU 激活

每层 7 个投影 × 36 层 = **252 次 AWQ GEMV 调用/token**，全部受益于 P0+P1 优化。

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
