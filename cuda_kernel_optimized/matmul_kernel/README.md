# matmul_kernel.cu 优化报告

## 目标平台
- **GPU**: NVIDIA Orin (SM 8.7, Ampere架构)
- **CUDA**: 12.6.68
- **内存带宽**: ~100 GB/s (LPDDR5)
- **SM数量**: 16个，每SM 128个CUDA核心

## 一、性能指标对比

### 1.1 NCU 级别 Kernel 性能指标

> 注: 本系统 `ncu` 需要 sudo 权限，使用 CUDA Events 替代收集高精度计时数据。

**CUDA Events Kernel-Level Timing（50次迭代平均值）：**

| 维度 (MxK) | 场景 | Kernel | 原始(ms) | 优化后(ms) | 加速比 |
|---|---|---|---|---|---|
| 4096×4096 | Qwen3 q/o投影 | FP32 GEMV | 0.924 | 0.576 | **1.60x** |
| 4096×4096 | | INT8 GEMV | 0.390 | 0.362 | **1.08x** |
| 4096×4096 | | Pure FP16 GEMV | 0.288 | 0.288 | 1.00x |
| 4096×4096 | | FP16→FP32 GEMV | 0.293 | 0.288 | 1.02x |
| 4096×512 | Qwen3 kv投影 | FP32 GEMV | 0.114 | 0.113 | 1.01x |
| 4096×512 | | INT8 GEMV | 0.106 | 0.098 | **1.08x** |
| 4096×512 | | FP16→FP32 GEMV | 0.048 | 0.041 | **1.18x** |
| 4096×12288 | Qwen3 FFN gate/up | FP32 GEMV | 2.481 | 1.706 | **1.45x** |
| 4096×12288 | | INT8 GEMV | 1.146 | 1.063 | **1.08x** |
| 4096×12288 | | FP16→FP32 GEMV | 0.854 | 0.611 | **1.40x** |
| 12288×4096 | Qwen3 FFN down | INT8 GEMV | 1.097 | 1.007 | **1.09x** |
| 12288×4096 | | FP16→FP32 GEMV | 0.788 | 0.611 | **1.29x** |
| 3584×3584 | Qwen2.5 qkv | FP32 GEMV | 0.673 | 0.441 | **1.53x** |
| 3584×3584 | | INT8 GEMV | 0.304 | 0.280 | **1.08x** |
| 3584×18944 | Qwen2.5 FFN | INT8 GEMV | 1.556 | 1.435 | **1.08x** |
| 18944×3584 | Qwen2.5 FFN down | INT8 GEMV | 1.466 | 1.343 | **1.09x** |
| 18944×3584 | | Pure FP16 GEMV | 0.859 | 0.822 | **1.05x** |

**NCU 基准命令（需sudo）：**
```bash
sudo ncu --set full -o matmul_orig ./bench_matmul_timing  # 原始版本
sudo ncu --set full -o matmul_opt ./bench_matmul_timing   # 优化版本
```

### 1.2 端到端模型推理性能对比

**参考项目 vs 优化后（头对头）：**

| 模型 | 指标 | 参考项目 | 优化后 | 变化 |
|---|---|---|---|---|
| **Qwen3-8B FP16** | Prefill (t/s) | 136.3 | 142.6 | **+4.6%** |
| | Decode (t/s) | 10.17 | 10.21 | +0.4% |
| **Qwen3-8B AWQ** | Prefill (t/s) | 131.5 | 131.3 | ≈0% |
| | Decode (t/s) | 9.20 | 9.23 | +0.3% |
| **Qwen2.5-7B INT8** | Prefill (t/s) | 6.09 | 6.09 | ≈0% |
| | Decode (t/s) | 5.72 | 5.71 | ≈0% |
| **Qwen2.5-7B FP16** | Prefill (t/s) | 149.6 | 152.7 | **+2.1%** |
| | Decode (t/s) | 10.92 | 10.95 | +0.3% |
| **Qwen3-VL-8B** | ViT Prefill (t/s) | 385 | **676** | **+75.6%** |
| | LLM Decode (t/s) | 9.32 | 9.58 | **+2.8%** |

> 注: Prefill数据有一定波动，取多次运行中间值。ViT Prefill提升极为显著。

---

## 二、每个 CUDA Kernel 优化原理

### 2.1 `matmul_kernel_cu_fp32` — FP32 GEMV (Decode)

**优化手段：**
1. **消除共享内存中间变量**: 原始代码使用 `sdata[THREAD_PER_BLOCK]` 共享内存数组作为累加器，然后传给CUB BlockReduce。优化后直接使用寄存器变量 `sum` 累加，避免了共享内存读写延迟和一次 `__syncthreads()` 同步
2. **`__ldg()` 只读缓存**: 权重矩阵只读，使用 `__ldg()` 走 L1 只读纹理缓存路径，减少 L1 数据缓存污染
3. **`fmaf()` 融合乘加**: 将 `a*b + c` 替换为单条 `fmaf` 指令，减少指令数量并提高精度
4. **`__restrict__` 指针限定**: 告诉编译器输入/输出无别名，允许更激进的优化

**效果**: 在 M=4096 K=4096 维度下 **1.60x** 加速

### 2.2 `matmul_kernel_cu_fp32int8` — INT8 量化 GEMV (Decode)

**优化手段：**
1. **`char4` 向量化加载**: 原始代码逐元素标量加载 `int8_t`，每次1字节。优化后使用 `char4` 一次加载4个int8值（4字节），减少75%的内存事务
2. **`float4` 输入向量化**: 对应的FP32输入也使用 `float4` 加载（16字节），与 `char4` 对齐
3. **`__ldg()` 只读缓存**: 权重、scale因子均通过只读缓存路径加载
4. **消除共享内存**: 同FP32优化，直接寄存器累加
5. **`fmaf()` 融合乘加**: 减少浮点指令数量

**效果**: 所有维度下一致 **1.08-1.09x** 加速

### 2.3 `batched_matmul_kernel_cu_fp32` — FP32 Batched GEMM (Prefill)

**优化手段：**
- 同 `matmul_kernel_cu_fp32`：`__ldg()`、`fmaf()`、寄存器累加、消除 `sdata`

### 2.4 `batched_matmul_fp16_weight_kernel` — 混合精度 Batched GEMM (Prefill)

**优化手段：**
1. **4元素向量化权重加载**: 从 `half2`（4字节/2元素）升级到 `2×half2`（8字节/4元素），配合 `float4` 输入加载
2. **`__ldg()` 只读缓存**: 权重通过纹理缓存加载
3. **`fmaf()` + 消除 `sdata`**: 直接寄存器累加

### 2.5 `fp32_to_fp16_kernel` / `fp16_to_fp32_kernel` — 精度转换

**优化手段：**
1. **4元素宽向量化**: 从每线程处理2元素升级到4元素
   - `fp32_to_fp16`: `float4` 读取 → 2×`half2` 写入
   - `fp16_to_fp32`: 2×`half2` 读取 → `float4` 写入
2. **`__ldg()` 输入加载**: 源数据走只读缓存

### 2.6 `gemv_pure_fp16_kernel_v2` — 纯FP16 GEMV (Decode)

**优化手段：**
1. **`__ldg()` 双向**: 权重和输入的 `float4` 加载均使用 `__ldg()`，利用只读缓存
2. **`fmaf()` 替换标量乘加**: 8个 `fmaf` 调用替代8个 `a*b + c` 表达式
3. **remainder路径优化**: 余数处理也使用 `__ldg()` + `fmaf()`

### 2.7 `batched_gemm_pure_fp16_kernel` — 纯FP16 Batched GEMM (Prefill)

**优化手段：**
1. **`half2` → `float4` 升级**: 从每次加载2个half（4字节）升级到8个half（16字节），4倍带宽利用率
2. **`__ldg()` 双向**: 权重和输入均走只读缓存
3. **消除 `sdata` 中间变量**: 直接寄存器累加传给CUB
4. **`fmaf()` 融合乘加**: 内层循环全部使用 `fmaf`

**效果**: 这是 **Qwen3-VL ViT 推理提速75%** 的主要贡献者

### 2.8 `gemv_fp16_input_fp16_weight_fp32_output` — FP16→FP32 GEMV (cls_logits)

**优化手段：**
1. **`half2` → `float4` 升级**: 从每次2个half升级到8个half（16字节），宽度4倍
2. **4路ILP累加器**: `sum0/sum1/sum2/sum3` 四个独立累加器，打破数据依赖链，提高指令级并行度
3. **`fmaf()` 嵌套**: `fmaf(a, b, fmaf(c, d, sum))` 形成流水线
4. **`int64_t` 偏移**: 防止 `row * M` 大vocabsize时int32溢出

**效果**: M=4096 K=12288 下 **1.40x** 加速

---

## 三、在 OrinMLLM 工程中的使用方式

### 3.1 调度架构

```
MatmulLayer::forward() (matmul.cpp)
├── FP32 input + FP32 weight     → matmul_kernel_cu()         → matmul_kernel_cu_fp32<128,1>
├── FP32 input + INT8 weight     → matmul_kernel_cu_qint8()   → matmul_kernel_cu_fp32int8<128,1>
├── FP32 input + FP16 weight     → matmul_kernel_cu_fp16_weight()
│   ├── M ≥ 2048                 → optimized::gemv_fp16_bandwidth_optimized<256,1>
│   └── M < 2048                 → optimized::gemv_fp16_optimized<8,4>
├── FP16 input + FP16 weight     → matmul_kernel_cu_pure_fp16()
│   └──                          → gemv_pure_fp16_kernel_v2<32,8>
└── FP16 input + FP16 weight → FP32 out → matmul_kernel_cu_fp16_input_fp16_weight()
    └──                          → gemv_fp16_input_fp16_weight_fp32_output<8,4>

BatchedMatmulLayer::forward() (batched_matmul.cpp)
├── FP32 path                    → batched_matmul_kernel_cu()  → batched_matmul_kernel_cu_fp32<128>
├── FP16 weight                  → batched_matmul_kernel_cu_fp16_weight()
│   ├── batch ≥ 8 + cuBLAS       → fp32_to_fp16_kernel → cublasHgemm → fp16_to_fp32_kernel
│   └── fallback                 → batched_matmul_fp16_weight_kernel<256>
├── Pure FP16                    → batched_matmul_kernel_cu_pure_fp16()
│   ├── cuBLAS available         → cublasHgemm (Tensor Core)
│   └── fallback                 → batched_gemm_pure_fp16_kernel<256>
└── FP16→FP32                   → batched_matmul_kernel_cu_fp16_input_fp16_weight()
    ├── cuBLAS                   → cublasGemmEx (Tensor Core)
    └── fallback                 → gemv_fp16_input_fp16_weight_fp32_output<8,4> (逐batch)
```

### 3.2 模型→Kernel路径映射

| 模型 | 权重格式 | Decode GEMV | Prefill Batched | cls_logits |
|---|---|---|---|---|
| Qwen3-8B FP16 | FP16 | `gemv_pure_fp16_kernel_v2` | `cublasHgemm` | `gemv_fp16_input_fp16_weight_fp32_output` |
| Qwen3-8B AWQ | INT8 | `matmul_kernel_cu_fp32int8` | `batched_matmul_kernel_cu_fp32` | FP32路径 |
| Qwen2.5-7B INT8 | INT8 | `matmul_kernel_cu_fp32int8` | `batched_matmul_kernel_cu_fp32` | FP32路径 |
| Qwen2.5-7B FP16 | FP16w+FP32io | `optimized::gemv_fp16_*` | `cublasHgemm`+conversion | FP16 weight路径 |
| Qwen3-VL-8B | FP16 | `gemv_pure_fp16_kernel_v2` | `cublasHgemm` / `batched_gemm_pure_fp16_kernel` | `gemv_fp16_input_fp16_weight_fp32_output` |

### 3.3 关键维度

| 模型 | dim | FFN intermediate | vocab_size | kv_dim |
|---|---|---|---|---|
| Qwen3-8B | 4096 | 12288 | 151936 | 512 |
| Qwen2.5-7B | 3584 | 18944 | 152064 | 512 |

---

## 四、优化后 Kernel 运行原理详解

### 4.1 `matmul_kernel_cu_fp32<128,1>` — FP32 GEMV

**Grid/Block 配置**: `<<<K, 128>>>`，每个block处理权重矩阵的一行

**Global Memory访问**:
- 权重: 每block读取一行 M 个float（通过 `float4` 向量化读取，每次16字节）
- 输入: 所有block共享读取同一个长度为M的向量
- 使用 `__ldg()` 走只读缓存路径加载权重，减少L1 数据缓存污染

**Threads执行流程**:
1. 128个线程协作处理一行M个元素
2. 每线程处理 `M/(128×4)` 组float4，使用寄存器变量 `sum` 累加
3. `fmaf()` 单指令完成乘加，4个fmaf串联处理一组float4
4. 通过CUB `BlockReduce` 将128个线程的部分和归约为最终结果

**Block/Shared Memory**:
- 仅使用CUB内部的 `TempStorage` 共享内存（~128 floats）
- 消除了原始版本的 `sdata[128]` 中间共享内存数组

### 4.2 `matmul_kernel_cu_fp32int8<128,1>` — INT8 GEMV

**Grid/Block 配置**: `<<<K, 128>>>`

**Global Memory访问**:
- INT8权重: 通过 `char4` 加载（4字节/次，4个元素），比标量加载减少75%事务
- FP32输入: 通过 `float4` 加载（16字节/次，4个元素）
- Scale因子: 通过 `__ldg()` 加载，利用只读缓存。同一group内的连续4个元素通常共享同一scale（group_size=128），L1缓存命中率极高

**Threads执行流程**:
1. 每线程循环处理 `M/(128×4)` 组 char4+float4
2. 对每组4个元素：加载char4权重→加载float4输入→查找4个scale→fmaf累加
3. Scale查找: `__ldg(&scales[base_idx / group_size])`，group_size=128时每32组才跨越一个group边界

**Block/Shared Memory**:
- 仅CUB TempStorage，无额外共享内存

### 4.3 `gemv_pure_fp16_kernel_v2<32,8>` — 纯FP16 GEMV

**Grid/Block 配置**: `<<<(K+7)/8, 256>>>` — 8个warp/block，每warp处理一行

**Global Memory访问**:
- 使用 `float4` 加载8个half（16字节/次）: `__ldg(weight_f4 + i)`, `__ldg(input_f4 + i)`
- 双向 `__ldg()`: 权重和输入均走只读纹理缓存

**Threads执行流程**:
1. 每个warp的32个lane协作处理一行M个元素
2. 4个独立累加器 `sum0/sum1/sum2/sum3` 打破数据依赖链
3. 内层: `float4` → 4组 `half2` → `__half22float2` → `fmaf` 嵌套累加
4. Warp shuffle归约: 5轮 `__shfl_down_sync`（32→16→8→4→2→1）

**Block/Shared Memory**:
- **零共享内存使用** — 完全寄存器+warp shuffle

### 4.4 `gemv_fp16_input_fp16_weight_fp32_output<8,4>` — FP16→FP32 GEMV

**Grid/Block 配置**: `<<<(K+7)/8, 256>>>` — 同FP16 GEMV

**Global Memory访问**:
- 升级为 `float4`（16字节/8个half），比原始 `half2`（4字节/2个half）宽4倍
- 双向 `__ldg()` 减少缓存污染

**Threads执行流程**:
1. 4路ILP累加器消除数据依赖: `s0 = fmaf(a, b, fmaf(c, d, s0))` 
2. 每次循环迭代处理8个元素 × 32个lane = 256个元素
3. 最终 `sum = s0+s1+s2+s3` + warp shuffle归约

### 4.5 `batched_gemm_pure_fp16_kernel<256>` — Batched FP16 (ViT关键路径)

**Grid/Block 配置**: `<<<dim3(K, batch_size), 256>>>`

**Global Memory访问**:
- **核心升级**: 从 `half2`（4字节/2元素）→ `float4`（16字节/8元素），带宽利用率提升4倍
- 双向 `__ldg()`: 权重和batch_input均走只读缓存

**这是 ViT 提速75%的关键**:
- ViT处理时 batch_size 较大（数百个patch token）
- 每个block处理 weight[row] × batch_input[batch_idx]
- float4宽加载显著减少了内存事务数量
- 消除 `sdata` 中间变量减少了共享内存读写

---

## 五、优化前后模型性能对比

### 5.1 Qwen3-VL-8B (ViT + LLM)

| 阶段 | 参考项目 | 优化后 | 提升 |
|---|---|---|---|
| **ViT Prefill** | 385 tokens/s | **676 tokens/s** | **+75.6%** |
| LLM Decode | 9.32 tokens/s | 9.58 tokens/s | **+2.8%** |
| 总时间 | ~29.5s | ~27.9s | **-5.4%** |

### 5.2 Qwen3-8B FP16

| 阶段 | 参考项目 | 优化后 | 提升 |
|---|---|---|---|
| Prefill | 136.3 tokens/s | 142.6 tokens/s | **+4.6%** |
| Decode | 10.17 tokens/s | 10.21 tokens/s | +0.4% |

### 5.3 Qwen3-8B AWQ (INT8)

| 阶段 | 参考项目 | 优化后 | 提升 |
|---|---|---|---|
| Prefill | 131.5 tokens/s | 131.3 tokens/s | ≈0% |
| Decode | 9.20 tokens/s | 9.23 tokens/s | +0.3% |

> AWQ decode使用 INT8 kernel (char4优化+1.08x)，但端到端受限于其他计算开销

### 5.4 Qwen2.5-7B FP16

| 阶段 | 参考项目 | 优化后 | 提升 |
|---|---|---|---|
| Prefill | 149.6 tokens/s | 152.7 tokens/s | **+2.1%** |
| Decode | 10.92 tokens/s | 10.95 tokens/s | +0.3% |

### 5.5 Qwen2.5-7B INT8

| 阶段 | 参考项目 | 优化后 | 提升 |
|---|---|---|---|
| Prefill | 6.09 tokens/s | 6.09 tokens/s | ≈0% |
| Decode | 5.72 tokens/s | 5.71 tokens/s | ≈0% |

> INT8模型受限于整体pipeline（embedding、attention等），matmul优化对端到端影响有限

---

## 六、代码清理

### 6.1 移除的死代码（5个device kernel）

| Kernel | 原因 |
|---|---|
| `optimized_gemv_kernel` | 无host函数调用，FP32 warp-based替代方案 |
| `gemv_large_m_kernel` | 无host函数调用，FP32 shared memory方案 |
| `gemv_fp16_weight_kernel` | 被 `optimized::gemv_fp16_optimized` 替代 |
| `gemv_fp16_weight_large_m_kernel` | 被 `optimized::gemv_fp16_bandwidth_optimized` 替代 |
| `gemv_pure_fp16_kernel` | 被 `gemv_pure_fp16_kernel_v2` 替代 |

### 6.2 代码行数变化

| 文件 | 优化前 | 优化后 | 变化 |
|---|---|---|---|
| matmul_kernel.cu | 1156行 | 918行 | **-238行 (-20.6%)** |

### 6.3 保留的活代码（8个device kernel + 2个optimized kernel）

| 类别 | Kernel | 调用方 |
|---|---|---|
| FP32 | `matmul_kernel_cu_fp32<128,1>` | `matmul_kernel_cu()` |
| INT8 | `matmul_kernel_cu_fp32int8<128,1>` | `matmul_kernel_cu_qint8()` |
| FP32 Batched | `batched_matmul_kernel_cu_fp32<128>` | `batched_matmul_kernel_cu()` |
| FP16w Mixed | `batched_matmul_fp16_weight_kernel<256>` | `batched_matmul_kernel_cu_fp16_weight()` fallback |
| FP16↔FP32 | `fp32_to_fp16_kernel`, `fp16_to_fp32_kernel` | cuBLAS转换路径 |
| Pure FP16 | `gemv_pure_fp16_kernel_v2<32,8>` | `matmul_kernel_cu_pure_fp16()` |
| Pure FP16 Batched | `batched_gemm_pure_fp16_kernel<256>` | cuBLAS fallback |
| FP16→FP32 | `gemv_fp16_input_fp16_weight_fp32_output<8,4>` | `matmul_kernel_cu_fp16_input_fp16_weight()` |
| Optimized Mixed | `optimized::gemv_fp16_bandwidth_optimized` | M≥2048 decode |
| Optimized Mixed | `optimized::gemv_fp16_optimized` | M<2048 decode |

---

## 七、优化总结

### 核心优化技术

| 技术 | 影响 | 适用场景 |
|---|---|---|
| `__ldg()` 只读缓存 | 减少L1数据缓存污染 | 所有权重/scale加载 |
| `fmaf()` 融合乘加 | 减少指令数、提高精度 | 所有点积运算 |
| `char4` INT8向量化 | 4倍减少内存事务 | INT8量化kernel |
| `float4` 宽加载(8 halfs) | 4倍提升FP16带宽利用 | batched FP16 kernel |
| 消除`sdata`共享内存 | 减少延迟、节省smem | block-reduce kernel |
| 4路ILP累加器 | 打破数据依赖链 | warp-based GEMV |
| `__restrict__` 指针 | 编译器更激进优化 | 全部kernel |

### 关键发现

1. **ViT处理受益最大** (+75.6%): `batched_gemm_pure_fp16_kernel` 的 `half2→float4` 升级在large batch时效果显著
2. **FP32 GEMV 改进显著** (1.45-1.60x): `__ldg()` + 消除共享内存中间变量是主要贡献
3. **INT8 一致改进** (1.08x): `char4` 向量化在所有维度下稳定提升
4. **Decode阶段受限于带宽**: 单token GEMV是memory-bound，Orin已接近带宽极限
5. **Prefill走cuBLAS路径**: 大batch时会走Tensor Core，自定义kernel优化对fallback路径有效
