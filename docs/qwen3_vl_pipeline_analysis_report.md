# Qwen3-VL 推理流水线源码深度分析报告

> 基于 `/mnt/ssd/workspace/OrinMLLM/kuiper/source/model/qwen3_vl.cpp` 源码分析  
> 生成日期：2026-03-03

---

## 目录

1. [Image Processing 流程分析](#1-image-processing-流程分析)
   - 1.1 [smart_resize](#11-smart_resize)
   - 1.2 [normalize_to_tensor](#12-normalize_to_tensor)
   - 1.3 [extract_patches_cu](#13-extract_patches_cu)
2. [encode_image 流程分析](#2-encode_image-流程分析)
   - 2.1 [Patch Embedding](#21-patch-embedding)
   - 2.2 [pos_embed_interpolate_cu](#22-pos_embed_interpolate_cu)
   - 2.3 [fused_split_rope_transpose_cu](#23-fused_split_rope_transpose_cu)
   - 2.4 [vision_merger](#24-vision_merger)
   - 2.5 [fused_multimodal_embed_cu](#25-fused_multimodal_embed_cu)
3. [Prefill 流程分析](#3-prefill-流程分析)
   - 3.1 [batched_mrope (M-RoPE)](#31-batched_mrope-m-rope)
   - 3.2 [DeepStack](#32-deepstack)
4. [DeepStack 原理详解](#4-deepstack-原理详解)

---

## 1. Image Processing 流程分析

Image Processing 是 Qwen3-VL 推理流水线的第一步，负责将原始图像文件转化为 Vision Encoder 可处理的 patch 张量。其入口函数为 `Qwen3VLModel::preprocess_image()`（第 1437-1479 行），整体流程如下：

```
原始图像文件
  ↓ stbi_load()                     → uint8 像素数组 [H, W, 3] (HWC)
  ↓ smart_resize()                  → uint8 像素数组 [H', W', 3]，保证 H'、W' 可被 patch_size 整除
  ↓ normalize_to_tensor()           → FP16 张量 [3, H', W'] (CHW)，已归一化到 [-1, 1]
  ↓ image_to_patches() → extract_patches_cu()  → FP16 张量 [num_patches, patch_dim]
```

### 1.1 smart_resize

**代码位置**：第 167-205 行

**原理**：
`smart_resize` 的目标是把图像缩放到一个合适的尺寸，使其满足以下约束：
1. **高宽均可被 `factor`（=patch_size=16）整除**，保证能被完整地切分为 patch。
2. **总像素数在 `[min_pixels, max_pixels]` 范围内**，平衡 ViT 推理速度与图像质量。

**算法步骤**（与 HuggingFace 官方实现一致）：

```
Step 1: 将 H、W 四舍五入到最近的 factor 倍数
        h_bar = round(H / factor) * factor
        w_bar = round(W / factor) * factor
        h_bar = max(h_bar, factor),  w_bar = max(w_bar, factor)

Step 2: 如果 h_bar * w_bar > max_pixels → 需要缩小
        β = sqrt(H * W / max_pixels)
        h_bar = max(factor, floor(H / β / factor) * factor)
        w_bar = max(factor, floor(W / β / factor) * factor)

Step 3: 如果 h_bar * w_bar < min_pixels → 需要放大
        β = sqrt(min_pixels / (H * W))
        h_bar = ceil(H * β / factor) * factor
        w_bar = ceil(W * β / factor) * factor
```

**代码实现**：
```cpp
int h_bar = static_cast<int>(std::round(static_cast<float>(src_height) / factor)) * factor;
int w_bar = static_cast<int>(std::round(static_cast<float>(src_width) / factor)) * factor;
h_bar = std::max(h_bar, factor);
w_bar = std::max(w_bar, factor);

if (h_bar * w_bar > max_pixels) {
    float beta = std::sqrt(static_cast<float>(src_height * src_width) / max_pixels);
    h_bar = std::max(factor, static_cast<int>(std::floor(src_height / beta / factor)) * factor);
    w_bar = std::max(factor, static_cast<int>(std::floor(src_width / beta / factor)) * factor);
} else if (h_bar * w_bar < min_pixels) {
    float beta = std::sqrt(static_cast<float>(min_pixels) / (src_height * src_width));
    h_bar = static_cast<int>(std::ceil(src_height * beta / factor)) * factor;
    w_bar = static_cast<int>(std::ceil(src_width * beta / factor)) * factor;
}
```

最终使用 `stbir_resize_uint8_linear()` 将图像缩放到 `(w_bar, h_bar)` 尺寸。

**默认参数（在 `preprocess_image` 调用处）**：
- `factor = 16`（即 `patch_size`）
- `min_pixels = 56 * 56 = 3136`
- `max_pixels = 1003520`（默认值，可通过参数调整）

### 1.2 normalize_to_tensor

**代码位置**：第 207-243 行

**原理**：
将 `uint8` 像素值转换为归一化后的 FP16 张量，同时完成 **HWC → CHW 格式变换**。

**归一化公式**：
$$\text{output}[c, h, w] = \frac{\text{pixel}[h, w, c] / 255.0 - \mu_c}{\sigma_c}$$

其中 Qwen3-VL 使用简单的 `mean = [0.5, 0.5, 0.5]`, `std = [0.5, 0.5, 0.5]`，即：
$$\text{output} = \frac{\text{pixel} / 255.0 - 0.5}{0.5} = \frac{\text{pixel}}{127.5} - 1.0$$

归一化后像素值范围为 $[-1, 1]$。

**实现流程**：
1. 在 CPU 侧创建 FP32 buffer `normalized[3 * H * W]`
2. 三重遍历 `(c, h, w)` 完成 HWC → CHW 重排 + 归一化
3. 逐元素转换 FP32 → FP16（使用 `float_to_half()` 位操作函数）
4. `cudaMemcpy` 将 FP16 数据拷贝到 GPU

**输出**：`tensor::Tensor [3, H', W']`（FP16，on GPU）

### 1.3 extract_patches_cu

**代码位置**：第 245-292 行（`image_to_patches` 函数）；CUDA kernel 声明在 `fused_kernels.cuh` 第 88-114 行

**原理**：
将归一化后的图像张量 `[3, H, W]` 切分为若干 patch，每个 patch 的维度为：
$$\text{patch\_dim} = C \times T_p \times P \times P = 3 \times 2 \times 16 \times 16 = 1536$$

其中 $T_p = 2$ 是时间维度（对于静态图像，会复制一帧填充为两帧），$P = 16$ 是 patch 空间尺寸。

**关键：2×2 block interleaved order**

Patch 不是按照简单的行扫描顺序排列的，而是按照 **2×2 block 交错顺序**排列，与 HuggingFace Qwen3-VL 的 `spatial_merge_size=2` 对应：

```
对于 grid_h=4, grid_w=4 的图像：
  Block (0,0): patch 0=(0,0), patch 1=(0,1), patch 2=(1,0), patch 3=(1,1)
  Block (0,1): patch 4=(0,2), patch 5=(0,3), patch 6=(1,2), patch 7=(1,3)
  Block (1,0): patch 8=(2,0), patch 9=(2,1), patch 10=(3,0), patch 11=(3,1)
  Block (1,1): patch 12=(2,2), patch 13=(2,3), patch 14=(3,2), patch 15=(3,3)
```

这种排列方式是为了后续 **spatial merge** 操作能直接将连续 4 个 patch 合并为 1 个 token，无需额外的重排操作。

**代码实现**：
```cpp
kernel::extract_patches_cu(
    image_tensor,   // [3, H, W] on GPU
    patches,        // [num_patches, patch_dim] on GPU
    channels, height, width,
    patch_size,            // 16
    temporal_patch_size,   // 2
    stream
);
```

整个 patch 提取在 GPU 上完成，避免了 D2H + CPU 处理 + H2D 的拷贝开销。

**输出**：`tensor::Tensor [num_patches, 1536]`（FP16，on GPU），其中 `num_patches = grid_h × grid_w`

---

## 2. encode_image 流程分析

`encode_image()` 是 Vision Encoder 的核心函数（第 1485-1631 行），将 patch 张量转化为可以注入 LLM 的视觉嵌入。整体流程：

```
pixel_values [num_patches, 1536]
  ↓ vision_patch_embed()                  → [num_patches, 1152]   Patch Embedding
  ↓ vision_add_pos_embed()                → [num_patches, 1152]   + 位置编码
  ↓ compute_vision_rotary_emb()           → cos/sin cache         RoPE 预计算
  ↓ 27× vision_transformer_block()        → [num_patches, 1152]   Transformer 层
      ├─ LayerNorm → QKV投影 → fused_split_rope_transpose_cu
      ├─ Attention → Output Projection → Residual
      └─ LayerNorm → MLP → Residual
      (在第 8/16/24 层输出 deepstack 特征)
  ↓ vision_merger()                       → [num_vision_tokens, 4096]  Main merger
```

### 2.1 Patch Embedding

**代码位置**：第 1636-1669 行（`vision_patch_embed`）

**原理**：
Patch Embedding 等价于一个 Conv3D 操作，将每个 patch 的原始像素特征映射到 `hidden_size=1152` 维的嵌入空间。由于 patch 已经被展平为 `[num_patches, patch_dim]` 格式，Conv3D 退化为一个简单的矩阵乘法：

$$\text{output} = \text{pixel\_values} \times W^T + b$$

其中：
- $W \in \mathbb{R}^{1152 \times 1536}$（`patch_embed_weight`）
- $b \in \mathbb{R}^{1152}$（`patch_embed_bias`）

**代码实现**：使用 cuBLAS `cublasHgemm` 完成 FP16 矩阵乘法，然后通过 `bias_add_residual_layer_` 添加 bias。

```cpp
// C = A @ B^T  →  [num_patches, 1152] = [num_patches, 1536] @ [1152, 1536]^T
cublasHgemm(cuda_config_->cublas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            hidden_size, num_patches, patch_dim,
            &alpha,
            vision_layers_->patch_embed_weight.ptr<half>(), patch_dim,
            image_data.pixel_values.ptr<half>(), patch_dim,
            &beta,
            output.ptr<half>(), hidden_size);
```

**输出**：`[num_patches, 1152]`

### 2.2 pos_embed_interpolate_cu

**代码位置**：第 1673-1694 行（`vision_add_pos_embed`）；kernel 声明在 `vision_encoder_kernel.cuh` 第 133-150 行

**原理**：
Qwen3-VL 的 Vision Encoder 使用**可学习的绝对位置编码**，预训练时对应一个 $48 \times 48 = 2304$ 的基网格。当实际图像的 grid 尺寸与预训练不同时，需要通过**双线性插值**将位置编码自适应到当前尺寸。

**算法**：
1. 基础位置编码表：`pos_embed_weight` 形状为 `[2304, 1152]`，对应 48×48 的空间网格
2. 对于输入图像的实际网格 `(grid_h, grid_w)`，计算每个 patch 在原始 48×48 空间中的浮点坐标
3. 使用**双线性插值**从基础表中获取对应位置的嵌入向量
4. 将插值结果与 patch embedding 相加

**关键细节**：插值时需考虑 `spatial_merge_size`，因为 patch 按 2×2 block 交错排列，位置坐标的映射需与此排列一致。

**代码实现**：
```cpp
vision_vl_layers_.pos_embed_interpolate_layer_->forward(
    patch_embeds,                          // [num_patches, 1152]
    vision_layers_->pos_embed_weight,      // [2304, 1152]
    output,                                // [num_patches, 1152]
    grid_h, grid_w, grid_t,
    num_grid_per_side,                     // 48 = sqrt(2304)
    vl_config_.vision.spatial_merge_size,  // 2
    cuda_config_->stream);
```

**输出**：`[num_patches, 1152]`（patch embedding + 位置编码）

### 2.3 fused_split_rope_transpose_cu

**代码位置**：Vision Transformer Block 内（第 1879-1891 行）；kernel 声明在 `vision_encoder_kernel.cuh` 第 182-192 行

**原理**：
这是一个**三合一融合 kernel**，将 3 个操作合并为 1 次 kernel launch，显著减少全局内存访问：

| 步骤 | 操作 | 输入 → 输出 |
|------|------|------------|
| 1 | **Split QKV** | `[N, 3H]` → Q `[N, H]`, K `[N, H]`, V `[N, H]` |
| 2 | **Apply RoPE** | 对 Q 和 K 应用旋转位置编码 |
| 3 | **Transpose** | `[N, H]` → `[num_heads, N, head_dim]`（为 batched GEMM 准备） |

**Vision RoPE 的特殊性**：
Vision Encoder 使用的 RoPE 与 LLM 不同：
- **基频 θ = 10000.0**（LLM 使用 5000000.0）
- **2D 位置编码**：每个 patch 的位置由 `(height, width)` 两个坐标决定
- **head_dim = 72** 的布局为：`[h_freq(18), w_freq(18), h_freq(18), w_freq(18)]`

RoPE 的核心计算为：
$$q'_{2i} = q_{2i} \cos\theta_i - q_{2i+1} \sin\theta_i$$
$$q'_{2i+1} = q_{2i+1} \cos\theta_i + q_{2i} \sin\theta_i$$

**代码实现**：
```cpp
vision_vl_layers_.fused_split_rope_transpose_layer_->forward(
    ws.qkv, cos_cache, sin_cache,
    vision_workspace_->q_transposed,   // [16, N, 72]
    vision_workspace_->k_transposed,   // [16, N, 72]
    vision_workspace_->v_transposed,   // [16, N, 72]
    num_tokens, num_heads, head_dim,
    cuda_config_->stream);
```

**性能意义**：将原本 5 次 kernel launch（split Q/K/V + RoPE Q + RoPE K）合并为 1 次，减少了多次全局内存遍历。

### 2.4 vision_merger

**代码位置**：第 1928-1988 行

**原理**：
Vision Merger 将 ViT 输出的 patch 特征投影到 LLM 能理解的维度。核心是**空间合并 + MLP 投射**：

```
[num_patches, 1152]
  ↓ LayerNorm                                   -- 对每个 patch 归一化
  ↓ Spatial Merge (2×2 → 1)                     -- 4 个相邻 patch 拼接
[num_vision_tokens, 4608]                         -- 4608 = 1152 × 4
  ↓ fc1 (Linear 4608 → 4608 + GELU)
  ↓ fc2 (Linear 4608 → 4096)
[num_vision_tokens, 4096]                         -- 匹配 LLM hidden_size
```

**Spatial Merge 步骤**：
1. `LayerNorm([num_patches, 1152])`：对每个 patch 特征做 Layer Normalization
2. `spatial_merge_cu`：将 2×2 空间相邻的 patch 合并为 1 个 token
   - 由于 patch 已按 2×2 block 交错排列（见 1.3 节），每连续 4 个 patch 天然就是一个 2×2 block
   - 直接拼接为 `[num_vision_tokens, 4 × 1152] = [num_vision_tokens, 4608]`
3. `MLP(fc1 + GELU + fc2)`：两层前馈网络，将 `4608 → 4608 → 4096`

**Main Merger vs Deepstack Merger 的区别**：
- **Main Merger**：在第 27 层（最后一层）输出后执行，LayerNorm 的维度为 `1152`（对原始 patch 维度归一化后再 merge）
- **Deepstack Merger**：在第 8/16/24 层输出后执行，LayerNorm 的维度为 `4608`（先 merge 再归一化），使用独立的权重

**代码实现**：
```cpp
// 1. LayerNorm
vision_vl_layers_.layernorm_with_bias_layer_->forward(
    hidden_states, merger->norm_weight, merger->norm_bias, 
    normed, 1e-6f, cuda_config_->stream);

// 2. Spatial merge: [num_patches, 1152] → [num_vision_tokens, 4608]
vision_vl_layers_.spatial_merge_layer_->forward(
    normed, merged, grid_t, grid_h, grid_w, 
    hidden_size, merge_size, cuda_config_->stream);

// 3. MLP: [num_vision_tokens, 4608] → [num_vision_tokens, 4096]
vision_vl_layers_.vision_merger_mlp_layer_->forward(
    merged, merger->fc1_weight, merger->fc1_bias,
    merger->fc2_weight, merger->fc2_bias,
    output, intermediate, cuda_config_.get());
```

### 2.5 fused_multimodal_embed_cu

**代码位置**：第 2057-2068 行（在 `prepare_multimodal_embeddings` 中调用）；kernel 声明在 `fused_kernels.cuh` 第 27-40 行

**原理**：
将文本嵌入和视觉嵌入**拼接**为统一的多模态嵌入序列。此操作替换了原本需要 3 次 `cudaMemcpyAsync` 的方案，用 1 个 CUDA kernel 完成。

**拼接逻辑**：

假设文本 token 序列中第 `p` 位置是 `<image_pad>` 占位符：

```
输入：
  text_embeds:   [t0, t1, ..., image_pad, t_{p+1}, ...]    shape: [text_len, dim]
  visual_embeds: [v0, v1, ..., v_{N-1}]                     shape: [N, dim]

输出（output）：
  [t0, t1, ..., t_{p-1}, v0, v1, ..., v_{N-1}, t_{p+1}, ...]   shape: [text_len - 1 + N, dim]
```

即：
- `output[0 : p]` ← `text_embeds[0 : p]`（图像前的文本）
- `output[p : p+N]` ← `visual_embeds[0 : N]`（视觉 token）
- `output[p+N :]` ← `text_embeds[p+1 :]`（图像后的文本，跳过占位符）

**代码实现**：
```cpp
vision_vl_layers_.fused_multimodal_embed_layer_->forward(
    embed_out.input_embeddings,  // text embeddings [tokens.size(), dim]
    visual_embeds,               // vision embeddings [num_vision_tokens, dim]
    multimodal_embeds,           // output [new_seq_len, dim]
    image_token_pos,             // 占位符位置
    num_vision_tokens,           // 视觉 token 数量
    static_cast<int>(tokens.size()),
    dim,
    cuda_config_->stream);
```

同时，此函数还会生成 **M-RoPE 3D 位置编码**，为后续 prefill 做准备（详见 3.1 节）。

---

## 3. Prefill 流程分析

Prefill 是 LLM 推理的关键阶段，一次性处理整个输入序列（包含文本和视觉 token）。入口函数为 `Qwen3VLModel::prefill()`（第 2169-2335 行）。

**整体流程**：

```
multimodal_embeddings [seq_len, 4096]
  ↓ 上传 M-RoPE 位置数组到 GPU
  ↓ 预分配所有 buffer（双缓冲 hidden state + QKV + FFN 中间结果）
  ↓ 36× Transformer Layer:
      ├─ RMSNorm
      ├─ Q/K/V投影 + q_norm/k_norm + batched_mrope + KV cache 更新
      ├─ Flash Attention (prefill)
      ├─ WO投影 + 残差连接
      ├─ FFN (RMSNorm + W1/W3 + SwiGLU + W2 + 残差)
      └─ DeepStack: 前 N 层添加视觉深层特征
  ↓ 取最后一个 token 的 hidden state → cls_logits → 采样
```

### 3.1 batched_mrope (M-RoPE)

**代码位置**：
- 位置生成：第 2075-2130 行（在 `prepare_multimodal_embeddings` 中）
- Kernel 调用：第 2834-2848 行（在 `batched_attention_qkv` 中）
- Kernel 实现：`rope_kernel.cuh` 第 73-85 行；`ncu_profile_rope.cu` 第 316-378 行

**原理**：
M-RoPE（Multimodal Rotary Position Embedding）是 Qwen3-VL 的核心创新之一。标准 RoPE 只有一维位置信息，但多模态场景下需要对**时间、高度、宽度**三个维度分别编码位置。

**M-RoPE 的 3D 位置分配**：

每个 token 的 128 维 head_dim 被拆分为 3 段，分别使用不同维度的位置 ID：

| 段 | 维度范围 | 对数（pairs） | 位置来源 |
|----|----------|--------------|----------|
| Section 0 | `[0, 48)` | 24 pairs | `pos_t`（时间/序列位置） |
| Section 1 | `[48, 88)` | 20 pairs | `pos_h`（高度位置） |
| Section 2 | `[88, 128)` | 20 pairs | `pos_w`（宽度位置） |

**位置 ID 的生成规则**：

```
文本 token（图像之前）：pos_t = pos_h = pos_w = seq_pos  (连续递增)
视觉 token：
  pos_t = visual_base_t               (所有视觉 token 共享同一时间位置)
  pos_h = visual_base_t + row          (行索引，范围 [0, merged_grid_h))
  pos_w = visual_base_t + col          (列索引，范围 [0, merged_grid_w))
文本 token（图像之后）：pos_t = pos_h = pos_w = visual_base_t + max(grid_h, grid_w) + offset
```

这种设计使得：
- 视觉 token 在空间维度上有合理的 2D 位置编码
- 所有 token 在时间维度上保持因果序列的连续性
- 文本 token 的三维位置退化为一维（三个位置相同）

**Kernel 实现** (`batched_mrope_kernel_cu_fp16_impl`)：

```
对于第 seq_idx 个 token, 第 head_idx 个 attention head 的第 pair_idx 对:
  1. 根据 pair_idx 所在的 section 确定使用哪个位置:
     d0 < section0*2  → pos = pos_t
     d0 < section0*2 + section1*2  → pos = pos_h
     否则 → pos = pos_w
  
  2. 从预计算的 sin/cos cache 中查表:
     sin0 = sin_cache[pos * head_size + freq_idx]
     cos0 = cos_cache[pos * head_size + freq_idx]
  
  3. 应用旋转变换:
     q'[d0] = q[d0] * cos0 - q[d1] * sin0
     q'[d1] = q[d1] * cos1 + q[d0] * sin1
     (K 同理，但仅对每个 KV head 的第一个 Q head 处理)
```

**代码调用**：
```cpp
qwen_layers_->batched_mrope_layer_->forward(
    seq_len, config_->dim_, config_->kv_dim_, config_->head_size_,
    section0, section1, section2,         // 24, 20, 20
    mrope_pos_t_gpu_ + start_pos,         // GPU 上的位置数组
    mrope_pos_h_gpu_ + start_pos,
    mrope_pos_w_gpu_ + start_pos,
    query_out, key_out,
    get_buffer(ModelBufferType::kSinCache),
    get_buffer(ModelBufferType::kCosCache));
```

**优化**：M-RoPE 位置数组使用 **pinned memory + 单次异步传输**上传到 GPU。将原本 3 次独立的 H2D 拷贝合并为 1 次连续传输（第 2208-2222 行）。

### 3.2 DeepStack

**代码位置**：第 2297-2308 行（在 prefill 主循环中）

**原理**：
DeepStack 是 Qwen3-VL 引入的多尺度视觉特征融合机制。在 LLM 的前 N 层（N = deepstack 特征数量 = 3），将 Vision Encoder 中间层输出的特征**加到对应的 LLM hidden state 上**。

**实现**：
```cpp
// 在 prefill 循环中，每个 LLM layer 的最后：
if (layer_idx < num_deepstack_layers && visual_pos_start_ >= 0) {
    int num_visual_tokens = visual_pos_end_ - visual_pos_start_;
    const auto& ds_feat = deepstack_features_[layer_idx];
    
    // 仅对视觉 token 位置添加 deepstack 特征
    // layer_output[visual_pos_start_:visual_pos_end_] += ds_feat
    half* hidden_ptr = layer_output->ptr<half>() + visual_pos_start_ * dim;
    const half* ds_ptr = ds_feat.ptr<half>();
    
    STATUS_CHECK(qwen_layers_->batched_add_layer_->forward_raw(
        hidden_ptr, ds_ptr, hidden_ptr, num_visual_tokens * dim));
}
```

**详细分析见第 4 节。**

---

## 4. DeepStack 原理详解

### 4.1 什么是 DeepStack？

DeepStack 是 Qwen3-VL 相较于 Qwen2-VL 的重要架构升级，旨在解决**单一视觉特征表征不足**的问题。

在传统 VLM 中，视觉编码器仅使用最后一层的输出作为视觉特征，传递给语言模型。但研究表明，ViT 的不同层捕获了不同抽象层级的信息：
- **浅层**（如第 8 层）：捕获低级视觉特征（边缘、纹理、颜色）
- **中层**（如第 16 层）：捕获中级语义特征（局部结构、部件）
- **深层**（如第 24 层）：捕获高级语义特征（物体、场景理解）

仅使用最后一层会丢失大量底层和中层视觉信息。

### 4.2 DeepStack 的工作流程

DeepStack 的完整工作流程跨越 `encode_image` 和 `prefill` 两个阶段：

**阶段一：在 Vision Encoder 中提取中间层特征**（`encode_image`，第 1578-1597 行）

在 27 层 Vision Transformer 的前向传播过程中，在预设的 3 个中间层（第 8、16、24 层）处"截取"特征：

```
Layer 0 → Layer 7 → [Layer 8: 提取 deepstack_feature_0]
→ Layer 9 → ... → Layer 15 → [Layer 16: 提取 deepstack_feature_1]
→ Layer 17 → ... → Layer 23 → [Layer 24: 提取 deepstack_feature_2]
→ Layer 25 → Layer 26 → [最终输出: main_output]
```

每个 deepstack 特征通过各自的 **Deepstack Merger** 处理：
```
中间层输出 [num_patches, 1152]
  ↓ Spatial Merge (2×2 → 1)
[num_vision_tokens, 4608]
  ↓ LayerNorm（注意：deepstack 的 LayerNorm 在 merge 之后）
  ↓ fc1 (4608 → 4608, GELU)
  ↓ fc2 (4608 → 4096)
[num_vision_tokens, 4096]
```

注意 Deepstack Merger 的 3 组权重是**各自独立**的，不与 Main Merger 共享。

**阶段二：在 LLM prefill 中注入多尺度特征**（`prefill`，第 2297-2308 行）

在 LLM 的前 3 层（layer 0, 1, 2）中，将对应的 deepstack 特征**逐元素加到**视觉 token 位置的 hidden state 上：

```
LLM Layer 0:
  hidden_state[visual_pos_start:visual_pos_end] += deepstack_features_[0]  (来自 ViT Layer 8)

LLM Layer 1:
  hidden_state[visual_pos_start:visual_pos_end] += deepstack_features_[1]  (来自 ViT Layer 16)

LLM Layer 2:
  hidden_state[visual_pos_start:visual_pos_end] += deepstack_features_[2]  (来自 ViT Layer 24)

LLM Layer 3 ~ 35:
  无 deepstack 操作，正常 Transformer 处理
```

### 4.3 为什么需要 DeepStack？

**1. 多尺度特征融合**

不同 ViT 层提供不同抽象层级的视觉信息：

| 来源层 | 特征层级 | 作用 |
|--------|---------|------|
| ViT Layer 8 → LLM Layer 0 | 低级特征 | 精确的空间信息、纹理细节、视觉 grounding |
| ViT Layer 16 → LLM Layer 1 | 中级特征 | 物体部件、局部语义关系 |
| ViT Layer 24 → LLM Layer 2 | 高级特征 | 全局语义理解、场景级信息 |
| ViT Layer 26（main） → 嵌入层 | 最终特征 | 综合视觉表征（作为基础嵌入） |

**2. 渐进式注入策略**

将多尺度特征注入到 LLM 的**浅层**，符合以下直觉：
- LLM 的浅层主要处理表征融合和对齐
- LLM 的深层更关注语义推理和生成
- 在浅层注入丰富的视觉信号，让后续层有更多信息可供推理

**3. 计算效率**

- DeepStack 的额外计算仅限于 3 个 Merger MLP 和 3 次向量加法
- 不需要修改 ViT 的任何结构（特征是正常前向传播的副产品）
- 不增加 LLM 的序列长度（特征加到已有的视觉 token 上，而非拼接新 token）
- 实际开销可忽略不计，但提供了显著更丰富的视觉信息

### 4.4 代码中的关键数据结构

```cpp
// 配置：指定从 ViT 哪些层提取特征
struct Qwen3VLVisionConfig {
    std::vector<int32_t> deepstack_visual_indexes = {8, 16, 24};
    // ...
};

// 权重：每个 deepstack 层有独立的 Merger 权重
struct Qwen3VLVisionLayers {
    Merger merger;                           // Main merger（最终层）
    std::vector<Merger> deepstack_mergers;   // 3 个 deepstack merger
};

// 运行时中间结果
class Qwen3VLModel {
    mutable std::vector<tensor::Tensor> deepstack_features_;  // [num_vision_tokens, 4096] × 3
    mutable int visual_pos_start_ = -1;  // 视觉 token 在序列中的起始位置
    mutable int visual_pos_end_ = -1;    // 视觉 token 在序列中的结束位置
};
```

### 4.5 总结

DeepStack 通过一种简洁而高效的方式实现了多尺度视觉特征的深度融合。其核心思想可以总结为：

> **在 Vision Encoder 的多个中间层"侧输出"特征，经过独立的 Merger 投影后，注入到 LLM 的浅层 hidden state 中，为语言模型提供从低级到高级的多粒度视觉信息。**

这种设计比简单拼接多尺度 token、交叉注意力等方案更轻量，且不增加序列长度开销，是 Qwen3-VL 在视觉理解能力上相对前代的重要改进。

---

## 附录：关键函数索引

| 函数/Kernel | 源码位置 | 作用 |
|-------------|---------|------|
| `preprocess_image()` | 第 1437-1479 行 | 图像预处理入口 |
| `smart_resize()` | 第 167-205 行 | 智能缩放 |
| `normalize_to_tensor()` | 第 207-243 行 | 归一化 + CHW 转换 |
| `extract_patches_cu()` | `fused_kernels.cuh` | GPU patch 提取 |
| `encode_image()` | 第 1485-1631 行 | Vision Encoder 入口 |
| `vision_patch_embed()` | 第 1636-1669 行 | Patch Embedding |
| `vision_add_pos_embed()` | 第 1673-1694 行 | 位置编码 |
| `pos_embed_interpolate_cu()` | `vision_encoder_kernel.cuh` | 位置编码插值 kernel |
| `fused_split_rope_transpose_cu()` | `vision_encoder_kernel.cuh` | QKV Split + RoPE + Transpose |
| `vision_merger()` | 第 1928-1988 行 | Spatial Merge + MLP |
| `fused_multimodal_embed_cu()` | `fused_kernels.cuh` | 多模态嵌入拼接 |
| `prepare_multimodal_embeddings()` | 第 1994-2144 行 | 多模态嵌入 + M-RoPE 位置 |
| `prefill()` | 第 2169-2335 行 | Prefill 主循环 |
| `batched_mrope_kernel_cu_fp16` | `rope_kernel.cuh` | M-RoPE kernel |
| `batched_attention_qkv()` | 第 2773-2859 行 | QKV + M-RoPE + KV cache |
