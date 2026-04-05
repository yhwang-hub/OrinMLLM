# Qwen3.5-9B-FP16 模型适配技术报告

> 平台：NVIDIA Jetson Orin AGX (sm_87, LPDDR5 170 GB/s)
>
> 源码工程：OrinMLLM
>
> 核心文件：`kuiper/source/model/qwen3_5.cpp`、`kuiper/include/model/qwen3_5.h`、`kuiper/source/op/kernels/cuda/gdn_kernel.cu`

---

## 目录

- [第一部分：Qwen3.5-9B-FP16 完整推理流程图](#第一部分qwen35-9b-fp16-完整推理流程图)
- [第二部分：适配过程详解——关键点与难点分析](#第二部分适配过程详解关键点与难点分析)
- [第三部分：ViT 视觉编码器流程与算子详解](#第三部分vit-视觉编码器流程与算子详解)
- [第四部分：Qwen3.5-9B 与 Qwen3-VL 的详细区别](#第四部分qwen35-9b-与-qwen3-vl-的详细区别)
- [第五部分：GDN (Gated Delta Networks) Linear Attention 详解](#第五部分gdn-gated-delta-networks-linear-attention-详解)

---

# 第一部分：Qwen3.5-9B-FP16 完整推理流程图

## 1.1 模型总体架构

Qwen3.5-9B 是一个**混合视觉-语言多模态模型**，其核心创新在于 LLM 部分采用了**混合注意力架构**：32 层 Transformer 中，8 层使用带 Output Gate 的 Full Attention（层索引 3,7,11,15,19,23,27,31），24 层使用 GDN (Gated Delta Net) Linear Attention。

### 模型全局配置（源码 `Qwen35TextConfig`，`qwen3_5.h:12-62`）

| 参数 | 值 | 说明 |
|------|------|------|
| hidden_size | 4096 | LLM 隐藏层维度 |
| intermediate_size | 12288 | FFN 中间维度 |
| num_hidden_layers | 32 | 总层数 |
| num_attention_heads | 16 | Q 头数（Full Attention） |
| num_key_value_heads | 4 | KV 头数（Full Attention，GQA） |
| head_dim | 256 | 每个注意力头维度 |
| vocab_size | 248320 | 词汇表大小 |
| rope_theta | 10,000,000 | RoPE 基频 |
| linear_num_key_heads | 16 | GDN K 头数 |
| linear_num_value_heads | 32 | GDN V 头数 |
| linear_key_head_dim | 128 | GDN K 头维度 |
| linear_value_head_dim | 128 | GDN V 头维度 |
| linear_conv_kernel_dim | 4 | GDN Conv1D 卷积核大小 |
| partial_rotary_factor | 0.25 | 部分 RoPE 比例 |
| mrope_section | [11, 11, 10] | M-RoPE 三维分段 |

### 计算维度推导（源码 `qwen3_5.h:46-54`）

```
q_dim     = num_attention_heads × head_dim        = 16 × 256 = 4096
kv_dim    = num_key_value_heads × head_dim        = 4 × 256  = 1024
q_gate_dim= 2 × q_dim                             = 8192（Q + OutputGate 交错）
conv_dim  = linear_num_key_heads × linear_key_head_dim × 2
          + linear_num_value_heads × linear_value_head_dim
          = 16 × 128 × 2 + 32 × 128               = 8192
partial_rope_dim = head_dim × partial_rotary_factor = 256 × 0.25 = 64
```

## 1.2 完整推理流程图（Decode 阶段，单 Token）

以下流程图对应 `qwen3_5.cpp` 中的 `decode_step_optimized()` → 逐层调用 `full_attn_decode()` 或 `linear_attn_decode()` → `q35_feed_forward()` → `q35_cls_logits()`。

```
输入: token_id (int32)
    │
    ▼
┌─────────────────────────────────────────────┐
│ Embedding Lookup                             │
│ embedding_layer_->forward()                  │
│ 输入: token_id [1]                           │
│ 权重: [248320, 4096] FP16                    │
│ 输出: decode_input [4096] FP16               │
└─────────────────────┬───────────────────────┘
                      │
        ┌─────────────┴──────────────┐
        │  FOR il = 0 to 31 (共32层)  │
        │  判断层类型:                 │
        │  Full Attn: il∈{3,7,11,...} │
        │  Linear Attn: 其余24层       │
        └─────────┬──────────┬───────┘
                  │          │
    ┌─────────────┘          └──────────────┐
    ▼ (Full Attention, 8层)                  ▼ (GDN Linear Attention, 24层)
```

### 1.2.1 Full Attention Decode 流程（`full_attn_decode()`，`qwen3_5.cpp:782-876`）

```
decode_input [4096] FP16
    │
    ▼
┌──────────────────────────────────────────────────┐
│ Step 1: RMSNorm (input_layernorm)                │
│ rmsnorm_layers_[layer_idx]->forward()            │
│ 输入: [4096] FP16                                │
│ 权重: [4096] FP16 (已+1.0偏置)                   │
│ 输出: rms_output [4096] FP16                     │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│ Step 2-3: Fused Q+K+V Projection (单次kernel)    │
│ fused_qkv_gemv_layer_->forward()                 │
│ 输入: rms_output [4096] FP16                     │
│ Q权重: [8192, 4096] FP16 (含Gate)                │
│ K权重: [1024, 4096] FP16                         │
│ V权重: [1024, 4096] FP16                         │
│ 输出: query_gate_buf [8192] FP16                 │
│       temp_key [1024] FP16                       │
│       temp_value [1024] FP16                     │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│ Step 3b: Q/Gate Deinterleave (逐头解交织)         │
│ deinterleave_q_gate_layer_->forward()            │
│ 输入: query_gate_buf [8192] FP16                 │
│   布局: [h0_q(256), h0_gate(256), h1_q(256), ...]│
│ 输出: full_attn_q_ [4096] FP16                  │
│       full_attn_gate_ [4096] FP16               │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│ Step 4: Per-Head Q Norm + K Norm (RMSNorm)       │
│ q_norm_layer->forward(query_view)                │
│ k_norm_layer->forward(temp_key)                  │
│ Q 输入: [16, 256] FP16 → [16, 256] FP16         │
│ K 输入: [4, 256] FP16 → [4, 256] FP16           │
│ 权重: [256] FP16 (逐头共享)                      │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│ Step 5: Partial M-RoPE (Interleaved)             │
│ partial_mrope_layer_->forward()                  │
│ 输入: Q [16×256], K [4×256] FP16                 │
│ sin/cos cache: [max_seq_len, 32] FP32            │
│ 仅旋转前 64 维 (partial_rope_dim=64)             │
│ M-RoPE sections: [11,11,10] 对→分配T/H/W位置     │
│ 输出: Q, K (原地修改前64维)                       │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│ Step 6: KV Cache Write                           │
│ cudaMemcpyAsync (D2D)                            │
│ K → key_cache[type_idx, pos, :] [1024] FP16      │
│ V → val_cache[type_idx, pos, :] [1024] FP16      │
│ Cache shape: [8, seq_len, 1024] FP16             │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│ Step 7: Flash Attention Decode                   │
│ flash_attention_decode_layer_->forward()         │
│ Q: [16, 256] FP16                                │
│ K_cache: [8, seq_len, 1024] → 扫描 [pos+1] 行    │
│ V_cache: [8, seq_len, 1024] → 扫描 [pos+1] 行    │
│ GQA: kv_mul=4 (每个KV头服务4个Q头)                │
│ 输出: mha_output [4096] FP16                     │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│ Step 8: Sigmoid Gate (Qwen3.5特有)               │
│ apply_sigmoid_gate_layer_->forward()             │
│ mha_output *= sigmoid(gate)                      │
│ 输入: mha_output [4096], gate [4096] FP16        │
│ 输出: mha_output [4096] FP16 (原地)              │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│ Step 9: Output Projection                        │
│ wo_layers_[type_idx]->forward()                  │
│ 输入: mha_output [4096] FP16                     │
│ 权重: [4096, 4096] FP16                          │
│ 输出: attn_output [4096] FP16                    │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
                   → FFN (见1.2.3)
```

### 1.2.2 GDN Linear Attention Decode 流程（`linear_attn_decode()`，`qwen3_5.cpp:966-1040`）

```
decode_input [4096] FP16
    │
    ▼
┌──────────────────────────────────────────────────┐
│ Step 1: RMSNorm (input_layernorm)                │
│ 同 Full Attention                                │
│ 输出: rms_output [4096] FP16                     │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│ Step 2: Fused QKV+Z GEMV + A/B Projection        │
│ fused_gdn_proj_gemv_layer_->forward()            │
│ 输入: rms_output [4096] FP16                     │
│ QKV权重: [8192, 4096] FP16                       │
│ Z权重:   [4096, 4096] FP16                       │
│ 输出: gdn_qkv_buf [8192] FP16                   │
│       gdn_z_buf   [4096] FP16                   │
│                                                  │
│ in_proj_a->forward(): [4096] → [32] FP16 (alpha) │
│ in_proj_b->forward(): [4096] → [32] FP16 (beta)  │
│ A权重: [32, 4096], B权重: [32, 4096] FP16        │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│ Step 3: Causal Conv1D + SiLU                     │
│ causal_conv1d_silu_layer_->forward()             │
│ 输入: gdn_qkv_buf [8192] FP16                   │
│ conv_state: [8192, 3] FP16 (kernel_size-1=3)    │
│ conv_weight: [8192, 4] FP16                      │
│ 操作: conv → shift_state → SiLU                  │
│ 输出: gdn_conv_out [8192] FP16                   │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│ Step 4: Split into Q, K, V (零拷贝指针分割)       │
│ Q = conv_out[0 : 2048]     → [16×128] FP16      │
│ K = conv_out[2048 : 4096]  → [16×128] FP16      │
│ V = conv_out[4096 : 8192]  → [32×128] FP16      │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│ Step 5: Per-Head L2 Normalize Q and K            │
│ l2_norm_per_head_layer_->forward()               │
│ Q: [16, 128] → L2Norm → [16, 128] FP16          │
│ K: [16, 128] → L2Norm → [16, 128] FP16          │
│ 公式: x_norm = x / sqrt(sum(x²) + eps)          │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│ Step 6: Compute GDN Gates                        │
│ compute_gdn_gates_layer_->forward()              │
│ 输入: alpha [32] FP16, dt_bias [32] FP16,       │
│       A_log [32] FP32, beta_raw [32] FP16        │
│ 计算:                                            │
│   alpha = alpha_raw + dt_bias                    │
│   softplus_alpha = log(1 + exp(alpha))           │
│   gate = exp(softplus_alpha × (−exp(A_log)))     │
│   beta = sigmoid(beta_raw)                       │
│ 输出: gate [32] FP32, beta [32] FP32            │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│ Step 7: Delta Net Decode Step                    │
│ gdn_decode_step_layer_->forward()                │
│ 输入: Q_norm [16×128], K_norm [16×128] FP16     │
│       V [32×128] FP16                            │
│       gate [32] FP32, beta [32] FP32             │
│       ssm_state [32, 128, 128] FP32              │
│ 算法 (每个 v_head):                              │
│   1. state' = state × gate                       │
│   2. kv_mem = state' @ k                         │
│   3. delta = beta × (v − kv_mem)                 │
│   4. state = state' + outer(k, delta)            │
│   5. output = state @ (q × 1/√k_dim)            │
│ 输出: gdn_attn_out [32×128=4096] FP16           │
│ 状态: ssm_state [32, 128, 128] FP32 (原地更新)   │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│ Step 8: Gated RMSNorm                            │
│ gated_rmsnorm_layer_->forward()                  │
│ 输入: attn_out [4096] FP16, z [4096] FP16       │
│ 权重: norm_weight [128] FP32 (逐head共享)        │
│ 公式: output = RMSNorm(attn_out) × SiLU(z)      │
│ 输出: gdn_normed_out [4096] FP16                │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│ Step 9: Output Projection                        │
│ out_proj->forward()                              │
│ 输入: gdn_normed_out [4096] FP16                │
│ 权重: [4096, 4096] FP16                          │
│ 输出: attn_output [4096] FP16                    │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
                   → FFN (见1.2.3)
```

### 1.2.3 Feed Forward Network（`q35_feed_forward()`，`qwen3_5.cpp:1042-1085`）

```
┌──────────────────────────────────────────────────┐
│ Residual Add #1                                  │
│ add_layer_->forward(input, attn_output, input)   │
│ input += attn_output                             │
│ [4096] + [4096] → [4096] FP16                   │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│ FFN RMSNorm (post_attention_layernorm)           │
│ rmsnorm_layers_[il + 32]->forward()              │
│ 输入: [4096] FP16 → 输出: ffn_norm [4096] FP16  │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│ Fused Gate+Up+SwiGLU (单次kernel)                │
│ fused_ffn_layer_->forward()                      │
│ 输入: ffn_norm [4096] FP16                       │
│ W1(gate): [12288, 4096] FP16                     │
│ W3(up):   [12288, 4096] FP16                     │
│ 计算: output = SiLU(W1 @ x) ⊙ (W3 @ x)          │
│ 输出: w1_out [12288] FP16                        │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│ Down Projection                                  │
│ w2_layers_[il]->forward()                        │
│ 输入: w1_out [12288] FP16                        │
│ W2(down): [4096, 12288] FP16                     │
│ 输出: w2_out [4096] FP16                         │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│ Residual Add #2                                  │
│ add_layer_->forward(input, w2_out, input)        │
│ input += w2_out                                  │
│ [4096] + [4096] → [4096] FP16                   │
└───────────────────────┘
```

### 1.2.4 CLS Logits（`q35_cls_logits()`，`qwen3_5.cpp:1087-1094`）

```
经过 32 层后的 decode_input [4096] FP16
    │
    ▼
┌──────────────────────────────────────────────────┐
│ Final RMSNorm                                    │
│ rmsnorm_layers_[64]->forward()                   │
│ 输入: [4096] FP16 → 输出: rms_out [4096] FP16   │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│ LM Head (Vocabulary Projection)                  │
│ cls_layer_->forward()                            │
│ 输入: rms_out [4096] FP16                        │
│ 权重: [248320, 4096] FP16                        │
│ 输出: logits [248320] FP32                       │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│ Argmax Sampling                                  │
│ sampler_->sample()                               │
│ 输入: logits [248320] FP32                       │
│ 输出: next_token_id (int32)                      │
└──────────────────────────────────────────────────┘
```

## 1.3 完整算子汇总表（Decode 单步）

| 算子名称 | 调用次数 | 输入维度 | 输出维度 | 权重维度 |
|----------|---------|---------|---------|---------|
| RMSNorm (input_layernorm) | 32 | [4096] | [4096] | [4096] |
| Fused QKV GEMV (Full Attn) | 8 | [4096]+3权重 | [8192]+[1024]+[1024] | [8192,4096]+[1024,4096]+[1024,4096] |
| Deinterleave Q/Gate | 8 | [8192] | [4096]+[4096] | - |
| Per-Head RMSNorm (Q) | 8 | [16,256] | [16,256] | [256] |
| Per-Head RMSNorm (K) | 8 | [4,256] | [4,256] | [256] |
| Partial M-RoPE | 8 | Q[16×256]+K[4×256] | 同输入(原地) | sin/cos cache |
| KV Cache Write | 8 | [1024]×2 | cache[pos] | - |
| Flash Attention Decode | 8 | Q[16,256]+KV cache | [4096] | - |
| Sigmoid Gate | 8 | [4096]×2 | [4096] | - |
| O Projection (Full Attn) | 8 | [4096] | [4096] | [4096,4096] |
| Fused QKV+Z GEMV (GDN) | 24 | [4096]+2权重 | [8192]+[4096] | [8192,4096]+[4096,4096] |
| A/B Projection (GDN) | 24×2 | [4096] | [32] | [32,4096] |
| Causal Conv1D + SiLU | 24 | [8192]+state | [8192] | [8192,4] |
| L2 Norm (Q and K) | 24×2 | [16,128] | [16,128] | - |
| Compute GDN Gates | 24 | [32]×4参数 | [32]×2 | A_log+dt_bias |
| Delta Net Decode Step | 24 | Q+K+V+gate+beta+state | [4096] | state[32,128,128] |
| Gated RMSNorm | 24 | [4096]+[4096] | [4096] | [128] FP32 |
| O Projection (GDN) | 24 | [4096] | [4096] | [4096,4096] |
| VecAdd (Residual) | 64 | [4096]+[4096] | [4096] | - |
| RMSNorm (FFN) | 32 | [4096] | [4096] | [4096] |
| Fused FFN (Gate+Up+SwiGLU) | 32 | [4096] | [12288] | [12288,4096]×2 |
| Down Projection | 32 | [12288] | [4096] | [4096,12288] |
| Final RMSNorm | 1 | [4096] | [4096] | [4096] |
| LM Head | 1 | [4096] | [248320] | [248320,4096] |

---

# 第二部分：适配过程详解——关键点与难点分析

## 2.1 适配总览

将 Qwen3.5-9B 适配到 OrinMLLM 推理框架，其核心挑战在于：该模型是首个引入**混合注意力（Full Attention + GDN Linear Attention）**的大型视觉语言模型。框架此前仅支持标准 Transformer 架构（Qwen2.5、Qwen3、Qwen3-VL），所有层都是同质的 Full Attention，而 Qwen3.5 有 75% 的层使用了全新的 GDN 线性注意力机制，需要从零实现。

### 适配步骤总览

```
Step 1: 模型结构分析与二进制格式设计
Step 2: 权重导出工具编写
Step 3: 继承架构设计 (Qwen35Model : Qwen3VLModel)
Step 4: 二进制模型加载实现
Step 5: GDN CUDA 算子实现 (14+ 个新算子)
Step 6: 内存布局与状态管理
Step 7: Decode 路径实现（Full Attention + GDN）
Step 8: Prefill 路径实现（Batched 版本）
Step 9: 正确性验证（对齐 Python HuggingFace 推理）
Step 10: 性能优化（Fused Kernel、CUDA Graph）
```

## 2.2 Step 1：模型结构分析

### 分析方法

首先阅读 HuggingFace 的 Qwen3.5 模型代码 (`modeling_qwen3_5.py`)，提取出模型配置和权重结构。关键发现：

**难点 1：混合层索引**

Qwen3.5 的 32 层不是按连续类型排列的。Full Attention 层出现在索引 {3, 7, 11, 15, 19, 23, 27, 31}，间隔 `full_attention_interval=4`。这意味着：
- `wq_layers_` / `wk_layers_` / `wv_layers_` / `wo_layers_` 只有 8 个元素（Full Attention 专属）
- 线性注意力的 24 层权重需要单独的 `LinearAttnWeights` 结构存储
- FFN 层（`w1`, `w2`, `w3`）则是 32 层全部共享

**解决方案**：设计了 `full_attn_type_idx()` 和 `linear_attn_type_idx()` 两个映射函数（`qwen3_5.cpp:50-70`），将全局层索引映射到类型内部索引：

```cpp
// qwen3_5.cpp:50-58
int Qwen35Model::full_attn_type_idx(int layer_idx) const {
  int idx = 0;
  for (auto li : q35_config_.full_attn_layer_indices) {
    if (li == layer_idx) return idx;
    ++idx;
  }
  LOG(FATAL) << "Layer " << layer_idx << " is not a full attention layer!";
  return -1;
}
```

## 2.3 Step 2-4：权重导出与模型加载

### 二进制格式设计

定义了新的 magic number `0x71333539`（"q359" 的 ASCII 编码），在 512 字节的 header 中存储所有配置参数：

```
Header (512 bytes):
  [0:4]   magic = 0x71333539
  [4:8]   version
  [8:52]  vision config (same as Qwen3-VL)
  [52:92] text config (dim, layers, heads, ...)
  [92:112] special tokens
  [112:148] hybrid attention config (full_attn indices)
  [148:172] linear attention config
  [172:192] MRoPE config
  [192:512] reserved
```

**难点 2：RMSNorm 权重偏移公式**

Qwen3.5 使用 `(1.0 + weight)` 的 RMSNorm 公式（类似 Gemma），而非标准的 `weight`。权重文件中存储的是初始化为零的偏移量，需要在加载后加 1.0。

关键代码（`qwen3_5.cpp:110-123`）：
```cpp
// Qwen3.5 RMSNorm uses (1.0 + weight) formula
int total_rms = qwen_layers_->rmsnorm_layers_.size();
for (int i = 0; i < total_rms; ++i) {
  auto lp = std::dynamic_pointer_cast<op::LayerParam>(qwen_layers_->rmsnorm_layers_[i]);
  auto& w = lp->get_weight(0);
  half* wptr = w.ptr<half>();
  for (int j = 0; j < numel; ++j) {
    wptr[j] = __float2half(__half2float(wptr[j]) + 1.0f);
  }
}
```

**注意**：GDN 的 `norm_weight` 不在 `rmsnorm_layers_` 中，它是 FP32 存储并单独管理的，不需要加 1.0。

**难点 3：mmap 权限**

由于需要原地修改 RMSNorm 权重（+1.0），使用了 `MAP_PRIVATE` 而非 `MAP_SHARED`，利用 Copy-On-Write 机制避免修改原始文件：

```cpp
// qwen3_5.cpp:218
vl_model_data_ = mmap(nullptr, vl_model_file_size_, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
```

## 2.4 Step 5：GDN CUDA 算子实现（核心难点）

GDN 线性注意力需要实现 14+ 个全新的 CUDA kernel，这是整个适配过程中最复杂的部分。

### 2.4.1 算子清单

| 算子 | 源码位置 | 复杂度 | 难度 |
|------|---------|-------|------|
| causal_conv1d_silu | gdn_kernel.cu:28-57 | ★★ | 状态管理顺序 |
| l2_norm_per_head | gdn_kernel.cu:100-155 | ★★ | 共享内存规约 |
| gdn_decode_step | gdn_kernel.cu:166-230 | ★★★★★ | 核心算法正确性 |
| gdn_prefill_transposed | gdn_kernel.cu:610-673 | ★★★★ | 内存布局优化 |
| gated_rmsnorm | gdn_kernel.cu:224-260 | ★★★ | 双输入融合 |
| compute_gdn_gates | gdn_kernel.cu:301-335 | ★★★ | 数值稳定性 |
| deinterleave_q_gate | gdn_kernel.cu:540-561 | ★★ | 头内交织布局 |
| partial_mrope_interleaved | gdn_kernel.cu:352-405 | ★★★★ | 交织RoPE+3D位置 |
| apply_sigmoid_gate | gdn_kernel.cu:338-347 | ★ | 简单逐元素 |
| fused_qkv_gemv | gdn_kernel.cu:678-784 | ★★★ | block分派融合 |
| fused_gdn_proj_gemv | gdn_kernel.cu:786-795 | ★★★ | 2-way融合 |
| gather_strided | gdn_kernel.cu:566-583 | ★★ | 跨步gather |
| transpose_state | gdn_kernel.cu:588-605 | ★★ | 三维转置 |
| batched_* (prefill variants) | 各处 | ★★★ | 批量扩展 |

### 2.4.2 关键难点：Delta Net 状态更新顺序

**问题描述**：GDN 的 Delta Net 核心算法中，状态更新的操作顺序极其关键。HuggingFace 参考实现中的操作顺序是：

```python
# HuggingFace (fla_modules/delta_net.py)
def delta_rule_recurrence(q, k, v, beta, state, gate):
    state = state * gate              # 1. 先衰减
    kv_mem = (state * k).sum(-1)      # 2. 从衰减后的状态计算记忆
    delta = beta * (v - kv_mem)       # 3. 计算 delta
    state = state + k.outer(delta)    # 4. 更新状态
    output = (state * q).sum(-1)      # 5. 从更新后的状态计算输出
    return output, state
```

**初始错误尝试**：最初实现时将状态衰减和状态更新分开，先衰减所有状态、再更新，导致 `kv_mem` 使用的是未完整衰减的状态，输出严重偏差。

**解决方法**：严格遵循 HuggingFace 的逐元素处理顺序，在 CUDA kernel 中对每个 `v_dim` 维度独立执行完整的 5 步操作（`gdn_kernel.cu:196-220`）：

```cuda
for (int vi = threadIdx.x; vi < head_v_dim; vi += blockDim.x) {
    float* state_row = state_head + vi * head_k_dim;
    
    // 1. Decay state and compute kv_mem from DECAYED state
    float sk_dot = 0.0f;
    for (int kj = 0; kj < head_k_dim; ++kj) {
      state_row[kj] *= gate_val;                              // step 1
      sk_dot += state_row[kj] * __half2float(k_head_ptr[kj]); // step 2
    }
    
    // 2. Delta from decayed state
    float v_val = __half2float(v_head_ptr[vi]);
    float delta = beta_val * (v_val - sk_dot);                 // step 3
    
    // 3. Update state and compute scaled output
    float dot_q = 0.0f;
    for (int kj = 0; kj < head_k_dim; ++kj) {
      float k_val = __half2float(k_head_ptr[kj]);
      state_row[kj] += delta * k_val;                         // step 4
      dot_q += state_row[kj] * (__half2float(q_head[kj]) * q_scale); // step 5
    }
    out_head[vi] = __float2half(dot_q);
}
```

### 2.4.3 关键难点：Conv1D 计算/状态更新顺序

**问题描述**：因果 Conv1D 的状态管理必须满足：先使用旧状态计算输出，再更新状态。

**初始错误**：先 shift state、再计算 conv，导致使用了包含当前输入的状态来计算当前输出。

**解决方案**（`gdn_kernel.cu:37-56`）：

```cuda
// 1. Compute convolution FIRST (before updating state)
float sum = 0.0f;
for (int j = 0; j < state_cols; ++j) {
  sum += conv_state[idx * state_cols + j] * conv_weight[idx * kernel_size + j];
}
sum += new_input[idx] * conv_weight[idx * kernel_size + kernel_size - 1];

// 2. THEN update state: shift left, insert new input
for (int j = 0; j < state_cols - 1; ++j) {
  conv_state[idx * state_cols + j] = conv_state[idx * state_cols + j + 1];
}
conv_state[idx * state_cols + state_cols - 1] = new_input[idx];
```

### 2.4.4 关键难点：Q/Gate 的逐头交织 (Deinterleave)

**问题描述**：Qwen3.5 Full Attention 的 `q_proj` 输出维度是 `q_gate_dim = 8192`，其中 Q 和 Gate 按**逐头交织**存储：

```
内存布局: [h0_q(256), h0_gate(256), h1_q(256), h1_gate(256), ..., h15_q(256), h15_gate(256)]
```

**初始错误尝试**：按全局前半/后半分割，即 `Q = output[0:4096], Gate = output[4096:8192]`。这导致第 0 个头的 Q 包含了 h0_q 和 h0_gate 的前面部分（混合了数据），输出完全错误。

**解决方案**：实现专门的 deinterleave kernel（`gdn_kernel.cu:540-561`），按头维度逐元素拆分：

```cuda
__global__ void deinterleave_q_gate_kernel(const half* interleaved, half* q_out, half* gate_out,
                                            int n_heads, int head_dim, int seq_len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n_heads * head_dim * seq_len;
  if (idx >= total) return;
  
  int t = idx / (n_heads * head_dim);
  int rem = idx % (n_heads * head_dim);
  int h = rem / head_dim;
  int d = rem % head_dim;
  
  // Source: interleaved layout [seq_len, n_heads * 2 * head_dim]
  // Pattern: [h0_q(hd), h0_gate(hd), h1_q(hd), h1_gate(hd), ...]
  int src_offset = t * (n_heads * 2 * head_dim) + h * 2 * head_dim;
  q_out[idx]    = interleaved[src_offset + d];
  gate_out[idx] = interleaved[src_offset + head_dim + d];
}
```

### 2.4.5 关键难点：Interleaved M-RoPE

**问题描述**：Qwen3.5 使用交织式 M-RoPE，与 Qwen3-VL 的标准 M-RoPE 不同：
- Qwen3-VL：按全局段分配 `[24 pairs for T, 20 for H, 20 for W]`
- Qwen3.5：按交织方式分配 `sections=[11,11,10]`，每对旋转的维度按 `pair_idx % 3` 决定属于 T/H/W 哪个分量

此外，Qwen3.5 使用 **half-split** 格式（前半 real 后半 imaginary），而非相邻配对格式。

**解决方案**（`gdn_kernel.cu:352-405`）：

```cuda
// For each rotation pair within a head
int pair = threadIdx.x;
if (pair >= num_pairs) return;

// Interleaved section assignment: pair 0,3,6,...→T; 1,4,7,...→H; 2,5,8,...→W
int section_idx = pair % 3;
int pos;
if (section_idx == 0) pos = pos_t;
else if (section_idx == 1) pos = pos_h;
else pos = pos_w;

// Half-split format: (x[i], x[i+num_pairs]) form a rotation pair
int idx_re = head * head_dim + pair;
int idx_im = head * head_dim + pair + num_pairs;

float cos_val = cos_cache[pos * num_pairs + pair];
float sin_val = sin_cache[pos * num_pairs + pair];
float re = __half2float(q[idx_re]);
float im = __half2float(q[idx_im]);
q[idx_re] = __float2half(re * cos_val - im * sin_val);
q[idx_im] = __float2half(re * sin_val + im * cos_val);
```

## 2.5 Step 6：内存布局与状态管理

### GDN 状态结构

每个 GDN 层维护两个持久状态（`qwen3_5.h:98-104`）：

```
GDNState {
  conv_state:  [8192, 3] FP16          — Conv1D 滑动窗口 (conv_dim × (kernel_size−1))
  ssm_state:   [32, 128, 128] FP32     — Delta Net 递推状态 (v_heads × v_dim × k_dim)
}
24 层共计: 24 × (8192×3×2 + 32×128×128×4) ≈ 24 × (48KB + 2MB) ≈ 49.2 MB
```

**关键设计决策**：`ssm_state` 使用 FP32 精度存储，因为 Delta Net 的状态会跨越所有 token 累积（递推），FP16 的精度不足会导致在长序列上出现严重的数值漂移。

### Prefill 状态优化

**难点**：标准的 `ssm_state` 布局 `[v_head, v_dim, k_dim]` 在 prefill 时每次内层循环访问一行 `state_row = state[vi, :]` 长度为 `k_dim=128`，对于 v_dim 方向的分步处理，不同线程访问的行跨度为 `k_dim×sizeof(float)=512 bytes`，这导致内存访问不连续。

**解决方案**：设计了转置版本 `gdn_prefill_transposed_kernel`（`gdn_kernel.cu:610-673`），在 prefill 前将状态转置为 `[v_head, k_dim, v_dim]`，使得同一 warp 的线程访问连续的 `v_dim` 元素（每个线程处理一个 v_dim 维度，访问步长为 4 bytes），实现完全合并的内存访问。

```cpp
// Transpose: [v_head, v_dim, k_dim] → [v_head, k_dim, v_dim]
transpose_state_layer_->forward(state.ssm_state, state_transposed, n_v_heads, v_head_dim, k_head_dim);

// Optimized GDN prefill with coalesced access
gdn_prefill_transposed_layer_->forward(..., state_transposed, ...);

// Transpose back for decode
transpose_state_layer_->forward(state_transposed, state.ssm_state, n_v_heads, k_head_dim, v_head_dim);
```

这个优化带来了约 **5.6×** 的 prefill 加速。

## 2.6 Step 7-8：Decode 与 Prefill 路径实现

### Layer 抽象模式

所有 GDN 算子均封装为 `Layer->forward()` 模式（`gdn_layers.h`），共 16 个 Layer wrapper 类：

```cpp
class GDNDecodeStepLayer : public Layer {
 public:
  explicit GDNDecodeStepLayer(base::DeviceType device_type);
  base::Status forward(const half* q, const half* k, const half* v,
                       const float* gate, const float* beta,
                       float* state, half* output,
                       int num_k_heads, int num_v_heads,
                       int head_k_dim, int head_v_dim);
};
```

每个 Layer 的 `forward()` 方法内部调用对应的 `kernel::xxx_cu()` 函数，并传入 `cuda_config_->stream`。

### CUDA Graph 兼容性

Full Attention 的 CUDA Graph 版本（`full_attn_decode_graph()`）需要所有位置参数存储在 GPU 上：
- `rope_pos_gpu`：RoPE 位置（通过 pinned→GPU H2D 拷贝更新）
- `kv_pos_gpu`：KV cache 写入位置

GDN 层天然兼容 CUDA Graph，因为它不依赖 CPU 传入的位置参数（状态全部在 GPU 上递推更新）。

## 2.7 Step 9：正确性验证

### 验证方法

编写了 Python HuggingFace 参考推理脚本（`hf_infer/qwen3_5_infer.py`），使用相同的图片和 prompt，逐 token 对比 C++ 与 Python 的输出。

### 关键验证发现

1. **Python 得到 172 个输入 token，C++ 得到 181 个**：这是因为 ViT 的图像 token 数量因 resize 算法差异略有不同（smart_resize 中 `factor = patch_size × spatial_merge_size`）
2. **两者输出均正确描述图片内容**：Python 输出包含较长的 `<think>` 推理过程，C++ 的 `<think>` 为空但最终回答准确
3. **完全数值一致不是目标**：FP16 精度下，微小的累积差异会导致不同的采样路径，但最终语义应一致

## 2.8 Step 10：性能优化

### 2.8.1 Fused QKV GEMV

将 Full Attention 的 3 次独立 GEMV（Q、K、V projection）合并为单次 kernel launch，使用 block-index dispatch（`gdn_kernel.cu:678-770`）：

```cuda
// Block dispatch: Q blocks → K blocks → V blocks
if (blockIdx.x < q_blocks) {
    // compute Q projection row
    weight = q_weight; output = q_output; N = q_dim;
} else if (blockIdx.x < q_blocks + k_blocks) {
    // compute K projection row
    weight = k_weight; output = k_output; N = kv_dim;
} else {
    // compute V projection row
    weight = v_weight; output = v_output; N = kv_dim;
}
```

### 2.8.2 Fused GDN Projection

将 GDN 的 QKV 和 Z 两个大型 projection 合并为单次 2-way dispatch（`gdn_kernel.cu:786-795`）。

### 2.8.3 Fused FFN

Gate + Up + SwiGLU 三合一融合（`fused_ffn_kernel.cu`），输入向量只读取一次。

### 性能结果（Decode 阶段）

| 配置 | 吞吐 | 延迟/token |
|------|------|-----------|
| 无 CUDA Graph | 8.97 tok/s | 111.4 ms |
| CUDA Graph | 9.10 tok/s | 109.9 ms |
| 理论极限 (带宽) | ~10.7 tok/s | ~93.5 ms |

---

# 第三部分：ViT 视觉编码器流程与算子详解

## 3.1 概述

Qwen3.5-9B 的视觉编码器完全复用 Qwen3-VL 的 ViT 架构。通过类继承 `Qwen35Model : public Qwen3VLModel`，直接使用父类的 `encode_image()`、`prepare_multimodal_embeddings()` 等方法。

### ViT 配置（`qwen3_vl.h:30-44`）

| 参数 | 值 | 说明 |
|------|------|------|
| hidden_size | 1152 | ViT 隐藏维度 |
| intermediate_size | 4304 | ViT MLP 中间维度 |
| num_heads | 16 | 注意力头数 |
| head_dim | 72 | 每头维度 (1152/16) |
| depth | 27 | Transformer block 数量 |
| patch_size | 16 | 图像 patch 边长 |
| temporal_patch_size | 2 | 时序维度（视频帧） |
| spatial_merge_size | 2 | 空间合并倍率 (4 patch → 1 token) |
| out_hidden_size | 4096 | 输出维度（与 LLM dim 对齐） |
| num_position_embeddings | 2304 | 位置编码数量 (48×48) |

## 3.2 完整 ViT 流程图

```
输入: 原始图像 (H×W×3, uint8)
    │
    ▼
┌──────────────────────────────────────────────────┐
│ Phase 0: 图像预处理 (CPU + GPU)                   │
│ (qwen3_vl.cpp: preprocess_image)                 │
│                                                  │
│ 0a. STB 加载图像 → [H, W, 3] uint8 RGB           │
│ 0b. Smart Resize:                                │
│     factor = patch_size × spatial_merge_size = 32│
│     确保 H/W 都是 32 的倍数                       │
│     像素数约束: 56²~1003520                       │
│     例: 2048×1365 → 1216×800                     │
│ 0c. GPU Upload: uint8 pixels → GPU buffer        │
│ 0d. Fused Normalize + Patch Extraction (GPU):    │
│     normalize: (pixel - 0.5) / 0.5              │
│     uint8 → FP16 转换                            │
│     抽 patch: [H,W,3] → [num_patches, 1536]     │
│     (1536 = 3×temp_ps×patch_size² = 3×2×16×16)  │
│ 输出: pixel_values [num_patches, 1536] FP16     │
│ 例: 1216×800 → grid 38×25 → 950 patches         │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│ Phase 1: Patch Embedding                         │
│ (qwen3_vl.cpp: vision_patch_embed)               │
│                                                  │
│ 算子: cuBLAS HGEMM                               │
│ 输入: pixel_values [num_patches, 1536] FP16     │
│ 权重: patch_embed_weight [1152, 1536] FP16      │
│ 偏置: patch_embed_bias [1152] FP16               │
│ 计算: output = input @ weight.T + bias           │
│ 输出: [num_patches, 1152] FP16                  │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│ Phase 2: Position Embedding (双线性插值)          │
│ (qwen3_vl.cpp: vision_add_pos_embed)             │
│                                                  │
│ 算子: pos_embed_interpolate_cu (自定义CUDA)       │
│ 输入: patch_embeds [num_patches, 1152] FP16     │
│ 权重: pos_embed [2304, 1152] FP16 (48×48 base)  │
│ 操作: 从 48×48 基础网格双线性插值到 grid_h×grid_w │
│       然后逐 patch 加上位置编码                   │
│ 输出: [num_patches, 1152] FP16 (原地加)         │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│ Phase 3: Vision RoPE 计算 (CPU → GPU)            │
│ (qwen3_vl.cpp: compute_vision_rotary_emb)        │
│                                                  │
│ theta = 10000.0 (vision 专用, 不同于 LLM)        │
│ head_dim = 72, 使用 36 个旋转对                   │
│ 布局: [18 height, 18 width, 18 height, 18 width] │
│ 为每个 patch 计算 (h_pos, w_pos) 的 cos/sin      │
│ 输出: vision_cos [num_patches, 72] FP16          │
│       vision_sin [num_patches, 72] FP16          │
│ CPU 计算后 async H2D                             │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────┐
│ Phase 4: 27 层 Vision Transformer Blocks (双缓冲)         │
│ (qwen3_vl.cpp: vision_transformer_block)                 │
│                                                          │
│ FOR block_idx = 0 to 26:                                 │
│                                                          │
│ ┌──────────────────────────────────────────────────────┐ │
│ │ Step 4a: LayerNorm (norm1)                           │ │
│ │ 算子: layernorm_with_bias_fp16_kernel (自定义CUDA)    │ │
│ │ 输入: hidden [num_patches, 1152] FP16               │ │
│ │ 权重: norm1_weight [1152], norm1_bias [1152] FP16   │ │
│ │ 输出: normed1 [num_patches, 1152] FP16              │ │
│ └──────────────────────┬───────────────────────────────┘ │
│                        │                                 │
│                        ▼                                 │
│ ┌──────────────────────────────────────────────────────┐ │
│ │ Step 4b: QKV Projection                              │ │
│ │ 算子: cuBLAS HGEMM                                   │ │
│ │ 输入: normed1 [num_patches, 1152] FP16              │ │
│ │ 权重: qkv_weight [3456, 1152] (3×1152) FP16        │ │
│ │ 偏置: qkv_bias [3456] FP16                          │ │
│ │ 输出: qkv [num_patches, 3456] FP16                  │ │
│ └──────────────────────┬───────────────────────────────┘ │
│                        │                                 │
│                        ▼                                 │
│ ┌──────────────────────────────────────────────────────┐ │
│ │ Step 4c: Fused Split + RoPE + Transpose              │ │
│ │ 算子: fused_split_rope_transpose_cu (超融合CUDA)      │ │
│ │ 单次 kernel 完成:                                     │ │
│ │   1. QKV 分头: [N, 3×1152] → Q,K,V [N, 16, 72]     │ │
│ │   2. Q,K 施加 Vision RoPE (cos/sin)                  │ │
│ │   3. 转置: [N, 16, 72] → [16, N, 72]                │ │
│ │ 输出: q_trans [16, N, 72] FP16     (含RoPE)         │ │
│ │       k_trans [16, N, 72] FP16     (含RoPE)         │ │
│ │       v_trans [16, N, 72] FP16     (无RoPE)         │ │
│ └──────────────────────┬───────────────────────────────┘ │
│                        │                                 │
│                        ▼                                 │
│ ┌──────────────────────────────────────────────────────┐ │
│ │ Step 4d: Batched Self-Attention                      │ │
│ │ 算子: vision_attention_pretransposed_cu               │ │
│ │                                                      │ │
│ │ Step 4d-1: Q @ K^T (Batched GEMM)                   │ │
│ │   cublasHgemmStridedBatched                          │ │
│ │   [16, N, 72] × [16, 72, N] → [16, N, N]           │ │
│ │   scale = 1/√72 ≈ 0.1179                            │ │
│ │                                                      │ │
│ │ Step 4d-2: Softmax (无 causal mask)                  │ │
│ │   vision_softmax_fp16_kernel                         │ │
│ │   [16, N, N] → [16, N, N]                           │ │
│ │                                                      │ │
│ │ Step 4d-3: Attn @ V (Batched GEMM)                  │ │
│ │   cublasHgemmStridedBatched                          │ │
│ │   [16, N, N] × [16, N, 72] → [16, N, 72]           │ │
│ │                                                      │ │
│ │ Step 4d-4: Transpose back                            │ │
│ │   transpose_head_token_kernel                        │ │
│ │   [16, N, 72] → [N, 16×72] = [N, 1152]             │ │
│ └──────────────────────┬───────────────────────────────┘ │
│                        │                                 │
│                        ▼                                 │
│ ┌──────────────────────────────────────────────────────┐ │
│ │ Step 4e: Output Projection + Residual                │ │
│ │ 算子: cuBLAS HGEMM + bias_add_residual_fp16_kernel   │ │
│ │ 输入: attn_out [N, 1152] FP16                       │ │
│ │ 权重: proj_weight [1152, 1152], proj_bias [1152]    │ │
│ │ 输出 = attn_out @ weight.T + bias + hidden (残差)   │ │
│ │ 输出: [N, 1152] FP16                                │ │
│ └──────────────────────┬───────────────────────────────┘ │
│                        │                                 │
│                        ▼                                 │
│ ┌──────────────────────────────────────────────────────┐ │
│ │ Step 4f: LayerNorm (norm2)                           │ │
│ │ 同 Step 4a                                           │ │
│ │ 输出: normed2 [N, 1152] FP16                        │ │
│ └──────────────────────┬───────────────────────────────┘ │
│                        │                                 │
│                        ▼                                 │
│ ┌──────────────────────────────────────────────────────┐ │
│ │ Step 4g: Vision MLP                                  │ │
│ │ 算子: vision_mlp_cu (融合实现)                        │ │
│ │                                                      │ │
│ │ FC1: cuBLAS HGEMM                                    │ │
│ │   [N, 1152] × [4304, 1152].T → [N, 4304]           │ │
│ │   权重: mlp_fc1_weight [4304, 1152] FP16            │ │
│ │                                                      │ │
│ │ Bias + GELU: bias_gelu_roundtrip_fp16_kernel         │ │
│ │   [N, 4304] + fc1_bias [4304] → GELU → [N, 4304]   │ │
│ │   (保持 FP16 round-trip 精度)                        │ │
│ │                                                      │ │
│ │ FC2: cuBLAS HGEMM                                    │ │
│ │   [N, 4304] × [1152, 4304].T → [N, 1152]           │ │
│ │   权重: mlp_fc2_weight [1152, 4304] FP16            │ │
│ │                                                      │ │
│ │ Bias + Residual: bias_add_residual_fp16_kernel       │ │
│ │   output = fc2_out + fc2_bias + hidden               │ │
│ │   输出: [N, 1152] FP16                              │ │
│ └──────────────────────────────────────────────────────┘ │
│                                                          │
│ (双缓冲: output ↔ output2 交替使用避免额外拷贝)           │
└───────────────────────┬──────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│ Phase 5: Spatial Merge (4 patches → 1 token)     │
│ (qwen3_vl.cpp: vision_merger)                    │
│                                                  │
│ Step 5a: LayerNorm                               │
│   normed = LN(hidden_states) [N, 1152] FP16     │
│                                                  │
│ Step 5b: Spatial Merge (2×2 block 合并)           │
│   spatial_merge_cu (D2D Memcpy based)            │
│   [num_patches, 1152] → [num_tokens, 4608]      │
│   4608 = 1152 × 2 × 2 (4 个 patch 拼接)         │
│   num_tokens = num_patches / 4                   │
│                                                  │
│ Step 5c: Merger MLP                              │
│   FC1: cuBLAS HGEMM                              │
│     [num_tokens, 4608] × [4608, 4608].T          │
│     → [num_tokens, 4608]                         │
│   Bias + GELU                                    │
│   FC2: cuBLAS HGEMM                              │
│     [num_tokens, 4608] × [4096, 4608].T          │
│     → [num_tokens, 4096]                         │
│                                                  │
│ 输出: vision_tokens [num_tokens, 4096] FP16      │
│ 例: 950 patches → 237 vision tokens              │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│ Phase 6: 多模态 Embedding 组装                    │
│ (qwen3_vl.cpp: prepare_multimodal_embeddings)    │
│                                                  │
│ Step 6a: Text Embedding                          │
│   embedding_layer_->forward(tokens)              │
│   输入: token_ids [text_len] int32               │
│   权重: [248320, 4096] FP16                      │
│   输出: text_embeds [text_len, 4096] FP16        │
│                                                  │
│ Step 6b: 拼接替换                                │
│   在 <image> token 位置插入 vision_tokens         │
│   [text_before] + [vision_tokens] + [text_after] │
│   输出: multimodal_embeds [total_len, 4096] FP16 │
│   total_len = text_len − 1 + num_vision_tokens   │
│                                                  │
│ Step 6c: M-RoPE 位置生成                         │
│   - 图片前文本: (pos, pos, pos)                   │
│   - 视觉 tokens: (t_base, t_base+row, t_base+col)│
│   - 图片后文本: (resume, resume, resume)          │
│   上传到 GPU: mrope_pos_t/h/w_gpu_               │
│                                                  │
│ 输出: multimodal_embeds [total_len, 4096] FP16   │
│       + M-RoPE 位置数组                           │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
                  → Prefill (LLM)
```

## 3.3 ViT 涉及的全部算子汇总

### 自定义 CUDA Kernel（vision_encoder_kernel.cu）

| 算子名称 | 功能 | 输入维度 | 输出维度 |
|----------|------|---------|---------|
| `fused_normalize_patches_kernel` | Normalize+Patch提取融合 | [H,W,3] uint8 | [N, 1536] FP16 |
| `pos_embed_interpolate_fp16_kernel` | 位置编码双线性插值 | [N,1152]+[2304,1152] | [N, 1152] FP16 |
| `layernorm_with_bias_fp16_kernel` | LayerNorm (含bias) | [N, D] FP16 | [N, D] FP16 |
| `fused_split_rope_transpose_kernel` | QKV分割+RoPE+转置 | [N, 3456] FP16 | [16,N,72]×3 FP16 |
| `vision_softmax_fp16_kernel` | Softmax (无causal) | [16, N, N] | [16, N, N] |
| `transpose_head_token_kernel` | 多头结果转置回拼 | [16, N, 72] | [N, 1152] |
| `bias_gelu_roundtrip_fp16_kernel` | Bias+GELU (FP16精度) | [N, D] | [N, D] |
| `bias_add_residual_fp16_kernel` | Bias+残差加 | [N,D]+[D]+[N,D] | [N, D] |
| `spatial_merge_cu` | 2×2空间合并 | [N, 1152] | [N/4, 4608] |
| `vision_mlp_cu` | ViT MLP (2 GEMM + GELU) | [N, 1152] | [N, 1152] |
| `vision_merger_mlp_cu` | Merger MLP | [N/4, 4608] | [N/4, 4096] |
| `vision_attention_pretransposed_cu` | 完整注意力 (GEMM×2+softmax) | Q,K,V [16,N,72] | [N, 1152] |

### cuBLAS 调用

| 操作 | 调用方式 | 场景 |
|------|---------|------|
| Patch Embed | `cublasHgemm` | Phase 1 |
| QKV Projection | `cublasHgemm` | 每个ViT block (×27) |
| Q@K^T | `cublasHgemmStridedBatched` | 每个ViT block (×27) |
| Attn@V | `cublasHgemmStridedBatched` | 每个ViT block (×27) |
| Output Projection | `cublasHgemm` | 每个ViT block (×27) |
| MLP FC1 | `cublasHgemm` | 每个ViT block (×27) |
| MLP FC2 | `cublasHgemm` | 每个ViT block (×27) |
| Merger FC1/FC2 | `cublasHgemm` | Phase 5 |

### 关键优化技术

1. **超级融合 Kernel**：`fused_split_rope_transpose_cu` 将原本 3 个独立操作（split、RoPE、transpose）合并为 1 次 kernel launch，消除中间缓冲区
2. **FP16 Round-Trip GELU**：特殊实现保持 GELU 在 FP16→FP32→FP16 转换中的数值一致性
3. **双缓冲**：27 层 ViT block 使用 output/output2 交替缓冲，避免每层的 D2D 拷贝
4. **预分配 Workspace**：`VisionWorkspace` 结构一次性分配所有中间 buffer（normed、qkv、attention scores 等），跨 block 复用
5. **Batched GEMM**：使用 `cublasHgemmStridedBatched` 对 16 个注意力头并行执行 Q@K^T 和 Attn@V

---

# 第四部分：Qwen3.5-9B 与 Qwen3-VL 的详细区别

## 4.1 总体架构对比

Qwen3.5-9B 通过类继承 `Qwen35Model : public Qwen3VLModel` 复用了 Qwen3-VL 的视觉编码器，但 LLM 部分进行了根本性重新设计。以下从多个维度进行系统对比。

## 4.2 模型配置参数对比

| 参数 | Qwen3-VL-8B | Qwen3.5-9B | 差异分析 |
|------|-------------|-------------|---------|
| hidden_size | 4096 | 4096 | 相同 |
| intermediate_size | 12288 | 12288 | 相同 |
| **num_hidden_layers** | **36** | **32** | Qwen3.5 少 4 层，但通过混合架构弥补 |
| **num_attention_heads** | **32** | **16** | Q 头数减半 |
| **num_key_value_heads** | **8** | **4** | KV 头数减半 |
| **head_dim** | **128** | **256** | 每头维度翻倍（保持 q_dim=4096 不变） |
| **vocab_size** | **151,936** | **248,320** | 词汇表增大 63.4% |
| **rope_theta** | **5,000,000** | **10,000,000** | 基频翻倍，长文本外推能力更强 |
| **mrope_section** | **[24, 20, 20]** | **[11, 11, 10]** | M-RoPE 总维度从 128 降至 64 |
| rms_norm_eps | 1e-6 | 1e-6 | 相同 |
| max_position_embeddings | 262144 | 262144 | 相同 |

**关键计算恒等式**：两个模型的 `q_dim` 和 `kv_dim` 相同，但通过不同分解实现：
- Qwen3-VL: `q_dim = 32 × 128 = 4096`, `kv_dim = 8 × 128 = 1024`
- Qwen3.5: `q_dim = 16 × 256 = 4096`, `kv_dim = 4 × 256 = 1024`

## 4.3 注意力机制对比（核心差异）

### 4.3.1 Qwen3-VL：同质 GQA 架构

Qwen3-VL 的 **36 层全部使用标准 Grouped Query Attention (GQA)**：

```
输入: hidden_state [4096]
  ↓
RMSNorm → Q proj [4096, 4096] → q_norm(per-head) → M-RoPE(全128维)
         K proj [1024, 4096] → k_norm(per-head) → M-RoPE(全128维) → KV Cache
         V proj [1024, 4096] → KV Cache
  ↓
Flash Attention Decode (32Qheads, 8KVheads, kv_mul=4)
  ↓
O proj [4096, 4096]
  ↓
输出 (无 gate)
```

**特点**：
- 所有 36 层结构完全一致，权重维度一致
- 标准注意力无 Output Gate
- Q/K/V 是 3 个独立矩阵乘

### 4.3.2 Qwen3.5 Full Attention（8 层，含 Output Gate）

Qwen3.5 的 Full Attention 层出现在索引 `{3, 7, 11, 15, 19, 23, 27, 31}`，有 **4 个关键区别**：

**区别 1：Q 与 Gate 交织合并**

Q projection 与 Output Gate 合并为单个权重矩阵 `[8192, 4096]`，输出按**逐头交织**排列：

```
Qwen3-VL:  Q_weight [4096, 4096] → Q [4096]  (仅 Q，无 gate)

Qwen3.5:   QGate_weight [8192, 4096] → interleaved [8192]
            布局: [h0_q(256), h0_gate(256), h1_q(256), h1_gate(256), ..., h15_q(256), h15_gate(256)]
            ↓ deinterleave
            Q [4096], Gate [4096]
```

**区别 2：Output Gate（Qwen3.5 独有）**

注意力输出被 sigmoid gate 调制：`output = attention_output * sigmoid(gate)`

这是对标准 Transformer 的改进，Gate 可以学习性地"关闭"某些维度的注意力输出，提升模型表达能力。Qwen3-VL 无此机制。

**区别 3：部分 RoPE (Partial RoPE)**

```
Qwen3-VL: head_dim=128, 全部 128 维旋转, 64 个旋转对
Qwen3.5:  head_dim=256, 仅前 64 维旋转 (partial_rotary_factor=0.25), 32 个旋转对
          后 192 维不做 RoPE（保持位置无关特征）
```

Partial RoPE 允许模型同时保留位置相关和位置无关的特征表示，在更大的 head_dim 下尤其有效。

**区别 4：Interleaved M-RoPE 分段方式**

```
Qwen3-VL: 按段连续分配
  pairs 0-23  → 时间维 T
  pairs 24-43 → 高度维 H  
  pairs 44-63 → 宽度维 W

Qwen3.5: 按交织模式分配 (pair_idx % 3)
  pair 0,3,6,9,...(mod3==0) → T (共11对)
  pair 1,4,7,10,..(mod3==1) → H (共11对)
  pair 2,5,8,11,..(mod3==2) → W (共10对)
```

交织分配使三个空间维度的信息在特征空间中更均匀地混合，避免了连续分段导致的信息聚集问题。

### 4.3.3 Qwen3.5 GDN Linear Attention（24 层，全新架构）

这是 Qwen3.5 最大的架构创新——75% 的层使用 **Gated Delta Net (GDN)** 替代传统注意力：

```
Qwen3-VL:  标准注意力 × 36 层（KV Cache 随序列长度线性增长）
Qwen3.5:   GDN 线性注意力 × 24 层（固定大小递推状态，不随序列增长）
```

Qwen3-VL **完全没有** GDN 相关组件。Qwen3.5 的 GDN 层引入了 9 个全新算子（详见第五部分）。

## 4.4 RMSNorm 差异

| 特性 | Qwen3-VL | Qwen3.5 |
|------|----------|---------|
| 公式 | `x * rsqrt(mean(x²)+eps) * weight` | `x * rsqrt(mean(x²)+eps) * (1 + weight)` |
| 权重存储 | FP16，直接使用 | FP16，加载后 CPU 端 +1.0 |
| 总层数 | 36×2+1=73 | 32×2+1+8q_norm+8k_norm=81 |
| mmap 权限 | `PROT_READ` | `PROT_READ\|PROT_WRITE + MAP_PRIVATE`（COW） |
| GDN Gated RMSNorm | 无 | 24 层，FP32 权重，`RMSNorm(x)*SiLU(z)` 逐头计算 |

Qwen3.5 的 `(1+weight)` 公式源自 Gemma 架构设计，`weight` 初始化为全零，训练过程中学习偏移量。这种设计使初始化时 RMSNorm 等效于单纯的缩放归一化（所有权重为 1.0），有利于训练初期的稳定性。

## 4.5 权重格式与加载差异

### 二进制格式

| 特征 | Qwen3-VL | Qwen3.5 |
|------|----------|---------|
| Magic | `0x71773376` ("qw3v") | `0x71333539` ("q359") |
| Header | 512 bytes | 512 bytes (扩展字段) |
| 扩展信息 | 无 | hybrid_attention 索引 + linear_attn 配置 + MRoPE 配置 |

### 权重矩阵数量对比

| 权重类型 | Qwen3-VL | Qwen3.5 | 说明 |
|----------|----------|---------|------|
| Embedding | [151936, 4096] | [248320, 4096] | Qwen3.5 大 63% |
| Q proj | [4096, 4096] × 36 = 36 | [8192, 4096] × 8 = 8 | Qwen3.5 含 Gate，但仅 8 层 |
| K proj | [1024, 4096] × 36 = 36 | [1024, 4096] × 8 = 8 | |
| V proj | [1024, 4096] × 36 = 36 | [1024, 4096] × 8 = 8 | |
| O proj (Full Attn) | [4096, 4096] × 36 = 36 | [4096, 4096] × 8 = 8 | |
| q_norm | [128] × 36 = 36 | [256] × 8 = 8 | head_dim 不同 |
| k_norm | [128] × 36 = 36 | [256] × 8 = 8 | |
| **GDN in_proj_qkv** | 无 | [8192, 4096] × 24 | GDN 独有 |
| **GDN in_proj_z** | 无 | [4096, 4096] × 24 | GDN 独有 |
| **GDN in_proj_a** | 无 | [32, 4096] × 24 | GDN 独有 |
| **GDN in_proj_b** | 无 | [32, 4096] × 24 | GDN 独有 |
| **GDN A_log** | 无 | [32] FP32 × 24 | GDN 独有 |
| **GDN dt_bias** | 无 | [32] FP16 × 24 | GDN 独有 |
| **GDN conv_weight** | 无 | [8192, 4] × 24 | GDN 独有 |
| **GDN norm_weight** | 无 | [128] FP32 × 24 | GDN 独有 |
| **GDN out_proj** | 无 | [4096, 4096] × 24 | GDN 独有 |
| FFN W1 | [12288, 4096] × 36 | [12288, 4096] × 32 | |
| FFN W2 | [4096, 12288] × 36 | [4096, 12288] × 32 | |
| FFN W3 | [12288, 4096] × 36 | [12288, 4096] × 32 | |
| RMSNorm | [4096] × 73 | [4096] × 65 | + 8 对 q_norm/k_norm [256] |
| LM Head | [151936, 4096] | [248320, 4096] | |

### Qwen3-VL 独有：Fused QKV 权重

Qwen3-VL 在加载后将 Q/K/V 三个权重物理拼接为 `[6144, 4096]` 的 fused 权重，decode 时单次 GEMV 产生所有 Q/K/V 输出：

```
Qwen3-VL: wqkv_fused [Q(4096)+K(1024)+V(1024), 4096] = [6144, 4096]
           → 单次 GEMV → 零拷贝指针分割 Q, K, V
```

Qwen3.5 的 Full Attention 由于 Q 包含 Gate（交织布局）,**无法使用相同的 fused QKV 方案**，改为 fused_qkv_gemv_kernel 的 block-dispatch 方式。

## 4.6 KV Cache 与 GDN 状态对比

### Qwen3-VL：纯 KV Cache

```
K Cache: [36 layers, max_seq_len, 1024] FP16
V Cache: [36 layers, max_seq_len, 1024] FP16
总计 (8K上下文): 36 × 2 × 8192 × 1024 × 2 = 1.15 GB
特点: 线性增长，序列越长内存越大
```

### Qwen3.5：混合状态

```
KV Cache (仅 8 层 Full Attn):
  K: [8, max_seq_len, 1024] FP16
  V: [8, max_seq_len, 1024] FP16
  总计 (8K上下文): 8 × 2 × 8192 × 1024 × 2 = 256 MB

GDN State (24 层 Linear Attn):
  conv_state:  [8192, 3] FP16 × 24 = 1.125 MB
  ssm_state:   [32, 128, 128] FP32 × 24 = 48 MB
  总计: 49.2 MB (固定大小，不随序列增长)

总内存: 305 MB vs Qwen3-VL 的 1.15 GB (3.8× 压缩)
```

**这是 GDN 架构的核心优势**：24 层的状态为固定大小的递推矩阵，不随输入序列长度增长，大幅降低了长序列推理的内存开销。

## 4.7 Decode 流程对比

### Qwen3-VL Decode

```
FOR il = 0 to 35 (全部同构):
  RMSNorm → 单次Fused GEMV(Q+K+V) → 零拷贝分割 → q_norm, k_norm
  → M-RoPE (全128维) → KV Cache Write → Flash Attention → O proj
  → Residual → FFN RMSNorm → Fused FFN → Residual
Final RMSNorm → LM Head [151936]
```

### Qwen3.5 Decode

```
FOR il = 0 to 31 (异构判断):
  IF Full Attention layer:
    RMSNorm → Fused QKV GEMV → Deinterleave Q/Gate → q_norm, k_norm
    → Partial M-RoPE (前64维, 交织) → KV Cache Write → Flash Attention
    → Sigmoid Gate → O proj → Residual → FFN → Residual
  ELSE (GDN layer):
    RMSNorm → Fused QKV+Z GEMV → A/B proj
    → Conv1D+SiLU → Split Q/K/V → L2 Norm → Gates
    → Delta Net Step (递推更新 ssm_state) → Gated RMSNorm → O proj
    → Residual → FFN → Residual
Final RMSNorm → LM Head [248320]
```

**Decode 复杂度对比**：
- Qwen3-VL：每层 ~6 个 kernel launch
- Qwen3.5 Full Attn：每层 ~10 个 kernel launch（多了 deinterleave、sigmoid gate、partial RoPE）
- Qwen3.5 GDN：每层 ~11 个 kernel launch（全新的算子流水线）

## 4.8 Prefill 流程对比

| 特性 | Qwen3-VL | Qwen3.5 |
|------|----------|---------|
| 注意力方式 | 全部 `cublasHgemmBatched` | Full Attn 用 cuBLAS Batched; GDN 用逐 token 递推 |
| GDN Prefill | 无 | 24 层需要 Conv1D+DeltaNet 逐 token 处理（开销大） |
| 状态转置优化 | 无 | Prefill 前转置 ssm_state 为 `[k_dim, v_dim]`，5.6× 加速 |
| DeepStack | 3 个 merger 在层 8/16/24 注入视觉特征 | 无 DeepStack |
| Prefill 速度 | ~916 tok/s (纯注意力全并行) | ~619 tok/s (受 GDN 递推限制) |

## 4.9 视觉编码器差异

### 共享部分（完全相同）

- 27 层 ViT 架构（hidden=1152, intermediate=4304, 16 heads, head_dim=72）
- Patch embedding, Position embedding（双线性插值）
- Vision RoPE（theta=10000, 2D）
- Spatial Merger（2×2 → 1 token，输出 4096 维）
- 所有 ViT CUDA kernel 完全复用

### DeepStack 差异

| | Qwen3-VL | Qwen3.5 |
|---|---|---|
| DeepStack Mergers | 3 个（层 8, 16, 24 后注入） | 0 个（显式清空） |
| 视觉特征注入 | Prefill 时在指定层添加到 visual token 位置 | 仅通过初始 embedding 融合 |
| 额外参数 | 3 × (LN + FC1 + FC2) | 无 |

Qwen3.5 在 `load_q35_model_file()` 中显式清空了 DeepStack 索引：
```cpp
vl_config_.deepstack_visual_indexes.clear();  // qwen3_5.cpp:423
```

## 4.10 性能对比

| 指标 | Qwen3-VL-8B-FP16 | Qwen3.5-9B-FP16 | 说明 |
|------|-------------------|------------------|------|
| Decode 速度 | ~9.87 tok/s | ~9.10 tok/s | Qwen3.5 因 vocab 更大、GDN 计算开销 |
| Prefill 速度 | ~916 tok/s | ~619 tok/s | GDN 递推限制 |
| ViT 编码 | ~416 ms | ~416 ms | 完全相同 |
| KV/State 内存 (8K) | ~1.15 GB | ~305 MB | **3.8× 压缩** |
| 模型参数量 | ~8B | ~9B | Qwen3.5 略大（GDN 权重 + 更大 vocab） |
| 带宽利用率 | ~84% | ~83% | 基本一致，均为带宽瓶颈 |

---

# 第五部分：GDN (Gated Delta Networks) Linear Attention 详解

## 5.1 GDN 算法原理

### 5.1.1 什么是 GDN

**Gated Delta Network (GDN)** 是一种线性注意力替代方案，将标准 Transformer 的 softmax 注意力（计算复杂度 $O(n^2)$）替换为递推状态更新机制（计算复杂度 $O(n)$）。其核心思想是使用一个固定大小的状态矩阵来替代 KV Cache，通过"增量规则"（Delta Rule）逐步更新状态。

### 5.1.2 GDN 与标准注意力的数学对比

**标准 Softmax 注意力**：
$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V$$

每一步 decode 需要扫描全部历史 KV：复杂度为 $O(n \cdot d)$，其中 $n$ 为序列长度。

**GDN Delta Rule 递推**：
$$S_t = \gamma_t \cdot S_{t-1} + \beta_t \cdot (v_t - S_{t-1}^T k_t) \otimes k_t$$
$$o_t = S_t^T \left(\frac{q_t}{\sqrt{d_k}}\right)$$

每一步 decode 只需访问固定大小的状态矩阵 $S \in \mathbb{R}^{d_v \times d_k}$：复杂度为 $O(d_v \cdot d_k)$，与序列长度无关。

### 5.1.3 GDN 完整推理流水线

以下是 Qwen3.5-9B 中 GDN 层的完整计算图（单 token decode）：

```
输入: x [4096] FP16 (residual stream)

┌─ Stage 1: 线性投影 ──────────────────────────────────────────────┐
│                                                                  │
│  RMSNorm(x) → [4096]                                            │
│      ↓                                                           │
│  in_proj_qkv: [8192, 4096] → qkv_raw [8192]                    │
│  in_proj_z:   [4096, 4096] → z [4096]  (gating 信号)           │
│  in_proj_a:   [32, 4096]   → α_raw [32] (decay gate 前置)      │
│  in_proj_b:   [32, 4096]   → β_raw [32] (write gate 前置)      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
                      ↓
┌─ Stage 2: 因果卷积 ─────────────────────────────────────────────┐
│                                                                  │
│  Causal Conv1D + SiLU:                                          │
│  conv_state: [8192, 3] FP16 (滑动窗口，保存 t-3, t-2, t-1)    │
│                                                                  │
│  计算: output_i = Σ(state[j] × w[j]) + input_i × w[3]          │
│        → SiLU(output)                                           │
│  更新: state ← [state[1], state[2], input]  (左移插入)          │
│                                                                  │
│  输出: conv_out [8192] FP16                                     │
│  关键: 先用旧状态计算，后更新状态（因果性保证）                    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
                      ↓
┌─ Stage 3: QKV 分割与归一化 ──────────────────────────────────────┐
│                                                                  │
│  零拷贝指针分割:                                                 │
│  Q = conv_out[0    : 2048]  → reshape [16, 128]                 │
│  K = conv_out[2048 : 4096]  → reshape [16, 128]                 │
│  V = conv_out[4096 : 8192]  → reshape [32, 128]                 │
│                                                                  │
│  Per-Head L2 Normalization:                                      │
│  Q_i = Q_i / √(Σ Q_i² + ε),  K_i = K_i / √(Σ K_i² + ε)      │
│                                                                  │
│  (注: 使用 L2 Norm 而非 RMSNorm，差异在于分母：                  │
│   L2 Norm: √(Σx²+ε),  RMSNorm: √(mean(x²)+ε))                │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
                      ↓
┌─ Stage 4: Gate 计算 ─────────────────────────────────────────────┐
│                                                                  │
│  对每个 v_head i (0..31):                                       │
│                                                                  │
│  Decay Gate (γ):                                                 │
│    α = α_raw[i] + dt_bias[i]                                    │
│    softplus_α = log(1 + exp(α))                                 │
│    γ_i = exp(softplus_α × (−exp(A_log[i])))                     │
│                                                                  │
│  Write Gate (β):                                                 │
│    β_i = sigmoid(β_raw[i])                                      │
│                                                                  │
│  γ ∈ (0, 1): 控制历史状态的衰减速率                              │
│  β ∈ (0, 1): 控制新信息的写入强度                                │
│                                                                  │
│  输出: gate [32] FP32, beta [32] FP32                           │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
                      ↓
┌─ Stage 5: Delta Net 递推 (核心算法) ─────────────────────────────┐
│                                                                  │
│  对每个 v_head h (0..31):                                       │
│  state_h: [128, 128] FP32 (v_dim × k_dim)                      │
│  k_head: k[h/2] (kv_mul=2, 每 2 个 v_head 共享 1 个 k_head)    │
│                                                                  │
│  Step 1 — 状态衰减:                                             │
│    state_h[v][k] *= γ_h,  ∀v,k                                  │
│                                                                  │
│  Step 2 — 记忆读取:                                             │
│    kv_mem[v] = Σ_k state_h[v][k] × k_head[k]                   │
│    (从衰减后的状态中提取与 key 相关的记忆)                        │
│                                                                  │
│  Step 3 — Delta 计算:                                            │
│    delta[v] = β_h × (v_head[v] − kv_mem[v])                    │
│    (计算"需要更新多少"：新值 v 与记忆中值的差异)                  │
│                                                                  │
│  Step 4 — 状态更新:                                             │
│    state_h[v][k] += delta[v] × k_head[k],  ∀v,k                │
│    (外积形式写入：关联 delta 与 key)                              │
│                                                                  │
│  Step 5 — 输出生成:                                             │
│    output_h[v] = Σ_k state_h[v][k] × q_head[k] / √d_k,  ∀v    │
│    (从更新后的状态中读取与 query 相关的信息)                      │
│                                                                  │
│  状态矩阵: [32, 128, 128] FP32 = 2 MB/层 (固定大小)            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
                      ↓
┌─ Stage 6: 输出后处理 ────────────────────────────────────────────┐
│                                                                  │
│  Gated RMSNorm:                                                  │
│    output = RMSNorm(attn_out, weight) × SiLU(z)                 │
│    RMSNorm 逐头计算 (32 头, 每头 128 维, FP32 权重)              │
│    SiLU(z) = z × sigmoid(z), z 为 Stage 1 的 in_proj_z 输出    │
│                                                                  │
│  Output Projection:                                              │
│    out_proj: [4096, 4096] → [4096]                              │
│                                                                  │
│  Residual Add: x = x + out_proj_output                          │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 5.1.4 GDN 各组件的物理意义

| 组件 | 物理意义 | 类比 |
|------|---------|------|
| **State S** | 固定容量的"记忆矩阵"，存储 key-value 关联信息 | 类似 KV Cache 的压缩版 |
| **Decay Gate γ** | 控制旧记忆的遗忘速率 (0→完全遗忘, 1→完全保留) | 类似 LSTM 的 forget gate |
| **Write Gate β** | 控制新信息的写入强度 (0→不写入, 1→完全写入) | 类似 LSTM 的 input gate |
| **Delta Rule** | "纠错式"更新：只写入新信息与当前记忆的差异 | 类似 Hebbian 学习的改进版 |
| **Conv1D** | 短程局部特征提取，弥补线性注意力的局部感知弱点 | 类似 S4/Mamba 的因果卷积 |
| **L2 Norm** | 稳定 Q/K 的范数，防止梯度爆炸/消失 | 类似注意力中除以 √d |
| **Gated RMSNorm** | Z 信号控制输出的信息流量 | 类似 GLU/Gating 机制 |

### 5.1.5 为什么混合 Full Attention + GDN

纯 GDN 的固定大小状态理论上无法完美记住所有历史信息（有限容量 vs 无限长度）。每隔 4 层插入一个 Full Attention 层（间隔 `full_attention_interval=4`），使模型在关键层可以"回顾"完整历史，弥补 GDN 的信息损失。

## 5.2 适配 GDN 过程中的困难与问题

### 5.2.1 难点一：Delta Net 状态更新的操作顺序

**问题描述**：GDN 核心的 5 步递推操作中，每一步都依赖前一步的结果，操作顺序不可交换。最初的实现尝试以 `v_dim` 为外循环、`k_dim` 为内循环并行处理，但由于将"衰减"和"更新"分成两个独立阶段，导致 Step 2 使用了错误的状态：

**错误实现**：
```
// 错误：先衰减所有状态
for all v, k: state[v][k] *= gate       // Step 1
// 错误：使用衰减后的完整状态计算读取和更新
for v:
  kv_mem = dot(state[v,:], k)            // Step 2 
  delta = beta * (v[v] - kv_mem)         // Step 3
  for k: state[v][k] += delta * k[k]    // Step 4
```

**问题**：这种实现在数学上等价于正确实现（因为 Step 1 对所有元素独立操作，顺序无关），但在 CUDA 实现中由于线程同步问题，不同线程可能看到部分衰减、部分未衰减的状态。

**正确实现**（`gdn_kernel.cu:196-220`）：使每个线程处理一个完整的 `v_dim` 维度，在单个 for 循环中顺序执行 5 步，确保每个线程看到一致的状态：

```cuda
for (int vi = threadIdx.x; vi < head_v_dim; vi += blockDim.x) {
    float* state_row = state_head + vi * head_k_dim;
    
    // Step 1+2: Decay AND compute kv_mem in single pass
    float sk_dot = 0.0f;
    for (int kj = 0; kj < head_k_dim; ++kj) {
        state_row[kj] *= gate_val;           // Step 1: decay
        sk_dot += state_row[kj] * k_val[kj]; // Step 2: memory read
    }
    
    // Step 3: delta
    float delta = beta_val * (v_val - sk_dot);
    
    // Step 4+5: Update AND output in single pass
    float dot_q = 0.0f;
    for (int kj = 0; kj < head_k_dim; ++kj) {
        state_row[kj] += delta * k_val[kj];              // Step 4: update
        dot_q += state_row[kj] * (q_val[kj] * q_scale);  // Step 5: output
    }
    out_head[vi] = __float2half(dot_q);
}
```

**调试过程**：发现此 bug 耗费了大量时间，因为错误输出并非完全随机——前几十个 token 几乎正确（状态较小时误差不明显），但在约 50 token 后开始明显发散。最终通过将 C++ 逐层输出与 HuggingFace Python 逐层输出进行 bit-level 对比（`hf_infer/debug_gdn_layer0.py`），精确定位到 `gdn_decode_step_kernel` 的操作顺序问题。

### 5.2.2 难点二：Conv1D 的计算-更新顺序

**问题描述**：因果卷积 `Causal Conv1D` 要求在计算当前时步输出时只使用历史信息（因果性）。具体而言，`conv_state` 保存 `[input[t-3], input[t-2], input[t-1]]`，当前输入为 `input[t]`：

```
output[t] = w[0]*input[t-3] + w[1]*input[t-2] + w[2]*input[t-1] + w[3]*input[t]
```

**初始错误**：先更新 state（将 `input[t]` 插入），再计算卷积。这导致 state 变成 `[input[t-2], input[t-1], input[t]]`，计算使用了"未来"信息 `input[t]` 两次（一次在 state 中，一次作为 new_input）：

```cuda
// 错误顺序
shift_left(conv_state);  conv_state[last] = new_input;  // 先更新
sum = dot(conv_state, conv_weight[0..K-2]) + new_input * conv_weight[K-1];  // 再计算
// 此时 conv_state 已包含 new_input，weight[K-2] 项用了 new_input 而非 input[t-1]
```

**修复**（`gdn_kernel.cu:37-56`）：先用 **旧状态** 计算卷积输出，再更新状态：

```cuda
// 正确顺序：先计算后更新
float sum = 0.0f;
for (int j = 0; j < state_cols; ++j)
    sum += conv_state[j] * conv_weight[j];
sum += new_input * conv_weight[kernel_size - 1];

// 然后更新
for (int j = 0; j < state_cols - 1; ++j)
    conv_state[j] = conv_state[j + 1];
conv_state[state_cols - 1] = new_input;
```

### 5.2.3 难点三：Q/Gate 逐头交织解码

**问题描述**：Qwen3.5 的 Full Attention Q projection 输出维度为 `[8192]`，包含 Q 和 Gate 按**逐头交织**存储。初始实现错误地按全局前后半分割：

```
错误假设: Q = output[0:4096], Gate = output[4096:8192]

实际布局: [h0_q(256), h0_gate(256), h1_q(256), h1_gate(256), ..., h15_q(256), h15_gate(256)]

错误分割导致:
  "Q" 的第 0 个头包含 h0_q(256) + h0_gate(256前128维) → 完全错误
```

**发现方法**：将 C++ 的 Q projection 输出与 Python 对比，发现每隔 256 个元素差异就翻转一次。追溯 HuggingFace 源码后发现 `attn.q_proj` 的权重实际上是 `[q_head_dim=256, gate_head_dim=256]` 交替排列的。

**修复**：实现了 `deinterleave_q_gate_kernel`（`gdn_kernel.cu:852-875`），按 `(head_idx, dim_in_head)` 精确索引提取 Q 和 Gate。

### 5.2.4 难点四：Interleaved M-RoPE 实现

**问题描述**：Qwen3.5 的 M-RoPE 与 Qwen3-VL 有 3 个层面的差异，需要从零推导正确实现：

**差异 1 — Partial Rotation**：仅旋转 head_dim=256 的前 64 维 (`partial_rotary_factor=0.25`)

**差异 2 — 交织分段**：
```
Qwen3-VL: 连续分段 [T×24, H×20, W×20]
Qwen3.5:  交织分段 pair[i%3==0]→T, pair[i%3==1]→H, pair[i%3==2]→W
```

**差异 3 — Half-Split 旋转对**：`(element[i], element[i+32])` 构成一个旋转对，而非 `(element[2i], element[2i+1])`。

**调试方法**：编写了 Python 参考实现，逐 pair 打印每个旋转对使用的 position 和 frequency_index，然后在 CUDA kernel 中复现相同逻辑。

### 5.2.5 难点五：GDN 状态的精度管理

**问题描述**：GDN 的 `ssm_state` 在 decode 过程中持续累积更新——每一步 Delta 更新都会修改状态矩阵，这与 KV Cache（写入后不修改）不同。如果使用 FP16 存储，经过数百个 token 的递推后，累积的量化误差会导致状态严重失真。

**具体问题表现**：
- 前 50 token：FP16 状态与 FP32 状态输出一致
- 50-100 token：部分 token 出现 1-2 bit 差异
- 200+ token：输出开始偏离，某些 token 完全错误
- 500+ token：生成乱码

**解决方案**：`ssm_state` 使用 FP32 精度存储（`qwen3_5.h:90`），代价是每层 2 MB（而非 FP16 的 1 MB），24 层共增加 24 MB。输入 Q/K/V 和输出仍为 FP16，仅状态矩阵使用 FP32 作为累积精度。

**gate/beta 也使用 FP32**（`gdn_kernel.cu:471-472`）：Gate 计算涉及 `exp(softplus × (-exp))` 的嵌套指数运算，FP16 的动态范围不足以表达中间值。

### 5.2.6 难点六：RMSNorm (1+weight) 公式差异

**问题描述**：Qwen3.5 的 RMSNorm 使用 `(1.0 + weight)` 公式，权重文件中存储的是初始化为零附近的偏移量。如果不加 1.0，归一化权重接近 0，输出几乎为零。

**初始现象**：模型加载后首次推理，所有层的 RMSNorm 输出均接近零向量，导致后续矩阵乘结果全为零、logits 全为零、argmax 永远输出 token 0。

**修复**（`qwen3_5.cpp:110-120`）：在 CPU 端对所有 81 个 RMSNorm 的权重执行 `weight[j] += 1.0f`。选择 CPU 端修改（而非 GPU kernel 修改）的原因：
1. 仅执行一次（init 时），不影响运行时性能
2. 使用 `MAP_PRIVATE` 的 mmap 支持 Copy-On-Write，不修改原始文件
3. 修改后的权重可直接复用已有的标准 RMSNorm kernel

**注意**：GDN 的 `norm_weight` 存储为 FP32 且**不需要**加 1.0——它使用的是标准 RMSNorm 公式。

## 5.3 GDN 适配中使用的优化方法

### 5.3.1 Fused GDN Projection GEMV

**问题**：每个 GDN 层需要 4 个矩阵乘投影（QKV、Z、A、B），其中 QKV 和 Z 都是大型矩阵乘（`[8192,4096]` 和 `[4096,4096]`），如果分开 launch 则有 4 次 kernel launch 开销。

**优化**：复用 Full Attention 的 fused_qkv_gemv 框架，将 QKV 和 Z 合并为 2-way block dispatch（`gdn_kernel.cu:1035-1048`）：

```cuda
void fused_fp16_gdn_proj_gemv_cu(...) {
    // 复用 fused_fp16_qkv_gemv_kernel，传入:
    //   "Q weight" = qkv_weight [8192, 4096]
    //   "K weight" = z_weight [4096, 4096]
    //   "V weight" = z_weight (dummy,不会被访问)
    // Block dispatch: blocks [0, qkv_blocks) → QKV, [qkv_blocks, total) → Z
    int total = (qkv_dim + WPB - 1) / WPB + (z_dim + WPB - 1) / WPB;
    fused_fp16_qkv_gemv_kernel<<<total, 256, 0, stream>>>(...);
}
```

**效果**：将 2 次大型 kernel launch 合并为 1 次，减少 launch 开销约 5-8μs/层。A 和 B 投影由于维度极小（`[32, 4096]`），执行时间仅约 1μs，保持独立 launch。

### 5.3.2 Fused QKV GEMV (Full Attention)

**优化**：Full Attention 的 Q（含 Gate）、K、V 三个投影通过 block-index dispatch 合并为单次 kernel launch（`gdn_kernel.cu:933-1033`）：

```
总 blocks = (8192+7)/8 + 2×(1024+7)/8 = 1024 + 256 = 1280 blocks

Block 0-1023:   → Q+Gate projection (8192 行)
Block 1024-1151: → K projection (1024 行 / 8 = 128 blocks)  
Block 1152-1279: → V projection (1024 行 / 8 = 128 blocks)
```

每个 block 内使用 8 个 warp (256 线程)，float4 向量化加载，4 个FP32 累加器实现 ILP，warp shuffle 规约。

### 5.3.3 GDN Prefill 状态转置优化

**问题**：`ssm_state` 默认布局 `[v_head, v_dim, k_dim]`，在 prefill 中每个线程处理一个 `v_dim` 维度，需要访问 `state_row = state[vi, 0..k_dim-1]`。不同线程的 `vi` 不同，内存访问步长为 `k_dim × sizeof(float) = 512 bytes`，导致完全非合并的 global memory 访问。

**优化**（`gdn_kernel.cu:917-985`）：转置状态为 `[v_head, k_dim, v_dim]`，使同一 warp 中相邻线程访问连续的 `v_dim` 元素：

```
原始布局 (非合并):
  Thread 0 reads state[0][0..127]  (at address base + 0)
  Thread 1 reads state[1][0..127]  (at address base + 512)  ← 512 byte 间距
  → 32 threads span 16 KB → 每次 load 需要 4 个 cache line

转置布局 (合并):
  Thread 0 reads state_t[kj][0]    (at address base + 0)
  Thread 1 reads state_t[kj][1]    (at address base + 4)   ← 4 byte 间距
  → 32 threads span 128 bytes → 单个 cache line 覆盖
```

**额外优化**：在转置版本中使用 shared memory 缓存 K 和 Q 向量（`gdn_kernel.cu:940-946`），避免每个线程重复从 global memory 读取：

```cuda
extern __shared__ float smem[];
float* sh_k = smem;                    // [head_k_dim = 128]
float* sh_q = smem + head_k_dim;       // [head_k_dim = 128]
for (int i = threadIdx.x; i < head_k_dim; i += blockDim.x) {
    sh_k[i] = __half2float(k_t[i]);
    sh_q[i] = __half2float(q_t[i]) * q_scale;
}
__syncthreads();
```

**性能对比**:
| 实现 | Prefill 时间 (511 tokens) | 说明 |
|------|--------------------------|------|
| 朴素实现 | ~4600 ms | 非合并访问 + 无 shared memory |
| 转置优化 | ~826 ms | 合并访问 + shared memory K/Q |
| **加速比** | **5.6×** | |

### 5.3.4 FP32 精度状态 + FP16 输入输出混合精度

**策略**：
- 状态矩阵 `ssm_state [32, 128, 128]` FP32 → 防止累积误差
- Gate/Beta 计算全程 FP32 → 防止 exp/sigmoid 溢出
- Q/K/V 输入为 FP16，从 global memory 加载后立即转 FP32 运算
- 输出使用 `__float2half()` 转回 FP16 写入

**代价 vs 收益**：
- 额外内存：24 层 × 2MB = 48 MB (FP32) vs 24MB (FP16)
- 但避免了长序列推理的数值发散，是必须的正确性保证

### 5.3.5 CUDA Graph 兼容性设计

**挑战**：CUDA Graph 要求所有 kernel 参数在 capture 时确定，不能每步从 CPU 传入变化的参数。

**Full Attention 层的解决方案**：
- RoPE position、KV Cache write position 使用 GPU 端存储 (`pos_gpu`)
- 实现 `partial_mrope_interleaved_gpu_pos_kernel` 和 `kv_cache_write_gpu_pos_kernel` 两个 GPU-pos 变体
- Graph 内包含 `increment_decode_pos_kernel`，每步自动递增位置

**GDN 层天然兼容**：
- 所有状态（`conv_state`、`ssm_state`）在 GPU 上原地更新
- 无位置参数依赖
- Gate/Beta 计算仅依赖层权重（固定）和投影输出（GPU 上产生）
- 因此不需要任何特殊处理即可被 CUDA Graph capture

**整体 Graph 结构**：
```
CUDA Graph capture:
  increment_decode_pos_kernel (1个)
  FOR il = 0..31:
    if full_attn: full_attn_decode_graph() (使用 GPU-pos kernels)
    else:         linear_attn_decode()     (天然兼容)
    q35_feed_forward()
  q35_cls_logits()
```

### 5.3.6 Batched 算子用于 Prefill

为 Prefill 阶段实现了所有 GDN 算子的 batched 版本：

| 算子 | Decode 版本 | Batched Prefill 版本 |
|------|------------|---------------------|
| Conv1D+SiLU | `causal_conv1d_silu_kernel` (单token) | `batched_causal_conv1d_silu_kernel` (逐token顺序) |
| L2 Norm | `l2_norm_per_head_kernel` | `batched_l2_norm_per_head_kernel` (grid 扩展) |
| Compute Gates | `compute_gdn_gates_kernel` | `batched_compute_gdn_gates_kernel` |
| Delta Net | `gdn_decode_step_kernel` | `gdn_prefill_transposed_kernel` (转置优化) |
| Gated RMSNorm | `gated_rmsnorm_kernel` | `batched_gated_rmsnorm_kernel` (grid 2D) |
| Sigmoid Gate | `apply_sigmoid_gate_kernel` | `batched_apply_sigmoid_gate_kernel` (连续内存) |
| M-RoPE | `partial_mrope_interleaved_kernel` | `batched_partial_mrope_interleaved_kernel` |
| Deinterleave | `deinterleave_q_gate_kernel` | 同一 kernel (seq_len 参数化) |
| RMSNorm | 复用框架层 | `batched_rmsnorm_fp16_kernel` (per-row 并行) |
| VecAdd | 框架 VecAddLayer | `batched_add_fp16_kernel` (逐元素并行) |

**Batched 策略**：
- Conv1D 和 Delta Net **无法**跨 token 并行（递推依赖），但可以跨 `conv_dim`/`v_head` 并行
- L2 Norm、Gates、RMSNorm、Sigmoid Gate 可以完全跨 token 并行（grid 扩展）
- Prefill 中的大型投影（QKV、Z）使用 cuBLAS HGEMM，天然支持 batch

### 5.3.7 Gather Strided 优化

**问题**：Conv1D 输出 `[seq_len, 8192]` 包含 Q、K、V 三部分交错存储。Prefill 中需要将它们分别 gather 到连续缓冲区以供后续 L2 Norm 和 Delta Net 使用。

**实现**（`gdn_kernel.cu:832-850`）：`gather_strided_kernel` 通过指定 `(inner_dim, outer_stride, src_offset)` 实现非连续→连续的数据搬移：

```cuda
// Q: inner_dim=2048, outer_stride=8192, src_offset=0
// K: inner_dim=2048, outer_stride=8192, src_offset=2048  
// V: inner_dim=4096, outer_stride=8192, src_offset=4096
```

这避免了 3 次 `cudaMemcpy2D` 调用，改用单次 kernel launch 完成跨步 gather。

---

## 附录：文件清单

| 文件路径 | 功能 |
|----------|------|
| `kuiper/include/model/qwen3_5.h` | Qwen3.5 模型头文件（配置、类声明） |
| `kuiper/source/model/qwen3_5.cpp` | 模型实现（init、decode、prefill） |
| `kuiper/include/op/gdn_layers.h` | 16 个 GDN Layer wrapper |
| `kuiper/source/op/gdn_layers.cpp` | Layer wrapper 实现 |
| `kuiper/source/op/kernels/cuda/gdn_kernel.cu` | GDN CUDA kernel 实现 |
| `kuiper/source/op/kernels/cuda/gdn_kernel.cuh` | GDN kernel 声明 |
| `kuiper/include/model/qwen3_vl.h` | Qwen3-VL 基类头文件 |
| `kuiper/source/model/qwen3_vl.cpp` | ViT 编码器实现 |
| `kuiper/source/op/kernels/cuda/vision_encoder_kernel.cu` | ViT CUDA kernel |
| `demo/main_qwen3_5.cpp` | 推理 Demo |
| `tools/export_qwen3_5.py` | 权重导出工具 |
