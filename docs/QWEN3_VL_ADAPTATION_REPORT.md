# Qwen3-VL-8B 模型适配报告

## 1. 项目概述

本报告记录了 Qwen3-VL-8B 视觉语言模型在 KuiperLLama 框架中的适配过程，包括技术分析、实现细节和当前状态。

### 1.1 模型架构

Qwen3-VL-8B 是一个先进的视觉语言模型，采用以下关键架构：

| 组件 | 规格 |
|------|------|
| **Vision Encoder (ViT)** | 27层, hidden_size=1152, 16头, head_dim=72 |
| **Language Model (LLM)** | 36层, hidden_size=4096, 32头, head_dim=128 |
| **Patch Size** | 16x16 (空间), 2 (时间) |
| **Spatial Merge Size** | 2x2 (4个 patch 合并为 1 个 token) |
| **Vocab Size** | 151,936 |

### 1.2 核心创新点

1. **DeepStack**: 从 ViT 层 8, 16, 24 提取特征，注入到 LLM 层 0, 1, 2
2. **Multimodal RoPE (MRoPE)**: 3D 位置编码 (时间, 高度, 宽度)
3. **Vision 2D RoPE**: 视觉编码器使用 2D 位置编码

---

## 2. 数据流分析

### 2.1 图像预处理

```
原始图像 (H × W × 3)
    ↓ smart_resize (保持宽高比，对齐到 patch_size)
调整后图像 (H' × W' × 3)  例如: (832 × 1216 × 3)
    ↓ 归一化 (mean=[0.48145..], std=[0.26862..])
    ↓ 转换为 patches
Patches tensor [num_patches, patch_dim]  例如: [3952, 1536]
    其中: num_patches = (H'/16) × (W'/16) = 52 × 76 = 3952
          patch_dim = 16 × 16 × 3 × 2 = 1536
```

### 2.2 Vision Encoder 流程

```
Patches [3952, 1536]
    ↓ Patch Embed (Conv3D)
Hidden [3952, 1152]
    ↓ + Position Embeddings (learned, interpolated)
Hidden [3952, 1152]
    ↓ 27 Transformer Blocks (with 2D RoPE)
    │   ├── 每个 Block:
    │   │   ├── LayerNorm
    │   │   ├── Self-Attention (16 heads, head_dim=72)
    │   │   │   ├── QKV Projection
    │   │   │   ├── 2D RoPE (height, width positions)
    │   │   │   └── Attention + Output Projection
    │   │   ├── Residual Add
    │   │   ├── LayerNorm
    │   │   ├── MLP (hidden=1152, intermediate=3072)
    │   │   └── Residual Add
    │   │
    │   ├── Layer 8: 提取 DeepStack Feature 0
    │   ├── Layer 16: 提取 DeepStack Feature 1
    │   └── Layer 24: 提取 DeepStack Feature 2
    │
Hidden [3952, 1152]
    ↓ Vision Merger (4 patches → 1 token)
    │   ├── Reshape: [3952, 1152] → [988, 4608]
    │   ├── LayerNorm
    │   ├── FC1: [4608 → 4608] + GELU
    │   └── FC2: [4608 → 4096]
Vision Tokens [988, 4096]

DeepStack Features:
    ├── Feature 0 (from layer 8): [988, 4096]
    ├── Feature 1 (from layer 16): [988, 4096]
    └── Feature 2 (from layer 24): [988, 4096]
```

### 2.3 Multimodal Embedding 构建

```
Text tokens: [<|im_start|>, system, ..., <vision_start>, <image>, <vision_end>, ..., user, ...]
               │                         │     ↑ 将被替换
               │                         │
Multimodal Embedding:
[text_embed_0..14] + [vision_embed_0..987] + [text_embed_16..24]
       ↑ 15 tokens      ↑ 988 tokens              ↑ 10 tokens
       
Total: ~1012 tokens
```

### 2.4 LLM 推理 (带 DeepStack)

```
Multimodal Embeddings [seq_len, 4096]
    ↓ 36 Transformer Layers
    │   ├── Layer 0: hidden += deepstack_feature_0[visual_positions]
    │   ├── Layer 1: hidden += deepstack_feature_1[visual_positions]
    │   ├── Layer 2: hidden += deepstack_feature_2[visual_positions]
    │   └── Layers 3-35: 标准 Transformer
    │
    │   每层包含:
    │   ├── RMSNorm + Self-Attention (需要 MRoPE)
    │   ├── Residual Add
    │   ├── RMSNorm + FFN
    │   └── Residual Add
    │
Final Hidden [seq_len, 4096]
    ↓ LM Head
Logits [seq_len, 151936]
    ↓ Sampling
Generated Token
```

---

## 3. 实现状态

### 3.1 已完成 ✅

| 组件 | 状态 | 说明 |
|------|------|------|
| 模型权重加载 | ✅ 完成 | 支持 FP16 格式 |
| 图像预处理 | ✅ 完成 | smart_resize, normalize, patch extraction |
| Vision Patch Embedding | ✅ 完成 | Conv3D 等效实现 |
| Vision Position Embedding | ✅ 完成 | 支持插值 |
| Vision 2D RoPE | ✅ 完成 | 使用 (height, width) 位置 |
| Vision Self-Attention | ✅ 完成 | cuBLAS 实现 |
| Vision MLP | ✅ 完成 | GELU 激活 |
| Vision Merger | ✅ 完成 | 4 patches → 1 token |
| DeepStack Features | ✅ 完成 | 层 8, 16, 24 提取 |
| Multimodal Embedding | ✅ 完成 | 视觉 + 文本融合 |
| LLM Inference | ✅ 完成 | 36 层 Transformer |
| DeepStack Injection | ✅ 完成 | 前 3 层注入 |

### 3.2 待优化 ⚠️

| 组件 | 状态 | 说明 |
|------|------|------|
| **LLM MRoPE** | ⚠️ 待实现 | 当前使用 1D RoPE，需要 3D MRoPE |
| Prefill 批处理 | ⚠️ 待优化 | 当前逐 token 处理，约 100 秒/1000 tokens |
| Vision Attention 优化 | ⚠️ 待优化 | 当前 cuBLAS 逐头处理 |

### 3.3 当前输出问题

**症状**: 生成的文本为乱码
```
输出示例: slides/net粳 invalid monkslb leakage为啥 Clyde肇庆...
```

**根因分析**: LLM 侧未实现 MRoPE (Multimodal RoPE)

---

## 4. MRoPE 技术分析

### 4.1 MRoPE vs 标准 RoPE

**标准 RoPE (1D)**:
- position_ids: [0, 1, 2, 3, ...]
- 所有头使用相同的位置编码

**MRoPE (3D)**:
- position_ids: [3, batch_size, seq_len]
  - position_ids[0]: temporal (时间)
  - position_ids[1]: height (高度)
  - position_ids[2]: width (宽度)
- 头维度分段: `mrope_section = [24, 20, 20]`
  - 维度 0-47: 使用 temporal 位置编码  
  - 维度 48-87: 使用 height 位置编码
  - 维度 88-127: 使用 width 位置编码

### 4.2 视觉 Token 的 3D 位置

对于 grid_thw = [1, 52, 76] (1帧, 52行, 76列 after merge):

```python
# Python 参考实现
# merged grid: 52/2=26 行, 76/2=38 列 = 988 tokens
t_index = [0, 0, 0, ...]  # 总是 0 (单帧)
h_index = [0, 0, ..., 0, 1, 1, ..., 1, ...]  # 每行 38 个 token
w_index = [0, 1, 2, ..., 37, 0, 1, 2, ...]   # 列索引循环
```

### 4.3 实现方案

需要修改的文件:
1. `rope_kernel.cu`: 添加 MRoPE kernel
2. `qwen3_vl.cpp`: 计算 3D position_ids
3. `qwen3_vl.h`: 添加 MRoPE 相关配置

```cpp
// 伪代码
void compute_mrope_positions(
    int image_pos_start,
    int num_vision_tokens,
    int grid_h, int grid_w,
    int* position_ids_t,  // [seq_len]
    int* position_ids_h,  // [seq_len]
    int* position_ids_w   // [seq_len]
) {
    int merged_h = grid_h / 2;
    int merged_w = grid_w / 2;
    
    for (int i = 0; i < seq_len; i++) {
        if (i >= image_pos_start && i < image_pos_start + num_vision_tokens) {
            // 视觉 token
            int visual_idx = i - image_pos_start;
            position_ids_t[i] = text_len + 0;  // 时间维度使用文本长度偏移
            position_ids_h[i] = text_len + visual_idx / merged_w;
            position_ids_w[i] = text_len + visual_idx % merged_w;
        } else {
            // 文本 token - 所有维度使用相同位置
            position_ids_t[i] = i;
            position_ids_h[i] = i;
            position_ids_w[i] = i;
        }
    }
}
```

---

## 5. 性能分析

### 5.1 当前性能 (NVIDIA Orin)

| 阶段 | 时间 | 说明 |
|------|------|------|
| 模型加载 | ~9.5s | 16GB FP16 权重 |
| 图像预处理 | ~0.5s | 包括 resize, normalize |
| Vision Encoder | ~13s | 27 层, 3952 tokens |
| Multimodal Embedding | <0.1s | 内存拷贝 |
| LLM Prefill | ~100s | 1012 tokens, 逐 token |
| LLM Decode | ~1.5s/token | 单 token 生成 |

### 5.2 优化机会

1. **Vision Attention**: 使用 Flash Attention 或 cuDNN 优化
2. **LLM Prefill**: 实现批量 prefill (parallel attention)
3. **内存优化**: 减少临时 tensor 分配

---

## 6. 文件清单

### 6.1 核心文件

```
kuiper/
├── include/
│   └── model/
│       └── qwen3_vl.h              # 模型头文件
└── source/
    ├── model/
    │   └── qwen3_vl.cpp            # 模型实现
    └── op/kernels/cuda/
        └── vision_encoder_kernel.cu # Vision CUDA kernels
```

### 6.2 主要 Kernel 函数

| 函数 | 用途 |
|------|------|
| `vision_patch_embed_cu` | Patch embedding |
| `vision_pos_embed_interpolate_cu` | 位置嵌入插值 |
| `vision_rope_cu` | Vision 2D RoPE |
| `vision_flash_attention_cu` | Vision self-attention |
| `vision_mlp_cu` | Vision MLP |
| `vision_merger_cu` | Patch merging |

---

## 7. 下一步计划

### 短期 (High Priority)

1. **实现 LLM MRoPE**
   - 添加 3D position_ids 计算
   - 修改 RoPE kernel 支持 mrope_section
   - 测试验证输出正确性

### 中期 (Medium Priority)

2. **性能优化**
   - 实现 batched prefill
   - 优化 Vision attention (Flash Attention)
   - 减少内存分配

### 长期 (Low Priority)

3. **功能扩展**
   - 视频输入支持
   - 多图像输入支持
   - AWQ 量化支持

---

## 8. 参考资源

1. [Qwen3-VL 官方仓库](https://github.com/QwenLM/Qwen3-VL)
2. [HuggingFace Transformers - Qwen3VL](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)
3. [DeepStack 论文](https://arxiv.org/abs/2406.04334)

---

## 附录 A: 特殊 Token ID

| Token | ID | 用途 |
|-------|-----|------|
| `<\|vision_start\|>` | 151652 | 视觉序列开始 |
| `<\|vision_end\|>` | 151653 | 视觉序列结束 |
| `<\|image\|>` | 151655 | 图像占位符 |
| `<\|video\|>` | 151656 | 视频占位符 |
| `<\|im_start\|>` | 151644 | 消息开始 |
| `<\|im_end\|>` | 151645 | 消息结束 |

---

## 附录 B: 测试命令

```bash
# 编译
cd build
cmake -DQWEN3_VL_SUPPORT=ON ..
make qwen3_vl_infer

# 运行推理
./demo/qwen3_vl_infer \
    /mnt/ssd/QwenModels/Qwen3-VL-8B-fp16.bin \
    /mnt/ssd/QwenModels/Qwen3-VL-8B-Instruct/tokenizer.json \
    --image /mnt/ssd/QwenModels/demo.jpg \
    --prompt "describe the picture" \
    --max-tokens 64 \
    --verbose
```

---

*报告生成时间: 2026-02-02*
*作者: KuiperLLama Team*
