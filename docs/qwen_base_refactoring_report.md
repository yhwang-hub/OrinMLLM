# Qwen 模型代码复用重构报告

## 一、重构背景与动机

重构前，`Qwen2Model`（2104行）、`Qwen3Model`（2095行）、`Qwen3VLModel`（3088行）三个模型文件各自独立继承 `Model` 基类，存在**约 1800 行几乎完全相同的代码**。

通过逐一对比三个模型的方法实现，发现差异集中在极少数地方：

| 方法 | Qwen2 vs Qwen3 差异 |
|------|---------------------|
| `forward`, `predict`, `embedding`, `cls_logits`, `post_processing` | **完全相同** |
| `attention_rms`, `attention_mha`, `attention_mha_with_graph` | **完全相同** |
| `feed_forward`, `feed_forward_fused` | Qwen3 多了 AWQ 分支判断，Qwen2 没有 → 用超集方案统一 |
| `batched_*`（6个方法）, `prefill`, `decode`, `clear_kv_cache` | 同上，Qwen3 多 AWQ 判断 |
| `set_attention_type` | Qwen3 额外向 `flash_attention_decode_gpu_pos_layer_` 传播 |
| **`attention_qkv` / `attention_qkv_with_graph` / `batched_attention_qkv`** | **核心差异**：Qwen2 有 Q/K/V bias，Qwen3 有 Q/K 逐头 RMSNorm + AWQ 支持 |

**唯一真正不同的只有 Q/K/V 投影相关的 3 个方法**，其余 20 个方法可以共享。

### 为什么 Qwen3VL 不纳入继承体系

Qwen3VL 与 Qwen2/Qwen3 存在根本性的 API 差异，强行统一会引入过多条件分支：

- 批量矩阵乘使用直接 `cublasHgemm` 调用而非 layer 抽象
- 使用 M-RoPE（多维位置编码）而非标准 RoPE
- Flash Attention 调用方式不同（`attention_qkv_with_graph` 接收 2 个 pos tensor）
- 有完全独立的视觉编码器（ViT）逻辑

---

## 二、重构策略：模板方法模式

采用经典的 **模板方法（Template Method）** 设计模式：

1. **提取公共基类 `QwenBaseModel`**：实现所有共享的推理逻辑
2. **将差异点定义为纯虚函数**：`attention_qkv()`, `attention_qkv_with_graph()`, `batched_attention_qkv()`
3. **通过多态访问层指针**：定义 `QwenBaseLayers` 基础结构体 + `get_base_layers()` 纯虚函数

### 继承关系变化

```
重构前:                              重构后:
Model                                Model
├── Qwen2Model   (2104行)            ├── QwenBaseModel (820行) 🆕
├── Qwen3Model   (2095行)            │   ├── Qwen2Model  (1145行) ⬇-959
└── Qwen3VLModel (3088行)            │   └── Qwen3Model  (1262行) ⬇-833
                                     └── Qwen3VLModel    (3088行) 不变
```

---

## 三、具体修改内容

### 3.1 新增 `kuiper/include/model/qwen_base.h`（192行）

定义两个核心抽象：

**`QwenBaseLayers` 结构体**：提取所有共享的 layer 指针

```cpp
struct QwenBaseLayers {
  // 非参数层（全模型共享单实例）
  std::shared_ptr<op::Layer> add_layer_, rope_layer_, swiglu_layer_, mha_layer_;

  // 每层参数权重层
  std::vector<std::shared_ptr<op::Layer>> wq_layers_, wk_layers_, wv_layers_, wo_layers_;
  std::vector<std::shared_ptr<op::Layer>> w1_layers_, w2_layers_, w3_layers_, rmsnorm_layers_;
  std::shared_ptr<op::Layer> cls_layer_, embedding_layer_;

  // Flash Attention / KV Cache / Fused FFN / Batched 层...
  virtual ~QwenBaseLayers() = default;  // 虚析构，支持多态
};
```

**`QwenBaseModel` 类**：声明共享方法 + 3 个纯虚接口

```cpp
class QwenBaseModel : public Model {
 protected:
  // 子类必须实现（模型差异点）
  virtual QwenBaseLayers* get_base_layers() const = 0;
  virtual void attention_qkv(...) const = 0;
  virtual void attention_qkv_with_graph(...) const = 0;
  virtual void batched_attention_qkv(...) const = 0;

  // 共享实现（20个方法）
  void attention_rms(...) const;
  void attention_mha(...) const;
  void attention_mha_with_graph(...) const;
  void feed_forward(...) const;
  void feed_forward_fused(...) const;
  void cls_logits(...) const;
  // ... 等等

  std::shared_ptr<kernel::CudaConfig> cuda_config_;
  bool use_fused_ffn_ = true;
};
```

### 3.2 新增 `kuiper/source/model/qwen_base.cpp`（820行）

实现了 20 个共享方法。关键设计决策：

**AWQ 兼容性处理（超集方案）**：原来 Qwen2 的 `feed_forward_fused` 不检查 AWQ，Qwen3 的会检查。统一采用 Qwen3 的方式——通过 `dynamic_pointer_cast<op::AWQMatmulLayer>` 尝试转型：

```cpp
auto w1_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(w1_layer);
if (w1_awq) {
    // AWQ 路径（Qwen3-AWQ 走这里）
} else {
    // 标准 MatmulLayer 路径（Qwen2 和 Qwen3-FP16 走这里）
}
```

对 Qwen2 而言，`dynamic_pointer_cast` 始终返回 `nullptr`，自然走标准路径——行为与原代码完全一致，无额外开销。

**层指针访问**通过 `get_base_layers()` 虚函数实现多态：

```cpp
void QwenBaseModel::attention_rms(int32_t layer_idx, const tensor::Tensor& input) const {
  auto* layers = get_base_layers();  // 多态调用，返回 Qwen2Layers* 或 Qwen3Layers*
  // 通过基类指针访问共有成员
  layers->rmsnorm_layers_.at(layer_idx)->forward(input, rmsnorm_output);
}
```

### 3.3 修改 `kuiper/include/model/qwen2.h`（165行 → 48行）

- `#include "model.h"` → `#include "qwen_base.h"`
- `class Qwen2Model : public Model` → `class Qwen2Model : public QwenBaseModel`
- `struct Qwen2Layers { 所有层... }` → `struct Qwen2Layers : public QwenBaseLayers { 仅2个特有层 }`
- 删除 20 个共享方法声明
- 添加 `get_base_layers() override` 和 3 个纯虚函数的 `override` 声明

`Qwen2Layers` 只保留 Qwen2 特有的层：

```cpp
struct Qwen2Layers : public QwenBaseLayers {
  std::shared_ptr<op::BatchedMatmulLayer> batched_matmul_layer_;  // Qwen2特有
  std::shared_ptr<op::BiasAddLayer> bias_add_layer_;               // Qwen2特有（Q/K/V bias）
};
```

### 3.4 修改 `kuiper/include/model/qwen3.h`（200行 → 74行）

同理，`Qwen3Layers` 只保留 Qwen3 特有的层：

```cpp
struct Qwen3Layers : public QwenBaseLayers {
  std::shared_ptr<op::MRoPELayer> mrope_layer_;                     // M-RoPE（VL用）
  std::shared_ptr<op::FlashAttentionDecodeGpuPosLayer> ...;         // GPU pos FA
  std::shared_ptr<op::RMSNormDimLayer> rmsnorm_dim_layer_;          // Q/K 逐头 RMSNorm
  // ... 等
};
```

Qwen3 额外 override 了 `set_attention_type`，因为它需要向 `flash_attention_decode_gpu_pos_layer_` 也传播 attention type（基类版本只传播到基础的 FA 层）。

### 3.5 修改 `kuiper/source/model/qwen2.cpp`（2104行 → 1145行，删除 959 行）

- **删除** 20 个已移至基类的方法实现
- **保留** `Qwen2Layers::to_cuda`、构造函数、`init`、`init_mem`、`create_*_layers`、以及 3 个 QKV 方法
- **修改** 构造函数委托：`Model(...)` → `QwenBaseModel(...)`

### 3.6 修改 `kuiper/source/model/qwen3.cpp`（2095行 → 1262行，删除 833 行）

同上处理。额外保留了 Qwen3 特有的 `set_attention_type` 和 `create_param_layers_awq`。

### 3.7 未修改 `kuiper/source/model/qwen3_vl.cpp`（3088行不变）

如上所述，Qwen3VL 的 API 差异过大，不纳入此继承体系。

---

## 四、重构后的完整类结构

```
Model (抽象基类)
│
├── QwenBaseModel (抽象基类，新增)
│   │
│   │  【共享成员】
│   │  #cuda_config_ : shared_ptr<CudaConfig>
│   │  #use_fused_ffn_ : bool
│   │
│   │  【共享方法实现 (20个)】
│   │  +forward(), +predict(), +embedding()
│   │  +prefill(), +decode()
│   │  +clear_kv_cache(), +set_attention_type()
│   │  #attention_rms(), #attention_mha(), #attention_mha_with_graph()
│   │  #feed_forward(), #feed_forward_fused()
│   │  #cls_logits(), #post_processing()
│   │  #batched_attention_rms() x2
│   │  #batched_attention_mha() x2
│   │  #batched_feed_forward(), #batched_feed_forward_optimized()
│   │
│   │  【纯虚接口 (4个)】
│   │  #get_base_layers() = 0
│   │  #attention_qkv() = 0
│   │  #attention_qkv_with_graph() = 0
│   │  #batched_attention_qkv() = 0
│   │
│   ├── Qwen2Model
│   │   - qwen_layers_ : unique_ptr<Qwen2Layers>
│   │   + get_base_layers() → qwen_layers_.get()
│   │   + attention_qkv()           【特有：Q/K/V bias】
│   │   + attention_qkv_with_graph() 【特有：Q/K/V bias + GPU pos】
│   │   + batched_attention_qkv()   【特有：批量 bias_add】
│   │   + init(), init_mem(), create_*_layers()
│   │
│   └── Qwen3Model
│       - qwen_layers_ : unique_ptr<Qwen3Layers>
│       + get_base_layers() → qwen_layers_.get()
│       + attention_qkv()           【特有：Q/K RMSNorm】
│       + attention_qkv_with_graph() 【特有：Q/K RMSNorm + GPU pos】
│       + batched_attention_qkv()   【特有：AWQ + Q/K RMSNorm】
│       + set_attention_type()      【特有：传播到 gpu_pos FA 层】
│       + init(), init_mem(), create_*_layers(), create_param_layers_awq()
│
└── Qwen3VLModel (独立，未变更)
    - 直接继承 Model，不纳入 QwenBaseModel 体系


QwenBaseLayers (数据结构体)
│  所有共用 layer 指针：add, rope, swiglu, mha,
│  wq/wk/wv/wo, w1/w2/w3, rmsnorm, cls, embedding,
│  flash_attention_decode/prefill, kv_cache_key/value,
│  fused_ffn, rope_gpu_pos, sin_cos_cache, mha_gpu_pos,
│  batched_rope/add/swiglu/mha, batched_matmul_helper
│
├── Qwen2Layers
│   + batched_matmul_layer_   (BatchedMatmulLayer)
│   + bias_add_layer_         (BiasAddLayer，Q/K/V bias 用)
│
└── Qwen3Layers
    + mrope_layer_                        (M-RoPE，VL 用)
    + mrope_gpu_pos_layer_                (M-RoPE GPU pos)
    + batched_mrope_layer_                (批量 M-RoPE)
    + fused_kv_cache_update_layer_        (融合 KV cache 更新)
    + rmsnorm_dim_layer_                  (Q/K 逐头 RMSNorm)
    + copy_to_kv_cache_layer_             (KV cache 拷贝)
    + flash_attention_decode_gpu_pos_layer_ (GPU pos Flash Attention)
```

---

## 五、重构后的推理调用流程

### 5.1 Decode 阶段（逐 token 生成）

```
应用层 (inference_common.h) 调用:
  model.decode(input, pos, next)
    └── QwenBaseModel::decode()              [qwen_base.cpp]
        │
        ├── 【CUDA Graph 路径】
        │   ├── 准备 GPU pos, decode_input buffer
        │   ├── 首次调用时捕获 Graph:
        │   │   └── for layer_idx in 0..N-1:
        │   │       ├── attention_rms()           [基类: RMSNorm → rmsnorm_output]
        │   │       ├── attention_qkv_with_graph() ──→ 【多态分发到子类】
        │   │       │   ├── Qwen2: WQ·x + bias, WK·x + bias, WV·x + bias, RoPE(GPU pos), KV cache
        │   │       │   └── Qwen3: WQ·x, WK·x, WV·x, Q/K RMSNorm, RoPE(GPU pos), KV cache
        │   │       ├── attention_mha_with_graph() [基类: Flash Attention → mha_output → WO投影]
        │   │       └── feed_forward_fused()       [基类: 残差 + RMSNorm + Fused W1·W3·SwiGLU + W2 + 残差]
        │   ├── cls_logits()                       [基类: 最终RMSNorm + 分类头]
        │   ├── graph.launch()                     (后续调用直接重放 Graph)
        │   └── argmax_sampler → next token
        │
        └── 【普通路径】(无 Graph 或 Graph 失败时)
            ├── for layer_idx in 0..N-1:
            │   ├── attention_rms()
            │   ├── attention_qkv()  ──→ 【多态分发到子类】
            │   ├── attention_mha()
            │   └── feed_forward / feed_forward_fused
            ├── cls_logits()
            └── post_processing() → next token
```

### 5.2 Prefill 阶段（批量处理 prompt）

```
应用层 (inference_common.h) 调用:
  model.prefill(embedding_output, seq_len, start_pos)
    └── QwenBaseModel::prefill()             [qwen_base.cpp]
        │
        ├── 分配 double-buffer: hidden_buf0, hidden_buf1 (交替使用，避免拷贝)
        ├── 分配 FFN buffer: ffn_norm, w1, w3, w2 (预分配复用，避免每层重分配)
        │
        └── for layer_idx in 0..N-1:
            ├── 确定 layer_input / layer_output (double-buffer 切换)
            ├── batched_attention_rms()          [基类: 批量 RMSNorm → rms_out]
            ├── batched_attention_qkv()           ──→ 【多态分发到子类】
            │   ├── Qwen2: BatchedMatmul + bias_add + BatchedRoPE + KV cache memcpy
            │   └── Qwen3: BatchedMatmul/AWQ + Q/K RMSNorm + BatchedRoPE + KV cache memcpy
            ├── batched_attention_mha()          [基类: FA prefill + WO投影 (AWQ兼容)]
            ├── batched_add (残差连接)            [基类]
            └── batched_feed_forward_optimized() [基类: 预分配buffer版 FFN]
        │
        ├── 取最后一个 token 的 hidden state
        └── cls_logits(last_hidden)              [基类: 最终RMSNorm + 分类头]
```

### 5.3 单步 Attention 内部流程

```
每一层 Transformer Layer 的执行流程:

input ──→ [RMSNorm] ──→ rmsnorm_output
                              │
                    ┌─────────┼─────────┐
                    ↓         ↓         ↓
                  [WQ·x]    [WK·x]    [WV·x]     ← 子类实现 (attention_qkv)
                    │         │         │
              ┌─────┤   ┌─────┤         │
              │ Qwen2: +bias  +bias     │
              │ Qwen3: Q-RMSNorm K-RMSNorm
              │     │         │         │
              │   [RoPE]    [RoPE]      │
              │     │         │         ↓
              │     │         └──→ [KV Cache 更新] ← 子类实现
              │     │         │         │
              │     ↓         ↓         ↓         ← 基类实现 (attention_mha)
              │   [Flash Attention / MHA] ──→ mha_output
              │                                │
              │                              [WO·x] ──→ attn_output
              │                                              │
              └───────────────────→ [残差 Add] ←─────────────┘
                                        │
                                   [FFN RMSNorm]              ← 基类实现 (feed_forward)
                                        │
                                 ┌──────┴──────┐
                                 ↓             ↓
                               [W1·x]       [W3·x]
                                 │             │
                                 └──→ [SwiGLU] ←┘
                                        │
                                      [W2·x]
                                        │
              input ──────────→ [残差 Add] ←──┘
                                    │
                                  output → 下一层的 input
```

---

## 六、关键设计决策

| 决策 | 原因 |
|------|------|
| 用 `dynamic_pointer_cast` 做 AWQ 判断 | Qwen2 无 AWQ 层，cast 返回 nullptr 走标准路径，零开销兼容 |
| `QwenBaseLayers` 用虚析构 + 运行时多态 | 运行时多态足够，无需 CRTP 编译期多态；简洁易维护 |
| Qwen3VL 不纳入继承 | API 差异太大（直接 cublas、M-RoPE、ViT），强行统一会引入过多条件分支 |
| `cuda_config_` 放在基类 | 两个子类都需要，避免重复声明 |
| `set_attention_type` 允许 Qwen3 再 override | Qwen3 需要额外传播到 `flash_attention_decode_gpu_pos_layer_`，调用链为 `Qwen3::set_attention_type` → `QwenBaseModel::set_attention_type` → `Model::set_attention_type` |
| `forward()` 中使用 `use_fused_ffn_` 标志 | 基类统一控制是否使用 Fused FFN 内核，子类无需关心 |

---

## 七、代码量变化汇总

| 文件 | 重构前 | 重构后 | 变化 |
|------|--------|--------|------|
| `qwen_base.h` | 0 | 192 | +192 |
| `qwen_base.cpp` | 0 | 820 | +820 |
| `qwen2.h` | 165 | 48 | -117 |
| `qwen2.cpp` | 2,104 | 1,145 | **-959** |
| `qwen3.h` | 200 | 74 | -126 |
| `qwen3.cpp` | 2,095 | 1,262 | **-833** |
| `qwen3_vl.h` | 511 | 511 | 0 |
| `qwen3_vl.cpp` | 3,088 | 3,088 | 0 |
| **总计** | **8,163** | **7,140** | **-1,023** |

净减少 **1,023 行代码**，消除了 20 个方法在两个文件中的重复实现。

---

## 八、测试验证

所有 5 个推理场景均通过测试，输出内容与重构前一致：

| # | 测试命令 | Prefill | Decode | 结果 |
|---|---------|---------|--------|------|
| 1 | `qwen3_infer` Qwen3-8B-fp16 | 131 tok/s | 10.3 tok/s | ✅ 通过 |
| 2 | `qwen3_infer` Qwen3-8B-awq | 158 tok/s | 10.2 tok/s | ✅ 通过 |
| 3 | `qwen_infer` Qwen2.5-7B (FP32) | 6.1 tok/s | 5.7 tok/s | ✅ 通过 |
| 4 | `qwen_infer` Qwen2.5-7B-fp16 | 150 tok/s | 11.0 tok/s | ✅ 通过 |
| 5 | `qwen3_vl_infer` Qwen3-VL-8B-fp16 | 499 tok/s | 9.8 tok/s | ✅ 通过 |

性能数据与重构前完全一致，重构仅改变代码组织结构，不影响运行时行为和性能。

---

## 九、后续维护收益

重构后，以下常见修改场景只需改动 **一处** 而非原来的 **两处**：

- 修改 decode/prefill 循环逻辑 → 改 `qwen_base.cpp`
- 修改 Flash Attention 调用方式 → 改 `qwen_base.cpp`
- 修改 CUDA Graph 捕获/重放策略 → 改 `qwen_base.cpp`
- 修改 KV cache 管理逻辑 → 改 `qwen_base.cpp`
- 修改 Fused FFN 内核调用 → 改 `qwen_base.cpp`
- 修改采样/后处理逻辑 → 改 `qwen_base.cpp`
- 添加新的共享优化（如 Attention 融合）→ 改 `qwen_base.cpp`，子类无需变动

如需添加新的 Qwen 变体（如 Qwen4），只需继承 `QwenBaseModel` 并实现 4 个纯虚函数即可获得完整的推理能力。
