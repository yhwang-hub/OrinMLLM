# Qwen 模型代码复用重构报告

## 一、重构背景与动机

重构前，`Qwen2Model`（2104行）、`Qwen3Model`（2095行）、`Qwen3VLModel`（3088行）三个模型文件各自独立继承 `Model` 基类，存在**约 1800 行几乎完全相同的代码**。

通过逐一对比三个模型的方法实现，发现差异集中在极少数地方：

| 方法 | Qwen2 vs Qwen3 差异 |
|------|---------------------|
| `forward`, `predict`, `embedding`, `cls_logits`, `post_processing` | **完全相同** |
| `attention_rms`, `attention_mha`, `attention_mha_with_graph` | **完全相同** |
| `feed_forward`, `feed_forward_fused` | **完全相同**（AWQ 已分离到子类） |
| `batched_*`（6个方法）, `prefill`, `decode`, `clear_kv_cache` | **完全相同**（AWQ 通过虚方法多态分发） |
| `set_attention_type` | Qwen3 额外向 `flash_attention_decode_gpu_pos_layer_` 传播 |
| **`attention_qkv` / `attention_qkv_with_graph` / `batched_attention_qkv`** | **核心差异**：Qwen2 有 Q/K/V bias，Qwen3 有 Q/K 逐头 RMSNorm |

**唯一真正不同的只有 Q/K/V 投影相关的 3 个方法**，其余 20 个方法可以共享。

### 为什么 Qwen3VL 不纳入继承体系

Qwen3VL 与 Qwen2/Qwen3 存在根本性的 API 差异，强行统一会引入过多条件分支：

- 批量矩阵乘使用直接 `cublasHgemm` 调用而非 layer 抽象
- 使用 M-RoPE（多维位置编码）而非标准 RoPE
- Flash Attention 调用方式不同（`attention_qkv_with_graph` 接收 2 个 pos tensor）
- 有完全独立的视觉编码器（ViT）逻辑

---

## 二、重构策略：模板方法模式 + AWQ 分离

采用经典的 **模板方法（Template Method）** 设计模式，并将 AWQ INT4 量化适配代码彻底分离到独立子类：

1. **提取公共基类 `QwenBaseModel`**：实现所有共享的推理逻辑
2. **将差异点定义为纯虚函数**：`attention_qkv()`, `attention_qkv_with_graph()`, `batched_attention_qkv()`
3. **通过多态访问层指针**：定义 `QwenBaseLayers` 基础结构体 + `get_base_layers()` 纯虚函数
4. **AWQ 彻底分离**：`Qwen3AWQModel` 继承 `Qwen3Model`，通过 override 虚方法实现 AWQ 特有的矩阵乘和权重加载，`qwen_base.cpp` 和 `qwen3.cpp` 中**不含任何 AWQ 符号引用**

### 继承关系变化

```
重构前:                              重构后:
Model                                Model
├── Qwen2Model   (2104行)            ├── QwenBaseModel (763行) 🆕
├── Qwen3Model   (2095行)            │   ├── Qwen2Model  (1145行) ⬇-959
└── Qwen3VLModel (3088行)            │   └── Qwen3Model  (1047行) ⬇-1048
                                     │       └── Qwen3AWQModel (250行) 🆕
                                     └── Qwen3VLModel    (3088行) 不变
```

---

## 三、具体修改内容

### 3.1 新增 `kuiper/include/model/qwen_base.h`（204行）

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

**`QwenBaseModel` 类**：声明共享方法 + 纯虚接口 + AWQ 多态分发虚方法

```cpp
class QwenBaseModel : public Model {
 protected:
  // 子类必须实现（模型差异点）
  virtual QwenBaseLayers* get_base_layers() const = 0;
  virtual void attention_qkv(...) const = 0;
  virtual void attention_qkv_with_graph(...) const = 0;
  virtual void batched_attention_qkv(...) const = 0;

  // AWQ 多态分发虚方法（基类提供 FP16/FP32 默认实现，Qwen3AWQModel override）
  virtual void batched_matmul_forward(const std::shared_ptr<op::Layer>& layer,
                                      const tensor::Tensor& input,
                                      const tensor::Tensor& output,
                                      int32_t seq_len) const;
  virtual void gate_up_swiglu(int32_t layer_idx,
                              const tensor::Tensor& input,
                              const tensor::Tensor& output) const;

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

### 3.2 新增 `kuiper/source/model/qwen_base.cpp`（763行）

实现了 20 个共享方法 + 2 个虚方法默认实现。关键设计决策：

**AWQ 多态分发方案**：通过虚方法替代 `dynamic_pointer_cast` 分支判断，使 `qwen_base.cpp` **完全不依赖 AWQ 头文件**。

**`batched_matmul_forward` 虚方法**——用于批量矩阵乘场景（wo/w1/w2/w3 投影）：

```cpp
// 基类默认实现：FP16/FP32 MatmulLayer 路径
void QwenBaseModel::batched_matmul_forward(const std::shared_ptr<op::Layer>& layer,
                                           const tensor::Tensor& input,
                                           const tensor::Tensor& output,
                                           int32_t seq_len) const {
  auto* layers = get_base_layers();
  auto matmul = std::dynamic_pointer_cast<op::MatmulLayer>(layer);
  CHECK_NE(matmul, nullptr) << "Layer is not a MatmulLayer";
  STATUS_CHECK(layers->batched_matmul_helper_layer_->forward(
      input, matmul->get_weight(0), output, seq_len, 1.f));
}
```

**`gate_up_swiglu` 虚方法**——用于 decode 阶段的 Fused FFN（W1 + W3 + SwiGLU 融合内核）：

```cpp
// 基类默认实现：使用 Fused FFN 内核
void QwenBaseModel::gate_up_swiglu(int32_t layer_idx,
                                   const tensor::Tensor& input,
                                   const tensor::Tensor& output) const {
  auto* layers = get_base_layers();
  // 获取 W1/W3 的 MatmulLayer 权重
  auto w1_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(w1_layer);
  auto w3_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(w3_layer);
  // 配置并执行 fused_ffn_layer_（单次 CUDA kernel 完成 W1·x, W3·x, SwiGLU）
  fused_ffn->set_input(0, input);
  fused_ffn->set_input(1, w1_weight);
  fused_ffn->set_input(2, w3_weight);
  fused_ffn->set_output(0, output);
  STATUS_CHECK(fused_ffn->forward());
}
```

调用方代码因此变得**极为简洁**，不含任何类型判断分支：

```cpp
void QwenBaseModel::feed_forward_fused(int32_t layer_idx, const tensor::Tensor& input) const {
  // ...
  tensor::Tensor w1_output = get_buffer(ModelBufferType::kW1Output);
  gate_up_swiglu(layer_idx, ffn_norm_output, w1_output);  // 虚调用，多态分发
  // w2 + residual ...
}

void QwenBaseModel::batched_feed_forward_optimized(...) const {
  // ...
  batched_matmul_forward(w1_layer, ffn_norm_out, w1_out, seq_len);  // 虚调用
  batched_matmul_forward(w3_layer, ffn_norm_out, w3_out, seq_len);  // 虚调用
  // SwiGLU ...
  batched_matmul_forward(w2_layer, w1_out, w2_out, seq_len);        // 虚调用
  // residual ...
}
```

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

### 3.4 修改 `kuiper/include/model/qwen3.h`（200行 → 76行）

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

关键新增：`Qwen3Model` 声明了 `batched_qkv_projection()` 虚方法，将 QKV 矩阵投影（FP16 vs AWQ）的分发点从 `batched_attention_qkv()` 中提取出来：

```cpp
class Qwen3Model : public QwenBaseModel {
 protected:
  // 将 QKV 矩阵投影提取为虚方法，供 Qwen3AWQModel override
  virtual void batched_qkv_projection(int32_t layer_idx, const tensor::Tensor& rms_out,
                                      const tensor::Tensor& query_out, const tensor::Tensor& key_out,
                                      const tensor::Tensor& value_out, int32_t seq_len) const;
  // ...
};
```

### 3.5 新增 `kuiper/include/model/qwen3_awq.h`（50行）

AWQ INT4 量化模型的子类声明：

```cpp
class Qwen3AWQModel : public Qwen3Model {
 protected:
  // 权重加载 override
  void create_param_layers() override;
  void create_param_quant_layers() override;

  // QKV 投影 override：使用 AWQMatmulLayer::forward()
  void batched_qkv_projection(...) const override;

  // 批量矩阵乘 override：AWQ 层使用 AWQMatmulLayer::forward()，非 AWQ 层回退基类
  void batched_matmul_forward(...) const override;

  // Gate/Up + SwiGLU override：AWQ 不支持 Fused FFN 内核，使用分离操作
  void gate_up_swiglu(...) const override;

 private:
  void create_param_layers_awq();
};

// 模型文件类型检测（读取二进制 header 中的 magic number）
bool is_awq_model_file(const std::string& model_path);
```

### 3.6 新增 `kuiper/source/model/qwen3_awq.cpp`（250行）

实现 AWQ 特有的所有逻辑：

**权重加载**（`create_param_layers_awq`）：读取 AWQ INT4 格式的 qweight/qzeros/scales 三元组权重。

**三个虚方法的 AWQ override**：

```cpp
// 批量矩阵乘：AWQ 层直接调用 AWQMatmulLayer::forward()
void Qwen3AWQModel::batched_matmul_forward(const std::shared_ptr<op::Layer>& layer,
                                            const tensor::Tensor& input,
                                            const tensor::Tensor& output,
                                            int32_t seq_len) const {
  auto awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(layer);
  if (awq) {
    STATUS_CHECK(awq->forward(input, output));  // AWQ 路径
  } else {
    Qwen3Model::batched_matmul_forward(layer, input, output, seq_len);  // 回退基类
  }
}

// Gate/Up + SwiGLU：AWQ 不支持 Fused FFN，使用分离的 W1·x, W3·x, SwiGLU 操作
void Qwen3AWQModel::gate_up_swiglu(int32_t layer_idx,
                                    const tensor::Tensor& input,
                                    const tensor::Tensor& output) const {
  auto* layers = get_base_layers();
  tensor::Tensor w3_output = get_buffer(ModelBufferType::kW3Output);
  STATUS_CHECK(w1_layer->forward(input, output));
  STATUS_CHECK(w3_layer->forward(input, w3_output));
  STATUS_CHECK(layers->swiglu_layer_->forward(output, w3_output, output));
}

// QKV 投影：使用 AWQMatmulLayer::forward() 替代 batched_matmul_helper
void Qwen3AWQModel::batched_qkv_projection(...) const {
  auto query_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(query_layer);
  STATUS_CHECK(query_awq->forward(rms_out, query_out));
  STATUS_CHECK(key_awq->forward(rms_out, key_out));
  STATUS_CHECK(value_awq->forward(rms_out, value_out));
}
```

### 3.7 修改 `kuiper/source/model/qwen2.cpp`（2104行 → 1145行，删除 959 行）

- **删除** 20 个已移至基类的方法实现
- **保留** `Qwen2Layers::to_cuda`、构造函数、`init`、`init_mem`、`create_*_layers`、以及 3 个 QKV 方法
- **修改** 构造函数委托：`Model(...)` → `QwenBaseModel(...)`

### 3.8 修改 `kuiper/source/model/qwen3.cpp`（2095行 → 1047行，删除 1048 行）

删除了移至基类的共享方法，并**彻底移除所有 AWQ 相关代码**：

- 移除 `#include <op/awq_matmul.h>` 和 `#include "../op/kernels/cuda/awq_gemm_tensorcore.cuh"`
- 移除原 `create_param_layers_awq()` 实现（移至 `qwen3_awq.cpp`）
- 移除 `batched_attention_qkv()` 中的 AWQ `dynamic_pointer_cast` 分支，改为调用虚方法 `batched_qkv_projection()`
- 移除 `create_param_layers()` 和 `create_param_quant_layers()` 中的 AWQ 分支

`qwen3.cpp` 中**不再包含任何 AWQ 头文件或 AWQ 符号引用**。

### 3.9 修改 `demo/main_qwen3.cpp`

添加模型文件自动检测，根据二进制 header 中的 magic number 自动选择实例化 `Qwen3Model` 或 `Qwen3AWQModel`：

```cpp
#include "model/qwen3.h"
#include "model/qwen3_awq.h"

int main(int argc, char* argv[]) {
    // ...
    if (argc >= 2 && model::is_awq_model_file(argv[1])) {
        return inference::run_model_inference<model::Qwen3AWQModel>(...);
    }
    return inference::run_model_inference<model::Qwen3Model>(...);
}
```

### 3.10 未修改 `kuiper/source/model/qwen3_vl.cpp`（3088行不变）

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
│   │  【AWQ 多态分发虚方法 (2个，提供 FP16/FP32 默认实现)】
│   │  #batched_matmul_forward()  [FP16: batched_matmul_helper, AWQ: AWQMatmulLayer]
│   │  #gate_up_swiglu()          [FP16: fused FFN kernel, AWQ: 分离 W1/W3/SwiGLU]
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
│       + batched_attention_qkv()   【特有：Q/K RMSNorm + batched RoPE + KV cache】
│       + batched_qkv_projection()  【虚方法：FP16 matmul，供 AWQ 子类 override】
│       + set_attention_type()      【特有：传播到 gpu_pos FA 层】
│       + init(), init_mem(), create_*_layers()
│       │
│       └── Qwen3AWQModel          🆕
│           + create_param_layers()      【override：AWQ INT4 权重加载】
│           + create_param_quant_layers() 【override：空操作】
│           + batched_qkv_projection()   【override：AWQMatmulLayer::forward()】
│           + batched_matmul_forward()   【override：AWQ 层用 AWQMatmulLayer，其余回退基类】
│           + gate_up_swiglu()           【override：分离 W1/W3/SwiGLU（AWQ 不支持 fused）】
│           + is_awq_model_file()        【静态检测：读取 magic number 判断 AWQ 格式】
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
        │   │       │   └── Qwen3/AWQ: WQ·x, WK·x, WV·x, Q/K RMSNorm, RoPE(GPU pos), KV cache
        │   │       ├── attention_mha_with_graph() [基类: Flash Attention → mha_output → WO投影]
        │   │       └── feed_forward_fused()       [基类: 残差 + RMSNorm + gate_up_swiglu() + W2 + 残差]
        │   │                                            └── 虚调用 gate_up_swiglu():
        │   │                                                ├── FP16: Fused W1·W3·SwiGLU 内核
        │   │                                                └── AWQ:  分离 W1·x, W3·x, SwiGLU
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
            │   └── Qwen3: batched_qkv_projection() + Q/K RMSNorm + BatchedRoPE + KV cache memcpy
            │              └── 虚调用 batched_qkv_projection():
            │                  ├── FP16: batched_matmul_helper->forward(weight)
            │                  └── AWQ:  AWQMatmulLayer->forward()
            ├── batched_attention_mha()          [基类: FA prefill + WO投影]
            │   └── WO 投影通过虚调用 batched_matmul_forward():
            │       ├── FP16: batched_matmul_helper->forward(weight)
            │       └── AWQ:  AWQMatmulLayer->forward()
            ├── batched_add (残差连接)            [基类]
            └── batched_feed_forward_optimized() [基类: 预分配buffer版 FFN]
                └── W1/W3/W2 投影通过虚调用 batched_matmul_forward()
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
                    │         │         │            Qwen3AWQ: AWQMatmulLayer::forward()
              ┌─────┤   ┌─────┤         │            Qwen3/Qwen2 FP16: MatmulLayer
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
              │                                │         (虚调用 batched_matmul_forward)
              └───────────────────→ [残差 Add] ←─────────────┘
                                        │
                                   [FFN RMSNorm]              ← 基类实现 (feed_forward)
                                        │
                                 ┌──────┴──────┐
                                 ↓             ↓
                               [W1·x]       [W3·x]       ← 虚调用 gate_up_swiglu()
                                 │             │             FP16: Fused FFN kernel
                                 └──→ [SwiGLU] ←┘            AWQ: 分离操作
                                        │
                                      [W2·x]             ← 虚调用 batched_matmul_forward()
                                        │
              input ──────────→ [残差 Add] ←──┘
                                    │
                                  output → 下一层的 input
```

---

## 六、关键设计决策

| 决策 | 原因 |
|------|------|
| AWQ 分离为独立子类 `Qwen3AWQModel` | 使 `qwen_base.cpp` 和 `qwen3.cpp` 完全不依赖 AWQ 头文件，代码更整洁 |
| 用虚方法替代 `dynamic_pointer_cast` 分支 | 基类中不再需要 AWQ 类型判断，FP16/AWQ 差异通过多态自然分发 |
| `batched_matmul_forward` + `gate_up_swiglu` 两个虚方法 | 精确覆盖 AWQ 与 FP16 在矩阵乘和 Fused FFN 上的差异，最小化虚方法数量 |
| AWQ `batched_matmul_forward` 中保留回退逻辑 | AWQ 模型的部分层（如 embedding、rmsnorm）仍是 MatmulLayer，需要回退到基类路径 |
| `batched_qkv_projection` 从 `batched_attention_qkv` 中提取 | QKV 投影是 AWQ 与 FP16 的分叉点，提取后共享的 Q/K Norm、RoPE、KV cache 代码不重复 |
| `QwenBaseLayers` 用虚析构 + 运行时多态 | 运行时多态足够，无需 CRTP 编译期多态；简洁易维护 |
| Qwen3VL 不纳入继承 | API 差异太大（直接 cublas、M-RoPE、ViT），强行统一会引入过多条件分支 |
| `cuda_config_` 放在基类 | 两个子类都需要，避免重复声明 |
| `set_attention_type` 允许 Qwen3 再 override | Qwen3 需要额外传播到 `flash_attention_decode_gpu_pos_layer_`，调用链为 `Qwen3::set_attention_type` → `QwenBaseModel::set_attention_type` → `Model::set_attention_type` |
| `forward()` 中使用 `use_fused_ffn_` 标志 | 基类统一控制是否使用 Fused FFN 内核，子类无需关心 |
| `is_awq_model_file()` 模型文件检测 | 读取二进制 header 中的 magic number（0x616b3438 = AWQ），main 函数自动选择实例化正确的模型类 |

---

## 七、代码量变化汇总

| 文件 | 重构前 | 重构后 | 变化 |
|------|--------|--------|------|
| `qwen_base.h` | 0 | 204 | +204 |
| `qwen_base.cpp` | 0 | 763 | +763 |
| `qwen3_awq.h` | 0 | 50 | +50 |
| `qwen3_awq.cpp` | 0 | 250 | +250 |
| `qwen2.h` | 165 | 48 | -117 |
| `qwen2.cpp` | 2,104 | 1,145 | **-959** |
| `qwen3.h` | 200 | 76 | -124 |
| `qwen3.cpp` | 2,095 | 1,047 | **-1,048** |
| `qwen3_vl.h` | 511 | 511 | 0 |
| `qwen3_vl.cpp` | 3,088 | 3,088 | 0 |
| **总计** | **8,163** | **7,182** | **-981** |

净减少 **981 行代码**，消除了 20 个方法在两个文件中的重复实现，并将 AWQ 代码从 3 个文件（`qwen_base.cpp`、`qwen3.cpp`、`qwen3.h`）集中到 2 个专用文件（`qwen3_awq.h`、`qwen3_awq.cpp`）。

---

## 八、测试验证

所有 5 个推理场景均通过测试，输出内容与重构前一致：

| # | 测试命令 | Prefill | Decode | 结果 |
|---|---------|---------|--------|------|
| 1 | `qwen3_infer` Qwen3-8B-fp16 | 136 tok/s | 10.2 tok/s | ✅ 通过 |
| 2 | `qwen3_infer` Qwen3-8B-awq | 159 tok/s | 10.6 tok/s | ✅ 通过 |
| 3 | `qwen_infer` Qwen2.5-7B (FP32) | 6.1 tok/s | 5.7 tok/s | ✅ 通过 |
| 4 | `qwen_infer` Qwen2.5-7B-fp16 | 154 tok/s | 10.9 tok/s | ✅ 通过 |
| 5 | `qwen3_vl_infer` Qwen3-VL-8B-fp16 | 499 tok/s | 9.5 tok/s | ✅ 通过 |

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
- 修改 AWQ 量化适配逻辑 → **只改 `qwen3_awq.cpp`**，`qwen_base.cpp` 和 `qwen3.cpp` 无需变动

如需添加新的 Qwen 变体（如 Qwen4），只需继承 `QwenBaseModel` 并实现 4 个纯虚函数即可获得完整的推理能力。如需添加新的量化格式（如 GPTQ），只需继承 `Qwen3Model`（类似 `Qwen3AWQModel`）并 override 3 个虚方法即可。
