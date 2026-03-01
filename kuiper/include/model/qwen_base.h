#ifndef KUIPER_INCLUDE_MODEL_QWEN_BASE_H_
#define KUIPER_INCLUDE_MODEL_QWEN_BASE_H_
#include <base/cuda_config.h>
#include "model.h"
#include "op/add.h"
#include "op/embedding.h"
#include "op/rope.h"
#include "op/swiglu.h"
#include "op/flash_attention.h"
#include "op/kv_cache.h"
#include "op/fused_ffn.h"
#include "op/batched_matmul.h"
#include "op/batched_rope.h"
#include "op/batched_add.h"
#include "op/misc_layers.h"

namespace model {

/**
 * @brief Common layer pointers shared by all Qwen model variants (Qwen2, Qwen3, Qwen3-VL).
 * 
 * Derived structs (Qwen2Layers, Qwen3Layers) add model-specific layers.
 */
struct QwenBaseLayers {
  // Core operator layers (non-parametric, shared instances)
  std::shared_ptr<op::Layer> add_layer_;
  std::shared_ptr<op::Layer> rope_layer_;
  std::shared_ptr<op::Layer> swiglu_layer_;
  std::shared_ptr<op::Layer> mha_layer_;

  // Per-layer parametric weight layers
  std::vector<std::shared_ptr<op::Layer>> wq_layers_;
  std::vector<std::shared_ptr<op::Layer>> wk_layers_;
  std::vector<std::shared_ptr<op::Layer>> wv_layers_;
  std::vector<std::shared_ptr<op::Layer>> wo_layers_;

  std::vector<std::shared_ptr<op::Layer>> w1_layers_;
  std::vector<std::shared_ptr<op::Layer>> w2_layers_;
  std::vector<std::shared_ptr<op::Layer>> w3_layers_;
  std::vector<std::shared_ptr<op::Layer>> rmsnorm_layers_;
  std::shared_ptr<op::Layer> cls_layer_;

  std::shared_ptr<op::Layer> embedding_layer_;

  // Flash attention layers (decode + prefill)
  std::shared_ptr<op::FlashAttentionDecodeLayer> flash_attention_decode_layer_;
  std::shared_ptr<op::FlashAttentionPrefillLayer> flash_attention_prefill_layer_;

  // KV cache layers
  std::shared_ptr<op::KVCacheLayer> kv_cache_key_layer_;
  std::shared_ptr<op::KVCacheLayer> kv_cache_value_layer_;

  // Fused/optimized layers
  std::shared_ptr<op::FusedFFNLayer> fused_ffn_layer_;
  std::shared_ptr<op::RoPEGpuPosLayer> rope_gpu_pos_layer_;
  std::shared_ptr<op::SinCosCacheLayer> sin_cos_cache_layer_;
  std::shared_ptr<op::MHAGpuPosLayer> mha_gpu_pos_layer_;

  // Batched layers (for prefill phase)
  std::shared_ptr<op::BatchedRoPELayer> batched_rope_layer_;
  std::shared_ptr<op::BatchedAddLayer> batched_add_layer_;
  std::shared_ptr<op::BatchedSwiGLULayer> batched_swiglu_layer_;
  std::shared_ptr<op::BatchedMHALayer> batched_mha_layer_;
  std::shared_ptr<op::BatchedMatmulHelperLayer> batched_matmul_helper_layer_;

  virtual ~QwenBaseLayers() = default;
};

/**
 * @brief Shared base class for Qwen2Model and Qwen3Model.
 * 
 * Provides common implementations for:
 * - Attention MHA (decode + CUDA Graph)
 * - Feed forward (standard + fused)
 * - Batched prefill operations
 * - Decode loop (with CUDA Graph support)
 * - Prefill loop (with double-buffering)
 * - Embedding, cls_logits, post_processing
 * - KV cache management
 * 
 * Subclasses must implement model-specific operations:
 * - attention_qkv / attention_qkv_with_graph (differ in bias vs q/k norms)
 * - batched_attention_qkv (differ in bias vs q/k norms + AWQ)
 * - Layer creation (create_param_layers, create_nonparam_layers, etc.)
 * - Memory initialization (init_mem)
 * - Model initialization (init)
 */
class QwenBaseModel : public Model {
 public:
  explicit QwenBaseModel(base::TokenizerType tokenizer_type, std::string token_path,
                         std::string model_path, bool is_quant_model);

  // === Model interface ===
  base::Status predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                       bool is_prompt, int& next) const override;

  base::Status forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                       int& next) const override;

  op::EmbeddingOutput embedding(const std::vector<int>& tokens) const override;

  // === Decode and Prefill ===
  base::Status prefill(const tensor::Tensor& input, int32_t seq_len, int32_t start_pos) const;
  base::Status decode(const tensor::Tensor& input, int32_t pos, int& next) const;

  // === CUDA Graph management ===
  void enable_cuda_graph(bool enable) {
    if (cuda_config_) {
      cuda_config_->use_cuda_graph = enable;
      if (enable && !cuda_config_->graph_context) {
        cuda_config_->graph_context = std::make_shared<base::CudaGraphContext>();
      }
    }
  }
  bool is_cuda_graph_enabled() const {
    return cuda_config_ && cuda_config_->use_cuda_graph;
  }
  void invalidate_cuda_graph() {
    if (cuda_config_) {
      cuda_config_->invalidate_graph();
    }
  }

  // === KV Cache management ===
  void clear_kv_cache();

  // === Configuration ===
  const TransformerConfig* get_config() const { return config_.get(); }
  std::shared_ptr<kernel::CudaConfig> get_cuda_config() const { return cuda_config_; }

  // === Fused FFN control ===
  void enable_fused_ffn(bool enable) { use_fused_ffn_ = enable; }
  bool is_fused_ffn_enabled() const { return use_fused_ffn_; }

  // === Attention type ===
  void set_attention_type(base::AttentionType type) override;

 protected:
  // --- Must be implemented by derived classes ---

  /// Return pointer to the model's layer struct (Qwen2Layers or Qwen3Layers)
  virtual QwenBaseLayers* get_base_layers() const = 0;

  /// Model-specific Q/K/V projection + position encoding (decode phase)
  virtual void attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const = 0;

  /// Model-specific Q/K/V projection + position encoding (CUDA Graph compatible)
  virtual void attention_qkv_with_graph(int32_t layer_idx,
                                        const tensor::Tensor& pos_tensor) const = 0;

  /// Model-specific batched Q/K/V projection + position encoding (prefill phase)
  virtual void batched_attention_qkv(int32_t layer_idx, const tensor::Tensor& rms_out,
                                     const tensor::Tensor& query_out,
                                     const tensor::Tensor& key_out,
                                     const tensor::Tensor& value_out,
                                     int32_t seq_len, int32_t start_pos) const = 0;

  // --- Shared implementations ---

  void attention_rms(int32_t layer_idx, const tensor::Tensor& input) const;
  void attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;
  void attention_mha_with_graph(int32_t layer_idx, const tensor::Tensor& pos_tensor_gpu) const;
  void feed_forward(int32_t layer_idx, const tensor::Tensor& input) const;
  void feed_forward_fused(int32_t layer_idx, const tensor::Tensor& input) const;
  void cls_logits(const tensor::Tensor& input) const;
  int32_t post_processing(const tensor::Tensor& pos, bool is_prompt) const override;

  // Batched operations (prefill phase)
  void batched_attention_rms(int32_t layer_idx, const tensor::Tensor& input,
                             int32_t seq_len) const;
  void batched_attention_rms(int32_t layer_idx, const tensor::Tensor& input,
                             const tensor::Tensor& output, int32_t seq_len) const;
  void batched_attention_mha(int32_t layer_idx, const tensor::Tensor& query,
                             const tensor::Tensor& mha_out,
                             int32_t seq_len, int32_t start_pos) const;
  void batched_attention_mha(int32_t layer_idx, const tensor::Tensor& query,
                             const tensor::Tensor& mha_out, tensor::Tensor& wo_out,
                             int32_t seq_len, int32_t start_pos) const;
  void batched_feed_forward(int32_t layer_idx, const tensor::Tensor& input,
                            int32_t seq_len) const;
  void batched_feed_forward_optimized(int32_t layer_idx, const tensor::Tensor& input,
                                      tensor::Tensor& ffn_norm_out, tensor::Tensor& w1_out,
                                      tensor::Tensor& w3_out, tensor::Tensor& w2_out,
                                      int32_t seq_len) const;

 protected:
  std::shared_ptr<kernel::CudaConfig> cuda_config_;
  bool use_fused_ffn_ = true;
};

}  // namespace model
#endif  // KUIPER_INCLUDE_MODEL_QWEN_BASE_H_
