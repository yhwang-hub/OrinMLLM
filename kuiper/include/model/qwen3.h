#ifndef KUIPER_INCLUDE_MODEL_QWEN3_H_
#define KUIPER_INCLUDE_MODEL_QWEN3_H_
#include <base/cuda_config.h>
#include "model.h"
#include "op/add.h"
#include "op/embedding.h"
#include "op/rope.h"
#include "op/swiglu.h"
#include "op/awq_matmul.h"
#include "op/flash_attention.h"
#include "op/kv_cache.h"
#include "op/fused_ffn.h"
#include "op/batched_matmul.h"
#include "op/batched_rope.h"
#include "op/batched_add.h"
#include "op/misc_layers.h"

namespace model {
struct QWen3TransformerConfig {
  int32_t kv_dim_ = 0;
  int32_t kv_mul_ = 0;
  int32_t head_size_ = 0;
  int32_t immediate_size_ = 0;
  int32_t vocab_size_ = 0;

  int32_t dim_ = 0;
  int32_t hidden_dim_ = 0;
  int32_t layer_num_ = 0;
  int32_t head_num_ = 0;
  int32_t kv_head_num_ = 0;
  int32_t seq_len_ = 0;
  bool is_shared_weight_ = false;
};

struct Qwen3Layers {
  std::shared_ptr<op::Layer> add_layer_;
  std::shared_ptr<op::Layer> rope_layer_;
  std::shared_ptr<op::Layer> swiglu_layer_;
  std::shared_ptr<op::Layer> mha_layer_;

  std::vector<std::shared_ptr<op::Layer>> wq_layers_;
  std::vector<std::shared_ptr<op::Layer>> wk_layers_;
  std::vector<std::shared_ptr<op::Layer>> wv_layers_;
  std::vector<std::shared_ptr<op::Layer>> wo_layers_;

  std::vector<std::shared_ptr<op::Layer>> w1_layers_;
  std::vector<std::shared_ptr<op::Layer>> w2_layers_;
  std::vector<std::shared_ptr<op::Layer>> rmsnorm_layers_;
  std::vector<std::shared_ptr<op::Layer>> w3_layers_;
  std::shared_ptr<op::Layer> cls_layer_;

  std::shared_ptr<op::Layer> embedding_layer_;
  
  // New layers for unified kernel access (use concrete types for setter access)
  std::shared_ptr<op::FlashAttentionDecodeLayer> flash_attention_decode_layer_;
  std::shared_ptr<op::FlashAttentionPrefillLayer> flash_attention_prefill_layer_;
  std::shared_ptr<op::KVCacheLayer> kv_cache_key_layer_;
  std::shared_ptr<op::KVCacheLayer> kv_cache_value_layer_;
  std::shared_ptr<op::FusedFFNLayer> fused_ffn_layer_;
  std::shared_ptr<op::RoPEGpuPosLayer> rope_gpu_pos_layer_;
  std::shared_ptr<op::BatchedRoPELayer> batched_rope_layer_;
  std::shared_ptr<op::BatchedAddLayer> batched_add_layer_;
  std::shared_ptr<op::BatchedSwiGLULayer> batched_swiglu_layer_;
  std::shared_ptr<op::SinCosCacheLayer> sin_cos_cache_layer_;
  std::shared_ptr<op::MHAGpuPosLayer> mha_gpu_pos_layer_;
  std::shared_ptr<op::BatchedMHALayer> batched_mha_layer_;
  std::shared_ptr<op::BatchedMatmulHelperLayer> batched_matmul_helper_layer_;
  
  // VL model specific layers for M-RoPE and vision
  std::shared_ptr<op::MRoPELayer> mrope_layer_;
  std::shared_ptr<op::MRoPEGpuPosLayer> mrope_gpu_pos_layer_;
  std::shared_ptr<op::BatchedMRoPELayer> batched_mrope_layer_;
  std::shared_ptr<op::FusedKVCacheUpdateLayer> fused_kv_cache_update_layer_;
  std::shared_ptr<op::RMSNormDimLayer> rmsnorm_dim_layer_;
  std::shared_ptr<op::CopyToKVCacheLayer> copy_to_kv_cache_layer_;
  std::shared_ptr<op::FlashAttentionDecodeGpuPosLayer> flash_attention_decode_gpu_pos_layer_;

  void to_cuda(std::shared_ptr<kernel::CudaConfig> config, bool keep_fp16_weights = true);
};

class Qwen3Model : public Model {
 public:
  explicit Qwen3Model(base::TokenizerType tokenizer_type, std::string token_path,
                      std::string model_path, bool is_quant_model);

  base::Status init(base::DeviceType device_type) override;

  base::Status predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                       bool is_prompt, int& next) const override;

  base::Status forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                       int& next) const override;

  op::EmbeddingOutput embedding(const std::vector<int>& tokens) const override;

  // Batched prefill: process all input tokens at once
  base::Status prefill(const tensor::Tensor& input, int32_t seq_len, int32_t start_pos) const;

  // Single token generation (decode phase)
  base::Status decode(const tensor::Tensor& input, int32_t pos, int& next) const;

  // CUDA Graph management for decode phase optimization
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
  
  // Clear KV cache (useful when starting a new conversation)
  void clear_kv_cache();
  
  // Get model configuration
  const TransformerConfig* get_config() const { return config_.get(); }

  std::shared_ptr<kernel::CudaConfig> get_cuda_config() const { return cuda_config_; }
  
  // Fused FFN optimization control (runtime switch)
  void enable_fused_ffn(bool enable) { use_fused_ffn_ = enable; }
  bool is_fused_ffn_enabled() const { return use_fused_ffn_; }

  // Override to propagate attention type to flash attention layers
  void set_attention_type(base::AttentionType type) override;

 private:
  void init_mem() override;

  base::Status create_layers() override;

  void create_param_layers() override;

  void create_param_layers_fp16();

  void create_param_layers_awq();

  void create_nonparam_layers() override;

  void create_param_quant_layers() override;

  void attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

  void attention_rms(int32_t layer_idx, const tensor::Tensor& input) const;

  void feed_forward(int32_t layer_idx, const tensor::Tensor& input) const;

  void attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;
  
  // CUDA Graph optimized version of attention_qkv
  void attention_qkv_with_graph(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;
  
  // CUDA Graph optimized version of attention_mha
  void attention_mha_with_graph(int32_t layer_idx, const tensor::Tensor& pos_tensor_gpu) const;
  
  // Fused FFN: combines W1+W3+SwiGLU into single kernel
  void feed_forward_fused(int32_t layer_idx, const tensor::Tensor& input) const;

  // Batched attention operations (for prefill phase)
  void batched_attention_rms(int32_t layer_idx, const tensor::Tensor& input, int32_t seq_len) const;
  // Optimized version with separate input/output buffers (no copy needed)
  void batched_attention_rms(int32_t layer_idx, const tensor::Tensor& input, 
                             const tensor::Tensor& output, int32_t seq_len) const;
  void batched_attention_qkv(int32_t layer_idx, const tensor::Tensor& rms_out,
                             const tensor::Tensor& query_out, const tensor::Tensor& key_out, 
                             const tensor::Tensor& value_out,
                             int32_t seq_len, int32_t start_pos) const;
  void batched_attention_mha(int32_t layer_idx, const tensor::Tensor& query,
                             const tensor::Tensor& mha_out, int32_t seq_len, int32_t start_pos) const;
  void batched_feed_forward(int32_t layer_idx, const tensor::Tensor& input, int32_t seq_len) const;
  // Optimized version with pre-allocated buffers (avoids per-layer allocation)
  void batched_feed_forward_optimized(int32_t layer_idx, const tensor::Tensor& input,
                                      tensor::Tensor& ffn_norm_out, tensor::Tensor& w1_out,
                                      tensor::Tensor& w3_out, tensor::Tensor& w2_out,
                                      int32_t seq_len) const;

  void cls_logits(const tensor::Tensor& input) const;

  int32_t post_processing(const tensor::Tensor& pos, bool is_prompt) const override;

 private:
  std::shared_ptr<kernel::CudaConfig> cuda_config_;
  std::unique_ptr<Qwen3Layers> qwen_layers_;
  
  // Fused FFN optimization (default enabled)
  bool use_fused_ffn_ = true;
};
}  // namespace model

#endif
