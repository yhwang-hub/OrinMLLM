#ifndef KUIPER_INCLUDE_MODEL_QWEN3_H_
#define KUIPER_INCLUDE_MODEL_QWEN3_H_
#include "qwen_base.h"
#include "op/awq_matmul.h"
#include "op/flash_attention.h"
#include "op/kv_cache.h"
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

struct Qwen3Layers : public QwenBaseLayers {
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

class Qwen3Model : public QwenBaseModel {
 public:
  explicit Qwen3Model(base::TokenizerType tokenizer_type, std::string token_path,
                      std::string model_path, bool is_quant_model);

  base::Status init(base::DeviceType device_type) override;

  // Override set_attention_type to propagate to Qwen3-specific layers
  void set_attention_type(base::AttentionType type) override;

 protected:
  // === QwenBaseModel interface ===
  QwenBaseLayers* get_base_layers() const override { return qwen_layers_.get(); }

  void attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const override;
  void attention_qkv_with_graph(int32_t layer_idx, const tensor::Tensor& pos_tensor) const override;
  void batched_attention_qkv(int32_t layer_idx, const tensor::Tensor& rms_out,
                             const tensor::Tensor& query_out, const tensor::Tensor& key_out,
                             const tensor::Tensor& value_out,
                             int32_t seq_len, int32_t start_pos) const override;

 private:
  void init_mem() override;
  base::Status create_layers() override;
  void create_param_layers() override;
  void create_param_layers_fp16();
  void create_param_layers_awq();
  void create_nonparam_layers() override;
  void create_param_quant_layers() override;

 private:
  std::unique_ptr<Qwen3Layers> qwen_layers_;
};
}  // namespace model

#endif
