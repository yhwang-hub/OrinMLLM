#ifndef KUIPER_INCLUDE_MODEL_LLAMA_H_
#define KUIPER_INCLUDE_MODEL_LLAMA_H_
#include "qwen_base.h"

namespace model {

struct Qwen2Layers : public QwenBaseLayers {
  // Qwen2-specific layers
  std::shared_ptr<op::BatchedMatmulLayer> batched_matmul_layer_;
  std::shared_ptr<op::BiasAddLayer> bias_add_layer_;

  void to_cuda(std::shared_ptr<kernel::CudaConfig> config, bool keep_fp16_weights = false);
};

class Qwen2Model : public QwenBaseModel {
 public:
  explicit Qwen2Model(base::TokenizerType tokenizer_type, std::string token_path,
                      std::string model_path, bool is_quant_model);

  base::Status init(base::DeviceType device_type) override;

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
  void create_nonparam_layers() override;
  void create_param_quant_layers() override;

  // FP16 model parameter layer creation
  void create_param_layers_fp16();

 private:
  std::unique_ptr<Qwen2Layers> qwen_layers_;
};
}  // namespace model

#endif
