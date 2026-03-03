#ifndef KUIPER_INCLUDE_MODEL_QWEN3_AWQ_H_
#define KUIPER_INCLUDE_MODEL_QWEN3_AWQ_H_
#include "qwen3.h"

namespace model {

/**
 * @brief Qwen3 model with AWQ INT4 quantization.
 *
 * Inherits from Qwen3Model and overrides weight loading to use
 * AWQ INT4 quantized weights (qweight/qzeros/scales triplets).
 * All inference logic (attention, feed-forward, prefill, decode)
 * is reused from the parent class via polymorphic dispatch.
 */
class Qwen3AWQModel : public Qwen3Model {
 public:
  explicit Qwen3AWQModel(base::TokenizerType tokenizer_type, std::string token_path,
                         std::string model_path, bool is_quant_model);

 protected:
  void create_param_layers() override;
  void create_param_quant_layers() override;

  void batched_qkv_projection(int32_t layer_idx, const tensor::Tensor& rms_out,
                               const tensor::Tensor& query_out, const tensor::Tensor& key_out,
                               const tensor::Tensor& value_out, int32_t seq_len) const override;

  void batched_matmul_forward(const std::shared_ptr<op::Layer>& layer,
                               const tensor::Tensor& input,
                               const tensor::Tensor& output,
                               int32_t seq_len) const override;

  void gate_up_swiglu(int32_t layer_idx,
                       const tensor::Tensor& input,
                       const tensor::Tensor& output) const override;

 private:
  void create_param_layers_awq();
};

/**
 * @brief Detect whether a model file is AWQ format by reading its header.
 * @param model_path Path to the model binary file
 * @return true if the file uses AWQ INT4 format (magic=0x616b3438, version=5)
 */
bool is_awq_model_file(const std::string& model_path);

}  // namespace model

#endif  // KUIPER_INCLUDE_MODEL_QWEN3_AWQ_H_
