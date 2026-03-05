#ifndef KUIPER_INCLUDE_MODEL_QWEN3_SQ_H_
#define KUIPER_INCLUDE_MODEL_QWEN3_SQ_H_
#include "qwen3.h"

namespace model {

/**
 * @brief Qwen3 model with SmoothQuant INT8 per-tensor quantization.
 *
 * Inherits from Qwen3Model and overrides weight loading to use
 * SmoothQuant INT8 quantized weights (qweight/weight_scale/input_scale).
 * All inference logic (attention, feed-forward, prefill, decode)
 * is reused from the parent class via polymorphic dispatch.
 */
class Qwen3SQModel : public Qwen3Model {
 public:
  explicit Qwen3SQModel(base::TokenizerType tokenizer_type, std::string token_path,
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
  void create_param_layers_sq();
};

/**
 * @brief Detect whether a model file is SmoothQuant format by reading its header.
 * @param model_path Path to the model binary file
 * @return true if the file uses SQ INT8 format (magic=0x73713438, version=6)
 */
bool is_sq_model_file(const std::string& model_path);

}  // namespace model

#endif  // KUIPER_INCLUDE_MODEL_QWEN3_SQ_H_
