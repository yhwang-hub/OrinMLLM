#ifndef KUIPER_INCLUDE_MODEL_QWEN3_FP8_H_
#define KUIPER_INCLUDE_MODEL_QWEN3_FP8_H_
#include "qwen3.h"

namespace model {

/**
 * @brief Qwen3 model with FP8 E4M3 block-quantized weights.
 *
 * Inherits from Qwen3Model and overrides weight loading to use
 * FP8 block-quantized weights (weight + weight_scale_inv per block).
 * FP8 provides ~2x memory bandwidth savings for decode (GEMV)
 * while maintaining accuracy via per-block scaling.
 */
class Qwen3FP8Model : public Qwen3Model {
 public:
  explicit Qwen3FP8Model(base::TokenizerType tokenizer_type, std::string token_path,
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
  void create_param_layers_fp8();
  int32_t block_size_ = 128;   // FP8 block quantization size
};

/**
 * @brief Detect whether a model file is FP8 format by reading its header.
 * @param model_path Path to the model binary file
 * @return true if the file uses FP8 format (magic=0x66703838, version=7)
 */
bool is_fp8_model_file(const std::string& model_path);

}  // namespace model

#endif  // KUIPER_INCLUDE_MODEL_QWEN3_FP8_H_
