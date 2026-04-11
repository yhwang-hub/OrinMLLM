#ifndef KUIPER_INCLUDE_MODEL_QWEN3_DFLASH_H_
#define KUIPER_INCLUDE_MODEL_QWEN3_DFLASH_H_
#ifdef QWEN3_SUPPORT
#include "qwen3.h"
#include <vector>

namespace model {

/**
 * @brief DFlash draft model configuration stored in the bin file header.
 */
struct DFlashConfig {
  int32_t block_size = 16;
  int32_t n_target_layers = 36;
  std::vector<int32_t> target_layer_ids;
  int32_t mask_token_id = 151669;
};

/**
 * @brief Utility: check if a file is a DFlash model by reading its magic number.
 */
bool is_dflash_model_file(const std::string& model_path);

/**
 * @brief DFlash draft model for speculative decoding.
 *
 * DFlash is a lightweight (5-layer) block-diffusion draft model that uses
 * cross-attention with target model hidden states. Key features:
 *   - fc layer fuses hidden states from 5 target layers into one vector
 *   - Cross-attention: Q from draft, K/V from context (target) + draft
 *   - Non-causal (bidirectional) attention
 *   - No embedding / lm_head (shared with target model)
 *   - Generates block_size tokens in parallel per step
 *
 * Inference flow:
 *   1. Target model prefill -> extract hidden_states from target_layer_ids
 *   2. DFlash draft model proposes block_size candidate tokens
 *   3. Target model verifies candidates -> accept consecutive matches
 *   4. Repeat from step 2
 */
class Qwen3DFlashModel : public Qwen3Model {
 public:
  explicit Qwen3DFlashModel(base::TokenizerType tokenizer_type, std::string token_path,
                            std::string model_path, bool is_quant_model);

  base::Status init(base::DeviceType device_type) override;

  /**
   * @brief Run the DFlash draft model forward pass.
   *
   * @param noise_embedding  Embedding of draft tokens [block_size, dim], on GPU
   * @param target_hidden    Fused target hidden states [context_len, dim], on GPU
   * @param context_len      Number of context tokens
   * @param block_size       Number of draft tokens (query tokens)
   * @param start_pos        Starting position for draft tokens
   * @return Draft model output hidden states [block_size, dim]
   */
  base::Status draft_forward(const tensor::Tensor& noise_embedding,
                             const tensor::Tensor& target_hidden,
                             int32_t context_len,
                             int32_t block_size,
                             int32_t start_pos);

  /**
   * @brief Extract and fuse target hidden states from specified layers.
   *
   * Takes the per-layer hidden states from the target model's prefill and
   * fuses them using fc + hidden_norm to produce the context features.
   *
   * @param per_layer_hidden  Vector of per-layer hidden states [seq_len, dim] each
   * @param seq_len           Sequence length
   * @return Fused hidden states tensor [seq_len, dim] on GPU
   */
  tensor::Tensor extract_and_fuse_context(
      const std::vector<tensor::Tensor>& per_layer_hidden,
      int32_t seq_len);

  const DFlashConfig& get_dflash_config() const { return dflash_config_; }

  /// Access draft model output (valid after draft_forward)
  tensor::Tensor& get_draft_output() { return draft_output_; }
  const tensor::Tensor& get_draft_output() const { return draft_output_; }

 protected:
  void init_mem() override;
  base::Status create_layers() override;
  void create_param_layers() override;
  void create_nonparam_layers() override;

 private:
  /// Read DFlash-specific header fields from the bin file
  base::Status read_dflash_header();

  /// Create the fc and hidden_norm layers
  void create_dflash_layers();

  DFlashConfig dflash_config_;

  // DFlash-specific layers
  std::shared_ptr<op::Layer> fc_layer_;           // Linear(n_target_layers * dim, dim)
  std::shared_ptr<op::Layer> hidden_norm_layer_;   // RMSNorm(dim)

  // Draft KV cache (separate from target model's KV cache)
  tensor::Tensor draft_key_cache_;
  tensor::Tensor draft_value_cache_;

  // Output buffer
  tensor::Tensor draft_output_;

  // Fused context buffer
  tensor::Tensor fused_context_;

  // Pre-allocated working buffers for draft_forward (Optimization 2)
  // Allocated once in init_mem(), reused across all draft_forward calls.
  tensor::Tensor draft_hidden_buf_;
  tensor::Tensor draft_rms_out_;
  tensor::Tensor draft_query_out_;
  tensor::Tensor draft_mha_out_;
  tensor::Tensor draft_wo_out_;
  tensor::Tensor draft_ffn_norm_out_;
  tensor::Tensor draft_w1_out_;
  tensor::Tensor draft_w3_out_;
  tensor::Tensor draft_w2_out_;
  tensor::Tensor draft_rope_dummy_k_;
  bool draft_buffers_allocated_ = false;
};

}  // namespace model

#endif  // QWEN3_SUPPORT
#endif  // KUIPER_INCLUDE_MODEL_QWEN3_DFLASH_H_
