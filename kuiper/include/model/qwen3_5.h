#ifndef KUIPER_INCLUDE_MODEL_QWEN3_5_H_
#define KUIPER_INCLUDE_MODEL_QWEN3_5_H_

#include "qwen3_vl.h"
#include "op/gdn_layers.h"

namespace model {

/**
 * @brief Qwen3.5-9B Text Config (overrides for hybrid architecture)
 */
struct Qwen35TextConfig {
  int32_t hidden_size = 4096;
  int32_t intermediate_size = 12288;
  int32_t num_hidden_layers = 32;
  int32_t num_attention_heads = 16;
  int32_t num_key_value_heads = 4;
  int32_t vocab_size = 248320;
  int32_t max_position_embeddings = 262144;
  int32_t head_dim = 256;
  float rms_norm_eps = 1e-6f;
  float rope_theta = 10000000.0f;
  
  // Hybrid attention config
  int32_t full_attention_interval = 4;
  bool attn_output_gate = true;
  
  // Full attention layer indices
  std::vector<int32_t> full_attn_layer_indices;
  
  // Linear attention config
  int32_t linear_conv_kernel_dim = 4;
  int32_t linear_key_head_dim = 128;
  int32_t linear_num_key_heads = 16;
  int32_t linear_num_value_heads = 32;
  int32_t linear_value_head_dim = 128;
  float partial_rotary_factor = 0.25f;
  
  // M-RoPE config
  bool mrope_interleaved = true;
  std::vector<int32_t> mrope_section = {11, 11, 10};
  
  // Computed dimensions
  int32_t q_dim() const { return num_attention_heads * head_dim; }  // 4096
  int32_t kv_dim() const { return num_key_value_heads * head_dim; }  // 1024
  int32_t q_gate_dim() const { return 2 * q_dim(); }  // 8192 (includes output gate)
  int32_t conv_dim() const {
    return linear_num_key_heads * linear_key_head_dim * 2 + 
           linear_num_value_heads * linear_value_head_dim;
  }  // 8192
  int32_t partial_rope_dim() const { 
    return static_cast<int32_t>(head_dim * partial_rotary_factor); 
  }  // 64
  
  bool is_full_attn_layer(int layer_idx) const {
    for (auto idx : full_attn_layer_indices) {
      if (idx == layer_idx) return true;
    }
    return false;
  }
};

/**
 * @brief Linear attention layer weights (GDN - Gated Delta Net)
 */
struct LinearAttnWeights {
  // Per-layer weights
  struct Layer {
    std::shared_ptr<op::Layer> in_proj_qkv;   // [conv_dim, hidden_size] FP16
    std::shared_ptr<op::Layer> in_proj_z;     // [hidden_size, hidden_size] FP16
    std::shared_ptr<op::Layer> in_proj_a;     // [num_v_heads, hidden_size] FP16
    std::shared_ptr<op::Layer> in_proj_b;     // [num_v_heads, hidden_size] FP16
    std::shared_ptr<op::Layer> out_proj;      // [hidden_size, hidden_size] FP16
    
    tensor::Tensor A_log;     // [num_v_heads] FP32, on GPU
    tensor::Tensor dt_bias;   // [num_v_heads] FP16, on GPU
    tensor::Tensor conv_weight; // [conv_dim, kernel_size] FP16, on GPU
    tensor::Tensor norm_weight; // [hidden_size] FP32, on GPU
  };
  std::vector<Layer> layers;
};

/**
 * @brief GDN recurrent state (per layer, per sequence)
 */
struct GDNState {
  // Conv1D state: [conv_dim, kernel_size-1] FP16
  tensor::Tensor conv_state;
  // SSM state: [num_v_heads, value_head_dim, key_head_dim] FP32
  tensor::Tensor ssm_state;
};

/**
 * @brief Qwen3.5-9B Model (Hybrid Vision-Language Model)
 * 
 * Architecture: ViT + Hybrid LLM (Linear Attention + Full Attention)
 * - Inherits vision encoder from Qwen3VLModel
 * - Adds GDN (Gated Delta Net) linear attention for 24/32 layers
 * - Full attention with output gate for 8/32 layers
 */
class Qwen35Model : public Qwen3VLModel {
public:
  explicit Qwen35Model(base::TokenizerType tokenizer_type,
                       std::string token_path,
                       std::string model_path);
  
  ~Qwen35Model();
  
  base::Status init(base::DeviceType device_type) override;
  
  // Override core methods for hybrid architecture
  base::Status prefill(const tensor::Tensor& input_embeddings,
                       int32_t seq_len, int32_t start_pos) const;
  
  base::Status decode_step_optimized(int32_t pos, int& next) const;
  
  int sample_first_token() const;
  
  // Configuration access
  const Qwen35TextConfig& get_q35_config() const { return q35_config_; }
  
  // Clear all state (KV cache + GDN state)
  void clear_all_state();
  
private:
  // Override model loading
  base::Status load_q35_model_file();
  void init_q35_mem();
  void create_q35_nonparam_layers();
  
  // Hybrid LLM layer forward (decode - single token)
  void full_attn_decode(int32_t layer_idx, const tensor::Tensor& input) const;
  void full_attn_decode_graph(int32_t layer_idx, const tensor::Tensor& input,
                              const int32_t* rope_pos_gpu, const int32_t* kv_pos_gpu) const;
  void linear_attn_decode(int32_t layer_idx, const tensor::Tensor& input) const;
  void q35_feed_forward(int32_t layer_idx, const tensor::Tensor& input) const;
  void q35_cls_logits(const tensor::Tensor& input) const;
  
  // Hybrid LLM layer forward (prefill - batched)
  void full_attn_prefill(int32_t layer_idx, const tensor::Tensor& rms_out,
                         const tensor::Tensor& query_out, const tensor::Tensor& key_out,
                         const tensor::Tensor& value_out,
                         int32_t seq_len, int32_t start_pos) const;
  void linear_attn_prefill(int32_t layer_idx, const tensor::Tensor& rms_out,
                           half* work_buf, int32_t seq_len) const;
  
  // Map from actual layer index to type-specific index
  int full_attn_type_idx(int layer_idx) const;
  int linear_attn_type_idx(int layer_idx) const;
  
private:
  Qwen35TextConfig q35_config_;
  
  // Linear attention weights
  std::unique_ptr<LinearAttnWeights> linear_attn_weights_;
  
  // Full attention weights override (q_proj includes gate)
  // wq_layers_[type_idx] has shape [q_gate_dim, hidden_size] for full attn layers
  // wk/wv/wo remain indexed by type_idx for full attn layers only
  
  // GDN state for linear attention layers (one per linear layer)
  mutable std::vector<GDNState> gdn_states_;
  
  // Intermediate buffers for GDN
  mutable tensor::Tensor gdn_qkv_buf_;     // [conv_dim] FP16
  mutable tensor::Tensor gdn_z_buf_;       // [hidden_size] FP16
  mutable tensor::Tensor gdn_alpha_buf_;   // [num_v_heads] FP16
  mutable tensor::Tensor gdn_beta_buf_;    // [num_v_heads] FP16
  mutable tensor::Tensor gdn_gate_buf_;    // [num_v_heads] FP32
  mutable tensor::Tensor gdn_beta_fp32_;   // [num_v_heads] FP32
  mutable tensor::Tensor gdn_conv_out_;    // [conv_dim] FP16
  mutable tensor::Tensor gdn_q_norm_;      // [num_k_heads, key_head_dim] FP16
  mutable tensor::Tensor gdn_k_norm_;      // [num_k_heads, key_head_dim] FP16
  mutable tensor::Tensor gdn_attn_out_;    // [num_v_heads * value_head_dim] FP16
  mutable tensor::Tensor gdn_normed_out_;  // [hidden_size] FP16
  
  // Full attention Q/gate buffers (for Q/gate deinterleave in decode)
  mutable tensor::Tensor full_attn_q_;     // [q_dim] FP16
  mutable tensor::Tensor full_attn_gate_;  // [q_dim] FP16
  
  // MRoPE sections on CPU for kernel calls  
  int mrope_sections_cpu_[3];

  // GDN layer wrappers (decode + prefill)
  std::shared_ptr<op::DeinterleaveQGateLayer> deinterleave_q_gate_layer_;
  std::shared_ptr<op::PartialMRoPEInterleavedLayer> partial_mrope_layer_;
  std::shared_ptr<op::KVCacheWriteGpuPosLayer> kv_cache_write_gpu_pos_layer_;
  std::shared_ptr<op::ApplySigmoidGateLayer> apply_sigmoid_gate_layer_;
  std::shared_ptr<op::CausalConv1dSiluLayer> causal_conv1d_silu_layer_;
  std::shared_ptr<op::L2NormPerHeadLayer> l2_norm_per_head_layer_;
  std::shared_ptr<op::ComputeGDNGatesLayer> compute_gdn_gates_layer_;
  std::shared_ptr<op::GDNDecodeStepLayer> gdn_decode_step_layer_;
  std::shared_ptr<op::GatedRMSNormLayer> gated_rmsnorm_layer_;
  std::shared_ptr<op::BatchedAddFP16Layer> batched_add_fp16_layer_;
  std::shared_ptr<op::BatchedRMSNormFP16Layer> batched_rmsnorm_fp16_layer_;
  std::shared_ptr<op::GatherStridedLayer> gather_strided_layer_;
  std::shared_ptr<op::TransposeStateLayer> transpose_state_layer_;
  std::shared_ptr<op::GDNPrefillTransposedLayer> gdn_prefill_transposed_layer_;
  std::shared_ptr<op::FusedQKVGemvLayer> fused_qkv_gemv_layer_;
  std::shared_ptr<op::FusedGDNProjGemvLayer> fused_gdn_proj_gemv_layer_;
};

}  // namespace model

#endif  // KUIPER_INCLUDE_MODEL_QWEN3_5_H_
