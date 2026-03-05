#ifdef QWEN3_SUPPORT
#include "model/qwen3_sq.h"
#include <glog/logging.h>
#include <cuda_fp16.h>
#include <op/matmul.h>
#include <op/rmsnorm.h>
#include <op/embedding.h>
#include <op/sq_matmul.h>

namespace model {

// ==================== Model file type detection ====================

bool is_sq_model_file(const std::string& model_path) {
  FILE* file = fopen(model_path.c_str(), "rb");
  if (!file) {
    return false;
  }

  uint32_t magic = 0;
  int32_t version = 0;
  bool is_sq = false;

  if (fread(&magic, sizeof(uint32_t), 1, file) == 1 &&
      fread(&version, sizeof(int32_t), 1, file) == 1) {
    // SQ format: magic=0x73713438 ("sq48"), version=6
    is_sq = (magic == 0x73713438 && version == 6);
  }

  fclose(file);
  return is_sq;
}

// ==================== Qwen3SQModel ====================

Qwen3SQModel::Qwen3SQModel(base::TokenizerType tokenizer_type, std::string token_path,
                             std::string model_path, bool is_quant_model)
    : Qwen3Model(tokenizer_type, std::move(token_path),
                 std::move(model_path), is_quant_model) {}

void Qwen3SQModel::create_param_quant_layers() {
  // SQ quantized layers are loaded in create_param_layers()
}

void Qwen3SQModel::create_param_layers() {
  create_param_layers_sq();
}

void Qwen3SQModel::create_param_layers_sq() {
  CHECK(qwen_layers_ != nullptr);
  LOG(INFO) << "Loading Qwen3 SmoothQuant INT8 model weights...";

  const uint8_t* base_ptr = static_cast<const uint8_t*>(raw_model_data_->weight_data);
  size_t pos = 0;
  
  int32_t dim = config_->dim_;
  int32_t kv_dim = config_->kv_dim_;
  int32_t hidden_dim = config_->hidden_dim_;
  int32_t immediate_dim = config_->immediate_dim_;
  auto cpu_device_type = base::DeviceType::kDeviceCPU;

  // SQ weight order (from export_qwen3-8B-sq.py):
  // == FP16 weights ==
  // 1. attention_norm (input_layernorm) for all layers - FP16
  // 2. ffn_norm (post_attention_layernorm) for all layers - FP16
  // 3. final norm - FP16
  // 4. token embeddings - FP16
  //
  // == SQ quantized weights (for each layer) ==
  // Each linear layer has: qweight (INT8), weight_scale (FP16 scalar), input_scale (FP32 scalar)
  // 5-11. wq, wk, wv, wo, w1, w2, w3 for all layers
  //
  // == FP16 weights ==
  // 12. lm_head - FP16 (if not shared)
  // 13. q_norm for all layers - FP16
  // 14. k_norm for all layers - FP16

  // 1. attention_norm layers (input_layernorm) - FP16
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto rms_norm_layer = std::make_shared<op::RmsNormLayer>(device_type_, dim);
    rms_norm_layer->set_weight_fp16(0, {dim}, base_ptr + pos, cpu_device_type);
    qwen_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    pos += dim * sizeof(uint16_t);
  }

  // 2. ffn_norm layers (post_attention_layernorm) - FP16
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto rms_norm_layer = std::make_shared<op::RmsNormLayer>(device_type_, dim);
    rms_norm_layer->set_weight_fp16(0, {dim}, base_ptr + pos, cpu_device_type);
    qwen_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    pos += dim * sizeof(uint16_t);
  }

  // 3. final norm - FP16
  {
    auto final_norm_layer = std::make_shared<op::RmsNormLayer>(device_type_, dim);
    final_norm_layer->set_weight_fp16(0, {dim}, base_ptr + pos, cpu_device_type);
    qwen_layers_->rmsnorm_layers_.push_back(final_norm_layer);
    pos += dim * sizeof(uint16_t);
  }

  // 4. token embeddings - FP16
  {
    auto emb_layer = std::make_shared<op::EmbeddingLayer>(
        device_type_, dim, config_->seq_len_, std::abs(config_->vocab_size_));
    emb_layer->set_weight_fp16(0, {std::abs(config_->vocab_size_), dim},
                               base_ptr + pos, cpu_device_type);
    qwen_layers_->embedding_layer_ = emb_layer;
  }
  pos += config_->vocab_size_ * dim * sizeof(uint16_t);

  // Helper function to load SQ quantized linear layer
  auto load_sq_layer = [&](int32_t in_features, int32_t out_features,
                           std::vector<std::shared_ptr<op::Layer>>& layer_list,
                           const std::string& name) {
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
      auto sq_layer = std::make_shared<op::SQMatmulLayer>(
          device_type_, in_features, out_features);
      
      // Read qweight [out_features, in_features] INT8
      const void* qweight_ptr = base_ptr + pos;
      size_t qweight_size = static_cast<size_t>(out_features) * in_features * sizeof(int8_t);
      pos += qweight_size;
      
      // Read weight_scale FP16 scalar (2 bytes)
      const void* weight_scale_ptr = base_ptr + pos;
      pos += sizeof(uint16_t);
      
      // Read input_scale FP32 scalar (4 bytes)
      const void* input_scale_ptr = base_ptr + pos;
      pos += sizeof(float);
      
      sq_layer->set_sq_weights(qweight_ptr, weight_scale_ptr, input_scale_ptr, cpu_device_type);
      layer_list.push_back(sq_layer);
    }
  };

  // 5. wq layers (q_proj) - SQ
  load_sq_layer(dim, dim, qwen_layers_->wq_layers_, "wq");

  // 6. wk layers (k_proj) - SQ
  load_sq_layer(dim, kv_dim, qwen_layers_->wk_layers_, "wk");

  // 7. wv layers (v_proj) - SQ
  load_sq_layer(dim, kv_dim, qwen_layers_->wv_layers_, "wv");

  // 8. wo layers (o_proj) - SQ
  load_sq_layer(dim, dim, qwen_layers_->wo_layers_, "wo");

  // 9. w1 layers (gate_proj) - SQ
  load_sq_layer(dim, immediate_dim, qwen_layers_->w1_layers_, "w1");

  // 10. w2 layers (down_proj) - SQ
  load_sq_layer(immediate_dim, dim, qwen_layers_->w2_layers_, "w2");

  // 11. w3 layers (up_proj) - SQ
  load_sq_layer(dim, immediate_dim, qwen_layers_->w3_layers_, "w3");

  // 12. output (lm_head) - FP16 (not quantized)
  if (!config_->is_shared_weight_) {
    auto lm_head = std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_, dim, false);
    lm_head->set_weight_fp16(0, {config_->vocab_size_, dim},
                             base_ptr + pos, cpu_device_type);
    qwen_layers_->cls_layer_ = lm_head;
    pos += config_->vocab_size_ * dim * sizeof(uint16_t);
  } else {
    // Share weights with embedding layer
    auto lm_head = std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_, dim, false);
    size_t emb_pos = (2 * config_->layer_num_ + 1) * dim * sizeof(uint16_t);
    lm_head->set_weight_fp16(0, {config_->vocab_size_, dim},
                             base_ptr + emb_pos, cpu_device_type);
    qwen_layers_->cls_layer_ = lm_head;
  }

  // 13. q_norm for all layers - FP16
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto rms_norm_layer = std::make_shared<op::RmsNormLayer>(device_type_, config_->head_size_);
    rms_norm_layer->set_weight_fp16(0, {config_->head_size_},
                                    base_ptr + pos, cpu_device_type);
    qwen_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    pos += config_->head_size_ * sizeof(uint16_t);
  }

  // 14. k_norm for all layers - FP16
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto rms_norm_layer = std::make_shared<op::RmsNormLayer>(device_type_, config_->head_size_);
    rms_norm_layer->set_weight_fp16(0, {config_->head_size_},
                                    base_ptr + pos, cpu_device_type);
    qwen_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    pos += config_->head_size_ * sizeof(uint16_t);
  }

  LOG(INFO) << "Qwen3 SmoothQuant INT8 model loaded successfully. Total bytes: " << pos;
}

void Qwen3SQModel::batched_qkv_projection(int32_t layer_idx, const tensor::Tensor& rms_out,
                                           const tensor::Tensor& query_out, const tensor::Tensor& key_out,
                                           const tensor::Tensor& value_out, int32_t seq_len) const {
  const auto& query_layer = qwen_layers_->wq_layers_.at(layer_idx);
  const auto& key_layer = qwen_layers_->wk_layers_.at(layer_idx);
  const auto& value_layer = qwen_layers_->wv_layers_.at(layer_idx);

  auto query_sq = std::dynamic_pointer_cast<op::SQMatmulLayer>(query_layer);
  auto key_sq = std::dynamic_pointer_cast<op::SQMatmulLayer>(key_layer);
  auto value_sq = std::dynamic_pointer_cast<op::SQMatmulLayer>(value_layer);

  CHECK_NE(query_sq, nullptr) << "Query layer is not an SQMatmulLayer";
  CHECK_NE(key_sq, nullptr) << "Key layer is not an SQMatmulLayer";
  CHECK_NE(value_sq, nullptr) << "Value layer is not an SQMatmulLayer";

  // For decode (M=1): shared quantization — quantize rms_out once, reuse for Q, K, V
  // Saves 6 kernel launches per layer (216 per decode step for 36 layers)
  int in_features = query_sq->in_features();
  int batch_size = rms_out.size() / in_features;

  if (batch_size == 1) {
    cudaStream_t stream = nullptr;
    if (cuda_config_) {
      stream = cuda_config_->stream;
    }
    // Quantize once: memset + absmax + quantize (3 kernels)
    op::SQMatmulLayer::quantize_input(rms_out, stream);
    // 3 GEMV calls with pre-quantized input (1 kernel each)
    STATUS_CHECK(op::SQMatmulLayer::forward_preq(query_out, *query_sq, stream));
    STATUS_CHECK(op::SQMatmulLayer::forward_preq(key_out, *key_sq, stream));
    STATUS_CHECK(op::SQMatmulLayer::forward_preq(value_out, *value_sq, stream));
    return;
  }

  // Fallback for M>1: separate SQ GEMM calls
  STATUS_CHECK(query_sq->forward(rms_out, query_out));
  STATUS_CHECK(key_sq->forward(rms_out, key_out));
  STATUS_CHECK(value_sq->forward(rms_out, value_out));
}

void Qwen3SQModel::batched_matmul_forward(const std::shared_ptr<op::Layer>& layer,
                                           const tensor::Tensor& input,
                                           const tensor::Tensor& output,
                                           int32_t seq_len) const {
  auto sq = std::dynamic_pointer_cast<op::SQMatmulLayer>(layer);
  if (sq) {
    STATUS_CHECK(sq->forward(input, output));
  } else {
    // Fallback to base class (MatmulLayer path)
    Qwen3Model::batched_matmul_forward(layer, input, output, seq_len);
  }
}

void Qwen3SQModel::gate_up_swiglu(int32_t layer_idx,
                                   const tensor::Tensor& input,
                                   const tensor::Tensor& output) const {
  auto* layers = get_base_layers();
  const auto& w1_layer = layers->w1_layers_.at(layer_idx);
  const auto& w3_layer = layers->w3_layers_.at(layer_idx);

  auto w1_sq = std::dynamic_pointer_cast<op::SQMatmulLayer>(w1_layer);
  auto w3_sq = std::dynamic_pointer_cast<op::SQMatmulLayer>(w3_layer);

  // Use fused FFN kernel for decode (M=1): saves 8 kernel launches
  if (w1_sq && w3_sq) {
    int in_features = w1_sq->in_features();
    int batch_size = input.size() / in_features;

    if (batch_size == 1) {
      // Fused path: quantize once + W1·x + W3·x + SwiGLU in 2 kernels
      cudaStream_t stream = nullptr;
      if (cuda_config_) {
        stream = cuda_config_->stream;
      }
      STATUS_CHECK(op::SQMatmulLayer::fused_ffn_forward(
          input, output, *w1_sq, *w3_sq, stream));
      return;
    }
  }

  // Fallback: separate SQ GEMM calls for W1, W3 + SwiGLU
  tensor::Tensor w3_output = get_buffer(ModelBufferType::kW3Output);
  STATUS_CHECK(w1_layer->forward(input, output));
  STATUS_CHECK(w3_layer->forward(input, w3_output));
  CHECK_NE(layers->swiglu_layer_, nullptr);
  STATUS_CHECK(layers->swiglu_layer_->forward(output, w3_output, output));
}

}  // namespace model

#endif
