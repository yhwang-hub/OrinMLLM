#ifdef QWEN3_SUPPORT
#include "model/qwen3_awq.h"
#include <glog/logging.h>
#include <op/matmul.h>
#include <op/rmsnorm.h>
#include <op/embedding.h>
#include <op/awq_matmul.h>

namespace model {

// ==================== Model file type detection ====================

bool is_awq_model_file(const std::string& model_path) {
  FILE* file = fopen(model_path.c_str(), "rb");
  if (!file) {
    return false;
  }

  uint32_t magic = 0;
  int32_t version = 0;
  bool is_awq = false;

  if (fread(&magic, sizeof(uint32_t), 1, file) == 1 &&
      fread(&version, sizeof(int32_t), 1, file) == 1) {
    // AWQ format: magic=0x616b3438 ("ak48"), version=5
    is_awq = (magic == 0x616b3438 && version == 5);
  }

  fclose(file);
  return is_awq;
}

// ==================== Qwen3AWQModel ====================

Qwen3AWQModel::Qwen3AWQModel(base::TokenizerType tokenizer_type, std::string token_path,
                               std::string model_path, bool is_quant_model)
    : Qwen3Model(tokenizer_type, std::move(token_path),
                 std::move(model_path), is_quant_model) {}

void Qwen3AWQModel::create_param_quant_layers() {
  // AWQ quantized layers are loaded in create_param_layers()
  // Nothing additional needed here
}

void Qwen3AWQModel::create_param_layers() {
  create_param_layers_awq();
}

void Qwen3AWQModel::create_param_layers_awq() {
  CHECK(qwen_layers_ != nullptr);
  LOG(INFO) << "Loading Qwen3 AWQ INT4 model weights...";

  // For AWQ, we need to use raw byte pointers since we mix INT32 and FP16 data
  // weight_data points to the start of weights (after 256-byte header)
  const uint8_t* base_ptr = static_cast<const uint8_t*>(raw_model_data_->weight_data);
  size_t pos = 0;  // position in bytes from base_ptr
  
  int32_t dim = config_->dim_;
  int32_t kv_dim = config_->kv_dim_;
  int32_t hidden_dim = config_->hidden_dim_;
  int32_t immediate_dim = config_->immediate_dim_;
  auto cpu_device_type = base::DeviceType::kDeviceCPU;

  // AWQ weight order (from export_qwen3-8B-awq.py):
  // == FP16 weights ==
  // 1. attention_norm (input_layernorm) for all layers - FP16, size: dim * 2 bytes
  // 2. ffn_norm (post_attention_layernorm) for all layers - FP16, size: dim * 2 bytes
  // 3. final norm - FP16, size: dim * 2 bytes
  // 4. token embeddings - FP16, size: vocab_size * dim * 2 bytes
  //
  // == AWQ quantized weights (for each layer) ==
  // Each linear layer has: qweight (INT32), qzeros (INT32), scales (FP16)
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

  // Helper function to load AWQ quantized linear layer
  auto load_awq_layer = [&](int32_t in_features, int32_t out_features, 
                            std::vector<std::shared_ptr<op::Layer>>& layer_list,
                            const std::string& name) {
    int32_t packed_out = out_features / 8;
    int32_t num_groups = in_features / group_size_;
    
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
      auto awq_layer = std::make_shared<op::AWQMatmulLayer>(
          device_type_, in_features, out_features, group_size_);
      
      // Read qweight, qzeros, scales in order using raw byte pointers
      const void* qweight_ptr = base_ptr + pos;
      size_t qweight_size = static_cast<size_t>(in_features) * packed_out * sizeof(int32_t);
      pos += qweight_size;
      
      const void* qzeros_ptr = base_ptr + pos;
      size_t qzeros_size = static_cast<size_t>(num_groups) * packed_out * sizeof(int32_t);
      pos += qzeros_size;
      
      const void* scales_ptr = base_ptr + pos;
      size_t scales_size = static_cast<size_t>(num_groups) * out_features * sizeof(uint16_t);
      pos += scales_size;
      
      awq_layer->set_awq_weights(qweight_ptr, qzeros_ptr, scales_ptr, cpu_device_type);
      layer_list.push_back(awq_layer);
    }
  };

  // 5. wq layers (q_proj) - AWQ
  load_awq_layer(dim, dim, qwen_layers_->wq_layers_, "wq");

  // 6. wk layers (k_proj) - AWQ
  load_awq_layer(dim, kv_dim, qwen_layers_->wk_layers_, "wk");

  // 7. wv layers (v_proj) - AWQ
  load_awq_layer(dim, kv_dim, qwen_layers_->wv_layers_, "wv");

  // 8. wo layers (o_proj) - AWQ
  load_awq_layer(dim, dim, qwen_layers_->wo_layers_, "wo");

  // 9. w1 layers (gate_proj) - AWQ
  load_awq_layer(dim, immediate_dim, qwen_layers_->w1_layers_, "w1");

  // 10. w2 layers (down_proj) - AWQ
  load_awq_layer(immediate_dim, dim, qwen_layers_->w2_layers_, "w2");

  // 11. w3 layers (up_proj) - AWQ
  load_awq_layer(dim, immediate_dim, qwen_layers_->w3_layers_, "w3");

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

  LOG(INFO) << "Qwen3 AWQ INT4 model loaded successfully. Total bytes: " << pos;
}

void Qwen3AWQModel::batched_qkv_projection(int32_t layer_idx, const tensor::Tensor& rms_out,
                                            const tensor::Tensor& query_out, const tensor::Tensor& key_out,
                                            const tensor::Tensor& value_out, int32_t seq_len) const {
  const auto& query_layer = qwen_layers_->wq_layers_.at(layer_idx);
  const auto& key_layer = qwen_layers_->wk_layers_.at(layer_idx);
  const auto& value_layer = qwen_layers_->wv_layers_.at(layer_idx);

  auto query_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(query_layer);
  auto key_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(key_layer);
  auto value_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(value_layer);

  CHECK_NE(query_awq, nullptr) << "Query layer is not an AWQMatmulLayer";
  CHECK_NE(key_awq, nullptr) << "Key layer is not an AWQMatmulLayer";
  CHECK_NE(value_awq, nullptr) << "Value layer is not an AWQMatmulLayer";

  STATUS_CHECK(query_awq->forward(rms_out, query_out));
  STATUS_CHECK(key_awq->forward(rms_out, key_out));
  STATUS_CHECK(value_awq->forward(rms_out, value_out));
}

void Qwen3AWQModel::batched_matmul_forward(const std::shared_ptr<op::Layer>& layer,
                                            const tensor::Tensor& input,
                                            const tensor::Tensor& output,
                                            int32_t seq_len) const {
  auto awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(layer);
  if (awq) {
    STATUS_CHECK(awq->forward(input, output));
  } else {
    // Fallback to base class (MatmulLayer path)
    Qwen3Model::batched_matmul_forward(layer, input, output, seq_len);
  }
}

void Qwen3AWQModel::gate_up_swiglu(int32_t layer_idx,
                                    const tensor::Tensor& input,
                                    const tensor::Tensor& output) const {
  auto* layers = get_base_layers();
  const auto& w1_layer = layers->w1_layers_.at(layer_idx);
  const auto& w3_layer = layers->w3_layers_.at(layer_idx);

  tensor::Tensor w3_output = get_buffer(ModelBufferType::kW3Output);
  STATUS_CHECK(w1_layer->forward(input, output));
  STATUS_CHECK(w3_layer->forward(input, w3_output));
  CHECK_NE(layers->swiglu_layer_, nullptr);
  STATUS_CHECK(layers->swiglu_layer_->forward(output, w3_output, output));
}

}  // namespace model

#endif
