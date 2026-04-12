#ifdef QWEN3_SUPPORT
#include "model/qwen3_fp8.h"
#include <glog/logging.h>
#include <cuda_fp16.h>
#include <op/matmul.h>
#include <op/rmsnorm.h>
#include <op/embedding.h>
#include <op/fp8_matmul.h>
#include <cmath>

namespace model {

// ==================== Model file type detection ====================

bool is_fp8_model_file(const std::string& model_path) {
  FILE* file = fopen(model_path.c_str(), "rb");
  if (!file) return false;

  uint32_t magic = 0;
  int32_t version = 0;
  bool is_fp8 = false;

  if (fread(&magic, sizeof(uint32_t), 1, file) == 1 &&
      fread(&version, sizeof(int32_t), 1, file) == 1) {
    is_fp8 = (magic == 0x66703838 && version == 7);
  }

  fclose(file);
  return is_fp8;
}

// ==================== Qwen3FP8Model ====================

Qwen3FP8Model::Qwen3FP8Model(base::TokenizerType tokenizer_type, std::string token_path,
                               std::string model_path, bool is_quant_model)
    : Qwen3Model(tokenizer_type, std::move(token_path),
                 std::move(model_path), is_quant_model) {}

void Qwen3FP8Model::create_param_quant_layers() {
  // FP8 quantized layers are loaded in create_param_layers()
}

void Qwen3FP8Model::create_param_layers() {
  create_param_layers_fp8();
}

void Qwen3FP8Model::create_param_layers_fp8() {
  CHECK(qwen_layers_ != nullptr);
  LOG(INFO) << "Loading Qwen3 FP8 E4M3 block-quantized model weights...";

  const uint8_t* base_ptr = static_cast<const uint8_t*>(raw_model_data_->weight_data);
  size_t pos = 0;

  int32_t dim = config_->dim_;  // hidden_size (2560 for 4B, 4096 for 8B)
  int32_t kv_dim = config_->kv_dim_;  // kv_head_num * head_dim
  int32_t attn_dim = config_->head_num_ * config_->head_size_;  // n_heads * head_dim
  int32_t hidden_dim = config_->hidden_dim_;
  int32_t immediate_dim = config_->immediate_dim_;
  auto cpu_device_type = base::DeviceType::kDeviceCPU;

  // Read block_size from header (offset 45 in header: magic(4)+version(4)+config(28+1+4) = 41 + 4 = 45)
  // Actually let's read it from the mmap'd file header
  {
    const uint8_t* file_start = static_cast<const uint8_t*>(raw_model_data_->data);
    // Header layout: magic(4) + version(4) + dim(4) + hidden_dim(4) + n_layers(4)
    //   + n_heads(4) + n_kv_heads(4) + vocab_size(4) + max_seq_len(4)
    //   + shared_classifier(1) + head_dim(4) + block_size(4)
    // Offset of block_size: 4+4+7*4+1+4 = 41 bytes
    int32_t bs;
    std::memcpy(&bs, file_start + 41, sizeof(int32_t));
    block_size_ = bs;
    LOG(INFO) << "FP8 block_size: " << block_size_;
  }

  // Weight layout (same as export_qwen3-8B-fp8.py):
  // == FP16 weights ==
  // 1. attention_norm for all layers
  // 2. ffn_norm for all layers
  // 3. final norm
  // 4. token embeddings
  //
  // == FP8 block-quantized weights ==
  // For each group (wq, wk, wv, wo, w1, w2, w3), for each layer:
  //   - weight [out, in] FP8 (1 byte)
  //   - scale_inv [scale_rows, scale_cols] FP16 (2 bytes)
  //
  // == FP16 weights ==
  // 12. lm_head (if not shared)
  // 13. q_norm for all layers
  // 14. k_norm for all layers

  // 1. attention_norm layers - FP16
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto rms_norm_layer = std::make_shared<op::RmsNormLayer>(device_type_, dim);
    rms_norm_layer->set_weight_fp16(0, {dim}, base_ptr + pos, cpu_device_type);
    qwen_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    pos += dim * sizeof(uint16_t);
  }

  // 2. ffn_norm layers - FP16
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
  size_t emb_pos = pos;  // Save embedding position for shared classifier
  pos += config_->vocab_size_ * dim * sizeof(uint16_t);

  // Helper to compute scale dimensions
  auto calc_scale_dim = [this](int32_t size) -> int32_t {
    return (size + block_size_ - 1) / block_size_;
  };

  // Helper to load FP8 block-quantized linear layer
  auto load_fp8_layer = [&](int32_t in_features, int32_t out_features,
                            std::vector<std::shared_ptr<op::Layer>>& layer_list,
                            const std::string& name) {
    int32_t scale_rows = calc_scale_dim(out_features);
    int32_t scale_cols = calc_scale_dim(in_features);

    for (int32_t i = 0; i < config_->layer_num_; ++i) {
      auto fp8_layer = std::make_shared<op::FP8MatmulLayer>(
          device_type_, in_features, out_features, block_size_);

      // FP8 weight [out_features, in_features] (1 byte each)
      const void* weight_ptr = base_ptr + pos;
      size_t weight_size = (size_t)out_features * in_features;
      pos += weight_size;

      // scale_inv [scale_rows, scale_cols] FP16 (2 bytes each)
      const void* scale_ptr = base_ptr + pos;
      size_t scale_size = (size_t)scale_rows * scale_cols * sizeof(uint16_t);
      pos += scale_size;

      fp8_layer->set_fp8_weights(weight_ptr, scale_ptr, scale_rows, scale_cols, cpu_device_type);
      layer_list.push_back(fp8_layer);
    }
  };

  // 5. wq (q_proj) for all layers - FP8
  // wq: q_proj [attn_dim, dim] - maps hidden_size to n_heads*head_dim
  load_fp8_layer(dim, attn_dim, qwen_layers_->wq_layers_, "wq");

  // 6. wk (k_proj) for all layers - FP8
  // wk: k_proj [kv_dim, dim] - maps hidden_size to kv_heads*head_dim
  load_fp8_layer(dim, kv_dim, qwen_layers_->wk_layers_, "wk");

  // 7. wv (v_proj) for all layers - FP8
  // wv: v_proj [kv_dim, dim] - maps hidden_size to kv_heads*head_dim
  load_fp8_layer(dim, kv_dim, qwen_layers_->wv_layers_, "wv");

  // 8. wo (o_proj) for all layers - FP8
  // wo: o_proj [dim, attn_dim] - maps n_heads*head_dim back to hidden_size
  load_fp8_layer(attn_dim, dim, qwen_layers_->wo_layers_, "wo");

  // 9. w1 (gate_proj) for all layers - FP8
  load_fp8_layer(dim, immediate_dim, qwen_layers_->w1_layers_, "w1");

  // 10. w2 (down_proj) for all layers - FP8
  load_fp8_layer(immediate_dim, dim, qwen_layers_->w2_layers_, "w2");

  // 11. w3 (up_proj) for all layers - FP8
  load_fp8_layer(dim, immediate_dim, qwen_layers_->w3_layers_, "w3");

  // 12. lm_head - FP16 (not quantized)
  if (!config_->is_shared_weight_) {
    auto lm_head = std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_, dim, false);
    lm_head->set_weight_fp16(0, {config_->vocab_size_, dim},
                             base_ptr + pos, cpu_device_type);
    qwen_layers_->cls_layer_ = lm_head;
    pos += config_->vocab_size_ * dim * sizeof(uint16_t);
  } else {
    auto lm_head = std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_, dim, false);
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

  LOG(INFO) << "Qwen3 FP8 E4M3 block-quantized model loaded. Total bytes: " << pos;
  int32_t fp8_attn_dim = config_->head_num_ * config_->head_size_;
  LOG(INFO) << "FP8 debug: dim_=" << config_->dim_ 
            << ", attn_dim=" << fp8_attn_dim
            << ", kv_dim_=" << config_->kv_dim_
            << ", head_size_=" << config_->head_size_
            << ", hidden_dim_=" << config_->hidden_dim_
            << ", immediate_dim_=" << config_->immediate_dim_;
}

void Qwen3FP8Model::batched_qkv_projection(int32_t layer_idx, const tensor::Tensor& rms_out,
                                            const tensor::Tensor& query_out, const tensor::Tensor& key_out,
                                            const tensor::Tensor& value_out, int32_t seq_len) const {
  const auto& query_layer = qwen_layers_->wq_layers_.at(layer_idx);
  const auto& key_layer = qwen_layers_->wk_layers_.at(layer_idx);
  const auto& value_layer = qwen_layers_->wv_layers_.at(layer_idx);

  auto query_fp8 = std::dynamic_pointer_cast<op::FP8MatmulLayer>(query_layer);
  auto key_fp8 = std::dynamic_pointer_cast<op::FP8MatmulLayer>(key_layer);
  auto value_fp8 = std::dynamic_pointer_cast<op::FP8MatmulLayer>(value_layer);

  CHECK_NE(query_fp8, nullptr) << "Query layer is not an FP8MatmulLayer";
  CHECK_NE(key_fp8, nullptr) << "Key layer is not an FP8MatmulLayer";
  CHECK_NE(value_fp8, nullptr) << "Value layer is not an FP8MatmulLayer";

  // FP8 GEMM/GEMV dispatch handles M=1 and M>1 internally
  STATUS_CHECK(query_fp8->forward(rms_out, query_out));
  STATUS_CHECK(key_fp8->forward(rms_out, key_out));
  STATUS_CHECK(value_fp8->forward(rms_out, value_out));
}

void Qwen3FP8Model::batched_matmul_forward(const std::shared_ptr<op::Layer>& layer,
                                            const tensor::Tensor& input,
                                            const tensor::Tensor& output,
                                            int32_t seq_len) const {
  auto fp8 = std::dynamic_pointer_cast<op::FP8MatmulLayer>(layer);
  if (fp8) {
    STATUS_CHECK(fp8->forward(input, output));
  } else {
    Qwen3Model::batched_matmul_forward(layer, input, output, seq_len);
  }
}

void Qwen3FP8Model::gate_up_swiglu(int32_t layer_idx,
                                   const tensor::Tensor& input,
                                   const tensor::Tensor& output) const {
  auto* layers = get_base_layers();
  const auto& w1_layer = layers->w1_layers_.at(layer_idx);
  const auto& w3_layer = layers->w3_layers_.at(layer_idx);

  // FP8 path: separate forward calls + SwiGLU
  // Cannot use fused FFN kernel (needs FP16 weight tensors)
  tensor::Tensor w3_output = get_buffer(ModelBufferType::kW3Output);
  STATUS_CHECK(w1_layer->forward(input, output));
  STATUS_CHECK(w3_layer->forward(input, w3_output));
  CHECK_NE(layers->swiglu_layer_, nullptr);
  STATUS_CHECK(layers->swiglu_layer_->forward(output, w3_output, output));
}

}  // namespace model

#endif
