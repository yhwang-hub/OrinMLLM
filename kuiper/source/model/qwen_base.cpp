#include "model/qwen_base.h"
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <op/matmul.h>
#include <op/mha.h>
#include <op/rmsnorm.h>
#include <op/embedding.h>
#include <op/awq_matmul.h>
#include <utility>
#include "sampler/argmax_sampler.h"
#include "base/tick.h"

namespace model {

QwenBaseModel::QwenBaseModel(base::TokenizerType tokenizer_type, std::string token_path,
                             std::string model_path, bool is_quant_model)
    : Model(tokenizer_type, base::ModelType::kModelTypeLLama2, std::move(token_path),
            std::move(model_path), is_quant_model) {}

// ==================== Forward / Predict ====================

base::Status QwenBaseModel::forward(const tensor::Tensor& input,
                                    const tensor::Tensor& pos_tensor, int& next) const {
  for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
    attention_rms(layer_idx, input);
    attention_qkv(layer_idx, pos_tensor);
    attention_mha(layer_idx, pos_tensor);
    if (use_fused_ffn_) {
      feed_forward_fused(layer_idx, input);
    } else {
      feed_forward(layer_idx, input);
    }
  }
  cls_logits(input);
  return base::error::Success();
}

base::Status QwenBaseModel::predict(const tensor::Tensor& input,
                                    const tensor::Tensor& pos_tensor,
                                    bool is_prompt, int& next) const {
  auto status = forward(input, pos_tensor, next);
  if (!status) {
    return status;
  }
  next = post_processing(pos_tensor, is_prompt);
  return base::error::Success();
}

// ==================== Embedding ====================

op::EmbeddingOutput QwenBaseModel::embedding(const std::vector<int>& tokens) const {
  auto* layers = get_base_layers();
  auto input_tokens = get_buffer(ModelBufferType::kInputTokens);
  auto input_embeddings = get_buffer(ModelBufferType::kInputEmbeddings);
  if (input_tokens.size() != tokens.size()) {
    input_tokens.reshape({static_cast<int32_t>(tokens.size())});
    input_embeddings.reshape({static_cast<int32_t>(tokens.size()), config_->dim_});
  }
  for (int32_t i = 0; i < tokens.size(); ++i) {
    input_tokens.index<int32_t>(i) = tokens.at(i);
  }

  auto input_token_num =
      tensor::Tensor(base::DataType::kDataTypeInt32, static_cast<int32_t>(tokens.size()));
  LOG_IF(FATAL, !layers->embedding_layer_)
      << "The embedding layer in the model is null pointer.";
  STATUS_CHECK(
      layers->embedding_layer_->forward(input_tokens, input_token_num, input_embeddings));

  op::EmbeddingOutput output(input_tokens, input_embeddings, input_token_num);
  return output;
}

// ==================== Attention RMS ====================

void QwenBaseModel::attention_rms(int32_t layer_idx, const tensor::Tensor& input) const {
  auto* layers = get_base_layers();
  CHECK(layers != nullptr);
  tensor::Tensor rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  std::shared_ptr<op::Layer> rmsnorm_layer = layers->rmsnorm_layers_.at(layer_idx);
  if (!rmsnorm_layer) {
    LOG(FATAL) << "The attention rmsnorm layer is a null pointer";
  }
  STATUS_CHECK(rmsnorm_layer->forward(input, rmsnorm_output));
}

// ==================== Attention MHA ====================

void QwenBaseModel::attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  auto* layers = get_base_layers();
  CHECK(layers != nullptr);

  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);
  tensor::Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);

  int pos = pos_tensor.index<int32_t>(0);

  // FP16 data always uses Flash Attention (MHA does not support FP16)
  if (query.data_type() == base::DataType::kDataTypeFp16 &&
      key_cache.data_type() == base::DataType::kDataTypeFp16) {
    auto flash_attn = layers->flash_attention_decode_layer_;
    flash_attn->set_layer_index(layer_idx);
    flash_attn->set_pos(pos);
    flash_attn->set_use_gpu_pos(false);
    flash_attn->set_input(0, query);
    flash_attn->set_input(1, mha_output);
    flash_attn->set_input(2, key_cache);
    flash_attn->set_input(3, val_cache);
    flash_attn->set_cuda_config(cuda_config_);
    STATUS_CHECK(flash_attn->forward());
  } else if (attention_type_ == base::AttentionType::kAttentionMHA) {
    // Standard FP32 MHA path
    tensor::Tensor score_storage = get_buffer(ModelBufferType::kScoreStorage);
    const auto& mha_layer = layers->mha_layer_;
    CHECK_NE(mha_layer, nullptr) << "The multi head attention layer is null pointer.";
    std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_pos(pos);
    std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_layer_idx(layer_idx);
    STATUS_CHECK(mha_layer->forward(query, score_storage, key_cache, val_cache, mha_output));
  } else {
    // FP32 Flash Attention path (FA1 or FA2)
    auto flash_attn = layers->flash_attention_decode_layer_;
    flash_attn->set_layer_index(layer_idx);
    flash_attn->set_pos(pos);
    flash_attn->set_use_gpu_pos(false);
    flash_attn->set_input(0, query);
    flash_attn->set_input(1, mha_output);
    flash_attn->set_input(2, key_cache);
    flash_attn->set_input(3, val_cache);
    flash_attn->set_cuda_config(cuda_config_);
    STATUS_CHECK(flash_attn->forward());
  }

  // wo @ attention output
  tensor::Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
  const auto& wo_layer = layers->wo_layers_.at(layer_idx);
  CHECK_NE(wo_layer, nullptr) << "The weight output layer is null pointer.";
  STATUS_CHECK(wo_layer->forward(mha_output, attn_output));
}

// ==================== Attention MHA with CUDA Graph ====================

void QwenBaseModel::attention_mha_with_graph(int32_t layer_idx,
                                             const tensor::Tensor& pos_tensor_gpu) const {
  auto* layers = get_base_layers();
  CHECK(layers != nullptr);
  CHECK(cuda_config_ != nullptr);

  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);
  tensor::Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);

  // FP16 data always uses Flash Attention
  if (query.data_type() == base::DataType::kDataTypeFp16 &&
      key_cache.data_type() == base::DataType::kDataTypeFp16) {
    auto flash_attn = layers->flash_attention_decode_layer_;
    flash_attn->set_layer_index(layer_idx);
    flash_attn->set_use_gpu_pos(true);
    flash_attn->set_input(0, query);
    flash_attn->set_input(1, mha_output);
    flash_attn->set_input(2, key_cache);
    flash_attn->set_input(3, val_cache);
    flash_attn->set_input(4, pos_tensor_gpu);
    flash_attn->set_cuda_config(cuda_config_);
    STATUS_CHECK(flash_attn->forward());
  } else if (attention_type_ == base::AttentionType::kAttentionMHA) {
    // Standard FP32 MHA with GPU pos
    tensor::Tensor score_storage = get_buffer(ModelBufferType::kScoreStorage);
    STATUS_CHECK(layers->mha_gpu_pos_layer_->forward(
        pos_tensor_gpu.ptr<int32_t>(),
        config_->head_num_, layer_idx, config_->seq_len_,
        config_->kv_dim_, config_->kv_mul_, config_->head_size_,
        mha_output, query, score_storage, key_cache, val_cache));
  } else {
    // FP32 + FA: fall back to MHA with GPU pos
    tensor::Tensor score_storage = get_buffer(ModelBufferType::kScoreStorage);
    STATUS_CHECK(layers->mha_gpu_pos_layer_->forward(
        pos_tensor_gpu.ptr<int32_t>(),
        config_->head_num_, layer_idx, config_->seq_len_,
        config_->kv_dim_, config_->kv_mul_, config_->head_size_,
        mha_output, query, score_storage, key_cache, val_cache));
  }

  // wo @ attention output
  tensor::Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
  const auto& wo_layer = layers->wo_layers_.at(layer_idx);
  CHECK_NE(wo_layer, nullptr);
  STATUS_CHECK(wo_layer->forward(mha_output, attn_output));
}

// ==================== Feed Forward ====================

void QwenBaseModel::feed_forward(int32_t layer_idx, const tensor::Tensor& input) const {
  auto* layers = get_base_layers();
  CHECK(layers != nullptr);

  // residual add
  CHECK_NE(layers->add_layer_, nullptr)
      << "The add layer in the feedforward block is null pointer";
  STATUS_CHECK(
      layers->add_layer_->forward(input, get_buffer(ModelBufferType::kAttnOutput), input));

  // ffn rmsnorm
  tensor::Tensor ffn_norm_output = get_buffer(ModelBufferType::kFFNRMSNorm);
  const auto& ffn_rmsnorm = layers->rmsnorm_layers_.at(layer_idx + config_->layer_num_);
  CHECK_NE(ffn_rmsnorm, nullptr)
      << "The final rmsnorm layer in the feedforward block is null pointer";
  STATUS_CHECK(ffn_rmsnorm->forward(input, ffn_norm_output));

  // w1
  tensor::Tensor w1_output = get_buffer(ModelBufferType::kW1Output);
  const auto& w1_layer = layers->w1_layers_.at(layer_idx);
  CHECK_NE(w1_layer, nullptr) << "The w1 layer in the feedforward block is null pointer";
  STATUS_CHECK(w1_layer->forward(ffn_norm_output, w1_output));

  // w3
  tensor::Tensor w3_ouput = get_buffer(ModelBufferType::kW3Output);
  const auto& w3_layer = layers->w3_layers_.at(layer_idx);
  CHECK_NE(w3_layer, nullptr) << "The w3 layer in the feedforward block is null pointer";
  STATUS_CHECK(w3_layer->forward(ffn_norm_output, w3_ouput));

  // SwiGLU
  CHECK_NE(layers->swiglu_layer_, nullptr)
      << "The swiglu layer in the feedforward block is null pointer";
  STATUS_CHECK(layers->swiglu_layer_->forward(w1_output, w3_ouput, w1_output));

  // w2
  tensor::Tensor w2_output = get_buffer(ModelBufferType::kW2Output);
  const auto& w2_layer = layers->w2_layers_.at(layer_idx);
  CHECK_NE(w2_layer, nullptr) << "The w2 layer in the feedforward block is null pointer";
  STATUS_CHECK(w2_layer->forward(w1_output, w2_output));

  // residual add
  STATUS_CHECK(layers->add_layer_->forward(input, w2_output, input));
}

// ==================== Fused Feed Forward ====================

void QwenBaseModel::feed_forward_fused(int32_t layer_idx, const tensor::Tensor& input) const {
  auto* layers = get_base_layers();
  CHECK(layers != nullptr);

  // residual add
  CHECK_NE(layers->add_layer_, nullptr);
  STATUS_CHECK(
      layers->add_layer_->forward(input, get_buffer(ModelBufferType::kAttnOutput), input));

  // ffn rmsnorm
  tensor::Tensor ffn_norm_output = get_buffer(ModelBufferType::kFFNRMSNorm);
  const auto& ffn_rmsnorm = layers->rmsnorm_layers_.at(layer_idx + config_->layer_num_);
  CHECK_NE(ffn_rmsnorm, nullptr);
  STATUS_CHECK(ffn_rmsnorm->forward(input, ffn_norm_output));

  // Fused W1 + W3 + SwiGLU kernel
  tensor::Tensor w1_output = get_buffer(ModelBufferType::kW1Output);
  const auto& w1_layer = layers->w1_layers_.at(layer_idx);
  const auto& w3_layer = layers->w3_layers_.at(layer_idx);
  CHECK_NE(w1_layer, nullptr);
  CHECK_NE(w3_layer, nullptr);

  // Check if AWQ layers (AWQ doesn't support fused kernel, fall back to standard)
  auto w1_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(w1_layer);
  auto w3_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(w3_layer);

  if (w1_awq || w3_awq) {
    // AWQ fallback: standard separate operations
    tensor::Tensor w3_output = get_buffer(ModelBufferType::kW3Output);
    STATUS_CHECK(w1_layer->forward(ffn_norm_output, w1_output));
    STATUS_CHECK(w3_layer->forward(ffn_norm_output, w3_output));
    CHECK_NE(layers->swiglu_layer_, nullptr);
    STATUS_CHECK(layers->swiglu_layer_->forward(w1_output, w3_output, w1_output));
  } else {
    // Standard path with fused FFN kernel
    auto w1_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(w1_layer);
    auto w3_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(w3_layer);
    CHECK_NE(w1_matmul, nullptr) << "W1 layer is not a MatmulLayer";
    CHECK_NE(w3_matmul, nullptr) << "W3 layer is not a MatmulLayer";

    const auto& w1_weight = w1_matmul->get_weight(0);
    const auto& w3_weight = w3_matmul->get_weight(0);

    auto fused_ffn = layers->fused_ffn_layer_;
    bool is_fp16 = input.data_type() == base::DataType::kDataTypeFp16 &&
                   w1_weight.data_type() == base::DataType::kDataTypeFp16;
    bool is_mixed = input.data_type() == base::DataType::kDataTypeFp32 &&
                    w1_weight.data_type() == base::DataType::kDataTypeFp16;
    fused_ffn->set_use_fp16(is_fp16);
    fused_ffn->set_use_mixed(is_mixed);
    fused_ffn->set_input(0, ffn_norm_output);
    fused_ffn->set_input(1, w1_weight);
    fused_ffn->set_input(2, w3_weight);
    fused_ffn->set_output(0, w1_output);
    fused_ffn->set_cuda_config(cuda_config_);
    STATUS_CHECK(fused_ffn->forward());
  }

  // w2 (down projection)
  tensor::Tensor w2_output = get_buffer(ModelBufferType::kW2Output);
  const auto& w2_layer = layers->w2_layers_.at(layer_idx);
  CHECK_NE(w2_layer, nullptr);
  STATUS_CHECK(w2_layer->forward(w1_output, w2_output));

  // residual add
  STATUS_CHECK(layers->add_layer_->forward(input, w2_output, input));
}

// ==================== Cls Logits ====================

void QwenBaseModel::cls_logits(const tensor::Tensor& input) const {
  auto* layers = get_base_layers();
  CHECK(layers != nullptr);
  const auto& norm = layers->rmsnorm_layers_.at(2 * config_->layer_num_);
  CHECK_NE(norm, nullptr);
  STATUS_CHECK(norm->forward(input, input));

  tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
  CHECK_NE(layers->cls_layer_, nullptr);
  STATUS_CHECK(layers->cls_layer_->forward(input, forward_output));
}

// ==================== Post Processing ====================

int32_t QwenBaseModel::post_processing(const tensor::Tensor& pos, bool is_prompt) const {
  tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
  const float* forward_logits = forward_output.ptr<float>();

  int32_t next = 0;
  if (is_prompt) {
    next = -1;
  } else {
    next = static_cast<int32_t>(sampler_->sample(forward_logits, forward_output.size(),
                                                 cuda_config_ ? cuda_config_->stream : nullptr));
  }
  return next;
}

// ==================== Batched Attention RMSNorm ====================

void QwenBaseModel::batched_attention_rms(int32_t layer_idx, const tensor::Tensor& input,
                                          int32_t seq_len) const {
  auto* layers = get_base_layers();
  CHECK(layers != nullptr);
  std::shared_ptr<op::Layer> rmsnorm_layer = layers->rmsnorm_layers_.at(layer_idx);
  if (!rmsnorm_layer) {
    LOG(FATAL) << "The attention rmsnorm layer is a null pointer";
  }
  STATUS_CHECK(rmsnorm_layer->forward(input, input));
}

void QwenBaseModel::batched_attention_rms(int32_t layer_idx, const tensor::Tensor& input,
                                          const tensor::Tensor& output, int32_t seq_len) const {
  auto* layers = get_base_layers();
  CHECK(layers != nullptr);
  std::shared_ptr<op::Layer> rmsnorm_layer = layers->rmsnorm_layers_.at(layer_idx);
  if (!rmsnorm_layer) {
    LOG(FATAL) << "The attention rmsnorm layer is a null pointer";
  }
  STATUS_CHECK(rmsnorm_layer->forward(input, output));
}

// ==================== Batched Attention MHA ====================

void QwenBaseModel::batched_attention_mha(int32_t layer_idx, const tensor::Tensor& query,
                                          const tensor::Tensor& mha_out,
                                          int32_t seq_len, int32_t start_pos) const {
  auto* layers = get_base_layers();
  CHECK(layers != nullptr);

  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);

  if (attention_type_ == base::AttentionType::kAttentionMHA &&
      query.data_type() != base::DataType::kDataTypeFp16) {
    // Standard batched MHA attention (FP32 only)
    std::shared_ptr<base::DeviceAllocator> score_alloc = base::CUDADeviceAllocatorFactory::get_instance();
    tensor::Tensor attn_scores(base::DataType::kDataTypeFp32,
                               seq_len, config_->head_num_, config_->seq_len_, true, score_alloc);
    STATUS_CHECK(layers->batched_mha_layer_->forward(
        start_pos, seq_len, config_->head_num_, layer_idx,
        config_->seq_len_, config_->dim_, config_->kv_dim_,
        config_->kv_mul_, config_->head_size_,
        const_cast<tensor::Tensor&>(mha_out), query, attn_scores, key_cache, val_cache));
  } else {
    // Flash Attention prefill (FA1 or FA2)
    auto prefill_layer = layers->flash_attention_prefill_layer_;
    prefill_layer->set_cur_seq_len(seq_len);
    prefill_layer->set_start_pos(start_pos);
    prefill_layer->set_layer_index(layer_idx);
    prefill_layer->set_use_fp16(query.data_type() == base::DataType::kDataTypeFp16 &&
                                key_cache.data_type() == base::DataType::kDataTypeFp16);
    prefill_layer->set_input(0, query);
    prefill_layer->set_input(1, mha_out);
    prefill_layer->set_input(2, key_cache);
    prefill_layer->set_input(3, val_cache);
    prefill_layer->set_cuda_config(cuda_config_);
    STATUS_CHECK(prefill_layer->forward());
  }

  // Batched wo projection
  base::DataType activation_dtype = mha_out.data_type();
  size_t elem_size = (activation_dtype == base::DataType::kDataTypeFp16)
      ? sizeof(uint16_t) : sizeof(float);

  const auto& wo_layer = layers->wo_layers_.at(layer_idx);
  std::shared_ptr<base::DeviceAllocator> alloc = base::CUDADeviceAllocatorFactory::get_instance();
  tensor::Tensor wo_out(activation_dtype, seq_len, config_->dim_, true, alloc);

  // Check if AWQ layer
  auto wo_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(wo_layer);
  if (wo_awq) {
    STATUS_CHECK(wo_awq->forward(mha_out, wo_out));
  } else {
    auto wo_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(wo_layer);
    CHECK_NE(wo_matmul, nullptr) << "WO layer is not a MatmulLayer";
    STATUS_CHECK(layers->batched_matmul_helper_layer_->forward(
        mha_out, wo_matmul->get_weight(0), wo_out, seq_len, 1.f));
  }

  // Copy back to mha_out
  cudaMemcpyAsync(const_cast<void*>(mha_out.get_buffer()->ptr()),
                  wo_out.get_buffer()->ptr(),
                  seq_len * config_->dim_ * elem_size,
                  cudaMemcpyDeviceToDevice, cuda_config_->stream);
}

// Optimized version with pre-allocated wo_out buffer
void QwenBaseModel::batched_attention_mha(int32_t layer_idx, const tensor::Tensor& query,
                                          const tensor::Tensor& mha_out, tensor::Tensor& wo_out,
                                          int32_t seq_len, int32_t start_pos) const {
  auto* layers = get_base_layers();
  CHECK(layers != nullptr);

  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);

  if (attention_type_ == base::AttentionType::kAttentionMHA &&
      query.data_type() != base::DataType::kDataTypeFp16) {
    std::shared_ptr<base::DeviceAllocator> score_alloc = base::CUDADeviceAllocatorFactory::get_instance();
    tensor::Tensor attn_scores(base::DataType::kDataTypeFp32,
                               seq_len, config_->head_num_, config_->seq_len_, true, score_alloc);
    STATUS_CHECK(layers->batched_mha_layer_->forward(
        start_pos, seq_len, config_->head_num_, layer_idx,
        config_->seq_len_, config_->dim_, config_->kv_dim_,
        config_->kv_mul_, config_->head_size_,
        const_cast<tensor::Tensor&>(mha_out), query, attn_scores, key_cache, val_cache));
  } else {
    auto prefill_layer = layers->flash_attention_prefill_layer_;
    prefill_layer->set_cur_seq_len(seq_len);
    prefill_layer->set_start_pos(start_pos);
    prefill_layer->set_layer_index(layer_idx);
    prefill_layer->set_use_fp16(query.data_type() == base::DataType::kDataTypeFp16 &&
                                key_cache.data_type() == base::DataType::kDataTypeFp16);
    prefill_layer->set_input(0, query);
    prefill_layer->set_input(1, mha_out);
    prefill_layer->set_input(2, key_cache);
    prefill_layer->set_input(3, val_cache);
    prefill_layer->set_cuda_config(cuda_config_);
    STATUS_CHECK(prefill_layer->forward());
  }

  // wo projection directly to pre-allocated wo_out
  const auto& wo_layer = layers->wo_layers_.at(layer_idx);
  auto wo_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(wo_layer);
  if (wo_awq) {
    STATUS_CHECK(wo_awq->forward(mha_out, wo_out));
  } else {
    auto wo_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(wo_layer);
    CHECK_NE(wo_matmul, nullptr) << "WO layer is not a MatmulLayer";
    STATUS_CHECK(layers->batched_matmul_helper_layer_->forward(
        mha_out, wo_matmul->get_weight(0), wo_out, seq_len, 1.f));
  }
}

// ==================== Batched Feed Forward ====================

void QwenBaseModel::batched_feed_forward(int32_t layer_idx, const tensor::Tensor& input,
                                         int32_t seq_len) const {
  auto* layers = get_base_layers();
  CHECK(layers != nullptr);

  base::DataType activation_dtype = input.data_type();
  std::shared_ptr<base::DeviceAllocator> alloc = base::CUDADeviceAllocatorFactory::get_instance();

  // FFN RMSNorm
  const auto& ffn_rmsnorm = layers->rmsnorm_layers_.at(layer_idx + config_->layer_num_);
  tensor::Tensor ffn_norm_out(activation_dtype, seq_len, config_->dim_, true, alloc);
  STATUS_CHECK(ffn_rmsnorm->forward(input, ffn_norm_out));

  // Batched W1 and W3
  const auto& w1_layer = layers->w1_layers_.at(layer_idx);
  const auto& w3_layer = layers->w3_layers_.at(layer_idx);

  int32_t hidden_dim = config_->hidden_dim_;
  tensor::Tensor w1_out(activation_dtype, seq_len, hidden_dim, true, alloc);
  tensor::Tensor w3_out(activation_dtype, seq_len, hidden_dim, true, alloc);

  // Check if AWQ layers
  auto w1_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(w1_layer);
  auto w3_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(w3_layer);

  if (w1_awq && w3_awq) {
    STATUS_CHECK(w1_awq->forward(ffn_norm_out, w1_out));
    STATUS_CHECK(w3_awq->forward(ffn_norm_out, w3_out));
  } else {
    auto w1_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(w1_layer);
    auto w3_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(w3_layer);
    CHECK_NE(w1_matmul, nullptr) << "W1 layer is not a MatmulLayer";
    CHECK_NE(w3_matmul, nullptr) << "W3 layer is not a MatmulLayer";
    STATUS_CHECK(layers->batched_matmul_helper_layer_->forward(
        ffn_norm_out, w1_matmul->get_weight(0), w1_out, seq_len, 1.f));
    STATUS_CHECK(layers->batched_matmul_helper_layer_->forward(
        ffn_norm_out, w3_matmul->get_weight(0), w3_out, seq_len, 1.f));
  }

  // Batched SwiGLU
  STATUS_CHECK(layers->batched_swiglu_layer_->forward(w1_out, w3_out, w1_out));

  // Batched W2
  const auto& w2_layer = layers->w2_layers_.at(layer_idx);
  tensor::Tensor w2_out(activation_dtype, seq_len, config_->dim_, true, alloc);

  auto w2_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(w2_layer);
  if (w2_awq) {
    STATUS_CHECK(w2_awq->forward(w1_out, w2_out));
  } else {
    auto w2_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(w2_layer);
    CHECK_NE(w2_matmul, nullptr) << "W2 layer is not a MatmulLayer";
    STATUS_CHECK(layers->batched_matmul_helper_layer_->forward(
        w1_out, w2_matmul->get_weight(0), w2_out, seq_len, 1.f));
  }

  // Residual add
  STATUS_CHECK(layers->batched_add_layer_->forward(input, w2_out, input));
}

// Optimized version with pre-allocated buffers
void QwenBaseModel::batched_feed_forward_optimized(
    int32_t layer_idx, const tensor::Tensor& input,
    tensor::Tensor& ffn_norm_out, tensor::Tensor& w1_out,
    tensor::Tensor& w3_out, tensor::Tensor& w2_out, int32_t seq_len) const {
  auto* layers = get_base_layers();
  CHECK(layers != nullptr);

  // FFN RMSNorm
  const auto& ffn_rmsnorm = layers->rmsnorm_layers_.at(layer_idx + config_->layer_num_);
  STATUS_CHECK(ffn_rmsnorm->forward(input, ffn_norm_out));

  // Batched W1 and W3
  const auto& w1_layer = layers->w1_layers_.at(layer_idx);
  const auto& w3_layer = layers->w3_layers_.at(layer_idx);

  auto w1_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(w1_layer);
  auto w3_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(w3_layer);

  if (w1_awq && w3_awq) {
    STATUS_CHECK(w1_awq->forward(ffn_norm_out, w1_out));
    STATUS_CHECK(w3_awq->forward(ffn_norm_out, w3_out));
  } else {
    auto w1_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(w1_layer);
    auto w3_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(w3_layer);
    CHECK_NE(w1_matmul, nullptr) << "W1 layer is not a MatmulLayer";
    CHECK_NE(w3_matmul, nullptr) << "W3 layer is not a MatmulLayer";
    STATUS_CHECK(layers->batched_matmul_helper_layer_->forward(
        ffn_norm_out, w1_matmul->get_weight(0), w1_out, seq_len, 1.f));
    STATUS_CHECK(layers->batched_matmul_helper_layer_->forward(
        ffn_norm_out, w3_matmul->get_weight(0), w3_out, seq_len, 1.f));
  }

  // Batched SwiGLU
  STATUS_CHECK(layers->batched_swiglu_layer_->forward(w1_out, w3_out, w1_out));

  // Batched W2
  const auto& w2_layer = layers->w2_layers_.at(layer_idx);
  auto w2_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(w2_layer);
  if (w2_awq) {
    STATUS_CHECK(w2_awq->forward(w1_out, w2_out));
  } else {
    auto w2_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(w2_layer);
    CHECK_NE(w2_matmul, nullptr) << "W2 layer is not a MatmulLayer";
    STATUS_CHECK(layers->batched_matmul_helper_layer_->forward(
        w1_out, w2_matmul->get_weight(0), w2_out, seq_len, 1.f));
  }

  // Residual add
  STATUS_CHECK(layers->batched_add_layer_->forward(input, w2_out, input));
}

// ==================== Prefill ====================

base::Status QwenBaseModel::prefill(const tensor::Tensor& input, int32_t seq_len,
                                    int32_t start_pos) const {
  if (input.is_empty()) {
    return base::error::InvalidArgument("The input tensor is empty.");
  }
  if (device_type_ != base::DeviceType::kDeviceCUDA) {
    return base::error::InternalError("Batched prefill only supports CUDA device");
  }

  auto* layers = get_base_layers();
  std::shared_ptr<base::DeviceAllocator> alloc = base::CUDADeviceAllocatorFactory::get_instance();

  base::DataType activation_dtype = input.data_type();
  size_t elem_size = (activation_dtype == base::DataType::kDataTypeFp16)
      ? sizeof(uint16_t) : sizeof(float);

  int32_t dim = config_->dim_;
  int32_t hidden_dim = config_->hidden_dim_;

  // Double-buffering for hidden states
  tensor::Tensor hidden_buf0(activation_dtype, seq_len, dim, true, alloc);
  tensor::Tensor hidden_buf1(activation_dtype, seq_len, dim, true, alloc);
  tensor::Tensor rms_out(activation_dtype, seq_len, dim, true, alloc);
  tensor::Tensor query_out(activation_dtype, seq_len, dim, true, alloc);
  tensor::Tensor key_out(activation_dtype, seq_len, config_->kv_dim_, true, alloc);
  tensor::Tensor value_out(activation_dtype, seq_len, config_->kv_dim_, true, alloc);
  tensor::Tensor mha_out(activation_dtype, seq_len, dim, true, alloc);
  tensor::Tensor wo_out(activation_dtype, seq_len, dim, true, alloc);

  // Pre-allocate FFN buffers
  tensor::Tensor ffn_norm_out(activation_dtype, seq_len, dim, true, alloc);
  tensor::Tensor w1_out(activation_dtype, seq_len, hidden_dim, true, alloc);
  tensor::Tensor w3_out(activation_dtype, seq_len, hidden_dim, true, alloc);
  tensor::Tensor w2_out(activation_dtype, seq_len, dim, true, alloc);

  tensor::Tensor* hidden_buffers[2] = {&hidden_buf0, &hidden_buf1};

  for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
    const tensor::Tensor* layer_input;
    tensor::Tensor* layer_output;

    if (layer_idx == 0) {
      layer_input = &input;
      layer_output = hidden_buffers[0];
    } else {
      layer_input = hidden_buffers[(layer_idx - 1) % 2];
      layer_output = hidden_buffers[layer_idx % 2];
    }

    // 1. Batched Attention RMSNorm
    batched_attention_rms(layer_idx, *layer_input, rms_out, seq_len);

    // 2. Q/K/V projections + position encoding + KV cache update
    batched_attention_qkv(layer_idx, rms_out, query_out, key_out, value_out,
                          seq_len, start_pos);

    // 3. Multi-head attention + wo projection
    batched_attention_mha(layer_idx, query_out, mha_out, wo_out, seq_len, start_pos);

    // 4. Residual add
    STATUS_CHECK(layers->batched_add_layer_->forward(*layer_input, wo_out, *layer_output));

    // 5. Feed forward with residual
    batched_feed_forward_optimized(layer_idx, *layer_output, ffn_norm_out,
                                   w1_out, w3_out, w2_out, seq_len);
  }

  // Final layer norm + cls_logits on last token
  tensor::Tensor* final_hidden = hidden_buffers[(config_->layer_num_ - 1) % 2];
  void* last_token_ptr = static_cast<char*>(
      const_cast<void*>(final_hidden->get_buffer()->ptr())) +
      (seq_len - 1) * dim * elem_size;
  tensor::Tensor last_hidden(activation_dtype, dim, false, nullptr, last_token_ptr);
  last_hidden.set_device_type(device_type_);

  cls_logits(last_hidden);

  return base::error::Success();
}

// ==================== Decode ====================

base::Status QwenBaseModel::decode(const tensor::Tensor& input, int32_t pos, int& next) const {
  if (input.is_empty()) {
    return base::error::InvalidArgument("The input tensor is empty.");
  }

  // Check if we should use CUDA Graph optimization
  bool use_graph = cuda_config_ && cuda_config_->should_use_graph();

  if (use_graph) {
    tensor::Tensor pos_tensor_gpu = get_buffer(ModelBufferType::kInputPosGPU);
    tensor::Tensor decode_input = get_buffer(ModelBufferType::kDecodeInput);
    tensor::Tensor pos_pinned = get_buffer(ModelBufferType::kInputPosPinned);
    tensor::Tensor argmax_output = get_buffer(ModelBufferType::kArgmaxOutput);
    tensor::Tensor argmax_pinned = get_buffer(ModelBufferType::kArgmaxOutputPinned);

    auto& graph_ctx = cuda_config_->graph_context;
    auto& graph = graph_ctx->decode_graph;
    bool need_capture = !graph->is_valid();

    // Determine element size based on data type
    size_t decode_elem_size = (decode_input.data_type() == base::DataType::kDataTypeFp16)
        ? sizeof(uint16_t) : sizeof(float);

    // Copy input to fixed decode_input buffer
    cudaMemcpyAsync(const_cast<void*>(decode_input.get_buffer()->ptr()),
                    input.get_buffer()->ptr(),
                    config_->dim_ * decode_elem_size,
                    cudaMemcpyDeviceToDevice, cuda_config_->stream);

    // Update position using pinned memory
    *const_cast<int32_t*>(pos_pinned.ptr<int32_t>()) = pos;
    cudaMemcpyAsync(const_cast<int32_t*>(pos_tensor_gpu.ptr<int32_t>()),
                    pos_pinned.ptr<int32_t>(), sizeof(int32_t),
                    cudaMemcpyHostToDevice, cuda_config_->stream);

    if (need_capture && !graph->is_disabled()) {
      cudaStreamSynchronize(cuda_config_->stream);

      if (graph->begin_capture(cuda_config_->stream)) {
        for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
          attention_rms(layer_idx, decode_input);
          attention_qkv_with_graph(layer_idx, pos_tensor_gpu);
          attention_mha_with_graph(layer_idx, pos_tensor_gpu);
          if (use_fused_ffn_) {
            feed_forward_fused(layer_idx, decode_input);
          } else {
            feed_forward(layer_idx, decode_input);
          }
        }
        cls_logits(decode_input);

        if (graph->end_capture(cuda_config_->stream)) {
          graph_ctx->graph_recaptures++;
        }
      }
    }

    if (graph->is_valid()) {
      if (graph->launch(cuda_config_->stream)) {
        graph_ctx->graph_launches++;

        tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
        auto* argmax_sampler = dynamic_cast<sampler::ArgmaxSampler*>(sampler_.get());
        if (argmax_sampler) {
          argmax_sampler->sample_prealloc(
              forward_output.ptr<float>(), forward_output.size(),
              reinterpret_cast<size_t*>(const_cast<int32_t*>(argmax_output.ptr<int32_t>())),
              reinterpret_cast<size_t*>(const_cast<int32_t*>(argmax_pinned.ptr<int32_t>())),
              cuda_config_->stream);
          cudaStreamSynchronize(cuda_config_->stream);
          next = static_cast<int32_t>(*reinterpret_cast<size_t*>(
              const_cast<int32_t*>(argmax_pinned.ptr<int32_t>())));
        } else {
          cudaStreamSynchronize(cuda_config_->stream);
          tensor::Tensor pos_tensor_cpu = get_buffer(ModelBufferType::kInputPos);
          next = post_processing(pos_tensor_cpu, false);
        }
        return base::error::Success();
      }
      graph_ctx->invalidate();
    }
  }

  // Normal execution path (no graph, or graph failed)
  tensor::Tensor pos_tensor = get_buffer(ModelBufferType::kInputPos);
  pos_tensor.index<int32_t>(0) = pos;

  for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
    attention_rms(layer_idx, input);
    attention_qkv(layer_idx, pos_tensor);
    attention_mha(layer_idx, pos_tensor);
    if (use_fused_ffn_) {
      feed_forward_fused(layer_idx, input);
    } else {
      feed_forward(layer_idx, input);
    }
  }

  cls_logits(input);

  if (cuda_config_ && cuda_config_->stream) {
    cudaStreamSynchronize(cuda_config_->stream);
  }

  next = post_processing(pos_tensor, false);
  return base::error::Success();
}

// ==================== KV Cache Management ====================

void QwenBaseModel::clear_kv_cache() {
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  tensor::Tensor value_cache = get_buffer(ModelBufferType::kValueCache);

  size_t elem_size = (key_cache.data_type() == base::DataType::kDataTypeFp16)
      ? sizeof(uint16_t) : sizeof(float);

  if (device_type_ == base::DeviceType::kDeviceCUDA && cuda_config_) {
    cudaMemsetAsync(const_cast<void*>(key_cache.get_buffer()->ptr()), 0,
                    key_cache.size() * elem_size, cuda_config_->stream);
    cudaMemsetAsync(const_cast<void*>(value_cache.get_buffer()->ptr()), 0,
                    value_cache.size() * elem_size, cuda_config_->stream);
    cudaStreamSynchronize(cuda_config_->stream);
    invalidate_cuda_graph();
  } else {
    memset(const_cast<void*>(key_cache.get_buffer()->ptr()), 0, key_cache.size() * elem_size);
    memset(const_cast<void*>(value_cache.get_buffer()->ptr()), 0, value_cache.size() * elem_size);
  }
}

// ==================== Attention Type ====================

void QwenBaseModel::set_attention_type(base::AttentionType type) {
  Model::set_attention_type(type);
  auto* layers = get_base_layers();
  if (layers) {
    if (layers->flash_attention_decode_layer_) {
      layers->flash_attention_decode_layer_->set_attention_type(type);
    }
    if (layers->flash_attention_prefill_layer_) {
      layers->flash_attention_prefill_layer_->set_attention_type(type);
    }
  }
}

}  // namespace model
