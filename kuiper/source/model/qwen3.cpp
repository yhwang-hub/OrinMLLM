#ifdef QWEN3_SUPPORT
#include "model/qwen3.h"
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <op/matmul.h>
#include <op/mha.h>
#include <op/rmsnorm.h>
#include <op/embedding.h>
#include <op/awq_matmul.h>
#include <sentencepiece_processor.h>
#include <utility>
#include <set>
#include "../op/kernels/cpu/rope_kernel.h"
#include "../op/kernels/cuda/rope_kernel.cuh"
#include "../op/kernels/cuda/matmul_kernel.cuh"
#include "../op/kernels/cuda/mha_kernel.cuh"
#include "../op/kernels/cuda/rmsnorm_kernel.cuh"
#include "../op/kernels/cuda/add_kernel.cuh"
#include "../op/kernels/cuda/swiglu_kernel.cuh"
#include "../op/kernels/cuda/flash_attention_kernel.cuh"
#include "../op/kernels/cuda/kv_cache_kernel.cuh"
#include "../op/kernels/cuda/fused_ffn_kernel.cuh"
#include "../op/kernels/cuda/fp16_convert_kernel.cuh"
#include "../op/kernels/cuda/argmax_kernel.cuh"
#include "../op/kernels/cuda/awq_gemm_tensorcore.cuh"
#include "sampler/argmax_sampler.h"
#include "base/tick.h"

// Set to 1 to use Flash Attention, 0 for standard batched attention
#define USE_FLASH_ATTENTION 1

namespace model {

void Qwen3Layers::to_cuda(std::shared_ptr<kernel::CudaConfig> config, bool keep_fp16_weights) {
  // Helper lambda to set keep_fp16_weights for LayerParam layers
  auto set_fp16_flag = [keep_fp16_weights](const std::shared_ptr<op::Layer>& layer) {
    if (auto layer_param = std::dynamic_pointer_cast<op::LayerParam>(layer)) {
      layer_param->set_keep_fp16_weights(keep_fp16_weights);
    }
  };

  if (add_layer_) {
    add_layer_->set_cuda_config(config);
    add_layer_->to_cuda();
  }

  if (rope_layer_) {
    rope_layer_->set_cuda_config(config);
    rope_layer_->to_cuda();
  }

  if (swiglu_layer_) {
    swiglu_layer_->set_cuda_config(config);
    swiglu_layer_->to_cuda();
  }

  if (cls_layer_) {
    set_fp16_flag(cls_layer_);
    cls_layer_->set_cuda_config(config);
    cls_layer_->to_cuda();
  }

  if (embedding_layer_) {
    set_fp16_flag(embedding_layer_);
    embedding_layer_->set_cuda_config(config);
    embedding_layer_->to_cuda();
  }

  if (mha_layer_) {
    mha_layer_->set_cuda_config(config);
    mha_layer_->to_cuda();
  }

  for (auto& weight_layer : wq_layers_) {
    if (weight_layer) {
      set_fp16_flag(weight_layer);
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : wk_layers_) {
    if (weight_layer) {
      set_fp16_flag(weight_layer);
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : wv_layers_) {
    if (weight_layer) {
      set_fp16_flag(weight_layer);
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : wo_layers_) {
    if (weight_layer) {
      set_fp16_flag(weight_layer);
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : w1_layers_) {
    if (weight_layer) {
      set_fp16_flag(weight_layer);
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : w2_layers_) {
    if (weight_layer) {
      set_fp16_flag(weight_layer);
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : w3_layers_) {
    if (weight_layer) {
      set_fp16_flag(weight_layer);
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& rms_norm_layer : rmsnorm_layers_) {
    if (rms_norm_layer) {
      set_fp16_flag(rms_norm_layer);
      rms_norm_layer->to_cuda();
      rms_norm_layer->set_cuda_config(config);
    }
  }

  // New layers for unified kernel calls
  if (flash_attention_decode_layer_) {
    flash_attention_decode_layer_->set_cuda_config(config);
    flash_attention_decode_layer_->to_cuda();
  }
  if (flash_attention_prefill_layer_) {
    flash_attention_prefill_layer_->set_cuda_config(config);
    flash_attention_prefill_layer_->to_cuda();
  }
  if (kv_cache_key_layer_) {
    kv_cache_key_layer_->set_cuda_config(config);
    kv_cache_key_layer_->to_cuda();
  }
  if (kv_cache_value_layer_) {
    kv_cache_value_layer_->set_cuda_config(config);
    kv_cache_value_layer_->to_cuda();
  }
  if (fused_ffn_layer_) {
    fused_ffn_layer_->set_cuda_config(config);
    fused_ffn_layer_->to_cuda();
  }
  if (rope_gpu_pos_layer_) {
    rope_gpu_pos_layer_->set_cuda_config(config);
    rope_gpu_pos_layer_->to_cuda();
  }
  if (batched_rope_layer_) {
    batched_rope_layer_->set_cuda_config(config);
    batched_rope_layer_->to_cuda();
  }
  if (batched_add_layer_) {
    batched_add_layer_->set_cuda_config(config);
    batched_add_layer_->to_cuda();
  }
  if (batched_swiglu_layer_) {
    batched_swiglu_layer_->set_cuda_config(config);
    batched_swiglu_layer_->to_cuda();
  }
  if (sin_cos_cache_layer_) {
    sin_cos_cache_layer_->set_cuda_config(config);
    sin_cos_cache_layer_->to_cuda();
  }
  if (mha_gpu_pos_layer_) {
    mha_gpu_pos_layer_->set_cuda_config(config);
    mha_gpu_pos_layer_->to_cuda();
  }
  if (batched_mha_layer_) {
    batched_mha_layer_->set_cuda_config(config);
    batched_mha_layer_->to_cuda();
  }
  if (batched_matmul_helper_layer_) {
    batched_matmul_helper_layer_->set_cuda_config(config);
    batched_matmul_helper_layer_->to_cuda();
  }
}

Qwen3Model::Qwen3Model(base::TokenizerType tokenizer_type, std::string token_path,
                       std::string model_path, bool is_quant_model)
    : Model(tokenizer_type, base::ModelType::kModelTypeLLama2, std::move(token_path),
            std::move(model_path), is_quant_model) {}

base::Status Qwen3Model::init(base::DeviceType device_type) {
  using namespace base;
  if (token_path_.empty()) {
    return error::PathNotValid(token_path_);
  }
  if (device_type == base::DeviceType::kDeviceCPU && is_quant_model_) {
    return error::InternalError("The cpu device do not support int8 quant model.");
  }

  device_type_ = device_type;
  if (device_type == DeviceType::kDeviceCUDA) {
    cudaSetDevice(0);
    cuda_config_ = std::make_shared<kernel::CudaConfig>();
    cudaStreamCreate(&cuda_config_->stream);
    
    // Create cuBLAS handle for optimized GEMM/GEMV operations
    cublasStatus_t cublas_status = cublasCreate(&cuda_config_->cublas_handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
      return error::InternalError("Failed to create cuBLAS handle.");
    }
    cublasSetStream(cuda_config_->cublas_handle, cuda_config_->stream);
    cublasSetMathMode(cuda_config_->cublas_handle, CUBLAS_DEFAULT_MATH);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      return error::InternalError("The cuda hanle create failed.");
    }
  }

  Status read_status = gen_model_from_file();
  if (!read_status) {
    return read_status;
  }

  init_mem();
  // Initialize sin/cos cache via layer forward
  CHECK_NE(qwen_layers_->sin_cos_cache_layer_, nullptr);
  qwen_layers_->sin_cos_cache_layer_->forward(config_->head_size_, config_->seq_len_,
                                              get_buffer(ModelBufferType::kSinCache),
                                              get_buffer(ModelBufferType::kCosCache));

  sampler_ = std::make_unique<sampler::ArgmaxSampler>(device_type_);
  return error::Success();
}

base::Status Qwen3Model::forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                                 int& next) const {
  if (input.is_empty()) {
    return base::error::InvalidArgument("The input tensor is empty.");
  }
  if (device_type_ == base::DeviceType::kDeviceCPU && is_quant_model_) {
    return base::error::InternalError("Unsupported int8 quant in the cpu device");
  }

  for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
    attention_rms(layer_idx, input);
    // attention (wq wk wv @ input)
    attention_qkv(layer_idx, pos_tensor);
    // multi-head attention
    attention_mha(layer_idx, pos_tensor);
    // feed forward
    feed_forward(layer_idx, input);
  }
  cls_logits(input);
  return base::error::Success();
}

void Qwen3Model::create_nonparam_layers() {
  CHECK(qwen_layers_ != nullptr);
  qwen_layers_->rope_layer_ = std::make_shared<op::RoPELayer>(
      device_type_, config_->dim_, config_->kv_dim_, config_->head_size_);

  qwen_layers_->mha_layer_ = std::make_shared<op::MultiHeadAttention>(
      device_type_, 0, config_->kv_mul_, config_->kv_dim_, config_->seq_len_, config_->head_num_,
      config_->head_size_);

  qwen_layers_->add_layer_ = std::make_shared<op::VecAddLayer>(device_type_);

  qwen_layers_->swiglu_layer_ =
      std::make_shared<op::SwiGLULayer>(device_type_, config_->immediate_dim_);

  // Create new layers for unified kernel calls with proper parameters
  qwen_layers_->flash_attention_decode_layer_ = std::make_shared<op::FlashAttentionDecodeLayer>(
      device_type_, config_->head_num_, config_->kv_head_num_, config_->head_size_,
      config_->kv_mul_, config_->seq_len_, config_->kv_dim_, is_fp16_model_);
  
  qwen_layers_->flash_attention_prefill_layer_ = std::make_shared<op::FlashAttentionPrefillLayer>(
      device_type_, config_->head_num_, config_->kv_head_num_, config_->head_size_,
      config_->seq_len_, is_fp16_model_);
  
  qwen_layers_->kv_cache_key_layer_ = std::make_shared<op::KVCacheLayer>(
      device_type_, config_->kv_dim_, config_->seq_len_, is_fp16_model_);
  
  qwen_layers_->kv_cache_value_layer_ = std::make_shared<op::KVCacheLayer>(
      device_type_, config_->kv_dim_, config_->seq_len_, is_fp16_model_);
  
  qwen_layers_->fused_ffn_layer_ = std::make_shared<op::FusedFFNLayer>(
      device_type_, config_->dim_, config_->hidden_dim_, is_fp16_model_, false);
  
  qwen_layers_->rope_gpu_pos_layer_ = std::make_shared<op::RoPEGpuPosLayer>(
      device_type_, config_->dim_, config_->kv_dim_, config_->head_size_, is_fp16_model_);
  
  qwen_layers_->batched_rope_layer_ = std::make_shared<op::BatchedRoPELayer>(
      device_type_, config_->dim_, config_->kv_dim_, config_->head_size_,
      config_->head_num_, config_->kv_head_num_);
  
  // Create batched add/swiglu layers for prefill (support any-dim tensors)
  qwen_layers_->batched_add_layer_ = std::make_shared<op::BatchedAddLayer>(device_type_);
  qwen_layers_->batched_swiglu_layer_ = std::make_shared<op::BatchedSwiGLULayer>(device_type_);
  
  // Create misc layers for unified kernel access
  qwen_layers_->sin_cos_cache_layer_ = std::make_shared<op::SinCosCacheLayer>(device_type_);
  qwen_layers_->mha_gpu_pos_layer_ = std::make_shared<op::MHAGpuPosLayer>(device_type_);
  qwen_layers_->batched_mha_layer_ = std::make_shared<op::BatchedMHALayer>(device_type_);
  qwen_layers_->batched_matmul_helper_layer_ = std::make_shared<op::BatchedMatmulHelperLayer>(device_type_);
}

void Qwen3Model::create_param_quant_layers() {
  // AWQ quantized layers
  if (is_awq_model_) {
    create_param_layers_awq();
  }
}

void Qwen3Model::create_param_layers() {
  // This function is for FP32 weights
  // For FP16 weights, use create_param_layers_fp16()
  // For AWQ weights, use create_param_layers_awq()
  if (is_awq_model_) {
    create_param_layers_awq();
    return;
  }
  if (is_fp16_model_) {
    create_param_layers_fp16();
    return;
  }

  CHECK(qwen_layers_ != nullptr);

  size_t pos = 0;
  int32_t dim = config_->dim_;
  int32_t kv_dim = config_->kv_dim_;
  int hidden_dim = config_->hidden_dim_;
  auto cpu_device_type = base::DeviceType::kDeviceCPU;
  float* weight_ptr = (float*)raw_model_data_->weight(pos);

  // rmsnorm attention attention, ffn, final
  for (int32_t i = 0; i < 2 * config_->layer_num_ + 1; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, hidden_dim);

    rms_norm_layer->set_weight(0, {hidden_dim}, weight_ptr, cpu_device_type);
    qwen_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    weight_ptr += hidden_dim;
  }
  pos += (2 * config_->layer_num_ + 1) * hidden_dim;

  // embedding layer
  qwen_layers_->embedding_layer_ = std::make_shared<op::EmbeddingLayer>(
      device_type_, hidden_dim, config_->seq_len_, std::abs(config_->vocab_size_));
  qwen_layers_->embedding_layer_->set_weight(0, {std::abs(config_->vocab_size_), hidden_dim},
                                             weight_ptr, cpu_device_type);
  pos += config_->vocab_size_ * hidden_dim;

  // query
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, hidden_dim, false);
    wq->set_weight(0, {dim, hidden_dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->wq_layers_.push_back(wq);
    pos = pos + hidden_dim * dim;
  }

  // key
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wk = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, kv_dim, false);
    wk->set_weight(0, {hidden_dim, kv_dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->wk_layers_.push_back(wk);
    pos = pos + hidden_dim * kv_dim;
  }

  // value
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wv = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, kv_dim, false);
    wv->set_weight(0, {hidden_dim, kv_dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->wv_layers_.push_back(wv);
    pos += kv_dim * hidden_dim;
  }

  // output
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wo = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim, false);
    wo->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->wo_layers_.push_back(wo);
    pos = pos + dim * hidden_dim;
  }

  // w1 layers
  int32_t immediate_dim = config_->immediate_dim_;
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w1 = std::make_shared<op::MatmulLayer>(device_type_, immediate_dim, hidden_dim, false);
    w1->set_weight(0, {immediate_dim, hidden_dim}, this->raw_model_data_->weight(pos),
                   cpu_device_type);
    qwen_layers_->w1_layers_.push_back(w1);
    pos = pos + hidden_dim * immediate_dim;
  }

  // w2 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w2 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, immediate_dim, false);
    w2->set_weight(0, {hidden_dim, immediate_dim}, this->raw_model_data_->weight(pos),
                   cpu_device_type);
    qwen_layers_->w2_layers_.push_back(w2);
    pos = pos + immediate_dim * hidden_dim;
  }

  // w3 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w3 = std::make_shared<op::MatmulLayer>(device_type_, immediate_dim, hidden_dim, false);
    w3->set_weight(0, {immediate_dim, hidden_dim}, this->raw_model_data_->weight(pos),
                   cpu_device_type);
    qwen_layers_->w3_layers_.push_back(w3);
    pos = pos + immediate_dim * hidden_dim;
  }

  // cls layer (lm_head)
  auto lm_head = std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_,
                                                   config_->hidden_dim_, false);
  lm_head->set_weight(0, {config_->vocab_size_, config_->hidden_dim_},
                      this->raw_model_data_->weight(pos), cpu_device_type);
  qwen_layers_->cls_layer_ = lm_head;
  pos = pos + config_->vocab_size_ * config_->hidden_dim_;

  // Qwen3 specific: q_norm and k_norm at the end
  // q_norm
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, config_->head_size_);
    rms_norm_layer->set_weight(0, {config_->head_size_}, this->raw_model_data_->weight(pos),
                               cpu_device_type);
    qwen_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    pos = pos + config_->head_size_;
  }

  // k_norm
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, config_->head_size_);
    rms_norm_layer->set_weight(0, {config_->head_size_}, this->raw_model_data_->weight(pos),
                               cpu_device_type);
    qwen_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    pos = pos + config_->head_size_;
  }
}

void Qwen3Model::create_param_layers_fp16() {
  CHECK(qwen_layers_ != nullptr);
  LOG(INFO) << "Loading Qwen3 FP16 model weights...";

  size_t pos = 0;  // position in FP16 elements
  int32_t dim = config_->dim_;
  int32_t kv_dim = config_->kv_dim_;
  int32_t hidden_dim = config_->hidden_dim_;
  int32_t immediate_dim = config_->immediate_dim_;
  auto cpu_device_type = base::DeviceType::kDeviceCPU;

  // Weight order in FP16 Qwen3 export (from export_qwen3-8B-fp16.py):
  // 1. attention_norm (input_layernorm) for all layers - size: dim
  // 2. ffn_norm (post_attention_layernorm) for all layers - size: dim
  // 3. final norm - size: dim
  // 4. token embeddings - size: vocab_size * dim
  // 5. wq for all layers - size: dim * dim
  // 6. wk for all layers - size: dim * kv_dim
  // 7. wv for all layers - size: dim * kv_dim
  // 8. wo for all layers - size: dim * dim
  // 9. w1 (gate_proj) for all layers - size: dim * immediate_dim
  // 10. w2 (down_proj) for all layers - size: immediate_dim * dim
  // 11. w3 (up_proj) for all layers - size: dim * immediate_dim
  // 12. output (lm_head) if not shared - size: vocab_size * dim
  // 13. q_norm for all layers - size: head_size
  // 14. k_norm for all layers - size: head_size

  // 1. attention_norm layers (input_layernorm)
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto rms_norm_layer = std::make_shared<op::RmsNormLayer>(device_type_, dim);
    rms_norm_layer->set_weight_fp16(0, {dim}, raw_model_data_->weight(pos),
                                    cpu_device_type);
    qwen_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    pos += dim;
  }

  // 2. ffn_norm layers (post_attention_layernorm)
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto rms_norm_layer = std::make_shared<op::RmsNormLayer>(device_type_, dim);
    rms_norm_layer->set_weight_fp16(0, {dim}, raw_model_data_->weight(pos),
                                    cpu_device_type);
    qwen_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    pos += dim;
  }

  // 3. final norm
  {
    auto final_norm_layer = std::make_shared<op::RmsNormLayer>(device_type_, dim);
    final_norm_layer->set_weight_fp16(0, {dim}, raw_model_data_->weight(pos),
                                      cpu_device_type);
    qwen_layers_->rmsnorm_layers_.push_back(final_norm_layer);
    pos += dim;
  }

  // 4. token embeddings
  {
    auto emb_layer = std::make_shared<op::EmbeddingLayer>(
        device_type_, dim, config_->seq_len_, std::abs(config_->vocab_size_));
    emb_layer->set_weight_fp16(0, {std::abs(config_->vocab_size_), dim},
                               raw_model_data_->weight(pos),
                               cpu_device_type);
    qwen_layers_->embedding_layer_ = emb_layer;
  }
  pos += config_->vocab_size_ * dim;

  // 5. wq layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, false);
    wq->set_weight_fp16(0, {dim, dim}, raw_model_data_->weight(pos),
                        cpu_device_type);
    qwen_layers_->wq_layers_.push_back(wq);
    pos += dim * dim;
  }

  // 6. wk layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    // wk: input dim -> output kv_dim, so MatmulLayer(output_dim=kv_dim, input_dim=dim)
    auto wk = std::make_shared<op::MatmulLayer>(device_type_, kv_dim, dim, false);
    wk->set_weight_fp16(0, {kv_dim, dim}, raw_model_data_->weight(pos),
                        cpu_device_type);
    qwen_layers_->wk_layers_.push_back(wk);
    pos += kv_dim * dim;
  }

  // 7. wv layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    // wv: input dim -> output kv_dim, so MatmulLayer(output_dim=kv_dim, input_dim=dim)
    auto wv = std::make_shared<op::MatmulLayer>(device_type_, kv_dim, dim, false);
    wv->set_weight_fp16(0, {kv_dim, dim}, raw_model_data_->weight(pos),
                        cpu_device_type);
    qwen_layers_->wv_layers_.push_back(wv);
    pos += kv_dim * dim;
  }

  // 8. wo layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wo = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, false);
    wo->set_weight_fp16(0, {dim, dim}, raw_model_data_->weight(pos),
                        cpu_device_type);
    qwen_layers_->wo_layers_.push_back(wo);
    pos += dim * dim;
  }

  // 9. w1 layers (gate_proj)
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w1 = std::make_shared<op::MatmulLayer>(device_type_, immediate_dim, dim, false);
    w1->set_weight_fp16(0, {immediate_dim, dim}, raw_model_data_->weight(pos),
                        cpu_device_type);
    qwen_layers_->w1_layers_.push_back(w1);
    pos += dim * immediate_dim;
  }

  // 10. w2 layers (down_proj)
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w2 = std::make_shared<op::MatmulLayer>(device_type_, dim, immediate_dim, false);
    w2->set_weight_fp16(0, {dim, immediate_dim}, raw_model_data_->weight(pos),
                        cpu_device_type);
    qwen_layers_->w2_layers_.push_back(w2);
    pos += immediate_dim * dim;
  }

  // 11. w3 layers (up_proj)
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w3 = std::make_shared<op::MatmulLayer>(device_type_, immediate_dim, dim, false);
    w3->set_weight_fp16(0, {immediate_dim, dim}, raw_model_data_->weight(pos),
                        cpu_device_type);
    qwen_layers_->w3_layers_.push_back(w3);
    pos += dim * immediate_dim;
  }

  // 12. output (lm_head) if not shared
  if (!config_->is_shared_weight_) {
    auto lm_head = std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_, dim, false);
    lm_head->set_weight_fp16(0, {config_->vocab_size_, dim},
                             raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->cls_layer_ = lm_head;
    pos += config_->vocab_size_ * dim;
  } else {
    // Share weights with embedding layer
    auto lm_head = std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_, dim, false);
    // Get embedding weight pointer (vocab_size * dim after initial norms)
    size_t emb_pos = (2 * config_->layer_num_ + 1) * dim;
    lm_head->set_weight_fp16(0, {config_->vocab_size_, dim},
                             raw_model_data_->weight(emb_pos), cpu_device_type);
    qwen_layers_->cls_layer_ = lm_head;
  }

  // 13. q_norm for all layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto rms_norm_layer = std::make_shared<op::RmsNormLayer>(device_type_, config_->head_size_);
    rms_norm_layer->set_weight_fp16(0, {config_->head_size_},
                                    raw_model_data_->weight(pos),
                                    cpu_device_type);
    qwen_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    pos += config_->head_size_;
  }

  // 14. k_norm for all layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto rms_norm_layer = std::make_shared<op::RmsNormLayer>(device_type_, config_->head_size_);
    rms_norm_layer->set_weight_fp16(0, {config_->head_size_},
                                    raw_model_data_->weight(pos),
                                    cpu_device_type);
    qwen_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    pos += config_->head_size_;
  }

  LOG(INFO) << "Qwen3 FP16 model loaded successfully. Total FP16 elements: " << pos;
}

void Qwen3Model::create_param_layers_awq() {
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
      
      if (i == 0) {
        LOG(INFO) << "  " << name << " layer loaded: [" << in_features << " x " << out_features << "]";
      }
    }
  };

  // 5. wq layers (q_proj) - AWQ
  LOG(INFO) << "Loading AWQ wq layers...";
  load_awq_layer(dim, dim, qwen_layers_->wq_layers_, "wq");

  // 6. wk layers (k_proj) - AWQ
  LOG(INFO) << "Loading AWQ wk layers...";
  load_awq_layer(dim, kv_dim, qwen_layers_->wk_layers_, "wk");

  // 7. wv layers (v_proj) - AWQ
  LOG(INFO) << "Loading AWQ wv layers...";
  load_awq_layer(dim, kv_dim, qwen_layers_->wv_layers_, "wv");

  // 8. wo layers (o_proj) - AWQ
  LOG(INFO) << "Loading AWQ wo layers...";
  load_awq_layer(dim, dim, qwen_layers_->wo_layers_, "wo");

  // 9. w1 layers (gate_proj) - AWQ
  LOG(INFO) << "Loading AWQ w1 (gate_proj) layers...";
  load_awq_layer(dim, immediate_dim, qwen_layers_->w1_layers_, "w1");

  // 10. w2 layers (down_proj) - AWQ
  LOG(INFO) << "Loading AWQ w2 (down_proj) layers...";
  load_awq_layer(immediate_dim, dim, qwen_layers_->w2_layers_, "w2");

  // 11. w3 layers (up_proj) - AWQ
  LOG(INFO) << "Loading AWQ w3 (up_proj) layers...";
  load_awq_layer(dim, immediate_dim, qwen_layers_->w3_layers_, "w3");

  // 12. output (lm_head) - FP16 (not quantized)
  if (!config_->is_shared_weight_) {
    LOG(INFO) << "Loading lm_head layer (FP16)...";
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
  LOG(INFO) << "Loading q_norm layers (FP16)...";
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto rms_norm_layer = std::make_shared<op::RmsNormLayer>(device_type_, config_->head_size_);
    rms_norm_layer->set_weight_fp16(0, {config_->head_size_},
                                    base_ptr + pos, cpu_device_type);
    qwen_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    pos += config_->head_size_ * sizeof(uint16_t);
  }

  // 14. k_norm for all layers - FP16
  LOG(INFO) << "Loading k_norm layers (FP16)...";
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto rms_norm_layer = std::make_shared<op::RmsNormLayer>(device_type_, config_->head_size_);
    rms_norm_layer->set_weight_fp16(0, {config_->head_size_},
                                    base_ptr + pos, cpu_device_type);
    qwen_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    pos += config_->head_size_ * sizeof(uint16_t);
  }

  LOG(INFO) << "Qwen3 AWQ INT4 model loaded successfully. Total bytes: " << pos;
}

void Qwen3Model::init_mem() {
  std::shared_ptr<base::DeviceAllocator> alloc;
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    alloc = base::CPUDeviceAllocatorFactory::get_instance();
  } else {
    alloc = base::CUDADeviceAllocatorFactory::get_instance();
  }

  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK_NE(cuda_config_, nullptr);
    // Keep FP16 weights on GPU for FP16 models to save memory and enable FP16 compute
    qwen_layers_->to_cuda(cuda_config_, is_fp16_model_);
    
    // Note: Pre-dequantization is disabled to preserve AWQ's memory advantage
    // The optimized W4A16 fused kernels are used instead for both GEMV (decode) and GEMM (prefill)
    // This keeps AWQ at ~6GB memory while achieving competitive performance
    if (is_awq_model_) {
      LOG(INFO) << "AWQ model loaded with optimized W4A16 fused kernels (no pre-dequant)";
    }
  }

  std::shared_ptr<base::DeviceAllocator> alloc_cpu =
      base::CPUDeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::DeviceAllocator> alloc_cu =
      base::CUDADeviceAllocatorFactory::get_instance();

  // For FP16 models, use FP16 activation buffers to avoid FP16<->FP32 conversion overhead
  // This enables a pure FP16 compute path from embedding to output
  base::DataType activation_dtype = is_fp16_model_ ? 
      base::DataType::kDataTypeFp16 : base::DataType::kDataTypeFp32;
  
  if (is_fp16_model_) {
    LOG(INFO) << "Using FP16 activation buffers for pure FP16 compute path";
  }

  // For Qwen3: dim_ is the model dimension (4096), hidden_dim_ is intermediate_size (12288)
  int32_t model_dim = config_->dim_;
  int32_t intermediate_dim = config_->immediate_dim_;

  tensor::Tensor input_tokens(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  tensor::Tensor input_embeddings(activation_dtype, 1, model_dim, true,
                                  alloc);

  CHECK(insert_buffer(ModelBufferType::kInputTokens, input_tokens));
  CHECK(insert_buffer(ModelBufferType::kInputEmbeddings, input_embeddings));
  LOG(INFO) << "Allocated input buffers and embeddings buffers.";

  tensor::Tensor sin_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_,
                           true, alloc);
  tensor::Tensor cos_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_,
                           true, alloc);

  CHECK(insert_buffer(ModelBufferType::kSinCache, sin_cache));
  CHECK(insert_buffer(ModelBufferType::kCosCache, cos_cache));
  LOG(INFO) << "Allocated RoPE sin/cos cache buffers.";

  tensor::Tensor rms_output(activation_dtype, model_dim, true, alloc);
  tensor::Tensor out_mha(activation_dtype, config_->dim_, true, alloc);

  CHECK(insert_buffer(ModelBufferType::kOutputRMSNorm, rms_output));
  LOG(INFO) << "Allocated output RMSNorm buffer.";
  CHECK(insert_buffer(ModelBufferType::kOutputMHA, out_mha));
  LOG(INFO) << "Allocated output MHA buffer.";
  CHECK(insert_buffer(ModelBufferType::kW2Output, rms_output));
  LOG(INFO) << "Allocated W2 output buffer.";
  CHECK(insert_buffer(ModelBufferType::kFFNRMSNorm, rms_output));
  LOG(INFO) << "Allocated FFN RMSNorm buffer.";
  LOG(INFO) << "Allocated intermediate layer output buffers.";

  tensor::Tensor w1_output(activation_dtype, intermediate_dim, true, alloc);
  tensor::Tensor w3_output(activation_dtype, intermediate_dim, true, alloc);

  CHECK(insert_buffer(ModelBufferType::kW1Output, w1_output));
  LOG(INFO) << "Allocated W1 output buffer.";
  CHECK(insert_buffer(ModelBufferType::kW3Output, w3_output));
  LOG(INFO) << "Allocated W3 output buffer.";

  // kv cache - use FP16 for memory efficiency and bandwidth when model is FP16
  tensor::Tensor key_cache(activation_dtype, config_->layer_num_, config_->seq_len_,
                           config_->kv_dim_, true, alloc);
  tensor::Tensor value_cache(activation_dtype, config_->layer_num_, config_->seq_len_,
                             config_->kv_dim_, true, alloc);

  CHECK(insert_buffer(ModelBufferType::kKeyCache, key_cache));
  CHECK(insert_buffer(ModelBufferType::kValueCache, value_cache));

  // Wq query output
  tensor::Tensor query(activation_dtype, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kQuery, query));
  LOG(INFO) << "Allocated query output buffer.";

  // Pos tensor - on CPU for normal path
  tensor::Tensor pos_tensor(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  CHECK(insert_buffer(ModelBufferType::kInputPos, pos_tensor));
  LOG(INFO) << "Allocated input position buffer on CPU.";
  
  // Pos tensor on GPU for CUDA Graph path
  tensor::Tensor pos_tensor_gpu(base::DataType::kDataTypeInt32, 1, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kInputPosGPU, pos_tensor_gpu));
  LOG(INFO) << "Allocated input position buffer on GPU.";

  // Temporary K/V buffers with fixed addresses for CUDA Graph optimization
  tensor::Tensor temp_key(activation_dtype, config_->kv_dim_, true, alloc);
  tensor::Tensor temp_value(activation_dtype, config_->kv_dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kTempKey, temp_key));
  LOG(INFO) << "Allocated temporary key buffer.";
  CHECK(insert_buffer(ModelBufferType::kTempValue, temp_value));
  LOG(INFO) << "Allocated temporary value buffer.";
  
  // Fixed decode input buffer for CUDA Graph optimization
  tensor::Tensor decode_input(activation_dtype, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kDecodeInput, decode_input));
  LOG(INFO) << "Allocated decode input buffer.";
  
  // Pinned memory buffers for efficient async Host-Device transfers
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    std::shared_ptr<base::DeviceAllocator> alloc_pinned = 
        base::CPUPinnedAllocatorFactory::get_instance();
    
    // Pinned pos buffer for async H2D transfer
    tensor::Tensor pos_pinned(base::DataType::kDataTypeInt32, 1, true, alloc_pinned);
    CHECK(insert_buffer(ModelBufferType::kInputPosPinned, pos_pinned));
    LOG(INFO) << "Allocated pinned input position buffer.";

    // Pre-allocated argmax output buffer on GPU
    tensor::Tensor argmax_output(base::DataType::kDataTypeInt32, 2, true, alloc);
    CHECK(insert_buffer(ModelBufferType::kArgmaxOutput, argmax_output));
    LOG(INFO) << "Allocated argmax output buffer on GPU.";
    
    // Pinned argmax result buffer for async D2H transfer
    tensor::Tensor argmax_pinned(base::DataType::kDataTypeInt32, 2, true, alloc_pinned);
    CHECK(insert_buffer(ModelBufferType::kArgmaxOutputPinned, argmax_pinned));
    LOG(INFO) << "Allocated pinned argmax output buffer.";
  }

  // Attention scores - keep FP32 for numerical stability in softmax
  tensor::Tensor attn(base::DataType::kDataTypeFp32, config_->head_num_, config_->seq_len_, true,
                      alloc);
  CHECK(insert_buffer(ModelBufferType::kScoreStorage, attn));
  LOG(INFO) << "Allocated attention score buffer.";

  // Attention output uses activation dtype
  tensor::Tensor attn_output(activation_dtype, model_dim, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kAttnOutput, attn_output));
  LOG(INFO) << "Allocated attention output buffer.";

  // final forward output
  tensor::Tensor forward_output(base::DataType::kDataTypeFp32, config_->vocab_size_, true, alloc);
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    tensor::Tensor forward_output_cpu(base::DataType::kDataTypeFp32, config_->vocab_size_, true,
                                      alloc_cpu);
    CHECK(insert_buffer(ModelBufferType::kForwardOutputCPU, forward_output_cpu));
    LOG(INFO) << "Allocated forward output buffer on CPU.";
  }

  CHECK(insert_buffer(ModelBufferType::kForwardOutput, forward_output));
  LOG(INFO) << "Allocated forward output buffer on device.";
}

base::Status Qwen3Model::create_layers() {
  using namespace base;
  if (!qwen_layers_) {
    qwen_layers_ = std::make_unique<Qwen3Layers>();
  }

  if (!is_quant_model_) {
    create_param_layers();
  } else {
    return error::FunctionNotImplement("");
  }
  create_nonparam_layers();

  if (!qwen_layers_->embedding_layer_) {
    return error::InternalError("Create the embedding layer for the llama model failed!");
  }

  if (qwen_layers_->rmsnorm_layers_.size() != 4 * config_->layer_num_ + 1) {
    // input norm
    return error::InternalError("Create the rmsnorm layers for the llama model failed!");
  }

  if (qwen_layers_->wq_layers_.size() != config_->layer_num_ ||
      qwen_layers_->wk_layers_.size() != config_->layer_num_ ||
      qwen_layers_->wv_layers_.size() != config_->layer_num_ ||
      qwen_layers_->wo_layers_.size() != config_->layer_num_) {
    return error::InternalError(
        "Create the matmul layer in the attention and ffn attention layers for "
        "the llama model "
        "failed.");
  }

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    if (!qwen_layers_->wq_layers_.at(i) || !qwen_layers_->wk_layers_.at(i) ||
        !qwen_layers_->wv_layers_.at(i) || !qwen_layers_->wo_layers_.at(i)) {
      return error::InternalError(
          "Create the matmul layer in the attention and ffn attention layers for "
          "the llama model "
          "failed.");
    }
  }

  if (qwen_layers_->w1_layers_.size() != config_->layer_num_ ||
      qwen_layers_->w2_layers_.size() != config_->layer_num_ ||
      qwen_layers_->w3_layers_.size() != config_->layer_num_) {
    return error::InternalError(
        "Create the matmul layer in the feedforward layers for the llama model "
        "failed.");
  }

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    if (!qwen_layers_->w1_layers_.at(i) || !qwen_layers_->w2_layers_.at(i) ||
        !qwen_layers_->w3_layers_.at(i)) {
      return error::InternalError(
          "Create the matmul layer in the feedforward layers for the llama model "
          "failed.");
    }
  }

  if (!qwen_layers_->rope_layer_) {
    return error::InternalError("Create the rope layer for the llama model failed!");
  }

  if (!qwen_layers_->add_layer_) {
    return error::InternalError("Create the add layer for the llama model failed!");
  }

  if (!qwen_layers_->mha_layer_) {
    return error::InternalError("Create the mha layer for the llama model failed!");
  }

  if (!qwen_layers_->swiglu_layer_) {
    return error::InternalError("Create the SwiGLU layer for the llama model failed!");
  }
  return error::Success();
}

void Qwen3Model::attention_rms(int32_t layer_idx, const tensor::Tensor& input) const {
  CHECK(qwen_layers_ != nullptr);
  // attn rmsnorm
  tensor::Tensor rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  std::shared_ptr<op::Layer> rmsnorm_layer = qwen_layers_->rmsnorm_layers_.at(layer_idx);
  if (!rmsnorm_layer) {
    LOG(FATAL) << "The attention rmsnorm layer is a null pointer in the llama2 model";
  }
  STATUS_CHECK(rmsnorm_layer->forward(input, rmsnorm_output));
}

void Qwen3Model::attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  CHECK(qwen_layers_ != nullptr);
  // kv cache
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
  int32_t pos = pos_tensor.index<int32_t>(0);
  // wq wk wv @ input
  auto [key, val] = slice_kv_cache(layer_idx, pos);

  // query
  const auto& query_layer = qwen_layers_->wq_layers_.at(layer_idx);
  CHECK_NE(query_layer, nullptr) << "The query layer in the attention block is null pointer.";

  auto rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  STATUS_CHECK(query_layer->forward(rmsnorm_output, query));

  // query norm
  auto query_norm = qwen_layers_->rmsnorm_layers_.at(layer_idx + 2 * config_->layer_num_ + 1);
  query.reshape({(int32_t)query.size() / config_->head_size_, config_->head_size_});
  query_norm->forward(query, query);
  query.reshape({(int32_t)query.size()});

  // key
  const auto& key_layer = qwen_layers_->wk_layers_.at(layer_idx);
  CHECK_NE(key_layer, nullptr) << "The key layer in the attention block is null pointer.";
  STATUS_CHECK(key_layer->forward(rmsnorm_output, key));

  // key norm
  auto key_norm = qwen_layers_->rmsnorm_layers_.at(layer_idx + 3 * config_->layer_num_ + 1);
  key.reshape({(int32_t)key.size() / config_->head_size_, config_->head_size_});
  key_norm->forward(key, key);
  key.reshape({(int32_t)key.size()});

  // value
  const auto& value_layer = qwen_layers_->wv_layers_.at(layer_idx);
  CHECK_NE(value_layer, nullptr) << "The value layer in the attention block is null pointer.";
  STATUS_CHECK(value_layer->forward(rmsnorm_output, val));

  // rope
  CHECK_NE(qwen_layers_->rope_layer_, nullptr)
      << "The RoPE layer in the attention block is null pointer.";
  STATUS_CHECK(qwen_layers_->rope_layer_->forward(
      query, key, pos_tensor, get_buffer(ModelBufferType::kSinCache),
      get_buffer(ModelBufferType::kCosCache), tensor::Tensor{}));
}

base::Status Qwen3Model::predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                                 bool is_prompt, int& next) const {
  auto status = forward(input, pos_tensor, next);
  if (!status) {
    return status;
  }
  next = post_processing(pos_tensor, is_prompt);
  return base::error::Success();
}

void Qwen3Model::attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  CHECK(qwen_layers_ != nullptr);
  // mha
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);

  tensor::Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);
  tensor::Tensor query = get_buffer(ModelBufferType::kQuery);
  
  int pos = pos_tensor.index<int32_t>(0);

  // FP16 data always uses Flash Attention (MHA does not support FP16)
  if (query.data_type() == base::DataType::kDataTypeFp16 &&
      key_cache.data_type() == base::DataType::kDataTypeFp16) {
    // Use Flash Attention FP16 for decode (FA1 or FA2 based on layer's attention_type_)
    auto flash_attn = qwen_layers_->flash_attention_decode_layer_;
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
    const auto& mha_layer = qwen_layers_->mha_layer_;
    CHECK_NE(mha_layer, nullptr) << "The multi head attention layer is null pointer.";
    std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_pos(pos);
    std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_layer_idx(layer_idx);
    STATUS_CHECK(mha_layer->forward(query, score_storage, key_cache, val_cache, mha_output));
  } else {
    // FP32 Flash Attention path (FA1 or FA2)
    auto flash_attn = qwen_layers_->flash_attention_decode_layer_;
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
  const auto& wo_layer = qwen_layers_->wo_layers_.at(layer_idx);
  CHECK_NE(wo_layer, nullptr) << "The weight output layer is null pointer.";
  STATUS_CHECK(wo_layer->forward(mha_output, attn_output));
}

void Qwen3Model::feed_forward(int32_t layer_idx, const tensor::Tensor& input) const {
  CHECK(qwen_layers_ != nullptr);

  // residual add
  CHECK_NE(qwen_layers_->add_layer_, nullptr)
      << "The add layer in the feedforward block is null pointer";
  STATUS_CHECK(
      qwen_layers_->add_layer_->forward(input, get_buffer(ModelBufferType::kAttnOutput), input));

  // ffn rmsnorm (post attention layernorm)
  tensor::Tensor ffn_norm_output = get_buffer(ModelBufferType::kFFNRMSNorm);
  const auto& ffn_rmsnorm = qwen_layers_->rmsnorm_layers_.at(layer_idx + config_->layer_num_);
  CHECK_NE(ffn_rmsnorm, nullptr)
      << "The final rmsnorm layer in the feedforward block is null pointer";
  STATUS_CHECK(ffn_rmsnorm->forward(input, ffn_norm_output));

  // w1
  tensor::Tensor w1_output = get_buffer(ModelBufferType::kW1Output);
  const auto& w1_layer = qwen_layers_->w1_layers_.at(layer_idx);
  CHECK_NE(w1_layer, nullptr) << "The w1 layer in the feedforward block is null pointer";
  STATUS_CHECK(w1_layer->forward(ffn_norm_output, w1_output));

  // w3
  tensor::Tensor w3_ouput = get_buffer(ModelBufferType::kW3Output);
  const auto& w3_layer = qwen_layers_->w3_layers_.at(layer_idx);
  CHECK_NE(w3_layer, nullptr) << "The w3 layer in the feedforward block is null pointer";
  STATUS_CHECK(w3_layer->forward(ffn_norm_output, w3_ouput));

  // SwiGLU
  CHECK_NE(qwen_layers_->swiglu_layer_, nullptr)
      << "The swiglu layer in the feedforward block is null pointer";
  STATUS_CHECK(qwen_layers_->swiglu_layer_->forward(w1_output, w3_ouput, w1_output));

  // w2
  tensor::Tensor w2_output = get_buffer(ModelBufferType::kW2Output);
  const auto& w2_layer = qwen_layers_->w2_layers_.at(layer_idx);
  CHECK_NE(w2_layer, nullptr) << "The w2 layer in the feedforward block is null pointer";
  STATUS_CHECK(w2_layer->forward(w1_output, w2_output));

  // residual add
  CHECK_NE(qwen_layers_->add_layer_, nullptr)
      << "The add layer in the feedforward block is null pointer";
  STATUS_CHECK(qwen_layers_->add_layer_->forward(input, w2_output, input));
}

op::EmbeddingOutput Qwen3Model::embedding(const std::vector<int>& tokens) const {
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
  LOG_IF(FATAL, !qwen_layers_->embedding_layer_)
      << "The embedding layer in the llama2 model is null pointer.";
  STATUS_CHECK(
      qwen_layers_->embedding_layer_->forward(input_tokens, input_token_num, input_embeddings));

  op::EmbeddingOutput output(input_tokens, input_embeddings, input_token_num);
  return output;
}

void Qwen3Model::cls_logits(const tensor::Tensor& input) const {
  CHECK(qwen_layers_ != nullptr);
  const auto& norm = qwen_layers_->rmsnorm_layers_.at(2 * config_->layer_num_);
  CHECK_NE(norm, nullptr);
  STATUS_CHECK(norm->forward(input, input));

  tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
  CHECK_NE(qwen_layers_->cls_layer_, nullptr);
  STATUS_CHECK(qwen_layers_->cls_layer_->forward(input, forward_output));
}

int32_t Qwen3Model::post_processing(const tensor::Tensor& pos, bool is_prompt) const {
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

// ==================== CUDA Graph Optimized Methods ====================

void Qwen3Model::attention_qkv_with_graph(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  CHECK(qwen_layers_ != nullptr);
  CHECK(cuda_config_ != nullptr);
  
  // Use fixed-address temporary buffers for CUDA Graph compatibility
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
  tensor::Tensor temp_key = this->get_buffer(ModelBufferType::kTempKey);
  tensor::Tensor temp_value = this->get_buffer(ModelBufferType::kTempValue);
  
  auto rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  
  // query
  const auto& query_layer = qwen_layers_->wq_layers_.at(layer_idx);
  CHECK_NE(query_layer, nullptr);
  STATUS_CHECK(query_layer->forward(rmsnorm_output, query));

  // Query norm (Qwen3 specific)
  auto query_norm = qwen_layers_->rmsnorm_layers_.at(layer_idx + 2 * config_->layer_num_ + 1);
  query.reshape({(int32_t)query.size() / config_->head_size_, config_->head_size_});
  query_norm->forward(query, query);
  query.reshape({(int32_t)query.size()});

  // key -> temp_key (fixed address)
  const auto& key_layer = qwen_layers_->wk_layers_.at(layer_idx);
  CHECK_NE(key_layer, nullptr);
  STATUS_CHECK(key_layer->forward(rmsnorm_output, temp_key));
  
  // Key norm (Qwen3 specific)
  auto key_norm = qwen_layers_->rmsnorm_layers_.at(layer_idx + 3 * config_->layer_num_ + 1);
  temp_key.reshape({(int32_t)temp_key.size() / config_->head_size_, config_->head_size_});
  key_norm->forward(temp_key, temp_key);
  temp_key.reshape({(int32_t)temp_key.size()});
  
  // value -> temp_value (fixed address)
  const auto& value_layer = qwen_layers_->wv_layers_.at(layer_idx);
  CHECK_NE(value_layer, nullptr);
  STATUS_CHECK(value_layer->forward(rmsnorm_output, temp_value));

  // RoPE - use GPU pos version for CUDA Graph compatibility via layer abstraction
  auto rope_layer = qwen_layers_->rope_gpu_pos_layer_;
  rope_layer->set_use_gpu_pos(true);
  rope_layer->set_use_fp16(query.data_type() == base::DataType::kDataTypeFp16);
  rope_layer->set_input(0, query);
  rope_layer->set_input(1, temp_key);
  rope_layer->set_input(2, pos_tensor);
  rope_layer->set_input(3, get_buffer(ModelBufferType::kSinCache));
  rope_layer->set_input(4, get_buffer(ModelBufferType::kCosCache));
  rope_layer->set_cuda_config(cuda_config_);
  STATUS_CHECK(rope_layer->forward());
  
  // Copy temp_key and temp_value to KV cache at correct position via layer
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);
  
  // Key cache copy
  auto key_cache_layer = qwen_layers_->kv_cache_key_layer_;
  key_cache_layer->set_layer_index(layer_idx);
  key_cache_layer->set_use_gpu_pos(true);
  key_cache_layer->set_use_fp16(key_cache.data_type() == base::DataType::kDataTypeFp16);
  key_cache_layer->set_input(0, temp_key);
  key_cache_layer->set_input(1, key_cache);
  key_cache_layer->set_input(2, pos_tensor);
  key_cache_layer->set_cuda_config(cuda_config_);
  STATUS_CHECK(key_cache_layer->forward());
  
  // Value cache copy
  auto value_cache_layer = qwen_layers_->kv_cache_value_layer_;
  value_cache_layer->set_layer_index(layer_idx);
  value_cache_layer->set_use_gpu_pos(true);
  value_cache_layer->set_use_fp16(val_cache.data_type() == base::DataType::kDataTypeFp16);
  value_cache_layer->set_input(0, temp_value);
  value_cache_layer->set_input(1, val_cache);
  value_cache_layer->set_input(2, pos_tensor);
  value_cache_layer->set_cuda_config(cuda_config_);
  STATUS_CHECK(value_cache_layer->forward());
}

void Qwen3Model::attention_mha_with_graph(int32_t layer_idx, const tensor::Tensor& pos_tensor_gpu) const {
  CHECK(qwen_layers_ != nullptr);
  CHECK(cuda_config_ != nullptr);
  
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);
  tensor::Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
  
  // FP16 data always uses Flash Attention (MHA does not support FP16)
  if (query.data_type() == base::DataType::kDataTypeFp16 &&
      key_cache.data_type() == base::DataType::kDataTypeFp16) {
    // Use GPU pos version for CUDA Graph compatibility with FP16 via layer
    auto flash_attn = qwen_layers_->flash_attention_decode_layer_;
    flash_attn->set_layer_index(layer_idx);
    flash_attn->set_use_gpu_pos(true);
    flash_attn->set_input(0, query);
    flash_attn->set_input(1, mha_output);
    flash_attn->set_input(2, key_cache);
    flash_attn->set_input(3, val_cache);
    flash_attn->set_input(4, pos_tensor_gpu);  // GPU position tensor
    flash_attn->set_cuda_config(cuda_config_);
    STATUS_CHECK(flash_attn->forward());
  } else if (attention_type_ == base::AttentionType::kAttentionMHA) {
    // Standard FP32 MHA path with GPU pos
    tensor::Tensor score_storage = get_buffer(ModelBufferType::kScoreStorage);
    STATUS_CHECK(qwen_layers_->mha_gpu_pos_layer_->forward(
        pos_tensor_gpu.ptr<int32_t>(),
        config_->head_num_,
        layer_idx,
        config_->seq_len_,
        config_->kv_dim_,
        config_->kv_mul_,
        config_->head_size_,
        mha_output,
        query,
        score_storage,
        key_cache,
        val_cache));
  } else {
    // FP32 + FA1/FA2: no GPU-pos flash attention kernel for FP32, fall back to MHA with GPU pos
    tensor::Tensor score_storage = get_buffer(ModelBufferType::kScoreStorage);
    STATUS_CHECK(qwen_layers_->mha_gpu_pos_layer_->forward(
        pos_tensor_gpu.ptr<int32_t>(),
        config_->head_num_,
        layer_idx,
        config_->seq_len_,
        config_->kv_dim_,
        config_->kv_mul_,
        config_->head_size_,
        mha_output,
        query,
        score_storage,
        key_cache,
        val_cache));
  }

  // wo @ attention output
  tensor::Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
  const auto& wo_layer = qwen_layers_->wo_layers_.at(layer_idx);
  CHECK_NE(wo_layer, nullptr);
  STATUS_CHECK(wo_layer->forward(mha_output, attn_output));
}

// ==================== Fused FFN Implementation ====================

void Qwen3Model::feed_forward_fused(int32_t layer_idx, const tensor::Tensor& input) const {
  CHECK(qwen_layers_ != nullptr);
  
  // residual add from attention output
  CHECK_NE(qwen_layers_->add_layer_, nullptr);
  STATUS_CHECK(
      qwen_layers_->add_layer_->forward(input, get_buffer(ModelBufferType::kAttnOutput), input));

  // ffn rmsnorm
  tensor::Tensor ffn_norm_output = get_buffer(ModelBufferType::kFFNRMSNorm);
  const auto& ffn_rmsnorm = qwen_layers_->rmsnorm_layers_.at(layer_idx + config_->layer_num_);
  CHECK_NE(ffn_rmsnorm, nullptr);
  STATUS_CHECK(ffn_rmsnorm->forward(input, ffn_norm_output));

  // Fused W1 + W3 + SwiGLU kernel
  tensor::Tensor w1_output = get_buffer(ModelBufferType::kW1Output);
  const auto& w1_layer = qwen_layers_->w1_layers_.at(layer_idx);
  const auto& w3_layer = qwen_layers_->w3_layers_.at(layer_idx);
  CHECK_NE(w1_layer, nullptr);
  CHECK_NE(w3_layer, nullptr);
  
  // Check if AWQ layers
  auto w1_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(w1_layer);
  auto w3_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(w3_layer);
  
  if (w1_awq || w3_awq) {
    // AWQ path: fall back to standard forward since fused kernel doesn't support AWQ
    tensor::Tensor w3_output = get_buffer(ModelBufferType::kW3Output);
    STATUS_CHECK(w1_layer->forward(ffn_norm_output, w1_output));
    STATUS_CHECK(w3_layer->forward(ffn_norm_output, w3_output));
        
    // SwiGLU
    CHECK_NE(qwen_layers_->swiglu_layer_, nullptr);
    
    STATUS_CHECK(qwen_layers_->swiglu_layer_->forward(w1_output, w3_output, w1_output));
  } else {
    // Standard path with fused FFN kernel via layer abstraction
    auto w1_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(w1_layer);
    auto w3_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(w3_layer);
    
    CHECK_NE(w1_matmul, nullptr) << "W1 layer is not a MatmulLayer";
    CHECK_NE(w3_matmul, nullptr) << "W3 layer is not a MatmulLayer";
    
    // Check input and weight data types
    const auto& w1_weight = w1_matmul->get_weight(0);
    const auto& w3_weight = w3_matmul->get_weight(0);
    
    // Use fused FFN layer
    auto fused_ffn = qwen_layers_->fused_ffn_layer_;
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
  const auto& w2_layer = qwen_layers_->w2_layers_.at(layer_idx);
  CHECK_NE(w2_layer, nullptr);
  
  STATUS_CHECK(w2_layer->forward(w1_output, w2_output));

  // residual add
  CHECK_NE(qwen_layers_->add_layer_, nullptr);
  STATUS_CHECK(qwen_layers_->add_layer_->forward(input, w2_output, input));
}

// ==================== Batched Prefill Implementation ====================

void Qwen3Model::batched_attention_rms(int32_t layer_idx, const tensor::Tensor& input, 
                                       int32_t seq_len) const {
  CHECK(qwen_layers_ != nullptr);
  std::shared_ptr<op::Layer> rmsnorm_layer = qwen_layers_->rmsnorm_layers_.at(layer_idx);
  if (!rmsnorm_layer) {
    LOG(FATAL) << "The attention rmsnorm layer is a null pointer";
  }
  
  // Use layer forward call instead of direct kernel call (in-place: input -> input)
  STATUS_CHECK(rmsnorm_layer->forward(input, input));
}

// Optimized version with separate input/output buffers (no copy needed)
void Qwen3Model::batched_attention_rms(int32_t layer_idx, const tensor::Tensor& input,
                                       const tensor::Tensor& output, int32_t seq_len) const {
  CHECK(qwen_layers_ != nullptr);
  std::shared_ptr<op::Layer> rmsnorm_layer = qwen_layers_->rmsnorm_layers_.at(layer_idx);
  if (!rmsnorm_layer) {
    LOG(FATAL) << "The attention rmsnorm layer is a null pointer";
  }
  
  // Use layer forward call instead of direct kernel call (input -> output)
  STATUS_CHECK(rmsnorm_layer->forward(input, output));
}

void Qwen3Model::batched_attention_qkv(int32_t layer_idx, const tensor::Tensor& rms_out,
                                       const tensor::Tensor& query_out, const tensor::Tensor& key_out, 
                                       const tensor::Tensor& value_out,
                                       int32_t seq_len, int32_t start_pos) const {
  CHECK(qwen_layers_ != nullptr);
  
  const auto& query_layer = qwen_layers_->wq_layers_.at(layer_idx);
  const auto& key_layer = qwen_layers_->wk_layers_.at(layer_idx);
  const auto& value_layer = qwen_layers_->wv_layers_.at(layer_idx);
  
  CHECK_NE(query_layer, nullptr);
  CHECK_NE(key_layer, nullptr);
  CHECK_NE(value_layer, nullptr);
  
  // Check if this is AWQ layer
  auto query_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(query_layer);
  auto key_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(key_layer);
  auto value_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(value_layer);
  
  if (query_awq && key_awq && value_awq) {
    // AWQ path: use layer forward which handles dispatch internally
    STATUS_CHECK(query_awq->forward(rms_out, query_out));
    STATUS_CHECK(key_awq->forward(rms_out, key_out));
    STATUS_CHECK(value_awq->forward(rms_out, value_out));
  } else {
    // Standard FP16/FP32 path
    auto query_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(query_layer);
    auto key_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(key_layer);
    auto value_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(value_layer);
    
    CHECK_NE(query_matmul, nullptr) << "Query layer is not a MatmulLayer";
    CHECK_NE(key_matmul, nullptr) << "Key layer is not a MatmulLayer";
    CHECK_NE(value_matmul, nullptr) << "Value layer is not a MatmulLayer";
    
    // Batched matmul
    STATUS_CHECK(qwen_layers_->batched_matmul_helper_layer_->forward(
        rms_out, query_matmul->get_weight(0), query_out, seq_len, 1.f));
    
    STATUS_CHECK(qwen_layers_->batched_matmul_helper_layer_->forward(
        rms_out, key_matmul->get_weight(0), key_out, seq_len, 1.f));
    
    STATUS_CHECK(qwen_layers_->batched_matmul_helper_layer_->forward(
        rms_out, value_matmul->get_weight(0), value_out, seq_len, 1.f));
  }
  
  // Apply Q/K norms for Qwen3 (per-head normalization)
  auto q_norm = qwen_layers_->rmsnorm_layers_.at(layer_idx + 2 * config_->layer_num_ + 1);
  auto k_norm = qwen_layers_->rmsnorm_layers_.at(layer_idx + 3 * config_->layer_num_ + 1);
  
  // Determine data type and element size dynamically
  base::DataType activation_dtype = query_out.data_type();
  size_t elem_size = (activation_dtype == base::DataType::kDataTypeFp16) 
      ? sizeof(uint16_t) : sizeof(float);
  
  // Create reshaped tensor views for per-head normalization
  // Query: original shape [seq_len, dim] -> view as [seq_len * head_num, head_size]
  // Key: original shape [seq_len, kv_dim] -> view as [seq_len * kv_head_num, head_size]
  auto q_buffer = std::make_shared<base::Buffer>(
      seq_len * config_->dim_ * elem_size, nullptr,
      const_cast<void*>(query_out.get_buffer()->ptr()), true);
  tensor::Tensor q_reshaped(activation_dtype, 
                            seq_len * config_->head_num_, config_->head_size_, 
                            false, nullptr, nullptr);
  q_reshaped.assign(q_buffer);
  q_reshaped.set_device_type(base::DeviceType::kDeviceCUDA);
  
  auto k_buffer = std::make_shared<base::Buffer>(
      seq_len * config_->kv_dim_ * elem_size, nullptr,
      const_cast<void*>(key_out.get_buffer()->ptr()), true);
  tensor::Tensor k_reshaped(activation_dtype, 
                            seq_len * config_->kv_head_num_, config_->head_size_, 
                            false, nullptr, nullptr);
  k_reshaped.assign(k_buffer);
  k_reshaped.set_device_type(base::DeviceType::kDeviceCUDA);
  
  // RMSNorm on reshaped query/key (per-head normalization) - use layer forward
  STATUS_CHECK(q_norm->forward(q_reshaped, q_reshaped));
  STATUS_CHECK(k_norm->forward(k_reshaped, k_reshaped));
  
  // Apply batched RoPE via layer abstraction
  auto batched_rope = qwen_layers_->batched_rope_layer_;
  batched_rope->set_seq_len(seq_len);
  batched_rope->set_start_pos(start_pos);
  batched_rope->set_input(0, query_out);
  batched_rope->set_input(1, key_out);
  batched_rope->set_input(2, get_buffer(ModelBufferType::kSinCache));
  batched_rope->set_input(3, get_buffer(ModelBufferType::kCosCache));
  batched_rope->set_cuda_config(cuda_config_);
  STATUS_CHECK(batched_rope->forward());
  
  // Copy to KV cache using cudaMemcpyAsync
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);
  
  int32_t layer_offset = layer_idx * config_->seq_len_ * config_->kv_dim_;
  
  // Reuse elem_size from above (already determined based on activation_dtype)
  
  // Copy value to cache first (before key, since value doesn't depend on RoPE)
  for (int i = 0; i < seq_len; ++i) {
    int32_t cache_offset = layer_offset + (start_pos + i) * config_->kv_dim_;
    void* dst = const_cast<void*>(val_cache.get_buffer()->ptr()) + cache_offset * elem_size;
    const void* src = value_out.get_buffer()->ptr() + i * config_->kv_dim_ * elem_size;
    cudaMemcpyAsync(dst, src,
                    config_->kv_dim_ * elem_size,
                    cudaMemcpyDeviceToDevice, cuda_config_->stream);
  }
  
  // Copy RoPE'd keys to cache
  for (int i = 0; i < seq_len; ++i) {
    int32_t cache_offset = layer_offset + (start_pos + i) * config_->kv_dim_;
    void* dst = const_cast<void*>(key_cache.get_buffer()->ptr()) + cache_offset * elem_size;
    const void* src = key_out.get_buffer()->ptr() + i * config_->kv_dim_ * elem_size;
    cudaMemcpyAsync(dst, src,
                    config_->kv_dim_ * elem_size,
                    cudaMemcpyDeviceToDevice, cuda_config_->stream);
  }
}

void Qwen3Model::batched_attention_mha(int32_t layer_idx, const tensor::Tensor& query,
                                       const tensor::Tensor& mha_out, int32_t seq_len, int32_t start_pos) const {
  CHECK(qwen_layers_ != nullptr);
  
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);
  
  if (attention_type_ == base::AttentionType::kAttentionMHA &&
      query.data_type() != base::DataType::kDataTypeFp16) {
    // Standard batched MHA attention (FP32 only)
    std::shared_ptr<base::DeviceAllocator> score_alloc = base::CUDADeviceAllocatorFactory::get_instance();
    tensor::Tensor attn_scores(base::DataType::kDataTypeFp32, 
                               seq_len, config_->head_num_, config_->seq_len_, true, score_alloc);
    STATUS_CHECK(qwen_layers_->batched_mha_layer_->forward(
        start_pos, seq_len,
        config_->head_num_, layer_idx, config_->seq_len_,
        config_->dim_, config_->kv_dim_, config_->kv_mul_, config_->head_size_,
        const_cast<tensor::Tensor&>(mha_out), query, attn_scores, key_cache, val_cache));
  } else {
    // Use Flash Attention prefill via layer abstraction (FA1 or FA2)
    auto prefill_layer = qwen_layers_->flash_attention_prefill_layer_;
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

  // Batched wo - use dynamic data type based on mha_out
  base::DataType activation_dtype = mha_out.data_type();
  size_t elem_size = (activation_dtype == base::DataType::kDataTypeFp16) 
      ? sizeof(uint16_t) : sizeof(float);
  
  const auto& wo_layer = qwen_layers_->wo_layers_.at(layer_idx);
  
  std::shared_ptr<base::DeviceAllocator> alloc = base::CUDADeviceAllocatorFactory::get_instance();
  tensor::Tensor wo_out(activation_dtype, seq_len, config_->dim_, true, alloc);
  
  // Check if AWQ layer
  auto wo_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(wo_layer);
  
  if (wo_awq) {
    // AWQ path: use layer forward which handles dispatch internally
    STATUS_CHECK(wo_awq->forward(mha_out, wo_out));
  } else {
    // Standard FP16/FP32 path
    auto wo_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(wo_layer);
    CHECK_NE(wo_matmul, nullptr) << "WO layer is not a MatmulLayer";
    STATUS_CHECK(qwen_layers_->batched_matmul_helper_layer_->forward(
        mha_out, wo_matmul->get_weight(0), wo_out, seq_len, 1.f));
  }
  
  // Copy back to mha_out (will be used for residual add)
  cudaMemcpyAsync(const_cast<void*>(mha_out.get_buffer()->ptr()), 
                  wo_out.get_buffer()->ptr(),
                  seq_len * config_->dim_ * elem_size, cudaMemcpyDeviceToDevice,
                  cuda_config_->stream);
}

void Qwen3Model::batched_feed_forward(int32_t layer_idx, const tensor::Tensor& input, int32_t seq_len) const {
  CHECK(qwen_layers_ != nullptr);
  // Note: Attention residual add is done in prefill() before calling this function
  
  // Use dynamic data type based on input
  base::DataType activation_dtype = input.data_type();
  
  // FFN RMSNorm - use layer forward call
  const auto& ffn_rmsnorm = qwen_layers_->rmsnorm_layers_.at(layer_idx + config_->layer_num_);
  
  std::shared_ptr<base::DeviceAllocator> alloc = base::CUDADeviceAllocatorFactory::get_instance();
  tensor::Tensor ffn_norm_out(activation_dtype, seq_len, config_->dim_, true, alloc);
  
  STATUS_CHECK(ffn_rmsnorm->forward(input, ffn_norm_out));
  
  // Batched W1 and W3
  const auto& w1_layer = qwen_layers_->w1_layers_.at(layer_idx);
  const auto& w3_layer = qwen_layers_->w3_layers_.at(layer_idx);
  
  int32_t hidden_dim = config_->hidden_dim_;
  tensor::Tensor w1_out(activation_dtype, seq_len, hidden_dim, true, alloc);
  tensor::Tensor w3_out(activation_dtype, seq_len, hidden_dim, true, alloc);
  
  // Check if this is AWQ layer
  auto w1_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(w1_layer);
  auto w3_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(w3_layer);
  
  if (w1_awq && w3_awq) {
    // AWQ path: use layer forward which handles dispatch internally
    STATUS_CHECK(w1_awq->forward(ffn_norm_out, w1_out));
    STATUS_CHECK(w3_awq->forward(ffn_norm_out, w3_out));
  } else {
    // Standard FP16/FP32 path
    auto w1_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(w1_layer);
    auto w3_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(w3_layer);
    
    CHECK_NE(w1_matmul, nullptr) << "W1 layer is not a MatmulLayer";
    CHECK_NE(w3_matmul, nullptr) << "W3 layer is not a MatmulLayer";
    
    STATUS_CHECK(qwen_layers_->batched_matmul_helper_layer_->forward(
        ffn_norm_out, w1_matmul->get_weight(0), w1_out, seq_len, 1.f));
    STATUS_CHECK(qwen_layers_->batched_matmul_helper_layer_->forward(
        ffn_norm_out, w3_matmul->get_weight(0), w3_out, seq_len, 1.f));
  }
  
  // Batched SwiGLU via batched_swiglu_layer_ (element-wise, works on full tensor)
  STATUS_CHECK(qwen_layers_->batched_swiglu_layer_->forward(w1_out, w3_out, w1_out));
  
  // Batched W2
  const auto& w2_layer = qwen_layers_->w2_layers_.at(layer_idx);
  tensor::Tensor w2_out(activation_dtype, seq_len, config_->dim_, true, alloc);
  
  // Check if this is AWQ layer
  auto w2_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(w2_layer);
  
  if (w2_awq) {
    // AWQ path: use layer forward which handles dispatch internally
    STATUS_CHECK(w2_awq->forward(w1_out, w2_out));
  } else {
    // Standard FP16/FP32 path
    auto w2_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(w2_layer);
    CHECK_NE(w2_matmul, nullptr) << "W2 layer is not a MatmulLayer";
    STATUS_CHECK(qwen_layers_->batched_matmul_helper_layer_->forward(
        w1_out, w2_matmul->get_weight(0), w2_out, seq_len, 1.f));
  }
  
  // Residual add via batched_add_layer_ (element-wise, works on full tensor)
  STATUS_CHECK(qwen_layers_->batched_add_layer_->forward(input, w2_out, input));
}

// Optimized version with pre-allocated buffers (avoids per-layer allocation overhead)
void Qwen3Model::batched_feed_forward_optimized(int32_t layer_idx, const tensor::Tensor& input,
                                                tensor::Tensor& ffn_norm_out, tensor::Tensor& w1_out,
                                                tensor::Tensor& w3_out, tensor::Tensor& w2_out,
                                                int32_t seq_len) const {
  CHECK(qwen_layers_ != nullptr);
  
  // FFN RMSNorm - use layer forward call
  const auto& ffn_rmsnorm = qwen_layers_->rmsnorm_layers_.at(layer_idx + config_->layer_num_);
  
  STATUS_CHECK(ffn_rmsnorm->forward(input, ffn_norm_out));
  
  // Batched W1 and W3
  const auto& w1_layer = qwen_layers_->w1_layers_.at(layer_idx);
  const auto& w3_layer = qwen_layers_->w3_layers_.at(layer_idx);
  
  // Check if this is AWQ layer
  auto w1_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(w1_layer);
  auto w3_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(w3_layer);
  
  if (w1_awq && w3_awq) {
    // AWQ path: use layer forward which handles dispatch internally
    STATUS_CHECK(w1_awq->forward(ffn_norm_out, w1_out));
    STATUS_CHECK(w3_awq->forward(ffn_norm_out, w3_out));
  } else {
    // Standard FP16/FP32 path
    auto w1_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(w1_layer);
    auto w3_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(w3_layer);
    
    CHECK_NE(w1_matmul, nullptr) << "W1 layer is not a MatmulLayer";
    CHECK_NE(w3_matmul, nullptr) << "W3 layer is not a MatmulLayer";
    
    STATUS_CHECK(qwen_layers_->batched_matmul_helper_layer_->forward(
        ffn_norm_out, w1_matmul->get_weight(0), w1_out, seq_len, 1.f));
    STATUS_CHECK(qwen_layers_->batched_matmul_helper_layer_->forward(
        ffn_norm_out, w3_matmul->get_weight(0), w3_out, seq_len, 1.f));
  }
  
  // Batched SwiGLU via batched_swiglu_layer_ (element-wise, works on full tensor)
  STATUS_CHECK(qwen_layers_->batched_swiglu_layer_->forward(w1_out, w3_out, w1_out));
  
  // Batched W2
  const auto& w2_layer = qwen_layers_->w2_layers_.at(layer_idx);
  
  // Check if this is AWQ layer
  auto w2_awq = std::dynamic_pointer_cast<op::AWQMatmulLayer>(w2_layer);
  
  if (w2_awq) {
    // AWQ path: use layer forward which handles dispatch internally
    STATUS_CHECK(w2_awq->forward(w1_out, w2_out));
  } else {
    // Standard FP16/FP32 path
    auto w2_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(w2_layer);
    CHECK_NE(w2_matmul, nullptr) << "W2 layer is not a MatmulLayer";
    STATUS_CHECK(qwen_layers_->batched_matmul_helper_layer_->forward(
        w1_out, w2_matmul->get_weight(0), w2_out, seq_len, 1.f));
  }
  
  // Residual add via batched_add_layer_ (element-wise, works on full tensor)
  STATUS_CHECK(qwen_layers_->batched_add_layer_->forward(input, w2_out, input));
}

// ==================== Prefill/Decode Implementation ====================

base::Status Qwen3Model::prefill(const tensor::Tensor& input, int32_t seq_len, int32_t start_pos) const {
  if (input.is_empty()) {
    return base::error::InvalidArgument("The input tensor is empty.");
  }
  if (device_type_ != base::DeviceType::kDeviceCUDA) {
    return base::error::InternalError("Batched prefill only supports CUDA device");
  }
  
  std::shared_ptr<base::DeviceAllocator> alloc = base::CUDADeviceAllocatorFactory::get_instance();
  
  // Use dynamic data type based on input (which follows model type)
  base::DataType activation_dtype = input.data_type();
  size_t elem_size = (activation_dtype == base::DataType::kDataTypeFp16) 
      ? sizeof(uint16_t) : sizeof(float);
  
  int32_t dim = config_->dim_;
  int32_t hidden_dim = config_->hidden_dim_;
  
  // OPTIMIZED: Use double-buffering for hidden states to avoid initialization copy
  // Layer 0: input=input (const, used directly), output=hidden_buf0
  // Layer 1: input=hidden_buf0, output=hidden_buf1
  // Layer 2: input=hidden_buf1, output=hidden_buf0
  // ...
  tensor::Tensor hidden_buf0(activation_dtype, seq_len, dim, true, alloc);
  tensor::Tensor hidden_buf1(activation_dtype, seq_len, dim, true, alloc);
  tensor::Tensor rms_out(activation_dtype, seq_len, dim, true, alloc);
  tensor::Tensor query_out(activation_dtype, seq_len, dim, true, alloc);
  tensor::Tensor key_out(activation_dtype, seq_len, config_->kv_dim_, true, alloc);
  tensor::Tensor value_out(activation_dtype, seq_len, config_->kv_dim_, true, alloc);
  tensor::Tensor mha_out(activation_dtype, seq_len, dim, true, alloc);
  
  // OPTIMIZED: Pre-allocate FFN buffers once (avoids per-layer allocation overhead)
  tensor::Tensor ffn_norm_out(activation_dtype, seq_len, dim, true, alloc);
  tensor::Tensor w1_out(activation_dtype, seq_len, hidden_dim, true, alloc);
  tensor::Tensor w3_out(activation_dtype, seq_len, hidden_dim, true, alloc);
  tensor::Tensor w2_out(activation_dtype, seq_len, dim, true, alloc);
  
  // Double-buffering pointers
  tensor::Tensor* hidden_buffers[2] = {&hidden_buf0, &hidden_buf1};
  
  // Process all layers with batched operations using double-buffering
  // Layer 0: input=input (const, used directly), output=hidden_buf0
  // Layer 1+: alternates between hidden_buf0 and hidden_buf1
  for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
    // Determine input and output buffers for this layer
    const tensor::Tensor* layer_input;
    tensor::Tensor* layer_output;
    
    if (layer_idx == 0) {
      // First layer: use input directly (avoid D2D copy)
      layer_input = &input;
      layer_output = hidden_buffers[0];  // Output to hidden_buf0
    } else {
      // Subsequent layers: alternate between buffers
      layer_input = hidden_buffers[(layer_idx - 1) % 2];
      layer_output = hidden_buffers[layer_idx % 2];
    }
    
    // 1. Batched Attention RMSNorm (using separate input/output to avoid copy)
    batched_attention_rms(layer_idx, *layer_input, rms_out, seq_len);
    
    // 2. Q/K/V projections + RoPE + KV cache update
    batched_attention_qkv(layer_idx, rms_out, query_out, key_out, value_out, 
                          seq_len, start_pos);
    
    // 3. Multi-head attention
    batched_attention_mha(layer_idx, query_out, mha_out, seq_len, start_pos);
    
    // 4. Residual add: layer_output = layer_input + mha_out (via batched_add_layer_)
    STATUS_CHECK(qwen_layers_->batched_add_layer_->forward(*layer_input, mha_out, *layer_output));
    
    // 5. Feed forward with residual (using pre-allocated buffers)
    batched_feed_forward_optimized(layer_idx, *layer_output, ffn_norm_out, 
                                   w1_out, w3_out, w2_out, seq_len);
  }
  
  // Determine which buffer holds the final hidden states
  tensor::Tensor* final_hidden = hidden_buffers[(config_->layer_num_ - 1) % 2];
  
  // Final layer norm and cls_logits on last token
  // Create a view into the last token's hidden state
  void* last_token_ptr = static_cast<char*>(const_cast<void*>(final_hidden->get_buffer()->ptr())) + 
                         (seq_len - 1) * dim * elem_size;
  tensor::Tensor last_hidden(activation_dtype, dim, false, nullptr,
                             last_token_ptr);
  last_hidden.set_device_type(device_type_);
  
  cls_logits(last_hidden);
  
  return base::error::Success();
}

base::Status Qwen3Model::decode(const tensor::Tensor& input, int32_t pos, int& next) const {
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
                    config_->dim_ * decode_elem_size, cudaMemcpyDeviceToDevice, cuda_config_->stream);
    
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
          // LOG(INFO) << "Qwen3 Graph capture successful! Total captures: " << graph_ctx->graph_recaptures;
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
          next = static_cast<int32_t>(*reinterpret_cast<size_t*>(const_cast<int32_t*>(argmax_pinned.ptr<int32_t>())));
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
  
  // Normal execution path
  tensor::Tensor pos_tensor = get_buffer(ModelBufferType::kInputPos);
  pos_tensor.index<int32_t>(0) = pos;
  
  static bool debug_layer_nan = false;
  static bool debug_layer2_detail = false;
  static bool debug_layer3_detail = false;
  
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

void Qwen3Model::clear_kv_cache() {
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  tensor::Tensor value_cache = get_buffer(ModelBufferType::kValueCache);
  
  // Determine element size based on data type
  size_t elem_size = (key_cache.data_type() == base::DataType::kDataTypeFp16) 
      ? sizeof(uint16_t) : sizeof(float);
  
  if (device_type_ == base::DeviceType::kDeviceCUDA && cuda_config_) {
    cudaMemsetAsync(const_cast<void*>(key_cache.get_buffer()->ptr()), 0, 
                    key_cache.size() * elem_size, cuda_config_->stream);
    cudaMemsetAsync(const_cast<void*>(value_cache.get_buffer()->ptr()), 0, 
                    value_cache.size() * elem_size, cuda_config_->stream);
    cudaStreamSynchronize(cuda_config_->stream);
    
    // 重要：使 CUDA Graph 失效，因为 KV cache 状态已改变
    // CUDA Graph 捕获的是特定的 KV cache 状态，清空后需要重新捕获
    invalidate_cuda_graph();
  } else {
    memset(const_cast<void*>(key_cache.get_buffer()->ptr()), 0, key_cache.size() * elem_size);
    memset(const_cast<void*>(value_cache.get_buffer()->ptr()), 0, value_cache.size() * elem_size);
  }
}

void Qwen3Model::set_attention_type(base::AttentionType type) {
  Model::set_attention_type(type);
  if (qwen_layers_) {
    if (qwen_layers_->flash_attention_decode_layer_) {
      qwen_layers_->flash_attention_decode_layer_->set_attention_type(type);
    }
    if (qwen_layers_->flash_attention_prefill_layer_) {
      qwen_layers_->flash_attention_prefill_layer_->set_attention_type(type);
    }
    if (qwen_layers_->flash_attention_decode_gpu_pos_layer_) {
      qwen_layers_->flash_attention_decode_gpu_pos_layer_->set_attention_type(type);
    }
  }
}

}  // namespace model

#endif
