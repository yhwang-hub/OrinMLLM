#include "model/qwen2.h"
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <op/matmul.h>
#include <op/mha.h>
#include <op/rmsnorm.h>
#include <op/embedding.h>
#include <sentencepiece_processor.h>
#include <utility>
#include "../op/kernels/cpu/rope_kernel.h"
#include "../op/kernels/cuda/rope_kernel.cuh"
#include "../op/kernels/cuda/matmul_kernel.cuh"
#include "../op/kernels/cuda/mha_kernel.cuh"
#include "../op/kernels/cuda/rmsnorm_kernel.cuh"
#include "../op/kernels/cuda/add_kernel.cuh"
#include "../op/kernels/cuda/swiglu_kernel.cuh"
#include "../op/kernels/cuda/flash_attention_kernel.cuh"
#include "../op/kernels/cuda/kv_cache_kernel.cuh"
#include "../op/kernels/cuda/fused_ffn_kernel.cuh"  // Fused FFN optimization
#include "../op/kernels/cuda/fp16_convert_kernel.cuh"  // FP16/FP32 conversion
#include "sampler/argmax_sampler.h"
#include "base/tick.h"

// Set to 1 to use Flash Attention, 0 for standard batched attention
#define USE_FLASH_ATTENTION 1

namespace model {

void Qwen2Layers::to_cuda(std::shared_ptr<kernel::CudaConfig> config, bool keep_fp16_weights) {
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
  
  // New layers for unified kernel access
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
  if (bias_add_layer_) {
    bias_add_layer_->set_cuda_config(config);
    bias_add_layer_->to_cuda();
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

Qwen2Model::Qwen2Model(base::TokenizerType tokenizer_type, std::string token_path,
                       std::string model_path, bool is_quant_model)
    : Model(tokenizer_type, base::ModelType::kModelTypeLLama2, std::move(token_path),
            std::move(model_path), is_quant_model) {}

base::Status Qwen2Model::init(base::DeviceType device_type) {
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
    // Enable Tensor Core math mode for better performance (on supported GPUs)
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

base::Status Qwen2Model::forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
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

void Qwen2Model::create_nonparam_layers() {
  CHECK(qwen_layers_ != nullptr);
  qwen_layers_->rope_layer_ = std::make_shared<op::RoPELayer>(
      device_type_, config_->dim_, config_->kv_dim_, config_->head_size_);

  qwen_layers_->mha_layer_ = std::make_shared<op::MultiHeadAttention>(
      device_type_, 0, config_->kv_mul_, config_->kv_dim_, config_->seq_len_, config_->head_num_,
      config_->head_size_);

  qwen_layers_->add_layer_ = std::make_shared<op::VecAddLayer>(device_type_);

  qwen_layers_->swiglu_layer_ =
      std::make_shared<op::SwiGLULayer>(device_type_, config_->hidden_dim_);
  
  // Create new layers for unified kernel access
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
  qwen_layers_->bias_add_layer_ = std::make_shared<op::BiasAddLayer>(device_type_);
  
  // Create misc layers for unified kernel access
  qwen_layers_->sin_cos_cache_layer_ = std::make_shared<op::SinCosCacheLayer>(device_type_);
  qwen_layers_->mha_gpu_pos_layer_ = std::make_shared<op::MHAGpuPosLayer>(device_type_);
  qwen_layers_->batched_mha_layer_ = std::make_shared<op::BatchedMHALayer>(device_type_);
  qwen_layers_->batched_matmul_helper_layer_ = std::make_shared<op::BatchedMatmulHelperLayer>(device_type_);
}

void Qwen2Model::create_param_quant_layers() {
  CHECK(is_quant_model_);
  CHECK(qwen_layers_ != nullptr);

  size_t pos = 0;
  int32_t dim = config_->dim_;
  auto cpu_device_type = base::DeviceType::kDeviceCPU;

  // query
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, true);
    wq->set_group_size(group_size_);
    wq->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->wq_layers_.push_back(wq);
    pos = pos + dim * dim + wq->get_scale_num() * sizeof(float);
  }

  // key
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wk = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim, true);
    wk->set_group_size(group_size_);
    wk->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->wk_layers_.push_back(wk);
    pos = pos + config_->kv_dim_ * dim + wk->get_scale_num() * sizeof(float);
  }

  // value
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wv = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim, true);
    wv->set_group_size(group_size_);
    wv->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->wv_layers_.push_back(wv);
    pos += config_->kv_dim_ * dim + wv->get_scale_num() * sizeof(float);
  }

  // output
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wo = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, true);
    wo->set_group_size(group_size_);
    wo->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->wo_layers_.push_back(wo);
    pos = pos + dim * dim + wo->get_scale_num() * sizeof(float);
  }

  // w1 layers
  int32_t hidden_dim = config_->hidden_dim_;
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w1 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim, true);
    w1->set_group_size(group_size_);
    w1->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->w1_layers_.push_back(w1);
    pos = pos + dim * hidden_dim + w1->get_scale_num() * sizeof(float);
  }

  // w2 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w2 = std::make_shared<op::MatmulLayer>(device_type_, dim, hidden_dim, true);
    w2->set_group_size(group_size_);
    w2->set_weight(0, {dim, hidden_dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->w2_layers_.push_back(w2);
    pos = pos + dim * hidden_dim + w2->get_scale_num() * sizeof(float);
  }

  // w3 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w3 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim, true);
    w3->set_group_size(group_size_);
    w3->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->w3_layers_.push_back(w3);
    pos = pos + dim * hidden_dim + w3->get_scale_num() * sizeof(float);
  }

  // wcls layer
  auto cls_layer = std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_, dim, true);
  cls_layer->set_group_size(group_size_);
  if (config_->is_shared_weight_) {
    // using token embedding weight
    cls_layer->set_weight(0, {config_->vocab_size_, dim}, this->raw_model_data_->weight(pos),
                          cpu_device_type);
  } else {
    // no shared
    cls_layer->set_weight(0, {config_->vocab_size_, dim}, this->raw_model_data_->weight(pos),
                          cpu_device_type);
    pos = pos + config_->vocab_size_ * dim + cls_layer->get_scale_num() * sizeof(float);
  }
  qwen_layers_->cls_layer_ = cls_layer;

  // embedding layer
  float* weight_ptr = (float*)raw_model_data_->weight(pos);
  qwen_layers_->embedding_layer_ = std::make_shared<op::EmbeddingLayer>(
      device_type_, config_->dim_, config_->seq_len_, std::abs(config_->vocab_size_));
  qwen_layers_->embedding_layer_->set_weight(0, {std::abs(config_->vocab_size_), dim}, weight_ptr,
                                             cpu_device_type);
  weight_ptr += config_->vocab_size_ * dim;

  // rmsnorm attention attention,ffn,final
  for (int32_t i = 0; i < 2 * config_->layer_num_ + 1; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, dim);

    rms_norm_layer->set_weight(0, {dim}, weight_ptr, cpu_device_type);
    qwen_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    weight_ptr += dim;
  }
}

void Qwen2Model::create_param_layers() {
  CHECK(!is_quant_model_);
  CHECK(qwen_layers_ != nullptr);
  // The embedding layer
  auto cpu_device_type = base::DeviceType::kDeviceCPU;
  qwen_layers_->embedding_layer_ = std::make_shared<op::EmbeddingLayer>(
      device_type_, config_->dim_, config_->seq_len_, std::abs(config_->vocab_size_));

  const void* weight_embedding = raw_model_data_->weight(0);
  qwen_layers_->embedding_layer_->set_weight(0, {std::abs(config_->vocab_size_), config_->dim_},
                                             weight_embedding, cpu_device_type);

  // create all matmul layer
  int32_t dim = config_->dim_;
  size_t pos = dim * std::abs(config_->vocab_size_) + dim * config_->layer_num_;
  // create weight matrix for query
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, false, true);
    wq->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    pos += dim * dim;
    wq->set_bias(0, dim, this->raw_model_data_->weight(pos), cpu_device_type);
    pos += dim;
    qwen_layers_->wq_layers_.push_back(wq);
  }

  // create weight matrix for key
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wk = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim, false, true);
    wk->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    pos += config_->kv_dim_ * dim;
    wk->set_bias(0, config_->kv_dim_, this->raw_model_data_->weight(pos), cpu_device_type);
    pos += config_->kv_dim_;
    qwen_layers_->wk_layers_.push_back(wk);
  }

  // create weight matrix for value
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wv = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim, false, true);
    wv->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    pos += config_->kv_dim_ * dim;
    wv->set_bias(0, config_->kv_dim_, this->raw_model_data_->weight(pos), cpu_device_type);
    pos += config_->kv_dim_;
    qwen_layers_->wv_layers_.push_back(wv);
  }

  // create weight matrix for output
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wo = std::make_shared<op::MatmulLayer>(device_type_, dim, dim);
    wo->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->wo_layers_.push_back(wo);
    pos += dim * dim;
  }

  // skip ffn rmsnorm
  pos += config_->layer_num_ * dim;

  // w1 layers
  int32_t hidden_dim = config_->hidden_dim_;
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w1 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim);
    w1->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->w1_layers_.push_back(w1);
    pos += dim * hidden_dim;
  }

  // w2 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w2 = std::make_shared<op::MatmulLayer>(device_type_, dim, hidden_dim);
    w2->set_weight(0, {dim, hidden_dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->w2_layers_.push_back(w2);
    pos += dim * hidden_dim;
  }

  // w3 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w3 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim);
    w3->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->w3_layers_.push_back(w3);
    pos += dim * hidden_dim;
  }

  // skip final rms weight
  pos += dim;
  // skip freqs_cos and freqs_sin weight
  // Use original_seq_len_ for correct weight offset (seq_len_ may be limited for memory)
  pos += config_->original_seq_len_ * config_->head_size_;

  qwen_layers_->cls_layer_ =
      std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_, dim);
  if (config_->is_shared_weight_) {
    // using token embedding weight
    qwen_layers_->cls_layer_->set_weight(0, {config_->vocab_size_, dim},
                                         this->raw_model_data_->weight(0), cpu_device_type);
  } else {
    qwen_layers_->cls_layer_->set_weight(0, {config_->vocab_size_, dim},
                                         this->raw_model_data_->weight(pos), cpu_device_type);
  }

  // create rmsnorm layer
  size_t rmsnorm_pos = config_->dim_ * std::abs(config_->vocab_size_);

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, config_->dim_);

    const void* weight_rmsnorm = raw_model_data_->weight(rmsnorm_pos);
    rms_norm_layer->set_weight(0, {config_->dim_}, weight_rmsnorm, cpu_device_type);
    qwen_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    rmsnorm_pos += config_->dim_;
  }

  // skip attention.wq attention.wk attention.wv attention.wo
  rmsnorm_pos += config_->layer_num_ * (config_->dim_ * config_->dim_ + config_->dim_);
  rmsnorm_pos += config_->layer_num_ * (config_->dim_ * config_->kv_dim_ + config_->kv_dim_);
  rmsnorm_pos += config_->layer_num_ * (config_->dim_ * config_->kv_dim_ + config_->kv_dim_);
  rmsnorm_pos += config_->layer_num_ * config_->dim_ * config_->dim_;

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, config_->dim_);
    const void* weight_rmsnorm = raw_model_data_->weight(rmsnorm_pos);
    rms_norm_layer->set_weight(0, {config_->dim_}, weight_rmsnorm, cpu_device_type);
    qwen_layers_->rmsnorm_layers_.push_back(rms_norm_layer);

    rmsnorm_pos += config_->dim_;
  }

  // skip ffn.w1 ffn.w2 ffn.w3
  rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;
  rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;
  rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;

  std::shared_ptr<op::RmsNormLayer> rms_final_layer =
      std::make_shared<op::RmsNormLayer>(device_type_, config_->dim_);

  const void* weight_rmsnorm_final = raw_model_data_->weight(rmsnorm_pos);
  rms_final_layer->set_weight(0, {config_->dim_}, weight_rmsnorm_final, cpu_device_type);
  qwen_layers_->rmsnorm_layers_.push_back(rms_final_layer);
}

void Qwen2Model::create_param_layers_fp16() {
  CHECK(is_fp16_model_);
  CHECK(qwen_layers_ != nullptr);
  LOG(INFO) << "Creating FP16 parameter layers...";
  
  auto cpu_device_type = base::DeviceType::kDeviceCPU;
  int32_t dim = config_->dim_;
  int32_t hidden_dim = config_->hidden_dim_;
  int32_t kv_dim = config_->kv_dim_;
  int32_t vocab_size = std::abs(config_->vocab_size_);
  int32_t layer_num = config_->layer_num_;
  
  /**
   * FP16 model file layout (after 256 byte header):
   * 1. attention_norm weights for all layers [layer_num * dim]
   * 2. ffn_norm weights for all layers [layer_num * dim]
   * 3. final_norm weight [dim]
   * 4. token_embeddings [vocab_size * dim]
   * 5. wq weights for all layers [layer_num * dim * dim]
   * 6. wk weights for all layers [layer_num * kv_dim * dim]
   * 7. wv weights for all layers [layer_num * kv_dim * dim]
   * 8. wo weights for all layers [layer_num * dim * dim]
   * 9. w1 (gate) weights for all layers [layer_num * hidden_dim * dim]
   * 10. w2 (down) weights for all layers [layer_num * dim * hidden_dim]
   * 11. w3 (up) weights for all layers [layer_num * hidden_dim * dim]
   * 12. output weights if not shared [vocab_size * dim]
   * 13. wq biases for all layers [layer_num * dim]
   * 14. wk biases for all layers [layer_num * kv_dim]
   * 15. wv biases for all layers [layer_num * kv_dim]
   * 
   * Note: pos is in terms of FP16 ELEMENTS (not bytes)
   */
  
  size_t pos = 0;  // Element offset (not byte offset!)
  
  // 1. attention_norm weights [layer_num * dim]
  for (int32_t i = 0; i < layer_num; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, dim);
    rms_norm_layer->set_weight_fp16(0, {dim}, raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    pos += dim;
  }
  
  // 2. ffn_norm weights [layer_num * dim]
  for (int32_t i = 0; i < layer_num; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, dim);
    rms_norm_layer->set_weight_fp16(0, {dim}, raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    pos += dim;
  }
  
  // 3. final_norm weight [dim]
  std::shared_ptr<op::RmsNormLayer> rms_final_layer =
      std::make_shared<op::RmsNormLayer>(device_type_, dim);
  rms_final_layer->set_weight_fp16(0, {dim}, raw_model_data_->weight(pos), cpu_device_type);
  qwen_layers_->rmsnorm_layers_.push_back(rms_final_layer);
  pos += dim;
  
  // 4. token_embeddings [vocab_size * dim]
  auto embedding_layer = std::make_shared<op::EmbeddingLayer>(
      device_type_, dim, config_->seq_len_, vocab_size);
  qwen_layers_->embedding_layer_ = embedding_layer;
  embedding_layer->set_weight_fp16(0, {vocab_size, dim}, raw_model_data_->weight(pos), cpu_device_type);
  size_t embedding_pos = pos;  // Save for shared classifier
  pos += static_cast<size_t>(vocab_size) * dim;
  
  // 5. wq weights for all layers [layer_num * dim * dim]
  for (int32_t i = 0; i < layer_num; ++i) {
    auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, false, true);
    wq->set_weight_fp16(0, {dim, dim}, raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->wq_layers_.push_back(wq);
    pos += static_cast<size_t>(dim) * dim;
  }
  
  // 6. wk weights for all layers [layer_num * kv_dim * dim]
  for (int32_t i = 0; i < layer_num; ++i) {
    auto wk = std::make_shared<op::MatmulLayer>(device_type_, kv_dim, dim, false, true);
    wk->set_weight_fp16(0, {kv_dim, dim}, raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->wk_layers_.push_back(wk);
    pos += static_cast<size_t>(kv_dim) * dim;
  }
  
  // 7. wv weights for all layers [layer_num * kv_dim * dim]
  for (int32_t i = 0; i < layer_num; ++i) {
    auto wv = std::make_shared<op::MatmulLayer>(device_type_, kv_dim, dim, false, true);
    wv->set_weight_fp16(0, {kv_dim, dim}, raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->wv_layers_.push_back(wv);
    pos += static_cast<size_t>(kv_dim) * dim;
  }
  
  // 8. wo weights for all layers [layer_num * dim * dim]
  for (int32_t i = 0; i < layer_num; ++i) {
    auto wo = std::make_shared<op::MatmulLayer>(device_type_, dim, dim);
    wo->set_weight_fp16(0, {dim, dim}, raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->wo_layers_.push_back(wo);
    pos += static_cast<size_t>(dim) * dim;
  }
  
  // 9. w1 (gate) weights for all layers [layer_num * hidden_dim * dim]
  for (int32_t i = 0; i < layer_num; ++i) {
    auto w1 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim);
    w1->set_weight_fp16(0, {hidden_dim, dim}, raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->w1_layers_.push_back(w1);
    pos += static_cast<size_t>(hidden_dim) * dim;
  }
  
  // 10. w2 (down) weights for all layers [layer_num * dim * hidden_dim]
  for (int32_t i = 0; i < layer_num; ++i) {
    auto w2 = std::make_shared<op::MatmulLayer>(device_type_, dim, hidden_dim);
    w2->set_weight_fp16(0, {dim, hidden_dim}, raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->w2_layers_.push_back(w2);
    pos += static_cast<size_t>(dim) * hidden_dim;
  }
  
  // 11. w3 (up) weights for all layers [layer_num * hidden_dim * dim]
  for (int32_t i = 0; i < layer_num; ++i) {
    auto w3 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim);
    w3->set_weight_fp16(0, {hidden_dim, dim}, raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->w3_layers_.push_back(w3);
    pos += static_cast<size_t>(hidden_dim) * dim;
  }
  
  // 12. output (cls) weights [vocab_size * dim] - may be shared with embeddings
  auto cls_layer = std::make_shared<op::MatmulLayer>(device_type_, vocab_size, dim);
  qwen_layers_->cls_layer_ = cls_layer;
  if (config_->is_shared_weight_) {
    // Using token embedding weight (shared classifier)
    cls_layer->set_weight_fp16(0, {vocab_size, dim}, raw_model_data_->weight(embedding_pos), cpu_device_type);
  } else {
    cls_layer->set_weight_fp16(0, {vocab_size, dim}, raw_model_data_->weight(pos), cpu_device_type);
    pos += static_cast<size_t>(vocab_size) * dim;
  }
  
  // 13. wq biases for all layers [layer_num * dim]
  for (int32_t i = 0; i < layer_num; ++i) {
    auto& wq = qwen_layers_->wq_layers_[i];
    auto wq_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(wq);
    wq_matmul->set_bias_fp16(0, dim, raw_model_data_->weight(pos), cpu_device_type);
    pos += dim;
  }
  
  // 14. wk biases for all layers [layer_num * kv_dim]
  for (int32_t i = 0; i < layer_num; ++i) {
    auto& wk = qwen_layers_->wk_layers_[i];
    auto wk_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(wk);
    wk_matmul->set_bias_fp16(0, kv_dim, raw_model_data_->weight(pos), cpu_device_type);
    pos += kv_dim;
  }
  
  // 15. wv biases for all layers [layer_num * kv_dim]
  for (int32_t i = 0; i < layer_num; ++i) {
    auto& wv = qwen_layers_->wv_layers_[i];
    auto wv_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(wv);
    wv_matmul->set_bias_fp16(0, kv_dim, raw_model_data_->weight(pos), cpu_device_type);
    pos += kv_dim;
  }
  
  // pos is element count, multiply by 2 for bytes
  LOG(INFO) << "FP16 parameter layers created successfully. Total elements: " << pos 
            << ", Total bytes: " << (pos * sizeof(uint16_t));
}

void Qwen2Model::init_mem() {
  std::shared_ptr<base::DeviceAllocator> alloc;
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    alloc = base::CPUDeviceAllocatorFactory::get_instance();
  } else {
    alloc = base::CUDADeviceAllocatorFactory::get_instance();
  }

  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK_NE(cuda_config_, nullptr);
    // For FP16 models, keep FP16 weights on GPU for pure FP16 compute path
    qwen_layers_->to_cuda(cuda_config_, is_fp16_model_);
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

  tensor::Tensor input_tokens(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  tensor::Tensor input_embeddings(activation_dtype, 1, config_->dim_, true, alloc);
  
  // sin/cos cache should remain FP32 for precision in RoPE computation
  tensor::Tensor sin_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_,
                           true, alloc);
  tensor::Tensor cos_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_,
                           true, alloc);

  CHECK(insert_buffer(ModelBufferType::kSinCache, sin_cache));
  CHECK(insert_buffer(ModelBufferType::kCosCache, cos_cache));

  CHECK(insert_buffer(ModelBufferType::kInputTokens, input_tokens));
  CHECK(insert_buffer(ModelBufferType::kInputEmbeddings, input_embeddings));

  tensor::Tensor rms_output(activation_dtype, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kOutputRMSNorm, rms_output));
  CHECK(insert_buffer(ModelBufferType::kOutputMHA, rms_output));
  CHECK(insert_buffer(ModelBufferType::kW2Output, rms_output));
  CHECK(insert_buffer(ModelBufferType::kFFNRMSNorm, rms_output));

  tensor::Tensor w1_output(activation_dtype, config_->hidden_dim_, true, alloc);
  tensor::Tensor w3_output(activation_dtype, config_->hidden_dim_, true, alloc);

  CHECK(insert_buffer(ModelBufferType::kW1Output, w1_output));
  CHECK(insert_buffer(ModelBufferType::kW3Output, w3_output));

  // kv cache - use FP16 for memory efficiency and bandwidth
  tensor::Tensor key_cache(activation_dtype, config_->layer_num_, config_->seq_len_,
                           config_->kv_dim_, true, alloc);
  tensor::Tensor value_cache(activation_dtype, config_->layer_num_, config_->seq_len_,
                             config_->kv_dim_, true, alloc);

  CHECK(insert_buffer(ModelBufferType::kKeyCache, key_cache));
  CHECK(insert_buffer(ModelBufferType::kValueCache, value_cache));

  // Wq query output
  tensor::Tensor query(activation_dtype, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kQuery, query));

  // Pos tensor - on CPU for normal path
  tensor::Tensor pos_tensor(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  CHECK(insert_buffer(ModelBufferType::kInputPos, pos_tensor));
  
  // Pos tensor on GPU for CUDA Graph path
  tensor::Tensor pos_tensor_gpu(base::DataType::kDataTypeInt32, 1, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kInputPosGPU, pos_tensor_gpu));

  // Temporary K/V buffers with fixed addresses for CUDA Graph optimization
  // These are used in decode phase to ensure memory addresses don't change
  tensor::Tensor temp_key(activation_dtype, config_->kv_dim_, true, alloc);
  tensor::Tensor temp_value(activation_dtype, config_->kv_dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kTempKey, temp_key));
  CHECK(insert_buffer(ModelBufferType::kTempValue, temp_value));
  
  // Fixed decode input buffer for CUDA Graph optimization
  // This ensures the input address remains constant across decode calls
  tensor::Tensor decode_input(activation_dtype, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kDecodeInput, decode_input));
  
  // Pinned memory buffers for efficient async Host-Device transfers
  // These enable true asynchronous memory transfers without blocking CPU
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    std::shared_ptr<base::DeviceAllocator> alloc_pinned = 
        base::CPUPinnedAllocatorFactory::get_instance();
    
    // Pinned pos buffer for async H2D transfer
    tensor::Tensor pos_pinned(base::DataType::kDataTypeInt32, 1, true, alloc_pinned);
    CHECK(insert_buffer(ModelBufferType::kInputPosPinned, pos_pinned));
    
    // Pre-allocated argmax output buffer on GPU (avoids per-decode allocation)
    // Use 2 int32_t to store size_t (8 bytes on 64-bit system)
    tensor::Tensor argmax_output(base::DataType::kDataTypeInt32, 2, true, alloc);
    CHECK(insert_buffer(ModelBufferType::kArgmaxOutput, argmax_output));
    
    // Pinned argmax result buffer for async D2H transfer
    tensor::Tensor argmax_pinned(base::DataType::kDataTypeInt32, 2, true, alloc_pinned);
    CHECK(insert_buffer(ModelBufferType::kArgmaxOutputPinned, argmax_pinned));
  }

  // Attention scores - keep FP32 for numerical stability in softmax
  tensor::Tensor attn(base::DataType::kDataTypeFp32, config_->head_num_, config_->seq_len_, true,
                      alloc);
  CHECK(insert_buffer(ModelBufferType::kScoreStorage, attn));
  CHECK(insert_buffer(ModelBufferType::kAttnOutput, query));

  // final forward output - keep FP32 for logits precision
  tensor::Tensor forward_output(base::DataType::kDataTypeFp32, config_->vocab_size_, true, alloc);
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    tensor::Tensor forward_output_cpu(base::DataType::kDataTypeFp32, config_->vocab_size_, true,
                                      alloc_cpu);
    CHECK(insert_buffer(ModelBufferType::kForwardOutputCPU, forward_output_cpu));
  }

  CHECK(insert_buffer(ModelBufferType::kForwardOutput, forward_output));
}

base::Status Qwen2Model::create_layers() {
  using namespace base;
  if (!qwen_layers_) {
    qwen_layers_ = std::make_unique<Qwen2Layers>();
  }

  if (is_fp16_model_) {
    // FP16 model (version 3)
    create_param_layers_fp16();
  } else if (!is_quant_model_) {
    create_param_layers();
  } else {
    create_param_quant_layers();
  }
  create_nonparam_layers();

  if (!qwen_layers_->embedding_layer_) {
    return error::InternalError("Create the embedding layer for the llama model failed!");
  }

  if (qwen_layers_->rmsnorm_layers_.size() != 2 * config_->layer_num_ + 1) {
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

void Qwen2Model::attention_rms(int32_t layer_idx, const tensor::Tensor& input) const {
  CHECK(qwen_layers_ != nullptr);
  // attn rmsnorm
  tensor::Tensor rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  std::shared_ptr<op::Layer> rmsnorm_layer = qwen_layers_->rmsnorm_layers_.at(layer_idx);
  if (!rmsnorm_layer) {
    LOG(FATAL) << "The attention rmsnorm layer is a null pointer in the llama2 model";
  }
  STATUS_CHECK(rmsnorm_layer->forward(input, rmsnorm_output));
}

void Qwen2Model::attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  CHECK(qwen_layers_ != nullptr);
  // kv cache
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
  int32_t pos = pos_tensor.index<int32_t>(0);
  // wq wk wv @ input
  const auto& [key, val] = slice_kv_cache(layer_idx, pos);
  // query
  const auto& query_layer = qwen_layers_->wq_layers_.at(layer_idx);
  CHECK_NE(query_layer, nullptr) << "The query layer in the attention block is null pointer.";

  auto rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  STATUS_CHECK(query_layer->forward(rmsnorm_output, query));

  // key
  const auto& key_layer = qwen_layers_->wk_layers_.at(layer_idx);
  CHECK_NE(key_layer, nullptr) << "The key layer in the attention block is null pointer.";
  STATUS_CHECK(key_layer->forward(rmsnorm_output, key));
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

void Qwen2Model::attention_qkv_with_graph(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  CHECK(qwen_layers_ != nullptr);
  CHECK(cuda_config_ != nullptr);
  
  // Use fixed-address temporary buffers for CUDA Graph compatibility
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
  tensor::Tensor temp_key = this->get_buffer(ModelBufferType::kTempKey);
  tensor::Tensor temp_value = this->get_buffer(ModelBufferType::kTempValue);
  
  auto rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  
  // query
  const auto& query_layer = qwen_layers_->wq_layers_.at(layer_idx);
  CHECK_NE(query_layer, nullptr) << "The query layer in the attention block is null pointer.";
  STATUS_CHECK(query_layer->forward(rmsnorm_output, query));

  // key -> temp_key (fixed address)
  const auto& key_layer = qwen_layers_->wk_layers_.at(layer_idx);
  CHECK_NE(key_layer, nullptr) << "The key layer in the attention block is null pointer.";
  STATUS_CHECK(key_layer->forward(rmsnorm_output, temp_key));
  
  // value -> temp_value (fixed address)
  const auto& value_layer = qwen_layers_->wv_layers_.at(layer_idx);
  CHECK_NE(value_layer, nullptr) << "The value layer in the attention block is null pointer.";
  STATUS_CHECK(value_layer->forward(rmsnorm_output, temp_value));

  // rope on query and temp_key - use GPU pos version for CUDA Graph compatibility
  CHECK_NE(qwen_layers_->rope_gpu_pos_layer_, nullptr)
      << "The RoPE GPU pos layer in the attention block is null pointer.";
  
  // Use layer for unified kernel access
  auto rope_layer = qwen_layers_->rope_gpu_pos_layer_;
  rope_layer->set_input(0, query);
  rope_layer->set_input(1, temp_key);
  rope_layer->set_input(2, pos_tensor);
  rope_layer->set_input(3, get_buffer(ModelBufferType::kSinCache));
  rope_layer->set_input(4, get_buffer(ModelBufferType::kCosCache));
  STATUS_CHECK(rope_layer->forward());
  
  // Copy temp_key and temp_value to KV cache at correct position
  // Position is read from GPU memory inside the kernel
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);
  
  // Use layer for unified KV cache copy
  auto key_cache_layer = qwen_layers_->kv_cache_key_layer_;
  key_cache_layer->set_layer_index(layer_idx);
  key_cache_layer->set_use_gpu_pos(true);
  key_cache_layer->set_input(0, temp_key);
  key_cache_layer->set_input(1, key_cache);
  key_cache_layer->set_input(2, pos_tensor);
  STATUS_CHECK(key_cache_layer->forward());
  
  auto value_cache_layer = qwen_layers_->kv_cache_value_layer_;
  value_cache_layer->set_layer_index(layer_idx);
  value_cache_layer->set_use_gpu_pos(true);
  value_cache_layer->set_input(0, temp_value);
  value_cache_layer->set_input(1, val_cache);
  value_cache_layer->set_input(2, pos_tensor);
  STATUS_CHECK(value_cache_layer->forward());
}

base::Status Qwen2Model::predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                                 bool is_prompt, int& next) const {
  auto status = forward(input, pos_tensor, next);
  if (!status) {
    return status;
  }
  next = post_processing(pos_tensor, is_prompt);
  return base::error::Success();
}

void Qwen2Model::attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  CHECK(qwen_layers_ != nullptr);
  // mha
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);

  tensor::Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
  
  int pos = pos_tensor.index<int32_t>(0);

  // Check if using pure FP16 path
  if (query.data_type() == base::DataType::kDataTypeFp16 &&
      key_cache.data_type() == base::DataType::kDataTypeFp16) {
    // Use Flash Attention layer for decode (FP16)
    auto flash_attn = qwen_layers_->flash_attention_decode_layer_;
    flash_attn->set_layer_index(layer_idx);
    flash_attn->set_pos(pos);
    flash_attn->set_use_gpu_pos(false);
    flash_attn->set_input(0, query);
    flash_attn->set_input(1, mha_output);
    flash_attn->set_input(2, key_cache);
    flash_attn->set_input(3, val_cache);
    // input[4] not needed when use_gpu_pos=false
    flash_attn->set_cuda_config(cuda_config_);
    STATUS_CHECK(flash_attn->forward());
  } else {
    // Standard FP32 MHA path
    tensor::Tensor score_storage = get_buffer(ModelBufferType::kScoreStorage);
    const auto& mha_layer = qwen_layers_->mha_layer_;
    CHECK_NE(mha_layer, nullptr) << "The multi head attention layer is null pointer.";
    std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_pos(pos);
    std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_layer_idx(layer_idx);
    STATUS_CHECK(mha_layer->forward(query, score_storage, key_cache, val_cache, mha_output));
  }

  // wo @ attention output
  tensor::Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
  const auto& wo_layer = qwen_layers_->wo_layers_.at(layer_idx);
  CHECK_NE(wo_layer, nullptr) << "The weight output layer is null pointer.";
  STATUS_CHECK(wo_layer->forward(mha_output, attn_output));
}

void Qwen2Model::attention_mha_with_graph(int32_t layer_idx, const tensor::Tensor& pos_tensor_gpu) const {
  CHECK(qwen_layers_ != nullptr);
  CHECK(cuda_config_ != nullptr);
  
  // Get KV caches and buffers
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);
  tensor::Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
  
  // Check if using pure FP16 path
  if (query.data_type() == base::DataType::kDataTypeFp16 &&
      key_cache.data_type() == base::DataType::kDataTypeFp16) {
    // Use Flash Attention layer with GPU pos for CUDA Graph compatibility
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
  } else {
    // Standard FP32 path using GPU pos kernel for CUDA Graph compatibility
    tensor::Tensor score_storage = get_buffer(ModelBufferType::kScoreStorage);
    STATUS_CHECK(qwen_layers_->mha_gpu_pos_layer_->forward(
        pos_tensor_gpu.ptr<int32_t>(),  // Position pointer in GPU memory
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
  CHECK_NE(wo_layer, nullptr) << "The weight output layer is null pointer.";
  STATUS_CHECK(wo_layer->forward(mha_output, attn_output));
}

void Qwen2Model::feed_forward(int32_t layer_idx, const tensor::Tensor& input) const {
  CHECK(qwen_layers_ != nullptr);
  
  // residual add
  CHECK_NE(qwen_layers_->add_layer_, nullptr)
      << "The add layer in the feedforward block is null pointer";
  STATUS_CHECK(
      qwen_layers_->add_layer_->forward(input, get_buffer(ModelBufferType::kAttnOutput), input));

  // ffn rmsnorm
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

/**
 * Fused Feed Forward implementation for decode phase (single token)
 * 
 * This fuses W1 @ x (gate), W3 @ x (up), and SwiGLU activation into a single kernel.
 * Benefits:
 * - Reduces kernel launches from 4 to 2 (fused_gate_up_swiglu + W2)
 * - Eliminates 2 intermediate memory writes (W1 output, W3 output)
 * - Input vector (ffn_norm_output) is read only once instead of twice
 * 
 * Memory bandwidth savings:
 * - Original: read input 2x (for W1 and W3) + write W1_out + write W3_out + read both for SwiGLU
 * - Fused: read input 1x, write directly to W1_out (reused for SwiGLU output)
 * 
 * For Qwen2.5-7B (dim=3584, hidden_dim=18944):
 * - Original: 2*3584 + 2*18944 + 2*18944 = 83,840 floats = 335KB
 * - Fused: 3584 + 18944 = 22,528 floats = 90KB (73% reduction)
 */
void Qwen2Model::feed_forward_fused(int32_t layer_idx, const tensor::Tensor& input) const {
  CHECK(qwen_layers_ != nullptr);
  
  // Debug code disabled for CUDA Graph compatibility
  #if 0
  // Debug: check inputs before residual add
  static int ff_fused_debug_count = 0;
  bool debug_this = (ff_fused_debug_count < 3 && layer_idx == 0 && is_fp16_model_);
  
  if (debug_this) {
    if (cuda_config_) cudaStreamSynchronize(cuda_config_->stream);
    std::vector<uint16_t> h_input(std::min((size_t)100, input.size()));
    std::vector<uint16_t> h_attn(std::min((size_t)100, get_buffer(ModelBufferType::kAttnOutput).size()));
    cudaMemcpy(h_input.data(), input.ptr<uint16_t>(), h_input.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_attn.data(), get_buffer(ModelBufferType::kAttnOutput).ptr<uint16_t>(), h_attn.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    
    float input_min = 1e30f, input_max = -1e30f, attn_min = 1e30f, attn_max = -1e30f;
    int input_nan = 0, attn_nan = 0;
    for (size_t i = 0; i < h_input.size(); i++) {
      half hv = *reinterpret_cast<half*>(&h_input[i]);
      float fv = __half2float(hv);
      if (__hisnan(hv)) input_nan++;
      else if (!isinf(fv)) { input_min = std::min(input_min, fv); input_max = std::max(input_max, fv); }
    }
    for (size_t i = 0; i < h_attn.size(); i++) {
      half hv = *reinterpret_cast<half*>(&h_attn[i]);
      float fv = __half2float(hv);
      if (__hisnan(hv)) attn_nan++;
      else if (!isinf(fv)) { attn_min = std::min(attn_min, fv); attn_max = std::max(attn_max, fv); }
    }
    LOG(INFO) << "FFN_FUSED BEFORE residual add (layer " << layer_idx << "):";
    LOG(INFO) << "  input: nan=" << input_nan << ", range=[" << input_min << ", " << input_max << "]";
    LOG(INFO) << "  attn_out: nan=" << attn_nan << ", range=[" << attn_min << ", " << attn_max << "]";
  }
  #endif
  
  // residual add from attention output
  CHECK_NE(qwen_layers_->add_layer_, nullptr)
      << "The add layer in the feedforward block is null pointer";
  STATUS_CHECK(
      qwen_layers_->add_layer_->forward(input, get_buffer(ModelBufferType::kAttnOutput), input));
  
  #if 0
  if (debug_this) {
    if (cuda_config_) cudaStreamSynchronize(cuda_config_->stream);
    std::vector<uint16_t> h_input(std::min((size_t)100, input.size()));
    cudaMemcpy(h_input.data(), input.ptr<uint16_t>(), h_input.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    float min_val = 1e30f, max_val = -1e30f;
    int nan_count = 0;
    for (size_t i = 0; i < h_input.size(); i++) {
      half hv = *reinterpret_cast<half*>(&h_input[i]);
      float fv = __half2float(hv);
      if (__hisnan(hv)) nan_count++;
      else if (!isinf(fv)) { min_val = std::min(min_val, fv); max_val = std::max(max_val, fv); }
    }
    LOG(INFO) << "FFN_FUSED AFTER residual add (layer " << layer_idx << "): nan=" << nan_count << ", range=[" << min_val << ", " << max_val << "]";
  }
  #endif

  // ffn rmsnorm
  tensor::Tensor ffn_norm_output = get_buffer(ModelBufferType::kFFNRMSNorm);
  const auto& ffn_rmsnorm = qwen_layers_->rmsnorm_layers_.at(layer_idx + config_->layer_num_);
  CHECK_NE(ffn_rmsnorm, nullptr)
      << "The final rmsnorm layer in the feedforward block is null pointer";
  STATUS_CHECK(ffn_rmsnorm->forward(input, ffn_norm_output));
  
  #if 0
  if (debug_this) {
    if (cuda_config_) cudaStreamSynchronize(cuda_config_->stream);
    std::vector<uint16_t> h_norm(std::min((size_t)100, ffn_norm_output.size()));
    cudaMemcpy(h_norm.data(), ffn_norm_output.ptr<uint16_t>(), h_norm.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    float min_val = 1e30f, max_val = -1e30f;
    int nan_count = 0;
    for (size_t i = 0; i < h_norm.size(); i++) {
      half hv = *reinterpret_cast<half*>(&h_norm[i]);
      float fv = __half2float(hv);
      if (__hisnan(hv)) nan_count++;
      else if (!isinf(fv)) { min_val = std::min(min_val, fv); max_val = std::max(max_val, fv); }
    }
    LOG(INFO) << "FFN_FUSED AFTER RMSNorm (layer " << layer_idx << "): nan=" << nan_count << ", range=[" << min_val << ", " << max_val << "]";
    ff_fused_debug_count++;
  }
  #endif

  // Fused W1 + W3 + SwiGLU kernel
  // Output goes to w1_output buffer (will be input to W2)
  tensor::Tensor w1_output = get_buffer(ModelBufferType::kW1Output);
  const auto& w1_layer = qwen_layers_->w1_layers_.at(layer_idx);
  const auto& w3_layer = qwen_layers_->w3_layers_.at(layer_idx);
  CHECK_NE(w1_layer, nullptr) << "The w1 layer in the feedforward block is null pointer";
  CHECK_NE(w3_layer, nullptr) << "The w3 layer in the feedforward block is null pointer";
  
  auto w1_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(w1_layer);
  auto w3_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(w3_layer);
  
  // Call fused FFN layer: output = silu(W1 @ input) * (W3 @ input)
  auto fused_ffn = qwen_layers_->fused_ffn_layer_;
  fused_ffn->set_input(0, ffn_norm_output);
  fused_ffn->set_input(1, w1_matmul->get_weight(0));
  fused_ffn->set_input(2, w3_matmul->get_weight(0));
  fused_ffn->set_output(0, w1_output);
  STATUS_CHECK(fused_ffn->forward());

  // w2 (down projection)
  tensor::Tensor w2_output = get_buffer(ModelBufferType::kW2Output);
  const auto& w2_layer = qwen_layers_->w2_layers_.at(layer_idx);
  CHECK_NE(w2_layer, nullptr) << "The w2 layer in the feedforward block is null pointer";
  STATUS_CHECK(w2_layer->forward(w1_output, w2_output));

  // residual add
  CHECK_NE(qwen_layers_->add_layer_, nullptr)
      << "The add layer in the feedforward block is null pointer";
  STATUS_CHECK(qwen_layers_->add_layer_->forward(input, w2_output, input));
}

op::EmbeddingOutput Qwen2Model::embedding(const std::vector<int>& tokens) const {
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


void Qwen2Model::cls_logits(const tensor::Tensor& input) const {
  CHECK(qwen_layers_ != nullptr);
  const auto& norm = qwen_layers_->rmsnorm_layers_.at(2 * config_->layer_num_);
  CHECK_NE(norm, nullptr);
  STATUS_CHECK(norm->forward(input, input));

  // Debug code disabled for CUDA Graph compatibility
  #if 0
  // Debug: Check input before cls_layer
  static int cls_call_count = 0;
  if (cls_call_count < 3) {
    if (cuda_config_ && cuda_config_->stream) {
      cudaStreamSynchronize(cuda_config_->stream);
    }
    
    if (input.data_type() == base::DataType::kDataTypeFp16) {
      std::vector<uint16_t> cpu_input(100);
      cudaMemcpy(cpu_input.data(), input.ptr<uint16_t>(), 
                 100 * sizeof(uint16_t), cudaMemcpyDeviceToHost);
      
      int nan_count = 0, inf_count = 0;
      float max_val = -1e30f, min_val = 1e30f;
      for (size_t i = 0; i < 100; ++i) {
        half h = *reinterpret_cast<half*>(&cpu_input[i]);
        float v = __half2float(h);
        if (std::isnan(v)) nan_count++;
        if (std::isinf(v)) inf_count++;
        if (!std::isnan(v) && !std::isinf(v)) {
          if (v > max_val) max_val = v;
          if (v < min_val) min_val = v;
        }
      }
      LOG(INFO) << "cls_logits input FP16 (call " << cls_call_count << "): "
                << "min=" << min_val << ", max=" << max_val
                << ", nan=" << nan_count << ", inf=" << inf_count;
    }
    cls_call_count++;
  }
  #endif

  tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
  CHECK_NE(qwen_layers_->cls_layer_, nullptr);
  STATUS_CHECK(qwen_layers_->cls_layer_->forward(input, forward_output));

  // Debug code disabled for production
  #if 0
  // Debug: Check output logits after cls_layer
  static int logits_debug_count = 0;
  if (logits_debug_count < 3) {
    if (cuda_config_ && cuda_config_->stream) {
      cudaStreamSynchronize(cuda_config_->stream);
    }
    
    std::vector<float> cpu_logits(1000);
    cudaMemcpy(cpu_logits.data(), forward_output.ptr<float>(), 
               1000 * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Find argmax in first 1000 logits
    int argmax = 0;
    float max_logit = cpu_logits[0];
    for (int i = 1; i < 1000; ++i) {
      if (cpu_logits[i] > max_logit) {
        max_logit = cpu_logits[i];
        argmax = i;
      }
    }
    
    // Find global argmax
    std::vector<float> all_logits(forward_output.size());
    cudaMemcpy(all_logits.data(), forward_output.ptr<float>(), 
               forward_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    int global_argmax = 0;
    float global_max = all_logits[0];
    for (size_t i = 1; i < all_logits.size(); ++i) {
      if (all_logits[i] > global_max) {
        global_max = all_logits[i];
        global_argmax = static_cast<int>(i);
      }
    }
    
    LOG(INFO) << "cls_logits OUTPUT (call " << logits_debug_count << "): "
              << "vocab_size=" << forward_output.size()
              << ", global_argmax=" << global_argmax << " (logit=" << global_max << ")"
              << ", first 10 logits: " << cpu_logits[0] << ", " << cpu_logits[1] << ", " << cpu_logits[2]
              << ", " << cpu_logits[3] << ", " << cpu_logits[4] << ", " << cpu_logits[5]
              << ", " << cpu_logits[6] << ", " << cpu_logits[7] << ", " << cpu_logits[8] << ", " << cpu_logits[9];
    logits_debug_count++;
  }
  #endif
}
int32_t Qwen2Model::post_processing(const tensor::Tensor& pos, bool is_prompt) const {
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

// ==================== Batched Prefill Implementation ====================

void Qwen2Model::batched_attention_rms(int32_t layer_idx, const tensor::Tensor& input, 
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
void Qwen2Model::batched_attention_rms(int32_t layer_idx, const tensor::Tensor& input,
                                       const tensor::Tensor& output, int32_t seq_len) const {
  CHECK(qwen_layers_ != nullptr);
  std::shared_ptr<op::Layer> rmsnorm_layer = qwen_layers_->rmsnorm_layers_.at(layer_idx);
  if (!rmsnorm_layer) {
    LOG(FATAL) << "The attention rmsnorm layer is a null pointer";
  }
  // Use layer forward call instead of direct kernel call (input -> output)
  STATUS_CHECK(rmsnorm_layer->forward(input, output));
}

void Qwen2Model::batched_attention_qkv(int32_t layer_idx, const tensor::Tensor& rms_out,
                                       const tensor::Tensor& query_out, const tensor::Tensor& key_out, 
                                       const tensor::Tensor& value_out,
                                       int32_t seq_len, int32_t start_pos) const {
  CHECK(qwen_layers_ != nullptr);
  
  // Get weight layers
  const auto& query_layer = qwen_layers_->wq_layers_.at(layer_idx);
  const auto& key_layer = qwen_layers_->wk_layers_.at(layer_idx);
  const auto& value_layer = qwen_layers_->wv_layers_.at(layer_idx);
  
  CHECK_NE(query_layer, nullptr);
  CHECK_NE(key_layer, nullptr);
  CHECK_NE(value_layer, nullptr);
  
  auto query_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(query_layer);
  auto key_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(key_layer);
  auto value_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(value_layer);
  
  // Batched matmul: rms_out [seq_len, dim] @ weight [out_dim, dim]^T -> output [seq_len, out_dim]
  STATUS_CHECK(qwen_layers_->batched_matmul_helper_layer_->forward(
      rms_out, query_matmul->get_weight(0), query_out, seq_len, 1.f));
  
  // Add bias if exists
  if (query_matmul->get_bias(0).size() > 0) {
    // Add bias via bias_add_layer_
    qwen_layers_->bias_add_layer_->set_dims(seq_len, config_->dim_);
    qwen_layers_->bias_add_layer_->set_input(0, query_out);
    qwen_layers_->bias_add_layer_->set_input(1, query_matmul->get_bias(0));
    qwen_layers_->bias_add_layer_->set_output(0, query_out);
    STATUS_CHECK(qwen_layers_->bias_add_layer_->forward());
  }
  
  STATUS_CHECK(qwen_layers_->batched_matmul_helper_layer_->forward(
      rms_out, key_matmul->get_weight(0), key_out, seq_len, 1.f));
  if (key_matmul->get_bias(0).size() > 0) {
    // Add bias via bias_add_layer_
    qwen_layers_->bias_add_layer_->set_dims(seq_len, config_->kv_dim_);
    qwen_layers_->bias_add_layer_->set_input(0, key_out);
    qwen_layers_->bias_add_layer_->set_input(1, key_matmul->get_bias(0));
    qwen_layers_->bias_add_layer_->set_output(0, key_out);
    STATUS_CHECK(qwen_layers_->bias_add_layer_->forward());
  }
  
  STATUS_CHECK(qwen_layers_->batched_matmul_helper_layer_->forward(
      rms_out, value_matmul->get_weight(0), value_out, seq_len, 1.f));
  if (value_matmul->get_bias(0).size() > 0) {
    // Add bias via bias_add_layer_
    qwen_layers_->bias_add_layer_->set_dims(seq_len, config_->kv_dim_);
    qwen_layers_->bias_add_layer_->set_input(0, value_out);
    qwen_layers_->bias_add_layer_->set_input(1, value_matmul->get_bias(0));
    qwen_layers_->bias_add_layer_->set_output(0, value_out);
    STATUS_CHECK(qwen_layers_->bias_add_layer_->forward());
  }
  
  // Copy value to KV cache (before RoPE since value doesn't need RoPE)
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  tensor::Tensor value_cache = get_buffer(ModelBufferType::kValueCache);
  
  int32_t layer_offset = layer_idx * config_->seq_len_ * config_->kv_dim_;
  
  // Check if using FP16 path
  if (value_cache.data_type() == base::DataType::kDataTypeFp16) {
    for (int i = 0; i < seq_len; ++i) {
      int32_t cache_offset = layer_offset + (start_pos + i) * config_->kv_dim_;
      cudaMemcpyAsync(const_cast<uint16_t*>(value_cache.ptr<uint16_t>(cache_offset)),
                      value_out.ptr<uint16_t>(i * config_->kv_dim_),
                      config_->kv_dim_ * sizeof(uint16_t),
                      cudaMemcpyDeviceToDevice, cuda_config_->stream);
    }
  } else {
    for (int i = 0; i < seq_len; ++i) {
      int32_t cache_offset = layer_offset + (start_pos + i) * config_->kv_dim_;
      cudaMemcpyAsync(const_cast<float*>(value_cache.ptr<float>(cache_offset)),
                      value_out.ptr<float>(i * config_->kv_dim_),
                      config_->kv_dim_ * sizeof(float),
                      cudaMemcpyDeviceToDevice, cuda_config_->stream);
    }
  }
  
  // Apply RoPE to query and key (in-place) via layer abstraction
  auto batched_rope = qwen_layers_->batched_rope_layer_;
  batched_rope->set_seq_len(seq_len);
  batched_rope->set_start_pos(start_pos);
  batched_rope->set_input(0, query_out);
  batched_rope->set_input(1, key_out);
  batched_rope->set_input(2, get_buffer(ModelBufferType::kSinCache));
  batched_rope->set_input(3, get_buffer(ModelBufferType::kCosCache));
  batched_rope->set_cuda_config(cuda_config_);
  STATUS_CHECK(batched_rope->forward());
  
  // DEBUG: Disabled for production
  #if 0
  // DEBUG: Check query values after RoPE (first layer only)
  static int qkv_debug_count = 0;
  if (qkv_debug_count < 3 && layer_idx == 0) {
    cudaStreamSynchronize(cuda_config_->stream);
    
    // Check query after RoPE
    if (query_out.data_type() == base::DataType::kDataTypeFp16) {
      const int check_size = std::min(128, seq_len * config_->dim_);
      std::vector<uint16_t> cpu_q(check_size);
      cudaMemcpy(cpu_q.data(), query_out.ptr<uint16_t>(), 
                 check_size * sizeof(uint16_t), cudaMemcpyDeviceToHost);
      
      int nan_count = 0, inf_count = 0;
      float max_val = -1e30f, min_val = 1e30f, sum = 0.0f;
      for (int i = 0; i < check_size; ++i) {
        half h = *reinterpret_cast<half*>(&cpu_q[i]);
        float v = __half2float(h);
        if (std::isnan(v)) nan_count++;
        else if (std::isinf(v)) inf_count++;
        else {
          if (v > max_val) max_val = v;
          if (v < min_val) min_val = v;
          sum += v;
        }
      }
      LOG(INFO) << "[FP16 QKV] Layer " << layer_idx << " query after RoPE: "
                << "min=" << min_val << ", max=" << max_val 
                << ", mean=" << (sum / check_size)
                << ", nan=" << nan_count << ", inf=" << inf_count;
    }
    qkv_debug_count++;
  }
  #endif
  
  // Copy RoPE'd keys to KV cache
  if (key_cache.data_type() == base::DataType::kDataTypeFp16) {
    for (int i = 0; i < seq_len; ++i) {
      int32_t cache_offset = layer_offset + (start_pos + i) * config_->kv_dim_;
      cudaMemcpyAsync(const_cast<uint16_t*>(key_cache.ptr<uint16_t>(cache_offset)),
                      key_out.ptr<uint16_t>(i * config_->kv_dim_),
                      config_->kv_dim_ * sizeof(uint16_t),
                      cudaMemcpyDeviceToDevice, cuda_config_->stream);
    }
  } else {
    for (int i = 0; i < seq_len; ++i) {
      int32_t cache_offset = layer_offset + (start_pos + i) * config_->kv_dim_;
      cudaMemcpyAsync(const_cast<float*>(key_cache.ptr<float>(cache_offset)),
                      key_out.ptr<float>(i * config_->kv_dim_),
                      config_->kv_dim_ * sizeof(float),
                      cudaMemcpyDeviceToDevice, cuda_config_->stream);
    }
  }
}

void Qwen2Model::batched_attention_mha(int32_t layer_idx, const tensor::Tensor& query,
                                       const tensor::Tensor& mha_out, 
                                       int32_t seq_len, int32_t start_pos) const {
  CHECK(qwen_layers_ != nullptr);
  
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  tensor::Tensor value_cache = get_buffer(ModelBufferType::kValueCache);
  
  std::shared_ptr<base::DeviceAllocator> alloc = base::CUDADeviceAllocatorFactory::get_instance();

#if USE_FLASH_ATTENTION
  // Use Flash Attention - more memory efficient via layer abstraction
  // Check if using pure FP16 path
  auto prefill_layer = qwen_layers_->flash_attention_prefill_layer_;
  prefill_layer->set_cur_seq_len(seq_len);
  prefill_layer->set_start_pos(start_pos);
  prefill_layer->set_layer_index(layer_idx);
  prefill_layer->set_use_fp16(query.data_type() == base::DataType::kDataTypeFp16 &&
                              key_cache.data_type() == base::DataType::kDataTypeFp16);
  prefill_layer->set_input(0, query);
  prefill_layer->set_input(1, mha_out);
  prefill_layer->set_input(2, key_cache);
  prefill_layer->set_input(3, value_cache);
  prefill_layer->set_cuda_config(cuda_config_);
  STATUS_CHECK(prefill_layer->forward());
    
  // DEBUG: Disabled for production FP16 path
    #if 0
    // DEBUG: Check FP16 flash attention output for numerical issues (first layer only)
    static int fp16_attn_debug_count = 0;
    if (fp16_attn_debug_count < 3 && layer_idx == 0) {
      cudaStreamSynchronize(cuda_config_->stream);
      
      const int check_size = std::min(128, seq_len * config_->dim_);
      std::vector<uint16_t> cpu_out(check_size);
      cudaMemcpy(cpu_out.data(), mha_out.ptr<uint16_t>(), 
                 check_size * sizeof(uint16_t), cudaMemcpyDeviceToHost);
      
      int nan_count = 0, inf_count = 0, zero_count = 0;
      float max_val = -1e30f, min_val = 1e30f, sum = 0.0f;
      for (int i = 0; i < check_size; ++i) {
        half h = *reinterpret_cast<half*>(&cpu_out[i]);
        float v = __half2float(h);
        if (std::isnan(v)) nan_count++;
        else if (std::isinf(v)) inf_count++;
        else {
          if (fabsf(v) < 1e-6f) zero_count++;
          if (v > max_val) max_val = v;
          if (v < min_val) min_val = v;
          sum += v;
        }
      }
      LOG(INFO) << "[FP16 Flash Attn] Layer " << layer_idx << " output: "
                << "min=" << min_val << ", max=" << max_val 
                << ", mean=" << (sum / check_size)
                << ", nan=" << nan_count << ", inf=" << inf_count 
                << ", zeros=" << zero_count << "/" << check_size;
      fp16_attn_debug_count++;
    }
    #endif
#else
  // Allocate temporary score storage for batched attention
  // Score shape: [seq_len, head_num, max_seq_len]
  tensor::Tensor score_storage(base::DataType::kDataTypeFp32, 
                               seq_len, config_->head_num_, config_->seq_len_, true, alloc);
  
  // Call batched MHA layer
  STATUS_CHECK(qwen_layers_->batched_mha_layer_->forward(
      start_pos, seq_len, config_->head_num_, layer_idx,
      config_->seq_len_, config_->dim_, config_->kv_dim_, 
      config_->kv_mul_, config_->head_size_, mha_out,
      query, score_storage, key_cache, value_cache));
#endif
  
  // wo @ mha_output for each position
  const auto& wo_layer = qwen_layers_->wo_layers_.at(layer_idx);
  CHECK_NE(wo_layer, nullptr);
  auto wo_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(wo_layer);
  
  // Use same data type as query for activation tensors
  base::DataType activation_dtype = query.data_type();
  
  // Batched matmul for wo: mha_out [seq_len, dim] @ wo [dim, dim] -> attn_out [seq_len, dim]
  tensor::Tensor attn_out(activation_dtype, seq_len, config_->dim_, true, alloc);
  STATUS_CHECK(qwen_layers_->batched_matmul_helper_layer_->forward(
      mha_out, wo_matmul->get_weight(0), attn_out, seq_len, 1.f));
  
  // Copy result back to mha_out (reuse buffer)
  size_t copy_size = seq_len * config_->dim_ * 
                     (activation_dtype == base::DataType::kDataTypeFp16 ? sizeof(uint16_t) : sizeof(float));
  cudaMemcpyAsync(const_cast<void*>(mha_out.get_buffer()->ptr()), attn_out.get_buffer()->ptr(),
                  copy_size, cudaMemcpyDeviceToDevice, cuda_config_->stream);
}

void Qwen2Model::batched_feed_forward(int32_t layer_idx, const tensor::Tensor& input,
                                      int32_t seq_len) const {
  CHECK(qwen_layers_ != nullptr);
  std::shared_ptr<base::DeviceAllocator> alloc = base::CUDADeviceAllocatorFactory::get_instance();
  
  // Use same data type as input for activation tensors
  base::DataType activation_dtype = input.data_type();
  
  // Allocate temporary buffers for batched FFN
  tensor::Tensor ffn_norm_out(activation_dtype, seq_len, config_->dim_, true, alloc);
  tensor::Tensor w1_out(activation_dtype, seq_len, config_->hidden_dim_, true, alloc);
  tensor::Tensor w3_out(activation_dtype, seq_len, config_->hidden_dim_, true, alloc);
  tensor::Tensor w2_out(activation_dtype, seq_len, config_->dim_, true, alloc);
  
  // FFN RMSNorm - use layer forward call
  const auto& ffn_rmsnorm = qwen_layers_->rmsnorm_layers_.at(layer_idx + config_->layer_num_);
  CHECK_NE(ffn_rmsnorm, nullptr);
  STATUS_CHECK(ffn_rmsnorm->forward(input, ffn_norm_out));
  
  // W1 matmul
  const auto& w1_layer = qwen_layers_->w1_layers_.at(layer_idx);
  CHECK_NE(w1_layer, nullptr);
  auto w1_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(w1_layer);
  STATUS_CHECK(qwen_layers_->batched_matmul_helper_layer_->forward(
      ffn_norm_out, w1_matmul->get_weight(0), w1_out, seq_len, 1.f));
  
  // W3 matmul
  const auto& w3_layer = qwen_layers_->w3_layers_.at(layer_idx);
  CHECK_NE(w3_layer, nullptr);
  auto w3_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(w3_layer);
  STATUS_CHECK(qwen_layers_->batched_matmul_helper_layer_->forward(
      ffn_norm_out, w3_matmul->get_weight(0), w3_out, seq_len, 1.f));
  
  // SwiGLU via batched_swiglu_layer_ (element-wise, works on full tensor)
  STATUS_CHECK(qwen_layers_->batched_swiglu_layer_->forward(w1_out, w3_out, w1_out));
  
  // W2 matmul
  const auto& w2_layer = qwen_layers_->w2_layers_.at(layer_idx);
  CHECK_NE(w2_layer, nullptr);
  auto w2_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(w2_layer);
  STATUS_CHECK(qwen_layers_->batched_matmul_helper_layer_->forward(
      w1_out, w2_matmul->get_weight(0), w2_out, seq_len, 1.f));
  
  // Residual add via batched_add_layer_
  STATUS_CHECK(qwen_layers_->batched_add_layer_->forward(input, w2_out, input));
}

// Optimized version with pre-allocated buffers (avoids per-layer allocation overhead)
void Qwen2Model::batched_feed_forward_optimized(int32_t layer_idx, const tensor::Tensor& input,
                                                tensor::Tensor& ffn_norm_out, tensor::Tensor& w1_out,
                                                tensor::Tensor& w3_out, tensor::Tensor& w2_out,
                                                int32_t seq_len) const {
  CHECK(qwen_layers_ != nullptr);
  
  // FFN RMSNorm - use layer forward call
  const auto& ffn_rmsnorm = qwen_layers_->rmsnorm_layers_.at(layer_idx + config_->layer_num_);
  CHECK_NE(ffn_rmsnorm, nullptr);
  STATUS_CHECK(ffn_rmsnorm->forward(input, ffn_norm_out));
  
  // W1 matmul
  const auto& w1_layer = qwen_layers_->w1_layers_.at(layer_idx);
  CHECK_NE(w1_layer, nullptr);
  auto w1_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(w1_layer);
  STATUS_CHECK(qwen_layers_->batched_matmul_helper_layer_->forward(
      ffn_norm_out, w1_matmul->get_weight(0), w1_out, seq_len, 1.f));
  
  // W3 matmul
  const auto& w3_layer = qwen_layers_->w3_layers_.at(layer_idx);
  CHECK_NE(w3_layer, nullptr);
  auto w3_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(w3_layer);
  STATUS_CHECK(qwen_layers_->batched_matmul_helper_layer_->forward(
      ffn_norm_out, w3_matmul->get_weight(0), w3_out, seq_len, 1.f));
  
  // SwiGLU via batched_swiglu_layer_ (element-wise, works on full tensor)
  STATUS_CHECK(qwen_layers_->batched_swiglu_layer_->forward(w1_out, w3_out, w1_out));
  
  // W2 matmul
  const auto& w2_layer = qwen_layers_->w2_layers_.at(layer_idx);
  CHECK_NE(w2_layer, nullptr);
  auto w2_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(w2_layer);
  STATUS_CHECK(qwen_layers_->batched_matmul_helper_layer_->forward(
      w1_out, w2_matmul->get_weight(0), w2_out, seq_len, 1.f));
  
  // Residual add via batched_add_layer_
  STATUS_CHECK(qwen_layers_->batched_add_layer_->forward(input, w2_out, input));
}

base::Status Qwen2Model::prefill(const tensor::Tensor& input, int32_t seq_len, 
                                 int32_t start_pos) const {
  if (input.is_empty()) {
    return base::error::InvalidArgument("The input tensor is empty.");
  }
  if (device_type_ != base::DeviceType::kDeviceCUDA) {
    return base::error::InternalError("Batched prefill only supports CUDA device");
  }
  
  std::shared_ptr<base::DeviceAllocator> alloc = base::CUDADeviceAllocatorFactory::get_instance();
  
  // Use same data type as input for activation tensors
  base::DataType activation_dtype = input.data_type();
  size_t elem_size = (activation_dtype == base::DataType::kDataTypeFp16) ? sizeof(uint16_t) : sizeof(float);
  
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
    
    // 2. Compute Q, K, V projections and apply RoPE
    batched_attention_qkv(layer_idx, rms_out, query_out, key_out, value_out, seq_len, start_pos);
    
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
  
  // Final layer norm and cls_logits on the last token
  // Create a view into the last token's hidden state
  void* last_token_ptr = static_cast<char*>(const_cast<void*>(final_hidden->get_buffer()->ptr())) + 
                         (seq_len - 1) * dim * elem_size;
  tensor::Tensor last_hidden(activation_dtype, dim, false, nullptr, last_token_ptr);
  last_hidden.set_device_type(device_type_);
  
  cls_logits(last_hidden);
  
  // Sync and check for any CUDA errors before returning
  // This ensures any errors are caught and cleared here rather than later
  if (cuda_config_ && cuda_config_->stream) {
    cudaStreamSynchronize(cuda_config_->stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      LOG(WARNING) << "CUDA error after prefill (cleared): " << cudaGetErrorString(err);
      // Error is now cleared by cudaGetLastError(), safe to continue
    }
  }
  
  return base::error::Success();
}

base::Status Qwen2Model::decode(const tensor::Tensor& input, int32_t pos, int& next) const {
  if (input.is_empty()) {
    return base::error::InvalidArgument("The input tensor is empty.");
  }
  
  // Check if we should use CUDA Graph optimization
  bool use_graph = cuda_config_ && cuda_config_->should_use_graph();
  
  if (use_graph) {
    // Use fixed-address buffers for CUDA Graph compatibility
    tensor::Tensor pos_tensor_gpu = get_buffer(ModelBufferType::kInputPosGPU);
    tensor::Tensor decode_input = get_buffer(ModelBufferType::kDecodeInput);
    tensor::Tensor pos_pinned = get_buffer(ModelBufferType::kInputPosPinned);
    tensor::Tensor argmax_output = get_buffer(ModelBufferType::kArgmaxOutput);
    tensor::Tensor argmax_pinned = get_buffer(ModelBufferType::kArgmaxOutputPinned);
    
    auto& graph_ctx = cuda_config_->graph_context;
    auto& graph = graph_ctx->decode_graph;
    
    // Check if we need to recapture the graph
    bool need_capture = !graph->is_valid();
    
    // Copy input embedding to fixed decode_input buffer
    // This ensures CUDA Graph uses the same address every time
    // Use correct size based on data type
    size_t copy_size = config_->dim_ * (is_fp16_model_ ? sizeof(uint16_t) : sizeof(float));
    cudaMemcpyAsync(const_cast<void*>(decode_input.get_buffer()->ptr()), 
                    input.get_buffer()->ptr(),
                    copy_size, cudaMemcpyDeviceToDevice, cuda_config_->stream);
    
    // Update position using pinned memory for true async H2D transfer
    // Write to pinned buffer first (CPU operation, no sync needed)
    *const_cast<int32_t*>(pos_pinned.ptr<int32_t>()) = pos;
    // Async copy from pinned to GPU - truly asynchronous since source is page-locked
    cudaMemcpyAsync(const_cast<int32_t*>(pos_tensor_gpu.ptr<int32_t>()), 
                    pos_pinned.ptr<int32_t>(), sizeof(int32_t), 
                    cudaMemcpyHostToDevice, cuda_config_->stream);
    
    if (need_capture && !graph->is_disabled()) {
      // Sync before capture to ensure copies are complete
      cudaStreamSynchronize(cuda_config_->stream);
      
      // Check for any CUDA errors before capture
      cudaError_t pre_err = cudaGetLastError();
      if (pre_err != cudaSuccess) {
        LOG(ERROR) << "CUDA error before graph capture: " << cudaGetErrorString(pre_err);
      }
      
      // Capture the graph using fixed-address buffers
      if (graph->begin_capture(cuda_config_->stream)) {
        LOG(INFO) << "Graph capture started successfully";
        for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
          attention_rms(layer_idx, decode_input);  // Use fixed decode_input
          attention_qkv_with_graph(layer_idx, pos_tensor_gpu);
          attention_mha_with_graph(layer_idx, pos_tensor_gpu);
          // Runtime switch for fused FFN optimization
          if (use_fused_ffn_) {
            feed_forward_fused(layer_idx, decode_input);  // Optimized: fused W1+W3+SwiGLU
          } else {
            feed_forward(layer_idx, decode_input);
          }
        }
        cls_logits(decode_input);  // Use fixed decode_input
        
        if (graph->end_capture(cuda_config_->stream)) {
          graph_ctx->graph_recaptures++;
          LOG(INFO) << "Graph capture SUCCESSFUL! Total captures: " << graph_ctx->graph_recaptures;
        } else {
          LOG(ERROR) << "Graph end_capture FAILED! Check for illegal operations in graph.";
          // Check for CUDA errors
          cudaError_t err = cudaGetLastError();
          if (err != cudaSuccess) {
            LOG(ERROR) << "CUDA error: " << cudaGetErrorString(err);
          }
        }
      } else {
        LOG(ERROR) << "Graph begin_capture FAILED!";
      }
    }
    
    if (graph->is_valid()) {
      // Launch the captured graph
      if (graph->launch(cuda_config_->stream)) {
        graph_ctx->graph_launches++;
        
        // Use optimized post_processing with pre-allocated buffers
        tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
        // Cast to ArgmaxSampler to use pre-allocated buffers
        auto* argmax_sampler = dynamic_cast<sampler::ArgmaxSampler*>(sampler_.get());
        if (argmax_sampler) {
          // Use pre-allocated buffers for async argmax
          argmax_sampler->sample_prealloc(
              forward_output.ptr<float>(), forward_output.size(),
              reinterpret_cast<size_t*>(const_cast<int32_t*>(argmax_output.ptr<int32_t>())),
              reinterpret_cast<size_t*>(const_cast<int32_t*>(argmax_pinned.ptr<int32_t>())),
              cuda_config_->stream);
          // Sync and read result from pinned memory
          cudaStreamSynchronize(cuda_config_->stream);
          next = static_cast<int32_t>(*reinterpret_cast<size_t*>(const_cast<int32_t*>(argmax_pinned.ptr<int32_t>())));
        } else {
          // Fallback to original path
          cudaStreamSynchronize(cuda_config_->stream);
          tensor::Tensor pos_tensor_cpu = get_buffer(ModelBufferType::kInputPos);
          next = post_processing(pos_tensor_cpu, false);
        }
        return base::error::Success();
      }
      // If launch failed, fall through to normal execution
      graph_ctx->invalidate();
    }
  }
  
  // Normal execution (no graph, or graph capture/launch failed)
  tensor::Tensor pos_tensor = get_buffer(ModelBufferType::kInputPos);
  pos_tensor.index<int32_t>(0) = pos;
  
  for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
    attention_rms(layer_idx, input);
    attention_qkv(layer_idx, pos_tensor);
    attention_mha(layer_idx, pos_tensor);
    
    // Runtime switch for fused FFN optimization
    if (use_fused_ffn_) {
      feed_forward_fused(layer_idx, input);  // Optimized: fused W1+W3+SwiGLU
    } else {
      feed_forward(layer_idx, input);
    }
  }
  
  cls_logits(input);
  
  // Sync stream if using CUDA
  if (cuda_config_ && cuda_config_->stream) {
    cudaStreamSynchronize(cuda_config_->stream);
  }
  
  // Sample next token
  next = post_processing(pos_tensor, false);
  return base::error::Success();
}

void Qwen2Model::clear_kv_cache() {
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  tensor::Tensor value_cache = get_buffer(ModelBufferType::kValueCache);
  
  // Determine element size based on data type
  const size_t elem_size = key_cache.data_type() == base::DataType::kDataTypeFp16 
                           ? sizeof(uint16_t)  // FP16 = 2 bytes
                           : sizeof(float);    // FP32 = 4 bytes
  
  if (device_type_ == base::DeviceType::kDeviceCUDA && cuda_config_) {
    cudaMemsetAsync(key_cache.ptr<void>(), 0, 
                    key_cache.size() * elem_size, cuda_config_->stream);
    cudaMemsetAsync(value_cache.ptr<void>(), 0, 
                    value_cache.size() * elem_size, cuda_config_->stream);
    cudaStreamSynchronize(cuda_config_->stream);
    
    // 重要：使 CUDA Graph 失效，因为 KV cache 状态已改变
    invalidate_cuda_graph();
  } else {
    memset(key_cache.ptr<void>(), 0, key_cache.size() * elem_size);
    memset(value_cache.ptr<void>(), 0, value_cache.size() * elem_size);
  }
}

}  // namespace model