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
    : QwenBaseModel(tokenizer_type, std::move(token_path),
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
  
  // 优化：KV cache中连续位置使用单次拷贝代替逐token循环
  int32_t cache_start_offset = layer_offset + start_pos * config_->kv_dim_;
  if (value_cache.data_type() == base::DataType::kDataTypeFp16) {
    cudaMemcpyAsync(const_cast<uint16_t*>(value_cache.ptr<uint16_t>(cache_start_offset)),
                    value_out.ptr<uint16_t>(0),
                    seq_len * config_->kv_dim_ * sizeof(uint16_t),
                    cudaMemcpyDeviceToDevice, cuda_config_->stream);
  } else {
    cudaMemcpyAsync(const_cast<float*>(value_cache.ptr<float>(cache_start_offset)),
                    value_out.ptr<float>(0),
                    seq_len * config_->kv_dim_ * sizeof(float),
                    cudaMemcpyDeviceToDevice, cuda_config_->stream);
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
  
  // 优化：key cache连续拷贝
  if (key_cache.data_type() == base::DataType::kDataTypeFp16) {
    cudaMemcpyAsync(const_cast<uint16_t*>(key_cache.ptr<uint16_t>(cache_start_offset)),
                    key_out.ptr<uint16_t>(0),
                    seq_len * config_->kv_dim_ * sizeof(uint16_t),
                    cudaMemcpyDeviceToDevice, cuda_config_->stream);
  } else {
    cudaMemcpyAsync(const_cast<float*>(key_cache.ptr<float>(cache_start_offset)),
                    key_out.ptr<float>(0),
                    seq_len * config_->kv_dim_ * sizeof(float),
                    cudaMemcpyDeviceToDevice, cuda_config_->stream);
  }
}


}  // namespace model