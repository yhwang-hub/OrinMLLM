/**
 * @file qwen3_vl_base.cpp
 * @brief Qwen3-VL Model Base Implementation
 * 
 * Contains initialization, setup, weight loading, and memory management functions
 * extracted from qwen3_vl.cpp for code organization.
 */

#ifdef QWEN3_VL_SUPPORT
#include "model/qwen3_vl.h"
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "op/matmul.h"
#include "op/mha.h"
#include "op/rmsnorm.h"
#include "op/batched_add.h"
#include "op/vision_layers.h"

namespace model {

// ============================================================================
// Vision Layers CUDA Transfer
// ============================================================================

void Qwen3VLVisionLayers::to_cuda(cudaStream_t stream) {
  auto copy_to_cuda = [stream](tensor::Tensor& tensor) {
    if (!tensor.is_empty() && tensor.device_type() != base::DeviceType::kDeviceCUDA) {
      tensor.to_cuda(stream);
    }
  };
  
  copy_to_cuda(patch_embed_weight);
  copy_to_cuda(patch_embed_bias);
  copy_to_cuda(pos_embed_weight);
  
  for (auto& block : blocks) {
    copy_to_cuda(block.norm1_weight);
    copy_to_cuda(block.norm1_bias);
    copy_to_cuda(block.norm2_weight);
    copy_to_cuda(block.norm2_bias);
    copy_to_cuda(block.qkv_weight);
    copy_to_cuda(block.qkv_bias);
    copy_to_cuda(block.proj_weight);
    copy_to_cuda(block.proj_bias);
    copy_to_cuda(block.mlp_fc1_weight);
    copy_to_cuda(block.mlp_fc1_bias);
    copy_to_cuda(block.mlp_fc2_weight);
    copy_to_cuda(block.mlp_fc2_bias);
  }
  
  auto copy_merger_to_cuda = [&copy_to_cuda](Merger& m) {
    copy_to_cuda(m.norm_weight);
    copy_to_cuda(m.norm_bias);
    copy_to_cuda(m.fc1_weight);
    copy_to_cuda(m.fc1_bias);
    copy_to_cuda(m.fc2_weight);
    copy_to_cuda(m.fc2_bias);
  };
  
  copy_merger_to_cuda(merger);
  for (auto& dm : deepstack_mergers) {
    copy_merger_to_cuda(dm);
  }
}

// ============================================================================
// Constructor / Destructor
// ============================================================================

Qwen3VLModel::Qwen3VLModel(base::TokenizerType tokenizer_type, 
                           std::string token_path,
                           std::string model_path)
    : Model(tokenizer_type, base::ModelType::kModelTypeLLama2, 
            std::move(token_path), std::move(model_path), false) {
  vision_layers_ = std::make_unique<Qwen3VLVisionLayers>();
  qwen_layers_ = std::make_unique<Qwen3Layers>();
}

Qwen3VLModel::~Qwen3VLModel() {
  // M-RoPE GPU arrays are now managed by GenerateMRoPEPositionsLayer
  mrope_pos_t_gpu_ = nullptr;
  mrope_pos_h_gpu_ = nullptr;
  mrope_pos_w_gpu_ = nullptr;
  
  // Clean up reusable GPU pixel buffer
  if (pixel_buf_gpu_) {
    cudaFree(pixel_buf_gpu_);
    pixel_buf_gpu_ = nullptr;
  }
  pixel_buf_gpu_capacity_ = 0;
  
  // Clean up mmap
  if (vl_model_data_ && vl_model_data_ != MAP_FAILED) {
    munmap(vl_model_data_, vl_model_file_size_);
    vl_model_data_ = nullptr;
  }
  if (vl_model_fd_ >= 0) {
    close(vl_model_fd_);
    vl_model_fd_ = -1;
  }
}

// ============================================================================
// Initialization
// ============================================================================

base::Status Qwen3VLModel::init(base::DeviceType device_type) {
  using namespace base;
  
  if (token_path_.empty()) {
    return error::PathNotValid(token_path_);
  }
  
  device_type_ = device_type;
  
  if (device_type == DeviceType::kDeviceCUDA) {
    cudaSetDevice(0);
    cuda_config_ = std::make_shared<kernel::CudaConfig>();
    cudaStreamCreate(&cuda_config_->stream);
    
    cublasStatus_t cublas_status = cublasCreate(&cuda_config_->cublas_handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
      return error::InternalError("Failed to create cuBLAS handle.");
    }
    cublasSetStream(cuda_config_->cublas_handle, cuda_config_->stream);
    cublasSetMathMode(cuda_config_->cublas_handle, CUBLAS_DEFAULT_MATH);
    
    // Pre-allocate cuBLAS workspace
    {
      const size_t cublas_workspace_size = 32 * 1024 * 1024;
      void* cublas_workspace = nullptr;
      cudaError_t alloc_err = cudaMalloc(&cublas_workspace, cublas_workspace_size);
      if (alloc_err == cudaSuccess && cublas_workspace) {
        cublasSetWorkspace(cuda_config_->cublas_handle, cublas_workspace, cublas_workspace_size);
        cuda_config_->cublas_workspace = cublas_workspace;
        cuda_config_->cublas_workspace_size = cublas_workspace_size;
      }
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      return error::InternalError("CUDA handle creation failed.");
    }
  }
  
  // Load model from binary file
  Status load_status = load_vl_model_file();
  if (!load_status) {
    return load_status;
  }
  
  // Create encode layer for tokenization
  Status encode_status = create_encode_layer();
  if (!encode_status) {
    return encode_status;
  }
  
  // Move LLM layers to CUDA
  if (device_type == DeviceType::kDeviceCUDA) {
    LOG(INFO) << "Moving LLM layers to CUDA...";
    
    auto move_layer = [this](const std::shared_ptr<op::Layer>& layer) {
      if (auto lp = std::dynamic_pointer_cast<op::LayerParam>(layer)) {
        lp->set_keep_fp16_weights(true);
      }
      layer->set_cuda_config(cuda_config_);
      layer->to_cuda();
    };
    
    auto move_layers = [&move_layer](const std::vector<std::shared_ptr<op::Layer>>& layers,
                                     const char* name) {
      LOG(INFO) << "  Moving " << layers.size() << " " << name << "...";
      for (auto& layer : layers) { move_layer(layer); }
    };
    
    move_layers(qwen_layers_->rmsnorm_layers_, "RMSNorm layers");
    
    if (qwen_layers_->embedding_layer_) {
      LOG(INFO) << "  Moving embedding layer...";
      move_layer(qwen_layers_->embedding_layer_);
    }
    
    move_layers(qwen_layers_->wq_layers_, "Q projections");
    move_layers(qwen_layers_->wk_layers_, "K projections");
    move_layers(qwen_layers_->wv_layers_, "V projections");
    move_layers(qwen_layers_->wo_layers_, "O projections");
    move_layers(qwen_layers_->w1_layers_, "W1 projections");
    move_layers(qwen_layers_->w2_layers_, "W2 projections");
    move_layers(qwen_layers_->w3_layers_, "W3 projections");
    
    if (qwen_layers_->cls_layer_) {
      LOG(INFO) << "  Moving LM head...";
      move_layer(qwen_layers_->cls_layer_);
    }
    
    cudaStreamSynchronize(cuda_config_->stream);
    LOG(INFO) << "Moved all LLM layers to CUDA";
  }
  
  // Create non-parameter layers
  create_nonparam_layers();
  
  // Initialize memory buffers
  LOG(INFO) << "Initializing memory buffers...";
  init_mem();
  
  // Initialize RoPE sin/cos cache
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK_NE(cuda_config_, nullptr);
    CHECK_NE(qwen_layers_->sin_cos_cache_layer_, nullptr);
    qwen_layers_->sin_cos_cache_layer_->forward(config_->head_size_, config_->seq_len_,
                                                get_buffer(ModelBufferType::kSinCache),
                                                get_buffer(ModelBufferType::kCosCache));
    LOG(INFO) << "Initialized RoPE sin/cos cache.";
  }
  
  // Create sampler
  sampler_ = std::make_unique<sampler::ArgmaxSampler>(device_type_);
  
  return error::Success();
}

// ============================================================================
// Model Loading
// ============================================================================

base::Status Qwen3VLModel::load_vl_model_file() {
  int fd = open(model_path_.c_str(), O_RDONLY);
  if (fd == -1) {
    return base::error::PathNotValid(model_path_);
  }
  
  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    close(fd);
    return base::error::ModelParseError("Failed to get file size for " + model_path_);
  }
  
  vl_model_file_size_ = sb.st_size;
  vl_model_fd_ = fd;
  vl_model_data_ = mmap(nullptr, vl_model_file_size_, PROT_READ, MAP_PRIVATE, fd, 0);
  
  if (vl_model_data_ == MAP_FAILED || vl_model_data_ == nullptr) {
    close(fd);
    return base::error::ModelParseError("Failed to mmap model file " + model_path_);
  }
  
  const int8_t* data = static_cast<const int8_t*>(vl_model_data_);
  size_t offset = 0;
  
  // Read header (512 bytes)
  uint32_t magic = *reinterpret_cast<const uint32_t*>(data + offset);
  offset += 4;
  
  if (magic != 0x71773376) {  // "qw3v"
    munmap(vl_model_data_, vl_model_file_size_);
    close(fd);
    return base::error::InvalidArgument("Invalid magic number for Qwen3-VL model");
  }
  
  int32_t version = *reinterpret_cast<const int32_t*>(data + offset);
  offset += 4;
  LOG(INFO) << "Qwen3-VL model version: " << version;
  
  // Vision config
  vl_config_.vision.hidden_size = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  vl_config_.vision.intermediate_size = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  vl_config_.vision.num_heads = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  vl_config_.vision.depth = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  vl_config_.vision.patch_size = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  vl_config_.vision.temporal_patch_size = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  vl_config_.vision.in_channels = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  vl_config_.vision.spatial_merge_size = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  vl_config_.vision.out_hidden_size = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  vl_config_.vision.num_position_embeddings = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  
  // Deepstack indexes
  vl_config_.vision.deepstack_visual_indexes.resize(3);
  for (int i = 0; i < 3; ++i) {
    vl_config_.vision.deepstack_visual_indexes[i] = *reinterpret_cast<const int32_t*>(data + offset);
    offset += 4;
  }
  
  // Text config
  vl_config_.text.hidden_size = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  vl_config_.text.intermediate_size = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  vl_config_.text.num_hidden_layers = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  vl_config_.text.num_attention_heads = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  vl_config_.text.num_key_value_heads = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  vl_config_.text.vocab_size = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  vl_config_.text.max_position_embeddings = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  vl_config_.text.head_dim = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  vl_config_.text.rms_norm_eps = *reinterpret_cast<const float*>(data + offset); offset += 4;
  vl_config_.text.rope_theta = *reinterpret_cast<const float*>(data + offset); offset += 4;
  
  // Special tokens
  vl_config_.special_tokens.image_token_id = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  vl_config_.special_tokens.video_token_id = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  vl_config_.special_tokens.vision_start_token_id = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  vl_config_.special_tokens.vision_end_token_id = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  vl_config_.special_tokens.eos_token_id = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  
  // Flags
  int32_t has_lm_head = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  vl_config_.has_lm_head = (has_lm_head != 0);
  
  // Skip to 512 bytes (header end)
  offset = 512;
  
  LOG(INFO) << "Qwen3-VL config loaded:";
  LOG(INFO) << "  Vision: hidden=" << vl_config_.vision.hidden_size 
            << ", depth=" << vl_config_.vision.depth
            << ", patch=" << vl_config_.vision.patch_size;
  LOG(INFO) << "  LLM: dim=" << vl_config_.text.hidden_size 
            << ", layers=" << vl_config_.text.num_hidden_layers
            << ", heads=" << vl_config_.text.num_attention_heads;
  
  // Create TransformerConfig for base class
  config_ = std::make_unique<TransformerConfig>();
  config_->dim_ = vl_config_.text.hidden_size;
  config_->hidden_dim_ = vl_config_.text.intermediate_size;
  config_->layer_num_ = vl_config_.text.num_hidden_layers;
  config_->head_num_ = vl_config_.text.num_attention_heads;
  config_->kv_head_num_ = vl_config_.text.num_key_value_heads;
  config_->vocab_size_ = vl_config_.text.vocab_size;
  config_->seq_len_ = 8192;
  config_->head_size_ = vl_config_.text.head_dim;
  config_->kv_dim_ = vl_config_.text.num_key_value_heads * vl_config_.text.head_dim;
  config_->kv_mul_ = vl_config_.text.num_attention_heads / vl_config_.text.num_key_value_heads;
  
  // Load Vision Encoder Weights (directly to GPU)
  auto read_fp16_tensor_to_gpu = [&data, &offset](tensor::Tensor& tensor, 
                                                    const std::vector<int>& dims,
                                                    std::shared_ptr<base::DeviceAllocator> alloc) {
    size_t numel = 1;
    for (int d : dims) numel *= d;
    
    tensor = tensor::Tensor(base::DataType::kDataTypeFp16, dims, true, alloc);
    cudaMemcpy(tensor.ptr<void>(), data + offset, numel * sizeof(uint16_t), cudaMemcpyHostToDevice);
    offset += numel * sizeof(uint16_t);
  };
  
  std::shared_ptr<base::DeviceAllocator> alloc_gpu = base::CUDADeviceAllocatorFactory::get_instance();
  
  int vit_hidden = vl_config_.vision.hidden_size;
  int vit_intermediate = vl_config_.vision.intermediate_size;
  int patch_size = vl_config_.vision.patch_size;
  int temporal_patch = vl_config_.vision.temporal_patch_size;
  int in_channels = vl_config_.vision.in_channels;
  int num_pos_embed = vl_config_.vision.num_position_embeddings;
  int vit_depth = vl_config_.vision.depth;
  int spatial_merge = vl_config_.vision.spatial_merge_size;
  int out_hidden = vl_config_.vision.out_hidden_size;
  int merged_hidden = vit_hidden * spatial_merge * spatial_merge;
  
  LOG(INFO) << "Loading vision encoder weights...";
  
  // Patch embedding
  read_fp16_tensor_to_gpu(vision_layers_->patch_embed_weight, 
                           {vit_hidden, in_channels, temporal_patch, patch_size, patch_size}, alloc_gpu);
  read_fp16_tensor_to_gpu(vision_layers_->patch_embed_bias, {vit_hidden}, alloc_gpu);
  
  // Position embedding
  read_fp16_tensor_to_gpu(vision_layers_->pos_embed_weight, {num_pos_embed, vit_hidden}, alloc_gpu);
  
  // Transformer blocks
  vision_layers_->blocks.resize(vit_depth);
  for (int i = 0; i < vit_depth; ++i) {
    auto& block = vision_layers_->blocks[i];
    
    read_fp16_tensor_to_gpu(block.norm1_weight, {vit_hidden}, alloc_gpu);
    read_fp16_tensor_to_gpu(block.norm1_bias, {vit_hidden}, alloc_gpu);
    read_fp16_tensor_to_gpu(block.norm2_weight, {vit_hidden}, alloc_gpu);
    read_fp16_tensor_to_gpu(block.norm2_bias, {vit_hidden}, alloc_gpu);
    
    read_fp16_tensor_to_gpu(block.qkv_weight, {3 * vit_hidden, vit_hidden}, alloc_gpu);
    read_fp16_tensor_to_gpu(block.qkv_bias, {3 * vit_hidden}, alloc_gpu);
    read_fp16_tensor_to_gpu(block.proj_weight, {vit_hidden, vit_hidden}, alloc_gpu);
    read_fp16_tensor_to_gpu(block.proj_bias, {vit_hidden}, alloc_gpu);
    
    read_fp16_tensor_to_gpu(block.mlp_fc1_weight, {vit_intermediate, vit_hidden}, alloc_gpu);
    read_fp16_tensor_to_gpu(block.mlp_fc1_bias, {vit_intermediate}, alloc_gpu);
    read_fp16_tensor_to_gpu(block.mlp_fc2_weight, {vit_hidden, vit_intermediate}, alloc_gpu);
    read_fp16_tensor_to_gpu(block.mlp_fc2_bias, {vit_hidden}, alloc_gpu);
    
    if (i % 9 == 0 || i == vit_depth - 1) {
      LOG(INFO) << "  Loaded vision block " << i;
    }
  }
  
  // Main merger
  auto load_main_merger = [&](Qwen3VLVisionLayers::Merger& m) {
    read_fp16_tensor_to_gpu(m.norm_weight, {vit_hidden}, alloc_gpu);
    read_fp16_tensor_to_gpu(m.norm_bias, {vit_hidden}, alloc_gpu);
    read_fp16_tensor_to_gpu(m.fc1_weight, {merged_hidden, merged_hidden}, alloc_gpu);
    read_fp16_tensor_to_gpu(m.fc1_bias, {merged_hidden}, alloc_gpu);
    read_fp16_tensor_to_gpu(m.fc2_weight, {out_hidden, merged_hidden}, alloc_gpu);
    read_fp16_tensor_to_gpu(m.fc2_bias, {out_hidden}, alloc_gpu);
  };
  
  auto load_deepstack_merger = [&](Qwen3VLVisionLayers::Merger& m) {
    read_fp16_tensor_to_gpu(m.norm_weight, {merged_hidden}, alloc_gpu);
    read_fp16_tensor_to_gpu(m.norm_bias, {merged_hidden}, alloc_gpu);
    read_fp16_tensor_to_gpu(m.fc1_weight, {merged_hidden, merged_hidden}, alloc_gpu);
    read_fp16_tensor_to_gpu(m.fc1_bias, {merged_hidden}, alloc_gpu);
    read_fp16_tensor_to_gpu(m.fc2_weight, {out_hidden, merged_hidden}, alloc_gpu);
    read_fp16_tensor_to_gpu(m.fc2_bias, {out_hidden}, alloc_gpu);
  };
  
  load_main_merger(vision_layers_->merger);
  LOG(INFO) << "  Loaded main merger";
  
  vision_layers_->deepstack_mergers.resize(vl_config_.vision.deepstack_visual_indexes.size());
  for (size_t i = 0; i < vision_layers_->deepstack_mergers.size(); ++i) {
    load_deepstack_merger(vision_layers_->deepstack_mergers[i]);
  }
  LOG(INFO) << "  Loaded " << vision_layers_->deepstack_mergers.size() << " deepstack mergers";
  
  // Load Language Model Weights (using mmap pointers)
  LOG(INFO) << "Loading language model weights...";
  
  int llm_dim = vl_config_.text.hidden_size;
  int llm_intermediate = vl_config_.text.intermediate_size;
  int llm_layers = vl_config_.text.num_hidden_layers;
  int vocab_size = vl_config_.text.vocab_size;
  int head_dim = vl_config_.text.head_dim;
  int kv_heads = vl_config_.text.num_key_value_heads;
  int q_heads = vl_config_.text.num_attention_heads;
  int kv_dim = kv_heads * head_dim;
  int q_dim = q_heads * head_dim;
  
  auto cpu_device_type = base::DeviceType::kDeviceCPU;
  
  // RMSNorm weights
  for (int i = 0; i < llm_layers; ++i) {
    auto rms_layer = std::make_shared<op::RmsNormLayer>(device_type_, llm_dim);
    rms_layer->set_weight_fp16(0, {llm_dim}, data + offset, cpu_device_type);
    qwen_layers_->rmsnorm_layers_.push_back(rms_layer);
    offset += llm_dim * sizeof(uint16_t);
  }
  
  for (int i = 0; i < llm_layers; ++i) {
    auto rms_layer = std::make_shared<op::RmsNormLayer>(device_type_, llm_dim);
    rms_layer->set_weight_fp16(0, {llm_dim}, data + offset, cpu_device_type);
    qwen_layers_->rmsnorm_layers_.push_back(rms_layer);
    offset += llm_dim * sizeof(uint16_t);
  }
  
  // Final norm
  {
    auto rms_layer = std::make_shared<op::RmsNormLayer>(device_type_, llm_dim);
    rms_layer->set_weight_fp16(0, {llm_dim}, data + offset, cpu_device_type);
    qwen_layers_->rmsnorm_layers_.push_back(rms_layer);
    offset += llm_dim * sizeof(uint16_t);
  }
  LOG(INFO) << "  Loaded " << qwen_layers_->rmsnorm_layers_.size() << " RMSNorm layers";
  
  // Token embeddings
  {
    auto embedding_layer = std::make_shared<op::EmbeddingLayer>(
        device_type_, llm_dim, config_->seq_len_, vocab_size);
    size_t embed_size = static_cast<size_t>(vocab_size) * llm_dim;
    embedding_layer->set_weight_fp16(0, {vocab_size, llm_dim}, data + offset, cpu_device_type);
    qwen_layers_->embedding_layer_ = embedding_layer;
    offset += embed_size * sizeof(uint16_t);
  }
  LOG(INFO) << "  Loaded token embeddings: [" << vocab_size << ", " << llm_dim << "]";
  
  // Q, K, V, O projection weights
  auto load_proj_weights = [&](std::vector<std::shared_ptr<op::Layer>>& layers, 
                               int out_dim, int in_dim) {
    size_t weight_size = static_cast<size_t>(out_dim) * in_dim;
    for (int i = 0; i < llm_layers; ++i) {
      auto matmul = std::make_shared<op::MatmulLayer>(device_type_, out_dim, in_dim, false);
      matmul->set_weight_fp16(0, {out_dim, in_dim}, data + offset, cpu_device_type);
      layers.push_back(matmul);
      offset += weight_size * sizeof(uint16_t);
    }
  };
  
  load_proj_weights(qwen_layers_->wq_layers_, q_dim, llm_dim);
  LOG(INFO) << "  Loaded Q projections";
  load_proj_weights(qwen_layers_->wk_layers_, kv_dim, llm_dim);
  LOG(INFO) << "  Loaded K projections";
  load_proj_weights(qwen_layers_->wv_layers_, kv_dim, llm_dim);
  LOG(INFO) << "  Loaded V projections";
  load_proj_weights(qwen_layers_->wo_layers_, llm_dim, q_dim);
  LOG(INFO) << "  Loaded O projections";
  
  // FFN weights
  load_proj_weights(qwen_layers_->w1_layers_, llm_intermediate, llm_dim);
  LOG(INFO) << "  Loaded gate projections";
  load_proj_weights(qwen_layers_->w2_layers_, llm_dim, llm_intermediate);
  LOG(INFO) << "  Loaded down projections";
  load_proj_weights(qwen_layers_->w3_layers_, llm_intermediate, llm_dim);
  LOG(INFO) << "  Loaded up projections";
  
  // LM head
  if (vl_config_.has_lm_head) {
    auto cls_layer = std::make_shared<op::MatmulLayer>(device_type_, vocab_size, llm_dim, false);
    size_t lm_head_size = static_cast<size_t>(vocab_size) * llm_dim;
    cls_layer->set_weight_fp16(0, {vocab_size, llm_dim}, data + offset, cpu_device_type);
    qwen_layers_->cls_layer_ = cls_layer;
    offset += lm_head_size * sizeof(uint16_t);
    LOG(INFO) << "  Loaded LM head: [" << vocab_size << ", " << llm_dim << "]";
  }
  
  // q_norm and k_norm
  for (int i = 0; i < llm_layers; ++i) {
    auto q_norm_layer = std::make_shared<op::RmsNormLayer>(device_type_, head_dim);
    q_norm_layer->set_weight_fp16(0, {head_dim}, data + offset, cpu_device_type);
    qwen_layers_->rmsnorm_layers_.push_back(q_norm_layer);
    offset += head_dim * sizeof(uint16_t);
  }
  
  for (int i = 0; i < llm_layers; ++i) {
    auto k_norm_layer = std::make_shared<op::RmsNormLayer>(device_type_, head_dim);
    k_norm_layer->set_weight_fp16(0, {head_dim}, data + offset, cpu_device_type);
    qwen_layers_->rmsnorm_layers_.push_back(k_norm_layer);
    offset += head_dim * sizeof(uint16_t);
  }
  LOG(INFO) << "  Loaded q_norm/k_norm: " << 2 * llm_layers << " tensors";
  
  LOG(INFO) << "Model loading complete! Total offset: " << offset << " bytes";
  
  return base::error::Success();
}

// ============================================================================
// Memory Initialization
// ============================================================================

void Qwen3VLModel::init_mem() {
  std::shared_ptr<base::DeviceAllocator> alloc;
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    alloc = base::CPUDeviceAllocatorFactory::get_instance();
  } else {
    alloc = base::CUDADeviceAllocatorFactory::get_instance();
  }
  
  std::shared_ptr<base::DeviceAllocator> alloc_cpu =
      base::CPUDeviceAllocatorFactory::get_instance();
  
  base::DataType activation_dtype = base::DataType::kDataTypeFp16;
  LOG(INFO) << "Using FP16 activation buffers for Qwen3-VL";
  
  int32_t model_dim = config_->dim_;
  int32_t intermediate_dim = config_->hidden_dim_;
  
  // Input token and embedding buffers
  tensor::Tensor input_tokens(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  tensor::Tensor input_embeddings(activation_dtype, 1, model_dim, true, alloc);
  
  CHECK(insert_buffer(ModelBufferType::kInputTokens, input_tokens));
  CHECK(insert_buffer(ModelBufferType::kInputEmbeddings, input_embeddings));
  
  // RoPE sin/cos cache
  tensor::Tensor sin_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_,
                           true, alloc);
  tensor::Tensor cos_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_,
                           true, alloc);
  CHECK(insert_buffer(ModelBufferType::kSinCache, sin_cache));
  CHECK(insert_buffer(ModelBufferType::kCosCache, cos_cache));
  
  // Intermediate buffers
  tensor::Tensor rms_output(activation_dtype, model_dim, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kOutputRMSNorm, rms_output));
  CHECK(insert_buffer(ModelBufferType::kOutputMHA, rms_output));
  CHECK(insert_buffer(ModelBufferType::kW2Output, rms_output));
  CHECK(insert_buffer(ModelBufferType::kFFNRMSNorm, rms_output));
  
  tensor::Tensor w1_output(activation_dtype, intermediate_dim, true, alloc);
  tensor::Tensor w3_output(activation_dtype, intermediate_dim, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kW1Output, w1_output));
  CHECK(insert_buffer(ModelBufferType::kW3Output, w3_output));
  
  // KV cache
  tensor::Tensor key_cache(activation_dtype, config_->layer_num_, config_->seq_len_,
                           config_->kv_dim_, true, alloc);
  tensor::Tensor value_cache(activation_dtype, config_->layer_num_, config_->seq_len_,
                             config_->kv_dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kKeyCache, key_cache));
  CHECK(insert_buffer(ModelBufferType::kValueCache, value_cache));
  
  tensor::Tensor query(activation_dtype, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kQuery, query));
  
  tensor::Tensor decode_input(activation_dtype, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kDecodeInput, decode_input));
  
  tensor::Tensor pos_tensor(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  CHECK(insert_buffer(ModelBufferType::kInputPos, pos_tensor));
  
  // GPU position buffers for CUDA Graph
  tensor::Tensor pos_tensor_gpu(base::DataType::kDataTypeInt32, 1, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kInputPosGPU, pos_tensor_gpu));
  
  tensor::Tensor kv_cache_pos_gpu(base::DataType::kDataTypeInt32, 1, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kKVCachePosGPU, kv_cache_pos_gpu));
  
  // Temporary K/V buffers for CUDA Graph
  tensor::Tensor temp_key(activation_dtype, config_->kv_dim_, true, alloc);
  tensor::Tensor temp_value(activation_dtype, config_->kv_dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kTempKey, temp_key));
  CHECK(insert_buffer(ModelBufferType::kTempValue, temp_value));
  
  // Pinned buffers
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    auto alloc_pinned = base::CPUPinnedAllocatorFactory::get_instance();
    
    tensor::Tensor pos_pinned(base::DataType::kDataTypeInt32, 1, true, alloc_pinned);
    tensor::Tensor kv_cache_pos_pinned(base::DataType::kDataTypeInt32, 1, true, alloc_pinned);
    CHECK(insert_buffer(ModelBufferType::kInputPosPinned, pos_pinned));
    CHECK(insert_buffer(ModelBufferType::kKVCachePosPinned, kv_cache_pos_pinned));
    
    tensor::Tensor argmax_output(base::DataType::kDataTypeInt32, 2, true, alloc);
    tensor::Tensor argmax_output_pinned(base::DataType::kDataTypeInt32, 2, true, alloc_pinned);
    CHECK(insert_buffer(ModelBufferType::kArgmaxOutput, argmax_output));
    CHECK(insert_buffer(ModelBufferType::kArgmaxOutputPinned, argmax_output_pinned));
  }
  
  // Attention scores
  tensor::Tensor attn(base::DataType::kDataTypeFp32, config_->head_num_, config_->seq_len_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kScoreStorage, attn));
  
  tensor::Tensor attn_output(activation_dtype, model_dim, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kAttnOutput, attn_output));
  
  // Forward output
  int vocab_size = vl_config_.text.vocab_size;
  tensor::Tensor forward_output(base::DataType::kDataTypeFp32, vocab_size, true, alloc);
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    tensor::Tensor forward_output_cpu(base::DataType::kDataTypeFp32, vocab_size, true, alloc_cpu);
    CHECK(insert_buffer(ModelBufferType::kForwardOutputCPU, forward_output_cpu));
  }
  CHECK(insert_buffer(ModelBufferType::kForwardOutput, forward_output));
  
  // Pre-allocate GPU pixel buffer
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    constexpr size_t kDefaultPixelBufSize = 32 * 1024 * 1024;
    cudaMalloc(&pixel_buf_gpu_, kDefaultPixelBufSize);
    pixel_buf_gpu_capacity_ = kDefaultPixelBufSize;
    LOG(INFO) << "Pre-allocated GPU pixel buffer: " << kDefaultPixelBufSize / (1024*1024) << " MB";
  }
  
  LOG(INFO) << "Memory initialization complete for Qwen3-VL.";
}

// ============================================================================
// Layer Creation
// ============================================================================

base::Status Qwen3VLModel::create_layers() {
  return base::error::Success();
}

void Qwen3VLModel::create_param_layers() {
  // Already created during load_vl_model_file
}

void Qwen3VLModel::create_nonparam_layers() {
  create_llm_nonparam_layers();
  create_vl_nonparam_layers();
  create_vision_nonparam_layers();
}

void Qwen3VLModel::create_llm_nonparam_layers() {
  qwen_layers_->rope_layer_ = std::make_shared<op::RoPELayer>(
      device_type_, config_->dim_, config_->kv_dim_, config_->head_size_);
  qwen_layers_->rope_layer_->set_cuda_config(cuda_config_);
  
  qwen_layers_->mha_layer_ = std::make_shared<op::MultiHeadAttention>(
      device_type_, 0, config_->kv_mul_, config_->kv_dim_, config_->seq_len_, 
      config_->head_num_, config_->head_size_);
  qwen_layers_->mha_layer_->set_cuda_config(cuda_config_);
  
  qwen_layers_->add_layer_ = std::make_shared<op::VecAddLayer>(device_type_);
  qwen_layers_->add_layer_->set_cuda_config(cuda_config_);
  
  qwen_layers_->swiglu_layer_ = std::make_shared<op::SwiGLULayer>(
      device_type_, config_->hidden_dim_);
  qwen_layers_->swiglu_layer_->set_cuda_config(cuda_config_);

  qwen_layers_->flash_attention_decode_layer_ =
      std::make_shared<op::FlashAttentionDecodeLayer>(device_type_);
  qwen_layers_->flash_attention_decode_layer_->set_cuda_config(cuda_config_);
  
  qwen_layers_->flash_attention_prefill_layer_ =
      std::make_shared<op::FlashAttentionPrefillLayer>(device_type_);
  qwen_layers_->flash_attention_prefill_layer_->set_cuda_config(cuda_config_);
  
  qwen_layers_->kv_cache_key_layer_ = std::make_shared<op::KVCacheLayer>(device_type_);
  qwen_layers_->kv_cache_key_layer_->set_cuda_config(cuda_config_);
  
  qwen_layers_->kv_cache_value_layer_ = std::make_shared<op::KVCacheLayer>(device_type_);
  qwen_layers_->kv_cache_value_layer_->set_cuda_config(cuda_config_);
  
  qwen_layers_->fused_ffn_layer_ = std::make_shared<op::FusedFFNLayer>(
      device_type_, config_->dim_, config_->hidden_dim_, true, false);
  qwen_layers_->fused_ffn_layer_->set_cuda_config(cuda_config_);
  
  qwen_layers_->rope_gpu_pos_layer_ = std::make_shared<op::RoPEGpuPosLayer>(device_type_);
  qwen_layers_->rope_gpu_pos_layer_->set_cuda_config(cuda_config_);
  
  qwen_layers_->batched_rope_layer_ = std::make_shared<op::BatchedRoPELayer>(device_type_);
  qwen_layers_->batched_rope_layer_->set_cuda_config(cuda_config_);
  
  qwen_layers_->batched_add_layer_ = std::make_shared<op::BatchedAddLayer>(device_type_);
  qwen_layers_->batched_add_layer_->set_cuda_config(cuda_config_);
  
  qwen_layers_->batched_swiglu_layer_ = std::make_shared<op::BatchedSwiGLULayer>(device_type_);
  qwen_layers_->batched_swiglu_layer_->set_cuda_config(cuda_config_);
  
  qwen_layers_->sin_cos_cache_layer_ = std::make_shared<op::SinCosCacheLayer>(device_type_);
  qwen_layers_->sin_cos_cache_layer_->set_cuda_config(cuda_config_);
}

void Qwen3VLModel::create_vl_nonparam_layers() {
  qwen_layers_->mrope_layer_ = std::make_shared<op::MRoPELayer>(device_type_);
  qwen_layers_->mrope_layer_->set_cuda_config(cuda_config_);
  
  qwen_layers_->mrope_gpu_pos_layer_ = std::make_shared<op::MRoPEGpuPosLayer>(device_type_);
  qwen_layers_->mrope_gpu_pos_layer_->set_cuda_config(cuda_config_);
  
  qwen_layers_->batched_mrope_layer_ = std::make_shared<op::BatchedMRoPELayer>(device_type_);
  qwen_layers_->batched_mrope_layer_->set_cuda_config(cuda_config_);
  
  qwen_layers_->fused_kv_cache_update_layer_ = std::make_shared<op::FusedKVCacheUpdateLayer>(device_type_);
  qwen_layers_->fused_kv_cache_update_layer_->set_cuda_config(cuda_config_);
  
  qwen_layers_->rmsnorm_dim_layer_ = std::make_shared<op::RMSNormDimLayer>(device_type_);
  qwen_layers_->rmsnorm_dim_layer_->set_cuda_config(cuda_config_);
  
  qwen_layers_->copy_to_kv_cache_layer_ = std::make_shared<op::CopyToKVCacheLayer>(device_type_);
  qwen_layers_->copy_to_kv_cache_layer_->set_cuda_config(cuda_config_);
  
  qwen_layers_->flash_attention_decode_gpu_pos_layer_ = std::make_shared<op::FlashAttentionDecodeGpuPosLayer>(device_type_);
  qwen_layers_->flash_attention_decode_gpu_pos_layer_->set_cuda_config(cuda_config_);
}

void Qwen3VLModel::create_vision_nonparam_layers() {
  vision_vl_layers_.extract_patches_layer_ = std::make_shared<op::ExtractPatchesLayer>(device_type_);
  vision_vl_layers_.extract_patches_layer_->set_cuda_config(cuda_config_);
  
  vision_vl_layers_.bias_add_residual_layer_ = std::make_shared<op::BiasAddResidualLayer>(device_type_);
  vision_vl_layers_.bias_add_residual_layer_->set_cuda_config(cuda_config_);
  
  vision_vl_layers_.pos_embed_interpolate_layer_ = std::make_shared<op::PosEmbedInterpolateLayer>(device_type_);
  vision_vl_layers_.pos_embed_interpolate_layer_->set_cuda_config(cuda_config_);
  
  vision_vl_layers_.layernorm_with_bias_layer_ = std::make_shared<op::LayerNormWithBiasLayer>(device_type_);
  vision_vl_layers_.layernorm_with_bias_layer_->set_cuda_config(cuda_config_);
  
  vision_vl_layers_.fused_split_rope_transpose_layer_ = std::make_shared<op::FusedSplitRopeTransposeLayer>(device_type_);
  vision_vl_layers_.fused_split_rope_transpose_layer_->set_cuda_config(cuda_config_);
  
  vision_vl_layers_.vision_attention_layer_ = std::make_shared<op::VisionAttentionLayer>(device_type_);
  vision_vl_layers_.vision_attention_layer_->set_cuda_config(cuda_config_);
  
  vision_vl_layers_.vision_mlp_layer_ = std::make_shared<op::VisionMLPLayer>(device_type_);
  vision_vl_layers_.vision_mlp_layer_->set_cuda_config(cuda_config_);
  
  vision_vl_layers_.spatial_merge_layer_ = std::make_shared<op::SpatialMergeLayer>(device_type_);
  vision_vl_layers_.spatial_merge_layer_->set_cuda_config(cuda_config_);
  
  vision_vl_layers_.vision_merger_mlp_layer_ = std::make_shared<op::VisionMergerMLPLayer>(device_type_);
  vision_vl_layers_.vision_merger_mlp_layer_->set_cuda_config(cuda_config_);
  
  vision_vl_layers_.fused_multimodal_embed_layer_ = std::make_shared<op::FusedMultimodalEmbedLayer>(device_type_);
  vision_vl_layers_.fused_multimodal_embed_layer_->set_cuda_config(cuda_config_);
  
  vision_vl_layers_.fused_normalize_patches_layer_ = std::make_shared<op::FusedNormalizePatchesLayer>(device_type_);
  vision_vl_layers_.fused_normalize_patches_layer_->set_cuda_config(cuda_config_);
  
  vision_vl_layers_.causal_softmax_layer_ = std::make_shared<op::CausalSoftmaxLayer>(device_type_);
  vision_vl_layers_.causal_softmax_layer_->set_cuda_config(cuda_config_);
  
  // New refactored layers
  vision_vl_layers_.load_image_layer_ = std::make_shared<op::LoadImageLayer>(base::DeviceType::kDeviceCPU);
  
  vision_vl_layers_.smart_resize_layer_ = std::make_shared<op::SmartResizeLayer>(base::DeviceType::kDeviceCPU);
  
  vision_vl_layers_.vision_rotary_emb_layer_ = std::make_shared<op::VisionRotaryEmbLayer>(base::DeviceType::kDeviceCPU);
  
  vision_vl_layers_.generate_mrope_positions_layer_ = std::make_shared<op::GenerateMRoPEPositionsLayer>(device_type_);
  vision_vl_layers_.generate_mrope_positions_layer_->set_cuda_config(cuda_config_);
  
  vision_vl_layers_.vision_patch_embed_layer_ = std::make_shared<op::VisionPatchEmbedLayer>(device_type_);
  vision_vl_layers_.vision_patch_embed_layer_->set_cuda_config(cuda_config_);
  
  vision_vl_layers_.batched_gemm_layer_ = std::make_shared<op::BatchedGemmLayer>(device_type_);
  vision_vl_layers_.batched_gemm_layer_->set_cuda_config(cuda_config_);
}

void Qwen3VLModel::create_param_quant_layers() {
  // Not used for FP16 model
}

}  // namespace model

#endif  // QWEN3_VL_SUPPORT
