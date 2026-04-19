/**
 * @file qwen3_5.cpp
 * @brief Qwen3.5-9B Hybrid Vision-Language Model Implementation
 *
 * Hybrid architecture: 24 GDN (Gated Delta Net) linear attention layers +
 * 8 full attention layers with output gating. Reuses Qwen3-VL's ViT pipeline.
 */

#ifdef QWEN3_VL_SUPPORT
#include "model/qwen3_5.h"
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cmath>
#include <algorithm>
#include <chrono>

#include "op/matmul.h"
#include "op/rmsnorm.h"
#include "op/flash_attention.h"
#include "op/vision_layers.h"
#include "op/fused_ffn.h"
#include "op/add.h"
#include "op/gdn_layers.h"
#include "sampler/argmax_sampler.h"

using namespace base;

namespace model {

// ==========================================================================
// Construction / Destruction
// ==========================================================================

Qwen35Model::Qwen35Model(base::TokenizerType tokenizer_type,
                          std::string token_path,
                          std::string model_path)
    : Qwen3VLModel(tokenizer_type, std::move(token_path), std::move(model_path)) {
  linear_attn_weights_ = std::make_unique<LinearAttnWeights>();
}

Qwen35Model::~Qwen35Model() = default;

// ==========================================================================
// Type index mapping
// ==========================================================================

int Qwen35Model::full_attn_type_idx(int layer_idx) const {
  int idx = 0;
  for (auto li : q35_config_.full_attn_layer_indices) {
    if (li == layer_idx) return idx;
    ++idx;
  }
  LOG(FATAL) << "Layer " << layer_idx << " is not a full attention layer!";
  return -1;
}

int Qwen35Model::linear_attn_type_idx(int layer_idx) const {
  int idx = 0;
  for (int i = 0; i < q35_config_.num_hidden_layers; ++i) {
    if (q35_config_.is_full_attn_layer(i)) continue;
    if (i == layer_idx) return idx;
    ++idx;
  }
  LOG(FATAL) << "Layer " << layer_idx << " is not a linear attention layer!";
  return -1;
}

// ==========================================================================
// Initialization
// ==========================================================================

base::Status Qwen35Model::init(base::DeviceType device_type) {
  device_type_ = device_type;

  if (device_type == DeviceType::kDeviceCUDA) {
    cudaSetDevice(0);
    cuda_config_ = std::make_shared<kernel::CudaConfig>();
    cudaStreamCreate(&cuda_config_->stream);
    cublasCreate(&cuda_config_->cublas_handle);
    cublasSetStream(cuda_config_->cublas_handle, cuda_config_->stream);
    cublasSetMathMode(cuda_config_->cublas_handle, CUBLAS_DEFAULT_MATH);

    const size_t cublas_ws = 32 * 1024 * 1024;
    void* ws = nullptr;
    if (cudaMalloc(&ws, cublas_ws) == cudaSuccess && ws) {
      cublasSetWorkspace(cuda_config_->cublas_handle, ws, cublas_ws);
      cuda_config_->cublas_workspace = ws;
      cuda_config_->cublas_workspace_size = cublas_ws;
    }
  }

  // Load model
  auto status = load_q35_model_file();
  if (!status) return status;

  // Create tokenizer
  status = create_encode_layer();
  if (!status) return status;

  // Move LLM layers to CUDA
  if (device_type == DeviceType::kDeviceCUDA) {
    // Qwen3.5 RMSNorm uses (1.0 + weight) formula, so add 1.0 to all regular RMSNorm weights
    // This applies to: input_layernorm (0..31), post_attention_layernorm (32..63), final_norm (64),
    //                   q_norm (65..72), k_norm (73..80)
    // The gated RMSNorm (GDN norm) stores FP32 weights separately, NOT in rmsnorm_layers_
    {
      int total_rms = qwen_layers_->rmsnorm_layers_.size();
      LOG(INFO) << "Adding 1.0 to " << total_rms << " regular RMSNorm weights (Qwen3.5 offset formula)";
      for (int i = 0; i < total_rms; ++i) {
        auto lp = std::dynamic_pointer_cast<op::LayerParam>(qwen_layers_->rmsnorm_layers_[i]);
        if (!lp) continue;
        auto& w = lp->get_weight(0);
        int numel = w.size();
        half* wptr = w.ptr<half>();
        for (int j = 0; j < numel; ++j) {
          wptr[j] = __float2half(__half2float(wptr[j]) + 1.0f);
        }
      }
    }

    LOG(INFO) << "Moving layers to CUDA...";
    auto move_layer = [this](const std::shared_ptr<op::Layer>& layer) {
      if (auto lp = std::dynamic_pointer_cast<op::LayerParam>(layer))
        lp->set_keep_fp16_weights(true);
      layer->set_cuda_config(cuda_config_);
      layer->to_cuda();
    };
    auto move_layers = [&](const std::vector<std::shared_ptr<op::Layer>>& layers, const char* n) {
      for (auto& l : layers) move_layer(l);
      LOG(INFO) << "  Moved " << layers.size() << " " << n;
    };

    move_layers(qwen_layers_->rmsnorm_layers_, "RMSNorm");
    if (qwen_layers_->embedding_layer_) move_layer(qwen_layers_->embedding_layer_);
    move_layers(qwen_layers_->wq_layers_, "Q proj (full attn, includes gate)");
    move_layers(qwen_layers_->wk_layers_, "K proj (full attn)");
    move_layers(qwen_layers_->wv_layers_, "V proj (full attn)");
    move_layers(qwen_layers_->wo_layers_, "O proj (full attn)");
    move_layers(qwen_layers_->w1_layers_, "gate proj (all layers)");
    move_layers(qwen_layers_->w2_layers_, "down proj (all layers)");
    move_layers(qwen_layers_->w3_layers_, "up proj (all layers)");
    if (qwen_layers_->cls_layer_) move_layer(qwen_layers_->cls_layer_);

    // Move linear attention layers
    for (auto& la : linear_attn_weights_->layers) {
      move_layer(la.in_proj_qkv);
      move_layer(la.in_proj_z);
      move_layer(la.in_proj_a);
      move_layer(la.in_proj_b);
      move_layer(la.out_proj);
    }
    LOG(INFO) << "  Moved " << linear_attn_weights_->layers.size() << " linear attention layers";

    cudaStreamSynchronize(cuda_config_->stream);
  }

  // Create non-param layers (reuse Qwen3VL's vision + LLM layers)
  create_nonparam_layers();
  create_q35_nonparam_layers();

  // Init memory
  init_q35_mem();

  // Init RoPE sin/cos cache with Qwen3.5's rope_theta (10000000)
  if (device_type_ == DeviceType::kDeviceCUDA) {
    int head_size = q35_config_.head_dim;  // 256
    int max_seq_len = config_->seq_len_;
    float rope_theta = q35_config_.rope_theta;
    int partial_dim = q35_config_.partial_rope_dim();  // 64
    int num_pairs = partial_dim / 2;  // 32
    
    // Compute sin/cos cache on CPU, then upload to GPU
    // Cache layout: [max_seq_len, num_pairs=32]
    // freq[i] = 1/(theta^(2*i/partial_dim)) for i=0..31
    std::vector<float> sin_cache_cpu(num_pairs * max_seq_len, 0.0f);
    std::vector<float> cos_cache_cpu(num_pairs * max_seq_len, 0.0f);
    
    for (int i = 0; i < num_pairs; ++i) {
      float freq = 1.0f / powf(rope_theta, static_cast<float>(2 * i) / static_cast<float>(partial_dim));
      for (int pos = 0; pos < max_seq_len; ++pos) {
        float val = static_cast<float>(pos) * freq;
        sin_cache_cpu[pos * num_pairs + i] = sinf(val);
        cos_cache_cpu[pos * num_pairs + i] = cosf(val);
      }
    }
    
    auto& sin_cache = get_buffer(ModelBufferType::kSinCache);
    auto& cos_cache = get_buffer(ModelBufferType::kCosCache);
    // Only fill the first num_pairs*max_seq_len elements of the larger buffer
    cudaMemcpy(sin_cache.ptr<float>(), sin_cache_cpu.data(), 
               num_pairs * max_seq_len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cos_cache.ptr<float>(), cos_cache_cpu.data(), 
               num_pairs * max_seq_len * sizeof(float), cudaMemcpyHostToDevice);
    LOG(INFO) << "Initialized RoPE sin/cos cache (theta=" << rope_theta 
              << ", partial_dim=" << partial_dim << ", num_pairs=" << num_pairs << ")";
  }

  sampler_ = std::make_unique<sampler::ArgmaxSampler>(device_type_);
  return error::Success();
}

// ==========================================================================
// Model File Loading (binary format)
// ==========================================================================

base::Status Qwen35Model::load_q35_model_file() {
  int fd = open(model_path_.c_str(), O_RDONLY);
  if (fd == -1) return base::error::PathNotValid(model_path_);

  struct stat sb;
  if (fstat(fd, &sb) == -1) { close(fd); return base::error::ModelParseError("fstat failed"); }

  vl_model_file_size_ = sb.st_size;
  vl_model_fd_ = fd;
  vl_model_data_ = mmap(nullptr, vl_model_file_size_, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
  if (vl_model_data_ == MAP_FAILED) { close(fd); return base::error::ModelParseError("mmap failed"); }

  const int8_t* data = static_cast<const int8_t*>(vl_model_data_);
  size_t offset = 0;

  // --- Header ---
  uint32_t magic = *reinterpret_cast<const uint32_t*>(data + offset); offset += 4;
  if (magic != 0x71333539) {
    munmap(vl_model_data_, vl_model_file_size_); close(fd);
    return base::error::InvalidArgument("Invalid magic for Qwen3.5 (expected 0x71333539)");
  }

  int32_t version = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  LOG(INFO) << "Qwen3.5 model version: " << version;

  // Vision config (same as Qwen3-VL)
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

  // Deepstack (3 padding zeros for Qwen3.5)
  vl_config_.vision.deepstack_visual_indexes.clear();
  for (int i = 0; i < 3; ++i) {
    int32_t idx = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
    if (idx > 0) vl_config_.vision.deepstack_visual_indexes.push_back(idx);
  }

  // Text config
  q35_config_.hidden_size = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  q35_config_.intermediate_size = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  q35_config_.num_hidden_layers = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  q35_config_.num_attention_heads = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  q35_config_.num_key_value_heads = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  q35_config_.vocab_size = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  q35_config_.max_position_embeddings = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  q35_config_.head_dim = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  q35_config_.rms_norm_eps = *reinterpret_cast<const float*>(data + offset); offset += 4;
  q35_config_.rope_theta = *reinterpret_cast<const float*>(data + offset); offset += 4;

  // Special tokens
  vl_config_.special_tokens.image_token_id = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  vl_config_.special_tokens.video_token_id = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  vl_config_.special_tokens.vision_start_token_id = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  vl_config_.special_tokens.vision_end_token_id = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  vl_config_.special_tokens.eos_token_id = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;

  // Flags
  int32_t has_lm_head = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  vl_config_.has_lm_head = (has_lm_head != 0);

  // Hybrid attention config
  int32_t num_full_attn = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  q35_config_.full_attn_layer_indices.clear();
  for (int i = 0; i < 8; ++i) {
    int32_t idx = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
    if (i < num_full_attn && idx >= 0) q35_config_.full_attn_layer_indices.push_back(idx);
  }

  // Linear attention config
  q35_config_.linear_conv_kernel_dim = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  q35_config_.linear_key_head_dim = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  q35_config_.linear_num_key_heads = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  q35_config_.linear_num_value_heads = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  q35_config_.linear_value_head_dim = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  q35_config_.partial_rotary_factor = *reinterpret_cast<const float*>(data + offset); offset += 4;

  // MRoPE config
  q35_config_.mrope_interleaved = (*reinterpret_cast<const int32_t*>(data + offset) != 0); offset += 4;
  q35_config_.mrope_section.resize(3);
  for (int i = 0; i < 3; ++i) {
    q35_config_.mrope_section[i] = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  }

  q35_config_.full_attention_interval = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  q35_config_.attn_output_gate = (*reinterpret_cast<const int32_t*>(data + offset) != 0); offset += 4;
  
  int32_t num_ds = *reinterpret_cast<const int32_t*>(data + offset); offset += 4;
  (void)num_ds;  // 0 for Qwen3.5

  // Skip to 512 bytes
  offset = 512;

  // Also copy to vl_config_.text for compatibility 
  vl_config_.text.hidden_size = q35_config_.hidden_size;
  vl_config_.text.intermediate_size = q35_config_.intermediate_size;
  vl_config_.text.num_hidden_layers = q35_config_.num_hidden_layers;
  vl_config_.text.num_attention_heads = q35_config_.num_attention_heads;
  vl_config_.text.num_key_value_heads = q35_config_.num_key_value_heads;
  vl_config_.text.vocab_size = q35_config_.vocab_size;
  vl_config_.text.head_dim = q35_config_.head_dim;
  vl_config_.text.rms_norm_eps = q35_config_.rms_norm_eps;
  vl_config_.text.rope_theta = q35_config_.rope_theta;
  vl_config_.text.mrope_section = q35_config_.mrope_section;

  // Set up TransformerConfig for base class
  config_ = std::make_unique<TransformerConfig>();
  config_->dim_ = q35_config_.hidden_size;
  config_->hidden_dim_ = q35_config_.intermediate_size;
  config_->layer_num_ = q35_config_.num_hidden_layers;
  config_->head_num_ = q35_config_.num_attention_heads;
  config_->kv_head_num_ = q35_config_.num_key_value_heads;
  config_->vocab_size_ = q35_config_.vocab_size;
  config_->seq_len_ = 8192;
  config_->head_size_ = q35_config_.head_dim;
  config_->kv_dim_ = q35_config_.kv_dim();
  config_->kv_mul_ = q35_config_.num_attention_heads / q35_config_.num_key_value_heads;

  // Copy mrope sections to CPU buffer
  mrope_sections_cpu_[0] = q35_config_.mrope_section[0];
  mrope_sections_cpu_[1] = q35_config_.mrope_section[1];
  mrope_sections_cpu_[2] = q35_config_.mrope_section[2];

  LOG(INFO) << "Qwen3.5-9B config:";
  LOG(INFO) << "  Vision: hidden=" << vl_config_.vision.hidden_size 
            << ", depth=" << vl_config_.vision.depth;
  LOG(INFO) << "  LLM: dim=" << q35_config_.hidden_size << ", layers=" << q35_config_.num_hidden_layers;
  LOG(INFO) << "  Full attn layers: " << q35_config_.full_attn_layer_indices.size();
  LOG(INFO) << "  Linear attn: conv_kernel=" << q35_config_.linear_conv_kernel_dim
            << " key_heads=" << q35_config_.linear_num_key_heads
            << " val_heads=" << q35_config_.linear_num_value_heads;
  LOG(INFO) << "  MRoPE interleaved=" << q35_config_.mrope_interleaved
            << " sections=[" << q35_config_.mrope_section[0] << ","
            << q35_config_.mrope_section[1] << "," << q35_config_.mrope_section[2] << "]";

  // =========================================
  // Load Vision Weights (reuse Qwen3VL loading code)
  // =========================================
  auto alloc_gpu = base::CUDADeviceAllocatorFactory::get_instance();
  
  auto read_fp16_tensor_to_gpu = [&data, &offset](tensor::Tensor& t,
                                                    const std::vector<int>& dims,
                                                    std::shared_ptr<base::DeviceAllocator> alloc) {
    size_t numel = 1;
    for (int d : dims) numel *= d;
    t = tensor::Tensor(base::DataType::kDataTypeFp16, dims, true, alloc);
    cudaMemcpy(t.ptr<void>(), data + offset, numel * sizeof(uint16_t), cudaMemcpyHostToDevice);
    offset += numel * sizeof(uint16_t);
  };

  auto read_fp32_tensor_to_gpu = [&data, &offset](tensor::Tensor& t,
                                                    const std::vector<int>& dims,
                                                    std::shared_ptr<base::DeviceAllocator> alloc) {
    size_t numel = 1;
    for (int d : dims) numel *= d;
    t = tensor::Tensor(base::DataType::kDataTypeFp32, dims, true, alloc);
    cudaMemcpy(t.ptr<void>(), data + offset, numel * sizeof(float), cudaMemcpyHostToDevice);
    offset += numel * sizeof(float);
  };

  int vit_hidden = vl_config_.vision.hidden_size;
  int vit_intermediate = vl_config_.vision.intermediate_size;
  int vit_depth = vl_config_.vision.depth;
  int spatial_merge = vl_config_.vision.spatial_merge_size;
  int out_hidden = vl_config_.vision.out_hidden_size;
  int merged_hidden = vit_hidden * spatial_merge * spatial_merge;

  LOG(INFO) << "Loading vision encoder weights...";
  // Patch embed
  read_fp16_tensor_to_gpu(vision_layers_->patch_embed_weight,
    {vit_hidden, vl_config_.vision.in_channels, vl_config_.vision.temporal_patch_size,
     vl_config_.vision.patch_size, vl_config_.vision.patch_size}, alloc_gpu);
  read_fp16_tensor_to_gpu(vision_layers_->patch_embed_bias, {vit_hidden}, alloc_gpu);
  // Pos embed
  read_fp16_tensor_to_gpu(vision_layers_->pos_embed_weight,
    {vl_config_.vision.num_position_embeddings, vit_hidden}, alloc_gpu);
  // ViT blocks
  vision_layers_->blocks.resize(vit_depth);
  for (int i = 0; i < vit_depth; ++i) {
    auto& b = vision_layers_->blocks[i];
    read_fp16_tensor_to_gpu(b.norm1_weight, {vit_hidden}, alloc_gpu);
    read_fp16_tensor_to_gpu(b.norm1_bias, {vit_hidden}, alloc_gpu);
    read_fp16_tensor_to_gpu(b.norm2_weight, {vit_hidden}, alloc_gpu);
    read_fp16_tensor_to_gpu(b.norm2_bias, {vit_hidden}, alloc_gpu);
    read_fp16_tensor_to_gpu(b.qkv_weight, {3*vit_hidden, vit_hidden}, alloc_gpu);
    read_fp16_tensor_to_gpu(b.qkv_bias, {3*vit_hidden}, alloc_gpu);
    read_fp16_tensor_to_gpu(b.proj_weight, {vit_hidden, vit_hidden}, alloc_gpu);
    read_fp16_tensor_to_gpu(b.proj_bias, {vit_hidden}, alloc_gpu);
    read_fp16_tensor_to_gpu(b.mlp_fc1_weight, {vit_intermediate, vit_hidden}, alloc_gpu);
    read_fp16_tensor_to_gpu(b.mlp_fc1_bias, {vit_intermediate}, alloc_gpu);
    read_fp16_tensor_to_gpu(b.mlp_fc2_weight, {vit_hidden, vit_intermediate}, alloc_gpu);
    read_fp16_tensor_to_gpu(b.mlp_fc2_bias, {vit_hidden}, alloc_gpu);
  }
  // Merger
  auto load_main_merger = [&](Qwen3VLVisionLayers::Merger& m) {
    read_fp16_tensor_to_gpu(m.norm_weight, {vit_hidden}, alloc_gpu);
    read_fp16_tensor_to_gpu(m.norm_bias, {vit_hidden}, alloc_gpu);
    read_fp16_tensor_to_gpu(m.fc1_weight, {merged_hidden, merged_hidden}, alloc_gpu);
    read_fp16_tensor_to_gpu(m.fc1_bias, {merged_hidden}, alloc_gpu);
    read_fp16_tensor_to_gpu(m.fc2_weight, {out_hidden, merged_hidden}, alloc_gpu);
    read_fp16_tensor_to_gpu(m.fc2_bias, {out_hidden}, alloc_gpu);
  };
  load_main_merger(vision_layers_->merger);
  // No deepstack for Qwen3.5
  vision_layers_->deepstack_mergers.clear();
  LOG(INFO) << "Vision encoder loaded";

  // =========================================
  // Load LLM Weights
  // =========================================
  int dim = q35_config_.hidden_size;
  int n_layers = q35_config_.num_hidden_layers;
  int vocab = q35_config_.vocab_size;
  int head_dim = q35_config_.head_dim;
  int n_kv_heads = q35_config_.num_key_value_heads;
  int n_heads = q35_config_.num_attention_heads;
  int kv_dim = q35_config_.kv_dim();
  int q_gate_dim = q35_config_.q_gate_dim();  // 8192
  int conv_dim = q35_config_.conv_dim();       // 8192
  int intermediate = q35_config_.intermediate_size;
  int n_full = q35_config_.full_attn_layer_indices.size();  // 8
  int n_linear = n_layers - n_full;  // 24
  int n_k_heads = q35_config_.linear_num_key_heads;
  int n_v_heads = q35_config_.linear_num_value_heads;
  int k_head_dim = q35_config_.linear_key_head_dim;
  int v_head_dim = q35_config_.linear_value_head_dim;

  auto cpu_dt = base::DeviceType::kDeviceCPU;

  LOG(INFO) << "Loading language model weights...";

  // 1. RMSNorm weights (input_layernorm for all 32 layers)
  for (int i = 0; i < n_layers; ++i) {
    auto rms = std::make_shared<op::RmsNormLayer>(device_type_, dim);
    rms->set_weight_fp16(0, {dim}, data + offset, cpu_dt);
    qwen_layers_->rmsnorm_layers_.push_back(rms);
    offset += dim * sizeof(uint16_t);
  }
  // 2. post_attention_layernorm for all 32 layers
  for (int i = 0; i < n_layers; ++i) {
    auto rms = std::make_shared<op::RmsNormLayer>(device_type_, dim);
    rms->set_weight_fp16(0, {dim}, data + offset, cpu_dt);
    qwen_layers_->rmsnorm_layers_.push_back(rms);
    offset += dim * sizeof(uint16_t);
  }
  // 3. Final norm
  {
    auto rms = std::make_shared<op::RmsNormLayer>(device_type_, dim);
    rms->set_weight_fp16(0, {dim}, data + offset, cpu_dt);
    qwen_layers_->rmsnorm_layers_.push_back(rms);
    offset += dim * sizeof(uint16_t);
  }
  LOG(INFO) << "  RMSNorm: " << qwen_layers_->rmsnorm_layers_.size() << " layers";

  // 4. Token embeddings
  {
    auto emb = std::make_shared<op::EmbeddingLayer>(device_type_, dim, config_->seq_len_, vocab);
    emb->set_weight_fp16(0, {vocab, dim}, data + offset, cpu_dt);
    qwen_layers_->embedding_layer_ = emb;
    offset += static_cast<size_t>(vocab) * dim * sizeof(uint16_t);
  }
  LOG(INFO) << "  Embeddings: [" << vocab << "," << dim << "]";

  // 5. Full attention weights (8 layers)
  // q_proj (includes gate: [q_gate_dim, dim] = [8192, 4096])
  for (int i = 0; i < n_full; ++i) {
    auto mm = std::make_shared<op::MatmulLayer>(device_type_, q_gate_dim, dim, false);
    mm->set_weight_fp16(0, {q_gate_dim, dim}, data + offset, cpu_dt);
    qwen_layers_->wq_layers_.push_back(mm);
    offset += static_cast<size_t>(q_gate_dim) * dim * sizeof(uint16_t);
  }
  // k_proj: [kv_dim, dim] = [1024, 4096]
  for (int i = 0; i < n_full; ++i) {
    auto mm = std::make_shared<op::MatmulLayer>(device_type_, kv_dim, dim, false);
    mm->set_weight_fp16(0, {kv_dim, dim}, data + offset, cpu_dt);
    qwen_layers_->wk_layers_.push_back(mm);
    offset += static_cast<size_t>(kv_dim) * dim * sizeof(uint16_t);
  }
  // v_proj: [kv_dim, dim]
  for (int i = 0; i < n_full; ++i) {
    auto mm = std::make_shared<op::MatmulLayer>(device_type_, kv_dim, dim, false);
    mm->set_weight_fp16(0, {kv_dim, dim}, data + offset, cpu_dt);
    qwen_layers_->wv_layers_.push_back(mm);
    offset += static_cast<size_t>(kv_dim) * dim * sizeof(uint16_t);
  }
  // o_proj: [dim, q_dim] = [4096, 4096]
  for (int i = 0; i < n_full; ++i) {
    auto mm = std::make_shared<op::MatmulLayer>(device_type_, dim, q35_config_.q_dim(), false);
    mm->set_weight_fp16(0, {dim, q35_config_.q_dim()}, data + offset, cpu_dt);
    qwen_layers_->wo_layers_.push_back(mm);
    offset += static_cast<size_t>(dim) * q35_config_.q_dim() * sizeof(uint16_t);
  }
  // q_norm: [head_dim] per full attn layer
  for (int i = 0; i < n_full; ++i) {
    auto rms = std::make_shared<op::RmsNormLayer>(device_type_, head_dim);
    rms->set_weight_fp16(0, {head_dim}, data + offset, cpu_dt);
    qwen_layers_->rmsnorm_layers_.push_back(rms);
    offset += head_dim * sizeof(uint16_t);
  }
  // k_norm: [head_dim] per full attn layer
  for (int i = 0; i < n_full; ++i) {
    auto rms = std::make_shared<op::RmsNormLayer>(device_type_, head_dim);
    rms->set_weight_fp16(0, {head_dim}, data + offset, cpu_dt);
    qwen_layers_->rmsnorm_layers_.push_back(rms);
    offset += head_dim * sizeof(uint16_t);
  }
  LOG(INFO) << "  Full attn: " << n_full << " layers (q_gate_dim=" << q_gate_dim << ")";

  // 6. Linear attention weights (24 layers)
  linear_attn_weights_->layers.resize(n_linear);
  // in_proj_qkv: [conv_dim, dim]
  for (int i = 0; i < n_linear; ++i) {
    auto mm = std::make_shared<op::MatmulLayer>(device_type_, conv_dim, dim, false);
    mm->set_weight_fp16(0, {conv_dim, dim}, data + offset, cpu_dt);
    linear_attn_weights_->layers[i].in_proj_qkv = mm;
    offset += static_cast<size_t>(conv_dim) * dim * sizeof(uint16_t);
  }
  // in_proj_z: [dim, dim]
  for (int i = 0; i < n_linear; ++i) {
    auto mm = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, false);
    mm->set_weight_fp16(0, {dim, dim}, data + offset, cpu_dt);
    linear_attn_weights_->layers[i].in_proj_z = mm;
    offset += static_cast<size_t>(dim) * dim * sizeof(uint16_t);
  }
  // in_proj_a: [n_v_heads, dim]
  for (int i = 0; i < n_linear; ++i) {
    auto mm = std::make_shared<op::MatmulLayer>(device_type_, n_v_heads, dim, false);
    mm->set_weight_fp16(0, {n_v_heads, dim}, data + offset, cpu_dt);
    linear_attn_weights_->layers[i].in_proj_a = mm;
    offset += static_cast<size_t>(n_v_heads) * dim * sizeof(uint16_t);
  }
  // in_proj_b: [n_v_heads, dim]
  for (int i = 0; i < n_linear; ++i) {
    auto mm = std::make_shared<op::MatmulLayer>(device_type_, n_v_heads, dim, false);
    mm->set_weight_fp16(0, {n_v_heads, dim}, data + offset, cpu_dt);
    linear_attn_weights_->layers[i].in_proj_b = mm;
    offset += static_cast<size_t>(n_v_heads) * dim * sizeof(uint16_t);
  }
  // A_log: [n_v_heads] FP32
  for (int i = 0; i < n_linear; ++i) {
    read_fp32_tensor_to_gpu(linear_attn_weights_->layers[i].A_log, {n_v_heads}, alloc_gpu);
  }
  // dt_bias: [n_v_heads] FP16
  for (int i = 0; i < n_linear; ++i) {
    read_fp16_tensor_to_gpu(linear_attn_weights_->layers[i].dt_bias, {n_v_heads}, alloc_gpu);
  }
  // conv1d: [conv_dim, 1, kernel_size] -> stored as [conv_dim, kernel_size] with 1 squeezed
  {
    int ks = q35_config_.linear_conv_kernel_dim;
    for (int i = 0; i < n_linear; ++i) {
      read_fp16_tensor_to_gpu(linear_attn_weights_->layers[i].conv_weight,
                              {conv_dim, ks}, alloc_gpu);
    }
  }
  // norm: [v_head_dim] FP32 (per-head, shared across all value heads)
  for (int i = 0; i < n_linear; ++i) {
    read_fp32_tensor_to_gpu(linear_attn_weights_->layers[i].norm_weight, {v_head_dim}, alloc_gpu);
  }
  // out_proj: [dim, dim]
  for (int i = 0; i < n_linear; ++i) {
    auto mm = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, false);
    mm->set_weight_fp16(0, {dim, dim}, data + offset, cpu_dt);
    linear_attn_weights_->layers[i].out_proj = mm;
    offset += static_cast<size_t>(dim) * dim * sizeof(uint16_t);
  }
  LOG(INFO) << "  Linear attn: " << n_linear << " layers (conv_dim=" << conv_dim << ")";

  // 7. FFN weights (all 32 layers)
  auto load_proj = [&](std::vector<std::shared_ptr<op::Layer>>& layers, int out, int in) {
    for (int i = 0; i < n_layers; ++i) {
      auto mm = std::make_shared<op::MatmulLayer>(device_type_, out, in, false);
      mm->set_weight_fp16(0, {out, in}, data + offset, cpu_dt);
      layers.push_back(mm);
      offset += static_cast<size_t>(out) * in * sizeof(uint16_t);
    }
  };
  load_proj(qwen_layers_->w1_layers_, intermediate, dim);
  load_proj(qwen_layers_->w2_layers_, dim, intermediate);
  load_proj(qwen_layers_->w3_layers_, intermediate, dim);
  LOG(INFO) << "  FFN: " << n_layers << " layers";

  // 8. LM head
  if (vl_config_.has_lm_head) {
    auto cls = std::make_shared<op::MatmulLayer>(device_type_, vocab, dim, false);
    cls->set_weight_fp16(0, {vocab, dim}, data + offset, cpu_dt);
    qwen_layers_->cls_layer_ = cls;
    offset += static_cast<size_t>(vocab) * dim * sizeof(uint16_t);
    LOG(INFO) << "  LM head: [" << vocab << "," << dim << "]";
  }

  LOG(INFO) << "Model loading complete. Offset: " << offset << " / " << vl_model_file_size_;
  return base::error::Success();
}

// ==========================================================================
// Memory Initialization
// ==========================================================================

void Qwen35Model::init_q35_mem() {
  auto alloc = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
  auto alloc_pinned = base::CPUPinnedAllocatorFactory::get_instance();
  auto act = base::DataType::kDataTypeFp16;

  int dim = q35_config_.hidden_size;
  int intermediate = q35_config_.intermediate_size;
  int kv_dim = q35_config_.kv_dim();
  int vocab = q35_config_.vocab_size;
  int n_layers = q35_config_.num_hidden_layers;
  int conv_dim = q35_config_.conv_dim();
  int n_v_heads = q35_config_.linear_num_value_heads;
  int n_k_heads = q35_config_.linear_num_key_heads;
  int k_head_dim = q35_config_.linear_key_head_dim;
  int v_head_dim = q35_config_.linear_value_head_dim;
  int ks = q35_config_.linear_conv_kernel_dim;
  int n_full = q35_config_.full_attn_layer_indices.size();
  int n_linear = n_layers - n_full;

  // Basic buffers (same as Qwen3VL)
  CHECK(insert_buffer(ModelBufferType::kInputTokens,
      tensor::Tensor(DataType::kDataTypeInt32, 1, true, alloc_cpu)));
  CHECK(insert_buffer(ModelBufferType::kInputEmbeddings,
      tensor::Tensor(act, 1, dim, true, alloc)));
  CHECK(insert_buffer(ModelBufferType::kSinCache,
      tensor::Tensor(DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_, true, alloc)));
  CHECK(insert_buffer(ModelBufferType::kCosCache,
      tensor::Tensor(DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_, true, alloc)));

  tensor::Tensor rms_out(act, dim, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kOutputRMSNorm, rms_out));
  CHECK(insert_buffer(ModelBufferType::kOutputMHA, rms_out));
  CHECK(insert_buffer(ModelBufferType::kW2Output, rms_out));
  CHECK(insert_buffer(ModelBufferType::kFFNRMSNorm, rms_out));

  CHECK(insert_buffer(ModelBufferType::kW1Output, tensor::Tensor(act, intermediate, true, alloc)));
  CHECK(insert_buffer(ModelBufferType::kW3Output, tensor::Tensor(act, intermediate, true, alloc)));

  // KV cache for FULL ATTENTION layers only
  // Shape: [n_full, seq_len, kv_dim]
  CHECK(insert_buffer(ModelBufferType::kKeyCache,
      tensor::Tensor(act, n_full, config_->seq_len_, kv_dim, true, alloc)));
  CHECK(insert_buffer(ModelBufferType::kValueCache,
      tensor::Tensor(act, n_full, config_->seq_len_, kv_dim, true, alloc)));

  CHECK(insert_buffer(ModelBufferType::kQuery, tensor::Tensor(act, q35_config_.q_gate_dim(), true, alloc)));
  CHECK(insert_buffer(ModelBufferType::kDecodeInput, tensor::Tensor(act, dim, true, alloc)));
  CHECK(insert_buffer(ModelBufferType::kInputPos, tensor::Tensor(DataType::kDataTypeInt32, 1, true, alloc_cpu)));
  CHECK(insert_buffer(ModelBufferType::kInputPosGPU, tensor::Tensor(DataType::kDataTypeInt32, 1, true, alloc)));
  CHECK(insert_buffer(ModelBufferType::kKVCachePosGPU, tensor::Tensor(DataType::kDataTypeInt32, 1, true, alloc)));
  CHECK(insert_buffer(ModelBufferType::kTempKey, tensor::Tensor(act, kv_dim, true, alloc)));
  CHECK(insert_buffer(ModelBufferType::kTempValue, tensor::Tensor(act, kv_dim, true, alloc)));

  CHECK(insert_buffer(ModelBufferType::kInputPosPinned, tensor::Tensor(DataType::kDataTypeInt32, 1, true, alloc_pinned)));
  CHECK(insert_buffer(ModelBufferType::kKVCachePosPinned, tensor::Tensor(DataType::kDataTypeInt32, 1, true, alloc_pinned)));
  CHECK(insert_buffer(ModelBufferType::kArgmaxOutput, tensor::Tensor(DataType::kDataTypeInt32, 2, true, alloc)));
  CHECK(insert_buffer(ModelBufferType::kArgmaxOutputPinned, tensor::Tensor(DataType::kDataTypeInt32, 2, true, alloc_pinned)));

  CHECK(insert_buffer(ModelBufferType::kScoreStorage,
      tensor::Tensor(DataType::kDataTypeFp32, config_->head_num_, config_->seq_len_, true, alloc)));
  CHECK(insert_buffer(ModelBufferType::kAttnOutput, tensor::Tensor(act, dim, true, alloc)));
  CHECK(insert_buffer(ModelBufferType::kForwardOutput,
      tensor::Tensor(DataType::kDataTypeFp32, vocab, true, alloc)));
  CHECK(insert_buffer(ModelBufferType::kForwardOutputCPU,
      tensor::Tensor(DataType::kDataTypeFp32, vocab, true, alloc_cpu)));

  // Pixel buffer for ViT
  constexpr size_t kPixelBuf = 32 * 1024 * 1024;
  cudaMalloc(&pixel_buf_gpu_, kPixelBuf);
  pixel_buf_gpu_capacity_ = kPixelBuf;

  // GDN intermediate buffers
  gdn_qkv_buf_ = tensor::Tensor(act, conv_dim, true, alloc);
  gdn_z_buf_ = tensor::Tensor(act, dim, true, alloc);
  gdn_alpha_buf_ = tensor::Tensor(act, n_v_heads, true, alloc);
  gdn_beta_buf_ = tensor::Tensor(act, n_v_heads, true, alloc);
  gdn_gate_buf_ = tensor::Tensor(DataType::kDataTypeFp32, n_v_heads, true, alloc);
  gdn_beta_fp32_ = tensor::Tensor(DataType::kDataTypeFp32, n_v_heads, true, alloc);
  gdn_conv_out_ = tensor::Tensor(act, conv_dim, true, alloc);
  gdn_q_norm_ = tensor::Tensor(act, n_k_heads * k_head_dim, true, alloc);
  gdn_k_norm_ = tensor::Tensor(act, n_k_heads * k_head_dim, true, alloc);
  gdn_attn_out_ = tensor::Tensor(act, n_v_heads * v_head_dim, true, alloc);
  gdn_normed_out_ = tensor::Tensor(act, dim, true, alloc);
  full_attn_q_ = tensor::Tensor(act, q35_config_.q_dim(), true, alloc);
  full_attn_gate_ = tensor::Tensor(act, q35_config_.q_dim(), true, alloc);

  // Create fused FFN layer for decode acceleration
  qwen_layers_->fused_ffn_layer_ = std::make_shared<op::FusedFFNLayer>(
      device_type_, dim, q35_config_.intermediate_size, true, false);
  qwen_layers_->fused_ffn_layer_->set_cuda_config(cuda_config_);

  // Create GPU-pos flash attention decode layer for CUDA Graph
  qwen_layers_->flash_attention_decode_gpu_pos_layer_ =
      std::make_shared<op::FlashAttentionDecodeGpuPosLayer>(device_type_);
  qwen_layers_->flash_attention_decode_gpu_pos_layer_->set_cuda_config(cuda_config_);

  // GDN states (one per linear layer)
  gdn_states_.resize(n_linear);
  for (int i = 0; i < n_linear; ++i) {
    gdn_states_[i].conv_state = tensor::Tensor(act, conv_dim, ks - 1, true, alloc);
    gdn_states_[i].ssm_state = tensor::Tensor(DataType::kDataTypeFp32,
        n_v_heads, v_head_dim, k_head_dim, true, alloc);
    // Zero-initialize states
    cudaMemset(gdn_states_[i].conv_state.ptr<void>(), 0,
               conv_dim * (ks-1) * sizeof(uint16_t));
    cudaMemset(gdn_states_[i].ssm_state.ptr<void>(), 0,
               n_v_heads * v_head_dim * k_head_dim * sizeof(float));
  }

  LOG(INFO) << "Qwen3.5 memory initialized. GDN states: " << n_linear << " layers";
}

void Qwen35Model::create_q35_nonparam_layers() {
  auto make_layer = [this](auto layer) {
    layer->set_cuda_config(cuda_config_);
    return layer;
  };

  deinterleave_q_gate_layer_ = make_layer(std::make_shared<op::DeinterleaveQGateLayer>(device_type_));
  partial_mrope_layer_ = make_layer(std::make_shared<op::PartialMRoPEInterleavedLayer>(device_type_));
  kv_cache_write_gpu_pos_layer_ = make_layer(std::make_shared<op::KVCacheWriteGpuPosLayer>(device_type_));
  apply_sigmoid_gate_layer_ = make_layer(std::make_shared<op::ApplySigmoidGateLayer>(device_type_));
  causal_conv1d_silu_layer_ = make_layer(std::make_shared<op::CausalConv1dSiluLayer>(device_type_));
  l2_norm_per_head_layer_ = make_layer(std::make_shared<op::L2NormPerHeadLayer>(device_type_));
  compute_gdn_gates_layer_ = make_layer(std::make_shared<op::ComputeGDNGatesLayer>(device_type_));
  gdn_decode_step_layer_ = make_layer(std::make_shared<op::GDNDecodeStepLayer>(device_type_));
  gated_rmsnorm_layer_ = make_layer(std::make_shared<op::GatedRMSNormLayer>(device_type_));
  batched_add_fp16_layer_ = make_layer(std::make_shared<op::BatchedAddFP16Layer>(device_type_));
  batched_rmsnorm_fp16_layer_ = make_layer(std::make_shared<op::BatchedRMSNormFP16Layer>(device_type_));
  gather_strided_layer_ = make_layer(std::make_shared<op::GatherStridedLayer>(device_type_));
  transpose_state_layer_ = make_layer(std::make_shared<op::TransposeStateLayer>(device_type_));
  gdn_prefill_transposed_layer_ = make_layer(std::make_shared<op::GDNPrefillTransposedLayer>(device_type_));
  fused_qkv_gemv_layer_ = make_layer(std::make_shared<op::FusedQKVGemvLayer>(device_type_));
  fused_gdn_proj_gemv_layer_ = make_layer(std::make_shared<op::FusedGDNProjGemvLayer>(device_type_));

  // Also create VecAddLayer for decode residual adds
  if (!qwen_layers_->add_layer_) {
    qwen_layers_->add_layer_ = make_layer(std::make_shared<op::VecAddLayer>(device_type_));
  }
}

void Qwen35Model::clear_all_state() {
  // Clear KV cache
  auto& key_cache = get_buffer(ModelBufferType::kKeyCache);
  auto& val_cache = get_buffer(ModelBufferType::kValueCache);
  cudaMemset(key_cache.ptr<void>(), 0, key_cache.byte_size());
  cudaMemset(val_cache.ptr<void>(), 0, val_cache.byte_size());

  // Clear GDN states
  for (auto& s : gdn_states_) {
    cudaMemset(s.conv_state.ptr<void>(), 0, s.conv_state.byte_size());
    cudaMemset(s.ssm_state.ptr<void>(), 0, s.ssm_state.byte_size());
  }
}

// ==========================================================================
// Decode: Full Attention Layer
// ==========================================================================

void Qwen35Model::full_attn_decode(int32_t layer_idx, const tensor::Tensor& input) const {
  auto stream = cuda_config_->stream;
  int type_idx = full_attn_type_idx(layer_idx);
  int n_layers = q35_config_.num_hidden_layers;
  int dim = q35_config_.hidden_size;
  int head_dim = q35_config_.head_dim;
  int n_heads = q35_config_.num_attention_heads;
  int n_kv_heads = q35_config_.num_key_value_heads;
  int kv_dim = q35_config_.kv_dim();
  int q_dim = q35_config_.q_dim();
  int q_gate_dim = q35_config_.q_gate_dim();
  int partial_dim = q35_config_.partial_rope_dim();

  // 1. Attention RMSNorm
  tensor::Tensor rms_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  qwen_layers_->rmsnorm_layers_[layer_idx]->forward(input, rms_output);

  // 2-3. Fused Q+K+V projection (single kernel launch)
  tensor::Tensor query_gate_buf = get_buffer(ModelBufferType::kQuery);  // [q_gate_dim=8192]
  tensor::Tensor temp_key = get_buffer(ModelBufferType::kTempKey);
  tensor::Tensor temp_value = get_buffer(ModelBufferType::kTempValue);

  auto wq_param = std::dynamic_pointer_cast<op::LayerParam>(qwen_layers_->wq_layers_[type_idx]);
  auto wk_param = std::dynamic_pointer_cast<op::LayerParam>(qwen_layers_->wk_layers_[type_idx]);
  auto wv_param = std::dynamic_pointer_cast<op::LayerParam>(qwen_layers_->wv_layers_[type_idx]);

  fused_qkv_gemv_layer_->forward(
      rms_output.ptr<half>(),
      wq_param->get_weight(0).ptr<half>(),
      wk_param->get_weight(0).ptr<half>(),
      wv_param->get_weight(0).ptr<half>(),
      query_gate_buf.ptr<half>(), temp_key.ptr<half>(), temp_value.ptr<half>(),
      dim, q_gate_dim, kv_dim);

  // Deinterleave Q and Gate per-head
  deinterleave_q_gate_layer_->forward(
      query_gate_buf.ptr<half>(), full_attn_q_.ptr<half>(), full_attn_gate_.ptr<half>(),
      n_heads, head_dim, 1);

  half* query_ptr = full_attn_q_.ptr<half>();
  half* gate_ptr = full_attn_gate_.ptr<half>();

  // 4. Q norm, K norm (per-head RMSNorm)
  int q_norm_idx = 2 * n_layers + 1 + type_idx;
  int k_norm_idx = 2 * n_layers + 1 + (int)q35_config_.full_attn_layer_indices.size() + type_idx;

  auto& q_norm_layer = qwen_layers_->rmsnorm_layers_[q_norm_idx];
  auto& k_norm_layer = qwen_layers_->rmsnorm_layers_[k_norm_idx];

  // Reshape to [n_heads, head_dim] for per-head norm then reshape back
  tensor::Tensor query_view(base::DataType::kDataTypeFp16, q_dim, false, nullptr, query_ptr);
  query_view.set_device_type(DeviceType::kDeviceCUDA);
  query_view.reshape({n_heads, head_dim});
  q_norm_layer->forward(query_view, query_view);
  query_view.reshape({q_dim});

  temp_key.reshape({n_kv_heads, head_dim});
  k_norm_layer->forward(temp_key, temp_key);
  temp_key.reshape({kv_dim});

  // 5. Partial M-RoPE (interleaved)
  tensor::Tensor pos_tensor = get_buffer(ModelBufferType::kInputPos);
  int pos = pos_tensor.index<int32_t>(0);
  int text_pos = mrope_max_text_pos_ + (pos - prefill_seq_len_) + 1;

  tensor::Tensor sin_cache = get_buffer(ModelBufferType::kSinCache);
  tensor::Tensor cos_cache = get_buffer(ModelBufferType::kCosCache);

  partial_mrope_layer_->forward(
      query_ptr, temp_key.ptr<half>(),
      sin_cache.ptr<float>(), cos_cache.ptr<float>(),
      text_pos, text_pos, text_pos,
      mrope_sections_cpu_, partial_dim,
      head_dim, n_heads, n_kv_heads);

  // 6. Write K, V to cache
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);
  size_t kv_offset = (static_cast<size_t>(type_idx) * config_->seq_len_ + pos) * kv_dim;
  cudaMemcpyAsync(key_cache.ptr<half>() + kv_offset, temp_key.ptr<half>(),
                  kv_dim * sizeof(half), cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(val_cache.ptr<half>() + kv_offset, temp_value.ptr<half>(),
                  kv_dim * sizeof(half), cudaMemcpyDeviceToDevice, stream);

  // 7. Flash Attention Decode
  tensor::Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);

  qwen_layers_->flash_attention_decode_layer_->forward(
      pos, n_heads, n_kv_heads, head_dim, config_->kv_mul_,
      type_idx, config_->seq_len_, kv_dim,
      query_view, mha_output, key_cache, val_cache);

  // 8. Apply sigmoid gate: mha_output *= sigmoid(gate)
  apply_sigmoid_gate_layer_->forward(mha_output.ptr<half>(), gate_ptr, q_dim);

  // 9. Output projection
  tensor::Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
  qwen_layers_->wo_layers_[type_idx]->forward(mha_output, attn_output);
}

// ==========================================================================
// Decode: Full Attention Layer (CUDA Graph compatible)
// ==========================================================================

void Qwen35Model::full_attn_decode_graph(int32_t layer_idx, const tensor::Tensor& input,
                                         const int32_t* rope_pos_gpu, const int32_t* kv_pos_gpu) const {
  int type_idx = full_attn_type_idx(layer_idx);
  int n_layers = q35_config_.num_hidden_layers;
  int dim = q35_config_.hidden_size;
  int head_dim = q35_config_.head_dim;
  int n_heads = q35_config_.num_attention_heads;
  int n_kv_heads = q35_config_.num_key_value_heads;
  int kv_dim = q35_config_.kv_dim();
  int q_dim = q35_config_.q_dim();
  int partial_dim = q35_config_.partial_rope_dim();

  // 1. Attention RMSNorm
  tensor::Tensor rms_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  qwen_layers_->rmsnorm_layers_[layer_idx]->forward(input, rms_output);

  // 2-3. Fused Q+K+V projection (single kernel launch)
  tensor::Tensor query_gate_buf = get_buffer(ModelBufferType::kQuery);
  tensor::Tensor temp_key = get_buffer(ModelBufferType::kTempKey);
  tensor::Tensor temp_value = get_buffer(ModelBufferType::kTempValue);

  fused_qkv_gemv_layer_->forward(
      rms_output.ptr<half>(),
      std::dynamic_pointer_cast<op::LayerParam>(qwen_layers_->wq_layers_[type_idx])->get_weight(0).ptr<half>(),
      std::dynamic_pointer_cast<op::LayerParam>(qwen_layers_->wk_layers_[type_idx])->get_weight(0).ptr<half>(),
      std::dynamic_pointer_cast<op::LayerParam>(qwen_layers_->wv_layers_[type_idx])->get_weight(0).ptr<half>(),
      query_gate_buf.ptr<half>(), temp_key.ptr<half>(), temp_value.ptr<half>(),
      dim, q35_config_.q_gate_dim(), kv_dim);

  // Deinterleave Q and Gate
  deinterleave_q_gate_layer_->forward(
      query_gate_buf.ptr<half>(), full_attn_q_.ptr<half>(), full_attn_gate_.ptr<half>(),
      n_heads, head_dim, 1);

  half* query_ptr = full_attn_q_.ptr<half>();
  half* gate_ptr = full_attn_gate_.ptr<half>();

  // 4. Q norm, K norm
  int q_norm_idx = 2 * n_layers + 1 + type_idx;
  int k_norm_idx = 2 * n_layers + 1 + (int)q35_config_.full_attn_layer_indices.size() + type_idx;
  tensor::Tensor query_view(base::DataType::kDataTypeFp16, q_dim, false, nullptr, query_ptr);
  query_view.set_device_type(DeviceType::kDeviceCUDA);
  query_view.reshape({n_heads, head_dim});
  qwen_layers_->rmsnorm_layers_[q_norm_idx]->forward(query_view, query_view);
  query_view.reshape({q_dim});

  temp_key.reshape({n_kv_heads, head_dim});
  qwen_layers_->rmsnorm_layers_[k_norm_idx]->forward(temp_key, temp_key);
  temp_key.reshape({kv_dim});

  // 5. Partial M-RoPE with GPU position
  tensor::Tensor sin_cache = get_buffer(ModelBufferType::kSinCache);
  tensor::Tensor cos_cache = get_buffer(ModelBufferType::kCosCache);
  partial_mrope_layer_->forward_gpu_pos(
      query_ptr, temp_key.ptr<half>(),
      sin_cache.ptr<float>(), cos_cache.ptr<float>(),
      rope_pos_gpu, mrope_sections_cpu_, partial_dim,
      head_dim, n_heads, n_kv_heads);

  // 6. Write K, V to cache using GPU position
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);
  kv_cache_write_gpu_pos_layer_->forward(key_cache.ptr<half>(), temp_key.ptr<half>(),
      kv_pos_gpu, kv_dim, type_idx, config_->seq_len_);
  kv_cache_write_gpu_pos_layer_->forward(val_cache.ptr<half>(), temp_value.ptr<half>(),
      kv_pos_gpu, kv_dim, type_idx, config_->seq_len_);

  // 7. Flash Attention Decode with GPU position
  tensor::Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);
  qwen_layers_->flash_attention_decode_gpu_pos_layer_->forward(
      kv_pos_gpu, n_heads, n_kv_heads, head_dim, config_->kv_mul_,
      type_idx, config_->seq_len_, kv_dim,
      query_view, mha_output, key_cache, val_cache);

  // 8. Apply sigmoid gate
  apply_sigmoid_gate_layer_->forward(mha_output.ptr<half>(), gate_ptr, q_dim);

  // 9. Output projection
  tensor::Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
  qwen_layers_->wo_layers_[type_idx]->forward(mha_output, attn_output);
}

// ==========================================================================
// Decode: Linear Attention Layer (GDN)
// ==========================================================================

void Qwen35Model::linear_attn_decode(int32_t layer_idx, const tensor::Tensor& input) const {
  int type_idx = linear_attn_type_idx(layer_idx);
  int n_layers = q35_config_.num_hidden_layers;
  int dim = q35_config_.hidden_size;
  int conv_dim = q35_config_.conv_dim();
  int n_k_heads = q35_config_.linear_num_key_heads;
  int n_v_heads = q35_config_.linear_num_value_heads;
  int k_head_dim = q35_config_.linear_key_head_dim;
  int v_head_dim = q35_config_.linear_value_head_dim;
  int ks = q35_config_.linear_conv_kernel_dim;
  
  auto& la = linear_attn_weights_->layers[type_idx];
  auto& state = gdn_states_[type_idx];

  // 1. RMSNorm
  tensor::Tensor rms_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  qwen_layers_->rmsnorm_layers_[layer_idx]->forward(input, rms_output);

  // 2. Projections: fused QKV+Z (single launch) + tiny A, B
  fused_gdn_proj_gemv_layer_->forward(
      rms_output.ptr<half>(),
      std::dynamic_pointer_cast<op::LayerParam>(la.in_proj_qkv)->get_weight(0).ptr<half>(),
      std::dynamic_pointer_cast<op::LayerParam>(la.in_proj_z)->get_weight(0).ptr<half>(),
      gdn_qkv_buf_.ptr<half>(), gdn_z_buf_.ptr<half>(),
      dim, conv_dim, dim);
  la.in_proj_a->forward(rms_output, gdn_alpha_buf_);     // [n_v_heads=32]
  la.in_proj_b->forward(rms_output, gdn_beta_buf_);      // [n_v_heads=32]

  // 3. Causal Conv1D + SiLU
  causal_conv1d_silu_layer_->forward(
      state.conv_state.ptr<half>(), gdn_qkv_buf_.ptr<half>(),
      la.conv_weight.ptr<half>(), gdn_conv_out_.ptr<half>(),
      conv_dim, ks);

  // 4. Split conv output into Q, K, V
  // Layout: [Q: n_k_heads*k_head_dim, K: n_k_heads*k_head_dim, V: n_v_heads*v_head_dim]
  int q_size = n_k_heads * k_head_dim;
  int k_size = n_k_heads * k_head_dim;
  int v_size = n_v_heads * v_head_dim;
  half* q_ptr = gdn_conv_out_.ptr<half>();
  half* k_ptr = q_ptr + q_size;
  half* v_ptr = k_ptr + k_size;

  // 5. L2 normalize Q and K
  l2_norm_per_head_layer_->forward(q_ptr, gdn_q_norm_.ptr<half>(), n_k_heads, k_head_dim, 1e-6f);
  l2_norm_per_head_layer_->forward(k_ptr, gdn_k_norm_.ptr<half>(), n_k_heads, k_head_dim, 1e-6f);

  // 6. Compute gates
  compute_gdn_gates_layer_->forward(
      gdn_alpha_buf_.ptr<half>(), la.dt_bias.ptr<half>(),
      la.A_log.ptr<float>(), gdn_beta_buf_.ptr<half>(),
      gdn_gate_buf_.ptr<float>(), gdn_beta_fp32_.ptr<float>(),
      n_v_heads);

  // 7. Delta Net decode step
  gdn_decode_step_layer_->forward(
      gdn_q_norm_.ptr<half>(), gdn_k_norm_.ptr<half>(), v_ptr,
      gdn_gate_buf_.ptr<float>(), gdn_beta_fp32_.ptr<float>(),
      state.ssm_state.ptr<float>(), gdn_attn_out_.ptr<half>(),
      n_k_heads, n_v_heads, k_head_dim, v_head_dim);

  // 8. Gated RMSNorm: RMSNorm(attn_out) * SiLU(z)
  gated_rmsnorm_layer_->forward(
      gdn_attn_out_.ptr<half>(), gdn_z_buf_.ptr<half>(),
      la.norm_weight.ptr<float>(), gdn_normed_out_.ptr<half>(),
      dim, q35_config_.rms_norm_eps);

  // 9. Output projection
  tensor::Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
  la.out_proj->forward(gdn_normed_out_, attn_output);
}

// ==========================================================================
// Decode: Feed Forward
// ==========================================================================

void Qwen35Model::q35_feed_forward(int32_t layer_idx, const tensor::Tensor& input) const {
  int n_layers = q35_config_.num_hidden_layers;

  tensor::Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
  
  // Residual add: input += attn_output (input buffer is mutable despite const ref)
  qwen_layers_->add_layer_->forward(input, attn_output, input);

  // FFN RMSNorm
  tensor::Tensor ffn_norm_output = get_buffer(ModelBufferType::kFFNRMSNorm);
  qwen_layers_->rmsnorm_layers_[layer_idx + n_layers]->forward(input, ffn_norm_output);

  // Gate + Up projections with optional fused FFN
  tensor::Tensor w1_out = get_buffer(ModelBufferType::kW1Output);
  auto w1_mm = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->w1_layers_[layer_idx]);
  auto w3_mm = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->w3_layers_[layer_idx]);

  auto fused_ffn = qwen_layers_->fused_ffn_layer_;
  if (fused_ffn && w1_mm && w3_mm) {
    // Fused gate+up+SwiGLU in one kernel
    fused_ffn->set_use_fp16(true);
    fused_ffn->set_input(0, ffn_norm_output);
    fused_ffn->set_input(1, w1_mm->get_weight(0));
    fused_ffn->set_input(2, w3_mm->get_weight(0));
    fused_ffn->set_output(0, w1_out);
    fused_ffn->set_cuda_config(cuda_config_);
    fused_ffn->forward();
  } else {
    qwen_layers_->w1_layers_[layer_idx]->forward(ffn_norm_output, w1_out);
    tensor::Tensor w3_out = get_buffer(ModelBufferType::kW3Output);
    qwen_layers_->w3_layers_[layer_idx]->forward(ffn_norm_output, w3_out);
    qwen_layers_->swiglu_layer_->forward(w1_out, w3_out, w1_out);
  }

  // Down projection
  tensor::Tensor w2_out = get_buffer(ModelBufferType::kW2Output);
  qwen_layers_->w2_layers_[layer_idx]->forward(w1_out, w2_out);

  // Residual add
  qwen_layers_->add_layer_->forward(input, w2_out, input);
}

// ==========================================================================
// Decode: CLS Logits
// ==========================================================================

void Qwen35Model::q35_cls_logits(const tensor::Tensor& input) const {
  int n_layers = q35_config_.num_hidden_layers;
  tensor::Tensor rms_out = get_buffer(ModelBufferType::kOutputRMSNorm);
  qwen_layers_->rmsnorm_layers_[2 * n_layers]->forward(input, rms_out);
  
  tensor::Tensor forward_out = get_buffer(ModelBufferType::kForwardOutput);
  qwen_layers_->cls_layer_->forward(rms_out, forward_out);
}

// ==========================================================================
// Decode: Full Step
// ==========================================================================

base::Status Qwen35Model::decode_step_optimized(int32_t pos, int& next) const {
  auto stream = cuda_config_->stream;
  int n_layers = q35_config_.num_hidden_layers;
  bool use_graph = cuda_config_ && cuda_config_->use_cuda_graph;

  tensor::Tensor decode_input = get_buffer(ModelBufferType::kDecodeInput);

  if (use_graph) {
    auto& graph_ctx = cuda_config_->graph_context;
    auto& graph = graph_ctx->decode_graph;

    tensor::Tensor pos_gpu = get_buffer(ModelBufferType::kInputPosGPU);
    tensor::Tensor kv_pos_gpu = get_buffer(ModelBufferType::kKVCachePosGPU);
    tensor::Tensor pos_pinned = get_buffer(ModelBufferType::kInputPosPinned);
    tensor::Tensor kv_pos_pinned = get_buffer(ModelBufferType::kKVCachePosPinned);
    tensor::Tensor argmax_output = get_buffer(ModelBufferType::kArgmaxOutput);
    tensor::Tensor argmax_pinned = get_buffer(ModelBufferType::kArgmaxOutputPinned);

    // Compute text_pos for RoPE (same as non-graph path)
    int32_t text_pos = mrope_max_text_pos_ + (pos - prefill_seq_len_) + 1;

    // Update GPU positions via pinned H2D copy
    *const_cast<int32_t*>(pos_pinned.ptr<int32_t>()) = text_pos;
    cudaMemcpyAsync(const_cast<int32_t*>(pos_gpu.ptr<int32_t>()),
                    pos_pinned.ptr<int32_t>(), sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
    *const_cast<int32_t*>(kv_pos_pinned.ptr<int32_t>()) = pos;
    cudaMemcpyAsync(const_cast<int32_t*>(kv_pos_gpu.ptr<int32_t>()),
                    kv_pos_pinned.ptr<int32_t>(), sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);

    bool need_capture = graph_ctx->needs_recapture || !graph->is_valid();

    if (need_capture && !graph->is_disabled()) {
      cudaStreamSynchronize(stream);

      if (graph->begin_capture(stream)) {
        for (int il = 0; il < n_layers; ++il) {
          if (q35_config_.is_full_attn_layer(il)) {
            full_attn_decode_graph(il, decode_input,
                                   pos_gpu.ptr<int32_t>(), kv_pos_gpu.ptr<int32_t>());
          } else {
            linear_attn_decode(il, decode_input);
          }
          q35_feed_forward(il, decode_input);
        }
        q35_cls_logits(decode_input);

        if (graph->end_capture(stream)) {
          graph_ctx->graph_recaptures++;
          graph_ctx->needs_recapture = false;
        }
      }
    }

    if (graph->is_valid()) {
      if (graph->launch(stream)) {
        graph_ctx->graph_launches++;

        tensor::Tensor forward_out = get_buffer(ModelBufferType::kForwardOutput);
        auto* argmax_sampler = dynamic_cast<sampler::ArgmaxSampler*>(sampler_.get());
        if (argmax_sampler) {
          argmax_sampler->sample_prealloc(
              forward_out.ptr<float>(), forward_out.size(),
              reinterpret_cast<size_t*>(const_cast<int32_t*>(argmax_output.ptr<int32_t>())),
              reinterpret_cast<size_t*>(const_cast<int32_t*>(argmax_pinned.ptr<int32_t>())),
              stream);
          cudaStreamSynchronize(stream);
          next = static_cast<int32_t>(*reinterpret_cast<size_t*>(
              const_cast<int32_t*>(argmax_pinned.ptr<int32_t>())));
        } else {
          cudaStreamSynchronize(stream);
          next = sampler_->sample(forward_out.ptr<float>(), forward_out.size(), stream);
        }
        return error::Success();
      }
      graph_ctx->invalidate();
    }
  }

  // Normal (non-graph) execution
  tensor::Tensor pos_tensor = get_buffer(ModelBufferType::kInputPos);
  *const_cast<int32_t*>(pos_tensor.ptr<int32_t>()) = pos;

  // Optional per-category timing (activated on step 5 to skip warmup)
  // Set Q35_PROFILE_DECODE=1 environment variable to enable
  static int profile_step_counter = 0;
  static bool profile_enabled = (std::getenv("Q35_PROFILE_DECODE") != nullptr);
  const bool do_profile = profile_enabled && (profile_step_counter == 5);
  profile_step_counter++;

  cudaEvent_t ev_start, ev_full_attn, ev_linear_attn, ev_ffn, ev_cls, ev_end;
  float t_full_attn_ms = 0, t_linear_attn_ms = 0, t_ffn_ms = 0, t_cls_ms = 0;
  if (do_profile) {
    cudaEventCreate(&ev_start); cudaEventCreate(&ev_full_attn);
    cudaEventCreate(&ev_linear_attn); cudaEventCreate(&ev_ffn);
    cudaEventCreate(&ev_cls); cudaEventCreate(&ev_end);
  }

  float total_full_attn = 0, total_linear_attn = 0, total_ffn = 0;

  for (int il = 0; il < n_layers; ++il) {
    if (do_profile) cudaEventRecord(ev_start, stream);

    if (q35_config_.is_full_attn_layer(il)) {
      full_attn_decode(il, decode_input);
      if (do_profile) {
        cudaEventRecord(ev_full_attn, stream);
        cudaEventSynchronize(ev_full_attn);
        float ms; cudaEventElapsedTime(&ms, ev_start, ev_full_attn);
        total_full_attn += ms;
      }
    } else {
      linear_attn_decode(il, decode_input);
      if (do_profile) {
        cudaEventRecord(ev_linear_attn, stream);
        cudaEventSynchronize(ev_linear_attn);
        float ms; cudaEventElapsedTime(&ms, ev_start, ev_linear_attn);
        total_linear_attn += ms;
      }
    }

    if (do_profile) cudaEventRecord(ev_start, stream);
    q35_feed_forward(il, decode_input);
    if (do_profile) {
      cudaEventRecord(ev_ffn, stream);
      cudaEventSynchronize(ev_ffn);
      float ms; cudaEventElapsedTime(&ms, ev_start, ev_ffn);
      total_ffn += ms;
    }
  }

  if (do_profile) cudaEventRecord(ev_start, stream);
  q35_cls_logits(decode_input);
  if (do_profile) {
    cudaEventRecord(ev_cls, stream);
    cudaEventSynchronize(ev_cls);
    float ms; cudaEventElapsedTime(&ms, ev_start, ev_cls);
    t_cls_ms = ms;
  }

  cudaStreamSynchronize(stream);

  if (do_profile) {
    float total = total_full_attn + total_linear_attn + total_ffn + t_cls_ms;
    LOG(INFO) << "\n=== Decode Step Profile (step 5) ===";
    LOG(INFO) << "  Full Attention (8 layers):  " << total_full_attn << " ms (" 
              << (100.0f * total_full_attn / total) << "%)";
    LOG(INFO) << "  Linear Attn/GDN (24 layers): " << total_linear_attn << " ms ("
              << (100.0f * total_linear_attn / total) << "%)";
    LOG(INFO) << "  FFN (32 layers):             " << total_ffn << " ms ("
              << (100.0f * total_ffn / total) << "%)";
    LOG(INFO) << "  LM Head (cls_logits):        " << t_cls_ms << " ms ("
              << (100.0f * t_cls_ms / total) << "%)";
    LOG(INFO) << "  Total GPU time:              " << total << " ms";
    LOG(INFO) << "=================================";
    cudaEventDestroy(ev_start); cudaEventDestroy(ev_full_attn);
    cudaEventDestroy(ev_linear_attn); cudaEventDestroy(ev_ffn);
    cudaEventDestroy(ev_cls); cudaEventDestroy(ev_end);
  }

  tensor::Tensor forward_out = get_buffer(ModelBufferType::kForwardOutput);
  next = sampler_->sample(forward_out.ptr<float>(), forward_out.size(), stream);

  return error::Success();
}

// ==========================================================================
// Prefill
// ==========================================================================

base::Status Qwen35Model::prefill(const tensor::Tensor& input_embeddings,
                                  int32_t seq_len, int32_t start_pos) const {
  auto stream = cuda_config_->stream;
  int n_layers = q35_config_.num_hidden_layers;
  int dim = q35_config_.hidden_size;
  int intermediate = q35_config_.intermediate_size;
  int n_heads = q35_config_.num_attention_heads;
  int n_kv_heads = q35_config_.num_key_value_heads;
  int kv_dim = q35_config_.kv_dim();
  int q_dim = q35_config_.q_dim();
  int q_gate_dim = q35_config_.q_gate_dim();
  int head_dim = q35_config_.head_dim;
  int partial_dim = q35_config_.partial_rope_dim();
  int conv_dim = q35_config_.conv_dim();
  int n_k_heads = q35_config_.linear_num_key_heads;
  int n_v_heads = q35_config_.linear_num_value_heads;
  int k_head_dim = q35_config_.linear_key_head_dim;
  int v_head_dim = q35_config_.linear_value_head_dim;
  int ks = q35_config_.linear_conv_kernel_dim;
  int kv_mul = config_->kv_mul_;
  int n_full = static_cast<int>(q35_config_.full_attn_layer_indices.size());
  int q_size = n_k_heads * k_head_dim;
  int k_size = n_k_heads * k_head_dim;
  int v_size = n_v_heads * v_head_dim;

  // Upload M-RoPE positions (generate text-only sequential positions if needed)
  if (mrope_pos_t_.empty()) {
    mrope_pos_t_.resize(seq_len);
    mrope_pos_h_.resize(seq_len);
    mrope_pos_w_.resize(seq_len);
    for (int i = 0; i < seq_len; ++i) {
      mrope_pos_t_[i] = i;
      mrope_pos_h_[i] = i;
      mrope_pos_w_[i] = i;
    }
    mrope_max_text_pos_ = seq_len - 1;
  }
  vision_vl_layers_.generate_mrope_positions_layer_->upload(
      mrope_pos_t_, mrope_pos_h_, mrope_pos_w_,
      mrope_pos_t_gpu_, mrope_pos_h_gpu_, mrope_pos_w_gpu_,
      cuda_config_->stream);

  auto alloc = base::CUDADeviceAllocatorFactory::get_instance();
  auto act = DataType::kDataTypeFp16;

  // ==========================================
  // Pre-allocate ALL working buffers (reused across layers)
  // ==========================================

  // Common buffers
  tensor::Tensor hidden0(act, seq_len, dim, true, alloc);
  tensor::Tensor hidden1(act, seq_len, dim, true, alloc);
  tensor::Tensor rms_out(act, seq_len, dim, true, alloc);
  tensor::Tensor ffn_norm(act, seq_len, dim, true, alloc);
  tensor::Tensor w1_out(act, seq_len, intermediate, true, alloc);
  tensor::Tensor w3_out(act, seq_len, intermediate, true, alloc);
  tensor::Tensor w2_out(act, seq_len, dim, true, alloc);
  tensor::Tensor attn_out_proj(act, seq_len, dim, true, alloc);

  // Full attention buffers (reused across 8 layers)
  tensor::Tensor query_gate(act, seq_len, q_gate_dim, true, alloc);
  tensor::Tensor key_buf(act, seq_len, kv_dim, true, alloc);
  tensor::Tensor val_buf(act, seq_len, kv_dim, true, alloc);
  tensor::Tensor mha_out(act, seq_len, dim, true, alloc);
  tensor::Tensor q_extracted(act, seq_len, q_dim, true, alloc);
  tensor::Tensor gate_extracted(act, seq_len, q_dim, true, alloc);

  int kv_len = start_pos + seq_len;
  half* score_buf = nullptr;
  cudaMalloc(&score_buf, (int64_t)n_heads * kv_len * seq_len * sizeof(half));

  // Linear attention buffers (reused across 24 layers)
  tensor::Tensor lin_qkv(act, seq_len, conv_dim, true, alloc);
  tensor::Tensor lin_z(act, seq_len, dim, true, alloc);
  tensor::Tensor lin_alpha(act, seq_len, n_v_heads, true, alloc);
  tensor::Tensor lin_beta(act, seq_len, n_v_heads, true, alloc);
  tensor::Tensor lin_conv_out(act, seq_len, conv_dim, true, alloc);
  tensor::Tensor lin_gdn_out(act, seq_len, n_v_heads * v_head_dim, true, alloc);
  tensor::Tensor lin_normed(act, seq_len, dim, true, alloc);
  tensor::Tensor gate_fp32(DataType::kDataTypeFp32, seq_len * n_v_heads, true, alloc);
  tensor::Tensor beta_fp32(DataType::kDataTypeFp32, seq_len * n_v_heads, true, alloc);
  tensor::Tensor q_normed(act, seq_len * q_size, true, alloc);
  tensor::Tensor k_normed(act, seq_len * k_size, true, alloc);
  tensor::Tensor v_gathered(act, seq_len * v_size, true, alloc);

  // Transposed state buffer for optimized coalesced GDN prefill
  // Standard: [v_head, v_dim, k_dim], Transposed: [v_head, k_dim, v_dim]
  tensor::Tensor state_transposed(DataType::kDataTypeFp32,
      n_v_heads * k_head_dim * v_head_dim, true, alloc);

  // ==========================================
  // Pre-compute batched attention pointer arrays for all full_attn layers
  // ==========================================
  const int ptrs_per_layer = 6 * n_heads;
  const size_t total_ptrs = n_full * ptrs_per_layer;

  half** d_attn_ptrs = nullptr;
  if (n_full > 0 && total_ptrs > 0) {
    cudaMalloc(&d_attn_ptrs, total_ptrs * sizeof(half*));
    half** h_ptrs = nullptr;
    cudaMallocHost(&h_ptrs, total_ptrs * sizeof(half*));

    half* kc_base = const_cast<half*>(get_buffer(ModelBufferType::kKeyCache).ptr<half>());
    half* vc_base = const_cast<half*>(get_buffer(ModelBufferType::kValueCache).ptr<half>());

    for (int ti = 0; ti < n_full; ++ti) {
      half* kc = kc_base + (int64_t)ti * config_->seq_len_ * kv_dim + start_pos * kv_dim;
      half* vc = vc_base + (int64_t)ti * config_->seq_len_ * kv_dim + start_pos * kv_dim;
      int base_off = ti * ptrs_per_layer;

      for (int h = 0; h < n_heads; ++h) {
        int kv_h = h / kv_mul;
        // Step 1: Q · K^T  (A=K, B=Q, C=Score)
        h_ptrs[base_off + h]                = kc + kv_h * head_dim;
        h_ptrs[base_off + n_heads + h]      = q_extracted.ptr<half>() + h * head_dim;
        h_ptrs[base_off + 2 * n_heads + h]  = score_buf + (int64_t)h * kv_len * seq_len;
        // Step 3: Score · V → mha_out  (A=V, B=Score, C=mha_out)
        h_ptrs[base_off + 3 * n_heads + h]  = vc + kv_h * head_dim;
        h_ptrs[base_off + 4 * n_heads + h]  = score_buf + (int64_t)h * kv_len * seq_len;
        h_ptrs[base_off + 5 * n_heads + h]  = mha_out.ptr<half>() + h * head_dim;
      }
    }

    cudaMemcpyAsync(d_attn_ptrs, h_ptrs, total_ptrs * sizeof(half*),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    cudaFreeHost(h_ptrs);
  }

  // ==========================================
  // Layer loop (double-buffered)
  // ==========================================
  tensor::Tensor* cur_in = nullptr;
  tensor::Tensor* cur_out = nullptr;

  for (int il = 0; il < n_layers; ++il) {
    if (il == 0) {
      cur_in = const_cast<tensor::Tensor*>(&input_embeddings);
      cur_out = &hidden0;
    } else {
      cur_in = (il % 2 == 1) ? &hidden0 : &hidden1;
      cur_out = (il % 2 == 1) ? &hidden1 : &hidden0;
    }

    // RMSNorm
    auto rms_il = std::dynamic_pointer_cast<op::LayerParam>(qwen_layers_->rmsnorm_layers_[il]);
    batched_rmsnorm_fp16_layer_->forward(cur_in->ptr<half>(), rms_out.ptr<half>(),
        rms_il->get_weight(0).ptr<half>(),
        dim, seq_len, q35_config_.rms_norm_eps);

    if (q35_config_.is_full_attn_layer(il)) {
      int ti = full_attn_type_idx(il);
      int q_norm_idx = 2 * n_layers + 1 + ti;
      int k_norm_idx = 2 * n_layers + 1 + n_full + ti;

      const half alpha_h = __float2half(1.0f);
      const half beta_h = __float2half(0.0f);

      // Q (with gate), K, V projections via cuBLAS
      auto wq = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->wq_layers_[ti]);
      auto wk = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->wk_layers_[ti]);
      auto wv = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->wv_layers_[ti]);

      cublasHgemm(cuda_config_->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
          q_gate_dim, seq_len, dim, &alpha_h,
          wq->get_weight(0).ptr<half>(), dim,
          rms_out.ptr<half>(), dim,
          &beta_h, query_gate.ptr<half>(), q_gate_dim);

      cublasHgemm(cuda_config_->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
          kv_dim, seq_len, dim, &alpha_h,
          wk->get_weight(0).ptr<half>(), dim,
          rms_out.ptr<half>(), dim,
          &beta_h, key_buf.ptr<half>(), kv_dim);

      cublasHgemm(cuda_config_->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
          kv_dim, seq_len, dim, &alpha_h,
          wv->get_weight(0).ptr<half>(), dim,
          rms_out.ptr<half>(), dim,
          &beta_h, val_buf.ptr<half>(), kv_dim);

      // Deinterleave Q and Gate per-head
      deinterleave_q_gate_layer_->forward(
          query_gate.ptr<half>(), q_extracted.ptr<half>(), gate_extracted.ptr<half>(),
          n_heads, head_dim, seq_len);

      // Per-head Q/K norm
      auto rms_qn = std::dynamic_pointer_cast<op::LayerParam>(qwen_layers_->rmsnorm_layers_[q_norm_idx]);
      auto rms_kn = std::dynamic_pointer_cast<op::LayerParam>(qwen_layers_->rmsnorm_layers_[k_norm_idx]);
      batched_rmsnorm_fp16_layer_->forward_dim(
          q_extracted.ptr<half>(), q_extracted.ptr<half>(),
          rms_qn->get_weight(0).ptr<half>(),
          head_dim, seq_len * n_heads, q35_config_.rms_norm_eps);
      batched_rmsnorm_fp16_layer_->forward_dim(
          key_buf.ptr<half>(), key_buf.ptr<half>(),
          rms_kn->get_weight(0).ptr<half>(),
          head_dim, seq_len * n_kv_heads, q35_config_.rms_norm_eps);

      // Partial M-RoPE (interleaved)
      partial_mrope_layer_->forward_batched(
          q_extracted.ptr<half>(), key_buf.ptr<half>(),
          get_buffer(ModelBufferType::kSinCache).ptr<float>(),
          get_buffer(ModelBufferType::kCosCache).ptr<float>(),
          mrope_pos_t_gpu_ + start_pos, mrope_pos_h_gpu_ + start_pos,
          mrope_pos_w_gpu_ + start_pos,
          mrope_sections_cpu_, partial_dim, head_dim, n_heads, n_kv_heads, seq_len);

      // Write K, V to cache
      half* kc_ptr = const_cast<half*>(get_buffer(ModelBufferType::kKeyCache).ptr<half>())
          + static_cast<size_t>(ti) * config_->seq_len_ * kv_dim + start_pos * kv_dim;
      half* vc_ptr = const_cast<half*>(get_buffer(ModelBufferType::kValueCache).ptr<half>())
          + static_cast<size_t>(ti) * config_->seq_len_ * kv_dim + start_pos * kv_dim;
      cudaMemcpyAsync(kc_ptr, key_buf.ptr<half>(), seq_len * kv_dim * sizeof(half),
                      cudaMemcpyDeviceToDevice, stream);
      cudaMemcpyAsync(vc_ptr, val_buf.ptr<half>(), seq_len * kv_dim * sizeof(half),
                      cudaMemcpyDeviceToDevice, stream);

      // Batched MHA via cublasHgemmBatched (replaces per-head loop)
      float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
      const half scale_h = __float2half(scale);
      half** layer_ptrs = d_attn_ptrs + ti * ptrs_per_layer;

      // Step 1: Q · K^T (batched over all heads)
      cublasHgemmBatched(cuda_config_->cublas_handle,
          CUBLAS_OP_T, CUBLAS_OP_N,
          kv_len, seq_len, head_dim, &scale_h,
          (const half**)layer_ptrs, kv_dim,
          (const half**)(layer_ptrs + n_heads), q_dim,
          &beta_h,
          layer_ptrs + 2 * n_heads, kv_len,
          n_heads);

      // Step 2: Causal softmax (all heads at once)
      vision_vl_layers_.causal_softmax_layer_->forward(
          score_buf, n_heads, seq_len, kv_len, start_pos, stream);

      // Step 3: Attn · V → mha_out (batched)
      half** step3 = layer_ptrs + 3 * n_heads;
      cublasHgemmBatched(cuda_config_->cublas_handle,
          CUBLAS_OP_N, CUBLAS_OP_N,
          head_dim, seq_len, kv_len, &alpha_h,
          (const half**)step3, kv_dim,
          (const half**)(step3 + n_heads), kv_len,
          &beta_h,
          step3 + 2 * n_heads, dim,
          n_heads);

      // Apply sigmoid gate
      apply_sigmoid_gate_layer_->forward_batched(mha_out.ptr<half>(), gate_extracted.ptr<half>(),
                                                  q_dim, seq_len);

      // WO projection
      auto wo = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->wo_layers_[ti]);
      cublasHgemm(cuda_config_->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
          dim, seq_len, q_dim, &alpha_h,
          wo->get_weight(0).ptr<half>(), q_dim,
          mha_out.ptr<half>(), q_dim,
          &beta_h, attn_out_proj.ptr<half>(), dim);

      // Residual
      batched_add_fp16_layer_->forward(cur_in->ptr<half>(), attn_out_proj.ptr<half>(),
                                       cur_out->ptr<half>(), dim, seq_len);

    } else {
      // Linear attention layer (GDN)
      int ti = linear_attn_type_idx(il);
      auto& la = linear_attn_weights_->layers[ti];
      auto& state = gdn_states_[ti];

      const half alpha_h = __float2half(1.0f);
      const half beta_h = __float2half(0.0f);

      // Projections via cuBLAS
      auto proj_qkv = std::dynamic_pointer_cast<op::MatmulLayer>(la.in_proj_qkv);
      auto proj_z = std::dynamic_pointer_cast<op::MatmulLayer>(la.in_proj_z);
      auto proj_a = std::dynamic_pointer_cast<op::MatmulLayer>(la.in_proj_a);
      auto proj_b = std::dynamic_pointer_cast<op::MatmulLayer>(la.in_proj_b);

      cublasHgemm(cuda_config_->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
          conv_dim, seq_len, dim, &alpha_h,
          proj_qkv->get_weight(0).ptr<half>(), dim,
          rms_out.ptr<half>(), dim, &beta_h,
          lin_qkv.ptr<half>(), conv_dim);

      cublasHgemm(cuda_config_->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
          dim, seq_len, dim, &alpha_h,
          proj_z->get_weight(0).ptr<half>(), dim,
          rms_out.ptr<half>(), dim, &beta_h,
          lin_z.ptr<half>(), dim);

      cublasHgemm(cuda_config_->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
          n_v_heads, seq_len, dim, &alpha_h,
          proj_a->get_weight(0).ptr<half>(), dim,
          rms_out.ptr<half>(), dim, &beta_h,
          lin_alpha.ptr<half>(), n_v_heads);

      cublasHgemm(cuda_config_->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
          n_v_heads, seq_len, dim, &alpha_h,
          proj_b->get_weight(0).ptr<half>(), dim,
          rms_out.ptr<half>(), dim, &beta_h,
          lin_beta.ptr<half>(), n_v_heads);

      // Conv1D
      causal_conv1d_silu_layer_->forward_batched(
          state.conv_state.ptr<half>(), lin_qkv.ptr<half>(),
          la.conv_weight.ptr<half>(), lin_conv_out.ptr<half>(),
          conv_dim, ks, seq_len);

      // Extract Q, K, V from strided conv_out into contiguous buffers (single kernel each)
      gather_strided_layer_->forward(lin_conv_out.ptr<half>(), q_normed.ptr<half>(),
                                      q_size, conv_dim, 0, seq_len);
      gather_strided_layer_->forward(lin_conv_out.ptr<half>(), k_normed.ptr<half>(),
                                      k_size, conv_dim, q_size, seq_len);
      gather_strided_layer_->forward(lin_conv_out.ptr<half>(), v_gathered.ptr<half>(),
                                      v_size, conv_dim, q_size + k_size, seq_len);

      // L2 norm per-head for all tokens at once (single kernel launch)
      l2_norm_per_head_layer_->forward(q_normed.ptr<half>(), q_normed.ptr<half>(),
                                        seq_len * n_k_heads, k_head_dim, 1e-6f);
      l2_norm_per_head_layer_->forward(k_normed.ptr<half>(), k_normed.ptr<half>(),
                                        seq_len * n_k_heads, k_head_dim, 1e-6f);

      // Gates
      compute_gdn_gates_layer_->forward_batched(
          lin_alpha.ptr<half>(), la.dt_bias.ptr<half>(),
          la.A_log.ptr<float>(), lin_beta.ptr<half>(),
          gate_fp32.ptr<float>(), beta_fp32.ptr<float>(),
          n_v_heads, seq_len);

      // Transpose state [v_head, v_dim, k_dim] → [v_head, k_dim, v_dim] for coalesced access
      transpose_state_layer_->forward(state.ssm_state.ptr<float>(),
                                       state_transposed.ptr<float>(),
                                       n_v_heads, v_head_dim, k_head_dim);

      // Optimized GDN prefill with transposed state (coalesced memory access)
      gdn_prefill_transposed_layer_->forward(
          q_normed.ptr<half>(), k_normed.ptr<half>(), v_gathered.ptr<half>(),
          gate_fp32.ptr<float>(), beta_fp32.ptr<float>(),
          state_transposed.ptr<float>(), lin_gdn_out.ptr<half>(),
          n_k_heads, n_v_heads, k_head_dim, v_head_dim, seq_len);

      // Transpose state back [v_head, k_dim, v_dim] → [v_head, v_dim, k_dim] for decode
      transpose_state_layer_->forward(state_transposed.ptr<float>(),
                                       state.ssm_state.ptr<float>(),
                                       n_v_heads, k_head_dim, v_head_dim);

      // Gated RMSNorm
      gated_rmsnorm_layer_->forward_batched(
          lin_gdn_out.ptr<half>(), lin_z.ptr<half>(),
          la.norm_weight.ptr<float>(), lin_normed.ptr<half>(),
          dim, seq_len, q35_config_.rms_norm_eps);

      // Output projection
      auto out_p = std::dynamic_pointer_cast<op::MatmulLayer>(la.out_proj);
      cublasHgemm(cuda_config_->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
          dim, seq_len, dim, &alpha_h,
          out_p->get_weight(0).ptr<half>(), dim,
          lin_normed.ptr<half>(), dim, &beta_h,
          attn_out_proj.ptr<half>(), dim);

      // Residual
      batched_add_fp16_layer_->forward(cur_in->ptr<half>(), attn_out_proj.ptr<half>(),
                                       cur_out->ptr<half>(), dim, seq_len);
    }

    // FFN (common for both layer types)
    auto rms_ffn = std::dynamic_pointer_cast<op::LayerParam>(qwen_layers_->rmsnorm_layers_[il + n_layers]);
    batched_rmsnorm_fp16_layer_->forward(cur_out->ptr<half>(), ffn_norm.ptr<half>(),
        rms_ffn->get_weight(0).ptr<half>(),
        dim, seq_len, q35_config_.rms_norm_eps);

    const half alpha_h = __float2half(1.0f);
    const half beta_h = __float2half(0.0f);

    auto w1 = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->w1_layers_[il]);
    auto w2 = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->w2_layers_[il]);
    auto w3 = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->w3_layers_[il]);

    cublasHgemm(cuda_config_->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
        intermediate, seq_len, dim, &alpha_h,
        w1->get_weight(0).ptr<half>(), dim,
        ffn_norm.ptr<half>(), dim, &beta_h,
        w1_out.ptr<half>(), intermediate);

    cublasHgemm(cuda_config_->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
        intermediate, seq_len, dim, &alpha_h,
        w3->get_weight(0).ptr<half>(), dim,
        ffn_norm.ptr<half>(), dim, &beta_h,
        w3_out.ptr<half>(), intermediate);

    // Batched SwiGLU
    qwen_layers_->batched_swiglu_layer_->forward(w1_out, w3_out, w1_out);

    cublasHgemm(cuda_config_->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
        dim, seq_len, intermediate, &alpha_h,
        w2->get_weight(0).ptr<half>(), intermediate,
        w1_out.ptr<half>(), intermediate, &beta_h,
        w2_out.ptr<half>(), dim);

    // Residual add
    batched_add_fp16_layer_->forward(cur_out->ptr<half>(), w2_out.ptr<half>(),
                                     cur_out->ptr<half>(), dim, seq_len);
  }

  // Free batched attention buffers
  if (score_buf) cudaFree(score_buf);
  if (d_attn_ptrs) cudaFree(d_attn_ptrs);

  // Copy last token's hidden state to decode input buffer
  tensor::Tensor& final_hidden = (n_layers % 2 == 1) ? hidden0 : hidden1;
  void* decode_dst = const_cast<void*>(get_buffer(ModelBufferType::kDecodeInput).ptr<void>());
  
  if (n_layers == 0) {
    cudaMemcpyAsync(decode_dst,
                    input_embeddings.ptr<half>() + (seq_len - 1) * dim,
                    dim * sizeof(half), cudaMemcpyDeviceToDevice, stream);
  } else {
    cudaMemcpyAsync(decode_dst,
                    final_hidden.ptr<half>() + (seq_len - 1) * dim,
                    dim * sizeof(half), cudaMemcpyDeviceToDevice, stream);
  }

  prefill_seq_len_ = seq_len;
  cudaStreamSynchronize(stream);
  return error::Success();
}

// ==========================================================================
// Sample First Token (after prefill)
// ==========================================================================

int Qwen35Model::sample_first_token() const {
  tensor::Tensor decode_input = get_buffer(ModelBufferType::kDecodeInput);
  q35_cls_logits(decode_input);
  cudaStreamSynchronize(cuda_config_->stream);
  
  tensor::Tensor forward_out = get_buffer(ModelBufferType::kForwardOutput);
  
  return sampler_->sample(forward_out.ptr<float>(), forward_out.size(), cuda_config_->stream);
}

}  // namespace model

#endif  // QWEN3_VL_SUPPORT
