/**
 * @file qwen3_5_base.cpp
 * @brief Qwen3.5-9B Base: Construction, Destruction, Init, Model Loading, Memory, State
 *
 * Extracted from qwen3_5.cpp to simplify the main inference file.
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

}  // namespace model

#endif  // QWEN3_VL_SUPPORT
