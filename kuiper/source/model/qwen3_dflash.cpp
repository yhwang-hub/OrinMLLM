#ifdef QWEN3_SUPPORT
#include "model/qwen3_dflash.h"
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <op/matmul.h>
#include <op/rmsnorm.h>
#include <op/embedding.h>
#include "../op/kernels/cuda/matmul_kernel.cuh"
#include "../op/kernels/cuda/rmsnorm_kernel.cuh"
#include "../op/kernels/cuda/add_kernel.cuh"
#include "../op/kernels/cuda/swiglu_kernel.cuh"
#include "../op/kernels/cuda/flash_attention_kernel.cuh"
#include "../op/kernels/cuda/kv_cache_kernel.cuh"
#include "../op/kernels/cuda/fused_ffn_kernel.cuh"
#include "../op/kernels/cuda/rope_kernel.cuh"
#include "sampler/argmax_sampler.h"
#include "base/tick.h"

namespace model {

// ==================== Model file type detection ====================

bool is_dflash_model_file(const std::string& model_path) {
  FILE* file = fopen(model_path.c_str(), "rb");
  if (!file) return false;

  uint32_t magic = 0;
  int32_t version = 0;
  bool is_dflash = false;

  if (fread(&magic, sizeof(uint32_t), 1, file) == 1 &&
      fread(&version, sizeof(int32_t), 1, file) == 1) {
    // DFlash format: magic=0x64663136 ("df16"), version=7
    is_dflash = (magic == 0x64663136 && version == 7);
  }

  fclose(file);
  return is_dflash;
}

// ==================== Constructor ====================

Qwen3DFlashModel::Qwen3DFlashModel(base::TokenizerType tokenizer_type, std::string token_path,
                                   std::string model_path, bool is_quant_model)
    : Qwen3Model(tokenizer_type, std::move(token_path), std::move(model_path), is_quant_model) {}

// ==================== Init ====================

base::Status Qwen3DFlashModel::init(base::DeviceType device_type) {
  using namespace base;
  if (token_path_.empty()) {
    return error::PathNotValid(token_path_);
  }
  if (device_type == base::DeviceType::kDeviceCPU) {
    return error::InternalError("DFlash only supports CUDA device.");
  }

  device_type_ = device_type;
  cudaSetDevice(0);
  cuda_config_ = std::make_shared<kernel::CudaConfig>();
  cudaStreamCreate(&cuda_config_->stream);

  cublasStatus_t cublas_status = cublasCreate(&cuda_config_->cublas_handle);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    return error::InternalError("Failed to create cuBLAS handle.");
  }
  cublasSetStream(cuda_config_->cublas_handle, cuda_config_->stream);
  cublasSetMathMode(cuda_config_->cublas_handle, CUBLAS_DEFAULT_MATH);

  Status read_status = gen_model_from_file();
  if (!read_status) return read_status;

  init_mem();

  // Initialize sin/cos cache
  CHECK_NE(qwen_layers_->sin_cos_cache_layer_, nullptr);
  qwen_layers_->sin_cos_cache_layer_->forward(config_->head_size_, config_->seq_len_,
                                              get_buffer(ModelBufferType::kSinCache),
                                              get_buffer(ModelBufferType::kCosCache));

  // Move DFlash layers to CUDA
  if (cuda_config_) {
    auto set_fp16_flag = [](const std::shared_ptr<op::Layer>& layer) {
      if (auto lp = std::dynamic_pointer_cast<op::LayerParam>(layer)) {
        lp->set_keep_fp16_weights(true);
      }
    };
    if (fc_layer_) {
      set_fp16_flag(fc_layer_);
      fc_layer_->set_cuda_config(cuda_config_);
      fc_layer_->to_cuda();
    }
    if (hidden_norm_layer_) {
      set_fp16_flag(hidden_norm_layer_);
      hidden_norm_layer_->set_cuda_config(cuda_config_);
      hidden_norm_layer_->to_cuda();
    }
  }

  // Read DFlash-specific config (already partially done in create_param_layers)
  LOG(INFO) << "DFlash model initialized:";
  LOG(INFO) << "  draft_layers=" << config_->layer_num_;
  LOG(INFO) << "  block_size=" << dflash_config_.block_size;
  LOG(INFO) << "  n_target_layers=" << dflash_config_.n_target_layers;
  LOG(INFO) << "  mask_token_id=" << dflash_config_.mask_token_id;
  std::string layer_ids_str;
  for (size_t i = 0; i < dflash_config_.target_layer_ids.size(); ++i) {
    if (i > 0) layer_ids_str += ", ";
    layer_ids_str += std::to_string(dflash_config_.target_layer_ids[i]);
  }
  LOG(INFO) << "  target_layer_ids=[" << layer_ids_str << "]";

  return error::Success();
}

// ==================== Read DFlash Header ====================

base::Status Qwen3DFlashModel::read_dflash_header() {
  // DFlash header is read in create_param_layers()
  return base::error::Success();
}

// ==================== Layer Creation ====================

void Qwen3DFlashModel::create_nonparam_layers() {
  // Create all standard Qwen3 non-parametric layers
  Qwen3Model::create_nonparam_layers();
}

void Qwen3DFlashModel::create_param_layers() {
  CHECK(qwen_layers_ != nullptr);
  LOG(INFO) << "Loading DFlash FP16 model weights...";

  int32_t dim = config_->dim_;
  int32_t kv_dim = config_->kv_dim_;
  int32_t hidden_dim = config_->hidden_dim_;
  int32_t immediate_dim = config_->immediate_dim_;
  auto cpu_device_type = base::DeviceType::kDeviceCPU;

  // DFlash-specific: read n_target_extract_layers from header
  // (We need this before loading weights to know fc input size)
  // Re-read dflash header for target_layer_ids count
  {
    FILE* hdr = fopen(model_path_.c_str(), "rb");
    if (hdr) {
      // Skip to block_size position: magic(4)+version(4)+config(28)+shared(1)+head_dim(4)=41
      fseek(hdr, 41, SEEK_SET);
      int32_t bs, ntl;
      fread(&bs, sizeof(int32_t), 1, hdr);
      fread(&ntl, sizeof(int32_t), 1, hdr);
      dflash_config_.block_size = bs;
      dflash_config_.n_target_layers = ntl;
      // Read target_layer_ids
      dflash_config_.target_layer_ids.resize(config_->layer_num_);
      for (int32_t i = 0; i < config_->layer_num_; ++i) {
        fread(&dflash_config_.target_layer_ids[i], sizeof(int32_t), 1, hdr);
      }
      int32_t mask_id;
      fread(&mask_id, sizeof(int32_t), 1, hdr);
      dflash_config_.mask_token_id = mask_id;
      fclose(hdr);
    }
  }

  int32_t n_target_extract_layers = static_cast<int32_t>(dflash_config_.target_layer_ids.size());
  if (n_target_extract_layers == 0) n_target_extract_layers = config_->layer_num_;
  int32_t fc_input_dim = n_target_extract_layers * dim;

  // Weight layout in DFlash bin (all FP16):
  // offset 0:          fc.weight [dim, fc_input_dim]
  // offset fc_size:    hidden_norm.weight [dim]
  // offset hdr_end:    <standard Qwen3 weights>
  //   attention_norm x layers [dim]
  //   ffn_norm x layers [dim]
  //   final norm [dim]
  //   wq x layers [dim, dim]
  //   wk x layers [kv_dim, dim]
  //   wv x layers [kv_dim, dim]
  //   wo x layers [dim, dim]
  //   w1 x layers [immediate_dim, dim]
  //   w2 x layers [dim, immediate_dim]
  //   w3 x layers [immediate_dim, dim]
  //   q_norm x layers [head_size]
  //   k_norm x layers [head_size]

  size_t pos = 0;

  // 1. fc layer
  fc_layer_ = std::make_shared<op::MatmulLayer>(device_type_, dim, fc_input_dim, false);
  auto fc_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(fc_layer_);
  fc_matmul->set_weight_fp16(0, {dim, fc_input_dim},
                             raw_model_data_->weight(pos), cpu_device_type);
  pos += (size_t)dim * fc_input_dim;

  // 2. hidden_norm layer
  hidden_norm_layer_ = std::make_shared<op::RmsNormLayer>(device_type_, dim);
  auto hidden_norm_rms = std::dynamic_pointer_cast<op::RmsNormLayer>(hidden_norm_layer_);
  hidden_norm_rms->set_weight_fp16(0, {dim},
                                   raw_model_data_->weight(pos), cpu_device_type);
  pos += dim;

  // === Standard Qwen3 weights follow ===

  // 3. attention_norm (input_layernorm) for all layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto rms = std::make_shared<op::RmsNormLayer>(device_type_, dim);
    rms->set_weight_fp16(0, {dim}, raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->rmsnorm_layers_.push_back(rms);
    pos += dim;
  }

  // 4. ffn_norm (post_attention_layernorm) for all layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto rms = std::make_shared<op::RmsNormLayer>(device_type_, dim);
    rms->set_weight_fp16(0, {dim}, raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->rmsnorm_layers_.push_back(rms);
    pos += dim;
  }

  // 5. final norm
  {
    auto rms = std::make_shared<op::RmsNormLayer>(device_type_, dim);
    rms->set_weight_fp16(0, {dim}, raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->rmsnorm_layers_.push_back(rms);
    pos += dim;
  }

  // Note: DFlash has no embedding or lm_head (shared with target model)
  // We create a placeholder embedding that won't be used directly
  // The target model's embedding and lm_head are used instead

  // 6. wq layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, false);
    wq->set_weight_fp16(0, {dim, dim}, raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->wq_layers_.push_back(wq);
    pos += (size_t)dim * dim;
  }

  // 7. wk layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wk = std::make_shared<op::MatmulLayer>(device_type_, kv_dim, dim, false);
    wk->set_weight_fp16(0, {kv_dim, dim}, raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->wk_layers_.push_back(wk);
    pos += (size_t)kv_dim * dim;
  }

  // 8. wv layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wv = std::make_shared<op::MatmulLayer>(device_type_, kv_dim, dim, false);
    wv->set_weight_fp16(0, {kv_dim, dim}, raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->wv_layers_.push_back(wv);
    pos += (size_t)kv_dim * dim;
  }

  // 9. wo layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wo = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, false);
    wo->set_weight_fp16(0, {dim, dim}, raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->wo_layers_.push_back(wo);
    pos += (size_t)dim * dim;
  }

  // 10. w1 layers (gate_proj)
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w1 = std::make_shared<op::MatmulLayer>(device_type_, immediate_dim, dim, false);
    w1->set_weight_fp16(0, {immediate_dim, dim}, raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->w1_layers_.push_back(w1);
    pos += (size_t)dim * immediate_dim;
  }

  // 11. w2 layers (down_proj)
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w2 = std::make_shared<op::MatmulLayer>(device_type_, dim, immediate_dim, false);
    w2->set_weight_fp16(0, {dim, immediate_dim}, raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->w2_layers_.push_back(w2);
    pos += (size_t)immediate_dim * dim;
  }

  // 12. w3 layers (up_proj)
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w3 = std::make_shared<op::MatmulLayer>(device_type_, immediate_dim, dim, false);
    w3->set_weight_fp16(0, {immediate_dim, dim}, raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->w3_layers_.push_back(w3);
    pos += (size_t)dim * immediate_dim;
  }

  // 13. q_norm for all layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto rms = std::make_shared<op::RmsNormLayer>(device_type_, config_->head_size_);
    rms->set_weight_fp16(0, {config_->head_size_},
                         raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->rmsnorm_layers_.push_back(rms);
    pos += config_->head_size_;
  }

  // 14. k_norm for all layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto rms = std::make_shared<op::RmsNormLayer>(device_type_, config_->head_size_);
    rms->set_weight_fp16(0, {config_->head_size_},
                         raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->rmsnorm_layers_.push_back(rms);
    pos += config_->head_size_;
  }

  LOG(INFO) << "DFlash FP16 model loaded. Total FP16 elements: " << pos
            << " (fc: " << dim * fc_input_dim << ", layers: 5)";
}

base::Status Qwen3DFlashModel::create_layers() {
  using namespace base;
  if (!qwen_layers_) {
    qwen_layers_ = std::make_unique<Qwen3Layers>();
  }

  // Create DFlash parametric layers (fc + hidden_norm + standard 5-layer weights)
  create_param_layers();
  // Create non-parametric layers (rope, mha, swiglu, etc.)
  create_nonparam_layers();

  // DFlash has no embedding layer - it shares with the target model
  // So we skip the embedding check that Qwen3Model::create_layers() does.

  // Validate layers
  if (qwen_layers_->rmsnorm_layers_.size() != 4 * config_->layer_num_ + 1) {
    return error::InternalError(
        "DFlash: rmsnorm layer count mismatch. Expected " +
        std::to_string(4 * config_->layer_num_ + 1) + ", got " +
        std::to_string(qwen_layers_->rmsnorm_layers_.size()));
  }

  if (qwen_layers_->wq_layers_.size() != config_->layer_num_ ||
      qwen_layers_->wk_layers_.size() != config_->layer_num_ ||
      qwen_layers_->wv_layers_.size() != config_->layer_num_ ||
      qwen_layers_->wo_layers_.size() != config_->layer_num_) {
    return error::InternalError("DFlash: attention layer count mismatch.");
  }

  if (qwen_layers_->w1_layers_.size() != config_->layer_num_ ||
      qwen_layers_->w2_layers_.size() != config_->layer_num_ ||
      qwen_layers_->w3_layers_.size() != config_->layer_num_) {
    return error::InternalError("DFlash: FFN layer count mismatch.");
  }

  if (!fc_layer_ || !hidden_norm_layer_) {
    return error::InternalError("DFlash: fc or hidden_norm layer not created.");
  }

  return error::Success();
}

// ==================== Memory Init ====================

void Qwen3DFlashModel::init_mem() {
  // Initialize base Qwen3 memory
  Qwen3Model::init_mem();

  // Allocate DFlash-specific buffers
  std::shared_ptr<base::DeviceAllocator> alloc;
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    alloc = base::CUDADeviceAllocatorFactory::get_instance();
  } else {
    alloc = base::CPUDeviceAllocatorFactory::get_instance();
  }

  base::DataType activation_dtype = is_fp16_model_ ?
      base::DataType::kDataTypeFp16 : base::DataType::kDataTypeFp32;

  int32_t block_size = dflash_config_.block_size;
  int32_t dim = config_->dim_;

  // Draft output buffer [block_size, dim]
  draft_output_ = tensor::Tensor(activation_dtype, block_size, dim, true, alloc);

  // Fused context buffer (dynamically resized based on context length)
  // Pre-allocate for max expected context
  int32_t max_context = config_->seq_len_;
  fused_context_ = tensor::Tensor(activation_dtype, max_context, dim, true, alloc);

  // Draft KV cache (for the 5-layer draft model)
  // The draft model uses its own KV cache separate from the target
  draft_key_cache_ = tensor::Tensor(activation_dtype,
      config_->layer_num_, config_->seq_len_, config_->kv_dim_, true, alloc);
  draft_value_cache_ = tensor::Tensor(activation_dtype,
      config_->layer_num_, config_->seq_len_, config_->kv_dim_, true, alloc);

  LOG(INFO) << "DFlash buffers allocated: draft_output[" << block_size << "," << dim
            << "], draft_kv_cache[" << config_->layer_num_ << "," << config_->seq_len_
            << "," << config_->kv_dim_ << "]";

  // Pre-allocate working buffers for draft_forward (Optimization 2)
  draft_hidden_buf_ = tensor::Tensor(activation_dtype, block_size, dim, true, alloc);
  draft_rms_out_ = tensor::Tensor(activation_dtype, block_size, dim, true, alloc);
  draft_query_out_ = tensor::Tensor(activation_dtype, block_size, dim, true, alloc);
  draft_mha_out_ = tensor::Tensor(activation_dtype, block_size, dim, true, alloc);
  draft_wo_out_ = tensor::Tensor(activation_dtype, block_size, dim, true, alloc);
  draft_ffn_norm_out_ = tensor::Tensor(activation_dtype, block_size, dim, true, alloc);
  draft_w1_out_ = tensor::Tensor(activation_dtype, block_size, config_->hidden_dim_, true, alloc);
  draft_w3_out_ = tensor::Tensor(activation_dtype, block_size, config_->hidden_dim_, true, alloc);
  draft_w2_out_ = tensor::Tensor(activation_dtype, block_size, dim, true, alloc);
  draft_rope_dummy_k_ = tensor::Tensor(activation_dtype, block_size, config_->kv_dim_, true, alloc);
  draft_buffers_allocated_ = true;
}

// ==================== Extract and Fuse Context ====================

tensor::Tensor Qwen3DFlashModel::extract_and_fuse_context(
    const std::vector<tensor::Tensor>& per_layer_hidden,
    int32_t seq_len) {
  CHECK(!dflash_config_.target_layer_ids.empty());
  int32_t n_extract = static_cast<int32_t>(dflash_config_.target_layer_ids.size());
  int32_t dim = config_->dim_;

  base::DataType activation_dtype = is_fp16_model_ ?
      base::DataType::kDataTypeFp16 : base::DataType::kDataTypeFp32;
  size_t elem_size = (activation_dtype == base::DataType::kDataTypeFp16)
      ? sizeof(uint16_t) : sizeof(float);

  std::shared_ptr<base::DeviceAllocator> alloc = base::CUDADeviceAllocatorFactory::get_instance();

  // Concatenate along last dimension: [seq_len, n_extract * dim]
  tensor::Tensor concat_hidden(activation_dtype, seq_len, n_extract * dim, true, alloc);

  // Interleaved copy: 5 source tensors → 1 output [seq_len, 5*dim]
  for (int32_t i = 0; i < n_extract; ++i) {
    CHECK_GE(per_layer_hidden[i].size(), seq_len * dim);
    void* dst = static_cast<char*>(
        const_cast<void*>(concat_hidden.get_buffer()->ptr())) +
        i * dim * elem_size;
    const void* src = per_layer_hidden[i].get_buffer()->ptr();
    cudaMemcpy2DAsync(
        dst, (size_t)n_extract * dim * elem_size,
        src, (size_t)dim * elem_size,
        (size_t)dim * elem_size,
        seq_len,
        cudaMemcpyDeviceToDevice, cuda_config_->stream);
  }

  // Apply fc: [seq_len, n_extract * dim] -> [seq_len, dim]
  // Use cublasGemmEx with FP32 output to avoid FP16 overflow (FC has 20480 input dim)
  // Then apply hidden_norm on FP32, then convert to FP16
  tensor::Tensor fc_output(activation_dtype, seq_len, dim, true, alloc);
  auto fc_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(fc_layer_);
  {
    auto& fc_weight = fc_matmul->get_weight(0);
    const int32_t K = fc_weight.get_dim(0);  // output dim = 4096
    const int32_t M = fc_weight.get_dim(1);  // input dim = 20480
    
    // FC output in FP32 to avoid FP16 overflow
    tensor::Tensor fc_fp32(base::DataType::kDataTypeFp32, seq_len, dim, true, alloc);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    cublasGemmEx(
        cuda_config_->cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        K, seq_len, M,
        &alpha,
        fc_weight.get_buffer()->ptr(), CUDA_R_16F, M,
        concat_hidden.get_buffer()->ptr(), CUDA_R_16F, M,
        &beta,
        fc_fp32.get_buffer()->ptr(), CUDA_R_32F, K,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    
    // Apply hidden_norm on FP32 data (uses row_rmsnorm_f32_fp16w_dim kernel)
    // hidden_norm weight is FP16, input/output FP32
    STATUS_CHECK(hidden_norm_layer_->forward(fc_fp32, fc_fp32));
    
    // Convert FP32 → FP16 with clamping
    kernel::fp32_to_fp16_clamp_cu(
        fc_fp32.ptr<float>(),
        const_cast<void*>(fc_output.get_buffer()->ptr()),
        seq_len * dim,
        cuda_config_->stream);
  }

  return fc_output;
}

// ==================== Draft Forward ====================

base::Status Qwen3DFlashModel::draft_forward(
    const tensor::Tensor& noise_embedding,
    const tensor::Tensor& target_hidden,
    int32_t context_len,
    int32_t block_size,
    int32_t start_pos) {
  // DFlash forward pass with zero-copy optimization:
  // K/V projections write directly into draft_key_cache_/draft_value_cache_
  // via view tensors, eliminating 6 cudaMemcpyAsync per layer + 2 global copies.

  if (noise_embedding.is_empty() || target_hidden.is_empty()) {
    return base::error::InvalidArgument("Empty input tensors for DFlash forward.");
  }

  auto* layers = get_base_layers();
  std::shared_ptr<base::DeviceAllocator> alloc = base::CUDADeviceAllocatorFactory::get_instance();

  base::DataType activation_dtype = noise_embedding.data_type();
  size_t elem_size = (activation_dtype == base::DataType::kDataTypeFp16)
      ? sizeof(uint16_t) : sizeof(float);

  int32_t dim = config_->dim_;
  int32_t kv_dim = config_->kv_dim_;
  int32_t total_kv_len = context_len + block_size;

  // Use pre-allocated buffers when block_size matches (Optimization 2)
  bool use_prealloc = draft_buffers_allocated_ && (block_size == dflash_config_.block_size);
  
  // Working buffers: pre-allocated or dynamically created
  // When use_prealloc=true, _dyn variants are default-constructed (empty) and not used.
  tensor::Tensor hidden_buf_dyn, rms_out_dyn, query_out_dyn, mha_out_dyn, wo_out_dyn;
  tensor::Tensor ffn_norm_out_dyn, w1_out_dyn, w3_out_dyn, w2_out_dyn, rope_dummy_k_dyn;
  if (!use_prealloc) {
    hidden_buf_dyn = tensor::Tensor(activation_dtype, block_size, dim, true, alloc);
    rms_out_dyn = tensor::Tensor(activation_dtype, block_size, dim, true, alloc);
    query_out_dyn = tensor::Tensor(activation_dtype, block_size, dim, true, alloc);
    mha_out_dyn = tensor::Tensor(activation_dtype, block_size, dim, true, alloc);
    wo_out_dyn = tensor::Tensor(activation_dtype, block_size, dim, true, alloc);
    ffn_norm_out_dyn = tensor::Tensor(activation_dtype, block_size, dim, true, alloc);
    w1_out_dyn = tensor::Tensor(activation_dtype, block_size, config_->hidden_dim_, true, alloc);
    w3_out_dyn = tensor::Tensor(activation_dtype, block_size, config_->hidden_dim_, true, alloc);
    w2_out_dyn = tensor::Tensor(activation_dtype, block_size, dim, true, alloc);
    rope_dummy_k_dyn = tensor::Tensor(activation_dtype, block_size, kv_dim, true, alloc);
  }
  tensor::Tensor& hidden_buf = use_prealloc ? draft_hidden_buf_ : hidden_buf_dyn;
  tensor::Tensor& rms_out = use_prealloc ? draft_rms_out_ : rms_out_dyn;
  tensor::Tensor& query_out = use_prealloc ? draft_query_out_ : query_out_dyn;
  tensor::Tensor& mha_out = use_prealloc ? draft_mha_out_ : mha_out_dyn;
  tensor::Tensor& wo_out = use_prealloc ? draft_wo_out_ : wo_out_dyn;
  tensor::Tensor& ffn_norm_out = use_prealloc ? draft_ffn_norm_out_ : ffn_norm_out_dyn;
  tensor::Tensor& w1_out = use_prealloc ? draft_w1_out_ : w1_out_dyn;
  tensor::Tensor& w3_out = use_prealloc ? draft_w3_out_ : w3_out_dyn;
  tensor::Tensor& w2_out = use_prealloc ? draft_w2_out_ : w2_out_dyn;
  tensor::Tensor& rope_dummy_k = use_prealloc ? draft_rope_dummy_k_ : rope_dummy_k_dyn;

  // For layer 0: hidden = noise_embedding (view, no copy)
  tensor::Tensor hidden(activation_dtype, block_size, dim, false, nullptr,
                        const_cast<void*>(noise_embedding.get_buffer()->ptr()));
  hidden.set_device_type(base::DeviceType::kDeviceCUDA);

  // rope_dummy_q depends on total_kv_len which varies, so always allocate
  tensor::Tensor rope_dummy_q(activation_dtype, total_kv_len, dim, true, alloc);

  for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
    // 1. Input LayerNorm
    const auto& attn_norm = layers->rmsnorm_layers_.at(layer_idx);
    STATUS_CHECK(attn_norm->forward(hidden, rms_out));

    // 2. Q projection
    const auto& wq = layers->wq_layers_.at(layer_idx);
    auto wq_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(wq);
    STATUS_CHECK(layers->batched_matmul_helper_layer_->forward(
        rms_out, wq_matmul->get_weight(0), query_out, block_size, 1.f));

    // Q norm
    auto q_norm = layers->rmsnorm_layers_.at(layer_idx + 2 * config_->layer_num_ + 1);
    tensor::Tensor q_reshaped(activation_dtype, block_size * config_->head_num_,
                              config_->head_size_, false, nullptr,
                              const_cast<void*>(query_out.get_buffer()->ptr()));
    q_reshaped.set_device_type(base::DeviceType::kDeviceCUDA);
    STATUS_CHECK(q_norm->forward(q_reshaped, q_reshaped));

    // === Zero-copy K/V: project directly into draft KV cache ===
    // KV cache layout: [layer_num, max_seq_len, kv_dim]
    // For this layer, K/V occupy [layer_idx * max_seq_len * kv_dim, ...]
    // We put k_ctx at offset 0 and k_noise at offset context_len within the layer's region.
    int32_t kv_cache_offset = layer_idx * config_->seq_len_ * kv_dim;
    void* key_layer_base = static_cast<char*>(
        const_cast<void*>(draft_key_cache_.get_buffer()->ptr())) +
        kv_cache_offset * elem_size;
    void* val_layer_base = static_cast<char*>(
        const_cast<void*>(draft_value_cache_.get_buffer()->ptr())) +
        kv_cache_offset * elem_size;

    // k_ctx view: points to cache[layer][0..context_len-1]
    tensor::Tensor k_ctx(activation_dtype, context_len, kv_dim, false, nullptr, key_layer_base);
    k_ctx.set_device_type(base::DeviceType::kDeviceCUDA);
    // k_noise view: points to cache[layer][context_len..context_len+block_size-1]
    void* k_noise_ptr = static_cast<char*>(key_layer_base) + (size_t)context_len * kv_dim * elem_size;
    tensor::Tensor k_noise(activation_dtype, block_size, kv_dim, false, nullptr, k_noise_ptr);
    k_noise.set_device_type(base::DeviceType::kDeviceCUDA);

    // v_ctx view: points to cache[layer][0..context_len-1]
    tensor::Tensor v_ctx(activation_dtype, context_len, kv_dim, false, nullptr, val_layer_base);
    v_ctx.set_device_type(base::DeviceType::kDeviceCUDA);
    // v_noise view: points to cache[layer][context_len..context_len+block_size-1]
    void* v_noise_ptr = static_cast<char*>(val_layer_base) + (size_t)context_len * kv_dim * elem_size;
    tensor::Tensor v_noise(activation_dtype, block_size, kv_dim, false, nullptr, v_noise_ptr);
    v_noise.set_device_type(base::DeviceType::kDeviceCUDA);

    // 3. K/V projection from context → writes directly to KV cache
    const auto& wk = layers->wk_layers_.at(layer_idx);
    const auto& wv = layers->wv_layers_.at(layer_idx);
    auto wk_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(wk);
    auto wv_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(wv);

    STATUS_CHECK(layers->batched_matmul_helper_layer_->forward(
        target_hidden, wk_matmul->get_weight(0), k_ctx, context_len, 1.f));
    STATUS_CHECK(layers->batched_matmul_helper_layer_->forward(
        target_hidden, wv_matmul->get_weight(0), v_ctx, context_len, 1.f));

    // 4. K/V projection from noise → writes directly to KV cache (contiguous after k_ctx)
    STATUS_CHECK(layers->batched_matmul_helper_layer_->forward(
        rms_out, wk_matmul->get_weight(0), k_noise, block_size, 1.f));
    STATUS_CHECK(layers->batched_matmul_helper_layer_->forward(
        rms_out, wv_matmul->get_weight(0), v_noise, block_size, 1.f));

    // K/V data is now contiguous in the KV cache: [k_ctx | k_noise] at cache[layer][0..total_kv_len-1]
    // No concat copy needed!

    // 5. K norm (on the contiguous K region in cache)
    auto k_norm = layers->rmsnorm_layers_.at(layer_idx + 3 * config_->layer_num_ + 1);
    tensor::Tensor k_reshaped(activation_dtype, total_kv_len * config_->kv_head_num_,
                              config_->head_size_, false, nullptr, key_layer_base);
    k_reshaped.set_device_type(base::DeviceType::kDeviceCUDA);
    STATUS_CHECK(k_norm->forward(k_reshaped, k_reshaped));

    // 6. RoPE on Q and K
    auto batched_rope = layers->batched_rope_layer_;

    // RoPE on query: positions [start_pos, start_pos + block_size)
    {
      batched_rope->set_seq_len(block_size);
      batched_rope->set_start_pos(start_pos);
      batched_rope->set_input(0, query_out);
      batched_rope->set_input(1, rope_dummy_k);
      batched_rope->set_input(2, get_buffer(ModelBufferType::kSinCache));
      batched_rope->set_input(3, get_buffer(ModelBufferType::kCosCache));
      batched_rope->set_cuda_config(cuda_config_);
      STATUS_CHECK(batched_rope->forward());
    }

    // RoPE on K (in-place in KV cache): positions [start_pos - context_len, start_pos + block_size)
    {
      // Create a view tensor over the K region in cache for RoPE
      tensor::Tensor k_cache_view(activation_dtype, total_kv_len, kv_dim, false, nullptr, key_layer_base);
      k_cache_view.set_device_type(base::DeviceType::kDeviceCUDA);
      int32_t k_start_pos = start_pos - context_len;
      if (k_start_pos < 0) k_start_pos = 0;
      batched_rope->set_seq_len(total_kv_len);
      batched_rope->set_start_pos(k_start_pos);
      batched_rope->set_input(0, rope_dummy_q);
      batched_rope->set_input(1, k_cache_view);
      batched_rope->set_input(2, get_buffer(ModelBufferType::kSinCache));
      batched_rope->set_input(3, get_buffer(ModelBufferType::kCosCache));
      batched_rope->set_cuda_config(cuda_config_);
      STATUS_CHECK(batched_rope->forward());
    }

    // 7. Non-causal attention reads directly from KV cache (data already there)
    auto prefill_layer = layers->flash_attention_prefill_layer_;
    prefill_layer->set_cur_seq_len(block_size);
    prefill_layer->set_layer_index(layer_idx);
    prefill_layer->set_use_fp16(activation_dtype == base::DataType::kDataTypeFp16);
    prefill_layer->set_is_causal(false);
    prefill_layer->set_dims(config_->head_num_, config_->kv_head_num_,
                            config_->head_size_, config_->seq_len_);
    prefill_layer->set_input(0, query_out);
    prefill_layer->set_input(1, mha_out);
    prefill_layer->set_input(2, draft_key_cache_);
    prefill_layer->set_input(3, draft_value_cache_);
    prefill_layer->set_cuda_config(cuda_config_);
    prefill_layer->set_start_pos(context_len);
    STATUS_CHECK(prefill_layer->forward());

    // 8. O projection
    const auto& wo = layers->wo_layers_.at(layer_idx);
    batched_matmul_forward(wo, mha_out, wo_out, block_size);

    // 9. Residual add: after layer 0, switch hidden to owned buffer
    if (layer_idx == 0) {
      STATUS_CHECK(layers->batched_add_layer_->forward(hidden, wo_out, hidden_buf));
      hidden = hidden_buf;
    } else {
      STATUS_CHECK(layers->batched_add_layer_->forward(hidden, wo_out, hidden));
    }

    // 10. Feed forward (with residual, writes back to hidden in-place)
    batched_feed_forward_optimized(layer_idx, hidden, ffn_norm_out,
                                   w1_out, w3_out, w2_out, block_size);
  }

  // Final norm: write directly to draft_output_ (zero-copy output)
  const auto& final_norm = layers->rmsnorm_layers_.at(2 * config_->layer_num_);
  STATUS_CHECK(final_norm->forward(hidden, draft_output_));

  return base::error::Success();
}

}  // namespace model

#endif  // QWEN3_SUPPORT
