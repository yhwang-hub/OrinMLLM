/**
 * @file qwen3_vl.cpp
 * @brief Qwen3-VL Vision-Language Model Implementation
 * 
 * This file implements the Qwen3-VL-8B multimodal model for KuiperLLama.
 * 
 * Model Architecture:
 * ==================
 * 1. Vision Encoder (ViT):
 *    - Patch Embedding: Conv3D [1152, 3, 2, 16, 16] -> [num_patches, 1152]
 *    - Position Embedding: Learnable [2304, 1152] with bilinear interpolation
 *    - 27 Transformer Blocks:
 *      - LayerNorm(1152) + Self-Attention (16 heads, head_dim=72)
 *      - LayerNorm(1152) + MLP (1152 -> 4304 -> 1152, GELU)
 *    - Merger: Projects 4 patches to 1 token [4*1152 -> 4*1152 -> 4096]
 *    - Deepstack: 3 additional mergers from layers [8, 16, 24]
 * 
 * 2. Language Model (Qwen3):
 *    - Token Embedding: [151936, 4096]
 *    - 36 Transformer Layers:
 *      - RMSNorm + Self-Attention (32 heads, 8 KV heads, head_dim=128)
 *      - q_norm, k_norm: RMSNorm on Q, K projections
 *      - RMSNorm + MLP (4096 -> 12288 -> 4096, SwiGLU)
 *    - Final RMSNorm + LM Head [4096 -> 151936]
 * 
 * 3. Multimodal Fusion:
 *    - Image tokens replace <image_pad> (token 151655) in input
 *    - Deepstack features provide multi-scale visual information
 *    - M-RoPE handles 3D position encoding (temporal, height, width)
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

// Include STB for image loading
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb/stb_image.h"
#include "stb/stb_image_resize2.h"

// CPU-side float32 to float16 conversion
// Uses a union-based bit manipulation approach
namespace {
inline uint16_t float_to_half(float value) {
  // Handle special cases
  if (value == 0.0f) return 0;
  if (std::isnan(value)) return 0x7e00;  // NaN
  if (std::isinf(value)) return (value > 0) ? 0x7c00 : 0xfc00;  // +/- Inf
  
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  
  uint32_t sign = (bits >> 31) & 0x1;
  int32_t exp = ((bits >> 23) & 0xff) - 127;
  uint32_t frac = bits & 0x7fffff;
  
  uint16_t h_sign = sign << 15;
  uint16_t h_exp;
  uint16_t h_frac;
  
  if (exp < -24) {
    // Underflow to zero
    return h_sign;
  } else if (exp < -14) {
    // Denormalized number
    h_exp = 0;
    h_frac = (frac | 0x800000) >> (14 - exp);
  } else if (exp > 15) {
    // Overflow to infinity
    return h_sign | 0x7c00;
  } else {
    h_exp = (exp + 15) << 10;
    h_frac = frac >> 13;
  }
  
  return h_sign | h_exp | h_frac;
}

inline float half_to_float(uint16_t h) {
  uint32_t sign = (h >> 15) & 0x1;
  uint32_t exp = (h >> 10) & 0x1f;
  uint32_t frac = h & 0x3ff;
  
  uint32_t bits;
  if (exp == 0) {
    if (frac == 0) {
      bits = sign << 31;
    } else {
      // Denormalized
      exp = 1;
      while ((frac & 0x400) == 0) {
        frac <<= 1;
        exp--;
      }
      frac &= 0x3ff;
      bits = (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13);
    }
  } else if (exp == 31) {
    bits = (sign << 31) | 0x7f800000 | (frac << 13);
  } else {
    bits = (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13);
  }
  
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}
}  // namespace

#include "op/matmul.h"
#include "op/mha.h"
#include "op/rmsnorm.h"
#include "op/batched_add.h"
#include "op/vision_layers.h"
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
#include "../op/kernels/cuda/vision_encoder_kernel.cuh"
#include "../op/kernels/cuda/rope_kernel.cuh"
#include "../op/kernels/cuda/fused_kernels.cuh"
#include "sampler/argmax_sampler.h"
#include "base/tick.h"

namespace model {

// ============================================================================
// Image Preprocessing Utilities
// ============================================================================

namespace image_utils {

std::vector<uint8_t> load_image(const std::string& path, 
                                 int& width, int& height, int& channels) {
  unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, 3);
  if (!data) {
    LOG(ERROR) << "Failed to load image: " << path;
    return {};
  }
  
  // Force RGB
  channels = 3;
  std::vector<uint8_t> pixels(data, data + width * height * channels);
  stbi_image_free(data);
  
  LOG(INFO) << "Loaded image: " << path << " (" << width << "x" << height << "x" << channels << ")";
  return pixels;
}

std::tuple<std::vector<uint8_t>, int, int> smart_resize(
    const std::vector<uint8_t>& pixels,
    int src_width, int src_height,
    int min_pixels, int max_pixels, int factor) {
  
  // HuggingFace-compatible smart_resize implementation
  // Step 1: Round to nearest factor first
  int h_bar = static_cast<int>(std::round(static_cast<float>(src_height) / factor)) * factor;
  int w_bar = static_cast<int>(std::round(static_cast<float>(src_width) / factor)) * factor;
  
  // Ensure minimum size
  h_bar = std::max(h_bar, factor);
  w_bar = std::max(w_bar, factor);
  
  // Step 2: Check if total pixels exceeds max_pixels
  if (h_bar * w_bar > max_pixels) {
    float beta = std::sqrt(static_cast<float>(src_height * src_width) / max_pixels);
    h_bar = std::max(factor, static_cast<int>(std::floor(src_height / beta / factor)) * factor);
    w_bar = std::max(factor, static_cast<int>(std::floor(src_width / beta / factor)) * factor);
  }
  // Step 3: Check if total pixels is below min_pixels
  else if (h_bar * w_bar < min_pixels) {
    float beta = std::sqrt(static_cast<float>(min_pixels) / (src_height * src_width));
    h_bar = static_cast<int>(std::ceil(src_height * beta / factor)) * factor;
    w_bar = static_cast<int>(std::ceil(src_width * beta / factor)) * factor;
  }
  
  LOG(INFO) << "Smart resize: " << src_width << "x" << src_height 
            << " -> " << w_bar << "x" << h_bar;
  
  // Resize using stb_image_resize
  std::vector<uint8_t> resized(w_bar * h_bar * 3);
  stbir_resize_uint8_linear(
      pixels.data(), src_width, src_height, src_width * 3,
      resized.data(), w_bar, h_bar, w_bar * 3,
      STBIR_RGB);
  
  return {resized, w_bar, h_bar};
}

tensor::Tensor normalize_to_tensor(const std::vector<uint8_t>& pixels,
                                    int width, int height) {
  // Qwen3-VL normalization constants (simple 0.5/0.5)
  const float mean[3] = {0.5f, 0.5f, 0.5f};
  const float std[3] = {0.5f, 0.5f, 0.5f};
  
  // Create FP16 tensor [3, height, width]
  std::shared_ptr<base::DeviceAllocator> alloc = base::CUDADeviceAllocatorFactory::get_instance();
  tensor::Tensor tensor(base::DataType::kDataTypeFp16, 3, height, width, true, alloc);
  
  // First create FP32 data on CPU
  std::vector<float> normalized(3 * height * width);
  
  for (int c = 0; c < 3; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        int src_idx = (h * width + w) * 3 + c;  // HWC format
        int dst_idx = c * height * width + h * width + w;  // CHW format
        
        float pixel = static_cast<float>(pixels[src_idx]) / 255.0f;
        normalized[dst_idx] = (pixel - mean[c]) / std[c];
      }
    }
  }
  
  // Convert to FP16 on CPU then copy to GPU
  std::vector<uint16_t> fp16_data(normalized.size());
  for (size_t i = 0; i < normalized.size(); ++i) {
    // Use CPU-side float to half conversion
    fp16_data[i] = float_to_half(normalized[i]);
  }
  
  // Copy to GPU
  cudaMemcpy(tensor.ptr<void>(), fp16_data.data(), 
             fp16_data.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
  
  return tensor;
}

tensor::Tensor image_to_patches(const tensor::Tensor& image_tensor,
                                 int patch_size, int temporal_patch_size,
                                 cudaStream_t stream = nullptr) {
  // Image tensor: [3, H, W]
  // Output: [num_patches, patch_dim]
  // where patch_dim = 3 * temporal_patch_size * patch_size * patch_size
  // 
  // IMPORTANT: The output patches are in 2x2 block interleaved order, matching HuggingFace!
  // For spatial_merge_size=2, the patch order is:
  //   block (0,0): patch 0=(0,0), patch 1=(0,1), patch 2=(1,0), patch 3=(1,1)
  //   block (0,1): patch 4=(0,2), patch 5=(0,3), patch 6=(1,2), patch 7=(1,3)
  //   ...
  
  int channels = image_tensor.get_dim(0);
  int height = image_tensor.get_dim(1);
  int width = image_tensor.get_dim(2);
  
  int grid_h = height / patch_size;
  int grid_w = width / patch_size;
  int num_patches = grid_h * grid_w;
  
  // For single image, temporal_patch_size is padded (repeat frame)
  int patch_dim = channels * temporal_patch_size * patch_size * patch_size;
  
  std::shared_ptr<base::DeviceAllocator> alloc = base::CUDADeviceAllocatorFactory::get_instance();
  tensor::Tensor patches(base::DataType::kDataTypeFp16, num_patches, patch_dim, true, alloc);
  
  // OPTIMIZED: Use GPU kernel to extract patches directly on GPU
  // This eliminates the D2H copy, CPU processing, and H2D copy
  kernel::extract_patches_cu(
      image_tensor,
      patches,
      channels,
      height,
      width,
      patch_size,
      temporal_patch_size,
      stream
  );
  
  LOG(INFO) << "Extracted patches (GPU): [" << num_patches << ", " << patch_dim << "] from ["
            << channels << ", " << height << ", " << width << "] (2x2 block interleaved order)";
  
  return patches;
}

} // namespace image_utils

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
// Qwen3VLModel Implementation
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
  // Clean up M-RoPE GPU arrays (now using single contiguous allocation)
  if (mrope_pos_gpu_) {
    cudaFree(mrope_pos_gpu_);
    mrope_pos_gpu_ = nullptr;
    mrope_pos_t_gpu_ = nullptr;  // These point into mrope_pos_gpu_
    mrope_pos_h_gpu_ = nullptr;
    mrope_pos_w_gpu_ = nullptr;
  }
  mrope_pos_gpu_capacity_ = 0;
  
  // Clean up pinned host memory for M-RoPE
  if (mrope_pos_pinned_) {
    cudaFreeHost(mrope_pos_pinned_);
    mrope_pos_pinned_ = nullptr;
  }
  mrope_pos_pinned_capacity_ = 0;
  
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
    
    // Helper lambda to set keep_fp16_weights for LayerParam layers
    auto set_fp16_flag = [](const std::shared_ptr<op::Layer>& layer) {
      if (auto layer_param = std::dynamic_pointer_cast<op::LayerParam>(layer)) {
        layer_param->set_keep_fp16_weights(true);
      }
    };
    
    // Set CUDA config for all layers and move weights to GPU
    LOG(INFO) << "  Moving " << qwen_layers_->rmsnorm_layers_.size() << " RMSNorm layers...";
    for (auto& layer : qwen_layers_->rmsnorm_layers_) {
      set_fp16_flag(layer);
      layer->set_cuda_config(cuda_config_);
      layer->to_cuda();
    }
    
    LOG(INFO) << "  Moving embedding layer...";
    if (qwen_layers_->embedding_layer_) {
      set_fp16_flag(qwen_layers_->embedding_layer_);
      qwen_layers_->embedding_layer_->set_cuda_config(cuda_config_);
      qwen_layers_->embedding_layer_->to_cuda();
    }
    
    LOG(INFO) << "  Moving Q projections...";
    for (auto& layer : qwen_layers_->wq_layers_) {
      set_fp16_flag(layer);
      layer->set_cuda_config(cuda_config_);
      layer->to_cuda();
    }
    
    LOG(INFO) << "  Moving K projections...";
    for (auto& layer : qwen_layers_->wk_layers_) {
      set_fp16_flag(layer);
      layer->set_cuda_config(cuda_config_);
      layer->to_cuda();
    }
    
    LOG(INFO) << "  Moving V projections...";
    for (auto& layer : qwen_layers_->wv_layers_) {
      set_fp16_flag(layer);
      layer->set_cuda_config(cuda_config_);
      layer->to_cuda();
    }
    
    LOG(INFO) << "  Moving O projections...";
    for (auto& layer : qwen_layers_->wo_layers_) {
      set_fp16_flag(layer);
      layer->set_cuda_config(cuda_config_);
      layer->to_cuda();
    }
    
    LOG(INFO) << "  Moving W1 projections...";
    for (auto& layer : qwen_layers_->w1_layers_) {
      set_fp16_flag(layer);
      layer->set_cuda_config(cuda_config_);
      layer->to_cuda();
    }
    
    LOG(INFO) << "  Moving W2 projections...";
    for (auto& layer : qwen_layers_->w2_layers_) {
      set_fp16_flag(layer);
      layer->set_cuda_config(cuda_config_);
      layer->to_cuda();
    }
    
    LOG(INFO) << "  Moving W3 projections...";
    for (auto& layer : qwen_layers_->w3_layers_) {
      set_fp16_flag(layer);
      layer->set_cuda_config(cuda_config_);
      layer->to_cuda();
    }
    
    LOG(INFO) << "  Moving LM head...";
    if (qwen_layers_->cls_layer_) {
      set_fp16_flag(qwen_layers_->cls_layer_);
      qwen_layers_->cls_layer_->set_cuda_config(cuda_config_);
      qwen_layers_->cls_layer_->to_cuda();
    }
    
    cudaStreamSynchronize(cuda_config_->stream);
    LOG(INFO) << "Moved all LLM layers to CUDA";
  }
  
  // Create non-parameter layers (RoPE, MHA, Add, SwiGLU)
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

base::Status Qwen3VLModel::load_vl_model_file() {
  // Use mmap to load model file
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
  config_->seq_len_ = 8192;  // Working context length
  config_->head_size_ = vl_config_.text.head_dim;
  config_->kv_dim_ = vl_config_.text.num_key_value_heads * vl_config_.text.head_dim;
  config_->kv_mul_ = vl_config_.text.num_attention_heads / vl_config_.text.num_key_value_heads;
  
  // =========================================
  // Load Vision Encoder Weights (directly to GPU)
  // =========================================
  
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
  
  // Main merger - note: main merger uses vit_hidden for norm, deepstack uses merged_hidden
  auto load_main_merger = [&](Qwen3VLVisionLayers::Merger& m) {
    read_fp16_tensor_to_gpu(m.norm_weight, {vit_hidden}, alloc_gpu);  // [1152]
    read_fp16_tensor_to_gpu(m.norm_bias, {vit_hidden}, alloc_gpu);    // [1152]
    read_fp16_tensor_to_gpu(m.fc1_weight, {merged_hidden, merged_hidden}, alloc_gpu);
    read_fp16_tensor_to_gpu(m.fc1_bias, {merged_hidden}, alloc_gpu);
    read_fp16_tensor_to_gpu(m.fc2_weight, {out_hidden, merged_hidden}, alloc_gpu);
    read_fp16_tensor_to_gpu(m.fc2_bias, {out_hidden}, alloc_gpu);
  };
  
  // Deepstack merger - uses merged_hidden for norm (after spatial merging)
  auto load_deepstack_merger = [&](Qwen3VLVisionLayers::Merger& m) {
    read_fp16_tensor_to_gpu(m.norm_weight, {merged_hidden}, alloc_gpu);  // [4608]
    read_fp16_tensor_to_gpu(m.norm_bias, {merged_hidden}, alloc_gpu);    // [4608]
    read_fp16_tensor_to_gpu(m.fc1_weight, {merged_hidden, merged_hidden}, alloc_gpu);
    read_fp16_tensor_to_gpu(m.fc1_bias, {merged_hidden}, alloc_gpu);
    read_fp16_tensor_to_gpu(m.fc2_weight, {out_hidden, merged_hidden}, alloc_gpu);
    read_fp16_tensor_to_gpu(m.fc2_bias, {out_hidden}, alloc_gpu);
  };
  
  load_main_merger(vision_layers_->merger);
  LOG(INFO) << "  Loaded main merger";
  
  // Deepstack mergers
  vision_layers_->deepstack_mergers.resize(vl_config_.vision.deepstack_visual_indexes.size());
  for (size_t i = 0; i < vision_layers_->deepstack_mergers.size(); ++i) {
    load_deepstack_merger(vision_layers_->deepstack_mergers[i]);
  }
  LOG(INFO) << "  Loaded " << vision_layers_->deepstack_mergers.size() << " deepstack mergers";
  
  // =========================================
  // Load Language Model Weights (using mmap pointers)
  // =========================================
  
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
  
  // RMSNorm weights (use mmap pointer directly)
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
  
  // FFN weights (gate, down, up)
  load_proj_weights(qwen_layers_->w1_layers_, llm_intermediate, llm_dim);  // gate
  LOG(INFO) << "  Loaded gate projections";
  load_proj_weights(qwen_layers_->w2_layers_, llm_dim, llm_intermediate);  // down
  LOG(INFO) << "  Loaded down projections";
  load_proj_weights(qwen_layers_->w3_layers_, llm_intermediate, llm_dim);  // up
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
  
  // Qwen3 specific: q_norm and k_norm
  // These are stored after LM head: q_norm for all layers, then k_norm for all layers
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

void Qwen3VLModel::init_mem() {
  // Initialize base model memory
  std::shared_ptr<base::DeviceAllocator> alloc;
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    alloc = base::CPUDeviceAllocatorFactory::get_instance();
  } else {
    alloc = base::CUDADeviceAllocatorFactory::get_instance();
  }
  
  std::shared_ptr<base::DeviceAllocator> alloc_cpu =
      base::CPUDeviceAllocatorFactory::get_instance();
  
  // Use FP16 for activations
  base::DataType activation_dtype = base::DataType::kDataTypeFp16;
  LOG(INFO) << "Using FP16 activation buffers for Qwen3-VL";
  
  int32_t model_dim = config_->dim_;
  int32_t intermediate_dim = config_->hidden_dim_;
  
  // Input token and embedding buffers
  tensor::Tensor input_tokens(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  tensor::Tensor input_embeddings(activation_dtype, 1, model_dim, true, alloc);
  
  CHECK(insert_buffer(ModelBufferType::kInputTokens, input_tokens));
  CHECK(insert_buffer(ModelBufferType::kInputEmbeddings, input_embeddings));
  LOG(INFO) << "Allocated input buffers and embeddings buffers.";
  
  // RoPE sin/cos cache
  tensor::Tensor sin_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_,
                           true, alloc);
  tensor::Tensor cos_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_,
                           true, alloc);
  CHECK(insert_buffer(ModelBufferType::kSinCache, sin_cache));
  CHECK(insert_buffer(ModelBufferType::kCosCache, cos_cache));
  LOG(INFO) << "Allocated RoPE sin/cos cache buffers.";
  
  // Intermediate buffers
  tensor::Tensor rms_output(activation_dtype, model_dim, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kOutputRMSNorm, rms_output));
  CHECK(insert_buffer(ModelBufferType::kOutputMHA, rms_output));
  CHECK(insert_buffer(ModelBufferType::kW2Output, rms_output));
  CHECK(insert_buffer(ModelBufferType::kFFNRMSNorm, rms_output));
  LOG(INFO) << "Allocated intermediate layer output buffers.";
  
  tensor::Tensor w1_output(activation_dtype, intermediate_dim, true, alloc);
  tensor::Tensor w3_output(activation_dtype, intermediate_dim, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kW1Output, w1_output));
  CHECK(insert_buffer(ModelBufferType::kW3Output, w3_output));
  LOG(INFO) << "Allocated W1/W3 output buffers.";
  
  // KV cache
  tensor::Tensor key_cache(activation_dtype, config_->layer_num_, config_->seq_len_,
                           config_->kv_dim_, true, alloc);
  tensor::Tensor value_cache(activation_dtype, config_->layer_num_, config_->seq_len_,
                             config_->kv_dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kKeyCache, key_cache));
  CHECK(insert_buffer(ModelBufferType::kValueCache, value_cache));
  LOG(INFO) << "Allocated KV cache buffers.";
  
  // Query output
  tensor::Tensor query(activation_dtype, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kQuery, query));
  
  // Decode input buffer
  tensor::Tensor decode_input(activation_dtype, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kDecodeInput, decode_input));
  
  // Position tensor (CPU for normal path)
  tensor::Tensor pos_tensor(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  CHECK(insert_buffer(ModelBufferType::kInputPos, pos_tensor));
  
  // Position tensor on GPU for CUDA Graph path
  tensor::Tensor pos_tensor_gpu(base::DataType::kDataTypeInt32, 1, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kInputPosGPU, pos_tensor_gpu));
  LOG(INFO) << "Allocated input position buffer on GPU (for M-RoPE text_pos).";
  
  // KV cache position on GPU (different from M-RoPE position for VL models)
  tensor::Tensor kv_cache_pos_gpu(base::DataType::kDataTypeInt32, 1, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kKVCachePosGPU, kv_cache_pos_gpu));
  LOG(INFO) << "Allocated KV cache position buffer on GPU.";
  
  // Temporary K/V buffers with fixed addresses for CUDA Graph optimization
  tensor::Tensor temp_key(activation_dtype, config_->kv_dim_, true, alloc);
  tensor::Tensor temp_value(activation_dtype, config_->kv_dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kTempKey, temp_key));
  CHECK(insert_buffer(ModelBufferType::kTempValue, temp_value));
  LOG(INFO) << "Allocated temporary K/V buffers for CUDA Graph.";
  
  // Pinned memory buffers for efficient async Host-Device transfers
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    std::shared_ptr<base::DeviceAllocator> alloc_pinned = 
        base::CPUPinnedAllocatorFactory::get_instance();
    
    // Pinned pos buffer for async H2D transfer (for M-RoPE text_pos)
    tensor::Tensor pos_pinned(base::DataType::kDataTypeInt32, 1, true, alloc_pinned);
    CHECK(insert_buffer(ModelBufferType::kInputPosPinned, pos_pinned));
    LOG(INFO) << "Allocated pinned input position buffer.";
    
    // Pinned KV cache pos buffer for async H2D transfer
    tensor::Tensor kv_cache_pos_pinned(base::DataType::kDataTypeInt32, 1, true, alloc_pinned);
    CHECK(insert_buffer(ModelBufferType::kKVCachePosPinned, kv_cache_pos_pinned));
    LOG(INFO) << "Allocated pinned KV cache position buffer.";

    // Pre-allocated argmax output buffer on GPU
    tensor::Tensor argmax_output(base::DataType::kDataTypeInt32, 2, true, alloc);
    CHECK(insert_buffer(ModelBufferType::kArgmaxOutput, argmax_output));
    LOG(INFO) << "Allocated argmax output buffer on GPU.";
    
    // Pinned argmax result buffer for async D2H transfer
    tensor::Tensor argmax_pinned(base::DataType::kDataTypeInt32, 2, true, alloc_pinned);
    CHECK(insert_buffer(ModelBufferType::kArgmaxOutputPinned, argmax_pinned));
    LOG(INFO) << "Allocated pinned argmax output buffer.";
  }
  
  // Attention scores (FP32 for numerical stability)
  tensor::Tensor attn(base::DataType::kDataTypeFp32, config_->head_num_, config_->seq_len_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kScoreStorage, attn));
  
  // Attention output
  tensor::Tensor attn_output(activation_dtype, model_dim, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kAttnOutput, attn_output));
  
  // Forward output - use vocab_size from vl_config_ for consistency with cls_layer_
  int vocab_size = vl_config_.text.vocab_size;
  tensor::Tensor forward_output(base::DataType::kDataTypeFp32, vocab_size, true, alloc);
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    tensor::Tensor forward_output_cpu(base::DataType::kDataTypeFp32, vocab_size, true, alloc_cpu);
    CHECK(insert_buffer(ModelBufferType::kForwardOutputCPU, forward_output_cpu));
  }
  CHECK(insert_buffer(ModelBufferType::kForwardOutput, forward_output));
  LOG(INFO) << "Allocated forward output buffers.";
  
  LOG(INFO) << "Memory initialization complete for Qwen3-VL.";
}

base::Status Qwen3VLModel::create_layers() {
  // Already created during load_vl_model_file
  return base::error::Success();
}

void Qwen3VLModel::create_param_layers() {
  // Already created during load_vl_model_file
}

void Qwen3VLModel::create_nonparam_layers() {
  // Create non-parameter layers
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

  // Create new layers for unified kernel calls
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
  
  // Create batched add/swiglu layers for prefill (support any-dim tensors)
  qwen_layers_->batched_add_layer_ = std::make_shared<op::BatchedAddLayer>(device_type_);
  qwen_layers_->batched_add_layer_->set_cuda_config(cuda_config_);
  
  qwen_layers_->batched_swiglu_layer_ = std::make_shared<op::BatchedSwiGLULayer>(device_type_);
  qwen_layers_->batched_swiglu_layer_->set_cuda_config(cuda_config_);
  
  qwen_layers_->sin_cos_cache_layer_ = std::make_shared<op::SinCosCacheLayer>(device_type_);
  qwen_layers_->sin_cos_cache_layer_->set_cuda_config(cuda_config_);
  
  // VL model specific layers for M-RoPE and vision
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
  
  // Vision-specific layers (stored in vision_vl_layers_)
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
}

void Qwen3VLModel::create_param_quant_layers() {
  // Not used for FP16 model
}

// ============================================================================
// LLM Forward Helper Functions
// ============================================================================

void Qwen3VLModel::attention_rms(int32_t layer_idx, const tensor::Tensor& input) const {
  CHECK(qwen_layers_ != nullptr);
  tensor::Tensor rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  std::shared_ptr<op::Layer> rmsnorm_layer = qwen_layers_->rmsnorm_layers_.at(layer_idx);
  CHECK_NE(rmsnorm_layer, nullptr) << "The attention rmsnorm layer is null";
  STATUS_CHECK(rmsnorm_layer->forward(input, rmsnorm_output));
}

void Qwen3VLModel::attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  CHECK(qwen_layers_ != nullptr);
  
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
  int32_t pos = pos_tensor.index<int32_t>(0);
  auto [key, val] = slice_kv_cache(layer_idx, pos);

  auto rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);

  // Query
  const auto& query_layer = qwen_layers_->wq_layers_.at(layer_idx);
  CHECK_NE(query_layer, nullptr) << "The query layer is null";
  STATUS_CHECK(query_layer->forward(rmsnorm_output, query));

  // Query norm (Qwen3 specific)
  auto query_norm = qwen_layers_->rmsnorm_layers_.at(layer_idx + 2 * config_->layer_num_ + 1);
  query.reshape({(int32_t)query.size() / config_->head_size_, config_->head_size_});
  query_norm->forward(query, query);
  query.reshape({(int32_t)query.size()});

  // Key
  const auto& key_layer = qwen_layers_->wk_layers_.at(layer_idx);
  CHECK_NE(key_layer, nullptr) << "The key layer is null";
  STATUS_CHECK(key_layer->forward(rmsnorm_output, key));

  // Key norm (Qwen3 specific)
  auto key_norm = qwen_layers_->rmsnorm_layers_.at(layer_idx + 3 * config_->layer_num_ + 1);
  key.reshape({(int32_t)key.size() / config_->head_size_, config_->head_size_});
  key_norm->forward(key, key);
  key.reshape({(int32_t)key.size()});

  // Value
  const auto& value_layer = qwen_layers_->wv_layers_.at(layer_idx);
  CHECK_NE(value_layer, nullptr) << "The value layer is null";
  STATUS_CHECK(value_layer->forward(rmsnorm_output, val));

  // M-RoPE: Use 3D position encoding for multimodal inputs
  // Check if we have M-RoPE positions computed (during prefill)
  if (!mrope_pos_t_.empty() && pos < static_cast<int32_t>(mrope_pos_t_.size())) {
    // Prefill phase with M-RoPE positions
    int32_t pos_t = mrope_pos_t_[pos];
    int32_t pos_h = mrope_pos_h_[pos];
    int32_t pos_w = mrope_pos_w_[pos];
    
    // Get mrope_section from config
    const auto& section = vl_config_.text.mrope_section;
    int32_t section0 = section[0];  // 24 pairs for temporal
    int32_t section1 = section[1];  // 20 pairs for height
    int32_t section2 = section[2];  // 20 pairs for width
    
    qwen_layers_->mrope_layer_->forward(
        pos_t, pos_h, pos_w,
        config_->dim_, config_->kv_dim_, config_->head_size_,
        section0, section1, section2,
        query, key,
        get_buffer(ModelBufferType::kSinCache),
        get_buffer(ModelBufferType::kCosCache));
  } else {
    // Decode phase: use sequential position for all dimensions
    // After prefill, new tokens use (pos, pos, pos) for t/h/w
    int32_t text_pos = mrope_max_text_pos_ + (pos - prefill_seq_len_) + 1;
    
    const auto& section = vl_config_.text.mrope_section;
    int32_t section0 = section[0];
    int32_t section1 = section[1];
    int32_t section2 = section[2];
    
    qwen_layers_->mrope_layer_->forward(
        text_pos, text_pos, text_pos,
        config_->dim_, config_->kv_dim_, config_->head_size_,
        section0, section1, section2,
        query, key,
        get_buffer(ModelBufferType::kSinCache),
        get_buffer(ModelBufferType::kCosCache));
  }
}

void Qwen3VLModel::attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  CHECK(qwen_layers_ != nullptr);
  
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);
  tensor::Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);
  tensor::Tensor query = get_buffer(ModelBufferType::kQuery);
  
  int pos = pos_tensor.index<int32_t>(0);

  // FP16 data always uses Flash Attention (MHA does not support FP16)
  if (query.data_type() == base::DataType::kDataTypeFp16 &&
      key_cache.data_type() == base::DataType::kDataTypeFp16) {
    // Use Flash Attention FP16 for decode (FA1 or FA2 based on layer's attention_type_)
    qwen_layers_->flash_attention_decode_layer_->forward(
        pos, config_->head_num_, config_->kv_head_num_,
        config_->head_size_, config_->kv_mul_, layer_idx,
        config_->seq_len_, config_->kv_dim_,
        query, mha_output, key_cache, val_cache);
  } else if (attention_type_ == base::AttentionType::kAttentionMHA) {
    tensor::Tensor score_storage = get_buffer(ModelBufferType::kScoreStorage);
    const auto& mha_layer = qwen_layers_->mha_layer_;
    CHECK_NE(mha_layer, nullptr) << "The MHA layer is null";
    std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_pos(pos);
    std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_layer_idx(layer_idx);
    STATUS_CHECK(mha_layer->forward(query, score_storage, key_cache, val_cache, mha_output));
  } else {
    // FP32 Flash Attention path (FA1 or FA2)
    qwen_layers_->flash_attention_decode_layer_->forward(
        pos, config_->head_num_, config_->kv_head_num_,
        config_->head_size_, config_->kv_mul_, layer_idx,
        config_->seq_len_, config_->kv_dim_,
        query, mha_output, key_cache, val_cache);
  }

  // WO @ attention output
  tensor::Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
  const auto& wo_layer = qwen_layers_->wo_layers_.at(layer_idx);
  CHECK_NE(wo_layer, nullptr) << "The WO layer is null";
  STATUS_CHECK(wo_layer->forward(mha_output, attn_output));
}

void Qwen3VLModel::attention_qkv_with_graph(int32_t layer_idx, 
                                             const tensor::Tensor& rope_pos_gpu,
                                             const tensor::Tensor& kv_cache_pos_gpu) const {
  CHECK(qwen_layers_ != nullptr);
  CHECK(cuda_config_ != nullptr);
  
  // Use fixed-address temporary buffers for CUDA Graph compatibility
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
  tensor::Tensor temp_key = this->get_buffer(ModelBufferType::kTempKey);
  tensor::Tensor temp_value = this->get_buffer(ModelBufferType::kTempValue);
  
  auto rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  
  // Query
  const auto& query_layer = qwen_layers_->wq_layers_.at(layer_idx);
  CHECK_NE(query_layer, nullptr);
  STATUS_CHECK(query_layer->forward(rmsnorm_output, query));

  // Query norm (Qwen3 specific)
  auto query_norm = qwen_layers_->rmsnorm_layers_.at(layer_idx + 2 * config_->layer_num_ + 1);
  query.reshape({(int32_t)query.size() / config_->head_size_, config_->head_size_});
  query_norm->forward(query, query);
  query.reshape({(int32_t)query.size()});

  // Key -> temp_key (fixed address)
  const auto& key_layer = qwen_layers_->wk_layers_.at(layer_idx);
  CHECK_NE(key_layer, nullptr);
  STATUS_CHECK(key_layer->forward(rmsnorm_output, temp_key));
  
  // Key norm (Qwen3 specific)
  auto key_norm = qwen_layers_->rmsnorm_layers_.at(layer_idx + 3 * config_->layer_num_ + 1);
  temp_key.reshape({(int32_t)temp_key.size() / config_->head_size_, config_->head_size_});
  key_norm->forward(temp_key, temp_key);
  temp_key.reshape({(int32_t)temp_key.size()});
  
  // Value -> temp_value (fixed address)
  const auto& value_layer = qwen_layers_->wv_layers_.at(layer_idx);
  CHECK_NE(value_layer, nullptr);
  STATUS_CHECK(value_layer->forward(rmsnorm_output, temp_value));

  // M-RoPE with GPU pos for CUDA Graph compatibility (decode phase uses same pos for t/h/w)
  const auto& section = vl_config_.text.mrope_section;
  qwen_layers_->mrope_gpu_pos_layer_->forward(
      rope_pos_gpu.ptr<int32_t>(),  // Use M-RoPE text position
      config_->dim_, config_->kv_dim_, config_->head_size_,
      section[0], section[1], section[2],
      query, temp_key,
      get_buffer(ModelBufferType::kSinCache),
      get_buffer(ModelBufferType::kCosCache));
  
  // Copy temp_key and temp_value to KV cache at correct position
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);
  
  qwen_layers_->copy_to_kv_cache_layer_->forward(
      key_cache, temp_key,
      kv_cache_pos_gpu.ptr<int32_t>(),  // Use KV cache position
      config_->kv_dim_,
      layer_idx,
      config_->seq_len_);
      
  qwen_layers_->copy_to_kv_cache_layer_->forward(
      val_cache, temp_value,
      kv_cache_pos_gpu.ptr<int32_t>(),  // Use KV cache position
      config_->kv_dim_,
      layer_idx,
      config_->seq_len_);
}

void Qwen3VLModel::attention_mha_with_graph(int32_t layer_idx, const tensor::Tensor& pos_tensor_gpu) const {
  CHECK(qwen_layers_ != nullptr);
  CHECK(cuda_config_ != nullptr);
  
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);
  tensor::Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
  
  // FP16 data always uses Flash Attention (MHA does not support FP16)
  if (query.data_type() == base::DataType::kDataTypeFp16 &&
      key_cache.data_type() == base::DataType::kDataTypeFp16) {
    // Use GPU pos version for CUDA Graph compatibility with FP16
    qwen_layers_->flash_attention_decode_gpu_pos_layer_->forward(
        pos_tensor_gpu.ptr<int32_t>(),  // GPU memory pointer
        config_->head_num_, config_->kv_head_num_,
        config_->head_size_, config_->kv_mul_, layer_idx,
        config_->seq_len_, config_->kv_dim_,
        query, mha_output, key_cache, val_cache);
  } else if (attention_type_ == base::AttentionType::kAttentionMHA) {
    // Standard FP32 MHA path with GPU pos
    tensor::Tensor score_storage = get_buffer(ModelBufferType::kScoreStorage);
    qwen_layers_->mha_gpu_pos_layer_->forward(
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
        val_cache);
  } else {
    // FP32 Flash Attention path with GPU pos (FA1 or FA2)
    qwen_layers_->flash_attention_decode_gpu_pos_layer_->forward(
        pos_tensor_gpu.ptr<int32_t>(),
        config_->head_num_, config_->kv_head_num_,
        config_->head_size_, config_->kv_mul_, layer_idx,
        config_->seq_len_, config_->kv_dim_,
        query, mha_output, key_cache, val_cache);
  }

  // WO @ attention output
  tensor::Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
  const auto& wo_layer = qwen_layers_->wo_layers_.at(layer_idx);
  CHECK_NE(wo_layer, nullptr);
  STATUS_CHECK(wo_layer->forward(mha_output, attn_output));
}

void Qwen3VLModel::feed_forward(int32_t layer_idx, const tensor::Tensor& input) const {
  CHECK(qwen_layers_ != nullptr);

  // Residual add
  CHECK_NE(qwen_layers_->add_layer_, nullptr) << "The add layer is null";
  STATUS_CHECK(qwen_layers_->add_layer_->forward(input, get_buffer(ModelBufferType::kAttnOutput), input));

  // FFN rmsnorm (post attention layernorm)
  tensor::Tensor ffn_norm_output = get_buffer(ModelBufferType::kFFNRMSNorm);
  const auto& ffn_rmsnorm = qwen_layers_->rmsnorm_layers_.at(layer_idx + config_->layer_num_);
  CHECK_NE(ffn_rmsnorm, nullptr) << "The FFN rmsnorm layer is null";
  STATUS_CHECK(ffn_rmsnorm->forward(input, ffn_norm_output));

  // Use fused Gate-Up-SwiGLU kernel for decode (single token) efficiency
  // Fused kernel reduces kernel launch overhead and only reads input once
  tensor::Tensor w1_output = get_buffer(ModelBufferType::kW1Output);
  const auto& w1_layer = qwen_layers_->w1_layers_.at(layer_idx);
  const auto& w3_layer = qwen_layers_->w3_layers_.at(layer_idx);
  
  auto w1_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(w1_layer);
  auto w3_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(w3_layer);
  
  if (w1_matmul && w3_matmul && 
      ffn_norm_output.data_type() == base::DataType::kDataTypeFp16) {
    // Use fused layer: W1 GEMV + W3 GEMV + SwiGLU in one kernel
    const auto& w1_weight = w1_matmul->get_weight(0);
    const auto& w3_weight = w3_matmul->get_weight(0);
    
    auto fused_ffn = qwen_layers_->fused_ffn_layer_;
    fused_ffn->set_use_fp16(true);
    fused_ffn->set_input(0, ffn_norm_output);
    fused_ffn->set_input(1, w1_weight);
    fused_ffn->set_input(2, w3_weight);
    fused_ffn->set_output(0, w1_output);
    fused_ffn->set_cuda_config(cuda_config_);
    STATUS_CHECK(fused_ffn->forward());
  } else {
    // Fallback to separate operations
    CHECK_NE(w1_layer, nullptr) << "The w1 layer is null";
    STATUS_CHECK(w1_layer->forward(ffn_norm_output, w1_output));
    
    tensor::Tensor w3_output = get_buffer(ModelBufferType::kW3Output);
    CHECK_NE(w3_layer, nullptr) << "The w3 layer is null";
    STATUS_CHECK(w3_layer->forward(ffn_norm_output, w3_output));
    
    CHECK_NE(qwen_layers_->swiglu_layer_, nullptr) << "The swiglu layer is null";
    STATUS_CHECK(qwen_layers_->swiglu_layer_->forward(w1_output, w3_output, w1_output));
  }

  // W2 (down)
  tensor::Tensor w2_output = get_buffer(ModelBufferType::kW2Output);
  const auto& w2_layer = qwen_layers_->w2_layers_.at(layer_idx);
  CHECK_NE(w2_layer, nullptr) << "The w2 layer is null";
  STATUS_CHECK(w2_layer->forward(w1_output, w2_output));

  // Residual add
  STATUS_CHECK(qwen_layers_->add_layer_->forward(input, w2_output, input));
}

void Qwen3VLModel::cls_logits(const tensor::Tensor& input) const {
  CHECK(qwen_layers_ != nullptr);
  
  // Final RMSNorm
  tensor::Tensor final_norm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  const auto& final_rmsnorm = qwen_layers_->rmsnorm_layers_.at(2 * config_->layer_num_);
  CHECK_NE(final_rmsnorm, nullptr) << "The final rmsnorm layer is null";
  STATUS_CHECK(final_rmsnorm->forward(input, final_norm_output));

  // LM head
  tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
  const auto& cls_layer = qwen_layers_->cls_layer_;
  CHECK_NE(cls_layer, nullptr) << "The cls layer is null";
  STATUS_CHECK(cls_layer->forward(final_norm_output, forward_output));
}

int32_t Qwen3VLModel::post_processing(const tensor::Tensor& pos, bool is_prompt) const {
  tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
  
  // Use sampler to get next token
  size_t next = sampler_->sample(forward_output.ptr<float>(), forward_output.size(),
                                  cuda_config_ ? cuda_config_->stream : nullptr);
  return static_cast<int32_t>(next);
}

// ============================================================================
// Image Preprocessing
// ============================================================================

ImageData Qwen3VLModel::preprocess_image(const std::string& image_path, int max_pixels) const {
  ImageData result;
  
  // 1. Load image
  int width, height, channels;
  auto pixels = image_utils::load_image(image_path, width, height, channels);
  if (pixels.empty()) {
    LOG(ERROR) << "Failed to load image: " << image_path;
    return result;
  }
  
  // 2. Smart resize - matching HuggingFace behavior
  // factor = patch_size (16) - dimensions must be divisible by patch_size
  // min_pixels = 56 * 56 = 3136 (default in HuggingFace)
  // max_pixels can be adjusted: lower = faster ViT, higher = better quality
  int factor = vl_config_.vision.patch_size;  // 16, not 32!
  constexpr int min_pixels = 56 * 56;  // 3136
  auto [resized_pixels, new_width, new_height] = image_utils::smart_resize(
      pixels, width, height, min_pixels, max_pixels, factor);
  
  // 3. Normalize and convert to tensor
  auto image_tensor = image_utils::normalize_to_tensor(resized_pixels, new_width, new_height);
  
  // 4. Calculate grid dimensions
  result.grid_h = new_height / vl_config_.vision.patch_size;
  result.grid_w = new_width / vl_config_.vision.patch_size;
  result.grid_t = 1;
  result.num_patches = result.grid_h * result.grid_w * result.grid_t;
  
  int merge_size = vl_config_.vision.spatial_merge_size;
  result.num_vision_tokens = result.num_patches / (merge_size * merge_size);
  
  // 5. Convert to patches (GPU-accelerated)
  result.pixel_values = image_utils::image_to_patches(
      image_tensor, vl_config_.vision.patch_size, vl_config_.vision.temporal_patch_size,
      cuda_config_ ? cuda_config_->stream : nullptr);
  
  LOG(INFO) << "Preprocessed image: " << image_path;
  LOG(INFO) << "  Grid: " << result.grid_t << "x" << result.grid_h << "x" << result.grid_w;
  LOG(INFO) << "  Patches: " << result.num_patches << " -> Vision tokens: " << result.num_vision_tokens;
  
  return result;
}

// ============================================================================
// Vision Encoder Forward
// ============================================================================

tensor::Tensor Qwen3VLModel::encode_image(const ImageData& image_data) const {
  // Vision encoder forward pass
  // Input: pixel_values [num_patches, patch_dim]
  // Output: visual_embeddings [num_vision_tokens, out_hidden * (1 + num_deepstack)]
  
  LOG(INFO) << "Running vision encoder...";
  
  // Start timing ViT forward
  auto vit_start = std::chrono::high_resolution_clock::now();
  
  int num_patches = image_data.num_patches;
  int hidden_size = vl_config_.vision.hidden_size;
  int intermediate_size = vl_config_.vision.intermediate_size;
  
  // Pre-allocate workspace buffers if needed (only once per session or if size changes)
  auto alloc = base::CUDADeviceAllocatorFactory::get_instance();
  int num_heads = vl_config_.vision.num_heads;  // 16
  int head_dim = hidden_size / num_heads;       // 72
  
  if (!vision_workspace_ || !vision_workspace_->is_valid_for(num_patches)) {
    LOG(INFO) << "Allocating vision workspace for " << num_patches << " patches...";
    vision_workspace_ = std::make_unique<VisionWorkspace>();
    vision_workspace_->max_patches = num_patches;
    vision_workspace_->normed1 = tensor::Tensor(base::DataType::kDataTypeFp16, 
                                                 num_patches, hidden_size, true, alloc);
    vision_workspace_->qkv = tensor::Tensor(base::DataType::kDataTypeFp16, 
                                             num_patches, 3 * hidden_size, true, alloc);
    vision_workspace_->query = tensor::Tensor(base::DataType::kDataTypeFp16, 
                                               num_patches, hidden_size, true, alloc);
    vision_workspace_->key = tensor::Tensor(base::DataType::kDataTypeFp16, 
                                             num_patches, hidden_size, true, alloc);
    vision_workspace_->value = tensor::Tensor(base::DataType::kDataTypeFp16, 
                                               num_patches, hidden_size, true, alloc);
    vision_workspace_->attn_out = tensor::Tensor(base::DataType::kDataTypeFp16, 
                                                  num_patches, hidden_size, true, alloc);
    vision_workspace_->normed2 = tensor::Tensor(base::DataType::kDataTypeFp16, 
                                                 num_patches, hidden_size, true, alloc);
    vision_workspace_->mlp_intermediate = tensor::Tensor(base::DataType::kDataTypeFp16, 
                                                          num_patches, intermediate_size, true, alloc);
    vision_workspace_->proj_out = tensor::Tensor(base::DataType::kDataTypeFp16, 
                                                  num_patches, hidden_size, true, alloc);
    vision_workspace_->output = tensor::Tensor(base::DataType::kDataTypeFp16, 
                                                num_patches, hidden_size, true, alloc);
    vision_workspace_->output2 = tensor::Tensor(base::DataType::kDataTypeFp16, 
                                                 num_patches, hidden_size, true, alloc);
    
    // Attention workspace buffers
    vision_workspace_->q_transposed = tensor::Tensor(base::DataType::kDataTypeFp16,
                                                      num_heads, num_patches, head_dim, true, alloc);
    vision_workspace_->k_transposed = tensor::Tensor(base::DataType::kDataTypeFp16,
                                                      num_heads, num_patches, head_dim, true, alloc);
    vision_workspace_->v_transposed = tensor::Tensor(base::DataType::kDataTypeFp16,
                                                      num_heads, num_patches, head_dim, true, alloc);
    vision_workspace_->out_transposed = tensor::Tensor(base::DataType::kDataTypeFp16,
                                                        num_heads, num_patches, head_dim, true, alloc);
    vision_workspace_->attn_scores = tensor::Tensor(base::DataType::kDataTypeFp16,
                                                     num_heads, num_patches, num_patches, true, alloc);
    
    LOG(INFO) << "Vision workspace allocated: " << num_patches << " patches, " 
              << hidden_size << " hidden, " << intermediate_size << " intermediate";
  }
  
  // 1. Patch embedding
  auto hidden_states = vision_patch_embed(image_data);
  
  // 2. Add position embeddings
  hidden_states = vision_add_pos_embed(hidden_states, image_data.grid_h, image_data.grid_w);
  
  // 3. Compute rotary position embeddings for attention
  auto [cos_cache, sin_cache] = compute_vision_rotary_emb(
      image_data.grid_h, image_data.grid_w, image_data.grid_t);
  
  // 4. Transformer blocks with double-buffering
  // Use output and output2 alternately to avoid cudaMemcpyAsync for residual
  const auto& deepstack_indexes = vl_config_.vision.deepstack_visual_indexes;
  std::vector<tensor::Tensor> deepstack_features;
  
  // Prepare cu_seqlens for attention
  // For single image: cu_seqlens = [0, num_patches]
  std::vector<int32_t> cu_seqlens_host = {0, image_data.num_patches};
  tensor::Tensor cu_seqlens(base::DataType::kDataTypeInt32, 2, true, 
                            base::CUDADeviceAllocatorFactory::get_instance());
  // Use async copy - subsequent kernels will wait on same stream
  cudaMemcpyAsync(cu_seqlens.ptr<void>(), cu_seqlens_host.data(), 
                  2 * sizeof(int32_t), cudaMemcpyHostToDevice, cuda_config_->stream);
  
  // Double buffering: alternate between output and output2
  // Layer 0: input=hidden_states, output=output
  // Layer 1: input=output, output=output2
  // Layer 2: input=output2, output=output
  // ...
  tensor::Tensor* current_input = &hidden_states;
  tensor::Tensor* buffers[2] = {&vision_workspace_->output, &vision_workspace_->output2};
  
  for (int layer_idx = 0; layer_idx < vl_config_.vision.depth; ++layer_idx) {
    tensor::Tensor* current_output = buffers[layer_idx % 2];
    
    vision_transformer_block(*current_input, *current_output, layer_idx, 
                              cu_seqlens, image_data.num_patches,
                              cos_cache, sin_cache, *vision_workspace_);
    
    // Check if this layer outputs deepstack features
    auto it = std::find(deepstack_indexes.begin(), deepstack_indexes.end(), layer_idx);
    if (it != deepstack_indexes.end()) {
      int merger_idx = std::distance(deepstack_indexes.begin(), it);
      auto deepstack_output = vision_merger(*current_output, 
                                             image_data.grid_h, image_data.grid_w, 
                                             image_data.grid_t, true, merger_idx);
      deepstack_features.push_back(deepstack_output);
    }
    
    // Next layer's input is current output
    current_input = current_output;
  }
  
  // The last output is in buffers[(depth-1) % 2]
  tensor::Tensor& final_hidden = *buffers[(vl_config_.vision.depth - 1) % 2];
  
  // 5. Final merger (main output for embeddings)
  auto main_output = vision_merger(final_hidden, 
                                    image_data.grid_h, image_data.grid_w,
                                    image_data.grid_t, false);
  
  // 6. Store deepstack features for use in LLM layers
  // According to Qwen3-VL architecture, deepstack features are added to 
  // LLM hidden states at the first N layers (where N = number of deepstack features)
  // Clear and store the deepstack features for later use in prefill
  deepstack_features_.clear();
  deepstack_features_ = std::move(deepstack_features);
  
  // End timing ViT forward
  cudaStreamSynchronize(cuda_config_->stream);
  auto vit_end = std::chrono::high_resolution_clock::now();
  double vit_encode_time = std::chrono::duration<double, std::milli>(vit_end - vit_start).count();
  
  LOG(INFO) << "Vision encoder complete. Main output: [" << image_data.num_vision_tokens 
            << ", " << vl_config_.vision.out_hidden_size << "]"
            << ", deepstack features: " << deepstack_features_.size() << " tensors";
  LOG(INFO) << "  [Timing] ViT encode time: " << std::fixed << std::setprecision(2) << vit_encode_time << " ms";
  
  // Return only the main output [num_vision_tokens, out_hidden_size]
  return main_output;
}

tensor::Tensor Qwen3VLModel::vision_patch_embed(const ImageData& image_data) const {
  // Apply Conv3D patch embedding via GEMM
  // Input: pixel_values [num_patches, patch_dim] where patch_dim = 3*2*16*16 = 1536
  // Output: [num_patches, hidden_size]
  
  int num_patches = image_data.num_patches;
  int hidden_size = vl_config_.vision.hidden_size;
  int patch_dim = 3 * vl_config_.vision.temporal_patch_size * 
                  vl_config_.vision.patch_size * vl_config_.vision.patch_size;
  
  tensor::Tensor output(base::DataType::kDataTypeFp16, num_patches, hidden_size, true,
                        base::CUDADeviceAllocatorFactory::get_instance());
  
  // Conv3D as GEMM: output = input @ weight.T + bias
  // weight: [hidden_size, patch_dim], need transpose for GEMM
  const half alpha = __float2half(1.0f);
  const half beta = __float2half(0.0f);
  
  // Using cuBLAS: C = A @ B^T
  // A: [num_patches, patch_dim], B: [hidden_size, patch_dim] -> C: [num_patches, hidden_size]
  cublasHgemm(cuda_config_->cublas_handle,
              CUBLAS_OP_T, CUBLAS_OP_N,
              hidden_size, num_patches, patch_dim,
              &alpha,
              vision_layers_->patch_embed_weight.ptr<half>(), patch_dim,
              image_data.pixel_values.ptr<half>(), patch_dim,
              &beta,
              output.ptr<half>(), hidden_size);
  
  // Add bias using layer
  vision_vl_layers_.bias_add_residual_layer_->forward(
      output, vision_layers_->patch_embed_bias, 
      tensor::Tensor(), output, cuda_config_->stream);
  
  LOG(INFO) << "Patch embed: [" << num_patches << ", " << patch_dim << "] -> ["
            << num_patches << ", " << hidden_size << "]";
  
  return output;
}

tensor::Tensor Qwen3VLModel::vision_add_pos_embed(const tensor::Tensor& patch_embeds,
                                                   int grid_h, int grid_w) const {
  // Add position embeddings with bilinear interpolation
  // Input: patch_embeds [num_patches, hidden_size]
  // Output: [num_patches, hidden_size]
  
  int num_patches = patch_embeds.get_dim(0);
  int hidden_size = patch_embeds.get_dim(1);
  int grid_t = 1;  // Single frame for image
  int num_grid_per_side = static_cast<int>(std::sqrt(vl_config_.vision.num_position_embeddings));
  
  tensor::Tensor output(base::DataType::kDataTypeFp16, num_patches, hidden_size, true,
                        base::CUDADeviceAllocatorFactory::get_instance());
  
  // Use layer for position embedding interpolation
  vision_vl_layers_.pos_embed_interpolate_layer_->forward(
      patch_embeds,
      vision_layers_->pos_embed_weight,
      output,
      grid_h, grid_w, grid_t,
      num_grid_per_side,
      vl_config_.vision.spatial_merge_size,
      cuda_config_->stream);
  
  LOG(INFO) << "Added position embeddings: grid=" << grid_h << "x" << grid_w 
            << ", base_grid=" << num_grid_per_side;
  
  return output;
}

std::pair<tensor::Tensor, tensor::Tensor> Qwen3VLModel::compute_vision_rotary_emb(
    int grid_h, int grid_w, int grid_t) const {
  // Compute rotary position embeddings for vision encoder
  // Based on Qwen3VLVisionModel.rot_pos_emb() from transformers
  //
  // HuggingFace implementation:
  //   dim = head_dim // 2 = 36  (for head_dim=72)
  //   inv_freq = 1 / (theta ** (arange(0, dim, 2) / dim))  # 18 frequencies
  //   rotary_pos_emb_full = outer(seq, inv_freq)  # [max_grid, 18]
  //   pos_ids = [h, w] for each token  # [num_tokens, 2]
  //   rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)  # [num_tokens, 36]
  //   emb = cat((rotary_pos_emb, rotary_pos_emb), dim=-1)  # [num_tokens, 72]
  //   cos = cos(emb), sin = sin(emb)
  //
  // So the layout of cos/sin is:
  //   [0:18]   = height frequencies
  //   [18:36]  = width frequencies  
  //   [36:54]  = height frequencies (repeat)
  //   [54:72]  = width frequencies (repeat)
  
  int merge_size = vl_config_.vision.spatial_merge_size;
  int num_heads = vl_config_.vision.num_heads;  // 16
  int hidden_size = vl_config_.vision.hidden_size;  // 1152
  int head_dim = hidden_size / num_heads;  // 72
  int half_head_dim = head_dim / 2;  // 36
  int quarter_head_dim = head_dim / 4;  // 18 (number of frequencies)
  
  // Number of tokens after spatial arrangement
  int merged_h = grid_h / merge_size;
  int merged_w = grid_w / merge_size;
  int num_tokens = grid_t * grid_h * grid_w;  // Before merge, each patch is a token
  
  // Maximum grid dimension
  int max_hw = std::max(grid_h, grid_w);
  
  // Rotary embedding base for Vision Encoder
  // IMPORTANT: Vision encoder uses theta=10000.0 (default), NOT 5000000 (text model)
  // This is defined in Qwen3VLVisionRotaryEmbedding.__init__(dim, theta=10000.0)
  float theta = 10000.0f;
  
  // Compute inverse frequencies: dim = head_dim/2 = 36, so we use 36 for division
  // HuggingFace: inv_freq = 1 / (theta ** (arange(0, dim, 2) / dim))
  // This gives 18 frequencies
  std::vector<float> inv_freq(quarter_head_dim);  // 18 frequencies
  for (int i = 0; i < quarter_head_dim; ++i) {
    inv_freq[i] = 1.0f / std::pow(theta, static_cast<float>(2 * i) / half_head_dim);
  }
  
  // Compute frequency table: freq_table[seq, i] = seq * inv_freq[i]
  std::vector<float> freq_table(max_hw * quarter_head_dim);
  for (int seq = 0; seq < max_hw; ++seq) {
    for (int i = 0; i < quarter_head_dim; ++i) {
      freq_table[seq * quarter_head_dim + i] = seq * inv_freq[i];
    }
  }
  
  // Compute position IDs for each token (height, width)
  // Tokens are arranged in spatial merge order
  std::vector<int32_t> pos_h(num_tokens);
  std::vector<int32_t> pos_w(num_tokens);
  
  int token_idx = 0;
  for (int t = 0; t < grid_t; ++t) {
    for (int block_h = 0; block_h < merged_h; ++block_h) {
      for (int block_w = 0; block_w < merged_w; ++block_w) {
        for (int local_h = 0; local_h < merge_size; ++local_h) {
          for (int local_w = 0; local_w < merge_size; ++local_w) {
            int h = block_h * merge_size + local_h;
            int w = block_w * merge_size + local_w;
            pos_h[token_idx] = h;
            pos_w[token_idx] = w;
            ++token_idx;
          }
        }
      }
    }
  }
  
  // Compute rotary embeddings with HuggingFace layout:
  // emb = cat(cat(h_freq, w_freq), cat(h_freq, w_freq))
  // Layout: [h_freq(18), w_freq(18), h_freq(18), w_freq(18)] = 72 dims
  std::vector<half> cos_data(num_tokens * head_dim);
  std::vector<half> sin_data(num_tokens * head_dim);
  
  for (int i = 0; i < num_tokens; ++i) {
    int h_pos = pos_h[i];
    int w_pos = pos_w[i];
    
    // [0:18]: height frequencies
    for (int j = 0; j < quarter_head_dim; ++j) {
      float freq = freq_table[h_pos * quarter_head_dim + j];
      *reinterpret_cast<uint16_t*>(&cos_data[i * head_dim + j]) = float_to_half(std::cos(freq));
      *reinterpret_cast<uint16_t*>(&sin_data[i * head_dim + j]) = float_to_half(std::sin(freq));
    }
    
    // [18:36]: width frequencies
    for (int j = 0; j < quarter_head_dim; ++j) {
      float freq = freq_table[w_pos * quarter_head_dim + j];
      *reinterpret_cast<uint16_t*>(&cos_data[i * head_dim + quarter_head_dim + j]) = float_to_half(std::cos(freq));
      *reinterpret_cast<uint16_t*>(&sin_data[i * head_dim + quarter_head_dim + j]) = float_to_half(std::sin(freq));
    }
    
    // [36:54]: height frequencies (repeat)
    for (int j = 0; j < quarter_head_dim; ++j) {
      float freq = freq_table[h_pos * quarter_head_dim + j];
      *reinterpret_cast<uint16_t*>(&cos_data[i * head_dim + half_head_dim + j]) = float_to_half(std::cos(freq));
      *reinterpret_cast<uint16_t*>(&sin_data[i * head_dim + half_head_dim + j]) = float_to_half(std::sin(freq));
    }
    
    // [54:72]: width frequencies (repeat)
    for (int j = 0; j < quarter_head_dim; ++j) {
      float freq = freq_table[w_pos * quarter_head_dim + j];
      *reinterpret_cast<uint16_t*>(&cos_data[i * head_dim + half_head_dim + quarter_head_dim + j]) = float_to_half(std::cos(freq));
      *reinterpret_cast<uint16_t*>(&sin_data[i * head_dim + half_head_dim + quarter_head_dim + j]) = float_to_half(std::sin(freq));
    }
  }
  
  // Create GPU tensors
  auto alloc = base::CUDADeviceAllocatorFactory::get_instance();
  tensor::Tensor cos_cache(base::DataType::kDataTypeFp16, num_tokens, head_dim, true, alloc);
  tensor::Tensor sin_cache(base::DataType::kDataTypeFp16, num_tokens, head_dim, true, alloc);
  
  // Use async copy - subsequent kernels will wait on same stream
  cudaMemcpyAsync(cos_cache.ptr<void>(), cos_data.data(), 
                  num_tokens * head_dim * sizeof(half), cudaMemcpyHostToDevice, 
                  cuda_config_->stream);
  cudaMemcpyAsync(sin_cache.ptr<void>(), sin_data.data(), 
                  num_tokens * head_dim * sizeof(half), cudaMemcpyHostToDevice,
                  cuda_config_->stream);
  
  LOG(INFO) << "Computed vision rotary embeddings: [" << num_tokens << ", " << head_dim << "]";
  
  return {cos_cache, sin_cache};
}

void Qwen3VLModel::vision_transformer_block(const tensor::Tensor& hidden_states,
                                             tensor::Tensor& output_buffer,
                                             int block_idx,
                                             const tensor::Tensor& cu_seqlens,
                                             int max_seqlen,
                                             const tensor::Tensor& cos_cache,
                                             const tensor::Tensor& sin_cache,
                                             VisionWorkspace& ws) const {
  // Vision transformer block forward (optimized with double-buffering)
  // x = x + attn(norm1(x))
  // x = x + mlp(norm2(x))
  //
  // Double-buffering: hidden_states and output_buffer are always different tensors,
  // so we can use hidden_states directly as residual without copying.
  
  const auto& block = vision_layers_->blocks[block_idx];
  int num_tokens = hidden_states.get_dim(0);
  int hidden_size = hidden_states.get_dim(1);
  int num_heads = vl_config_.vision.num_heads;
  int head_dim = hidden_size / num_heads;
  
  // 1. LayerNorm (norm1)
  vision_vl_layers_.layernorm_with_bias_layer_->forward(
      hidden_states, block.norm1_weight, block.norm1_bias,
      ws.normed1, 1e-6f, cuda_config_->stream);
  
  // 2. QKV projection: [num_tokens, hidden_size] -> [num_tokens, 3*hidden_size]
  const half alpha = __float2half(1.0f);
  const half beta = __float2half(0.0f);
  
  cublasHgemm(cuda_config_->cublas_handle,
              CUBLAS_OP_T, CUBLAS_OP_N,
              3 * hidden_size, num_tokens, hidden_size,
              &alpha,
              block.qkv_weight.ptr<half>(), hidden_size,
              ws.normed1.ptr<half>(), hidden_size,
              &beta,
              ws.qkv.ptr<half>(), 3 * hidden_size);
  
  // Add QKV bias
  vision_vl_layers_.bias_add_residual_layer_->forward(
      ws.qkv, block.qkv_bias, tensor::Tensor(), ws.qkv, cuda_config_->stream);
  
  // 3. Self-Attention with RoPE (OPTIMIZED: fused split + RoPE + transpose)
  // This fuses 3 operations into 1 kernel, saving memory bandwidth
  vision_vl_layers_.fused_split_rope_transpose_layer_->forward(
      ws.qkv, cos_cache, sin_cache,
      vision_workspace_->q_transposed,
      vision_workspace_->k_transposed,
      vision_workspace_->v_transposed,
      num_tokens, num_heads, head_dim,
      cuda_config_->stream);
  
  // Compute attention with pre-transposed Q, K, V using cuBLAS
  // Note: Flash Attention kernel was 18x slower than cuBLAS on Orin
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  vision_vl_layers_.vision_attention_layer_->forward(
      vision_workspace_->q_transposed,
      vision_workspace_->k_transposed,
      vision_workspace_->v_transposed,
      ws.attn_out,
      vision_workspace_->out_transposed,
      vision_workspace_->attn_scores,
      num_tokens, num_heads, head_dim, scale,
      cuda_config_.get());
  
  // 4. Output projection
  cublasHgemm(cuda_config_->cublas_handle,
              CUBLAS_OP_T, CUBLAS_OP_N,
              hidden_size, num_tokens, hidden_size,
              &alpha,
              block.proj_weight.ptr<half>(), hidden_size,
              ws.attn_out.ptr<half>(), hidden_size,
              &beta,
              ws.proj_out.ptr<half>(), hidden_size);

  // Add bias and residual: output_buffer = proj_out + bias + hidden_states (residual)
  // Since hidden_states != output_buffer (double buffering), this is safe
  vision_vl_layers_.bias_add_residual_layer_->forward(
      ws.proj_out, block.proj_bias, hidden_states, output_buffer, cuda_config_->stream);
  
  // 5. LayerNorm (norm2)
  vision_vl_layers_.layernorm_with_bias_layer_->forward(
      output_buffer, block.norm2_weight, block.norm2_bias,
      ws.normed2, 1e-6f, cuda_config_->stream);
  
  // 6. MLP: fc1 + fused(bias+GELU) + fc2
  // MLP output adds residual from output_buffer (after attention)
  vision_vl_layers_.vision_mlp_layer_->forward(
      ws.normed2, 
      block.mlp_fc1_weight, block.mlp_fc1_bias,
      block.mlp_fc2_weight, block.mlp_fc2_bias,
      output_buffer, output_buffer, ws.mlp_intermediate, cuda_config_.get());
}

tensor::Tensor Qwen3VLModel::vision_merger(const tensor::Tensor& hidden_states,
                                            int grid_h, int grid_w, int grid_t,
                                            bool is_deepstack, int merger_idx) const {
  // Merge spatial patches: 4 patches -> 1 token
  // Input: [num_patches, hidden_size]
  // Output: [num_vision_tokens, out_hidden_size]
  // 
  // HuggingFace flow (when use_postshuffle_norm=False):
  // 1. LayerNorm on [num_patches, 1152]
  // 2. view(-1, 4608) -> [num_vision_tokens, 4608]
  // 3. fc1 -> GELU -> fc2
  
  int merge_size = vl_config_.vision.spatial_merge_size;
  int num_patches = hidden_states.get_dim(0);
  int hidden_size = hidden_states.get_dim(1);
  int num_vision_tokens = (grid_h * grid_w * grid_t) / (merge_size * merge_size);
  int merged_hidden = hidden_size * merge_size * merge_size;  // 4*1152 = 4608
  int out_hidden = vl_config_.vision.out_hidden_size;         // 4096
  
  auto alloc = base::CUDADeviceAllocatorFactory::get_instance();
  
  // Select merger weights
  const Qwen3VLVisionLayers::Merger* merger;
  if (is_deepstack) {
    CHECK(merger_idx < vision_layers_->deepstack_mergers.size());
    merger = &vision_layers_->deepstack_mergers[merger_idx];
  } else {
    merger = &vision_layers_->merger;
  }
  
  // 1. LayerNorm on [num_patches, hidden_size] BEFORE spatial merge
  tensor::Tensor normed(base::DataType::kDataTypeFp16, num_patches, hidden_size, true, alloc);
  vision_vl_layers_.layernorm_with_bias_layer_->forward(
      hidden_states, merger->norm_weight, merger->norm_bias, 
      normed, 1e-6f, cuda_config_->stream);
  
  // 2. Spatial merge: [num_patches, hidden_size] -> [num_vision_tokens, merged_hidden]
  tensor::Tensor merged(base::DataType::kDataTypeFp16, num_vision_tokens, merged_hidden, true, alloc);
  vision_vl_layers_.spatial_merge_layer_->forward(
      normed, merged, grid_t, grid_h, grid_w, 
      hidden_size, merge_size, cuda_config_->stream);
  
  // 3. Apply MLP: fc1 + GELU -> fc2
  tensor::Tensor output(base::DataType::kDataTypeFp16, num_vision_tokens, out_hidden, true, alloc);
  tensor::Tensor intermediate(base::DataType::kDataTypeFp16, num_vision_tokens, merged_hidden, true, alloc);
  
  vision_vl_layers_.vision_merger_mlp_layer_->forward(
      merged,
      merger->fc1_weight, merger->fc1_bias,
      merger->fc2_weight, merger->fc2_bias,
      output, intermediate, cuda_config_.get());
  
  LOG(INFO) << "Vision merger: [" << num_patches << ", " << hidden_size << "] -> ["
            << num_vision_tokens << ", " << out_hidden << "]"
            << (is_deepstack ? " (deepstack " + std::to_string(merger_idx) + ")" : "");
  
  return output;
}

// ============================================================================
// Multimodal Embedding
// ============================================================================

tensor::Tensor Qwen3VLModel::prepare_multimodal_embeddings(
    const std::vector<int>& tokens,
    const ImageData* image_data) const {
  
  // Start timing text embedding
  auto text_embed_start = std::chrono::high_resolution_clock::now();
  
  // Get text embeddings
  auto embed_out = embedding(tokens);
  
  auto text_embed_end = std::chrono::high_resolution_clock::now();
  double text_embed_time = std::chrono::duration<double, std::milli>(text_embed_end - text_embed_start).count();
  LOG(INFO) << "  [Timing] Text embedding time: " << std::fixed << std::setprecision(2) << text_embed_time << " ms";
  
  if (!image_data || image_data->pixel_values.is_empty()) {
    return embed_out.input_embeddings;
  }
  
  // Start timing ViT encode
  auto vit_start = std::chrono::high_resolution_clock::now();
  
  // Get visual embeddings from vision encoder
  // This also populates deepstack_features_ for use in prefill
  auto visual_embeds = encode_image(*image_data);
  
  auto vit_end = std::chrono::high_resolution_clock::now();
  double vit_time = std::chrono::duration<double, std::milli>(vit_end - vit_start).count();
  LOG(INFO) << "  [Timing] ViT encode total time: " << std::fixed << std::setprecision(2) << vit_time << " ms";
  
  // visual_embeds shape: [num_vision_tokens, dim] (main merger output only)
  // deepstack_features_ contains 3 tensors [num_vision_tokens, dim] each
  // These will be added to LLM hidden states in layers 0, 1, 2
  
  int num_vision_tokens = image_data->num_vision_tokens;
  int dim = config_->dim_;
  
  // Find the image token position
  int image_token_id = vl_config_.special_tokens.image_token_id;
  
  int image_token_pos = -1;
  for (int i = 0; i < tokens.size(); ++i) {
    if (tokens[i] == image_token_id) {
      image_token_pos = i;
      break;
    }
  }
  
  if (image_token_pos < 0) {
    LOG(WARNING) << "No image token found in input, using text-only embeddings";
    return embed_out.input_embeddings;
  }
  
  // Store visual position mask for deepstack processing
  // This marks which positions in the final sequence are visual tokens
  visual_pos_start_ = image_token_pos;
  visual_pos_end_ = image_token_pos + num_vision_tokens;
  
  // Create new embedding tensor with space for vision tokens
  // new_seq_len = tokens.size() - 1 (remove image token) + num_vision_tokens
  int new_seq_len = static_cast<int>(tokens.size()) - 1 + num_vision_tokens;
  
  auto alloc = base::CUDADeviceAllocatorFactory::get_instance();
  tensor::Tensor multimodal_embeds(base::DataType::kDataTypeFp16, new_seq_len, dim, true, alloc);
  
  // OPTIMIZED: Use fused layer to assemble multimodal embeddings
  // Replaces 3 separate cudaMemcpyAsync calls with a single kernel launch
  vision_vl_layers_.fused_multimodal_embed_layer_->forward(
      embed_out.input_embeddings,  // text embeddings [tokens.size(), dim]
      visual_embeds,               // vision embeddings [num_vision_tokens, dim]
      multimodal_embeds,           // output [new_seq_len, dim]
      image_token_pos,
      num_vision_tokens,
      static_cast<int>(tokens.size()),
      dim,
      cuda_config_->stream
  );
  
  // ========== Generate M-RoPE 3D positions ==========
  // For visual tokens: (t=0, h=row_idx, w=col_idx) with 2D spatial positions
  // For text tokens: (t=pos, h=pos, w=pos) using sequential position
  //
  // The key insight from HuggingFace Qwen3-VL implementation:
  // - Visual tokens have spatial positions from their grid location
  // - Text tokens before/after visuals have sequential positions
  // - All positions share the same continuous sequence to maintain causality
  
  mrope_pos_t_.resize(new_seq_len);
  mrope_pos_h_.resize(new_seq_len);
  mrope_pos_w_.resize(new_seq_len);
  
  int merged_grid_h = image_data->grid_h / vl_config_.vision.spatial_merge_size;
  int merged_grid_w = image_data->grid_w / vl_config_.vision.spatial_merge_size;
  
  // Track text position - starts at 0 for tokens before image
  int text_pos = 0;
  
  // Text tokens before image
  for (int i = 0; i < image_token_pos; ++i) {
    mrope_pos_t_[i] = text_pos;
    mrope_pos_h_[i] = text_pos;
    mrope_pos_w_[i] = text_pos;
    text_pos++;
  }
  
  // Visual tokens - use 2D spatial positions
  // Following HuggingFace: for images, t=same for all, h/w vary by grid position
  // The t-position for visual tokens is the current text_pos (preserved for continuity)
  int visual_base_t = text_pos;
  for (int v = 0; v < num_vision_tokens; ++v) {
    int row = v / merged_grid_w;
    int col = v % merged_grid_w;
    mrope_pos_t_[image_token_pos + v] = visual_base_t;  // All visual tokens share same t
    mrope_pos_h_[image_token_pos + v] = visual_base_t + row;  // h varies by row
    mrope_pos_w_[image_token_pos + v] = visual_base_t + col;  // w varies by col
  }
  
  // After visual tokens, text continues from max visual position
  // max visual position = visual_base_t + max(grid_h, grid_w) - 1
  int max_visual_extent = std::max(merged_grid_h, merged_grid_w);
  text_pos = visual_base_t + max_visual_extent;
  
  // Text tokens after image
  int after_image_start = image_token_pos + num_vision_tokens;
  for (int i = after_image_start; i < new_seq_len; ++i) {
    mrope_pos_t_[i] = text_pos;
    mrope_pos_h_[i] = text_pos;
    mrope_pos_w_[i] = text_pos;
    text_pos++;
  }
  
  // Store the max position for decode phase
  mrope_max_text_pos_ = text_pos - 1;  // Last position used
  
  LOG(INFO) << "M-RoPE positions generated: "
            << "visual_base_t=" << visual_base_t
            << ", merged_grid=" << merged_grid_h << "x" << merged_grid_w
            << ", max_text_pos=" << mrope_max_text_pos_;
  
  // End timing embedding assembly
  cudaStreamSynchronize(cuda_config_->stream);
  auto embed_assembly_end = std::chrono::high_resolution_clock::now();
  double embed_assembly_time = std::chrono::duration<double, std::milli>(embed_assembly_end - vit_end).count();
  LOG(INFO) << "  [Timing] Embedding assembly time: " << std::fixed << std::setprecision(2) << embed_assembly_time << " ms";
  
  LOG(INFO) << "Created multimodal embeddings: text tokens=" << tokens.size()
            << ", vision tokens=" << num_vision_tokens
            << ", total=" << new_seq_len
            << ", image_pos=" << image_token_pos
            << ", deepstack features=" << deepstack_features_.size();
  
  return multimodal_embeds;
}

// ============================================================================
// Prefill and Decode
// ============================================================================

base::Status Qwen3VLModel::multimodal_prefill(const std::vector<int>& tokens,
                                               const std::string& image_path) const {
  // Preprocess image
  ImageData image_data;
  if (!image_path.empty()) {
    image_data = preprocess_image(image_path);
  }
  
  // Prepare multimodal embeddings
  auto embeddings = prepare_multimodal_embeddings(tokens, 
      image_path.empty() ? nullptr : &image_data);
  
  // Run prefill
  int seq_len = static_cast<int>(tokens.size());
  if (!image_data.pixel_values.is_empty()) {
    // Adjust for vision tokens
    seq_len = seq_len - 1 + image_data.num_vision_tokens;  // Replace 1 image token
  }
  
  return prefill(embeddings, seq_len, 0);
}

base::Status Qwen3VLModel::prefill(const tensor::Tensor& input_embeddings,
                                    int32_t seq_len, int32_t start_pos) const {
  // ==========================================================================
  // OPTIMIZED BATCHED PREFILL with DeepStack support
  // ==========================================================================
  // Instead of processing tokens one-by-one, we process all tokens in parallel
  // using batched matrix operations and Flash Attention prefill kernel
  // ==========================================================================
  
  LOG(INFO) << "Batched Prefill: seq_len=" << seq_len << ", start_pos=" << start_pos;
  if (visual_pos_start_ >= 0 && visual_pos_end_ > visual_pos_start_) {
    LOG(INFO) << "  Visual positions: [" << visual_pos_start_ << ", " << visual_pos_end_ << ")";
    LOG(INFO) << "  Deepstack features: " << deepstack_features_.size();
  }
  
  // OPTIMIZED: Upload M-RoPE position arrays using single contiguous transfer
  // Instead of 3 separate cudaMemcpyAsync + sync, use pinned memory + single transfer
  size_t total_positions = mrope_pos_t_.size();
  if (total_positions > 0 && total_positions > mrope_pos_gpu_capacity_) {
    // Free old GPU allocation
    if (mrope_pos_gpu_) cudaFree(mrope_pos_gpu_);
    
    // Allocate contiguous GPU memory for all 3 arrays
    cudaMalloc(&mrope_pos_gpu_, 3 * total_positions * sizeof(int32_t));
    mrope_pos_t_gpu_ = mrope_pos_gpu_;
    mrope_pos_h_gpu_ = mrope_pos_gpu_ + total_positions;
    mrope_pos_w_gpu_ = mrope_pos_gpu_ + 2 * total_positions;
    mrope_pos_gpu_capacity_ = total_positions;
    
    // Allocate/resize pinned memory for async transfer
    if (total_positions > mrope_pos_pinned_capacity_) {
      if (mrope_pos_pinned_) cudaFreeHost(mrope_pos_pinned_);
      cudaMallocHost(&mrope_pos_pinned_, 3 * total_positions * sizeof(int32_t));
      mrope_pos_pinned_capacity_ = total_positions;
    }
  }
  
  if (total_positions > 0) {
    // Pack positions into contiguous pinned memory
    int32_t* pinned_t = mrope_pos_pinned_;
    int32_t* pinned_h = mrope_pos_pinned_ + total_positions;
    int32_t* pinned_w = mrope_pos_pinned_ + 2 * total_positions;
    memcpy(pinned_t, mrope_pos_t_.data(), total_positions * sizeof(int32_t));
    memcpy(pinned_h, mrope_pos_h_.data(), total_positions * sizeof(int32_t));
    memcpy(pinned_w, mrope_pos_w_.data(), total_positions * sizeof(int32_t));
    
    // Single async H2D transfer for all 3 arrays
    cudaMemcpyAsync(mrope_pos_gpu_, mrope_pos_pinned_,
                    3 * total_positions * sizeof(int32_t), cudaMemcpyHostToDevice,
                    cuda_config_->stream);
    // Note: No sync needed here as subsequent kernels on the same stream will wait
  }
  
  auto alloc = base::CUDADeviceAllocatorFactory::get_instance();
  base::DataType activation_dtype = base::DataType::kDataTypeFp16;
  size_t elem_size = sizeof(uint16_t);  // FP16
  
  int dim = config_->dim_;
  int kv_dim = config_->kv_dim_;
  int hidden_dim = config_->hidden_dim_;
  int head_size = config_->head_size_;
  
  // Number of deepstack layers = min(num_deepstack_features, config_->layer_num_)
  int num_deepstack_layers = std::min(static_cast<int>(deepstack_features_.size()), 
                                       config_->layer_num_);
  
  // Pre-allocate ALL batched buffers once to avoid per-layer allocation overhead
  // This significantly reduces memory allocation time and improves cache efficiency
  // OPTIMIZED: Use double-buffering for hidden states to avoid initialization copy
  // Layer 0: input=input_embeddings, output=hidden_buf0
  // Layer 1: input=hidden_buf0, output=hidden_buf1  
  // Layer 2: input=hidden_buf1, output=hidden_buf0
  // ...
  tensor::Tensor hidden_buf0(activation_dtype, seq_len, dim, true, alloc);
  tensor::Tensor hidden_buf1(activation_dtype, seq_len, dim, true, alloc);
  tensor::Tensor rms_out(activation_dtype, seq_len, dim, true, alloc);
  tensor::Tensor query_out(activation_dtype, seq_len, dim, true, alloc);
  tensor::Tensor key_out(activation_dtype, seq_len, kv_dim, true, alloc);
  tensor::Tensor value_out(activation_dtype, seq_len, kv_dim, true, alloc);
  tensor::Tensor mha_out(activation_dtype, seq_len, dim, true, alloc);
  
  // Pre-allocate FFN buffers (previously allocated per-layer in batched_feed_forward)
  tensor::Tensor ffn_norm_out(activation_dtype, seq_len, dim, true, alloc);
  tensor::Tensor w1_out(activation_dtype, seq_len, hidden_dim, true, alloc);
  tensor::Tensor w3_out(activation_dtype, seq_len, hidden_dim, true, alloc);
  tensor::Tensor w2_out(activation_dtype, seq_len, dim, true, alloc);
  
  // OPTIMIZED: No copy needed - use input_embeddings directly as first layer input
  // Double-buffering pointers
  tensor::Tensor* hidden_buffers[2] = {&hidden_buf0, &hidden_buf1};
  
  // Process all layers with batched operations using double-buffering
  // Layer 0: input=input_embeddings (const, used directly), output=hidden_buf0
  // Layer 1+: alternates between hidden_buf0 and hidden_buf1
  tensor::Tensor* final_hidden = nullptr;
  for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
    // Determine input and output buffers for this layer
    const tensor::Tensor* layer_input;
    tensor::Tensor* layer_output;
    
    if (layer_idx == 0) {
      // First layer: use input_embeddings directly (avoid D2D copy)
      layer_input = &input_embeddings;
      layer_output = hidden_buffers[0];  // Output to hidden_buf0
    } else {
      // Subsequent layers: alternate between buffers
      layer_input = hidden_buffers[(layer_idx - 1) % 2];
      layer_output = hidden_buffers[layer_idx % 2];
    }
    
    // 1. Batched Attention RMSNorm
    batched_attention_rms(layer_idx, *layer_input, rms_out, seq_len);
    
    // 2. Batched Q/K/V projections + RoPE + KV cache update
    batched_attention_qkv(layer_idx, rms_out, query_out, key_out, value_out, 
                          seq_len, start_pos);
    
    // 3. Batched Multi-head attention using Flash Attention
    batched_attention_mha(layer_idx, query_out, mha_out, seq_len, start_pos);
    
    // 4. Residual add: layer_output = layer_input + mha_out (via batched_add_layer_)
    STATUS_CHECK(qwen_layers_->batched_add_layer_->forward(*layer_input, mha_out, *layer_output));
    
    // 5. Batched Feed Forward with residual (modifies layer_output in-place)
    batched_feed_forward_optimized(layer_idx, *layer_output, ffn_norm_out, 
                                   w1_out, w3_out, w2_out, seq_len);
    
    // 6. DeepStack: Add visual features after first N layers
    if (layer_idx < num_deepstack_layers && visual_pos_start_ >= 0) {
      int num_visual_tokens = visual_pos_end_ - visual_pos_start_;
      const auto& ds_feat = deepstack_features_[layer_idx];
      
      // Add deepstack features to visual token positions
      // layer_output[visual_pos_start_:visual_pos_end_] += ds_feat
      half* hidden_ptr = layer_output->ptr<half>() + visual_pos_start_ * dim;
      const half* ds_ptr = ds_feat.ptr<half>();
      
      STATUS_CHECK(qwen_layers_->batched_add_layer_->forward_raw(
          hidden_ptr, ds_ptr, hidden_ptr, num_visual_tokens * dim));
    }
    
    final_hidden = layer_output;
  }
  
  // OPTIMIZED: Use pointer slice to access last token directly, avoiding D2D copy
  // The final hidden state is in final_hidden buffer, we can pass its last token slice
  // directly to cls_logits and sampling
  tensor::Tensor decode_input = get_buffer(ModelBufferType::kDecodeInput);
  void* last_token_ptr = final_hidden->ptr<uint8_t>() + (seq_len - 1) * dim * elem_size;
  
  // Still need to copy to decode_input for subsequent decode steps to use
  // But this is a small copy (single token = 8KB for dim=4096)
  cudaMemcpyAsync(decode_input.ptr<void>(), last_token_ptr,
                  dim * elem_size, cudaMemcpyDeviceToDevice, cuda_config_->stream);
  
  // Save prefill sequence length for decode
  prefill_seq_len_ = seq_len;
  
  cudaStreamSynchronize(cuda_config_->stream);
  LOG(INFO) << "Batched Prefill complete";
  
  return base::error::Success();
}

// Sample the first token from prefill output
int Qwen3VLModel::sample_first_token() const {
  tensor::Tensor input = get_buffer(ModelBufferType::kDecodeInput);
  tensor::Tensor pos_tensor = get_buffer(ModelBufferType::kInputPos);
  
  // The input buffer already has the last token's hidden state from prefill
  cls_logits(input);
  
  if (cuda_config_ && cuda_config_->stream) {
    cudaStreamSynchronize(cuda_config_->stream);
  }
  
  return post_processing(pos_tensor, false);
}

base::Status Qwen3VLModel::decode_step(const tensor::Tensor& input,
                                        int32_t pos, int& next) const {
  // Check if CUDA Graph is enabled
  bool use_graph = cuda_config_ && cuda_config_->use_cuda_graph && 
                   cuda_config_->graph_context;
  
  if (use_graph) {
    auto& graph_ctx = cuda_config_->graph_context;
    auto& graph = graph_ctx->decode_graph;
    
    // Get fixed-address buffers for CUDA Graph
    tensor::Tensor decode_input = get_buffer(ModelBufferType::kDecodeInput);
    tensor::Tensor pos_tensor_gpu = get_buffer(ModelBufferType::kInputPosGPU);      // For M-RoPE (text_pos)
    tensor::Tensor kv_cache_pos_gpu = get_buffer(ModelBufferType::kKVCachePosGPU);  // For KV cache (original pos)
    tensor::Tensor pos_pinned = get_buffer(ModelBufferType::kInputPosPinned);
    tensor::Tensor kv_cache_pos_pinned = get_buffer(ModelBufferType::kKVCachePosPinned);
    tensor::Tensor argmax_output = get_buffer(ModelBufferType::kArgmaxOutput);
    tensor::Tensor argmax_pinned = get_buffer(ModelBufferType::kArgmaxPinned);
    
    size_t elem_size = sizeof(uint16_t);  // FP16
    
    // Copy input embedding to fixed decode_input buffer
    cudaMemcpyAsync(decode_input.ptr<void>(), input.ptr<void>(),
                    config_->dim_ * elem_size, cudaMemcpyDeviceToDevice, 
                    cuda_config_->stream);
    
    // Calculate text position for M-RoPE (decode uses same pos for t/h/w)
    // text_pos = mrope_max_text_pos_ + (pos - prefill_seq_len_) + 1
    int32_t text_pos = mrope_max_text_pos_ + (pos - prefill_seq_len_) + 1;
    
    // Update M-RoPE position using pinned memory for async H2D transfer
    *const_cast<int32_t*>(pos_pinned.ptr<int32_t>()) = text_pos;
    cudaMemcpyAsync(const_cast<int32_t*>(pos_tensor_gpu.ptr<int32_t>()), 
                    pos_pinned.ptr<int32_t>(), sizeof(int32_t), 
                    cudaMemcpyHostToDevice, cuda_config_->stream);
    
    // Update KV cache position using pinned memory for async H2D transfer
    *const_cast<int32_t*>(kv_cache_pos_pinned.ptr<int32_t>()) = pos;
    cudaMemcpyAsync(const_cast<int32_t*>(kv_cache_pos_gpu.ptr<int32_t>()), 
                    kv_cache_pos_pinned.ptr<int32_t>(), sizeof(int32_t), 
                    cudaMemcpyHostToDevice, cuda_config_->stream);
    
    bool need_capture = graph_ctx->needs_recapture || !graph->is_valid();
    
    if (need_capture && !graph->is_disabled()) {
      // Sync before capture
      cudaStreamSynchronize(cuda_config_->stream);
      
      // Capture the graph
      if (graph->begin_capture(cuda_config_->stream)) {
        for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
          attention_rms(layer_idx, decode_input);
          attention_qkv_with_graph(layer_idx, pos_tensor_gpu, kv_cache_pos_gpu);
          attention_mha_with_graph(layer_idx, kv_cache_pos_gpu);
          feed_forward(layer_idx, decode_input);
        }
        cls_logits(decode_input);
        
        if (graph->end_capture(cuda_config_->stream)) {
          graph_ctx->graph_recaptures++;
          graph_ctx->needs_recapture = false;
        }
      }
    }
    
    if (graph->is_valid()) {
      // Launch the captured graph
      if (graph->launch(cuda_config_->stream)) {
        graph_ctx->graph_launches++;
        
        // Use optimized post_processing with pre-allocated buffers
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
      // If launch failed, fall through to normal execution
      graph_ctx->invalidate();
    }
  }
  
  // Normal execution (no graph, or graph capture/launch failed)
  tensor::Tensor pos_tensor = get_buffer(ModelBufferType::kInputPos);
  pos_tensor.index<int32_t>(0) = pos;
  
  tensor::Tensor decode_input = get_buffer(ModelBufferType::kDecodeInput);
  size_t elem_size = sizeof(uint16_t);  // FP16
  
  // Copy input embedding to decode input buffer
  cudaMemcpyAsync(decode_input.ptr<void>(), input.ptr<void>(),
                  config_->dim_ * elem_size, cudaMemcpyDeviceToDevice, 
                  cuda_config_->stream);
  
  // Run transformer layers
  for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
    attention_rms(layer_idx, decode_input);
    attention_qkv(layer_idx, pos_tensor);
    attention_mha(layer_idx, pos_tensor);
    feed_forward(layer_idx, decode_input);
  }
  
  // LM head and sampling
  cls_logits(decode_input);
  
  if (cuda_config_ && cuda_config_->stream) {
    cudaStreamSynchronize(cuda_config_->stream);
  }
  
  next = post_processing(pos_tensor, false);
  return base::error::Success();
}

base::Status Qwen3VLModel::decode_step_optimized(int32_t pos, int& next) const {
  // OPTIMIZED decode step that assumes embedding is already in decode_input buffer
  // This avoids the D2D copy from input to decode_input
  
  // Check if CUDA Graph is enabled
  bool use_graph = cuda_config_ && cuda_config_->use_cuda_graph && 
                   cuda_config_->graph_context;
  
  if (use_graph) {
    auto& graph_ctx = cuda_config_->graph_context;
    auto& graph = graph_ctx->decode_graph;
    
    // Get fixed-address buffers for CUDA Graph
    tensor::Tensor decode_input = get_buffer(ModelBufferType::kDecodeInput);
    tensor::Tensor pos_tensor_gpu = get_buffer(ModelBufferType::kInputPosGPU);
    tensor::Tensor kv_cache_pos_gpu = get_buffer(ModelBufferType::kKVCachePosGPU);
    tensor::Tensor pos_pinned = get_buffer(ModelBufferType::kInputPosPinned);
    tensor::Tensor kv_cache_pos_pinned = get_buffer(ModelBufferType::kKVCachePosPinned);
    tensor::Tensor argmax_output = get_buffer(ModelBufferType::kArgmaxOutput);
    tensor::Tensor argmax_pinned = get_buffer(ModelBufferType::kArgmaxPinned);
    
    // NOTE: No D2D copy needed - embedding is already in decode_input buffer
    
    // Calculate text position for M-RoPE
    int32_t text_pos = mrope_max_text_pos_ + (pos - prefill_seq_len_) + 1;
    
    // Update M-RoPE position using pinned memory for async H2D transfer
    *const_cast<int32_t*>(pos_pinned.ptr<int32_t>()) = text_pos;
    cudaMemcpyAsync(const_cast<int32_t*>(pos_tensor_gpu.ptr<int32_t>()), 
                    pos_pinned.ptr<int32_t>(), sizeof(int32_t), 
                    cudaMemcpyHostToDevice, cuda_config_->stream);
    
    // Update KV cache position using pinned memory for async H2D transfer
    *const_cast<int32_t*>(kv_cache_pos_pinned.ptr<int32_t>()) = pos;
    cudaMemcpyAsync(const_cast<int32_t*>(kv_cache_pos_gpu.ptr<int32_t>()), 
                    kv_cache_pos_pinned.ptr<int32_t>(), sizeof(int32_t), 
                    cudaMemcpyHostToDevice, cuda_config_->stream);
    
    bool need_capture = graph_ctx->needs_recapture || !graph->is_valid();
    
    if (need_capture && !graph->is_disabled()) {
      cudaStreamSynchronize(cuda_config_->stream);
      
      if (graph->begin_capture(cuda_config_->stream)) {
        for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
          attention_rms(layer_idx, decode_input);
          attention_qkv_with_graph(layer_idx, pos_tensor_gpu, kv_cache_pos_gpu);
          attention_mha_with_graph(layer_idx, kv_cache_pos_gpu);
          feed_forward(layer_idx, decode_input);
        }
        cls_logits(decode_input);
        
        if (graph->end_capture(cuda_config_->stream)) {
          graph_ctx->graph_recaptures++;
          graph_ctx->needs_recapture = false;
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
  
  // Normal execution (no graph, or graph capture/launch failed)
  tensor::Tensor pos_tensor = get_buffer(ModelBufferType::kInputPos);
  pos_tensor.index<int32_t>(0) = pos;
  
  tensor::Tensor decode_input = get_buffer(ModelBufferType::kDecodeInput);
  // NOTE: No D2D copy needed - embedding is already in decode_input buffer
  
  // Run transformer layers
  for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
    attention_rms(layer_idx, decode_input);
    attention_qkv(layer_idx, pos_tensor);
    attention_mha(layer_idx, pos_tensor);
    feed_forward(layer_idx, decode_input);
  }
  
  cls_logits(decode_input);
  
  if (cuda_config_ && cuda_config_->stream) {
    cudaStreamSynchronize(cuda_config_->stream);
  }
  
  next = post_processing(pos_tensor, false);
  return base::error::Success();
}

// ============================================================================
// Generation
// ============================================================================

std::string Qwen3VLModel::generate(const std::string& image_path,
                                    const std::string& prompt,
                                    int max_tokens) const {
  // Tokenize prompt
  auto tokens = encode(prompt);
  
  // Calculate prefill sequence length - this includes vision tokens
  // Preprocess image once to get vision token count
  int prefill_seq_len = static_cast<int>(tokens.size());
  if (!image_path.empty()) {
    // Calculate vision tokens based on image path
    ImageData image_data = preprocess_image(image_path);
    // Adjust for vision tokens: replace 1 image token with num_vision_tokens
    prefill_seq_len = static_cast<int>(tokens.size()) - 1 + image_data.num_vision_tokens;
    // Store for later use
    const_cast<Qwen3VLModel*>(this)->cached_image_data_ = std::move(image_data);
  }
  
  // Multimodal prefill
  auto status = multimodal_prefill(tokens, image_path);
  if (!status) {
    LOG(ERROR) << "Prefill failed: " << status.get_err_code();
    return "";
  }
  
  // Sample first token using the output from the last prefill token
  int first_token = sample_first_token();
  LOG(INFO) << "First generated token: " << first_token;
  
  if (first_token == vl_config_.special_tokens.eos_token_id) {
    return "";
  }
  
  // Decode loop
  std::vector<int32_t> generated;
  generated.push_back(static_cast<int32_t>(first_token));
  
  int next_token = -1;
  
  for (int i = 1; i < max_tokens; ++i) {
    // OPTIMIZED: Embed token directly into decode_input buffer (avoids D2D copy)
    embedding_to_decode_input(generated.back());
    
    // Decode step - position is prefill_seq_len + number of generated tokens - 1
    // (because the first generated token is at position prefill_seq_len - 1's output)
    int pos = prefill_seq_len_ + static_cast<int>(generated.size()) - 1;
    status = decode_step_optimized(pos, next_token);
    if (!status) {
      LOG(ERROR) << "Decode step failed at pos=" << pos;
      break;
    }
    
    if (next_token == vl_config_.special_tokens.eos_token_id) {
      break;
    }
    
    generated.push_back(static_cast<int32_t>(next_token));
  }
  
  LOG(INFO) << "Total generated tokens: " << generated.size();
  
  // Decode tokens to text using base class method
  return Model::decode(generated);
}

// ============================================================================
// Base Model Methods
// ============================================================================

base::Status Qwen3VLModel::predict(const tensor::Tensor& input, 
                                    const tensor::Tensor& pos_tensor,
                                    bool is_prompt, int& next) const {
  return forward(input, pos_tensor, next);
}

base::Status Qwen3VLModel::forward(const tensor::Tensor& input,
                                    const tensor::Tensor& pos_tensor,
                                    int& next) const {
  // Standard forward pass (for decode)
  // TODO: Implement
  return base::error::Success();
}

op::EmbeddingOutput Qwen3VLModel::embedding(const std::vector<int>& tokens) const {
  // Get embeddings from token ids - similar to Qwen3Model::embedding
  auto input_tokens = get_buffer(ModelBufferType::kInputTokens);
  auto input_embeddings = get_buffer(ModelBufferType::kInputEmbeddings);
  
  if (input_tokens.size() != tokens.size()) {
    input_tokens.reshape({static_cast<int32_t>(tokens.size())});
    input_embeddings.reshape({static_cast<int32_t>(tokens.size()), config_->dim_});
  }
  
  for (int32_t i = 0; i < tokens.size(); ++i) {
    input_tokens.index<int32_t>(i) = tokens.at(i);
  }
  
  auto input_token_num = tensor::Tensor(base::DataType::kDataTypeInt32, 
                                         static_cast<int32_t>(tokens.size()));
  
  if (qwen_layers_->embedding_layer_) {
    STATUS_CHECK(qwen_layers_->embedding_layer_->forward(
        input_tokens, input_token_num, input_embeddings));
  }
  
  op::EmbeddingOutput output(input_tokens, input_embeddings, input_token_num);
  return output;
}

void Qwen3VLModel::embedding_to_decode_input(int token_id) const {
  // OPTIMIZED: Embed single token directly into decode_input buffer
  // This avoids the D2D copy that would otherwise be needed in decode_step
  auto input_tokens = get_buffer(ModelBufferType::kInputTokens);
  auto decode_input = get_buffer(ModelBufferType::kDecodeInput);
  
  // Ensure buffers are sized for single token
  if (input_tokens.size() != 1) {
    input_tokens.reshape({1});
  }
  
  input_tokens.index<int32_t>(0) = token_id;
  
  auto input_token_num = tensor::Tensor(base::DataType::kDataTypeInt32, 1);
  
  if (qwen_layers_->embedding_layer_) {
    // Output directly to decode_input buffer (fixed address for CUDA Graph)
    STATUS_CHECK(qwen_layers_->embedding_layer_->forward(
        input_tokens, input_token_num, decode_input));
  }
}

void Qwen3VLModel::enable_cuda_graph(bool enable) {
  if (cuda_config_) {
    cuda_config_->use_cuda_graph = enable;
    if (enable && !cuda_config_->graph_context) {
      // Create CUDA Graph context
      cuda_config_->graph_context = std::make_unique<base::CudaGraphContext>();
      cuda_config_->graph_context->needs_recapture = true;
      LOG(INFO) << "Created CUDA Graph context for Qwen3-VL";
    }
  }
}

// ============================================================================
// Batched Operations for Prefill Optimization
// ============================================================================

void Qwen3VLModel::batched_attention_rms(int32_t layer_idx, const tensor::Tensor& input,
                                          const tensor::Tensor& output, int32_t seq_len) const {
  CHECK(qwen_layers_ != nullptr);
  const auto& rmsnorm_layer = qwen_layers_->rmsnorm_layers_.at(layer_idx);
  
  // Use layer forward call instead of direct kernel call
  STATUS_CHECK(rmsnorm_layer->forward(input, output));
}

void Qwen3VLModel::batched_attention_qkv(int32_t layer_idx, const tensor::Tensor& rms_out,
                                          const tensor::Tensor& query_out, 
                                          const tensor::Tensor& key_out, 
                                          const tensor::Tensor& value_out,
                                          int32_t seq_len, int32_t start_pos) const {
  CHECK(qwen_layers_ != nullptr);
  
  auto alloc = base::CUDADeviceAllocatorFactory::get_instance();
  base::DataType activation_dtype = rms_out.data_type();
  size_t elem_size = (activation_dtype == base::DataType::kDataTypeFp16) 
      ? sizeof(uint16_t) : sizeof(float);
  
  // Batched Q projection
  const auto& wq_layer = qwen_layers_->wq_layers_.at(layer_idx);
  auto wq_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(wq_layer);
  CHECK_NE(wq_matmul, nullptr) << "WQ layer is not a MatmulLayer";
  
  // Use cuBLAS for batched matrix multiplication
  const half alpha = __float2half(1.0f);
  const half beta = __float2half(0.0f);
  
  // Q: [seq_len, dim] = [seq_len, dim] @ [dim, dim]^T
  cublasHgemm(cuda_config_->cublas_handle,
              CUBLAS_OP_T, CUBLAS_OP_N,
              config_->dim_, seq_len, config_->dim_,
              &alpha,
              wq_matmul->get_weight(0).ptr<half>(), config_->dim_,
              rms_out.ptr<half>(), config_->dim_,
              &beta,
              const_cast<half*>(query_out.ptr<half>()), config_->dim_);
  
  // K: [seq_len, kv_dim] = [seq_len, dim] @ [dim, kv_dim]^T
  const auto& wk_layer = qwen_layers_->wk_layers_.at(layer_idx);
  auto wk_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(wk_layer);
  CHECK_NE(wk_matmul, nullptr) << "WK layer is not a MatmulLayer";
  
  cublasHgemm(cuda_config_->cublas_handle,
              CUBLAS_OP_T, CUBLAS_OP_N,
              config_->kv_dim_, seq_len, config_->dim_,
              &alpha,
              wk_matmul->get_weight(0).ptr<half>(), config_->dim_,
              rms_out.ptr<half>(), config_->dim_,
              &beta,
              const_cast<half*>(key_out.ptr<half>()), config_->kv_dim_);
  
  // V: [seq_len, kv_dim] = [seq_len, dim] @ [dim, kv_dim]^T
  const auto& wv_layer = qwen_layers_->wv_layers_.at(layer_idx);
  auto wv_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(wv_layer);
  CHECK_NE(wv_matmul, nullptr) << "WV layer is not a MatmulLayer";
  
  cublasHgemm(cuda_config_->cublas_handle,
              CUBLAS_OP_T, CUBLAS_OP_N,
              config_->kv_dim_, seq_len, config_->dim_,
              &alpha,
              wv_matmul->get_weight(0).ptr<half>(), config_->dim_,
              rms_out.ptr<half>(), config_->dim_,
              &beta,
              const_cast<half*>(value_out.ptr<half>()), config_->kv_dim_);
  
  // Apply q_norm and k_norm (per-head RMSNorm)
  // q_norm is at index: layer_idx + 2 * layer_num + 1
  // k_norm is at index: layer_idx + 3 * layer_num + 1
  const auto& q_norm_layer = qwen_layers_->rmsnorm_layers_.at(layer_idx + 2 * config_->layer_num_ + 1);
  const auto& k_norm_layer = qwen_layers_->rmsnorm_layers_.at(layer_idx + 3 * config_->layer_num_ + 1);
  
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
  
  const auto& q_weight = std::dynamic_pointer_cast<op::RmsNormLayer>(q_norm_layer)->get_weight(0);
  const auto& k_weight = std::dynamic_pointer_cast<op::RmsNormLayer>(k_norm_layer)->get_weight(0);
  
  qwen_layers_->rmsnorm_dim_layer_->forward(
      q_reshaped, q_weight, q_reshaped, config_->head_size_);
  qwen_layers_->rmsnorm_dim_layer_->forward(
      k_reshaped, k_weight, k_reshaped, config_->head_size_);
  
  // Apply batched M-RoPE using GPU-resident position arrays
  const auto& section = vl_config_.text.mrope_section;
  int32_t section0 = section[0];  // 24 pairs for temporal
  int32_t section1 = section[1];  // 20 pairs for height  
  int32_t section2 = section[2];  // 20 pairs for width
  
  // Use batched M-RoPE layer (all tokens in one kernel launch)
  qwen_layers_->batched_mrope_layer_->forward(
      seq_len, config_->dim_, config_->kv_dim_, config_->head_size_,
      section0, section1, section2,
      mrope_pos_t_gpu_ + start_pos, mrope_pos_h_gpu_ + start_pos, mrope_pos_w_gpu_ + start_pos,
      query_out, key_out,
      get_buffer(ModelBufferType::kSinCache),
      get_buffer(ModelBufferType::kCosCache));
  
  // OPTIMIZED: Use fused layer to update both K and V caches in a single launch
  // Replaces 2 separate cudaMemcpyAsync calls with one kernel
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);
  
  qwen_layers_->fused_kv_cache_update_layer_->forward(
      key_out, value_out,
      key_cache, val_cache,
      layer_idx,
      start_pos,
      seq_len,
      config_->kv_dim_,
      config_->seq_len_);
}

void Qwen3VLModel::batched_attention_mha(int32_t layer_idx, const tensor::Tensor& query,
                                          tensor::Tensor& mha_out, 
                                          int32_t seq_len, int32_t start_pos) const {
  CHECK(qwen_layers_ != nullptr);
  
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);
  
  // Flash Attention outputs to query buffer (which is no longer needed after QKV)
  // This avoids an extra copy since we can directly use query as attention output
  qwen_layers_->flash_attention_prefill_layer_->forward(
      start_pos, seq_len,
      config_->head_num_, config_->kv_head_num_,
      config_->head_size_, config_->kv_mul_, layer_idx,
      config_->seq_len_, config_->kv_dim_,
      query, const_cast<tensor::Tensor&>(query), key_cache, val_cache);
  
  // WO projection directly to mha_out (final output)
  const auto& wo_layer = qwen_layers_->wo_layers_.at(layer_idx);
  auto wo_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(wo_layer);
  CHECK_NE(wo_matmul, nullptr) << "WO layer is not a MatmulLayer";
  
  const half alpha = __float2half(1.0f);
  const half beta = __float2half(0.0f);
  
  // WO: [seq_len, dim] = attention_out @ WO^T
  // Input: query (now contains attention output), Output: mha_out
  cublasHgemm(cuda_config_->cublas_handle,
              CUBLAS_OP_T, CUBLAS_OP_N,
              config_->dim_, seq_len, config_->dim_,
              &alpha,
              wo_matmul->get_weight(0).ptr<half>(), config_->dim_,
              query.ptr<half>(), config_->dim_,
              &beta,
              mha_out.ptr<half>(), config_->dim_);
  
  // No copy needed - mha_out now contains the final result
}

void Qwen3VLModel::batched_feed_forward(int32_t layer_idx, const tensor::Tensor& input, 
                                         int32_t seq_len) const {
  CHECK(qwen_layers_ != nullptr);
  
  auto alloc = base::CUDADeviceAllocatorFactory::get_instance();
  base::DataType activation_dtype = input.data_type();
  size_t elem_size = (activation_dtype == base::DataType::kDataTypeFp16) 
      ? sizeof(uint16_t) : sizeof(float);
  
  // FFN RMSNorm - use layer forward call
  const auto& ffn_rmsnorm = qwen_layers_->rmsnorm_layers_.at(layer_idx + config_->layer_num_);
  
  tensor::Tensor ffn_norm_out(activation_dtype, seq_len, config_->dim_, true, alloc);
  STATUS_CHECK(ffn_rmsnorm->forward(input, ffn_norm_out));
  
  // Batched W1 (gate) and W3 (up)
  const auto& w1_layer = qwen_layers_->w1_layers_.at(layer_idx);
  const auto& w3_layer = qwen_layers_->w3_layers_.at(layer_idx);
  
  auto w1_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(w1_layer);
  auto w3_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(w3_layer);
  CHECK_NE(w1_matmul, nullptr) << "W1 layer is not a MatmulLayer";
  CHECK_NE(w3_matmul, nullptr) << "W3 layer is not a MatmulLayer";
  
  int32_t hidden_dim = config_->hidden_dim_;
  tensor::Tensor w1_out(activation_dtype, seq_len, hidden_dim, true, alloc);
  tensor::Tensor w3_out(activation_dtype, seq_len, hidden_dim, true, alloc);
  
  const half alpha = __float2half(1.0f);
  const half beta = __float2half(0.0f);
  
  // W1: [seq_len, hidden_dim] = [seq_len, dim] @ [dim, hidden_dim]^T
  cublasHgemm(cuda_config_->cublas_handle,
              CUBLAS_OP_T, CUBLAS_OP_N,
              hidden_dim, seq_len, config_->dim_,
              &alpha,
              w1_matmul->get_weight(0).ptr<half>(), config_->dim_,
              ffn_norm_out.ptr<half>(), config_->dim_,
              &beta,
              w1_out.ptr<half>(), hidden_dim);
  
  // W3: [seq_len, hidden_dim] = [seq_len, dim] @ [dim, hidden_dim]^T
  cublasHgemm(cuda_config_->cublas_handle,
              CUBLAS_OP_T, CUBLAS_OP_N,
              hidden_dim, seq_len, config_->dim_,
              &alpha,
              w3_matmul->get_weight(0).ptr<half>(), config_->dim_,
              ffn_norm_out.ptr<half>(), config_->dim_,
              &beta,
              w3_out.ptr<half>(), hidden_dim);
  
  // Batched SwiGLU via batched_swiglu_layer_: w1_out = silu(w1_out) * w3_out
  STATUS_CHECK(qwen_layers_->batched_swiglu_layer_->forward(w1_out, w3_out, w1_out));
  
  // Batched W2 (down)
  const auto& w2_layer = qwen_layers_->w2_layers_.at(layer_idx);
  auto w2_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(w2_layer);
  CHECK_NE(w2_matmul, nullptr) << "W2 layer is not a MatmulLayer";
  
  tensor::Tensor w2_out(activation_dtype, seq_len, config_->dim_, true, alloc);
  
  // W2: [seq_len, dim] = [seq_len, hidden_dim] @ [hidden_dim, dim]^T
  cublasHgemm(cuda_config_->cublas_handle,
              CUBLAS_OP_T, CUBLAS_OP_N,
              config_->dim_, seq_len, hidden_dim,
              &alpha,
              w2_matmul->get_weight(0).ptr<half>(), hidden_dim,
              w1_out.ptr<half>(), hidden_dim,
              &beta,
              w2_out.ptr<half>(), config_->dim_);
  
  // Residual add via batched_add_layer_: input = input + w2_out
  STATUS_CHECK(qwen_layers_->batched_add_layer_->forward(input, w2_out, input));
}

void Qwen3VLModel::batched_feed_forward_optimized(
    int32_t layer_idx, 
    const tensor::Tensor& input,
    tensor::Tensor& ffn_norm_out,
    tensor::Tensor& w1_out,
    tensor::Tensor& w3_out,
    tensor::Tensor& w2_out,
    int32_t seq_len) const {
  // Optimized version using pre-allocated buffers to avoid per-layer allocation overhead
  CHECK(qwen_layers_ != nullptr);
  
  // FFN RMSNorm - use layer forward call
  const auto& ffn_rmsnorm = qwen_layers_->rmsnorm_layers_.at(layer_idx + config_->layer_num_);
  
  STATUS_CHECK(ffn_rmsnorm->forward(input, ffn_norm_out));
  
  // Batched W1 (gate) and W3 (up)
  const auto& w1_layer = qwen_layers_->w1_layers_.at(layer_idx);
  const auto& w3_layer = qwen_layers_->w3_layers_.at(layer_idx);
  
  auto w1_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(w1_layer);
  auto w3_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(w3_layer);
  
  int32_t hidden_dim = config_->hidden_dim_;
  
  const half alpha = __float2half(1.0f);
  const half beta = __float2half(0.0f);
  
  // W1: [seq_len, hidden_dim] = [seq_len, dim] @ [dim, hidden_dim]^T
  cublasHgemm(cuda_config_->cublas_handle,
              CUBLAS_OP_T, CUBLAS_OP_N,
              hidden_dim, seq_len, config_->dim_,
              &alpha,
              w1_matmul->get_weight(0).ptr<half>(), config_->dim_,
              ffn_norm_out.ptr<half>(), config_->dim_,
              &beta,
              w1_out.ptr<half>(), hidden_dim);
  
  // W3: [seq_len, hidden_dim] = [seq_len, dim] @ [dim, hidden_dim]^T
  cublasHgemm(cuda_config_->cublas_handle,
              CUBLAS_OP_T, CUBLAS_OP_N,
              hidden_dim, seq_len, config_->dim_,
              &alpha,
              w3_matmul->get_weight(0).ptr<half>(), config_->dim_,
              ffn_norm_out.ptr<half>(), config_->dim_,
              &beta,
              w3_out.ptr<half>(), hidden_dim);
  
  // Batched SwiGLU via batched_swiglu_layer_: w1_out = silu(w1_out) * w3_out
  STATUS_CHECK(qwen_layers_->batched_swiglu_layer_->forward(w1_out, w3_out, w1_out));
  
  // Batched W2 (down)
  const auto& w2_layer = qwen_layers_->w2_layers_.at(layer_idx);
  auto w2_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(w2_layer);
  
  // W2: [seq_len, dim] = [seq_len, hidden_dim] @ [hidden_dim, dim]^T
  cublasHgemm(cuda_config_->cublas_handle,
              CUBLAS_OP_T, CUBLAS_OP_N,
              config_->dim_, seq_len, hidden_dim,
              &alpha,
              w2_matmul->get_weight(0).ptr<half>(), hidden_dim,
              w1_out.ptr<half>(), hidden_dim,
              &beta,
              w2_out.ptr<half>(), config_->dim_);
  
  // Residual add via batched_add_layer_: input = input + w2_out
  STATUS_CHECK(qwen_layers_->batched_add_layer_->forward(input, w2_out, input));
}

bool Qwen3VLModel::is_cuda_graph_enabled() const {
  return cuda_config_ && cuda_config_->use_cuda_graph;
}

void Qwen3VLModel::invalidate_cuda_graph() {
  if (cuda_config_) {
    cuda_config_->invalidate_graph();
  }
}

void Qwen3VLModel::clear_kv_cache() {
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  tensor::Tensor value_cache = get_buffer(ModelBufferType::kValueCache);
  
  size_t elem_size = sizeof(uint16_t);  // FP16
  
  if (device_type_ == base::DeviceType::kDeviceCUDA && cuda_config_) {
    cudaMemsetAsync(const_cast<void*>(key_cache.get_buffer()->ptr()), 0, 
                    key_cache.size() * elem_size, cuda_config_->stream);
    cudaMemsetAsync(const_cast<void*>(value_cache.get_buffer()->ptr()), 0, 
                    value_cache.size() * elem_size, cuda_config_->stream);
    cudaStreamSynchronize(cuda_config_->stream);
  }
}

void Qwen3VLModel::set_attention_type(base::AttentionType type) {
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

#endif  // QWEN3_VL_SUPPORT
