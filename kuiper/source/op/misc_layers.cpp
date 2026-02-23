#include "op/misc_layers.h"
#include "kernels/kernels_interface.h"
#include "kernels/cpu/rope_kernel.h"
#include "kernels/cuda/rope_kernel.cuh"
#include "kernels/cuda/mha_kernel.cuh"
#include "kernels/cuda/matmul_kernel.cuh"
#include "kernels/cuda/flash_attention_kernel.cuh"
#include "kernels/cuda/paged_attention_kernel.cuh"
#include "kernels/cuda/rmsnorm_kernel.cuh"
#include "kernels/cuda/kv_cache_kernel.cuh"
#include "kernels/cuda/fused_kernels.cuh"

namespace op {

// ==================== SinCosCacheLayer ====================

SinCosCacheLayer::SinCosCacheLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerUnknown, "SinCosCache") {
  reset_input_size(0);
  reset_output_size(2);  // sin_cache, cos_cache
}

base::Status SinCosCacheLayer::check() const {
  return base::error::Success();
}

base::Status SinCosCacheLayer::forward() {
  return base::error::InvalidArgument("Use forward(head_size, seq_len, sin_cache, cos_cache)");
}

base::Status SinCosCacheLayer::forward(int32_t head_size, int32_t seq_len,
                                       const tensor::Tensor& sin_cache,
                                       const tensor::Tensor& cos_cache) {
  if (sin_cache.is_empty() || cos_cache.is_empty()) {
    return base::error::InvalidArgument("Empty sin/cos cache tensors");
  }
  
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    kernel::sin_cos_cache_calc_cpu(head_size, seq_len,
                                   const_cast<float*>(sin_cache.ptr<float>()),
                                   const_cast<float*>(cos_cache.ptr<float>()));
  } else {
    kernel::sin_cos_cache_calc_cu(head_size, seq_len, sin_cache, cos_cache,
                                  cuda_config_ ? cuda_config_->stream : nullptr);
  }
  return base::error::Success();
}

// ==================== MHAGpuPosLayer ====================

MHAGpuPosLayer::MHAGpuPosLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerMHA, "MHAGpuPos") {
  reset_input_size(4);   // query, score_storage, key_cache, val_cache
  reset_output_size(1);  // mha_output
}

base::Status MHAGpuPosLayer::check() const {
  return base::error::Success();
}

base::Status MHAGpuPosLayer::forward() {
  return base::error::InvalidArgument("Use forward(...) with parameters");
}

base::Status MHAGpuPosLayer::forward(const int32_t* pos_ptr, int32_t head_num, 
                                     int32_t layer_idx, int32_t seq_len,
                                     int32_t kv_dim, int32_t kv_mul, int32_t head_size,
                                     tensor::Tensor& mha_output, const tensor::Tensor& query,
                                     tensor::Tensor& score_storage, 
                                     const tensor::Tensor& key_cache,
                                     const tensor::Tensor& val_cache) {
  if (pos_ptr == nullptr) {
    return base::error::InvalidArgument("Position pointer is null");
  }
  
  kernel::mha_kernel_cu_gpu_pos(pos_ptr, head_num, layer_idx, seq_len,
                                kv_dim, kv_mul, head_size, mha_output, query,
                                score_storage, key_cache, val_cache,
                                device_type_, cuda_config_.get());
  return base::error::Success();
}

// ==================== BatchedMHALayer ====================

BatchedMHALayer::BatchedMHALayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerMHA, "BatchedMHA") {
  reset_input_size(4);   // query, score_storage, key_cache, val_cache
  reset_output_size(1);  // mha_output
}

base::Status BatchedMHALayer::check() const {
  return base::error::Success();
}

base::Status BatchedMHALayer::forward() {
  return base::error::InvalidArgument("Use forward(...) with parameters");
}

base::Status BatchedMHALayer::forward(int32_t start_pos, int32_t seq_len, int32_t head_num,
                                      int32_t layer_idx, int32_t max_seq_len, int32_t dim,
                                      int32_t kv_dim, int32_t kv_mul, int32_t head_size,
                                      tensor::Tensor& mha_output, const tensor::Tensor& query,
                                      tensor::Tensor& score_storage, 
                                      const tensor::Tensor& key_cache,
                                      const tensor::Tensor& val_cache) {
  kernel::batched_mha_kernel_cu(start_pos, seq_len, head_num, layer_idx,
                                max_seq_len, dim, kv_dim, kv_mul, head_size,
                                mha_output, query, score_storage, key_cache, val_cache,
                                device_type_, cuda_config_.get());
  return base::error::Success();
}

// ==================== BatchedMatmulHelperLayer ====================

BatchedMatmulHelperLayer::BatchedMatmulHelperLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerMatmul, "BatchedMatmulHelper") {
  reset_input_size(2);  // input, weight
  reset_output_size(1); // output
}

base::Status BatchedMatmulHelperLayer::check() const {
  return base::error::Success();
}

base::Status BatchedMatmulHelperLayer::forward() {
  return base::error::InvalidArgument("Use forward(input, weight, output, batch_size, scale)");
}

base::Status BatchedMatmulHelperLayer::forward(const tensor::Tensor& input,
                                               const tensor::Tensor& weight,
                                               const tensor::Tensor& output,
                                               int32_t batch_size, float scale) {
  if (input.is_empty() || weight.is_empty() || output.is_empty()) {
    return base::error::InvalidArgument("Empty tensors in batched matmul");
  }
  if (batch_size <= 0 || scale <= 0) {
    return base::error::InvalidArgument("Invalid batch_size or scale");
  }
  
  // Dispatch based on data types
  if (input.data_type() == base::DataType::kDataTypeFp16 &&
      output.data_type() == base::DataType::kDataTypeFp16 &&
      weight.data_type() == base::DataType::kDataTypeFp16) {
    // Pure FP16 path
    kernel::batched_matmul_kernel_cu_pure_fp16(input, weight, output, batch_size, scale,
                                               cuda_config_.get());
  } else if (weight.data_type() == base::DataType::kDataTypeFp16) {
    // Mixed path: FP32 input x FP16 weight -> FP32 output
    kernel::batched_matmul_kernel_cu_fp16_weight(input, weight, output, batch_size, scale,
                                                 cuda_config_.get());
  } else {
    // Pure FP32 path
    kernel::batched_matmul_kernel_cu(input, weight, output, batch_size, scale,
                                     cuda_config_.get());
  }
  
  return base::error::Success();
}

// ==================== MRoPELayer ====================

MRoPELayer::MRoPELayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerUnknown, "MRoPE") {
  reset_input_size(4);  // query, key, sin_cache, cos_cache
  reset_output_size(0);
}

base::Status MRoPELayer::check() const {
  return base::error::Success();
}

base::Status MRoPELayer::forward() {
  return base::error::InvalidArgument("Use forward(...) with parameters");
}

base::Status MRoPELayer::forward(int32_t pos_t, int32_t pos_h, int32_t pos_w,
                                  int32_t dim, int32_t kv_dim, int32_t head_size,
                                  int32_t section0, int32_t section1, int32_t section2,
                                  const tensor::Tensor& query, const tensor::Tensor& key,
                                  const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache) {
  kernel::mrope_kernel_cu_fp16(pos_t, pos_h, pos_w, dim, kv_dim, head_size,
                                section0, section1, section2,
                                query, key, sin_cache, cos_cache,
                                cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

// ==================== MRoPEGpuPosLayer ====================

MRoPEGpuPosLayer::MRoPEGpuPosLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerUnknown, "MRoPEGpuPos") {
  reset_input_size(4);
  reset_output_size(0);
}

base::Status MRoPEGpuPosLayer::check() const {
  return base::error::Success();
}

base::Status MRoPEGpuPosLayer::forward() {
  return base::error::InvalidArgument("Use forward(...) with parameters");
}

base::Status MRoPEGpuPosLayer::forward(const int32_t* rope_pos_gpu,
                                        int32_t dim, int32_t kv_dim, int32_t head_size,
                                        int32_t section0, int32_t section1, int32_t section2,
                                        const tensor::Tensor& query, const tensor::Tensor& key,
                                        const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache) {
  kernel::mrope_kernel_cu_fp16_gpu_pos(rope_pos_gpu, dim, kv_dim, head_size,
                                       section0, section1, section2,
                                       query, key, sin_cache, cos_cache,
                                       cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

// ==================== BatchedMRoPELayer ====================

BatchedMRoPELayer::BatchedMRoPELayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerUnknown, "BatchedMRoPE") {
  reset_input_size(4);
  reset_output_size(0);
}

base::Status BatchedMRoPELayer::check() const {
  return base::error::Success();
}

base::Status BatchedMRoPELayer::forward() {
  return base::error::InvalidArgument("Use forward(...) with parameters");
}

base::Status BatchedMRoPELayer::forward(int32_t seq_len, int32_t dim, int32_t kv_dim, int32_t head_size,
                                         int32_t section0, int32_t section1, int32_t section2,
                                         const int32_t* pos_t, const int32_t* pos_h, const int32_t* pos_w,
                                         const tensor::Tensor& query, const tensor::Tensor& key,
                                         const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache) {
  kernel::batched_mrope_kernel_cu_fp16(seq_len, dim, kv_dim, head_size,
                                       section0, section1, section2,
                                       pos_t, pos_h, pos_w,
                                       query, key, sin_cache, cos_cache,
                                       cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

// ==================== FusedKVCacheUpdateLayer ====================

FusedKVCacheUpdateLayer::FusedKVCacheUpdateLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerUnknown, "FusedKVCacheUpdate") {
  reset_input_size(4);  // key, value, key_cache, val_cache
  reset_output_size(0);
}

base::Status FusedKVCacheUpdateLayer::check() const {
  return base::error::Success();
}

base::Status FusedKVCacheUpdateLayer::forward() {
  return base::error::InvalidArgument("Use forward(...) with parameters");
}

base::Status FusedKVCacheUpdateLayer::forward(const tensor::Tensor& key, const tensor::Tensor& value,
                                               const tensor::Tensor& key_cache, const tensor::Tensor& val_cache,
                                               int32_t layer_idx, int32_t start_pos, int32_t seq_len,
                                               int32_t kv_dim, int32_t max_seq_len) {
  kernel::fused_kv_cache_update_cu(key, value, 
                                   const_cast<tensor::Tensor&>(key_cache), 
                                   const_cast<tensor::Tensor&>(val_cache),
                                   layer_idx, start_pos, seq_len, kv_dim, max_seq_len,
                                   cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

// ==================== RMSNormDimLayer ====================

RMSNormDimLayer::RMSNormDimLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerRMSNorm, "RMSNormDim") {
  reset_input_size(2);  // input, weight
  reset_output_size(1); // output
}

base::Status RMSNormDimLayer::check() const {
  return base::error::Success();
}

base::Status RMSNormDimLayer::forward() {
  return base::error::InvalidArgument("Use forward(...) with parameters");
}

base::Status RMSNormDimLayer::forward(const tensor::Tensor& input, const tensor::Tensor& weight,
                                       const tensor::Tensor& output, int32_t dim) {
  kernel::rmsnorm_kernel_cu_dim(input, weight, output, dim,
                                cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

// ==================== CopyToKVCacheLayer ====================

CopyToKVCacheLayer::CopyToKVCacheLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerUnknown, "CopyToKVCache") {
  reset_input_size(2);  // kv_cache, kv_data
  reset_output_size(0);
}

base::Status CopyToKVCacheLayer::check() const {
  return base::error::Success();
}

base::Status CopyToKVCacheLayer::forward() {
  return base::error::InvalidArgument("Use forward(...) with parameters");
}

base::Status CopyToKVCacheLayer::forward(const tensor::Tensor& kv_cache, const tensor::Tensor& kv_data,
                                          const int32_t* pos_gpu, int32_t kv_dim, int32_t layer_idx,
                                          int32_t seq_len) {
  kernel::copy_to_kv_cache_kernel_fp16(
      reinterpret_cast<half*>(const_cast<uint16_t*>(kv_cache.ptr<uint16_t>())),
      reinterpret_cast<const half*>(kv_data.ptr<uint16_t>()),
      pos_gpu, kv_dim, layer_idx, seq_len,
      cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

// ==================== FlashAttentionDecodeGpuPosLayer ====================

FlashAttentionDecodeGpuPosLayer::FlashAttentionDecodeGpuPosLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerMHA, "FlashAttentionDecodeGpuPos") {
  reset_input_size(4);  // query, mha_output, key_cache, val_cache
  reset_output_size(0);
}

base::Status FlashAttentionDecodeGpuPosLayer::check() const {
  return base::error::Success();
}

base::Status FlashAttentionDecodeGpuPosLayer::forward() {
  return base::error::InvalidArgument("Use forward(...) with parameters");
}

base::Status FlashAttentionDecodeGpuPosLayer::forward(const int32_t* pos_gpu, 
                                                       int32_t head_num, int32_t kv_head_num,
                                                       int32_t head_size, int32_t kv_mul, int32_t layer_idx,
                                                       int32_t seq_len, int32_t kv_dim,
                                                       const tensor::Tensor& query, const tensor::Tensor& mha_output,
                                                       const tensor::Tensor& key_cache, const tensor::Tensor& val_cache) {
  // Paged attention path
  if (paged_mode_) {
    kernel::paged_flash_attention_decode_fp16_gpu_pos_cu(
        pos_gpu, head_num, kv_head_num, head_size, kv_mul, layer_idx,
        kv_dim, page_size_, max_blocks_per_seq_,
        query, mha_output, key_pool_, value_pool_, block_table_,
        cuda_config_.get());
    return base::error::Success();
  }
  if (attention_type_ == base::AttentionType::kAttentionFlash2) {
    kernel::flash_attention2_decode_fp16_gpu_pos_cu(pos_gpu, head_num, kv_head_num,
                                                    head_size, kv_mul, layer_idx,
                                                    seq_len, kv_dim,
                                                    query, mha_output, key_cache, val_cache,
                                                    cuda_config_.get());
  } else {
    kernel::flash_attention_decode_fp16_gpu_pos_cu(pos_gpu, head_num, kv_head_num,
                                                   head_size, kv_mul, layer_idx,
                                                   seq_len, kv_dim,
                                                   query, mha_output, key_cache, val_cache,
                                                   cuda_config_.get());
  }
  return base::error::Success();
}

}  // namespace op
