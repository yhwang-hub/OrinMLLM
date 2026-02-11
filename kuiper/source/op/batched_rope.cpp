#include "op/batched_rope.h"
#include <cuda_runtime_api.h>
#include "kernels/cuda/rope_kernel.cuh"

namespace op {

// ==================== RoPEGpuPosLayer ====================

RoPEGpuPosLayer::RoPEGpuPosLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerRoPe, "RoPEGpuPos") {
  reset_input_size(5);  // query, key, pos, sin_cache, cos_cache
  reset_output_size(0); // in-place modification
}

RoPEGpuPosLayer::RoPEGpuPosLayer(base::DeviceType device_type, int32_t dim, 
                                 int32_t kv_dim, int32_t head_size, bool use_fp16)
    : Layer(device_type, LayerType::kLayerRoPe, "RoPEGpuPos"),
      dim_(dim),
      kv_dim_(kv_dim),
      head_size_(head_size),
      use_fp16_(use_fp16) {
  reset_input_size(5);  // query, key, pos, sin_cache, cos_cache
  reset_output_size(0); // in-place modification
}

base::Status RoPEGpuPosLayer::check() const {
  if (dim_ <= 0 || kv_dim_ <= 0 || head_size_ <= 0) {
    return base::error::InvalidArgument("Invalid RoPE parameters");
  }
  return base::error::Success();
}

base::Status RoPEGpuPosLayer::forward() {
  auto status = check();
  if (!status) {
    return status;
  }

  tensor::Tensor& query = const_cast<tensor::Tensor&>(get_input(0));
  tensor::Tensor& key = const_cast<tensor::Tensor&>(get_input(1));
  const tensor::Tensor& pos_tensor = get_input(2);
  const tensor::Tensor& sin_cache = get_input(3);
  const tensor::Tensor& cos_cache = get_input(4);

  if (use_fp16_) {
    kernel::rope_kernel_cu_fp16_gpu_pos(
        dim_, kv_dim_, head_size_,
        query, key, pos_tensor.ptr<int32_t>(),
        sin_cache, cos_cache,
        cuda_config_ ? cuda_config_->stream : nullptr);
  } else {
    kernel::rope_kernel_cu_gpu_pos(
        dim_, kv_dim_, head_size_,
        query, key, pos_tensor.ptr<int32_t>(),
        sin_cache, cos_cache,
        cuda_config_ ? cuda_config_->stream : nullptr);
  }

  return base::error::Success();
}

// ==================== BatchedRoPELayer ====================

BatchedRoPELayer::BatchedRoPELayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerRoPe, "BatchedRoPE") {
  reset_input_size(4);  // query, key, sin_cache, cos_cache
  reset_output_size(0); // in-place modification
}

BatchedRoPELayer::BatchedRoPELayer(base::DeviceType device_type, int32_t dim, 
                                   int32_t kv_dim, int32_t head_size,
                                   int32_t head_num, int32_t kv_head_num)
    : Layer(device_type, LayerType::kLayerRoPe, "BatchedRoPE"),
      dim_(dim),
      kv_dim_(kv_dim),
      head_size_(head_size),
      head_num_(head_num),
      kv_head_num_(kv_head_num) {
  reset_input_size(4);  // query, key, sin_cache, cos_cache
  reset_output_size(0); // in-place modification
}

base::Status BatchedRoPELayer::check() const {
  if (dim_ <= 0 || kv_dim_ <= 0 || head_size_ <= 0) {
    return base::error::InvalidArgument("Invalid batched RoPE parameters");
  }
  return base::error::Success();
}

base::Status BatchedRoPELayer::forward() {
  auto status = check();
  if (!status) {
    return status;
  }

  tensor::Tensor& query = const_cast<tensor::Tensor&>(get_input(0));
  tensor::Tensor& key = const_cast<tensor::Tensor&>(get_input(1));
  const tensor::Tensor& sin_cache = get_input(2);
  const tensor::Tensor& cos_cache = get_input(3);

  kernel::batched_rope_kernel_cu(
      start_pos_, seq_len_, dim_, kv_dim_, head_size_,
      query, key, sin_cache, cos_cache,
      cuda_config_ ? cuda_config_->stream : nullptr);

  return base::error::Success();
}

}  // namespace op
