#include "op/kv_cache.h"
#include <cuda_runtime_api.h>
#include "kernels/cuda/kv_cache_kernel.cuh"

namespace op {

KVCacheLayer::KVCacheLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerUnknown, "KVCache") {
  reset_input_size(3);  // input, kv_cache, pos
  reset_output_size(0); // in-place update
}

KVCacheLayer::KVCacheLayer(base::DeviceType device_type, int32_t kv_dim, 
                           int32_t seq_len, bool use_fp16)
    : Layer(device_type, LayerType::kLayerUnknown, "KVCache"),
      kv_dim_(kv_dim),
      seq_len_(seq_len),
      use_fp16_(use_fp16) {
  reset_input_size(3);  // input, kv_cache, pos
  reset_output_size(0); // in-place update
}

base::Status KVCacheLayer::check() const {
  if (kv_dim_ <= 0 || seq_len_ <= 0) {
    return base::error::InvalidArgument("Invalid KV cache parameters");
  }
  return base::error::Success();
}

base::Status KVCacheLayer::forward() {
  auto status = check();
  if (!status) {
    return status;
  }

  const tensor::Tensor& input = get_input(0);
  tensor::Tensor& kv_cache = const_cast<tensor::Tensor&>(get_input(1));
  const tensor::Tensor& pos_tensor = get_input(2);

  if (use_fp16_) {
    kernel::copy_to_kv_cache_kernel_fp16(
        reinterpret_cast<half*>(const_cast<uint16_t*>(kv_cache.ptr<uint16_t>())),
        reinterpret_cast<const half*>(input.ptr<uint16_t>()),
        pos_tensor.ptr<int32_t>(),
        kv_dim_,
        layer_index_,
        seq_len_,
        cuda_config_ ? cuda_config_->stream : nullptr);
  } else {
    kernel::copy_to_kv_cache_kernel(
        const_cast<float*>(kv_cache.ptr<float>()),
        input.ptr<float>(),
        pos_tensor.ptr<int32_t>(),
        kv_dim_,
        layer_index_,
        seq_len_,
        cuda_config_ ? cuda_config_->stream : nullptr);
  }

  return base::error::Success();
}

}  // namespace op
