#include "op/fused_ffn.h"
#include <cuda_runtime_api.h>
#include "kernels/cuda/fused_ffn_kernel.cuh"

namespace op {

FusedFFNLayer::FusedFFNLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerSwiGLU, "FusedFFN") {
  reset_input_size(3);  // input, W1, W3
  reset_output_size(1); // activated output
}

FusedFFNLayer::FusedFFNLayer(base::DeviceType device_type, int32_t dim, 
                             int32_t hidden_dim, bool use_fp16, bool use_mixed)
    : Layer(device_type, LayerType::kLayerSwiGLU, "FusedFFN"),
      dim_(dim),
      hidden_dim_(hidden_dim),
      use_fp16_(use_fp16),
      use_mixed_(use_mixed) {
  reset_input_size(3);  // input, W1, W3
  reset_output_size(1); // activated output
}

base::Status FusedFFNLayer::check() const {
  if (dim_ <= 0 || hidden_dim_ <= 0) {
    return base::error::InvalidArgument("Invalid FFN dimensions");
  }
  return base::error::Success();
}

base::Status FusedFFNLayer::forward() {
  auto status = check();
  if (!status) {
    return status;
  }

  const tensor::Tensor& input = get_input(0);
  const tensor::Tensor& w1 = get_input(1);
  const tensor::Tensor& w3 = get_input(2);
  tensor::Tensor& output = get_output(0);

  if (use_fp16_) {
    kernel::fused_gate_up_swiglu_kernel_cu_fp16(
        input, w1, w3, output,
        cuda_config_.get());
  } else if (use_mixed_) {
    kernel::fused_gate_up_swiglu_kernel_cu_mixed(
        input, w1, w3, output,
        cuda_config_.get());
  } else {
    kernel::fused_gate_up_swiglu_kernel_cu(
        input, w1, w3, output,
        cuda_config_.get());
  }

  return base::error::Success();
}

}  // namespace op
