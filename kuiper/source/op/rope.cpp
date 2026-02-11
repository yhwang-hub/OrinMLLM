#include "op/rope.h"
#include <cmath>
#include "kernels/cpu/rope_kernel.h"
#include "kernels/kernels_interface.h"
namespace op {
RoPELayer::RoPELayer(base::DeviceType device_type, int32_t dim, int32_t kv_dim, int32_t head_size)
    : Layer(device_type, LayerType::kLayerRoPe, "RoPe"),
      dim_(dim),
      kv_dim_(kv_dim),
      head_size_(head_size) {
  reset_input_size(5);
  reset_output_size(1);
}

base::Status RoPELayer::forward() {
  base::Status status = check();
  if (!status) {
    return status;
  }

  tensor::Tensor input_q = this->get_input(0);
  tensor::Tensor input_k = this->get_input(1);
  tensor::Tensor input_pos = this->get_input(2);

  tensor::Tensor sin_cache = this->get_input(3);
  tensor::Tensor cos_cache = this->get_input(4);

  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK(cuda_config_ != nullptr);
  }
  kernel::get_rope_kernel(device_type_)(dim_, kv_dim_, head_size_, input_q, input_k, input_pos,
                                        sin_cache, cos_cache,
                                        cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

base::Status RoPELayer::check() const {
  // pos tensor
  auto status = check_tensor_with_dim(get_input(2), base::DeviceType::kDeviceCPU,
                                      base::DataType::kDataTypeInt32, 1);
  if (!status) {
    LOG(ERROR) << "The input tensor 2 error in the add layer.";
    return status;
  }

  // Allow both FP32 and FP16 for Q and K tensors (for pure FP16 compute path)
  auto input_k_dtype = get_input(1).data_type();
  if (input_k_dtype != data_type_ && input_k_dtype != base::DataType::kDataTypeFp16) {
    LOG(ERROR) << "The input tensor 1 error in the add layer.";
    return base::error::InvalidArgument("Input K must be FP32 or FP16");
  }
  status = check_tensor_with_dim(get_input(1), device_type_, input_k_dtype, kv_dim_);
  if (!status) {
    LOG(ERROR) << "The input tensor 1 error in the add layer.";
    return status;
  }

  auto input_q_dtype = get_input(0).data_type();
  if (input_q_dtype != data_type_ && input_q_dtype != base::DataType::kDataTypeFp16) {
    LOG(ERROR) << "The input tensor 0 error in the add layer.";
    return base::error::InvalidArgument("Input Q must be FP32 or FP16");
  }
  status = check_tensor_with_dim(get_input(0), device_type_, input_q_dtype, dim_);
  if (!status) {
    LOG(ERROR) << "The input tensor 0 error in the add layer.";
    return status;
  }
  return base::error::Success();
}

}  // namespace op