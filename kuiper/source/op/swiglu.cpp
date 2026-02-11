#include "op/swiglu.h"
#include "kernels/cpu/swiglu_kernel.h"
#include "kernels/kernels_interface.h"
#include "op/layer.h"
namespace op {
SwiGLULayer::SwiGLULayer(base::DeviceType device_type, int32_t hidden_dim)
    : Layer(device_type, op::LayerType::kLayerSwiGLU, "SwiGLU"), hidden_dim_(hidden_dim) {
  reset_input_size(2);
  reset_output_size(1);
}

base::Status SwiGLULayer::check() const {
  base::Status status;
  const int32_t input_tensor_num = 2;
  
  // Allow both FP32 and FP16 input (for pure FP16 compute path)
  auto input_dtype = get_input(0).data_type();
  if (input_dtype != data_type_ && input_dtype != base::DataType::kDataTypeFp16) {
    LOG(ERROR) << "The input tensor 0 error in the swiglu layer.";
    return base::error::InvalidArgument("Input must be FP32 or FP16");
  }
  
  for (int32_t i = 0; i < input_tensor_num; ++i) {
    status = check_tensor_with_dim(get_input(i), device_type_, input_dtype, hidden_dim_);
    if (!status) {
      LOG(ERROR) << "The input tensor " << std::to_string(i) << " error in the swiglu layer.";
      return status;
    }
  }

  // Allow both FP32 and FP16 output
  auto output_dtype = get_output(0).data_type();
  if (output_dtype != data_type_ && output_dtype != base::DataType::kDataTypeFp16) {
    LOG(ERROR) << "The output tensor error in the swiglu layer.";
    return base::error::InvalidArgument("Output must be FP32 or FP16");
  }
  status = check_tensor_with_dim(get_output(0), device_type_, output_dtype, hidden_dim_);
  if (!status) {
    LOG(ERROR) << "The output tensor error in the swiglu layer.";
    return status;
  }
  return base::error::Success();
}

base::Status SwiGLULayer::forward() {
  auto status = check();
  if (!status) {
    return status;
  }
  auto input1 = this->get_input(0);
  auto input2 = this->get_input(1);
  auto output = this->get_output(0);
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK(cuda_config_ != nullptr);
  }
  kernel::get_swiglu_kernel(device_type_)(input1, input2, output,
                                          cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

}  // namespace op
