#include "op/add.h"
#include "kernels/kernels_interface.h"
namespace op {
VecAddLayer::VecAddLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerAdd, "Add") {
  reset_input_size(2);
  reset_output_size(1);
}

base::Status VecAddLayer::check() const {
  tensor::Tensor input1 = this->get_input(0);
  tensor::Tensor input2 = this->get_input(1);
  int32_t size = input1.size();
  base::Status status;
  
  // Allow both FP32 and FP16 for inputs and output (for pure FP16 compute path)
  auto input1_dtype = input1.data_type();
  if (input1_dtype != data_type_ && input1_dtype != base::DataType::kDataTypeFp16) {
    LOG(ERROR) << "The input tensor 1 error in the add layer.";
    return base::error::InvalidArgument("Input 1 must be FP32 or FP16");
  }
  status = check_tensor_with_dim(input1, device_type_, input1_dtype, size);
  if (!status) {
    LOG(ERROR) << "The input tensor 1 error in the add layer.";
    return status;
  }

  auto input2_dtype = input2.data_type();
  if (input2_dtype != data_type_ && input2_dtype != base::DataType::kDataTypeFp16) {
    LOG(ERROR) << "The input tensor 2 error in the add layer.";
    return base::error::InvalidArgument("Input 2 must be FP32 or FP16");
  }
  status = check_tensor_with_dim(input2, device_type_, input2_dtype, size);
  if (!status) {
    LOG(ERROR) << "The input tensor 2 error in the add layer.";
    return status;
  }

  auto output_dtype = get_output(0).data_type();
  if (output_dtype != data_type_ && output_dtype != base::DataType::kDataTypeFp16) {
    LOG(ERROR) << "The output tensor error in the add layer.";
    return base::error::InvalidArgument("Output must be FP32 or FP16");
  }
  status = check_tensor_with_dim(get_output(0), device_type_, output_dtype, size);
  if (!status) {
    LOG(ERROR) << "The output tensor error in the add layer.";
    return status;
  }
  return base::error::Success();
}

base::Status VecAddLayer::forward() {
  auto status = this->check();
  if (!status) {
    return status;
  }
  auto input1 = this->get_input(0);
  auto input2 = this->get_input(1);
  auto output = this->get_output(0);
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK(cuda_config_ != nullptr);
  }
  kernel::get_add_kernel(device_type_)(input1, input2, output,
                                       cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

}  // namespace op