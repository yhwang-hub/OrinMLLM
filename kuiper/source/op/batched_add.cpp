#include "op/batched_add.h"
#include "kernels/kernels_interface.h"
#include "kernels/cuda/swiglu_kernel.cuh"
#include "kernels/cuda/add_kernel.cuh"

namespace op {

// ==================== BatchedAddLayer ====================

BatchedAddLayer::BatchedAddLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerAdd, "BatchedAdd") {
  reset_input_size(2);
  reset_output_size(1);
}

base::Status BatchedAddLayer::check() const {
  const tensor::Tensor& input1 = this->get_input(0);
  const tensor::Tensor& input2 = this->get_input(1);
  const tensor::Tensor& output = this->get_output(0);
  
  // Check non-empty
  if (input1.is_empty() || input2.is_empty() || output.is_empty()) {
    LOG(ERROR) << "BatchedAddLayer: Empty tensor";
    return base::error::InvalidArgument("Empty tensor in batched add layer");
  }
  
  // Check size match (element-wise operation)
  if (input1.size() != input2.size() || input1.size() != output.size()) {
    LOG(ERROR) << "BatchedAddLayer: Size mismatch: " << input1.size() 
               << " vs " << input2.size() << " vs " << output.size();
    return base::error::InvalidArgument("Size mismatch in batched add layer");
  }
  
  // Check device
  if (input1.device_type() != device_type_ || 
      input2.device_type() != device_type_ ||
      output.device_type() != device_type_) {
    LOG(ERROR) << "BatchedAddLayer: Device mismatch";
    return base::error::InvalidArgument("Device mismatch in batched add layer");
  }
  
  // Check data type (allow FP16 and FP32)
  auto dt1 = input1.data_type();
  auto dt2 = input2.data_type();
  auto dto = output.data_type();
  bool valid = (dt1 == base::DataType::kDataTypeFp16 || dt1 == base::DataType::kDataTypeFp32) &&
               (dt2 == base::DataType::kDataTypeFp16 || dt2 == base::DataType::kDataTypeFp32) &&
               (dto == base::DataType::kDataTypeFp16 || dto == base::DataType::kDataTypeFp32);
  if (!valid) {
    LOG(ERROR) << "BatchedAddLayer: Invalid data type";
    return base::error::InvalidArgument("Invalid data type in batched add layer");
  }
  
  return base::error::Success();
}

base::Status BatchedAddLayer::forward() {
  auto status = this->check();
  if (!status) {
    return status;
  }
  
  const tensor::Tensor& input1 = this->get_input(0);
  const tensor::Tensor& input2 = this->get_input(1);
  tensor::Tensor& output = const_cast<tensor::Tensor&>(this->get_output(0));
  
  kernel::get_add_kernel(device_type_)(input1, input2, output,
                                       cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

base::Status BatchedAddLayer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                      const tensor::Tensor& output1) {
  // Direct forward without using internal buffers
  if (input1.is_empty() || input2.is_empty() || output1.is_empty()) {
    return base::error::InvalidArgument("Empty tensor in batched add forward");
  }
  if (input1.size() != input2.size() || input1.size() != output1.size()) {
    return base::error::InvalidArgument("Size mismatch in batched add forward");
  }
  
  kernel::get_add_kernel(device_type_)(input1, input2, output1,
                                       cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

base::Status BatchedAddLayer::forward_raw(half* a, const half* b, half* output, int n) {
  if (a == nullptr || b == nullptr || output == nullptr || n <= 0) {
    return base::error::InvalidArgument("Invalid arguments in forward_raw");
  }
  
  kernel::add_cu(a, b, output, n, cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

// ==================== BatchedSwiGLULayer ====================

BatchedSwiGLULayer::BatchedSwiGLULayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerSwiGLU, "BatchedSwiGLU") {
  reset_input_size(2);
  reset_output_size(1);
}

base::Status BatchedSwiGLULayer::check() const {
  const tensor::Tensor& input1 = this->get_input(0);
  const tensor::Tensor& input2 = this->get_input(1);
  const tensor::Tensor& output = this->get_output(0);
  
  // Check non-empty
  if (input1.is_empty() || input2.is_empty() || output.is_empty()) {
    LOG(ERROR) << "BatchedSwiGLULayer: Empty tensor";
    return base::error::InvalidArgument("Empty tensor in batched swiglu layer");
  }
  
  // Check size match (element-wise operation)
  if (input1.size() != input2.size() || input1.size() != output.size()) {
    LOG(ERROR) << "BatchedSwiGLULayer: Size mismatch";
    return base::error::InvalidArgument("Size mismatch in batched swiglu layer");
  }
  
  // Check device
  if (input1.device_type() != device_type_ || 
      input2.device_type() != device_type_ ||
      output.device_type() != device_type_) {
    LOG(ERROR) << "BatchedSwiGLULayer: Device mismatch";
    return base::error::InvalidArgument("Device mismatch in batched swiglu layer");
  }
  
  return base::error::Success();
}

base::Status BatchedSwiGLULayer::forward() {
  auto status = this->check();
  if (!status) {
    return status;
  }
  
  const tensor::Tensor& input1 = this->get_input(0);
  const tensor::Tensor& input2 = this->get_input(1);
  tensor::Tensor& output = const_cast<tensor::Tensor&>(this->get_output(0));
  
  kernel::swiglu_kernel_cu(input1, input2, output,
                           cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

base::Status BatchedSwiGLULayer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                         const tensor::Tensor& output1) {
  // Direct forward without using internal buffers
  if (input1.is_empty() || input2.is_empty() || output1.is_empty()) {
    return base::error::InvalidArgument("Empty tensor in batched swiglu forward");
  }
  if (input1.size() != input2.size() || input1.size() != output1.size()) {
    return base::error::InvalidArgument("Size mismatch in batched swiglu forward");
  }
  
  kernel::swiglu_kernel_cu(input1, input2, output1,
                           cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

// ==================== BiasAddLayer ====================

BiasAddLayer::BiasAddLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerAdd, "BiasAdd") {
  reset_input_size(2);  // input [rows, cols], bias [cols]
  reset_output_size(1); // output [rows, cols]
}

void BiasAddLayer::set_dims(int32_t rows, int32_t cols) {
  rows_ = rows;
  cols_ = cols;
}

base::Status BiasAddLayer::check() const {
  if (rows_ <= 0 || cols_ <= 0) {
    return base::error::InvalidArgument("Invalid dimensions for bias add layer");
  }
  return base::error::Success();
}

base::Status BiasAddLayer::forward() {
  auto status = this->check();
  if (!status) {
    return status;
  }
  
  const tensor::Tensor& input = this->get_input(0);
  const tensor::Tensor& bias = this->get_input(1);
  tensor::Tensor& output = const_cast<tensor::Tensor&>(this->get_output(0));
  
  // FP16 path uses broadcast kernel
  if (input.data_type() == base::DataType::kDataTypeFp16) {
    kernel::broadcast_add_bias_fp16_cu(input, bias, output, rows_, cols_,
                                       cuda_config_ ? cuda_config_->stream : nullptr);
  } else {
    // FP32 path: add bias to each row
    for (int i = 0; i < rows_; ++i) {
      tensor::Tensor input_row(base::DataType::kDataTypeFp32, cols_, false, nullptr,
                               const_cast<float*>(input.ptr<float>(i * cols_)));
      input_row.set_device_type(device_type_);
      tensor::Tensor output_row(base::DataType::kDataTypeFp32, cols_, false, nullptr,
                                const_cast<float*>(output.ptr<float>(i * cols_)));
      output_row.set_device_type(device_type_);
      kernel::get_add_kernel(device_type_)(input_row, bias, output_row,
                                           cuda_config_ ? cuda_config_->stream : nullptr);
    }
  }
  
  return base::error::Success();
}

}  // namespace op
