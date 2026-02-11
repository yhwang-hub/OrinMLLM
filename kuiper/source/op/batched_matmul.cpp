#include "op/batched_matmul.h"
#include <cuda_runtime_api.h>
#include "kernels/cuda/matmul_kernel.cuh"

namespace op {

BatchedMatmulLayer::BatchedMatmulLayer(base::DeviceType device_type, 
                                       int32_t dim0, int32_t dim1)
    : LayerParam(device_type, LayerType::kLayerMatmul, false, "BatchedMatmul"),
      dim0_(dim0),
      dim1_(dim1) {
  reset_input_size(1);
  reset_output_size(1);
  reset_weight_size(1);
}

base::Status BatchedMatmulLayer::check() const {
  if (dim0_ <= 0 || dim1_ <= 0 || batch_size_ <= 0) {
    return base::error::InvalidArgument("Invalid batched matmul dimensions");
  }
  return base::error::Success();
}

base::Status BatchedMatmulLayer::forward() {
  auto status = check();
  if (!status) {
    return status;
  }

  const tensor::Tensor& input = get_input(0);
  const tensor::Tensor& weight = get_weight(0);
  tensor::Tensor& output = get_output(0);

  // Dispatch based on data types
  if (input.data_type() == base::DataType::kDataTypeFp16 &&
      output.data_type() == base::DataType::kDataTypeFp16 &&
      weight.data_type() == base::DataType::kDataTypeFp16) {
    // Pure FP16 path
    kernel::batched_matmul_kernel_cu_pure_fp16(
        input, weight, output, batch_size_, scale_, cuda_config_.get());
  } else if (weight.data_type() == base::DataType::kDataTypeFp16) {
    // Mixed path: FP32 input x FP16 weight -> FP32 output
    kernel::batched_matmul_kernel_cu_fp16_weight(
        input, weight, output, batch_size_, scale_, cuda_config_.get());
  } else {
    // Pure FP32 path
    kernel::batched_matmul_kernel_cu(
        input, weight, output, batch_size_, scale_, cuda_config_.get());
  }

  return base::error::Success();
}

}  // namespace op
