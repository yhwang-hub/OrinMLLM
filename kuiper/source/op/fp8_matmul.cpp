//
// FP8 E4M3 Block-Quantized Matrix Multiplication Layer Implementation
//

#include "op/fp8_matmul.h"
#include "kernels/cuda/fp8_gemm_kernel.cuh"
#include "base/alloc.h"
#include <glog/logging.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>

namespace op {

FP8MatmulLayer::FP8MatmulLayer(base::DeviceType device_type,
                               int32_t in_features,
                               int32_t out_features,
                               int32_t block_size)
    : Layer(device_type, LayerType::kLayerMatmul, "FP8Matmul"),
      in_features_(in_features),
      out_features_(out_features),
      block_size_(block_size) {
  reset_input_size(1);
  reset_output_size(1);
}

base::Status FP8MatmulLayer::check() const {
  if (in_features_ <= 0 || out_features_ <= 0) {
    return base::error::InternalError("Invalid dimensions for FP8 matmul");
  }
  if (fp8_weight_.is_empty()) {
    return base::error::InternalError("FP8 weights not set");
  }
  if (scale_inv_.is_empty()) {
    return base::error::InternalError("FP8 scale_inv not set");
  }
  return base::error::Success();
}

void FP8MatmulLayer::set_fp8_weights(const void* fp8_weight_ptr,
                                      const void* scale_inv_ptr,
                                      int32_t scale_rows,
                                      int32_t scale_cols,
                                      base::DeviceType src_device) {
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  scale_rows_ = scale_rows;
  scale_cols_ = scale_cols;

  // Load FP8 weight [out_features, in_features] as raw bytes (uint8)
  int32_t weight_size = out_features_ * in_features_;
  fp8_weight_ = tensor::Tensor(base::DataType::kDataTypeInt8, weight_size, true, alloc);
  std::memcpy(fp8_weight_.ptr<void>(), fp8_weight_ptr, weight_size);

  // Load scale_inv [scale_rows, scale_cols] as FP16
  int32_t scale_size = scale_rows * scale_cols;
  scale_inv_ = tensor::Tensor(base::DataType::kDataTypeFp16, scale_size, true, alloc);
  std::memcpy(scale_inv_.ptr<void>(), scale_inv_ptr, scale_size * sizeof(uint16_t));
}

void FP8MatmulLayer::to_cuda() {
  if (device_type_ != base::DeviceType::kDeviceCUDA) {
    return;
  }

  auto cuda_alloc = base::CUDADeviceAllocatorFactory::get_instance();

  if (!fp8_weight_.is_empty()) {
    int32_t weight_size = out_features_ * in_features_;
    tensor::Tensor gpu_weight(base::DataType::kDataTypeInt8, weight_size, true, cuda_alloc);
    cudaMemcpy(gpu_weight.ptr<void>(), fp8_weight_.ptr<void>(),
               weight_size, cudaMemcpyHostToDevice);
    fp8_weight_ = std::move(gpu_weight);
  }

  if (!scale_inv_.is_empty()) {
    int32_t scale_size = scale_rows_ * scale_cols_;
    tensor::Tensor gpu_scale(base::DataType::kDataTypeFp16, scale_size, true, cuda_alloc);
    cudaMemcpy(gpu_scale.ptr<void>(), scale_inv_.ptr<void>(),
               scale_size * sizeof(uint16_t), cudaMemcpyHostToDevice);
    scale_inv_ = std::move(gpu_scale);
  }

  // Initialize shared dequant buffer for prefill GEMM (sized for this layer)
  size_t weight_elements = (size_t)out_features_ * in_features_;
  kernel::fp8_init_dequant_buffer(weight_elements);
}

base::Status FP8MatmulLayer::forward(const tensor::Tensor& input, const tensor::Tensor& output) {
  if (input.is_empty() || output.is_empty()) {
    return base::error::InvalidArgument("Empty tensors in FP8 forward");
  }

  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK(!fp8_weight_.is_empty()) << "FP8 weight not on GPU. Call to_cuda() first.";
    CHECK(!scale_inv_.is_empty()) << "FP8 scale_inv not on GPU. Call to_cuda() first.";

    int batch_size = input.size() / in_features_;

    cudaStream_t stream = nullptr;
    cublasHandle_t cublas_handle = nullptr;
    if (cuda_config_) {
      stream = cuda_config_->stream;
      cublas_handle = cuda_config_->cublas_handle;
    }

    kernel::fp8_gemm_cu(
        fp8_weight_.ptr<uint8_t>(),
        scale_inv_.ptr<half>(),
        input.ptr<half>(),
        const_cast<half*>(output.ptr<half>()),
        batch_size,
        out_features_,
        in_features_,
        block_size_,
        scale_cols_,
        cublas_handle,
        stream);
  } else {
    return base::error::InternalError("FP8 only supports CUDA device");
  }

  return base::error::Success();
}

base::Status FP8MatmulLayer::forward() {
  auto status = check();
  if (!status) {
    LOG(ERROR) << "FP8 check failed: " << status.get_err_msg();
    return status;
  }

  const tensor::Tensor& input = get_input(0);
  tensor::Tensor& output = get_output(0);
  return forward(input, output);
}

}  // namespace op
