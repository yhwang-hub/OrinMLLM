//
// AWQ INT4 Quantized Matrix Multiplication Layer Implementation
//

#include "op/awq_matmul.h"
#include "base/alloc.h"
#include "kernels/cuda/awq_gemm_tensorcore.cuh"
#include <glog/logging.h>
#include <cuda_runtime.h>

namespace op {

AWQMatmulLayer::AWQMatmulLayer(base::DeviceType device_type, 
                               int32_t in_features, 
                               int32_t out_features,
                               int32_t group_size)
    : Layer(device_type, LayerType::kLayerMatmul, "AWQMatmul"),
      in_features_(in_features),
      out_features_(out_features),
      group_size_(group_size) {
  // Initialize tensors with correct sizes
  reset_input_size(1);
  reset_output_size(1);
}

base::Status AWQMatmulLayer::check() const {
  if (in_features_ <= 0 || out_features_ <= 0) {
    return base::error::InternalError("Invalid dimensions for AWQ matmul");
  }
  if (qweight_.is_empty() || qzeros_.is_empty() || scales_.is_empty()) {
    return base::error::InternalError("AWQ weights not set");
  }
  return base::error::Success();
}

void AWQMatmulLayer::set_awq_weights(const void* qweight_ptr, 
                                      const void* qzeros_ptr,
                                      const void* scales_ptr,
                                      base::DeviceType src_device) {
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  
  // Calculate dimensions
  int32_t packed_out = out_features_ / 8;  // 8 INT4 per INT32
  int32_t num_groups = in_features_ / group_size_;
  
  // qweight: [in_features, out_features/8] INT32
  int32_t qweight_size = in_features_ * packed_out;
  qweight_ = tensor::Tensor(base::DataType::kDataTypeInt32, qweight_size, true, alloc);
  std::memcpy(qweight_.ptr<void>(), qweight_ptr, qweight_size * sizeof(int32_t));
  
  // qzeros: [num_groups, out_features/8] INT32
  int32_t qzeros_size = num_groups * packed_out;
  qzeros_ = tensor::Tensor(base::DataType::kDataTypeInt32, qzeros_size, true, alloc);
  std::memcpy(qzeros_.ptr<void>(), qzeros_ptr, qzeros_size * sizeof(int32_t));
  
  // scales: [num_groups, out_features] FP16
  int32_t scales_size = num_groups * out_features_;
  scales_ = tensor::Tensor(base::DataType::kDataTypeFp16, scales_size, true, alloc);
  std::memcpy(scales_.ptr<void>(), scales_ptr, scales_size * sizeof(uint16_t));
}

void AWQMatmulLayer::to_cuda() {
  if (device_type_ != base::DeviceType::kDeviceCUDA) {
    return;
  }
  
  auto cuda_alloc = base::CUDADeviceAllocatorFactory::get_instance();
  
  // Move qweight to CUDA
  if (!qweight_.is_empty()) {
    tensor::Tensor cuda_qweight(base::DataType::kDataTypeInt32, 
                                 qweight_.size(), true, cuda_alloc);
    cudaMemcpy(cuda_qweight.ptr<void>(), qweight_.ptr<void>(),
               qweight_.byte_size(), cudaMemcpyHostToDevice);
    qweight_ = std::move(cuda_qweight);
  }
  
  // Move qzeros to CUDA
  int32_t num_groups = in_features_ / group_size_;
  if (!qzeros_.is_empty()) {
    tensor::Tensor cuda_qzeros(base::DataType::kDataTypeInt32, 
                                qzeros_.size(), true, cuda_alloc);
    cudaMemcpy(cuda_qzeros.ptr<void>(), qzeros_.ptr<void>(),
               qzeros_.byte_size(), cudaMemcpyHostToDevice);
    qzeros_ = std::move(cuda_qzeros);
  }
  
  // Move scales to CUDA
  if (!scales_.is_empty()) {
    tensor::Tensor cuda_scales(base::DataType::kDataTypeFp16, 
                                scales_.size(), true, cuda_alloc);
    cudaMemcpy(cuda_scales.ptr<void>(), scales_.ptr<void>(),
               scales_.byte_size(), cudaMemcpyHostToDevice);
    scales_ = std::move(cuda_scales);
  }
  
  LOG(INFO) << "AWQ weights loaded to CUDA: [" 
            << in_features_ << ", " << out_features_ << "]";
}

base::Status AWQMatmulLayer::forward(const tensor::Tensor& input, const tensor::Tensor& output) {
  if (input.is_empty() || output.is_empty()) {
    return base::error::InvalidArgument("Empty tensors in AWQ forward");
  }
  
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    int batch_size = input.size() / in_features_;
    
    cudaStream_t stream = nullptr;
    if (cuda_config_) {
      stream = cuda_config_->stream;
    }
    
    int split_k_iters = (batch_size == 1) ? 4 : 1;
    
    kernel::awq_gemm_tensorcore_cu(
        reinterpret_cast<const half*>(input.ptr<uint16_t>()),
        qweight_.ptr<int32_t>(),
        qzeros_.ptr<int32_t>(),
        reinterpret_cast<const half*>(scales_.ptr<uint16_t>()),
        reinterpret_cast<half*>(const_cast<uint16_t*>(output.ptr<uint16_t>())),
        batch_size,
        in_features_,
        out_features_,
        group_size_,
        split_k_iters,
        stream
    );
  } else {
    return base::error::InternalError("AWQ only supports CUDA device");
  }
  
  return base::error::Success();
}

base::Status AWQMatmulLayer::forward() {
  auto status = check();
  if (!status) {
    LOG(ERROR) << "AWQ check failed: " << status.get_err_msg();
    return status;
  }
  
  const tensor::Tensor& input = get_input(0);
  tensor::Tensor& output = get_output(0);
  
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    // Get batch size (assuming input is [batch, in_features])
    int batch_size = input.size() / in_features_;
    
    cudaStream_t stream = nullptr;
    if (cuda_config_) {
      stream = cuda_config_->stream;
    }
    
    // Adaptive split-K strategy for decode optimization
    int split_k_iters = 1;
    if (batch_size == 1) {
      split_k_iters = 4;   // Optimal for decode
    }
    
    // Use Tensor Core GEMM kernel
    kernel::awq_gemm_tensorcore_cu(
        reinterpret_cast<const half*>(input.ptr<uint16_t>()),
        qweight_.ptr<int32_t>(),
        qzeros_.ptr<int32_t>(),
        reinterpret_cast<const half*>(scales_.ptr<uint16_t>()),
        reinterpret_cast<half*>(const_cast<uint16_t*>(output.ptr<uint16_t>())),
        batch_size,
        in_features_,
        out_features_,
        group_size_,
        split_k_iters,
        stream
    );
  } else {
    return base::error::InternalError("AWQ only supports CUDA device");
  }
  
  return base::error::Success();
}

}  // namespace op
