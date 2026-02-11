#include "op/matmul.h"
#include <base/alloc.h>
#include "kernels/cpu/matmul_kernel.h"
#include "kernels/kernels_interface.h"
#include "kernels/cuda/fp16_convert_kernel.cuh"
#include "kernels/cuda/matmul_kernel.cuh"
namespace op {
MatmulLayer::MatmulLayer(base::DeviceType device_type, int32_t dim0, int32_t dim1,
                         bool is_quant_layer, bool has_bias)
    : LayerParam(device_type, LayerType::kLayerMatmul, is_quant_layer, "Matmul"),
      dim0_(dim0),
      dim1_(dim1),
      has_bias_(has_bias) {
  reset_input_size(1);
  reset_output_size(1);
  reset_weight_size(1);
  if (has_bias_) {
    bias_.resize(1);
  }
}

base::Status MatmulLayer::check() const {
  // Allow both FP32 and FP16 input (for pure FP16 compute path)
  auto input_dtype = get_input(0).data_type();
  if (input_dtype != data_type_ && input_dtype != base::DataType::kDataTypeFp16) {
    LOG(ERROR) << "The input tensor error in the matmul layer.";
    return base::error::InvalidArgument("Input must be FP32 or FP16");
  }
  auto status = check_tensor_with_dim(get_input(0), device_type_, input_dtype, dim1_);
  if (!status) {
    LOG(ERROR) << "The input tensor error in the matmul layer.";
    return status;
  }

  if (!is_quant_layer_) {
    // Allow both FP32 and FP16 weights for non-quantized layers
    auto weight_dtype = get_weight(0).data_type();
    if (weight_dtype == base::DataType::kDataTypeFp32) {
      status = check_tensor_with_dim(get_weight(0), device_type_, base::DataType::kDataTypeFp32, dim0_, dim1_);
    } else if (weight_dtype == base::DataType::kDataTypeFp16) {
      status = check_tensor_with_dim(get_weight(0), device_type_, base::DataType::kDataTypeFp16, dim0_, dim1_);
    } else {
      LOG(ERROR) << "Unsupported weight data type in matmul layer.";
      return base::error::InvalidArgument("Unsupported weight data type");
    }
    if (!status) {
      LOG(ERROR) << "The weight tensor error in the matmul layer.";
      return status;
    }
  } else {
    status = check_tensor_with_dim(get_weight(0), device_type_, base::DataType::kDataTypeInt8,
                                   dim0_, dim1_);
    if (!status) {
      LOG(ERROR) << "The weight tensor error in the matmul layer.";
      return status;
    }
  }

  if (is_quant_layer_) {
    status = check_tensor_with_dim(scales_, device_type_, base::DataType::kDataTypeFp32, scales_.size());
    if (!status) {
      LOG(ERROR) << "The scale tensor error in the matmul layer.";
      return status;
    }
  }

  // Allow both FP32 and FP16 output (for pure FP16 compute path)
  auto output_dtype = get_output(0).data_type();
  if (output_dtype != data_type_ && output_dtype != base::DataType::kDataTypeFp16) {
    LOG(ERROR) << "The output tensor error in the matmul layer.";
    return base::error::InvalidArgument("Output must be FP32 or FP16");
  }
  status = check_tensor_with_dim(get_output(0), device_type_, output_dtype, dim0_);
  if (!status) {
    LOG(ERROR) << "The output tensor error in the matmul layer.";
    return status;
  }
  return base::error::Success();
}

base::Status MatmulLayer::forward() {
  auto status = check();
  if (!status) {
    return status;
  }
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK(cuda_config_ != nullptr);
  }
  if (is_quant_layer_) {
    kernel::get_matmul_kernel_quant8(device_type_)(get_input(0), get_weight(0), get_output(0),
                                                   group_size_, scales_,
                                                   cuda_config_ ? cuda_config_.get() : nullptr);
  } else {
    // Check for pure FP16 path (FP16 input, FP16 weight, FP16 output)
    if (device_type_ == base::DeviceType::kDeviceCUDA &&
        get_input(0).data_type() == base::DataType::kDataTypeFp16 &&
        get_weight(0).data_type() == base::DataType::kDataTypeFp16 &&
        get_output(0).data_type() == base::DataType::kDataTypeFp16) {
      // Pure FP16 path using Tensor Core HGEMM
      kernel::matmul_kernel_cu_pure_fp16(get_input(0), get_weight(0), get_output(0), 1.f,
                                          cuda_config_ ? cuda_config_.get() : nullptr);
    } else if (device_type_ == base::DeviceType::kDeviceCUDA &&
               get_input(0).data_type() == base::DataType::kDataTypeFp16 &&
               get_weight(0).data_type() == base::DataType::kDataTypeFp16 &&
               get_output(0).data_type() == base::DataType::kDataTypeFp32) {
      // FP16 input x FP16 weight -> FP32 output (for cls_logits layer)
      kernel::matmul_kernel_cu_fp16_input_fp16_weight(get_input(0), get_weight(0), get_output(0), 1.f,
                                                       cuda_config_ ? cuda_config_.get() : nullptr);
    } else if (device_type_ == base::DeviceType::kDeviceCUDA && 
               get_weight(0).data_type() == base::DataType::kDataTypeFp16) {
      // Use FP16 weight kernel (mixed precision: FP16 weight x FP32 input -> FP32 output)
      kernel::matmul_kernel_cu_fp16_weight(get_input(0), get_weight(0), get_output(0), 1.f,
                                           cuda_config_ ? cuda_config_.get() : nullptr);
    } else {
      kernel::get_matmul_kernel(device_type_)(get_input(0), get_weight(0), get_output(0), 1.f,
                                              cuda_config_ ? cuda_config_.get() : nullptr);
    }
  }

  if (has_bias_) {
    // Debug: check types for bias addition
    static int bias_debug_count = 0;
    if (bias_debug_count < 3 && device_type_ == base::DeviceType::kDeviceCUDA) {
      // LOG(INFO) << "Bias addition: output_type=" << (int)get_output(0).data_type()
      //           << ", bias_type=" << (int)get_bias(0).data_type()
      //           << ", output_size=" << get_output(0).size()
      //           << ", bias_size=" << get_bias(0).size();
      bias_debug_count++;
    }
    kernel::get_add_kernel(device_type_)(get_output(0), get_bias(0), get_output(0),
                                            cuda_config_ ? cuda_config_->stream : nullptr);
  }

  return base::error::Success();
}

base::Status MatmulLayer::set_bias(int32_t idx, int32_t& dim, const void* bias_ptr,
                                   base::DeviceType device_type) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, bias_.size());
  CHECK_NE(bias_ptr, nullptr);

  size_t size = dim * sizeof(float);
  std::shared_ptr<base::Buffer> buffer =
      std::make_shared<base::Buffer>(size, nullptr, const_cast<void*>(bias_ptr), true);
  if (device_type != base::DeviceType::kDeviceUnknown) {
    buffer->set_device_type(device_type);
  }

  if (!is_quant_layer_) {
    tensor::Tensor bias(base::DataType::kDataTypeFp32, dim);
    bias.set_device_type(device_type);
    CHECK(bias.assign(buffer));
    // LOG(INFO) << "bias:" << bias.index<float>(0);
    bias_.at(idx) = bias;
  } else {
    // is quant layer
    tensor::Tensor bias(base::DataType::kDataTypeInt8, dim);
    bias.set_device_type(device_type);
    CHECK(bias.assign(buffer));
    bias_.at(idx) = bias;

    const int32_t bias_size = static_cast<int32_t>(bias.size());
    CHECK(bias_size % group_size_ == 0);

    int32_t scale_nums = bias_size / group_size_;
    scales_ = tensor::Tensor{base::DataType::kDataTypeFp32, scale_nums, false, nullptr,
                             reinterpret_cast<float*>((int8_t*)bias_ptr + bias_size)};
    scales_.set_device_type(device_type);
  }

  return base::error::Success();
}

// FP16 bias support
base::Status MatmulLayer::set_bias_fp16(int32_t idx, int32_t& dim, const void* bias_ptr,
                                        base::DeviceType device_type) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, bias_.size());
  CHECK_NE(bias_ptr, nullptr);

  // FP16 uses 2 bytes per element
  size_t size = dim * sizeof(uint16_t);
  std::shared_ptr<base::Buffer> buffer =
      std::make_shared<base::Buffer>(size, nullptr, const_cast<void*>(bias_ptr), true);
  if (device_type != base::DeviceType::kDeviceUnknown) {
    buffer->set_device_type(device_type);
  }

  tensor::Tensor bias(base::DataType::kDataTypeFp16, dim);
  bias.set_device_type(device_type);
  CHECK(bias.assign(buffer));
  bias_.at(idx) = bias;

  return base::error::Success();
}

tensor::Tensor& MatmulLayer::get_bias(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, bias_.size());
  return bias_.at(idx);
}

const tensor::Tensor& MatmulLayer::get_bias(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, bias_.size());
  return bias_.at(idx);
}

void MatmulLayer::to_cuda() {
  LayerParam::to_cuda();
  if (has_bias_) {
    cudaStream_t stream = cuda_config_ ? cuda_config_->stream : nullptr;
    for (auto& bias : bias_) {
      // Keep FP16 bias as FP16 for pure FP16 model support
      bias.to_cuda(stream);
    }
  }
}

}  // namespace op