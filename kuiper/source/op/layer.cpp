#include "op/layer.h"
#include <base/alloc.h>
#include <base/cuda_config.h>
#include <glog/logging.h>
#include <cstdarg>
#include <numeric>
#include <utility>
#include "kernels/cuda/fp16_convert_kernel.cuh"

namespace op {
BaseLayer::BaseLayer(base::DeviceType device_type, LayerType layer_type, base::DataType data_type,
                     std::string layer_name)
    : device_type_(device_type),
      layer_type_(layer_type),
      data_type_(data_type),
      layer_name_(std::move(layer_name)) {}

base::DataType BaseLayer::data_type() const { return data_type_; }

LayerType BaseLayer::layer_type() const { return layer_type_; }

base::Status BaseLayer::set_weight(int32_t idx, const tensor::Tensor& weight) {
  return base::error::FunctionNotImplement();
}

base::Status BaseLayer::set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                   const void* weight_ptr, base::DeviceType device_type) {
  return base::error::FunctionNotImplement();
}

const std::string& BaseLayer::get_layer_name() const { return layer_name_; }

void BaseLayer::set_layer_name(const std::string& layer_name) { layer_name_ = layer_name; }
base::DeviceType BaseLayer::device_type() const { return device_type_; }

void BaseLayer::set_device_type(base::DeviceType device_type) { device_type_ = device_type; }

Layer::Layer(base::DeviceType device_type, LayerType layer_type, std::string layer_name)
    : BaseLayer(device_type, layer_type, base::DataType::kDataTypeFp32, std::move(layer_name)) {}

base::Status Layer::init() { return base::error::Success(); }

base::Status Layer::forward() { return base::error::FunctionNotImplement(""); }

base::Status Layer::check_tensor(const tensor::Tensor& tensor, base::DeviceType device_type,
                                 base::DataType data_type) const {
  if (tensor.is_empty()) {
    return base::error::InvalidArgument("The tensor parameter is empty.");
  }
  if (tensor.device_type() != device_type) {
    return base::error::InvalidArgument("The tensor has a wrong device type.");
  }
  if (tensor.data_type() != data_type) {
    return base::error::InvalidArgument("The tensor has a wrong data type.");
  }
  return base::error::Success();
}

base::Status Layer::check_tensor_with_dim(const tensor::Tensor& tensor,
                                          base::DeviceType device_type, base::DataType data_type,
                                          ...) const {
  std::va_list args;
  if (tensor.is_empty()) {
    return base::error::InvalidArgument("The tensor parameter is empty.");
  }
  if (tensor.device_type() != device_type) {
    return base::error::InvalidArgument("The tensor has a wrong device type.");
  }
  if (tensor.data_type() != data_type) {
    return base::error::InvalidArgument("The tensor has a wrong data type.");
  }

  va_start(args, data_type);
  int32_t dims = tensor.dims_size();
  for (int32_t i = 0; i < dims; ++i) {
    int32_t dim = va_arg(args, int32_t);
    if (dim != tensor.get_dim(i)) {
      return base::error::InvalidArgument("The tensor has a wrong dim in dim" + std::to_string(i));
    }
  }
  va_end(args);
  return base::error::Success();
}

void Layer::set_input(int32_t idx, const tensor::Tensor& input) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  this->inputs_.at(idx) = input;
}

void Layer::set_output(int32_t idx, const tensor::Tensor& output) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  this->outputs_.at(idx) = output;
}

const tensor::Tensor& Layer::get_input(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  return inputs_.at(idx);
}

tensor::Tensor& Layer::get_input(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  return inputs_.at(idx);
}

tensor::Tensor& Layer::get_output(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  return outputs_.at(idx);
}

base::Status Layer::check() const {
  return base::error::FunctionNotImplement("The check function is not implement yet");
}

const tensor::Tensor& Layer::get_output(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  return outputs_.at(idx);
}

void Layer::reset_input_size(size_t size) { inputs_.resize(size); }

void Layer::reset_output_size(size_t size) { outputs_.resize(size); }

void Layer::to_cuda() {
  for (auto& input : inputs_) {
    if (!input.is_empty()) {
      input.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
    }
  }
  for (auto& output : outputs_) {
    if (!output.is_empty()) {
      output.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
    }
  }
}

void Layer::set_cuda_config(std::shared_ptr<kernel::CudaConfig> config) {
  if (!config) {
    return;
  }
  this->cuda_config_ = config;
}

std::shared_ptr<kernel::CudaConfig> Layer::cuda_config() const { return cuda_config_; }

size_t Layer::input_size() const { return inputs_.size(); }

size_t Layer::output_size() const { return outputs_.size(); }

LayerParam::LayerParam(base::DeviceType device_type, LayerType layer_type, bool is_quant_layer,
                       std::string layer_name)
    : Layer(device_type, layer_type, std::move(layer_name)), is_quant_layer_(is_quant_layer) {}

base::Status LayerParam::set_weight(int32_t idx, const tensor::Tensor& weight) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  CHECK(weight.data_type() == base::DataType::kDataTypeFp32);
  if (!weight.is_empty()) {
    CHECK(weight.device_type() == device_type_);
  }
  weights_.at(idx) = weight;
  return base::error::Success();
}

const tensor::Tensor& LayerParam::get_weight(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  return weights_.at(idx);
}

void LayerParam::to_cuda() {
  Layer::to_cuda();
  cudaStream_t stream = cuda_config_ ? cuda_config_->stream : nullptr;
  
  for (size_t w_idx = 0; w_idx < weights_.size(); ++w_idx) {
    auto& weight = weights_[w_idx];
    if (weight.is_empty()) {
      continue;
    }
    if (weight.data_type() == base::DataType::kDataTypeFp16) {
      if (keep_fp16_weights_) {
        // Keep FP16 weights on GPU for pure FP16 compute path
        size_t num_elements = weight.size();
        std::vector<int32_t> dims = weight.dims();
        
        // Allocate FP16 GPU buffer
        auto cu_alloc = base::CUDADeviceAllocatorFactory::get_instance();
        size_t fp16_byte_size = num_elements * sizeof(uint16_t);
        auto fp16_buffer = std::make_shared<base::Buffer>(fp16_byte_size, cu_alloc);
        fp16_buffer->set_device_type(base::DeviceType::kDeviceCUDA);
        
        // Get FP16 data pointer from CPU
        const uint16_t* fp16_cpu_ptr = weight.ptr<uint16_t>();
        uint16_t* fp16_gpu_ptr = static_cast<uint16_t*>(fp16_buffer->ptr());
        
        if (fp16_cpu_ptr == nullptr) {
          LOG(ERROR) << "FP16 weight pointer is null!";
          continue;
        }
        
        // Copy FP16 data directly to GPU
        cudaMemcpyAsync(fp16_gpu_ptr, fp16_cpu_ptr, fp16_byte_size, cudaMemcpyHostToDevice, stream);
        
        // Update tensor to use FP16 buffer on GPU
        weight = tensor::Tensor(base::DataType::kDataTypeFp16, dims);
        weight.set_device_type(base::DeviceType::kDeviceCUDA);
        weight.assign(fp16_buffer);
      } else {
        // Convert FP16 weights to FP32 for computation
        // This ensures numerical stability and compatibility with FP32 activations
        size_t num_elements = weight.size();
        std::vector<int32_t> dims = weight.dims();
        
        // Allocate FP32 GPU buffer
        auto cu_alloc = base::CUDADeviceAllocatorFactory::get_instance();
        size_t fp32_byte_size = num_elements * sizeof(float);
        auto fp32_buffer = std::make_shared<base::Buffer>(fp32_byte_size, cu_alloc);
        fp32_buffer->set_device_type(base::DeviceType::kDeviceCUDA);
        
        // Get FP16 data pointer from CPU
        const uint16_t* fp16_ptr = weight.ptr<uint16_t>();
        float* fp32_gpu_ptr = static_cast<float*>(fp32_buffer->ptr());
        
        if (fp16_ptr == nullptr) {
          LOG(ERROR) << "FP16 weight pointer is null!";
          continue;
        }
        
        // Convert FP16 to FP32 on GPU
        kernel::fp16_cpu_to_fp32_gpu(fp16_ptr, fp32_gpu_ptr, num_elements, stream);
        
        // Update tensor to use FP32 buffer (converted from FP16)
        weight = tensor::Tensor(base::DataType::kDataTypeFp32, dims);
        weight.set_device_type(base::DeviceType::kDeviceCUDA);
        weight.assign(fp32_buffer);
      }
    } else {
      // Normal to_cuda for FP32 or other types
      weight.to_cuda(stream);
    }
  }
  if (!scales_.is_empty()) {
    scales_.to_cuda(stream);
  }
}

base::Status LayerParam::set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                    const void* weight_ptr, base::DeviceType device_type) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  CHECK_NE(weight_ptr, nullptr);

  size_t size = std::accumulate(dims.begin(), dims.end(), sizeof(float), std::multiplies<>());
  std::shared_ptr<base::Buffer> buffer =
      std::make_shared<base::Buffer>(size, nullptr, const_cast<void*>(weight_ptr), true);
  if (device_type != base::DeviceType::kDeviceUnknown) {
    buffer->set_device_type(device_type);
  }

  if (!is_quant_layer_) {
    tensor::Tensor weight(base::DataType::kDataTypeFp32, dims);
    weight.set_device_type(device_type);
    CHECK(weight.assign(buffer));
    weights_.at(idx) = weight;
  } else {
    // is quant layer
    tensor::Tensor weight(base::DataType::kDataTypeInt8, dims);
    weight.set_device_type(device_type);
    CHECK(weight.assign(buffer));
    weights_.at(idx) = weight;

    const int32_t weight_size = static_cast<int32_t>(weight.size());
    CHECK(weight_size % group_size_ == 0);

    int32_t scale_nums = weight_size / group_size_;
    scales_ = tensor::Tensor{base::DataType::kDataTypeFp32, scale_nums, false, nullptr,
                             reinterpret_cast<float*>((int8_t*)weight_ptr + weight_size)};
    scales_.set_device_type(device_type);
  }

  return base::error::Success();
}

// FP16 weight support: stores weights as FP16 (half) precision
base::Status LayerParam::set_weight_fp16(int32_t idx, const std::vector<int32_t>& dims,
                                         const void* weight_ptr, base::DeviceType device_type) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  CHECK_NE(weight_ptr, nullptr);

  // FP16 uses 2 bytes per element (sizeof(__half) = 2)
  size_t size = std::accumulate(dims.begin(), dims.end(), sizeof(uint16_t), std::multiplies<>());
  std::shared_ptr<base::Buffer> buffer =
      std::make_shared<base::Buffer>(size, nullptr, const_cast<void*>(weight_ptr), true);
  if (device_type != base::DeviceType::kDeviceUnknown) {
    buffer->set_device_type(device_type);
  }

  // Create tensor with FP16 data type
  tensor::Tensor weight(base::DataType::kDataTypeFp16, dims);
  weight.set_device_type(device_type);
  CHECK(weight.assign(buffer));
  weights_.at(idx) = weight;

  return base::error::Success();
}

void LayerParam::set_scales(const tensor::Tensor& scales) {
  CHECK(!scales.is_empty());
  this->scales_ = scales;
}

void LayerParam::set_group_size(int32_t group_size) { this->group_size_ = group_size; }

int32_t LayerParam::get_scale_num() const {
  CHECK(!scales_.is_empty());
  return static_cast<int32_t>(scales_.size());
}

void LayerParam::reset_weight_size(size_t size) { weights_.resize(size); }

size_t LayerParam::weight_size() const { return weights_.size(); }

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_output(0, output1);
  return this->forward();
}

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);

  this->set_output(0, output1);
  return this->forward();
}

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& input3, const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);
  this->set_input(2, input3);

  this->set_output(0, output1);
  return this->forward();
}

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& input3, const tensor::Tensor& input4,
                            const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);
  this->set_input(2, input3);
  this->set_input(3, input4);

  this->set_output(0, output1);
  return this->forward();
}

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& input3, const tensor::Tensor& input4,
                            const tensor::Tensor& input5, const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);
  this->set_input(2, input3);
  this->set_input(3, input4);
  this->set_input(4, input5);

  this->set_output(0, output1);
  return this->forward();
}

tensor::Tensor& LayerParam::get_weight(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  return weights_.at(idx);
}

}  // namespace op