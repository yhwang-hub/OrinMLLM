#include "op/embedding.h"
#include "kernels/cpu/emb_kernel.h"
#include "kernels/kernels_interface.h"
#include "op/layer.h"
namespace op {
EmbeddingLayer::EmbeddingLayer(base::DeviceType device_type, int32_t dim, int32_t seq_len,
                               int32_t vocab_size)
    : dim_(dim),
      seq_len_(seq_len),
      vocab_size_(vocab_size),
      LayerParam(device_type, LayerType::kLayerEmbedding, false, "Embedding") {
  reset_weight_size(1);
  reset_input_size(2);
  reset_output_size(1);
}

base::Status EmbeddingLayer::check() const {
  const auto& input_tensor = get_input(0);
  const auto& token_size = get_input(1).size();
  if (token_size > input_tensor.size()) {
    return base::error::InvalidArgument("The number of input tensor is greater than seq len.");
  }

  base::Status status = check_tensor_with_dim(input_tensor, base::DeviceType::kDeviceCPU,
                                              base::DataType::kDataTypeInt32, token_size);
  if (!status) {
    LOG(ERROR) << "The input tensor error in the embedding layer.";
    return status;
  }

  // Allow both FP32 and FP16 weights for embedding layer
  auto weight_dtype = get_weight(0).data_type();
  if (weight_dtype == base::DataType::kDataTypeFp32) {
    status = check_tensor_with_dim(get_weight(0), device_type_, base::DataType::kDataTypeFp32, vocab_size_, dim_);
  } else if (weight_dtype == base::DataType::kDataTypeFp16) {
    status = check_tensor_with_dim(get_weight(0), device_type_, base::DataType::kDataTypeFp16, vocab_size_, dim_);
  } else {
    LOG(ERROR) << "Unsupported weight data type in embedding layer.";
    return base::error::InvalidArgument("Unsupported weight data type");
  }
  if (!status) {
    LOG(ERROR) << "The weight tensor error in the embedding layer.";
    return status;
  }

  // Allow output to match either the default data_type_ or the weight data type (for pure FP16 path)
  auto output_dtype = get_output(0).data_type();
  if (output_dtype == weight_dtype || output_dtype == data_type_) {
    status = check_tensor_with_dim(get_output(0), device_type_, output_dtype, token_size, dim_);
  } else {
    LOG(ERROR) << "Output tensor type must match either weight type or layer default type.";
    return base::error::InvalidArgument("Output tensor type mismatch");
  }
  if (!status) {
    LOG(ERROR) << "The output tensor error in the embedding layer.";
    return status;
  }
  return base::error::Success();
}

base::Status EmbeddingLayer::forward() {
  base::Status status = check();
  if (!status) {
    return status;
  }
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK(cuda_config_ != nullptr);
  }
  kernel::get_emb_kernel(device_type_)(get_input(0), get_weight(0), get_output(0), vocab_size_,
                                       cuda_config_ ? cuda_config_->stream : nullptr);
  return base::StatusCode::kSuccess;
}
}  // namespace op