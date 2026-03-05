//
// SmoothQuant INT8 Quantized Matrix Multiplication Layer Implementation
//
// Strategy: Keep INT8 weights on GPU, dynamic per-tensor activation quantization,
// CUTLASS INT8 Tensor Core GEMM with fused epilogue dequantization.
//
// Key improvements over the previous dequant-at-load approach:
// 1. Fast loading: INT8 weights uploaded directly to GPU (no CPU dequantization)
// 2. True INT8 GEMM: ~2x throughput via INT8 Tensor Core MMA instructions
// 3. Lower GPU memory: INT8 weights use 2x less memory than FP16
//

#include "op/sq_matmul.h"
#include "kernels/cuda/sq_gemm_kernel.cuh"
#include "base/alloc.h"
#include <glog/logging.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>

namespace op {

SQMatmulLayer::SQMatmulLayer(base::DeviceType device_type,
                             int32_t in_features,
                             int32_t out_features)
    : Layer(device_type, LayerType::kLayerMatmul, "SQMatmul"),
      in_features_(in_features),
      out_features_(out_features) {
  reset_input_size(1);
  reset_output_size(1);
}

base::Status SQMatmulLayer::check() const {
  if (in_features_ <= 0 || out_features_ <= 0) {
    return base::error::InternalError("Invalid dimensions for SQ matmul");
  }
  if (qweight_.is_empty()) {
    return base::error::InternalError("SQ weights not set");
  }
  if (weight_scale_ == 0.0f) {
    return base::error::InternalError("SQ weight_scale not set");
  }
  return base::error::Success();
}

// Convert FP16 bits (uint16_t) to FP32 value on host (no CUDA intrinsics needed)
static float fp16_bits_to_float(uint16_t h) {
  uint32_t sign = (h >> 15) & 1;
  uint32_t exponent = (h >> 10) & 0x1F;
  uint32_t mantissa = h & 0x3FF;
  uint32_t fp32_bits;
  if (exponent == 0) {
    if (mantissa == 0) {
      fp32_bits = sign << 31;
    } else {
      exponent = 1;
      while (!(mantissa & 0x400)) {
        mantissa <<= 1;
        exponent--;
      }
      mantissa &= 0x3FF;
      fp32_bits = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
  } else if (exponent == 0x1F) {
    fp32_bits = (sign << 31) | (0xFF << 23) | (mantissa << 13);
  } else {
    fp32_bits = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  }
  float result;
  std::memcpy(&result, &fp32_bits, sizeof(float));
  return result;
}

void SQMatmulLayer::set_sq_weights(const void* qweight_ptr,
                                    const void* weight_scale_ptr,
                                    const void* input_scale_ptr,
                                    base::DeviceType src_device) {
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();

  // Load INT8 quantized weight [out_features, in_features]
  int32_t qweight_size = out_features_ * in_features_;
  qweight_ = tensor::Tensor(base::DataType::kDataTypeInt8, qweight_size, true, alloc);
  std::memcpy(qweight_.ptr<void>(), qweight_ptr, qweight_size * sizeof(int8_t));

  // Load weight_scale (FP16 scalar → convert to FP32)
  uint16_t ws_fp16;
  std::memcpy(&ws_fp16, weight_scale_ptr, sizeof(uint16_t));
  weight_scale_ = fp16_bits_to_float(ws_fp16);

  // Load input_scale (FP32 scalar) – kept for reference but not used at runtime
  // (we use dynamic per-tensor absmax quantization instead)
  std::memcpy(&input_scale_, input_scale_ptr, sizeof(float));
}

void SQMatmulLayer::to_cuda() {
  if (device_type_ != base::DeviceType::kDeviceCUDA) {
    return;
  }

  if (!qweight_.is_empty()) {
    // Direct INT8 upload to GPU – no CPU dequantization needed!
    // This is the key speedup for model loading time.
    auto cuda_alloc = base::CUDADeviceAllocatorFactory::get_instance();
    int32_t total = out_features_ * in_features_;

    tensor::Tensor gpu_qweight(base::DataType::kDataTypeInt8, total, true, cuda_alloc);
    cudaMemcpy(gpu_qweight.ptr<void>(), qweight_.ptr<void>(),
               total * sizeof(int8_t), cudaMemcpyHostToDevice);

    // Replace CPU tensor with GPU tensor
    qweight_ = std::move(gpu_qweight);
  }
}

base::Status SQMatmulLayer::forward(const tensor::Tensor& input, const tensor::Tensor& output) {
  if (input.is_empty() || output.is_empty()) {
    return base::error::InvalidArgument("Empty tensors in SQ forward");
  }

  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK(!qweight_.is_empty()) << "INT8 weight not available. Call to_cuda() first.";

    int batch_size = input.size() / in_features_;

    cudaStream_t stream = nullptr;
    if (cuda_config_) {
      stream = cuda_config_->stream;
    }

    // CUTLASS INT8 Tensor Core GEMM with dynamic per-tensor quantization
    // Full pipeline on GPU: FP16→INT8 quantize → INT8 GEMM → FP16 dequant (fused epilogue)
    kernel::sq_gemm_cu(
        input.ptr<half>(),           // FP16 input [M, K]
        qweight_.ptr<int8_t>(),      // INT8 weight [N, K] on GPU
        const_cast<half*>(output.ptr<half>()),  // FP16 output [M, N]
        weight_scale_,               // per-tensor weight scale
        batch_size,                  // M
        in_features_,                // K
        out_features_,               // N
        stream);
  } else {
    return base::error::InternalError("SQ only supports CUDA device");
  }

  return base::error::Success();
}

base::Status SQMatmulLayer::forward() {
  auto status = check();
  if (!status) {
    LOG(ERROR) << "SQ check failed: " << status.get_err_msg();
    return status;
  }

  const tensor::Tensor& input = get_input(0);
  tensor::Tensor& output = get_output(0);

  return forward(input, output);
}

base::Status SQMatmulLayer::fused_ffn_forward(const tensor::Tensor& input,
                                               const tensor::Tensor& output,
                                               const SQMatmulLayer& w1_layer,
                                               const SQMatmulLayer& w3_layer,
                                               cudaStream_t stream) {
  if (input.is_empty() || output.is_empty()) {
    return base::error::InvalidArgument("Empty tensors in SQ fused FFN forward");
  }

  kernel::sq_fused_ffn_cu(
      input.ptr<half>(),
      w1_layer.qweight_ptr(),
      w3_layer.qweight_ptr(),
      const_cast<half*>(output.ptr<half>()),
      w1_layer.weight_scale(),
      w3_layer.weight_scale(),
      w1_layer.in_features(),
      w1_layer.out_features(),
      stream);

  return base::error::Success();
}

void SQMatmulLayer::quantize_input(const tensor::Tensor& input, cudaStream_t stream) {
  kernel::sq_quantize_input_cu(input.ptr<half>(), input.size(), stream);
}

base::Status SQMatmulLayer::forward_preq(const tensor::Tensor& output,
                                          const SQMatmulLayer& layer,
                                          cudaStream_t stream) {
  if (output.is_empty()) {
    return base::error::InvalidArgument("Empty output tensor in SQ forward_preq");
  }

  kernel::sq_gemv_preq_cu(
      layer.qweight_ptr(),
      const_cast<half*>(output.ptr<half>()),
      layer.weight_scale(),
      layer.in_features(),
      layer.out_features(),
      stream);

  return base::error::Success();
}

}  // namespace op
