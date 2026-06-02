#include <tensor/tensor.h>
#include <cuda_fp16.h>
#include "swiglu_kernel.cuh"
namespace kernel {

// ============================================================================
// Optimized FP32 SwiGLU Kernel
// - Removed unnecessary shared memory (element-wise op needs no data sharing)
// - float4 vectorization: 128-bit loads/stores, 4 elements per thread
// - __expf / __fdividef fast math intrinsics
// - 256 threads/block for better occupancy on Orin (SM 8.7)
// ============================================================================
__global__ void __launch_bounds__(256)
swiglu_kernel_cu_fp32(int size, const float* __restrict__ in1,
                      const float* __restrict__ in2, float* __restrict__ out) {
  const int idx = (threadIdx.x + blockDim.x * blockIdx.x) * 4;

  if (idx + 3 < size) {
    // Vectorized path: load 4 floats (128 bits) at once
    float4 v1 = *reinterpret_cast<const float4*>(in1 + idx);
    float4 v2 = *reinterpret_cast<const float4*>(in2 + idx);

    float4 result;
    result.x = v1.x * __fdividef(1.0f, 1.0f + __expf(-v1.x)) * v2.x;
    result.y = v1.y * __fdividef(1.0f, 1.0f + __expf(-v1.y)) * v2.y;
    result.z = v1.z * __fdividef(1.0f, 1.0f + __expf(-v1.z)) * v2.z;
    result.w = v1.w * __fdividef(1.0f, 1.0f + __expf(-v1.w)) * v2.w;

    *reinterpret_cast<float4*>(out + idx) = result;
  } else {
    // Tail elements
    for (int i = idx; i < size && i < idx + 4; i++) {
      float val1 = in1[i];
      float val2 = in2[i];
      out[i] = val1 * __fdividef(1.0f, 1.0f + __expf(-val1)) * val2;
    }
  }
}

// ============================================================================
// Optimized FP16 Vectorized SwiGLU Kernel
// - float4 loads/stores: 128-bit transactions, 8 half elements per thread
// - FP32 intermediate computation for numerical stability
// - __expf / __fdividef fast math intrinsics
// - #pragma unroll for inner loop
// - 256 threads/block for optimal Orin occupancy
// ============================================================================
__global__ void __launch_bounds__(256)
swiglu_kernel_cu_fp16_vec(int size, const half* __restrict__ in1,
                          const half* __restrict__ in2, half* __restrict__ out) {
  const int idx = (threadIdx.x + blockDim.x * blockIdx.x) * 8;

  if (idx + 7 < size) {
    // Load 8 half elements (128 bits) via float4
    float4 raw1 = *reinterpret_cast<const float4*>(in1 + idx);
    float4 raw2 = *reinterpret_cast<const float4*>(in2 + idx);

    const half2* h1 = reinterpret_cast<const half2*>(&raw1);
    const half2* h2 = reinterpret_cast<const half2*>(&raw2);

    float4 out_raw;
    half2* h_out = reinterpret_cast<half2*>(&out_raw);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
      float2 f1 = __half22float2(h1[i]);
      float2 f2 = __half22float2(h2[i]);

      float2 r;
      r.x = f1.x * __fdividef(1.0f, 1.0f + __expf(-f1.x)) * f2.x;
      r.y = f1.y * __fdividef(1.0f, 1.0f + __expf(-f1.y)) * f2.y;

      h_out[i] = __float22half2_rn(r);
    }

    *reinterpret_cast<float4*>(out + idx) = out_raw;
  } else {
    // Tail elements
    for (int i = idx; i < size && i < idx + 8; i++) {
      float val1 = __half2float(in1[i]);
      float val2 = __half2float(in2[i]);
      out[i] = __float2half(val1 * __fdividef(1.0f, 1.0f + __expf(-val1)) * val2);
    }
  }
}

// ============================================================================
// Dispatch function: selects FP16 or FP32 kernel based on tensor data types
// ============================================================================
void swiglu_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                      const tensor::Tensor& output, void* stream) {
  CHECK_EQ(input1.is_empty(), false);
  CHECK(input1.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK_EQ(input2.is_empty(), false);
  CHECK(input2.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK_EQ(output.is_empty(), false);
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

  const int size = static_cast<int32_t>(input1.size());
  constexpr int threads = 256;

  // Pure FP16 path
  if (input1.data_type() == base::DataType::kDataTypeFp16 &&
      input2.data_type() == base::DataType::kDataTypeFp16 &&
      output.data_type() == base::DataType::kDataTypeFp16) {
    constexpr int elems_per_thread = 8;
    const int blocks = (size + threads * elems_per_thread - 1) / (threads * elems_per_thread);

    const half* in1_ptr = reinterpret_cast<const half*>(input1.ptr<uint16_t>());
    const half* in2_ptr = reinterpret_cast<const half*>(input2.ptr<uint16_t>());
    half* out_ptr = reinterpret_cast<half*>(const_cast<uint16_t*>(output.ptr<uint16_t>()));

    if (!stream) {
      swiglu_kernel_cu_fp16_vec<<<blocks, threads>>>(size, in1_ptr, in2_ptr, out_ptr);
    } else {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      swiglu_kernel_cu_fp16_vec<<<blocks, threads, 0, stream_>>>(size, in1_ptr, in2_ptr, out_ptr);
    }
    return;
  }

  // FP32 path: float4 vectorized, no shared memory
  constexpr int elems_per_thread = 4;
  const int blocks = (size + threads * elems_per_thread - 1) / (threads * elems_per_thread);
  if (!stream) {
    swiglu_kernel_cu_fp32<<<blocks, threads>>>(
        size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
  } else {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    swiglu_kernel_cu_fp32<<<blocks, threads, 0, stream_>>>(
        size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
  }
}

}  // namespace kernel