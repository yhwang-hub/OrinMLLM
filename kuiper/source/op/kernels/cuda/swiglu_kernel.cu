#include <tensor/tensor.h>
#include <cuda_fp16.h>
#include "swiglu_kernel.cuh"
namespace kernel {
__global__ void swiglu_kernel_cu_fp32(int size, const float* in1, const float* in2, float* out) {
  int tid = threadIdx.x;
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= size) {
    return;
  }
  extern __shared__ float shared_mem[];
  float* smem1 = shared_mem;
  float* smem2 = shared_mem + blockDim.x;

  smem1[tid] = in1[idx];
  smem2[tid] = in2[idx];
  __syncthreads();

  float value = 1.0f / (1.0f + exp(-smem1[tid]));
  smem1[tid] = smem1[tid] * value;

  out[idx] = smem1[tid] * smem2[tid];
}

// Pure FP16 SwiGLU kernel - computes in FP32 for precision, uses FP16 I/O
__global__ void swiglu_kernel_cu_fp16_impl(int size, const half* in1, const half* in2, half* out) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= size) {
    return;
  }
  
  // Load FP16, compute in FP32 for better precision
  float val1 = __half2float(in1[idx]);
  float val2 = __half2float(in2[idx]);
  
  // SiLU: x * sigmoid(x)
  float sigmoid = 1.0f / (1.0f + expf(-val1));
  float silu = val1 * sigmoid;
  
  // Multiply with gate
  float result = silu * val2;
  
  // Store as FP16
  out[idx] = __float2half(result);
}

// Vectorized pure FP16 SwiGLU using half2 for better throughput
__global__ void swiglu_kernel_cu_fp16_vec(int size, const half* in1, const half* in2, half* out) {
  int idx = (threadIdx.x + blockDim.x * blockIdx.x) * 2;
  
  if (idx + 1 < size) {
    half2 v1 = *reinterpret_cast<const half2*>(in1 + idx);
    half2 v2 = *reinterpret_cast<const half2*>(in2 + idx);
    
    // Convert to float for precision
    float2 f1 = __half22float2(v1);
    float2 f2 = __half22float2(v2);
    
    // SiLU + multiply for both elements
    float sigmoid0 = 1.0f / (1.0f + expf(-f1.x));
    float sigmoid1 = 1.0f / (1.0f + expf(-f1.y));
    float silu0 = f1.x * sigmoid0;
    float silu1 = f1.y * sigmoid1;
    
    float2 result;
    result.x = silu0 * f2.x;
    result.y = silu1 * f2.y;
    
    *reinterpret_cast<half2*>(out + idx) = __float22half2_rn(result);
  } else if (idx < size) {
    float val1 = __half2float(in1[idx]);
    float val2 = __half2float(in2[idx]);
    float sigmoid = 1.0f / (1.0f + expf(-val1));
    out[idx] = __float2half(val1 * sigmoid * val2);
  }
}

void swiglu_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                      const tensor::Tensor& output, void* stream) {
  CHECK_EQ(input1.is_empty(), false);
  CHECK(input1.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK_EQ(input2.is_empty(), false);
  CHECK(input2.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK_EQ(output.is_empty(), false);
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

  int size = static_cast<int32_t>(input1.size());
  
  // Check if this is pure FP16 path
  if (input1.data_type() == base::DataType::kDataTypeFp16 &&
      input2.data_type() == base::DataType::kDataTypeFp16 &&
      output.data_type() == base::DataType::kDataTypeFp16) {
    // Use vectorized FP16 kernel
    int threads = 256;
    int elements_per_thread = 2;
    int blocks = (size + threads * elements_per_thread - 1) / (threads * elements_per_thread);
    
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
  
  // FP32 path
  int threads = 128;
  int blocks = (size + threads - 1) / threads;
  const size_t shmem = threads * sizeof(float) * 2;
  if (!stream) {
    swiglu_kernel_cu_fp32<<<blocks, threads, shmem>>>(
        size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
  } else {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    swiglu_kernel_cu_fp32<<<blocks, threads, shmem, stream_>>>(
        size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
  }
}

void swiglu_kernel_cu_pure_fp16(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                 const tensor::Tensor& output, void* stream) {
  CHECK_EQ(input1.is_empty(), false);
  CHECK(input1.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(input1.data_type() == base::DataType::kDataTypeFp16);

  CHECK_EQ(input2.is_empty(), false);
  CHECK(input2.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(input2.data_type() == base::DataType::kDataTypeFp16);

  CHECK_EQ(output.is_empty(), false);
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(output.data_type() == base::DataType::kDataTypeFp16);

  int size = static_cast<int32_t>(input1.size());
  
  // Use vectorized kernel for better throughput
  int threads = 256;
  int elements_per_thread = 2;
  int blocks = (size + threads * elements_per_thread - 1) / (threads * elements_per_thread);
  
  const half* in1_ptr = reinterpret_cast<const half*>(input1.ptr<uint16_t>());
  const half* in2_ptr = reinterpret_cast<const half*>(input2.ptr<uint16_t>());
  half* out_ptr = reinterpret_cast<half*>(const_cast<uint16_t*>(output.ptr<uint16_t>()));
  
  if (!stream) {
    swiglu_kernel_cu_fp16_vec<<<blocks, threads>>>(size, in1_ptr, in2_ptr, out_ptr);
  } else {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    swiglu_kernel_cu_fp16_vec<<<blocks, threads, 0, stream_>>>(size, in1_ptr, in2_ptr, out_ptr);
  }
}
}  // namespace kernel