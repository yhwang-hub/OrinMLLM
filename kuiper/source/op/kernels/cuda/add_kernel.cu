#include "add_kernel.cuh"
#include <cuda_fp16.h>

namespace kernel {
__global__ void add_kernel_cu_fp32(int32_t size, const float* in1, const float* in2, float* out) {
  int32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= size) {
    return;
  }
  float in_val1 = in1[tid];
  float in_val2 = in2[tid];
  out[tid] = in_val1 + in_val2;
}

// Pure FP16 add kernel using half2 for vectorized operations
__global__ void add_kernel_cu_fp16_impl(int32_t size, const half* in1, const half* in2, half* out) {
  int32_t idx = (threadIdx.x + blockDim.x * blockIdx.x) * 2;
  
  // Process 2 elements at a time using half2
  if (idx + 1 < size) {
    half2 val1 = *reinterpret_cast<const half2*>(in1 + idx);
    half2 val2 = *reinterpret_cast<const half2*>(in2 + idx);
    *reinterpret_cast<half2*>(out + idx) = __hadd2(val1, val2);
  } else if (idx < size) {
    out[idx] = __hadd(in1[idx], in2[idx]);
  }
}

void add_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                   const tensor::Tensor& output, void* stream) {
  CHECK_EQ(input1.is_empty(), false);
  CHECK_EQ(input2.is_empty(), false);
  CHECK_EQ(output.is_empty(), false);
  int32_t size = static_cast<int32_t>(input1.size());
  CHECK_EQ(size, input2.size());
  CHECK_EQ(size, output.size());
  
  // Check if this is pure FP16 path
  if (input1.data_type() == base::DataType::kDataTypeFp16 &&
      input2.data_type() == base::DataType::kDataTypeFp16 &&
      output.data_type() == base::DataType::kDataTypeFp16) {
    // Process 2 elements per thread with half2
    int32_t thread_num = 256;
    int32_t elements_per_thread = 2;
    int32_t block_num = (size + thread_num * elements_per_thread - 1) / (thread_num * elements_per_thread);
    
    const half* in1_ptr = reinterpret_cast<const half*>(input1.ptr<uint16_t>());
    const half* in2_ptr = reinterpret_cast<const half*>(input2.ptr<uint16_t>());
    half* out_ptr = reinterpret_cast<half*>(const_cast<uint16_t*>(output.ptr<uint16_t>()));
    
    if (stream) {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      add_kernel_cu_fp16_impl<<<block_num, thread_num, 0, stream_>>>(size, in1_ptr, in2_ptr, out_ptr);
    } else {
      add_kernel_cu_fp16_impl<<<block_num, thread_num>>>(size, in1_ptr, in2_ptr, out_ptr);
    }
    return;
  }
  
  // FP32 path
  int32_t thread_num = 512;
  int32_t block_num = (size + thread_num - 1) / thread_num;
  if (stream) {
    cudaStream_t stream_ = static_cast<CUstream_st*>(stream);
    add_kernel_cu_fp32<<<block_num, thread_num, 0, stream_>>>(
        size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
  } else {
    add_kernel_cu_fp32<<<block_num, thread_num>>>(size, input1.ptr<float>(), input2.ptr<float>(),
                                                  const_cast<float*>(output.ptr<float>()));
  }
}

void add_kernel_cu_pure_fp16(const tensor::Tensor& input1, const tensor::Tensor& input2,
                              const tensor::Tensor& output, void* stream) {
  CHECK_EQ(input1.is_empty(), false);
  CHECK_EQ(input2.is_empty(), false);
  CHECK_EQ(output.is_empty(), false);
  CHECK(input1.data_type() == base::DataType::kDataTypeFp16);
  CHECK(input2.data_type() == base::DataType::kDataTypeFp16);
  CHECK(output.data_type() == base::DataType::kDataTypeFp16);
  
  int32_t size = static_cast<int32_t>(input1.size());
  CHECK_EQ(size, input2.size());
  CHECK_EQ(size, output.size());
  
  // Process 2 elements per thread with half2
  int32_t thread_num = 256;
  int32_t elements_per_thread = 2;
  int32_t block_num = (size + thread_num * elements_per_thread - 1) / (thread_num * elements_per_thread);
  
  const half* in1_ptr = reinterpret_cast<const half*>(input1.ptr<uint16_t>());
  const half* in2_ptr = reinterpret_cast<const half*>(input2.ptr<uint16_t>());
  half* out_ptr = reinterpret_cast<half*>(const_cast<uint16_t*>(output.ptr<uint16_t>()));
  
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    add_kernel_cu_fp16_impl<<<block_num, thread_num, 0, stream_>>>(size, in1_ptr, in2_ptr, out_ptr);
  } else {
    add_kernel_cu_fp16_impl<<<block_num, thread_num>>>(size, in1_ptr, in2_ptr, out_ptr);
  }
}

// FP16 broadcast add kernel: adds bias to each row of matrix
// matrix: [rows, cols], bias: [cols], output: [rows, cols]
__global__ void broadcast_add_bias_fp16_kernel(
    const half* __restrict__ matrix,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int32_t rows,
    int32_t cols
) {
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t total = rows * cols;
  
  if (idx < total) {
    int32_t col = idx % cols;
    output[idx] = __hadd(matrix[idx], bias[col]);
  }
}

void broadcast_add_bias_fp16_cu(
    const tensor::Tensor& matrix,
    const tensor::Tensor& bias,
    const tensor::Tensor& output,
    int32_t rows,
    int32_t cols,
    void* stream
) {
  CHECK(!matrix.is_empty());
  CHECK(!bias.is_empty());
  CHECK(!output.is_empty());
  CHECK(matrix.data_type() == base::DataType::kDataTypeFp16);
  CHECK(bias.data_type() == base::DataType::kDataTypeFp16);
  CHECK(output.data_type() == base::DataType::kDataTypeFp16);
  CHECK_EQ(bias.size(), cols);
  CHECK_EQ(matrix.size(), rows * cols);
  CHECK_EQ(output.size(), rows * cols);
  
  int32_t total = rows * cols;
  int32_t thread_num = 256;
  int32_t block_num = (total + thread_num - 1) / thread_num;
  
  const half* matrix_ptr = reinterpret_cast<const half*>(matrix.ptr<uint16_t>());
  const half* bias_ptr = reinterpret_cast<const half*>(bias.ptr<uint16_t>());
  half* output_ptr = reinterpret_cast<half*>(const_cast<uint16_t*>(output.ptr<uint16_t>()));
  
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    broadcast_add_bias_fp16_kernel<<<block_num, thread_num, 0, stream_>>>(
        matrix_ptr, bias_ptr, output_ptr, rows, cols);
  } else {
    broadcast_add_bias_fp16_kernel<<<block_num, thread_num>>>(
        matrix_ptr, bias_ptr, output_ptr, rows, cols);
  }
}

// Simple FP16 vector add kernel
__global__ void add_vec_fp16_kernel(half* a, const half* b, half* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = __hadd(a[idx], b[idx]);
  }
}

void add_cu(half* a, const half* b, half* output, int n, void* stream) {
  int thread_num = 256;
  int block_num = (n + thread_num - 1) / thread_num;
  
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    add_vec_fp16_kernel<<<block_num, thread_num, 0, stream_>>>(a, b, output, n);
  } else {
    add_vec_fp16_kernel<<<block_num, thread_num>>>(a, b, output, n);
  }
}

}  // namespace kernel
