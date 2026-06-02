#include "add_kernel.cuh"
#include <cuda_fp16.h>

namespace kernel {

// ============================================================================
// Optimized CUDA kernels for NVIDIA Orin (SM 8.7 Ampere)
// Key optimizations:
//   1. 128-bit (float4) vectorized memory access for max bandwidth
//   2. __ldg() read-only cache intrinsic for input data
//   3. __restrict__ pointer aliasing hints for compiler optimization
//   4. 2D grid for broadcast kernel (eliminates expensive integer modulo)
//   5. #pragma unroll for inner compute loops
// ============================================================================

// FP32 vectorized add: float4 (128-bit) = 4 floats per thread
__global__ void add_kernel_cu_fp32(int32_t size, const float* __restrict__ in1,
                                   const float* __restrict__ in2, float* __restrict__ out) {
  const int VEC = 4;
  int32_t idx = (threadIdx.x + blockDim.x * blockIdx.x) * VEC;

  if (idx + (VEC - 1) < size) {
    float4 a = __ldg(reinterpret_cast<const float4*>(in1 + idx));
    float4 b = __ldg(reinterpret_cast<const float4*>(in2 + idx));
    float4 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    c.z = a.z + b.z;
    c.w = a.w + b.w;
    *reinterpret_cast<float4*>(out + idx) = c;
  } else {
    // Scalar tail for remaining 0-3 elements
    #pragma unroll
    for (int32_t i = idx; i < size; i++) {
      out[i] = __ldg(in1 + i) + __ldg(in2 + i);
    }
  }
}

// FP16 vectorized add: float4 (128-bit) = 8 halfs per thread, computed as 4x half2
__global__ void add_kernel_cu_fp16_impl(int32_t size, const half* __restrict__ in1,
                                        const half* __restrict__ in2, half* __restrict__ out) {
  const int VEC = 8;  // 8 halfs = 16 bytes = 128 bits
  int32_t idx = (threadIdx.x + blockDim.x * blockIdx.x) * VEC;

  if (idx + (VEC - 1) < size) {
    float4 a4 = __ldg(reinterpret_cast<const float4*>(in1 + idx));
    float4 b4 = __ldg(reinterpret_cast<const float4*>(in2 + idx));

    half2* a = reinterpret_cast<half2*>(&a4);
    half2* b = reinterpret_cast<half2*>(&b4);
    float4 c4;
    half2* c = reinterpret_cast<half2*>(&c4);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
      c[i] = __hadd2(a[i], b[i]);
    }

    *reinterpret_cast<float4*>(out + idx) = c4;
  } else {
    // Scalar tail for remaining 0-7 elements
    for (int32_t i = idx; i < size; i++) {
      out[i] = __hadd(in1[i], in2[i]);
    }
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
    // 8 halfs per thread (128-bit vectorized)
    int32_t thread_num = 256;
    int32_t elements_per_thread = 8;
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

  // FP32 path: 4 floats per thread (128-bit vectorized)
  int32_t thread_num = 256;
  int32_t elements_per_thread = 4;
  int32_t block_num = (size + thread_num * elements_per_thread - 1) / (thread_num * elements_per_thread);
  if (stream) {
    cudaStream_t stream_ = static_cast<CUstream_st*>(stream);
    add_kernel_cu_fp32<<<block_num, thread_num, 0, stream_>>>(
        size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
  } else {
    add_kernel_cu_fp32<<<block_num, thread_num>>>(size, input1.ptr<float>(), input2.ptr<float>(),
                                                  const_cast<float*>(output.ptr<float>()));
  }
}

// FP16 broadcast add bias: 2D grid eliminates integer modulo
// matrix: [rows, cols], bias: [cols], output: [rows, cols]
// blockIdx.y = row index, blockIdx.x * blockDim.x + threadIdx.x = col block
// Each thread processes 8 consecutive columns via float4 (128-bit) vectorized access
__global__ void broadcast_add_bias_fp16_kernel(
    const half* __restrict__ matrix,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int32_t rows,
    int32_t cols
) {
  const int VEC = 8;
  int32_t col_base = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
  int32_t row = blockIdx.y;

  if (row >= rows || col_base >= cols) return;

  int32_t idx = row * cols + col_base;

  if (col_base + VEC <= cols) {
    // Vectorized path: load 8 halfs at once
    float4 m = __ldg(reinterpret_cast<const float4*>(matrix + idx));
    float4 b = __ldg(reinterpret_cast<const float4*>(bias + col_base));

    half2* mh = reinterpret_cast<half2*>(&m);
    half2* bh = reinterpret_cast<half2*>(&b);
    float4 result;
    half2* rh = reinterpret_cast<half2*>(&result);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
      rh[i] = __hadd2(mh[i], bh[i]);
    }

    *reinterpret_cast<float4*>(output + idx) = result;
  } else {
    // Scalar tail for remaining columns
    for (int32_t c = col_base; c < cols; c++) {
      int32_t i = row * cols + c;
      output[i] = __hadd(__ldg(matrix + i), __ldg(bias + c));
    }
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

  int32_t thread_num = 256;
  int32_t elements_per_thread = 8;
  // 2D grid: x-dim covers columns, y-dim covers rows
  int32_t col_blocks = (cols + thread_num * elements_per_thread - 1) / (thread_num * elements_per_thread);
  dim3 grid(col_blocks, rows);

  const half* matrix_ptr = reinterpret_cast<const half*>(matrix.ptr<uint16_t>());
  const half* bias_ptr = reinterpret_cast<const half*>(bias.ptr<uint16_t>());
  half* output_ptr = reinterpret_cast<half*>(const_cast<uint16_t*>(output.ptr<uint16_t>()));

  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    broadcast_add_bias_fp16_kernel<<<grid, thread_num, 0, stream_>>>(
        matrix_ptr, bias_ptr, output_ptr, rows, cols);
  } else {
    broadcast_add_bias_fp16_kernel<<<grid, thread_num>>>(
        matrix_ptr, bias_ptr, output_ptr, rows, cols);
  }
}

// FP16 vector add: float4 (128-bit) = 8 halfs per thread
__global__ void add_vec_fp16_kernel(half* __restrict__ a, const half* __restrict__ b,
                                    half* __restrict__ output, int n) {
  const int VEC = 8;
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;

  if (idx + (VEC - 1) < n) {
    // Use __ldg for both inputs through readonly cache
    float4 av = __ldg(reinterpret_cast<const float4*>(a + idx));
    float4 bv = __ldg(reinterpret_cast<const float4*>(b + idx));

    half2* ah = reinterpret_cast<half2*>(&av);
    half2* bh = reinterpret_cast<half2*>(&bv);
    float4 cv;
    half2* ch = reinterpret_cast<half2*>(&cv);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
      ch[i] = __hadd2(ah[i], bh[i]);
    }

    *reinterpret_cast<float4*>(output + idx) = cv;
  } else {
    // Scalar tail for remaining 0-7 elements
    for (int i = idx; i < n; i++) {
      output[i] = __hadd(a[i], b[i]);
    }
  }
}

void add_cu(half* a, const half* b, half* output, int n, void* stream) {
  int thread_num = 256;
  int elements_per_thread = 8;
  int block_num = (n + thread_num * elements_per_thread - 1) / (thread_num * elements_per_thread);

  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    add_vec_fp16_kernel<<<block_num, thread_num, 0, stream_>>>(a, b, output, n);
  } else {
    add_vec_fp16_kernel<<<block_num, thread_num>>>(a, b, output, n);
  }
}

}  // namespace kernel
