#include <tensor/tensor.h>
#include <cub/block/block_reduce.cuh>
#include <cuda_fp16.h>
#include <mma.h>
#include "../kernels_interface.h"
#include "matmul_kernel.cuh"
#include "optimized_fp16_kernels.cuh"
namespace kernel {

// Forward declaration for FP32 to FP16 conversion kernel (vectorized)
__global__ void fp32_to_fp16_kernel(const float* __restrict__ input, 
                                     half* __restrict__ output,
                                     const int size);

// FP16 to FP32 conversion kernel - OPTIMIZED with 4-element vectorized ops
__global__ void fp16_to_fp32_kernel(const half* __restrict__ input, 
                                     float* __restrict__ output,
                                     const int size) {
    // Process 4 elements at a time with float4 output
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < size) {
        const half2* h2_ptr = reinterpret_cast<const half2*>(input + idx);
        half2 in0 = __ldg(h2_ptr);
        half2 in1 = __ldg(h2_ptr + 1);
        float2 f0 = __half22float2(in0);
        float2 f1 = __half22float2(in1);
        float4 out = make_float4(f0.x, f0.y, f1.x, f1.y);
        *reinterpret_cast<float4*>(output + idx) = out;
    } else {
        // Handle remainder
        for (int i = idx; i < size; i++) {
            output[i] = __half2float(__ldg(input + i));
        }
    }
}

template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32(const float* __restrict__ input, const float* __restrict__ weight, float* output, int M,
                                      int K) {
  unsigned int tid = threadIdx.x;

  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }

  constexpr int pack_size = 4;
  const int pack_num = M / pack_size;
  const int pack_off = pack_size * pack_num;

#pragma unroll
  for (int p = start_row; p < end_row; ++p) {
    float sum = 0.0f;
    int row_offset = p * M;
    const float4* input_float4_ptr = reinterpret_cast<const float4*>(input);
    const float4* weight_float4_ptr = reinterpret_cast<const float4*>(weight + row_offset);

#pragma unroll
    for (int i = tid; i < pack_num; i += blockDim.x) {
      float4 input_float4 = input_float4_ptr[i];
      float4 weight_float4 = __ldg(weight_float4_ptr + i);
      sum = fmaf(input_float4.x, weight_float4.x, sum);
      sum = fmaf(input_float4.y, weight_float4.y, sum);
      sum = fmaf(input_float4.z, weight_float4.z, sum);
      sum = fmaf(input_float4.w, weight_float4.w, sum);
    }

    for (int i = pack_off + tid; i < M; i += blockDim.x) {
      sum = fmaf(input[i], __ldg(weight + row_offset + i), sum);
    }

    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sum);
    __syncthreads();

    if (tid == 0) {
      output[p] = part_sum;
    }
    __syncthreads();
  }
}

template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32int8(const float* __restrict__ input, const int8_t* __restrict__ weight,
                                          const float* __restrict__ scales, const int32_t group_size,
                                          float* output, int M, int K) {
  unsigned int tid = threadIdx.x;

  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }

  constexpr int pack_size = 4;
  const int pack_num = M / pack_size;
  const int pack_off = pack_size * pack_num;

  for (int p = start_row; p < end_row; ++p) {
    float sum = 0.0f;
    const int row_offset = p * M;
    const char4* weight_char4 = reinterpret_cast<const char4*>(weight + row_offset);
    const float4* input_float4 = reinterpret_cast<const float4*>(input);

    #pragma unroll
    for (int i = tid; i < pack_num; i += THREAD_PER_BLOCK) {
      char4 w4 = __ldg(weight_char4 + i);
      float4 x4 = input_float4[i];
      int base_idx = row_offset + i * 4;
      float s0 = __ldg(&scales[base_idx / group_size]);
      float s1 = __ldg(&scales[(base_idx + 1) / group_size]);
      float s2 = __ldg(&scales[(base_idx + 2) / group_size]);
      float s3 = __ldg(&scales[(base_idx + 3) / group_size]);
      sum = fmaf(x4.x, s0 * static_cast<float>(w4.x), sum);
      sum = fmaf(x4.y, s1 * static_cast<float>(w4.y), sum);
      sum = fmaf(x4.z, s2 * static_cast<float>(w4.z), sum);
      sum = fmaf(x4.w, s3 * static_cast<float>(w4.w), sum);
    }

    for (int i = pack_off + tid; i < M; i += THREAD_PER_BLOCK) {
      const int weight_idx = row_offset + i;
      sum = fmaf(input[i], __ldg(&scales[weight_idx / group_size]) * static_cast<float>(__ldg(weight + weight_idx)), sum);
    }

    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sum);
    __syncthreads();

    if (tid == 0) {
      output[p] = part_sum;
    }
    __syncthreads();
  }
}

/**
 * Optimized GEMV for decode phase (single token): y = W * x
 * Weight layout: [K, M] (row-major)
 * Uses custom CUDA kernel for row-major data layout
 * Note: cuBLAS SGEMV expects column-major which doesn't match our row-major storage,
 *       so we use custom kernel for correctness.
 */
void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, const float scale, const CudaConfig* config) {
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  const int32_t K = weight.get_dim(0);  // output dim (rows)
  const int32_t M = weight.get_dim(1);  // input dim (cols)

  CHECK_EQ(M, input.get_dim(0));
  
  cudaStream_t stream = config ? config->stream : nullptr;
  
  // Use custom kernel optimized for row-major data layout
  if (stream) {
    matmul_kernel_cu_fp32<128, 1><<<K, 128, 0, stream>>>(
        input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), M, K);
  } else {
    matmul_kernel_cu_fp32<128, 1><<<K, 128>>>(input.ptr<float>(), weight.ptr<float>(),
                                              const_cast<float*>(output.ptr<float>()), M, K);
  }
}

void matmul_kernel_cu_qint8(const tensor::Tensor& input, const tensor::Tensor& weight,
                            const tensor::Tensor& output, int32_t group_size,
                            const tensor::Tensor& scale, const CudaConfig* config) {
  CHECK(config != nullptr);
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  const int32_t K = weight.get_dim(0);  // row
  const int32_t M = weight.get_dim(1);  // col
  int packet_size = 4;
  CHECK_EQ(M % packet_size, 0);
  CHECK_EQ(M, input.get_dim(0));
  if (config->stream) {
    matmul_kernel_cu_fp32int8<128, 1><<<K, 128, 0, config->stream>>>(
        input.ptr<float>(), weight.ptr<int8_t>(), scale.ptr<float>(), group_size,
        const_cast<float*>(output.ptr<float>()), M, K);
  } else {
    matmul_kernel_cu_fp32int8<128, 1><<<K, 128>>>(input.ptr<float>(), weight.ptr<int8_t>(),
                                                  scale.ptr<float>(), group_size,
                                                  const_cast<float*>(output.ptr<float>()), M, K);
  }
}

// Batched matmul kernel for batch prefill: input [batch_size, M], weight [K, M], output [batch_size, K]
template <int THREAD_PER_BLOCK>
__global__ void batched_matmul_kernel_cu_fp32(const float* __restrict__ input, const float* __restrict__ weight, float* output,
                                              int batch_size, int M, int K) {
  int row = blockIdx.x;
  int batch_idx = blockIdx.y;
  
  if (row >= K || batch_idx >= batch_size) {
    return;
  }

  unsigned int tid = threadIdx.x;
  
  const float* batch_input = input + batch_idx * M;
  float* batch_output = output + batch_idx * K;
  
  constexpr int pack_size = 4;
  const int pack_num = M / pack_size;
  const int pack_off = pack_size * pack_num;
  
  float sum = 0.0f;
  int row_offset = row * M;
  const float4* input_float4_ptr = reinterpret_cast<const float4*>(batch_input);
  const float4* weight_float4_ptr = reinterpret_cast<const float4*>(weight + row_offset);

#pragma unroll
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 input_float4 = input_float4_ptr[i];
    float4 weight_float4 = __ldg(weight_float4_ptr + i);
    sum = fmaf(input_float4.x, weight_float4.x, sum);
    sum = fmaf(input_float4.y, weight_float4.y, sum);
    sum = fmaf(input_float4.z, weight_float4.z, sum);
    sum = fmaf(input_float4.w, weight_float4.w, sum);
  }

  for (int i = pack_off + tid; i < M; i += blockDim.x) {
    sum = fmaf(batch_input[i], __ldg(weight + row_offset + i), sum);
  }

  using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
  __shared__ typename BlockReduce::TempStorage temp;
  float part_sum = BlockReduce(temp).Sum(sum);

  if (tid == 0) {
    batch_output[row] = part_sum;
  }
}

/**
 * Batched GEMM kernel for FP16 weight (prefill phase) - OPTIMIZED VERSION
 * Mixed precision: FP16 weight x FP32 input -> FP32 output
 * Uses half2 + float2 vectorized loads for 2x memory bandwidth
 * Enhanced unrolling for better instruction-level parallelism
 */
template <int THREAD_PER_BLOCK>
__global__ void batched_matmul_fp16_weight_kernel(
    const float* __restrict__ input, 
    const half* __restrict__ weight, 
    float* __restrict__ output,
    const int batch_size, const int M, const int K
) {
    const int row = blockIdx.x;
    const int batch_idx = blockIdx.y;
    
    if (row >= K || batch_idx >= batch_size) return;
    
    const unsigned int tid = threadIdx.x;
    
    const float* batch_input = input + static_cast<int64_t>(batch_idx) * M;
    float* batch_output = output + static_cast<int64_t>(batch_idx) * K;
    
    const half* row_ptr = weight + static_cast<int64_t>(row) * M;
    
    // Use 4-element vectorized weight loads (2 x half2 = 4 halfs = 8 bytes)
    const int num_h4 = M / 4;
    
    float sum = 0.0f;
    
    #pragma unroll 4
    for (int i = tid; i < num_h4; i += THREAD_PER_BLOCK) {
        const half2* h2_ptr = reinterpret_cast<const half2*>(row_ptr + i * 4);
        half2 w0 = __ldg(h2_ptr);
        half2 w1 = __ldg(h2_ptr + 1);
        float4 x = *reinterpret_cast<const float4*>(batch_input + i * 4);
        float2 fw0 = __half22float2(w0);
        float2 fw1 = __half22float2(w1);
        sum = fmaf(fw0.x, x.x, sum);
        sum = fmaf(fw0.y, x.y, sum);
        sum = fmaf(fw1.x, x.z, sum);
        sum = fmaf(fw1.y, x.w, sum);
    }
    
    // Handle remaining elements
    const int base = num_h4 * 4;
    for (int i = base + tid; i < M; i += THREAD_PER_BLOCK) {
        sum = fmaf(__half2float(__ldg(row_ptr + i)), batch_input[i], sum);
    }
    
    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sum);
    
    if (tid == 0) {
        batch_output[row] = part_sum;
    }
}

/**
 * FP16 weight GEMV for decode phase - ULTRA OPTIMIZED VERSION
 * Uses custom kernel for mixed precision (FP16 weight x FP32 input -> FP32 output)
 * 
 * Optimizations applied:
 * - Aggressive unrolling for instruction-level parallelism
 * - half2 vectorized loads for 2x memory bandwidth
 * - __ldg() intrinsic for read-only cache utilization
 * - Multiple elements per thread to increase arithmetic intensity
 */
void matmul_kernel_cu_fp16_weight(const tensor::Tensor& input, const tensor::Tensor& weight,
                                   const tensor::Tensor& output, const float scale, 
                                   const CudaConfig* config) {
    CHECK(input.is_empty() == false && input.dims_size() <= 2);
    CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(weight.is_empty() == false && weight.dims_size() == 2);
    CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(weight.data_type() == base::DataType::kDataTypeFp16);
    
    const int32_t K = weight.get_dim(0);  // output dim
    const int32_t M = weight.get_dim(1);  // input dim
    
    CHECK_EQ(M, input.get_dim(0));
    
    cudaStream_t stream = config ? config->stream : nullptr;
    
    // For large M (like 3584 or 18944), use bandwidth-optimized kernel
    // This kernel uses full block for each row, better for memory-bound operations
    if (M >= 2048) {
        constexpr int BLOCK_SIZE = 256;
        optimized::gemv_fp16_bandwidth_optimized<BLOCK_SIZE, 1><<<K, BLOCK_SIZE, 0, stream>>>(
            input.ptr<float>(), 
            reinterpret_cast<const half*>(weight.ptr<uint16_t>()), 
            const_cast<float*>(output.ptr<float>()), M, K);
    } else {
        // For smaller M, use warp-based kernel
        constexpr int WARPS_PER_BLOCK = 8;
        constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
        const int num_blocks = (K + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        
        optimized::gemv_fp16_optimized<WARPS_PER_BLOCK, 4><<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
            input.ptr<float>(), 
            reinterpret_cast<const half*>(weight.ptr<uint16_t>()), 
            const_cast<float*>(output.ptr<float>()), M, K);
    }
}

// FP32 to FP16 conversion kernel - OPTIMIZED with 4-element vectorized ops
__global__ void fp32_to_fp16_kernel(const float* __restrict__ input, 
                                     half* __restrict__ output,
                                     const int size) {
    // Process 4 elements at a time with float4 input
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < size) {
        float4 in = __ldg(reinterpret_cast<const float4*>(input + idx));
        half2 out0 = __float22half2_rn(make_float2(in.x, in.y));
        half2 out1 = __float22half2_rn(make_float2(in.z, in.w));
        reinterpret_cast<half2*>(output + idx)[0] = out0;
        reinterpret_cast<half2*>(output + idx)[1] = out1;
    } else {
        // Handle remainder
        for (int i = idx; i < size; i++) {
            output[i] = __float2half(__ldg(input + i));
        }
    }
}

/**
 * Batched FP16 weight matmul for prefill phase - OPTIMIZED VERSION
 * Uses cuBLAS GemmEx with mixed precision for maximum throughput on Tensor Core
 * FP16 weight x FP32 input -> FP32 output
 * 
 * Strategy:
 * - For larger batch sizes: Use cuBLAS HGEMM with FP16 conversion (better Tensor Core utilization)
 * - For small batch sizes: Use custom kernel (avoids conversion overhead)
 */
void batched_matmul_kernel_cu_fp16_weight(const tensor::Tensor& input, const tensor::Tensor& weight,
                                           const tensor::Tensor& output, int32_t batch_size, 
                                           float scale, const CudaConfig* config) {
    CHECK(input.is_empty() == false);
    CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(weight.is_empty() == false && weight.dims_size() == 2);
    CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(weight.data_type() == base::DataType::kDataTypeFp16);
    
    const int32_t K = weight.get_dim(0);  // output dim
    const int32_t M = weight.get_dim(1);  // input dim
    
    cudaStream_t stream = config ? config->stream : nullptr;
    
    // Use cuBLAS HGEMM for prefill - FP16 Tensor Core is significantly faster
    if (config && config->cublas_handle && batch_size >= 8) {
        const half alpha_h = __float2half(1.0f);
        const half beta_h = __float2half(0.0f);
        
        size_t input_size = static_cast<size_t>(batch_size) * M;
        size_t output_size = static_cast<size_t>(batch_size) * K;
        
        CudaConfig* mutable_config = const_cast<CudaConfig*>(config);
        if (mutable_config->ensure_fp16_workspace(input_size, output_size)) {
            half* input_fp16 = mutable_config->fp16_input_workspace;
            half* output_fp16 = mutable_config->fp16_output_workspace;
            
            // Convert FP32 input to FP16 (vectorized, 4 elements per thread)
            int threads = 256;
            int elements_per_thread = 4;
            int blocks = (input_size + threads * elements_per_thread - 1) / (threads * elements_per_thread);
            fp32_to_fp16_kernel<<<blocks, threads, 0, stream>>>(
                input.ptr<float>(), input_fp16, input_size);
            
            // Enable Tensor Core for HGEMM
            cublasSetMathMode(config->cublas_handle, CUBLAS_TENSOR_OP_MATH);
            
            // cuBLAS HGEMM: C[K, batch] = W[K, M] @ X[M, batch]
            // Weight is stored as [K, M] row-major, which cuBLAS sees as [M, K] col-major
            // So we use CUBLAS_OP_T to get the transpose: [K, M]
            cublasStatus_t status = cublasHgemm(
                config->cublas_handle,
                CUBLAS_OP_T,          // W transpose
                CUBLAS_OP_N,          // X
                K,                    // m
                batch_size,           // n
                M,                    // k
                &alpha_h,
                reinterpret_cast<const half*>(weight.ptr<uint16_t>()),
                M,                    // lda
                input_fp16,           
                M,                    // ldb
                &beta_h,
                output_fp16,          
                K                     // ldc
            );
            
            if (status == CUBLAS_STATUS_SUCCESS) {
                // Convert FP16 output back to FP32 (4 elements per thread)
                blocks = (output_size + threads * elements_per_thread - 1) / (threads * elements_per_thread);
                fp16_to_fp32_kernel<<<blocks, threads, 0, stream>>>(
                    output_fp16, const_cast<float*>(output.ptr<float>()), output_size);
                return;
            }
        }
    }
    
    // Fallback: Use optimized custom kernel
    dim3 grid(K, batch_size);
    batched_matmul_fp16_weight_kernel<256><<<grid, 256, 0, stream>>>(
        input.ptr<float>(), 
        reinterpret_cast<const half*>(weight.ptr<uint16_t>()), 
        const_cast<float*>(output.ptr<float>()),
        batch_size, M, K);
}

/**
 * Batched Matrix Multiplication for FP32 path
 * Uses optimized custom CUDA kernel for row-major data layout
 * 
 * Input A: [batch_size, M] - batch of input vectors (FP32, row-major)
 * Weight B: [K, M] (row-major) - weight matrix (FP32)
 * Output C: [batch_size, K] - batch of output vectors (FP32, row-major)
 * 
 * Note: cuBLAS requires column-major layout which doesn't match our data format,
 * so we use custom kernel for correctness with row-major tensors.
 */
void batched_matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                              const tensor::Tensor& output, int32_t batch_size, float scale,
                              const CudaConfig* config) {
  CHECK(input.is_empty() == false);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  
  const int32_t K = weight.get_dim(0);  // output dim
  const int32_t M = weight.get_dim(1);  // input dim
  
  // Use custom kernel optimized for row-major data layout
  dim3 grid(K, batch_size);
  cudaStream_t stream = config ? config->stream : nullptr;
  if (stream) {
    batched_matmul_kernel_cu_fp32<128><<<grid, 128, 0, stream>>>(
        input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()),
        batch_size, M, K);
  } else {
    batched_matmul_kernel_cu_fp32<128><<<grid, 128>>>(
        input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()),
        batch_size, M, K);
  }
}

// ==================== Pure FP16 Kernels Implementation ====================

/**
 * Optimized Pure FP16 GEMV kernel with float4 vectorization
 * Loads 8 halfs at a time using float4 (16 bytes)
 * Uses FP32 accumulation for numerical stability
 */
template <int WARP_SIZE = 32, int WARPS_PER_BLOCK = 4>
__global__ void gemv_pure_fp16_kernel_v2(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    const int M,  // input dimension  
    const int K   // output dimension
) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    const int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    
    if (row >= K) return;
    
    const half* row_ptr = weight + static_cast<int64_t>(row) * M;
    
    // Use multiple accumulators for better ILP
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    
    // Process 8 halfs at a time using float4 (which contains 4 floats = 8 halfs)
    const int num_float4 = M / 8;
    const float4* weight_f4 = reinterpret_cast<const float4*>(row_ptr);
    const float4* input_f4 = reinterpret_cast<const float4*>(input);
    
    #pragma unroll 4
    for (int i = lane_id; i < num_float4; i += WARP_SIZE) {
        float4 w = __ldg(weight_f4 + i);
        float4 x = __ldg(input_f4 + i);
        
        // Cast to half2 pairs and accumulate with separate accumulators
        const half2* w_h2 = reinterpret_cast<const half2*>(&w);
        const half2* x_h2 = reinterpret_cast<const half2*>(&x);
        
        float2 wf0 = __half22float2(w_h2[0]);
        float2 xf0 = __half22float2(x_h2[0]);
        float2 wf1 = __half22float2(w_h2[1]);
        float2 xf1 = __half22float2(x_h2[1]);
        float2 wf2 = __half22float2(w_h2[2]);
        float2 xf2 = __half22float2(x_h2[2]);
        float2 wf3 = __half22float2(w_h2[3]);
        float2 xf3 = __half22float2(x_h2[3]);
        
        sum0 = fmaf(wf0.x, xf0.x, fmaf(wf0.y, xf0.y, sum0));
        sum1 = fmaf(wf1.x, xf1.x, fmaf(wf1.y, xf1.y, sum1));
        sum2 = fmaf(wf2.x, xf2.x, fmaf(wf2.y, xf2.y, sum2));
        sum3 = fmaf(wf3.x, xf3.x, fmaf(wf3.y, xf3.y, sum3));
    }
    
    float sum = sum0 + sum1 + sum2 + sum3;
    
    // Handle remainder (M % 8 elements) with half2
    const int base8 = num_float4 * 8;
    const int num_h2_remaining = (M - base8) / 2;
    const half2* weight_h2_rem = reinterpret_cast<const half2*>(row_ptr + base8);
    const half2* input_h2_rem = reinterpret_cast<const half2*>(input + base8);
    
    for (int i = lane_id; i < num_h2_remaining; i += WARP_SIZE) {
        float2 wf = __half22float2(__ldg(weight_h2_rem + i));
        float2 xf = __half22float2(__ldg(input_h2_rem + i));
        sum = fmaf(wf.x, xf.x, fmaf(wf.y, xf.y, sum));
    }
    
    // Handle odd remainder (if M % 2 != 0)
    const int base2 = base8 + num_h2_remaining * 2;
    for (int i = base2 + lane_id; i < M; i += WARP_SIZE) {
        sum = fmaf(__half2float(__ldg(row_ptr + i)), __half2float(__ldg(input + i)), sum);
    }
    
    // Warp reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (lane_id == 0) {
        output[row] = __float2half(sum);
    }
}

/**
 * Batched pure FP16 GEMM kernel for prefill phase
 * FP16 input x FP16 weight -> FP16 output
 * Uses float4 vectorized loads (8 halfs at a time), FP32 accumulation for precision
 */
template <int THREAD_PER_BLOCK>
__global__ void batched_gemm_pure_fp16_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    const int batch_size, const int M, const int K
) {
    const int row = blockIdx.x;
    const int batch_idx = blockIdx.y;
    
    if (row >= K || batch_idx >= batch_size) return;
    
    const unsigned int tid = threadIdx.x;
    
    const half* batch_input = input + static_cast<int64_t>(batch_idx) * M;
    half* batch_output = output + static_cast<int64_t>(batch_idx) * K;
    const half* row_ptr = weight + static_cast<int64_t>(row) * M;
    
    // Process 8 halfs at a time using float4 (16 bytes)
    const int num_float4 = M / 8;
    const float4* weight_f4 = reinterpret_cast<const float4*>(row_ptr);
    const float4* input_f4 = reinterpret_cast<const float4*>(batch_input);
    
    float sum = 0.0f;
    
    #pragma unroll 4
    for (int i = tid; i < num_float4; i += THREAD_PER_BLOCK) {
        float4 w = __ldg(weight_f4 + i);
        float4 x = __ldg(input_f4 + i);
        const half2* w_h2 = reinterpret_cast<const half2*>(&w);
        const half2* x_h2 = reinterpret_cast<const half2*>(&x);
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            float2 wf = __half22float2(w_h2[k]);
            float2 xf = __half22float2(x_h2[k]);
            sum = fmaf(wf.x, xf.x, fmaf(wf.y, xf.y, sum));
        }
    }
    
    // Handle remainder with half2
    const int base8 = num_float4 * 8;
    const int num_h2_rem = (M - base8) / 2;
    const half2* weight_h2_rem = reinterpret_cast<const half2*>(row_ptr + base8);
    const half2* input_h2_rem = reinterpret_cast<const half2*>(batch_input + base8);
    
    for (int i = tid; i < num_h2_rem; i += THREAD_PER_BLOCK) {
        float2 wf = __half22float2(__ldg(weight_h2_rem + i));
        float2 xf = __half22float2(__ldg(input_h2_rem + i));
        sum = fmaf(wf.x, xf.x, fmaf(wf.y, xf.y, sum));
    }
    
    // Handle odd remainder
    const int base2 = base8 + num_h2_rem * 2;
    for (int i = base2 + tid; i < M; i += THREAD_PER_BLOCK) {
        sum = fmaf(__half2float(__ldg(row_ptr + i)), __half2float(__ldg(batch_input + i)), sum);
    }
    
    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sum);
    
    if (tid == 0) {
        batch_output[row] = __float2half(part_sum);
    }
}

/**
 * Pure FP16 matmul for decode phase
 * Uses cuBLAS HGEMV/HGEMM for optimal Tensor Core utilization
 */
void matmul_kernel_cu_pure_fp16(const tensor::Tensor& input, const tensor::Tensor& weight,
                                 const tensor::Tensor& output, const float scale, 
                                 const CudaConfig* config) {
    CHECK(input.is_empty() == false && input.dims_size() <= 2);
    CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(input.data_type() == base::DataType::kDataTypeFp16);
    CHECK(weight.is_empty() == false && weight.dims_size() == 2);
    CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(weight.data_type() == base::DataType::kDataTypeFp16);
    CHECK(output.data_type() == base::DataType::kDataTypeFp16);
    
    const int32_t K = weight.get_dim(0);  // output dim
    const int32_t M = weight.get_dim(1);  // input dim
    
    CHECK_EQ(M, input.get_dim(0));
    
    cudaStream_t stream = config ? config->stream : nullptr;
    
    // Use optimized GEMV kernel with float4 vectorization (8 halfs per load)
    // Use 8 warps per block to maximize occupancy
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
    const int num_blocks = (K + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    
    gemv_pure_fp16_kernel_v2<32, WARPS_PER_BLOCK><<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        reinterpret_cast<const half*>(input.ptr<uint16_t>()), 
        reinterpret_cast<const half*>(weight.ptr<uint16_t>()), 
        reinterpret_cast<half*>(const_cast<uint16_t*>(output.ptr<uint16_t>())), 
        M, K);
}

/**
 * Batched pure FP16 matmul for prefill phase
 * Uses cuBLAS HGEMM for optimal Tensor Core utilization
 */
void batched_matmul_kernel_cu_pure_fp16(const tensor::Tensor& input, const tensor::Tensor& weight,
                                         const tensor::Tensor& output, int32_t batch_size, 
                                         float scale, const CudaConfig* config) {
    CHECK(input.is_empty() == false);
    CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(input.data_type() == base::DataType::kDataTypeFp16);
    CHECK(weight.is_empty() == false && weight.dims_size() == 2);
    CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(weight.data_type() == base::DataType::kDataTypeFp16);
    CHECK(output.data_type() == base::DataType::kDataTypeFp16);
    
    const int32_t K = weight.get_dim(0);  // output dim
    const int32_t M = weight.get_dim(1);  // input dim
    
    cudaStream_t stream = config ? config->stream : nullptr;
    
    // Use cuBLAS HGEMM for batched prefill
    if (config && config->cublas_handle) {
        const half alpha = __float2half(scale);
        const half beta = __float2half(0.0f);
        
        // Enable Tensor Core
        cublasSetMathMode(config->cublas_handle, CUBLAS_TENSOR_OP_MATH);
        
        // cuBLAS HGEMM: C[K, batch] = W[K, M] @ X[M, batch]
        cublasStatus_t status = cublasHgemm(
            config->cublas_handle,
            CUBLAS_OP_T,          // W^T
            CUBLAS_OP_N,          // X
            K,                    // m
            batch_size,           // n
            M,                    // k
            &alpha,
            reinterpret_cast<const half*>(weight.ptr<uint16_t>()),
            M,                    // lda
            reinterpret_cast<const half*>(input.ptr<uint16_t>()),
            M,                    // ldb
            &beta,
            reinterpret_cast<half*>(const_cast<uint16_t*>(output.ptr<uint16_t>())),
            K                     // ldc
        );
        
        if (status == CUBLAS_STATUS_SUCCESS) {
            return;
        }
    }
    
    // Fallback: custom kernel
    dim3 grid(K, batch_size);
    batched_gemm_pure_fp16_kernel<256><<<grid, 256, 0, stream>>>(
        reinterpret_cast<const half*>(input.ptr<uint16_t>()), 
        reinterpret_cast<const half*>(weight.ptr<uint16_t>()), 
        reinterpret_cast<half*>(const_cast<uint16_t*>(output.ptr<uint16_t>())),
        batch_size, M, K);
}

/**
 * FP16 input × FP16 weight → FP32 output kernel
 * Used for cls_logits layer where we have FP16 activations but need FP32 logits
 * Strategy: Convert FP16 input to FP32 on-the-fly during GEMV computation
 */
template<int WARPS_PER_BLOCK, int ELEMENTS_PER_THREAD>
__global__ void gemv_fp16_input_fp16_weight_fp32_output(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    float* __restrict__ output,
    const int M,
    const int K) {
    
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    
    if (row >= K) return;
    
    const half* row_ptr = weight + static_cast<int64_t>(row) * M;
    
    // Use multiple accumulators for ILP
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    
    // Process 8 halfs at a time using float4 (16 bytes)
    const int num_float4 = M / 8;
    const float4* weight_f4 = reinterpret_cast<const float4*>(row_ptr);
    const float4* input_f4 = reinterpret_cast<const float4*>(input);
    
    #pragma unroll 4
    for (int i = lane_id; i < num_float4; i += 32) {
        float4 w = __ldg(weight_f4 + i);
        float4 x = __ldg(input_f4 + i);
        
        const half2* w_h2 = reinterpret_cast<const half2*>(&w);
        const half2* x_h2 = reinterpret_cast<const half2*>(&x);
        
        float2 wf0 = __half22float2(w_h2[0]);
        float2 xf0 = __half22float2(x_h2[0]);
        float2 wf1 = __half22float2(w_h2[1]);
        float2 xf1 = __half22float2(x_h2[1]);
        float2 wf2 = __half22float2(w_h2[2]);
        float2 xf2 = __half22float2(x_h2[2]);
        float2 wf3 = __half22float2(w_h2[3]);
        float2 xf3 = __half22float2(x_h2[3]);
        
        sum0 = fmaf(wf0.x, xf0.x, fmaf(wf0.y, xf0.y, sum0));
        sum1 = fmaf(wf1.x, xf1.x, fmaf(wf1.y, xf1.y, sum1));
        sum2 = fmaf(wf2.x, xf2.x, fmaf(wf2.y, xf2.y, sum2));
        sum3 = fmaf(wf3.x, xf3.x, fmaf(wf3.y, xf3.y, sum3));
    }
    
    float sum = sum0 + sum1 + sum2 + sum3;
    
    // Handle remainder with half2
    const int base8 = num_float4 * 8;
    const int num_h2_rem = (M - base8) / 2;
    const half2* weight_h2_rem = reinterpret_cast<const half2*>(row_ptr + base8);
    const half2* input_h2_rem = reinterpret_cast<const half2*>(input + base8);
    
    for (int i = lane_id; i < num_h2_rem; i += 32) {
        float2 wf = __half22float2(__ldg(weight_h2_rem + i));
        float2 xf = __half22float2(__ldg(input_h2_rem + i));
        sum = fmaf(wf.x, xf.x, fmaf(wf.y, xf.y, sum));
    }
    
    // Handle odd remainder
    const int base2 = base8 + num_h2_rem * 2;
    if (lane_id == 0 && base2 < M) {
        float in_v = __half2float(__ldg(&input[base2]));
        float w_v = __half2float(__ldg(&row_ptr[base2]));
        sum = fmaf(in_v, w_v, sum);
    }
    
    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (lane_id == 0) {
        output[row] = sum;
    }
}

void matmul_kernel_cu_fp16_input_fp16_weight(const tensor::Tensor& input, const tensor::Tensor& weight,
                                              const tensor::Tensor& output, const float scale,
                                              const CudaConfig* config) {
    CHECK(input.is_empty() == false && input.dims_size() <= 2);
    CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(input.data_type() == base::DataType::kDataTypeFp16);
    CHECK(weight.is_empty() == false && weight.dims_size() == 2);
    CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(weight.data_type() == base::DataType::kDataTypeFp16);
    CHECK(output.data_type() == base::DataType::kDataTypeFp32);
    
    const int32_t K = weight.get_dim(0);  // output dim
    const int32_t M = weight.get_dim(1);  // input dim
    
    CHECK_EQ(M, input.get_dim(0));
    
    cudaStream_t stream = config ? config->stream : nullptr;
    
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
    const int num_blocks = (K + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    
    gemv_fp16_input_fp16_weight_fp32_output<WARPS_PER_BLOCK, 4><<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        reinterpret_cast<const half*>(input.ptr<uint16_t>()),
        reinterpret_cast<const half*>(weight.ptr<uint16_t>()),
        const_cast<float*>(output.ptr<float>()),
        M, K);
}

// Batched version: input [batch_size, dim], weight [vocab_size, dim], output [batch_size, vocab_size]
// Uses cuBLAS GEMM for efficient batch processing
void batched_matmul_kernel_cu_fp16_input_fp16_weight(const tensor::Tensor& input, const tensor::Tensor& weight,
                                                      const tensor::Tensor& output, int32_t batch_size,
                                                      const float scale, const CudaConfig* config) {
    CHECK(input.is_empty() == false);
    CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(input.data_type() == base::DataType::kDataTypeFp16);
    CHECK(weight.is_empty() == false && weight.dims_size() == 2);
    CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(weight.data_type() == base::DataType::kDataTypeFp16);
    CHECK(output.data_type() == base::DataType::kDataTypeFp32);
    
    const int32_t vocab_size = weight.get_dim(0);  // output dim (K)
    const int32_t dim = weight.get_dim(1);         // input dim (M)
    
    cudaStream_t stream = config ? config->stream : nullptr;
    cublasHandle_t handle = config ? config->cublas_handle : nullptr;
    
    if (handle) {
        // Use cuBLAS GEMM: C = alpha * A * B^T + beta * C
        // A: input [batch_size, dim], B: weight [vocab_size, dim] (row major)
        // C: output [batch_size, vocab_size]
        // In cuBLAS column-major: we compute C^T = B * A^T
        // So: m=vocab_size, n=batch_size, k=dim
        const float alpha = scale;
        const float beta = 0.0f;
        
        // cuBLAS GemmEx for mixed precision
        cublasGemmEx(handle,
                     CUBLAS_OP_T,  // B^T
                     CUBLAS_OP_N,  // A
                     vocab_size,   // m
                     batch_size,   // n
                     dim,          // k
                     &alpha,
                     weight.ptr<void>(), CUDA_R_16F, dim,      // B: [vocab_size, dim] -> transposed
                     input.ptr<void>(), CUDA_R_16F, dim,       // A: [batch_size, dim]
                     &beta,
                     const_cast<void*>(output.ptr<void>()), CUDA_R_32F, vocab_size,  // C: [batch_size, vocab_size]
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    } else {
        // Fallback: process each row individually
        for (int32_t b = 0; b < batch_size; ++b) {
            const half* input_row = reinterpret_cast<const half*>(input.ptr<uint16_t>()) + b * dim;
            float* output_row = const_cast<float*>(output.ptr<float>()) + b * vocab_size;
            
            constexpr int WARPS_PER_BLOCK = 8;
            constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
            const int num_blocks = (vocab_size + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
            
            gemv_fp16_input_fp16_weight_fp32_output<WARPS_PER_BLOCK, 4><<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
                input_row,
                reinterpret_cast<const half*>(weight.ptr<uint16_t>()),
                output_row,
                dim, vocab_size);
        }
    }
}

}  // namespace kernel