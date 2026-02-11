/**
 * @file fp16_gemv_kernel.cu
 * @brief Highly optimized FP16 GEMV kernels for decode phase
 * 
 * Key optimizations:
 * 1. FP16 native input/output to reduce memory bandwidth
 * 2. half2 vectorized operations
 * 3. Warp-level reduction using shuffle
 * 4. Multiple rows per block for better occupancy
 * 
 * This kernel is designed to match llama.cpp performance.
 */
#include <cuda_fp16.h>
#include <cub/block/block_reduce.cuh>
#include "fp16_gemv_kernel.cuh"

namespace kernel {

/**
 * Highly optimized FP16 GEMV kernel for decode phase
 * FP16 weight x FP16 input -> FP16/FP32 output
 * 
 * This kernel processes multiple rows per block using warp-level parallelism.
 * Each warp handles one output element with vectorized half2 operations.
 */
template <int WARP_SIZE = 32, int WARPS_PER_BLOCK = 8>
__global__ void fp16_gemv_kernel_optimized(
    const half* __restrict__ input,      // [M]
    const half* __restrict__ weight,     // [K, M]
    half* __restrict__ output,           // [K]
    const int M,                         // input dimension
    const int K                          // output dimension
) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    const int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    
    if (row >= K) return;
    
    const half* row_ptr = weight + static_cast<int64_t>(row) * M;
    
    // Use half2 for vectorized operations (2x throughput)
    const int num_h2 = M / 2;
    const half2* input_h2 = reinterpret_cast<const half2*>(input);
    const half2* weight_h2 = reinterpret_cast<const half2*>(row_ptr);
    
    // Accumulate in FP32 for numerical stability
    float sum = 0.0f;
    
    // Main loop: process 2 elements per iteration with half2
    #pragma unroll 4
    for (int i = lane_id; i < num_h2; i += WARP_SIZE) {
        half2 w = weight_h2[i];
        half2 x = input_h2[i];
        
        // Convert to float for accumulation
        float2 wf = __half22float2(w);
        float2 xf = __half22float2(x);
        sum += wf.x * xf.x + wf.y * xf.y;
    }
    
    // Handle remainder (if M is odd)
    const int base = num_h2 * 2;
    for (int i = base + lane_id; i < M; i += WARP_SIZE) {
        sum += __half2float(row_ptr[i]) * __half2float(input[i]);
    }
    
    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // Lane 0 writes the result
    if (lane_id == 0) {
        output[row] = __float2half(sum);
    }
}

/**
 * FP16 GEMV with FP32 output for better precision
 * Useful when output needs to be FP32 (e.g., for RMSNorm)
 */
template <int WARP_SIZE = 32, int WARPS_PER_BLOCK = 8>
__global__ void fp16_gemv_fp32_output_kernel(
    const half* __restrict__ input,      // [M] FP16
    const half* __restrict__ weight,     // [K, M] FP16
    float* __restrict__ output,          // [K] FP32
    const int M,
    const int K
) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    const int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    
    if (row >= K) return;
    
    const half* row_ptr = weight + static_cast<int64_t>(row) * M;
    
    const int num_h2 = M / 2;
    const half2* input_h2 = reinterpret_cast<const half2*>(input);
    const half2* weight_h2 = reinterpret_cast<const half2*>(row_ptr);
    
    float sum = 0.0f;
    
    #pragma unroll 4
    for (int i = lane_id; i < num_h2; i += WARP_SIZE) {
        half2 w = weight_h2[i];
        half2 x = input_h2[i];
        float2 wf = __half22float2(w);
        float2 xf = __half22float2(x);
        sum += wf.x * xf.x + wf.y * xf.y;
    }
    
    const int base = num_h2 * 2;
    for (int i = base + lane_id; i < M; i += WARP_SIZE) {
        sum += __half2float(row_ptr[i]) * __half2float(input[i]);
    }
    
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (lane_id == 0) {
        output[row] = sum;
    }
}

/**
 * Large M kernel with shared memory caching for FP16 input
 * Better for large input dimensions where input doesn't fit in L1 cache
 */
template <int BLOCK_SIZE = 256>
__global__ void fp16_gemv_large_m_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    const int M,
    const int K
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (row >= K) return;
    
    // Shared memory for input caching
    extern __shared__ half s_input[];
    
    // Cooperatively load input to shared memory
    for (int i = tid; i < M; i += BLOCK_SIZE) {
        s_input[i] = input[i];
    }
    __syncthreads();
    
    const half* row_ptr = weight + static_cast<int64_t>(row) * M;
    
    const int num_h2 = M / 2;
    const half2* weight_h2 = reinterpret_cast<const half2*>(row_ptr);
    const half2* shared_h2 = reinterpret_cast<const half2*>(s_input);
    
    float sum = 0.0f;
    
    #pragma unroll 4
    for (int i = tid; i < num_h2; i += BLOCK_SIZE) {
        half2 w = weight_h2[i];
        half2 x = shared_h2[i];
        float2 wf = __half22float2(w);
        float2 xf = __half22float2(x);
        sum += wf.x * xf.x + wf.y * xf.y;
    }
    
    const int base = num_h2 * 2;
    for (int i = base + tid; i < M; i += BLOCK_SIZE) {
        sum += __half2float(row_ptr[i]) * __half2float(s_input[i]);
    }
    
    // Block-level reduction
    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    sum = BlockReduce(temp_storage).Sum(sum);
    
    if (tid == 0) {
        output[row] = __float2half(sum);
    }
}

/**
 * Mixed precision GEMV: FP16 weight x FP32 input -> FP32 output
 * Used when input activations are in FP32 format
 */
template <int WARP_SIZE = 32, int WARPS_PER_BLOCK = 8>
__global__ void fp16_weight_fp32_io_gemv_kernel(
    const float* __restrict__ input,     // [M] FP32
    const half* __restrict__ weight,     // [K, M] FP16
    float* __restrict__ output,          // [K] FP32
    const int M,
    const int K
) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    const int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    
    if (row >= K) return;
    
    const half* row_ptr = weight + static_cast<int64_t>(row) * M;
    
    const int num_h2 = M / 2;
    const half2* weight_h2 = reinterpret_cast<const half2*>(row_ptr);
    const float2* input_f2 = reinterpret_cast<const float2*>(input);
    
    float sum = 0.0f;
    
    #pragma unroll 4
    for (int i = lane_id; i < num_h2; i += WARP_SIZE) {
        half2 w = weight_h2[i];
        float2 x = input_f2[i];
        float2 wf = __half22float2(w);
        sum += wf.x * x.x + wf.y * x.y;
    }
    
    const int base = num_h2 * 2;
    for (int i = base + lane_id; i < M; i += WARP_SIZE) {
        sum += __half2float(row_ptr[i]) * input[i];
    }
    
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (lane_id == 0) {
        output[row] = sum;
    }
}

// ===================== Host API Functions =====================

void fp16_gemv_kernel_cu(
    const half* input, const half* weight, half* output,
    int M, int K, cudaStream_t stream
) {
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
    const int num_blocks = (K + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    
    fp16_gemv_kernel_optimized<32, WARPS_PER_BLOCK><<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        input, weight, output, M, K);
}

void fp16_gemv_fp32_output_kernel_cu(
    const half* input, const half* weight, float* output,
    int M, int K, cudaStream_t stream
) {
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
    const int num_blocks = (K + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    
    fp16_gemv_fp32_output_kernel<32, WARPS_PER_BLOCK><<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        input, weight, output, M, K);
}

void fp16_gemv_large_m_kernel_cu(
    const half* input, const half* weight, half* output,
    int M, int K, cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    const size_t shared_mem_size = M * sizeof(half);
    
    // Check shared memory limit (48KB typical)
    if (shared_mem_size <= 49152) {
        fp16_gemv_large_m_kernel<BLOCK_SIZE><<<K, BLOCK_SIZE, shared_mem_size, stream>>>(
            input, weight, output, M, K);
    } else {
        // Fall back to standard kernel
        fp16_gemv_kernel_cu(input, weight, output, M, K, stream);
    }
}

void fp16_weight_fp32_io_gemv_kernel_cu(
    const float* input, const half* weight, float* output,
    int M, int K, cudaStream_t stream
) {
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
    const int num_blocks = (K + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    
    fp16_weight_fp32_io_gemv_kernel<32, WARPS_PER_BLOCK><<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        input, weight, output, M, K);
}

}  // namespace kernel
