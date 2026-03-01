/**
 * Optimized FP16 CUDA Kernels for KuiperLLama
 * 
 * Mixed-precision GEMV kernels for decode phase:
 * FP32 input × FP16 weight → FP32 output
 * 
 * Target: NVIDIA Orin (SM 8.7, Ampere-based)
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>

namespace kernel {
namespace optimized {

constexpr int WARP_SIZE = 32;

//=============================================================================
// Mixed-Precision GEMV Kernels (FP32 input × FP16 weight → FP32 output)
//=============================================================================

/**
 * Warp-level GEMV kernel for decode phase (small-to-medium M)
 * - Uses half2 vectorized loads for 2x memory bandwidth
 * - Warp shuffle reduction (fastest reduction method)
 * - Multiple elements per thread for ILP
 * 
 * Used by: matmul_kernel_cu_fp16_weight when M < 2048
 */
template <int WARPS_PER_BLOCK = 8, int ELEMENTS_PER_THREAD = 8>
__global__ void gemv_fp16_optimized(
    const float* __restrict__ input,
    const half* __restrict__ weight,
    float* __restrict__ output,
    const int M,
    const int K
) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    
    if (row >= K) return;
    
    const half* row_ptr = weight + static_cast<int64_t>(row) * M;
    const half2* weight_h2 = reinterpret_cast<const half2*>(row_ptr);
    const float2* input_f2 = reinterpret_cast<const float2*>(input);
    
    const int num_h2 = M / 2;
    
    // Accumulate in registers - process multiple elements per thread for ILP
    float sum = 0.0f;
    
    // Main loop with aggressive unrolling for instruction-level parallelism
    const int stride = WARP_SIZE * ELEMENTS_PER_THREAD;
    const int base = lane_id * ELEMENTS_PER_THREAD;
    
    #pragma unroll 2
    for (int offset = 0; offset < num_h2; offset += stride) {
        #pragma unroll
        for (int e = 0; e < ELEMENTS_PER_THREAD; e++) {
            const int i = base + offset + e * WARP_SIZE;
            if (i < num_h2) {
                half2 w_h2 = __ldg(&weight_h2[i]);
                float2 x_f2 = input_f2[i];
                float2 w_f2 = __half22float2(w_h2);
                sum += w_f2.x * x_f2.x + w_f2.y * x_f2.y;
            }
        }
    }
    
    // Handle remainder
    const int rem_base = num_h2 * 2;
    for (int i = rem_base + lane_id; i < M; i += WARP_SIZE) {
        sum += __half2float(row_ptr[i]) * input[i];
    }
    
    // Fast warp reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (lane_id == 0) {
        output[row] = sum;
    }
}

/**
 * Block-level GEMV kernel for decode phase (large M >= 2048)
 * - Full block collaborates on each output row
 * - CUB block reduction for inter-warp summation
 * - half4 (8-byte) loads for maximum memory throughput
 * 
 * Used by: matmul_kernel_cu_fp16_weight when M >= 2048
 */
template <int BLOCK_SIZE = 256, int ROWS_PER_BLOCK = 1>
__global__ void gemv_fp16_bandwidth_optimized(
    const float* __restrict__ input,
    const half* __restrict__ weight,
    float* __restrict__ output,
    const int M,
    const int K
) {
    const int row = blockIdx.x * ROWS_PER_BLOCK;
    const int tid = threadIdx.x;
    
    if (row >= K) return;
    
    const half* row_ptr = weight + static_cast<int64_t>(row) * M;
    
    // Use half4 (4 halfs = 8 bytes) for maximum memory throughput
    const int num_h4 = M / 4;
    
    float sum = 0.0f;
    
    // Process 4 half values at a time
    for (int i = tid; i < num_h4; i += BLOCK_SIZE) {
        // Load 4 half values (8 bytes) - coalesced access
        const half2* h2_ptr = reinterpret_cast<const half2*>(row_ptr + i * 4);
        half2 w0 = __ldg(h2_ptr);
        half2 w1 = __ldg(h2_ptr + 1);
        
        // Load 4 float values (16 bytes) 
        const float4* f4_ptr = reinterpret_cast<const float4*>(input + i * 4);
        float4 x = *f4_ptr;
        
        // Convert and accumulate
        float2 fw0 = __half22float2(w0);
        float2 fw1 = __half22float2(w1);
        
        sum += fw0.x * x.x + fw0.y * x.y + fw1.x * x.z + fw1.y * x.w;
    }
    
    // Handle remaining elements
    const int base = num_h4 * 4;
    for (int i = base + tid; i < M; i += BLOCK_SIZE) {
        sum += __half2float(row_ptr[i]) * input[i];
    }
    
    // Block-level reduction using CUB
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    sum = BlockReduce(temp_storage).Sum(sum);
    
    if (tid == 0) {
        output[row] = sum;
    }
}

}  // namespace optimized
}  // namespace kernel
