/*
 * AWQ High-Performance W4A16 Fused Kernels - Version 3.0
 * 
 * Key Design Goals:
 * 1. NO pre-dequantization - keep AWQ memory advantage (~6GB vs 16GB FP16)
 * 2. Outperform FP16 in BOTH prefill and decode phases
 * 3. Maintain inference accuracy
 * 
 * Key Optimizations:
 * 1. GEMV (M=1, decode): Vectorized INT4 load + parallel reduction
 *    - INT4 has 4x memory bandwidth advantage over FP16
 *    - Bottleneck is memory bandwidth, not compute
 * 
 * 2. GEMM (M>1, prefill): Shared memory tiled dequant + GEMM fusion
 *    - Dequant once in shared memory, reuse M times
 *    - Amortize dequant cost over batch dimension
 * 
 * AWQ Format:
 * - Weights: INT4 packed 8 per INT32, with reverse order {0,4,1,5,2,6,3,7}
 * - Scales/Zeros: per-group (group_size=128 typically)
 * - Dequant: w_fp16 = scale * (w_int4 - zero)
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>

namespace kernel {

// AWQ reverse bit order: maps output index i to bit position awq_order[i]*4
__device__ __constant__ int c_awq_order[8] = {0, 4, 1, 5, 2, 6, 3, 7};

// =============================================================================
// Ultra-Fast GEMV Kernel for Decode Phase (M=1)
// =============================================================================
// 
// Design: Each thread block computes 128 output elements
// - Block: 256 threads (8 warps)
// - Each warp: handles 16 output elements
// - Each thread: handles 16 input elements per iteration
// 
// Memory optimization:
// - Use __ldg for read-only cache (faster than L1 for streaming)
// - Vectorized load: read 2 packed INT32 (16 weights) at once
// - Coalesced access patterns

__global__ __launch_bounds__(256, 4)
void awq_gemv_v3_kernel(
    const half* __restrict__ X,           // [in_features]
    const int32_t* __restrict__ qweight,  // [in_features, out_features/8]
    const int32_t* __restrict__ qzeros,   // [n_groups, out_features/8]
    const half* __restrict__ scales,      // [n_groups, out_features]
    half* __restrict__ Y,                 // [out_features]
    int in_features,
    int out_features,
    int group_size
) {
    // Thread/warp configuration
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = 8;  // 256 threads / 32
    
    // Output assignment: each block handles 128 outputs (16 per warp)
    const int block_out_base = blockIdx.x * 128;
    const int warp_out_base = block_out_base + warp_id * 16;
    
    if (warp_out_base >= out_features) return;
    
    const int packed_N = out_features / 8;
    const int n_groups = in_features / group_size;
    
    // AWQ order for fast unpacking
    const int awq_order[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    
    // Accumulators: 16 outputs per warp, 2 packed groups per warp
    // Each lane computes partial sums that will be reduced
    float acc[16] = {0};
    
    // Process all groups
    for (int g = 0; g < n_groups; g++) {
        const int group_start = g * group_size;
        
        // Preload scales and zeros for this warp's 16 output channels
        float s[16], neg_sz[16];
        
        #pragma unroll
        for (int i = 0; i < 2; i++) {  // 2 packed INT32s = 16 outputs
            const int packed_idx = (warp_out_base / 8) + i;
            if (packed_idx < packed_N) {
                const int32_t qz = __ldg(&qzeros[g * packed_N + packed_idx]);
                
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    const int out_idx = i * 8 + j;
                    float scale = __half2float(__ldg(&scales[g * out_features + warp_out_base + out_idx]));
                    float zero = (float)((qz >> (awq_order[j] * 4)) & 0xF);
                    s[out_idx] = scale;
                    neg_sz[out_idx] = -scale * zero;
                }
            }
        }
        
        // Process input features in this group
        // Each lane processes different input indices, then reduce
        for (int k = lane_id; k < group_size; k += 32) {
            const int in_idx = group_start + k;
            if (in_idx >= in_features) continue;
            
            float x = __half2float(__ldg(&X[in_idx]));
            
            // Load 2 packed INT32s for 16 output channels
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                const int packed_idx = (warp_out_base / 8) + i;
                if (packed_idx < packed_N) {
                    const int32_t w_packed = __ldg(&qweight[in_idx * packed_N + packed_idx]);
                    
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        const int out_idx = i * 8 + j;
                        float w = (float)((w_packed >> (awq_order[j] * 4)) & 0xF);
                        acc[out_idx] += x * (s[out_idx] * w + neg_sz[out_idx]);
                    }
                }
            }
        }
    }
    
    // Warp-level reduction using shuffle
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            acc[i] += __shfl_down_sync(0xffffffff, acc[i], offset);
        }
    }
    
    // Write results (lane 0 of each warp)
    if (lane_id == 0) {
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            const int out_idx = warp_out_base + i;
            if (out_idx < out_features) {
                Y[out_idx] = __float2half(acc[i]);
            }
        }
    }
}

// =============================================================================
// High-Performance GEMM Kernel for Prefill Phase (M>1)
// =============================================================================
// 
// Design: Tiled GEMM with shared memory for weight dequantization
// - Block tile: [TILE_M, TILE_N] = [32, 64]
// - K-dimension processing with shared memory caching
// 
// Key insight: Dequantize weights to shared memory ONCE,
// then reuse for TILE_M rows of input, amortizing dequant cost

constexpr int TILE_M = 32;
constexpr int TILE_N = 64;
constexpr int TILE_K = 32;  // Must align with group_size for efficiency

__global__ __launch_bounds__(256)
void awq_gemm_v3_kernel(
    const half* __restrict__ X,           // [M, in_features]
    const int32_t* __restrict__ qweight,  // [in_features, out_features/8]
    const int32_t* __restrict__ qzeros,   // [n_groups, out_features/8]
    const half* __restrict__ scales,      // [n_groups, out_features]
    half* __restrict__ Y,                 // [M, out_features]
    int M,
    int in_features,
    int out_features,
    int group_size
) {
    // Block/thread configuration
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    // Block handles [TILE_M, TILE_N] of output
    const int block_m = blockIdx.y * TILE_M;
    const int block_n = blockIdx.x * TILE_N;
    
    if (block_m >= M || block_n >= out_features) return;
    
    const int packed_N = out_features / 8;
    const int n_groups = in_features / group_size;
    
    // AWQ order
    const int awq_order[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    
    // Shared memory for dequantized weights: [TILE_K, TILE_N]
    __shared__ half smem_weights[TILE_K][TILE_N];
    // Shared memory for input tile: [TILE_M, TILE_K]
    __shared__ half smem_input[TILE_M][TILE_K];
    
    // Accumulators for each thread
    // Thread block layout: 256 threads
    // Each thread handles a 4x4 tile of the output
    constexpr int THREAD_M = 4;
    constexpr int THREAD_N = 4;
    
    const int thread_row = (tid / 16) * THREAD_M;  // 0, 4, 8, ..., 28
    const int thread_col = (tid % 16) * THREAD_N;  // 0, 4, 8, ..., 60
    
    float acc[THREAD_M][THREAD_N] = {0};
    
    // Process K dimension in tiles
    for (int k_base = 0; k_base < in_features; k_base += TILE_K) {
        // Determine group for this K tile
        const int g = k_base / group_size;
        
        // --- Load and dequant weights to shared memory ---
        // Each thread handles multiple elements
        // TILE_K * TILE_N / 256 = 32 * 64 / 256 = 8 elements per thread
        for (int elem = tid; elem < TILE_K * TILE_N; elem += 256) {
            const int k_local = elem / TILE_N;
            const int n_local = elem % TILE_N;
            const int k_global = k_base + k_local;
            const int n_global = block_n + n_local;
            
            if (k_global < in_features && n_global < out_features) {
                const int packed_n_idx = n_global / 8;
                const int n_in_pack = n_global % 8;
                
                // Load packed weight
                const int32_t w_packed = __ldg(&qweight[k_global * packed_N + packed_n_idx]);
                const int32_t z_packed = __ldg(&qzeros[g * packed_N + packed_n_idx]);
                const half scale = __ldg(&scales[g * out_features + n_global]);
                
                // Dequantize with AWQ order
                int w = (w_packed >> (awq_order[n_in_pack] * 4)) & 0xF;
                int z = (z_packed >> (awq_order[n_in_pack] * 4)) & 0xF;
                smem_weights[k_local][n_local] = __hmul(__float2half((float)(w - z)), scale);
            } else {
                smem_weights[k_local][n_local] = __float2half(0.0f);
            }
        }
        
        // --- Load input to shared memory ---
        // TILE_M * TILE_K / 256 = 32 * 32 / 256 = 4 elements per thread
        for (int elem = tid; elem < TILE_M * TILE_K; elem += 256) {
            const int m_local = elem / TILE_K;
            const int k_local = elem % TILE_K;
            const int m_global = block_m + m_local;
            const int k_global = k_base + k_local;
            
            if (m_global < M && k_global < in_features) {
                smem_input[m_local][k_local] = __ldg(&X[m_global * in_features + k_global]);
            } else {
                smem_input[m_local][k_local] = __float2half(0.0f);
            }
        }
        
        __syncthreads();
        
        // --- Compute partial products ---
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            #pragma unroll
            for (int m = 0; m < THREAD_M; m++) {
                float a = __half2float(smem_input[thread_row + m][k]);
                #pragma unroll
                for (int n = 0; n < THREAD_N; n++) {
                    float b = __half2float(smem_weights[k][thread_col + n]);
                    acc[m][n] += a * b;
                }
            }
        }
        
        __syncthreads();
    }
    
    // --- Write output ---
    #pragma unroll
    for (int m = 0; m < THREAD_M; m++) {
        const int m_global = block_m + thread_row + m;
        if (m_global < M) {
            #pragma unroll
            for (int n = 0; n < THREAD_N; n++) {
                const int n_global = block_n + thread_col + n;
                if (n_global < out_features) {
                    Y[m_global * out_features + n_global] = __float2half(acc[m][n]);
                }
            }
        }
    }
}

// =============================================================================
// Dispatcher Function
// =============================================================================

void awq_gemm_v3_cu(
    const half* input,
    const int32_t* qweight,
    const int32_t* qzeros,
    const half* scales,
    half* output,
    int M,
    int in_features,
    int out_features,
    int group_size,
    cudaStream_t stream
) {
    if (M == 1) {
        // GEMV: optimized for memory bandwidth
        // Each block handles 128 outputs
        const int num_blocks = (out_features + 127) / 128;
        awq_gemv_v3_kernel<<<num_blocks, 256, 0, stream>>>(
            input, qweight, qzeros, scales, output,
            in_features, out_features, group_size
        );
    } else if (M <= 4) {
        // Small batch: use simple GEMM kernel per row to maximize memory bandwidth
        // Launch one GEMV kernel per row
        for (int m = 0; m < M; m++) {
            const int num_blocks = (out_features + 127) / 128;
            awq_gemv_v3_kernel<<<num_blocks, 256, 0, stream>>>(
                input + m * in_features,
                qweight, qzeros, scales,
                output + m * out_features,
                in_features, out_features, group_size
            );
        }
    } else {
        // Larger batch: use tiled GEMM kernel
        dim3 grid((out_features + TILE_N - 1) / TILE_N,
                  (M + TILE_M - 1) / TILE_M);
        awq_gemm_v3_kernel<<<grid, 256, 0, stream>>>(
            input, qweight, qzeros, scales, output,
            M, in_features, out_features, group_size
        );
    }
}

}  // namespace kernel
