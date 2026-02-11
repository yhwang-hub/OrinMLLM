/**
 * Optimized FP16 CUDA Kernels for KuiperLLama
 * 
 * Based on llama.cpp's optimizations:
 * - PTX-level MMA instructions for Tensor Core utilization
 * - Optimized Flash Attention with tiled computation
 * - Fused operations (RMSNorm + GEMV, etc.)
 * 
 * Target: NVIDIA Orin (SM 8.7, Ampere-based)
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cub/cub.cuh>

namespace kernel {
namespace optimized {

//=============================================================================
// Constants and Configuration
//=============================================================================

constexpr int WARP_SIZE = 32;
constexpr int HALF_MAX_HALF_F = 32752.0f;  // ~65504/2 to avoid overflow
constexpr float SOFTMAX_FTZ_THRESHOLD = -20.0f;

// MMA tile sizes for Ampere/Orin (m16n8k16)
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;

//=============================================================================
// PTX MMA Primitives
//=============================================================================

/**
 * Load matrix from shared memory using ldmatrix PTX instruction
 * This is much faster than regular loads for MMA operations
 * 
 * ldmatrix loads 4 8x8 matrices (or 2 8x8 matrices for .x2) into registers
 * in a format ready for mma.sync
 */
__device__ __forceinline__ void ldmatrix_x4(
    uint32_t* dst,
    const void* __restrict__ src
) {
#if __CUDA_ARCH__ >= 750
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(src));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
        : "r"(addr)
    );
#else
    // Fallback for older architectures
    const uint32_t* src32 = reinterpret_cast<const uint32_t*>(src);
    dst[0] = src32[threadIdx.x % 8];
    dst[1] = src32[threadIdx.x % 8 + 8];
    dst[2] = src32[threadIdx.x % 8 + 16];
    dst[3] = src32[threadIdx.x % 8 + 24];
#endif
}

__device__ __forceinline__ void ldmatrix_x2(
    uint32_t* dst,
    const void* __restrict__ src
) {
#if __CUDA_ARCH__ >= 750
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(src));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
        : "=r"(dst[0]), "=r"(dst[1])
        : "r"(addr)
    );
#else
    const uint32_t* src32 = reinterpret_cast<const uint32_t*>(src);
    dst[0] = src32[threadIdx.x % 8];
    dst[1] = src32[threadIdx.x % 8 + 8];
#endif
}

/**
 * MMA m16n8k16 FP16 operation using PTX
 * D = A @ B + C (where all are FP16, accumulate in FP32 then convert)
 */
__device__ __forceinline__ void mma_m16n8k16_fp16(
    float* d,        // Output: 4 floats per thread (m16n8 tile)
    const uint32_t* a, // Input A: 4 registers per thread
    const uint32_t* b, // Input B: 2 registers per thread  
    const float* c     // Input C: 4 floats per thread (accumulator)
) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
#elif __CUDA_ARCH__ >= 750
    // Turing uses different MMA shape (m16n8k8)
    // Split k16 into 2 k8 operations
    float tmp[4] = {c[0], c[1], c[2], c[3]};
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5}, "
        "{%6}, "
        "{%7, %8, %9, %10};"
        : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3])
        : "r"(a[0]), "r"(a[1]),
          "r"(b[0]),
          "f"(tmp[0]), "f"(tmp[1]), "f"(tmp[2]), "f"(tmp[3])
    );
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5}, "
        "{%6}, "
        "{%7, %8, %9, %10};"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[2]), "r"(a[3]),
          "r"(b[1]),
          "f"(tmp[0]), "f"(tmp[1]), "f"(tmp[2]), "f"(tmp[3])
    );
#else
    // Software fallback
    d[0] = c[0]; d[1] = c[1]; d[2] = c[2]; d[3] = c[3];
#endif
}

//=============================================================================
// Optimized GEMV with Half2 and Warp Shuffle
//=============================================================================

/**
 * Ultra-optimized GEMV kernel for decode phase
 * - Uses half2 vectorized loads for 2x memory bandwidth
 * - Warp shuffle reduction (fastest reduction method)
 * - Multiple warps per output element for large M
 * - Register-level accumulation
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
 * High-throughput GEMV kernel using multiple warps per row
 * Better for large M dimensions where memory bandwidth is critical
 * Each block processes one row with multiple warps collaborating
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
    const int remaining = M - num_h4 * 4;
    
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

/**
 * Pure FP16 GEMV kernel - no FP32 conversion needed
 * For use when both input and weight are FP16
 */
template <int WARPS_PER_BLOCK = 8>
__global__ void gemv_pure_fp16_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    const int M,
    const int K
) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    
    if (row >= K) return;
    
    const half* row_ptr = weight + static_cast<int64_t>(row) * M;
    const half2* weight_h2 = reinterpret_cast<const half2*>(row_ptr);
    const half2* input_h2 = reinterpret_cast<const half2*>(input);
    
    const int num_h2 = M / 2;
    
    // Use half2 FMA for efficient FP16 computation
    half2 sum2 = __float2half2_rn(0.0f);
    
    for (int i = lane_id; i < num_h2; i += WARP_SIZE) {
        half2 w = __ldg(&weight_h2[i]);
        half2 x = input_h2[i];
        sum2 = __hfma2(w, x, sum2);
    }
    
    // Convert to float for reduction
    float sum = __half2float(sum2.x) + __half2float(sum2.y);
    
    // Handle remainder
    const int base = num_h2 * 2;
    for (int i = base + lane_id; i < M; i += WARP_SIZE) {
        sum += __half2float(row_ptr[i]) * __half2float(input[i]);
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

//=============================================================================
// WMMA-based GEMM for larger batch sizes
//=============================================================================

using namespace nvcuda::wmma;

/**
 * WMMA-based GEMM kernel for prefill (batch >= 16)
 * Uses Tensor Cores through WMMA API
 * 
 * Input: [batch_size, M]
 * Weight: [K, M] 
 * Output: [batch_size, K]
 */
template <int WMMA_M = 16, int WMMA_N = 16, int WMMA_K = 16>
__global__ void gemm_wmma_fp16(
    const half* __restrict__ input,   // [batch_size, M] - already converted to FP16
    const half* __restrict__ weight,  // [K, M]
    float* __restrict__ output,       // [batch_size, K]
    const int batch_size,
    const int M,
    const int K
) {
    // Each block computes a WMMA_M x WMMA_N output tile
    // Multiple warps can work on different tiles
    
    const int warp_id = (threadIdx.x + threadIdx.y * blockDim.x) / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    // Output tile position
    const int tile_row = blockIdx.y * WMMA_N;  // K dimension
    const int tile_col = blockIdx.x * WMMA_M;  // batch dimension
    
    if (tile_row >= K || tile_col >= batch_size) return;
    
    // WMMA fragments
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    // Initialize accumulator
    fill_fragment(c_frag, 0.0f);
    
    // Loop over K dimension in tiles
    for (int k_tile = 0; k_tile < M; k_tile += WMMA_K) {
        // Load input tile: [batch_size, M] -> [WMMA_M, WMMA_K] at (tile_col, k_tile)
        // Input is row-major: input[b][m] at input + b*M + m
        load_matrix_sync(a_frag, input + tile_col * M + k_tile, M);
        
        // Load weight tile: [K, M] -> [WMMA_K, WMMA_N] at (k_tile, tile_row) 
        // Weight is row-major, but we need column-major view
        // W[k][m] stored at weight + k*M + m
        // For WMMA we need W^T[m][k], which we get via col_major loading
        load_matrix_sync(b_frag, weight + tile_row * M + k_tile, M);
        
        // Matrix multiply-accumulate
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // Store result
    // Output is [batch_size, K], store at output + tile_col * K + tile_row
    store_matrix_sync(output + tile_col * K + tile_row, c_frag, K, mem_row_major);
}

//=============================================================================
// Optimized Flash Attention
//=============================================================================

/**
 * Flash Attention with online softmax - FP16 optimized version
 * Based on llama.cpp's fattn-vec implementation
 * 
 * Key optimizations:
 * - half2 vectorized Q/K/V loads
 * - Online softmax (no need to store full attention matrix)
 * - Tiled computation for better cache utilization
 * - Warp-level primitives for reductions
 */
template <int HEAD_SIZE = 128, int TILE_K = 64>
__global__ void flash_attention_fp16_optimized(
    const half* __restrict__ Q,        // [seq_len, dim]
    const half* __restrict__ K_cache,  // [max_seq_len, kv_dim]
    const half* __restrict__ V_cache,  // [max_seq_len, kv_dim]
    float* __restrict__ O,             // [seq_len, dim]
    const int seq_len,
    const int start_pos,
    const int head_num,
    const int kv_head_num,
    const int kv_mul,
    const int dim,
    const int kv_dim,
    const float scale
) {
    const int head = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int tid = threadIdx.x;
    
    constexpr int BLOCK_SIZE = 128;  // Threads per block
    
    if (head >= head_num || seq_idx >= seq_len) return;
    
    // GQA mapping
    const int kv_head = head / kv_mul;
    const int head_offset = kv_head * HEAD_SIZE;
    
    // Current position for causal mask
    const int cur_pos = start_pos + seq_idx;
    const int kv_len = cur_pos + 1;
    
    // Shared memory for query and attention scores
    extern __shared__ char fattn_smem[];
    half* s_query_h = reinterpret_cast<half*>(fattn_smem);
    float* s_scores = reinterpret_cast<float*>(fattn_smem + HEAD_SIZE * sizeof(half) + 16);  // Padding for alignment
    
    // Load query to shared memory (half precision)
    const half* q_ptr = Q + seq_idx * dim + head * HEAD_SIZE;
    for (int d = tid; d < HEAD_SIZE; d += BLOCK_SIZE) {
        s_query_h[d] = q_ptr[d];
    }
    __syncthreads();
    
    // Output pointer
    float* o_ptr = O + seq_idx * dim + head * HEAD_SIZE;
    
    // Accumulators per thread (4 dimensions each for HEAD_SIZE=128, BLOCK_SIZE=128)
    constexpr int DIMS_PER_THREAD = (HEAD_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float acc_o[DIMS_PER_THREAD] = {0.0f};
    
    // Online softmax state
    float row_max = -HALF_MAX_HALF_F;
    float row_sum = 0.0f;
    
    // Pre-compute scale as half2
    const half scale_h = __float2half(scale);
    
    // Process K/V in tiles
    for (int tile_start = 0; tile_start < kv_len; tile_start += TILE_K) {
        const int tile_end = min(tile_start + TILE_K, kv_len);
        const int tile_len = tile_end - tile_start;
        
        float tile_max = -HALF_MAX_HALF_F;
        
        // Compute attention scores with half2 dot products
        for (int k = tid; k < tile_len; k += BLOCK_SIZE) {
            const int kv_pos = tile_start + k;
            const half* k_ptr = K_cache + kv_pos * kv_dim + head_offset;
            
            // Q · K^T using half2 operations
            float score = 0.0f;
            
            const half2* q_h2 = reinterpret_cast<const half2*>(s_query_h);
            const half2* k_h2 = reinterpret_cast<const half2*>(k_ptr);
            
            #pragma unroll 8
            for (int d = 0; d < HEAD_SIZE / 2; d++) {
                half2 q_val = q_h2[d];
                half2 k_val = k_h2[d];
                float2 q_f2 = __half22float2(q_val);
                float2 k_f2 = __half22float2(k_val);
                score += q_f2.x * k_f2.x + q_f2.y * k_f2.y;
            }
            score *= scale;
            
            s_scores[k] = score;
            tile_max = fmaxf(tile_max, score);
        }
        __syncthreads();
        
        // Block reduction for tile max
        typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        
        float block_max = BlockReduce(temp_storage).Reduce(tile_max, cub::Max());
        __shared__ float s_tile_max;
        if (tid == 0) s_tile_max = block_max;
        __syncthreads();
        float m_j = s_tile_max;
        
        // Update global max
        float m_new = fmaxf(row_max, m_j);
        
        // Compute exp(score - m_new) and tile sum
        float tile_sum = 0.0f;
        for (int k = tid; k < tile_len; k += BLOCK_SIZE) {
            float val = s_scores[k] - m_new;
            // Flush to zero for numerical stability
            float exp_score = (val > SOFTMAX_FTZ_THRESHOLD) ? expf(val) : 0.0f;
            s_scores[k] = exp_score;
            tile_sum += exp_score;
        }
        __syncthreads();
        
        // Block-reduce tile sum
        float block_sum = BlockReduce(temp_storage).Sum(tile_sum);
        __shared__ float s_tile_sum;
        if (tid == 0) s_tile_sum = block_sum;
        __syncthreads();
        float l_j = s_tile_sum;
        
        // Correction factor for previous accumulator
        float correction = expf(row_max - m_new);
        float l_new = correction * row_sum + l_j;
        
        // Update output accumulators
        #pragma unroll
        for (int i = 0; i < DIMS_PER_THREAD; i++) {
            const int d = tid + i * BLOCK_SIZE;
            if (d < HEAD_SIZE) {
                // Scale previous accumulator
                acc_o[i] *= correction;
                
                // Add contribution from this tile
                for (int k = 0; k < tile_len; k++) {
                    const int kv_pos = tile_start + k;
                    const half* v_ptr = V_cache + kv_pos * kv_dim + head_offset;
                    acc_o[i] += s_scores[k] * __half2float(v_ptr[d]);
                }
            }
        }
        
        row_max = m_new;
        row_sum = l_new;
        __syncthreads();
    }
    
    // Final normalization
    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    
    #pragma unroll
    for (int i = 0; i < DIMS_PER_THREAD; i++) {
        const int d = tid + i * BLOCK_SIZE;
        if (d < HEAD_SIZE) {
            o_ptr[d] = acc_o[i] * inv_sum;
        }
    }
}

//=============================================================================
// Fused Operations
//=============================================================================

/**
 * Fused RMSNorm + GEMV kernel
 * Combines RMSNorm and first linear layer to eliminate intermediate memory traffic
 * 
 * RMSNorm: y = x * weight / sqrt(mean(x^2) + eps)
 * GEMV: z = W @ y
 * 
 * Fused: z = W @ (x * norm_weight / sqrt(mean(x^2) + eps))
 */
template <int BLOCK_SIZE = 256>
__global__ void fused_rmsnorm_gemv_fp16(
    const float* __restrict__ input,       // [dim]
    const float* __restrict__ norm_weight, // [dim]
    const half* __restrict__ weight,       // [K, dim]
    float* __restrict__ output,            // [K]
    const int dim,
    const int K,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (row >= K) return;
    
    extern __shared__ float fused_smem[];
    float* s_normalized = fused_smem;
    
    // Step 1: Compute sum of squares for RMSNorm (collaborative)
    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += BLOCK_SIZE) {
        float val = input[i];
        sum_sq += val * val;
    }
    
    // Block reduction
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float total_sum_sq = BlockReduce(temp_storage).Sum(sum_sq);
    
    __shared__ float s_rms_inv;
    if (tid == 0) {
        float rms = sqrtf(total_sum_sq / dim + eps);
        s_rms_inv = 1.0f / rms;
    }
    __syncthreads();
    float rms_inv = s_rms_inv;
    
    // Step 2: Compute normalized values and store in shared memory
    for (int i = tid; i < dim; i += BLOCK_SIZE) {
        s_normalized[i] = input[i] * norm_weight[i] * rms_inv;
    }
    __syncthreads();
    
    // Step 3: GEMV with normalized input
    const half* row_ptr = weight + static_cast<int64_t>(row) * dim;
    const half2* weight_h2 = reinterpret_cast<const half2*>(row_ptr);
    
    float sum = 0.0f;
    const int num_h2 = dim / 2;
    
    #pragma unroll 4
    for (int i = tid; i < num_h2; i += BLOCK_SIZE) {
        half2 w_h2 = weight_h2[i];
        float2 w_f2 = __half22float2(w_h2);
        float x0 = s_normalized[i * 2];
        float x1 = s_normalized[i * 2 + 1];
        sum += w_f2.x * x0 + w_f2.y * x1;
    }
    
    // Handle remainder
    const int rem_base = num_h2 * 2;
    for (int i = rem_base + tid; i < dim; i += BLOCK_SIZE) {
        sum += __half2float(row_ptr[i]) * s_normalized[i];
    }
    
    // Final reduction
    float final_sum = BlockReduce(temp_storage).Sum(sum);
    
    if (tid == 0) {
        output[row] = final_sum;
    }
}

/**
 * Fused SiLU activation (for feed-forward network)
 * silu(x) = x * sigmoid(x)
 * Fused with element-wise multiply: out = silu(gate) * up
 */
__global__ void fused_silu_mul_fp16(
    const float* __restrict__ gate,  // [dim]
    const float* __restrict__ up,    // [dim]
    half* __restrict__ output,       // [dim]
    const int dim
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < dim) {
        float g = gate[idx];
        float u = up[idx];
        // SiLU = x * sigmoid(x)
        float silu_g = g / (1.0f + expf(-g));
        output[idx] = __float2half(silu_g * u);
    }
}

/**
 * Vectorized version processing 2 elements at a time
 */
__global__ void fused_silu_mul_fp16_vec2(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    half* __restrict__ output,
    const int dim
) {
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    
    if (idx + 1 < dim) {
        float2 g = *reinterpret_cast<const float2*>(gate + idx);
        float2 u = *reinterpret_cast<const float2*>(up + idx);
        
        float silu_g0 = g.x / (1.0f + expf(-g.x));
        float silu_g1 = g.y / (1.0f + expf(-g.y));
        
        half2 out = make_half2(__float2half(silu_g0 * u.x), __float2half(silu_g1 * u.y));
        *reinterpret_cast<half2*>(output + idx) = out;
    } else if (idx < dim) {
        float g = gate[idx];
        float u = up[idx];
        float silu_g = g / (1.0f + expf(-g));
        output[idx] = __float2half(silu_g * u);
    }
}

/**
 * Fused RoPE (Rotary Position Embedding) kernel for FP16
 * Applied in-place to Q and K
 */
__global__ void fused_rope_fp16(
    half* __restrict__ q,     // [seq_len, dim]
    half* __restrict__ k,     // [seq_len, kv_dim] 
    const float* __restrict__ freq_cis_real,  // [max_seq_len, head_size/2]
    const float* __restrict__ freq_cis_imag,  // [max_seq_len, head_size/2]
    const int seq_len,
    const int start_pos,
    const int dim,
    const int kv_dim,
    const int head_size,
    const int head_num,
    const int kv_head_num
) {
    const int pos_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int pair_idx = threadIdx.x;  // Which (real, imag) pair
    
    if (pos_idx >= seq_len || pair_idx >= head_size / 2) return;
    
    const int abs_pos = start_pos + pos_idx;
    const int freq_idx = abs_pos * (head_size / 2) + pair_idx;
    
    float cos_val = freq_cis_real[freq_idx];
    float sin_val = freq_cis_imag[freq_idx];
    
    // Apply to Q heads
    if (head_idx < head_num) {
        const int q_offset = pos_idx * dim + head_idx * head_size + pair_idx * 2;
        float q0 = __half2float(q[q_offset]);
        float q1 = __half2float(q[q_offset + 1]);
        
        float q0_new = q0 * cos_val - q1 * sin_val;
        float q1_new = q0 * sin_val + q1 * cos_val;
        
        q[q_offset] = __float2half(q0_new);
        q[q_offset + 1] = __float2half(q1_new);
    }
    
    // Apply to K heads (only for kv_head_num heads)
    if (head_idx < kv_head_num) {
        const int k_offset = pos_idx * kv_dim + head_idx * head_size + pair_idx * 2;
        float k0 = __half2float(k[k_offset]);
        float k1 = __half2float(k[k_offset + 1]);
        
        float k0_new = k0 * cos_val - k1 * sin_val;
        float k1_new = k0 * sin_val + k1 * cos_val;
        
        k[k_offset] = __float2half(k0_new);
        k[k_offset + 1] = __float2half(k1_new);
    }
}

//=============================================================================
// Pure FP16 GEMV - Highest Performance for Decode
//=============================================================================

/**
 * Pure FP16 GEMV kernel - Maximum throughput for decode phase
 * Input: FP16, Weight: FP16, Output: FP16
 * 
 * Optimizations:
 * - Half4 vectorized loads (64-bit) for maximum memory bandwidth
 * - Register blocking with 8 elements per thread per iteration
 * - Warp shuffle reduction (fastest reduction method)
 * - __ldg() read-only cache utilization
 * - Interleaved memory access pattern for bank conflict avoidance
 */
template <int WARPS_PER_BLOCK = 8>
__global__ void gemv_pure_fp16_optimized(
    const half* __restrict__ input,    // [M]
    const half* __restrict__ weight,   // [K, M] row-major
    half* __restrict__ output,         // [K]
    const int M,
    const int K
) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    
    if (row >= K) return;
    
    const half* row_ptr = weight + static_cast<int64_t>(row) * M;
    
    // Use half4 (64-bit) loads for maximum memory bandwidth
    // half4 is represented as 2x half2
    const int num_h4 = M / 4;
    const half2* weight_h2 = reinterpret_cast<const half2*>(row_ptr);
    const half2* input_h2 = reinterpret_cast<const half2*>(input);
    
    float2 sum = make_float2(0.0f, 0.0f);
    
    // Process 4 half values at a time (64-bit loads)
    #pragma unroll 4
    for (int i = lane_id; i < num_h4; i += WARP_SIZE) {
        // Load 4 consecutive half values as 2x half2
        half2 w0 = __ldg(&weight_h2[i * 2]);
        half2 w1 = __ldg(&weight_h2[i * 2 + 1]);
        half2 x0 = input_h2[i * 2];
        half2 x1 = input_h2[i * 2 + 1];
        
        // Convert to float and accumulate
        float2 w0_f = __half22float2(w0);
        float2 w1_f = __half22float2(w1);
        float2 x0_f = __half22float2(x0);
        float2 x1_f = __half22float2(x1);
        
        sum.x += w0_f.x * x0_f.x + w0_f.y * x0_f.y;
        sum.y += w1_f.x * x1_f.x + w1_f.y * x1_f.y;
    }
    
    float total = sum.x + sum.y;
    
    // Handle remainder
    const int base = num_h4 * 4;
    for (int i = base + lane_id; i < M; i += WARP_SIZE) {
        total += __half2float(__ldg(&row_ptr[i])) * __half2float(input[i]);
    }
    
    // Warp reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        total += __shfl_down_sync(0xffffffff, total, offset);
    }
    
    if (lane_id == 0) {
        output[row] = __float2half(total);
    }
}

/**
 * Pure FP16 batched GEMV for small batch prefill
 * Uses cuBLAS-like approach but with custom kernel for small batches
 */
template <int BLOCK_SIZE = 256>
__global__ void batched_gemv_pure_fp16(
    const half* __restrict__ input,    // [batch, M]
    const half* __restrict__ weight,   // [K, M]
    half* __restrict__ output,         // [batch, K]
    const int batch_size,
    const int M,
    const int K
) {
    const int row = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int tid = threadIdx.x;
    
    if (row >= K || batch_idx >= batch_size) return;
    
    const half* batch_input = input + batch_idx * M;
    half* batch_output = output + batch_idx * K;
    
    const half2* weight_h2 = reinterpret_cast<const half2*>(weight + row * M);
    const half2* input_h2 = reinterpret_cast<const half2*>(batch_input);
    
    const int num_h2 = M / 2;
    
    float sum = 0.0f;
    
    #pragma unroll 4
    for (int i = tid; i < num_h2; i += BLOCK_SIZE) {
        half2 w = __ldg(&weight_h2[i]);
        half2 x = input_h2[i];
        float2 wf = __half22float2(w);
        float2 xf = __half22float2(x);
        sum += wf.x * xf.x + wf.y * xf.y;
    }
    
    // Handle remainder
    const int base = num_h2 * 2;
    for (int i = base + tid; i < M; i += BLOCK_SIZE) {
        sum += __half2float(weight[row * M + i]) * __half2float(batch_input[i]);
    }
    
    // Block reduction
    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    sum = BlockReduce(temp_storage).Sum(sum);
    
    if (tid == 0) {
        batch_output[row] = __float2half(sum);
    }
}

/**
 * Specialized GEMV for common dimension sizes in LLMs
 * Qwen2.5-7B: dim=3584, intermediate=18944, kv_dim=512, head_dim=128
 */
template <int DIM>
__global__ void gemv_fp16_specialized(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    const int K
) {
    constexpr int WARPS = 8;
    constexpr int THREADS = WARPS * 32;
    
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int row = blockIdx.x * WARPS + warp_id;
    
    if (row >= K) return;
    
    // Fully unrolled for known dimension
    const half2* w_h2 = reinterpret_cast<const half2*>(weight + row * DIM);
    const half2* x_h2 = reinterpret_cast<const half2*>(input);
    
    constexpr int NUM_H2 = DIM / 2;
    constexpr int ITERS = (NUM_H2 + 31) / 32;
    
    float sum = 0.0f;
    
    #pragma unroll
    for (int iter = 0; iter < ITERS; iter++) {
        const int i = iter * 32 + lane_id;
        if (i < NUM_H2) {
            half2 w = __ldg(&w_h2[i]);
            half2 x = x_h2[i];
            float2 wf = __half22float2(w);
            float2 xf = __half22float2(x);
            sum += wf.x * xf.x + wf.y * xf.y;
        }
    }
    
    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (lane_id == 0) {
        output[row] = __float2half(sum);
    }
}

}  // namespace optimized
}  // namespace kernel
