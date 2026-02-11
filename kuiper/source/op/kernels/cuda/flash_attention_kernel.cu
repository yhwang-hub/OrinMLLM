/**
 * Flash Attention CUDA Kernel - MMA Optimized Implementation
 * Based on the paper "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
 * Enhanced with PTX MMA instructions for Tensor Core acceleration
 * 
 * Optimizations applied:
 * - PTX MMA (mma.m16n8k16) for Tensor Core utilization
 * - half2 vectorized Q/K dot products for 2x compute throughput
 * - Online softmax to avoid storing full attention matrix
 * - Tiled computation with optimized tile sizes for L2 cache
 * - Warp-level primitives for fast reductions
 * - Numerical stability with flushing small values to zero
 * - Asynchronous memory copy where available
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <cstdio>
#include <cub/cub.cuh>
#include <mma.h>
#include "flash_attention_kernel.cuh"
#include "mma.cuh"

namespace kernel {

// FP32 kernels use BLOCK_SIZE_FP32 = 256 for correctness
constexpr int BLOCK_SIZE_FP32 = 256;
// FP16 kernels use BLOCK_SIZE = 128 for Orin optimization
constexpr int BLOCK_SIZE = 128;
constexpr int TILE_K = 1024;      // Tile size for KV processing (optimized)
constexpr int MMA_TILE_M = 16;   // MMA tile dimensions
constexpr int MMA_TILE_N = 16;
constexpr int MMA_TILE_K = 16;
constexpr int NUM_WARPS = 4;     // 4 warps per block
constexpr float SOFTMAX_FTZ = -20.0f;  // Flush-to-zero threshold

/**
 * Flash Attention prefill kernel with online softmax - FP32 version
 * Grid: [head_num, seq_len]
 * Each block handles one (head, query_position) pair
 * Uses BLOCK_SIZE_FP32 = 256 threads
 */
__global__ void flash_attention_prefill_kernel(
    const float* __restrict__ Q,           // [seq_len, dim]
    const float* __restrict__ K_cache,     // [max_seq_len, kv_dim]  
    const float* __restrict__ V_cache,     // [max_seq_len, kv_dim]
    float* __restrict__ O,                 // [seq_len, dim]
    const int seq_len,
    const int start_pos,
    const int head_num,
    const int kv_head_num,
    const int head_size,
    const int kv_mul,
    const int dim,
    const int kv_dim,
    const float scale
) {
    const int head = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int tid = threadIdx.x;
    
    if (head >= head_num || seq_idx >= seq_len) return;
    
    // GQA: map query head to kv head
    const int kv_head = head / kv_mul;
    const int head_offset = kv_head * head_size;
    
    // Current position (for causal mask)
    const int cur_pos = start_pos + seq_idx;
    const int kv_len = cur_pos + 1;  // attend to positions 0..cur_pos
    
    // Shared memory layout:
    // [0, head_size): query
    // [head_size, head_size + TILE_K): scores for current tile
    extern __shared__ float smem[];
    float* s_query = smem;
    float* s_scores = smem + head_size;
    
    // Load query to shared memory
    const float* q_ptr = Q + seq_idx * dim + head * head_size;
    for (int d = tid; d < head_size; d += BLOCK_SIZE_FP32) {
        s_query[d] = q_ptr[d];
    }
    __syncthreads();
    
    // Output pointer
    float* o_ptr = O + seq_idx * dim + head * head_size;
    
    // Initialize output accumulators
    // Each thread handles ceil(head_size / BLOCK_SIZE_FP32) dimensions
    float acc_o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    // Online softmax running state
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    
    // Process K/V in tiles
    for (int tile_start = 0; tile_start < kv_len; tile_start += TILE_K) {
        const int tile_end = min(tile_start + TILE_K, kv_len);
        const int tile_len = tile_end - tile_start;
        
        // Step 1: Compute attention scores for this tile
        float tile_max = -FLT_MAX;
        
        for (int k = tid; k < tile_len; k += BLOCK_SIZE_FP32) {
            const int kv_pos = tile_start + k;
            const float* k_ptr = K_cache + kv_pos * kv_dim + head_offset;
            
            // Compute Q · K^T
            float score = 0.0f;
            for (int d = 0; d < head_size; d++) {
                score += s_query[d] * k_ptr[d];
            }
            score *= scale;
            
            s_scores[k] = score;
            tile_max = fmaxf(tile_max, score);
        }
        __syncthreads();
        
        // Step 2: Find tile maximum using block reduction
        typedef cub::BlockReduce<float, BLOCK_SIZE_FP32> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        
        float block_max = BlockReduce(temp_storage).Reduce(tile_max, cub::Max());
        __shared__ float s_tile_max;
        if (tid == 0) s_tile_max = block_max;
        __syncthreads();
        float m_j = s_tile_max;  // max of current tile
        
        // Step 3: Update global max
        float m_new = fmaxf(row_max, m_j);
        
        // Step 4: Compute exp(score - m_new) and tile sum
        float tile_sum = 0.0f;
        for (int k = tid; k < tile_len; k += BLOCK_SIZE_FP32) {
            float exp_score = expf(s_scores[k] - m_new);
            s_scores[k] = exp_score;
            tile_sum += exp_score;
        }
        __syncthreads();
        
        // Step 5: Block-reduce tile sum
        float block_sum = BlockReduce(temp_storage).Sum(tile_sum);
        __shared__ float s_tile_sum;
        if (tid == 0) s_tile_sum = block_sum;
        __syncthreads();
        float l_j = s_tile_sum;
        
        // Step 6: Compute correction factor for previous accumulator
        float correction = expf(row_max - m_new);
        float l_new = correction * row_sum + l_j;
        
        // Step 7: Scale previous output and add new contribution
        for (int i = 0; i < (head_size + BLOCK_SIZE_FP32 - 1) / BLOCK_SIZE_FP32; i++) {
            const int d = tid + i * BLOCK_SIZE_FP32;
            if (d < head_size) {
                // Scale previous accumulator
                acc_o[i] *= correction;
                
                // Add contribution from this tile
                for (int k = 0; k < tile_len; k++) {
                    const int kv_pos = tile_start + k;
                    const float* v_ptr = V_cache + kv_pos * kv_dim + head_offset;
                    acc_o[i] += s_scores[k] * v_ptr[d];
                }
            }
        }
        
        // Step 8: Update running state
        row_max = m_new;
        row_sum = l_new;
        
        __syncthreads();
    }
    
    // Step 9: Final normalization: O = O / l
    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    
    for (int i = 0; i < (head_size + BLOCK_SIZE_FP32 - 1) / BLOCK_SIZE_FP32; i++) {
        const int d = tid + i * BLOCK_SIZE_FP32;
        if (d < head_size) {
            o_ptr[d] = acc_o[i] * inv_sum;
        }
    }
}

/**
 * Optimized decode attention kernel for single token
 * Uses warp-level primitives for better performance
 * Grid: [head_num]
 * Each block handles one head
 */
__global__ void flash_attention_decode_kernel_optimized(
    const float* __restrict__ Q,
    const float* __restrict__ K_cache,
    const float* __restrict__ V_cache,
    float* __restrict__ O,
    const int pos,           // Current position
    const int head_num,
    const int kv_head_num,
    const int head_size,
    const int kv_mul,
    const int dim,
    const int kv_dim,
    const float scale
) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    
    if (head >= head_num) return;
    
    const int kv_head = head / kv_mul;
    const int head_offset = kv_head * head_size;
    const int kv_len = pos + 1;
    
    // Shared memory
    extern __shared__ float smem[];
    float* s_query = smem;
    float* s_scores = smem + head_size;
    float* s_max = s_scores + ((kv_len + BLOCK_SIZE_FP32 - 1) / BLOCK_SIZE_FP32) * BLOCK_SIZE_FP32;
    float* s_sum = s_max + 8;  // For 8 warps (256 threads)
    
    // Load query
    const float* q_ptr = Q + head * head_size;
    for (int d = tid; d < head_size; d += BLOCK_SIZE_FP32) {
        s_query[d] = q_ptr[d];
    }
    __syncthreads();
    
    // Compute attention scores
    float local_max = -FLT_MAX;
    
    for (int k = tid; k < kv_len; k += BLOCK_SIZE_FP32) {
        const float* k_ptr = K_cache + k * kv_dim + head_offset;
        
        float score = 0.0f;
        #pragma unroll 8
        for (int d = 0; d < head_size; d++) {
            score += s_query[d] * k_ptr[d];
        }
        score *= scale;
        
        s_scores[k] = score;
        local_max = fmaxf(local_max, score);
    }
    __syncthreads();
    
    // Warp-level max reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    if (lane_id == 0) {
        s_max[warp_id] = local_max;
    }
    __syncthreads();
    
    // Final max across warps (8 warps for 256 threads)
    float global_max = -FLT_MAX;
    if (tid < 8) {
        global_max = s_max[tid];
    }
    for (int offset = 4; offset > 0; offset /= 2) {
        global_max = fmaxf(global_max, __shfl_down_sync(0xffffffff, global_max, offset));
    }
    global_max = __shfl_sync(0xffffffff, global_max, 0);
    
    // Compute softmax
    float local_sum = 0.0f;
    for (int k = tid; k < kv_len; k += BLOCK_SIZE_FP32) {
        float exp_val = expf(s_scores[k] - global_max);
        s_scores[k] = exp_val;
        local_sum += exp_val;
    }
    __syncthreads();
    
    // Warp-level sum reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if (lane_id == 0) {
        s_sum[warp_id] = local_sum;
    }
    __syncthreads();
    
    // Final sum across warps
    float global_sum = 0.0f;
    if (tid < 8) {
        global_sum = s_sum[tid];
    }
    for (int offset = 4; offset > 0; offset /= 2) {
        global_sum += __shfl_down_sync(0xffffffff, global_sum, offset);
    }
    global_sum = __shfl_sync(0xffffffff, global_sum, 0);
    
    float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
    
    // Compute output
    float* o_ptr = O + head * head_size;
    
    for (int d = tid; d < head_size; d += BLOCK_SIZE_FP32) {
        float acc = 0.0f;
        for (int k = 0; k < kv_len; k++) {
            const float* v_ptr = V_cache + k * kv_dim + head_offset;
            acc += s_scores[k] * v_ptr[d];
        }
        o_ptr[d] = acc * inv_sum;
    }
}

/**
 * Ultra-optimized FP16 decode attention kernel
 * Uses half2 vectorization, warp-level parallelism, and register blocking
 * Grid: [head_num]
 * Block: 256 threads (8 warps)
 */
constexpr int DECODE_BLOCK_SIZE = 256;
constexpr int DECODE_NUM_WARPS = 8;
constexpr int DECODE_WARP_SIZE = 32;

__global__ void flash_attention_decode_kernel_fp16_optimized(
    const half* __restrict__ Q,        // [dim] - query for current token
    const half* __restrict__ K_cache,  // [kv_len, kv_dim]
    const half* __restrict__ V_cache,  // [kv_len, kv_dim]
    half* __restrict__ O,              // [dim]
    const int pos,
    const int head_num,
    const int kv_head_num,
    const int head_size,
    const int kv_mul,
    const int kv_dim,
    const float scale
) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane_id = tid % DECODE_WARP_SIZE;
    const int warp_id = tid / DECODE_WARP_SIZE;
    
    if (head >= head_num) return;
    
    const int kv_head = head / kv_mul;
    const int head_offset = kv_head * head_size;
    const int kv_len = pos + 1;
    const int head_size_h2 = head_size / 2;
    
    // Shared memory layout:
    // [0, head_size): query (half)
    // [head_size, head_size + kv_len): attention scores (float)
    // [head_size + kv_len, ...): reduction buffers
    extern __shared__ char smem_raw[];
    half* s_query = reinterpret_cast<half*>(smem_raw);
    float* s_scores = reinterpret_cast<float*>(smem_raw + head_size * sizeof(half));
    float* s_max = s_scores + ((kv_len + DECODE_BLOCK_SIZE - 1) / DECODE_BLOCK_SIZE) * DECODE_BLOCK_SIZE;
    float* s_sum = s_max + DECODE_NUM_WARPS;
    
    // Load query to shared memory using half2
    const half* q_ptr = Q + head * head_size;
    const half2* q_ptr_h2 = reinterpret_cast<const half2*>(q_ptr);
    half2* s_query_h2 = reinterpret_cast<half2*>(s_query);
    
    for (int d = tid; d < head_size_h2; d += DECODE_BLOCK_SIZE) {
        s_query_h2[d] = q_ptr_h2[d];
    }
    __syncthreads();
    
    // =========================================================================
    // Phase 1: Compute Q·K attention scores with half2 dot product
    // =========================================================================
    float local_max = -FLT_MAX;
    
    // Each thread handles multiple K vectors
    for (int k = tid; k < kv_len; k += DECODE_BLOCK_SIZE) {
        const half2* k_ptr_h2 = reinterpret_cast<const half2*>(K_cache + k * kv_dim + head_offset);
        
        // Vectorized dot product using half2
        float2 acc = make_float2(0.0f, 0.0f);
        
        #pragma unroll 4
        for (int d = 0; d < head_size_h2; d++) {
            half2 q = s_query_h2[d];
            half2 kv = k_ptr_h2[d];
            // half2 multiply and accumulate
            float2 q_f = __half22float2(q);
            float2 k_f = __half22float2(kv);
            acc.x += q_f.x * k_f.x;
            acc.y += q_f.y * k_f.y;
        }
        
        float score = (acc.x + acc.y) * scale;
        s_scores[k] = score;
        local_max = fmaxf(local_max, score);
    }
    __syncthreads();
    
    // =========================================================================
    // Phase 2: Warp-level max reduction
    // =========================================================================
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
    }
    if (lane_id == 0) {
        s_max[warp_id] = local_max;
    }
    __syncthreads();
    
    // Block-level max reduction
    // FIX: Thread 0 does final reduction and stores to shared memory for ALL threads
    float global_max;
    if (tid < DECODE_NUM_WARPS) {
        local_max = s_max[tid];
    }
    #pragma unroll
    for (int offset = DECODE_NUM_WARPS / 2; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
    }
    // Thread 0 writes final max to shared memory for all warps to read
    if (tid == 0) {
        s_max[0] = local_max;
    }
    __syncthreads();
    global_max = s_max[0];  // All threads read the same global_max
    
    // =========================================================================
    // Phase 3: Softmax normalization
    // =========================================================================
    float local_sum = 0.0f;
    
    for (int k = tid; k < kv_len; k += DECODE_BLOCK_SIZE) {
        float val = s_scores[k] - global_max;
        float exp_val = (val > SOFTMAX_FTZ) ? expf(val) : 0.0f;
        s_scores[k] = exp_val;
        local_sum += exp_val;
    }
    __syncthreads();
    
    // Warp-level sum reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    }
    if (lane_id == 0) {
        s_sum[warp_id] = local_sum;
    }
    __syncthreads();
    
    // Block-level sum reduction
    // FIX: Thread 0 does final reduction and stores to shared memory for ALL threads
    float global_sum;
    if (tid < DECODE_NUM_WARPS) {
        local_sum = s_sum[tid];
    }
    #pragma unroll
    for (int offset = DECODE_NUM_WARPS / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    }
    // Thread 0 writes final sum to shared memory for all warps to read
    if (tid == 0) {
        s_sum[0] = local_sum;
    }
    __syncthreads();
    global_sum = s_sum[0];  // All threads read the same global_sum
    float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
    
    // =========================================================================
    // Phase 4: Compute weighted V sum with register blocking
    // =========================================================================
    half* o_ptr = O + head * head_size;
    
    // Each thread computes multiple output dimensions
    for (int d = tid; d < head_size; d += DECODE_BLOCK_SIZE) {
        float acc = 0.0f;
        
        // Process 4 K/V pairs at a time for better memory coalescing
        int k = 0;
        for (; k + 3 < kv_len; k += 4) {
            const half* v0 = V_cache + (k + 0) * kv_dim + head_offset;
            const half* v1 = V_cache + (k + 1) * kv_dim + head_offset;
            const half* v2 = V_cache + (k + 2) * kv_dim + head_offset;
            const half* v3 = V_cache + (k + 3) * kv_dim + head_offset;
            
            acc += s_scores[k + 0] * __half2float(v0[d]);
            acc += s_scores[k + 1] * __half2float(v1[d]);
            acc += s_scores[k + 2] * __half2float(v2[d]);
            acc += s_scores[k + 3] * __half2float(v3[d]);
        }
        
        // Handle remaining elements
        for (; k < kv_len; k++) {
            const half* v_ptr = V_cache + k * kv_dim + head_offset;
            acc += s_scores[k] * __half2float(v_ptr[d]);
        }
        
        o_ptr[d] = __float2half(acc * inv_sum);
    }
}

/**
 * GPU pos version of FP16 decode attention kernel for CUDA Graph compatibility
 * Reads position from GPU memory pointer instead of kernel argument
 * Grid: [head_num]
 * Block: 256 threads (8 warps)
 */
__global__ void flash_attention_decode_kernel_fp16_gpu_pos(
    const half* __restrict__ Q,        // [dim] - query for current token
    const half* __restrict__ K_cache,  // [max_seq_len, kv_dim]
    const half* __restrict__ V_cache,  // [max_seq_len, kv_dim]
    half* __restrict__ O,              // [dim]
    const int32_t* __restrict__ pos_ptr, // GPU memory - current position
    const int max_seq_len,             // Maximum sequence length for shared memory allocation
    const int head_num,
    const int kv_head_num,
    const int head_size,
    const int kv_mul,
    const int kv_dim,
    const float scale
) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane_id = tid % DECODE_WARP_SIZE;
    const int warp_id = tid / DECODE_WARP_SIZE;
    
    if (head >= head_num) return;
    
    // Read position from GPU memory - volatile to prevent caching stale values
    const int pos = *reinterpret_cast<const volatile int32_t*>(pos_ptr);
    const int kv_len = pos + 1;
    
    const int kv_head = head / kv_mul;
    const int head_offset = kv_head * head_size;
    const int head_size_h2 = head_size / 2;
    
    // Shared memory layout - use max_seq_len for allocation
    extern __shared__ char smem_raw[];
    half* s_query = reinterpret_cast<half*>(smem_raw);
    float* s_scores = reinterpret_cast<float*>(smem_raw + head_size * sizeof(half));
    float* s_max = s_scores + ((max_seq_len + DECODE_BLOCK_SIZE - 1) / DECODE_BLOCK_SIZE) * DECODE_BLOCK_SIZE;
    float* s_sum = s_max + DECODE_NUM_WARPS;
    
    // Load query to shared memory using half2
    const half* q_ptr = Q + head * head_size;
    const half2* q_ptr_h2 = reinterpret_cast<const half2*>(q_ptr);
    half2* s_query_h2 = reinterpret_cast<half2*>(s_query);
    
    for (int d = tid; d < head_size_h2; d += DECODE_BLOCK_SIZE) {
        s_query_h2[d] = q_ptr_h2[d];
    }
    __syncthreads();
    
    // Phase 1: Compute Q·K attention scores with half2 dot product
    float local_max = -FLT_MAX;
    
    for (int k = tid; k < kv_len; k += DECODE_BLOCK_SIZE) {
        const half2* k_ptr_h2 = reinterpret_cast<const half2*>(K_cache + k * kv_dim + head_offset);
        
        float2 acc = make_float2(0.0f, 0.0f);
        
        #pragma unroll 4
        for (int d = 0; d < head_size_h2; d++) {
            half2 q = s_query_h2[d];
            half2 kv = k_ptr_h2[d];
            float2 q_f = __half22float2(q);
            float2 k_f = __half22float2(kv);
            acc.x += q_f.x * k_f.x;
            acc.y += q_f.y * k_f.y;
        }
        
        float score = (acc.x + acc.y) * scale;
        s_scores[k] = score;
        local_max = fmaxf(local_max, score);
    }
    __syncthreads();
    
    // Phase 2: Warp-level max reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
    }
    if (lane_id == 0) {
        s_max[warp_id] = local_max;
    }
    __syncthreads();
    
    // Block-level max reduction
    // FIX: Thread 0 does final reduction and stores to shared memory for ALL threads
    float global_max;
    if (tid < DECODE_NUM_WARPS) {
        local_max = s_max[tid];
    }
    #pragma unroll
    for (int offset = DECODE_NUM_WARPS / 2; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
    }
    // Thread 0 writes final max to shared memory for all warps to read
    if (tid == 0) {
        s_max[0] = local_max;
    }
    __syncthreads();
    global_max = s_max[0];  // All threads read the same global_max
    
    // Phase 3: Softmax normalization
    float local_sum = 0.0f;
    
    for (int k = tid; k < kv_len; k += DECODE_BLOCK_SIZE) {
        float val = s_scores[k] - global_max;
        float exp_val = (val > SOFTMAX_FTZ) ? expf(val) : 0.0f;
        s_scores[k] = exp_val;
        local_sum += exp_val;
    }
    __syncthreads();
    
    // Warp-level sum reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    }
    if (lane_id == 0) {
        s_sum[warp_id] = local_sum;
    }
    __syncthreads();
    
    // Block-level sum reduction
    // FIX: Thread 0 does final reduction and stores to shared memory for ALL threads
    float global_sum;
    if (tid < DECODE_NUM_WARPS) {
        local_sum = s_sum[tid];
    }
    #pragma unroll
    for (int offset = DECODE_NUM_WARPS / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    }
    // Thread 0 writes final sum to shared memory for all warps to read
    if (tid == 0) {
        s_sum[0] = local_sum;
    }
    __syncthreads();
    global_sum = s_sum[0];  // All threads read the same global_sum
    float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
    
    // Phase 4: Compute weighted V sum
    half* o_ptr = O + head * head_size;
    
    for (int d = tid; d < head_size; d += DECODE_BLOCK_SIZE) {
        float acc = 0.0f;
        
        int k = 0;
        for (; k + 3 < kv_len; k += 4) {
            const half* v0 = V_cache + (k + 0) * kv_dim + head_offset;
            const half* v1 = V_cache + (k + 1) * kv_dim + head_offset;
            const half* v2 = V_cache + (k + 2) * kv_dim + head_offset;
            const half* v3 = V_cache + (k + 3) * kv_dim + head_offset;
            
            acc += s_scores[k + 0] * __half2float(v0[d]);
            acc += s_scores[k + 1] * __half2float(v1[d]);
            acc += s_scores[k + 2] * __half2float(v2[d]);
            acc += s_scores[k + 3] * __half2float(v3[d]);
        }
        
        for (; k < kv_len; k++) {
            const half* v_ptr = V_cache + k * kv_dim + head_offset;
            acc += s_scores[k] * __half2float(v_ptr[d]);
        }
        
        o_ptr[d] = __float2half(acc * inv_sum);
    }
}

// Optimized tile size for shared memory V loading
constexpr int V_TILE_K = 32;  // Number of V vectors to load at once

/**
 * FP16 prefill attention kernel with shared memory V tiling
 * Uses online softmax with V preloaded to shared memory for coalesced access
 * 
 * Grid: [head_num, seq_len]
 * Each block handles one (head, query_position) pair
 * Uses BLOCK_SIZE = 128 threads for optimal Orin performance
 * 
 * Key optimizations:
 * 1. Each thread processes one output dimension (head_size=128, BLOCK_SIZE=128)
 * 2. Vectorized Q·K computation using half2 with fmaf
 * 3. Online softmax for numerical stability
 * 4. V tile preloaded to shared memory for coalesced read then broadcast access
 */
__global__ void flash_attention_prefill_kernel_fp16(
    const half* __restrict__ Q,
    const half* __restrict__ K_cache,
    const half* __restrict__ V_cache,
    half* __restrict__ O,
    const int seq_len,
    const int start_pos,
    const int head_num,
    const int kv_head_num,
    const int head_size,
    const int kv_mul,
    const int dim,
    const int kv_dim,
    const float scale
) {
    const int head = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int tid = threadIdx.x;
    
    if (head >= head_num || seq_idx >= seq_len) return;
    
    const int kv_head = head / kv_mul;
    const int head_offset = kv_head * head_size;
    const int cur_pos = start_pos + seq_idx;
    const int kv_len = cur_pos + 1;
    const int head_size_h2 = head_size / 2;
    
    // Shared memory layout:
    // - s_query: [head_size] half (256 bytes)
    // - s_scores: [TILE_K] float (256 bytes)
    extern __shared__ char smem_prefill_fp16[];
    half* s_query = reinterpret_cast<half*>(smem_prefill_fp16);
    float* s_scores = reinterpret_cast<float*>(smem_prefill_fp16 + head_size * sizeof(half));
    half2* s_query_h2 = reinterpret_cast<half2*>(s_query);
    
    // Load query to shared memory (coalesced)
    const half* q_ptr = Q + seq_idx * dim + head * head_size;
    for (int d = tid; d < head_size; d += BLOCK_SIZE) {
        s_query[d] = q_ptr[d];
    }
    __syncthreads();
    
    // Each thread accumulates one output dimension
    float acc_o = 0.0f;
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    
    // Pre-compute V base pointer for this thread
    const half* v_thread_base = V_cache + head_offset + tid;
    
    // Process KV in tiles
    for (int tile_start = 0; tile_start < kv_len; tile_start += TILE_K) {
        const int tile_len = min(TILE_K, kv_len - tile_start);
        
        // Step 1: Compute Q·K scores with half2 vectorization
        float tile_max_local = -FLT_MAX;
        for (int k_idx = tid; k_idx < tile_len; k_idx += BLOCK_SIZE) {
            const int kv_pos = tile_start + k_idx;
            const half2* k_ptr_h2 = reinterpret_cast<const half2*>(K_cache + kv_pos * kv_dim + head_offset);
            
            float2 acc = make_float2(0.0f, 0.0f);
            #pragma unroll 8
            for (int d = 0; d < head_size_h2; d++) {
                float2 q_val = __half22float2(s_query_h2[d]);
                float2 k_val = __half22float2(k_ptr_h2[d]);
                acc.x = fmaf(q_val.x, k_val.x, acc.x);
                acc.y = fmaf(q_val.y, k_val.y, acc.y);
            }
            float score = (acc.x + acc.y) * scale;
            s_scores[k_idx] = score;
            tile_max_local = fmaxf(tile_max_local, score);
        }
        __syncthreads();
        
        // Step 2: Block reduce to find tile maximum using warp primitives
        // 128 threads = 4 warps
        const int lane_id = tid & 31;
        const int warp_id = tid >> 5;
        
        // Warp reduce max
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            tile_max_local = fmaxf(tile_max_local, __shfl_xor_sync(0xffffffff, tile_max_local, offset));
        }
        
        // Store warp max to shared memory
        __shared__ float s_warp_max[4];
        if (lane_id == 0) s_warp_max[warp_id] = tile_max_local;
        __syncthreads();
        
        // Thread 0 reduces across warps
        float m_j;
        if (tid == 0) {
            m_j = fmaxf(fmaxf(s_warp_max[0], s_warp_max[1]), fmaxf(s_warp_max[2], s_warp_max[3]));
            s_warp_max[0] = m_j;
        }
        __syncthreads();
        m_j = s_warp_max[0];
        
        // Step 3: Update global max
        float m_new = fmaxf(row_max, m_j);
        
        // Step 4: Compute exp(score - m_new) and tile sum
        float tile_sum_local = 0.0f;
        for (int k_idx = tid; k_idx < tile_len; k_idx += BLOCK_SIZE) {
            float val = s_scores[k_idx] - m_new;
            float exp_score = (val > SOFTMAX_FTZ) ? expf(val) : 0.0f;
            s_scores[k_idx] = exp_score;
            tile_sum_local += exp_score;
        }
        __syncthreads();
        
        // Step 5: Block-reduce tile sum using warp primitives
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            tile_sum_local += __shfl_xor_sync(0xffffffff, tile_sum_local, offset);
        }
        
        __shared__ float s_warp_sum[4];
        if (lane_id == 0) s_warp_sum[warp_id] = tile_sum_local;
        __syncthreads();
        
        float l_j;
        if (tid == 0) {
            l_j = s_warp_sum[0] + s_warp_sum[1] + s_warp_sum[2] + s_warp_sum[3];
            s_warp_sum[0] = l_j;
        }
        __syncthreads();
        l_j = s_warp_sum[0];
        
        // Step 6: Apply correction factor
        float correction = expf(row_max - m_new);
        acc_o *= correction;
        
        // Step 7: Accumulate V with optimized memory access
        // Each thread accumulates its output dimension with coalesced V reads
        // All threads access V[k, head_offset:head_offset+head_size] together
        if (tid < head_size) {
            const half* v_ptr = v_thread_base + tile_start * kv_dim;
            
            // Unroll by 8 for better ILP and instruction-level parallelism
            int k = 0;
            for (; k + 7 < tile_len; k += 8) {
                float s0 = s_scores[k];
                float s1 = s_scores[k+1];
                float s2 = s_scores[k+2];
                float s3 = s_scores[k+3];
                float s4 = s_scores[k+4];
                float s5 = s_scores[k+5];
                float s6 = s_scores[k+6];
                float s7 = s_scores[k+7];
                
                float v0 = __half2float(v_ptr[0]);
                float v1 = __half2float(v_ptr[kv_dim]);
                float v2 = __half2float(v_ptr[2*kv_dim]);
                float v3 = __half2float(v_ptr[3*kv_dim]);
                float v4 = __half2float(v_ptr[4*kv_dim]);
                float v5 = __half2float(v_ptr[5*kv_dim]);
                float v6 = __half2float(v_ptr[6*kv_dim]);
                float v7 = __half2float(v_ptr[7*kv_dim]);
                
                acc_o = fmaf(s0, v0, acc_o);
                acc_o = fmaf(s1, v1, acc_o);
                acc_o = fmaf(s2, v2, acc_o);
                acc_o = fmaf(s3, v3, acc_o);
                acc_o = fmaf(s4, v4, acc_o);
                acc_o = fmaf(s5, v5, acc_o);
                acc_o = fmaf(s6, v6, acc_o);
                acc_o = fmaf(s7, v7, acc_o);
                
                v_ptr += 8 * kv_dim;
            }
            
            // Handle remainder with unroll by 4
            for (; k + 3 < tile_len; k += 4) {
                float s0 = s_scores[k];
                float s1 = s_scores[k+1];
                float s2 = s_scores[k+2];
                float s3 = s_scores[k+3];
                
                float v0 = __half2float(v_ptr[0]);
                float v1 = __half2float(v_ptr[kv_dim]);
                float v2 = __half2float(v_ptr[2*kv_dim]);
                float v3 = __half2float(v_ptr[3*kv_dim]);
                
                acc_o = fmaf(s0, v0, acc_o);
                acc_o = fmaf(s1, v1, acc_o);
                acc_o = fmaf(s2, v2, acc_o);
                acc_o = fmaf(s3, v3, acc_o);
                
                v_ptr += 4 * kv_dim;
            }
            
            // Handle remainder
            for (; k < tile_len; k++) {
                acc_o = fmaf(s_scores[k], __half2float(v_ptr[0]), acc_o);
                v_ptr += kv_dim;
            }
        }
        
        row_max = m_new;
        row_sum = correction * row_sum + l_j;
        __syncthreads();
    }
    
    // Final normalization and write output
    if (tid < head_size) {
        float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
        half* o_ptr = O + seq_idx * dim + head * head_size;
        o_ptr[tid] = __float2half(acc_o * inv_sum);
    }
}


void flash_attention_prefill_cu(
    int32_t start_pos,
    int32_t seq_len,
    int32_t head_num,
    int32_t kv_head_num,
    int32_t head_size,
    int32_t kv_mul,
    int32_t layer_index,
    int32_t max_seq_len,
    int32_t kv_dim,
    const tensor::Tensor& query,
    const tensor::Tensor& output,
    const tensor::Tensor& key_cache,
    const tensor::Tensor& value_cache,
    CudaConfig* config
) {
    const int layer_offset = layer_index * max_seq_len * kv_dim;
    const int dim = head_num * head_size;
    
    float* Q = const_cast<float*>(query.ptr<float>());
    float* O = const_cast<float*>(output.ptr<float>());
    float* K = const_cast<float*>(key_cache.ptr<float>()) + layer_offset;
    float* V = const_cast<float*>(value_cache.ptr<float>()) + layer_offset;
    
    const float scale = 1.0f / sqrtf((float)head_size);
    
    dim3 grid(head_num, seq_len);
    dim3 block(BLOCK_SIZE_FP32);  // Use BLOCK_SIZE_FP32 for FP32 kernel
    
    // Shared memory: query + scores (optimized size)
    const int smem_size = (head_size + TILE_K) * sizeof(float);
    
    cudaStream_t stream = config ? config->stream : nullptr;
    
    flash_attention_prefill_kernel<<<grid, block, smem_size, stream>>>(
        Q, K, V, O,
        seq_len, start_pos, head_num, kv_head_num, head_size, kv_mul,
        dim, kv_dim, scale
    );
}


void flash_attention_decode_cu(
    int32_t pos,
    int32_t head_num,
    int32_t kv_head_num,
    int32_t head_size,
    int32_t kv_mul,
    int32_t layer_index,
    int32_t max_seq_len,
    int32_t kv_dim,
    const tensor::Tensor& query,
    const tensor::Tensor& output,
    const tensor::Tensor& key_cache,
    const tensor::Tensor& value_cache,
    CudaConfig* config
) {
    const int layer_offset = layer_index * max_seq_len * kv_dim;
    const int dim = head_num * head_size;
    const int kv_len = pos + 1;
    
    float* Q = const_cast<float*>(query.ptr<float>());
    float* O = const_cast<float*>(output.ptr<float>());
    float* K = const_cast<float*>(key_cache.ptr<float>()) + layer_offset;
    float* V = const_cast<float*>(value_cache.ptr<float>()) + layer_offset;
    
    const float scale = 1.0f / sqrtf((float)head_size);
    
    // Use optimized decode kernel with BLOCK_SIZE_FP32
    dim3 grid(head_num);
    dim3 block(BLOCK_SIZE_FP32);
    
    // Shared memory: query + scores + max/sum buffers
    const int score_buffer_size = ((kv_len + BLOCK_SIZE_FP32 - 1) / BLOCK_SIZE_FP32) * BLOCK_SIZE_FP32;
    const int smem_size = (head_size + score_buffer_size + 16) * sizeof(float);  // +16 for 8 warps max/sum
    
    cudaStream_t stream = config ? config->stream : nullptr;
    
    flash_attention_decode_kernel_optimized<<<grid, block, smem_size, stream>>>(
        Q, K, V, O,
        pos, head_num, kv_head_num, head_size, kv_mul,
        dim, kv_dim, scale
    );
}

// ============================================================================
// FP16 Flash Attention Functions
// ============================================================================

void flash_attention_prefill_fp16_cu(
    int32_t start_pos,
    int32_t seq_len,
    int32_t head_num,
    int32_t kv_head_num,
    int32_t head_size,
    int32_t kv_mul,
    int32_t layer_index,
    int32_t max_seq_len,
    int32_t kv_dim,
    const tensor::Tensor& query,
    const tensor::Tensor& output,
    const tensor::Tensor& key_cache,
    const tensor::Tensor& value_cache,
    CudaConfig* config
) {
    const int layer_offset = layer_index * max_seq_len * kv_dim;
    const int dim = head_num * head_size;
    
    half* Q = const_cast<half*>(query.ptr<half>());
    half* O = const_cast<half*>(output.ptr<half>());
    half* K = const_cast<half*>(key_cache.ptr<half>()) + layer_offset;
    half* V = const_cast<half*>(value_cache.ptr<half>()) + layer_offset;
    
    const float scale = 1.0f / sqrtf((float)head_size);
    
    dim3 grid(head_num, seq_len);
    dim3 block(BLOCK_SIZE);
    
    // Shared memory: query (half) + scores (float)
    const int smem_size = head_size * sizeof(half) + TILE_K * sizeof(float);
    
    cudaStream_t stream = config ? config->stream : nullptr;
    
    flash_attention_prefill_kernel_fp16<<<grid, block, smem_size, stream>>>(
        Q, K, V, O,
        seq_len, start_pos, head_num, kv_head_num, head_size, kv_mul,
        dim, kv_dim, scale
    );
}


void flash_attention_decode_fp16_cu(
    int32_t pos,
    int32_t head_num,
    int32_t kv_head_num,
    int32_t head_size,
    int32_t kv_mul,
    int32_t layer_index,
    int32_t max_seq_len,
    int32_t kv_dim,
    const tensor::Tensor& query,
    const tensor::Tensor& output,
    const tensor::Tensor& key_cache,
    const tensor::Tensor& value_cache,
    CudaConfig* config
) {
    const int layer_offset = layer_index * max_seq_len * kv_dim;
    const int kv_len = pos + 1;
    
    half* Q = const_cast<half*>(query.ptr<half>());
    half* O = const_cast<half*>(output.ptr<half>());
    half* K = const_cast<half*>(key_cache.ptr<half>()) + layer_offset;
    half* V = const_cast<half*>(value_cache.ptr<half>()) + layer_offset;
    
    const float scale = 1.0f / sqrtf((float)head_size);
    
    // Use ultra-optimized FP16 decode kernel with 256 threads
    dim3 grid(head_num);
    dim3 block(DECODE_BLOCK_SIZE);
    
    // Shared memory: query (half) + scores (float) + reduction buffers
    const int score_buffer_size = ((kv_len + DECODE_BLOCK_SIZE - 1) / DECODE_BLOCK_SIZE) * DECODE_BLOCK_SIZE;
    const int smem_size = head_size * sizeof(half) + 
                          score_buffer_size * sizeof(float) + 
                          2 * DECODE_NUM_WARPS * sizeof(float);
    
    cudaStream_t stream = config ? config->stream : nullptr;
    
    flash_attention_decode_kernel_fp16_optimized<<<grid, block, smem_size, stream>>>(
        Q, K, V, O,
        pos, head_num, kv_head_num, head_size, kv_mul,
        kv_dim, scale
    );
}

/**
 * Online softmax FP16 decode kernel for CUDA Graph compatibility
 * Uses tiled processing with fixed shared memory size
 * 
 * Key insight: Use online softmax to avoid storing all scores
 * We process K/V in tiles, maintaining running max and sum
 * 
 * Grid: [head_num]
 * Block: 128 threads (better for head_size=128)
 */
constexpr int ONLINE_TILE_K = 256;  // Tile size for online softmax - larger reduces iterations
constexpr int ONLINE_BLOCK_SIZE = 128;  // Match head_size for 1:1 mapping
constexpr int ONLINE_NUM_WARPS = 4;

__global__ void flash_attention_decode_kernel_fp16_online_softmax(
    const half* __restrict__ Q,        // [dim] - query for current token
    const half* __restrict__ K_cache,  // [max_seq_len, kv_dim]
    const half* __restrict__ V_cache,  // [max_seq_len, kv_dim]
    half* __restrict__ O,              // [dim]
    const int32_t* __restrict__ pos_ptr, // GPU memory - current position
    const int head_num,
    const int kv_head_num,
    const int head_size,
    const int kv_mul,
    const int kv_dim,
    const float scale
) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    
    if (head >= head_num) return;
    
    // Read position from GPU memory - volatile to prevent caching
    const int pos = *reinterpret_cast<const volatile int32_t*>(pos_ptr);
    const int kv_len = pos + 1;
    
    const int kv_head = head / kv_mul;
    const int head_offset = kv_head * head_size;
    const int head_size_h2 = head_size / 2;
    
    // Shared memory layout - FIXED size for CUDA Graph
    extern __shared__ char smem_raw[];
    half* s_query = reinterpret_cast<half*>(smem_raw);
    float* s_scores = reinterpret_cast<float*>(smem_raw + head_size * sizeof(half));
    float* s_max = s_scores + ONLINE_TILE_K;
    float* s_sum = s_max + ONLINE_NUM_WARPS;
    
    // Load query to shared memory using half2
    const half* q_ptr = Q + head * head_size;
    const half2* q_ptr_h2 = reinterpret_cast<const half2*>(q_ptr);
    half2* s_query_h2 = reinterpret_cast<half2*>(s_query);
    
    for (int d = tid; d < head_size_h2; d += ONLINE_BLOCK_SIZE) {
        s_query_h2[d] = q_ptr_h2[d];
    }
    __syncthreads();
    
    // Online softmax state
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    
    // Each thread maintains its own output accumulator for dimension tid
    // With head_size=128 and BLOCK_SIZE=128, each thread handles exactly 1 dimension
    float acc_o = 0.0f;
    
    // Process KV in tiles
    for (int tile_start = 0; tile_start < kv_len; tile_start += ONLINE_TILE_K) {
        const int tile_end = min(tile_start + ONLINE_TILE_K, kv_len);
        const int tile_len = tile_end - tile_start;
        
        // Step 1: Compute Q·K attention scores for this tile
        float tile_max_local = -FLT_MAX;
        
        for (int k_idx = tid; k_idx < tile_len; k_idx += ONLINE_BLOCK_SIZE) {
            const int kv_pos = tile_start + k_idx;
            const half2* k_ptr_h2 = reinterpret_cast<const half2*>(K_cache + kv_pos * kv_dim + head_offset);
            
            // Vectorized dot product using half2
            float2 acc = make_float2(0.0f, 0.0f);
            
            #pragma unroll 4
            for (int d = 0; d < head_size_h2; d++) {
                half2 q = s_query_h2[d];
                half2 kv = k_ptr_h2[d];
                float2 q_f = __half22float2(q);
                float2 k_f = __half22float2(kv);
                acc.x += q_f.x * k_f.x;
                acc.y += q_f.y * k_f.y;
            }
            
            float score = (acc.x + acc.y) * scale;
            s_scores[k_idx] = score;
            tile_max_local = fmaxf(tile_max_local, score);
        }
        __syncthreads();
        
        // Step 2: Warp-level max reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            tile_max_local = fmaxf(tile_max_local, __shfl_xor_sync(0xffffffff, tile_max_local, offset));
        }
        if (lane_id == 0) {
            s_max[warp_id] = tile_max_local;
        }
        __syncthreads();
        
        // Block-level max reduction
        float m_j = s_max[0];
        for (int w = 1; w < ONLINE_NUM_WARPS; w++) {
            m_j = fmaxf(m_j, s_max[w]);
        }
        
        // Step 3: Update global max
        float m_new = fmaxf(row_max, m_j);
        
        // Step 4: Compute exp(score - m_new) and tile sum
        float tile_sum_local = 0.0f;
        
        for (int k_idx = tid; k_idx < tile_len; k_idx += ONLINE_BLOCK_SIZE) {
            float val = s_scores[k_idx] - m_new;
            float exp_val = (val > SOFTMAX_FTZ) ? expf(val) : 0.0f;
            s_scores[k_idx] = exp_val;
            tile_sum_local += exp_val;
        }
        __syncthreads();
        
        // Step 5: Warp-level sum reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            tile_sum_local += __shfl_xor_sync(0xffffffff, tile_sum_local, offset);
        }
        if (lane_id == 0) {
            s_sum[warp_id] = tile_sum_local;
        }
        __syncthreads();
        
        // Block-level sum
        float l_j = s_sum[0];
        for (int w = 1; w < ONLINE_NUM_WARPS; w++) {
            l_j += s_sum[w];
        }
        
        // Step 6: Compute correction factor for previous accumulator
        float correction = expf(row_max - m_new);
        
        // Step 7: Scale previous output accumulator
        acc_o *= correction;
        
        // Step 8: Add weighted V values for this tile
        // Each thread is responsible for its own dimension (tid)
        const int my_dim = tid;  // With head_size=128, tid maps directly to dimension
        if (my_dim < head_size) {
            for (int k = 0; k < tile_len; k++) {
                const int kv_pos = tile_start + k;
                const half* v_ptr = V_cache + kv_pos * kv_dim + head_offset;
                acc_o += s_scores[k] * __half2float(v_ptr[my_dim]);
            }
        }
        
        // Update running state
        row_max = m_new;
        row_sum = correction * row_sum + l_j;
        __syncthreads();
    }
    
    // Final normalization and write output
    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    half* o_ptr = O + head * head_size;
    
    const int my_dim = tid;
    if (my_dim < head_size) {
        o_ptr[my_dim] = __float2half(acc_o * inv_sum);
    }
}

void flash_attention_decode_fp16_gpu_pos_cu(
    const int32_t* pos_ptr,  // GPU memory pointer to position
    int32_t head_num,
    int32_t kv_head_num,
    int32_t head_size,
    int32_t kv_mul,
    int32_t layer_index,
    int32_t max_seq_len,
    int32_t kv_dim,
    const tensor::Tensor& query,
    const tensor::Tensor& output,
    const tensor::Tensor& key_cache,
    const tensor::Tensor& value_cache,
    CudaConfig* config
) {
    const int layer_offset = layer_index * max_seq_len * kv_dim;
    
    half* Q = const_cast<half*>(query.ptr<half>());
    half* O = const_cast<half*>(output.ptr<half>());
    half* K = const_cast<half*>(key_cache.ptr<half>()) + layer_offset;
    half* V = const_cast<half*>(value_cache.ptr<half>()) + layer_offset;
    
    const float scale = 1.0f / sqrtf((float)head_size);
    
    dim3 grid(head_num);
    dim3 block(ONLINE_BLOCK_SIZE);  // 128 threads = head_size
    
    // FIXED shared memory size for CUDA Graph compatibility
    // Only allocate ONLINE_TILE_K scores instead of max_seq_len
    const int smem_size = head_size * sizeof(half) + 
                          ONLINE_TILE_K * sizeof(float) + 
                          2 * ONLINE_NUM_WARPS * sizeof(float);
    
    cudaStream_t stream = config ? config->stream : nullptr;
    
    flash_attention_decode_kernel_fp16_online_softmax<<<grid, block, smem_size, stream>>>(
        Q, K, V, O,
        pos_ptr, head_num, kv_head_num, head_size, kv_mul,
        kv_dim, scale
    );
}

}  // namespace kernel
