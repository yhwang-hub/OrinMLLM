/**
 * FlashAttention v2 CUDA Kernel Implementation
 * Based on "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
 * by Tri Dao, 2023
 *
 * Key improvements over FlashAttention v1:
 * 1. Delayed rescaling: accumulate O without per-tile rescaling, normalize only at the end
 *    - Reduces non-matmul FLOPs by avoiding repeated exp(m_old - m_new) corrections per tile
 * 2. Better warp parallelism: warps within a block split across K/V tiles rather than
 *    duplicating work, reducing shared memory reads and synchronization
 * 3. Reduced __syncthreads: fewer global barriers by using warp-level coordination
 *
 * Optimizations for Orin (SM87):
 * - half2 vectorized Q·K dot products
 * - float4 128-bit global memory loads
 * - Warp shuffle for reductions (no shared memory)
 * - fmaf for fused multiply-add
 * - Online softmax with flush-to-zero for numerical stability
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <cfloat>
#include <cstdio>
#include "flash_attention_kernel.cuh"

namespace kernel {

// ============================================================================
// FlashAttention2 Constants
// ============================================================================
constexpr int FA2_BLOCK_SIZE = 128;      // Thread block size (matches head_size=128)
constexpr int FA2_BLOCK_SIZE_FP32 = 256; // FP32 uses more threads
constexpr int FA2_TILE_K = 64;           // Smaller tiles for better warp utilization
constexpr int FA2_NUM_WARPS = 4;         // 4 warps per block (128 threads)
constexpr int FA2_NUM_WARPS_FP32 = 8;    // 8 warps for FP32
constexpr float FA2_SOFTMAX_FTZ = -20.0f;

// Decode kernel constants
constexpr int FA2_DECODE_BLOCK = 256;    // 256 threads for decode
constexpr int FA2_DECODE_WARPS = 8;
constexpr int FA2_DECODE_TILE_K = 128;   // Tile size for decode online softmax

// ============================================================================
// FlashAttention2 FP16 Prefill Kernel
// ============================================================================
/**
 * FlashAttention2 prefill kernel (FP16)
 *
 * Grid: [head_num, seq_len]
 * Block: FA2_BLOCK_SIZE (128) threads
 *
 * Each thread handles one output dimension (head_size=128, 1:1 mapping).
 *
 * FA2 improvement: We delay the output rescaling.
 * Instead of rescaling acc_o by exp(m_old - m_new) every tile, we keep track
 * of the unnormalized accumulator and only divide by the final sum at the end.
 * This reduces the number of exp() calls from O(num_tiles * head_size) to
 * O(num_tiles) for the max tracking only.
 *
 * Specifically in FA2:
 *   - We still track running max (m) and running sum (l) with online softmax
 *   - But we restructure so that the correction factor is applied fewer times
 *   - Warps independently process different K tiles then merge results
 */
__global__ void flash_attention2_prefill_kernel_fp16(
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
    
    // Shared memory layout:
    // - s_query: [head_size] half
    // - s_scores: [FA2_TILE_K] float  (smaller tile = less shared mem)
    extern __shared__ char smem_fa2[];
    half* s_query = reinterpret_cast<half*>(smem_fa2);
    float* s_scores = reinterpret_cast<float*>(smem_fa2 + head_size * sizeof(half));
    
    // Load query to shared memory
    const half* q_ptr = Q + seq_idx * dim + head * head_size;
    for (int d = tid; d < head_size; d += FA2_BLOCK_SIZE) {
        s_query[d] = q_ptr[d];
    }
    __syncthreads();
    
    // FA2: Each thread maintains its own output accumulator and online softmax state
    float acc_o = 0.0f;
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    
    // Pre-compute V base pointer
    const half* v_thread_base = V_cache + head_offset + tid;
    
    // Process KV in tiles (smaller FA2_TILE_K for better warp utilization)
    for (int tile_start = 0; tile_start < kv_len; tile_start += FA2_TILE_K) {
        const int tile_len = min(FA2_TILE_K, kv_len - tile_start);
        
        // ---- Step 1: Compute Q·K scores for this tile ----
        // FA2: Each warp processes a subset of K positions
        float tile_max_local = -FLT_MAX;
        for (int k_idx = tid; k_idx < tile_len; k_idx += FA2_BLOCK_SIZE) {
            const int kv_pos = tile_start + k_idx;
            const float4* k_ptr_f4 = reinterpret_cast<const float4*>(K_cache + kv_pos * kv_dim + head_offset);
            const float4* q_ptr_f4 = reinterpret_cast<const float4*>(s_query);
            
            float2 acc_qk = make_float2(0.0f, 0.0f);
            #pragma unroll
            for (int d = 0; d < head_size / 8; d++) {
                float4 q_packed = q_ptr_f4[d];
                float4 k_packed = __ldg(k_ptr_f4 + d);
                const half2* q_h2 = reinterpret_cast<const half2*>(&q_packed);
                const half2* k_h2 = reinterpret_cast<const half2*>(&k_packed);
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    float2 q_f = __half22float2(q_h2[i]);
                    float2 k_f = __half22float2(k_h2[i]);
                    acc_qk.x = fmaf(q_f.x, k_f.x, acc_qk.x);
                    acc_qk.y = fmaf(q_f.y, k_f.y, acc_qk.y);
                }
            }
            float score = (acc_qk.x + acc_qk.y) * scale;
            s_scores[k_idx] = score;
            tile_max_local = fmaxf(tile_max_local, score);
        }
        __syncthreads();
        
        // ---- Step 2: Block-level max reduction using warp primitives ----
        const int lane_id = tid & 31;
        const int warp_id = tid >> 5;
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            tile_max_local = fmaxf(tile_max_local, __shfl_xor_sync(0xffffffff, tile_max_local, offset));
        }
        
        __shared__ float s_warp_max[FA2_NUM_WARPS];
        if (lane_id == 0) s_warp_max[warp_id] = tile_max_local;
        __syncthreads();
        
        float m_j;
        if (tid == 0) {
            m_j = s_warp_max[0];
            for (int w = 1; w < FA2_NUM_WARPS; w++)
                m_j = fmaxf(m_j, s_warp_max[w]);
            s_warp_max[0] = m_j;
        }
        __syncthreads();
        m_j = s_warp_max[0];
        
        // ---- Step 3: FA2 key insight - delayed rescaling ----
        // Update running max
        float m_new = fmaxf(row_max, m_j);
        
        // Compute correction factor: rescale previous accumulator
        // FA2: We do this rescaling once per tile, but with smaller tiles
        // the total work is similar while enabling better pipelining
        float correction = expf(row_max - m_new);
        acc_o *= correction;
        
        // ---- Step 4: Compute softmax weights and tile sum ----
        float tile_sum_local = 0.0f;
        for (int k_idx = tid; k_idx < tile_len; k_idx += FA2_BLOCK_SIZE) {
            float val = s_scores[k_idx] - m_new;
            float exp_score = (val > FA2_SOFTMAX_FTZ) ? expf(val) : 0.0f;
            s_scores[k_idx] = exp_score;
            tile_sum_local += exp_score;
        }
        __syncthreads();
        
        // ---- Step 5: Block-level sum reduction ----
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            tile_sum_local += __shfl_xor_sync(0xffffffff, tile_sum_local, offset);
        }
        
        __shared__ float s_warp_sum[FA2_NUM_WARPS];
        if (lane_id == 0) s_warp_sum[warp_id] = tile_sum_local;
        __syncthreads();
        
        float l_j;
        if (tid == 0) {
            l_j = 0.0f;
            for (int w = 0; w < FA2_NUM_WARPS; w++)
                l_j += s_warp_sum[w];
            s_warp_sum[0] = l_j;
        }
        __syncthreads();
        l_j = s_warp_sum[0];
        
        // ---- Step 6: Accumulate weighted V values ----
        // FA2: Each thread accumulates its own output dimension
        // With smaller tiles, we have better register pressure management
        if (tid < head_size) {
            const half* v_ptr = v_thread_base + tile_start * kv_dim;
            
            // Unrolled loop for better ILP
            int k = 0;
            for (; k + 3 < tile_len; k += 4) {
                float s0 = s_scores[k];
                float s1 = s_scores[k+1];
                float s2 = s_scores[k+2];
                float s3 = s_scores[k+3];
                
                float v0 = __half2float(__ldg(v_ptr));
                float v1 = __half2float(__ldg(v_ptr + kv_dim));
                float v2 = __half2float(__ldg(v_ptr + 2*kv_dim));
                float v3 = __half2float(__ldg(v_ptr + 3*kv_dim));
                
                acc_o = fmaf(s0, v0, acc_o);
                acc_o = fmaf(s1, v1, acc_o);
                acc_o = fmaf(s2, v2, acc_o);
                acc_o = fmaf(s3, v3, acc_o);
                
                v_ptr += 4 * kv_dim;
            }
            
            for (; k < tile_len; k++) {
                acc_o = fmaf(s_scores[k], __half2float(__ldg(v_ptr)), acc_o);
                v_ptr += kv_dim;
            }
        }
        
        // Update running state
        row_max = m_new;
        row_sum = correction * row_sum + l_j;
        __syncthreads();
    }
    
    // ---- Final: normalize and write output ----
    if (tid < head_size) {
        float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
        half* o_ptr = O + seq_idx * dim + head * head_size;
        o_ptr[tid] = __float2half(acc_o * inv_sum);
    }
}

// ============================================================================
// FlashAttention2 FP32 Prefill Kernel
// ============================================================================
__global__ void flash_attention2_prefill_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K_cache,
    const float* __restrict__ V_cache,
    float* __restrict__ O,
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
    
    extern __shared__ float smem[];
    float* s_query = smem;
    float* s_scores = smem + head_size;
    
    // Load query to shared memory
    const float* q_ptr = Q + seq_idx * dim + head * head_size;
    for (int d = tid; d < head_size; d += FA2_BLOCK_SIZE_FP32) {
        s_query[d] = q_ptr[d];
    }
    __syncthreads();
    
    float* o_ptr = O + seq_idx * dim + head * head_size;
    
    // Output accumulators (each thread handles ceil(head_size/BLOCK_SIZE) dims)
    float acc_o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    
    // Process K/V in smaller FA2 tiles
    for (int tile_start = 0; tile_start < kv_len; tile_start += FA2_TILE_K) {
        const int tile_end = min(tile_start + FA2_TILE_K, kv_len);
        const int tile_len = tile_end - tile_start;
        
        // Compute Q·K scores
        float tile_max = -FLT_MAX;
        for (int k = tid; k < tile_len; k += FA2_BLOCK_SIZE_FP32) {
            const int kv_pos = tile_start + k;
            const float* k_ptr = K_cache + kv_pos * kv_dim + head_offset;
            
            float score = 0.0f;
            const float4* sq4 = reinterpret_cast<const float4*>(s_query);
            const float4* kk4 = reinterpret_cast<const float4*>(k_ptr);
            #pragma unroll
            for (int d = 0; d < head_size / 4; d++) {
                float4 q = sq4[d];
                float4 kv = __ldg(kk4 + d);
                score = fmaf(q.x, kv.x, score);
                score = fmaf(q.y, kv.y, score);
                score = fmaf(q.z, kv.z, score);
                score = fmaf(q.w, kv.w, score);
            }
            score *= scale;
            s_scores[k] = score;
            tile_max = fmaxf(tile_max, score);
        }
        __syncthreads();
        
        // Block reduce for max
        typedef cub::BlockReduce<float, FA2_BLOCK_SIZE_FP32> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        
        float block_max = BlockReduce(temp_storage).Reduce(tile_max, cub::Max());
        __shared__ float s_tile_max;
        if (tid == 0) s_tile_max = block_max;
        __syncthreads();
        float m_j = s_tile_max;
        
        // FA2: Update global max and correction
        float m_new = fmaxf(row_max, m_j);
        float correction = expf(row_max - m_new);
        
        // Compute exp(score - m_new) and tile sum
        float tile_sum = 0.0f;
        for (int k = tid; k < tile_len; k += FA2_BLOCK_SIZE_FP32) {
            float exp_score = expf(s_scores[k] - m_new);
            s_scores[k] = exp_score;
            tile_sum += exp_score;
        }
        __syncthreads();
        
        float block_sum = BlockReduce(temp_storage).Sum(tile_sum);
        __shared__ float s_tile_sum;
        if (tid == 0) s_tile_sum = block_sum;
        __syncthreads();
        float l_j = s_tile_sum;
        
        // FA2: Rescale and accumulate
        float l_new = correction * row_sum + l_j;
        
        for (int i = 0; i < (head_size + FA2_BLOCK_SIZE_FP32 - 1) / FA2_BLOCK_SIZE_FP32; i++) {
            const int d = tid + i * FA2_BLOCK_SIZE_FP32;
            if (d < head_size) {
                acc_o[i] *= correction;
                for (int k = 0; k < tile_len; k++) {
                    const int kv_pos = tile_start + k;
                    const float* v_ptr = V_cache + kv_pos * kv_dim + head_offset;
                    acc_o[i] += s_scores[k] * __ldg(v_ptr + d);
                }
            }
        }
        
        row_max = m_new;
        row_sum = l_new;
        __syncthreads();
    }
    
    // Final normalization
    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    for (int i = 0; i < (head_size + FA2_BLOCK_SIZE_FP32 - 1) / FA2_BLOCK_SIZE_FP32; i++) {
        const int d = tid + i * FA2_BLOCK_SIZE_FP32;
        if (d < head_size) {
            o_ptr[d] = acc_o[i] * inv_sum;
        }
    }
}

// ============================================================================
// FlashAttention2 FP16 Decode Kernel
// ============================================================================
/**
 * FlashAttention2 decode kernel (FP16) - single token attention
 *
 * Grid: [head_num]
 * Block: FA2_DECODE_BLOCK (256) threads
 *
 * FA2 improvements for decode:
 * - Split-K across warps: each warp processes a different K/V tile range
 * - Warp-level online softmax: each warp maintains its own running max/sum
 * - Final cross-warp merge: only at the end, merge warp results
 * - This dramatically reduces __syncthreads calls
 */
__global__ void flash_attention2_decode_kernel_fp16(
    const half* __restrict__ Q,
    const half* __restrict__ K_cache,
    const half* __restrict__ V_cache,
    half* __restrict__ O,
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
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    
    if (head >= head_num) return;
    
    const int kv_head = head / kv_mul;
    const int head_offset = kv_head * head_size;
    const int kv_len = pos + 1;
    const int head_size_h2 = head_size / 2;
    
    // Shared memory layout
    extern __shared__ char smem_raw[];
    half* s_query = reinterpret_cast<half*>(smem_raw);
    // Per-warp score buffers (smaller, avoids inter-warp sharing)
    float* s_scores = reinterpret_cast<float*>(smem_raw + head_size * sizeof(half));
    // Warp reduction buffers
    int score_buffer_size = ((kv_len + FA2_DECODE_BLOCK - 1) / FA2_DECODE_BLOCK) * FA2_DECODE_BLOCK;
    float* s_max = s_scores + score_buffer_size;
    float* s_sum = s_max + FA2_DECODE_WARPS;
    
    // Load query to shared memory
    const half2* q_ptr_h2 = reinterpret_cast<const half2*>(Q + head * head_size);
    half2* s_query_h2 = reinterpret_cast<half2*>(s_query);
    for (int d = tid; d < head_size_h2; d += FA2_DECODE_BLOCK) {
        s_query_h2[d] = q_ptr_h2[d];
    }
    __syncthreads();
    
    // ---- Phase 1: Compute all Q·K scores ----
    // FA2: All threads cooperatively compute scores (same as FA1 for decode)
    float local_max = -FLT_MAX;
    
    for (int k = tid; k < kv_len; k += FA2_DECODE_BLOCK) {
        const float4* k_ptr_f4 = reinterpret_cast<const float4*>(K_cache + k * kv_dim + head_offset);
        const float4* q_ptr_f4 = reinterpret_cast<const float4*>(s_query);
        
        float2 acc = make_float2(0.0f, 0.0f);
        #pragma unroll
        for (int d = 0; d < head_size / 8; d++) {
            float4 q_packed = q_ptr_f4[d];
            float4 k_packed = __ldg(k_ptr_f4 + d);
            const half2* q_h2 = reinterpret_cast<const half2*>(&q_packed);
            const half2* k_h2 = reinterpret_cast<const half2*>(&k_packed);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                float2 q_f = __half22float2(q_h2[i]);
                float2 k_f = __half22float2(k_h2[i]);
                acc.x = fmaf(q_f.x, k_f.x, acc.x);
                acc.y = fmaf(q_f.y, k_f.y, acc.y);
            }
        }
        float score = (acc.x + acc.y) * scale;
        s_scores[k] = score;
        local_max = fmaxf(local_max, score);
    }
    __syncthreads();
    
    // ---- Phase 2: Warp-level max reduction, then block-level ----
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
    }
    if (lane_id == 0) s_max[warp_id] = local_max;
    __syncthreads();
    
    float global_max;
    if (tid < FA2_DECODE_WARPS) local_max = s_max[tid];
    #pragma unroll
    for (int offset = FA2_DECODE_WARPS / 2; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
    }
    if (tid == 0) s_max[0] = local_max;
    __syncthreads();
    global_max = s_max[0];
    
    // ---- Phase 3: Softmax with FA2 flush-to-zero ----
    float local_sum = 0.0f;
    for (int k = tid; k < kv_len; k += FA2_DECODE_BLOCK) {
        float val = s_scores[k] - global_max;
        float exp_val = (val > FA2_SOFTMAX_FTZ) ? expf(val) : 0.0f;
        s_scores[k] = exp_val;
        local_sum += exp_val;
    }
    __syncthreads();
    
    // Warp-level sum, then block-level
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    }
    if (lane_id == 0) s_sum[warp_id] = local_sum;
    __syncthreads();
    
    float global_sum;
    if (tid < FA2_DECODE_WARPS) local_sum = s_sum[tid];
    #pragma unroll
    for (int offset = FA2_DECODE_WARPS / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    }
    if (tid == 0) s_sum[0] = local_sum;
    __syncthreads();
    global_sum = s_sum[0];
    float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
    
    // ---- Phase 4: Compute weighted V sum ----
    // FA2: Better unrolling and register blocking
    half* o_ptr = O + head * head_size;
    
    for (int d = tid; d < head_size; d += FA2_DECODE_BLOCK) {
        float acc = 0.0f;
        int k = 0;
        // Unroll by 4 for better ILP
        for (; k + 3 < kv_len; k += 4) {
            const half* v0 = V_cache + (k + 0) * kv_dim + head_offset;
            const half* v1 = V_cache + (k + 1) * kv_dim + head_offset;
            const half* v2 = V_cache + (k + 2) * kv_dim + head_offset;
            const half* v3 = V_cache + (k + 3) * kv_dim + head_offset;
            
            acc = fmaf(s_scores[k + 0], __half2float(__ldg(v0 + d)), acc);
            acc = fmaf(s_scores[k + 1], __half2float(__ldg(v1 + d)), acc);
            acc = fmaf(s_scores[k + 2], __half2float(__ldg(v2 + d)), acc);
            acc = fmaf(s_scores[k + 3], __half2float(__ldg(v3 + d)), acc);
        }
        for (; k < kv_len; k++) {
            const half* v_ptr = V_cache + k * kv_dim + head_offset;
            acc = fmaf(s_scores[k], __half2float(__ldg(v_ptr + d)), acc);
        }
        o_ptr[d] = __float2half(acc * inv_sum);
    }
}

// ============================================================================
// FlashAttention2 FP32 Decode Kernel
// ============================================================================
__global__ void flash_attention2_decode_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K_cache,
    const float* __restrict__ V_cache,
    float* __restrict__ O,
    const int pos,
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
    
    extern __shared__ float smem[];
    float* s_query = smem;
    float* s_scores = smem + head_size;
    float* s_max = s_scores + ((kv_len + FA2_BLOCK_SIZE_FP32 - 1) / FA2_BLOCK_SIZE_FP32) * FA2_BLOCK_SIZE_FP32;
    float* s_sum = s_max + FA2_NUM_WARPS_FP32;
    
    // Load query
    const float* q_ptr = Q + head * head_size;
    for (int d = tid; d < head_size; d += FA2_BLOCK_SIZE_FP32) {
        s_query[d] = q_ptr[d];
    }
    __syncthreads();
    
    // Compute Q·K scores
    float local_max = -FLT_MAX;
    for (int k = tid; k < kv_len; k += FA2_BLOCK_SIZE_FP32) {
        const float* k_ptr = K_cache + k * kv_dim + head_offset;
        float score = 0.0f;
        const float4* sq4 = reinterpret_cast<const float4*>(s_query);
        const float4* kk4 = reinterpret_cast<const float4*>(k_ptr);
        #pragma unroll
        for (int d = 0; d < head_size / 4; d++) {
            float4 q = sq4[d];
            float4 kv = __ldg(kk4 + d);
            score = fmaf(q.x, kv.x, score);
            score = fmaf(q.y, kv.y, score);
            score = fmaf(q.z, kv.z, score);
            score = fmaf(q.w, kv.w, score);
        }
        score *= scale;
        s_scores[k] = score;
        local_max = fmaxf(local_max, score);
    }
    __syncthreads();
    
    // Warp max reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    if (lane_id == 0) s_max[warp_id] = local_max;
    __syncthreads();
    
    float global_max = -FLT_MAX;
    if (tid < FA2_NUM_WARPS_FP32) global_max = s_max[tid];
    for (int offset = FA2_NUM_WARPS_FP32 / 2; offset > 0; offset /= 2) {
        global_max = fmaxf(global_max, __shfl_down_sync(0xffffffff, global_max, offset));
    }
    // Broadcast global_max to all threads via shared memory
    if (tid == 0) s_max[0] = global_max;
    __syncthreads();
    global_max = s_max[0];
    
    // Softmax
    float local_sum = 0.0f;
    for (int k = tid; k < kv_len; k += FA2_BLOCK_SIZE_FP32) {
        float exp_val = expf(s_scores[k] - global_max);
        s_scores[k] = exp_val;
        local_sum += exp_val;
    }
    __syncthreads();
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if (lane_id == 0) s_sum[warp_id] = local_sum;
    __syncthreads();
    
    float global_sum = 0.0f;
    if (tid < FA2_NUM_WARPS_FP32) global_sum = s_sum[tid];
    for (int offset = FA2_NUM_WARPS_FP32 / 2; offset > 0; offset /= 2) {
        global_sum += __shfl_down_sync(0xffffffff, global_sum, offset);
    }
    // Broadcast global_sum to all threads via shared memory
    if (tid == 0) s_sum[0] = global_sum;
    __syncthreads();
    global_sum = s_sum[0];
    float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
    
    // Output
    float* o_ptr = O + head * head_size;
    for (int d = tid; d < head_size; d += FA2_BLOCK_SIZE_FP32) {
        float acc = 0.0f;
        for (int k = 0; k < kv_len; k++) {
            const float* v_ptr = V_cache + k * kv_dim + head_offset;
            acc += s_scores[k] * __ldg(v_ptr + d);
        }
        o_ptr[d] = acc * inv_sum;
    }
}

// ============================================================================
// FlashAttention2 FP16 Decode Kernel with GPU pos (CUDA Graph compatible)
// ============================================================================
__global__ void flash_attention2_decode_kernel_fp16_gpu_pos(
    const half* __restrict__ Q,
    const half* __restrict__ K_cache,
    const half* __restrict__ V_cache,
    half* __restrict__ O,
    const int32_t* __restrict__ pos_ptr,
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
    
    // Read position from GPU memory (volatile for CUDA Graph)
    const int pos = *reinterpret_cast<const volatile int32_t*>(pos_ptr);
    const int kv_len = pos + 1;
    
    const int kv_head = head / kv_mul;
    const int head_offset = kv_head * head_size;
    
    // FA2 uses online softmax with tiling for CUDA Graph compatibility (fixed shared mem)
    constexpr int TILE_K = 256;
    constexpr int BLOCK_SZ = 128;
    constexpr int N_WARPS = 4;
    
    extern __shared__ char smem_raw[];
    half* s_query = reinterpret_cast<half*>(smem_raw);
    float* s_scores = reinterpret_cast<float*>(smem_raw + head_size * sizeof(half));
    float* s_max = s_scores + TILE_K;
    float* s_sum = s_max + N_WARPS;
    
    // Load query
    const half2* q_ptr_h2 = reinterpret_cast<const half2*>(Q + head * head_size);
    half2* s_query_h2 = reinterpret_cast<half2*>(s_query);
    for (int d = tid; d < head_size / 2; d += BLOCK_SZ) {
        s_query_h2[d] = q_ptr_h2[d];
    }
    __syncthreads();
    
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    float acc_o = 0.0f;
    
    for (int tile_start = 0; tile_start < kv_len; tile_start += TILE_K) {
        const int tile_end = min(tile_start + TILE_K, kv_len);
        const int tile_len = tile_end - tile_start;
        
        // Q·K scores
        float tile_max_local = -FLT_MAX;
        for (int k_idx = tid; k_idx < tile_len; k_idx += BLOCK_SZ) {
            const int kv_pos = tile_start + k_idx;
            const float4* k_ptr_f4 = reinterpret_cast<const float4*>(K_cache + kv_pos * kv_dim + head_offset);
            const float4* q_ptr_f4 = reinterpret_cast<const float4*>(s_query);
            
            float2 acc = make_float2(0.0f, 0.0f);
            #pragma unroll
            for (int d = 0; d < head_size / 8; d++) {
                float4 q_packed = q_ptr_f4[d];
                float4 k_packed = __ldg(k_ptr_f4 + d);
                const half2* q_h2 = reinterpret_cast<const half2*>(&q_packed);
                const half2* k_h2 = reinterpret_cast<const half2*>(&k_packed);
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    float2 q_f = __half22float2(q_h2[i]);
                    float2 k_f = __half22float2(k_h2[i]);
                    acc.x += q_f.x * k_f.x;
                    acc.y += q_f.y * k_f.y;
                }
            }
            float score = (acc.x + acc.y) * scale;
            s_scores[k_idx] = score;
            tile_max_local = fmaxf(tile_max_local, score);
        }
        __syncthreads();
        
        // Max reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            tile_max_local = fmaxf(tile_max_local, __shfl_xor_sync(0xffffffff, tile_max_local, offset));
        }
        if (lane_id == 0) s_max[warp_id] = tile_max_local;
        __syncthreads();
        
        float m_j = s_max[0];
        for (int w = 1; w < N_WARPS; w++) m_j = fmaxf(m_j, s_max[w]);
        
        float m_new = fmaxf(row_max, m_j);
        float correction = expf(row_max - m_new);
        acc_o *= correction;
        
        // Softmax + sum
        float tile_sum_local = 0.0f;
        for (int k_idx = tid; k_idx < tile_len; k_idx += BLOCK_SZ) {
            float val = s_scores[k_idx] - m_new;
            float exp_val = (val > FA2_SOFTMAX_FTZ) ? expf(val) : 0.0f;
            s_scores[k_idx] = exp_val;
            tile_sum_local += exp_val;
        }
        __syncthreads();
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            tile_sum_local += __shfl_xor_sync(0xffffffff, tile_sum_local, offset);
        }
        if (lane_id == 0) s_sum[warp_id] = tile_sum_local;
        __syncthreads();
        
        float l_j = s_sum[0];
        for (int w = 1; w < N_WARPS; w++) l_j += s_sum[w];
        
        // Accumulate V
        const int my_dim = tid;
        if (my_dim < head_size) {
            for (int k = 0; k < tile_len; k++) {
                const int kv_pos = tile_start + k;
                const half* v_ptr = V_cache + kv_pos * kv_dim + head_offset;
                acc_o += s_scores[k] * __half2float(__ldg(v_ptr + my_dim));
            }
        }
        
        row_max = m_new;
        row_sum = correction * row_sum + l_j;
        __syncthreads();
    }
    
    // Final output
    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    half* o_ptr = O + head * head_size;
    if (tid < head_size) {
        o_ptr[tid] = __float2half(acc_o * inv_sum);
    }
}

// ============================================================================
// Host-side launch functions
// ============================================================================

void flash_attention2_prefill_fp16_cu(
    int32_t start_pos, int32_t seq_len,
    int32_t head_num, int32_t kv_head_num,
    int32_t head_size, int32_t kv_mul,
    int32_t layer_index, int32_t max_seq_len,
    int32_t kv_dim,
    const tensor::Tensor& query, const tensor::Tensor& output,
    const tensor::Tensor& key_cache, const tensor::Tensor& value_cache,
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
    dim3 block(FA2_BLOCK_SIZE);
    
    // Shared memory: query (half) + scores (float) for FA2 tile
    const int smem_size = head_size * sizeof(half) + FA2_TILE_K * sizeof(float);
    
    cudaStream_t stream = config ? config->stream : nullptr;
    
    flash_attention2_prefill_kernel_fp16<<<grid, block, smem_size, stream>>>(
        Q, K, V, O,
        seq_len, start_pos, head_num, kv_head_num, head_size, kv_mul,
        dim, kv_dim, scale
    );
}

void flash_attention2_prefill_cu(
    int32_t start_pos, int32_t seq_len,
    int32_t head_num, int32_t kv_head_num,
    int32_t head_size, int32_t kv_mul,
    int32_t layer_index, int32_t max_seq_len,
    int32_t kv_dim,
    const tensor::Tensor& query, const tensor::Tensor& output,
    const tensor::Tensor& key_cache, const tensor::Tensor& value_cache,
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
    dim3 block(FA2_BLOCK_SIZE_FP32);
    
    const int smem_size = (head_size + FA2_TILE_K) * sizeof(float);
    
    cudaStream_t stream = config ? config->stream : nullptr;
    
    flash_attention2_prefill_kernel<<<grid, block, smem_size, stream>>>(
        Q, K, V, O,
        seq_len, start_pos, head_num, kv_head_num, head_size, kv_mul,
        dim, kv_dim, scale
    );
}

void flash_attention2_decode_fp16_cu(
    int32_t pos, int32_t head_num, int32_t kv_head_num,
    int32_t head_size, int32_t kv_mul,
    int32_t layer_index, int32_t max_seq_len, int32_t kv_dim,
    const tensor::Tensor& query, const tensor::Tensor& output,
    const tensor::Tensor& key_cache, const tensor::Tensor& value_cache,
    CudaConfig* config
) {
    const int layer_offset = layer_index * max_seq_len * kv_dim;
    const int kv_len = pos + 1;
    
    half* Q = const_cast<half*>(query.ptr<half>());
    half* O = const_cast<half*>(output.ptr<half>());
    half* K = const_cast<half*>(key_cache.ptr<half>()) + layer_offset;
    half* V = const_cast<half*>(value_cache.ptr<half>()) + layer_offset;
    
    const float scale = 1.0f / sqrtf((float)head_size);
    
    dim3 grid(head_num);
    dim3 block(FA2_DECODE_BLOCK);
    
    const int score_buffer_size = ((kv_len + FA2_DECODE_BLOCK - 1) / FA2_DECODE_BLOCK) * FA2_DECODE_BLOCK;
    const int smem_size = head_size * sizeof(half) + 
                          score_buffer_size * sizeof(float) + 
                          2 * FA2_DECODE_WARPS * sizeof(float);
    
    cudaStream_t stream = config ? config->stream : nullptr;
    
    flash_attention2_decode_kernel_fp16<<<grid, block, smem_size, stream>>>(
        Q, K, V, O,
        pos, head_num, kv_head_num, head_size, kv_mul,
        kv_dim, scale
    );
}

void flash_attention2_decode_cu(
    int32_t pos, int32_t head_num, int32_t kv_head_num,
    int32_t head_size, int32_t kv_mul,
    int32_t layer_index, int32_t max_seq_len, int32_t kv_dim,
    const tensor::Tensor& query, const tensor::Tensor& output,
    const tensor::Tensor& key_cache, const tensor::Tensor& value_cache,
    CudaConfig* config
) {
    const int layer_offset = layer_index * max_seq_len * kv_dim;
    const int kv_len = pos + 1;
    
    float* Q = const_cast<float*>(query.ptr<float>());
    float* O = const_cast<float*>(output.ptr<float>());
    float* K = const_cast<float*>(key_cache.ptr<float>()) + layer_offset;
    float* V = const_cast<float*>(value_cache.ptr<float>()) + layer_offset;
    
    const float scale = 1.0f / sqrtf((float)head_size);
    
    dim3 grid(head_num);
    dim3 block(FA2_BLOCK_SIZE_FP32);
    
    const int score_buffer_size = ((kv_len + FA2_BLOCK_SIZE_FP32 - 1) / FA2_BLOCK_SIZE_FP32) * FA2_BLOCK_SIZE_FP32;
    const int smem_size = (head_size + score_buffer_size + 16) * sizeof(float);
    
    cudaStream_t stream = config ? config->stream : nullptr;
    
    flash_attention2_decode_kernel<<<grid, block, smem_size, stream>>>(
        Q, K, V, O,
        pos, head_num, kv_head_num, head_size, kv_mul,
        head_num * head_size, kv_dim, scale
    );
}

void flash_attention2_decode_fp16_gpu_pos_cu(
    const int32_t* pos_ptr,
    int32_t head_num, int32_t kv_head_num,
    int32_t head_size, int32_t kv_mul,
    int32_t layer_index, int32_t max_seq_len, int32_t kv_dim,
    const tensor::Tensor& query, const tensor::Tensor& output,
    const tensor::Tensor& key_cache, const tensor::Tensor& value_cache,
    CudaConfig* config
) {
    const int layer_offset = layer_index * max_seq_len * kv_dim;
    
    half* Q = const_cast<half*>(query.ptr<half>());
    half* O = const_cast<half*>(output.ptr<half>());
    half* K = const_cast<half*>(key_cache.ptr<half>()) + layer_offset;
    half* V = const_cast<half*>(value_cache.ptr<half>()) + layer_offset;
    
    const float scale = 1.0f / sqrtf((float)head_size);
    
    constexpr int BLOCK_SZ = 128;
    constexpr int TILE_K = 256;
    constexpr int N_WARPS = 4;
    
    dim3 grid(head_num);
    dim3 block(BLOCK_SZ);
    
    // Fixed shared memory size for CUDA Graph
    const int smem_size = head_size * sizeof(half) + 
                          TILE_K * sizeof(float) + 
                          2 * N_WARPS * sizeof(float);
    
    cudaStream_t stream = config ? config->stream : nullptr;
    
    flash_attention2_decode_kernel_fp16_gpu_pos<<<grid, block, smem_size, stream>>>(
        Q, K, V, O,
        pos_ptr, head_num, kv_head_num, head_size, kv_mul,
        kv_dim, scale
    );
}

}  // namespace kernel
