#include "fused_rope_kv_kernel.cuh"
#include <cuda_fp16.h>
#include <cstdint>
#include <cfloat>

namespace kernel {

/**
 * Fused M-RoPE + KV Cache Write CUDA Kernel (FP16, Decode Phase)
 *
 * This kernel fuses three separate operations:
 *   1. Apply M-RoPE to Q (in-place)
 *   2. Apply M-RoPE to K → write directly to key_cache
 *   3. Copy V → write directly to val_cache
 *
 * Thread organization (inspired by RMinte's applyRopeWriteKV):
 *   - blockIdx.y selects head: [0, num_q_heads) for Q heads, [num_q_heads, num_q_heads + num_kv_heads) for KV heads
 *   - blockIdx.x * blockDim.y + threadIdx.y selects token (always 0 for single-token decode)
 *   - threadIdx.x processes head_dim / vec_size elements per thread
 *
 * For decode phase (single token), we simplify:
 *   - gridDim.x = 1 (single token)
 *   - gridDim.y = num_q_heads + num_kv_heads
 *   - blockDim.x = head_size / 2 (each thread handles one pair of elements for RoPE)
 *   - blockDim.y = 1 (single token)
 *
 * M-RoPE position mapping (Qwen3-VL):
 *   Decode phase uses same text_pos for all 3 dimensions (t=h=w=text_pos)
 *   - Dimensions [0, section0*2): temporal position
 *   - Dimensions [section0*2, (section0+section1)*2): height position
 *   - Dimensions [(section0+section1)*2, head_size): width position
 *
 * RoPE formula (non-interleaved, half-split):
 *   For pair (d, d + half_head_size):
 *     q[d]                = q[d] * cos - q[d+half_head_size] * sin
 *     q[d+half_head_size] = q[d+half_head_size] * cos + q[d] * sin
 */
__global__ void fused_mrope_kv_write_fp16_kernel(
    const int32_t* __restrict__ pos_gpu,
    const int32_t* __restrict__ kv_cache_pos_gpu,
    half* __restrict__ query,
    const half* __restrict__ key,
    const half* __restrict__ value,
    half* __restrict__ key_cache,
    half* __restrict__ val_cache,
    const float* __restrict__ sin_cache,
    const float* __restrict__ cos_cache,
    int32_t dim,
    int32_t kv_dim,
    int32_t head_size,
    int32_t section0,
    int32_t section1,
    int32_t section2,
    int32_t layer_idx,
    int32_t seq_len)
{
  // Read positions from GPU memory (CUDA Graph compatible)
  int32_t rope_pos = *reinterpret_cast<const volatile int32_t*>(pos_gpu);
  int32_t kv_pos = *reinterpret_cast<const volatile int32_t*>(kv_cache_pos_gpu);

  int32_t num_q_heads = dim / head_size;
  int32_t half_head_size = head_size / 2;

  int32_t head_idx_global = blockIdx.y;  // Which head this CTA processes
  int32_t pair_idx = threadIdx.x;        // Which RoPE pair within the head

  if (pair_idx >= half_head_size) return;

  // Compute dimension indices for this pair
  int32_t d0 = pair_idx;                  // First dimension of pair (0..63)
  int32_t d1 = pair_idx + half_head_size; // Second dimension of pair (64..127)

  // M-RoPE: determine position based on which section d0/d1 falls in
  // For decode phase, all positions equal rope_pos, but we keep the section logic
  // for correctness (and it costs essentially nothing since it's all the same value)
  int32_t dim_threshold0 = section0 * 2;  // 48 for Qwen3-VL
  int32_t dim_threshold1 = dim_threshold0 + section1 * 2;  // 88 for Qwen3-VL

  int32_t pos0, pos1;
  // Position for d0
  if (d0 < dim_threshold0) {
    pos0 = rope_pos;  // temporal
  } else if (d0 < dim_threshold1) {
    pos0 = rope_pos;  // height (same as temporal in decode)
  } else {
    pos0 = rope_pos;  // width (same as temporal in decode)
  }
  // Position for d1
  if (d1 < dim_threshold0) {
    pos1 = rope_pos;
  } else if (d1 < dim_threshold1) {
    pos1 = rope_pos;
  } else {
    pos1 = rope_pos;
  }

  // Look up sin/cos values
  int32_t freq_idx = pair_idx * 2;
  float sin0 = sin_cache[pos0 * head_size + freq_idx];
  float cos0 = cos_cache[pos0 * head_size + freq_idx];
  float sin1 = sin_cache[pos1 * head_size + freq_idx];
  float cos1 = cos_cache[pos1 * head_size + freq_idx];

  if (head_idx_global < num_q_heads) {
    // ============ Q head: Apply M-RoPE in-place ============
    int32_t q_head_idx = head_idx_global;
    int32_t q_offset = q_head_idx * head_size;

    float v0 = __half2float(query[q_offset + d0]);
    float v1 = __half2float(query[q_offset + d1]);

    query[q_offset + d0] = __float2half(v0 * cos0 - v1 * sin0);
    query[q_offset + d1] = __float2half(v1 * cos1 + v0 * sin1);
  } else {
    // ============ KV head: Apply M-RoPE to K + write K,V to cache ============
    int32_t kv_head_idx = head_idx_global - num_q_heads;

    // Apply M-RoPE to K
    int32_t k_offset = kv_head_idx * head_size;
    float k0 = __half2float(key[k_offset + d0]);
    float k1 = __half2float(key[k_offset + d1]);
    half k_rope_d0 = __float2half(k0 * cos0 - k1 * sin0);
    half k_rope_d1 = __float2half(k1 * cos1 + k0 * sin1);

    // Write RoPE'd K directly to key_cache at correct position
    // Layout: [layer_num, seq_len, kv_dim]
    int64_t cache_base = static_cast<int64_t>(layer_idx) * seq_len * kv_dim +
                         static_cast<int64_t>(kv_pos) * kv_dim +
                         kv_head_idx * head_size;
    key_cache[cache_base + d0] = k_rope_d0;
    key_cache[cache_base + d1] = k_rope_d1;

    // Write V directly to val_cache (no RoPE needed for V)
    half v_d0 = value[k_offset + d0];
    half v_d1 = value[k_offset + d1];
    val_cache[cache_base + d0] = v_d0;
    val_cache[cache_base + d1] = v_d1;
  }
}

void fused_mrope_kv_write_fp16(
    const int32_t* pos_gpu,
    const int32_t* kv_cache_pos_gpu,
    half* query,
    const half* key,
    const half* value,
    half* key_cache,
    half* val_cache,
    const float* sin_cache,
    const float* cos_cache,
    int32_t dim,
    int32_t kv_dim,
    int32_t head_size,
    int32_t section0,
    int32_t section1,
    int32_t section2,
    int32_t layer_idx,
    int32_t seq_len,
    cudaStream_t stream)
{
  int32_t num_q_heads = dim / head_size;
  int32_t num_kv_heads = kv_dim / head_size;
  int32_t half_head_size = head_size / 2;

  // Grid: each block handles one head (Q or KV)
  // Block: each thread handles one RoPE pair
  dim3 grid(1, num_q_heads + num_kv_heads);
  dim3 block(half_head_size, 1);

  fused_mrope_kv_write_fp16_kernel<<<grid, block, 0, stream>>>(
      pos_gpu, kv_cache_pos_gpu,
      query, key, value,
      key_cache, val_cache,
      sin_cache, cos_cache,
      dim, kv_dim, head_size,
      section0, section1, section2,
      layer_idx, seq_len);
}

// ============================================================================
// Fused GQA + MRoPE + KV Cache Read/Write Decode Kernel (FP16)
// ============================================================================

// Constants for the fused GQA kernel
constexpr int FUSED_GQA_TILE_K = 512;
constexpr int FUSED_GQA_BLOCK_SIZE = 256;
constexpr int FUSED_GQA_NUM_WARPS = 8;
constexpr float FUSED_GQA_SOFTMAX_FTZ = -20.0f;

/**
 * Fused GQA + MRoPE + KV Cache Read/Write CUDA Kernel (FP16, Decode Phase)
 * =========================================================================
 * Multi-Q-per-block GQA optimization:
 *
 *   Grid:  (num_q_heads / q_per_block,)
 *   Block: 256 threads (8 warps)
 *
 * Each block processes q_per_block Q heads that share the same KV head.
 * Core optimization: K/V cache data is loaded ONCE per block and reused
 * for q_per_block Q·K dot products and V accumulations, reducing
 * L2 cache traffic and increasing arithmetic intensity.
 *
 * q_per_block configurations:
 *   q_per_block=4 (kv_mul): 8 blocks, 4x K/V reuse, 50% SM utilization
 *   q_per_block=2:         16 blocks, 2x K/V reuse, 100% SM utilization
 *   q_per_block=1:         32 blocks, 1x K/V reuse, 100% (like original FA)
 *
 * Shared memory layout:
 *   s_query[q_per_block * head_size] half    — MRoPE'd Q for q_per_block heads
 *   s_k_current[head_size] half              — MRoPE'd K for current token
 *   s_scores[q_per_block * TILE_K] float     — attention scores for q_per_block Q heads
 *   s_reduce[NUM_WARPS * q_per_block] float  — warp reduction workspace
 */
template<int Q_PER_BLOCK>
__global__ void fused_gqa_mrope_kv_decode_fp16_kernel(
    const int32_t* __restrict__ pos_gpu,
    const int32_t* __restrict__ kv_pos_gpu,
    const half* __restrict__ Q_in,
    const half* __restrict__ K_in,
    const half* __restrict__ V_in,
    half* __restrict__ K_cache,
    half* __restrict__ V_cache,
    half* __restrict__ O,
    const float* __restrict__ sin_cache,
    const float* __restrict__ cos_cache,
    const int head_num,
    const int kv_head_num,
    const int head_size,
    const int kv_mul,
    const int kv_dim,
    const float scale,
    const int section0,
    const int section1,
    const int section2)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;

    // Compute which KV head and which Q-heads this block handles
    const int groups_per_kv = kv_mul / Q_PER_BLOCK;  // e.g., 4/2=2 groups per KV head
    const int kv_head = bid / groups_per_kv;
    const int group_idx = bid % groups_per_kv;
    const int first_q_head = kv_head * kv_mul + group_idx * Q_PER_BLOCK;
    // Whether this is the first block for this KV head (responsible for cache write)
    const int is_first_group = (group_idx == 0);

    if (kv_head >= kv_head_num) return;

    const int rope_pos = *reinterpret_cast<const volatile int32_t*>(pos_gpu);
    const int kv_pos = *reinterpret_cast<const volatile int32_t*>(kv_pos_gpu);
    const int kv_past_len = kv_pos;

    const int head_offset = kv_head * head_size;
    const int half_head_size = head_size / 2;

    // ========== Shared Memory Layout ==========
    extern __shared__ char smem_raw[];
    half* s_query = reinterpret_cast<half*>(smem_raw);  // [Q_PER_BLOCK * head_size]
    half* s_k_current = s_query + Q_PER_BLOCK * head_size;   // [head_size]
    float* s_scores = reinterpret_cast<float*>(s_k_current + head_size);  // [Q_PER_BLOCK * TILE_K]
    float* s_reduce = s_scores + Q_PER_BLOCK * FUSED_GQA_TILE_K;  // [NUM_WARPS * Q_PER_BLOCK]

    // ========== Phase 0: MRoPE for q_per_block Q heads + K ==========
    // q_per_block * half_head_size pairs mapped to threads
    {
        const int qi = tid / half_head_size;   // which Q head within group
        const int pair = tid % half_head_size; // which RoPE pair

        if (qi < Q_PER_BLOCK) {
            const int d0 = pair;
            const int d1 = pair + half_head_size;
            const int freq_idx = pair * 2;
            const int q_head = first_q_head + qi;

            float sin_val = sin_cache[rope_pos * head_size + freq_idx];
            float cos_val = cos_cache[rope_pos * head_size + freq_idx];

            // MRoPE Q → shared memory
            float q0 = __half2float(Q_in[q_head * head_size + d0]);
            float q1 = __half2float(Q_in[q_head * head_size + d1]);
            s_query[qi * head_size + d0] = __float2half(q0 * cos_val - q1 * sin_val);
            s_query[qi * head_size + d1] = __float2half(q1 * cos_val + q0 * sin_val);

            // First group of threads also computes K MRoPE
            if (qi == 0) {
                float k0 = __half2float(K_in[head_offset + d0]);
                float k1 = __half2float(K_in[head_offset + d1]);
                s_k_current[d0] = __float2half(k0 * cos_val - k1 * sin_val);
                s_k_current[d1] = __float2half(k1 * cos_val + k0 * sin_val);
            }
        }
    }
    __syncthreads();

    // ========== Phase 0b: Write K/V to KV cache (first group only) ==========
    if (is_first_group && tid < head_size) {
        K_cache[(int64_t)kv_pos * kv_dim + head_offset + tid] = s_k_current[tid];
        V_cache[(int64_t)kv_pos * kv_dim + head_offset + tid] = V_in[head_offset + tid];
    }

    // ========== Phase 0c: Q·K_current for q_per_block Q heads ==========
    float score_current[4];  // max kv_mul=4
    if (tid < Q_PER_BLOCK) {
        const float4* q_f4 = reinterpret_cast<const float4*>(s_query + tid * head_size);
        const float4* k_f4 = reinterpret_cast<const float4*>(s_k_current);
        float2 dot = make_float2(0.0f, 0.0f);
        #pragma unroll
        for (int d = 0; d < head_size / 8; d++) {
            float4 q_packed = q_f4[d];
            float4 k_packed = k_f4[d];
            const half2* qh = reinterpret_cast<const half2*>(&q_packed);
            const half2* kh = reinterpret_cast<const half2*>(&k_packed);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                float2 qf = __half22float2(qh[i]);
                float2 kf = __half22float2(kh[i]);
                dot.x = fmaf(qf.x, kf.x, dot.x);
                dot.y = fmaf(qf.y, kf.y, dot.y);
            }
        }
        s_reduce[tid] = (dot.x + dot.y) * scale;
    }
    __syncthreads();
    // All threads read the Q_PER_BLOCK scores
    for (int qi = 0; qi < Q_PER_BLOCK; qi++)
        score_current[qi] = s_reduce[qi];

    // ========== Phases 1-3: Tiled online softmax with multi-Q batching ==========
    const int my_dim = tid % head_size;
    float acc_o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float row_max[4], row_sum[4];
    for (int qi = 0; qi < Q_PER_BLOCK; qi++) {
        row_max[qi] = -FLT_MAX;
        row_sum[qi] = 0.0f;
    }

    for (int tile_start = 0; tile_start < kv_past_len; tile_start += FUSED_GQA_TILE_K) {
        const int tile_end = min(tile_start + FUSED_GQA_TILE_K, kv_past_len);
        const int tile_len = tile_end - tile_start;

        // === Scoring: load K once, compute q_per_block dot products ===
        float tile_max_local[4];
        for (int qi = 0; qi < Q_PER_BLOCK; qi++)
            tile_max_local[qi] = -FLT_MAX;

        for (int k_idx = tid; k_idx < tile_len; k_idx += FUSED_GQA_BLOCK_SIZE) {
            const float4* k_ptr_f4 = reinterpret_cast<const float4*>(
                K_cache + (int64_t)(tile_start + k_idx) * kv_dim + head_offset);

            // Compute Q_PER_BLOCK dot products with Q_PER_BLOCK Q vectors
            #pragma unroll
            for (int qi = 0; qi < Q_PER_BLOCK; qi++) {
                const float4* q_f4 = reinterpret_cast<const float4*>(s_query + qi * head_size);
                float2 dot = make_float2(0.0f, 0.0f);
                #pragma unroll
                for (int d = 0; d < head_size / 8; d++) {
                    float4 q_packed = q_f4[d];
                    float4 k_packed = __ldg(k_ptr_f4 + d);
                    const half2* qh = reinterpret_cast<const half2*>(&q_packed);
                    const half2* kh = reinterpret_cast<const half2*>(&k_packed);
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        float2 qf = __half22float2(qh[i]);
                        float2 kf = __half22float2(kh[i]);
                        dot.x = fmaf(qf.x, kf.x, dot.x);
                        dot.y = fmaf(qf.y, kf.y, dot.y);
                    }
                }
                float score = (dot.x + dot.y) * scale;
                s_scores[qi * FUSED_GQA_TILE_K + k_idx] = score;
                tile_max_local[qi] = fmaxf(tile_max_local[qi], score);
            }
        }

        // Max reduction for each Q head
        for (int qi = 0; qi < Q_PER_BLOCK; qi++) {
            float m = tile_max_local[qi];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, offset));
            if (lane_id == 0) s_reduce[qi * FUSED_GQA_NUM_WARPS + warp_id] = m;
        }
        __syncthreads();

        float m_j[4], m_new[4];
        if (tid == 0) {
            for (int qi = 0; qi < Q_PER_BLOCK; qi++) {
                m_j[qi] = s_reduce[qi * FUSED_GQA_NUM_WARPS];
                #pragma unroll
                for (int w = 1; w < FUSED_GQA_NUM_WARPS; w++)
                    m_j[qi] = fmaxf(m_j[qi], s_reduce[qi * FUSED_GQA_NUM_WARPS + w]);
                s_reduce[qi] = m_j[qi];
            }
        }
        __syncthreads();
        for (int qi = 0; qi < Q_PER_BLOCK; qi++) {
            m_j[qi] = s_reduce[qi];
            m_new[qi] = fmaxf(row_max[qi], m_j[qi]);
        }

        // === Exp + Sum for each Q head ===
        float tile_sum_local[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int k_idx = tid; k_idx < tile_len; k_idx += FUSED_GQA_BLOCK_SIZE) {
            for (int qi = 0; qi < Q_PER_BLOCK; qi++) {
                float val = s_scores[qi * FUSED_GQA_TILE_K + k_idx] - m_new[qi];
                float exp_val = (val > FUSED_GQA_SOFTMAX_FTZ) ? expf(val) : 0.0f;
                s_scores[qi * FUSED_GQA_TILE_K + k_idx] = exp_val;
                tile_sum_local[qi] += exp_val;
            }
        }

        for (int qi = 0; qi < Q_PER_BLOCK; qi++) {
            float s = tile_sum_local[qi];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                s += __shfl_xor_sync(0xffffffff, s, offset);
            if (lane_id == 0) s_reduce[qi * FUSED_GQA_NUM_WARPS + warp_id] = s;
        }
        __syncthreads();

        float l_j[4];
        if (tid == 0) {
            for (int qi = 0; qi < Q_PER_BLOCK; qi++) {
                l_j[qi] = s_reduce[qi * FUSED_GQA_NUM_WARPS];
                #pragma unroll
                for (int w = 1; w < FUSED_GQA_NUM_WARPS; w++)
                    l_j[qi] += s_reduce[qi * FUSED_GQA_NUM_WARPS + w];
                s_reduce[qi] = l_j[qi];
            }
        }
        __syncthreads();
        for (int qi = 0; qi < Q_PER_BLOCK; qi++)
            l_j[qi] = s_reduce[qi];

        // === Rescale + V accumulation: load V once, use for q_per_block heads ===
        float correction[4];
        for (int qi = 0; qi < Q_PER_BLOCK; qi++) {
            correction[qi] = expf(row_max[qi] - m_new[qi]);
            acc_o[qi] *= correction[qi];
        }

        if (my_dim < head_size) {
            const half* v_base = V_cache + head_offset + my_dim;
            int k = 0;
            for (; k + 3 < tile_len; k += 4) {
                const int64_t bp = (int64_t)(tile_start + k) * kv_dim;
                // Load V once
                float v0 = __half2float(__ldg(v_base + bp));
                float v1 = __half2float(__ldg(v_base + bp + kv_dim));
                float v2 = __half2float(__ldg(v_base + bp + 2 * kv_dim));
                float v3 = __half2float(__ldg(v_base + bp + 3 * kv_dim));
                // Accumulate for Q_PER_BLOCK Q heads
                for (int qi = 0; qi < Q_PER_BLOCK; qi++) {
                    float s0 = s_scores[qi * FUSED_GQA_TILE_K + k];
                    float s1 = s_scores[qi * FUSED_GQA_TILE_K + k + 1];
                    float s2 = s_scores[qi * FUSED_GQA_TILE_K + k + 2];
                    float s3 = s_scores[qi * FUSED_GQA_TILE_K + k + 3];
                    acc_o[qi] = fmaf(s0, v0, acc_o[qi]);
                    acc_o[qi] = fmaf(s1, v1, acc_o[qi]);
                    acc_o[qi] = fmaf(s2, v2, acc_o[qi]);
                    acc_o[qi] = fmaf(s3, v3, acc_o[qi]);
                }
            }
            for (; k < tile_len; k++) {
                float v_val = __half2float(__ldg(v_base + (int64_t)(tile_start + k) * kv_dim));
                for (int qi = 0; qi < Q_PER_BLOCK; qi++)
                    acc_o[qi] = fmaf(s_scores[qi * FUSED_GQA_TILE_K + k], v_val, acc_o[qi]);
            }
        }

        for (int qi = 0; qi < Q_PER_BLOCK; qi++) {
            row_max[qi] = m_new[qi];
            row_sum[qi] = fmaf(correction[qi], row_sum[qi], l_j[qi]);
        }
        __syncthreads();
    }

    for (int qi = 0; qi < Q_PER_BLOCK; qi++) {
        float m_new_qi = fmaxf(row_max[qi], score_current[qi]);
        float corr = expf(row_max[qi] - m_new_qi);
        float exp_cur = expf(score_current[qi] - m_new_qi);
        acc_o[qi] *= corr;
        if (my_dim < head_size) {
            float v_cur = __half2float(V_in[head_offset + my_dim]);
            acc_o[qi] = fmaf(exp_cur, v_cur, acc_o[qi]);
        }
        row_max[qi] = m_new_qi;
        row_sum[qi] = fmaf(corr, row_sum[qi], exp_cur);
    }

    // ========== Phase 5: Write q_per_block output vectors ==========
    if (my_dim < head_size && tid < head_size) {
        for (int qi = 0; qi < Q_PER_BLOCK; qi++) {
            float inv_sum = (row_sum[qi] > 0.0f) ? (1.0f / row_sum[qi]) : 0.0f;
            O[(first_q_head + qi) * head_size + my_dim] = __float2half(acc_o[qi] * inv_sum);
        }
    }
}

void fused_gqa_mrope_kv_decode_fp16(
    const int32_t* pos_gpu,
    const int32_t* kv_pos_gpu,
    const half* query,
    const half* key,
    const half* value,
    half* key_cache,
    half* val_cache,
    half* output,
    const float* sin_cache,
    const float* cos_cache,
    int32_t dim,
    int32_t kv_dim,
    int32_t head_size,
    int32_t section0,
    int32_t section1,
    int32_t section2,
    int32_t layer_idx,
    int32_t max_seq_len,
    cudaStream_t stream)
{
  int32_t num_q_heads = dim / head_size;
  int32_t num_kv_heads = kv_dim / head_size;
  int32_t kv_mul = num_q_heads / num_kv_heads;
  float scale = 1.0f / sqrtf((float)head_size);

  int64_t layer_offset = (int64_t)layer_idx * max_seq_len * kv_dim;

  // Choose q_per_block to maximize SM utilization on Orin (16 SMs)
  // q_per_block=1: 32 blocks (best occupancy, L2 provides K/V reuse for short-medium seqs)
  // q_per_block=2: 16 blocks (2x K/V reuse, useful for long sequences exceeding L2)
  // q_per_block=4: 8 blocks (4x K/V reuse, but low SM occupancy)
  constexpr int q_per_block = 1;
  int num_blocks = num_q_heads / q_per_block;

  dim3 grid(num_blocks);
  dim3 block(FUSED_GQA_BLOCK_SIZE);

  // Shared memory: q_per_block Q vectors + K current + q_per_block score arrays + reduce workspace
  int smem_size = (q_per_block * head_size + head_size) * sizeof(half) +
                  q_per_block * FUSED_GQA_TILE_K * sizeof(float) +
                  FUSED_GQA_NUM_WARPS * q_per_block * sizeof(float);

  fused_gqa_mrope_kv_decode_fp16_kernel<q_per_block><<<grid, block, smem_size, stream>>>(
      pos_gpu, kv_pos_gpu,
      query, key, value,
      key_cache + layer_offset,
      val_cache + layer_offset,
      output,
      sin_cache, cos_cache,
      num_q_heads, num_kv_heads, head_size, kv_mul, kv_dim, scale,
      section0, section1, section2);
}

}  // namespace kernel
