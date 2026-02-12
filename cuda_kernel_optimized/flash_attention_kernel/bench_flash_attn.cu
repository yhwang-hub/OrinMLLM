// Standalone benchmark for flash attention kernels - original vs optimized
// Compile: nvcc -O3 -arch=sm_87 -o bench_flash_attn bench_flash_attn.cu
// Profile: sudo /usr/local/cuda-12.6/bin/ncu --set full \
//   --kernel-name "regex:original::.*|optimized::.*" \
//   --launch-skip 0 --launch-count 6 -o ncu_flash_attn_report ./bench_flash_attn
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// ============== Common constants ==============
constexpr int HEAD_SIZE = 128;
constexpr int HEAD_NUM = 32;
constexpr int KV_HEAD_NUM = 8;
constexpr int KV_MUL = 4;
constexpr int KV_DIM = 1024;
constexpr float SOFTMAX_FTZ = -20.0f;

// ====================== ORIGINAL (BEFORE) KERNELS ======================
namespace original {

// --- FP16 Decode (256 threads, non-graph) - ORIGINAL ---
constexpr int DECODE_BLOCK_SIZE = 256;
constexpr int DECODE_NUM_WARPS = 8;
constexpr int DECODE_WARP_SIZE = 32;

__global__ void flash_attention_decode_kernel_fp16_optimized(
    const half* __restrict__ Q,
    const half* __restrict__ K_cache,
    const half* __restrict__ V_cache,
    half* __restrict__ O,
    const int pos, const int head_num, const int kv_head_num,
    const int head_size, const int kv_mul, const int kv_dim, const float scale
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

    extern __shared__ char smem_raw[];
    half* s_query = reinterpret_cast<half*>(smem_raw);
    float* s_scores = reinterpret_cast<float*>(smem_raw + head_size * sizeof(half));
    float* s_max = s_scores + ((kv_len + DECODE_BLOCK_SIZE - 1) / DECODE_BLOCK_SIZE) * DECODE_BLOCK_SIZE;
    float* s_sum = s_max + DECODE_NUM_WARPS;

    const half* q_ptr = Q + head * head_size;
    const half2* q_ptr_h2 = reinterpret_cast<const half2*>(q_ptr);
    half2* s_query_h2 = reinterpret_cast<half2*>(s_query);
    for (int d = tid; d < head_size_h2; d += DECODE_BLOCK_SIZE) {
        s_query_h2[d] = q_ptr_h2[d];
    }
    __syncthreads();

    // Phase 1: Q·K with half2 (ORIGINAL - no float4, no __ldg)
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

    // Phase 2: max reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
    if (lane_id == 0) s_max[warp_id] = local_max;
    __syncthreads();
    float global_max;
    if (tid < DECODE_NUM_WARPS) local_max = s_max[tid];
    #pragma unroll
    for (int offset = DECODE_NUM_WARPS / 2; offset > 0; offset /= 2)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
    if (tid == 0) s_max[0] = local_max;
    __syncthreads();
    global_max = s_max[0];

    // Phase 3: softmax
    float local_sum = 0.0f;
    for (int k = tid; k < kv_len; k += DECODE_BLOCK_SIZE) {
        float val = s_scores[k] - global_max;
        float exp_val = (val > SOFTMAX_FTZ) ? expf(val) : 0.0f;
        s_scores[k] = exp_val;
        local_sum += exp_val;
    }
    __syncthreads();
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    if (lane_id == 0) s_sum[warp_id] = local_sum;
    __syncthreads();
    float global_sum;
    if (tid < DECODE_NUM_WARPS) local_sum = s_sum[tid];
    #pragma unroll
    for (int offset = DECODE_NUM_WARPS / 2; offset > 0; offset /= 2)
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    if (tid == 0) s_sum[0] = local_sum;
    __syncthreads();
    global_sum = s_sum[0];
    float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;

    // Phase 4: V accumulation (ORIGINAL - no __ldg)
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

// --- FP16 Decode Online-Softmax (128 threads, CUDA Graph) - ORIGINAL ---
constexpr int ONLINE_TILE_K = 256;
constexpr int ONLINE_BLOCK_SIZE = 128;
constexpr int ONLINE_NUM_WARPS = 4;

__global__ void flash_attention_decode_kernel_fp16_online_softmax(
    const half* __restrict__ Q, const half* __restrict__ K_cache,
    const half* __restrict__ V_cache, half* __restrict__ O,
    const int32_t* __restrict__ pos_ptr,
    const int head_num, const int kv_head_num, const int head_size,
    const int kv_mul, const int kv_dim, const float scale
) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    if (head >= head_num) return;
    const int pos = *reinterpret_cast<const volatile int32_t*>(pos_ptr);
    const int kv_len = pos + 1;
    const int kv_head = head / kv_mul;
    const int head_offset = kv_head * head_size;
    const int head_size_h2 = head_size / 2;

    extern __shared__ char smem_raw[];
    half* s_query = reinterpret_cast<half*>(smem_raw);
    float* s_scores = reinterpret_cast<float*>(smem_raw + head_size * sizeof(half));
    float* s_max = s_scores + ONLINE_TILE_K;
    float* s_sum = s_max + ONLINE_NUM_WARPS;

    const half* q_ptr = Q + head * head_size;
    const half2* q_ptr_h2 = reinterpret_cast<const half2*>(q_ptr);
    half2* s_query_h2 = reinterpret_cast<half2*>(s_query);
    for (int d = tid; d < head_size_h2; d += ONLINE_BLOCK_SIZE) {
        s_query_h2[d] = q_ptr_h2[d];
    }
    __syncthreads();

    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    float acc_o = 0.0f;

    for (int tile_start = 0; tile_start < kv_len; tile_start += ONLINE_TILE_K) {
        const int tile_end = min(tile_start + ONLINE_TILE_K, kv_len);
        const int tile_len = tile_end - tile_start;

        // Q·K: ORIGINAL half2 (no float4, no __ldg)
        float tile_max_local = -FLT_MAX;
        for (int k_idx = tid; k_idx < tile_len; k_idx += ONLINE_BLOCK_SIZE) {
            const int kv_pos = tile_start + k_idx;
            const half2* k_ptr_h2 = reinterpret_cast<const half2*>(K_cache + kv_pos * kv_dim + head_offset);
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

        // Max reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            tile_max_local = fmaxf(tile_max_local, __shfl_xor_sync(0xffffffff, tile_max_local, offset));
        if (lane_id == 0) s_max[warp_id] = tile_max_local;
        __syncthreads();
        float m_j = s_max[0];
        for (int w = 1; w < ONLINE_NUM_WARPS; w++) m_j = fmaxf(m_j, s_max[w]);
        float m_new = fmaxf(row_max, m_j);

        // Softmax
        float tile_sum_local = 0.0f;
        for (int k_idx = tid; k_idx < tile_len; k_idx += ONLINE_BLOCK_SIZE) {
            float val = s_scores[k_idx] - m_new;
            float exp_val = (val > SOFTMAX_FTZ) ? expf(val) : 0.0f;
            s_scores[k_idx] = exp_val;
            tile_sum_local += exp_val;
        }
        __syncthreads();
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            tile_sum_local += __shfl_xor_sync(0xffffffff, tile_sum_local, offset);
        if (lane_id == 0) s_sum[warp_id] = tile_sum_local;
        __syncthreads();
        float l_j = s_sum[0];
        for (int w = 1; w < ONLINE_NUM_WARPS; w++) l_j += s_sum[w];

        float correction = expf(row_max - m_new);
        acc_o *= correction;

        // V accumulation: ORIGINAL (no __ldg)
        const int my_dim = tid;
        if (my_dim < head_size) {
            for (int k = 0; k < tile_len; k++) {
                const int kv_pos = tile_start + k;
                const half* v_ptr = V_cache + kv_pos * kv_dim + head_offset;
                acc_o += s_scores[k] * __half2float(v_ptr[my_dim]);
            }
        }
        row_max = m_new;
        row_sum = correction * row_sum + l_j;
        __syncthreads();
    }

    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    half* o_ptr = O + head * head_size;
    const int my_dim = tid;
    if (my_dim < head_size) {
        o_ptr[my_dim] = __float2half(acc_o * inv_sum);
    }
}

// --- FP16 Prefill (128 threads) - ORIGINAL ---
constexpr int BLOCK_SIZE = 128;
constexpr int TILE_K = 1024;

__global__ void flash_attention_prefill_kernel_fp16(
    const half* __restrict__ Q, const half* __restrict__ K_cache,
    const half* __restrict__ V_cache, half* __restrict__ O,
    const int seq_len, const int start_pos, const int head_num,
    const int kv_head_num, const int head_size, const int kv_mul,
    const int dim, const int kv_dim, const float scale
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

    extern __shared__ char smem_prefill_fp16[];
    half* s_query = reinterpret_cast<half*>(smem_prefill_fp16);
    float* s_scores = reinterpret_cast<float*>(smem_prefill_fp16 + head_size * sizeof(half));
    half2* s_query_h2 = reinterpret_cast<half2*>(s_query);

    const half* q_ptr = Q + seq_idx * dim + head * head_size;
    for (int d = tid; d < head_size; d += BLOCK_SIZE) {
        s_query[d] = q_ptr[d];
    }
    __syncthreads();

    float acc_o = 0.0f;
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    const half* v_thread_base = V_cache + head_offset + tid;

    for (int tile_start = 0; tile_start < kv_len; tile_start += TILE_K) {
        const int tile_len = min(TILE_K, kv_len - tile_start);

        // Q·K: ORIGINAL half2 with fmaf (no float4, no __ldg)
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

        // Max reduction
        const int lane_id = tid & 31;
        const int warp_id = tid >> 5;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            tile_max_local = fmaxf(tile_max_local, __shfl_xor_sync(0xffffffff, tile_max_local, offset));
        __shared__ float s_warp_max[4];
        if (lane_id == 0) s_warp_max[warp_id] = tile_max_local;
        __syncthreads();
        float m_j;
        if (tid == 0) {
            m_j = fmaxf(fmaxf(s_warp_max[0], s_warp_max[1]), fmaxf(s_warp_max[2], s_warp_max[3]));
            s_warp_max[0] = m_j;
        }
        __syncthreads();
        m_j = s_warp_max[0];
        float m_new = fmaxf(row_max, m_j);

        // Softmax
        float tile_sum_local = 0.0f;
        for (int k_idx = tid; k_idx < tile_len; k_idx += BLOCK_SIZE) {
            float val = s_scores[k_idx] - m_new;
            float exp_score = (val > SOFTMAX_FTZ) ? expf(val) : 0.0f;
            s_scores[k_idx] = exp_score;
            tile_sum_local += exp_score;
        }
        __syncthreads();
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            tile_sum_local += __shfl_xor_sync(0xffffffff, tile_sum_local, offset);
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

        float correction = expf(row_max - m_new);
        acc_o *= correction;

        // V accumulation: ORIGINAL (no __ldg)
        if (tid < head_size) {
            const half* v_ptr = v_thread_base + tile_start * kv_dim;
            int k = 0;
            for (; k + 7 < tile_len; k += 8) {
                float s0 = s_scores[k], s1 = s_scores[k+1], s2 = s_scores[k+2], s3 = s_scores[k+3];
                float s4 = s_scores[k+4], s5 = s_scores[k+5], s6 = s_scores[k+6], s7 = s_scores[k+7];
                float v0 = __half2float(v_ptr[0]);
                float v1 = __half2float(v_ptr[kv_dim]);
                float v2 = __half2float(v_ptr[2*kv_dim]);
                float v3 = __half2float(v_ptr[3*kv_dim]);
                float v4 = __half2float(v_ptr[4*kv_dim]);
                float v5 = __half2float(v_ptr[5*kv_dim]);
                float v6 = __half2float(v_ptr[6*kv_dim]);
                float v7 = __half2float(v_ptr[7*kv_dim]);
                acc_o = fmaf(s0, v0, acc_o); acc_o = fmaf(s1, v1, acc_o);
                acc_o = fmaf(s2, v2, acc_o); acc_o = fmaf(s3, v3, acc_o);
                acc_o = fmaf(s4, v4, acc_o); acc_o = fmaf(s5, v5, acc_o);
                acc_o = fmaf(s6, v6, acc_o); acc_o = fmaf(s7, v7, acc_o);
                v_ptr += 8 * kv_dim;
            }
            for (; k + 3 < tile_len; k += 4) {
                float s0 = s_scores[k], s1 = s_scores[k+1], s2 = s_scores[k+2], s3 = s_scores[k+3];
                float v0 = __half2float(v_ptr[0]);
                float v1 = __half2float(v_ptr[kv_dim]);
                float v2 = __half2float(v_ptr[2*kv_dim]);
                float v3 = __half2float(v_ptr[3*kv_dim]);
                acc_o = fmaf(s0, v0, acc_o); acc_o = fmaf(s1, v1, acc_o);
                acc_o = fmaf(s2, v2, acc_o); acc_o = fmaf(s3, v3, acc_o);
                v_ptr += 4 * kv_dim;
            }
            for (; k < tile_len; k++) {
                acc_o = fmaf(s_scores[k], __half2float(v_ptr[0]), acc_o);
                v_ptr += kv_dim;
            }
        }
        row_max = m_new;
        row_sum = correction * row_sum + l_j;
        __syncthreads();
    }

    if (tid < head_size) {
        float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
        half* o_ptr = O + seq_idx * dim + head * head_size;
        o_ptr[tid] = __float2half(acc_o * inv_sum);
    }
}

}  // namespace original

// ====================== OPTIMIZED (AFTER) KERNELS ======================
namespace optimized {

// --- FP16 Decode (256 threads, non-graph) - OPTIMIZED ---
constexpr int DECODE_BLOCK_SIZE = 256;
constexpr int DECODE_NUM_WARPS = 8;
constexpr int DECODE_WARP_SIZE = 32;

__global__ void flash_attention_decode_kernel_fp16_optimized(
    const half* __restrict__ Q,
    const half* __restrict__ K_cache,
    const half* __restrict__ V_cache,
    half* __restrict__ O,
    const int pos, const int head_num, const int kv_head_num,
    const int head_size, const int kv_mul, const int kv_dim, const float scale
) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane_id = tid % DECODE_WARP_SIZE;
    const int warp_id = tid / DECODE_WARP_SIZE;
    if (head >= head_num) return;
    const int kv_head = head / kv_mul;
    const int head_offset = kv_head * head_size;
    const int kv_len = pos + 1;

    extern __shared__ char smem_raw[];
    half* s_query = reinterpret_cast<half*>(smem_raw);
    float* s_scores = reinterpret_cast<float*>(smem_raw + head_size * sizeof(half));
    float* s_max = s_scores + ((kv_len + DECODE_BLOCK_SIZE - 1) / DECODE_BLOCK_SIZE) * DECODE_BLOCK_SIZE;
    float* s_sum = s_max + DECODE_NUM_WARPS;

    const half* q_ptr = Q + head * head_size;
    const half2* q_ptr_h2 = reinterpret_cast<const half2*>(q_ptr);
    half2* s_query_h2 = reinterpret_cast<half2*>(s_query);
    for (int d = tid; d < head_size / 2; d += DECODE_BLOCK_SIZE) {
        s_query_h2[d] = q_ptr_h2[d];
    }
    __syncthreads();

    // Phase 1: Q·K with float4 vectorization + __ldg (OPTIMIZED)
    float local_max = -FLT_MAX;
    for (int k = tid; k < kv_len; k += DECODE_BLOCK_SIZE) {
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
                acc.x += q_f.x * k_f.x;
                acc.y += q_f.y * k_f.y;
            }
        }
        float score = (acc.x + acc.y) * scale;
        s_scores[k] = score;
        local_max = fmaxf(local_max, score);
    }
    __syncthreads();

    // Phase 2: max reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
    if (lane_id == 0) s_max[warp_id] = local_max;
    __syncthreads();
    float global_max;
    if (tid < DECODE_NUM_WARPS) local_max = s_max[tid];
    #pragma unroll
    for (int offset = DECODE_NUM_WARPS / 2; offset > 0; offset /= 2)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
    if (tid == 0) s_max[0] = local_max;
    __syncthreads();
    global_max = s_max[0];

    // Phase 3: softmax
    float local_sum = 0.0f;
    for (int k = tid; k < kv_len; k += DECODE_BLOCK_SIZE) {
        float val = s_scores[k] - global_max;
        float exp_val = (val > SOFTMAX_FTZ) ? expf(val) : 0.0f;
        s_scores[k] = exp_val;
        local_sum += exp_val;
    }
    __syncthreads();
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    if (lane_id == 0) s_sum[warp_id] = local_sum;
    __syncthreads();
    float global_sum;
    if (tid < DECODE_NUM_WARPS) local_sum = s_sum[tid];
    #pragma unroll
    for (int offset = DECODE_NUM_WARPS / 2; offset > 0; offset /= 2)
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    if (tid == 0) s_sum[0] = local_sum;
    __syncthreads();
    global_sum = s_sum[0];
    float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;

    // Phase 4: V accumulation with __ldg (OPTIMIZED)
    half* o_ptr = O + head * head_size;
    for (int d = tid; d < head_size; d += DECODE_BLOCK_SIZE) {
        float acc = 0.0f;
        int k = 0;
        for (; k + 3 < kv_len; k += 4) {
            const half* v0 = V_cache + (k + 0) * kv_dim + head_offset;
            const half* v1 = V_cache + (k + 1) * kv_dim + head_offset;
            const half* v2 = V_cache + (k + 2) * kv_dim + head_offset;
            const half* v3 = V_cache + (k + 3) * kv_dim + head_offset;
            acc += s_scores[k + 0] * __half2float(__ldg(v0 + d));
            acc += s_scores[k + 1] * __half2float(__ldg(v1 + d));
            acc += s_scores[k + 2] * __half2float(__ldg(v2 + d));
            acc += s_scores[k + 3] * __half2float(__ldg(v3 + d));
        }
        for (; k < kv_len; k++) {
            const half* v_ptr = V_cache + k * kv_dim + head_offset;
            acc += s_scores[k] * __half2float(__ldg(v_ptr + d));
        }
        o_ptr[d] = __float2half(acc * inv_sum);
    }
}

// --- FP16 Decode Online-Softmax (128 threads, CUDA Graph) - OPTIMIZED ---
constexpr int ONLINE_TILE_K = 256;
constexpr int ONLINE_BLOCK_SIZE = 128;
constexpr int ONLINE_NUM_WARPS = 4;

__global__ void flash_attention_decode_kernel_fp16_online_softmax(
    const half* __restrict__ Q, const half* __restrict__ K_cache,
    const half* __restrict__ V_cache, half* __restrict__ O,
    const int32_t* __restrict__ pos_ptr,
    const int head_num, const int kv_head_num, const int head_size,
    const int kv_mul, const int kv_dim, const float scale
) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    if (head >= head_num) return;
    const int pos = *reinterpret_cast<const volatile int32_t*>(pos_ptr);
    const int kv_len = pos + 1;
    const int kv_head = head / kv_mul;
    const int head_offset = kv_head * head_size;

    extern __shared__ char smem_raw[];
    half* s_query = reinterpret_cast<half*>(smem_raw);
    float* s_scores = reinterpret_cast<float*>(smem_raw + head_size * sizeof(half));
    float* s_max = s_scores + ONLINE_TILE_K;
    float* s_sum = s_max + ONLINE_NUM_WARPS;

    const half* q_ptr = Q + head * head_size;
    const half2* q_ptr_h2 = reinterpret_cast<const half2*>(q_ptr);
    half2* s_query_h2 = reinterpret_cast<half2*>(s_query);
    for (int d = tid; d < head_size / 2; d += ONLINE_BLOCK_SIZE) {
        s_query_h2[d] = q_ptr_h2[d];
    }
    __syncthreads();

    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    float acc_o = 0.0f;

    for (int tile_start = 0; tile_start < kv_len; tile_start += ONLINE_TILE_K) {
        const int tile_end = min(tile_start + ONLINE_TILE_K, kv_len);
        const int tile_len = tile_end - tile_start;

        // Q·K: float4 vectorization + __ldg (OPTIMIZED)
        float tile_max_local = -FLT_MAX;
        for (int k_idx = tid; k_idx < tile_len; k_idx += ONLINE_BLOCK_SIZE) {
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
        for (int offset = 16; offset > 0; offset /= 2)
            tile_max_local = fmaxf(tile_max_local, __shfl_xor_sync(0xffffffff, tile_max_local, offset));
        if (lane_id == 0) s_max[warp_id] = tile_max_local;
        __syncthreads();
        float m_j = s_max[0];
        for (int w = 1; w < ONLINE_NUM_WARPS; w++) m_j = fmaxf(m_j, s_max[w]);
        float m_new = fmaxf(row_max, m_j);

        // Softmax
        float tile_sum_local = 0.0f;
        for (int k_idx = tid; k_idx < tile_len; k_idx += ONLINE_BLOCK_SIZE) {
            float val = s_scores[k_idx] - m_new;
            float exp_val = (val > SOFTMAX_FTZ) ? expf(val) : 0.0f;
            s_scores[k_idx] = exp_val;
            tile_sum_local += exp_val;
        }
        __syncthreads();
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            tile_sum_local += __shfl_xor_sync(0xffffffff, tile_sum_local, offset);
        if (lane_id == 0) s_sum[warp_id] = tile_sum_local;
        __syncthreads();
        float l_j = s_sum[0];
        for (int w = 1; w < ONLINE_NUM_WARPS; w++) l_j += s_sum[w];

        float correction = expf(row_max - m_new);
        acc_o *= correction;

        // V accumulation: __ldg (OPTIMIZED)
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

    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    half* o_ptr = O + head * head_size;
    const int my_dim = tid;
    if (my_dim < head_size) {
        o_ptr[my_dim] = __float2half(acc_o * inv_sum);
    }
}

// --- FP16 Prefill (128 threads) - OPTIMIZED ---
constexpr int BLOCK_SIZE = 128;
constexpr int TILE_K = 1024;

__global__ void flash_attention_prefill_kernel_fp16(
    const half* __restrict__ Q, const half* __restrict__ K_cache,
    const half* __restrict__ V_cache, half* __restrict__ O,
    const int seq_len, const int start_pos, const int head_num,
    const int kv_head_num, const int head_size, const int kv_mul,
    const int dim, const int kv_dim, const float scale
) {
    const int head = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int tid = threadIdx.x;
    if (head >= head_num || seq_idx >= seq_len) return;
    const int kv_head = head / kv_mul;
    const int head_offset = kv_head * head_size;
    const int cur_pos = start_pos + seq_idx;
    const int kv_len = cur_pos + 1;

    extern __shared__ char smem_prefill_fp16[];
    half* s_query = reinterpret_cast<half*>(smem_prefill_fp16);
    float* s_scores = reinterpret_cast<float*>(smem_prefill_fp16 + head_size * sizeof(half));

    const half* q_ptr = Q + seq_idx * dim + head * head_size;
    for (int d = tid; d < head_size; d += BLOCK_SIZE) {
        s_query[d] = q_ptr[d];
    }
    __syncthreads();

    float acc_o = 0.0f;
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    const half* v_thread_base = V_cache + head_offset + tid;

    for (int tile_start = 0; tile_start < kv_len; tile_start += TILE_K) {
        const int tile_len = min(TILE_K, kv_len - tile_start);

        // Q·K: float4 vectorization + __ldg + fmaf (OPTIMIZED)
        float tile_max_local = -FLT_MAX;
        for (int k_idx = tid; k_idx < tile_len; k_idx += BLOCK_SIZE) {
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
                    acc.x = fmaf(q_f.x, k_f.x, acc.x);
                    acc.y = fmaf(q_f.y, k_f.y, acc.y);
                }
            }
            float score = (acc.x + acc.y) * scale;
            s_scores[k_idx] = score;
            tile_max_local = fmaxf(tile_max_local, score);
        }
        __syncthreads();

        // Max reduction
        const int lane_id = tid & 31;
        const int warp_id = tid >> 5;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            tile_max_local = fmaxf(tile_max_local, __shfl_xor_sync(0xffffffff, tile_max_local, offset));
        __shared__ float s_warp_max[4];
        if (lane_id == 0) s_warp_max[warp_id] = tile_max_local;
        __syncthreads();
        float m_j;
        if (tid == 0) {
            m_j = fmaxf(fmaxf(s_warp_max[0], s_warp_max[1]), fmaxf(s_warp_max[2], s_warp_max[3]));
            s_warp_max[0] = m_j;
        }
        __syncthreads();
        m_j = s_warp_max[0];
        float m_new = fmaxf(row_max, m_j);

        // Softmax
        float tile_sum_local = 0.0f;
        for (int k_idx = tid; k_idx < tile_len; k_idx += BLOCK_SIZE) {
            float val = s_scores[k_idx] - m_new;
            float exp_score = (val > SOFTMAX_FTZ) ? expf(val) : 0.0f;
            s_scores[k_idx] = exp_score;
            tile_sum_local += exp_score;
        }
        __syncthreads();
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            tile_sum_local += __shfl_xor_sync(0xffffffff, tile_sum_local, offset);
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

        float correction = expf(row_max - m_new);
        acc_o *= correction;

        // V accumulation: __ldg (OPTIMIZED)
        if (tid < head_size) {
            const half* v_ptr = v_thread_base + tile_start * kv_dim;
            int k = 0;
            for (; k + 7 < tile_len; k += 8) {
                float s0 = s_scores[k], s1 = s_scores[k+1], s2 = s_scores[k+2], s3 = s_scores[k+3];
                float s4 = s_scores[k+4], s5 = s_scores[k+5], s6 = s_scores[k+6], s7 = s_scores[k+7];
                float v0 = __half2float(__ldg(v_ptr));
                float v1 = __half2float(__ldg(v_ptr + kv_dim));
                float v2 = __half2float(__ldg(v_ptr + 2*kv_dim));
                float v3 = __half2float(__ldg(v_ptr + 3*kv_dim));
                float v4 = __half2float(__ldg(v_ptr + 4*kv_dim));
                float v5 = __half2float(__ldg(v_ptr + 5*kv_dim));
                float v6 = __half2float(__ldg(v_ptr + 6*kv_dim));
                float v7 = __half2float(__ldg(v_ptr + 7*kv_dim));
                acc_o = fmaf(s0, v0, acc_o); acc_o = fmaf(s1, v1, acc_o);
                acc_o = fmaf(s2, v2, acc_o); acc_o = fmaf(s3, v3, acc_o);
                acc_o = fmaf(s4, v4, acc_o); acc_o = fmaf(s5, v5, acc_o);
                acc_o = fmaf(s6, v6, acc_o); acc_o = fmaf(s7, v7, acc_o);
                v_ptr += 8 * kv_dim;
            }
            for (; k + 3 < tile_len; k += 4) {
                float s0 = s_scores[k], s1 = s_scores[k+1], s2 = s_scores[k+2], s3 = s_scores[k+3];
                float v0 = __half2float(__ldg(v_ptr));
                float v1 = __half2float(__ldg(v_ptr + kv_dim));
                float v2 = __half2float(__ldg(v_ptr + 2*kv_dim));
                float v3 = __half2float(__ldg(v_ptr + 3*kv_dim));
                acc_o = fmaf(s0, v0, acc_o); acc_o = fmaf(s1, v1, acc_o);
                acc_o = fmaf(s2, v2, acc_o); acc_o = fmaf(s3, v3, acc_o);
                v_ptr += 4 * kv_dim;
            }
            for (; k < tile_len; k++) {
                acc_o = fmaf(s_scores[k], __half2float(__ldg(v_ptr)), acc_o);
                v_ptr += kv_dim;
            }
        }
        row_max = m_new;
        row_sum = correction * row_sum + l_j;
        __syncthreads();
    }

    if (tid < head_size) {
        float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
        half* o_ptr = O + seq_idx * dim + head * head_size;
        o_ptr[tid] = __float2half(acc_o * inv_sum);
    }
}

}  // namespace optimized

// ======================== BENCHMARK DRIVER ===================================
int main() {
    // Realistic LLM attention parameters (Qwen3-8B)
    const int head_size = HEAD_SIZE;       // 128
    const int head_num = HEAD_NUM;         // 32
    const int kv_head_num = KV_HEAD_NUM;   // 8
    const int kv_mul = KV_MUL;             // 4
    const int kv_dim = KV_DIM;             // 1024
    const int dim = head_num * head_size;  // 4096

    // Decode context: pos=290 (typical short generation)
    const int decode_pos = 290;
    const int decode_kv_len = decode_pos + 1;  // 291

    // Prefill context: seq_len=8
    const int prefill_seq_len = 8;
    const int prefill_start_pos = 0;

    const float scale = 1.0f / sqrtf((float)head_size);

    // Allocate device memory
    half *d_Q, *d_K, *d_V, *d_O;
    int32_t *d_pos;

    // Q for decode: [dim] = [4096]
    cudaMalloc(&d_Q, dim * sizeof(half));
    // Q for prefill: [seq_len, dim]
    half* d_Q_prefill;
    cudaMalloc(&d_Q_prefill, prefill_seq_len * dim * sizeof(half));
    // K_cache, V_cache: [max_kv_len, kv_dim]
    const int max_kv_len = 512;
    cudaMalloc(&d_K, max_kv_len * kv_dim * sizeof(half));
    cudaMalloc(&d_V, max_kv_len * kv_dim * sizeof(half));
    // Output
    cudaMalloc(&d_O, dim * sizeof(half));
    half* d_O_prefill;
    cudaMalloc(&d_O_prefill, prefill_seq_len * dim * sizeof(half));
    // GPU position pointer
    cudaMalloc(&d_pos, sizeof(int32_t));

    // Initialize with random data
    {
        const int total = max_kv_len * kv_dim;
        half* h_data = (half*)malloc(total * sizeof(half));
        for (int i = 0; i < total; i++)
            h_data[i] = __float2half((float)(rand() % 1000 - 500) / 5000.0f);
        cudaMemcpy(d_K, h_data, total * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, h_data, total * sizeof(half), cudaMemcpyHostToDevice);
        
        half* h_q = (half*)malloc(prefill_seq_len * dim * sizeof(half));
        for (int i = 0; i < prefill_seq_len * dim; i++)
            h_q[i] = __float2half((float)(rand() % 1000 - 500) / 5000.0f);
        cudaMemcpy(d_Q, h_q, dim * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Q_prefill, h_q, prefill_seq_len * dim * sizeof(half), cudaMemcpyHostToDevice);
        
        int h_pos = decode_pos;
        cudaMemcpy(d_pos, &h_pos, sizeof(int32_t), cudaMemcpyHostToDevice);
        
        free(h_data);
        free(h_q);
    }
    cudaDeviceSynchronize();

    printf("=== Flash Attention Benchmark ===\n");
    printf("head_size=%d, head_num=%d, kv_head_num=%d, kv_dim=%d\n",
           head_size, head_num, kv_head_num, kv_dim);
    printf("decode pos=%d (kv_len=%d), prefill seq_len=%d\n\n",
           decode_pos, decode_kv_len, prefill_seq_len);

    // =================== ORIGINAL KERNELS ===================
    printf("=== Running ORIGINAL kernels ===\n");

    // 1. Decode FP16 (256 threads) - ORIGINAL
    {
        int score_buf = ((decode_kv_len + original::DECODE_BLOCK_SIZE - 1) / original::DECODE_BLOCK_SIZE)
                        * original::DECODE_BLOCK_SIZE;
        int smem = head_size * sizeof(half) + score_buf * sizeof(float)
                 + 2 * original::DECODE_NUM_WARPS * sizeof(float);
        original::flash_attention_decode_kernel_fp16_optimized
            <<<head_num, original::DECODE_BLOCK_SIZE, smem>>>(
            d_Q, d_K, d_V, d_O, decode_pos, head_num, kv_head_num,
            head_size, kv_mul, kv_dim, scale);
    }
    cudaDeviceSynchronize();

    // 2. Decode Online-Softmax (128 threads) - ORIGINAL
    {
        int smem = head_size * sizeof(half) + original::ONLINE_TILE_K * sizeof(float)
                 + 2 * original::ONLINE_NUM_WARPS * sizeof(float);
        original::flash_attention_decode_kernel_fp16_online_softmax
            <<<head_num, original::ONLINE_BLOCK_SIZE, smem>>>(
            d_Q, d_K, d_V, d_O, d_pos, head_num, kv_head_num,
            head_size, kv_mul, kv_dim, scale);
    }
    cudaDeviceSynchronize();

    // 3. Prefill FP16 (128 threads) - ORIGINAL
    {
        int smem = head_size * sizeof(half) + original::TILE_K * sizeof(float);
        dim3 grid(head_num, prefill_seq_len);
        original::flash_attention_prefill_kernel_fp16
            <<<grid, original::BLOCK_SIZE, smem>>>(
            d_Q_prefill, d_K, d_V, d_O_prefill, prefill_seq_len,
            prefill_start_pos, head_num, kv_head_num, head_size,
            kv_mul, dim, kv_dim, scale);
    }
    cudaDeviceSynchronize();

    // =================== OPTIMIZED KERNELS ===================
    printf("=== Running OPTIMIZED kernels ===\n");

    // 1. Decode FP16 (256 threads) - OPTIMIZED
    {
        int score_buf = ((decode_kv_len + optimized::DECODE_BLOCK_SIZE - 1) / optimized::DECODE_BLOCK_SIZE)
                        * optimized::DECODE_BLOCK_SIZE;
        int smem = head_size * sizeof(half) + score_buf * sizeof(float)
                 + 2 * optimized::DECODE_NUM_WARPS * sizeof(float);
        optimized::flash_attention_decode_kernel_fp16_optimized
            <<<head_num, optimized::DECODE_BLOCK_SIZE, smem>>>(
            d_Q, d_K, d_V, d_O, decode_pos, head_num, kv_head_num,
            head_size, kv_mul, kv_dim, scale);
    }
    cudaDeviceSynchronize();

    // 2. Decode Online-Softmax (128 threads) - OPTIMIZED
    {
        int smem = head_size * sizeof(half) + optimized::ONLINE_TILE_K * sizeof(float)
                 + 2 * optimized::ONLINE_NUM_WARPS * sizeof(float);
        optimized::flash_attention_decode_kernel_fp16_online_softmax
            <<<head_num, optimized::ONLINE_BLOCK_SIZE, smem>>>(
            d_Q, d_K, d_V, d_O, d_pos, head_num, kv_head_num,
            head_size, kv_mul, kv_dim, scale);
    }
    cudaDeviceSynchronize();

    // 3. Prefill FP16 (128 threads) - OPTIMIZED
    {
        int smem = head_size * sizeof(half) + optimized::TILE_K * sizeof(float);
        dim3 grid(head_num, prefill_seq_len);
        optimized::flash_attention_prefill_kernel_fp16
            <<<grid, optimized::BLOCK_SIZE, smem>>>(
            d_Q_prefill, d_K, d_V, d_O_prefill, prefill_seq_len,
            prefill_start_pos, head_num, kv_head_num, head_size,
            kv_mul, dim, kv_dim, scale);
    }
    cudaDeviceSynchronize();

    printf("=== All kernels completed ===\n");

    cudaFree(d_Q); cudaFree(d_Q_prefill);
    cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_O); cudaFree(d_O_prefill);
    cudaFree(d_pos);
    return 0;
}
