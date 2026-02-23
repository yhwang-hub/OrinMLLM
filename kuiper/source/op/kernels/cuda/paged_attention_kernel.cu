/**
 * Paged Attention CUDA Kernels
 *
 * Block-based KV cache layout:
 *   key_pool / value_pool: [num_blocks, PAGE_SIZE, kv_dim]
 *   block_table (per-layer): [max_blocks_per_seq]  maps logical_block -> physical_block
 *
 * Address translation for token at position `pos`:
 *   physical_block = block_table[pos >> PAGE_SHIFT]
 *   element_offset = (physical_block * PAGE_SIZE + (pos & PAGE_MASK)) * kv_dim
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <cstdio>
#include <cub/cub.cuh>
#include "paged_attention_kernel.cuh"

namespace kernel {

// Page configuration (must be power of 2)
constexpr int PAGE_SIZE = 16;
constexpr int PAGE_SHIFT = 4;
constexpr int PAGE_MASK = 15;

// Block sizes matching the contiguous FA1 kernels
constexpr int PG_BLOCK_FP32 = 256;
constexpr int PG_BLOCK_FP16 = 128;
constexpr int PG_DECODE_BLOCK = 256;
constexpr int PG_DECODE_WARPS = 8;
constexpr int PG_ONLINE_BLOCK = 128;
constexpr int PG_ONLINE_WARPS = 4;
constexpr int PG_TILE_K = 1024;
constexpr int PG_ONLINE_TILE_K = 256;
constexpr float PG_SOFTMAX_FTZ = -20.0f;

// ============================================================================
// Helper: paged KV offset (element count, not bytes)
// ============================================================================
__device__ __forceinline__ size_t paged_off(const int32_t* bt, int32_t pos, int32_t kv_dim) {
  return ((size_t)bt[pos >> PAGE_SHIFT] * PAGE_SIZE + (pos & PAGE_MASK)) * kv_dim;
}

// ============================================================================
// 1. FP32 Prefill Kernel (Paged)
// ============================================================================
__global__ void paged_prefill_fp32_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ key_pool,
    const float* __restrict__ value_pool,
    float* __restrict__ O,
    const int32_t* __restrict__ block_table,
    const int seq_len, const int start_pos,
    const int head_num, const int kv_head_num,
    const int head_size, const int kv_mul,
    const int dim, const int kv_dim, const float scale)
{
  const int head = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int tid = threadIdx.x;
  if (head >= head_num || seq_idx >= seq_len) return;

  const int kv_head = head / kv_mul;
  const int head_offset = kv_head * head_size;
  const int cur_pos = start_pos + seq_idx;
  const int kv_len = cur_pos + 1;

  extern __shared__ float smem[];
  float* s_query  = smem;
  float* s_scores = smem + head_size;

  const float* q_ptr = Q + seq_idx * dim + head * head_size;
  for (int d = tid; d < head_size; d += PG_BLOCK_FP32) s_query[d] = q_ptr[d];
  __syncthreads();

  float* o_ptr = O + seq_idx * dim + head * head_size;
  float acc_o[4] = {0,0,0,0};
  float row_max = -FLT_MAX, row_sum = 0.f;

  for (int tile_start = 0; tile_start < kv_len; tile_start += PG_TILE_K) {
    const int tile_end = min(tile_start + PG_TILE_K, kv_len);
    const int tile_len = tile_end - tile_start;
    float tile_max = -FLT_MAX;

    for (int k = tid; k < tile_len; k += PG_BLOCK_FP32) {
      const int kv_pos = tile_start + k;
      const float* k_ptr = key_pool + paged_off(block_table, kv_pos, kv_dim) + head_offset;
      float score = 0.f;
      const float4* sq4 = reinterpret_cast<const float4*>(s_query);
      const float4* kk4 = reinterpret_cast<const float4*>(k_ptr);
      #pragma unroll
      for (int d = 0; d < head_size / 4; d++) {
        float4 q = sq4[d]; float4 kk = __ldg(kk4 + d);
        score += q.x*kk.x + q.y*kk.y + q.z*kk.z + q.w*kk.w;
      }
      s_scores[k] = score * scale;
      tile_max = fmaxf(tile_max, s_scores[k]);
    }
    __syncthreads();

    typedef cub::BlockReduce<float, PG_BLOCK_FP32> BR;
    __shared__ typename BR::TempStorage ts;
    float bm = BR(ts).Reduce(tile_max, cub::Max());
    __shared__ float s_tm;
    if (tid == 0) s_tm = bm;
    __syncthreads();
    float m_j = s_tm;
    float m_new = fmaxf(row_max, m_j);

    float tsum = 0.f;
    for (int k = tid; k < tile_len; k += PG_BLOCK_FP32) {
      float e = expf(s_scores[k] - m_new); s_scores[k] = e; tsum += e;
    }
    __syncthreads();
    float bs = BR(ts).Sum(tsum);
    __shared__ float s_ts;
    if (tid == 0) s_ts = bs;
    __syncthreads();
    float l_j = s_ts;
    float correction = expf(row_max - m_new);
    float l_new = correction * row_sum + l_j;

    for (int i = 0; i < (head_size + PG_BLOCK_FP32 - 1) / PG_BLOCK_FP32; i++) {
      const int d = tid + i * PG_BLOCK_FP32;
      if (d < head_size) {
        acc_o[i] *= correction;
        for (int k = 0; k < tile_len; k++) {
          const float* v_ptr = value_pool + paged_off(block_table, tile_start + k, kv_dim) + head_offset;
          acc_o[i] += s_scores[k] * __ldg(v_ptr + d);
        }
      }
    }
    row_max = m_new; row_sum = l_new;
    __syncthreads();
  }

  float inv = (row_sum > 0.f) ? (1.f / row_sum) : 0.f;
  for (int i = 0; i < (head_size + PG_BLOCK_FP32 - 1) / PG_BLOCK_FP32; i++) {
    const int d = tid + i * PG_BLOCK_FP32;
    if (d < head_size) o_ptr[d] = acc_o[i] * inv;
  }
}

// ============================================================================
// 2. FP32 Decode Kernel (Paged)
// ============================================================================
__global__ void paged_decode_fp32_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ key_pool,
    const float* __restrict__ value_pool,
    float* __restrict__ O,
    const int32_t* __restrict__ block_table,
    const int pos, const int head_num, const int kv_head_num,
    const int head_size, const int kv_mul, const int dim,
    const int kv_dim, const float scale)
{
  const int head = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid % 32, warp = tid / 32;
  if (head >= head_num) return;

  const int kv_head = head / kv_mul;
  const int head_offset = kv_head * head_size;
  const int kv_len = pos + 1;

  extern __shared__ float smem[];
  float* s_query = smem;
  float* s_scores = smem + head_size;
  float* s_max = s_scores + ((kv_len + PG_BLOCK_FP32 - 1) / PG_BLOCK_FP32) * PG_BLOCK_FP32;
  float* s_sum = s_max + 8;

  const float* q_ptr = Q + head * head_size;
  for (int d = tid; d < head_size; d += PG_BLOCK_FP32) s_query[d] = q_ptr[d];
  __syncthreads();

  float lmax = -FLT_MAX;
  for (int k = tid; k < kv_len; k += PG_BLOCK_FP32) {
    const float* k_ptr = key_pool + paged_off(block_table, k, kv_dim) + head_offset;
    float score = 0.f;
    const float4* sq4 = reinterpret_cast<const float4*>(s_query);
    const float4* kk4 = reinterpret_cast<const float4*>(k_ptr);
    #pragma unroll
    for (int d = 0; d < head_size / 4; d++) {
      float4 q = sq4[d]; float4 kk = __ldg(kk4 + d);
      score += q.x*kk.x + q.y*kk.y + q.z*kk.z + q.w*kk.w;
    }
    score *= scale; s_scores[k] = score; lmax = fmaxf(lmax, score);
  }
  __syncthreads();

  #pragma unroll
  for (int o = 16; o > 0; o /= 2) lmax = fmaxf(lmax, __shfl_down_sync(0xffffffff, lmax, o));
  if (lane == 0) s_max[warp] = lmax;
  __syncthreads();
  float gmax = -FLT_MAX;
  if (tid < 8) gmax = s_max[tid];
  for (int o = 4; o > 0; o /= 2) gmax = fmaxf(gmax, __shfl_down_sync(0xffffffff, gmax, o));
  if (tid == 0) s_max[0] = gmax;
  __syncthreads();
  gmax = s_max[0];

  float lsum = 0.f;
  for (int k = tid; k < kv_len; k += PG_BLOCK_FP32) {
    float e = expf(s_scores[k] - gmax); s_scores[k] = e; lsum += e;
  }
  __syncthreads();
  #pragma unroll
  for (int o = 16; o > 0; o /= 2) lsum += __shfl_down_sync(0xffffffff, lsum, o);
  if (lane == 0) s_sum[warp] = lsum;
  __syncthreads();
  float gsum = 0.f;
  if (tid < 8) gsum = s_sum[tid];
  for (int o = 4; o > 0; o /= 2) gsum += __shfl_down_sync(0xffffffff, gsum, o);
  if (tid == 0) s_sum[0] = gsum;
  __syncthreads();
  gsum = s_sum[0];
  float inv = (gsum > 0.f) ? (1.f / gsum) : 0.f;

  float* o_ptr = O + head * head_size;
  for (int d = tid; d < head_size; d += PG_BLOCK_FP32) {
    float acc = 0.f;
    for (int k = 0; k < kv_len; k++) {
      const float* v_ptr = value_pool + paged_off(block_table, k, kv_dim) + head_offset;
      acc += s_scores[k] * __ldg(v_ptr + d);
    }
    o_ptr[d] = acc * inv;
  }
}

// ============================================================================
// 3. FP16 Decode Kernel (Paged, 256 threads, full softmax)
// ============================================================================
__global__ void paged_decode_fp16_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ key_pool,
    const half* __restrict__ value_pool,
    half* __restrict__ O,
    const int32_t* __restrict__ block_table,
    const int pos, const int head_num, const int kv_head_num,
    const int head_size, const int kv_mul, const int kv_dim, const float scale)
{
  const int head = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid % 32, warp = tid / 32;
  if (head >= head_num) return;

  const int kv_head = head / kv_mul;
  const int head_offset = kv_head * head_size;
  const int kv_len = pos + 1;
  const int hs_h2 = head_size / 2;

  extern __shared__ char smem_raw[];
  half* s_query = reinterpret_cast<half*>(smem_raw);
  float* s_scores = reinterpret_cast<float*>(smem_raw + head_size * sizeof(half));
  float* s_max = s_scores + ((kv_len + PG_DECODE_BLOCK - 1) / PG_DECODE_BLOCK) * PG_DECODE_BLOCK;
  float* s_sum = s_max + PG_DECODE_WARPS;

  // Load query
  const half2* q_h2 = reinterpret_cast<const half2*>(Q + head * head_size);
  half2* sq_h2 = reinterpret_cast<half2*>(s_query);
  for (int d = tid; d < hs_h2; d += PG_DECODE_BLOCK) sq_h2[d] = q_h2[d];
  __syncthreads();

  // Phase 1: Q·K scores
  float lmax = -FLT_MAX;
  for (int k = tid; k < kv_len; k += PG_DECODE_BLOCK) {
    const float4* kf4 = reinterpret_cast<const float4*>(key_pool + paged_off(block_table, k, kv_dim) + head_offset);
    const float4* qf4 = reinterpret_cast<const float4*>(s_query);
    float2 acc = make_float2(0.f, 0.f);
    #pragma unroll
    for (int d = 0; d < head_size / 8; d++) {
      float4 qp = qf4[d]; float4 kp = __ldg(kf4 + d);
      const half2* qh = reinterpret_cast<const half2*>(&qp);
      const half2* kh = reinterpret_cast<const half2*>(&kp);
      #pragma unroll
      for (int i = 0; i < 4; i++) {
        float2 qf = __half22float2(qh[i]), kf = __half22float2(kh[i]);
        acc.x += qf.x * kf.x; acc.y += qf.y * kf.y;
      }
    }
    float score = (acc.x + acc.y) * scale;
    s_scores[k] = score; lmax = fmaxf(lmax, score);
  }
  __syncthreads();

  // Warp max
  #pragma unroll
  for (int o = 16; o > 0; o /= 2) lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffff, lmax, o));
  if (lane == 0) s_max[warp] = lmax;
  __syncthreads();
  float gmax;
  if (tid < PG_DECODE_WARPS) lmax = s_max[tid];
  #pragma unroll
  for (int o = PG_DECODE_WARPS / 2; o > 0; o /= 2) lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffff, lmax, o));
  if (tid == 0) s_max[0] = lmax;
  __syncthreads();
  gmax = s_max[0];

  // Softmax
  float lsum = 0.f;
  for (int k = tid; k < kv_len; k += PG_DECODE_BLOCK) {
    float v = s_scores[k] - gmax;
    float e = (v > PG_SOFTMAX_FTZ) ? expf(v) : 0.f;
    s_scores[k] = e; lsum += e;
  }
  __syncthreads();
  #pragma unroll
  for (int o = 16; o > 0; o /= 2) lsum += __shfl_xor_sync(0xffffffff, lsum, o);
  if (lane == 0) s_sum[warp] = lsum;
  __syncthreads();
  float gsum;
  if (tid < PG_DECODE_WARPS) lsum = s_sum[tid];
  #pragma unroll
  for (int o = PG_DECODE_WARPS / 2; o > 0; o /= 2) lsum += __shfl_xor_sync(0xffffffff, lsum, o);
  if (tid == 0) s_sum[0] = lsum;
  __syncthreads();
  gsum = s_sum[0];
  float inv = (gsum > 0.f) ? (1.f / gsum) : 0.f;

  // Phase 3: Weighted V sum
  half* o_ptr = O + head * head_size;
  for (int d = tid; d < head_size; d += PG_DECODE_BLOCK) {
    float acc = 0.f;
    int k = 0;
    for (; k + 3 < kv_len; k += 4) {
      const half* v0 = value_pool + paged_off(block_table, k+0, kv_dim) + head_offset;
      const half* v1 = value_pool + paged_off(block_table, k+1, kv_dim) + head_offset;
      const half* v2 = value_pool + paged_off(block_table, k+2, kv_dim) + head_offset;
      const half* v3 = value_pool + paged_off(block_table, k+3, kv_dim) + head_offset;
      acc += s_scores[k+0] * __half2float(__ldg(v0 + d));
      acc += s_scores[k+1] * __half2float(__ldg(v1 + d));
      acc += s_scores[k+2] * __half2float(__ldg(v2 + d));
      acc += s_scores[k+3] * __half2float(__ldg(v3 + d));
    }
    for (; k < kv_len; k++) {
      const half* vp = value_pool + paged_off(block_table, k, kv_dim) + head_offset;
      acc += s_scores[k] * __half2float(__ldg(vp + d));
    }
    o_ptr[d] = __float2half(acc * inv);
  }
}

// ============================================================================
// 4. FP16 Prefill Kernel (Paged, 128 threads, online softmax)
// ============================================================================
__global__ void paged_prefill_fp16_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ key_pool,
    const half* __restrict__ value_pool,
    half* __restrict__ O,
    const int32_t* __restrict__ block_table,
    const int seq_len, const int start_pos,
    const int head_num, const int kv_head_num,
    const int head_size, const int kv_mul,
    const int dim, const int kv_dim, const float scale)
{
  const int head = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int tid = threadIdx.x;
  if (head >= head_num || seq_idx >= seq_len) return;

  const int kv_head = head / kv_mul;
  const int head_offset = kv_head * head_size;
  const int cur_pos = start_pos + seq_idx;
  const int kv_len = cur_pos + 1;

  extern __shared__ char smem_pf16[];
  half* s_query = reinterpret_cast<half*>(smem_pf16);
  float* s_scores = reinterpret_cast<float*>(smem_pf16 + head_size * sizeof(half));

  const half* q_ptr = Q + seq_idx * dim + head * head_size;
  for (int d = tid; d < head_size; d += PG_BLOCK_FP16) s_query[d] = q_ptr[d];
  __syncthreads();

  float acc_o = 0.f, row_max = -FLT_MAX, row_sum = 0.f;

  for (int tile_start = 0; tile_start < kv_len; tile_start += PG_TILE_K) {
    const int tile_len = min(PG_TILE_K, kv_len - tile_start);

    // Q·K scores
    float tmax_local = -FLT_MAX;
    for (int ki = tid; ki < tile_len; ki += PG_BLOCK_FP16) {
      const int kv_pos = tile_start + ki;
      const float4* kf4 = reinterpret_cast<const float4*>(key_pool + paged_off(block_table, kv_pos, kv_dim) + head_offset);
      const float4* qf4 = reinterpret_cast<const float4*>(s_query);
      float2 acc = make_float2(0.f, 0.f);
      #pragma unroll
      for (int d = 0; d < head_size / 8; d++) {
        float4 qp = qf4[d]; float4 kp = __ldg(kf4 + d);
        const half2* qh = reinterpret_cast<const half2*>(&qp);
        const half2* kh = reinterpret_cast<const half2*>(&kp);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
          float2 qf = __half22float2(qh[i]), kf = __half22float2(kh[i]);
          acc.x = fmaf(qf.x, kf.x, acc.x); acc.y = fmaf(qf.y, kf.y, acc.y);
        }
      }
      float score = (acc.x + acc.y) * scale;
      s_scores[ki] = score; tmax_local = fmaxf(tmax_local, score);
    }
    __syncthreads();

    // Block max (4 warps)
    const int lane = tid & 31, warp_id = tid >> 5;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) tmax_local = fmaxf(tmax_local, __shfl_xor_sync(0xffffffff, tmax_local, o));
    __shared__ float s_wm[4];
    if (lane == 0) s_wm[warp_id] = tmax_local;
    __syncthreads();
    float m_j;
    if (tid == 0) { m_j = fmaxf(fmaxf(s_wm[0], s_wm[1]), fmaxf(s_wm[2], s_wm[3])); s_wm[0] = m_j; }
    __syncthreads();
    m_j = s_wm[0];
    float m_new = fmaxf(row_max, m_j);

    // Exp + sum
    float tsum_local = 0.f;
    for (int ki = tid; ki < tile_len; ki += PG_BLOCK_FP16) {
      float v = s_scores[ki] - m_new;
      float e = (v > PG_SOFTMAX_FTZ) ? expf(v) : 0.f;
      s_scores[ki] = e; tsum_local += e;
    }
    __syncthreads();
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) tsum_local += __shfl_xor_sync(0xffffffff, tsum_local, o);
    __shared__ float s_ws[4];
    if (lane == 0) s_ws[warp_id] = tsum_local;
    __syncthreads();
    float l_j;
    if (tid == 0) { l_j = s_ws[0] + s_ws[1] + s_ws[2] + s_ws[3]; s_ws[0] = l_j; }
    __syncthreads();
    l_j = s_ws[0];

    float correction = expf(row_max - m_new);
    acc_o *= correction;

    // Accumulate V
    if (tid < head_size) {
      for (int k = 0; k < tile_len; k++) {
        int32_t kv_pos = tile_start + k;
        const half* v_ptr = value_pool + paged_off(block_table, kv_pos, kv_dim) + head_offset;
        acc_o = fmaf(s_scores[k], __half2float(__ldg(v_ptr + tid)), acc_o);
      }
    }

    row_max = m_new;
    row_sum = correction * row_sum + l_j;
    __syncthreads();
  }

  if (tid < head_size) {
    float inv = (row_sum > 0.f) ? (1.f / row_sum) : 0.f;
    half* o_ptr = O + seq_idx * dim + head * head_size;
    o_ptr[tid] = __float2half(acc_o * inv);
  }
}

// ============================================================================
// 5. FP16 Decode with GPU Position (Paged, online softmax, CUDA Graph)
// ============================================================================
__global__ void paged_decode_fp16_gpu_pos_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ key_pool,
    const half* __restrict__ value_pool,
    half* __restrict__ O,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ pos_ptr,
    const int head_num, const int kv_head_num,
    const int head_size, const int kv_mul,
    const int kv_dim, const float scale)
{
  const int head = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid % 32, warp_id = tid / 32;
  if (head >= head_num) return;

  const int pos = *reinterpret_cast<const volatile int32_t*>(pos_ptr);
  const int kv_len = pos + 1;
  const int kv_head = head / kv_mul;
  const int head_offset = kv_head * head_size;

  extern __shared__ char smem_gp[];
  half* s_query = reinterpret_cast<half*>(smem_gp);
  float* s_scores = reinterpret_cast<float*>(smem_gp + head_size * sizeof(half));
  float* s_max = s_scores + PG_ONLINE_TILE_K;
  float* s_sum = s_max + PG_ONLINE_WARPS;

  const half2* q_h2 = reinterpret_cast<const half2*>(Q + head * head_size);
  half2* sq_h2 = reinterpret_cast<half2*>(s_query);
  for (int d = tid; d < head_size / 2; d += PG_ONLINE_BLOCK) sq_h2[d] = q_h2[d];
  __syncthreads();

  float row_max = -FLT_MAX, row_sum = 0.f, acc_o = 0.f;

  for (int tile_start = 0; tile_start < kv_len; tile_start += PG_ONLINE_TILE_K) {
    const int tile_end = min(tile_start + PG_ONLINE_TILE_K, kv_len);
    const int tile_len = tile_end - tile_start;

    float tmax_local = -FLT_MAX;
    for (int ki = tid; ki < tile_len; ki += PG_ONLINE_BLOCK) {
      const int kv_pos = tile_start + ki;
      const float4* kf4 = reinterpret_cast<const float4*>(key_pool + paged_off(block_table, kv_pos, kv_dim) + head_offset);
      const float4* qf4 = reinterpret_cast<const float4*>(s_query);
      float2 acc = make_float2(0.f, 0.f);
      #pragma unroll
      for (int d = 0; d < head_size / 8; d++) {
        float4 qp = qf4[d]; float4 kp = __ldg(kf4 + d);
        const half2* qh = reinterpret_cast<const half2*>(&qp);
        const half2* kh = reinterpret_cast<const half2*>(&kp);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
          float2 qf = __half22float2(qh[i]), kf = __half22float2(kh[i]);
          acc.x += qf.x * kf.x; acc.y += qf.y * kf.y;
        }
      }
      float score = (acc.x + acc.y) * scale;
      s_scores[ki] = score; tmax_local = fmaxf(tmax_local, score);
    }
    __syncthreads();

    #pragma unroll
    for (int o = 16; o > 0; o /= 2) tmax_local = fmaxf(tmax_local, __shfl_xor_sync(0xffffffff, tmax_local, o));
    if (lane == 0) s_max[warp_id] = tmax_local;
    __syncthreads();
    float m_j = s_max[0];
    for (int w = 1; w < PG_ONLINE_WARPS; w++) m_j = fmaxf(m_j, s_max[w]);
    float m_new = fmaxf(row_max, m_j);

    float tsum_local = 0.f;
    for (int ki = tid; ki < tile_len; ki += PG_ONLINE_BLOCK) {
      float v = s_scores[ki] - m_new;
      float e = (v > PG_SOFTMAX_FTZ) ? expf(v) : 0.f;
      s_scores[ki] = e; tsum_local += e;
    }
    __syncthreads();
    #pragma unroll
    for (int o = 16; o > 0; o /= 2) tsum_local += __shfl_xor_sync(0xffffffff, tsum_local, o);
    if (lane == 0) s_sum[warp_id] = tsum_local;
    __syncthreads();
    float l_j = s_sum[0];
    for (int w = 1; w < PG_ONLINE_WARPS; w++) l_j += s_sum[w];

    float correction = expf(row_max - m_new);
    acc_o *= correction;

    const int my_dim = tid;
    if (my_dim < head_size) {
      for (int k = 0; k < tile_len; k++) {
        const int kv_pos = tile_start + k;
        const half* v_ptr = value_pool + paged_off(block_table, kv_pos, kv_dim) + head_offset;
        acc_o += s_scores[k] * __half2float(__ldg(v_ptr + my_dim));
      }
    }

    row_max = m_new;
    row_sum = correction * row_sum + l_j;
    __syncthreads();
  }

  float inv = (row_sum > 0.f) ? (1.f / row_sum) : 0.f;
  half* o_ptr = O + head * head_size;
  if (tid < head_size) o_ptr[tid] = __float2half(acc_o * inv);
}

// ============================================================================
// 6. Paged KV Cache Write Kernels
// ============================================================================
__global__ void paged_copy_kv_fp32_cu(
    float* kv_pool, const float* src, const int32_t* pos,
    const int32_t* block_table, int32_t kv_dim,
    int32_t layer_idx, int32_t max_blocks_per_seq)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < kv_dim) {
    int32_t position = *reinterpret_cast<const volatile int32_t*>(pos);
    const int32_t* bt = block_table + layer_idx * max_blocks_per_seq;
    size_t off = paged_off(bt, position, kv_dim);
    kv_pool[off + idx] = src[idx];
  }
}

__global__ void paged_copy_kv_fp16_cu(
    half* kv_pool, const half* src, const int32_t* pos,
    const int32_t* block_table, int32_t kv_dim,
    int32_t layer_idx, int32_t max_blocks_per_seq)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < kv_dim) {
    int32_t position = *reinterpret_cast<const volatile int32_t*>(pos);
    const int32_t* bt = block_table + layer_idx * max_blocks_per_seq;
    size_t off = paged_off(bt, position, kv_dim);
    kv_pool[off + idx] = src[idx];
  }
}

// ============================================================================
// Host Launch Functions
// ============================================================================

void paged_flash_attention_prefill_cu(
    int32_t start_pos, int32_t seq_len,
    int32_t head_num, int32_t kv_head_num,
    int32_t head_size, int32_t kv_mul, int32_t layer_index,
    int32_t kv_dim, int32_t page_size, int32_t max_blocks_per_seq,
    const tensor::Tensor& query, const tensor::Tensor& output,
    const void* key_pool, const void* value_pool, const int32_t* block_table,
    CudaConfig* config)
{
  const int32_t* layer_bt = block_table + layer_index * max_blocks_per_seq;
  const int dim = head_num * head_size;
  float scale = 1.0f / sqrtf((float)head_size);

  dim3 grid(head_num, seq_len);
  dim3 block(PG_BLOCK_FP32);
  int smem = (head_size + PG_TILE_K) * sizeof(float);
  cudaStream_t stream = config ? config->stream : nullptr;

  paged_prefill_fp32_kernel<<<grid, block, smem, stream>>>(
      query.ptr<float>(), (const float*)key_pool, (const float*)value_pool,
      const_cast<float*>(output.ptr<float>()), layer_bt,
      seq_len, start_pos, head_num, kv_head_num, head_size, kv_mul, dim, kv_dim, scale);
}

void paged_flash_attention_decode_cu(
    int32_t pos, int32_t head_num, int32_t kv_head_num,
    int32_t head_size, int32_t kv_mul, int32_t layer_index,
    int32_t kv_dim, int32_t page_size, int32_t max_blocks_per_seq,
    const tensor::Tensor& query, const tensor::Tensor& output,
    const void* key_pool, const void* value_pool, const int32_t* block_table,
    CudaConfig* config)
{
  const int32_t* layer_bt = block_table + layer_index * max_blocks_per_seq;
  const int dim = head_num * head_size;
  const int kv_len = pos + 1;
  float scale = 1.0f / sqrtf((float)head_size);

  dim3 grid(head_num);
  dim3 block(PG_BLOCK_FP32);
  int score_buf = ((kv_len + PG_BLOCK_FP32 - 1) / PG_BLOCK_FP32) * PG_BLOCK_FP32;
  int smem = (head_size + score_buf + 16) * sizeof(float);
  cudaStream_t stream = config ? config->stream : nullptr;

  paged_decode_fp32_kernel<<<grid, block, smem, stream>>>(
      query.ptr<float>(), (const float*)key_pool, (const float*)value_pool,
      const_cast<float*>(output.ptr<float>()), layer_bt,
      pos, head_num, kv_head_num, head_size, kv_mul, dim, kv_dim, scale);
}

void paged_flash_attention_prefill_fp16_cu(
    int32_t start_pos, int32_t seq_len,
    int32_t head_num, int32_t kv_head_num,
    int32_t head_size, int32_t kv_mul, int32_t layer_index,
    int32_t kv_dim, int32_t page_size, int32_t max_blocks_per_seq,
    const tensor::Tensor& query, const tensor::Tensor& output,
    const void* key_pool, const void* value_pool, const int32_t* block_table,
    CudaConfig* config)
{
  const int32_t* layer_bt = block_table + layer_index * max_blocks_per_seq;
  const int dim = head_num * head_size;
  float scale = 1.0f / sqrtf((float)head_size);

  dim3 grid(head_num, seq_len);
  dim3 block(PG_BLOCK_FP16);
  int smem = head_size * sizeof(half) + PG_TILE_K * sizeof(float);
  cudaStream_t stream = config ? config->stream : nullptr;

  paged_prefill_fp16_kernel<<<grid, block, smem, stream>>>(
      query.ptr<half>(), (const half*)key_pool, (const half*)value_pool,
      const_cast<half*>(output.ptr<half>()), layer_bt,
      seq_len, start_pos, head_num, kv_head_num, head_size, kv_mul, dim, kv_dim, scale);
}

void paged_flash_attention_decode_fp16_cu(
    int32_t pos, int32_t head_num, int32_t kv_head_num,
    int32_t head_size, int32_t kv_mul, int32_t layer_index,
    int32_t kv_dim, int32_t page_size, int32_t max_blocks_per_seq,
    const tensor::Tensor& query, const tensor::Tensor& output,
    const void* key_pool, const void* value_pool, const int32_t* block_table,
    CudaConfig* config)
{
  const int32_t* layer_bt = block_table + layer_index * max_blocks_per_seq;
  const int kv_len = pos + 1;
  float scale = 1.0f / sqrtf((float)head_size);

  dim3 grid(head_num);
  dim3 block(PG_DECODE_BLOCK);
  int score_buf = ((kv_len + PG_DECODE_BLOCK - 1) / PG_DECODE_BLOCK) * PG_DECODE_BLOCK;
  int smem = head_size * sizeof(half) + score_buf * sizeof(float) + 2 * PG_DECODE_WARPS * sizeof(float);
  cudaStream_t stream = config ? config->stream : nullptr;

  paged_decode_fp16_kernel<<<grid, block, smem, stream>>>(
      query.ptr<half>(), (const half*)key_pool, (const half*)value_pool,
      const_cast<half*>(output.ptr<half>()), layer_bt,
      pos, head_num, kv_head_num, head_size, kv_mul, kv_dim, scale);
}

void paged_flash_attention_decode_fp16_gpu_pos_cu(
    const int32_t* pos_ptr, int32_t head_num, int32_t kv_head_num,
    int32_t head_size, int32_t kv_mul, int32_t layer_index,
    int32_t kv_dim, int32_t page_size, int32_t max_blocks_per_seq,
    const tensor::Tensor& query, const tensor::Tensor& output,
    const void* key_pool, const void* value_pool, const int32_t* block_table,
    CudaConfig* config)
{
  const int32_t* layer_bt = block_table + layer_index * max_blocks_per_seq;
  float scale = 1.0f / sqrtf((float)head_size);

  dim3 grid(head_num);
  dim3 block(PG_ONLINE_BLOCK);
  int smem = head_size * sizeof(half) + PG_ONLINE_TILE_K * sizeof(float) + 2 * PG_ONLINE_WARPS * sizeof(float);
  cudaStream_t stream = config ? config->stream : nullptr;

  paged_decode_fp16_gpu_pos_kernel<<<grid, block, smem, stream>>>(
      query.ptr<half>(), (const half*)key_pool, (const half*)value_pool,
      const_cast<half*>(output.ptr<half>()), layer_bt, pos_ptr,
      head_num, kv_head_num, head_size, kv_mul, kv_dim, scale);
}

void paged_copy_to_kv_cache_kernel(float* kv_pool, const float* src, const int32_t* pos,
    const int32_t* block_table, int32_t kv_dim, int32_t layer_idx,
    int32_t max_blocks_per_seq, int32_t page_size, cudaStream_t stream)
{
  const int bs = 256;
  const int gs = (kv_dim + bs - 1) / bs;
  paged_copy_kv_fp32_cu<<<gs, bs, 0, stream>>>(
      kv_pool, src, pos, block_table, kv_dim, layer_idx, max_blocks_per_seq);
}

void paged_copy_to_kv_cache_kernel_fp16(half* kv_pool, const half* src, const int32_t* pos,
    const int32_t* block_table, int32_t kv_dim, int32_t layer_idx,
    int32_t max_blocks_per_seq, int32_t page_size, cudaStream_t stream)
{
  const int bs = 256;
  const int gs = (kv_dim + bs - 1) / bs;
  paged_copy_kv_fp16_cu<<<gs, bs, 0, stream>>>(
      kv_pool, src, pos, block_table, kv_dim, layer_idx, max_blocks_per_seq);
}

}  // namespace kernel
