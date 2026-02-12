/**
 * NCU profiling program for RoPE kernels
 * Launches each kernel exactly once for clean NCU profiling
 *
 * Compile:
 * nvcc -O3 -arch=sm_87 -DQWEN2_SUPPORT -DQWEN3_SUPPORT \
 *   ncu_profile_rope.cu -o ncu_profile_rope
 *
 * Profile:
 * sudo /usr/local/cuda/bin/ncu --set full -o ncu_rope_report ./ncu_profile_rope
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Qwen3-8B parameters
static const int HEAD_SIZE = 128;
static const int NUM_Q_HEADS = 32;
static const int NUM_KV_HEADS = 8;
static const int DIM = NUM_Q_HEADS * HEAD_SIZE;     // 4096
static const int KV_DIM = NUM_KV_HEADS * HEAD_SIZE; // 1024
static const int MAX_SEQ_LEN = 32768;
static const int SEQ_LEN = 128;
static const int POS = 42;

// ============= Copy kernel implementations here =============
// These are copied from the source to allow standalone compilation

#if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)

__global__ void rope_kernel_cu_fp32(int pos, int dim, int kv_dim, int head_size,
                                    const float* input_q, const float* input_k,
                                    const float* __restrict__ sin_cache, const float* __restrict__ cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int num_heads = dim / head_size;
  int head_pair_count = head_size / 2;
  int total_pairs = num_heads * head_pair_count;
  if (idx >= total_pairs) return;

  int head_idx = idx / head_pair_count;
  int head_dim = idx % head_pair_count;
  int i = head_idx * head_size;
  int v0_idx = i + head_dim;
  int v1_idx = i + head_dim + head_size / 2;

  float fci = sin_cache[pos * head_size + head_dim * 2];
  float fcr = cos_cache[pos * head_size + head_dim * 2];
  int rotn = i < kv_dim ? 2 : 1;

  for (int v = 0; v < rotn; v++) {
    float* vec = const_cast<float*>(v == 0 ? input_q : input_k);
    float v0 = vec[v0_idx];
    float v1 = vec[v1_idx];
    vec[v0_idx] = fcr * v0 - fci * v1;
    vec[v1_idx] = fcr * v1 + fci * v0;
  }
}

__global__ void sin_cos_calc(int head_size, int max_seq_len, float* sin_cache, float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int head_dim = idx % head_size;
  float freq = 1.0f / powf(1000000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
  for (int pos = 0; pos < max_seq_len; ++pos) {
    float val = static_cast<float>(pos) * freq;
    float fci, fcr;
    __sincosf(val, &fci, &fcr);
    *(sin_cache + pos * head_size + head_dim) = fci;
    *(cos_cache + pos * head_size + head_dim) = fcr;
  }
}

__global__ void rope_kernel_cu_fp32_gpu_pos(const int32_t* pos_ptr, int dim, int kv_dim, int head_size,
                                            const float* input_q, const float* input_k,
                                            const float* __restrict__ sin_cache, const float* __restrict__ cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int num_heads = dim / head_size;
  int head_pair_count = head_size / 2;
  int total_pairs = num_heads * head_pair_count;
  if (idx >= total_pairs) return;
  int pos = *reinterpret_cast<const volatile int32_t*>(pos_ptr);
  int head_idx = idx / head_pair_count;
  int head_dim = idx % head_pair_count;
  int i = head_idx * head_size;
  int v0_idx = i + head_dim;
  int v1_idx = i + head_dim + head_size / 2;
  float fci = sin_cache[pos * head_size + head_dim * 2];
  float fcr = cos_cache[pos * head_size + head_dim * 2];
  int rotn = i < kv_dim ? 2 : 1;
  for (int v = 0; v < rotn; v++) {
    float* vec = const_cast<float*>(v == 0 ? input_q : input_k);
    float v0 = vec[v0_idx];
    float v1 = vec[v1_idx];
    vec[v0_idx] = fcr * v0 - fci * v1;
    vec[v1_idx] = fcr * v1 + fci * v0;
  }
}

__global__ void rope_kernel_cu_fp16_impl(int pos, int dim, int kv_dim, int head_size,
                                          half* input_q, half* input_k,
                                          const float* __restrict__ sin_cache, const float* __restrict__ cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int num_heads = dim / head_size;
  int head_pair_count = head_size / 2;
  int total_pairs = num_heads * head_pair_count;
  if (idx >= total_pairs) return;
  int head_idx = idx / head_pair_count;
  int head_dim = idx % head_pair_count;
  int i = head_idx * head_size;
  int v0_idx = i + head_dim;
  int v1_idx = i + head_dim + head_size / 2;
  float fci = sin_cache[pos * head_size + head_dim * 2];
  float fcr = cos_cache[pos * head_size + head_dim * 2];
  int rotn = i < kv_dim ? 2 : 1;
  for (int v = 0; v < rotn; v++) {
    half* vec = (v == 0) ? input_q : input_k;
    float v0 = __half2float(vec[v0_idx]);
    float v1 = __half2float(vec[v1_idx]);
    vec[v0_idx] = __float2half(fcr * v0 - fci * v1);
    vec[v1_idx] = __float2half(fcr * v1 + fci * v0);
  }
}

__global__ void rope_kernel_cu_fp16_gpu_pos_impl(const int32_t* pos_ptr, int dim, int kv_dim, int head_size,
                                                  half* input_q, half* input_k,
                                                  const float* __restrict__ sin_cache, const float* __restrict__ cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int num_heads = dim / head_size;
  int head_pair_count = head_size / 2;
  int total_pairs = num_heads * head_pair_count;
  if (idx >= total_pairs) return;
  int pos = *reinterpret_cast<const volatile int32_t*>(pos_ptr);
  int head_idx = idx / head_pair_count;
  int head_dim = idx % head_pair_count;
  int i = head_idx * head_size;
  int v0_idx = i + head_dim;
  int v1_idx = i + head_dim + head_size / 2;
  float fci = sin_cache[pos * head_size + head_dim * 2];
  float fcr = cos_cache[pos * head_size + head_dim * 2];
  int rotn = i < kv_dim ? 2 : 1;
  for (int v = 0; v < rotn; v++) {
    half* vec = (v == 0) ? input_q : input_k;
    float v0 = __half2float(vec[v0_idx]);
    float v1 = __half2float(vec[v1_idx]);
    vec[v0_idx] = __float2half(fcr * v0 - fci * v1);
    vec[v1_idx] = __float2half(fcr * v1 + fci * v0);
  }
}

__global__ void batched_rope_kernel_cu_fp32(int start_pos, int seq_len, int dim, int kv_dim, int head_size,
                                            float* input_q, float* input_k,
                                            const float* __restrict__ sin_cache, const float* __restrict__ cos_cache) {
  int seq_idx = blockIdx.x;
  if (seq_idx >= seq_len) return;
  int pos = start_pos + seq_idx;
  int idx = threadIdx.x + blockDim.x * blockIdx.y;
  int num_heads = dim / head_size;
  int head_pair_count = head_size / 2;
  int total_pairs = num_heads * head_pair_count;
  if (idx >= total_pairs) return;
  int head_idx = idx / head_pair_count;
  int head_dim = idx % head_pair_count;
  int i = head_idx * head_size;
  int v0_idx = i + head_dim;
  int v1_idx = i + head_dim + head_size / 2;
  float fci = sin_cache[pos * head_size + head_dim * 2];
  float fcr = cos_cache[pos * head_size + head_dim * 2];
  int rotn = i < kv_dim ? 2 : 1;
  int q_offset = seq_idx * dim;
  int k_offset = seq_idx * kv_dim;
  for (int v = 0; v < rotn; v++) {
    float* vec = (v == 0) ? (input_q + q_offset) : (input_k + k_offset);
    int actual_v0_idx = (v == 0) ? v0_idx : (v0_idx < kv_dim ? v0_idx : -1);
    int actual_v1_idx = (v == 0) ? v1_idx : (v1_idx < kv_dim ? v1_idx : -1);
    if (actual_v0_idx >= 0 && actual_v1_idx >= 0) {
      float v0 = vec[actual_v0_idx];
      float v1 = vec[actual_v1_idx];
      vec[actual_v0_idx] = fcr * v0 - fci * v1;
      vec[actual_v1_idx] = fcr * v1 + fci * v0;
    }
  }
}

__global__ void batched_rope_kernel_cu_fp16_impl(int start_pos, int seq_len, int dim, int kv_dim, int head_size,
                                                  half* input_q, half* input_k,
                                                  const float* __restrict__ sin_cache, const float* __restrict__ cos_cache) {
  int seq_idx = blockIdx.x;
  if (seq_idx >= seq_len) return;
  int pos = start_pos + seq_idx;
  int idx = threadIdx.x + blockDim.x * blockIdx.y;
  int num_heads = dim / head_size;
  int head_pair_count = head_size / 2;
  int total_pairs = num_heads * head_pair_count;
  if (idx >= total_pairs) return;
  int head_idx = idx / head_pair_count;
  int head_dim = idx % head_pair_count;
  int i = head_idx * head_size;
  int v0_idx = i + head_dim;
  int v1_idx = i + head_dim + head_size / 2;
  float fci = sin_cache[pos * head_size + head_dim * 2];
  float fcr = cos_cache[pos * head_size + head_dim * 2];
  int rotn = i < kv_dim ? 2 : 1;
  int q_offset = seq_idx * dim;
  int k_offset = seq_idx * kv_dim;
  for (int v = 0; v < rotn; v++) {
    half* vec = (v == 0) ? (input_q + q_offset) : (input_k + k_offset);
    int actual_v0_idx = (v == 0) ? v0_idx : (v0_idx < kv_dim ? v0_idx : -1);
    int actual_v1_idx = (v == 0) ? v1_idx : (v1_idx < kv_dim ? v1_idx : -1);
    if (actual_v0_idx >= 0 && actual_v1_idx >= 0) {
      float v0 = __half2float(vec[actual_v0_idx]);
      float v1 = __half2float(vec[actual_v1_idx]);
      vec[actual_v0_idx] = __float2half(fcr * v0 - fci * v1);
      vec[actual_v1_idx] = __float2half(fcr * v1 + fci * v0);
    }
  }
}

// M-RoPE kernels
__global__ void mrope_kernel_cu_fp16_impl(
    int pos_t, int pos_h, int pos_w,
    int dim, int kv_dim, int head_size,
    int section0, int section1, int section2,
    half* input_q, half* input_k,
    const float* __restrict__ sin_cache, const float* __restrict__ cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int num_heads = dim / head_size;
  int num_kv_heads = kv_dim / head_size;
  int half_head_size = head_size / 2;
  int total_pairs = num_heads * half_head_size;
  if (idx >= total_pairs) return;
  int head_idx = idx / half_head_size;
  int pair_idx = idx % half_head_size;
  int i = head_idx * head_size;
  int d0 = pair_idx;
  int d1 = pair_idx + half_head_size;
  int v0_idx = i + d0;
  int v1_idx = i + d1;
  int dim_threshold0 = section0 * 2;
  int dim_threshold1 = dim_threshold0 + section1 * 2;
  int pos0 = (d0 < dim_threshold0) ? pos_t : ((d0 < dim_threshold1) ? pos_h : pos_w);
  int pos1 = (d1 < dim_threshold0) ? pos_t : ((d1 < dim_threshold1) ? pos_h : pos_w);
  int freq_idx = pair_idx * 2;
  float sin0 = sin_cache[pos0 * head_size + freq_idx];
  float cos0 = cos_cache[pos0 * head_size + freq_idx];
  float sin1 = sin_cache[pos1 * head_size + freq_idx];
  float cos1 = cos_cache[pos1 * head_size + freq_idx];
  {
    float v0 = __half2float(input_q[v0_idx]);
    float v1 = __half2float(input_q[v1_idx]);
    input_q[v0_idx] = __float2half(v0 * cos0 - v1 * sin0);
    input_q[v1_idx] = __float2half(v1 * cos1 + v0 * sin1);
  }
  int kv_mul = num_heads / num_kv_heads;
  int kv_head_idx = head_idx / kv_mul;
  if (head_idx % kv_mul == 0) {
    int kv_i = kv_head_idx * head_size;
    int kv_v0_idx = kv_i + d0;
    int kv_v1_idx = kv_i + d1;
    float v0 = __half2float(input_k[kv_v0_idx]);
    float v1 = __half2float(input_k[kv_v1_idx]);
    input_k[kv_v0_idx] = __float2half(v0 * cos0 - v1 * sin0);
    input_k[kv_v1_idx] = __float2half(v1 * cos1 + v0 * sin1);
  }
}

__global__ void mrope_kernel_cu_fp16_gpu_pos_impl(
    const int32_t* pos_ptr, int dim, int kv_dim, int head_size,
    int section0, int section1, int section2,
    half* input_q, half* input_k,
    const float* __restrict__ sin_cache, const float* __restrict__ cos_cache) {
  int pos = pos_ptr[0];
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int num_heads = dim / head_size;
  int num_kv_heads = kv_dim / head_size;
  int half_head_size = head_size / 2;
  int total_pairs = num_heads * half_head_size;
  if (idx >= total_pairs) return;
  int head_idx = idx / half_head_size;
  int pair_idx = idx % half_head_size;
  int i = head_idx * head_size;
  int d0 = pair_idx;
  int d1 = pair_idx + half_head_size;
  int v0_idx = i + d0;
  int v1_idx = i + d1;
  int pos_t = pos, pos_h = pos, pos_w = pos;
  int dim_threshold0 = section0 * 2;
  int dim_threshold1 = dim_threshold0 + section1 * 2;
  int pos0 = (d0 < dim_threshold0) ? pos_t : ((d0 < dim_threshold1) ? pos_h : pos_w);
  int pos1 = (d1 < dim_threshold0) ? pos_t : ((d1 < dim_threshold1) ? pos_h : pos_w);
  int freq_idx = pair_idx * 2;
  float sin0 = sin_cache[pos0 * head_size + freq_idx];
  float cos0 = cos_cache[pos0 * head_size + freq_idx];
  float sin1 = sin_cache[pos1 * head_size + freq_idx];
  float cos1 = cos_cache[pos1 * head_size + freq_idx];
  {
    float v0 = __half2float(input_q[v0_idx]);
    float v1 = __half2float(input_q[v1_idx]);
    input_q[v0_idx] = __float2half(v0 * cos0 - v1 * sin0);
    input_q[v1_idx] = __float2half(v1 * cos1 + v0 * sin1);
  }
  int kv_mul = num_heads / num_kv_heads;
  int kv_head_idx = head_idx / kv_mul;
  if (head_idx % kv_mul == 0) {
    int kv_i = kv_head_idx * head_size;
    int kv_v0_idx = kv_i + d0;
    int kv_v1_idx = kv_i + d1;
    float v0 = __half2float(input_k[kv_v0_idx]);
    float v1 = __half2float(input_k[kv_v1_idx]);
    input_k[kv_v0_idx] = __float2half(v0 * cos0 - v1 * sin0);
    input_k[kv_v1_idx] = __float2half(v1 * cos1 + v0 * sin1);
  }
}

__global__ void batched_mrope_kernel_cu_fp16_impl(
    int seq_len, int dim, int kv_dim, int head_size,
    int section0, int section1, int section2,
    const int32_t* pos_t_arr, const int32_t* pos_h_arr, const int32_t* pos_w_arr,
    half* input_q, half* input_k,
    const float* __restrict__ sin_cache, const float* __restrict__ cos_cache) {
  int seq_idx = blockIdx.x;
  if (seq_idx >= seq_len) return;
  int idx = threadIdx.x + blockDim.x * blockIdx.y;
  int num_heads = dim / head_size;
  int num_kv_heads = kv_dim / head_size;
  int half_head_size = head_size / 2;
  int total_pairs = num_heads * half_head_size;
  if (idx >= total_pairs) return;
  int pos_t = pos_t_arr[seq_idx];
  int pos_h = pos_h_arr[seq_idx];
  int pos_w = pos_w_arr[seq_idx];
  int head_idx = idx / half_head_size;
  int pair_idx = idx % half_head_size;
  int i = head_idx * head_size;
  int d0 = pair_idx;
  int d1 = pair_idx + half_head_size;
  int dim_threshold0 = section0 * 2;
  int dim_threshold1 = dim_threshold0 + section1 * 2;
  int pos0 = (d0 < dim_threshold0) ? pos_t : ((d0 < dim_threshold1) ? pos_h : pos_w);
  int pos1 = (d1 < dim_threshold0) ? pos_t : ((d1 < dim_threshold1) ? pos_h : pos_w);
  int freq_idx = pair_idx * 2;
  float sin0 = sin_cache[pos0 * head_size + freq_idx];
  float cos0 = cos_cache[pos0 * head_size + freq_idx];
  float sin1 = sin_cache[pos1 * head_size + freq_idx];
  float cos1 = cos_cache[pos1 * head_size + freq_idx];
  int q_offset = seq_idx * dim;
  int k_offset = seq_idx * kv_dim;
  int v0_idx = i + d0;
  int v1_idx = i + d1;
  {
    float v0 = __half2float(input_q[q_offset + v0_idx]);
    float v1 = __half2float(input_q[q_offset + v1_idx]);
    input_q[q_offset + v0_idx] = __float2half(v0 * cos0 - v1 * sin0);
    input_q[q_offset + v1_idx] = __float2half(v1 * cos1 + v0 * sin1);
  }
  int kv_mul = num_heads / num_kv_heads;
  int kv_head_idx = head_idx / kv_mul;
  if (head_idx % kv_mul == 0) {
    int kv_i = kv_head_idx * head_size;
    int kv_v0_idx = kv_i + d0;
    int kv_v1_idx = kv_i + d1;
    float v0 = __half2float(input_k[k_offset + kv_v0_idx]);
    float v1 = __half2float(input_k[k_offset + kv_v1_idx]);
    input_k[k_offset + kv_v0_idx] = __float2half(v0 * cos0 - v1 * sin0);
    input_k[k_offset + kv_v1_idx] = __float2half(v1 * cos1 + v0 * sin1);
  }
}

#endif // QWEN2_SUPPORT || QWEN3_SUPPORT

void init_sin_cos_cache(float* sin_cache, float* cos_cache, int head_size, int max_seq_len) {
  for (int pos = 0; pos < max_seq_len; pos++) {
    for (int i = 0; i < head_size; i++) {
      float freq = 1.0f / powf(1000000.0f, (float)i / (float)head_size);
      float val = (float)pos * freq;
      sin_cache[pos * head_size + i] = sinf(val);
      cos_cache[pos * head_size + i] = cosf(val);
    }
  }
}

int main() {
  // Allocate host memory
  int cache_size = MAX_SEQ_LEN * HEAD_SIZE;
  float* h_sin = (float*)calloc(cache_size, sizeof(float));
  float* h_cos = (float*)calloc(cache_size, sizeof(float));
  init_sin_cos_cache(h_sin, h_cos, HEAD_SIZE, MAX_SEQ_LEN);

  float* h_q_f32 = (float*)calloc(DIM, sizeof(float));
  float* h_k_f32 = (float*)calloc(KV_DIM, sizeof(float));
  half* h_q_f16 = (half*)calloc(DIM, sizeof(half));
  half* h_k_f16 = (half*)calloc(KV_DIM, sizeof(half));
  
  // Init with random
  for (int i = 0; i < DIM; i++) {
    h_q_f32[i] = (float)rand() / RAND_MAX - 0.5f;
    h_q_f16[i] = __float2half(h_q_f32[i]);
  }
  for (int i = 0; i < KV_DIM; i++) {
    h_k_f32[i] = (float)rand() / RAND_MAX - 0.5f;
    h_k_f16[i] = __float2half(h_k_f32[i]);
  }

  // Allocate device memory
  float *d_sin, *d_cos, *d_q_f32, *d_k_f32;
  half *d_q_f16, *d_k_f16;
  int32_t *d_pos;
  
  cudaMalloc(&d_sin, cache_size * sizeof(float));
  cudaMalloc(&d_cos, cache_size * sizeof(float));
  cudaMalloc(&d_q_f32, DIM * sizeof(float));
  cudaMalloc(&d_k_f32, KV_DIM * sizeof(float));
  cudaMalloc(&d_q_f16, DIM * sizeof(half));
  cudaMalloc(&d_k_f16, KV_DIM * sizeof(half));
  cudaMalloc(&d_pos, sizeof(int32_t));
  
  cudaMemcpy(d_sin, h_sin, cache_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_cos, h_cos, cache_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_q_f32, h_q_f32, DIM * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_k_f32, h_k_f32, KV_DIM * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_q_f16, h_q_f16, DIM * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_k_f16, h_k_f16, KV_DIM * sizeof(half), cudaMemcpyHostToDevice);
  int32_t h_pos = POS;
  cudaMemcpy(d_pos, &h_pos, sizeof(int32_t), cudaMemcpyHostToDevice);

  // Batched buffers
  float *d_bq_f32, *d_bk_f32;
  half *d_bq_f16, *d_bk_f16;
  cudaMalloc(&d_bq_f32, SEQ_LEN * DIM * sizeof(float));
  cudaMalloc(&d_bk_f32, SEQ_LEN * KV_DIM * sizeof(float));
  cudaMalloc(&d_bq_f16, SEQ_LEN * DIM * sizeof(half));
  cudaMalloc(&d_bk_f16, SEQ_LEN * KV_DIM * sizeof(half));
  
  // M-RoPE position arrays
  int32_t *d_pos_t, *d_pos_h, *d_pos_w;
  cudaMalloc(&d_pos_t, SEQ_LEN * sizeof(int32_t));
  cudaMalloc(&d_pos_h, SEQ_LEN * sizeof(int32_t));
  cudaMalloc(&d_pos_w, SEQ_LEN * sizeof(int32_t));
  int32_t* h_pos_arr = (int32_t*)calloc(SEQ_LEN, sizeof(int32_t));
  for (int i = 0; i < SEQ_LEN; i++) h_pos_arr[i] = POS + i;
  cudaMemcpy(d_pos_t, h_pos_arr, SEQ_LEN * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_pos_h, h_pos_arr, SEQ_LEN * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_pos_w, h_pos_arr, SEQ_LEN * sizeof(int32_t), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();

  int threads = 128;
  int total_pairs = NUM_Q_HEADS * (HEAD_SIZE / 2);
  int blocks = (total_pairs + threads - 1) / threads;
  
  // ---- sin_cos_calc ----
  printf("Launching sin_cos_calc\n");
  sin_cos_calc<<<1, HEAD_SIZE>>>(HEAD_SIZE, MAX_SEQ_LEN, d_sin, d_cos);
  cudaDeviceSynchronize();

  // ---- rope_kernel_cu_fp32 ----
  printf("Launching rope_kernel_cu_fp32\n");
  cudaMemcpy(d_q_f32, h_q_f32, DIM * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_k_f32, h_k_f32, KV_DIM * sizeof(float), cudaMemcpyHostToDevice);
  rope_kernel_cu_fp32<<<blocks, threads>>>(POS, DIM, KV_DIM, HEAD_SIZE, d_q_f32, d_k_f32, d_sin, d_cos);
  cudaDeviceSynchronize();

  // ---- rope_kernel_cu_fp32_gpu_pos ----
  printf("Launching rope_kernel_cu_fp32_gpu_pos\n");
  cudaMemcpy(d_q_f32, h_q_f32, DIM * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_k_f32, h_k_f32, KV_DIM * sizeof(float), cudaMemcpyHostToDevice);
  rope_kernel_cu_fp32_gpu_pos<<<blocks, threads>>>(d_pos, DIM, KV_DIM, HEAD_SIZE, d_q_f32, d_k_f32, d_sin, d_cos);
  cudaDeviceSynchronize();

  // ---- rope_kernel_cu_fp16_impl ----
  printf("Launching rope_kernel_cu_fp16_impl\n");
  cudaMemcpy(d_q_f16, h_q_f16, DIM * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_k_f16, h_k_f16, KV_DIM * sizeof(half), cudaMemcpyHostToDevice);
  rope_kernel_cu_fp16_impl<<<blocks, threads>>>(POS, DIM, KV_DIM, HEAD_SIZE, d_q_f16, d_k_f16, d_sin, d_cos);
  cudaDeviceSynchronize();

  // ---- rope_kernel_cu_fp16_gpu_pos_impl ----
  printf("Launching rope_kernel_cu_fp16_gpu_pos_impl\n");
  cudaMemcpy(d_q_f16, h_q_f16, DIM * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_k_f16, h_k_f16, KV_DIM * sizeof(half), cudaMemcpyHostToDevice);
  rope_kernel_cu_fp16_gpu_pos_impl<<<blocks, threads>>>(d_pos, DIM, KV_DIM, HEAD_SIZE, d_q_f16, d_k_f16, d_sin, d_cos);
  cudaDeviceSynchronize();

  // ---- batched_rope_kernel_cu_fp32 ----
  printf("Launching batched_rope_kernel_cu_fp32\n");
  int blocks_y = (total_pairs + threads - 1) / threads;
  dim3 grid_batch(SEQ_LEN, blocks_y);
  batched_rope_kernel_cu_fp32<<<grid_batch, threads>>>(POS, SEQ_LEN, DIM, KV_DIM, HEAD_SIZE, d_bq_f32, d_bk_f32, d_sin, d_cos);
  cudaDeviceSynchronize();

  // ---- batched_rope_kernel_cu_fp16_impl ----
  printf("Launching batched_rope_kernel_cu_fp16_impl\n");
  batched_rope_kernel_cu_fp16_impl<<<grid_batch, threads>>>(POS, SEQ_LEN, DIM, KV_DIM, HEAD_SIZE, d_bq_f16, d_bk_f16, d_sin, d_cos);
  cudaDeviceSynchronize();

  // ---- mrope_kernel_cu_fp16_impl ----
  printf("Launching mrope_kernel_cu_fp16_impl\n");
  cudaMemcpy(d_q_f16, h_q_f16, DIM * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_k_f16, h_k_f16, KV_DIM * sizeof(half), cudaMemcpyHostToDevice);
  mrope_kernel_cu_fp16_impl<<<blocks, threads>>>(POS, POS+1, POS+2, DIM, KV_DIM, HEAD_SIZE, 24, 20, 20, d_q_f16, d_k_f16, d_sin, d_cos);
  cudaDeviceSynchronize();

  // ---- mrope_kernel_cu_fp16_gpu_pos_impl ----
  printf("Launching mrope_kernel_cu_fp16_gpu_pos_impl\n");
  cudaMemcpy(d_q_f16, h_q_f16, DIM * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_k_f16, h_k_f16, KV_DIM * sizeof(half), cudaMemcpyHostToDevice);
  mrope_kernel_cu_fp16_gpu_pos_impl<<<blocks, threads>>>(d_pos, DIM, KV_DIM, HEAD_SIZE, 24, 20, 20, d_q_f16, d_k_f16, d_sin, d_cos);
  cudaDeviceSynchronize();

  // ---- batched_mrope_kernel_cu_fp16_impl ----
  printf("Launching batched_mrope_kernel_cu_fp16_impl\n");
  cudaMemcpy(d_bq_f16, h_q_f16, DIM * sizeof(half), cudaMemcpyHostToDevice); // just first token
  cudaMemcpy(d_bk_f16, h_k_f16, KV_DIM * sizeof(half), cudaMemcpyHostToDevice);
  batched_mrope_kernel_cu_fp16_impl<<<grid_batch, threads>>>(SEQ_LEN, DIM, KV_DIM, HEAD_SIZE, 24, 20, 20, d_pos_t, d_pos_h, d_pos_w, d_bq_f16, d_bk_f16, d_sin, d_cos);
  cudaDeviceSynchronize();

  printf("All kernels launched successfully\n");

  // Cleanup
  cudaFree(d_sin); cudaFree(d_cos);
  cudaFree(d_q_f32); cudaFree(d_k_f32);
  cudaFree(d_q_f16); cudaFree(d_k_f16);
  cudaFree(d_pos);
  cudaFree(d_bq_f32); cudaFree(d_bk_f32);
  cudaFree(d_bq_f16); cudaFree(d_bk_f16);
  cudaFree(d_pos_t); cudaFree(d_pos_h); cudaFree(d_pos_w);
  free(h_sin); free(h_cos);
  free(h_q_f32); free(h_k_f32);
  free(h_q_f16); free(h_k_f16);
  free(h_pos_arr);
  return 0;
}
