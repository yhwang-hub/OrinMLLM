// MHA Kernel Benchmark: Compares original (cub::BlockReduce) vs optimized (warp shuffle + __ldg)
// Usage: nvcc -O3 -arch=sm_87 bench_mha_kernels.cu -o bench_mha -lcudart && ./bench_mha
// NCU:  sudo /usr/local/cuda/bin/ncu --set full -o ncu_mha_report ./bench_mha

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cmath>
#include <chrono>

// ============================================================
// Original kernels (baseline) - uses cub::BlockReduce
// ============================================================
namespace original {

constexpr static int thread_num = 256;

__device__ void softmax_gpu(float* __restrict__ x, int size) {
  int tid = threadIdx.x;
  int step = blockDim.x;
  float max_val = tid < size ? x[tid] : -FLT_MAX;
  for (int i = tid + step; i < size; i += step) {
    if (x[i] > max_val) max_val = x[i];
  }
  using BlockReduce = cub::BlockReduce<float, 256>;
  __shared__ BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
  if (threadIdx.x == 0) shared_val = max_val;
  __syncthreads();
  max_val = shared_val;
  float sum = 0.0f;
  for (int i = tid; i < size; i += step) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) shared_val = sum;
  __syncthreads();
  sum = shared_val;
  for (int i = tid; i < size; i += step) x[i] /= sum;
}

__device__ void softmax_gpu_causal(float* __restrict__ x, int size, int cur_pos, int total_pos) {
  int tid = threadIdx.x;
  int step = blockDim.x;
  for (int i = tid; i <= total_pos; i += step) {
    if (i > cur_pos) x[i] = -FLT_MAX;
  }
  __syncthreads();
  float max_val = tid <= cur_pos ? x[tid] : -FLT_MAX;
  for (int i = tid + step; i <= cur_pos; i += step) {
    if (x[i] > max_val) max_val = x[i];
  }
  using BlockReduce = cub::BlockReduce<float, 256>;
  __shared__ BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
  if (threadIdx.x == 0) shared_val = max_val;
  __syncthreads();
  max_val = shared_val;
  float sum = 0.0f;
  for (int i = tid; i <= cur_pos; i += step) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  for (int i = tid; i <= total_pos; i += step) {
    if (i > cur_pos) x[i] = 0.0f;
  }
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) shared_val = sum;
  __syncthreads();
  sum = shared_val;
  for (int i = tid; i <= cur_pos; i += step) x[i] /= sum;
}

__global__ void multi_head_attention_kernel(
    int32_t pos, int32_t seq_len, float* query, float* score_ptr, float* output,
    float* key_cache, float* value_cache, int32_t kv_dim, int32_t kv_mul,
    int32_t head_num, int32_t head_size, int32_t layer_offset) {
  int head = blockIdx.x;
  if (head >= head_num) return;
  extern __shared__ float s_query_head[];
  float scale = 1.f / sqrtf(float(head_size));
  float* query_head = query + head * head_size;
  for (int i = threadIdx.x; i < head_size; i += blockDim.x)
    s_query_head[i] = query_head[i];
  __syncthreads();
  float* score_head = score_ptr + head * seq_len;
  int head_offset = (head / kv_mul) * head_size;
  for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
    float* key_head = key_cache + layer_offset + t * kv_dim + head_offset;
    float score = 0.0f;
    for (int i = 0; i < head_size; i += 4) {
      float4 key_val = *reinterpret_cast<float4*>(key_head + i);
      float4 query_val = *reinterpret_cast<float4*>(s_query_head + i);
      score += key_val.x * query_val.x + key_val.y * query_val.y +
               key_val.z * query_val.z + key_val.w * query_val.w;
    }
    score *= scale;
    score_head[t] = score;
  }
  __syncthreads();
  softmax_gpu(score_head, pos + 1);
  __syncthreads();
  float* output_head = output + head * head_size;
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    float value = 0.0f;
    for (int t = 0; t <= pos; t++) {
      float* value_head = value_cache + layer_offset + t * kv_dim + head_offset;
      value += score_head[t] * value_head[i];
    }
    output_head[i] = value;
  }
}

__global__ void multi_head_attention_kernel_gpu_pos(
    const int32_t* pos_ptr, int32_t seq_len, float* query, float* score_ptr, float* output,
    float* key_cache, float* value_cache, int32_t kv_dim, int32_t kv_mul,
    int32_t head_num, int32_t head_size, int32_t layer_offset) {
  int head = blockIdx.x;
  if (head >= head_num) return;
  int32_t pos = *reinterpret_cast<const volatile int32_t*>(pos_ptr);
  extern __shared__ float s_query_head[];
  float scale = 1.f / sqrtf(float(head_size));
  float* query_head = query + head * head_size;
  for (int i = threadIdx.x; i < head_size; i += blockDim.x)
    s_query_head[i] = query_head[i];
  __syncthreads();
  float* score_head = score_ptr + head * seq_len;
  int head_offset = (head / kv_mul) * head_size;
  for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
    float* key_head = key_cache + layer_offset + t * kv_dim + head_offset;
    float score = 0.0f;
    for (int i = 0; i < head_size; i += 4) {
      float4 key_val = *reinterpret_cast<float4*>(key_head + i);
      float4 query_val = *reinterpret_cast<float4*>(s_query_head + i);
      score += key_val.x * query_val.x + key_val.y * query_val.y +
               key_val.z * query_val.z + key_val.w * query_val.w;
    }
    score *= scale;
    score_head[t] = score;
  }
  __syncthreads();
  softmax_gpu(score_head, pos + 1);
  __syncthreads();
  float* output_head = output + head * head_size;
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    float value = 0.0f;
    for (int t = 0; t <= pos; t++) {
      float* value_head = value_cache + layer_offset + t * kv_dim + head_offset;
      value += score_head[t] * value_head[i];
    }
    output_head[i] = value;
  }
}

__global__ void batched_multi_head_attention_kernel(
    int32_t start_pos, int32_t input_seq_len, int32_t max_seq_len, float* query,
    float* score_ptr, float* output, float* key_cache, float* value_cache,
    int32_t dim, int32_t kv_dim, int32_t kv_mul, int32_t head_num, int32_t head_size,
    int32_t layer_offset) {
  int head = blockIdx.x;
  int seq_idx = blockIdx.y;
  if (head >= head_num || seq_idx >= input_seq_len) return;
  extern __shared__ float s_query_head[];
  float scale = 1.f / sqrtf(float(head_size));
  int cur_pos = start_pos + seq_idx;
  float* query_head = query + seq_idx * dim + head * head_size;
  for (int i = threadIdx.x; i < head_size; i += blockDim.x)
    s_query_head[i] = query_head[i];
  __syncthreads();
  float* score_head = score_ptr + seq_idx * head_num * max_seq_len + head * max_seq_len;
  int head_offset = (head / kv_mul) * head_size;
  for (int t = threadIdx.x; t <= cur_pos; t += blockDim.x) {
    float* key_head = key_cache + layer_offset + t * kv_dim + head_offset;
    float score = 0.0f;
    for (int i = 0; i < head_size; i += 4) {
      float4 key_val = *reinterpret_cast<float4*>(key_head + i);
      float4 query_val = *reinterpret_cast<float4*>(s_query_head + i);
      score += key_val.x * query_val.x + key_val.y * query_val.y +
               key_val.z * query_val.z + key_val.w * query_val.w;
    }
    score *= scale;
    score_head[t] = score;
  }
  __syncthreads();
  softmax_gpu_causal(score_head, cur_pos + 1, cur_pos, cur_pos);
  __syncthreads();
  float* output_head = output + seq_idx * dim + head * head_size;
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    float value = 0.0f;
    for (int t = 0; t <= cur_pos; t++) {
      float* value_head = value_cache + layer_offset + t * kv_dim + head_offset;
      value += score_head[t] * value_head[i];
    }
    output_head[i] = value;
  }
}

}  // namespace original

// ============================================================
// Optimized kernels - warp shuffle reduction + __ldg
// ============================================================
namespace optimized {

constexpr static int thread_num = 256;
constexpr static int WARP_SIZE = 32;
constexpr static int NUM_WARPS = thread_num / WARP_SIZE;

__device__ __forceinline__ float warp_reduce_max(float val) {
  #pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
  return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
  #pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
  return val;
}

__device__ void softmax_gpu(float* __restrict__ x, int size) {
  int tid = threadIdx.x;
  int step = blockDim.x;
  int warp_id = tid >> 5;
  int lane_id = tid & 31;
  __shared__ float s_warp[NUM_WARPS];
  __shared__ float s_val;

  float max_val = tid < size ? x[tid] : -FLT_MAX;
  for (int i = tid + step; i < size; i += step)
    max_val = fmaxf(max_val, x[i]);
  max_val = warp_reduce_max(max_val);
  if (lane_id == 0) s_warp[warp_id] = max_val;
  __syncthreads();
  if (warp_id == 0) {
    max_val = (lane_id < NUM_WARPS) ? s_warp[lane_id] : -FLT_MAX;
    max_val = warp_reduce_max(max_val);
    if (lane_id == 0) s_val = max_val;
  }
  __syncthreads();
  max_val = s_val;

  float sum = 0.0f;
  for (int i = tid; i < size; i += step) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  sum = warp_reduce_sum(sum);
  if (lane_id == 0) s_warp[warp_id] = sum;
  __syncthreads();
  if (warp_id == 0) {
    sum = (lane_id < NUM_WARPS) ? s_warp[lane_id] : 0.0f;
    sum = warp_reduce_sum(sum);
    if (lane_id == 0) s_val = sum;
  }
  __syncthreads();
  sum = s_val;
  for (int i = tid; i < size; i += step) x[i] /= sum;
}

__device__ void softmax_gpu_causal(float* __restrict__ x, int size, int cur_pos, int total_pos) {
  int tid = threadIdx.x;
  int step = blockDim.x;
  int warp_id = tid >> 5;
  int lane_id = tid & 31;
  __shared__ float s_warp[NUM_WARPS];
  __shared__ float s_val;

  float max_val = tid <= cur_pos ? x[tid] : -FLT_MAX;
  for (int i = tid + step; i <= cur_pos; i += step)
    max_val = fmaxf(max_val, x[i]);
  max_val = warp_reduce_max(max_val);
  if (lane_id == 0) s_warp[warp_id] = max_val;
  __syncthreads();
  if (warp_id == 0) {
    max_val = (lane_id < NUM_WARPS) ? s_warp[lane_id] : -FLT_MAX;
    max_val = warp_reduce_max(max_val);
    if (lane_id == 0) s_val = max_val;
  }
  __syncthreads();
  max_val = s_val;

  float sum = 0.0f;
  for (int i = tid; i <= cur_pos; i += step) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  for (int i = tid; i <= total_pos; i += step) {
    if (i > cur_pos) x[i] = 0.0f;
  }
  sum = warp_reduce_sum(sum);
  if (lane_id == 0) s_warp[warp_id] = sum;
  __syncthreads();
  if (warp_id == 0) {
    sum = (lane_id < NUM_WARPS) ? s_warp[lane_id] : 0.0f;
    sum = warp_reduce_sum(sum);
    if (lane_id == 0) s_val = sum;
  }
  __syncthreads();
  sum = s_val;
  for (int i = tid; i <= cur_pos; i += step) x[i] /= sum;
}

__global__ void multi_head_attention_kernel(
    int32_t pos, int32_t seq_len,
    float* __restrict__ query, float* __restrict__ score_ptr,
    float* __restrict__ output, float* __restrict__ key_cache,
    float* __restrict__ value_cache, int32_t kv_dim, int32_t kv_mul,
    int32_t head_num, int32_t head_size, int32_t layer_offset) {
  int head = blockIdx.x;
  if (head >= head_num) return;
  extern __shared__ float s_query_head[];
  float scale = 1.f / sqrtf(float(head_size));
  float* query_head = query + head * head_size;
  for (int i = threadIdx.x; i < head_size; i += blockDim.x)
    s_query_head[i] = query_head[i];
  __syncthreads();
  float* score_head = score_ptr + head * seq_len;
  int head_offset = (head / kv_mul) * head_size;
  const float* key_base = key_cache + layer_offset + head_offset;
  for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
    const float* key_head = key_base + t * kv_dim;
    float score = 0.0f;
    #pragma unroll 4
    for (int i = 0; i < head_size; i += 4) {
      float4 key_val = __ldg(reinterpret_cast<const float4*>(key_head + i));
      float4 query_val = *reinterpret_cast<const float4*>(s_query_head + i);
      score += key_val.x * query_val.x + key_val.y * query_val.y +
               key_val.z * query_val.z + key_val.w * query_val.w;
    }
    score *= scale;
    score_head[t] = score;
  }
  __syncthreads();
  softmax_gpu(score_head, pos + 1);
  __syncthreads();
  float* output_head = output + head * head_size;
  const float* value_base = value_cache + layer_offset + head_offset;
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    float value = 0.0f;
    for (int t = 0; t <= pos; t++) {
      value += score_head[t] * __ldg(value_base + t * kv_dim + i);
    }
    output_head[i] = value;
  }
}

__global__ void multi_head_attention_kernel_gpu_pos(
    const int32_t* __restrict__ pos_ptr, int32_t seq_len,
    float* __restrict__ query, float* __restrict__ score_ptr,
    float* __restrict__ output, float* __restrict__ key_cache,
    float* __restrict__ value_cache, int32_t kv_dim, int32_t kv_mul,
    int32_t head_num, int32_t head_size, int32_t layer_offset) {
  int head = blockIdx.x;
  if (head >= head_num) return;
  int32_t pos = *reinterpret_cast<const volatile int32_t*>(pos_ptr);
  extern __shared__ float s_query_head[];
  float scale = 1.f / sqrtf(float(head_size));
  float* query_head = query + head * head_size;
  for (int i = threadIdx.x; i < head_size; i += blockDim.x)
    s_query_head[i] = query_head[i];
  __syncthreads();
  float* score_head = score_ptr + head * seq_len;
  int head_offset = (head / kv_mul) * head_size;
  const float* key_base = key_cache + layer_offset + head_offset;
  for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
    const float* key_head = key_base + t * kv_dim;
    float score = 0.0f;
    #pragma unroll 4
    for (int i = 0; i < head_size; i += 4) {
      float4 key_val = __ldg(reinterpret_cast<const float4*>(key_head + i));
      float4 query_val = *reinterpret_cast<const float4*>(s_query_head + i);
      score += key_val.x * query_val.x + key_val.y * query_val.y +
               key_val.z * query_val.z + key_val.w * query_val.w;
    }
    score *= scale;
    score_head[t] = score;
  }
  __syncthreads();
  softmax_gpu(score_head, pos + 1);
  __syncthreads();
  float* output_head = output + head * head_size;
  const float* value_base = value_cache + layer_offset + head_offset;
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    float value = 0.0f;
    for (int t = 0; t <= pos; t++) {
      value += score_head[t] * __ldg(value_base + t * kv_dim + i);
    }
    output_head[i] = value;
  }
}

__global__ void batched_multi_head_attention_kernel(
    int32_t start_pos, int32_t input_seq_len, int32_t max_seq_len,
    float* __restrict__ query, float* __restrict__ score_ptr,
    float* __restrict__ output, float* __restrict__ key_cache,
    float* __restrict__ value_cache, int32_t dim, int32_t kv_dim, int32_t kv_mul,
    int32_t head_num, int32_t head_size, int32_t layer_offset) {
  int head = blockIdx.x;
  int seq_idx = blockIdx.y;
  if (head >= head_num || seq_idx >= input_seq_len) return;
  extern __shared__ float s_query_head[];
  float scale = 1.f / sqrtf(float(head_size));
  int cur_pos = start_pos + seq_idx;
  float* query_head = query + seq_idx * dim + head * head_size;
  for (int i = threadIdx.x; i < head_size; i += blockDim.x)
    s_query_head[i] = query_head[i];
  __syncthreads();
  float* score_head = score_ptr + seq_idx * head_num * max_seq_len + head * max_seq_len;
  int head_offset = (head / kv_mul) * head_size;
  const float* key_base = key_cache + layer_offset + head_offset;
  for (int t = threadIdx.x; t <= cur_pos; t += blockDim.x) {
    const float* key_head = key_base + t * kv_dim;
    float score = 0.0f;
    #pragma unroll 4
    for (int i = 0; i < head_size; i += 4) {
      float4 key_val = __ldg(reinterpret_cast<const float4*>(key_head + i));
      float4 query_val = *reinterpret_cast<const float4*>(s_query_head + i);
      score += key_val.x * query_val.x + key_val.y * query_val.y +
               key_val.z * query_val.z + key_val.w * query_val.w;
    }
    score *= scale;
    score_head[t] = score;
  }
  __syncthreads();
  softmax_gpu_causal(score_head, cur_pos + 1, cur_pos, cur_pos);
  __syncthreads();
  float* output_head = output + seq_idx * dim + head * head_size;
  const float* value_base = value_cache + layer_offset + head_offset;
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    float value = 0.0f;
    for (int t = 0; t <= cur_pos; t++) {
      value += score_head[t] * __ldg(value_base + t * kv_dim + i);
    }
    output_head[i] = value;
  }
}

}  // namespace optimized

// ============================================================
// Helper functions
// ============================================================

void init_random(float* data, int size) {
  for (int i = 0; i < size; i++) {
    data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
  }
}

float max_diff(float* a, float* b, int size) {
  float md = 0.0f;
  for (int i = 0; i < size; i++) {
    float d = fabsf(a[i] - b[i]);
    if (d > md) md = d;
  }
  return md;
}

float relative_error(float* a, float* b, int size) {
  float max_rel = 0.0f;
  for (int i = 0; i < size; i++) {
    float denom = fmaxf(fabsf(a[i]), fabsf(b[i]));
    if (denom > 1e-8f) {
      float rel = fabsf(a[i] - b[i]) / denom;
      if (rel > max_rel) max_rel = rel;
    }
  }
  return max_rel;
}

// ============================================================
// Benchmark runner
// ============================================================

struct BenchConfig {
  const char* name;
  int head_num;
  int head_size;
  int kv_mul;
  int seq_len;
  int pos;
  int input_seq_len;  // for prefill
};

void benchmark_decode(const BenchConfig& cfg, int warmup = 5, int repeat = 20) {
  printf("\n=== %s (Decode) ===\n", cfg.name);
  printf("heads=%d, head_size=%d, kv_mul=%d, seq_len=%d, pos=%d\n",
         cfg.head_num, cfg.head_size, cfg.kv_mul, cfg.seq_len, cfg.pos);

  int dim = cfg.head_num * cfg.head_size;
  int kv_heads = cfg.head_num / cfg.kv_mul;
  int kv_dim = kv_heads * cfg.head_size;
  int layer_offset = 0;

  // Allocate host memory
  float* h_query = (float*)malloc(dim * sizeof(float));
  float* h_key_cache = (float*)malloc(cfg.seq_len * kv_dim * sizeof(float));
  float* h_val_cache = (float*)malloc(cfg.seq_len * kv_dim * sizeof(float));
  float* h_out_orig = (float*)malloc(dim * sizeof(float));
  float* h_out_opt = (float*)malloc(dim * sizeof(float));

  srand(42);
  init_random(h_query, dim);
  init_random(h_key_cache, cfg.seq_len * kv_dim);
  init_random(h_val_cache, cfg.seq_len * kv_dim);

  // Allocate device memory
  float *d_query, *d_key, *d_val, *d_score_orig, *d_score_opt, *d_out_orig, *d_out_opt;
  int32_t *d_pos;
  cudaMalloc(&d_query, dim * sizeof(float));
  cudaMalloc(&d_key, cfg.seq_len * kv_dim * sizeof(float));
  cudaMalloc(&d_val, cfg.seq_len * kv_dim * sizeof(float));
  cudaMalloc(&d_score_orig, cfg.head_num * cfg.seq_len * sizeof(float));
  cudaMalloc(&d_score_opt, cfg.head_num * cfg.seq_len * sizeof(float));
  cudaMalloc(&d_out_orig, dim * sizeof(float));
  cudaMalloc(&d_out_opt, dim * sizeof(float));
  cudaMalloc(&d_pos, sizeof(int32_t));

  cudaMemcpy(d_query, h_query, dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_key, h_key_cache, cfg.seq_len * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_val, h_val_cache, cfg.seq_len * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_pos, &cfg.pos, sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemset(d_score_orig, 0, cfg.head_num * cfg.seq_len * sizeof(float));
  cudaMemset(d_score_opt, 0, cfg.head_num * cfg.seq_len * sizeof(float));

  int smem = cfg.head_size * sizeof(float);

  // Correctness check
  original::multi_head_attention_kernel<<<cfg.head_num, 256, smem>>>(
      cfg.pos, cfg.seq_len, d_query, d_score_orig, d_out_orig, d_key, d_val,
      kv_dim, cfg.kv_mul, cfg.head_num, cfg.head_size, layer_offset);
  cudaDeviceSynchronize();

  optimized::multi_head_attention_kernel<<<cfg.head_num, 256, smem>>>(
      cfg.pos, cfg.seq_len, d_query, d_score_opt, d_out_opt, d_key, d_val,
      kv_dim, cfg.kv_mul, cfg.head_num, cfg.head_size, layer_offset);
  cudaDeviceSynchronize();

  cudaMemcpy(h_out_orig, d_out_orig, dim * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_out_opt, d_out_opt, dim * sizeof(float), cudaMemcpyDeviceToHost);

  float md = max_diff(h_out_orig, h_out_opt, dim);
  float re = relative_error(h_out_orig, h_out_opt, dim);
  printf("Correctness: max_diff=%.2e, max_rel_error=%.2e %s\n", md, re,
         (re < 1e-4f) ? "PASS" : "FAIL");

  // GPU-pos correctness
  cudaMemset(d_score_opt, 0, cfg.head_num * cfg.seq_len * sizeof(float));
  optimized::multi_head_attention_kernel_gpu_pos<<<cfg.head_num, 256, smem>>>(
      d_pos, cfg.seq_len, d_query, d_score_opt, d_out_opt, d_key, d_val,
      kv_dim, cfg.kv_mul, cfg.head_num, cfg.head_size, layer_offset);
  cudaDeviceSynchronize();
  cudaMemcpy(h_out_opt, d_out_opt, dim * sizeof(float), cudaMemcpyDeviceToHost);
  md = max_diff(h_out_orig, h_out_opt, dim);
  re = relative_error(h_out_orig, h_out_opt, dim);
  printf("GPU-pos:     max_diff=%.2e, max_rel_error=%.2e %s\n", md, re,
         (re < 1e-4f) ? "PASS" : "FAIL");

  // Timing - original
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (int i = 0; i < warmup; i++) {
    cudaMemset(d_score_orig, 0, cfg.head_num * cfg.seq_len * sizeof(float));
    original::multi_head_attention_kernel<<<cfg.head_num, 256, smem>>>(
        cfg.pos, cfg.seq_len, d_query, d_score_orig, d_out_orig, d_key, d_val,
        kv_dim, cfg.kv_mul, cfg.head_num, cfg.head_size, layer_offset);
  }
  cudaDeviceSynchronize();

  cudaEventRecord(start);
  for (int i = 0; i < repeat; i++) {
    cudaMemset(d_score_orig, 0, cfg.head_num * cfg.seq_len * sizeof(float));
    original::multi_head_attention_kernel<<<cfg.head_num, 256, smem>>>(
        cfg.pos, cfg.seq_len, d_query, d_score_orig, d_out_orig, d_key, d_val,
        kv_dim, cfg.kv_mul, cfg.head_num, cfg.head_size, layer_offset);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float orig_ms;
  cudaEventElapsedTime(&orig_ms, start, stop);
  orig_ms /= repeat;

  // Timing - optimized
  for (int i = 0; i < warmup; i++) {
    cudaMemset(d_score_opt, 0, cfg.head_num * cfg.seq_len * sizeof(float));
    optimized::multi_head_attention_kernel<<<cfg.head_num, 256, smem>>>(
        cfg.pos, cfg.seq_len, d_query, d_score_opt, d_out_opt, d_key, d_val,
        kv_dim, cfg.kv_mul, cfg.head_num, cfg.head_size, layer_offset);
  }
  cudaDeviceSynchronize();

  cudaEventRecord(start);
  for (int i = 0; i < repeat; i++) {
    cudaMemset(d_score_opt, 0, cfg.head_num * cfg.seq_len * sizeof(float));
    optimized::multi_head_attention_kernel<<<cfg.head_num, 256, smem>>>(
        cfg.pos, cfg.seq_len, d_query, d_score_opt, d_out_opt, d_key, d_val,
        kv_dim, cfg.kv_mul, cfg.head_num, cfg.head_size, layer_offset);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float opt_ms;
  cudaEventElapsedTime(&opt_ms, start, stop);
  opt_ms /= repeat;

  // Timing - optimized gpu_pos
  for (int i = 0; i < warmup; i++) {
    cudaMemset(d_score_opt, 0, cfg.head_num * cfg.seq_len * sizeof(float));
    optimized::multi_head_attention_kernel_gpu_pos<<<cfg.head_num, 256, smem>>>(
        d_pos, cfg.seq_len, d_query, d_score_opt, d_out_opt, d_key, d_val,
        kv_dim, cfg.kv_mul, cfg.head_num, cfg.head_size, layer_offset);
  }
  cudaDeviceSynchronize();

  cudaEventRecord(start);
  for (int i = 0; i < repeat; i++) {
    cudaMemset(d_score_opt, 0, cfg.head_num * cfg.seq_len * sizeof(float));
    optimized::multi_head_attention_kernel_gpu_pos<<<cfg.head_num, 256, smem>>>(
        d_pos, cfg.seq_len, d_query, d_score_opt, d_out_opt, d_key, d_val,
        kv_dim, cfg.kv_mul, cfg.head_num, cfg.head_size, layer_offset);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float gpu_pos_ms;
  cudaEventElapsedTime(&gpu_pos_ms, start, stop);
  gpu_pos_ms /= repeat;

  printf("Original decode:       %.3f ms\n", orig_ms);
  printf("Optimized decode:      %.3f ms (%.1f%% speedup)\n", opt_ms,
         (orig_ms - opt_ms) / orig_ms * 100);
  printf("Optimized gpu_pos:     %.3f ms (%.1f%% speedup)\n", gpu_pos_ms,
         (orig_ms - gpu_pos_ms) / orig_ms * 100);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  free(h_query); free(h_key_cache); free(h_val_cache);
  free(h_out_orig); free(h_out_opt);
  cudaFree(d_query); cudaFree(d_key); cudaFree(d_val);
  cudaFree(d_score_orig); cudaFree(d_score_opt);
  cudaFree(d_out_orig); cudaFree(d_out_opt); cudaFree(d_pos);
}

void benchmark_prefill(const BenchConfig& cfg, int warmup = 5, int repeat = 20) {
  printf("\n=== %s (Prefill) ===\n", cfg.name);
  printf("heads=%d, head_size=%d, kv_mul=%d, max_seq_len=%d, input_seq_len=%d\n",
         cfg.head_num, cfg.head_size, cfg.kv_mul, cfg.seq_len, cfg.input_seq_len);

  int dim = cfg.head_num * cfg.head_size;
  int kv_heads = cfg.head_num / cfg.kv_mul;
  int kv_dim = kv_heads * cfg.head_size;
  int layer_offset = 0;
  int start_pos = 0;

  float* h_query = (float*)malloc(cfg.input_seq_len * dim * sizeof(float));
  float* h_key_cache = (float*)malloc(cfg.seq_len * kv_dim * sizeof(float));
  float* h_val_cache = (float*)malloc(cfg.seq_len * kv_dim * sizeof(float));
  float* h_out_orig = (float*)malloc(cfg.input_seq_len * dim * sizeof(float));
  float* h_out_opt = (float*)malloc(cfg.input_seq_len * dim * sizeof(float));

  srand(42);
  init_random(h_query, cfg.input_seq_len * dim);
  init_random(h_key_cache, cfg.seq_len * kv_dim);
  init_random(h_val_cache, cfg.seq_len * kv_dim);

  float *d_query, *d_key, *d_val, *d_score_orig, *d_score_opt, *d_out_orig, *d_out_opt;
  int score_size = cfg.input_seq_len * cfg.head_num * cfg.seq_len;
  cudaMalloc(&d_query, cfg.input_seq_len * dim * sizeof(float));
  cudaMalloc(&d_key, cfg.seq_len * kv_dim * sizeof(float));
  cudaMalloc(&d_val, cfg.seq_len * kv_dim * sizeof(float));
  cudaMalloc(&d_score_orig, score_size * sizeof(float));
  cudaMalloc(&d_score_opt, score_size * sizeof(float));
  cudaMalloc(&d_out_orig, cfg.input_seq_len * dim * sizeof(float));
  cudaMalloc(&d_out_opt, cfg.input_seq_len * dim * sizeof(float));

  cudaMemcpy(d_query, h_query, cfg.input_seq_len * dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_key, h_key_cache, cfg.seq_len * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_val, h_val_cache, cfg.seq_len * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(d_score_orig, 0, score_size * sizeof(float));
  cudaMemset(d_score_opt, 0, score_size * sizeof(float));

  int smem = cfg.head_size * sizeof(float);
  dim3 grid(cfg.head_num, cfg.input_seq_len);

  // Correctness
  original::batched_multi_head_attention_kernel<<<grid, 256, smem>>>(
      start_pos, cfg.input_seq_len, cfg.seq_len, d_query, d_score_orig, d_out_orig,
      d_key, d_val, dim, kv_dim, cfg.kv_mul, cfg.head_num, cfg.head_size, layer_offset);
  cudaDeviceSynchronize();

  optimized::batched_multi_head_attention_kernel<<<grid, 256, smem>>>(
      start_pos, cfg.input_seq_len, cfg.seq_len, d_query, d_score_opt, d_out_opt,
      d_key, d_val, dim, kv_dim, cfg.kv_mul, cfg.head_num, cfg.head_size, layer_offset);
  cudaDeviceSynchronize();

  int out_size = cfg.input_seq_len * dim;
  cudaMemcpy(h_out_orig, d_out_orig, out_size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_out_opt, d_out_opt, out_size * sizeof(float), cudaMemcpyDeviceToHost);

  float md = max_diff(h_out_orig, h_out_opt, out_size);
  float re = relative_error(h_out_orig, h_out_opt, out_size);
  printf("Correctness: max_diff=%.2e, max_rel_error=%.2e %s\n", md, re,
         (re < 1e-4f) ? "PASS" : "FAIL");

  // Timing
  cudaEvent_t start_e, stop_e;
  cudaEventCreate(&start_e);
  cudaEventCreate(&stop_e);

  for (int i = 0; i < warmup; i++) {
    cudaMemset(d_score_orig, 0, score_size * sizeof(float));
    original::batched_multi_head_attention_kernel<<<grid, 256, smem>>>(
        start_pos, cfg.input_seq_len, cfg.seq_len, d_query, d_score_orig, d_out_orig,
        d_key, d_val, dim, kv_dim, cfg.kv_mul, cfg.head_num, cfg.head_size, layer_offset);
  }
  cudaDeviceSynchronize();
  cudaEventRecord(start_e);
  for (int i = 0; i < repeat; i++) {
    cudaMemset(d_score_orig, 0, score_size * sizeof(float));
    original::batched_multi_head_attention_kernel<<<grid, 256, smem>>>(
        start_pos, cfg.input_seq_len, cfg.seq_len, d_query, d_score_orig, d_out_orig,
        d_key, d_val, dim, kv_dim, cfg.kv_mul, cfg.head_num, cfg.head_size, layer_offset);
  }
  cudaEventRecord(stop_e);
  cudaEventSynchronize(stop_e);
  float orig_ms;
  cudaEventElapsedTime(&orig_ms, start_e, stop_e);
  orig_ms /= repeat;

  for (int i = 0; i < warmup; i++) {
    cudaMemset(d_score_opt, 0, score_size * sizeof(float));
    optimized::batched_multi_head_attention_kernel<<<grid, 256, smem>>>(
        start_pos, cfg.input_seq_len, cfg.seq_len, d_query, d_score_opt, d_out_opt,
        d_key, d_val, dim, kv_dim, cfg.kv_mul, cfg.head_num, cfg.head_size, layer_offset);
  }
  cudaDeviceSynchronize();
  cudaEventRecord(start_e);
  for (int i = 0; i < repeat; i++) {
    cudaMemset(d_score_opt, 0, score_size * sizeof(float));
    optimized::batched_multi_head_attention_kernel<<<grid, 256, smem>>>(
        start_pos, cfg.input_seq_len, cfg.seq_len, d_query, d_score_opt, d_out_opt,
        d_key, d_val, dim, kv_dim, cfg.kv_mul, cfg.head_num, cfg.head_size, layer_offset);
  }
  cudaEventRecord(stop_e);
  cudaEventSynchronize(stop_e);
  float opt_ms;
  cudaEventElapsedTime(&opt_ms, start_e, stop_e);
  opt_ms /= repeat;

  printf("Original prefill:      %.3f ms\n", orig_ms);
  printf("Optimized prefill:     %.3f ms (%.1f%% speedup)\n", opt_ms,
         (orig_ms - opt_ms) / orig_ms * 100);

  cudaEventDestroy(start_e);
  cudaEventDestroy(stop_e);
  free(h_query); free(h_key_cache); free(h_val_cache);
  free(h_out_orig); free(h_out_opt);
  cudaFree(d_query); cudaFree(d_key); cudaFree(d_val);
  cudaFree(d_score_orig); cudaFree(d_score_opt);
  cudaFree(d_out_orig); cudaFree(d_out_opt);
}

int main() {
  printf("MHA Kernel Benchmark: Original (cub) vs Optimized (warp shuffle + __ldg)\n");
  printf("==========================================================================\n");

  // Qwen3-8B / Qwen2.5-7B typical configuration
  // Qwen3-8B: head_num=32, head_size=128, kv_heads=8, kv_mul=4
  // Qwen2.5-7B: head_num=28, head_size=128, kv_heads=4, kv_mul=7

  // Decode benchmarks at different sequence lengths
  BenchConfig decode_configs[] = {
    {"Qwen3-8B pos=100",  32, 128, 4, 32768, 100, 0},
    {"Qwen3-8B pos=500",  32, 128, 4, 32768, 500, 0},
    {"Qwen3-8B pos=1000", 32, 128, 4, 32768, 1000, 0},
    {"Qwen3-8B pos=2000", 32, 128, 4, 32768, 2000, 0},
    {"Qwen2.5-7B pos=100",  28, 128, 7, 32768, 100, 0},
    {"Qwen2.5-7B pos=500",  28, 128, 7, 32768, 500, 0},
    {"Qwen2.5-7B pos=1000", 28, 128, 7, 32768, 1000, 0},
    {"Qwen2.5-7B pos=2000", 28, 128, 7, 32768, 2000, 0},
  };

  for (auto& cfg : decode_configs) {
    benchmark_decode(cfg);
  }

  // Prefill benchmarks
  BenchConfig prefill_configs[] = {
    {"Qwen3-8B prefill=32",  32, 128, 4, 32768, 32768, 32},
    {"Qwen3-8B prefill=128", 32, 128, 4, 32768, 32768, 128},
    {"Qwen2.5-7B prefill=32",  28, 128, 7, 32768, 32768, 32},
    {"Qwen2.5-7B prefill=128", 28, 128, 7, 32768, 32768, 128},
  };

  for (auto& cfg : prefill_configs) {
    benchmark_prefill(cfg);
  }

  printf("\n=== DONE ===\n");
  return 0;
}
