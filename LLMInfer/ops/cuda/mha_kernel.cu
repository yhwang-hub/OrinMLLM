#include <base/cuda_config.h>
#include <tensor/tensor.h>
#include <cfloat>
#include "mha_kernel.cuh"
namespace kernel {
constexpr static int thread_num = 256;
constexpr static int WARP_SIZE = 32;
constexpr static int NUM_WARPS = thread_num / WARP_SIZE;

// =============================================================================
// Warp-level reduction primitives (replace cub::BlockReduce for lower overhead)
// Uses __shfl_xor_sync for intra-warp communication (no shared memory needed)
// =============================================================================

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

// =============================================================================
// Optimized softmax using warp shuffle reduction
// Reduces shared memory usage vs cub::BlockReduce (36 bytes vs ~1KB)
// Same reduction tree structure as cub WARP_REDUCTIONS for bit-exact results
// =============================================================================

__device__ void softmax_gpu(float* __restrict__ x, int size) {
  int tid = threadIdx.x;
  int step = blockDim.x;
  int warp_id = tid >> 5;
  int lane_id = tid & 31;

  __shared__ float s_warp[NUM_WARPS];
  __shared__ float s_val;

  // Find max value (for numerical stability)
  float max_val = tid < size ? x[tid] : -FLT_MAX;
  for (int i = tid + step; i < size; i += step)
    max_val = fmaxf(max_val, x[i]);

  // Two-level reduction: intra-warp shuffle, then inter-warp via shared memory
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

  // Compute exp and sum
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

  // Normalize
  for (int i = tid; i < size; i += step) {
    x[i] /= sum;
  }
}

// =============================================================================
// Causal mask softmax for prefill
// Optimized: removed redundant -FLT_MAX mask writes (the max/sum loops already
// limit to [0..cur_pos], so masking is implicit)
// =============================================================================

__device__ void softmax_gpu_causal(float* __restrict__ x, int size, int cur_pos, int total_pos) {
  int tid = threadIdx.x;
  int step = blockDim.x;
  int warp_id = tid >> 5;
  int lane_id = tid & 31;

  __shared__ float s_warp[NUM_WARPS];
  __shared__ float s_val;

  // Find max over valid positions [0..cur_pos] - no need for explicit mask
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

  // Compute exp and sum over valid positions only
  float sum = 0.0f;
  for (int i = tid; i <= cur_pos; i += step) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }

  // Zero out masked positions (i > cur_pos)
  for (int i = tid; i <= total_pos; i += step) {
    if (i > cur_pos) {
      x[i] = 0.0f;
    }
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

  // Normalize valid positions
  for (int i = tid; i <= cur_pos; i += step) {
    x[i] /= sum;
  }
}


// =============================================================================
// Decode MHA Kernel (standard, pos as value parameter)
// Optimizations:
//   1. __ldg() for K/V cache reads → L1 read-only cache path
//   2. __restrict__ on all pointers → better compiler alias analysis
//   3. Pre-computed key/value base addresses → reduced per-iteration arithmetic
//   4. #pragma unroll for Q·K inner loop → instruction-level parallelism
//   5. fmaxf() in softmax → hardware intrinsic
// =============================================================================

__global__ void multi_head_attention_kernel(
    int32_t pos, int32_t seq_len,
    float* __restrict__ query,
    float* __restrict__ score_ptr,
    float* __restrict__ output,
    float* __restrict__ key_cache,
    float* __restrict__ value_cache,
    int32_t kv_dim, int32_t kv_mul,
    int32_t head_num, int32_t head_size,
    int32_t layer_offset) {
  int head = blockIdx.x;
  if (head >= head_num) {
    return;
  }

  extern __shared__ float s_query_head[];
  float scale = 1.f / sqrtf(float(head_size));
  float* query_head = query + head * head_size;

  // Load query to shared memory
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    s_query_head[i] = query_head[i];
  }
  __syncthreads();

  float* score_head = score_ptr + head * seq_len;
  int head_offset = (head / kv_mul) * head_size;
  // Pre-compute base addresses to reduce per-iteration arithmetic
  const float* key_base = key_cache + layer_offset + head_offset;

  // Compute attention scores with __ldg for read-only K cache access
  for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
    const float* key_head = key_base + t * kv_dim;
    float score = 0.0f;
    #pragma unroll 4
    for (int i = 0; i < head_size; i += 4) {
      // __ldg routes K cache reads through L1 read-only cache (texture path)
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

  // Weighted sum of values with __ldg for read-only V cache access
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    float value = 0.0f;
    for (int t = 0; t <= pos; t++) {
      // __ldg for V cache: reduces L1 cache pollution, improves streaming reads
      value += score_head[t] * __ldg(value_base + t * kv_dim + i);
    }
    output_head[i] = value;
  }
}

void mha_kernel_cu(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                   int32_t kv_dim, int32_t kv_mul, int32_t head_size, const tensor::Tensor& mha_out,
                   const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
                   const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
                   base::DeviceType device_type, CudaConfig* config) {
  UNUSED(device_type);
  int32_t layer_offset = layer_index * seq_len * kv_dim;
  float* query = const_cast<float*>(query_tensor.ptr<float>());
  float* score = const_cast<float*>(score_tensor.ptr<float>());
  float* output = const_cast<float*>(mha_out.ptr<float>());

  float* key_cache = const_cast<float*>(key_cache_tensor.ptr<float>());
  float* value_cache = const_cast<float*>(value_cache_tensor.ptr<float>());

  cudaStream_t stream = config->stream;
  multi_head_attention_kernel<<<head_num, thread_num, head_size * sizeof(float), stream>>>(
      pos, seq_len, query, score, output, key_cache, value_cache, kv_dim, kv_mul, head_num,
      head_size, layer_offset);
}

// =============================================================================
// Decode MHA Kernel (GPU pos for CUDA Graph compatibility)
// Same optimizations as standard kernel, plus volatile pos read from device mem
// =============================================================================

__global__ void multi_head_attention_kernel_gpu_pos(
    const int32_t* __restrict__ pos_ptr, int32_t seq_len,
    float* __restrict__ query,
    float* __restrict__ score_ptr,
    float* __restrict__ output,
    float* __restrict__ key_cache,
    float* __restrict__ value_cache,
    int32_t kv_dim, int32_t kv_mul,
    int32_t head_num, int32_t head_size,
    int32_t layer_offset) {
  int head = blockIdx.x;
  if (head >= head_num) {
    return;
  }

  // Read position from GPU memory using volatile to prevent optimization
  int32_t pos = *reinterpret_cast<const volatile int32_t*>(pos_ptr);

  extern __shared__ float s_query_head[];
  float scale = 1.f / sqrtf(float(head_size));
  float* query_head = query + head * head_size;

  // Load query to shared memory
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    s_query_head[i] = query_head[i];
  }
  __syncthreads();

  float* score_head = score_ptr + head * seq_len;
  int head_offset = (head / kv_mul) * head_size;
  const float* key_base = key_cache + layer_offset + head_offset;

  // Compute attention scores with __ldg for read-only K cache
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

  // Weighted sum of values with __ldg for read-only V cache
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    float value = 0.0f;
    for (int t = 0; t <= pos; t++) {
      value += score_head[t] * __ldg(value_base + t * kv_dim + i);
    }
    output_head[i] = value;
  }
}

void mha_kernel_cu_gpu_pos(const int32_t* pos_ptr, int32_t head_num, int32_t layer_index, 
                           int32_t seq_len, int32_t kv_dim, int32_t kv_mul, int32_t head_size, 
                           const tensor::Tensor& mha_out, const tensor::Tensor& query_tensor, 
                           const tensor::Tensor& score_tensor, const tensor::Tensor& key_cache_tensor, 
                           const tensor::Tensor& value_cache_tensor, base::DeviceType device_type, 
                           CudaConfig* config) {
  UNUSED(device_type);
  int32_t layer_offset = layer_index * seq_len * kv_dim;
  float* query = const_cast<float*>(query_tensor.ptr<float>());
  float* score = const_cast<float*>(score_tensor.ptr<float>());
  float* output = const_cast<float*>(mha_out.ptr<float>());

  float* key_cache = const_cast<float*>(key_cache_tensor.ptr<float>());
  float* value_cache = const_cast<float*>(value_cache_tensor.ptr<float>());

  cudaStream_t stream = config->stream;
  multi_head_attention_kernel_gpu_pos<<<head_num, thread_num, head_size * sizeof(float), stream>>>(
      pos_ptr, seq_len, query, score, output, key_cache, value_cache, kv_dim, kv_mul, head_num,
      head_size, layer_offset);
}

// =============================================================================
// Batched MHA Kernel for Prefill Phase
// query: [seq_len, dim], key_cache/value_cache: [layer_num, max_seq_len, kv_dim]
// output: [seq_len, dim]
// Same optimizations: __ldg, __restrict__, pre-computed addresses, unroll
// =============================================================================

__global__ void batched_multi_head_attention_kernel(
    int32_t start_pos, int32_t input_seq_len,
    int32_t max_seq_len,
    float* __restrict__ query,
    float* __restrict__ score_ptr,
    float* __restrict__ output,
    float* __restrict__ key_cache,
    float* __restrict__ value_cache,
    int32_t dim, int32_t kv_dim, int32_t kv_mul,
    int32_t head_num, int32_t head_size,
    int32_t layer_offset) {
  // blockIdx.x: head index
  // blockIdx.y: sequence position in current input
  int head = blockIdx.x;
  int seq_idx = blockIdx.y;

  if (head >= head_num || seq_idx >= input_seq_len) {
    return;
  }

  extern __shared__ float s_query_head[];
  float scale = 1.f / sqrtf(float(head_size));

  // Current position in the full sequence
  int cur_pos = start_pos + seq_idx;

  // Query for current sequence position and head
  float* query_head = query + seq_idx * dim + head * head_size;

  // Load query to shared memory
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    s_query_head[i] = query_head[i];
  }
  __syncthreads();

  // Score storage for this head and sequence position
  float* score_head = score_ptr + seq_idx * head_num * max_seq_len + head * max_seq_len;

  int head_offset = (head / kv_mul) * head_size;
  const float* key_base = key_cache + layer_offset + head_offset;

  // Compute attention scores with __ldg for read-only K cache (causal mask)
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

  // Apply causal softmax
  softmax_gpu_causal(score_head, cur_pos + 1, cur_pos, cur_pos);
  __syncthreads();

  // Output for current sequence position and head
  float* output_head = output + seq_idx * dim + head * head_size;
  const float* value_base = value_cache + layer_offset + head_offset;

  // Weighted sum of values with __ldg for read-only V cache
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    float value = 0.0f;
    for (int t = 0; t <= cur_pos; t++) {
      value += score_head[t] * __ldg(value_base + t * kv_dim + i);
    }
    output_head[i] = value;
  }
}

void batched_mha_kernel_cu(int32_t start_pos, int32_t seq_len, int32_t head_num, int32_t layer_index,
                           int32_t max_seq_len, int32_t dim, int32_t kv_dim, int32_t kv_mul, 
                           int32_t head_size, const tensor::Tensor& mha_out,
                           const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
                           const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
                           base::DeviceType device_type, CudaConfig* config) {
  UNUSED(device_type);
  int32_t layer_offset = layer_index * max_seq_len * kv_dim;
  float* query = const_cast<float*>(query_tensor.ptr<float>());
  float* score = const_cast<float*>(score_tensor.ptr<float>());
  float* output = const_cast<float*>(mha_out.ptr<float>());

  float* key_cache = const_cast<float*>(key_cache_tensor.ptr<float>());
  float* value_cache = const_cast<float*>(value_cache_tensor.ptr<float>());

  cudaStream_t stream = config->stream;
  dim3 grid(head_num, seq_len);
  batched_multi_head_attention_kernel<<<grid, thread_num, head_size * sizeof(float), stream>>>(
      start_pos, seq_len, max_seq_len, query, score, output, key_cache, value_cache, 
      dim, kv_dim, kv_mul, head_num, head_size, layer_offset);
}

}  // namespace kernel