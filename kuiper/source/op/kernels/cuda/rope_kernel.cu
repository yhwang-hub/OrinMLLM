#include "rope_kernel.cuh"
#include <cuda_fp16.h>
namespace kernel {

#if defined (LLAMA3_SUPPORT)
__global__ void rope_kernel_cu_fp32(int pos, int dim, int kv_dim, int head_size,
                                    const float* input_q, const float* input_k,
                                    const float* sin_cache, const float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  int num_heads = dim / head_size;
  int head_pair_count = head_size / 2;
  int total_pairs = num_heads * head_pair_count;
  if (idx > total_pairs) {
    return;
  }

  int head_idx = idx / head_pair_count;
  int head_dim = idx % head_pair_count;

  int i = head_idx * head_size;
  int v0_idx = i + head_dim;
  int v1_idx = i + head_dim + head_size / 2;

  float fci = sin_cache[pos * head_size + head_dim * 2];
  float fcr = cos_cache[pos * head_size + head_dim * 2];

  int rotn = i < kv_dim ? 2 : 1;

  for (int v = 0; v < rotn; v++) {
    float* vec = const_cast<float*>(v == 0 ? input_q : input_k);  // the vector to rotate (query or key)
    float v0 = vec[v0_idx];
    float v1 = vec[v1_idx];
    vec[v0_idx] = fcr * v0 - fci * v1;
    vec[v1_idx] = fcr * v1 + fci * v0;
  }
}

__global__ void sin_cos_calc(int head_size, int max_seq_len, float* sin_cache, float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int head_dim = idx % head_size;
  for (int pos = 0; pos < max_seq_len; ++pos) {
    float freq = 1.0f / pow(500000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
    float val = static_cast<float>(pos) * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    *(sin_cache + pos * head_size + head_dim) = fci;
    *(cos_cache + pos * head_size + head_dim) = fcr;
  }
}
#elif defined (QWEN2_SUPPORT) || defined (QWEN3_SUPPORT)
__global__ void rope_kernel_cu_fp32(int pos, int dim, int kv_dim, int head_size,
                                    const float* input_q, const float* input_k,
                                    const float* sin_cache, const float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  int num_heads = dim / head_size;
  int head_pair_count = head_size / 2;
  int total_pairs = num_heads * head_pair_count;
  if (idx >= total_pairs) {  // Fixed: was > which caused out-of-bounds access
    return;
  }

  int head_idx = idx / head_pair_count;
  int head_dim = idx % head_pair_count;

  int i = head_idx * head_size;
  int v0_idx = i + head_dim;
  int v1_idx = i + head_dim + head_size / 2;

  float fci = sin_cache[pos * head_size + head_dim * 2];
  float fcr = cos_cache[pos * head_size + head_dim * 2];

  int rotn = i < kv_dim ? 2 : 1;

  for (int v = 0; v < rotn; v++) {
    float* vec = const_cast<float*>(v == 0 ? input_q : input_k);  // the vector to rotate (query or key)
    float v0 = vec[v0_idx];
    float v1 = vec[v1_idx];
    vec[v0_idx] = fcr * v0 - fci * v1;
    vec[v1_idx] = fcr * v1 + fci * v0;
  }
}

__global__ void sin_cos_calc(int head_size, int max_seq_len, float* sin_cache, float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int head_dim = idx % head_size;
  for (int pos = 0; pos < max_seq_len; ++pos) {
    float freq = 1.0f / pow(1000000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
    float val = static_cast<float>(pos) * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    *(sin_cache + pos * head_size + head_dim) = fci;
    *(cos_cache + pos * head_size + head_dim) = fcr;
  }
}
#else
__device__ void rope_calc(float fcr, float fci, float* vec, int32_t idx) {
  float2* vec_ptr = reinterpret_cast<float2*>(vec + idx);
  float2 vec_value = *vec_ptr;
  *vec_ptr =
      make_float2(vec_value.x * fcr - vec_value.y * fci, vec_value.x * fci + vec_value.y * fcr);
}

__global__ void rope_kernel_cu_fp32(int pos, int dim, int kv_dim, int head_size,
                                    const float* input_q, const float* input_k,
                                    const float* sin_cache, const float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  idx = idx * 2;
  if (idx >= dim) {
    return;
  }

  int head_dim = idx % head_size;
  float fci = *(sin_cache + pos * head_size + head_dim);
  float fcr = *(cos_cache + pos * head_size + head_dim);

  rope_calc(fcr, fci, const_cast<float*>(input_q), idx);
  if (idx >= kv_dim) {
    return;
  }
  rope_calc(fcr, fci, const_cast<float*>(input_k), idx);
}

__global__ void sin_cos_calc(int head_size, int max_seq_len, float* sin_cache, float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int head_dim = idx % head_size;
  for (int pos = 0; pos < max_seq_len; ++pos) {
    // Qwen3-VL uses rope_theta = 5000000 (from config.json)
    float freq = 1.0f / pow(5000000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
    float val = static_cast<float>(pos) * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    *(sin_cache + pos * head_size + head_dim) = fci;
    *(cos_cache + pos * head_size + head_dim) = fcr;
  }
}
#endif

// Forward declarations for pure FP16 RoPE kernels
#if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
__global__ void rope_kernel_cu_fp16_impl(int pos, int dim, int kv_dim, int head_size,
                                          half* input_q, half* input_k,
                                          const float* sin_cache, const float* cos_cache);

__global__ void batched_rope_kernel_cu_fp16_impl(int start_pos, int seq_len, int dim, int kv_dim, int head_size,
                                                  half* input_q, half* input_k,
                                                  const float* sin_cache, const float* cos_cache);
#endif

void sin_cos_cache_calc_cu(int head_size, int max_seq_len, const tensor::Tensor& sin_cache,
                           const tensor::Tensor& cos_cache, cudaStream_t stream) {
  CHECK_EQ(sin_cache.is_empty(), false);
  CHECK_EQ(cos_cache.is_empty(), false);
  int threads = head_size;
  if (stream) {
    sin_cos_calc<<<1, threads, 0, stream>>>(head_size, max_seq_len,
                                            const_cast<float*>(sin_cache.ptr<float>()),
                                            const_cast<float*>(cos_cache.ptr<float>()));
  } else {
    sin_cos_calc<<<1, threads>>>(head_size, max_seq_len, const_cast<float*>(sin_cache.ptr<float>()),
                                 const_cast<float*>(cos_cache.ptr<float>()));
  }
}

void rope_kernel_cu(int32_t dim, int32_t kv_dim, int32_t head_size, const tensor::Tensor& input_q,
                    const tensor::Tensor& input_k, const tensor::Tensor& input_pos,
                    const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
                    void* stream) {
  const int32_t pos = *input_pos.ptr<int32_t>(0);
  int threads = 128;
  int blocks = (dim + threads - 1) / threads;
  
#if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
  // Check if this is pure FP16 path
  if (input_q.data_type() == base::DataType::kDataTypeFp16 &&
      input_k.data_type() == base::DataType::kDataTypeFp16) {
    half* q_ptr = reinterpret_cast<half*>(const_cast<uint16_t*>(input_q.ptr<uint16_t>()));
    half* k_ptr = reinterpret_cast<half*>(const_cast<uint16_t*>(input_k.ptr<uint16_t>()));
    
    if (stream) {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      rope_kernel_cu_fp16_impl<<<blocks, threads, 0, stream_>>>(
          pos, dim, kv_dim, head_size, q_ptr, k_ptr,
          sin_cache.ptr<float>(), cos_cache.ptr<float>());
    } else {
      rope_kernel_cu_fp16_impl<<<blocks, threads>>>(
          pos, dim, kv_dim, head_size, q_ptr, k_ptr,
          sin_cache.ptr<float>(), cos_cache.ptr<float>());
    }
    return;
  }
#endif
  
  // FP32 path
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    rope_kernel_cu_fp32<<<blocks, threads, 0, stream_>>>(
        pos, dim, kv_dim, head_size, input_q.ptr<float>(), input_k.ptr<float>(),
        sin_cache.ptr<float>(), cos_cache.ptr<float>());
  } else {
    rope_kernel_cu_fp32<<<blocks, threads>>>(pos, dim, kv_dim, head_size, input_q.ptr<float>(),
                                             input_k.ptr<float>(), sin_cache.ptr<float>(),
                                             cos_cache.ptr<float>());
  }
}

#if defined (QWEN2_SUPPORT) || defined (QWEN3_SUPPORT)

// RoPE kernel that reads position from GPU memory (for CUDA Graph optimization)
__global__ void rope_kernel_cu_fp32_gpu_pos(const int32_t* pos_ptr, int dim, int kv_dim, int head_size,
                                            const float* input_q, const float* input_k,
                                            const float* sin_cache, const float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  int num_heads = dim / head_size;
  int head_pair_count = head_size / 2;
  int total_pairs = num_heads * head_pair_count;
  if (idx >= total_pairs) {  // Fixed: was > which caused out-of-bounds access
    return;
  }
  
  // Read position from GPU memory using volatile to prevent optimization
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

void rope_kernel_cu_gpu_pos(int32_t dim, int32_t kv_dim, int32_t head_size, 
                            const tensor::Tensor& input_q, const tensor::Tensor& input_k, 
                            const int32_t* pos_ptr, const tensor::Tensor& sin_cache, 
                            const tensor::Tensor& cos_cache, void* stream) {
  int threads = 128;
  int blocks = (dim + threads - 1) / threads;
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    rope_kernel_cu_fp32_gpu_pos<<<blocks, threads, 0, stream_>>>(
        pos_ptr, dim, kv_dim, head_size, input_q.ptr<float>(), input_k.ptr<float>(),
        sin_cache.ptr<float>(), cos_cache.ptr<float>());
  } else {
    rope_kernel_cu_fp32_gpu_pos<<<blocks, threads>>>(
        pos_ptr, dim, kv_dim, head_size, input_q.ptr<float>(), input_k.ptr<float>(),
        sin_cache.ptr<float>(), cos_cache.ptr<float>());
  }
}

// FP16 RoPE kernel that reads position from GPU memory (for CUDA Graph optimization)
__global__ void rope_kernel_cu_fp16_gpu_pos_impl(const int32_t* pos_ptr, int dim, int kv_dim, int head_size,
                                                  half* input_q, half* input_k,
                                                  const float* sin_cache, const float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  int num_heads = dim / head_size;
  int head_pair_count = head_size / 2;
  int total_pairs = num_heads * head_pair_count;
  if (idx >= total_pairs) {
    return;
  }
  
  // Read position from GPU memory using volatile to prevent optimization
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
    // Load FP16, compute in FP32, store back to FP16
    float v0 = __half2float(vec[v0_idx]);
    float v1 = __half2float(vec[v1_idx]);
    vec[v0_idx] = __float2half(fcr * v0 - fci * v1);
    vec[v1_idx] = __float2half(fcr * v1 + fci * v0);
  }
}

void rope_kernel_cu_fp16_gpu_pos(int32_t dim, int32_t kv_dim, int32_t head_size, 
                                  const tensor::Tensor& input_q, const tensor::Tensor& input_k, 
                                  const int32_t* pos_ptr, const tensor::Tensor& sin_cache, 
                                  const tensor::Tensor& cos_cache, void* stream) {
  int threads = 128;
  int blocks = (dim + threads - 1) / threads;
  half* q_ptr = reinterpret_cast<half*>(const_cast<uint16_t*>(input_q.ptr<uint16_t>()));
  half* k_ptr = reinterpret_cast<half*>(const_cast<uint16_t*>(input_k.ptr<uint16_t>()));
  
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    rope_kernel_cu_fp16_gpu_pos_impl<<<blocks, threads, 0, stream_>>>(
        pos_ptr, dim, kv_dim, head_size, q_ptr, k_ptr,
        sin_cache.ptr<float>(), cos_cache.ptr<float>());
  } else {
    rope_kernel_cu_fp16_gpu_pos_impl<<<blocks, threads>>>(
        pos_ptr, dim, kv_dim, head_size, q_ptr, k_ptr,
        sin_cache.ptr<float>(), cos_cache.ptr<float>());
  }
}

// Batched RoPE kernel for prefill phase
__global__ void batched_rope_kernel_cu_fp32(int start_pos, int seq_len, int dim, int kv_dim, int head_size,
                                            float* input_q, float* input_k,
                                            const float* sin_cache, const float* cos_cache) {
  // blockIdx.x: position in sequence
  // threadIdx.x + blockIdx.y * blockDim.x: head pair index
  int seq_idx = blockIdx.x;
  if (seq_idx >= seq_len) {
    return;
  }
  
  int pos = start_pos + seq_idx;
  int idx = threadIdx.x + blockDim.x * blockIdx.y;

  int num_heads = dim / head_size;
  int head_pair_count = head_size / 2;
  int total_pairs = num_heads * head_pair_count;
  if (idx >= total_pairs) {
    return;
  }

  int head_idx = idx / head_pair_count;
  int head_dim = idx % head_pair_count;

  int i = head_idx * head_size;
  int v0_idx = i + head_dim;
  int v1_idx = i + head_dim + head_size / 2;

  float fci = sin_cache[pos * head_size + head_dim * 2];
  float fcr = cos_cache[pos * head_size + head_dim * 2];

  int rotn = i < kv_dim ? 2 : 1;

  // Calculate offset for current sequence position
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

void batched_rope_kernel_cu(int32_t start_pos, int32_t seq_len, int32_t dim, int32_t kv_dim, 
                            int32_t head_size, const tensor::Tensor& input_q,
                            const tensor::Tensor& input_k, const tensor::Tensor& sin_cache,
                            const tensor::Tensor& cos_cache, void* stream) {
  int num_heads = dim / head_size;
  int head_pair_count = head_size / 2;
  int total_pairs = num_heads * head_pair_count;
  
  int threads = 128;
  int blocks_y = (total_pairs + threads - 1) / threads;
  dim3 grid(seq_len, blocks_y);
  
#if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
  // Check if this is pure FP16 path
  if (input_q.data_type() == base::DataType::kDataTypeFp16 &&
      input_k.data_type() == base::DataType::kDataTypeFp16) {
    half* q_ptr = reinterpret_cast<half*>(const_cast<uint16_t*>(input_q.ptr<uint16_t>()));
    half* k_ptr = reinterpret_cast<half*>(const_cast<uint16_t*>(input_k.ptr<uint16_t>()));
    
    if (stream) {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      batched_rope_kernel_cu_fp16_impl<<<grid, threads, 0, stream_>>>(
          start_pos, seq_len, dim, kv_dim, head_size, q_ptr, k_ptr,
          sin_cache.ptr<float>(), cos_cache.ptr<float>());
    } else {
      batched_rope_kernel_cu_fp16_impl<<<grid, threads>>>(
          start_pos, seq_len, dim, kv_dim, head_size, q_ptr, k_ptr,
          sin_cache.ptr<float>(), cos_cache.ptr<float>());
    }
    return;
  }
#endif
  
  // FP32 path
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    batched_rope_kernel_cu_fp32<<<grid, threads, 0, stream_>>>(
        start_pos, seq_len, dim, kv_dim, head_size,
        const_cast<float*>(input_q.ptr<float>()), const_cast<float*>(input_k.ptr<float>()),
        sin_cache.ptr<float>(), cos_cache.ptr<float>());
  } else {
    batched_rope_kernel_cu_fp32<<<grid, threads>>>(
        start_pos, seq_len, dim, kv_dim, head_size,
        const_cast<float*>(input_q.ptr<float>()), const_cast<float*>(input_k.ptr<float>()),
        sin_cache.ptr<float>(), cos_cache.ptr<float>());
  }
}

// ==================== Pure FP16 RoPE Implementation ====================

// Pure FP16 RoPE kernel for single position (decode phase)
// Note: sin/cos cache remains FP32 for precision, Q/K are FP16
__global__ void rope_kernel_cu_fp16_impl(int pos, int dim, int kv_dim, int head_size,
                                          half* input_q, half* input_k,
                                          const float* sin_cache, const float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  int num_heads = dim / head_size;
  int head_pair_count = head_size / 2;
  int total_pairs = num_heads * head_pair_count;
  if (idx >= total_pairs) {
    return;
  }

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
    // Load FP16, compute in FP32, store back to FP16
    float v0 = __half2float(vec[v0_idx]);
    float v1 = __half2float(vec[v1_idx]);
    vec[v0_idx] = __float2half(fcr * v0 - fci * v1);
    vec[v1_idx] = __float2half(fcr * v1 + fci * v0);
  }
}

// Batched pure FP16 RoPE kernel for prefill phase
__global__ void batched_rope_kernel_cu_fp16_impl(int start_pos, int seq_len, int dim, int kv_dim, int head_size,
                                                  half* input_q, half* input_k,
                                                  const float* sin_cache, const float* cos_cache) {
  int seq_idx = blockIdx.x;
  if (seq_idx >= seq_len) {
    return;
  }
  
  int pos = start_pos + seq_idx;
  int idx = threadIdx.x + blockDim.x * blockIdx.y;

  int num_heads = dim / head_size;
  int head_pair_count = head_size / 2;
  int total_pairs = num_heads * head_pair_count;
  if (idx >= total_pairs) {
    return;
  }

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

void rope_kernel_cu_pure_fp16(int32_t dim, int32_t kv_dim, int32_t head_size, 
                               const tensor::Tensor& input_q, const tensor::Tensor& input_k, 
                               int32_t pos, const tensor::Tensor& sin_cache, 
                               const tensor::Tensor& cos_cache, void* stream) {
  CHECK(input_q.data_type() == base::DataType::kDataTypeFp16);
  CHECK(input_k.data_type() == base::DataType::kDataTypeFp16);
  
  int threads = 128;
  int blocks = (dim + threads - 1) / threads;
  
  half* q_ptr = reinterpret_cast<half*>(const_cast<uint16_t*>(input_q.ptr<uint16_t>()));
  half* k_ptr = reinterpret_cast<half*>(const_cast<uint16_t*>(input_k.ptr<uint16_t>()));
  
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    rope_kernel_cu_fp16_impl<<<blocks, threads, 0, stream_>>>(
        pos, dim, kv_dim, head_size, q_ptr, k_ptr,
        sin_cache.ptr<float>(), cos_cache.ptr<float>());
  } else {
    rope_kernel_cu_fp16_impl<<<blocks, threads>>>(
        pos, dim, kv_dim, head_size, q_ptr, k_ptr,
        sin_cache.ptr<float>(), cos_cache.ptr<float>());
  }
}

void batched_rope_kernel_cu_pure_fp16(int32_t start_pos, int32_t seq_len, int32_t dim, int32_t kv_dim,
                                       int32_t head_size, const tensor::Tensor& input_q,
                                       const tensor::Tensor& input_k, const tensor::Tensor& sin_cache,
                                       const tensor::Tensor& cos_cache, void* stream) {
  CHECK(input_q.data_type() == base::DataType::kDataTypeFp16);
  CHECK(input_k.data_type() == base::DataType::kDataTypeFp16);
  
  int num_heads = dim / head_size;
  int head_pair_count = head_size / 2;
  int total_pairs = num_heads * head_pair_count;
  
  int threads = 128;
  int blocks_y = (total_pairs + threads - 1) / threads;
  dim3 grid(seq_len, blocks_y);
  
  half* q_ptr = reinterpret_cast<half*>(const_cast<uint16_t*>(input_q.ptr<uint16_t>()));
  half* k_ptr = reinterpret_cast<half*>(const_cast<uint16_t*>(input_k.ptr<uint16_t>()));
  
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    batched_rope_kernel_cu_fp16_impl<<<grid, threads, 0, stream_>>>(
        start_pos, seq_len, dim, kv_dim, head_size, q_ptr, k_ptr,
        sin_cache.ptr<float>(), cos_cache.ptr<float>());
  } else {
    batched_rope_kernel_cu_fp16_impl<<<grid, threads>>>(
        start_pos, seq_len, dim, kv_dim, head_size, q_ptr, k_ptr,
        sin_cache.ptr<float>(), cos_cache.ptr<float>());
  }
}

// ==================== M-RoPE (Multimodal RoPE) Implementation ====================
// M-RoPE uses 3D position encoding: (temporal, height, width)
// mrope_section = [24, 20, 20] for Qwen3-VL (head_size=128)
// After *2: [48, 40, 40] = 128 dimensions
// - Dimensions [0, 48): use temporal position (t)
// - Dimensions [48, 88): use height position (h)  
// - Dimensions [88, 128): use width position (w)
// For visual tokens: t=base, h=base+row, w=base+col
// For text tokens: t=h=w=sequential_position
//
// The rotation is applied in pairs: (x_0, x_1), (x_2, x_3), ...
// For each pair (x_i, x_{i+1}):
//   x_i' = x_i * cos - x_{i+1} * sin
//   x_{i+1}' = x_i * sin + x_{i+1} * cos
// This is the standard RoPE rotation.

/**
 * @brief M-RoPE kernel for single token (decode phase)
 * Processes all heads for one token with 3D position.
 * 
 * @param pos_t Temporal position
 * @param pos_h Height position  
 * @param pos_w Width position
 * @param dim Total Q dimension (head_num * head_size)
 * @param kv_dim Total K dimension (kv_head_num * head_size)
 * @param head_size Size of each attention head (128 for Qwen3-VL)
 * @param section0 First section size (24*2=48 dims for t-dimension)
 * @param section1 Second section size (20*2=40 dims for h-dimension)
 * @param section2 Third section size (20*2=40 dims for w-dimension)
 * @param input_q Query tensor [dim]
 * @param input_k Key tensor [kv_dim]
 * @param sin_cache Sin cache [max_seq_len, head_size]
 * @param cos_cache Cos cache [max_seq_len, head_size]
 */
__global__ void mrope_kernel_cu_fp16_impl(
    int pos_t, int pos_h, int pos_w,
    int dim, int kv_dim, int head_size,
    int section0, int section1, int section2,
    half* input_q, half* input_k,
    const float* sin_cache, const float* cos_cache) {
  
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  
  int num_heads = dim / head_size;
  int num_kv_heads = kv_dim / head_size;
  int half_head_size = head_size / 2;  // 64 for head_size=128
  int total_pairs = num_heads * half_head_size;
  
  if (idx >= total_pairs) {
    return;
  }
  
  int head_idx = idx / half_head_size;
  int pair_idx = idx % half_head_size;  // Which pair within the head (0-63)
  
  // Get the two dimension indices for this pair (interleaved half-split: (d, d+64))
  int i = head_idx * head_size;
  int d0 = pair_idx;                   // First dimension of pair (0-63)
  int d1 = pair_idx + half_head_size;  // Second dimension of pair (64-127)
  int v0_idx = i + d0;
  int v1_idx = i + d1;
  
  // Convert section pair counts to dimension thresholds
  // section0=24 pairs -> dims [0, 48) use T, dims [64, 112) also map to same freq range
  // section1=20 pairs -> dims [48, 88) use H
  // section2=20 pairs -> dims [88, 128) use W
  int dim_threshold0 = section0 * 2;  // 48
  int dim_threshold1 = dim_threshold0 + section1 * 2;  // 88
  
  // Determine position for d0 based on which section it falls in
  int pos0;
  if (d0 < dim_threshold0) {
    pos0 = pos_t;  // dims [0, 48)
  } else if (d0 < dim_threshold1) {
    pos0 = pos_h;  // dims [48, 88) - but d0 only goes to 63, so this won't happen
  } else {
    pos0 = pos_w;  // dims [88, 128) - but d0 only goes to 63, so this won't happen
  }
  
  // Determine position for d1 based on which section it falls in
  int pos1;
  if (d1 < dim_threshold0) {
    pos1 = pos_t;  // dims [0, 48) - but d1 starts at 64, so this won't happen
  } else if (d1 < dim_threshold1) {
    pos1 = pos_h;  // dims [64, 88) - d1 in [64, 88) means pair_idx in [0, 24)
  } else {
    pos1 = pos_w;  // dims [88, 128) - d1 in [88, 128) means pair_idx in [24, 64)
  }
  
  // For HuggingFace MRoPE, both dimensions in a pair use the SAME frequency
  // (due to cat((freqs, freqs), dim=-1)), but different positions
  // Frequency index is pair_idx * 2 (matching standard RoPE cache layout)
  int freq_idx = pair_idx * 2;
  
  // Look up sin/cos for each dimension using the frequency but different positions
  float sin0 = sin_cache[pos0 * head_size + freq_idx];
  float cos0 = cos_cache[pos0 * head_size + freq_idx];
  float sin1 = sin_cache[pos1 * head_size + freq_idx];
  float cos1 = cos_cache[pos1 * head_size + freq_idx];
  
  // Apply RoPE to Q
  // HuggingFace rotate_half: [x1, x2] -> [-x2, x1]
  // q_embed = q * cos + rotate_half(q) * sin
  // q_embed[d0] = q[d0] * cos0 + (-q[d1]) * sin0
  // q_embed[d1] = q[d1] * cos1 + q[d0] * sin1
  {
    float v0 = __half2float(input_q[v0_idx]);
    float v1 = __half2float(input_q[v1_idx]);
    input_q[v0_idx] = __float2half(v0 * cos0 - v1 * sin0);
    input_q[v1_idx] = __float2half(v1 * cos1 + v0 * sin1);
  }
  
  // Apply RoPE to K - with GQA, K has fewer heads
  int kv_mul = num_heads / num_kv_heads;
  int kv_head_idx = head_idx / kv_mul;
  
  // Only the first Q head in each group should update the K head
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

void mrope_kernel_cu_fp16(
    int32_t pos_t, int32_t pos_h, int32_t pos_w,
    int32_t dim, int32_t kv_dim, int32_t head_size,
    int32_t section0, int32_t section1, int32_t section2,
    const tensor::Tensor& input_q, const tensor::Tensor& input_k,
    const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
    void* stream) {
  
  CHECK(input_q.data_type() == base::DataType::kDataTypeFp16);
  CHECK(input_k.data_type() == base::DataType::kDataTypeFp16);
  
  int num_heads = dim / head_size;
  int half_head_size = head_size / 2;
  int total_pairs = num_heads * half_head_size;
  
  int threads = 128;
  int blocks = (total_pairs + threads - 1) / threads;
  
  half* q_ptr = reinterpret_cast<half*>(const_cast<uint16_t*>(input_q.ptr<uint16_t>()));
  half* k_ptr = reinterpret_cast<half*>(const_cast<uint16_t*>(input_k.ptr<uint16_t>()));
  
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    mrope_kernel_cu_fp16_impl<<<blocks, threads, 0, stream_>>>(
        pos_t, pos_h, pos_w, dim, kv_dim, head_size,
        section0, section1, section2,
        q_ptr, k_ptr,
        sin_cache.ptr<float>(), cos_cache.ptr<float>());
  } else {
    mrope_kernel_cu_fp16_impl<<<blocks, threads>>>(
        pos_t, pos_h, pos_w, dim, kv_dim, head_size,
        section0, section1, section2,
        q_ptr, k_ptr,
        sin_cache.ptr<float>(), cos_cache.ptr<float>());
  }
}

/**
 * @brief Batched M-RoPE kernel for multiple tokens (prefill phase)
 * Each token has its own 3D position (pos_t, pos_h, pos_w).
 */
__global__ void batched_mrope_kernel_cu_fp16_impl(
    int seq_len, int dim, int kv_dim, int head_size,
    int section0, int section1, int section2,
    const int32_t* pos_t_arr, const int32_t* pos_h_arr, const int32_t* pos_w_arr,
    half* input_q, half* input_k,
    const float* sin_cache, const float* cos_cache) {
  
  // Grid: (seq_len, blocks_for_pairs)
  int seq_idx = blockIdx.x;
  if (seq_idx >= seq_len) return;
  
  int idx = threadIdx.x + blockDim.x * blockIdx.y;
  
  int num_heads = dim / head_size;
  int num_kv_heads = kv_dim / head_size;
  int half_head_size = head_size / 2;
  int total_pairs = num_heads * half_head_size;
  
  if (idx >= total_pairs) return;
  
  // Get 3D positions for this token
  int pos_t = pos_t_arr[seq_idx];
  int pos_h = pos_h_arr[seq_idx];
  int pos_w = pos_w_arr[seq_idx];
  
  int head_idx = idx / half_head_size;
  int pair_idx = idx % half_head_size;
  
  int i = head_idx * head_size;
  int d0 = pair_idx;
  int d1 = pair_idx + half_head_size;
  
  // Dimension thresholds
  int dim_threshold0 = section0 * 2;  // 48
  int dim_threshold1 = dim_threshold0 + section1 * 2;  // 88
  
  // Position for first half (d0 in [0, 64))
  int pos0;
  if (d0 < dim_threshold0) {
    pos0 = pos_t;
  } else if (d0 < dim_threshold1) {
    pos0 = pos_h;
  } else {
    pos0 = pos_w;
  }
  
  // Position for second half (d1 in [64, 128))
  int pos1;
  if (d1 < dim_threshold0) {
    pos1 = pos_t;
  } else if (d1 < dim_threshold1) {
    pos1 = pos_h;
  } else {
    pos1 = pos_w;
  }
  
  int freq_idx = pair_idx * 2;
  
  float sin0 = sin_cache[pos0 * head_size + freq_idx];
  float cos0 = cos_cache[pos0 * head_size + freq_idx];
  float sin1 = sin_cache[pos1 * head_size + freq_idx];
  float cos1 = cos_cache[pos1 * head_size + freq_idx];
  
  int q_offset = seq_idx * dim;
  int k_offset = seq_idx * kv_dim;
  
  int v0_idx = i + d0;
  int v1_idx = i + d1;
  
  // Apply RoPE to Q
  {
    float v0 = __half2float(input_q[q_offset + v0_idx]);
    float v1 = __half2float(input_q[q_offset + v1_idx]);
    input_q[q_offset + v0_idx] = __float2half(v0 * cos0 - v1 * sin0);
    input_q[q_offset + v1_idx] = __float2half(v1 * cos1 + v0 * sin1);
  }
  
  // Apply RoPE to K - with GQA
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

void batched_mrope_kernel_cu_fp16(
    int32_t seq_len, int32_t dim, int32_t kv_dim, int32_t head_size,
    int32_t section0, int32_t section1, int32_t section2,
    const int32_t* pos_t_arr, const int32_t* pos_h_arr, const int32_t* pos_w_arr,
    const tensor::Tensor& input_q, const tensor::Tensor& input_k,
    const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
    void* stream) {
  
  CHECK(input_q.data_type() == base::DataType::kDataTypeFp16);
  CHECK(input_k.data_type() == base::DataType::kDataTypeFp16);
  
  int num_heads = dim / head_size;
  int half_head_size = head_size / 2;
  int total_pairs = num_heads * half_head_size;
  
  int threads = 128;
  int blocks_y = (total_pairs + threads - 1) / threads;
  dim3 grid(seq_len, blocks_y);
  
  half* q_ptr = reinterpret_cast<half*>(const_cast<uint16_t*>(input_q.ptr<uint16_t>()));
  half* k_ptr = reinterpret_cast<half*>(const_cast<uint16_t*>(input_k.ptr<uint16_t>()));
  
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    batched_mrope_kernel_cu_fp16_impl<<<grid, threads, 0, stream_>>>(
        seq_len, dim, kv_dim, head_size,
        section0, section1, section2,
        pos_t_arr, pos_h_arr, pos_w_arr,
        q_ptr, k_ptr,
        sin_cache.ptr<float>(), cos_cache.ptr<float>());
  } else {
    batched_mrope_kernel_cu_fp16_impl<<<grid, threads>>>(
        seq_len, dim, kv_dim, head_size,
        section0, section1, section2,
        pos_t_arr, pos_h_arr, pos_w_arr,
        q_ptr, k_ptr,
        sin_cache.ptr<float>(), cos_cache.ptr<float>());
  }
}

/**
 * @brief M-RoPE kernel with GPU-resident position for CUDA Graph decode phase
 * All 3 positions (t/h/w) use the same value for text tokens
 */
__global__ void mrope_kernel_cu_fp16_gpu_pos_impl(
    const int32_t* pos_ptr,  // GPU pointer to position
    int dim, int kv_dim, int head_size,
    int section0, int section1, int section2,
    half* input_q, half* input_k,
    const float* sin_cache, const float* cos_cache) {
  
  int pos = pos_ptr[0];  // Read position from GPU memory
  
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  
  int num_heads = dim / head_size;
  int num_kv_heads = kv_dim / head_size;
  int half_head_size = head_size / 2;
  int total_pairs = num_heads * half_head_size;
  
  if (idx >= total_pairs) {
    return;
  }
  
  int head_idx = idx / half_head_size;
  int pair_idx = idx % half_head_size;
  
  int i = head_idx * head_size;
  int d0 = pair_idx;
  int d1 = pair_idx + half_head_size;
  int v0_idx = i + d0;
  int v1_idx = i + d1;
  
  // For decode phase, all positions are the same text_pos
  int pos_t = pos;
  int pos_h = pos;
  int pos_w = pos;
  
  int dim_threshold0 = section0 * 2;
  int dim_threshold1 = dim_threshold0 + section1 * 2;
  
  int pos0;
  if (d0 < dim_threshold0) {
    pos0 = pos_t;
  } else if (d0 < dim_threshold1) {
    pos0 = pos_h;
  } else {
    pos0 = pos_w;
  }
  
  int pos1;
  if (d1 < dim_threshold0) {
    pos1 = pos_t;
  } else if (d1 < dim_threshold1) {
    pos1 = pos_h;
  } else {
    pos1 = pos_w;
  }
  
  int freq_idx = pair_idx * 2;
  
  float sin0 = sin_cache[pos0 * head_size + freq_idx];
  float cos0 = cos_cache[pos0 * head_size + freq_idx];
  float sin1 = sin_cache[pos1 * head_size + freq_idx];
  float cos1 = cos_cache[pos1 * head_size + freq_idx];
  
  // Apply RoPE to Q
  {
    float v0 = __half2float(input_q[v0_idx]);
    float v1 = __half2float(input_q[v1_idx]);
    input_q[v0_idx] = __float2half(v0 * cos0 - v1 * sin0);
    input_q[v1_idx] = __float2half(v1 * cos1 + v0 * sin1);
  }
  
  // Apply RoPE to K
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

void mrope_kernel_cu_fp16_gpu_pos(
    const int32_t* pos_gpu,
    int32_t dim, int32_t kv_dim, int32_t head_size,
    int32_t section0, int32_t section1, int32_t section2,
    const tensor::Tensor& input_q, const tensor::Tensor& input_k,
    const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
    void* stream) {
  
  CHECK(input_q.data_type() == base::DataType::kDataTypeFp16);
  CHECK(input_k.data_type() == base::DataType::kDataTypeFp16);
  
  int num_heads = dim / head_size;
  int half_head_size = head_size / 2;
  int total_pairs = num_heads * half_head_size;
  
  int threads = 128;
  int blocks = (total_pairs + threads - 1) / threads;
  
  half* q_ptr = reinterpret_cast<half*>(const_cast<uint16_t*>(input_q.ptr<uint16_t>()));
  half* k_ptr = reinterpret_cast<half*>(const_cast<uint16_t*>(input_k.ptr<uint16_t>()));
  
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    mrope_kernel_cu_fp16_gpu_pos_impl<<<blocks, threads, 0, stream_>>>(
        pos_gpu, dim, kv_dim, head_size,
        section0, section1, section2,
        q_ptr, k_ptr,
        sin_cache.ptr<float>(), cos_cache.ptr<float>());
  } else {
    mrope_kernel_cu_fp16_gpu_pos_impl<<<blocks, threads>>>(
        pos_gpu, dim, kv_dim, head_size,
        section0, section1, section2,
        q_ptr, k_ptr,
        sin_cache.ptr<float>(), cos_cache.ptr<float>());
  }
}

#endif
}  // namespace kernel