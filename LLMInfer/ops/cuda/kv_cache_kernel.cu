#include "kv_cache_kernel.cuh"
#include <cuda_fp16.h>
#include <stdio.h>

namespace kernel {

/**
 * @brief CUDA kernel to copy data from temp buffer to KV cache (FP32)
 * 
 * All pointer parameters are fixed addresses, only content changes.
 * Position is read from GPU memory to allow CUDA Graph reuse.
 */
__global__ void copy_to_kv_cache_cu(float* kv_cache, const float* src, const int32_t* pos,
                                     int32_t kv_dim, int32_t layer_idx, int32_t seq_len) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < kv_dim) {
    // Read position from GPU memory
    // Use volatile to prevent compiler optimization
    int32_t position = *reinterpret_cast<const volatile int32_t*>(pos);
    
    // Calculate offset: layer_idx * seq_len * kv_dim + position * kv_dim + idx
    int32_t offset = layer_idx * seq_len * kv_dim + position * kv_dim + idx;
    kv_cache[offset] = src[idx];
  }
}

/**
 * @brief CUDA kernel to copy data from temp buffer to KV cache (FP16)
 */
__global__ void copy_to_kv_cache_fp16_cu(half* kv_cache, const half* src, const int32_t* pos,
                                          int32_t kv_dim, int32_t layer_idx, int32_t seq_len) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < kv_dim) {
    int32_t position = *reinterpret_cast<const volatile int32_t*>(pos);
    int64_t offset = static_cast<int64_t>(layer_idx) * seq_len * kv_dim + 
                     static_cast<int64_t>(position) * kv_dim + idx;
    kv_cache[offset] = src[idx];
  }
}

void copy_to_kv_cache_kernel(float* kv_cache, const float* src, const int32_t* pos,
                             int32_t kv_dim, int32_t layer_idx, int32_t seq_len,
                             cudaStream_t stream) {
  const int block_size = 256;
  const int grid_size = (kv_dim + block_size - 1) / block_size;
  copy_to_kv_cache_cu<<<grid_size, block_size, 0, stream>>>(
      kv_cache, src, pos, kv_dim, layer_idx, seq_len);
}

void copy_to_kv_cache_kernel_fp16(half* kv_cache, const half* src, const int32_t* pos,
                                   int32_t kv_dim, int32_t layer_idx, int32_t seq_len,
                                   cudaStream_t stream) {
  const int block_size = 256;
  const int grid_size = (kv_dim + block_size - 1) / block_size;
  copy_to_kv_cache_fp16_cu<<<grid_size, block_size, 0, stream>>>(
      kv_cache, src, pos, kv_dim, layer_idx, seq_len);
}

}  // namespace kernel
