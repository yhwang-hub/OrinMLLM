#include "../kernels_interface.h"
#include "argmax_kernel.cuh"
#include "tensor/tensor.h"
namespace kernel {
__forceinline__ __device__ void warp_reduce_argmax(float& val, size_t& ptr) {
  float tmp_val;
  size_t tmp_ptr;
  unsigned int mask = __ballot_sync(0xFFFFFFFF, true);
  for (unsigned int k = (warpSize >> 1); k > 0; k >>= 1) {
    tmp_val = __shfl_down_sync(mask, val, k, warpSize);
    tmp_ptr = __shfl_down_sync(mask, ptr, k, warpSize);
    if (ptr == SIZE_MAX || tmp_ptr == SIZE_MAX) continue;
    if (tmp_val > val) {
      val = tmp_val;
      ptr = tmp_ptr;
    } else if (tmp_val == val && tmp_ptr < ptr) {
      ptr = tmp_ptr;
    }
  }
}

__forceinline__ __device__ void block_reduce_argmax(float& val, size_t& ptr, float* shared_value,
                                                    size_t* shared_ptr) {
  int lane_id = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;

  warp_reduce_argmax(val, ptr);

  __syncthreads();
  if (lane_id == 0) {
    shared_value[warp_id] = val;
    shared_ptr[warp_id] = ptr;
  }

  __syncthreads();
  if (threadIdx.x < blockDim.x / warpSize) {
    val = shared_value[lane_id];
    ptr = shared_ptr[lane_id];
  } else {
    val = 0;
    ptr = SIZE_MAX;
  }

  if (warp_id == 0) {
    warp_reduce_argmax(val, ptr);
  }
}

__global__ void argmax_kernel_fp32(const float* input_ptr, size_t size, size_t* output_idx) {
  __shared__ size_t shared_max_ptr[32];
  __shared__ float shared_max_value[32];
  uint32_t tid = threadIdx.x;
  if (tid >= size) {
    return;
  }

  size_t max_index = threadIdx.x;
  float max_value = input_ptr[max_index];
  for (size_t i = tid; i < size; i += blockDim.x) {
    if (input_ptr[i] > max_value) {
      max_index = i;
      max_value = input_ptr[i];
    }
  }

  block_reduce_argmax(max_value, max_index, shared_max_value, shared_max_ptr);
  __syncthreads();
  if (threadIdx.x == 0) {
    *output_idx = max_index;
  }
}

size_t argmax_kernel_cu(const float* input_ptr, size_t size, void* stream) {
  std::shared_ptr<base::DeviceAllocator> alloc_cu =
      base::CUDADeviceAllocatorFactory::get_instance();
  size_t* index = static_cast<size_t*>(alloc_cu->allocate(sizeof(size_t)));
  size_t output_index = 0;
  if (!stream) {
    argmax_kernel_fp32<<<1, 512>>>(input_ptr, size, index);
    cudaMemcpy(&output_index, index, sizeof(size_t), cudaMemcpyDeviceToHost);
  } else {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    argmax_kernel_fp32<<<1, 512, 0, stream_>>>(input_ptr, size, index);
    cudaMemcpyAsync(&output_index, index, sizeof(size_t), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
  }
  return output_index;
}

// Optimized version using pre-allocated buffers
// This avoids memory allocation overhead and enables true async D2H transfer with pinned memory
void argmax_kernel_cu_prealloc(const float* input_ptr, size_t size, 
                                size_t* output_gpu, size_t* output_pinned,
                                void* stream) {
  cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
  
  if (stream_) {
    argmax_kernel_fp32<<<1, 512, 0, stream_>>>(input_ptr, size, output_gpu);
    // Async copy to pinned memory - truly asynchronous since output_pinned is page-locked
    cudaMemcpyAsync(output_pinned, output_gpu, sizeof(size_t), cudaMemcpyDeviceToHost, stream_);
  } else {
    argmax_kernel_fp32<<<1, 512>>>(input_ptr, size, output_gpu);
    cudaMemcpy(output_pinned, output_gpu, sizeof(size_t), cudaMemcpyDeviceToHost);
  }
}

// Batched argmax kernel: one block per row
__global__ void batched_argmax_kernel_fp32(const float* input_ptr, int32_t* output_idx,
                                            int32_t batch_size, int32_t vocab_size) {
  __shared__ size_t shared_max_ptr[32];
  __shared__ float shared_max_value[32];
  
  int batch_idx = blockIdx.x;
  if (batch_idx >= batch_size) return;
  
  const float* row_input = input_ptr + batch_idx * vocab_size;
  uint32_t tid = threadIdx.x;
  
  size_t max_index = tid < vocab_size ? tid : 0;
  float max_value = tid < vocab_size ? row_input[tid] : -1e30f;
  
  for (int i = tid; i < vocab_size; i += blockDim.x) {
    if (row_input[i] > max_value) {
      max_index = i;
      max_value = row_input[i];
    }
  }
  
  block_reduce_argmax(max_value, max_index, shared_max_value, shared_max_ptr);
  __syncthreads();
  
  if (threadIdx.x == 0) {
    output_idx[batch_idx] = static_cast<int32_t>(max_index);
  }
}

void batched_argmax_kernel_cu(const float* input_ptr, int32_t* output_gpu,
                               int32_t batch_size, int32_t vocab_size, void* stream) {
  cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
  
  // One block per row, 512 threads per block
  dim3 grid(batch_size);
  dim3 block(512);
  
  if (stream_) {
    batched_argmax_kernel_fp32<<<grid, block, 0, stream_>>>(input_ptr, output_gpu, batch_size, vocab_size);
  } else {
    batched_argmax_kernel_fp32<<<grid, block>>>(input_ptr, output_gpu, batch_size, vocab_size);
  }
}

// Single row argmax on GPU - result stays on GPU
void single_argmax_kernel_cu(const float* input_ptr, int32_t* output_gpu,
                              int32_t vocab_size, void* stream) {
  // Reuse batched kernel with batch_size=1
  batched_argmax_kernel_cu(input_ptr, output_gpu, 1, vocab_size, stream);
}

// Argmax + D2T mapping kernel: find max index and apply offset
__global__ void argmax_d2t_kernel_fp32(const float* input_ptr, const int32_t* d2t_gpu,
                                         int32_t* output, int32_t vocab_size) {
  __shared__ size_t shared_max_ptr[32];
  __shared__ float shared_max_value[32];
  
  uint32_t tid = threadIdx.x;
  
  size_t max_index = tid < vocab_size ? tid : 0;
  float max_value = tid < vocab_size ? input_ptr[tid] : -1e30f;
  
  for (int i = tid; i < vocab_size; i += blockDim.x) {
    if (input_ptr[i] > max_value) {
      max_index = i;
      max_value = input_ptr[i];
    }
  }
  
  block_reduce_argmax(max_value, max_index, shared_max_value, shared_max_ptr);
  __syncthreads();
  
  if (threadIdx.x == 0) {
    // Apply D2T mapping: target_token = draft_idx + d2t[draft_idx]
    int32_t draft_idx = static_cast<int32_t>(max_index);
    *output = draft_idx + d2t_gpu[draft_idx];
  }
}

void argmax_d2t_kernel_cu(const float* input_ptr, const int32_t* d2t_gpu,
                           int32_t* output_gpu, int32_t vocab_size, void* stream) {
  cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
  
  if (stream_) {
    argmax_d2t_kernel_fp32<<<1, 512, 0, stream_>>>(input_ptr, d2t_gpu, output_gpu, vocab_size);
  } else {
    argmax_d2t_kernel_fp32<<<1, 512>>>(input_ptr, d2t_gpu, output_gpu, vocab_size);
  }
}

// Top-K + D2T mapping kernel: find top-k indices and apply offset
// output: [k] - top-k target tokens (with D2T mapping applied)
__global__ void topk_d2t_kernel_fp32(const float* input_ptr, const int32_t* d2t_gpu,
                                      int32_t* output, int32_t vocab_size, int32_t k) {
  // Use insertion sort to maintain top-k (k is small, typically 4-8)
  __shared__ float top_values[16];  // Max k=16
  __shared__ int32_t top_indices[16];
  
  // Initialize with minimum values
  if (threadIdx.x < k) {
    top_values[threadIdx.x] = -1e30f;
    top_indices[threadIdx.x] = 0;
  }
  __syncthreads();
  
  // Each thread processes a subset of vocab
  for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
    float val = input_ptr[i];
    
    // Check if this value should be in top-k (only thread 0 does insertion)
    // Use atomicMax pattern - find if val > min(top_values)
    // For simplicity, use atomic operations
    for (int j = 0; j < k; j++) {
      float old_val = top_values[j];
      if (val > old_val) {
        // Try to swap
        float expected = old_val;
        float replaced = atomicCAS(reinterpret_cast<unsigned int*>(&top_values[j]),
                                    __float_as_uint(expected), __float_as_uint(val));
        if (__uint_as_float(replaced) == expected) {
          // Successfully replaced, now update index
          atomicExch(&top_indices[j], i);
          val = old_val;  // Continue trying to insert old value
        }
      }
    }
  }
  __syncthreads();
  
  // Sort top-k by value (bubble sort, k is small)
  if (threadIdx.x == 0) {
    for (int i = 0; i < k - 1; i++) {
      for (int j = i + 1; j < k; j++) {
        if (top_values[j] > top_values[i]) {
          float tmp_v = top_values[i];
          int32_t tmp_i = top_indices[i];
          top_values[i] = top_values[j];
          top_indices[i] = top_indices[j];
          top_values[j] = tmp_v;
          top_indices[j] = tmp_i;
        }
      }
    }
    
    // Apply D2T mapping and write output
    for (int i = 0; i < k; i++) {
      int32_t draft_idx = top_indices[i];
      output[i] = draft_idx + d2t_gpu[draft_idx];
    }
  }
}

void topk_d2t_kernel_cu(const float* input_ptr, const int32_t* d2t_gpu,
                         int32_t* output_gpu, int32_t vocab_size, int32_t k, void* stream) {
  cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
  
  if (stream_) {
    topk_d2t_kernel_fp32<<<1, 256, 0, stream_>>>(input_ptr, d2t_gpu, output_gpu, vocab_size, k);
  } else {
    topk_d2t_kernel_fp32<<<1, 256>>>(input_ptr, d2t_gpu, output_gpu, vocab_size, k);
  }
}
}  // namespace kernel