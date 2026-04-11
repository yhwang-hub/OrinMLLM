#include "../kernels_interface.h"
#include "argmax_kernel.cuh"
#include "tensor/tensor.h"
#include <float.h>
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
    cudaMemcpyAsync(output_pinned, output_gpu, sizeof(size_t), cudaMemcpyDeviceToHost, stream_);
  } else {
    argmax_kernel_fp32<<<1, 512>>>(input_ptr, size, output_gpu);
    cudaMemcpy(output_pinned, output_gpu, sizeof(size_t), cudaMemcpyDeviceToHost);
  }
}

// Batched FP16 argmax kernel: one block per row
// Input: [batch, row_size] FP16, Output: [batch] int32
__global__ void batched_argmax_fp16_kernel(
    const half* __restrict__ input,   // [batch, row_size]
    int32_t* __restrict__ output,     // [batch]
    int32_t row_size) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const half* row_ptr = input + (size_t)row * row_size;

  float best_val = -FLT_MAX;
  int32_t best_idx = 0;

  for (int i = tid; i < row_size; i += blockDim.x) {
    float v = __half2float(row_ptr[i]);
    if (v > best_val) { best_val = v; best_idx = i; }
  }

  // Warp-level reduction
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    float other_val = __shfl_down_sync(0xFFFFFFFF, best_val, offset);
    int32_t other_idx = __shfl_down_sync(0xFFFFFFFF, best_idx, offset);
    if (other_val > best_val) { best_val = other_val; best_idx = other_idx; }
  }

  // Block-level reduction via shared memory
  __shared__ float s_val[32];
  __shared__ int32_t s_idx[32];
  int lane = tid % warpSize;
  int wid = tid / warpSize;
  if (lane == 0) { s_val[wid] = best_val; s_idx[wid] = best_idx; }
  __syncthreads();

  if (tid < blockDim.x / warpSize) {
    best_val = s_val[tid]; best_idx = s_idx[tid];
    for (int offset = (blockDim.x / warpSize) / 2; offset > 0; offset >>= 1) {
      float other_val = __shfl_down_sync(0xFFFFFFFF, best_val, offset);
      int32_t other_idx = __shfl_down_sync(0xFFFFFFFF, best_idx, offset);
      if (other_val > best_val) { best_val = other_val; best_idx = other_idx; }
    }
    if (tid == 0) output[row] = best_idx;
  }
}

void batched_argmax_fp16_cu(const void* input, int32_t* output_gpu, int32_t* output_cpu,
                            int32_t batch, int32_t row_size, void* stream) {
  cudaStream_t s = stream ? static_cast<cudaStream_t>(stream) : nullptr;
  batched_argmax_fp16_kernel<<<batch, 256, 0, s>>>(
      static_cast<const half*>(input), output_gpu, row_size);
  cudaMemcpyAsync(output_cpu, output_gpu, batch * sizeof(int32_t),
                  cudaMemcpyDeviceToHost, s);
}

}  // namespace kernel