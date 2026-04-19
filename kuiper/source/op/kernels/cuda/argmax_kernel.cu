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

// ---------------------------------------------------------------------------
// 9.8 optimization: fast two-stage argmax over large vocab (e.g. 151936).
// Original single-block argmax_kernel_fp32 underutilizes the GPU (~253us/call
// with <<<1, 512>>> on Jetson Orin).  This version uses multi-block parallel
// reduction to a small partials array, then a final reduction block.
// ---------------------------------------------------------------------------
namespace {
constexpr int kArgmaxStage1Blocks = 32;
constexpr int kArgmaxStage1Threads = 256;

__global__ void argmax_stage1_fp32_kernel(const float* __restrict__ input,
                                          size_t size,
                                          float* __restrict__ partial_vals,
                                          size_t* __restrict__ partial_idxs) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const size_t gtid = static_cast<size_t>(bid) * blockDim.x + tid;
  const size_t gstride = static_cast<size_t>(gridDim.x) * blockDim.x;

  float best_val = -FLT_MAX;
  size_t best_idx = SIZE_MAX;
  for (size_t i = gtid; i < size; i += gstride) {
    float v = input[i];
    if (v > best_val || (v == best_val && i < best_idx)) {
      best_val = v;
      best_idx = i;
    }
  }

  __shared__ float s_val[32];
  __shared__ size_t s_idx[32];
  const int lane = tid & 31;
  const int wid  = tid >> 5;

  // Warp reduce
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    float v_other   = __shfl_down_sync(0xFFFFFFFF, best_val, offset);
    size_t i_other  = __shfl_down_sync(0xFFFFFFFF, best_idx, offset);
    if (v_other > best_val || (v_other == best_val && i_other < best_idx)) {
      best_val = v_other;
      best_idx = i_other;
    }
  }
  if (lane == 0) { s_val[wid] = best_val; s_idx[wid] = best_idx; }
  __syncthreads();

  const int num_warps = blockDim.x / 32;
  if (wid == 0) {
    if (tid < num_warps) {
      best_val = s_val[tid];
      best_idx = s_idx[tid];
    } else {
      best_val = -FLT_MAX;
      best_idx = SIZE_MAX;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      float v_other  = __shfl_down_sync(0xFFFFFFFF, best_val, offset);
      size_t i_other = __shfl_down_sync(0xFFFFFFFF, best_idx, offset);
      if (v_other > best_val || (v_other == best_val && i_other < best_idx)) {
        best_val = v_other;
        best_idx = i_other;
      }
    }
    if (tid == 0) {
      partial_vals[bid] = best_val;
      partial_idxs[bid] = best_idx;
    }
  }
}

__global__ void argmax_stage2_fp32_kernel(const float* __restrict__ partial_vals,
                                          const size_t* __restrict__ partial_idxs,
                                          int num_partials,
                                          size_t* __restrict__ output_idx) {
  const int tid = threadIdx.x;
  float best_val = -FLT_MAX;
  size_t best_idx = SIZE_MAX;
  if (tid < num_partials) {
    best_val = partial_vals[tid];
    best_idx = partial_idxs[tid];
  }
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    float v_other  = __shfl_down_sync(0xFFFFFFFF, best_val, offset);
    size_t i_other = __shfl_down_sync(0xFFFFFFFF, best_idx, offset);
    if (v_other > best_val || (v_other == best_val && i_other < best_idx)) {
      best_val = v_other;
      best_idx = i_other;
    }
  }
  if (tid == 0) *output_idx = best_idx;
}

// Persistent scratch for the two-stage argmax (one per process, small).
struct ArgmaxScratch {
  float*  partial_vals = nullptr;
  size_t* partial_idxs = nullptr;
  int capacity = 0;
};
static ArgmaxScratch g_argmax_scratch;

void ensure_argmax_scratch(int num_partials) {
  if (g_argmax_scratch.capacity >= num_partials) return;
  if (g_argmax_scratch.partial_vals) cudaFree(g_argmax_scratch.partial_vals);
  if (g_argmax_scratch.partial_idxs) cudaFree(g_argmax_scratch.partial_idxs);
  cudaMalloc(&g_argmax_scratch.partial_vals, num_partials * sizeof(float));
  cudaMalloc(&g_argmax_scratch.partial_idxs, num_partials * sizeof(size_t));
  g_argmax_scratch.capacity = num_partials;
}
}  // anonymous namespace

void argmax_fp32_fast_cu(const float* input_ptr, size_t size,
                         size_t* output_gpu, size_t* output_pinned,
                         void* stream) {
  cudaStream_t s = stream ? static_cast<cudaStream_t>(stream) : nullptr;
  // Size threshold: only use two-stage for large vocab.  Below this, a single
  // block is already fast enough.
  const int num_blocks = (size >= 32 * 1024)
                        ? kArgmaxStage1Blocks
                        : 1;
  if (num_blocks == 1) {
    argmax_kernel_fp32<<<1, 512, 0, s>>>(input_ptr, size, output_gpu);
  } else {
    ensure_argmax_scratch(num_blocks);
    argmax_stage1_fp32_kernel<<<num_blocks, kArgmaxStage1Threads, 0, s>>>(
        input_ptr, size,
        g_argmax_scratch.partial_vals,
        g_argmax_scratch.partial_idxs);
    // Stage 2: final reduction in a single warp.  num_blocks must be <= 32.
    argmax_stage2_fp32_kernel<<<1, 32, 0, s>>>(
        g_argmax_scratch.partial_vals,
        g_argmax_scratch.partial_idxs,
        num_blocks,
        output_gpu);
  }
  cudaMemcpyAsync(output_pinned, output_gpu, sizeof(size_t),
                  cudaMemcpyDeviceToHost, s);
}


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