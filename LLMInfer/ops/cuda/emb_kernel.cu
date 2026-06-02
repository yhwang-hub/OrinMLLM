#include "emb_kernel.cuh"
#include <cuda_fp16.h>
namespace kernel {

// ============================================================================
// Optimized Embedding Kernels for NVIDIA Orin (SM 8.7 Ampere)
// Embedding = table lookup = copying a row from weight matrix to output.
// These are pure memory-copy kernels → optimize for maximum memory bandwidth.
// Key optimizations:
//   1. float4 (128-bit) vectorized load/store = 4x fewer memory transactions
//   2. __ldg() read-only cache for weight table (large, random-access)
//   3. __restrict__ pointer aliasing hints
//   4. Increased thread count (128 → 256) for better latency hiding
//   5. For FP16→FP32: batch convert 8 halfs → 2 x float4 stores
// ============================================================================

// FP32 weight → FP32 output: float4 (128-bit) vectorized copy
__global__ void emb_kernel_cu_fp32(int32_t vocab_size, int32_t token_num, int32_t weight_dim,
                                   const int32_t* __restrict__ input_ptr,
                                   const float* __restrict__ weight_ptr,
                                   float* __restrict__ output_ptr) {
  int32_t token_idx = blockIdx.x;
  if (token_idx >= token_num) return;
  int32_t token = __ldg(input_ptr + token_idx);
  if (token >= vocab_size) return;

  float* output_ptr_start = output_ptr + token_idx * weight_dim;
  const float* weight_ptr_start = weight_ptr + token * weight_dim;

  // float4 vectorized copy: 4 floats (16 bytes) per iteration
  const int VEC = 4;
  const int num_vecs = weight_dim / VEC;

  for (int32_t i = threadIdx.x; i < num_vecs; i += blockDim.x) {
    float4 w = __ldg(reinterpret_cast<const float4*>(weight_ptr_start) + i);
    reinterpret_cast<float4*>(output_ptr_start)[i] = w;
  }

  // Scalar tail for remaining 0-3 elements
  for (int32_t i = num_vecs * VEC + threadIdx.x; i < weight_dim; i += blockDim.x) {
    output_ptr_start[i] = __ldg(weight_ptr_start + i);
  }
}

// FP16 weight → FP32 output: float4 load (8 halfs) → convert → 2x float4 store
__global__ void emb_kernel_cu_fp16(int32_t vocab_size, int32_t token_num, int32_t weight_dim,
                                   const int32_t* __restrict__ input_ptr,
                                   const half* __restrict__ weight_ptr,
                                   float* __restrict__ output_ptr) {
  int32_t token_idx = blockIdx.x;
  if (token_idx >= token_num) return;
  int32_t token = __ldg(input_ptr + token_idx);
  if (token >= vocab_size) return;

  float* output_ptr_start = output_ptr + token_idx * weight_dim;
  const half* weight_ptr_start = weight_ptr + token * weight_dim;

  // float4 load = 8 halfs → convert → 2x float4 store = 8 floats
  const int VEC = 8;  // 8 halfs per iteration
  const int num_vecs = weight_dim / VEC;

  for (int32_t i = threadIdx.x; i < num_vecs; i += blockDim.x) {
    // Load 128 bits = 8 halfs via __ldg
    float4 packed = __ldg(reinterpret_cast<const float4*>(weight_ptr_start) + i);
    const half2* h2 = reinterpret_cast<const half2*>(&packed);

    // Convert 8 halfs → 8 floats, store as 2x float4
    float4 out_lo, out_hi;
    out_lo.x = __half2float(h2[0].x);
    out_lo.y = __half2float(h2[0].y);
    out_lo.z = __half2float(h2[1].x);
    out_lo.w = __half2float(h2[1].y);
    out_hi.x = __half2float(h2[2].x);
    out_hi.y = __half2float(h2[2].y);
    out_hi.z = __half2float(h2[3].x);
    out_hi.w = __half2float(h2[3].y);

    reinterpret_cast<float4*>(output_ptr_start)[i * 2]     = out_lo;
    reinterpret_cast<float4*>(output_ptr_start)[i * 2 + 1] = out_hi;
  }

  // Scalar tail for remaining elements
  for (int32_t i = num_vecs * VEC + threadIdx.x; i < weight_dim; i += blockDim.x) {
    output_ptr_start[i] = __half2float(__ldg(weight_ptr_start + i));
  }
}

// Pure FP16 weight → FP16 output: float4 (128-bit) vectorized copy (no conversion)
__global__ void emb_kernel_cu_pure_fp16_impl(int32_t vocab_size, int32_t token_num, int32_t weight_dim,
                                              const int32_t* __restrict__ input_ptr,
                                              const half* __restrict__ weight_ptr,
                                              half* __restrict__ output_ptr) {
  int32_t token_idx = blockIdx.x;
  if (token_idx >= token_num) return;
  int32_t token = __ldg(input_ptr + token_idx);
  if (token >= vocab_size) return;

  half* output_ptr_start = output_ptr + static_cast<int64_t>(token_idx) * weight_dim;
  const half* weight_ptr_start = weight_ptr + static_cast<int64_t>(token) * weight_dim;

  // float4 vectorized copy: 8 halfs (16 bytes) per iteration
  const int VEC = 8;
  const int num_vecs = weight_dim / VEC;

  for (int32_t i = threadIdx.x; i < num_vecs; i += blockDim.x) {
    float4 w = __ldg(reinterpret_cast<const float4*>(weight_ptr_start) + i);
    reinterpret_cast<float4*>(output_ptr_start)[i] = w;
  }

  // Scalar tail for remaining 0-7 elements
  for (int32_t i = num_vecs * VEC + threadIdx.x; i < weight_dim; i += blockDim.x) {
    output_ptr_start[i] = __ldg(weight_ptr_start + i);
  }
}

void emb_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                   const tensor::Tensor& output, int32_t vocab_size, void* stream) {
  tensor::Tensor input_cu;
  const int32_t* in_ptr = nullptr;
  if (input.device_type() != base::DeviceType::kDeviceCUDA) {
    input_cu = input.clone();
    input_cu.to_cuda();
    in_ptr = input_cu.ptr<int32_t>();
  } else {
    in_ptr = input.ptr<int32_t>();
  }
  const int32_t input_num = static_cast<int32_t>(input.size());
  const int32_t weight_dim = weight.get_dim(1);
  CHECK(weight.device_type() == output.device_type());
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

  // 使用实际的 token 数量作为 grid size，而不是固定的 max_seq_len
  const int32_t grid_size = input_num;
  constexpr int32_t thread_num = 256;  // 256 threads for better latency hiding with float4 loads
  
  cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
  
  // Check output data type to determine compute path
  if (output.data_type() == base::DataType::kDataTypeFp16) {
    // Pure FP16 path: FP16 weight -> FP16 output (no conversion)
    CHECK(weight.data_type() == base::DataType::kDataTypeFp16) 
        << "FP16 output requires FP16 weight";
    const half* wei_ptr = reinterpret_cast<const half*>(weight.ptr<uint16_t>());
    half* out_ptr = reinterpret_cast<half*>(const_cast<uint16_t*>(output.ptr<uint16_t>()));
    
    if (stream_) {
      emb_kernel_cu_pure_fp16_impl<<<grid_size, thread_num, 0, stream_>>>(
          vocab_size, input_num, weight_dim, in_ptr, wei_ptr, out_ptr);
    } else {
      emb_kernel_cu_pure_fp16_impl<<<grid_size, thread_num>>>(
          vocab_size, input_num, weight_dim, in_ptr, wei_ptr, out_ptr);
    }
  } else {
    // FP32 output path
    float* out_ptr = const_cast<float*>(output.ptr<float>());
    
    // Check weight data type and dispatch to appropriate kernel
    if (weight.data_type() == base::DataType::kDataTypeFp16) {
      const half* wei_ptr = reinterpret_cast<const half*>(weight.ptr<uint16_t>());
      if (stream_) {
        emb_kernel_cu_fp16<<<grid_size, thread_num, 0, stream_>>>(vocab_size, input_num, weight_dim,
                                                                     in_ptr, wei_ptr, out_ptr);
      } else {
        emb_kernel_cu_fp16<<<grid_size, thread_num>>>(vocab_size, input_num, weight_dim, in_ptr,
                                                         wei_ptr, out_ptr);
      }
    } else {
      float* wei_ptr = const_cast<float*>(weight.ptr<float>());
      if (stream_) {
        emb_kernel_cu_fp32<<<grid_size, thread_num, 0, stream_>>>(vocab_size, input_num, weight_dim,
                                                                     in_ptr, wei_ptr, out_ptr);
      } else {
        emb_kernel_cu_fp32<<<grid_size, thread_num>>>(vocab_size, input_num, weight_dim, in_ptr,
                                                         wei_ptr, out_ptr);
      }
    }
  }
}

}  // namespace kernel