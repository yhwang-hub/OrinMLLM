#include "emb_kernel.cuh"
#include <cuda_fp16.h>
namespace kernel {
__global__ void emb_kernel_cu_fp32(int32_t vocab_size, int32_t token_num, int32_t weight_dim,
                                   const int32_t* input_ptr, const float* weight_ptr,
                                   float* output_ptr) {
  int32_t token_idx = blockIdx.x;
  if (token_idx >= token_num) {
    return;
  }
  int32_t token = input_ptr[token_idx];
  if (token >= vocab_size) {
    return;
  }

  float* output_ptr_start = output_ptr + token_idx * weight_dim;
  const float* weight_ptr_start = weight_ptr + token * weight_dim;

  for (int32_t i = threadIdx.x; i < weight_dim; i += blockDim.x) {
    output_ptr_start[i] = weight_ptr_start[i];
  }
}

// FP16 weight embedding kernel: converts FP16 weight to FP32 output
__global__ void emb_kernel_cu_fp16(int32_t vocab_size, int32_t token_num, int32_t weight_dim,
                                   const int32_t* input_ptr, const half* weight_ptr,
                                   float* output_ptr) {
  int32_t token_idx = blockIdx.x;
  if (token_idx >= token_num) {
    return;
  }
  int32_t token = input_ptr[token_idx];
  if (token >= vocab_size) {
    return;
  }

  float* output_ptr_start = output_ptr + token_idx * weight_dim;
  const half* weight_ptr_start = weight_ptr + token * weight_dim;

  // Use half2 for better memory throughput
  const int vec_size = 2;
  const int num_vecs = weight_dim / vec_size;
  const half2* weight_h2 = reinterpret_cast<const half2*>(weight_ptr_start);

  for (int32_t i = threadIdx.x; i < num_vecs; i += blockDim.x) {
    half2 w = weight_h2[i];
    output_ptr_start[i * 2] = __half2float(w.x);
    output_ptr_start[i * 2 + 1] = __half2float(w.y);
  }
  
  // Handle remainder
  for (int32_t i = num_vecs * vec_size + threadIdx.x; i < weight_dim; i += blockDim.x) {
    output_ptr_start[i] = __half2float(weight_ptr_start[i]);
  }
}

// Pure FP16 embedding kernel: FP16 weight -> FP16 output (no conversion)
__global__ void emb_kernel_cu_pure_fp16_impl(int32_t vocab_size, int32_t token_num, int32_t weight_dim,
                                              const int32_t* input_ptr, const half* weight_ptr,
                                              half* output_ptr) {
  int32_t token_idx = blockIdx.x;
  if (token_idx >= token_num) {
    return;
  }
  int32_t token = input_ptr[token_idx];
  if (token >= vocab_size) {
    return;
  }

  half* output_ptr_start = output_ptr + static_cast<int64_t>(token_idx) * weight_dim;
  const half* weight_ptr_start = weight_ptr + static_cast<int64_t>(token) * weight_dim;

  // Use half2 for vectorized copy (2x bandwidth)
  const int vec_size = 2;
  const int num_vecs = weight_dim / vec_size;
  const half2* weight_h2 = reinterpret_cast<const half2*>(weight_ptr_start);
  half2* output_h2 = reinterpret_cast<half2*>(output_ptr_start);

  for (int32_t i = threadIdx.x; i < num_vecs; i += blockDim.x) {
    output_h2[i] = weight_h2[i];
  }
  
  // Handle remainder
  for (int32_t i = num_vecs * vec_size + threadIdx.x; i < weight_dim; i += blockDim.x) {
    output_ptr_start[i] = weight_ptr_start[i];
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
  constexpr int32_t thread_num = 128;
  
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

// Pure FP16 embedding: FP16 weight -> FP16 output (no conversion)
void emb_kernel_cu_pure_fp16(const tensor::Tensor& input, const tensor::Tensor& weight,
                              const tensor::Tensor& output, int32_t vocab_size, void* stream) {
  tensor::Tensor input_cu;
  if (input.device_type() != base::DeviceType::kDeviceCUDA) {
    input_cu = input.clone();
    input_cu.to_cuda();
  }
  
  const int32_t input_num = static_cast<int32_t>(input.size());
  const int32_t weight_dim = weight.get_dim(1);
  
  CHECK(weight.device_type() == output.device_type());
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(weight.data_type() == base::DataType::kDataTypeFp16);
  CHECK(output.data_type() == base::DataType::kDataTypeFp16);

  // 使用实际的 token 数量作为 grid size
  const int32_t grid_size = input_num;
  constexpr int32_t thread_num = 128;
  
  int32_t* in_ptr = input_cu.ptr<int32_t>();
  const half* wei_ptr = reinterpret_cast<const half*>(weight.ptr<uint16_t>());
  half* out_ptr = reinterpret_cast<half*>(const_cast<uint16_t*>(output.ptr<uint16_t>()));
  
  cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
  
  if (stream_) {
    emb_kernel_cu_pure_fp16_impl<<<grid_size, thread_num, 0, stream_>>>(
        vocab_size, input_num, weight_dim, in_ptr, wei_ptr, out_ptr);
  } else {
    emb_kernel_cu_pure_fp16_impl<<<grid_size, thread_num>>>(
        vocab_size, input_num, weight_dim, in_ptr, wei_ptr, out_ptr);
  }
}
}  // namespace kernel