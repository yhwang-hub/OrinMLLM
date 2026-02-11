#include <device_launch_parameters.h>
#include <cub/block/block_reduce.cuh>
#include <cuda_fp16.h>
#include "rmsnorm_kernel.cuh"
namespace kernel {

/**
 * FP16 weight RMSNorm kernel (FP32 input × FP16 weight → FP32 output)
 * weight is FP16, input/output are FP32
 */
template <int32_t BLOCK_DIM>
static __global__ void row_rmsnorm_f32_fp16w(float* in, const half* wei, float* out, int size, float eps) {
  const int tid = threadIdx.x;

  constexpr int pack_size = 4;
  const int pack_num = size / pack_size;
  const int pack_off = pack_size * pack_num;

  float sum = 0.0f;
  float4* in_pack = reinterpret_cast<float4*>(in);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    sum += in_float4.x * in_float4.x;
    sum += in_float4.y * in_float4.y;
    sum += in_float4.z * in_float4.z;
    sum += in_float4.w * in_float4.w;
  }

  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    sum += in[i] * in[i];
  }

  using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

  // Load FP16 weight and convert to FP32 for computation
  // Pack 4 half values as 2 half2 for each float4 output
  float4* out_pack = reinterpret_cast<float4*>(out);
  
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    // Load 4 consecutive half values
    int base_idx = i * 4;
    float w0 = __half2float(wei[base_idx]);
    float w1 = __half2float(wei[base_idx + 1]);
    float w2 = __half2float(wei[base_idx + 2]);
    float w3 = __half2float(wei[base_idx + 3]);
    
    *(out_pack + i) =
        make_float4(scale * in_float4.x * w0, scale * in_float4.y * w1,
                    scale * in_float4.z * w2, scale * in_float4.w * w3);
  }

  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    out[i] = __half2float(wei[i]) * in[i] * scale;
  }
}

/**
 * FP16 weight RMSNorm kernel for multi-dim input
 */
static __global__ void row_rmsnorm_f32_fp16w_dim(float* in, const half* wei, float* out, int dim_size,
                                                  int size, float eps) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  if (bid >= dim_size) {
    return;
  }

  float* block_in = in + bid * size;
  float* block_out = out + bid * size;
  constexpr int pack_size = 4;
  const int pack_num = size / pack_size;
  const int pack_off = pack_size * pack_num;

  float sum = 0.0f;
  float4* in_pack = reinterpret_cast<float4*>(block_in);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    sum += in_float4.x * in_float4.x;
    sum += in_float4.y * in_float4.y;
    sum += in_float4.z * in_float4.z;
    sum += in_float4.w * in_float4.w;
  }

  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    sum += block_in[i] * block_in[i];
  }

  using BlockReduce = cub::BlockReduce<float, 128>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

  float4* out_pack = reinterpret_cast<float4*>(block_out);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    // Load 4 consecutive half values
    int base_idx = i * 4;
    float w0 = __half2float(wei[base_idx]);
    float w1 = __half2float(wei[base_idx + 1]);
    float w2 = __half2float(wei[base_idx + 2]);
    float w3 = __half2float(wei[base_idx + 3]);
    
    *(out_pack + i) =
        make_float4(scale * in_float4.x * w0, scale * in_float4.y * w1,
                    scale * in_float4.z * w2, scale * in_float4.w * w3);
  }

  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    block_out[i] = __half2float(wei[i]) * block_in[i] * scale;
  }
}

/**
 * 计算多维输入 in = (dim1, dim2), 计算在dim2维度上的rmsnorm
 */
static __global__ void row_rmsnorm_f32_dim(float* in, float* wei, float* out, int dim_size,
                                           int size, float eps) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  if (bid >= dim_size) {
    return;
  }

  float* block_in = in + bid * size;
  float* block_out = out + bid * size;
  constexpr int pack_size = 4;
  const int pack_num = size / pack_size;
  const int pack_off = pack_size * pack_num;

  float sum = 0.0f;
  float4* in_pack = reinterpret_cast<float4*>(block_in);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    sum += in_float4.x * in_float4.x;
    sum += in_float4.y * in_float4.y;
    sum += in_float4.z * in_float4.z;
    sum += in_float4.w * in_float4.w;
  }

  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    sum += block_in[i] * block_in[i];
  }

  using BlockReduce = cub::BlockReduce<float, 128>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

  float4* wei_pack = reinterpret_cast<float4*>(wei);
  float4* out_pack = reinterpret_cast<float4*>(block_out);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    float4 wei_float4 = *(wei_pack + i);
    *(out_pack + i) =
        make_float4(scale * in_float4.x * wei_float4.x, scale * in_float4.y * wei_float4.y,
                    scale * in_float4.z * wei_float4.z, scale * in_float4.w * wei_float4.w);
  }

  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    block_out[i] = wei[i] * block_in[i] * scale;
  }
}

template <int32_t BLOCK_DIM>
static __global__ void row_rmsnorm_f32(float* in, float* wei, float* out, int size, float eps) {
  const int tid = threadIdx.x;

  constexpr int pack_size = 4;
  const int pack_num = size / pack_size;
  const int pack_off = pack_size * pack_num;

  float sum = 0.0f;
  float4* in_pack = reinterpret_cast<float4*>(in);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    sum += in_float4.x * in_float4.x;
    sum += in_float4.y * in_float4.y;
    sum += in_float4.z * in_float4.z;
    sum += in_float4.w * in_float4.w;
  }

  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    sum += in[i] * in[i];
  }

  using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

  float4* wei_pack = reinterpret_cast<float4*>(wei);
  float4* out_pack = reinterpret_cast<float4*>(out);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    float4 wei_float4 = *(wei_pack + i);
    *(out_pack + i) =
        make_float4(scale * in_float4.x * wei_float4.x, scale * in_float4.y * wei_float4.y,
                    scale * in_float4.z * wei_float4.z, scale * in_float4.w * wei_float4.w);
  }

  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    out[i] = wei[i] * in[i] * scale;
  }
}

// Forward declarations for pure FP16 RMSNorm kernels
template <int32_t BLOCK_DIM>
static __global__ void row_rmsnorm_pure_fp16(const half* in, const half* wei, half* out, 
                                              int size, float eps);

template <int32_t BLOCK_DIM>
static __global__ void row_rmsnorm_pure_fp16_dim(const half* in, const half* wei, half* out, 
                                                  int dim_size, int size, float eps);

void rmsnorm_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA &&
        weight.device_type() == base::DeviceType::kDeviceCUDA &&
        output.device_type() == base::DeviceType::kDeviceCUDA);

#if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
  const float eps = 1e-6f;
#else
  const float eps = 1e-5f;
#endif
  const int32_t size = static_cast<int32_t>(input.size());
  constexpr int threads_num = 128;
  
  // Check if this is pure FP16 path (FP16 input, FP16 weight, FP16 output)
  if (input.data_type() == base::DataType::kDataTypeFp16 &&
      output.data_type() == base::DataType::kDataTypeFp16 &&
      weight.data_type() == base::DataType::kDataTypeFp16) {
    const half* in_ptr = reinterpret_cast<const half*>(input.ptr<uint16_t>());
    const half* wei_ptr = reinterpret_cast<const half*>(weight.ptr<uint16_t>());
    half* out_ptr = reinterpret_cast<half*>(const_cast<uint16_t*>(output.ptr<uint16_t>()));
    
    if (stream) {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      row_rmsnorm_pure_fp16<128><<<1, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    } else {
      row_rmsnorm_pure_fp16<128><<<1, threads_num>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    }
    return;
  }
  
  // FP32 path
  float* in_ptr = const_cast<float*>(input.ptr<float>());
  float* out_ptr = const_cast<float*>(output.ptr<float>());
  
  // Check if weight is FP16
  if (weight.data_type() == base::DataType::kDataTypeFp16) {
    const half* wei_ptr = reinterpret_cast<const half*>(weight.ptr<uint16_t>());
    if (stream) {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      row_rmsnorm_f32_fp16w<128><<<1, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    } else {
      row_rmsnorm_f32_fp16w<128><<<1, threads_num>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    }
  } else {
    float* wei_ptr = const_cast<float*>(weight.ptr<float>());
    if (stream) {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      row_rmsnorm_f32<128><<<1, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    } else {
      row_rmsnorm_f32<128><<<1, threads_num>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    }
  }
}

void rmsnorm_kernel_cu_dim(const tensor::Tensor& input, const tensor::Tensor& weight,
                           const tensor::Tensor& output, int32_t dim, void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA &&
        weight.device_type() == base::DeviceType::kDeviceCUDA &&
        output.device_type() == base::DeviceType::kDeviceCUDA);

  const float eps = 1e-6f;
  const int32_t total_size = static_cast<int32_t>(input.size());
  const int32_t size = input.get_dim(input.dims_size() - 1);
  const int32_t dim_size = total_size / size;
  constexpr int threads_num = 128;

  // Check if this is pure FP16 path
  if (input.data_type() == base::DataType::kDataTypeFp16 &&
      output.data_type() == base::DataType::kDataTypeFp16 &&
      weight.data_type() == base::DataType::kDataTypeFp16) {
    const half* in_ptr = reinterpret_cast<const half*>(input.ptr<uint16_t>());
    const half* wei_ptr = reinterpret_cast<const half*>(weight.ptr<uint16_t>());
    half* out_ptr = reinterpret_cast<half*>(const_cast<uint16_t*>(output.ptr<uint16_t>()));
    
    if (stream) {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      row_rmsnorm_pure_fp16_dim<128><<<dim_size, threads_num, 0, stream_>>>(
          in_ptr, wei_ptr, out_ptr, dim_size, size, eps);
    } else {
      row_rmsnorm_pure_fp16_dim<128><<<dim_size, threads_num>>>(
          in_ptr, wei_ptr, out_ptr, dim_size, size, eps);
    }
    return;
  }

  // FP32 path
  float* in_ptr = const_cast<float*>(input.ptr<float>());
  float* out_ptr = const_cast<float*>(output.ptr<float>());
  
  // Check if weight is FP16
  if (weight.data_type() == base::DataType::kDataTypeFp16) {
    const half* wei_ptr = reinterpret_cast<const half*>(weight.ptr<uint16_t>());
    if (stream) {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      row_rmsnorm_f32_fp16w_dim<<<dim_size, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, dim_size, size, eps);
    } else {
      row_rmsnorm_f32_fp16w_dim<<<dim_size, threads_num>>>(in_ptr, wei_ptr, out_ptr, dim_size, size, eps);
    }
  } else {
    float* wei_ptr = const_cast<float*>(weight.ptr<float>());
    if (stream) {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      row_rmsnorm_f32_dim<<<dim_size, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, dim_size, size, eps);
    } else {
      row_rmsnorm_f32_dim<<<dim_size, threads_num>>>(in_ptr, wei_ptr, out_ptr, dim_size, size, eps);
    }
  }
}

// ==================== Pure FP16 RMSNorm Implementation ====================

/**
 * Pure FP16 RMSNorm kernel: FP16 input x FP16 weight -> FP16 output
 * Computation is done in FP32 internally for better precision
 */
template <int32_t BLOCK_DIM>
static __global__ void row_rmsnorm_pure_fp16(const half* in, const half* wei, half* out, 
                                              int size, float eps) {
  const int tid = threadIdx.x;

  // Use half2 for vectorized loads
  const int num_h2 = size / 2;
  const half2* in_h2 = reinterpret_cast<const half2*>(in);
  
  float sum = 0.0f;
  
  // Calculate sum of squares
  for (int i = tid; i < num_h2; i += blockDim.x) {
    half2 val = in_h2[i];
    float2 fval = __half22float2(val);
    sum += fval.x * fval.x + fval.y * fval.y;
  }
  
  // Handle remainder
  const int base = num_h2 * 2;
  for (int i = base + tid; i < size; i += blockDim.x) {
    float fval = __half2float(in[i]);
    sum += fval * fval;
  }

  // Block reduction
  using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

  // Apply normalization and weight
  const half2* wei_h2 = reinterpret_cast<const half2*>(wei);
  half2* out_h2 = reinterpret_cast<half2*>(out);
  
  for (int i = tid; i < num_h2; i += blockDim.x) {
    half2 val = in_h2[i];
    half2 w = wei_h2[i];
    float2 fval = __half22float2(val);
    float2 fw = __half22float2(w);
    float2 result;
    result.x = scale * fval.x * fw.x;
    result.y = scale * fval.y * fw.y;
    out_h2[i] = __float22half2_rn(result);
  }
  
  for (int i = base + tid; i < size; i += blockDim.x) {
    float fval = __half2float(in[i]);
    float fw = __half2float(wei[i]);
    out[i] = __float2half(scale * fval * fw);
  }
}

/**
 * Pure FP16 RMSNorm kernel for multi-row input (batched)
 */
template <int32_t BLOCK_DIM>
static __global__ void row_rmsnorm_pure_fp16_dim(const half* in, const half* wei, half* out, 
                                                  int dim_size, int size, float eps) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  if (bid >= dim_size) return;

  const half* block_in = in + static_cast<int64_t>(bid) * size;
  half* block_out = out + static_cast<int64_t>(bid) * size;
  
  const int num_h2 = size / 2;
  const half2* in_h2 = reinterpret_cast<const half2*>(block_in);
  
  float sum = 0.0f;
  
  for (int i = tid; i < num_h2; i += blockDim.x) {
    half2 val = in_h2[i];
    float2 fval = __half22float2(val);
    sum += fval.x * fval.x + fval.y * fval.y;
  }
  
  const int base = num_h2 * 2;
  for (int i = base + tid; i < size; i += blockDim.x) {
    float fval = __half2float(block_in[i]);
    sum += fval * fval;
  }

  using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

  const half2* wei_h2 = reinterpret_cast<const half2*>(wei);
  half2* out_h2 = reinterpret_cast<half2*>(block_out);
  
  for (int i = tid; i < num_h2; i += blockDim.x) {
    half2 val = in_h2[i];
    half2 w = wei_h2[i];
    float2 fval = __half22float2(val);
    float2 fw = __half22float2(w);
    float2 result;
    result.x = scale * fval.x * fw.x;
    result.y = scale * fval.y * fw.y;
    out_h2[i] = __float22half2_rn(result);
  }
  
  for (int i = base + tid; i < size; i += blockDim.x) {
    float fval = __half2float(block_in[i]);
    float fw = __half2float(wei[i]);
    block_out[i] = __float2half(scale * fval * fw);
  }
}

void rmsnorm_kernel_cu_pure_fp16(const tensor::Tensor& input, const tensor::Tensor& weight,
                                  const tensor::Tensor& output, void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());
  CHECK(input.data_type() == base::DataType::kDataTypeFp16);
  CHECK(weight.data_type() == base::DataType::kDataTypeFp16);
  CHECK(output.data_type() == base::DataType::kDataTypeFp16);

  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA &&
        weight.device_type() == base::DeviceType::kDeviceCUDA &&
        output.device_type() == base::DeviceType::kDeviceCUDA);

  const float eps = 1e-6f;
  const int32_t size = static_cast<int32_t>(input.size());
  const half* in_ptr = reinterpret_cast<const half*>(input.ptr<uint16_t>());
  const half* wei_ptr = reinterpret_cast<const half*>(weight.ptr<uint16_t>());
  half* out_ptr = reinterpret_cast<half*>(const_cast<uint16_t*>(output.ptr<uint16_t>()));
  
  constexpr int threads_num = 128;
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    row_rmsnorm_pure_fp16<128><<<1, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, size, eps);
  } else {
    row_rmsnorm_pure_fp16<128><<<1, threads_num>>>(in_ptr, wei_ptr, out_ptr, size, eps);
  }
}

void rmsnorm_kernel_cu_pure_fp16_dim(const tensor::Tensor& input, const tensor::Tensor& weight,
                                      const tensor::Tensor& output, int32_t dim, void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());
  CHECK(input.data_type() == base::DataType::kDataTypeFp16);
  CHECK(weight.data_type() == base::DataType::kDataTypeFp16);
  CHECK(output.data_type() == base::DataType::kDataTypeFp16);

  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA &&
        weight.device_type() == base::DeviceType::kDeviceCUDA &&
        output.device_type() == base::DeviceType::kDeviceCUDA);

  const float eps = 1e-6f;
  const int32_t total_size = static_cast<int32_t>(input.size());
  const int32_t size = input.get_dim(input.dims_size() - 1);
  const int32_t dim_size = total_size / size;

  const half* in_ptr = reinterpret_cast<const half*>(input.ptr<uint16_t>());
  const half* wei_ptr = reinterpret_cast<const half*>(weight.ptr<uint16_t>());
  half* out_ptr = reinterpret_cast<half*>(const_cast<uint16_t*>(output.ptr<uint16_t>()));
  
  constexpr int threads_num = 128;
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    row_rmsnorm_pure_fp16_dim<128><<<dim_size, threads_num, 0, stream_>>>(
        in_ptr, wei_ptr, out_ptr, dim_size, size, eps);
  } else {
    row_rmsnorm_pure_fp16_dim<128><<<dim_size, threads_num>>>(
        in_ptr, wei_ptr, out_ptr, dim_size, size, eps);
  }
}
}  // namespace kernel