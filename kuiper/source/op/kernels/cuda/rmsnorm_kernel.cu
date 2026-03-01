#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include "rmsnorm_kernel.cuh"

namespace kernel {

// =======================================================================
// Optimized warp-shuffle based block reduction
// Replaces cub::BlockReduce: lower latency, less shared memory,
// fewer __syncthreads calls
// =======================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

// Block-level sum reduction with broadcast to all threads
// Uses only NUM_WARPS * 4 bytes shared memory (vs ~512 bytes for cub)
template <int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float val) {
  static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be multiple of 32");
  constexpr int NUM_WARPS = BLOCK_SIZE / 32;
  __shared__ float smem_reduce[NUM_WARPS];

  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;

  // Intra-warp reduction via shuffle
  val = warp_reduce_sum(val);

  // Store each warp's partial sum
  if (lane == 0) smem_reduce[wid] = val;
  __syncthreads();

  // First warp reduces all partial sums
  val = (threadIdx.x < NUM_WARPS) ? smem_reduce[threadIdx.x] : 0.0f;
  if (wid == 0) val = warp_reduce_sum(val);

  // Broadcast final result to all threads
  if (threadIdx.x == 0) smem_reduce[0] = val;
  __syncthreads();
  return smem_reduce[0];
}

// =======================================================================
// row_rmsnorm_f32_fp16w: FP32 input × FP16 weight → FP32 output (1 row)
// Optimizations: warp shuffle, uint2 vectorized FP16 weight load,
//                __ldg for read-only weight, __restrict__, #pragma unroll
// =======================================================================
template <int32_t BLOCK_DIM>
static __global__ void row_rmsnorm_f32_fp16w(float* __restrict__ in,
                                              const half* __restrict__ wei,
                                              float* __restrict__ out,
                                              int size, float eps) {
  const int tid = threadIdx.x;
  const int pack_num = size >> 2;
  const int pack_off = pack_num << 2;

  // Phase 1: Sum of squares with float4 vectorized loads
  float sum = 0.0f;
  const float4* in_pack = reinterpret_cast<const float4*>(in);

#pragma unroll 4
  for (int i = tid; i < pack_num; i += BLOCK_DIM) {
    float4 v = in_pack[i];
    sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }
  for (int i = pack_off + tid; i < size; i += BLOCK_DIM) {
    float v = in[i];
    sum += v * v;
  }

  // Phase 2: Warp-shuffle block reduction
  sum = block_reduce_sum<BLOCK_DIM>(sum);
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

  // Phase 3: Normalize and scale with vectorized FP16 weight load
  float4* out_pack = reinterpret_cast<float4*>(out);
  const uint2* wei_u2 = reinterpret_cast<const uint2*>(wei);

#pragma unroll 4
  for (int i = tid; i < pack_num; i += BLOCK_DIM) {
    float4 v = in_pack[i];
    // Load 4 half weights as one 64-bit load via __ldg
    uint2 w_raw = __ldg(wei_u2 + i);
    float2 fw01 = __half22float2(*reinterpret_cast<const half2*>(&w_raw.x));
    float2 fw23 = __half22float2(*reinterpret_cast<const half2*>(&w_raw.y));

    out_pack[i] = make_float4(scale * v.x * fw01.x, scale * v.y * fw01.y,
                               scale * v.z * fw23.x, scale * v.w * fw23.y);
  }
  for (int i = pack_off + tid; i < size; i += BLOCK_DIM) {
    out[i] = __half2float(__ldg(wei + i)) * in[i] * scale;
  }
}

// =======================================================================
// row_rmsnorm_f32_fp16w_dim: FP32 input × FP16 weight → FP32 output (batched)
// =======================================================================
template <int32_t BLOCK_DIM>
static __global__ void row_rmsnorm_f32_fp16w_dim(float* __restrict__ in,
                                                  const half* __restrict__ wei,
                                                  float* __restrict__ out,
                                                  int dim_size, int size, float eps) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  if (bid >= dim_size) return;

  float* block_in = in + bid * size;
  float* block_out = out + bid * size;
  const int pack_num = size >> 2;
  const int pack_off = pack_num << 2;

  float sum = 0.0f;
  const float4* in_pack = reinterpret_cast<const float4*>(block_in);

#pragma unroll 4
  for (int i = tid; i < pack_num; i += BLOCK_DIM) {
    float4 v = in_pack[i];
    sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }
  for (int i = pack_off + tid; i < size; i += BLOCK_DIM) {
    sum += block_in[i] * block_in[i];
  }

  sum = block_reduce_sum<BLOCK_DIM>(sum);
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

  float4* out_pack = reinterpret_cast<float4*>(block_out);
  const uint2* wei_u2 = reinterpret_cast<const uint2*>(wei);

#pragma unroll 4
  for (int i = tid; i < pack_num; i += BLOCK_DIM) {
    float4 v = in_pack[i];
    uint2 w_raw = __ldg(wei_u2 + i);
    float2 fw01 = __half22float2(*reinterpret_cast<const half2*>(&w_raw.x));
    float2 fw23 = __half22float2(*reinterpret_cast<const half2*>(&w_raw.y));

    out_pack[i] = make_float4(scale * v.x * fw01.x, scale * v.y * fw01.y,
                               scale * v.z * fw23.x, scale * v.w * fw23.y);
  }
  for (int i = pack_off + tid; i < size; i += BLOCK_DIM) {
    block_out[i] = __half2float(__ldg(wei + i)) * block_in[i] * scale;
  }
}

// =======================================================================
// row_rmsnorm_f32_dim: FP32 input × FP32 weight → FP32 output (batched)
// =======================================================================
template <int32_t BLOCK_DIM>
static __global__ void row_rmsnorm_f32_dim(float* __restrict__ in,
                                            float* __restrict__ wei,
                                            float* __restrict__ out,
                                            int dim_size, int size, float eps) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  if (bid >= dim_size) return;

  float* block_in = in + bid * size;
  float* block_out = out + bid * size;
  const int pack_num = size >> 2;
  const int pack_off = pack_num << 2;

  float sum = 0.0f;
  const float4* in_pack = reinterpret_cast<const float4*>(block_in);

#pragma unroll 4
  for (int i = tid; i < pack_num; i += BLOCK_DIM) {
    float4 v = in_pack[i];
    sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }
  for (int i = pack_off + tid; i < size; i += BLOCK_DIM) {
    sum += block_in[i] * block_in[i];
  }

  sum = block_reduce_sum<BLOCK_DIM>(sum);
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

  const float4* wei_pack = reinterpret_cast<const float4*>(wei);
  float4* out_pack = reinterpret_cast<float4*>(block_out);

#pragma unroll 4
  for (int i = tid; i < pack_num; i += BLOCK_DIM) {
    float4 v = in_pack[i];
    float4 w = __ldg(wei_pack + i);
    out_pack[i] = make_float4(scale * v.x * w.x, scale * v.y * w.y,
                               scale * v.z * w.z, scale * v.w * w.w);
  }
  for (int i = pack_off + tid; i < size; i += BLOCK_DIM) {
    block_out[i] = wei[i] * block_in[i] * scale;
  }
}

// =======================================================================
// row_rmsnorm_f32: FP32 input × FP32 weight → FP32 output (single row)
// =======================================================================
template <int32_t BLOCK_DIM>
static __global__ void row_rmsnorm_f32(float* __restrict__ in,
                                        float* __restrict__ wei,
                                        float* __restrict__ out,
                                        int size, float eps) {
  const int tid = threadIdx.x;
  const int pack_num = size >> 2;
  const int pack_off = pack_num << 2;

  float sum = 0.0f;
  const float4* in_pack = reinterpret_cast<const float4*>(in);

#pragma unroll 4
  for (int i = tid; i < pack_num; i += BLOCK_DIM) {
    float4 v = in_pack[i];
    sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }
  for (int i = pack_off + tid; i < size; i += BLOCK_DIM) {
    sum += in[i] * in[i];
  }

  sum = block_reduce_sum<BLOCK_DIM>(sum);
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

  const float4* wei_pack = reinterpret_cast<const float4*>(wei);
  float4* out_pack = reinterpret_cast<float4*>(out);

#pragma unroll 4
  for (int i = tid; i < pack_num; i += BLOCK_DIM) {
    float4 v = in_pack[i];
    float4 w = __ldg(wei_pack + i);
    out_pack[i] = make_float4(scale * v.x * w.x, scale * v.y * w.y,
                               scale * v.z * w.z, scale * v.w * w.w);
  }
  for (int i = pack_off + tid; i < size; i += BLOCK_DIM) {
    out[i] = wei[i] * in[i] * scale;
  }
}

// =======================================================================
// row_rmsnorm_pure_fp16: Pure FP16 (single row)
// Key optimization: 128-bit vectorized loads (uint4 = 8 halfs per load)
// for maximum memory bandwidth on Orin
// =======================================================================
template <int32_t BLOCK_DIM>
static __global__ void row_rmsnorm_pure_fp16(const half* __restrict__ in,
                                              const half* __restrict__ wei,
                                              half* __restrict__ out,
                                              int size, float eps) {
  const int tid = threadIdx.x;

  // Phase 1: Sum of squares with 128-bit vectorized loads
  const int num_vec8 = size >> 3;  // size / 8
  const int vec8_off = num_vec8 << 3;
  const uint4* in_vec = reinterpret_cast<const uint4*>(in);

  float sum = 0.0f;

#pragma unroll 2
  for (int i = tid; i < num_vec8; i += BLOCK_DIM) {
    uint4 raw = in_vec[i];
    float2 f0 = __half22float2(*reinterpret_cast<const half2*>(&raw.x));
    float2 f1 = __half22float2(*reinterpret_cast<const half2*>(&raw.y));
    float2 f2 = __half22float2(*reinterpret_cast<const half2*>(&raw.z));
    float2 f3 = __half22float2(*reinterpret_cast<const half2*>(&raw.w));
    sum += f0.x * f0.x + f0.y * f0.y + f1.x * f1.x + f1.y * f1.y +
           f2.x * f2.x + f2.y * f2.y + f3.x * f3.x + f3.y * f3.y;
  }
  // half2 remainder
  const int num_h2 = size >> 1;
  const half2* in_h2 = reinterpret_cast<const half2*>(in);
  for (int i = (vec8_off >> 1) + tid; i < num_h2; i += BLOCK_DIM) {
    float2 fv = __half22float2(in_h2[i]);
    sum += fv.x * fv.x + fv.y * fv.y;
  }
  // Scalar remainder
  for (int i = (num_h2 << 1) + tid; i < size; i += BLOCK_DIM) {
    float v = __half2float(in[i]);
    sum += v * v;
  }

  // Phase 2: Warp-shuffle reduction
  sum = block_reduce_sum<BLOCK_DIM>(sum);
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

  // Phase 3: Normalize with 128-bit vectorized stores
  const uint4* wei_vec = reinterpret_cast<const uint4*>(wei);
  uint4* out_vec = reinterpret_cast<uint4*>(out);

#pragma unroll 2
  for (int i = tid; i < num_vec8; i += BLOCK_DIM) {
    uint4 in_raw = in_vec[i];
    uint4 w_raw = __ldg(wei_vec + i);

    float2 fi0 = __half22float2(*reinterpret_cast<const half2*>(&in_raw.x));
    float2 fi1 = __half22float2(*reinterpret_cast<const half2*>(&in_raw.y));
    float2 fi2 = __half22float2(*reinterpret_cast<const half2*>(&in_raw.z));
    float2 fi3 = __half22float2(*reinterpret_cast<const half2*>(&in_raw.w));
    float2 fw0 = __half22float2(*reinterpret_cast<const half2*>(&w_raw.x));
    float2 fw1 = __half22float2(*reinterpret_cast<const half2*>(&w_raw.y));
    float2 fw2 = __half22float2(*reinterpret_cast<const half2*>(&w_raw.z));
    float2 fw3 = __half22float2(*reinterpret_cast<const half2*>(&w_raw.w));

    half2 r0 = __float22half2_rn(make_float2(scale * fi0.x * fw0.x, scale * fi0.y * fw0.y));
    half2 r1 = __float22half2_rn(make_float2(scale * fi1.x * fw1.x, scale * fi1.y * fw1.y));
    half2 r2 = __float22half2_rn(make_float2(scale * fi2.x * fw2.x, scale * fi2.y * fw2.y));
    half2 r3 = __float22half2_rn(make_float2(scale * fi3.x * fw3.x, scale * fi3.y * fw3.y));

    uint4 result;
    result.x = *reinterpret_cast<const unsigned int*>(&r0);
    result.y = *reinterpret_cast<const unsigned int*>(&r1);
    result.z = *reinterpret_cast<const unsigned int*>(&r2);
    result.w = *reinterpret_cast<const unsigned int*>(&r3);
    out_vec[i] = result;
  }
  // half2 remainder
  const half2* wei_h2 = reinterpret_cast<const half2*>(wei);
  half2* out_h2 = reinterpret_cast<half2*>(out);
  for (int i = (vec8_off >> 1) + tid; i < num_h2; i += BLOCK_DIM) {
    float2 fv = __half22float2(in_h2[i]);
    float2 fw = __half22float2(__ldg(wei_h2 + i));
    out_h2[i] = __float22half2_rn(make_float2(scale * fv.x * fw.x, scale * fv.y * fw.y));
  }
  for (int i = (num_h2 << 1) + tid; i < size; i += BLOCK_DIM) {
    out[i] = __float2half(scale * __half2float(in[i]) * __half2float(__ldg(wei + i)));
  }
}

// =======================================================================
// row_rmsnorm_pure_fp16_dim: Pure FP16 (batched, prefill path)
// 128-bit vectorized loads for maximum throughput
// =======================================================================
template <int32_t BLOCK_DIM>
static __global__ void row_rmsnorm_pure_fp16_dim(const half* __restrict__ in,
                                                  const half* __restrict__ wei,
                                                  half* __restrict__ out,
                                                  int dim_size, int size, float eps) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  if (bid >= dim_size) return;

  const half* block_in = in + static_cast<int64_t>(bid) * size;
  half* block_out = out + static_cast<int64_t>(bid) * size;

  // Phase 1: Sum of squares with 128-bit loads
  const int num_vec8 = size >> 3;
  const int vec8_off = num_vec8 << 3;
  const uint4* in_vec = reinterpret_cast<const uint4*>(block_in);

  float sum = 0.0f;

#pragma unroll 2
  for (int i = tid; i < num_vec8; i += BLOCK_DIM) {
    uint4 raw = in_vec[i];
    float2 f0 = __half22float2(*reinterpret_cast<const half2*>(&raw.x));
    float2 f1 = __half22float2(*reinterpret_cast<const half2*>(&raw.y));
    float2 f2 = __half22float2(*reinterpret_cast<const half2*>(&raw.z));
    float2 f3 = __half22float2(*reinterpret_cast<const half2*>(&raw.w));
    sum += f0.x * f0.x + f0.y * f0.y + f1.x * f1.x + f1.y * f1.y +
           f2.x * f2.x + f2.y * f2.y + f3.x * f3.x + f3.y * f3.y;
  }
  const int num_h2 = size >> 1;
  const half2* in_h2 = reinterpret_cast<const half2*>(block_in);
  for (int i = (vec8_off >> 1) + tid; i < num_h2; i += BLOCK_DIM) {
    float2 fv = __half22float2(in_h2[i]);
    sum += fv.x * fv.x + fv.y * fv.y;
  }
  for (int i = (num_h2 << 1) + tid; i < size; i += BLOCK_DIM) {
    float v = __half2float(block_in[i]);
    sum += v * v;
  }

  // Phase 2: Reduction
  sum = block_reduce_sum<BLOCK_DIM>(sum);
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

  // Phase 3: Normalize with 128-bit vectorized stores
  const uint4* wei_vec = reinterpret_cast<const uint4*>(wei);
  uint4* out_vec = reinterpret_cast<uint4*>(block_out);

#pragma unroll 2
  for (int i = tid; i < num_vec8; i += BLOCK_DIM) {
    uint4 in_raw = in_vec[i];
    uint4 w_raw = __ldg(wei_vec + i);

    float2 fi0 = __half22float2(*reinterpret_cast<const half2*>(&in_raw.x));
    float2 fi1 = __half22float2(*reinterpret_cast<const half2*>(&in_raw.y));
    float2 fi2 = __half22float2(*reinterpret_cast<const half2*>(&in_raw.z));
    float2 fi3 = __half22float2(*reinterpret_cast<const half2*>(&in_raw.w));
    float2 fw0 = __half22float2(*reinterpret_cast<const half2*>(&w_raw.x));
    float2 fw1 = __half22float2(*reinterpret_cast<const half2*>(&w_raw.y));
    float2 fw2 = __half22float2(*reinterpret_cast<const half2*>(&w_raw.z));
    float2 fw3 = __half22float2(*reinterpret_cast<const half2*>(&w_raw.w));

    half2 r0 = __float22half2_rn(make_float2(scale * fi0.x * fw0.x, scale * fi0.y * fw0.y));
    half2 r1 = __float22half2_rn(make_float2(scale * fi1.x * fw1.x, scale * fi1.y * fw1.y));
    half2 r2 = __float22half2_rn(make_float2(scale * fi2.x * fw2.x, scale * fi2.y * fw2.y));
    half2 r3 = __float22half2_rn(make_float2(scale * fi3.x * fw3.x, scale * fi3.y * fw3.y));

    uint4 result;
    result.x = *reinterpret_cast<const unsigned int*>(&r0);
    result.y = *reinterpret_cast<const unsigned int*>(&r1);
    result.z = *reinterpret_cast<const unsigned int*>(&r2);
    result.w = *reinterpret_cast<const unsigned int*>(&r3);
    out_vec[i] = result;
  }
  const half2* wei_h2 = reinterpret_cast<const half2*>(wei);
  half2* out_h2 = reinterpret_cast<half2*>(block_out);
  for (int i = (vec8_off >> 1) + tid; i < num_h2; i += BLOCK_DIM) {
    float2 fv = __half22float2(in_h2[i]);
    float2 fw = __half22float2(__ldg(wei_h2 + i));
    out_h2[i] = __float22half2_rn(make_float2(scale * fv.x * fw.x, scale * fv.y * fw.y));
  }
  for (int i = (num_h2 << 1) + tid; i < size; i += BLOCK_DIM) {
    block_out[i] =
        __float2half(scale * __half2float(block_in[i]) * __half2float(__ldg(wei + i)));
  }
}

// =======================================================================
// Host launch functions
// =======================================================================

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

  // Pure FP16 path (FP16 input, FP16 weight, FP16 output)
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

  // FP32 input + FP16 weight
  if (weight.data_type() == base::DataType::kDataTypeFp16) {
    const half* wei_ptr = reinterpret_cast<const half*>(weight.ptr<uint16_t>());
    if (stream) {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      row_rmsnorm_f32_fp16w<128><<<1, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    } else {
      row_rmsnorm_f32_fp16w<128><<<1, threads_num>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    }
  } else {
    // FP32 input + FP32 weight
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

  // Pure FP16 path
  if (input.data_type() == base::DataType::kDataTypeFp16 &&
      output.data_type() == base::DataType::kDataTypeFp16 &&
      weight.data_type() == base::DataType::kDataTypeFp16) {
    const half* in_ptr = reinterpret_cast<const half*>(input.ptr<uint16_t>());
    const half* wei_ptr = reinterpret_cast<const half*>(weight.ptr<uint16_t>());
    half* out_ptr = reinterpret_cast<half*>(const_cast<uint16_t*>(output.ptr<uint16_t>()));

    if (stream) {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      row_rmsnorm_pure_fp16_dim<128>
          <<<dim_size, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, dim_size, size, eps);
    } else {
      row_rmsnorm_pure_fp16_dim<128>
          <<<dim_size, threads_num>>>(in_ptr, wei_ptr, out_ptr, dim_size, size, eps);
    }
    return;
  }

  // FP32 path
  float* in_ptr = const_cast<float*>(input.ptr<float>());
  float* out_ptr = const_cast<float*>(output.ptr<float>());

  if (weight.data_type() == base::DataType::kDataTypeFp16) {
    const half* wei_ptr = reinterpret_cast<const half*>(weight.ptr<uint16_t>());
    if (stream) {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      row_rmsnorm_f32_fp16w_dim<128>
          <<<dim_size, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, dim_size, size, eps);
    } else {
      row_rmsnorm_f32_fp16w_dim<128>
          <<<dim_size, threads_num>>>(in_ptr, wei_ptr, out_ptr, dim_size, size, eps);
    }
  } else {
    float* wei_ptr = const_cast<float*>(weight.ptr<float>());
    if (stream) {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      row_rmsnorm_f32_dim<128>
          <<<dim_size, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, dim_size, size, eps);
    } else {
      row_rmsnorm_f32_dim<128>
          <<<dim_size, threads_num>>>(in_ptr, wei_ptr, out_ptr, dim_size, size, eps);
    }
  }
}

}  // namespace kernel
