#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/block/block_reduce.cuh>
#include <cstdio>
#include <cmath>
#include <cstdlib>

#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
    } \
  } while (0)

constexpr int HIDDEN_SIZE = 4096;

// ============ Original kernels (for NCU) ============

template <int32_t BLOCK_DIM>
__global__ void row_rmsnorm_f32(float* in, float* wei, float* out, int size, float eps) {
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
  for (int i = pack_off + tid; i < size; i += blockDim.x) sum += in[i] * in[i];

  using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) shared_val = sum;
  __syncthreads();
  sum = shared_val;
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

  float4* wei_pack = reinterpret_cast<float4*>(wei);
  float4* out_pack = reinterpret_cast<float4*>(out);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_f4 = *(in_pack + i);
    float4 w_f4 = *(wei_pack + i);
    *(out_pack + i) = make_float4(scale*in_f4.x*w_f4.x, scale*in_f4.y*w_f4.y,
                                   scale*in_f4.z*w_f4.z, scale*in_f4.w*w_f4.w);
  }
  for (int i = pack_off + tid; i < size; i += blockDim.x) out[i] = wei[i]*in[i]*scale;
}

__global__ void row_rmsnorm_f32_dim(float* in, float* wei, float* out, int dim_size, int size, float eps) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  if (bid >= dim_size) return;
  float* block_in = in + bid * size;
  float* block_out = out + bid * size;
  constexpr int pack_size = 4;
  const int pack_num = size / pack_size;
  const int pack_off = pack_size * pack_num;

  float sum = 0.0f;
  float4* in_pack = reinterpret_cast<float4*>(block_in);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 f4 = *(in_pack + i);
    sum += f4.x*f4.x + f4.y*f4.y + f4.z*f4.z + f4.w*f4.w;
  }
  for (int i = pack_off + tid; i < size; i += blockDim.x) sum += block_in[i]*block_in[i];

  using BlockReduce = cub::BlockReduce<float, 128>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) shared_val = sum;
  __syncthreads();
  sum = shared_val;
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

  float4* wei_pack = reinterpret_cast<float4*>(wei);
  float4* out_pack = reinterpret_cast<float4*>(block_out);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 f4 = *(in_pack + i);
    float4 w4 = *(wei_pack + i);
    *(out_pack + i) = make_float4(scale*f4.x*w4.x, scale*f4.y*w4.y,
                                   scale*f4.z*w4.z, scale*f4.w*w4.w);
  }
  for (int i = pack_off + tid; i < size; i += blockDim.x) block_out[i] = wei[i]*block_in[i]*scale;
}

template <int32_t BLOCK_DIM>
__global__ void row_rmsnorm_pure_fp16(const half* in, const half* wei, half* out, int size, float eps) {
  const int tid = threadIdx.x;
  const int num_h2 = size / 2;
  const half2* in_h2 = reinterpret_cast<const half2*>(in);
  float sum = 0.0f;
  for (int i = tid; i < num_h2; i += blockDim.x) {
    float2 fv = __half22float2(in_h2[i]);
    sum += fv.x*fv.x + fv.y*fv.y;
  }
  const int base = num_h2*2;
  for (int i = base+tid; i < size; i += blockDim.x) { float v = __half2float(in[i]); sum += v*v; }

  using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) shared_val = sum;
  __syncthreads();
  sum = shared_val;
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

  const half2* wei_h2 = reinterpret_cast<const half2*>(wei);
  half2* out_h2 = reinterpret_cast<half2*>(out);
  for (int i = tid; i < num_h2; i += blockDim.x) {
    float2 fv = __half22float2(in_h2[i]);
    float2 fw = __half22float2(wei_h2[i]);
    out_h2[i] = __float22half2_rn(make_float2(scale*fv.x*fw.x, scale*fv.y*fw.y));
  }
  for (int i = base+tid; i < size; i += blockDim.x)
    out[i] = __float2half(scale * __half2float(in[i]) * __half2float(wei[i]));
}

int main() {
  float *d_in_f32, *d_wei_f32, *d_out_f32;
  CUDA_CHECK(cudaMalloc(&d_in_f32, HIDDEN_SIZE * 64 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_wei_f32, HIDDEN_SIZE * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out_f32, HIDDEN_SIZE * 64 * sizeof(float)));
  
  half *d_in_fp16, *d_wei_fp16, *d_out_fp16;
  CUDA_CHECK(cudaMalloc(&d_in_fp16, HIDDEN_SIZE * 64 * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_wei_fp16, HIDDEN_SIZE * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_out_fp16, HIDDEN_SIZE * 64 * sizeof(half)));
  
  float* h_data = new float[HIDDEN_SIZE * 64];
  for (int i = 0; i < HIDDEN_SIZE * 64; i++) h_data[i] = (float)rand()/RAND_MAX - 0.5f;
  CUDA_CHECK(cudaMemcpy(d_in_f32, h_data, HIDDEN_SIZE*64*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_wei_f32, h_data, HIDDEN_SIZE*sizeof(float), cudaMemcpyHostToDevice));
  
  half* h_fp16 = new half[HIDDEN_SIZE*64];
  for (int i=0; i<HIDDEN_SIZE*64; i++) h_fp16[i] = __float2half(h_data[i]);
  CUDA_CHECK(cudaMemcpy(d_in_fp16, h_fp16, HIDDEN_SIZE*64*sizeof(half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_wei_fp16, h_fp16, HIDDEN_SIZE*sizeof(half), cudaMemcpyHostToDevice));
  
  const float eps = 1e-6f;

  // Launch each kernel exactly once for NCU profiling
  row_rmsnorm_f32<128><<<1, 128>>>(d_in_f32, d_wei_f32, d_out_f32, HIDDEN_SIZE, eps);
  CUDA_CHECK(cudaDeviceSynchronize());
  
  row_rmsnorm_f32_dim<<<4, 128>>>(d_in_f32, d_wei_f32, d_out_f32, 4, HIDDEN_SIZE, eps);
  CUDA_CHECK(cudaDeviceSynchronize());
  
  row_rmsnorm_pure_fp16<128><<<1, 128>>>(d_in_fp16, d_wei_fp16, d_out_fp16, HIDDEN_SIZE, eps);
  CUDA_CHECK(cudaDeviceSynchronize());
  
  printf("NCU profiling kernels launched.\n");
  
  CUDA_CHECK(cudaFree(d_in_f32));
  CUDA_CHECK(cudaFree(d_wei_f32));
  CUDA_CHECK(cudaFree(d_out_f32));
  CUDA_CHECK(cudaFree(d_in_fp16));
  CUDA_CHECK(cudaFree(d_wei_fp16));
  CUDA_CHECK(cudaFree(d_out_fp16));
  delete[] h_data;
  delete[] h_fp16;
  return 0;
}
