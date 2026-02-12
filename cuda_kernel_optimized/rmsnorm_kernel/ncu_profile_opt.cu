#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
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

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

template <int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float val) {
  constexpr int NUM_WARPS = BLOCK_SIZE / 32;
  __shared__ float smem_reduce[NUM_WARPS];
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  val = warp_reduce_sum(val);
  if (lane == 0) smem_reduce[wid] = val;
  __syncthreads();
  val = (threadIdx.x < NUM_WARPS) ? smem_reduce[threadIdx.x] : 0.0f;
  if (wid == 0) val = warp_reduce_sum(val);
  if (threadIdx.x == 0) smem_reduce[0] = val;
  __syncthreads();
  return smem_reduce[0];
}

template <int32_t BLOCK_DIM>
__global__ void opt_row_rmsnorm_f32(float* __restrict__ in, float* __restrict__ wei,
                                     float* __restrict__ out, int size, float eps) {
  const int tid = threadIdx.x;
  const int pack_num = size >> 2;
  const int pack_off = pack_num << 2;
  float sum = 0.0f;
  const float4* in_pack = reinterpret_cast<const float4*>(in);
#pragma unroll 4
  for (int i = tid; i < pack_num; i += BLOCK_DIM) {
    float4 v = in_pack[i];
    sum += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
  }
  for (int i = pack_off + tid; i < size; i += BLOCK_DIM) sum += in[i]*in[i];
  sum = block_reduce_sum<BLOCK_DIM>(sum);
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);
  const float4* wei_pack = reinterpret_cast<const float4*>(wei);
  float4* out_pack = reinterpret_cast<float4*>(out);
#pragma unroll 4
  for (int i = tid; i < pack_num; i += BLOCK_DIM) {
    float4 v = in_pack[i];
    float4 w = __ldg(wei_pack + i);
    out_pack[i] = make_float4(scale*v.x*w.x, scale*v.y*w.y, scale*v.z*w.z, scale*v.w*w.w);
  }
  for (int i = pack_off + tid; i < size; i += BLOCK_DIM) out[i] = wei[i]*in[i]*scale;
}

template <int32_t BLOCK_DIM>
__global__ void opt_row_rmsnorm_f32_dim(float* __restrict__ in, float* __restrict__ wei,
                                         float* __restrict__ out, int dim_size, int size, float eps) {
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
    sum += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
  }
  for (int i = pack_off + tid; i < size; i += BLOCK_DIM) sum += block_in[i]*block_in[i];
  sum = block_reduce_sum<BLOCK_DIM>(sum);
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);
  const float4* wei_pack = reinterpret_cast<const float4*>(wei);
  float4* out_pack = reinterpret_cast<float4*>(block_out);
#pragma unroll 4
  for (int i = tid; i < pack_num; i += BLOCK_DIM) {
    float4 v = in_pack[i];
    float4 w = __ldg(wei_pack + i);
    out_pack[i] = make_float4(scale*v.x*w.x, scale*v.y*w.y, scale*v.z*w.z, scale*v.w*w.w);
  }
  for (int i = pack_off + tid; i < size; i += BLOCK_DIM) block_out[i] = wei[i]*block_in[i]*scale;
}

template <int32_t BLOCK_DIM>
__global__ void opt_row_rmsnorm_pure_fp16(const half* __restrict__ in, const half* __restrict__ wei,
                                           half* __restrict__ out, int size, float eps) {
  const int tid = threadIdx.x;
  const int num_vec8 = size >> 3;
  const int vec8_off = num_vec8 << 3;
  const uint4* in_vec = reinterpret_cast<const uint4*>(in);
  float sum = 0.0f;
#pragma unroll 2
  for (int i = tid; i < num_vec8; i += BLOCK_DIM) {
    uint4 raw = in_vec[i];
    float2 f0=__half22float2(*reinterpret_cast<const half2*>(&raw.x));
    float2 f1=__half22float2(*reinterpret_cast<const half2*>(&raw.y));
    float2 f2=__half22float2(*reinterpret_cast<const half2*>(&raw.z));
    float2 f3=__half22float2(*reinterpret_cast<const half2*>(&raw.w));
    sum += f0.x*f0.x+f0.y*f0.y+f1.x*f1.x+f1.y*f1.y+f2.x*f2.x+f2.y*f2.y+f3.x*f3.x+f3.y*f3.y;
  }
  const int num_h2 = size >> 1;
  const half2* in_h2 = reinterpret_cast<const half2*>(in);
  for (int i=(vec8_off>>1)+tid; i<num_h2; i+=BLOCK_DIM) {
    float2 fv=__half22float2(in_h2[i]); sum+=fv.x*fv.x+fv.y*fv.y;
  }
  for (int i=(num_h2<<1)+tid; i<size; i+=BLOCK_DIM) { float v=__half2float(in[i]); sum+=v*v; }
  sum = block_reduce_sum<BLOCK_DIM>(sum);
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);
  const uint4* wei_vec = reinterpret_cast<const uint4*>(wei);
  uint4* out_vec = reinterpret_cast<uint4*>(out);
#pragma unroll 2
  for (int i = tid; i < num_vec8; i += BLOCK_DIM) {
    uint4 in_raw = in_vec[i];
    uint4 w_raw = __ldg(wei_vec + i);
    float2 fi0=__half22float2(*reinterpret_cast<const half2*>(&in_raw.x));
    float2 fi1=__half22float2(*reinterpret_cast<const half2*>(&in_raw.y));
    float2 fi2=__half22float2(*reinterpret_cast<const half2*>(&in_raw.z));
    float2 fi3=__half22float2(*reinterpret_cast<const half2*>(&in_raw.w));
    float2 fw0=__half22float2(*reinterpret_cast<const half2*>(&w_raw.x));
    float2 fw1=__half22float2(*reinterpret_cast<const half2*>(&w_raw.y));
    float2 fw2=__half22float2(*reinterpret_cast<const half2*>(&w_raw.z));
    float2 fw3=__half22float2(*reinterpret_cast<const half2*>(&w_raw.w));
    half2 r0=__float22half2_rn(make_float2(scale*fi0.x*fw0.x,scale*fi0.y*fw0.y));
    half2 r1=__float22half2_rn(make_float2(scale*fi1.x*fw1.x,scale*fi1.y*fw1.y));
    half2 r2=__float22half2_rn(make_float2(scale*fi2.x*fw2.x,scale*fi2.y*fw2.y));
    half2 r3=__float22half2_rn(make_float2(scale*fi3.x*fw3.x,scale*fi3.y*fw3.y));
    uint4 result;
    result.x=*reinterpret_cast<const unsigned int*>(&r0);
    result.y=*reinterpret_cast<const unsigned int*>(&r1);
    result.z=*reinterpret_cast<const unsigned int*>(&r2);
    result.w=*reinterpret_cast<const unsigned int*>(&r3);
    out_vec[i] = result;
  }
  const half2* wei_h2=reinterpret_cast<const half2*>(wei);
  half2* out_h2=reinterpret_cast<half2*>(out);
  for (int i=(vec8_off>>1)+tid; i<num_h2; i+=BLOCK_DIM) {
    float2 fv=__half22float2(in_h2[i]); float2 fw=__half22float2(__ldg(wei_h2+i));
    out_h2[i]=__float22half2_rn(make_float2(scale*fv.x*fw.x,scale*fv.y*fw.y));
  }
  for (int i=(num_h2<<1)+tid; i<size; i+=BLOCK_DIM)
    out[i]=__float2half(scale*__half2float(in[i])*__half2float(__ldg(wei+i)));
}

int main() {
  float *d_in_f32, *d_wei_f32, *d_out_f32;
  CUDA_CHECK(cudaMalloc(&d_in_f32, HIDDEN_SIZE*64*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_wei_f32, HIDDEN_SIZE*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out_f32, HIDDEN_SIZE*64*sizeof(float)));
  half *d_in_fp16, *d_wei_fp16, *d_out_fp16;
  CUDA_CHECK(cudaMalloc(&d_in_fp16, HIDDEN_SIZE*64*sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_wei_fp16, HIDDEN_SIZE*sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_out_fp16, HIDDEN_SIZE*64*sizeof(half)));

  float* h_data = new float[HIDDEN_SIZE*64];
  for (int i=0; i<HIDDEN_SIZE*64; i++) h_data[i]=(float)rand()/RAND_MAX-0.5f;
  CUDA_CHECK(cudaMemcpy(d_in_f32,h_data,HIDDEN_SIZE*64*sizeof(float),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_wei_f32,h_data,HIDDEN_SIZE*sizeof(float),cudaMemcpyHostToDevice));
  half* h_fp16=new half[HIDDEN_SIZE*64];
  for (int i=0;i<HIDDEN_SIZE*64;i++) h_fp16[i]=__float2half(h_data[i]);
  CUDA_CHECK(cudaMemcpy(d_in_fp16,h_fp16,HIDDEN_SIZE*64*sizeof(half),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_wei_fp16,h_fp16,HIDDEN_SIZE*sizeof(half),cudaMemcpyHostToDevice));

  const float eps = 1e-6f;
  // Launch each kernel once for NCU
  opt_row_rmsnorm_f32<128><<<1,128>>>(d_in_f32,d_wei_f32,d_out_f32,HIDDEN_SIZE,eps);
  CUDA_CHECK(cudaDeviceSynchronize());
  opt_row_rmsnorm_f32_dim<128><<<4,128>>>(d_in_f32,d_wei_f32,d_out_f32,4,HIDDEN_SIZE,eps);
  CUDA_CHECK(cudaDeviceSynchronize());
  opt_row_rmsnorm_pure_fp16<128><<<1,128>>>(d_in_fp16,d_wei_fp16,d_out_fp16,HIDDEN_SIZE,eps);
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("NCU profiling kernels launched.\n");

  CUDA_CHECK(cudaFree(d_in_f32)); CUDA_CHECK(cudaFree(d_wei_f32)); CUDA_CHECK(cudaFree(d_out_f32));
  CUDA_CHECK(cudaFree(d_in_fp16)); CUDA_CHECK(cudaFree(d_wei_fp16)); CUDA_CHECK(cudaFree(d_out_fp16));
  delete[] h_data; delete[] h_fp16;
  return 0;
}
