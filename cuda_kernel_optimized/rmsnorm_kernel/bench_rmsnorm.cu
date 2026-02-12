#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/block/block_reduce.cuh>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <functional>

// Timing helper
#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
    } \
  } while (0)

// ======================== Benchmark Parameters ========================
// Typical Qwen3-8B hidden_size = 4096
constexpr int HIDDEN_SIZE = 4096;
constexpr int BATCH_SIZES[] = {1, 4, 16, 64};
constexpr int NUM_BATCH_SIZES = 4;
constexpr int WARMUP_ITERS = 50;
constexpr int BENCH_ITERS = 200;

// ======================== Include kernel implementations ========================
// We'll directly include the kernel file to benchmark its internal kernels
// For NCU profiling, we create wrapper kernels

// FP32 kernel
template <int32_t BLOCK_DIM>
__global__ void bench_row_rmsnorm_f32(float* in, float* wei, float* out, int size, float eps) {
  const int tid = threadIdx.x;
  constexpr int pack_size = 4;
  const int pack_num = size / pack_size;
  const int pack_off = pack_size * pack_num;

  float sum = 0.0f;
  float4* in_pack = reinterpret_cast<float4*>(in);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    sum += in_float4.x * in_float4.x + in_float4.y * in_float4.y +
           in_float4.z * in_float4.z + in_float4.w * in_float4.w;
  }
  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    sum += in[i] * in[i];
  }

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
  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    out[i] = wei[i] * in[i] * scale;
  }
}

// FP32 dim kernel 
__global__ void bench_row_rmsnorm_f32_dim(float* in, float* wei, float* out, int dim_size, int size, float eps) {
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
    float4 in_f4 = *(in_pack + i);
    sum += in_f4.x*in_f4.x + in_f4.y*in_f4.y + in_f4.z*in_f4.z + in_f4.w*in_f4.w;
  }
  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    sum += block_in[i] * block_in[i];
  }

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
    float4 in_f4 = *(in_pack + i);
    float4 w_f4 = *(wei_pack + i);
    *(out_pack + i) = make_float4(scale*in_f4.x*w_f4.x, scale*in_f4.y*w_f4.y,
                                   scale*in_f4.z*w_f4.z, scale*in_f4.w*w_f4.w);
  }
  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    block_out[i] = wei[i] * block_in[i] * scale;
  }
}

// FP16 kernel
template <int32_t BLOCK_DIM>
__global__ void bench_row_rmsnorm_pure_fp16(const half* in, const half* wei, half* out, int size, float eps) {
  const int tid = threadIdx.x;
  const int num_h2 = size / 2;
  const half2* in_h2 = reinterpret_cast<const half2*>(in);
  
  float sum = 0.0f;
  for (int i = tid; i < num_h2; i += blockDim.x) {
    half2 val = in_h2[i];
    float2 fval = __half22float2(val);
    sum += fval.x * fval.x + fval.y * fval.y;
  }
  const int base = num_h2 * 2;
  for (int i = base + tid; i < size; i += blockDim.x) {
    float fval = __half2float(in[i]);
    sum += fval * fval;
  }

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

float benchmark_kernel_ms(std::function<void()> kernel_launch, int warmup, int iters) {
  for (int i = 0; i < warmup; i++) kernel_launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; i++) kernel_launch();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  
  float ms;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / iters;
}

int main() {
  printf("=== RMSNorm Kernel Benchmark (Baseline) ===\n");
  printf("Hidden Size: %d\n\n", HIDDEN_SIZE);
  
  // Allocate FP32 buffers
  float *d_in_f32, *d_wei_f32, *d_out_f32;
  CUDA_CHECK(cudaMalloc(&d_in_f32, HIDDEN_SIZE * 64 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_wei_f32, HIDDEN_SIZE * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out_f32, HIDDEN_SIZE * 64 * sizeof(float)));
  
  // Allocate FP16 buffers
  half *d_in_fp16, *d_wei_fp16, *d_out_fp16;
  CUDA_CHECK(cudaMalloc(&d_in_fp16, HIDDEN_SIZE * 64 * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_wei_fp16, HIDDEN_SIZE * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_out_fp16, HIDDEN_SIZE * 64 * sizeof(half)));
  
  // Initialize with random data
  float* h_data = new float[HIDDEN_SIZE * 64];
  for (int i = 0; i < HIDDEN_SIZE * 64; i++) h_data[i] = (float)rand() / RAND_MAX - 0.5f;
  CUDA_CHECK(cudaMemcpy(d_in_f32, h_data, HIDDEN_SIZE * 64 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_wei_f32, h_data, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  
  // Convert to FP16
  half* h_fp16 = new half[HIDDEN_SIZE * 64];
  for (int i = 0; i < HIDDEN_SIZE * 64; i++) h_fp16[i] = __float2half(h_data[i]);
  CUDA_CHECK(cudaMemcpy(d_in_fp16, h_fp16, HIDDEN_SIZE * 64 * sizeof(half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_wei_fp16, h_fp16, HIDDEN_SIZE * sizeof(half), cudaMemcpyHostToDevice));
  
  const float eps = 1e-6f;
  
  // Benchmark row_rmsnorm_f32 (single row)
  printf("--- row_rmsnorm_f32 (1 row, %d elements) ---\n", HIDDEN_SIZE);
  auto f32_time = benchmark_kernel_ms([&](){
    bench_row_rmsnorm_f32<128><<<1, 128>>>(d_in_f32, d_wei_f32, d_out_f32, HIDDEN_SIZE, eps);
  }, WARMUP_ITERS, BENCH_ITERS);
  printf("  Time: %.3f us\n\n", f32_time * 1000);
  
  // Benchmark row_rmsnorm_f32_dim (batched)
  for (int bi = 0; bi < NUM_BATCH_SIZES; bi++) {
    int bs = BATCH_SIZES[bi];
    printf("--- row_rmsnorm_f32_dim (batch=%d, hidden=%d) ---\n", bs, HIDDEN_SIZE);
    auto dim_time = benchmark_kernel_ms([&](){
      bench_row_rmsnorm_f32_dim<<<bs, 128>>>(d_in_f32, d_wei_f32, d_out_f32, bs, HIDDEN_SIZE, eps);
    }, WARMUP_ITERS, BENCH_ITERS);
    printf("  Time: %.3f us\n\n", dim_time * 1000);
  }
  
  // Benchmark row_rmsnorm_pure_fp16 (single row)
  printf("--- row_rmsnorm_pure_fp16 (1 row, %d elements) ---\n", HIDDEN_SIZE);
  auto fp16_time = benchmark_kernel_ms([&](){
    bench_row_rmsnorm_pure_fp16<128><<<1, 128>>>(d_in_fp16, d_wei_fp16, d_out_fp16, HIDDEN_SIZE, eps);
  }, WARMUP_ITERS, BENCH_ITERS);
  printf("  Time: %.3f us\n\n", fp16_time * 1000);
  
  // Benchmark row_rmsnorm_pure_fp16_dim (batched) - reuse bench_row_rmsnorm_pure_fp16 with blockIdx
  // (simplified: use same kernel logic)
  
  // Cleanup
  CUDA_CHECK(cudaFree(d_in_f32));
  CUDA_CHECK(cudaFree(d_wei_f32));
  CUDA_CHECK(cudaFree(d_out_f32));
  CUDA_CHECK(cudaFree(d_in_fp16));
  CUDA_CHECK(cudaFree(d_wei_fp16));
  CUDA_CHECK(cudaFree(d_out_fp16));
  delete[] h_data;
  delete[] h_fp16;
  
  printf("=== Benchmark Complete ===\n");
  return 0;
}
