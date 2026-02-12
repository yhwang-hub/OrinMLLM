// SwiGLU Kernel Benchmark
// Benchmarks FP32 and FP16 SwiGLU kernels with typical model sizes
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <chrono>
#include <vector>
#include <string>

// ============ Original Kernels (Baseline) ============

__global__ void swiglu_kernel_cu_fp32_orig(int size, const float* in1, const float* in2, float* out) {
  int tid = threadIdx.x;
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= size) return;
  
  extern __shared__ float shared_mem[];
  float* smem1 = shared_mem;
  float* smem2 = shared_mem + blockDim.x;
  
  smem1[tid] = in1[idx];
  smem2[tid] = in2[idx];
  __syncthreads();
  
  float value = 1.0f / (1.0f + exp(-smem1[tid]));
  smem1[tid] = smem1[tid] * value;
  out[idx] = smem1[tid] * smem2[tid];
}

__global__ void swiglu_kernel_cu_fp16_vec_orig(int size, const half* in1, const half* in2, half* out) {
  int idx = (threadIdx.x + blockDim.x * blockIdx.x) * 2;
  
  if (idx + 1 < size) {
    half2 v1 = *reinterpret_cast<const half2*>(in1 + idx);
    half2 v2 = *reinterpret_cast<const half2*>(in2 + idx);
    float2 f1 = __half22float2(v1);
    float2 f2 = __half22float2(v2);
    float sigmoid0 = 1.0f / (1.0f + expf(-f1.x));
    float sigmoid1 = 1.0f / (1.0f + expf(-f1.y));
    float silu0 = f1.x * sigmoid0;
    float silu1 = f1.y * sigmoid1;
    float2 result;
    result.x = silu0 * f2.x;
    result.y = silu1 * f2.y;
    *reinterpret_cast<half2*>(out + idx) = __float22half2_rn(result);
  } else if (idx < size) {
    float val1 = __half2float(in1[idx]);
    float val2 = __half2float(in2[idx]);
    float sigmoid = 1.0f / (1.0f + expf(-val1));
    out[idx] = __float2half(val1 * sigmoid * val2);
  }
}

// ============ Optimized Kernels ============

// FP32: Remove shared memory, use float4 vectorization, fast math
__global__ void __launch_bounds__(256)
swiglu_kernel_cu_fp32_opt(int size, const float* __restrict__ in1, const float* __restrict__ in2, float* __restrict__ out) {
  int idx = (threadIdx.x + blockDim.x * blockIdx.x) * 4;
  
  if (idx + 3 < size) {
    float4 v1 = *reinterpret_cast<const float4*>(in1 + idx);
    float4 v2 = *reinterpret_cast<const float4*>(in2 + idx);
    
    float s0 = __fdividef(1.0f, 1.0f + __expf(-v1.x));
    float s1 = __fdividef(1.0f, 1.0f + __expf(-v1.y));
    float s2 = __fdividef(1.0f, 1.0f + __expf(-v1.z));
    float s3 = __fdividef(1.0f, 1.0f + __expf(-v1.w));
    
    float4 result;
    result.x = v1.x * s0 * v2.x;
    result.y = v1.y * s1 * v2.y;
    result.z = v1.z * s2 * v2.z;
    result.w = v1.w * s3 * v2.w;
    
    *reinterpret_cast<float4*>(out + idx) = result;
  } else {
    // Handle tail elements
    for (int i = idx; i < size && i < idx + 4; i++) {
      float val1 = in1[i];
      float val2 = in2[i];
      float sigmoid = __fdividef(1.0f, 1.0f + __expf(-val1));
      out[i] = val1 * sigmoid * val2;
    }
  }
}

// FP16: float4 loads 8 half elements, process 8 per thread
__global__ void __launch_bounds__(256)
swiglu_kernel_cu_fp16_vec_opt(int size, const half* __restrict__ in1, const half* __restrict__ in2, half* __restrict__ out) {
  int idx = (threadIdx.x + blockDim.x * blockIdx.x) * 8;
  
  if (idx + 7 < size) {
    // Load 8 half elements (128 bits) via float4
    float4 raw1 = *reinterpret_cast<const float4*>(in1 + idx);
    float4 raw2 = *reinterpret_cast<const float4*>(in2 + idx);
    
    half2* h1 = reinterpret_cast<half2*>(&raw1);
    half2* h2 = reinterpret_cast<half2*>(&raw2);
    
    float4 out_raw;
    half2* h_out = reinterpret_cast<half2*>(&out_raw);
    
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      float2 f1 = __half22float2(h1[i]);
      float2 f2 = __half22float2(h2[i]);
      
      float sig0 = __fdividef(1.0f, 1.0f + __expf(-f1.x));
      float sig1 = __fdividef(1.0f, 1.0f + __expf(-f1.y));
      
      float2 r;
      r.x = f1.x * sig0 * f2.x;
      r.y = f1.y * sig1 * f2.y;
      
      h_out[i] = __float22half2_rn(r);
    }
    
    *reinterpret_cast<float4*>(out + idx) = out_raw;
  } else {
    // Handle tail
    for (int i = idx; i < size && i < idx + 8; i++) {
      float val1 = __half2float(in1[i]);
      float val2 = __half2float(in2[i]);
      float sigmoid = __fdividef(1.0f, 1.0f + __expf(-val1));
      out[i] = __float2half(val1 * sigmoid * val2);
    }
  }
}

// ============ Benchmark Utilities ============

template <typename T>
void fill_random(T* d_ptr, int size) {
  std::vector<float> host(size);
  for (int i = 0; i < size; i++) {
    host[i] = (float(rand()) / RAND_MAX) * 2.0f - 1.0f;
  }
  if constexpr (std::is_same_v<T, float>) {
    cudaMemcpy(d_ptr, host.data(), size * sizeof(float), cudaMemcpyHostToDevice);
  } else {
    std::vector<half> host_half(size);
    for (int i = 0; i < size; i++) {
      host_half[i] = __float2half(host[i]);
    }
    cudaMemcpy(d_ptr, host_half.data(), size * sizeof(half), cudaMemcpyHostToDevice);
  }
}

struct BenchResult {
  float avg_us;
  float min_us;
  float max_us;
};

template <typename Func>
BenchResult benchmark(Func func, int warmup = 50, int iters = 200) {
  // Warmup
  for (int i = 0; i < warmup; i++) func();
  cudaDeviceSynchronize();
  
  std::vector<float> times(iters);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  for (int i = 0; i < iters; i++) {
    cudaEventRecord(start);
    func();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&times[i], start, stop);
    times[i] *= 1000.0f; // ms -> us
  }
  
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  
  BenchResult r;
  r.min_us = times[0];
  r.max_us = times[0];
  double sum = 0;
  for (float t : times) {
    sum += t;
    if (t < r.min_us) r.min_us = t;
    if (t > r.max_us) r.max_us = t;
  }
  r.avg_us = sum / iters;
  return r;
}

void bench_fp32(int size, const char* label) {
  float *d_in1, *d_in2, *d_out;
  cudaMalloc(&d_in1, size * sizeof(float));
  cudaMalloc(&d_in2, size * sizeof(float));
  cudaMalloc(&d_out, size * sizeof(float));
  fill_random<float>(d_in1, size);
  fill_random<float>(d_in2, size);
  
  // Original
  int threads_orig = 128;
  int blocks_orig = (size + threads_orig - 1) / threads_orig;
  size_t shmem = threads_orig * sizeof(float) * 2;
  
  auto r_orig = benchmark([&]() {
    swiglu_kernel_cu_fp32_orig<<<blocks_orig, threads_orig, shmem>>>(size, d_in1, d_in2, d_out);
  });
  
  // Optimized
  int threads_opt = 256;
  int blocks_opt = (size + threads_opt * 4 - 1) / (threads_opt * 4);
  
  auto r_opt = benchmark([&]() {
    swiglu_kernel_cu_fp32_opt<<<blocks_opt, threads_opt>>>(size, d_in1, d_in2, d_out);
  });
  
  printf("FP32 %s (size=%d):\n", label, size);
  printf("  Original: avg=%.2f us, min=%.2f us, max=%.2f us  [blocks=%d, threads=%d, shmem=%zu]\n",
         r_orig.avg_us, r_orig.min_us, r_orig.max_us, blocks_orig, threads_orig, shmem);
  printf("  Optimized: avg=%.2f us, min=%.2f us, max=%.2f us  [blocks=%d, threads=%d, shmem=0]\n",
         r_opt.avg_us, r_opt.min_us, r_opt.max_us, blocks_opt, threads_opt);
  printf("  Speedup: %.2fx\n\n", r_orig.avg_us / r_opt.avg_us);
  
  cudaFree(d_in1);
  cudaFree(d_in2);
  cudaFree(d_out);
}

void bench_fp16(int size, const char* label) {
  half *d_in1, *d_in2, *d_out;
  cudaMalloc(&d_in1, size * sizeof(half));
  cudaMalloc(&d_in2, size * sizeof(half));
  cudaMalloc(&d_out, size * sizeof(half));
  fill_random<half>(d_in1, size);
  fill_random<half>(d_in2, size);
  
  // Original (half2, 2 elements per thread)
  int threads_orig = 256;
  int blocks_orig = (size + threads_orig * 2 - 1) / (threads_orig * 2);
  
  auto r_orig = benchmark([&]() {
    swiglu_kernel_cu_fp16_vec_orig<<<blocks_orig, threads_orig>>>(size, d_in1, d_in2, d_out);
  });
  
  // Optimized (float4, 8 elements per thread)
  int threads_opt = 256;
  int blocks_opt = (size + threads_opt * 8 - 1) / (threads_opt * 8);
  
  auto r_opt = benchmark([&]() {
    swiglu_kernel_cu_fp16_vec_opt<<<blocks_opt, threads_opt>>>(size, d_in1, d_in2, d_out);
  });
  
  printf("FP16 %s (size=%d):\n", label, size);
  printf("  Original: avg=%.2f us, min=%.2f us, max=%.2f us  [blocks=%d, threads=%d]\n",
         r_orig.avg_us, r_orig.min_us, r_orig.max_us, blocks_orig, threads_orig);
  printf("  Optimized: avg=%.2f us, min=%.2f us, max=%.2f us  [blocks=%d, threads=%d]\n",
         r_opt.avg_us, r_opt.min_us, r_opt.max_us, blocks_opt, threads_opt);
  printf("  Speedup: %.2fx\n\n", r_orig.avg_us / r_opt.avg_us);
  
  cudaFree(d_in1);
  cudaFree(d_in2);
  cudaFree(d_out);
}

int main() {
  printf("=== SwiGLU Kernel Benchmark on Orin (SM 8.7) ===\n\n");
  
  // Typical model sizes (decode: 1 token)
  bench_fp32(18944, "Qwen2.5-7B decode");
  bench_fp32(12288, "Qwen3-8B decode");
  bench_fp16(18944, "Qwen2.5-7B decode");
  bench_fp16(12288, "Qwen3-8B decode");
  
  // Prefill sizes (e.g., 128 tokens)
  bench_fp32(18944 * 128, "Qwen2.5-7B prefill-128");
  bench_fp32(12288 * 128, "Qwen3-8B prefill-128");
  bench_fp16(18944 * 128, "Qwen2.5-7B prefill-128");
  bench_fp16(12288 * 128, "Qwen3-8B prefill-128");
  
  printf("=== Benchmark Complete ===\n");
  return 0;
}
