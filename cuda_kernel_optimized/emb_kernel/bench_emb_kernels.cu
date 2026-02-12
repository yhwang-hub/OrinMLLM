// Standalone benchmark for embedding kernels - original vs optimized
// Compile: nvcc -O3 -arch=sm_87 -o bench_emb bench_emb_kernels.cu
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// ======================= ORIGINAL KERNELS ==========================
namespace original {

__global__ void emb_kernel_cu_fp32(int32_t vocab_size, int32_t token_num, int32_t weight_dim,
                                   const int32_t* input_ptr, const float* weight_ptr,
                                   float* output_ptr) {
  int32_t token_idx = blockIdx.x;
  if (token_idx >= token_num) return;
  int32_t token = input_ptr[token_idx];
  if (token >= vocab_size) return;
  float* out = output_ptr + token_idx * weight_dim;
  const float* w = weight_ptr + token * weight_dim;
  for (int32_t i = threadIdx.x; i < weight_dim; i += blockDim.x) {
    out[i] = w[i];
  }
}

__global__ void emb_kernel_cu_fp16(int32_t vocab_size, int32_t token_num, int32_t weight_dim,
                                   const int32_t* input_ptr, const half* weight_ptr,
                                   float* output_ptr) {
  int32_t token_idx = blockIdx.x;
  if (token_idx >= token_num) return;
  int32_t token = input_ptr[token_idx];
  if (token >= vocab_size) return;
  float* out = output_ptr + token_idx * weight_dim;
  const half* w = weight_ptr + token * weight_dim;
  const int vec_size = 2;
  const int num_vecs = weight_dim / vec_size;
  const half2* wh2 = reinterpret_cast<const half2*>(w);
  for (int32_t i = threadIdx.x; i < num_vecs; i += blockDim.x) {
    half2 wv = wh2[i];
    out[i * 2] = __half2float(wv.x);
    out[i * 2 + 1] = __half2float(wv.y);
  }
  for (int32_t i = num_vecs * vec_size + threadIdx.x; i < weight_dim; i += blockDim.x) {
    out[i] = __half2float(w[i]);
  }
}

__global__ void emb_kernel_cu_pure_fp16_impl(int32_t vocab_size, int32_t token_num, int32_t weight_dim,
                                              const int32_t* input_ptr, const half* weight_ptr,
                                              half* output_ptr) {
  int32_t token_idx = blockIdx.x;
  if (token_idx >= token_num) return;
  int32_t token = input_ptr[token_idx];
  if (token >= vocab_size) return;
  half* out = output_ptr + static_cast<int64_t>(token_idx) * weight_dim;
  const half* w = weight_ptr + static_cast<int64_t>(token) * weight_dim;
  const int vec_size = 2;
  const int num_vecs = weight_dim / vec_size;
  const half2* wh2 = reinterpret_cast<const half2*>(w);
  half2* oh2 = reinterpret_cast<half2*>(out);
  for (int32_t i = threadIdx.x; i < num_vecs; i += blockDim.x) {
    oh2[i] = wh2[i];
  }
  for (int32_t i = num_vecs * vec_size + threadIdx.x; i < weight_dim; i += blockDim.x) {
    out[i] = w[i];
  }
}

}  // namespace original

// ======================= OPTIMIZED KERNELS ==========================
namespace optimized {

__global__ void emb_kernel_cu_fp32(int32_t vocab_size, int32_t token_num, int32_t weight_dim,
                                   const int32_t* __restrict__ input_ptr,
                                   const float* __restrict__ weight_ptr,
                                   float* __restrict__ output_ptr) {
  int32_t token_idx = blockIdx.x;
  if (token_idx >= token_num) return;
  int32_t token = __ldg(input_ptr + token_idx);
  if (token >= vocab_size) return;
  float* out = output_ptr + token_idx * weight_dim;
  const float* w = weight_ptr + token * weight_dim;
  const int VEC = 4;
  const int num_vecs = weight_dim / VEC;
  for (int32_t i = threadIdx.x; i < num_vecs; i += blockDim.x) {
    float4 wv = __ldg(reinterpret_cast<const float4*>(w) + i);
    reinterpret_cast<float4*>(out)[i] = wv;
  }
  for (int32_t i = num_vecs * VEC + threadIdx.x; i < weight_dim; i += blockDim.x) {
    out[i] = __ldg(w + i);
  }
}

__global__ void emb_kernel_cu_fp16(int32_t vocab_size, int32_t token_num, int32_t weight_dim,
                                   const int32_t* __restrict__ input_ptr,
                                   const half* __restrict__ weight_ptr,
                                   float* __restrict__ output_ptr) {
  int32_t token_idx = blockIdx.x;
  if (token_idx >= token_num) return;
  int32_t token = __ldg(input_ptr + token_idx);
  if (token >= vocab_size) return;
  float* out = output_ptr + token_idx * weight_dim;
  const half* w = weight_ptr + token * weight_dim;
  const int VEC = 8;
  const int num_vecs = weight_dim / VEC;
  for (int32_t i = threadIdx.x; i < num_vecs; i += blockDim.x) {
    float4 packed = __ldg(reinterpret_cast<const float4*>(w) + i);
    const half2* h2 = reinterpret_cast<const half2*>(&packed);
    float4 out_lo, out_hi;
    out_lo.x = __half2float(h2[0].x); out_lo.y = __half2float(h2[0].y);
    out_lo.z = __half2float(h2[1].x); out_lo.w = __half2float(h2[1].y);
    out_hi.x = __half2float(h2[2].x); out_hi.y = __half2float(h2[2].y);
    out_hi.z = __half2float(h2[3].x); out_hi.w = __half2float(h2[3].y);
    reinterpret_cast<float4*>(out)[i * 2]     = out_lo;
    reinterpret_cast<float4*>(out)[i * 2 + 1] = out_hi;
  }
  for (int32_t i = num_vecs * VEC + threadIdx.x; i < weight_dim; i += blockDim.x) {
    out[i] = __half2float(__ldg(w + i));
  }
}

__global__ void emb_kernel_cu_pure_fp16_impl(int32_t vocab_size, int32_t token_num, int32_t weight_dim,
                                              const int32_t* __restrict__ input_ptr,
                                              const half* __restrict__ weight_ptr,
                                              half* __restrict__ output_ptr) {
  int32_t token_idx = blockIdx.x;
  if (token_idx >= token_num) return;
  int32_t token = __ldg(input_ptr + token_idx);
  if (token >= vocab_size) return;
  half* out = output_ptr + static_cast<int64_t>(token_idx) * weight_dim;
  const half* w = weight_ptr + static_cast<int64_t>(token) * weight_dim;
  const int VEC = 8;
  const int num_vecs = weight_dim / VEC;
  for (int32_t i = threadIdx.x; i < num_vecs; i += blockDim.x) {
    float4 wv = __ldg(reinterpret_cast<const float4*>(w) + i);
    reinterpret_cast<float4*>(out)[i] = wv;
  }
  for (int32_t i = num_vecs * VEC + threadIdx.x; i < weight_dim; i += blockDim.x) {
    out[i] = __ldg(w + i);
  }
}

}  // namespace optimized

// ======================== BENCHMARK DRIVER ===========================
int main() {
  // Typical LLM embedding: vocab=151936, dim=4096, tokens=512 (prefill)
  const int VOCAB = 151936;
  const int DIM = 4096;
  const int TOKEN_NUM = 512;

  // Allocate weight table (large!)
  float *d_w_fp32;
  half *d_w_fp16;
  int32_t *d_input;
  float *d_out_fp32;
  half *d_out_fp16;

  cudaMalloc(&d_w_fp32, (size_t)VOCAB * DIM * sizeof(float));
  cudaMalloc(&d_w_fp16, (size_t)VOCAB * DIM * sizeof(half));
  cudaMalloc(&d_input, TOKEN_NUM * sizeof(int32_t));
  cudaMalloc(&d_out_fp32, TOKEN_NUM * DIM * sizeof(float));
  cudaMalloc(&d_out_fp16, TOKEN_NUM * DIM * sizeof(half));

  // Init with random tokens
  {
    int32_t* h_input = (int32_t*)malloc(TOKEN_NUM * sizeof(int32_t));
    for (int i = 0; i < TOKEN_NUM; i++) h_input[i] = rand() % VOCAB;
    cudaMemcpy(d_input, h_input, TOKEN_NUM * sizeof(int32_t), cudaMemcpyHostToDevice);
    free(h_input);
    // Zero-init weights (content doesn't matter for perf)
    cudaMemset(d_w_fp32, 0, (size_t)VOCAB * DIM * sizeof(float));
    cudaMemset(d_w_fp16, 0, (size_t)VOCAB * DIM * sizeof(half));
  }
  cudaDeviceSynchronize();

  // =================== ORIGINAL KERNELS ===================
  printf("=== Running ORIGINAL kernels ===\n");

  // 1. FP32 embedding - original (128 threads)
  original::emb_kernel_cu_fp32<<<TOKEN_NUM, 128>>>(VOCAB, TOKEN_NUM, DIM, d_input, d_w_fp32, d_out_fp32);
  cudaDeviceSynchronize();

  // 2. FP16→FP32 embedding - original (128 threads)
  original::emb_kernel_cu_fp16<<<TOKEN_NUM, 128>>>(VOCAB, TOKEN_NUM, DIM, d_input, d_w_fp16, d_out_fp32);
  cudaDeviceSynchronize();

  // 3. Pure FP16 embedding - original (128 threads)
  original::emb_kernel_cu_pure_fp16_impl<<<TOKEN_NUM, 128>>>(VOCAB, TOKEN_NUM, DIM, d_input, d_w_fp16, d_out_fp16);
  cudaDeviceSynchronize();

  // =================== OPTIMIZED KERNELS ===================
  printf("=== Running OPTIMIZED kernels ===\n");

  // 1. FP32 embedding - optimized (256 threads)
  optimized::emb_kernel_cu_fp32<<<TOKEN_NUM, 256>>>(VOCAB, TOKEN_NUM, DIM, d_input, d_w_fp32, d_out_fp32);
  cudaDeviceSynchronize();

  // 2. FP16→FP32 embedding - optimized (256 threads)
  optimized::emb_kernel_cu_fp16<<<TOKEN_NUM, 256>>>(VOCAB, TOKEN_NUM, DIM, d_input, d_w_fp16, d_out_fp32);
  cudaDeviceSynchronize();

  // 3. Pure FP16 embedding - optimized (256 threads)
  optimized::emb_kernel_cu_pure_fp16_impl<<<TOKEN_NUM, 256>>>(VOCAB, TOKEN_NUM, DIM, d_input, d_w_fp16, d_out_fp16);
  cudaDeviceSynchronize();

  printf("=== All kernels completed ===\n");

  cudaFree(d_w_fp32); cudaFree(d_w_fp16);
  cudaFree(d_input); cudaFree(d_out_fp32); cudaFree(d_out_fp16);
  return 0;
}
