// Standalone benchmark for add kernels - both original and optimized versions
// Compile: nvcc -O3 -arch=sm_87 -o bench_add bench_add_kernels.cu
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ======================= ORIGINAL (BEFORE) KERNELS ==========================
namespace original {

__global__ void add_kernel_cu_fp32(int32_t size, const float* in1, const float* in2, float* out) {
  int32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= size) return;
  out[tid] = in1[tid] + in2[tid];
}

__global__ void add_kernel_cu_fp16_impl(int32_t size, const half* in1, const half* in2, half* out) {
  int32_t idx = (threadIdx.x + blockDim.x * blockIdx.x) * 2;
  if (idx + 1 < size) {
    half2 val1 = *reinterpret_cast<const half2*>(in1 + idx);
    half2 val2 = *reinterpret_cast<const half2*>(in2 + idx);
    *reinterpret_cast<half2*>(out + idx) = __hadd2(val1, val2);
  } else if (idx < size) {
    out[idx] = __hadd(in1[idx], in2[idx]);
  }
}

__global__ void broadcast_add_bias_fp16_kernel(
    const half* __restrict__ matrix, const half* __restrict__ bias,
    half* __restrict__ output, int32_t rows, int32_t cols) {
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t total = rows * cols;
  if (idx < total) {
    int32_t col = idx % cols;
    output[idx] = __hadd(matrix[idx], bias[col]);
  }
}

__global__ void add_vec_fp16_kernel(half* a, const half* b, half* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = __hadd(a[idx], b[idx]);
  }
}

}  // namespace original

// ======================= OPTIMIZED (AFTER) KERNELS ==========================
namespace optimized {

__global__ void add_kernel_cu_fp32(int32_t size, const float* __restrict__ in1,
                                   const float* __restrict__ in2, float* __restrict__ out) {
  const int VEC = 4;
  int32_t idx = (threadIdx.x + blockDim.x * blockIdx.x) * VEC;
  if (idx + (VEC - 1) < size) {
    float4 a = __ldg(reinterpret_cast<const float4*>(in1 + idx));
    float4 b = __ldg(reinterpret_cast<const float4*>(in2 + idx));
    float4 c;
    c.x = a.x + b.x; c.y = a.y + b.y; c.z = a.z + b.z; c.w = a.w + b.w;
    *reinterpret_cast<float4*>(out + idx) = c;
  } else {
    #pragma unroll
    for (int32_t i = idx; i < size; i++) {
      out[i] = __ldg(in1 + i) + __ldg(in2 + i);
    }
  }
}

__global__ void add_kernel_cu_fp16_impl(int32_t size, const half* __restrict__ in1,
                                        const half* __restrict__ in2, half* __restrict__ out) {
  const int VEC = 8;
  int32_t idx = (threadIdx.x + blockDim.x * blockIdx.x) * VEC;
  if (idx + (VEC - 1) < size) {
    float4 a4 = __ldg(reinterpret_cast<const float4*>(in1 + idx));
    float4 b4 = __ldg(reinterpret_cast<const float4*>(in2 + idx));
    half2* a = reinterpret_cast<half2*>(&a4);
    half2* b = reinterpret_cast<half2*>(&b4);
    float4 c4;
    half2* c = reinterpret_cast<half2*>(&c4);
    #pragma unroll
    for (int i = 0; i < 4; i++) c[i] = __hadd2(a[i], b[i]);
    *reinterpret_cast<float4*>(out + idx) = c4;
  } else {
    for (int32_t i = idx; i < size; i++) out[i] = __hadd(in1[i], in2[i]);
  }
}

__global__ void broadcast_add_bias_fp16_kernel(
    const half* __restrict__ matrix, const half* __restrict__ bias,
    half* __restrict__ output, int32_t rows, int32_t cols) {
  const int VEC = 8;
  int32_t col_base = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
  int32_t row = blockIdx.y;
  if (row >= rows || col_base >= cols) return;
  int32_t idx = row * cols + col_base;
  if (col_base + VEC <= cols) {
    float4 m = __ldg(reinterpret_cast<const float4*>(matrix + idx));
    float4 b = __ldg(reinterpret_cast<const float4*>(bias + col_base));
    half2* mh = reinterpret_cast<half2*>(&m);
    half2* bh = reinterpret_cast<half2*>(&b);
    float4 result;
    half2* rh = reinterpret_cast<half2*>(&result);
    #pragma unroll
    for (int i = 0; i < 4; i++) rh[i] = __hadd2(mh[i], bh[i]);
    *reinterpret_cast<float4*>(output + idx) = result;
  } else {
    for (int32_t c = col_base; c < cols; c++) {
      int32_t i = row * cols + c;
      output[i] = __hadd(__ldg(matrix + i), __ldg(bias + c));
    }
  }
}

__global__ void add_vec_fp16_kernel(half* __restrict__ a, const half* __restrict__ b,
                                    half* __restrict__ output, int n) {
  const int VEC = 8;
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
  if (idx + (VEC - 1) < n) {
    float4 av = __ldg(reinterpret_cast<const float4*>(a + idx));
    float4 bv = __ldg(reinterpret_cast<const float4*>(b + idx));
    half2* ah = reinterpret_cast<half2*>(&av);
    half2* bh = reinterpret_cast<half2*>(&bv);
    float4 cv;
    half2* ch = reinterpret_cast<half2*>(&cv);
    #pragma unroll
    for (int i = 0; i < 4; i++) ch[i] = __hadd2(ah[i], bh[i]);
    *reinterpret_cast<float4*>(output + idx) = cv;
  } else {
    for (int i = idx; i < n; i++) output[i] = __hadd(a[i], b[i]);
  }
}

}  // namespace optimized

// ======================== BENCHMARK DRIVER ===================================
int main() {
  // Typical LLM shapes: dim=4096, hidden_dim=12288, seq_len=512
  const int N_FP32 = 4096 * 512;        // 2M elements
  const int N_FP16 = 4096 * 512;        // 2M elements (residual connection)
  const int BIAS_ROWS = 512;            // seq_len
  const int BIAS_COLS = 4096;           // dim (QKV bias add)
  const int N_VEC = 4096 * 512;         // 2M elements (DeepStack add)

  float *d_f1, *d_f2, *d_fo;
  half *d_h1, *d_h2, *d_ho;
  half *d_mat, *d_bias, *d_bout;
  half *d_va, *d_vb, *d_vo;

  cudaMalloc(&d_f1, N_FP32 * sizeof(float));
  cudaMalloc(&d_f2, N_FP32 * sizeof(float));
  cudaMalloc(&d_fo, N_FP32 * sizeof(float));
  cudaMalloc(&d_h1, N_FP16 * sizeof(half));
  cudaMalloc(&d_h2, N_FP16 * sizeof(half));
  cudaMalloc(&d_ho, N_FP16 * sizeof(half));
  cudaMalloc(&d_mat, BIAS_ROWS * BIAS_COLS * sizeof(half));
  cudaMalloc(&d_bias, BIAS_COLS * sizeof(half));
  cudaMalloc(&d_bout, BIAS_ROWS * BIAS_COLS * sizeof(half));
  cudaMalloc(&d_va, N_VEC * sizeof(half));
  cudaMalloc(&d_vb, N_VEC * sizeof(half));
  cudaMalloc(&d_vo, N_VEC * sizeof(half));

  // Initialize with random data
  {
    float* h_f = (float*)malloc(N_FP32 * sizeof(float));
    for (int i = 0; i < N_FP32; i++) h_f[i] = (float)(rand() % 1000) / 1000.0f;
    cudaMemcpy(d_f1, h_f, N_FP32 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f2, h_f, N_FP32 * sizeof(float), cudaMemcpyHostToDevice);
    free(h_f);
    
    half* h_h = (half*)malloc(N_FP16 * sizeof(half));
    for (int i = 0; i < N_FP16; i++) h_h[i] = __float2half((float)(rand() % 1000) / 1000.0f);
    cudaMemcpy(d_h1, h_h, N_FP16 * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_h2, h_h, N_FP16 * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat, h_h, BIAS_ROWS * BIAS_COLS * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_va, h_h, N_VEC * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vb, h_h, N_VEC * sizeof(half), cudaMemcpyHostToDevice);
    
    half* h_bias = (half*)malloc(BIAS_COLS * sizeof(half));
    for (int i = 0; i < BIAS_COLS; i++) h_bias[i] = __float2half(0.01f);
    cudaMemcpy(d_bias, h_bias, BIAS_COLS * sizeof(half), cudaMemcpyHostToDevice);
    free(h_h);
    free(h_bias);
  }
  cudaDeviceSynchronize();

  // =================== ORIGINAL KERNELS ===================
  printf("=== Running ORIGINAL kernels ===\n");
  
  // 1. FP32 add - original
  {
    int tn = 512, bn = (N_FP32 + tn - 1) / tn;
    original::add_kernel_cu_fp32<<<bn, tn>>>(N_FP32, d_f1, d_f2, d_fo);
  }
  cudaDeviceSynchronize();

  // 2. FP16 add - original
  {
    int tn = 256, ept = 2, bn = (N_FP16 + tn*ept - 1) / (tn*ept);
    original::add_kernel_cu_fp16_impl<<<bn, tn>>>(N_FP16, d_h1, d_h2, d_ho);
  }
  cudaDeviceSynchronize();

  // 3. Broadcast bias add - original
  {
    int total = BIAS_ROWS * BIAS_COLS;
    int tn = 256, bn = (total + tn - 1) / tn;
    original::broadcast_add_bias_fp16_kernel<<<bn, tn>>>(d_mat, d_bias, d_bout, BIAS_ROWS, BIAS_COLS);
  }
  cudaDeviceSynchronize();

  // 4. Vec FP16 add - original
  {
    int tn = 256, bn = (N_VEC + tn - 1) / tn;
    original::add_vec_fp16_kernel<<<bn, tn>>>(d_va, d_vb, d_vo, N_VEC);
  }
  cudaDeviceSynchronize();

  // =================== OPTIMIZED KERNELS ===================
  printf("=== Running OPTIMIZED kernels ===\n");

  // 1. FP32 add - optimized
  {
    int tn = 256, ept = 4, bn = (N_FP32 + tn*ept - 1) / (tn*ept);
    optimized::add_kernel_cu_fp32<<<bn, tn>>>(N_FP32, d_f1, d_f2, d_fo);
  }
  cudaDeviceSynchronize();

  // 2. FP16 add - optimized
  {
    int tn = 256, ept = 8, bn = (N_FP16 + tn*ept - 1) / (tn*ept);
    optimized::add_kernel_cu_fp16_impl<<<bn, tn>>>(N_FP16, d_h1, d_h2, d_ho);
  }
  cudaDeviceSynchronize();

  // 3. Broadcast bias add - optimized (2D grid)
  {
    int tn = 256, ept = 8;
    int col_blocks = (BIAS_COLS + tn*ept - 1) / (tn*ept);
    dim3 grid(col_blocks, BIAS_ROWS);
    optimized::broadcast_add_bias_fp16_kernel<<<grid, tn>>>(d_mat, d_bias, d_bout, BIAS_ROWS, BIAS_COLS);
  }
  cudaDeviceSynchronize();

  // 4. Vec FP16 add - optimized
  {
    int tn = 256, ept = 8, bn = (N_VEC + tn*ept - 1) / (tn*ept);
    optimized::add_vec_fp16_kernel<<<bn, tn>>>(d_va, d_vb, d_vo, N_VEC);
  }
  cudaDeviceSynchronize();

  printf("=== All kernels completed ===\n");

  cudaFree(d_f1); cudaFree(d_f2); cudaFree(d_fo);
  cudaFree(d_h1); cudaFree(d_h2); cudaFree(d_ho);
  cudaFree(d_mat); cudaFree(d_bias); cudaFree(d_bout);
  cudaFree(d_va); cudaFree(d_vb); cudaFree(d_vo);
  return 0;
}
