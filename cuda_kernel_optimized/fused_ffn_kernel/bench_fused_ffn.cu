/**
 * bench_fused_ffn.cu — NCU benchmark for Fused FFN (Gate-Up-SwiGLU) kernels
 *
 * Benchmarks 3 actually-used kernels (original vs optimized):
 *   1. fused_gate_up_swiglu_kernel<256>       (FP32, block-level)  — Qwen2.5-7B INT8
 *   2. fused_gate_up_swiglu_kernel_mixed<256>  (FP16w + FP32 I/O) — Qwen3-8B AWQ
 *   3. fused_gate_up_swiglu_kernel_fp16_v2<32,8> (FP16, warp)     — Qwen3-8B/Qwen2.5-7B FP16/Qwen3-VL
 *
 * Build:
 *   nvcc -O3 -arch=sm_87 -lineinfo -o bench_fused_ffn bench_fused_ffn.cu -I/mnt/ssd/third_party/cub
 *
 * NCU profiling:
 *   sudo /usr/local/NVIDIA-Nsight-Compute/ncu --set full --target-processes all \
 *       -o fused_ffn_ncu_report ./bench_fused_ffn
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/block/block_reduce.cuh>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// ============================================================================
// ORIGINAL kernels (without __ldg / fmaf)
// ============================================================================

// 1. FP32 block-level (original)
template <int BLOCK_SIZE = 256>
__global__ void orig_fused_gate_up_swiglu_fp32(
    const float* __restrict__ input,
    const float* __restrict__ w1,
    const float* __restrict__ w3,
    float* __restrict__ output,
    const int M, const int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= K) return;

    const float* w1_row = w1 + row * M;
    const float* w3_row = w3 + row * M;
    float sum_gate = 0.0f, sum_up = 0.0f;

    constexpr int vec_size = 4;
    const int num_vecs = M / vec_size;
    const float4* input_vec = reinterpret_cast<const float4*>(input);
    const float4* w1_vec = reinterpret_cast<const float4*>(w1_row);
    const float4* w3_vec = reinterpret_cast<const float4*>(w3_row);

    #pragma unroll 4
    for (int i = tid; i < num_vecs; i += BLOCK_SIZE) {
        float4 x = input_vec[i];
        float4 g = w1_vec[i];
        float4 u = w3_vec[i];
        sum_gate += g.x * x.x + g.y * x.y + g.z * x.z + g.w * x.w;
        sum_up += u.x * x.x + u.y * x.y + u.z * x.z + u.w * x.w;
    }
    const int base = num_vecs * vec_size;
    for (int i = base + tid; i < M; i += BLOCK_SIZE) {
        float x_val = input[i];
        sum_gate += w1_row[i] * x_val;
        sum_up += w3_row[i] * x_val;
    }

    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage ts_g, ts_u;
    sum_gate = BlockReduce(ts_g).Sum(sum_gate);
    __syncthreads();
    sum_up = BlockReduce(ts_u).Sum(sum_up);
    if (tid == 0) {
        float ga = sum_gate / (1.0f + expf(-sum_gate));
        output[row] = ga * sum_up;
    }
}

// 2. Mixed precision (original)
template <int BLOCK_SIZE = 256>
__global__ void orig_fused_gate_up_swiglu_mixed(
    const float* __restrict__ input,
    const half* __restrict__ w1,
    const half* __restrict__ w3,
    float* __restrict__ output,
    const int M, const int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= K) return;

    const half* w1_row = w1 + row * M;
    const half* w3_row = w3 + row * M;
    float sum_gate = 0.0f, sum_up = 0.0f;

    constexpr int vec_size = 4;
    const int num_vecs = M / vec_size;
    const float4* input_vec = reinterpret_cast<const float4*>(input);
    const half2* w1_vec = reinterpret_cast<const half2*>(w1_row);
    const half2* w3_vec = reinterpret_cast<const half2*>(w3_row);

    #pragma unroll 4
    for (int i = tid; i < num_vecs; i += BLOCK_SIZE) {
        float4 x = input_vec[i];
        half2 g0 = w1_vec[i * 2];
        half2 g1 = w1_vec[i * 2 + 1];
        half2 u0 = w3_vec[i * 2];
        half2 u1 = w3_vec[i * 2 + 1];
        sum_gate += __half2float(g0.x) * x.x + __half2float(g0.y) * x.y
                  + __half2float(g1.x) * x.z + __half2float(g1.y) * x.w;
        sum_up += __half2float(u0.x) * x.x + __half2float(u0.y) * x.y
                + __half2float(u1.x) * x.z + __half2float(u1.y) * x.w;
    }
    const int base = num_vecs * vec_size;
    for (int i = base + tid; i < M; i += BLOCK_SIZE) {
        float x_val = input[i];
        sum_gate += __half2float(w1_row[i]) * x_val;
        sum_up += __half2float(w3_row[i]) * x_val;
    }

    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage ts_g, ts_u;
    sum_gate = BlockReduce(ts_g).Sum(sum_gate);
    __syncthreads();
    sum_up = BlockReduce(ts_u).Sum(sum_up);
    if (tid == 0) {
        float ga = sum_gate / (1.0f + expf(-sum_gate));
        output[row] = ga * sum_up;
    }
}

// 3. FP16 warp-level v2 (original)
template <int WARP_SIZE = 32, int WARPS_PER_BLOCK = 8>
__global__ void orig_fused_gate_up_swiglu_fp16_v2(
    const half* __restrict__ input,
    const half* __restrict__ w1,
    const half* __restrict__ w3,
    half* __restrict__ output,
    const int M, const int K)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= K) return;

    const half* w1_row = w1 + static_cast<int64_t>(row) * M;
    const half* w3_row = w3 + static_cast<int64_t>(row) * M;

    float sum_gate0 = 0.0f, sum_gate1 = 0.0f, sum_gate2 = 0.0f, sum_gate3 = 0.0f;
    float sum_up0 = 0.0f, sum_up1 = 0.0f, sum_up2 = 0.0f, sum_up3 = 0.0f;

    const int num_float4 = M / 8;
    const float4* input_f4 = reinterpret_cast<const float4*>(input);
    const float4* w1_f4 = reinterpret_cast<const float4*>(w1_row);
    const float4* w3_f4 = reinterpret_cast<const float4*>(w3_row);

    #pragma unroll 4
    for (int i = lane_id; i < num_float4; i += WARP_SIZE) {
        float4 x_f4 = input_f4[i];
        float4 g_f4 = w1_f4[i];
        float4 u_f4 = w3_f4[i];
        const half2* x_h2 = reinterpret_cast<const half2*>(&x_f4);
        const half2* g_h2 = reinterpret_cast<const half2*>(&g_f4);
        const half2* u_h2 = reinterpret_cast<const half2*>(&u_f4);

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 xf = __half22float2(x_h2[j]);
            float2 gf = __half22float2(g_h2[j]);
            float2 uf = __half22float2(u_h2[j]);
            if (j == 0) { sum_gate0 += gf.x*xf.x + gf.y*xf.y; sum_up0 += uf.x*xf.x + uf.y*xf.y; }
            else if (j == 1) { sum_gate1 += gf.x*xf.x + gf.y*xf.y; sum_up1 += uf.x*xf.x + uf.y*xf.y; }
            else if (j == 2) { sum_gate2 += gf.x*xf.x + gf.y*xf.y; sum_up2 += uf.x*xf.x + uf.y*xf.y; }
            else { sum_gate3 += gf.x*xf.x + gf.y*xf.y; sum_up3 += uf.x*xf.x + uf.y*xf.y; }
        }
    }
    float sum_gate = sum_gate0 + sum_gate1 + sum_gate2 + sum_gate3;
    float sum_up = sum_up0 + sum_up1 + sum_up2 + sum_up3;
    const int base2 = num_float4 * 8;
    for (int i = base2 + lane_id; i < M; i += WARP_SIZE) {
        float x_val = __half2float(input[i]);
        sum_gate += __half2float(w1_row[i]) * x_val;
        sum_up += __half2float(w3_row[i]) * x_val;
    }
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        sum_gate += __shfl_down_sync(0xffffffff, sum_gate, offset);
        sum_up   += __shfl_down_sync(0xffffffff, sum_up,   offset);
    }
    if (lane_id == 0) {
        float ga = sum_gate / (1.0f + expf(-sum_gate));
        output[row] = __float2half(ga * sum_up);
    }
}

// ============================================================================
// OPTIMIZED kernels (with __ldg + fmaf + branch elimination)
// ============================================================================

// 1. FP32 block-level (optimized)
template <int BLOCK_SIZE = 256>
__global__ void opt_fused_gate_up_swiglu_fp32(
    const float* __restrict__ input,
    const float* __restrict__ w1,
    const float* __restrict__ w3,
    float* __restrict__ output,
    const int M, const int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= K) return;

    const float* w1_row = w1 + row * M;
    const float* w3_row = w3 + row * M;
    float sum_gate = 0.0f, sum_up = 0.0f;

    constexpr int vec_size = 4;
    const int num_vecs = M / vec_size;
    const float4* input_vec = reinterpret_cast<const float4*>(input);
    const float4* w1_vec = reinterpret_cast<const float4*>(w1_row);
    const float4* w3_vec = reinterpret_cast<const float4*>(w3_row);

    #pragma unroll 4
    for (int i = tid; i < num_vecs; i += BLOCK_SIZE) {
        float4 x = __ldg(input_vec + i);
        float4 g = __ldg(w1_vec + i);
        float4 u = __ldg(w3_vec + i);
        sum_gate = fmaf(g.x, x.x, fmaf(g.y, x.y, fmaf(g.z, x.z, fmaf(g.w, x.w, sum_gate))));
        sum_up = fmaf(u.x, x.x, fmaf(u.y, x.y, fmaf(u.z, x.z, fmaf(u.w, x.w, sum_up))));
    }
    const int base = num_vecs * vec_size;
    for (int i = base + tid; i < M; i += BLOCK_SIZE) {
        float x_val = __ldg(input + i);
        sum_gate = fmaf(__ldg(w1_row + i), x_val, sum_gate);
        sum_up = fmaf(__ldg(w3_row + i), x_val, sum_up);
    }

    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage ts_g, ts_u;
    sum_gate = BlockReduce(ts_g).Sum(sum_gate);
    __syncthreads();
    sum_up = BlockReduce(ts_u).Sum(sum_up);
    if (tid == 0) {
        float ga = sum_gate / (1.0f + expf(-sum_gate));
        output[row] = ga * sum_up;
    }
}

// 2. Mixed precision (optimized: float4 weight loads + __ldg + fmaf)
template <int BLOCK_SIZE = 256>
__global__ void opt_fused_gate_up_swiglu_mixed(
    const float* __restrict__ input,
    const half* __restrict__ w1,
    const half* __restrict__ w3,
    float* __restrict__ output,
    const int M, const int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= K) return;

    const half* w1_row = w1 + row * M;
    const half* w3_row = w3 + row * M;
    float sum_gate = 0.0f, sum_up = 0.0f;

    const float4* input_vec = reinterpret_cast<const float4*>(input);
    const float4* w1_f4 = reinterpret_cast<const float4*>(w1_row);
    const float4* w3_f4 = reinterpret_cast<const float4*>(w3_row);
    const int num_f4_w = M / 8;

    #pragma unroll 4
    for (int i = tid; i < num_f4_w; i += BLOCK_SIZE) {
        float4 x0 = __ldg(input_vec + i * 2);
        float4 x1 = __ldg(input_vec + i * 2 + 1);
        float4 g_packed = __ldg(w1_f4 + i);
        float4 u_packed = __ldg(w3_f4 + i);
        const half2* g_h2 = reinterpret_cast<const half2*>(&g_packed);
        const half2* u_h2 = reinterpret_cast<const half2*>(&u_packed);

        sum_gate = fmaf(__half2float(g_h2[0].x), x0.x, fmaf(__half2float(g_h2[0].y), x0.y,
                  fmaf(__half2float(g_h2[1].x), x0.z, fmaf(__half2float(g_h2[1].y), x0.w,
                  fmaf(__half2float(g_h2[2].x), x1.x, fmaf(__half2float(g_h2[2].y), x1.y,
                  fmaf(__half2float(g_h2[3].x), x1.z, fmaf(__half2float(g_h2[3].y), x1.w, sum_gate))))))));
        sum_up = fmaf(__half2float(u_h2[0].x), x0.x, fmaf(__half2float(u_h2[0].y), x0.y,
                 fmaf(__half2float(u_h2[1].x), x0.z, fmaf(__half2float(u_h2[1].y), x0.w,
                 fmaf(__half2float(u_h2[2].x), x1.x, fmaf(__half2float(u_h2[2].y), x1.y,
                 fmaf(__half2float(u_h2[3].x), x1.z, fmaf(__half2float(u_h2[3].y), x1.w, sum_up))))))));
    }
    const int base = num_f4_w * 8;
    for (int i = base + tid; i < M; i += BLOCK_SIZE) {
        float x_val = __ldg(input + i);
        sum_gate = fmaf(__half2float(__ldg(w1_row + i)), x_val, sum_gate);
        sum_up = fmaf(__half2float(__ldg(w3_row + i)), x_val, sum_up);
    }

    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage ts_g, ts_u;
    sum_gate = BlockReduce(ts_g).Sum(sum_gate);
    __syncthreads();
    sum_up = BlockReduce(ts_u).Sum(sum_up);
    if (tid == 0) {
        float ga = sum_gate / (1.0f + expf(-sum_gate));
        output[row] = ga * sum_up;
    }
}

// 3. FP16 warp-level v2 (optimized: __ldg + fmaf + branch elimination)
template <int WARP_SIZE = 32, int WARPS_PER_BLOCK = 8>
__global__ void opt_fused_gate_up_swiglu_fp16_v2(
    const half* __restrict__ input,
    const half* __restrict__ w1,
    const half* __restrict__ w3,
    half* __restrict__ output,
    const int M, const int K)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= K) return;

    const half* w1_row = w1 + static_cast<int64_t>(row) * M;
    const half* w3_row = w3 + static_cast<int64_t>(row) * M;

    float sum_gate0 = 0.0f, sum_gate1 = 0.0f, sum_gate2 = 0.0f, sum_gate3 = 0.0f;
    float sum_up0 = 0.0f, sum_up1 = 0.0f, sum_up2 = 0.0f, sum_up3 = 0.0f;

    const int num_float4 = M / 8;
    const float4* input_f4 = reinterpret_cast<const float4*>(input);
    const float4* w1_f4 = reinterpret_cast<const float4*>(w1_row);
    const float4* w3_f4 = reinterpret_cast<const float4*>(w3_row);

    #pragma unroll 4
    for (int i = lane_id; i < num_float4; i += WARP_SIZE) {
        float4 x_f4 = __ldg(input_f4 + i);
        float4 g_f4 = __ldg(w1_f4 + i);
        float4 u_f4 = __ldg(w3_f4 + i);
        const half2* x_h2 = reinterpret_cast<const half2*>(&x_f4);
        const half2* g_h2 = reinterpret_cast<const half2*>(&g_f4);
        const half2* u_h2 = reinterpret_cast<const half2*>(&u_f4);

        float2 xf0 = __half22float2(x_h2[0]);
        float2 gf0 = __half22float2(g_h2[0]);
        float2 uf0 = __half22float2(u_h2[0]);
        sum_gate0 = fmaf(gf0.x, xf0.x, fmaf(gf0.y, xf0.y, sum_gate0));
        sum_up0 = fmaf(uf0.x, xf0.x, fmaf(uf0.y, xf0.y, sum_up0));

        float2 xf1 = __half22float2(x_h2[1]);
        float2 gf1 = __half22float2(g_h2[1]);
        float2 uf1 = __half22float2(u_h2[1]);
        sum_gate1 = fmaf(gf1.x, xf1.x, fmaf(gf1.y, xf1.y, sum_gate1));
        sum_up1 = fmaf(uf1.x, xf1.x, fmaf(uf1.y, xf1.y, sum_up1));

        float2 xf2 = __half22float2(x_h2[2]);
        float2 gf2 = __half22float2(g_h2[2]);
        float2 uf2 = __half22float2(u_h2[2]);
        sum_gate2 = fmaf(gf2.x, xf2.x, fmaf(gf2.y, xf2.y, sum_gate2));
        sum_up2 = fmaf(uf2.x, xf2.x, fmaf(uf2.y, xf2.y, sum_up2));

        float2 xf3 = __half22float2(x_h2[3]);
        float2 gf3 = __half22float2(g_h2[3]);
        float2 uf3 = __half22float2(u_h2[3]);
        sum_gate3 = fmaf(gf3.x, xf3.x, fmaf(gf3.y, xf3.y, sum_gate3));
        sum_up3 = fmaf(uf3.x, xf3.x, fmaf(uf3.y, xf3.y, sum_up3));
    }
    float sum_gate = sum_gate0 + sum_gate1 + sum_gate2 + sum_gate3;
    float sum_up = sum_up0 + sum_up1 + sum_up2 + sum_up3;
    const int base2 = num_float4 * 8;
    for (int i = base2 + lane_id; i < M; i += WARP_SIZE) {
        float x_val = __half2float(__ldg(input + i));
        sum_gate = fmaf(__half2float(__ldg(w1_row + i)), x_val, sum_gate);
        sum_up = fmaf(__half2float(__ldg(w3_row + i)), x_val, sum_up);
    }
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        sum_gate += __shfl_down_sync(0xffffffff, sum_gate, offset);
        sum_up   += __shfl_down_sync(0xffffffff, sum_up,   offset);
    }
    if (lane_id == 0) {
        float ga = sum_gate / (1.0f + expf(-sum_gate));
        output[row] = __float2half(ga * sum_up);
    }
}

// ============================================================================
// Benchmark driver
// ============================================================================

void init_fp32(float* d, int n) {
    float* h = new float[n];
    for (int i = 0; i < n; i++) h[i] = 0.001f * (i % 1000 - 500);
    cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h;
}

void init_fp16(half* d, int n) {
    half* h = new half[n];
    for (int i = 0; i < n; i++) h[i] = __float2half(0.001f * (i % 1000 - 500));
    cudaMemcpy(d, h, n * sizeof(half), cudaMemcpyHostToDevice);
    delete[] h;
}

int main() {
    // ===== Kernel 1: FP32 block-level (Qwen2.5-7B INT8: M=3584, K=18944) =====
    {
        const int M = 3584, K = 18944;
        float *d_input, *d_w1, *d_w3, *d_out;
        cudaMalloc(&d_input, M * sizeof(float));
        cudaMalloc(&d_w1, (int64_t)K * M * sizeof(float));
        cudaMalloc(&d_w3, (int64_t)K * M * sizeof(float));
        cudaMalloc(&d_out, K * sizeof(float));
        init_fp32(d_input, M);
        init_fp32(d_w1, (int64_t)K * M);
        init_fp32(d_w3, (int64_t)K * M);

        // Warmup
        orig_fused_gate_up_swiglu_fp32<256><<<K, 256>>>(d_input, d_w1, d_w3, d_out, M, K);
        cudaDeviceSynchronize();

        // Original — NCU will capture this
        orig_fused_gate_up_swiglu_fp32<256><<<K, 256>>>(d_input, d_w1, d_w3, d_out, M, K);
        cudaDeviceSynchronize();

        // Optimized — NCU will capture this
        opt_fused_gate_up_swiglu_fp32<256><<<K, 256>>>(d_input, d_w1, d_w3, d_out, M, K);
        cudaDeviceSynchronize();

        cudaFree(d_input); cudaFree(d_w1); cudaFree(d_w3); cudaFree(d_out);
        printf("[FP32 Block] M=%d K=%d done\n", M, K);
    }

    // ===== Kernel 2: Mixed precision (Qwen3-8B AWQ: M=4096, K=12288) =====
    {
        const int M = 4096, K = 12288;
        float *d_input, *d_out;
        half *d_w1, *d_w3;
        cudaMalloc(&d_input, M * sizeof(float));
        cudaMalloc(&d_w1, (int64_t)K * M * sizeof(half));
        cudaMalloc(&d_w3, (int64_t)K * M * sizeof(half));
        cudaMalloc(&d_out, K * sizeof(float));
        init_fp32(d_input, M);
        init_fp16(d_w1, (int64_t)K * M);
        init_fp16(d_w3, (int64_t)K * M);

        // Warmup
        orig_fused_gate_up_swiglu_mixed<256><<<K, 256>>>(d_input, d_w1, d_w3, d_out, M, K);
        cudaDeviceSynchronize();

        // Original
        orig_fused_gate_up_swiglu_mixed<256><<<K, 256>>>(d_input, d_w1, d_w3, d_out, M, K);
        cudaDeviceSynchronize();

        // Optimized
        opt_fused_gate_up_swiglu_mixed<256><<<K, 256>>>(d_input, d_w1, d_w3, d_out, M, K);
        cudaDeviceSynchronize();

        cudaFree(d_input); cudaFree(d_w1); cudaFree(d_w3); cudaFree(d_out);
        printf("[Mixed] M=%d K=%d done\n", M, K);
    }

    // ===== Kernel 3: FP16 warp v2 (Qwen3-8B FP16: M=4096, K=12288) =====
    {
        const int M = 4096, K = 12288;
        constexpr int WARPS = 8;
        constexpr int THREADS = 32 * WARPS;
        const int blocks = (K + WARPS - 1) / WARPS;

        half *d_input, *d_w1, *d_w3, *d_out;
        cudaMalloc(&d_input, M * sizeof(half));
        cudaMalloc(&d_w1, (int64_t)K * M * sizeof(half));
        cudaMalloc(&d_w3, (int64_t)K * M * sizeof(half));
        cudaMalloc(&d_out, K * sizeof(half));
        init_fp16(d_input, M);
        init_fp16(d_w1, (int64_t)K * M);
        init_fp16(d_w3, (int64_t)K * M);

        // Warmup
        orig_fused_gate_up_swiglu_fp16_v2<32, WARPS><<<blocks, THREADS>>>(d_input, d_w1, d_w3, d_out, M, K);
        cudaDeviceSynchronize();

        // Original
        orig_fused_gate_up_swiglu_fp16_v2<32, WARPS><<<blocks, THREADS>>>(d_input, d_w1, d_w3, d_out, M, K);
        cudaDeviceSynchronize();

        // Optimized
        opt_fused_gate_up_swiglu_fp16_v2<32, WARPS><<<blocks, THREADS>>>(d_input, d_w1, d_w3, d_out, M, K);
        cudaDeviceSynchronize();

        cudaFree(d_input); cudaFree(d_w1); cudaFree(d_w3); cudaFree(d_out);
        printf("[FP16 Warp v2] M=%d K=%d done\n", M, K);
    }

    // ===== Kernel 3b: FP16 warp v2 (Qwen2.5-7B FP16: M=3584, K=18944) =====
    {
        const int M = 3584, K = 18944;
        constexpr int WARPS = 8;
        constexpr int THREADS = 32 * WARPS;
        const int blocks = (K + WARPS - 1) / WARPS;

        half *d_input, *d_w1, *d_w3, *d_out;
        cudaMalloc(&d_input, M * sizeof(half));
        cudaMalloc(&d_w1, (int64_t)K * M * sizeof(half));
        cudaMalloc(&d_w3, (int64_t)K * M * sizeof(half));
        cudaMalloc(&d_out, K * sizeof(half));
        init_fp16(d_input, M);
        init_fp16(d_w1, (int64_t)K * M);
        init_fp16(d_w3, (int64_t)K * M);

        // Warmup
        orig_fused_gate_up_swiglu_fp16_v2<32, WARPS><<<blocks, THREADS>>>(d_input, d_w1, d_w3, d_out, M, K);
        cudaDeviceSynchronize();

        // Original
        orig_fused_gate_up_swiglu_fp16_v2<32, WARPS><<<blocks, THREADS>>>(d_input, d_w1, d_w3, d_out, M, K);
        cudaDeviceSynchronize();

        // Optimized
        opt_fused_gate_up_swiglu_fp16_v2<32, WARPS><<<blocks, THREADS>>>(d_input, d_w1, d_w3, d_out, M, K);
        cudaDeviceSynchronize();

        cudaFree(d_input); cudaFree(d_w1); cudaFree(d_w3); cudaFree(d_out);
        printf("[FP16 Warp v2 Qwen2.5] M=%d K=%d done\n", M, K);
    }

    printf("All benchmarks complete.\n");
    return 0;
}
