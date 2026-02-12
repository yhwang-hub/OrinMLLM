/**
 * bench_fused_ffn_timing.cu — CUDA Events precise timing benchmark
 * for Fused FFN (Gate-Up-SwiGLU) kernels
 *
 * Compares original vs optimized versions of 3 actually-used kernels.
 * Uses CUDA Events for microsecond-level timing without sudo.
 *
 * Build:
 *   nvcc -O3 -arch=sm_87 -lineinfo -o bench_timing bench_fused_ffn_timing.cu -I/mnt/ssd/third_party/cub
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/block/block_reduce.cuh>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>

// ============================================================================
// ORIGINAL kernels (without __ldg / fmaf)
// ============================================================================

template <int BLOCK_SIZE = 256>
__global__ void orig_fused_gate_up_swiglu_fp32(
    const float* __restrict__ input, const float* __restrict__ w1,
    const float* __restrict__ w3, float* __restrict__ output,
    const int M, const int K)
{
    const int row = blockIdx.x, tid = threadIdx.x;
    if (row >= K) return;
    const float* w1_row = w1 + row * M;
    const float* w3_row = w3 + row * M;
    float sg = 0.0f, su = 0.0f;
    const int nv = M / 4;
    const float4* iv = reinterpret_cast<const float4*>(input);
    const float4* gv = reinterpret_cast<const float4*>(w1_row);
    const float4* uv = reinterpret_cast<const float4*>(w3_row);
    #pragma unroll 4
    for (int i = tid; i < nv; i += BLOCK_SIZE) {
        float4 x = iv[i], g = gv[i], u = uv[i];
        sg += g.x*x.x + g.y*x.y + g.z*x.z + g.w*x.w;
        su += u.x*x.x + u.y*x.y + u.z*x.z + u.w*x.w;
    }
    for (int i = nv*4 + tid; i < M; i += BLOCK_SIZE) {
        float xv = input[i]; sg += w1_row[i]*xv; su += w3_row[i]*xv;
    }
    using BR = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BR::TempStorage tg, tu;
    sg = BR(tg).Sum(sg); __syncthreads(); su = BR(tu).Sum(su);
    if (tid == 0) { float a = sg/(1.0f+expf(-sg)); output[row] = a*su; }
}

template <int BLOCK_SIZE = 256>
__global__ void orig_fused_gate_up_swiglu_mixed(
    const float* __restrict__ input, const half* __restrict__ w1,
    const half* __restrict__ w3, float* __restrict__ output,
    const int M, const int K)
{
    const int row = blockIdx.x, tid = threadIdx.x;
    if (row >= K) return;
    const half* w1r = w1 + row * M, *w3r = w3 + row * M;
    float sg = 0.0f, su = 0.0f;
    const int nv = M / 4;
    const float4* iv = reinterpret_cast<const float4*>(input);
    const half2* gv = reinterpret_cast<const half2*>(w1r);
    const half2* uv = reinterpret_cast<const half2*>(w3r);
    #pragma unroll 4
    for (int i = tid; i < nv; i += BLOCK_SIZE) {
        float4 x = iv[i];
        half2 g0 = gv[i*2], g1 = gv[i*2+1], u0 = uv[i*2], u1 = uv[i*2+1];
        sg += __half2float(g0.x)*x.x + __half2float(g0.y)*x.y + __half2float(g1.x)*x.z + __half2float(g1.y)*x.w;
        su += __half2float(u0.x)*x.x + __half2float(u0.y)*x.y + __half2float(u1.x)*x.z + __half2float(u1.y)*x.w;
    }
    for (int i = nv*4 + tid; i < M; i += BLOCK_SIZE) {
        float xv = input[i]; sg += __half2float(w1r[i])*xv; su += __half2float(w3r[i])*xv;
    }
    using BR = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BR::TempStorage tg, tu;
    sg = BR(tg).Sum(sg); __syncthreads(); su = BR(tu).Sum(su);
    if (tid == 0) { float a = sg/(1.0f+expf(-sg)); output[row] = a*su; }
}

template <int WARP_SIZE = 32, int WPB = 8>
__global__ void orig_fused_gate_up_swiglu_fp16_v2(
    const half* __restrict__ input, const half* __restrict__ w1,
    const half* __restrict__ w3, half* __restrict__ output,
    const int M, const int K)
{
    const int wid = threadIdx.x / WARP_SIZE, lid = threadIdx.x % WARP_SIZE;
    const int row = blockIdx.x * WPB + wid;
    if (row >= K) return;
    const half* w1r = w1 + static_cast<int64_t>(row)*M, *w3r = w3 + static_cast<int64_t>(row)*M;
    float sg0=0,sg1=0,sg2=0,sg3=0, su0=0,su1=0,su2=0,su3=0;
    const int nf = M / 8;
    const float4* xf = reinterpret_cast<const float4*>(input);
    const float4* gf = reinterpret_cast<const float4*>(w1r);
    const float4* uf = reinterpret_cast<const float4*>(w3r);
    #pragma unroll 4
    for (int i = lid; i < nf; i += WARP_SIZE) {
        float4 xv = xf[i], gv = gf[i], uv = uf[i];
        const half2* xh = reinterpret_cast<const half2*>(&xv);
        const half2* gh = reinterpret_cast<const half2*>(&gv);
        const half2* uh = reinterpret_cast<const half2*>(&uv);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 x2 = __half22float2(xh[j]), g2 = __half22float2(gh[j]), u2 = __half22float2(uh[j]);
            if (j==0) { sg0 += g2.x*x2.x + g2.y*x2.y; su0 += u2.x*x2.x + u2.y*x2.y; }
            else if (j==1) { sg1 += g2.x*x2.x + g2.y*x2.y; su1 += u2.x*x2.x + u2.y*x2.y; }
            else if (j==2) { sg2 += g2.x*x2.x + g2.y*x2.y; su2 += u2.x*x2.x + u2.y*x2.y; }
            else { sg3 += g2.x*x2.x + g2.y*x2.y; su3 += u2.x*x2.x + u2.y*x2.y; }
        }
    }
    float sg = sg0+sg1+sg2+sg3, su = su0+su1+su2+su3;
    for (int i = nf*8+lid; i < M; i += WARP_SIZE) {
        float xv = __half2float(input[i]);
        sg += __half2float(w1r[i])*xv; su += __half2float(w3r[i])*xv;
    }
    #pragma unroll
    for (int o = WARP_SIZE/2; o > 0; o /= 2) {
        sg += __shfl_down_sync(0xffffffff, sg, o);
        su += __shfl_down_sync(0xffffffff, su, o);
    }
    if (lid == 0) { float a = sg/(1.0f+expf(-sg)); output[row] = __float2half(a*su); }
}

// ============================================================================
// OPTIMIZED kernels (__ldg + fmaf + branch elimination)
// ============================================================================

template <int BLOCK_SIZE = 256>
__global__ void opt_fused_gate_up_swiglu_fp32(
    const float* __restrict__ input, const float* __restrict__ w1,
    const float* __restrict__ w3, float* __restrict__ output,
    const int M, const int K)
{
    const int row = blockIdx.x, tid = threadIdx.x;
    if (row >= K) return;
    const float* w1r = w1 + row * M, *w3r = w3 + row * M;
    float sg = 0.0f, su = 0.0f;
    const int nv = M / 4;
    const float4* iv = reinterpret_cast<const float4*>(input);
    const float4* gv = reinterpret_cast<const float4*>(w1r);
    const float4* uv = reinterpret_cast<const float4*>(w3r);
    #pragma unroll 4
    for (int i = tid; i < nv; i += BLOCK_SIZE) {
        float4 x = __ldg(iv + i), g = __ldg(gv + i), u = __ldg(uv + i);
        sg = fmaf(g.x, x.x, fmaf(g.y, x.y, fmaf(g.z, x.z, fmaf(g.w, x.w, sg))));
        su = fmaf(u.x, x.x, fmaf(u.y, x.y, fmaf(u.z, x.z, fmaf(u.w, x.w, su))));
    }
    for (int i = nv*4 + tid; i < M; i += BLOCK_SIZE) {
        float xv = __ldg(input + i);
        sg = fmaf(__ldg(w1r + i), xv, sg); su = fmaf(__ldg(w3r + i), xv, su);
    }
    using BR = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BR::TempStorage tg, tu;
    sg = BR(tg).Sum(sg); __syncthreads(); su = BR(tu).Sum(su);
    if (tid == 0) { float a = sg/(1.0f+expf(-sg)); output[row] = a*su; }
}

template <int BLOCK_SIZE = 256>
__global__ void opt_fused_gate_up_swiglu_mixed(
    const float* __restrict__ input, const half* __restrict__ w1,
    const half* __restrict__ w3, float* __restrict__ output,
    const int M, const int K)
{
    const int row = blockIdx.x, tid = threadIdx.x;
    if (row >= K) return;
    const half* w1r = w1 + row * M, *w3r = w3 + row * M;
    float sg = 0.0f, su = 0.0f;
    const float4* iv = reinterpret_cast<const float4*>(input);
    const float4* w1f = reinterpret_cast<const float4*>(w1r);
    const float4* w3f = reinterpret_cast<const float4*>(w3r);
    const int nf = M / 8;
    #pragma unroll 4
    for (int i = tid; i < nf; i += BLOCK_SIZE) {
        float4 x0 = __ldg(iv + i*2), x1 = __ldg(iv + i*2 + 1);
        float4 gp = __ldg(w1f + i), up = __ldg(w3f + i);
        const half2* gh = reinterpret_cast<const half2*>(&gp);
        const half2* uh = reinterpret_cast<const half2*>(&up);
        sg = fmaf(__half2float(gh[0].x), x0.x, fmaf(__half2float(gh[0].y), x0.y,
             fmaf(__half2float(gh[1].x), x0.z, fmaf(__half2float(gh[1].y), x0.w,
             fmaf(__half2float(gh[2].x), x1.x, fmaf(__half2float(gh[2].y), x1.y,
             fmaf(__half2float(gh[3].x), x1.z, fmaf(__half2float(gh[3].y), x1.w, sg))))))));
        su = fmaf(__half2float(uh[0].x), x0.x, fmaf(__half2float(uh[0].y), x0.y,
             fmaf(__half2float(uh[1].x), x0.z, fmaf(__half2float(uh[1].y), x0.w,
             fmaf(__half2float(uh[2].x), x1.x, fmaf(__half2float(uh[2].y), x1.y,
             fmaf(__half2float(uh[3].x), x1.z, fmaf(__half2float(uh[3].y), x1.w, su))))))));
    }
    for (int i = nf*8 + tid; i < M; i += BLOCK_SIZE) {
        float xv = __ldg(input + i);
        sg = fmaf(__half2float(__ldg(w1r + i)), xv, sg);
        su = fmaf(__half2float(__ldg(w3r + i)), xv, su);
    }
    using BR = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BR::TempStorage tg, tu;
    sg = BR(tg).Sum(sg); __syncthreads(); su = BR(tu).Sum(su);
    if (tid == 0) { float a = sg/(1.0f+expf(-sg)); output[row] = a*su; }
}

template <int WARP_SIZE = 32, int WPB = 8>
__global__ void opt_fused_gate_up_swiglu_fp16_v2(
    const half* __restrict__ input, const half* __restrict__ w1,
    const half* __restrict__ w3, half* __restrict__ output,
    const int M, const int K)
{
    const int wid = threadIdx.x / WARP_SIZE, lid = threadIdx.x % WARP_SIZE;
    const int row = blockIdx.x * WPB + wid;
    if (row >= K) return;
    const half* w1r = w1 + static_cast<int64_t>(row)*M, *w3r = w3 + static_cast<int64_t>(row)*M;
    float sg0=0,sg1=0,sg2=0,sg3=0, su0=0,su1=0,su2=0,su3=0;
    const int nf = M / 8;
    const float4* xf = reinterpret_cast<const float4*>(input);
    const float4* gf = reinterpret_cast<const float4*>(w1r);
    const float4* uf = reinterpret_cast<const float4*>(w3r);
    #pragma unroll 4
    for (int i = lid; i < nf; i += WARP_SIZE) {
        float4 xv = __ldg(xf + i), gv = __ldg(gf + i), uv = __ldg(uf + i);
        const half2* xh = reinterpret_cast<const half2*>(&xv);
        const half2* gh = reinterpret_cast<const half2*>(&gv);
        const half2* uh = reinterpret_cast<const half2*>(&uv);
        float2 x0 = __half22float2(xh[0]), g0 = __half22float2(gh[0]), u0 = __half22float2(uh[0]);
        sg0 = fmaf(g0.x, x0.x, fmaf(g0.y, x0.y, sg0));
        su0 = fmaf(u0.x, x0.x, fmaf(u0.y, x0.y, su0));
        float2 x1 = __half22float2(xh[1]), g1 = __half22float2(gh[1]), u1 = __half22float2(uh[1]);
        sg1 = fmaf(g1.x, x1.x, fmaf(g1.y, x1.y, sg1));
        su1 = fmaf(u1.x, x1.x, fmaf(u1.y, x1.y, su1));
        float2 x2 = __half22float2(xh[2]), g2 = __half22float2(gh[2]), u2 = __half22float2(uh[2]);
        sg2 = fmaf(g2.x, x2.x, fmaf(g2.y, x2.y, sg2));
        su2 = fmaf(u2.x, x2.x, fmaf(u2.y, x2.y, su2));
        float2 x3 = __half22float2(xh[3]), g3 = __half22float2(gh[3]), u3 = __half22float2(uh[3]);
        sg3 = fmaf(g3.x, x3.x, fmaf(g3.y, x3.y, sg3));
        su3 = fmaf(u3.x, x3.x, fmaf(u3.y, x3.y, su3));
    }
    float sg = sg0+sg1+sg2+sg3, su = su0+su1+su2+su3;
    for (int i = nf*8+lid; i < M; i += WARP_SIZE) {
        float xv = __half2float(__ldg(input + i));
        sg = fmaf(__half2float(__ldg(w1r + i)), xv, sg);
        su = fmaf(__half2float(__ldg(w3r + i)), xv, su);
    }
    #pragma unroll
    for (int o = WARP_SIZE/2; o > 0; o /= 2) {
        sg += __shfl_down_sync(0xffffffff, sg, o);
        su += __shfl_down_sync(0xffffffff, su, o);
    }
    if (lid == 0) { float a = sg/(1.0f+expf(-sg)); output[row] = __float2half(a*su); }
}

// ============================================================================
// Timing utility
// ============================================================================

struct BenchResult {
    float avg_ms;
    float min_ms;
};

template <typename Func>
BenchResult benchmark(Func func, int warmup = 5, int iters = 20) {
    // Warmup
    for (int i = 0; i < warmup; i++) func();
    cudaDeviceSynchronize();

    float total = 0.0f, min_t = FLT_MAX;
    for (int i = 0; i < iters; i++) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        func();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        total += ms;
        if (ms < min_t) min_t = ms;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    return {total / iters, min_t};
}

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
    printf("=================================================================\n");
    printf("  Fused FFN (Gate-Up-SwiGLU) Kernel Benchmark\n");
    printf("  Platform: NVIDIA Orin (SM 8.7)\n");
    printf("=================================================================\n\n");

    // ===== Kernel 1: FP32 block-level (Qwen2.5-7B INT8: M=3584, K=18944) =====
    {
        const int M = 3584, K = 18944;
        printf("--- Kernel 1: FP32 Block (Qwen2.5-7B INT8, M=%d K=%d) ---\n", M, K);
        float *d_in, *d_w1, *d_w3, *d_out;
        cudaMalloc(&d_in, M * sizeof(float));
        cudaMalloc(&d_w1, (int64_t)K * M * sizeof(float));
        cudaMalloc(&d_w3, (int64_t)K * M * sizeof(float));
        cudaMalloc(&d_out, K * sizeof(float));
        init_fp32(d_in, M); init_fp32(d_w1, (int64_t)K*M); init_fp32(d_w3, (int64_t)K*M);

        auto r_orig = benchmark([&]() {
            orig_fused_gate_up_swiglu_fp32<256><<<K, 256>>>(d_in, d_w1, d_w3, d_out, M, K);
        });
        auto r_opt = benchmark([&]() {
            opt_fused_gate_up_swiglu_fp32<256><<<K, 256>>>(d_in, d_w1, d_w3, d_out, M, K);
        });
        printf("  Original: avg=%.3f ms, min=%.3f ms\n", r_orig.avg_ms, r_orig.min_ms);
        printf("  Optimized: avg=%.3f ms, min=%.3f ms\n", r_opt.avg_ms, r_opt.min_ms);
        printf("  Speedup: %.2fx (avg), %.2fx (min)\n\n",
               r_orig.avg_ms / r_opt.avg_ms, r_orig.min_ms / r_opt.min_ms);

        cudaFree(d_in); cudaFree(d_w1); cudaFree(d_w3); cudaFree(d_out);
    }

    // ===== Kernel 2: Mixed (Qwen3-8B AWQ: M=4096, K=12288) =====
    {
        const int M = 4096, K = 12288;
        printf("--- Kernel 2: Mixed FP16w+FP32io (Qwen3-8B AWQ, M=%d K=%d) ---\n", M, K);
        float *d_in, *d_out; half *d_w1, *d_w3;
        cudaMalloc(&d_in, M * sizeof(float));
        cudaMalloc(&d_w1, (int64_t)K * M * sizeof(half));
        cudaMalloc(&d_w3, (int64_t)K * M * sizeof(half));
        cudaMalloc(&d_out, K * sizeof(float));
        init_fp32(d_in, M); init_fp16(d_w1, (int64_t)K*M); init_fp16(d_w3, (int64_t)K*M);

        auto r_orig = benchmark([&]() {
            orig_fused_gate_up_swiglu_mixed<256><<<K, 256>>>(d_in, d_w1, d_w3, d_out, M, K);
        });
        auto r_opt = benchmark([&]() {
            opt_fused_gate_up_swiglu_mixed<256><<<K, 256>>>(d_in, d_w1, d_w3, d_out, M, K);
        });
        printf("  Original: avg=%.3f ms, min=%.3f ms\n", r_orig.avg_ms, r_orig.min_ms);
        printf("  Optimized: avg=%.3f ms, min=%.3f ms\n", r_opt.avg_ms, r_opt.min_ms);
        printf("  Speedup: %.2fx (avg), %.2fx (min)\n\n",
               r_orig.avg_ms / r_opt.avg_ms, r_orig.min_ms / r_opt.min_ms);

        cudaFree(d_in); cudaFree(d_w1); cudaFree(d_w3); cudaFree(d_out);
    }

    // ===== Kernel 3a: FP16 warp v2 (Qwen3-8B FP16: M=4096, K=12288) =====
    {
        const int M = 4096, K = 12288;
        constexpr int W = 8, T = 32 * W;
        const int blocks = (K + W - 1) / W;
        printf("--- Kernel 3a: FP16 Warp v2 (Qwen3-8B FP16, M=%d K=%d) ---\n", M, K);
        half *d_in, *d_w1, *d_w3, *d_out;
        cudaMalloc(&d_in, M * sizeof(half));
        cudaMalloc(&d_w1, (int64_t)K * M * sizeof(half));
        cudaMalloc(&d_w3, (int64_t)K * M * sizeof(half));
        cudaMalloc(&d_out, K * sizeof(half));
        init_fp16(d_in, M); init_fp16(d_w1, (int64_t)K*M); init_fp16(d_w3, (int64_t)K*M);

        auto r_orig = benchmark([&]() {
            orig_fused_gate_up_swiglu_fp16_v2<32, W><<<blocks, T>>>(d_in, d_w1, d_w3, d_out, M, K);
        });
        auto r_opt = benchmark([&]() {
            opt_fused_gate_up_swiglu_fp16_v2<32, W><<<blocks, T>>>(d_in, d_w1, d_w3, d_out, M, K);
        });
        printf("  Original: avg=%.3f ms, min=%.3f ms\n", r_orig.avg_ms, r_orig.min_ms);
        printf("  Optimized: avg=%.3f ms, min=%.3f ms\n", r_opt.avg_ms, r_opt.min_ms);
        printf("  Speedup: %.2fx (avg), %.2fx (min)\n\n",
               r_orig.avg_ms / r_opt.avg_ms, r_orig.min_ms / r_opt.min_ms);

        cudaFree(d_in); cudaFree(d_w1); cudaFree(d_w3); cudaFree(d_out);
    }

    // ===== Kernel 3b: FP16 warp v2 (Qwen2.5-7B FP16: M=3584, K=18944) =====
    {
        const int M = 3584, K = 18944;
        constexpr int W = 8, T = 32 * W;
        const int blocks = (K + W - 1) / W;
        printf("--- Kernel 3b: FP16 Warp v2 (Qwen2.5-7B FP16, M=%d K=%d) ---\n", M, K);
        half *d_in, *d_w1, *d_w3, *d_out;
        cudaMalloc(&d_in, M * sizeof(half));
        cudaMalloc(&d_w1, (int64_t)K * M * sizeof(half));
        cudaMalloc(&d_w3, (int64_t)K * M * sizeof(half));
        cudaMalloc(&d_out, K * sizeof(half));
        init_fp16(d_in, M); init_fp16(d_w1, (int64_t)K*M); init_fp16(d_w3, (int64_t)K*M);

        auto r_orig = benchmark([&]() {
            orig_fused_gate_up_swiglu_fp16_v2<32, W><<<blocks, T>>>(d_in, d_w1, d_w3, d_out, M, K);
        });
        auto r_opt = benchmark([&]() {
            opt_fused_gate_up_swiglu_fp16_v2<32, W><<<blocks, T>>>(d_in, d_w1, d_w3, d_out, M, K);
        });
        printf("  Original: avg=%.3f ms, min=%.3f ms\n", r_orig.avg_ms, r_orig.min_ms);
        printf("  Optimized: avg=%.3f ms, min=%.3f ms\n", r_opt.avg_ms, r_opt.min_ms);
        printf("  Speedup: %.2fx (avg), %.2fx (min)\n\n",
               r_orig.avg_ms / r_opt.avg_ms, r_orig.min_ms / r_opt.min_ms);

        cudaFree(d_in); cudaFree(d_w1); cudaFree(d_w3); cudaFree(d_out);
    }

    printf("=================================================================\n");
    printf("  All benchmarks complete.\n");
    printf("=================================================================\n");
    return 0;
}
