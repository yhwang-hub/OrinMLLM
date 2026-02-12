/**
 * CUDA Events Kernel-Level Timing Benchmark for matmul_kernel optimizations
 * Compares original vs optimized kernel performance on NVIDIA Orin
 *
 * Build: nvcc -O3 -arch=sm_87 -o bench_matmul_timing bench_matmul_timing.cu
 * Run:   ./bench_matmul_timing
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/block/block_reduce.cuh>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ===================== ORIGINAL KERNELS =====================

// Original FP32 GEMV (sdata intermediate, no __ldg, no fmaf)
template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void orig_matmul_kernel_cu_fp32(const float* input, const float* weight, float* output, int M, int K) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    int start_row = blockIdx.x * ROW_PER_BLOCK;
    if (start_row >= K) return;
    constexpr int pack_size = 4;
    const int pack_num = M / pack_size;
    const int pack_off = pack_size * pack_num;
    for (int p = start_row; p < start_row + ROW_PER_BLOCK; ++p) {
        sdata[tid] = 0;
        float4* input_f4 = (float4*)input;
        float4* weight_f4 = (float4*)(weight + p * M);
        for (int i = tid; i < pack_num; i += blockDim.x) {
            float4 x = *(input_f4 + i);
            float4 w = *(weight_f4 + i);
            sdata[tid] += x.x*w.x + x.y*w.y + x.z*w.z + x.w*w.w;
        }
        for (int i = pack_off + tid; i < M; i += blockDim.x)
            sdata[tid] += input[i] * weight[p*M+i];
        __syncthreads();
        using BR = cub::BlockReduce<float, THREAD_PER_BLOCK>;
        __shared__ typename BR::TempStorage tmp;
        float s = BR(tmp).Sum(sdata[tid]);
        __syncthreads();
        if (tid == 0) output[p] = s;
        __syncthreads();
    }
}

// Original INT8 GEMV (scalar, no vectorization)
template <int TPB, int RPB>
__global__ void orig_matmul_fp32int8(const float* input, const int8_t* weight, const float* scales, int32_t group_size, float* output, int M, int K) {
    __shared__ float sdata[TPB];
    unsigned int tid = threadIdx.x;
    int start_row = blockIdx.x * RPB;
    if (start_row >= K) return;
    for (int p = start_row; p < start_row + RPB; ++p) {
        sdata[tid] = 0;
        for (int i = tid; i < M; i += TPB) {
            int wi = p * M + i;
            sdata[tid] += input[i] * scales[wi / group_size] * static_cast<float>(weight[wi]);
        }
        __syncthreads();
        using BR = cub::BlockReduce<float, TPB>;
        __shared__ typename BR::TempStorage tmp;
        float s = BR(tmp).Sum(sdata[tid]);
        __syncthreads();
        if (tid == 0) output[p] = s;
        __syncthreads();
    }
}

// Original pure FP16 GEMV v2 (no __ldg, no fmaf)
template <int WARP_SIZE = 32, int WARPS_PER_BLOCK = 4>
__global__ void orig_gemv_pure_fp16_v2(const half* input, const half* weight, half* output, int M, int K) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= K) return;
    const half* row_ptr = weight + (int64_t)row * M;
    float s0=0, s1=0, s2=0, s3=0;
    int nf4 = M / 8;
    const float4* wf4 = reinterpret_cast<const float4*>(row_ptr);
    const float4* xf4 = reinterpret_cast<const float4*>(input);
    for (int i = lane_id; i < nf4; i += WARP_SIZE) {
        float4 w = wf4[i]; float4 x = xf4[i];
        const half2* wh = reinterpret_cast<const half2*>(&w);
        const half2* xh = reinterpret_cast<const half2*>(&x);
        float2 a0=__half22float2(wh[0]), b0=__half22float2(xh[0]);
        float2 a1=__half22float2(wh[1]), b1=__half22float2(xh[1]);
        float2 a2=__half22float2(wh[2]), b2=__half22float2(xh[2]);
        float2 a3=__half22float2(wh[3]), b3=__half22float2(xh[3]);
        s0 += a0.x*b0.x + a0.y*b0.y;
        s1 += a1.x*b1.x + a1.y*b1.y;
        s2 += a2.x*b2.x + a2.y*b2.y;
        s3 += a3.x*b3.x + a3.y*b3.y;
    }
    float sum = s0+s1+s2+s3;
    for (int off = WARP_SIZE/2; off > 0; off /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, off);
    if (lane_id == 0) output[row] = __float2half(sum);
}

// Original FP16in FP16w FP32out (half2, no float4)
template<int WPB, int EPT>
__global__ void orig_gemv_fp16_fp32out(const half* input, const half* weight, float* output, int M, int K) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int row = blockIdx.x * WPB + warp_id;
    if (row >= K) return;
    const half* rp = weight + row * M;
    float sum = 0.0f;
    int h2M = M / 2;
    const half2* ih2 = reinterpret_cast<const half2*>(input);
    const half2* wh2 = reinterpret_cast<const half2*>(rp);
    for (int i = lane_id; i < h2M; i += 32) {
        half2 in = __ldg(&ih2[i]);
        half2 w = __ldg(&wh2[i]);
        float2 inf = __half22float2(in);
        float2 wf = __half22float2(w);
        sum += inf.x * wf.x + inf.y * wf.y;
    }
    int base = h2M * 2;
    if (lane_id == 0 && base < M)
        sum += __half2float(__ldg(&input[base])) * __half2float(__ldg(&rp[base]));
    for (int off = 16; off > 0; off /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, off);
    if (lane_id == 0) output[row] = sum;
}

// ===================== OPTIMIZED KERNELS =====================

// Optimized FP32 GEMV (__ldg, fmaf, register accumulation)
template <int TPB, int RPB>
__global__ void opt_matmul_fp32(const float* __restrict__ input, const float* __restrict__ weight, float* output, int M, int K) {
    unsigned int tid = threadIdx.x;
    int start_row = blockIdx.x * RPB;
    if (start_row >= K) return;
    constexpr int ps = 4;
    int pn = M / ps, po = ps * pn;
    for (int p = start_row; p < start_row + RPB; ++p) {
        float sum = 0.0f;
        const float4* xf4 = reinterpret_cast<const float4*>(input);
        const float4* wf4 = reinterpret_cast<const float4*>(weight + p * M);
        for (int i = tid; i < pn; i += blockDim.x) {
            float4 x = xf4[i];
            float4 w = __ldg(wf4 + i);
            sum = fmaf(x.x, w.x, sum); sum = fmaf(x.y, w.y, sum);
            sum = fmaf(x.z, w.z, sum); sum = fmaf(x.w, w.w, sum);
        }
        for (int i = po + tid; i < M; i += blockDim.x)
            sum = fmaf(input[i], __ldg(weight + p*M + i), sum);
        using BR = cub::BlockReduce<float, TPB>;
        __shared__ typename BR::TempStorage tmp;
        float s = BR(tmp).Sum(sum);
        __syncthreads();
        if (tid == 0) output[p] = s;
        __syncthreads();
    }
}

// Optimized INT8 GEMV (char4, __ldg, fmaf)
template <int TPB, int RPB>
__global__ void opt_matmul_fp32int8(const float* __restrict__ input, const int8_t* __restrict__ weight,
                                     const float* __restrict__ scales, int32_t group_size, float* output, int M, int K) {
    unsigned int tid = threadIdx.x;
    int start_row = blockIdx.x * RPB;
    if (start_row >= K) return;
    int pn = M / 4, po = pn * 4;
    for (int p = start_row; p < start_row + RPB; ++p) {
        float sum = 0.0f;
        int ro = p * M;
        const char4* wc4 = reinterpret_cast<const char4*>(weight + ro);
        const float4* xf4 = reinterpret_cast<const float4*>(input);
        for (int i = tid; i < pn; i += TPB) {
            char4 w4 = __ldg(wc4 + i);
            float4 x4 = xf4[i];
            int bi = ro + i * 4;
            float s0 = __ldg(&scales[bi/group_size]);
            float s1 = __ldg(&scales[(bi+1)/group_size]);
            float s2 = __ldg(&scales[(bi+2)/group_size]);
            float s3 = __ldg(&scales[(bi+3)/group_size]);
            sum = fmaf(x4.x, s0*(float)w4.x, sum);
            sum = fmaf(x4.y, s1*(float)w4.y, sum);
            sum = fmaf(x4.z, s2*(float)w4.z, sum);
            sum = fmaf(x4.w, s3*(float)w4.w, sum);
        }
        for (int i = po + tid; i < M; i += TPB) {
            int wi = ro + i;
            sum = fmaf(input[i], __ldg(&scales[wi/group_size]) * (float)__ldg(weight+wi), sum);
        }
        using BR = cub::BlockReduce<float, TPB>;
        __shared__ typename BR::TempStorage tmp;
        float s = BR(tmp).Sum(sum);
        __syncthreads();
        if (tid == 0) output[p] = s;
        __syncthreads();
    }
}

// Optimized pure FP16 GEMV v2 (__ldg, fmaf)
template <int WARP_SIZE = 32, int WPB = 4>
__global__ void opt_gemv_pure_fp16_v2(const half* __restrict__ input, const half* __restrict__ weight, half* output, int M, int K) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row = blockIdx.x * WPB + warp_id;
    if (row >= K) return;
    const half* rp = weight + (int64_t)row * M;
    float s0=0, s1=0, s2=0, s3=0;
    int nf4 = M / 8;
    const float4* wf4 = reinterpret_cast<const float4*>(rp);
    const float4* xf4 = reinterpret_cast<const float4*>(input);
    for (int i = lane_id; i < nf4; i += WARP_SIZE) {
        float4 w = __ldg(wf4 + i);
        float4 x = __ldg(xf4 + i);
        const half2* wh = reinterpret_cast<const half2*>(&w);
        const half2* xh = reinterpret_cast<const half2*>(&x);
        float2 a0=__half22float2(wh[0]), b0=__half22float2(xh[0]);
        float2 a1=__half22float2(wh[1]), b1=__half22float2(xh[1]);
        float2 a2=__half22float2(wh[2]), b2=__half22float2(xh[2]);
        float2 a3=__half22float2(wh[3]), b3=__half22float2(xh[3]);
        s0 = fmaf(a0.x, b0.x, fmaf(a0.y, b0.y, s0));
        s1 = fmaf(a1.x, b1.x, fmaf(a1.y, b1.y, s1));
        s2 = fmaf(a2.x, b2.x, fmaf(a2.y, b2.y, s2));
        s3 = fmaf(a3.x, b3.x, fmaf(a3.y, b3.y, s3));
    }
    float sum = s0+s1+s2+s3;
    for (int off = WARP_SIZE/2; off > 0; off /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, off);
    if (lane_id == 0) output[row] = __float2half(sum);
}

// Optimized FP16in FP16w FP32out (float4, multiple accumulators, fmaf)
template<int WPB, int EPT>
__global__ void opt_gemv_fp16_fp32out(const half* __restrict__ input, const half* __restrict__ weight, float* output, int M, int K) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int row = blockIdx.x * WPB + warp_id;
    if (row >= K) return;
    const half* rp = weight + (int64_t)row * M;
    float s0=0, s1=0, s2=0, s3=0;
    int nf4 = M / 8;
    const float4* wf4 = reinterpret_cast<const float4*>(rp);
    const float4* xf4 = reinterpret_cast<const float4*>(input);
    for (int i = lane_id; i < nf4; i += 32) {
        float4 w = __ldg(wf4+i); float4 x = __ldg(xf4+i);
        const half2* wh = reinterpret_cast<const half2*>(&w);
        const half2* xh = reinterpret_cast<const half2*>(&x);
        float2 a0=__half22float2(wh[0]), b0=__half22float2(xh[0]);
        float2 a1=__half22float2(wh[1]), b1=__half22float2(xh[1]);
        float2 a2=__half22float2(wh[2]), b2=__half22float2(xh[2]);
        float2 a3=__half22float2(wh[3]), b3=__half22float2(xh[3]);
        s0 = fmaf(a0.x, b0.x, fmaf(a0.y, b0.y, s0));
        s1 = fmaf(a1.x, b1.x, fmaf(a1.y, b1.y, s1));
        s2 = fmaf(a2.x, b2.x, fmaf(a2.y, b2.y, s2));
        s3 = fmaf(a3.x, b3.x, fmaf(a3.y, b3.y, s3));
    }
    float sum = s0+s1+s2+s3;
    for (int off = 16; off > 0; off /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, off);
    if (lane_id == 0) output[row] = sum;
}

// ===================== BENCHMARK HARNESS =====================

float benchmark_kernel(void (*launch_fn)(void*, int, int), void* args, int warmup, int iters) {
    // Warmup
    for (int i = 0; i < warmup; i++) {
        launch_fn(args, 0, 0);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        launch_fn(args, 0, 0);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return ms / iters;
}

struct KernelArgs {
    float* d_input_f32;
    half* d_input_fp16;
    float* d_weight_f32;
    half* d_weight_fp16;
    int8_t* d_weight_int8;
    float* d_scales;
    float* d_output_f32;
    half* d_output_fp16;
    int M, K;
    int group_size;
};

// Launch wrappers
void launch_orig_fp32(void* a, int, int) { KernelArgs* p = (KernelArgs*)a; orig_matmul_kernel_cu_fp32<128,1><<<p->K, 128>>>(p->d_input_f32, p->d_weight_f32, p->d_output_f32, p->M, p->K); }
void launch_opt_fp32(void* a, int, int) { KernelArgs* p = (KernelArgs*)a; opt_matmul_fp32<128,1><<<p->K, 128>>>(p->d_input_f32, p->d_weight_f32, p->d_output_f32, p->M, p->K); }

void launch_orig_int8(void* a, int, int) { KernelArgs* p = (KernelArgs*)a; orig_matmul_fp32int8<128,1><<<p->K, 128>>>(p->d_input_f32, p->d_weight_int8, p->d_scales, p->group_size, p->d_output_f32, p->M, p->K); }
void launch_opt_int8(void* a, int, int) { KernelArgs* p = (KernelArgs*)a; opt_matmul_fp32int8<128,1><<<p->K, 128>>>(p->d_input_f32, p->d_weight_int8, p->d_scales, p->group_size, p->d_output_f32, p->M, p->K); }

void launch_orig_fp16v2(void* a, int, int) { KernelArgs* p = (KernelArgs*)a; int nb=(p->K+7)/8; orig_gemv_pure_fp16_v2<32,8><<<nb, 256>>>(p->d_input_fp16, p->d_weight_fp16, p->d_output_fp16, p->M, p->K); }
void launch_opt_fp16v2(void* a, int, int) { KernelArgs* p = (KernelArgs*)a; int nb=(p->K+7)/8; opt_gemv_pure_fp16_v2<32,8><<<nb, 256>>>(p->d_input_fp16, p->d_weight_fp16, p->d_output_fp16, p->M, p->K); }

void launch_orig_fp16_fp32out(void* a, int, int) { KernelArgs* p = (KernelArgs*)a; int nb=(p->K+7)/8; orig_gemv_fp16_fp32out<8,4><<<nb, 256>>>(p->d_input_fp16, p->d_weight_fp16, p->d_output_f32, p->M, p->K); }
void launch_opt_fp16_fp32out(void* a, int, int) { KernelArgs* p = (KernelArgs*)a; int nb=(p->K+7)/8; opt_gemv_fp16_fp32out<8,4><<<nb, 256>>>(p->d_input_fp16, p->d_weight_fp16, p->d_output_f32, p->M, p->K); }

int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  matmul_kernel CUDA Events Timing Benchmark (Orin SM 8.7)  ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    struct TestCase { int M; int K; const char* desc; };
    TestCase cases[] = {
        {4096, 4096, "Qwen3 q/o projection"},
        {4096, 512,  "Qwen3 kv projection"},
        {4096, 12288,"Qwen3 FFN gate/up"},
        {12288,4096, "Qwen3 FFN down"},
        {3584, 3584, "Qwen2.5 qkv projection"},
        {3584, 18944,"Qwen2.5 FFN gate/up"},
        {18944,3584, "Qwen2.5 FFN down"},
    };

    int warmup = 10, iters = 50;
    int group_size = 128;

    for (auto& tc : cases) {
        int M = tc.M, K = tc.K;
        printf("━━━ %s (M=%d, K=%d) ━━━\n", tc.desc, M, K);

        KernelArgs args;
        args.M = M; args.K = K; args.group_size = group_size;

        // Allocate
        CHECK_CUDA(cudaMalloc(&args.d_input_f32, M * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&args.d_input_fp16, M * sizeof(half)));
        CHECK_CUDA(cudaMalloc(&args.d_weight_f32, (size_t)K * M * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&args.d_weight_fp16, (size_t)K * M * sizeof(half)));
        CHECK_CUDA(cudaMalloc(&args.d_weight_int8, (size_t)K * M));
        int num_groups = ((size_t)K * M + group_size - 1) / group_size;
        CHECK_CUDA(cudaMalloc(&args.d_scales, num_groups * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&args.d_output_f32, K * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&args.d_output_fp16, K * sizeof(half)));

        // Initialize with random data
        float* h_buf = (float*)malloc(std::max((size_t)K*M*sizeof(float), (size_t)num_groups*sizeof(float)));
        for (size_t i = 0; i < (size_t)K*M; i++) h_buf[i] = (rand()/(float)RAND_MAX - 0.5f) * 0.1f;
        CHECK_CUDA(cudaMemcpy(args.d_input_f32, h_buf, M*sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(args.d_weight_f32, h_buf, (size_t)K*M*sizeof(float), cudaMemcpyHostToDevice));
        // FP16 versions
        half* h_fp16 = (half*)malloc((size_t)K*M*sizeof(half));
        for (size_t i = 0; i < (size_t)K*M; i++) h_fp16[i] = __float2half(h_buf[i]);
        CHECK_CUDA(cudaMemcpy(args.d_input_fp16, h_fp16, M*sizeof(half), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(args.d_weight_fp16, h_fp16, (size_t)K*M*sizeof(half), cudaMemcpyHostToDevice));
        // INT8 + scales
        int8_t* h_int8 = (int8_t*)malloc((size_t)K*M);
        for (size_t i = 0; i < (size_t)K*M; i++) h_int8[i] = (int8_t)(rand() % 256 - 128);
        CHECK_CUDA(cudaMemcpy(args.d_weight_int8, h_int8, (size_t)K*M, cudaMemcpyHostToDevice));
        for (int i = 0; i < num_groups; i++) h_buf[i] = 0.01f;
        CHECK_CUDA(cudaMemcpy(args.d_scales, h_buf, num_groups*sizeof(float), cudaMemcpyHostToDevice));

        // Benchmark each kernel type
        float t_orig, t_opt;

        // 1. FP32 GEMV
        t_orig = benchmark_kernel(launch_orig_fp32, &args, warmup, iters);
        t_opt = benchmark_kernel(launch_opt_fp32, &args, warmup, iters);
        printf("  FP32 GEMV:         orig=%.3fms  opt=%.3fms  speedup=%.2fx\n", t_orig, t_opt, t_orig/t_opt);

        // 2. INT8 GEMV
        t_orig = benchmark_kernel(launch_orig_int8, &args, warmup, iters);
        t_opt = benchmark_kernel(launch_opt_int8, &args, warmup, iters);
        printf("  INT8 GEMV:         orig=%.3fms  opt=%.3fms  speedup=%.2fx\n", t_orig, t_opt, t_orig/t_opt);

        // 3. Pure FP16 GEMV v2
        t_orig = benchmark_kernel(launch_orig_fp16v2, &args, warmup, iters);
        t_opt = benchmark_kernel(launch_opt_fp16v2, &args, warmup, iters);
        printf("  Pure FP16 GEMV:    orig=%.3fms  opt=%.3fms  speedup=%.2fx\n", t_orig, t_opt, t_orig/t_opt);

        // 4. FP16in FP16w FP32out
        t_orig = benchmark_kernel(launch_orig_fp16_fp32out, &args, warmup, iters);
        t_opt = benchmark_kernel(launch_opt_fp16_fp32out, &args, warmup, iters);
        printf("  FP16→FP32 GEMV:    orig=%.3fms  opt=%.3fms  speedup=%.2fx\n", t_orig, t_opt, t_orig/t_opt);

        printf("\n");

        // Cleanup
        cudaFree(args.d_input_f32); cudaFree(args.d_input_fp16);
        cudaFree(args.d_weight_f32); cudaFree(args.d_weight_fp16);
        cudaFree(args.d_weight_int8); cudaFree(args.d_scales);
        cudaFree(args.d_output_f32); cudaFree(args.d_output_fp16);
        free(h_buf); free(h_fp16); free(h_int8);
    }

    return 0;
}
