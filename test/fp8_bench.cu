#include <vector>
// FP8 GEMV Kernel Microbenchmark
// Measures kernel execution time for different layer sizes
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

// Include FP8 GEMM kernel header
#include "kuiper/source/op/kernels/cuda/fp8_gemm_kernel.cuh"

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
}

// Benchmark a single GEMV configuration
void benchmark_gemv(int N, int K, int block_size, int warmup_iters, int test_iters,
                    cublasHandle_t cublas_handle, cudaStream_t stream) {
    int scale_rows = (N + block_size - 1) / block_size;
    int scale_cols = (K + block_size - 1) / block_size;

    // Allocate device memory
    uint8_t* d_weight;
    half* d_scale;
    half* d_input;
    half* d_output;

    CHECK_CUDA(cudaMalloc(&d_weight, (size_t)N * K));
    CHECK_CUDA(cudaMalloc(&d_scale, (size_t)scale_rows * scale_cols * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_input, K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_output, N * sizeof(half)));

    // Initialize with random data
    {
        std::vector<uint8_t> h_weight(N * K);
        std::vector<uint16_t> h_scale(scale_rows * scale_cols);
        std::vector<uint16_t> h_input(K);
        for (auto& v : h_weight) v = rand() % 256;
        for (auto& v : h_scale) v = 0x3C00; // 1.0 in FP16
        for (auto& v : h_input) v = 0x3C00;
        cudaMemcpy(d_weight, h_weight.data(), N * K, cudaMemcpyHostToDevice);
        cudaMemcpy(d_scale, h_scale.data(), scale_rows * scale_cols * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_input, h_input.data(), K * sizeof(half), cudaMemcpyHostToDevice);
    }

    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        kernel::fp8_gemm_cu(d_weight, d_scale, d_input, d_output,
                            1, N, K, block_size, scale_cols, cublas_handle, stream);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Time GEMV
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < test_iters; i++) {
        kernel::fp8_gemm_cu(d_weight, d_scale, d_input, d_output,
                            1, N, K, block_size, scale_cols, cublas_handle, stream);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    float avg_us = elapsed_ms * 1000.0f / test_iters;

    // Calculate memory bandwidth utilization
    double weight_bytes = (double)N * K;  // FP8 = 1 byte
    double scale_bytes = (double)scale_rows * scale_cols * 2;  // FP16
    double input_bytes = K * 2.0;  // FP16
    double output_bytes = N * 2.0;  // FP16
    double total_bytes = weight_bytes + scale_bytes + input_bytes + output_bytes;
    double bandwidth_gbps = total_bytes / (avg_us * 1e-6) / 1e9;

    printf("GEMV [%5d x %5d]: %7.1f us, %6.1f GB/s, weight=%.1f MB\n",
           N, K, avg_us, bandwidth_gbps, weight_bytes / 1e6);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_weight));
    CHECK_CUDA(cudaFree(d_scale));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
}

// Benchmark GEMM (prefill) with dequant + cuBLAS
void benchmark_gemm(int M, int N, int K, int block_size, int warmup_iters, int test_iters,
                    cublasHandle_t cublas_handle, cudaStream_t stream) {
    int scale_rows = (N + block_size - 1) / block_size;
    int scale_cols = (K + block_size - 1) / block_size;

    uint8_t* d_weight;
    half* d_scale;
    half* d_input;
    half* d_output;

    CHECK_CUDA(cudaMalloc(&d_weight, (size_t)N * K));
    CHECK_CUDA(cudaMalloc(&d_scale, (size_t)scale_rows * scale_cols * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_input, (size_t)M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_output, (size_t)M * N * sizeof(half)));

    kernel::fp8_init_dequant_buffer((size_t)N * K);

    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        kernel::fp8_gemm_cu(d_weight, d_scale, d_input, d_output,
                            M, N, K, block_size, scale_cols, cublas_handle, stream);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < test_iters; i++) {
        kernel::fp8_gemm_cu(d_weight, d_scale, d_input, d_output,
                            M, N, K, block_size, scale_cols, cublas_handle, stream);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    float avg_us = elapsed_ms * 1000.0f / test_iters;

    // Pure cuBLAS FP16 reference (no dequant overhead)
    half* d_weight_fp16;
    CHECK_CUDA(cudaMalloc(&d_weight_fp16, (size_t)N * K * sizeof(half)));
    for (int i = 0; i < warmup_iters; i++) {
        half alpha = __float2half(1.0f), beta = __float2half(0.0f);
        cublasSetStream(cublas_handle, stream);
        cublasHgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     N, M, K, &alpha, d_weight_fp16, K, d_input, K, &beta, d_output, N);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < test_iters; i++) {
        half alpha = __float2half(1.0f), beta = __float2half(0.0f);
        cublasSetStream(cublas_handle, stream);
        cublasHgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     N, M, K, &alpha, d_weight_fp16, K, d_input, K, &beta, d_output, N);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float cublas_ms;
    CHECK_CUDA(cudaEventElapsedTime(&cublas_ms, start, stop));
    float cublas_avg_us = cublas_ms * 1000.0f / test_iters;

    printf("GEMM [M=%4d, %5d x %5d]: FP8=%7.1f us, cuBLAS_FP16=%7.1f us, overhead=%.1fx\n",
           M, N, K, avg_us, cublas_avg_us, avg_us / cublas_avg_us);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_weight));
    CHECK_CUDA(cudaFree(d_scale));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_weight_fp16));
}

int main() {
    cudaStream_t stream;
    cublasHandle_t cublas;
    CHECK_CUDA(cudaStreamCreate(&stream));
    cublasCreate(&cublas);

    printf("=== RTX 5070 FP8 GEMV Microbenchmark (Qwen3-4B layers) ===\n");
    printf("Layer dimensions: dim=2560, attn_dim=4096, kv_dim=1024, imm_dim=9728\n\n");

    // Decode GEMV: all layer sizes for Qwen3-4B
    struct { const char* name; int N; int K; } gemv_layers[] = {
        {"wq (q_proj)",   4096, 2560},
        {"wk (k_proj)",   1024, 2560},
        {"wv (v_proj)",   1024, 2560},
        {"wo (o_proj)",   2560, 4096},
        {"w1 (gate_proj)",9728, 2560},
        {"w2 (down_proj)",2560, 9728},
        {"w3 (up_proj)",  9728, 2560},
    };

    printf("--- GEMV (Decode, M=1) ---\n");
    float total_gemv_us = 0;
    for (auto& layer : gemv_layers) {
        benchmark_gemv(layer.N, layer.K, 128, 50, 200, cublas, stream);
        // Run again to get stable number for total
    }

    printf("\n--- Per-layer decode time (36 layers) ---\n");
    float layer_total_us = 0;
    for (auto& layer : gemv_layers) {
        // Get stable measurement
        int scale_cols = (layer.K + 127) / 128;
        uint8_t* d_w; half* d_s; half* d_i; half* d_o;
        cudaMalloc(&d_w, (size_t)layer.N * layer.K);
        cudaMalloc(&d_s, ((layer.N+127)/128) * scale_cols * sizeof(half));
        cudaMalloc(&d_i, layer.K * sizeof(half));
        cudaMalloc(&d_o, layer.N * sizeof(half));
        
        for (int i = 0; i < 100; i++)
            kernel::fp8_gemm_cu(d_w, d_s, d_i, d_o, 1, layer.N, layer.K, 128, scale_cols, cublas, stream);
        cudaStreamSynchronize(stream);
        
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0, stream);
        for (int i = 0; i < 500; i++)
            kernel::fp8_gemm_cu(d_w, d_s, d_i, d_o, 1, layer.N, layer.K, 128, scale_cols, cublas, stream);
        cudaEventRecord(t1, stream);
        cudaEventSynchronize(t1);
        float ms;
        cudaEventElapsedTime(&ms, t0, t1);
        float us = ms * 1000.0f / 500;
        layer_total_us += us;
        printf("  %-15s: %6.1f us\n", layer.name, us);
        
        cudaEventDestroy(t0); cudaEventDestroy(t1);
        cudaFree(d_w); cudaFree(d_s); cudaFree(d_i); cudaFree(d_o);
    }
    printf("  Total per layer: %.1f us\n", layer_total_us);
    printf("  Total 36 layers: %.1f us = %.2f ms\n", layer_total_us * 36, layer_total_us * 36 / 1000.0f);

    printf("\n--- GEMM (Prefill) ---\n");
    int prefill_sizes[] = {32, 64, 128, 256, 512};
    for (int M : prefill_sizes) {
        benchmark_gemm(M, 9728, 2560, 128, 20, 50, cublas, stream);  // gate_proj (largest)
    }

    cublasDestroy(cublas);
    cudaStreamDestroy(stream);
    return 0;
}
