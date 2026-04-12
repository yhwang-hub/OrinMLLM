#include "fp8_gemm_kernel.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

namespace kernel {

// ======================== FP8 E4M3 → FP32 Conversion ========================
__device__ __forceinline__ float fp8e4m3_to_float(uint8_t val) {
    uint32_t s = (val >> 7);
    uint32_t e = (val >> 3) & 0xF;
    uint32_t m = val & 0x7;
    if (e == 0 && m == 0) return s ? -0.0f : 0.0f;
    if (e == 0) {
        float f = (float)m * 1.953125e-3f;
        return s ? -f : f;
    }
    uint32_t fp32 = (s << 31) | ((e + 120) << 23) | (m << 20);
    return __uint_as_float(fp32);
}

// ======================== Shared Dequant Buffer ========================
static half* g_dequant_buffer = nullptr;
static size_t g_dequant_buffer_size = 0;

void fp8_init_dequant_buffer(size_t max_weight_elements) {
    if (g_dequant_buffer && g_dequant_buffer_size >= max_weight_elements) return;
    if (g_dequant_buffer) cudaFree(g_dequant_buffer);
    cudaMalloc(&g_dequant_buffer, max_weight_elements * sizeof(half));
    g_dequant_buffer_size = max_weight_elements;
}

void fp8_free_dequant_buffer() {
    if (g_dequant_buffer) {
        cudaFree(g_dequant_buffer);
        g_dequant_buffer = nullptr;
        g_dequant_buffer_size = 0;
    }
}

// ======================== Optimized GEMV (M=1) V2 ========================
// Key optimization: use shared memory to cache the input vector tile.
// This reduces redundant global memory reads since all threads in a block
// access the same input elements (but different weight rows in the per-row version
// already only has 1 row... so input caching helps across warps reading same K range).
//
// Actually the main bottleneck is weight bandwidth. For N-large/K-small cases
// (w1/w3: [9728, 2560]), each block reads K=2560 FP8 bytes of weight plus
// ~2560*2 FP16 bytes of input. Total per block: 2560 + 5120 = 7680 bytes.
// With 128 threads and vectorized 16-byte loads, this is ~480 loads per block.
//
// The input vector is shared across ALL N blocks, so it will be cached in L2.
// The bottleneck is pure weight bandwidth.
//
// Alternative approach: process ROWS_PER_BLOCK rows per block to amortize
// any remaining overhead and improve occupancy.

// V2: Multi-row GEMV kernel.
// Each block processes ROWS_PER_BLOCK output rows.
// When ROWS_PER_BLOCK=4: each row gets 1 warp (32 threads).
// When ROWS_PER_BLOCK=2: each row gets 2 warps (64 threads) with cross-warp reduction.
template<int BLOCK_DIM = 128, int ROWS_PER_BLOCK = 4>
__global__ void fp8_gemv_multirow(
    const uint8_t* __restrict__ weight,     // [N, K] FP8
    const half* __restrict__ scale_inv,     // [scale_rows, scale_cols] FP16
    const half* __restrict__ input,         // [K] FP16
    half* __restrict__ output,              // [N] FP16
    int N, int K,
    int block_size, int scale_cols)
{
    const int block_row_start = blockIdx.x * ROWS_PER_BLOCK;
    const int tid = threadIdx.x;

    // Generalized row/thread mapping
    constexpr int THREADS_PER_ROW = BLOCK_DIM / ROWS_PER_BLOCK;
    const int row_in_block = tid / THREADS_PER_ROW;
    const int tid_in_row = tid % THREADS_PER_ROW;
    const int row = block_row_start + row_in_block;

    if (row >= N) return;

    const int scale_row_idx = row / block_size;
    const half* scale_row_base = scale_inv + scale_row_idx * scale_cols;

    float sum = 0.0f;

    // Process K dimension
    const int k_vec16 = K / 16;
    const uint4* w_row_v = reinterpret_cast<const uint4*>(weight + (size_t)row * K);

    for (int kv = tid_in_row; kv < k_vec16; kv += THREADS_PER_ROW) {
        const int k_base = kv * 16;
        uint4 w128 = w_row_v[kv];
        const uint8_t* wb = reinterpret_cast<const uint8_t*>(&w128);

        const int sc_start = k_base / block_size;
        const int sc_end = (k_base + 15) / block_size;

        if (sc_start == sc_end) {
            const float s = __half2float(__ldg(scale_row_base + sc_start));
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                sum += fp8e4m3_to_float(wb[i]) * s * __half2float(input[k_base + i]);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                const int sc = (k_base + i) / block_size;
                const float s = __half2float(__ldg(scale_row_base + sc));
                sum += fp8e4m3_to_float(wb[i]) * s * __half2float(input[k_base + i]);
            }
        }
    }

    // Remainder
    for (int k = k_vec16 * 16 + tid_in_row; k < K; k += THREADS_PER_ROW) {
        const float s = __half2float(__ldg(scale_row_base + k / block_size));
        sum += fp8e4m3_to_float(weight[(size_t)row * K + k]) * s * __half2float(input[k]);
    }

    // Reduction within THREADS_PER_ROW threads for this row
    const int lane_id = tid_in_row & 31;
    // Phase 1: warp-level shuffle reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if constexpr (THREADS_PER_ROW <= 32) {
        // Single warp per row: lane 0 writes result
        if (lane_id == 0) {
            output[row] = __float2half(sum);
        }
    } else {
        // Multi-warp per row: need cross-warp reduction
        constexpr int WARPS_PER_ROW = THREADS_PER_ROW / 32;
        __shared__ float smem[ROWS_PER_BLOCK * WARPS_PER_ROW];
        const int warp_in_row = tid_in_row / 32;

        if (lane_id == 0) {
            smem[row_in_block * WARPS_PER_ROW + warp_in_row] = sum;
        }
        __syncthreads();

        if (warp_in_row == 0 && lane_id < WARPS_PER_ROW) {
            sum = smem[row_in_block * WARPS_PER_ROW + lane_id];
            #pragma unroll
            for (int offset = WARPS_PER_ROW / 2; offset > 0; offset >>= 1) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }
            if (lane_id == 0) {
                output[row] = __float2half(sum);
            }
        }
    }
}

// Original single-row GEMV for smaller N layers (wk, wv with N=1024)
// where multirow has too few blocks for good occupancy
template<int BLOCK_DIM = 128>
__global__ void fp8_block_gemv_kernel(
    const uint8_t* __restrict__ weight,
    const half* __restrict__ scale_inv,
    const half* __restrict__ input,
    half* __restrict__ output,
    int N, int K,
    int block_size, int scale_cols)
{
    const int row = blockIdx.x;
    if (row >= N) return;

    const int tid = threadIdx.x;
    const int scale_row_idx = row / block_size;
    const half* scale_row_base = scale_inv + scale_row_idx * scale_cols;

    float sum = 0.0f;

    const int k_vec16 = K / 16;
    const uint4* w_row_v = reinterpret_cast<const uint4*>(weight + (size_t)row * K);

    for (int kv = tid; kv < k_vec16; kv += BLOCK_DIM) {
        const int k_base = kv * 16;
        uint4 w128 = w_row_v[kv];
        const uint8_t* wb = reinterpret_cast<const uint8_t*>(&w128);

        const int sc_start = k_base / block_size;
        const int sc_end = (k_base + 15) / block_size;

        if (sc_start == sc_end) {
            const float s = __half2float(__ldg(scale_row_base + sc_start));
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                sum += fp8e4m3_to_float(wb[i]) * s * __half2float(input[k_base + i]);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                const int sc = (k_base + i) / block_size;
                const float s = __half2float(__ldg(scale_row_base + sc));
                sum += fp8e4m3_to_float(wb[i]) * s * __half2float(input[k_base + i]);
            }
        }
    }

    for (int k = k_vec16 * 16 + tid; k < K; k += BLOCK_DIM) {
        const float s = __half2float(__ldg(scale_row_base + k / block_size));
        sum += fp8e4m3_to_float(weight[(size_t)row * K + k]) * s * __half2float(input[k]);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    constexpr int NUM_WARPS = BLOCK_DIM / 32;
    __shared__ float warp_sums[NUM_WARPS];

    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    if (lane_id == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane_id < NUM_WARPS) ? warp_sums[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = NUM_WARPS / 2; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane_id == 0) output[row] = __float2half(sum);
    }
}

// ======================== Dequant Kernel ========================
__global__ void fp8_dequant_kernel_v2(
    const uint8_t* __restrict__ fp8_weight,
    const half* __restrict__ scale_inv,
    half* __restrict__ fp16_weight,
    int N, int K,
    int block_size, int scale_cols)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * K;
    const int base = idx * 8;

    if (base + 7 < total) {
        uint2 w8 = *reinterpret_cast<const uint2*>(fp8_weight + base);
        const uint8_t* wb = reinterpret_cast<const uint8_t*>(&w8);

        half results[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int gidx = base + i;
            const int row = gidx / K;
            const int col = gidx % K;
            const float s = __half2float(__ldg(scale_inv + (row / block_size) * scale_cols + col / block_size));
            results[i] = __float2half(fp8e4m3_to_float(wb[i]) * s);
        }
        *reinterpret_cast<uint4*>(fp16_weight + base) = *reinterpret_cast<uint4*>(results);
    } else {
        for (int i = 0; i < 8 && base + i < total; i++) {
            const int gidx = base + i;
            const int row = gidx / K;
            const int col = gidx % K;
            const float s = __half2float(__ldg(scale_inv + (row / block_size) * scale_cols + col / block_size));
            fp16_weight[gidx] = __float2half(fp8e4m3_to_float(fp8_weight[gidx]) * s);
        }
    }
}

// ======================== Main Dispatch ========================
void fp8_gemm_cu(const uint8_t* fp8_weight,
                 const half* scale_inv,
                 const half* input_fp16,
                 half* output_fp16,
                 int M, int N, int K,
                 int block_size,
                 int scale_cols,
                 cublasHandle_t cublas_handle,
                 cudaStream_t stream)
{
    if (M == 1) {
        // GEMV path: choose kernel based on N
        if (N >= 8192) {
            // Very large N (w1/w3-like): 4 rows/block, each row gets 1 warp (32 threads)
            constexpr int ROWS = 4;
            int num_blocks = (N + ROWS - 1) / ROWS;
            fp8_gemv_multirow<128, ROWS><<<num_blocks, 128, 0, stream>>>(
                fp8_weight, scale_inv, input_fp16, output_fp16,
                N, K, block_size, scale_cols);
        } else if (N >= 2560) {
            // Medium N (wq/wo/w2-like): 2 rows/block, each row gets 2 warps (64 threads)
            constexpr int ROWS = 2;
            int num_blocks = (N + ROWS - 1) / ROWS;
            fp8_gemv_multirow<128, ROWS><<<num_blocks, 128, 0, stream>>>(
                fp8_weight, scale_inv, input_fp16, output_fp16,
                N, K, block_size, scale_cols);
        } else {
            // Small N (wk/wv): single-row, 128 threads
            fp8_block_gemv_kernel<128><<<N, 128, 0, stream>>>(
                fp8_weight, scale_inv, input_fp16, output_fp16,
                N, K, block_size, scale_cols);
        }
    } else {
        // GEMM: dequant + cuBLAS
        if (!g_dequant_buffer || g_dequant_buffer_size < (size_t)N * K) {
            fp8_init_dequant_buffer((size_t)N * K);
        }

        int total = N * K;
        int total8 = (total + 7) / 8;
        int threads = 256;
        int blocks = (total8 + threads - 1) / threads;
        fp8_dequant_kernel_v2<<<blocks, threads, 0, stream>>>(
            fp8_weight, scale_inv, g_dequant_buffer,
            N, K, block_size, scale_cols);

        half alpha_h = __float2half(1.0f);
        half beta_h = __float2half(0.0f);
        cublasSetStream(cublas_handle, stream);
        cublasHgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     N, M, K,
                     &alpha_h,
                     g_dequant_buffer, K,
                     input_fp16, K,
                     &beta_h,
                     output_fp16, N);
    }
}

}  // namespace kernel
