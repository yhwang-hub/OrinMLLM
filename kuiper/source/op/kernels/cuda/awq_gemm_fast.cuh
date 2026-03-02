/*
 * AWQ Fast W4A16 GEMV Kernel - M=1 Decode Optimized
 *
 * Uses LOP3-based INT4 to FP16 conversion for fast dequantization.
 * Designed exclusively for decode phase where M=1 (single token generation).
 *
 * For M>1 (prefill), use awq_gemm_vllm_cu (Tensor Core MMA path).
 */

#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace kernel {

/**
 * AWQ GEMV - Ultra-optimized for M=1 decode with LOP3 dequantization
 *
 * Uses LOP3 bit manipulation for fast INT4→FP16 conversion,
 * half2 vectorized accumulation, and high-occupancy GEMV structure
 * to maximize memory bandwidth utilization.
 *
 * @param input Input activations [1, K] FP16
 * @param qweight Quantized weights [K, N/8] INT32 (8 INT4 packed)
 * @param qzeros Quantized zeros [K/group_size, N/8] INT32
 * @param scales Scale factors [K/group_size, N] FP16
 * @param output Output [1, N] FP16
 * @param M Batch size (must be 1)
 * @param K Input dimension
 * @param N Output dimension
 * @param group_size AWQ group size (typically 128)
 * @param stream CUDA stream
 */
void awq_gemm_fast_cu(
    const half* input,
    const int32_t* qweight,
    const int32_t* qzeros,
    const half* scales,
    half* output,
    int M,
    int K,
    int N,
    int group_size,
    cudaStream_t stream
);

}  // namespace kernel
