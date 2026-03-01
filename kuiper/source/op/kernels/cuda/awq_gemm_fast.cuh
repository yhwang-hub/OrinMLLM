/*
 * AWQ Ultra-Fast W4A16 Fused Kernels - Version 4.0
 * 
 * Goals: 
 * - Outperform FP16 in BOTH prefill and decode phases
 * - Maintain AWQ memory advantage (no pre-dequantization)
 * - Keep inference accuracy
 * 
 * Key Optimizations:
 * 1. PTX-optimized INT4 to FP16 conversion (minimal instructions)
 * 2. Async double-buffering to hide dequant latency
 * 3. Optimized memory access patterns for INT4 data
 * 4. Vectorized loads to maximize memory bandwidth utilization
 */

#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cublas_v2.h>

namespace kernel {

/**
 * AWQ GEMM/GEMV Fast - Ultra-optimized W4A16 Fused Kernel
 * 
 * This implementation uses a hybrid strategy:
 * - M=1: Memory-bandwidth optimized GEMV (INT4 gives 4x bandwidth advantage)
 * - M>1: Async pipelined fused GEMM (hides dequant behind compute)
 * 
 * @param input Input activations [M, K] FP16
 * @param qweight Quantized weights [K, N/8] INT32 (8 INT4 packed)
 * @param qzeros Quantized zeros [K/group_size, N/8] INT32
 * @param scales Scale factors [K/group_size, N] FP16
 * @param output Output [M, N] FP16
 * @param M Batch size (number of input rows)
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
