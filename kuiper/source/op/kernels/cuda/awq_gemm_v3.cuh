/*
 * AWQ High-Performance W4A16 Fused Kernels - Version 3.0
 * 
 * Header for the optimized AWQ GEMM/GEMV kernels that:
 * 1. Keep AWQ memory advantage (no pre-dequantization)
 * 2. Outperform FP16 in decode phase (GEMV, M=1)
 * 3. Competitive prefill performance (GEMM, M>1)
 */

#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace kernel {

/**
 * AWQ GEMM/GEMV V3 - Optimized W4A16 Fused Kernel
 * 
 * For M=1 (decode): Uses high-bandwidth GEMV kernel
 * For M>1 (prefill): Uses tiled GEMM with shared memory dequant
 * 
 * @param input Input activations [M, in_features] FP16
 * @param qweight Quantized weights [in_features, out_features/8] INT32
 * @param qzeros Quantized zeros [in_features/group_size, out_features/8] INT32
 * @param scales Scale factors [in_features/group_size, out_features] FP16
 * @param output Output [M, out_features] FP16
 * @param M Batch size
 * @param in_features Input dimension (K)
 * @param out_features Output dimension (N)
 * @param group_size AWQ group size (typically 128)
 * @param stream CUDA stream
 */
void awq_gemm_v3_cu(
    const half* input,
    const int32_t* qweight,
    const int32_t* qzeros,
    const half* scales,
    half* output,
    int M,
    int in_features,
    int out_features,
    int group_size,
    cudaStream_t stream
);

}  // namespace kernel
