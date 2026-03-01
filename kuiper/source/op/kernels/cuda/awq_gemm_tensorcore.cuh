/*
 * AWQ Tensor Core GEMM kernel for KuiperLLama
 * 
 * High-performance INT4 quantized GEMM using Tensor Cores
 * 
 * Version 2.0 Features:
 * - Fast LOP3 dequantization (2-3x faster)
 * - cuBLAS HGEMM integration for prefill
 * - Optimized GEMV for decode
 */

#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>

namespace kernel {

/**
 * AWQ GEMM/GEMV - Fused kernel (for smaller M) or cuBLAS with runtime dequant
 * 
 * Computes: C = A @ dequant(B_quant)
 * Where dequant(w) = scales * (w - zeros)
 * 
 * @param input Input activations [M, in_features] FP16
 * @param qweight Quantized weights [in_features, out_features/8] INT32
 * @param qzeros Quantized zeros [in_features/group_size, out_features/8] INT32
 * @param scales Scale factors [in_features/group_size, out_features] FP16
 * @param output Output [M, out_features] FP16
 * @param M Batch size
 * @param in_features Input channels
 * @param out_features Output channels
 * @param group_size AWQ group size (128)
 * @param split_k_iters Split-K parallelism (not used in v2)
 * @param stream CUDA stream
 */
void awq_gemm_tensorcore_cu(
    const half* input,
    const int32_t* qweight,
    const int32_t* qzeros,
    const half* scales,
    half* output,
    int M,
    int in_features,
    int out_features,
    int group_size,
    int split_k_iters,
    cudaStream_t stream
);

}  // namespace kernel
