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

/**
 * AWQ GEMM with pre-dequantized weights - Uses cuBLAS HGEMM directly
 * 
 * This is the fastest path for prefill when weights are pre-dequantized.
 * No runtime dequantization overhead.
 * 
 * @param input Input activations [M, in_features] FP16
 * @param dequant_weight Pre-dequantized weights [in_features, out_features] FP16
 * @param output Output [M, out_features] FP16
 * @param M Batch size
 * @param in_features Input channels
 * @param out_features Output channels
 * @param stream CUDA stream
 */
void awq_gemm_with_dequant_cu(
    const half* input,
    const half* dequant_weight,
    half* output,
    int M,
    int in_features,
    int out_features,
    cudaStream_t stream
);

/**
 * AWQ Weight Dequantization - Pre-dequantize weights for cuBLAS HGEMM
 * 
 * Call this once per layer at model load time to pre-dequantize weights.
 * The dequantized weights can then be reused for multiple forward passes.
 * 
 * @param qweight Quantized weights [in_features, out_features/8] INT32
 * @param qzeros Quantized zeros [in_features/group_size, out_features/8] INT32
 * @param scales Scale factors [in_features/group_size, out_features] FP16
 * @param dequant_weight Output buffer [in_features, out_features] FP16
 * @param in_features Input channels (K)
 * @param out_features Output channels (N)
 * @param group_size AWQ group size (128)
 * @param stream CUDA stream
 */
void awq_dequant_weight_cu(
    const int32_t* qweight,
    const int32_t* qzeros,
    const half* scales,
    half* dequant_weight,
    int in_features,
    int out_features,
    int group_size,
    cudaStream_t stream
);

/**
 * AWQ GEMM with cuBLAS - Dequant + HGEMM (optimal for large M)
 * 
 * This is the fastest approach for prefill (M > 8):
 * 1. Dequantize weights to FP16 (fast LOP3 kernel)
 * 2. Use cuBLAS HGEMM with Tensor Core
 * 
 * @param input Input activations [M, in_features] FP16
 * @param qweight Quantized weights [in_features, out_features/8] INT32
 * @param qzeros Quantized zeros [in_features/group_size, out_features/8] INT32
 * @param scales Scale factors [in_features/group_size, out_features] FP16
 * @param dequant_weight Temp buffer for dequantized weights [in_features, out_features] FP16
 * @param output Output [M, out_features] FP16
 * @param M Batch size
 * @param in_features Input channels
 * @param out_features Output channels
 * @param group_size AWQ group size (128)
 * @param cublas_handle cuBLAS handle
 * @param stream CUDA stream
 */
void awq_gemm_cublas_cu(
    const half* input,
    const int32_t* qweight,
    const int32_t* qzeros,
    const half* scales,
    half* dequant_weight,
    half* output,
    int M,
    int in_features,
    int out_features,
    int group_size,
    cublasHandle_t cublas_handle,
    cudaStream_t stream
);

/**
 * AWQ Weight Repacking - Convert Kuiper AWQ format to vllm format
 * 
 * Kuiper packing: {0,4,1,5,2,6,3,7}
 * vllm packing:   {0,2,4,6,1,3,5,7}
 * 
 * This enables using vllm's fast LOP3 dequantization which extracts
 * values in a way compatible with half2 FMA operations.
 * 
 * @param kuiper_weights Original weights in Kuiper format [K, N/8] INT32
 * @param vllm_weights Output buffer for repacked weights [K, N/8] INT32
 * @param K Input features (in_features)
 * @param N Output features (out_features)
 * @param stream CUDA stream
 */
void awq_repack_weights_cu(
    const int32_t* kuiper_weights,
    int32_t* vllm_weights,
    int K,
    int N,
    cudaStream_t stream
);

}  // namespace kernel
