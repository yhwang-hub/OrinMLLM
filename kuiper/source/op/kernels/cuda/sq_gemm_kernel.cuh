#ifndef SQ_GEMM_KERNEL_CUH
#define SQ_GEMM_KERNEL_CUH
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace kernel {

/**
 * SmoothQuant INT8 GEMM with Optimized Dual-Path Architecture v2
 *
 * Dispatch:
 *   M=1 (decode): INT8 GEMV kernel (__dp4a, 128-bit loads, warp reduction)
 *   M>1 (prefill): CUTLASS INT8 Tensor Core GEMM (adaptive tile)
 *
 * Full pipeline on GPU, CUDA graph compatible.
 */
void sq_gemm_cu(const half* input_fp16,
                const int8_t* qweight,
                half* output_fp16,
                float weight_scale,
                int batch_size,
                int in_features,
                int out_features,
                cudaStream_t stream);

/**
 * Fused SQ FFN: Gate(W1) + Up(W3) + SwiGLU in a single fused operation.
 * For decode (M=1): quantize input once, then fused W1+W3 GEMV + SwiGLU.
 * Saves 6 kernel launches vs. separate w1, w3 SQ GEMM calls.
 */
void sq_fused_ffn_cu(const half* input_fp16,
                     const int8_t* w1_int8,
                     const int8_t* w3_int8,
                     half* output_fp16,
                     float w1_weight_scale,
                     float w3_weight_scale,
                     int in_features,
                     int hidden_dim,
                     cudaStream_t stream);

/**
 * Quantize input for shared use across multiple GEMV calls (e.g., Q, K, V).
 * Call once, then use sq_gemv_preq_cu() for each projection.
 * Stores quantized input (INT8) and input_scale in internal workspace.
 *
 * Saves 6 kernel launches per layer for QKV (216 per decode step for 36 layers).
 */
void sq_quantize_input_cu(const half* input_fp16,
                          int K,
                          cudaStream_t stream);

/**
 * GEMV with pre-quantized input (from sq_quantize_input_cu).
 * Reads input_scale from workspace and multiplies by per-layer weight_scale.
 */
void sq_gemv_preq_cu(const int8_t* qweight,
                     half* output_fp16,
                     float weight_scale,
                     int K,
                     int N,
                     cudaStream_t stream);

}  // namespace kernel

#endif  // SQ_GEMM_KERNEL_CUH
