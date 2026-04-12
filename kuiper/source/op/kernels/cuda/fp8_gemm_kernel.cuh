#ifndef FP8_GEMM_KERNEL_CUH
#define FP8_GEMM_KERNEL_CUH
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace kernel {

/**
 * FP8 E4M3 Block-Quantized GEMM with Dual-Path Architecture
 *
 * Dispatch:
 *   M=1 (decode): FP8 GEMV kernel (on-the-fly block dequant, warp reduction)
 *   M>1 (prefill): FP8→FP16 dequant kernel + cuBLAS FP16 Tensor Core GEMM
 *
 * Block quantization: weight[N,K] in FP8 with per-block scale_inv of shape
 *   [ceil(N/block_size), ceil(K/block_size)] in FP16
 * Dequant formula: fp16_val = fp8_val * scale_inv[row/block_size, col/block_size]
 */
void fp8_gemm_cu(const uint8_t* fp8_weight,    // [N, K] FP8 E4M3
                 const half* scale_inv,          // [scale_rows, scale_cols] FP16
                 const half* input_fp16,         // [M, K] FP16
                 half* output_fp16,              // [M, N] FP16
                 int M, int N, int K,
                 int block_size,
                 int scale_cols,
                 cublasHandle_t cublas_handle,
                 cudaStream_t stream);

/**
 * Initialize FP8 shared dequant buffer for prefill GEMM.
 * Called once at model init with max_weight_elements = max(N*K across all layers).
 */
void fp8_init_dequant_buffer(size_t max_weight_elements);

/**
 * Free the shared dequant buffer.
 */
void fp8_free_dequant_buffer();

}  // namespace kernel

#endif  // FP8_GEMM_KERNEL_CUH
