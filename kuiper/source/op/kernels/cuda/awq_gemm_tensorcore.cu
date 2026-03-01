/*
 * AWQ Optimized GEMM/GEMV for KuiperLLama
 * 
 * High-performance INT4 quantized GEMM/GEMV dispatch:
 * - M=1 (decode): awq_gemm_fast_cu (fused GEMV, bandwidth-optimized)
 * - M>1 (prefill): awq_gemm_vllm_cu (Tensor Core MMA with LOP3 dequant)
 * 
 * AWQ uses a specific bit packing order that is NOT sequential.
 * For 8 INT4 values in one INT32:
 *   Output index:  0  1  2  3  4  5  6  7
 *   Bit position:  0  4  8  12 16 20 24 28  (i * 4)
 *   AWQ mapping:   0  4  1  5  2  6  3  7   (reverse order)
 */

#include "awq_gemm_tensorcore.cuh"
#include "awq_gemm_fast.cuh"
#include "awq_gemm_vllm.cuh"
#include <cuda_fp16.h>

namespace kernel {

// ============================================================================
// Weight Dequantization Kernel for cuBLAS HGEMM
// ============================================================================

// ============================================================================
// Global cuBLAS handle (lazy initialization)
// ============================================================================

static cublasHandle_t g_cublas_handle = nullptr;
static bool g_initialized = false;

static void ensure_initialized() {
    if (g_initialized) return;
    
    cublasCreate(&g_cublas_handle);
    cublasSetMathMode(g_cublas_handle, CUBLAS_TENSOR_OP_MATH);
    
    g_initialized = true;
}

// ============================================================================
// Main AWQ GEMM Dispatch
// ============================================================================

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
) {
    ensure_initialized();
    
    if (M == 1) {
        // Decode: use fast fused GEMV kernel (exploits INT4 bandwidth advantage)
        awq_gemm_fast_cu(
            input, qweight, qzeros, scales, output,
            M, in_features, out_features, group_size, stream
        );
    } else {
        // Prefill: use Tensor Core MMA kernel with fused LOP3 dequant
        awq_gemm_vllm_cu(
            input, qweight, qzeros, scales, output,
            M, in_features, out_features, group_size, stream
        );
    }
}

}  // namespace kernel
